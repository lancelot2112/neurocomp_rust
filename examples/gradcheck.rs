//! Gradient check: is there a gradient from an island's best genome? Measure only, nothing
//! evolves, nothing is written but `GC_OUT`.
//!
//! The setup (every `ISL_*` variable, the ladders, a checkpoint by `ISL_RESUME`) is the same as
//! `examples/islands.rs`; the code up to its measure-only modes is copied here unchanged, so a
//! rung here is exactly the rung there. Then, with `GC_KIDS=n` (0, the default: do nothing), for
//! each island (not the frontier) its first genome as resumed (the best) is the parent, and:
//!
//! - `GC_KIDS` children of one mutation each (`TagGenome::mutate`, the operator breeding uses);
//! - `GC_BREED` children made as breeding makes them from that parent: a crossover with another of
//!   the first `ISL_ELITE` genomes one time in four, then one mutation and more with chance 1/2;
//! - `GC_WALKS` random walks of `GC_WALK_LEN` (3) single mutations, every step scored;
//!
//! all scored with the parent on `GC_WORLDS` (24) fresh worlds (70000...) on the island's current
//! rung (its probes, its lesson); the `GC_TOP` (3) children with the highest mean gain on the
//! rung's goal probe as selection grades it (`graded`) are scored again with the parent on
//! `GC_WORLDS2` (24) other fresh worlds (80000...). With `GC_LIFE=1` the parent is also scored on
//! the first worlds with half and twice `ISL_TRIALS`: does a longer life score more (learning)?
//! `GC_PICK=n`: the parent is the best of the first `ISL_ELITE` on the graded goal over n other
//! fresh worlds (90000...), not the first-ranked genome. `GC_SEED` (1): the children's random stream. Every lifetime's probes go to `GC_OUT`
//! (gradcheck.tsv), one line per genome, world and probe: island, rung, set, genome, kind,
//! whether its text equals the parent's, world, probe, value, graded value, work.

#![allow(unused, dead_code)]
use std::cell::{Cell, RefCell};
use std::io::Write;

use neurocomp::evolve::tags::{Pop, TagGenome, Token};
use neurocomp::evolve::{evolve_islands, probe_all_in, probe_means, Island, Lesson, Physics, Stage, ALL_PROBES};
use neurocomp::world::observe::{ProbeRow, PROBES};
use rand::rngs::StdRng;
use rand::Rng;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn row(m: &ProbeRow, only: &[usize]) -> String {
    (0..m.len()).map(|p| if only.contains(&p) { format!("{:>7}", m[p]) } else { format!("{:>7}", ".") }).collect::<Vec<_>>().join("")
}

/// A moved population's id: the top bit, its plasmid's number and its place in the plasmid.
const TAGGED: u32 = 1 << 31;

fn plasmid_of(id: u32) -> Option<u32> {
    (id & TAGGED != 0).then_some((id & !TAGGED) >> 8)
}

/// A probe score change this large counts (a single genome on a few worlds is noisy).
const NOTICE: i32 = 40;

/// A plasmid moved between islands, followed until no genome of its island carries it.
struct Moved {
    n: u32,
    /// the donor island (none: moved before a resume)
    from: Option<usize>,
    to: usize,
    at: usize,
    ids: Vec<u32>,
    peak: usize,
    alive: bool,
    kind: String,
}

/// Whether a genome carries any of a plasmid's populations, and all of them.
fn carries(g: &TagGenome, ids: &[u32]) -> (bool, bool) {
    let has: Vec<bool> = ids.iter().map(|id| g.tokens.iter().any(|t| matches!(t, Token::Pop(p) if p.id == *id))).collect();
    (has.iter().any(|&h| h), has.iter().all(|&h| h))
}

/// What a plasmid is made of: its size, what it reads and writes (ports, or "host": channels it
/// names but does not declare, which resolve among the recipient's), and what could hold state;
/// and its role from that wiring.
fn measure(tokens: &[Token]) -> (String, &'static str) {
    let g = TagGenome { tokens: tokens.to_vec(), next_id: 0 };
    let declared: Vec<u8> = tokens.iter().filter_map(|t| if let Token::Channel { tag, .. } = t { Some(*tag) } else { None }).collect();
    let pops: Vec<&Pop> = tokens.iter().filter_map(|t| if let Token::Pop(p) = t { Some(p) } else { None }).collect();
    let host = |t: u8| !declared.contains(&t);
    let (mut reads, mut writes) = g.ports();
    if pops.iter().any(|p| (!p.from_port && host(p.from)) || p.gate.is_some_and(|(t, _)| host(t)) || p.pause.is_some_and(|(t, _)| host(t))) {
        reads.push("host".into());
    }
    if pops.iter().any(|p| !p.to_port && host(p.to)) {
        writes.push("host".into());
    }
    let mut memory = Vec::new();
    let holds = tokens.iter().filter(|t| matches!(t, Token::Channel { hold: true, .. })).count();
    if holds > 0 { memory.push("holds") }
    if pops.iter().any(|p| !p.from_port && !p.to_port && p.from == p.to) { memory.push("loop") }
    if pops.iter().any(|p| p.lag > 1 || p.span > 1) { memory.push("history") }
    if pops.iter().any(|p| p.repeat) { memory.push("repeat") }
    if pops.iter().any(|p| p.grow) { memory.push("grows") }
    let gated = pops.iter().any(|p| p.gate.is_some() || p.pause.is_some());
    let kernels: usize = pops.iter().filter(|p| !p.wire).map(|p| p.count as usize).sum();
    let wires = pops.iter().filter(|p| p.wire).count();
    let sense = reads.iter().any(|r| ["S", "eye", "ear", "touch", "affect"].contains(&r.as_str()));
    let act = writes.iter().any(|w| ["A", "voice", "hand"].contains(&w.as_str()));
    let (host_in, host_out) = (reads.iter().any(|r| r == "host"), writes.iter().any(|w| w == "host"));
    let role = match (sense, act) {
        (true, true) => "reflex (senses to act)",
        (true, false) if host_out => "sensor (senses into the host)",
        (true, false) => "sealed sensor (reaches nothing)",
        (false, true) if host_in => "driver (host to act)",
        (false, true) => "babbler (acts on no sense)",
        (false, false) if host_in && host_out => "relay (host to host)",
        _ => "sealed (reaches nothing)",
    };
    let list = |v: &[String]| if v.is_empty() { "nothing".to_string() } else { v.join("+") };
    let text = format!(
        "{} populations, {} channels{}, {kernels} kernels, {wires} wires; reads {}; writes {}; memory {}{}",
        pops.len(), declared.len(), if holds > 0 { format!(" ({holds} hold)") } else { String::new() },
        list(&reads), list(&writes), if memory.is_empty() { "none".to_string() } else { memory.join("+") }, if gated { "; gated" } else { "" },
    );
    (text, role)
}

/// The ops some genomes write with, most used first (share of their populations), and how
/// recursive the first is (`TagGenome::loops`): populations on a loop, self-loops among them, and
/// the largest loop's channels.
fn ops_and_loops(genomes: &[TagGenome]) -> String {
    let mut count: Vec<(String, usize)> = Vec::new();
    let mut total = 0usize;
    for g in genomes {
        for t in &g.tokens {
            if let Token::Pop(p) = t {
                let name = p.op.text().split(':').next().unwrap_or("").to_string();
                match count.iter_mut().find(|c| c.0 == name) {
                    Some(c) => c.1 += 1,
                    None => count.push((name, 1)),
                }
                total += 1;
            }
        }
    }
    count.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let ops: Vec<String> = count.iter().take(5).map(|(n, k)| format!("{n} {}%", k * 100 / total.max(1))).collect();
    // what each population reads its input by: a wire copies it; kernels match by their trigger
    // (with ISL_ZEROS, "+0" for a two-mask kernel that expects zeros too)
    let mut inputs: Vec<(String, usize)> = Vec::new();
    for g in genomes {
        for t in &g.tokens {
            if let Token::Pop(p) = t {
                let name = if p.wire { "wire".to_string() } else {
                    let t = ["at-least", "at-most", "band", "tile", "int>=", "int<=", "int~", "int-learned", "lanes>", "lanes<"][(p.trigger as usize).min(9)];
                    // (the width used: two lanes compared are at most 32 bits each)
                    let lane = if p.trigger >= 8 { format!("/{}", 1u32 << p.lane.clamp(3, 5)) } else if p.trigger >= 4 { format!("/{}", 1u32 << p.lane.clamp(3, 6)) } else { String::new() };
                    format!("{t}{lane}{}", if p.expect { "+0" } else { "" })
                };
                match inputs.iter_mut().find(|c| c.0 == name) {
                    Some(c) => c.1 += 1,
                    None => inputs.push((name, 1)),
                }
            }
        }
    }
    inputs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let inputs: Vec<String> = inputs.iter().take(6).map(|(n, k)| format!("{n} {}%", k * 100 / total.max(1))).collect();
    let (on, selfs, largest, pops) = genomes.first().map_or((0, 0, 0, 0), |g| g.loops());
    // (explored: populations kept from exploration; marked: populations with a mark, and the mean)
    let all: Vec<&Pop> = genomes.iter().flat_map(|g| g.tokens.iter().filter_map(|t| if let Token::Pop(p) = t { Some(p) } else { None })).collect();
    let explored = all.iter().filter(|p| p.explored).count();
    let marked: Vec<u32> = all.iter().filter(|p| p.mark > 0).map(|p| p.mark as u32).collect();
    let marks = if marked.is_empty() { String::new() } else { format!("; marked {} (mean {})", marked.len(), marked.iter().sum::<u32>() / marked.len() as u32) };
    // (with ISL_ZEROS: the share of kernel populations that expect zeros as well as ones)
    let kernels = all.iter().filter(|p| !p.wire).count();
    let expect = if neurocomp::evolve::tags::zeros() { format!("; expecting {}%", all.iter().filter(|p| p.expect && !p.wire).count() * 100 / kernels.max(1)) } else { String::new() };
    // (with ISL_ACCIDENT: the share of kernel populations prone to accidents, and their mean
    // exponent: one accident every 2^k steps)
    let prone: Vec<u32> = all.iter().filter(|p| !p.wire && p.accident > 0).map(|p| p.accident as u32).collect();
    let accident = if neurocomp::evolve::tags::accident() { format!("; accident-prone {}% (1 in 2^{})", prone.len() * 100 / kernels.max(1), prone.iter().sum::<u32>() / (prone.len() as u32).max(1)) } else { String::new() };
    // (with ISL_PORTGATE: populations gated or paused on a port, and writing a sense)
    let ported = if neurocomp::evolve::tags::port_gates() {
        let gp = all.iter().filter(|p| p.gate.is_some() && p.gate_at > 0).count();
        let pp = all.iter().filter(|p| p.pause.is_some() && p.pause_at > 0).count();
        let wi = all.iter().filter(|p| p.to_port && p.to_input).count();
        let ro = all.iter().filter(|p| p.from_port && p.from_output).count();
        let mut reading = if neurocomp::evolve::tags::hint_use() || neurocomp::evolve::tags::read_outputs() { format!(", reading an output {}%", ro * 100 / all.len().max(1)) } else { String::new() };
        // (with ISL_INHIBIT: populations with an inhibit input)
        if neurocomp::evolve::tags::inhibit_inputs() {
            let held = all.iter().filter(|p| !p.wire && p.inhibitor.is_some()).count();
            reading += &format!(", inhibited {}%", held * 100 / all.len().max(1));
        }
        // (with ISL_OVERLAP: the overlap rule, in quarters, among the kernel populations)
        if neurocomp::evolve::tags::overlap_genes() {
            let q: Vec<usize> = (1..=4).map(|k| all.iter().filter(|p| !p.wire && p.inhibit_quarters.clamp(1, 4) == k).count()).collect();
            reading += &format!(", overlap quarters 1:{} 2:{} 3:{} 4:{}", q[0], q[1], q[2], q[3]);
        }
        // (with ISL_HEAT: populations with a heat input, warm and cool)
        if neurocomp::evolve::tags::heat_inputs() {
            let warm = all.iter().filter(|p| !p.wire && p.heat.is_some_and(|h| h.1)).count();
            let cool = all.iter().filter(|p| !p.wire && p.heat.is_some_and(|h| !h.1)).count();
            reading += &format!(", heated {}% (warm {}, cool {})", (warm + cool) * 100 / all.len().max(1), warm, cool);
        }
        format!("; port-gated {}%, port-paused {}%, writing a sense {}%{reading}", gp * 100 / all.len().max(1), pp * 100 / all.len().max(1), wi * 100 / all.len().max(1))
    } else { String::new() };
    // (with ISL_SHADOW: predictors kept from shadow explorers, and what they predict)
    let predictors = if neurocomp::program::neurocomp::shadow() {
        let mut what: Vec<(String, usize)> = Vec::new();
        for p in all.iter().filter(|p| p.teacher.is_some()) {
            let (t, at) = p.teacher.unwrap();
            let name = match at { 1 => ["S", "eye", "ear", "touch", "affect", "M"][[0x00u8, 0x07, 0x38, 0xC1, 0xE0, 0x80].iter().enumerate().min_by_key(|p| (*p.1 ^ t).count_ones()).unwrap().0].to_string(), 2 => ["A", "voice", "hand"][[0xFFu8, 0x1F, 0xF8].iter().enumerate().min_by_key(|p| (*p.1 ^ t).count_ones()).unwrap().0].to_string(), _ => "channel".to_string() };
            match what.iter_mut().find(|w| w.0 == name) { Some(w) => w.1 += 1, None => what.push((name, 1)) }
        }
        what.sort_by(|a, b| b.1.cmp(&a.1));
        format!("; predictors {}{}", what.iter().map(|w| w.1).sum::<usize>(), if what.is_empty() { String::new() } else { format!(" ({})", what.iter().map(|w| format!("{} {}", w.0, w.1)).collect::<Vec<_>>().join(", ")) })
    } else { String::new() };
    // (with ISL_SHADOW_HINT: populations carrying a hint, and how many gate or pause on it)
    let hints = if neurocomp::evolve::tags::shadow_hints() {
        let h = all.iter().filter(|p| p.hint.is_some()).count();
        let used = all.iter().filter(|p| p.hint.is_some_and(|x| p.gate.is_some_and(|g| g.0 == x.0 && p.gate_at == x.1) || p.pause.is_some_and(|g| g.0 == x.0 && p.pause_at == x.1))).count();
        format!("; hinted {}% (gated or paused on the hint {})", h * 100 / all.len().max(1), used)
    } else { String::new() };
    format!("ops {} (written); inputs {} (read by); loops: {on} of {pops} wired populations on a loop ({selfs} self-loops), the largest {largest} channels; explored {explored}{marks}{expect}{accident}{ported}{predictors}{hints}", if ops.is_empty() { "none".into() } else { ops.join(", ") }, if inputs.is_empty() { "none".into() } else { inputs.join(", ") })
}

/// The probes where `a` scores at least `NOTICE` above `b`, named.
fn above(a: &ProbeRow, b: &ProbeRow, only: &[usize]) -> Vec<String> {
    only.iter().filter(|&&p| a[p] - b[p] >= NOTICE).map(|&p| PROBES[p].to_string()).collect()
}

/// `name a/b` for each probe in `only` where either is non-zero.
fn pairs(a: &ProbeRow, b: &ProbeRow, only: &[usize]) -> String {
    let v: Vec<String> = only.iter().filter(|&&p| a[p] != 0 || b[p] != 0).map(|&p| format!("{} {}/{}", PROBES[p], a[p], b[p])).collect();
    if v.is_empty() { "all 0".into() } else { v.join(" ") }
}

/// A probe by (the start of) its name.
fn probe_named(n: &str) -> usize {
    PROBES.iter().position(|p| p.starts_with(n)).unwrap_or_else(|| panic!("no probe named {n}"))
}

/// A ladder from its text (see the header).
fn ladder(text: &str) -> Vec<Stage> {
    text.split(';').filter(|x| !x.trim().is_empty()).map(|st| {
        let mut it = st.split_whitespace();
        let name = it.next().unwrap_or("stage").to_string();
        let mut stage = Stage { name, probes: vec![], lesson: Lesson::default(), goal: vec![], hold: env("ISL_HOLD", 3) };
        for kv in it {
            let (k, v) = kv.split_once('=').expect("key=value");
            match k {
                "probes" => stage.probes = v.split('+').map(probe_named).collect(),
                "tokens" => stage.lesson.tokens = v.parse().expect("tokens"),
                "teach" => stage.lesson.teach = v.parse().expect("teach"),
                "swaps" => stage.lesson.swaps = v.parse().expect("swaps"),
                "delay" => stage.lesson.delay = v.parse().expect("delay"),
                "vary" => stage.lesson.vary_delay = v.parse().expect("vary"),
                "distract" => stage.lesson.distract = v == "1",
                "two" => stage.lesson.two_lights = v == "1",
                "go" => stage.lesson.go_tone = v == "1",
                "period" => stage.lesson.period = v.parse().expect("period"),
                "noise" => stage.lesson.noise = v.parse().expect("noise"),
                "blur" => stage.lesson.blur = v.parse().expect("blur"),
                "shift" => stage.lesson.shift = v.parse().expect("shift"),
                "maze" => stage.lesson.maze_len = v.parse().expect("maze"),
                "gone" => stage.lesson.cue_gone = v == "1",
                "symbol" => stage.lesson.symbol = v == "1",
                "fixed" => stage.lesson.maze_fixed = v == "1",
                "nocue" => stage.lesson.maze_nocue = v == "1",
                "mirror" => stage.lesson.mirror = v.parse().expect("mirror"),
                "quiet" => stage.lesson.quiet = v == "1",
                "motor" => stage.lesson.motor_place = v == "1",
                "babble" => stage.lesson.babble = v.parse().expect("babble"),
                "senses" => stage.lesson.sense_place = v == "1",
                "cue" => stage.lesson.mixed_cue = v == "1",
                "block" => stage.lesson.mixed_block = v.parse().expect("block"),
                "sw" => stage.lesson.screen_w = v.parse().expect("sw"),
                "sh" => stage.lesson.screen_h = v.parse().expect("sh"),
                "lit" => stage.lesson.screen_lit = v.parse().expect("lit"),
                "cell" => stage.lesson.screen_cell = v.parse().expect("cell"),
                "walls" => stage.lesson.screen_walls = v.parse().expect("walls"),
                "torch" => stage.lesson.screen_torch = v == "1",
                "dark" => stage.lesson.screen_dark = v == "1",
                "hold" => stage.hold = v.parse().expect("hold"),
                "goal" => stage.goal = v.split('+').map(|g| {
                    let (p, x) = g.split_once('@').expect("probe@score");
                    (probe_named(p), x.parse().expect("score"))
                }).collect(),
                _ => panic!("unknown key {k}"),
            }
        }
        stage
    }).collect()
}

/// Reading's ladder: a varied voice, then echoing (sound to voice, no glyph) with more and more
/// tokens, then the teacher fading so the glyph must carry more of the work.
const READING_LADDER: &str = "babble probes=babble tokens=2 teach=8 goal=babble@768; \
    echo2 probes=babble+echo tokens=2 teach=8 goal=echo@150; \
    echo4 probes=babble+echo tokens=4 teach=8 goal=echo@150; \
    echo8 probes=babble+echo tokens=8 teach=8 goal=echo@150; \
    fade6 probes=echo+copy tokens=8 teach=6 goal=copy@100; \
    fade4 probes=echo+copy tokens=8 teach=4 goal=copy@100; \
    fade2 probes=echo+copy tokens=8 teach=2 goal=copy@100; \
    read probes=echo+copy tokens=8 teach=1";

/// Switching's ladder: see a light, tell the two apart, repeat what was stroked, prefer the
/// paying light, then follow one swap, then three.
const SWITCHING_LADDER: &str = "respond probes=respond swaps=0 goal=respond@100; \
    tell probes=respond+tell swaps=0 goal=tell@100; \
    repeat probes=respond+tell+repeat swaps=0 maze=0 fixed=1 nocue=1 goal=repeat@300; \
    winstay probes=respond+tell+win-stay swaps=0 goal=win-stay@80; \
    discrim probes=respond+tell+win-stay+discriminate swaps=0 goal=discriminate@100; \
    swap1 probes=respond+tell+win-stay+discriminate+reverse swaps=1 goal=reverse@80; \
    swap3 probes=respond+tell+win-stay+discriminate+reverse swaps=3";

/// Recall's ladder: click at a light (respond, as switching starts: a population that never
/// clicks at a light scores 0 on every lag, and one that does passes lag1 already), then after a
/// light gone 1 step before the window, then 2, then 3 (the plain task), then 4.
const RECALL_LADDER: &str = "respond probes=respond swaps=0 goal=respond@100; \
    lag1 probes=respond+remember delay=1 goal=remember@150; \
    lag2 probes=respond+remember delay=2 goal=remember@150; \
    lag3 probes=respond+remember delay=3 goal=remember@150; \
    lag4 probes=respond+remember delay=4";

/// Recall's long ladder (`ISL_RECALL_LONG=1`): lag4 and on to longer delays (the trial lengthened
/// to fit), a distracting flash of the other light during the delay, a delay drawn per trial (the
/// memory held until the window, not timed), and two lights (which one came, not that one did).
const RECALL_LONG_LADDER: &str = "respond probes=respond swaps=0 goal=respond@100; \
    lag1 probes=respond+remember delay=1 goal=remember@150; \
    lag2 probes=respond+remember delay=2 goal=remember@150; \
    lag3 probes=respond+remember delay=3 goal=remember@150; \
    lag4 probes=respond+remember delay=4 goal=remember@150; \
    lag5 probes=respond+remember delay=5 goal=remember@150; \
    lag8 probes=respond+remember delay=8 goal=remember@150; \
    lag16 probes=respond+remember delay=16 goal=remember@150; \
    distract probes=respond+remember delay=8 distract=1 goal=remember@150; \
    vary probes=respond+remember vary=8 goal=remember@150; \
    two probes=respond+remember delay=8 two=1";

/// Recall's long ladder with fair rungs (`ISL_RECALL_FAIR=1`): after lag16, two lights (which one
/// came), then a distractor of the other light during the delay (a latch on the latest light is
/// wrong; with the other light on no-cue trials, clicking a fixed time after any light scores 0),
/// then a delay drawn per trial with a tone as the window opens (which light came held until the
/// tone; a delay line from the light hits only when the drawn delay matches it).
const RECALL_FAIR_LADDER: &str = "respond probes=respond swaps=0 goal=respond@100; \
    lag1 probes=respond+remember delay=1 goal=remember@150; \
    lag2 probes=respond+remember delay=2 goal=remember@150; \
    lag3 probes=respond+remember delay=3 goal=remember@150; \
    lag4 probes=respond+remember delay=4 goal=remember@150; \
    lag5 probes=respond+remember delay=5 goal=remember@150; \
    lag8 probes=respond+remember delay=8 goal=remember@150; \
    lag16 probes=respond+remember delay=16 goal=remember@150; \
    two probes=respond+remember delay=8 two=1 goal=remember@150; \
    distract probes=respond+remember delay=8 two=1 distract=1 goal=remember@150; \
    vary probes=respond+remember vary=8 two=1 go=1";

/// Prediction's ladder: a stream of tokens in a cycle, one in eight replaced by another; the
/// program is scored on clicking after a token that broke the cycle (surprise), with no code to
/// learn for any token. A break in a cycle of k is a token unlike the one k steps back, so each
/// stage asks for a longer reach into the past; the last ones need the two tokens before (a cycle
/// of 6 over 3 tokens repeats tokens) and a longer cycle.
const PREDICTION_LADDER: &str = "cycle1 probes=surprise period=1 noise=1 goal=surprise@200; \
    cycle2 probes=surprise period=2 noise=1 goal=surprise@200; \
    cycle3 probes=surprise period=3 noise=1 goal=surprise@200; \
    cycle4 probes=surprise period=4 noise=1 goal=surprise@200; \
    order2 probes=surprise period=6 tokens=3 noise=1 goal=surprise@150; \
    cycle8 probes=surprise period=8 noise=1";

/// Shapes' ladder: the two lights become shapes, a fresh exemplar every trial (each line moved
/// within its word with chance `blur` in 8, the whole pattern rotated by up to `shift` lines,
/// which carries it into other words and tiles), and the program must still tell them apart;
/// then repeat what was stroked (win-stay, as switching's ladder: shapes' best told the shapes
/// apart, tell 191 held out, but clicked the stroked one less, win-stay -112, discriminate
/// -108), then prefer the paying one, then follow one swap of it.
const SHAPES_LADDER: &str = "clean probes=respond+tell swaps=0 goal=tell@150; \
    blur1 probes=respond+tell swaps=0 blur=1 goal=tell@150; \
    blur2 probes=respond+tell swaps=0 blur=2 goal=tell@150; \
    shift8 probes=respond+tell swaps=0 blur=1 shift=8 goal=tell@120; \
    shift64 probes=respond+tell swaps=0 blur=2 shift=64 goal=tell@100; \
    repeat probes=respond+tell+repeat swaps=0 blur=2 shift=16 maze=0 fixed=1 nocue=1 goal=repeat@300; \
    winstay probes=respond+tell+win-stay swaps=0 blur=2 shift=16 goal=win-stay@80; \
    paying probes=respond+tell+win-stay+discriminate swaps=0 blur=2 shift=16 goal=discriminate@100; \
    swap1 probes=respond+tell+discriminate+reverse swaps=1 blur=2 shift=16";

/// Maze's ladder: a T-maze, stroked at the food's arm. Turn to the side the cue is on with no stem,
/// then walk a stem of 2 first, then with the cue only at the stem's foot (remember the side), a
/// longer stem, then a cue that is a light's code (which side it means is learned from the
/// stroke), then that code only at the foot. The hand's codes are drawn each lifetime, so even
/// the first stage is learned within a lifetime, not wired in.
const MAZE_LADDER: &str = "learn probes=repeat+maze maze=0 fixed=1 nocue=1 goal=repeat@300; \
    turn probes=maze maze=0 goal=maze@200; \
    walk probes=maze maze=2 goal=maze@200; \
    gone probes=maze maze=2 gone=1 goal=maze@150; \
    long probes=maze maze=4 gone=1 goal=maze@150; \
    symbol probes=maze maze=0 symbol=1 goal=maze@150; \
    symbolgone probes=maze maze=2 gone=1 symbol=1";

/// Screen's ladder: move a cursor to the glyph on the screen and click it (stroked on the target).
/// Two cells side by side in the dark with a torch (no cursor mark, and the glyph shows only
/// while the cursor is on it: the eye is empty unless it is on the target, so clicking at what
/// is seen is a respond; runs 4-6 scored 0-24 on the torch alone, where the cursor's mark is
/// always seen and the glyph must be told from it), then the torch with the cursor shown (on the
/// plain two cells, telling the cursor on the glyph from beside it, the population scored 0-54
/// of 300 for 200 generations), then two cells plain,
/// then four in a row, then a 2 x 2 grid (up and down too), then 4 x 2;
/// then a 2-D maze: a 4 x 4 grid of 32-bit cells, then with 3 walls, then 6 (a move into a wall
/// does nothing; the way round must be found), then 8 x 4 of 16-bit cells with 10 walls; then a
/// distractor beside the target (which glyph is the target is learned from the stroke), then
/// three distractors on the full screen.
const SCREEN_LADDER: &str = "glow probes=screen sw=2 sh=1 lit=1 torch=1 dark=1 goal=screen@300; \
    torch probes=screen sw=2 sh=1 lit=1 torch=1 goal=screen@300; \
    two probes=screen sw=2 sh=1 lit=1 goal=screen@300; \
    row probes=screen sw=4 sh=1 lit=1 goal=screen@250; \
    square probes=screen sw=2 sh=2 lit=1 goal=screen@250; \
    full probes=screen sw=4 sh=2 lit=1 goal=screen@200; \
    grid probes=screen sw=4 sh=4 cell=32 lit=1 goal=screen@200; \
    walls probes=screen sw=4 sh=4 cell=32 walls=3 lit=1 goal=screen@150; \
    maze probes=screen sw=4 sh=4 cell=32 walls=6 lit=1 goal=screen@120; \
    bigmaze probes=screen sw=8 sh=4 cell=16 walls=10 lit=1 goal=screen@100; \
    pick probes=screen sw=2 sh=2 lit=2 goal=screen@150; \
    crowd probes=screen sw=4 sh=2 lit=4";

/// Task selection's ladder: one lifetime, one network, blocks alternating responding (click at a
/// light, the hand) and babbling (vary the voice, the tokens shown), scored by `select` (the two
/// tasks' mean minus doing the other's thing). First with each block's task felt as a cue on the
/// touch channel, then with none (the task read from the scene), then short blocks.
/// Selection's step rung (`ISL_SELECT_STEP=1`): before cued, the same mixed lifetime scored on
/// responding in its conditioning blocks alone (`mixclick`, the respond part of select), so
/// clicking at the lights pays before it has to be gated by the task. No run's selection island
/// ever held cued's goal on its training worlds: its elites babbled and never clicked.
const SELECT_STEP: &str = "mixclick probes=mixclick tokens=2 cue=1 goal=mixclick@100;";

const SELECTION_LADDER: &str = "cued probes=select tokens=2 cue=1 goal=select@600; \
    uncued probes=select tokens=2 cue=0 goal=select@600; \
    short probes=select tokens=2 cue=0 block=10";

/// The world the generalist and the cross-island table use: the plain world, with a cycle of 3
/// broken one time in 8 for surprise (the plain stream has no pattern to break).
fn plain() -> Lesson {
    Lesson { period: 3, noise: 1, ..Lesson::default() }
}

fn main() {
    neurocomp::program::neurocomp::set_in_place(env("ISL_INPLACE", 0u8) == 1);
    neurocomp::program::neurocomp::set_count_reads(env("ISL_READS", 0u8) == 1);
    neurocomp::evolve::set_carry(env("ISL_CARRY", 0u64));
    neurocomp::evolve::set_carry_energy(env("ISL_CARRY_ENERGY", 0u8) == 1);
    neurocomp::kernel::class::set_op_choice(env("ISL_OPS", 0u8) == 1);
    neurocomp::evolve::set_epigenetics(env("ISL_EPI", 0u8) == 1);
    neurocomp::program::neurocomp::set_explore(env("ISL_EXPLORE", 0u8) == 1);
    neurocomp::evolve::tags::set_zeros(env("ISL_ZEROS", 0u8) == 1);
    neurocomp::evolve::tags::set_accident(env("ISL_ACCIDENT", 0u8) == 1);
    neurocomp::evolve::tags::set_port_gates(env("ISL_PORTGATE", 0u8) == 1);
    neurocomp::program::neurocomp::set_shadow(env("ISL_SHADOW", 0u8) == 1);
    neurocomp::evolve::tags::set_pause_draws(env("ISL_PAUSES", 0u8) == 1);
    neurocomp::evolve::tags::set_shadow_hints(env("ISL_SHADOW_HINT", 0u8) == 1);
    neurocomp::evolve::tags::set_hint_use(env("ISL_HINT_USE", 0u8) == 1);
    neurocomp::evolve::tags::set_read_outputs(env("ISL_READ_OUT", 0u8) == 1);
    neurocomp::evolve::tags::set_hint_band(env("ISL_HINT_BAND", 0u8) == 1);
    neurocomp::evolve::tags::set_self_writes(env("ISL_SELF_WRITE", 0u8) == 1);
    neurocomp::evolve::tags::set_close_loops(env("ISL_LOOPS", 0u8) == 1);
    neurocomp::evolve::tags::set_heat_inputs(env("ISL_HEAT", 0u8) == 1);
    neurocomp::evolve::tags::set_inhibit_inputs(env("ISL_INHIBIT", 0u8) == 1);
    neurocomp::evolve::tags::set_overlap_genes(env("ISL_OVERLAP", 0u8) == 1);
    neurocomp::evolve::set_elite_pass(env("ISL_ELITE_PASS", 0u8) == 1);
    neurocomp::evolve::set_partial(env("ISL_PARTIAL", 0u8) == 1);
    neurocomp::evolve::set_partial_remember(env("ISL_PARTIAL_REMEMBER", 0u8) == 1);
    neurocomp::evolve::set_partial_select(env("ISL_PARTIAL_SELECT", 0u8) == 1);
    neurocomp::evolve::set_fair_sequences(env("ISL_FAIR_SEQUENCES", 0u8) == 1);
    neurocomp::evolve::set_varied_orders(env("ISL_VARIED_ORDERS", 0u8) == 1);
    neurocomp::world::observe::set_fair_said(env("ISL_FAIR_SAID", 0u8) == 1);
    neurocomp::world::observe::set_fair_win_stay(env("ISL_FAIR_WINSTAY", 0u8) == 1);
    neurocomp::program::neurocomp::set_clock_drawn(env("ISL_CLOCK_DRAWN", 0u8) == 1);
    neurocomp::evolve::set_fresh_held(env("ISL_FRESH_HELD", 0u8) == 1);
    neurocomp::evolve::set_one_word_voice(env("ISL_ONE_WORD_VOICE", 0u8) == 1);
    neurocomp::evolve::set_migrate(env("ISL_MIGRATE", 0u8) == 1);
    neurocomp::evolve::set_migrate_up(env("ISL_MIGRATE_UP", 0usize));
    neurocomp::evolve::set_migrate_up_n(env("ISL_MIGRATE_UP_N", 1usize));
    neurocomp::evolve::set_generalist_room(env("ISL_GENERALIST_ROOM", 0usize));
    neurocomp::evolve::set_maze_mirror(env("ISL_MAZE_MIRROR", 0u8) == 1);
    neurocomp::evolve::set_same_word_lights(env("ISL_SAME_WORD", 0u8) == 1);
    neurocomp::evolve::set_food_from_world(env("ISL_FOOD_WORLD", 0u8) == 1);
    neurocomp::evolve::set_keep_rungs(env("ISL_KEEP_RUNGS", 0u8) == 1);
    neurocomp::evolve::set_size_classes(env("ISL_SIZE_CLASSES", 0u8) == 1);
    neurocomp::evolve::set_thrift_relax(env("ISL_THRIFT_RELAX", 0usize));
    neurocomp::evolve::set_wire_cost(env("ISL_WIRE_COST", 0u64));
    neurocomp::program::neurocomp::set_write_compete(env("ISL_COMPETE", 0u8) == 1);
    neurocomp::evolve::tags::set_branches(env("ISL_BRANCH", 0u8) == 1);
    neurocomp::evolve::set_muscle(env("ISL_MUSCLE", 0u8) == 1);
    neurocomp::evolve::set_sentences(env("ISL_SENTENCE", 0u8) == 1);
    neurocomp::evolve::set_one_word(env("ISL_ONE_WORD", 0usize));
    neurocomp::evolve::set_body(env("ISL_BODY", 0u64));
    neurocomp::evolve::tags::set_eligibility(env("ISL_ELIG", 0u8) == 1);
    neurocomp::evolve::tags::set_rule_tables(env("ISL_RULES", 0u8) == 1);
    neurocomp::evolve::tags::set_triggers(env("ISL_TRIGGERS", 0u8) == 1);
    neurocomp::evolve::tags::set_motions(env("ISL_MOTIONS", 0u8) == 1);
    neurocomp::evolve::set_frontier_passed(env("ISL_FRONTIER_PASSED", 0u8) == 1);
    let frontier_ladders = env("ISL_FRONTIER_LADDERS", 0u8) == 1;
    neurocomp::evolve::tags::set_birth_lags(env("ISL_BIRTH_LAG", 0u8) == 1);
    neurocomp::program::neurocomp::set_prune(env("ISL_COMPETE", 0u8) == 1 && env("ISL_PRUNE", 0u8) == 1);
    // (energy: a ration a step, and food in rations for each 1024 points scored; before any
    // lesson is made)
    let ration: u64 = env("ISL_RATION", 0);
    neurocomp::evolve::set_energy(ration, ration * env("ISL_FOOD", 32u64));
    let pop: usize = env("ISL_POP", 16);
    let elite: usize = env("ISL_ELITE", 4);
    let gens: usize = env("ISL_GENS", 300);
    let lives: u64 = env("ISL_LIVES", 12);
    let held_n: u64 = env("ISL_HELD", 12);
    let trials: usize = env("ISL_TRIALS", 200);
    let gap: usize = env("ISL_GAP", 6);
    let every: usize = env("ISL_EVERY", 20);
    let plasmids: usize = env("ISL_PLASMIDS", 2);
    let report_every: usize = env("ISL_REPORT", 10);
    let seed: u64 = env("ISL_SEED", 7);
    let thrift = env("ISL_THRIFT", 1u8) == 1;
    let log_path = std::env::var("ISL_LOG").unwrap_or_else(|_| "plasmids.txt".into());
    let resume = std::env::var("ISL_RESUME").ok();
    let mut log = if resume.is_some() {
        std::fs::OpenOptions::new().create(true).append(true).open(&log_path).expect("plasmid log")
    } else {
        std::fs::File::create(&log_path).expect("plasmid log")
    };
    let ck_path = std::env::var("ISL_CHECKPOINT").unwrap_or_else(|_| "checkpoint.txt".into());
    let ckpt: usize = env("ISL_CKPT", 0);
    let train: Vec<u64> = (0..lives).collect();
    let held: Vec<u64> = (10_000..10_000 + held_n).collect();
    let mut islands = vec![
        Island::plain("switching", vec![0, 1, 2, 3, 5]),
        Island::plain("recall", vec![0, 4]),
        Island::plain("reading", vec![6, 7]),
        Island { name: "generalist".into(), frontier: env("ISL_FRONTIER", 0u8) == 1, stages: vec![Stage { name: "all".into(), probes: ALL_PROBES.to_vec(), lesson: plain(), goal: vec![], hold: 0 }] },
    ];
    // (`ISL_PREDICTION=1`: a fifth island, prediction, on the surprise probe; it starts naked, or
    // from a checkpoint that has it)
    if env("ISL_PREDICTION", 0u8) == 1 {
        islands.push(Island { name: "prediction".into(), frontier: false, stages: vec![Stage { name: "all".into(), probes: vec![9], lesson: plain(), goal: vec![], hold: 0 }] });
    }
    // (`ISL_MAZE=1`: a T-maze; starts naked or from a checkpoint)
    if env("ISL_MAZE", 0u8) == 1 {
        islands.push(Island { name: "maze".into(), frontier: false, stages: vec![Stage { name: "all".into(), probes: vec![10], lesson: plain(), goal: vec![], hold: 0 }] });
    }
    // (`ISL_SCREEN=1`: a screen with a cursor; starts naked or from a checkpoint)
    if env("ISL_SCREEN", 0u8) == 1 {
        islands.push(Island { name: "screen".into(), frontier: false, stages: vec![Stage { name: "all".into(), probes: vec![11], lesson: plain(), goal: vec![], hold: 0 }] });
    }
    // (`ISL_SHAPES=1`: shapes, telling two distorted lights apart; starts naked or from a checkpoint)
    if env("ISL_SHAPES", 0u8) == 1 {
        let lesson = Lesson { swaps: 0, blur: 2, shift: 16, ..Lesson::default() };
        islands.push(Island { name: "shapes".into(), frontier: false, stages: vec![Stage { name: "all".into(), probes: vec![0, 1, 2], lesson, goal: vec![], hold: 0 }] });
    }
    // (`ISL_SELECTION=1`: task selection, one lifetime of blocks alternating two tasks; starts
    // naked or from a checkpoint)
    if env("ISL_SELECTION", 0u8) == 1 {
        let lesson = Lesson { tokens: 2, mixed_cue: true, ..Lesson::default() };
        islands.push(Island { name: "selection".into(), frontier: false, stages: vec![Stage { name: "all".into(), probes: vec![13], lesson, goal: vec![], hold: 0 }] });
    }
    // (`ISL_LADDERS=1`: the default ladders; `ISL_LADDER_<island>`: one's own)
    let ladders = env("ISL_LADDERS", 0u8) == 1;
    for isl in islands.iter_mut() {
        if let Ok(text) = std::env::var(format!("ISL_LADDER_{}", isl.name)) {
            isl.stages = ladder(&text);
        } else if ladders {
            match isl.name.as_str() {
                "reading" => isl.stages = ladder(READING_LADDER),
                "switching" => isl.stages = ladder(SWITCHING_LADDER),
                "recall" => isl.stages = ladder(if env("ISL_RECALL_LONG", 0u8) == 1 && env("ISL_RECALL_FAIR", 0u8) == 1 { RECALL_FAIR_LADDER } else if env("ISL_RECALL_LONG", 0u8) == 1 { RECALL_LONG_LADDER } else { RECALL_LADDER }),
                "prediction" => isl.stages = ladder(PREDICTION_LADDER),
                "shapes" => isl.stages = ladder(SHAPES_LADDER),
                "maze" => isl.stages = ladder(MAZE_LADDER),
                "screen" => isl.stages = ladder(SCREEN_LADDER),
                "selection" => isl.stages = ladder(&if env("ISL_SELECT_STEP", 0u8) == 1 { format!("{SELECT_STEP} {SELECTION_LADDER}") } else { SELECTION_LADDER.to_string() }),
                _ => {}
            }
        }
    }
    // (`ISL_FEEL=1`: a feel rung before each repeat rung, switching's, shapes' and the maze's:
    // strokes at random, click after one; the rung it is copied from, with feel for repeat)
    if env("ISL_FEEL", 0u8) == 1 {
        let (feel, repeat, maze) = (probe_named("feel"), probe_named("repeat"), probe_named("maze"));
        for isl in islands.iter_mut() {
            if let Some(k) = isl.stages.iter().position(|s| s.name == "repeat" || s.name == "learn") {
                let mut st = isl.stages[k].clone();
                st.name = "feel".into();
                st.probes = st.probes.iter().copied().filter(|&p| p != repeat && p != maze).chain([feel]).collect();
                st.goal = vec![(feel, env("ISL_FEEL_GOAL", 300))];
                isl.stages.insert(k, st);
            }
        }
    }
    // (`ISL_MIRROR=1`: reading's easier rungs before echo2: the same lesson with each token heard
    // as it is said, then less and less like it, so echoing starts as copying the ear into the
    // voice and moves toward learning the sound of each token)
    if env("ISL_MIRROR", 0u8) == 1 {
        for isl in islands.iter_mut() {
            if let Some(k) = isl.stages.iter().position(|s| s.name == "echo2") {
                let base = isl.stages[k].clone();
                for (j, (name, m, quiet)) in [("echo2q", 8usize, true), ("echo2m8", 8, false), ("echo2m4", 4, false)].into_iter().enumerate() {
                    let mut st = base.clone();
                    st.name = name.into();
                    st.lesson.mirror = m;
                    st.lesson.quiet = quiet;
                    isl.stages.insert(k + j, st);
                }
            }
        }
    }
    // (`ISL_PAIRS=1`: reading's paired-associate rungs after echo2q (or before echo2): glyph to
    // token pairs drawn afresh each lifetime, 8 and then 32 of them, the teacher on six tokens in
    // eight after the first quarter, each token heard like its voice code and one's own voice not
    // heard (so a wire from the ear echoes the teacher exactly), goal copy 100: where a reader under
    // the three-factor rule learns 128 pairs by hand, `examples/pairscheck`)
    if env("ISL_PAIRS", 0u8) == 1 {
        let probe = |n: &str| neurocomp::world::observe::PROBES.iter().position(|p| *p == n).unwrap();
        let (echo, copy) = (probe("echo"), probe("copy"));
        for isl in islands.iter_mut() {
            let at = isl.stages.iter().position(|s| s.name == "echo2q").map(|k| k + 1).or_else(|| isl.stages.iter().position(|s| s.name == "echo2"));
            if let Some(k) = at {
                let base = isl.stages[k.min(isl.stages.len() - 1)].clone();
                for (j, n) in [8usize, 32].into_iter().enumerate() {
                    let mut st = base.clone();
                    st.name = format!("pairs{n}");
                    st.lesson.tokens = n;
                    st.lesson.teach = 6;
                    st.lesson.babble = trials / 4;
                    st.lesson.mirror = 8;
                    st.lesson.quiet = true;
                    st.probes = vec![echo, copy];
                    st.goal = vec![(copy, 100)];
                    isl.stages.insert(k + j, st);
                }
            }
        }
    }
    // (`ISL_STEPS=rung=goal+goal,...`: steps up to a rung: copies of it put before it, the same
    // world and probes, with lower goals, so a rung no run passes is climbed in parts)
    if let Ok(spec) = std::env::var("ISL_STEPS") {
        for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (name, goals) = part.split_once('=').expect("ISL_STEPS: rung=goal+goal");
            let goals: Vec<i32> = goals.split('+').map(|g| g.parse().expect("ISL_STEPS: a goal")).collect();
            for isl in islands.iter_mut() {
                if let Some(k) = isl.stages.iter().position(|s| s.name == name) {
                    let base = isl.stages[k].clone();
                    for (j, &g) in goals.iter().enumerate() {
                        let mut st = base.clone();
                        st.name = format!("{name}-{g}");
                        st.goal = st.goal.iter().map(|&(p, _)| (p, g)).collect();
                        isl.stages.insert(k + j, st);
                    }
                }
            }
        }
    }
    // (`ISL_ONLY=a,b`: only these islands, for a short test of one ladder)
    if let Ok(only) = std::env::var("ISL_ONLY") {
        let keep: Vec<&str> = only.split(',').map(str::trim).filter(|x| !x.is_empty()).collect();
        islands.retain(|i| keep.contains(&i.name.as_str()));
        assert!(!islands.is_empty(), "ISL_ONLY names no island");
    }
    let tokens = (0..4).map(|i| Token::Channel { tag: (0x11u8).wrapping_mul(i as u8 + 1) ^ 0x5A, width: 3, hold: false }).collect();
    let naked = TagGenome { tokens, next_id: 1 };
    // starting populations: naked, or a checkpoint's (by island name)
    let mut start: Vec<Vec<TagGenome>> = vec![vec![naked.clone()]; islands.len()];
    let mut first = 0usize;
    let mut stages = vec![0usize; islands.len()];
    // (`ISL_FRONTIER_LADDERS`) the generalist's own rung on each ladder: a checkpoint's
    // `ladders a,b,...`, else the bottom of each
    let mut ladders = vec![0usize; islands.len()];
    if let Some(path) = &resume {
        let text = std::fs::read_to_string(path).expect("checkpoint");
        for part in text.split("island ").skip(1) {
            let (head, body) = part.split_once('\n').unwrap_or((part, ""));
            let mut h = head.split_whitespace();
            let name = h.next().unwrap_or("");
            let g: usize = h.nth(1).and_then(|x| x.parse().ok()).unwrap_or(0);
            // (and its stage, if it was climbing a ladder: `stage n`)
            let st: usize = h.nth(1).and_then(|x| x.parse().ok()).unwrap_or(0);
            if let (Some("ladders"), Some(v)) = (h.next(), h.next()) {
                ladders = v.split(',').map(|x| x.parse().unwrap_or(0)).collect();
            }
            if let Some(i) = islands.iter().position(|isl| isl.name == name) {
                start[i] = TagGenome::from_text(body).expect("checkpoint genomes");
                first = first.max(g + 1);
                stages[i] = st.min(islands[i].stages.len() - 1);
                println!("resumed {name} from {path}: {} genomes, generation {g}, stage {} ({})", start[i].len(), stages[i], islands[i].stages[stages[i]].name);
            }
        }
    }
    println!("islands: {} of {pop}, elite {elite}, {lives} worlds, {held_n} held out, {trials} trials, gaps 0-{gap}, exchange every {every} ({plasmids} plasmids an island), thrift {thrift}, reads counted {}, channels held in place {}, carrying cost {} a token, plasmids of at most {} populations (0: any), exclusion {}, op choice {}, ration {} a step (0: no energy), food {} for 1024 points scored, fresh training worlds {}, epigenetic marks {}, exploration {}{}",
        islands.iter().map(|i| i.name.as_str()).collect::<Vec<_>>().join(", "), neurocomp::program::neurocomp::count_reads(), neurocomp::program::neurocomp::in_place(), neurocomp::evolve::carry(), env("ISL_PLASMID_MAX", 0usize), env("ISL_EXCLUDE", 0u8) == 1, neurocomp::kernel::class::op_choice(), neurocomp::evolve::energy().0, neurocomp::evolve::energy().1, env("ISL_FRESH", 0u8) == 1, neurocomp::evolve::epigenetics(), neurocomp::program::neurocomp::explore(), format!("{}{}{}", if neurocomp::evolve::tags::zeros() { ", expecting zeros true" } else { "" }, if neurocomp::evolve::tags::accident() { ", accidents true" } else { "" }, if neurocomp::evolve::tags::port_gates() { ", port gates true" } else { "" }) + if neurocomp::program::neurocomp::shadow() { ", shadow explorers true" } else { "" } + if neurocomp::evolve::tags::pause_draws() { ", pauses drawn true" } else { "" } + if neurocomp::evolve::tags::shadow_hints() { ", shadow hints true" } else { "" } + if neurocomp::evolve::tags::hint_use() { ", edges on hints true" } else { "" } + if neurocomp::evolve::tags::read_outputs() { ", outputs read true" } else { "" } + if neurocomp::evolve::tags::hint_band() { ", hint bands true" } else { "" } + if neurocomp::evolve::tags::self_writes() { ", self writes true" } else { "" } + if neurocomp::evolve::tags::close_loops() { ", loops closed true" } else { "" } + if neurocomp::evolve::tags::heat_inputs() { ", heat inputs true" } else { "" } + if neurocomp::evolve::carry_energy() { ", carrying paid as energy true" } else { "" } + if neurocomp::evolve::elite_pass() { ", goals judged on the elite true" } else { "" } + if neurocomp::evolve::partial() { ", partial credit true" } else { "" } + if neurocomp::evolve::partial_remember() { ", partial credit on remember true" } else { "" } + if neurocomp::evolve::partial_select() { ", partial credit on select true" } else { "" } + if neurocomp::evolve::fair_sequences() { ", fair prediction streams true" } else { "" } + if neurocomp::evolve::varied_orders() { ", order cycles of varied shape true" } else { "" } + if neurocomp::world::observe::fair_said() { ", echo and copy per token shown true" } else { "" } + if neurocomp::world::observe::fair_win_stay() { ", win-stay against the other light true" } else { "" } + if neurocomp::program::neurocomp::clock_drawn() { ", the clock drawn true" } else { "" } + if neurocomp::evolve::fresh_held() { ", held-out worlds fresh at every check true" } else { "" } + if neurocomp::evolve::one_word_voice() { ", the voice one-word everywhere true" } else { "" } + if neurocomp::evolve::migrate() { ", the generalist's passes migrate true" } else { "" } + &if neurocomp::evolve::migrate_up() > 0 { format!(", the islands' best {} into the generalist every {} (room {})", neurocomp::evolve::migrate_up_n(), neurocomp::evolve::migrate_up(), neurocomp::evolve::generalist_room()) } else { String::new() } + if neurocomp::evolve::maze_mirror() { ", the maze cue mirrored per lifetime true" } else { "" } + if neurocomp::evolve::same_word_lights() { ", the shapes' lights in one word true" } else { "" } + if neurocomp::evolve::food_from_world() { ", food from the world's strokes true" } else { "" } + if neurocomp::evolve::keep_rungs() { ", passed rungs kept true" } else { "" } + if neurocomp::evolve::tags::inhibit_inputs() { ", inhibit inputs true" } else { "" } + if neurocomp::evolve::tags::overlap_genes() { ", overlap genes true" } else { "" } + if neurocomp::evolve::size_classes() { ", size classes true" } else { "" } + &if neurocomp::evolve::thrift_relax() > 0 { format!(", thrift relaxed while stuck {}", neurocomp::evolve::thrift_relax()) } else { String::new() } + &if neurocomp::evolve::wire_cost() > 0 { format!(", wiring cost {} a bit of tag distance", neurocomp::evolve::wire_cost()) } else { String::new() } + if neurocomp::program::neurocomp::write_compete() { ", populations compete for their destination true" } else { "" } + if neurocomp::program::neurocomp::prune() { ", pruning true" } else { "" } + if neurocomp::evolve::tags::branches() { ", branches true" } else { "" } + if neurocomp::evolve::muscle() { ", the muscle's response true" } else { "" } + if neurocomp::evolve::tags::birth_lags() { ", birth lags true" } else { "" } + if neurocomp::evolve::sentences() { ", sentences true" } else { "" } + if neurocomp::evolve::body() != 0 { ", a fixed body true" } else { "" } + if neurocomp::evolve::tags::eligibility() { ", the three-factor output rule true" } else { "" } + if neurocomp::evolve::tags::rule_tables() { ", rule tables true" } else { "" } + if neurocomp::evolve::tags::triggers() { ", triggers true" } else { "" } + if neurocomp::evolve::tags::motions() { ", motions true" } else { "" } + if neurocomp::evolve::frontier_passed() { ", the generalist on passed rungs too true" } else { "" } + &if neurocomp::evolve::one_word() > 0 { format!(", one-word codes of {} lines", neurocomp::evolve::one_word()) } else { String::new() });
    let names: String = PROBES.iter().map(|p| format!("{:>7}", &p[..p.len().min(6)])).collect();
    for isl in islands.iter().filter(|i| i.stages.len() > 1) {
        println!("{} climbs: {}", isl.name, isl.stages.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(" > "));
    }
    let kids: usize = env("GC_KIDS", 0);
    if kids == 0 {
        println!("GC_KIDS=0: nothing to do");
        return;
    }
    use neurocomp::evolve::Evolvable;
    use rand::SeedableRng;
    let nw: u64 = env("GC_WORLDS", 24);
    let nw2: u64 = env("GC_WORLDS2", 24);
    let breed: usize = env("GC_BREED", 0);
    let walks: usize = env("GC_WALKS", 0);
    let walk_len: usize = env("GC_WALK_LEN", 3);
    let top: usize = env("GC_TOP", 3);
    let life = env("GC_LIFE", 0u8) == 1;
    let gseed: u64 = env("GC_SEED", 1);
    let out_path = std::env::var("GC_OUT").unwrap_or_else(|_| "gradcheck.tsv".into());
    let mut out = std::fs::File::create(&out_path).expect("GC_OUT");
    writeln!(out, "island\trung\tset\tg\tkind\tsame\tworld\tprobe\tvalue\tgraded\twork").unwrap();
    let w1: Vec<u64> = (70_000..70_000 + nw).collect();
    let w2: Vec<u64> = (80_000..80_000 + nw2).collect();
    fn dump(out: &mut std::fs::File, isl: &str, rung: &str, set: &str, kinds: &[(String, bool)], scored: &[neurocomp::evolve::Probed<TagGenome>], worlds: &[u64], probes: &[usize]) {
        for (g, s) in scored.iter().enumerate() {
            for (k, r) in s.probes.iter().enumerate() {
                for &p in probes {
                    writeln!(out, "{isl}\t{rung}\t{set}\t{g}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", kinds[g].0, kinds[g].1 as u8, worlds[k], PROBES[p], r[p], r[neurocomp::evolve::graded(p)], s.works[k]).unwrap();
                }
            }
        }
    }
    for (i, isl) in islands.iter().enumerate() {
        if isl.frontier || start[i].is_empty() {
            continue;
        }
        let st = &isl.stages[stages[i]];
        let Some(&(goal, _)) = st.goal.first() else { continue };
        let gp = neurocomp::evolve::graded(goal);
        // (`GC_PICK=n`: the parent is the one of the first `ISL_ELITE` genomes with the highest
        // graded goal on n other fresh worlds (90000...), not the first-ranked; 0: the first)
        let pick: u64 = env("GC_PICK", 0);
        let pi = if pick > 0 {
            let e = elite.max(1).min(start[i].len());
            let pw: Vec<u64> = (90_000..90_000 + pick).collect();
            let s = probe_all_in(start[i][..e].to_vec(), trials, Physics::Varied, &pw, gap, &st.probes, st.lesson.clone());
            let m: Vec<f64> = s.iter().map(|x| x.probes.iter().map(|r| r[gp] as f64).sum::<f64>() / pick as f64).collect();
            println!("{:<11} elite on {pick} worlds, graded goal: {}", isl.name, m.iter().map(|v| format!("{v:.0}")).collect::<Vec<_>>().join(" "));
            (0..e).max_by(|&a, &b| m[a].partial_cmp(&m[b]).unwrap()).unwrap()
        } else {
            0
        };
        let parent = start[i][pi].clone();
        let ptext = parent.to_text();
        let mut rng = StdRng::seed_from_u64(gseed ^ (i as u64).wrapping_mul(0x9E37_79B9));
        let mut genomes = vec![parent.clone()];
        let mut kinds = vec![("parent".to_string(), true)];
        let mut push = |g: TagGenome, kind: String, genomes: &mut Vec<TagGenome>, kinds: &mut Vec<(String, bool)>| {
            kinds.push((kind, g.to_text() == ptext));
            genomes.push(g);
        };
        for _ in 0..kids {
            let c = parent.mutate(&mut rng);
            push(c, "mut1".into(), &mut genomes, &mut kinds);
        }
        let others = elite.max(2).min(start[i].len());
        for _ in 0..breed {
            let mut c = parent.clone();
            let mut kind = "breed".to_string();
            if others > 1 && rng.gen_range(0..4) == 0 {
                let mut o = rng.gen_range(0..others);
                if o == pi { o = (o + 1) % others }
                c = c.crossover(&start[i][o], &mut rng);
                kind = "breedx".into();
            }
            loop {
                c = c.mutate(&mut rng);
                if rng.gen_bool(0.5) {
                    break;
                }
            }
            push(c, kind, &mut genomes, &mut kinds);
        }
        for w in 0..walks {
            let mut c = parent.clone();
            for s in 1..=walk_len {
                c = c.mutate(&mut rng);
                push(c.clone(), format!("walk{w}.{s}"), &mut genomes, &mut kinds);
            }
        }
        let t0 = std::time::Instant::now();
        let scored = probe_all_in(genomes.clone(), trials, Physics::Varied, &w1, gap, &st.probes, st.lesson.clone());
        dump(&mut out, &isl.name, &st.name, "base", &kinds, &scored, &w1, &st.probes);
        let mean = |s: &neurocomp::evolve::Probed<TagGenome>| s.probes.iter().map(|r| r[gp] as f64).sum::<f64>() / s.probes.len().max(1) as f64;
        let pm = mean(&scored[0]);
        // the children (not walks) with the highest mean gain on the graded goal, scored again
        let mut idx: Vec<usize> = (1..1 + kids + breed).collect();
        idx.sort_by(|&a, &b| mean(&scored[b]).partial_cmp(&mean(&scored[a])).unwrap());
        let mut walk_best: Vec<usize> = (1 + kids + breed..genomes.len()).collect();
        walk_best.sort_by(|&a, &b| mean(&scored[b]).partial_cmp(&mean(&scored[a])).unwrap());
        let again: Vec<usize> = std::iter::once(0).chain(idx.iter().take(top).copied()).chain(walk_best.iter().take(top.min(walk_best.len())).copied()).collect();
        let s2 = probe_all_in(again.iter().map(|&g| genomes[g].clone()).collect(), trials, Physics::Varied, &w2, gap, &st.probes, st.lesson.clone());
        let k2: Vec<(String, bool)> = again.iter().map(|&g| (format!("{}#{g}", kinds[g].0), kinds[g].1)).collect();
        dump(&mut out, &isl.name, &st.name, "rescore", &k2, &s2, &w2, &st.probes);
        if life {
            for (name, t) in [("half", trials / 2), ("twice", trials * 2)] {
                let s = probe_all_in(vec![parent.clone()], t, Physics::Varied, &w1, gap, &st.probes, st.lesson.clone());
                dump(&mut out, &isl.name, &st.name, &format!("life-{name}"), &[("parent".into(), true)], &s, &w1, &st.probes);
            }
        }
        println!("{:<11} parent #{pi} {:<12} {} genomes in {:.0}s: parent {} (graded {}) {pm:.1}; best child {:.1}", isl.name, st.name, genomes.len(), t0.elapsed().as_secs_f64(), PROBES[goal], PROBES[gp], mean(&scored[idx[0]]));
    }
}
