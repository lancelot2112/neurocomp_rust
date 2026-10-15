//! Islands: populations pressed on different capacities, trading circuits.
//!
//! Each island is the tag genome from a naked start, evolving by epsilon-lexicase on its own
//! probes (`observe::probes`):
//! - switching: respond, tell, discriminate, reverse, win-stay (serial reversal);
//! - recall: respond, remember (a light gone a few steps ago);
//! - reading: copy (say the token shown on the eye a few steps before) and echo (say the token
//!   the teacher said a few steps before: the stepping stone, sound to voice with no glyph);
//! - generalist: all ten (surprise on a cycle of 3 broken one time in 8);
//! - prediction (`ISL_PREDICTION=1`): surprise, clicking after a token that broke the stream's
//!   cycle;
//! - maze (`ISL_MAZE=1`): a T-maze, turning at the junction to the side of the food the cue shows;
//! - screen (`ISL_SCREEN=1`): move a cursor over a grid of cells to the target glyph and click,
//!   later round walls (a 2-D maze);
//! - shapes (`ISL_SHAPES=1`): respond, tell and discriminate with the lights distorted into
//!   shapes, a fresh exemplar every trial.
//! Every `ISL_EVERY` generations each island takes `ISL_PLASMIDS` plasmids (one population of a
//! donor with the circuit of any channel it touches, made private; or the population alone if it
//! only reads and writes ports) from the best genomes of other islands into its new children, so a circuit grown under one pressure is tried under another. Each plasmid is
//! recorded (from, to, the circuit) in `ISL_LOG`; at each exchange every island's best genome is
//! scored on every probe on held-out worlds (does a circuit carry over?).
//!
//! Each plasmid is numbered (`#n`): its populations' ids carry the number, so its carriers can be
//! counted every generation until none is left (mutation and one-point crossover keep ids; a
//! copied segment or a circuit taken within an island by crossover gets new ones, so spread inside
//! an island is undercounted). At the exchange each is measured (populations, channels, kernels,
//! wires, what it reads and writes, what could hold state) and classified: its role from its
//! wiring (reflex, sensor, driver, babbler, relay, sealed), what it scores alone against its blind
//! control (`ISL_PLASMID_WORLDS` held-out worlds, 16), and the recipient's stage probes with it and
//! without it (helps, hurts, mixed: gains on some probes and losses on others, or neutral: a
//! change of `NOTICE`).
//!
//! As bacteria keep plasmids: `ISL_PLASMID_MAX` (0: any size) cuts a circuit of more populations
//! to the chosen one and its nearest neighbours in it (a small mobile unit, not a busy channel's
//! tangle); `ISL_EXCLUDE=1` sends nothing to a recipient that already carries every population
//! of it (one plasmid of a kind); `ISL_CARRY` (0) charges that much work per step for each token
//! a genome carries, so what does nothing costs something.
//!
//! `ISL_OPS=1`: the op choice. A population's op is drawn again by a mutation of its own (one in
//! sixteen), not only as one of its many properties, and ops are drawn without `rotr` (the same as
//! `rotl` by 64 - n), so `clear`, `xor` and the rest are tried far more often.
//!
//! Learning from reward has a rung of its own (`repeat`), below switching's and shapes'
//! win-stay and maze's turn: a maze with no stem, the food on one side all lifetime and no cue
//! (ladder keys `fixed` and `nocue`), so which turn pays is learned from the stroke alone; the
//! `repeat` probe scores a learner high and a turner that ignores strokes 0. Win-stay, with the
//! probe fixed, averaged about 0 in run 9: no genome learned from reward.
//!
//! `ISL_EPI=1`: epigenetic marks. Each population's lifetimes count its predictions that came
//! true; its mark becomes half the old one plus 16 for each doubling, children inherit it, and
//! mutation duplicates a marked population more often and deletes it less often.
//! `ISL_EXPLORE=1` (with `ISL_EPI`): a population whose kernels all go 64 steps without firing
//! grows, once a lifetime, a trial connection (an XOR population reading its destination and
//! writing back into its source, or the destination); one whose predictions come true 16 times
//! or more over a generation's lifetimes is added to the genome, tagged `explored`.
//!
//! `ISL_FRONTIER=1`: the generalist is scored at the frontier, on each other island's current
//! stage (its probes, in its world; each probe from the first island that has it), not on all
//! twelve probes in the plain world. In the plain world most probes are the top of another
//! island's ladder and score about 0 for every genome (run 8's best: babble 845, tell 293, the
//! other nine 0-13), so lexicase found a gradient only in babble and tell.
//!
//! `ISL_FRESH=1`: the training worlds are new every generation (the same for every genome in
//! it). With fixed training worlds a population can fit them: run 6's maze elite scored 476-658
//! on its 12 training worlds and -69 to -177 on fresh ones, having wired the hand's codes those
//! worlds drew instead of learning them each lifetime.
//!
//! `ISL_RATION` (0: none): energy. Each lifetime step a network gets this much and pays its work;
//! it eats what it scores: every 50 steps its last block is scored by its task's probes, and each
//! 1024 points are `ISL_FOOD` (32) rations. With its store below 0 it rests (does not run or act)
//! until the ration refills it. Compute beyond the ration must be earned by scoring, so cost and
//! score have to make sense together, and a runaway genome no longer sets a generation's time.
//!
//! Every `ISL_REPORT` generations, per island: its population's means on its own probes (training
//! worlds), its best genome's probes on held-out worlds, its work per step and which ports its
//! genomes read and write.
//!
//! `ISL_POP` (16 a island), `ISL_ELITE` (4), `ISL_GENS` (300), `ISL_LIVES` (12 training worlds),
//! `ISL_HELD` (12), `ISL_TRIALS` (200), `ISL_GAP` (6), `ISL_EVERY` (20), `ISL_PLASMIDS` (2),
//! `ISL_REPORT` (10), `ISL_LOG` (plasmids.txt), `ISL_SEED` (7), `ISL_THRIFT` (1), `ISL_READS` (0; 1: the words a kernel
//! population reads count as its work, so thrift sees a wide read), `ISL_INPLACE` (0; 1: every declared channel held in
//! place, one frame, never cleared or copied: only clearing writes take lines away, and a write
//! is seen by the populations after it in the same step; no history: every lag reads it as it is
//! now, so a delay is built by the network, e.g. a `set` wire placed before the write it delays,
//! or a latch: a `set` population that runs only while an enable channel is on).
//!
//! Curricula (`ISL_LADDERS=1`): an island can climb a ladder of stages, each its own probes, its
//! own world (`Lesson`: tokens and how often the teacher speaks) and a goal; it moves to the next
//! stage once its population's training mean meets the goal for `hold` generations in a row
//! (`evolve_islands`). Switching climbs respond, tell, win-stay, discriminate (no swaps), then one
//! swap, then three; recall clicks at a light, then after one gone 1, 2, 3 (the plain task), then 4 steps
//! before the window. Reading's: babble, then echo with 2, 4 and 8 tokens
//! with the teacher on every token, then the teacher fading (6, 4, 2, 1 in 8) so the glyph must
//! carry more of the work. `ISL_LADDER_<island>` replaces an island's ladder: stages separated by
//! `;`, each `name key=value ...` with keys `probes` (probe names joined by `+`), `tokens`,
//! `teach` (eighths), `period` and `noise` (surprise's stream), `blur` and `shift` (shapes), `maze`, `gone` and `symbol` (the T-maze), `sw`, `sh`, `lit`, `cell`, `walls`, `torch` and `dark` (the screen), `swaps` (of the paying light in a reversal lifetime), `delay` (steps from the
//! delayed light to the response window), `goal` (`probe@score`, joined by `+`) and `hold` (`ISL_HOLD`, 3: generations in a row the goal is met on the training worlds before the held-out check). The cross-island table
//! at each exchange is always in the plain world, on every probe.
//!
//! Checkpoints (also every `ISL_CKPT` generations, if set): at every exchange each island's whole population is written to `ISL_CHECKPOINT`
//! (checkpoint.txt, as text); `ISL_RESUME=<file>` starts every island named in it from its saved
//! population and generation (islands not in it start naked), so a change to the world or the
//! probes can be injected into a running experiment: stop it, change, resume.

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
    let (on, selfs, largest, pops) = genomes.first().map_or((0, 0, 0, 0), |g| g.loops());
    // (explored: populations kept from exploration; marked: populations with a mark, and the mean)
    let all: Vec<&Pop> = genomes.iter().flat_map(|g| g.tokens.iter().filter_map(|t| if let Token::Pop(p) = t { Some(p) } else { None })).collect();
    let explored = all.iter().filter(|p| p.explored).count();
    let marked: Vec<u32> = all.iter().filter(|p| p.mark > 0).map(|p| p.mark as u32).collect();
    let marks = if marked.is_empty() { String::new() } else { format!("; marked {} (mean {})", marked.len(), marked.iter().sum::<u32>() / marked.len() as u32) };
    format!("ops {}; loops: {on} of {pops} wired populations on a loop ({selfs} self-loops), the largest {largest} channels; explored {explored}{marks}", if ops.is_empty() { "none".into() } else { ops.join(", ") })
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
                "period" => stage.lesson.period = v.parse().expect("period"),
                "noise" => stage.lesson.noise = v.parse().expect("noise"),
                "blur" => stage.lesson.blur = v.parse().expect("blur"),
                "shift" => stage.lesson.shift = v.parse().expect("shift"),
                "maze" => stage.lesson.maze_len = v.parse().expect("maze"),
                "gone" => stage.lesson.cue_gone = v == "1",
                "symbol" => stage.lesson.symbol = v == "1",
                "fixed" => stage.lesson.maze_fixed = v == "1",
                "nocue" => stage.lesson.maze_nocue = v == "1",
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

/// The world the generalist and the cross-island table use: the plain world, with a cycle of 3
/// broken one time in 8 for surprise (the plain stream has no pattern to break).
fn plain() -> Lesson {
    Lesson { period: 3, noise: 1, ..Lesson::default() }
}

fn main() {
    neurocomp::program::neurocomp::set_in_place(env("ISL_INPLACE", 0u8) == 1);
    neurocomp::program::neurocomp::set_count_reads(env("ISL_READS", 0u8) == 1);
    neurocomp::evolve::set_carry(env("ISL_CARRY", 0u64));
    neurocomp::kernel::class::set_op_choice(env("ISL_OPS", 0u8) == 1);
    neurocomp::evolve::set_epigenetics(env("ISL_EPI", 0u8) == 1);
    neurocomp::program::neurocomp::set_explore(env("ISL_EXPLORE", 0u8) == 1);
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
    // (`ISL_LADDERS=1`: the default ladders; `ISL_LADDER_<island>`: one's own)
    let ladders = env("ISL_LADDERS", 0u8) == 1;
    for isl in islands.iter_mut() {
        if let Ok(text) = std::env::var(format!("ISL_LADDER_{}", isl.name)) {
            isl.stages = ladder(&text);
        } else if ladders {
            match isl.name.as_str() {
                "reading" => isl.stages = ladder(READING_LADDER),
                "switching" => isl.stages = ladder(SWITCHING_LADDER),
                "recall" => isl.stages = ladder(RECALL_LADDER),
                "prediction" => isl.stages = ladder(PREDICTION_LADDER),
                "shapes" => isl.stages = ladder(SHAPES_LADDER),
                "maze" => isl.stages = ladder(MAZE_LADDER),
                "screen" => isl.stages = ladder(SCREEN_LADDER),
                _ => {}
            }
        }
    }
    let tokens = (0..4).map(|i| Token::Channel { tag: (0x11u8).wrapping_mul(i as u8 + 1) ^ 0x5A, width: 3, hold: false }).collect();
    let naked = TagGenome { tokens, next_id: 1 };
    // starting populations: naked, or a checkpoint's (by island name)
    let mut start: Vec<Vec<TagGenome>> = vec![vec![naked.clone()]; islands.len()];
    let mut first = 0usize;
    let mut stages = vec![0usize; islands.len()];
    if let Some(path) = &resume {
        let text = std::fs::read_to_string(path).expect("checkpoint");
        for part in text.split("island ").skip(1) {
            let (head, body) = part.split_once('\n').unwrap_or((part, ""));
            let mut h = head.split_whitespace();
            let name = h.next().unwrap_or("");
            let g: usize = h.nth(1).and_then(|x| x.parse().ok()).unwrap_or(0);
            // (and its stage, if it was climbing a ladder: `stage n`)
            let st: usize = h.nth(1).and_then(|x| x.parse().ok()).unwrap_or(0);
            if let Some(i) = islands.iter().position(|isl| isl.name == name) {
                start[i] = TagGenome::from_text(body).expect("checkpoint genomes");
                first = first.max(g + 1);
                stages[i] = st.min(islands[i].stages.len() - 1);
                println!("resumed {name} from {path}: {} genomes, generation {g}, stage {} ({})", start[i].len(), stages[i], islands[i].stages[stages[i]].name);
            }
        }
    }
    println!("islands: {} of {pop}, elite {elite}, {lives} worlds, {held_n} held out, {trials} trials, gaps 0-{gap}, exchange every {every} ({plasmids} plasmids an island), thrift {thrift}, reads counted {}, channels held in place {}, carrying cost {} a token, plasmids of at most {} populations (0: any), exclusion {}, op choice {}, ration {} a step (0: no energy), food {} for 1024 points scored, fresh training worlds {}, epigenetic marks {}, exploration {}",
        islands.iter().map(|i| i.name.as_str()).collect::<Vec<_>>().join(", "), neurocomp::program::neurocomp::count_reads(), neurocomp::program::neurocomp::in_place(), neurocomp::evolve::carry(), env("ISL_PLASMID_MAX", 0usize), env("ISL_EXCLUDE", 0u8) == 1, neurocomp::kernel::class::op_choice(), neurocomp::evolve::energy().0, neurocomp::evolve::energy().1, env("ISL_FRESH", 0u8) == 1, neurocomp::evolve::epigenetics(), neurocomp::program::neurocomp::explore());
    let names: String = PROBES.iter().map(|p| format!("{:>7}", &p[..p.len().min(6)])).collect();
    for isl in islands.iter().filter(|i| i.stages.len() > 1) {
        println!("{} climbs: {}", isl.name, isl.stages.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(" > "));
    }
    let mut was = stages.clone();
    // plasmids: those already in the populations (moved before a resume), then each one moved
    let plasmid_worlds: Vec<u64> = held.iter().copied().take(env("ISL_PLASMID_WORLDS", 16usize)).collect();
    let plasmid_max: usize = env("ISL_PLASMID_MAX", 0);
    let exclude = env("ISL_EXCLUDE", 0u8) == 1;
    let mut followed: Vec<Moved> = Vec::new();
    for (i, popn) in start.iter().enumerate() {
        let mut found: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
        for g in popn {
            for t in &g.tokens {
                if let Token::Pop(p) = t {
                    if let Some(n) = plasmid_of(p.id) {
                        let ids = found.entry(n).or_default();
                        if !ids.contains(&p.id) { ids.push(p.id) }
                    }
                }
            }
        }
        for (n, ids) in found {
            followed.push(Moved { n, from: None, to: i, at: first.saturating_sub(1), ids, peak: 0, alive: true, kind: "moved before the resume".into() });
        }
    }
    let number = Cell::new(followed.iter().map(|m| m.n + 1).max().unwrap_or(0));
    // (each transfer's plasmid number, recipient, child, tokens and the plasmids it came from)
    let pending: RefCell<Vec<(u32, TagGenome, TagGenome, Vec<Token>, Vec<u32>)>> = RefCell::new(Vec::new());
    // each generation's line carries the UTC time and the seconds since the last one
    let clock = || { let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) % 86400; format!("{:02}:{:02}:{:02}Z", t / 3600, t / 60 % 60, t % 60) };
    let mut last = std::time::Instant::now();
    println!("started {}", clock());
    // a plasmid: one circuit of the donor (a random declared channel and what touches it)
    let transfer = |recipient: &TagGenome, donor: &TagGenome, rng: &mut StdRng| -> (TagGenome, String) {
        let n = number.get();
        number.set(n + 1);
        if donor.populations() == 0 {
            pending.borrow_mut().push((n, recipient.clone(), recipient.clone(), vec![], vec![]));
            return (recipient.clone(), "(nothing: the donor has no population)".into());
        }
        let mut child = recipient.clone();
        let k = rng.gen_range(0..donor.populations());
        let moved = child.take_population_plasmid_upto(donor, k, plasmid_max, rng);
        // (the same population but for its id and channel tags)
        let bare = |p: &Pop| Pop { id: 0, from: 0, to: 0, gate: p.gate.map(|(_, on)| (0, on)), pause: p.pause.map(|(_, on)| (0, on)), mark: 0, explored: false, ..p.clone() };
        // exclusion: a recipient that already carries every population of it takes nothing (two
        // copies of one circuit would pile up, as a cell holds one plasmid of a kind)
        if exclude && moved.iter().all(|t| match t { Token::Pop(q) => recipient.tokens.iter().any(|r| matches!(r, Token::Pop(p) if bare(p) == bare(q))), _ => true }) {
            pending.borrow_mut().push((n, recipient.clone(), recipient.clone(), vec![], vec![]));
            return (recipient.clone(), "(excluded: the recipient already carries it)".into());
        }
        // the plasmid's populations get numbered ids; one that was itself moved before names the
        // plasmid it came from
        let at = child.tokens.len() - moved.len();
        let mut parents = Vec::new();
        for (j, t) in child.tokens[at..].iter_mut().enumerate() {
            if let Token::Pop(q) = t {
                for d in &donor.tokens {
                    if let Token::Pop(p) = d {
                        if let Some(m) = plasmid_of(p.id) {
                            if bare(p) == bare(q) && !parents.contains(&m) { parents.push(m) }
                        }
                    }
                }
                q.id = TAGGED | (n << 8) | (j.min(255) as u32);
            }
        }
        let tokens = child.tokens[at..].to_vec();
        let text = TagGenome { tokens: tokens.clone(), next_id: 0 }.describe();
        pending.borrow_mut().push((n, recipient.clone(), child.clone(), tokens, parents));
        (child, text)
    };
    evolve_islands(&islands, start, stages, first, pop, trials, Physics::Varied, &train, env("ISL_FRESH", 0u8) == 1, &held, gap, elite, gens, seed, thrift, every, plasmids, transfer, |g, r| {
        // the plasmids followed: how many of their island's genomes carry them (this generation's
        // new ones are in children, scored from the next)
        for m in followed.iter_mut().filter(|m| m.alive && m.at < g) {
            let c = r.ranked[m.to].iter().filter(|x| carries(x, &m.ids).0).count();
            m.peak = m.peak.max(c);
            if c == 0 {
                m.alive = false;
                println!("{g:>4}  plasmid #{} gone from {} after {} generations (at most {} of {pop} carried it)", m.n, islands[m.to].name, g - m.at, m.peak);
                let _ = writeln!(log, "generation {g}: #{} gone from {} after {} generations (peak {} of {pop})", m.n, islands[m.to].name, g - m.at, m.peak);
            }
        }
        // the plasmids moved this generation: measured, scored alone against their blind
        // controls, and their recipients scored with and without them
        let fresh: Vec<_> = pending.borrow_mut().drain(..).collect();
        assert_eq!(fresh.len(), r.moved.len(), "one pending plasmid a move");
        let mut moved_lines = Vec::new();
        if !fresh.is_empty() {
            let alone: Vec<TagGenome> = fresh.iter().flat_map(|f| { let a = TagGenome { tokens: f.3.clone(), next_id: 0 }; [a.blind(), a] }).collect();
            let scored = probe_all_in(alone, trials, Physics::Varied, &plasmid_worlds, gap, &ALL_PROBES, plain());
            for (k, ((n, recipient, child, tokens, parents), (from, to, what))) in fresh.into_iter().zip(&r.moved).enumerate() {
                let (from, to) = (*from, *to);
                let route = format!("#{n} {}->{}", islands[from].name, islands[to].name);
                if tokens.is_empty() {
                    moved_lines.push(format!("{route}: {}", what.trim_matches(|c| c == '(' || c == ')')));
                    let _ = writeln!(log, "generation {g}: {route}\n{what}");
                    continue;
                }
                let (made, role) = measure(&tokens);
                let blind = probe_means(&scored[2 * k].probes);
                let solo = probe_means(&scored[2 * k + 1].probes);
                let sees = above(&solo, &blind, &ALL_PROBES);
                let artefact: Vec<String> = ALL_PROBES.iter().filter(|&&p| blind[p] >= NOTICE && solo[p] - blind[p] < NOTICE).map(|&p| PROBES[p].to_string()).collect();
                let st = &islands[to].stages[r.stage[to]];
                let host = probe_all_in(vec![recipient, child], trials, Physics::Varied, &plasmid_worlds, gap, &st.probes, st.lesson);
                let (without, with) = (probe_means(&host[0].probes), probe_means(&host[1].probes));
                let judged: Vec<usize> = if st.goal.is_empty() { st.probes.clone() } else { st.goal.iter().map(|&(p, _)| p).collect() };
                // (an island scored on many probes can keep a plasmid that gains on one and loses
                // on others: lexicase takes each probe as its own case, so that is "mixed")
                let gains = judged.iter().any(|&p| with[p] - without[p] >= NOTICE);
                let losses = judged.iter().any(|&p| with[p] - without[p] <= -NOTICE);
                let verdict = match (gains, losses) { (true, true) => "mixed", (false, true) => "hurts", (true, false) => "helps", _ => "neutral" };
                let deltas: Vec<String> = judged.iter().filter(|&&p| with[p] != without[p]).map(|&p| format!("{} {:+}", PROBES[p], with[p] - without[p])).collect();
                let kind = format!(
                    "{role}; {}{}; {verdict} in {} ({})",
                    if sees.is_empty() { "senses for nothing alone".to_string() } else { format!("senses for {}", sees.join("+")) },
                    if artefact.is_empty() { String::new() } else { format!("; scores {} blind", artefact.join("+")) },
                    islands[to].name, if deltas.is_empty() { "no change".to_string() } else { deltas.join(", ") },
                );
                let pops = tokens.iter().filter(|t| matches!(t, Token::Pop(_))).count();
                moved_lines.push(format!("{route} ({pops} populations): {kind}"));
                let from_text = if parents.is_empty() { String::new() } else { format!(" (from {})", parents.iter().map(|m| format!("#{m}")).collect::<Vec<_>>().join(", ")) };
                let _ = writeln!(log, "generation {g}: {route}{from_text}\n  made of: {made}\n  alone/blind: {}\n  in {} on {}, with/without: {}\n  kind: {kind}\n{what}",
                    pairs(&solo, &blind, &ALL_PROBES), islands[to].name, st.name, pairs(&with, &without, &st.probes));
                let ids = tokens.iter().filter_map(|t| if let Token::Pop(p) = t { Some(p.id) } else { None }).collect();
                followed.push(Moved { n, from: Some(from), to, at: g, ids, peak: 0, alive: true, kind });
            }
        }
        // a checkpoint at every exchange, every `ISL_CKPT` generations, and at the end: each
        // island's population, best first
        if (every > 0 && g % every == 0) || (ckpt > 0 && g % ckpt == 0) || g + 1 == gens {
            let mut ck = String::new();
            for (i, isl) in islands.iter().enumerate() {
                ck += &format!("island {} generation {g} stage {}\n", isl.name, r.stage[i]);
                for x in &r.ranked[i] {
                    ck += &x.to_text();
                }
            }
            let _ = std::fs::write(&ck_path, ck);
        }
        // every generation, one line: each climbing island's goal probes against their goals
        let goals: Vec<String> = islands.iter().enumerate().filter(|(_, isl)| isl.stages.len() > 1).filter_map(|(i, isl)| {
            let st = &isl.stages[r.stage[i]];
            if st.goal.is_empty() {
                return None;
            }
            let g: Vec<String> = st.goal.iter().map(|&(p, x)| format!("{} {}/{x}", PROBES[p], r.means[i][p])).collect();
            Some(format!("{} {} {} (met {}/{})", isl.name, st.name, g.join(" "), r.met[i], st.hold.max(1)))
        }).collect();
        let took = last.elapsed().as_secs_f64();
        last = std::time::Instant::now();
        println!("{g:>4}  {} {took:.0}s{}", clock(), if goals.is_empty() { String::new() } else { format!("  goals: {}", goals.join("; ")) });
        // an island that held its goal on the training worlds, checked on the held-out ones
        for (i, isl) in islands.iter().enumerate() {
            if let Some(m) = &r.rechecked[i] {
                let st = &isl.stages[r.stage[i]];
                let pass = st.goal.iter().all(|&(p, x)| m[p] >= x);
                let g2: Vec<String> = st.goal.iter().map(|&(p, x)| format!("{} {}/{x}", PROBES[p], m[p])).collect();
                println!("{g:>4}  {} held its goal on training; its elite held out: {} ({})", isl.name, g2.join(" "), if pass { "moves on" } else { "stays" });
            }
        }
        // an island that met its stage's goal: it is scored on the next stage from this generation
        for (i, isl) in islands.iter().enumerate() {
            if r.stage[i] != was[i] {
                println!("{g:>4}  {} moves to stage {} ({}): goal met", isl.name, r.stage[i], isl.stages[r.stage[i]].name);
                was[i] = r.stage[i];
            }
        }
        if !r.moved.is_empty() {
            println!("{g:>4}  plasmids moved: {}", r.moved.iter().map(|(f, t, w)| format!("{}->{} ({} populations)", islands[*f].name, islands[*t].name, w.lines().filter(|l| l.contains("->")).count())).collect::<Vec<_>>().join(", "));
            for l in &moved_lines {
                println!("      {l}");
            }
            // the earlier plasmids still carried, and by how many (in the elite: the kept)
            let alive: Vec<String> = followed.iter().filter(|m| m.alive && m.at < g).map(|m| {
                let c = r.ranked[m.to].iter().filter(|x| carries(x, &m.ids).0).count();
                let e = r.ranked[m.to].iter().take(elite).filter(|x| carries(x, &m.ids).0).count();
                let whole = r.ranked[m.to].iter().filter(|x| carries(x, &m.ids).1).count();
                format!("#{} {}->{} (generation {}) {c}/{pop}, {e} in the elite, {whole} whole", m.n, m.from.map_or("?", |f| islands[f].name.as_str()), islands[m.to].name, m.at)
            }).collect();
            let gone = followed.iter().filter(|m| !m.alive).count();
            println!("      plasmids still carried: {} ({gone} gone){}", alive.len(), if alive.is_empty() { String::new() } else { format!(": {}", alive.join("; ")) });
            // every island's best on every probe, held out: does a circuit carry over?
            println!("      every island's best on every probe, held out:{}", "");
            println!("      {:<11}{names}", "");
            for (i, isl) in islands.iter().enumerate() {
                let h = probe_all_in(vec![r.ranked[i][0].clone()], trials, Physics::Varied, &held, gap, &ALL_PROBES, plain());
                println!("      {:<11}{}", isl.name, row(&probe_means(&h[0].probes), &ALL_PROBES));
            }
        }
        if g % report_every == 0 || g + 1 == gens {
            println!("{g:>4}  {:<11}{names}   work   size  ports", "");
            for (i, isl) in islands.iter().enumerate() {
                let best = &r.ranked[i][0];
                let st = &isl.stages[r.stage[i]];
                let h = probe_all_in(vec![best.clone()], trials, Physics::Varied, &held, gap, &st.probes, st.lesson);
                let (rd, wr) = best.ports();
                let at = if isl.stages.len() > 1 { format!("  [stage {} {}, {} generations]", r.stage[i], st.name, r.held[i]) } else { String::new() };
                println!("      {:<11}{}  {:>5}  {:>5}  reads {} writes {}  (population, training){at}", isl.name, row(&r.means[i], &st.probes), r.work[i], best.tokens.len(), rd.join("+"), wr.join("+"));
                println!("      {:<11}{}  (its best, held out)", "", row(&probe_means(&h[0].probes), &st.probes));
                println!("      {:<11}{}  (the elite's ops; its best's loops)", "", ops_and_loops(&r.ranked[i][..elite.min(r.ranked[i].len())]));
            }
        }
    });
    // the plasmids' fates: those still carried at the end, with what they are
    let all = followed.iter().filter(|m| m.from.is_some()).count();
    let kept: Vec<&Moved> = followed.iter().filter(|m| m.alive && m.from.is_some()).collect();
    println!("plasmids: {all} moved, {} still carried at the end", kept.len());
    for m in kept {
        println!("      #{} {}->{} (generation {}, peak {}/{pop}): {}", m.n, m.from.map_or("?", |f| islands[f].name.as_str()), islands[m.to].name, m.at, m.peak, m.kind);
    }
    println!("plasmids recorded in {log_path}");
}
