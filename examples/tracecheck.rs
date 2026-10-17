//! One lifetime of partscheck's hand-made three-part genome, step by step, to see where a glyph and
//! its sound fail to pair: each step what was shown and taught, what was said, which tokens' sounds
//! the auditory channel H holds, and how many kernels of each population fired; every `EVERY`
//! steps (and at the end) what each glyph kernel learned (the glyph its synapses lie on, the sound
//! its output writes) and each listener kernel (the sound it hears, the voice code it writes).
//! `cargo run --release --example tracecheck -- runs/echo-wires/wire.txt > trace.txt`
//! (`SEED=n`: the world, 90000 by default; `TOKENS`, `TEACH`, `BABBLE`, `TRIALS`: the lesson,
//! fade6 by default; `ONE_WORD=1`: every code in word 0, `LINES=k` lines each; `GLYPH_WIRE=1`, `LGATE=0`: see below; `LAG`, `BIRTH`, `GATE`: the listener's, 1 2 1 by default; `EVERY`: 500).
use neurocomp::evolve::tags::TagGenome;
use neurocomp::evolve::{live_lesson, set_trace, Lesson, Physics, Task};
use neurocomp::world::{World, ACTIVE};
use neurocomp::program::neurocomp::RuntimeNetwork;

fn set(line: &str, key: &str, val: &str) -> String {
    let pat = format!(" {key}=");
    match line.find(&pat) {
        Some(i) => {
            let rest = &line[i + pat.len()..];
            let end = rest.find(' ').unwrap_or(rest.len());
            format!("{}{pat}{val}{}", &line[..i], &rest[end..])
        }
        None => format!("{line} {key}={val}"),
    }
}

fn var(k: &str, d: u64) -> u64 {
    std::env::var(k).ok().and_then(|x| x.parse().ok()).unwrap_or(d)
}

/// The tokens whose code has at least half its lines in `words` (8 words, one channel).
fn tokens_in(words: &[u64], codes: &[Vec<u64>]) -> Vec<usize> {
    codes
        .iter()
        .enumerate()
        .filter(|(_, c)| c.iter().zip(words).map(|(a, b)| (a & b).count_ones() as usize).sum::<usize>() * 2 >= ACTIVE)
        .map(|(t, _)| t)
        .collect()
}

/// The token whose code overlaps `words` most (ties, the lower), with the overlap.
fn best(words: &[u64], codes: &[Vec<u64>]) -> (Option<usize>, u32) {
    let mut b = (None, 0u32);
    for (t, c) in codes.iter().enumerate() {
        let o: u32 = c.iter().zip(words).map(|(a, b)| (a & b).count_ones()).sum();
        if o > b.1 {
            b = (Some(t), o);
        }
    }
    b
}

/// A kernel's mask laid at its word in a channel of 8 words.
fn laid(mask: &[u64], at: usize, base: usize) -> Vec<u64> {
    let mut v = vec![0u64; 8];
    for (i, &m) in mask.iter().enumerate() {
        if let Some(x) = (at + i).checked_sub(base).filter(|&j| j < 8) {
            v[x] |= m;
        }
    }
    v
}

fn show(t: Option<usize>) -> String {
    t.map_or("-".into(), |x| x.to_string())
}

fn main() {
    neurocomp::evolve::tags::set_birth_lags(true);
    neurocomp::evolve::tags::set_branches(true);
    neurocomp::evolve::tags::set_port_gates(true);
    let text = std::fs::read_to_string(std::env::args().nth(1).expect("a genome file with one population")).unwrap();
    let tpl = text.lines().find(|l| l.starts_with("pop ")).expect("a pop line").to_string();
    let pop = |id: u32, kv: &[(&str, String)]| {
        let mut l = tpl.replacen("pop id=871 ", &format!("pop id={id} "), 1);
        for (k, v) in kv {
            l = set(&l, k, v);
        }
        l
    };
    let (e, h) = (74u8, 150u8);
    let (lag, birth, gate) = (var("LAG", 1), var("BIRTH", 2), var("GATE", 1));
    let babbler = pop(1, &[("from", "128".into()), ("to", "31".into()), ("wire", "0".into()), ("count", "4".into()), ("tonic", "2".into()), ("branch", e.to_string()), ("branch_at", "0".into()), ("pause", "56:0".into()), ("pause_at", "1".into()), ("gate", "-".into())]);
    let ear = pop(2, &[("from", "56".into()), ("to", h.to_string()), ("to_port", "0".into()), ("wire", "1".into()), ("op", "or".into()), ("gate", "-".into())]);
    let listener = pop(3, &[("from", h.to_string()), ("from_port", "0".into()), ("to", "31".into()), ("wire", "0".into()), ("count", "0".into()), ("grow", "1".into()), ("lag", lag.to_string()), ("birth_lag", birth.to_string()), ("gate", format!("{e}:1")), ("gate_lag", gate.to_string()), ("gate_at", "0".into())]);
    // (`LOOP=d`: the listener does not answer its own voice: its fired kernels branch into an
    // efference channel F (tag 200), wires carry it down a chain of d channels (200, 201, ...),
    // and it is paused while the last had lines on a frame ago; its voice comes back about three
    // steps after it is made)
    let looped: u8 = std::env::var("LOOP").ok().and_then(|x| x.parse().ok()).unwrap_or(0);
    let mut chain = String::new();
    let listener = if looped == 0 { listener } else {
        for k in 0..looped {
            chain += &format!("channel {} 3 0\n", 200 + k);
        }
        for k in 1..looped {
            chain += &pop(10 + k as u32, &[("from", (199 + k).to_string()), ("from_port", "0".into()), ("to", (200 + k).to_string()), ("to_port", "0".into()), ("wire", "1".into()), ("op", "or".into()), ("gate", "-".into())]);
            chain.push('\n');
        }
        let l = set(&listener, "branch", "200");
        let l = set(&l, "branch_at", "0");
        let l = set(&l, "pause", &format!("{}:0", 199 + looped));
        set(&l, "pause_at", "0")
    };
    let glyphs = pop(4, &[("from", "7".into()), ("to", h.to_string()), ("to_port", "0".into()), ("wire", "0".into()), ("count", "0".into()), ("grow", "1".into()), ("gate", "-".into()), ("repeat", "0".into())]);
    // (`GLYPH_WIRE=1`: no glyph kernels; a wire copies the eye into H one word along, so a glyph
    // in the eye's word 0 lands in H's word 1, beside its sound in word 0, inside one 128-bit tile
    // of the listener; `LGATE=0`: the listener learns whenever it fires, not only while babbling)
    let glyph_wire = std::env::var("GLYPH_WIRE").is_ok_and(|v| v == "1");
    let glyphs = if glyph_wire { pop(4, &[("from", "7".into()), ("to", h.to_string()), ("to_port", "0".into()), ("wire", "1".into()), ("op", "or".into()), ("gate", "-".into()), ("cover", "0".into()), ("at", "0".into()), ("field", "3".into()), ("place", "1".into())]) } else { glyphs };
    let listener = if std::env::var("LGATE").is_ok_and(|v| v == "0") { set(&listener, "gate", "-") } else { listener };
    // (`LQ=q`: the listener's overlap rule, kernels covering more than q quarters of each other's
    // masks hold each other down; 2, more than half, by default)
    let listener = match std::env::var("LQ").ok() { Some(q) => { neurocomp::evolve::tags::set_overlap_genes(true); set(&listener, "inhibit_quarters", &q) } None => listener };
    // (`READER=l,b,g`: a reader, kernels from the eye (the glyph `l` frames ago) into the voice,
    // each born copying the move made `b` frames ago (the network's echo of the teacher naming
    // the glyph), learning and giving birth only while the ear had lines on `g` frames ago, and
    // paused while either efference copy (the babbler's E and the listener's F, both wired into
    // channel 210, carried down 211 and 212) had lines on a frame ago: it learns from the
    // teacher's voice, never its own. Needs `LOOP`)
    let mut reader = String::new();
    if let Some(r) = std::env::var("READER").ok().filter(|_| looped > 0) {
        let v: Vec<u8> = r.split(',').map(|t| t.parse().unwrap()).collect();
        for t in [210, 211, 212] {
            reader += &format!("channel {t} 3 0\n");
        }
        for (id, from, to) in [(21u32, e, 210u8), (22, 200, 210), (23, 210, 211), (24, 211, 212)] {
            reader += &pop(id, &[("from", from.to_string()), ("from_port", "0".into()), ("to", to.to_string()), ("to_port", "0".into()), ("wire", "1".into()), ("op", "or".into()), ("gate", "-".into())]);
            reader.push('\n');
        }
        reader += &pop(25, &[("from", "7".into()), ("to", "31".into()), ("wire", "0".into()), ("count", "0".into()), ("grow", "1".into()), ("lag", v[0].to_string()), ("birth_lag", v[1].to_string()), ("gate", "56:1".into()), ("gate_lag", v[2].to_string()), ("gate_at", "1".into()), ("pause", "212:0".into()), ("pause_at", "0".into())]);
        reader.push('\n');
    }
    let g = format!("genome 5\nchannel {e} 3 0\nchannel {h} 3 0\n{chain}{babbler}\n{ear}\n{glyphs}\n{listener}\n{reader}");
    let g = TagGenome::from_text(&g).unwrap_or_else(|x| panic!("{x}")).remove(0);
    println!("{}", g.describe());
    let every = var("EVERY", 500) as usize;
    let mut step = 0usize;
    set_trace(Some(Box::new(move |w: &World, net: &RuntimeNetwork| {
        let (gl, so) = w.codes();
        let words = |v: &neurocomp::bitvec::BitVector| v.as_words().to_vec();
        let (gl, so): (Vec<Vec<u64>>, Vec<Vec<u64>>) = (gl.iter().map(words).collect(), so.iter().map(words).collect());
        let vo: Vec<Vec<u64>> = (0..w.body().voice.len()).map(|t| words(w.body().voice.command(t))).collect();
        if step == 0 {
            for (i, n) in net.nodes.iter().enumerate() {
                println!("node {i}: {} bits, {} frames", n.current().as_words().len() * 64, n.frames());
            }
            for (i, x) in net.edges.iter().enumerate() {
                println!("edge {i}: {} -> {} ({:?}, window {:?})", x.src.idx, x.dst, x.kind, x.window);
            }
            println!("step shown taught said | H holds (sound tokens) | E on | fired/kernels by edge");
        }
        step += 1;
        let ev = w.events.last().unwrap();
        // H: the auditory channel, node 2 (senses, E, H, act)
        let hh = net.nodes[2].current().as_words().to_vec();
        let fired: Vec<String> = net
            .edges
            .iter()
            .map(|x| {
                let ks: Vec<_> = x.group.classes().iter().flat_map(|c| c.kernels().iter()).collect();
                format!("{}/{}", ks.iter().filter(|k| k.age == 0).count(), ks.len())
            })
            .collect();
        println!(
            "{step:5} {:>2} {} {:>2} | {:?} | {} | {}",
            show(ev.shown),
            if ev.taught { "T" } else { "." },
            show(ev.said),
            tokens_in(&hh, &so),
            net.nodes[1].current().count_ones(),
            fired.join(" ")
        );
        if step % every == 0 {
            println!("-- step {step}: what each population's kernels learned");
            for (i, x) in net.edges.iter().enumerate() {
                let ks: Vec<_> = x.group.classes().iter().flat_map(|c| c.kernels().iter()).collect();
                if ks.is_empty() {
                    continue;
                }
                // what it reads: the eye (senses words 0-7), H (whole), or the self channel
                let (inc, inbase) = match x.src.idx {
                    0 => (&gl, 0usize),
                    2 => (&so, 0),
                    _ => (&gl, usize::MAX),
                };
                let (outc, outbase) = if x.dst == 2 { (&so, 0usize) } else { (&vo, 0) };
                let mut rows: Vec<String> = ks
                    .iter()
                    .map(|k| {
                        let i = if inbase == usize::MAX { (None, 0) } else { best(&laid(k.input_mask.as_words(), k.input_idx, inbase), inc) };
                        let o = best(&laid(k.output_mask.as_words(), k.output_idx, outbase), outc);
                        if x.src.idx == 2 {
                            // (reading H) its lines on the glyph a wire lands one word along, and
                            // its threshold: can the glyph alone fire it?
                            let gh: Vec<Vec<u64>> = gl.iter().map(|c| { let mut v = vec![0u64; 8]; v[1] = c[0]; v }).collect();
                            let m = laid(k.input_mask.as_words(), k.input_idx, 0);
                            let g = best(&m, &gh);
                            let n: u32 = m.iter().map(|w| w.count_ones()).sum();
                            return format!("{}({})+g{}({})/{}th{}>{}({})", show(i.0), i.1, show(g.0), g.1, n, k.threshold, show(o.0), o.1);
                        }
                        format!("{}({})>{}({})", show(i.0), i.1, show(o.0), o.1)
                    })
                    .collect();
                rows.sort();
                println!("  edge {i} ({} kernels, in(overlap)>out(overlap)): {}", ks.len(), rows.join(" "));
            }
        }
    })));
    let lesson = Lesson { tokens: var("TOKENS", 8) as usize, teach: var("TEACH", 6) as u32, babble: var("BABBLE", 200) as usize, one_word: var("ONE_WORD", 0) == 1, code_lines: std::env::var("LINES").ok().and_then(|x| x.parse().ok()).unwrap_or(0), ..Lesson::default() };
    let trials = var("TRIALS", 400) as usize;
    let (ev, _) = live_lesson(&g, Task::Reading(trials), Physics::Varied, var("SEED", 90_000), 6, lesson);
    let copy = neurocomp::world::observe::copy(&ev);
    let echo = neurocomp::world::observe::echo(&ev);
    println!("copy {copy} echo {echo}");
}
