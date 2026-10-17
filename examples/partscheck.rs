//! Three things to learn, by three parts of one hand-made genome (two tokens' sounds, glyphs and
//! voice codes drawn afresh each lifetime, the teacher silent for the first half):
//! - what my voice sounds like: a babbler (self channel to voice, spontaneous, quiet while the ear
//!   is busy) with a branch into an efference-copy channel E; a listener (auditory channel H to
//!   voice) born on what it hears, copying the move made `birth_lag` frames before, learning only
//!   while E was on (`gate_lag` frames ago): the sound of a move, mapped back to the move;
//! - what the teacher wants: the same listener, the teacher's sound arriving in H by a wire from
//!   the ear: echo;
//! - what a glyph sounds like: kernels from the eye into H, born writing what H holds (the sound
//!   heard with the glyph), so a glyph shown untaught puts its sound in H for the listener: copy.
//! Scored on 24 worlds on echo (two tokens, all taught after the babbling half) and on copy and
//! echo with eight tokens, the teacher saying six in eight (fade6), by the listener's lag, birth
//! lag and gate lag (`ONLY=l,b,g`: one; `GLYPH_REPEAT=k`: glyph births copy H's sound k frames
//! back; `ONE_WORD=1`: all codes in word 0). `cargo run --release --example partscheck -- runs/echo-wires/wire.txt`.
use neurocomp::evolve::tags::TagGenome;
use neurocomp::evolve::{probe_all_in, Lesson, Physics};
use neurocomp::world::observe::PROBES;

/// The line with `key` set to `val` (added if it is not there).
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

fn main() {
    // (`SENTENCE=1`) sentences in every lifetime (`evolve::set_sentences`)
    neurocomp::evolve::set_sentences(std::env::var("SENTENCE").is_ok_and(|v| v == "1"));
    neurocomp::evolve::tags::set_birth_lags(true);
    neurocomp::evolve::tags::set_branches(true);
    neurocomp::evolve::tags::set_port_gates(true);
    let text = std::fs::read_to_string(std::env::args().nth(1).expect("a genome file with one population")).unwrap();
    let tpl = text.lines().find(|l| l.starts_with("pop ")).expect("a pop line").to_string();
    let probe = |n: &str| PROBES.iter().position(|p| *p == n).unwrap();
    let (echo, copy) = (probe("echo"), probe("copy"));
    let worlds: Vec<u64> = (90_000..90_024).collect();
    let pop = |id: u32, kv: &[(&str, String)]| {
        let mut l = tpl.replacen("pop id=871 ", &format!("pop id={id} "), 1);
        for (k, v) in kv {
            l = set(&l, k, v);
        }
        l
    };
    let (e, h) = (74u8, 150u8);
    // (`ONE_WORD=1`: every token's glyph, sound and voice code in word 0, `World::one_word`)
    let one_word = std::env::var("ONE_WORD").is_ok_and(|v| v == "1");
    println!("listener lag, birth lag, gate lag: echo2 echo | fade6 copy, echo   (goals: echo 150, copy 100)");
    // (`GLYPH_REPEAT=k`: the glyph kernels' births copy what H held k frames ago, its busiest
    // word: the sound; 0, they write at their own place)
    // (`BABBLERS=n`: the babbler's kernels, 4 by default)
    let babblers: u8 = std::env::var("BABBLERS").ok().and_then(|x| x.parse().ok()).unwrap_or(4);
    let rep: u8 = std::env::var("GLYPH_REPEAT").ok().and_then(|x| x.parse().ok()).unwrap_or(0);
    let only: Option<(u8, u8, u8)> = std::env::var("ONLY").ok().map(|x| { let v: Vec<u8> = x.split(',').map(|t| t.parse().unwrap()).collect(); (v[0], v[1], v[2]) });
    for lag in [1u8, 2] {
        for birth in [0u8, 1, 2, 3] {
            for gate in [1u8, 2] {
                if only.is_some_and(|o| o != (lag, birth, gate)) {
                    continue;
                }
                let babbler = pop(1, &[("from", "128".into()), ("to", "31".into()), ("wire", "0".into()), ("count", babblers.to_string()), ("tonic", "2".into()), ("branch", e.to_string()), ("branch_at", "0".into()), ("pause", "56:0".into()), ("pause_at", "1".into()), ("gate", "-".into())]);
                let ear = pop(2, &[("from", "56".into()), ("to", h.to_string()), ("to_port", "0".into()), ("wire", "1".into()), ("op", "or".into()), ("gate", "-".into())]);
                let listener = pop(3, &[("from", h.to_string()), ("from_port", "0".into()), ("to", "31".into()), ("wire", "0".into()), ("count", "0".into()), ("grow", "1".into()), ("lag", lag.to_string()), ("birth_lag", birth.to_string()), ("gate", format!("{e}:1")), ("gate_lag", gate.to_string()), ("gate_at", "0".into())]);
                let glyphs = pop(4, &[("from", "7".into()), ("to", h.to_string()), ("to_port", "0".into()), ("wire", "0".into()), ("count", "0".into()), ("grow", "1".into()), ("gate", "-".into()), ("repeat", if rep > 0 { "1".into() } else { "0".into() }), ("repeat_lag", rep.max(1).to_string())]);
                let g = format!("genome 5\nchannel {e} 3 0\nchannel {h} 3 0\n{babbler}\n{ear}\n{glyphs}\n{listener}\n");
                let g = TagGenome::from_text(&g).unwrap_or_else(|x| panic!("{x}")).remove(0);
                let score = |lesson: Lesson, p: usize, trials: usize| {
                    let s = probe_all_in(vec![g.clone()], trials, Physics::Varied, &worlds, 6, &[p], lesson);
                    s[0].probes.iter().map(|r| r[p] as i64).sum::<i64>() / worlds.len() as i64
                };
                let e2 = score(Lesson { tokens: 2, teach: 8, babble: 100, one_word, ..Lesson::default() }, echo, 200);
                let f6 = Lesson { tokens: 8, teach: 6, babble: 200, one_word, ..Lesson::default() };
                let (c6, e6) = (score(f6.clone(), copy, 400), score(f6, echo, 400));
                println!("  {lag} {birth} {gate}: {e2} | {c6}, {e6}");
            }
        }
    }
}
