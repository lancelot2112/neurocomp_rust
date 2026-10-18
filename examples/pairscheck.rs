//! The paired-associate capacity curve: how many arbitrary glyph-to-token pairs a lifetime can
//! learn. Reading lifetimes of N tokens (N = 2, 8, 32, 128), each token's glyph drawn afresh every
//! lifetime, the teacher saying six tokens in eight after the first quarter (fade6), scored by
//! `copy` (saying the token whose glyph was shown, untaught, 1 to 4 steps later, minus 9 to 12 steps
//! later, per token said), at two lifetime lengths (`TRIALS`, 400, and eight times that), on 24
//! worlds (`MIRROR=m`: tokens heard like their voice codes). Under the run-30 body by default (`BODY=1`: a fixed body; `ONE_WORD=8`: every code 8
//! lines in one word; 0 turns either off).
//!
//! The learners:
//! - **lookup** (the reference): a table written in the harness, not a network. It learns each
//!   glyph's token on taught steps (as a perfect ear would) and says, on each step after an untaught
//!   one, that glyph's token, if it has one (silent after a taught step, as copy scores);
//! - **blind** (the floor): a random token each step;
//! - **networks**: the best genomes in a checkpoint (`CKPT`, runs/islands30/checkpoint.txt; the
//!   islands in `ISLANDS`, reading,generalist,recall; the first `TOP` genomes of each, 3, the best
//!   of them reported) and any genome files given as arguments.
//!
//! `cargo run --release --example pairscheck -- [genome files]`
use neurocomp::evolve::tags::TagGenome;
use neurocomp::evolve::{live_lesson, probe_all_in, Lesson, Physics, Task};
use neurocomp::world::observe::{self, PROBES};
use neurocomp::world::Event;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn var(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|x| x.parse().ok()).unwrap_or(d)
}

fn lesson(n: usize, trials: usize) -> Lesson {
    // (`MIRROR=m`: each token heard like the voice's code for it, m eighths of its lines, one's own
    // voice not heard: a wire from the ear to the voice echoes the teacher exactly)
    let m = var("MIRROR", 0);
    Lesson { tokens: n, teach: 6, babble: trials / 4, mirror: m, quiet: m > 0, ..Lesson::default() }
}

/// The world's events for a lifetime with a learner written in the harness saying `say(t)` at
/// step t (the shown and taught tokens do not depend on what the network says).
fn replay(events: &[Event], mut say: impl FnMut(usize, &[Event]) -> Option<usize>) -> Vec<Event> {
    let mut out: Vec<Event> = events.to_vec();
    for t in 0..out.len() {
        let s = say(t, events);
        out[t].said = s;
        out[t].sentence.clear();
    }
    out
}

fn main() {
    neurocomp::evolve::set_body(var("BODY", 1) as u64);
    neurocomp::evolve::set_one_word(var("ONE_WORD", 8));
    // the run-30 runtime switches its genomes were evolved under
    neurocomp::evolve::tags::set_port_gates(true);
    neurocomp::evolve::tags::set_birth_lags(true);
    neurocomp::evolve::tags::set_eligibility(true);
    neurocomp::evolve::tags::set_rule_tables(true);
    neurocomp::evolve::tags::set_triggers(true);
    neurocomp::evolve::tags::set_branches(true);
    neurocomp::evolve::tags::set_inhibit_inputs(true);
    neurocomp::evolve::tags::set_overlap_genes(true);
    neurocomp::evolve::tags::set_read_outputs(true);
    neurocomp::program::neurocomp::set_write_compete(true);
    neurocomp::program::neurocomp::set_prune(true);
    let copy = PROBES.iter().position(|p| *p == "copy").unwrap();
    // (`WORLDS=s`: the first of 24 world seeds, 70000 by default)
    let w0 = var("WORLDS", 70_000) as u64;
    let worlds: Vec<u64> = (w0..w0 + 24).collect();
    let base = var("TRIALS", 400);
    // the networks: from a checkpoint's islands, and files
    let mut nets: Vec<(String, TagGenome)> = Vec::new();
    let ckpt = std::env::var("CKPT").unwrap_or("runs/islands30/checkpoint.txt".into());
    let islands = std::env::var("ISLANDS").unwrap_or("reading,generalist,recall".into());
    if let Ok(text) = std::fs::read_to_string(&ckpt) {
        let blocks: Vec<&str> = text.split("\nisland ").collect();
        for name in islands.split(',').filter(|x| !x.is_empty()) {
            if let Some(b) = blocks.iter().find(|b| b.trim_start_matches("island ").starts_with(&format!("{name} "))) {
                let body = &b[b.find('\n').map_or(b.len(), |i| i + 1)..];
                if let Ok(gs) = TagGenome::from_text(body) {
                    for (i, g) in gs.into_iter().take(var("TOP", 3)).enumerate() {
                        nets.push((format!("{name} #{i}"), g));
                    }
                }
            }
        }
    }
    for f in std::env::args().skip(1) {
        let g = TagGenome::from_text(&std::fs::read_to_string(&f).unwrap()).unwrap().remove(0);
        nets.push((f, g));
    }
    let empty = TagGenome::from_text("genome 0\n").unwrap().remove(0);
    println!("copy (goal 100; lookup a perfect table, blind the floor), mean of {} worlds; body {}, one-word codes {}", worlds.len(), var("BODY", 1), var("ONE_WORD", 8));
    println!("{:>4} {:>6} | {:>7} {:>6} | networks", "N", "trials", "lookup", "blind");
    for n in [2usize, 8, 32, 128] {
        for trials in [base, base * 8] {
            let (mut look, mut blind) = (0i64, 0i64);
            for &w in &worlds {
                let (ev, _) = live_lesson(&empty, Task::Reading(trials), Physics::Varied, w, 6, lesson(n, trials));
                let mut table: Vec<Option<usize>> = vec![None; n];
                let l = replay(&ev, |t, e| {
                    // what it has learned by the step before, from the taught steps it saw
                    if t > 0 && e[t - 1].taught {
                        if let Some(s) = e[t - 1].shown {
                            table[s.min(n - 1)] = Some(s);
                        }
                    }
                    // (silent when it heard the teacher: copy counts only untaught tokens, and a
                    // token said on a taught step counts against it)
                    if t == 0 || e[t - 1].taught { None } else { e[t - 1].shown.and_then(|s| table[s.min(n - 1)]) }
                });
                look += observe::copy(&l) as i64;
                let mut r = StdRng::seed_from_u64(w ^ 0xB11D);
                let b = replay(&ev, |_, _| Some(r.gen_range(0..n)));
                blind += observe::copy(&b) as i64;
            }
            let k = worlds.len() as i64;
            let mut row: Vec<String> = Vec::new();
            for (name, g) in &nets {
                let s = probe_all_in(vec![g.clone()], trials, Physics::Varied, &worlds, 6, &[copy], lesson(n, trials));
                row.push(format!("{name} {}", s[0].probes.iter().map(|r| r[copy] as i64).sum::<i64>() / k));
            }
            println!("{n:>4} {trials:>6} | {:>7} {:>6} | {}", look / k, blind / k, row.join(", "));
        }
    }
}
