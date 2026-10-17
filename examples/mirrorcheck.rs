//! The echo of a hand-made genome (a file of `TagGenome` text, e.g. `runs/echo-wires/`) in reading
//! lifetimes of two tokens, all taught, at each `Lesson::mirror` and `quiet`, over 24 worlds:
//! what the easier reading rungs (`ISL_MIRROR`) ask for.
//! `cargo run --release --example mirrorcheck -- runs/echo-wires/wire.txt`
use neurocomp::evolve::tags::TagGenome;
use neurocomp::evolve::{live_lesson, Lesson, Physics, Task};
use neurocomp::world::observe;

fn main() {
    neurocomp::evolve::tags::set_read_outputs(true);
    // wires from the ear to the voice (and what the voice said a step ago taken away)
    let g = TagGenome::from_text(&std::fs::read_to_string(std::env::args().nth(1).unwrap()).unwrap()).unwrap().remove(0);
    for (m, quiet) in [(8usize, true), (6, true), (4, true), (8, false), (6, false), (4, false), (0, false)] {
        let mut e = 0i64;
        let mut b = 0i64;
        for seed in 0..24u64 {
            let (ev, _) = live_lesson(&g, Task::Reading(200), Physics::Varied, seed, 0, Lesson { tokens: 2, teach: 8, mirror: m, quiet, ..Lesson::default() });
            e += observe::echo(&ev) as i64;
            b += observe::babble(&ev) as i64;
        }
        println!("mirror {m} quiet {quiet}: echo {} babble {}", e / 24, b / 24);
    }
}
