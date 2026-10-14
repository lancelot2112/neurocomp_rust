//! Does a network's cost follow what it reads, or the width of its vectors?
//!
//! The context circuit (tag grammar, experiment 06) with its widest channel Y set to 2^w words,
//! w from 3 (512 bits) to 11 (131,072 bits). Three cases per width:
//! - `fixed`: the conjunction population Z reads a 64-word field of Y, as in the seed;
//! - `whole`: Z's field is all of Y;
//! - `sparse`: as `whole`, with Z's kernels sparse (each visits only the words its lines are in).
//!
//! Each row: milliseconds per lifetime (reversal, `WIDE_TRIALS` trials), steps per second, and the
//! held-out discrimination (mean over `WIDE_LIVES` lifetimes) as a check that it still runs.
//!
//! `WIDE_TRIALS`, `WIDE_LIVES`, `WIDE_MAX` (widest w).

use std::time::Instant;

use neurocomp::evolve::tags::{TagGenome, Token};
use neurocomp::evolve::{live, Physics, Task};
use neurocomp::world::observe;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

const Y: u8 = 0x33;
const Z: u8 = 0x55;

fn genome(width: u8, whole: bool, sparse: bool) -> TagGenome {
    let mut g = TagGenome::context_seed();
    for t in g.tokens.iter_mut() {
        match t {
            Token::Channel { tag, width: w, .. } if *tag == Y => *w = width,
            Token::Pop(p) if p.from == Y && p.to == Z && whole => { p.field = width; p.sparse = sparse }
            _ => {}
        }
    }
    g
}

fn main() {
    let trials: usize = env("WIDE_TRIALS", 60);
    let lives: u64 = env("WIDE_LIVES", 4);
    let max: u8 = env("WIDE_MAX", 11);
    println!("{:>6} {:>8} {:>6} {:>10} {:>10} {:>8}", "w", "bits", "field", "ms/life", "steps/s", "held");
    for (whole, sparse) in [(false, false), (true, false), (true, true)] {
        for w in 3..=max {
            let g = genome(w, whole, sparse);
            let task = Task::Reversal { trials, period: (trials / 4).max(1), worst_phase: false };
            let t0 = Instant::now();
            let (mut steps, mut score) = (0usize, 0i64);
            for s in 0..lives {
                let ev = live(&g, task, Physics::Varied, 1000 + s);
                steps += ev.len();
                score += observe::responding(&ev).discrimination() as i64;
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            println!(
                "{:>6} {:>8} {:>6} {:>10.1} {:>10.0} {:>8}",
                w,
                64usize << w,
                if sparse { "sparse" } else if whole { "whole" } else { "fixed" },
                ms / lives as f64,
                steps as f64 / (ms / 1000.0),
                score / lives as i64
            );
        }
    }
}
