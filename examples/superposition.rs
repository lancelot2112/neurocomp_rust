//! Superposition and clean-up in a Hebbian population (`Associate`): how many events can
//! one population store before recall fails, with and without presynaptic scaling and
//! centering, for random codes and for language-like codes (shared common words).
//!
//! Each event: a presynaptic pattern (3 words of 16 bits in 2,048: two common words drawn
//! from a skewed (Zipf) list of 50, one rare word from 5,000; or, for random codes, 48
//! random bits) and a random target of 32 of 2,048 cells. All events are stored, then
//! each is recalled from its full pattern; right = at least 80% of its target recovered.
//!
//! cargo run --release --example superposition

use neurocomp::bitvec::BitVector;
use neurocomp::program::modules::{Associate, Ctx, Module, Readout, Scale};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

const BITS: usize = 2048;
const CELLS: usize = 2048;
const K: usize = 32;

fn word(i: usize) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(i as u64 * 7919 + 13);
    (0..BITS).collect::<Vec<_>>().choose_multiple(&mut rng, 16).copied().collect()
}

fn run(events: usize, language: bool, scale: Scale, center: bool, seed: u64) -> usize {
    let mut rng = StdRng::seed_from_u64(seed);
    // Zipf weights over 50 common words
    let zipf: Vec<f64> = (1..=50).map(|r| 1.0 / r as f64).collect(); // float: report
    let ztotal: f64 = zipf.iter().sum(); // float: report
    let mut pick = |rng: &mut StdRng| {
        let mut u = rng.r#gen::<f64>() * ztotal; // float: report
        for (i, w) in zipf.iter().enumerate() {
            if u < *w {
                return i;
            }
            u -= w;
        }
        49
    };
    let mut data = Vec::new();
    for _ in 0..events {
        let pre: Vec<usize> = if language {
            let mut v: Vec<usize> = [pick(&mut rng), pick(&mut rng), 1000 + rng.gen_range(0..5000)].iter().flat_map(|&w| word(w)).collect();
            v.sort_unstable();
            v.dedup();
            v
        } else {
            (0..BITS).collect::<Vec<_>>().choose_multiple(&mut rng, 48).copied().collect()
        };
        let post: Vec<usize> = (0..CELLS).collect::<Vec<_>>().choose_multiple(&mut rng, K).copied().collect();
        data.push((BitVector::from_bits(&pre, BITS), BitVector::from_bits(&post, CELLS)));
    }
    let mut a = Associate::new(CELLS, &[scale], 0, Readout::TopK(K), 100_000, 0, K);
    a.set_center(center);
    let mut r = StdRng::seed_from_u64(0);
    let mut ctx = Ctx { rng: &mut r, learn: true };
    let none = BitVector::EMPTY;
    for (pre, post) in &data {
        a.tick(&[post, &none, pre], &mut ctx);
    }
    let mut right = 0;
    for (pre, post) in &data {
        a.tick(&[&none, &none, pre], &mut ctx);
        let got = a.output(0);
        let ov: usize = got.as_words().iter().zip(post.as_words()).map(|(x, y)| (x & y).count_ones() as usize).sum();
        if ov * 10 >= K * 8 {
            right += 1;
        }
    }
    right
}

fn main() {
    println!("| codes | events | counts | shift-scaled | 1/n-scaled | centered | shift + centered | 1/n + centered |");
    println!("|---|---|---|---|---|---|---|---|");
    for language in [false, true] {
        for events in [250, 500, 1000, 2000, 4000] {
            let row: Vec<String> = [(Scale::None, false), (Scale::Shift, false), (Scale::Inverse, false), (Scale::None, true), (Scale::Shift, true), (Scale::Inverse, true)]
                .iter()
                .map(|&(s, c)| format!("{}%", run(events, language, s, c, 1) * 100 / events))
                .collect();
            println!("| {} | {} | {} |", if language { "language-like" } else { "random" }, events, row.join(" | "));
        }
    }
}
