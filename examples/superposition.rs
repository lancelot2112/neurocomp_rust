//! Superposition and clean-up in a Hebbian population: how many events can one
//! population store before recall fails? Counts (`Associate`) with and without
//! presynaptic scaling and centering, and phase-coded weights (`PhaseAssociate`), for
//! random codes and for language-like codes (shared common words).
//!
//! Each event: an input pattern (3 words of 16 bits in 2,048: two common words drawn
//! from a skewed (Zipf) list of 50 and one rare word from 5,000; or, for random codes, 48
//! random bits) and a target of 32 of 2,048 cells (each with a random phase of 16, for
//! the phase populations). All events are stored, then each is recalled from its full
//! input; right = at least 80% of its target cells recovered (for the phase populations,
//! at the right phase, unless the column says "cells").
//!
//! cargo run --release --example superposition

use neurocomp::bitvec::BitVector;
use neurocomp::program::{EpisodicCircuit, IndexConfig, IndexMemory};
use neurocomp::program::modules::{phase_cells, phase_code, Associate, Center, Ctx, Module, PhaseAssociate, Readout, Scale};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

const BITS: usize = 2048;
const CELLS: usize = 2048;
const K: usize = 32;
const PHASES: usize = 16;

fn word(i: usize) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(i as u64 * 7919 + 13);
    (0..BITS).collect::<Vec<_>>().choose_multiple(&mut rng, 16).copied().collect()
}

#[derive(Clone, Copy, PartialEq)]
enum InPhase {
    /// Plain input bits (phase 0).
    None,
    /// Each event's input bits get random phases (the same at recall): an upper bound.
    Random,
    /// A bit's phase is set by the slot its word fills (phase binding of word to role):
    /// the same word in the same slot has the same phases in every event.
    Slot,
}

#[derive(Clone, Copy)]
enum Variant {
    Counts(Scale, Center),
    /// Phase-coded weights: 1/n row weights, the input phases, score on cells only.
    Phase(bool, InPhase, bool),
    /// One row per event, winner-take-all (`IndexMemory`); 1/n cue weights or plain overlap.
    Index(bool),
}

struct Event {
    pre: BitVector,
    /// (input bit, slot) for language-like events; slot 0 for random ones.
    slots: Vec<(usize, usize)>,
    post: Vec<(usize, usize)>,
}

fn events(n: usize, language: bool, seed: u64) -> Vec<Event> {
    let mut rng = StdRng::seed_from_u64(seed);
    let zipf: Vec<u64> = (1..=50u64).map(|r| 1_000_000 / r).collect();
    let ztotal: u64 = zipf.iter().sum();
    let pick = |rng: &mut StdRng| {
        let mut u = rng.gen_range(0..ztotal);
        for (i, &w) in zipf.iter().enumerate() {
            if u < w {
                return i;
            }
            u -= w;
        }
        49
    };
    (0..n)
        .map(|_| {
            let mut slots: Vec<(usize, usize)> = if language {
                let ws = [pick(&mut rng), pick(&mut rng), 1000 + rng.gen_range(0..5000)];
                ws.iter().enumerate().flat_map(|(s, &w)| word(w).into_iter().map(move |b| (b, s))).collect()
            } else {
                (0..BITS).collect::<Vec<_>>().choose_multiple(&mut rng, 48).map(|&b| (b, 0)).collect()
            };
            slots.sort_unstable();
            slots.dedup_by_key(|x| x.0);
            let bits: Vec<usize> = slots.iter().map(|x| x.0).collect();
            let mut post: Vec<(usize, usize)> =
                (0..CELLS).collect::<Vec<_>>().choose_multiple(&mut rng, K).map(|&j| (j, rng.gen_range(0..PHASES))).collect();
            post.sort_unstable();
            Event { pre: BitVector::from_bits(&bits, BITS), slots, post }
        })
        .collect()
}

fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn run(n: usize, language: bool, v: Variant) -> usize {
    let data = events(n, language, 1);
    let mut r = StdRng::seed_from_u64(0);
    let mut ctx = Ctx { rng: &mut r, learn: true };
    let none = BitVector::EMPTY;
    let mut right = 0;
    match v {
        Variant::Counts(scale, center) => {
            let mut a = Associate::new(CELLS, &[scale], 0, Readout::TopK(K), 100_000, 0, K);
            a.set_center(center);
            for e in &data {
                let cells: Vec<usize> = e.post.iter().map(|p| p.0).collect();
                a.tick(&[&BitVector::from_bits(&cells, CELLS), &none, &e.pre], &mut ctx);
            }
            for e in &data {
                a.tick(&[&none, &none, &e.pre], &mut ctx);
                let got = a.output(0);
                if e.post.iter().filter(|p| got.bit_get(p.0)).count() * 10 >= K * 8 {
                    right += 1;
                }
            }
        }
        Variant::Index(inverse) => {
            let mut m = IndexMemory::new(IndexConfig { inverse, cap: usize::MAX, min_overlap: 1, period: u32::MAX, ..IndexConfig::default() });
            for e in &data {
                let cells: Vec<usize> = e.post.iter().map(|p| p.0).collect();
                let keys: Vec<usize> = (0..BITS).filter(|&b| e.pre.bit_get(b)).collect();
                m.store_split(&keys, &[], &cells);
            }
            for e in &data {
                let keys: Vec<usize> = (0..BITS).filter(|&b| e.pre.bit_get(b)).collect();
                let got = m.recall(&keys).ec;
                if e.post.iter().filter(|p| got.contains(&p.0)).count() * 10 >= K * 8 {
                    right += 1;
                }
            }
        }
        Variant::Phase(inverse, input, cells_only) => {
            let mut a = PhaseAssociate::new(CELLS, K, PHASES, if input == InPhase::None { 1 } else { PHASES }, inverse);
            let mut prng = StdRng::seed_from_u64(5);
            let inputs: Vec<BitVector> = data
                .iter()
                .map(|e| match input {
                    InPhase::None => e.pre.clone(),
                    InPhase::Random => phase_code(&e.slots.iter().map(|&(b, _)| (b, prng.gen_range(0..PHASES))).collect::<Vec<_>>(), BITS, PHASES),
                    InPhase::Slot => phase_code(&e.slots.iter().map(|&(b, s)| (b, (mix((b * 8 + s) as u64 + 1) % PHASES as u64) as usize)).collect::<Vec<_>>(), BITS, PHASES),
                })
                .collect();
            for (e, x) in data.iter().zip(&inputs) {
                a.tick(&[&phase_code(&e.post, CELLS, PHASES), x], &mut ctx);
            }
            for (e, x) in data.iter().zip(&inputs) {
                a.tick(&[&none, x], &mut ctx);
                let got = phase_cells(a.output(0), PHASES);
                let hit = |p: &(usize, usize)| if cells_only { got.iter().any(|g| g.0 == p.0) } else { got.contains(p) };
                if e.post.iter().filter(|p| hit(p)).count() * 10 >= K * 8 {
                    right += 1;
                }
            }
        }
    }
    right
}

fn main() {
    let variants = [
        ("counts", Variant::Counts(Scale::None, Center::Off)),
        ("1/n", Variant::Counts(Scale::Inverse, Center::Off)),
        ("centered (homeostatic)", Variant::Counts(Scale::None, Center::Homeostatic(16))),
        ("1/n + centered (homeostatic)", Variant::Counts(Scale::Inverse, Center::Homeostatic(16))),
        ("index (overlap)", Variant::Index(false)),
        ("index (1/n)", Variant::Index(true)),
        ("phase in (random) + out, cells", Variant::Phase(false, InPhase::Random, true)),
        ("phase in (slot) + out + 1/n, cells", Variant::Phase(true, InPhase::Slot, true)),
    ];
    let names: Vec<&str> = variants.iter().map(|v| v.0).collect();
    println!("| codes | events | {} |", names.join(" | "));
    println!("|---|---|{}", "---|".repeat(names.len()));
    for language in [false, true] {
        for n in [1000, 2000, 4000, 8000, 16000] {
            let row: Vec<String> = variants.iter().map(|&(_, v)| format!("{}%", run(n, language, v) * 100 / n)).collect();
            println!("| {} | {} | {} |", if language { "language-like" } else { "random" }, n, row.join(" | "));
        }
    }
}
