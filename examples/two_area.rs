//! Two cortical areas, the same column, different settings: does the upper one hold a context
//! across a gap and a distractor while the lower one stays driven and clean?
//!
//! - **Lower area:** layer 4 over the input, no held population (a sensory area).
//! - **Upper area:** layer 4 over the lower area's code, with a held population (an association
//!   or prefrontal area): what it did not expect is held, maintained by completion.
//! - **A trial:** a familiar scene for 3 steps; a cue (A or B) joins it for one step; a gap of
//!   `GAP` steps (the scene, or silence with `SILENT=1`); with `DISTRACT=1` a distractor C joins
//!   the scene in the middle of the gap. At the gap's end: is the upper area's held state closer
//!   to this trial's cue (as the upper area coded it at the cue step) than to the other cue? And
//!   how much of the cue is left in the lower area's code (it should be none)?
//!
//! TRIALS (400), GAP (4), DISTRACT (1), SILENT (0), SEED (0).

use neurocomp::bitvec::BitVector;
use neurocomp::program::area::Pool;
use neurocomp::program::column::Column;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

const BITS: usize = 2048;
const K: usize = 32;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn code(rng: &mut StdRng) -> BitVector {
    let all: Vec<usize> = (0..BITS).collect();
    BitVector::from_bits(&all.choose_multiple(rng, K).copied().collect::<Vec<_>>(), BITS)
}

fn or(a: &BitVector, b: &BitVector) -> BitVector {
    BitVector::from_words(a.as_words().iter().zip(b.as_words()).map(|(x, y)| x | y).collect())
}

fn overlap(a: &BitVector, b: &BitVector) -> u32 {
    a.as_words().iter().zip(b.as_words()).map(|(x, y)| (x & y).count_ones()).sum()
}

fn main() {
    let trials: usize = env("TRIALS", 400);
    let gap: usize = env("GAP", 4);
    let distract: bool = env("DISTRACT", 1) == 1;
    let silent_gap: bool = env("SILENT", 0) == 1;
    let seed: u64 = env("SEED", 0);
    let mut rng = StdRng::seed_from_u64(seed);
    let (scene, a, b, c) = (code(&mut rng), code(&mut rng), code(&mut rng), code(&mut rng));
    let silent = BitVector::new(BITS, Some(0));
    let mut lower = Column::granular(BITS, Pool::new(1, BITS, BITS, K, 1, seed ^ 1), 8192);
    let mut upper = Column::granular(BITS, Pool::new(1, BITS, BITS, K, 1, seed ^ 2), 8192);
    upper.set_recurrence(3);
    // every part of the input gets cells (as the mind's senses have): a cue joining a familiar
    // scene changes the code; COVER=0: off
    let cover = env("COVER", 1) == 1;
    lower.l4.as_mut().unwrap().coverage = cover;
    upper.l4.as_mut().unwrap().coverage = cover;
    println!("two areas: {trials} trials, gap {gap}, distractor {distract}, silent gap {silent_gap}");
    // the upper area's code at the latest cue step, per cue
    let mut cue_code: [Option<BitVector>; 2] = [None, None];
    let block = 50;
    // per block: (trials, held closer to its own cue, own-cue overlap, other-cue overlap,
    // the lower area's code: overlap with its cue-step code at the gap's end)
    let mut stats = vec![(0usize, 0usize, 0u64, 0u64, 0u64); trials / block + 1];
    let mut lower_cue = BitVector::new(BITS, Some(0));
    for t in 0..trials {
        let which = (rand::Rng::gen_range(&mut rng, 0..2)) as usize;
        let cue = if which == 0 { &a } else { &b };
        let mut step = |x: &BitVector, lower: &mut Column, upper: &mut Column, rng: &mut StdRng| -> (BitVector, BitVector) {
            let l = if x.count_ones() > 0 { lower.sense(&[(0, x)], None, true, rng) } else { lower.sense_code(&silent, None, true, rng) };
            let u = upper.sense(&[(0, &l)], None, true, rng);
            (l, u)
        };
        let mut l_scene = BitVector::new(BITS, Some(0));
        for _ in 0..3 {
            l_scene = step(&scene, &mut lower, &mut upper, &mut rng).0;
        }
        let (l, u) = step(&or(&scene, cue), &mut lower, &mut upper, &mut rng);
        // the cue's own bits in the lower area's code (what the scene alone does not have)
        lower_cue = BitVector::from_words(l.as_words().iter().zip(l_scene.as_words()).map(|(x, y)| x & !y).collect());
        // the upper area's driven code for the cue step (its layer 4's; the held part left out)
        let _ = u;
        cue_code[which] = Some(upper.l4.as_ref().unwrap().encode(&[(0, &l)]));
        let mut last_lower = BitVector::new(BITS, Some(0));
        for g in 0..gap {
            let base = if silent_gap { silent.clone() } else { scene.clone() };
            let x = if distract && g == gap / 2 { or(&base, &c) } else { base };
            let (l, _) = step(&x, &mut lower, &mut upper, &mut rng);
            last_lower = l;
        }
        if let (Some(own), Some(other)) = (cue_code[which].as_ref(), cue_code[1 - which].as_ref()) {
            let held = upper.held();
            // the cue's own part of the code (what differs from the other cue's code)
            let own_part = BitVector::from_words(own.as_words().iter().zip(other.as_words()).map(|(x, y)| x & !y).collect());
            let other_part = BitVector::from_words(other.as_words().iter().zip(own.as_words()).map(|(x, y)| x & !y).collect());
            let (o, n) = (overlap(held, &own_part), overlap(held, &other_part));
            let s = &mut stats[t / block];
            s.0 += 1;
            s.1 += (o > n) as usize;
            s.2 += o as u64;
            s.3 += n as u64;
            s.4 += overlap(&last_lower, &lower_cue) as u64;
        }
    }
    for (i, s) in stats.iter().enumerate().filter(|x| x.1 .0 > 0) {
        let n = s.0 as f64;
        println!(
            "  trials {:4}-{:4}: held closer to its own cue {:.0}% (own cue's bits held {:.1}, the other's {:.1}); the cue's bits left in the lower area's code at the gap's end {:.1}",
            i * block,
            (i + 1) * block,
            100.0 * s.1 as f64 / n,
            s.2 as f64 / n,
            s.3 as f64 / n,
            s.4 as f64 / n
        );
    }
    println!("upper area loads: {} on surprise, {} by the gate", upper.loads.0, upper.loads.1);
}
