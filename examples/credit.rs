//! Do we need a credit-assignment operation?
//!
//! Run:  cargo run --release --example credit
//!
//! Task (synthetic, long gap): episodes `cue  f f f ... f  ?  answer`, where the
//! cue is one of A..D, 3-6 filler symbols are drawn from 12 fillers, and the
//! answer is a fixed function of the cue (A->w, B->x, C->y, D->z). Scored on
//! predicting the answer after `?`.
//!
//! The predictor (a predictive KernelClass) sees the current and previous
//! symbol, plus one hidden frame. The hidden frame comes from K persistent
//! memory units: each unit watches for one symbol and, once it sees it, stays
//! on for 8 ticks. With K smaller than the number of symbols, something has to
//! decide which symbols are worth remembering. Policies:
//! - fixed random: units keep their initial random symbols;
//! - activity (Hebbian-like): periodically re-point the least active unit at
//!   the most frequent unwatched symbol ("fire together, keep together");
//! - credit: each unit scores how often the predictor's correct kernels used its
//!   bits (`KernelClass::credited_inputs`); periodically re-point the least
//!   credited unit at a random unwatched symbol.
//!
//! - three-factor: eligibility (unit active) x reward (prediction right) minus a
//!   running baseline; periodically re-point the unit with the lowest utility at
//!   a random unwatched symbol. Unlike "credit", a unit that is merely present
//!   whenever things go right earns nothing unless things go *better* than usual.
//! - ablation: counterfactual credit. After each prediction, re-predict with one
//!   active unit's bits removed (`KernelClass::peek`, no state change). +1 if
//!   the right answer becomes wrong without the unit, -1 if a wrong answer
//!   becomes right; re-point the lowest-scoring unit at a random unwatched symbol.
//!
//! Credit-guided growth (`KernelClass::set_growth_mask`): optionally, a newly
//! grown kernel may sample hidden bits from only ONE active unit, chosen with
//! odds credit + 1, instead of from every active unit. Kernels then depend on a
//! single unit, so ablating an incidental unit no longer flips them.
//!
//! Crossed with synapse-level credit inside the predictor (`GrowthConfig::generalize`):
//! a kernel that nearly matched and would have been right drops its silent
//! connections, so it stops requiring the particular fillers it was grown on.

mod common;

use common::Encoder;
use neurocomp::bitvec::BitVector;
use neurocomp::kernel::{GrowthConfig, KernelClass, SimpleKernel};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

const BITS: usize = 512; // per frame
const CUES: usize = 4;
const FILLERS: usize = 12;
// symbols: 0..4 cues, 4..16 fillers, 16 = '?', 17..21 answers
const QUERY: usize = CUES + FILLERS;
const N_SYMBOLS: usize = CUES + FILLERS + 1 + CUES;
const PERSIST: usize = 8;
const REVIEW_EVERY: usize = 100; // episodes between unit re-assignments
const UNIT_BITS: usize = 32;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Policy {
    Oracle, // units watch exactly the cues (upper bound on what memory can give)
    Partial(usize), // diagnostic: units fixed to this many cues, the rest fillers
    FixedRandom,
    Activity,
    Credit,
    ThreeFactor,
    Ablation,
}

struct Memory {
    watch: Vec<usize>,      // symbol each unit watches
    on_until: Vec<usize>,   // tick until which the unit is on
    out_bits: Vec<Vec<usize>>, // hidden-frame bits each unit drives
    activity: Vec<f64>,
    credit: Vec<f64>,
    age: Vec<usize>, // reviews since the unit was last re-pointed
}

impl Memory {
    fn new(k: usize, rng: &mut StdRng) -> Self {
        let mut symbols: Vec<usize> = (0..N_SYMBOLS).collect();
        symbols.shuffle(rng);
        let mut positions: Vec<usize> = (0..BITS).collect();
        positions.shuffle(rng);
        Self {
            watch: symbols[..k].to_vec(),
            on_until: vec![0; k],
            out_bits: (0..k).map(|u| positions[u * UNIT_BITS..(u + 1) * UNIT_BITS].to_vec()).collect(),
            activity: vec![0.0; k],
            credit: vec![0.0; k],
            age: vec![0; k],
        }
    }

    fn step(&mut self, t: usize, symbol: usize) -> BitVector {
        let mut frame = BitVector::new(BITS, Some(0));
        for u in 0..self.watch.len() {
            if self.watch[u] == symbol {
                self.on_until[u] = t + PERSIST;
            }
            if t < self.on_until[u] {
                self.activity[u] += 1.0;
                for &b in &self.out_bits[u] {
                    frame.bit_set(b);
                }
            }
        }
        frame
    }

    fn reassign(&mut self, u: usize, symbol: usize) {
        self.watch[u] = symbol;
        self.on_until[u] = 0;
        self.activity[u] = 0.0;
        self.credit[u] = 0.0;
        self.age[u] = 0;
    }

    fn unwatched(&self) -> Vec<usize> {
        (0..N_SYMBOLS).filter(|s| !self.watch.contains(s)).collect()
    }
}

fn episode(rng: &mut StdRng) -> (Vec<usize>, usize) {
    let cue = rng.gen_range(0..CUES);
    let mut seq = vec![cue];
    for _ in 0..rng.gen_range(3..=6) {
        seq.push(CUES + rng.gen_range(0..FILLERS));
    }
    seq.push(QUERY);
    let answer_at = seq.len();
    seq.push(QUERY + 1 + cue);
    (seq, answer_at)
}

fn run(policy: Policy, k: usize, generalize: Option<f32>, patient: bool, guided: bool, episodes: usize, seed: u64) -> (f64, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let enc = Encoder::new(N_SYMBOLS, BITS, 32, &mut rng);
    let mut mem = Memory::new(k, &mut rng);
    if policy == Policy::Oracle {
        for u in 0..k {
            mem.watch[u] = if u < CUES { u } else { QUERY };
        }
    }
    if let Policy::Partial(n) = policy {
        for u in 0..k {
            mem.watch[u] = if u < n { u } else { CUES + u };
        }
    }
    let mut class: KernelClass<SimpleKernel> = KernelClass::predictive(GrowthConfig {
        max_kernels: 20_000,
        frame_words: BITS / 64,
        max_frames: 3,
        sample_bits: 16,
        match_fraction: 0.8,
        surprise_fraction: 0.5,
        generalize,
        generalize_after: 1,
    });
    let mut symbol_counts = vec![0f64; N_SYMBOLS];
    let (mut t, mut prev) = (0usize, None::<usize>);
    let (mut hits, mut total) = (0usize, 0usize);
    let mut baseline = 0.0f64; // running average of "prediction was right"
    for e in 0..episodes {
        let (seq, answer_at) = episode(&mut rng);
        for i in 0..seq.len() - 1 {
            let s = seq[i];
            symbol_counts[s] += 1.0;
            let hidden = mem.step(t, s);
            // frames: current symbol, previous symbol, hidden memory
            let mut words = enc.codes[s].as_words().to_vec();
            words.extend_from_slice(prev.map_or(&[0u64; BITS / 64][..], |p| enc.codes[p].as_words()));
            words.extend_from_slice(hidden.as_words());
            let input = BitVector::from_words(words);

            let mut out = BitVector::new(BITS, Some(0));
            class.process_predictive(&input, &mut out);
            let next = seq[i + 1];
            if policy == Policy::ThreeFactor {
                let reward = (enc.decode(&out) == Some(next)) as u8 as f64;
                for u in 0..k {
                    if t < mem.on_until[u] {
                        mem.credit[u] += reward - baseline;
                    }
                }
                baseline += 0.01 * (reward - baseline);
            }
            if i + 1 == answer_at && e >= episodes - 1000 {
                total += 1;
                hits += (enc.decode(&out) == Some(next)) as usize;
            }
            if guided {
                // Credit-guided growth: a new kernel may use the two symbol frames
                // plus the bits of ONE active unit, picked with odds credit + 1.
                let active: Vec<usize> = (0..k).filter(|&u| t < mem.on_until[u]).collect();
                let mut mask = BitVector::new(3 * BITS, Some(0));
                for b in 0..2 * BITS {
                    mask.bit_set(b);
                }
                if !active.is_empty() {
                    let weights: Vec<f64> = active.iter().map(|&u| mem.credit[u].max(0.0) + 1.0).collect();
                    let mut r = rng.gen_range(0.0..weights.iter().sum::<f64>());
                    let mut pick = active[0];
                    for (&u, &w) in active.iter().zip(&weights) {
                        if r < w {
                            pick = u;
                            break;
                        }
                        r -= w;
                    }
                    for &b in &mem.out_bits[pick] {
                        mask.bit_set(2 * BITS + b);
                    }
                }
                class.set_growth_mask(Some(mask));
            }
            if policy == Policy::Ablation {
                let right = enc.decode(&out) == Some(next);
                for u in 0..k {
                    if t >= mem.on_until[u] {
                        continue; // inactive units can't have mattered
                    }
                    let mut ablated = input.clone();
                    for &b in &mem.out_bits[u] {
                        ablated.bit_clear(2 * BITS + b);
                    }
                    let right_without = class.peek(&ablated).and_then(|o| enc.decode(o)) == Some(next);
                    mem.credit[u] += (right && !right_without) as u8 as f64 - (!right && right_without) as u8 as f64;
                }
            }
            class.feedback(&input, &enc.codes[next], &mut rng);

            if policy == Policy::Credit {
                let credited = class.credited_inputs(3 * BITS);
                for u in 0..k {
                    if mem.out_bits[u].iter().any(|&b| credited.bit_get(2 * BITS + b)) {
                        mem.credit[u] += 1.0;
                    }
                }
            }
            prev = Some(s);
            t += 1;
        }
        prev = Some(*seq.last().unwrap());
        t += 1;

        if e % REVIEW_EVERY == REVIEW_EVERY - 1 {
            let free = mem.unwatched();
            match policy {
                Policy::FixedRandom | Policy::Oracle | Policy::Partial(_) => {}
                Policy::Activity => {
                    // least active unit -> most frequent unwatched symbol
                    let u = (0..k).min_by(|&a, &b| mem.activity[a].partial_cmp(&mem.activity[b]).unwrap()).unwrap();
                    let s = *free.iter().max_by(|&&a, &&b| symbol_counts[a].partial_cmp(&symbol_counts[b]).unwrap()).unwrap();
                    if symbol_counts[s] > symbol_counts[mem.watch[u]] {
                        mem.reassign(u, s);
                    }
                }
                Policy::Credit | Policy::ThreeFactor | Policy::Ablation => {
                    // least credited unit -> a random unwatched symbol (explore).
                    // Patient exploration: new units get 3 reviews to earn credit,
                    // and a unit that has earned credit (>= 1) is never replaced.
                    let candidates: Vec<usize> = (0..k).filter(|&u| !patient || mem.age[u] >= 3).collect();
                    let worst = candidates.into_iter().min_by(|&a, &b| mem.credit[a].partial_cmp(&mem.credit[b]).unwrap());
                    if let Some(u) = worst {
                        if !patient || mem.credit[u] < 1.0 {
                            mem.reassign(u, *free.choose(&mut rng).unwrap());
                        }
                    }
                }
            }
            for u in 0..k {
                mem.age[u] += 1;
                mem.activity[u] *= 0.5;
                mem.credit[u] *= 0.5;
            }
        }
    }
    let mut watched = mem.watch.clone();
    watched.sort();
    (100.0 * hits as f64 / total as f64, watched)
}

fn name(s: usize) -> String {
    match s {
        s if s < CUES => format!("cue{}", (b'A' + s as u8) as char),
        s if s < QUERY => format!("f{}", s - CUES),
        s if s == QUERY => "?".into(),
        s => format!("ans{}", s - QUERY - 1),
    }
}

fn main() {
    let episodes: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(6000);
    println!("answer accuracy over the last 1000 of {episodes} episodes (chance 25% once '?' is recognized)");
    println!("predictor sees: current symbol, previous symbol, and K persistent memory units");
    println!();
    let all = [Policy::Oracle, Policy::FixedRandom, Policy::Activity, Policy::Credit, Policy::ThreeFactor, Policy::Ablation];
    let credit_based = [Policy::Credit, Policy::ThreeFactor, Policy::Ablation];
    let partial = [Policy::Partial(3), Policy::Partial(2), Policy::Partial(1)];
    let guided_set = [Policy::FixedRandom, Policy::ThreeFactor, Policy::Ablation];
    let configs: [(usize, Option<f32>, bool, bool, &[Policy]); 6] = [
        (4, None, false, false, &all),
        (4, Some(0.5), false, false, &all),
        (4, None, true, false, &credit_based),
        (4, None, false, false, &partial),
        (4, None, false, true, &guided_set),
        (4, None, true, true, &guided_set),
    ];
    let skip: usize = std::env::var("SKIP_CONFIGS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    for (k, generalize, patient, guided, policies) in configs.into_iter().skip(skip) {
        println!(
            "K={k} memory units, synapse-level credit (generalize near misses): {}{}{}",
            generalize.map_or("off".to_string(), |f| format!("on, near = matched >= {f} of connections")),
            if patient { "; patient exploration (3-review grace, keep units with credit >= 1)" } else { "" },
            if guided { "; credit-guided growth (one unit per new kernel)" } else { "" }
        );
        for &policy in policies {
            let mut accs = Vec::new();
            let mut example = Vec::new();
            for seed in 0..5 {
                let (acc, watched) = run(policy, k, generalize, patient, guided, episodes, seed);
                accs.push(acc);
                if seed == 0 {
                    example = watched;
                }
            }
            let mean = accs.iter().sum::<f64>() / accs.len() as f64;
            println!(
                "  {:<12} mean {mean:5.1}%  runs [{}]  e.g. units watch {{{}}}",
                format!("{policy:?}").replace("Partial(", "Fixed ").replace(')', " cues"),
                accs.iter().map(|a| format!("{a:.0}")).collect::<Vec<_>>().join(" "),
                example.iter().map(|&s| name(s)).collect::<Vec<_>>().join(", ")
            );
        }
        println!();
    }
}
