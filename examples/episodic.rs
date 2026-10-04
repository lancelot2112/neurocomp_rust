//! Experiment 11: episodic autoassociative memory for fact binding.
//!
//! Run:  cargo run --release --example episodic
//!
//! Stories as in experiments 09-10 (random filler words between stories, 1-N
//! facts about distinct people, a question about a random one, one place per
//! name held out of training), in three forms:
//!   short:  "X went to the P ."                          "where is X ?"
//!   long:   "X went [all the way over] to the P ."        "where is X right now ?"
//!   varied: "X [quickly|slowly] went [all the way over] to the [big|old] P ."
//!           with every bracket optional, so the place sits at a variable
//!           distance from the name and no fixed offset finds it.
//!
//! Episodic memory (`EpisodicMemory`): each sentence (up to ".") is stored once as
//! the union of its word codes. While reading, the current sentence so far,
//! minus habituated (frequent) bits, is the cue; the most recent episode with the
//! largest overlap is recalled, and its novel bits, minus the cue, are given to
//! the predictor. No offsets, no routes, no gradual learning in the memory.
//! The memory keeps storing at test time (storing is reading); the predictor's
//! learning is off at test.
//!
//! Predictor input: [current word | memory (or relay) frames | previous word].
//!
//! Experiment 12 adds `Ca3` policies: episodes are stored in Hebbian weights
//! (EC->CA3, CA3 recurrent, CA3->EC; decaying palimpsest) under a dentate-gyrus
//! code, and recalled by driving CA3 from the cue and letting it settle. Low vs high
//! pattern separation = few dense vs many sparse granule/CA3 cells.

mod common;

use std::collections::HashMap;

use common::Encoder;
use neurocomp::bitvec::BitVector;
use neurocomp::kernel::{GrowthConfig, KernelClass, SimpleKernel};
use neurocomp::program::{Ca3Memory, DentateGyrus, EpisodicMemory, RelayChannel, Thalamus};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

const BITS: usize = 8192; // sparse enough that words rarely share bits (habituation is per bit)
const NAMES: &[&str] = &["mary", "john", "sandra", "daniel", "anna", "peter"];
const PLACES: &[&str] = &["kitchen", "garden", "office", "hallway", "bathroom", "bedroom"];
const FILLERS: &[&str] = &["then", "later", "so", "next", "after"];
const TRAIN: usize = 3000;
const TEST: usize = 1000;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Task {
    Short,
    Long,
    Varied,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Policy {
    NoMemory,
    FixedRelay, // hand-set thalamic routes from experiments 09-10: (1,4), (3,4), (3,8)
    Episodic,
    /// Experiment 12: dentate-gyrus code + Hebbian CA3 store instead of a list.
    /// (granule cells, active cells, recurrent settle steps)
    Ca3 { cells: usize, k: usize, settle: usize },
}

struct Story {
    words: Vec<&'static str>,
    answer_at: usize,
    held_out: bool,
}

fn allowed(n: usize, p: usize) -> bool {
    p != n % PLACES.len()
}

fn story(rng: &mut StdRng, task: Task, max_facts: usize, want_held_out: bool) -> Story {
    loop {
        let mut words: Vec<&'static str> = Vec::new();
        for _ in 0..rng.gen_range(0..=4) {
            words.push(FILLERS.choose(rng).unwrap());
        }
        let facts = rng.gen_range(1..=max_facts);
        let mut names: Vec<usize> = (0..NAMES.len()).collect();
        names.shuffle(rng);
        let mut loc = Vec::new();
        for &n in &names[..facts] {
            let p = rng.gen_range(0..PLACES.len());
            loc.push((n, p));
            words.push(NAMES[n]);
            if task == Task::Varied && rng.gen_bool(0.5) {
                words.push(if rng.gen_bool(0.5) { "quickly" } else { "slowly" });
            }
            words.push("went");
            if task != Task::Short && rng.gen_bool(0.5) {
                words.extend(["all", "the", "way", "over"]);
            }
            words.extend(["to", "the"]);
            if task == Task::Varied && rng.gen_bool(0.5) {
                words.push(if rng.gen_bool(0.5) { "big" } else { "old" });
            }
            words.extend([PLACES[p], "."]);
        }
        let (q, a) = loc[rng.gen_range(0..facts)];
        let ok = want_held_out || loc.iter().all(|&(n, p)| allowed(n, p));
        if !ok || allowed(q, a) == want_held_out {
            continue;
        }
        words.extend(["where", "is", NAMES[q]]);
        if task != Task::Short {
            words.extend(["right", "now"]);
        }
        words.push("?");
        let answer_at = words.len();
        words.extend([PLACES[a], "."]);
        return Story { words, answer_at, held_out: want_held_out };
    }
}

struct Outcome {
    seen: f64,
    held_out: f64,
}

fn run(policy: Policy, task: Task, max_facts: usize, seed: u64) -> Outcome {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vocab: Vec<&str> = vec![
        "went", "to", "the", ".", "where", "is", "?", "all", "way", "over", "right", "now", "quickly", "slowly", "big", "old",
    ];
    vocab.extend(NAMES);
    vocab.extend(PLACES);
    vocab.extend(FILLERS);
    let index: HashMap<&str, usize> = vocab.iter().enumerate().map(|(i, w)| (*w, i)).collect();
    let enc = Encoder::new(vocab.len(), BITS, 32, &mut rng);

    let routes = vec![
        RelayChannel { query_lag: 1, value_offset: 4 },
        RelayChannel { query_lag: 3, value_offset: 4 },
        RelayChannel { query_lag: 3, value_offset: 8 },
    ];
    let mid_frames = match policy {
        Policy::NoMemory | Policy::Episodic | Policy::Ca3 { .. } => 1,
        Policy::FixedRelay => routes.len(),
    };
    let mut th = Thalamus::new(BITS, 60, routes);
    let mut memory = EpisodicMemory::new(BITS, 200); // also keeps the habituation statistics
    let (dg, mut ca3) = match policy {
        Policy::Ca3 { cells, k, settle } => {
            (Some(DentateGyrus::new(BITS, cells, 300, k, seed + 100)), Some(Ca3Memory::new(BITS, cells, k, 0.97, settle)))
        }
        _ => (None, None),
    };
    let mut sentence = BitVector::new(BITS, Some(0)); // bag of the current sentence so far
    let habituation = 0.4; // recalled content: drop bits in more than 40% of episodes
    let min_overlap = 8; // floor for the recall threshold

    let mut class: KernelClass<SimpleKernel> = KernelClass::predictive(GrowthConfig {
        max_kernels: 100_000,
        frame_words: BITS / 64,
        max_frames: mid_frames + 2,
        sample_bits: 16,
        match_fraction: 0.8,
        surprise_fraction: 0.5,
        generalize: None,
        generalize_after: 1,
    });

    let mut prev: Option<usize> = None;
    let (mut seen, mut held) = ((0usize, 0usize), (0usize, 0usize));
    let full_stop = index["."];
    for s_i in 0..TRAIN + TEST {
        let testing = s_i >= TRAIN;
        let s = story(&mut rng, task, max_facts, testing && s_i % 2 == 1);
        let ids: Vec<usize> = s.words.iter().map(|w| index[w]).collect();
        for t in 0..ids.len() {
            let code = &enc.codes[ids[t]];
            th.observe(code);
            sentence.or_mut(code);

            if t + 1 < ids.len() {
                let mut words = code.as_words().to_vec();
                match policy {
                    Policy::NoMemory => words.extend(std::iter::repeat(0).take(BITS / 64)),
                    Policy::FixedRelay => words.extend_from_slice(th.relay().as_words()),
                    Policy::Ca3 { .. } => {
                        let cue = memory.rarest(&sentence, 0.1, 1.5);
                        let mut recalled = BitVector::new(BITS, Some(0));
                        let cue_bits = set_bits(&cue);
                        if !cue_bits.is_empty() {
                            let (bits, strength) = ca3.as_ref().unwrap().recall(&cue_bits, BITS);
                            if strength > 0.0 {
                                recalled = memory.novel(&BitVector::from_bits(&bits, BITS), habituation);
                                for (r, &c) in recalled.as_words_mut().iter_mut().zip(cue.as_words()) {
                                    *r &= !c;
                                }
                            }
                        }
                        words.extend_from_slice(recalled.as_words());
                    }
                    Policy::Episodic => {
                        let cue = memory.rarest(&sentence, 0.1, 1.5);
                        let mut recalled = BitVector::new(BITS, Some(0));
                        if cue.count_ones() > 0 {
                            // recall needs most of the cue to be present in the episode
                            let need = ((cue.count_ones() as f32 * 0.7) as u32).max(min_overlap);
                            if let Some(ep) = memory.recall(&cue, need) {
                                recalled = memory.novel(ep, habituation);
                                // what the memory adds beyond the cue
                                for (r, &c) in recalled.as_words_mut().iter_mut().zip(cue.as_words()) {
                                    *r &= !c;
                                }
                            }
                        }
                        if std::env::var("TRACE").is_ok() && testing && t + 1 == s.answer_at && s_i < TRAIN + 3 {
                            let names = |bv: &BitVector| -> Vec<&str> {
                                (0..vocab.len())
                                    .filter(|&i| enc.codes[i].as_words().iter().zip(bv.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24)
                                    .map(|i| vocab[i])
                                    .collect()
                            };
                            eprintln!(
                                "{:?}\n  cue {:?} ({} bits; sentence {} bits) -> recalled novel {:?}",
                                s.words,
                                names(&cue),
                                cue.count_ones(),
                                sentence.count_ones(),
                                names(&recalled)
                            );
                            let freqs: Vec<String> = s.words[..=t].iter().map(|w| format!("{w}:{:.2}", memory.frequency(&enc.codes[index[w]]))).collect();
                            eprintln!("  word frequency over {} stored episodes: {}", memory.len(), freqs.join(" "));
                        }
                        words.extend_from_slice(recalled.as_words());
                    }
                }
                words.extend_from_slice(prev.map_or(&[0u64; BITS / 64][..], |p| enc.codes[p].as_words()));
                let input = BitVector::from_words(words);

                let mut out = BitVector::new(BITS, Some(0));
                class.process_predictive(&input, &mut out);
                let next = ids[t + 1];
                if testing && t + 1 == s.answer_at {
                    let right = enc.decode(&out) == Some(next);
                    let r = if s.held_out { &mut held } else { &mut seen };
                    r.0 += right as usize;
                    r.1 += 1;
                }
                if !testing {
                    class.feedback(&input, &enc.codes[next], &mut rng);
                }
            }

            if ids[t] == full_stop {
                memory.store(&sentence); // one-shot: the whole sentence is one episode
                if let (Some(dg), Some(ca3)) = (&dg, &mut ca3) {
                    let x = set_bits(&sentence);
                    ca3.store(&x, &dg.separate(&x));
                }
                sentence = BitVector::new(BITS, Some(0));
            }
            prev = Some(ids[t]);
        }
    }
    let pct = |r: (usize, usize)| 100.0 * r.0 as f64 / r.1.max(1) as f64;
    Outcome { seen: pct(seen), held_out: pct(held) }
}

fn set_bits(bv: &BitVector) -> Vec<usize> {
    (0..bv.bit_len()).filter(|&b| bv.bit_get(b)).collect()
}

fn main() {
    let seeds: u64 = std::env::var("SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let tasks: Vec<Task> = match std::env::var("TASK").as_deref() {
        Ok("short") => vec![Task::Short],
        Ok("long") => vec![Task::Long],
        Ok("varied") => vec![Task::Varied],
        _ => vec![Task::Short, Task::Long, Task::Varied],
    };
    println!("answer accuracy on {TEST} test stories after {TRAIN} training stories (predictor learning off at test); chance 1/6");
    for task in tasks {
        for max_facts in [2usize, 3] {
            println!();
            println!("{task:?} stories, 1-{max_facts} facts, question about a random one");
            let policies: Vec<Policy> = match std::env::var("POLICIES").as_deref() {
                Ok("ca3") => vec![
                    Policy::Episodic,
                    Policy::Ca3 { cells: 1024, k: 64, settle: 2 },
                    Policy::Ca3 { cells: 16384, k: 32, settle: 2 },
                    Policy::Ca3 { cells: 16384, k: 32, settle: 0 },
                ],
                _ => vec![Policy::NoMemory, Policy::FixedRelay, Policy::Episodic],
            };
            for policy in policies {
                let runs: Vec<Outcome> = (0..seeds).map(|seed| run(policy, task, max_facts, seed)).collect();
                let mean = |f: fn(&Outcome) -> f64| runs.iter().map(f).sum::<f64>() / runs.len() as f64;
                println!(
                    "  {:<40} seen pairs {:5.1}%   held-out pairs {:5.1}%   held-out runs [{}]",
                    format!("{policy:?}"),
                    mean(|o| o.seen),
                    mean(|o| o.held_out),
                    runs.iter().map(|o| format!("{:.0}", o.held_out)).collect::<Vec<_>>().join(" ")
                );
            }
        }
    }
}
