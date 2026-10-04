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
use neurocomp::program::{BasalGanglia, Ca3Memory, DentateGyrus, EpisodicMemory, RelayChannel, Thalamus};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

const BITS: usize = 8192; // sparse enough that words rarely share bits (habituation is per bit)
const NAMES: &[&str] = &["mary", "john", "sandra", "daniel", "anna", "peter"];
const PLACES: &[&str] = &["kitchen", "garden", "office", "hallway", "bathroom", "bedroom"];
const FILLERS: &[&str] = &["then", "later", "so", "next", "after"];
const OBJECTS: &[&str] = &["ball", "apple", "book", "key", "cup", "box"];
const TRAIN: usize = 3000;
const TEST: usize = 1000;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Task {
    Short,
    Long,
    Varied,
    /// Two-hop questions: "X picked up the O" + "X went to the P" ... "where is the O ?"
    TwoHop,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Policy {
    NoMemory,
    FixedRelay, // hand-set thalamic routes from experiments 09-10: (1,4), (3,4), (3,8)
    Episodic,
    /// Experiment 12: dentate-gyrus code + Hebbian CA3 store instead of a list.
    /// (granule cells, active cells, recurrent settle steps)
    Ca3 { cells: usize, k: usize, settle: usize },
    /// Big-loop recall (`EpisodicMemory::recall_chain`) with this many hops; one
    /// predictor frame per hop.
    Loop(usize),
    /// Branching big loop (`EpisodicMemory::recall_branches`): hop 1 plus this many
    /// hop-2 branches, one per rare item of hop 1; one frame each.
    Branch(usize),
    /// Big loop where a basal-ganglia selector (`BasalGanglia`) picks which item of
    /// hop 1 to follow, learned from reward (did hop 2 recall the next word?).
    /// Frames: hop 1, hop 2.
    Select,
}

struct Story {
    words: Vec<&'static str>,
    answer_at: usize,
    held_out: bool,
}

fn allowed(n: usize, p: usize) -> bool {
    p != n % PLACES.len()
}

/// Two-hop story: 2-3 people move around (1-2 moves each) and 1-2 of them pick up an
/// object, in random order; the question asks where an object is, i.e. the last place
/// its holder went. Held-out: (object, place) answers never seen in training.
fn two_hop_story(rng: &mut StdRng, want_held_out: bool) -> Story {
    loop {
        let mut words: Vec<&'static str> = Vec::new();
        for _ in 0..rng.gen_range(0..=4) {
            words.push(FILLERS.choose(rng).unwrap());
        }
        let mut names: Vec<usize> = (0..NAMES.len()).collect();
        names.shuffle(rng);
        let people = &names[..rng.gen_range(2..=3)];
        let mut objects: Vec<usize> = (0..OBJECTS.len()).collect();
        objects.shuffle(rng);
        let held: Vec<(usize, usize)> = objects[..rng.gen_range(1..=2)].iter().map(|&o| (o, *people.choose(rng).unwrap())).collect();
        let mut sentences: Vec<(Vec<&'static str>, Option<(usize, usize)>)> = Vec::new(); // (words, (person, place) if a move)
        for &p in people {
            for _ in 0..rng.gen_range(1..=2) {
                let place = rng.gen_range(0..PLACES.len());
                sentences.push((vec![NAMES[p], "went", "to", "the", PLACES[place], "."], Some((p, place))));
            }
        }
        for &(o, p) in &held {
            sentences.push((vec![NAMES[p], "picked", "up", "the", OBJECTS[o], "."], None));
        }
        sentences.shuffle(rng);
        let (o, holder) = held[rng.gen_range(0..held.len())];
        let answer = sentences.iter().filter_map(|(_, m)| *m).filter(|&(p, _)| p == holder).last().unwrap().1;
        if (answer != o % PLACES.len()) == want_held_out {
            continue; // held-out answers: place == object index (a diagonal)
        }
        for (w, _) in sentences {
            words.extend(w);
        }
        words.extend(["where", "is", "the", OBJECTS[o], "?"]);
        let answer_at = words.len();
        words.extend([PLACES[answer], "."]);
        return Story { words, answer_at, held_out: want_held_out };
    }
}

fn story(rng: &mut StdRng, task: Task, max_facts: usize, want_held_out: bool) -> Story {
    if task == Task::TwoHop {
        return two_hop_story(rng, want_held_out);
    }
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
    /// % of test questions where the memory (or relay) frames contained the answer word
    recall: f64,
}

fn run(policy: Policy, task: Task, max_facts: usize, seed: u64) -> Outcome {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vocab: Vec<&str> = vec![
        "went", "to", "the", ".", "where", "is", "?", "all", "way", "over", "right", "now", "quickly", "slowly", "big", "old", "picked", "up",
    ];
    vocab.extend(OBJECTS);
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
        Policy::Loop(hops) => hops,
        Policy::Branch(b) => b + 1,
        Policy::Select => 2,
        Policy::FixedRelay => routes.len(),
    };
    let mut th = Thalamus::new(BITS, 60, routes);
    let mut memory = EpisodicMemory::new(BITS, 200); // also keeps the habituation statistics
    let (dg, mut ca3) = match policy {
        Policy::Ca3 { cells, k, settle } => {
            let env = |name: &str, default: f32| std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
            let fan_in = env("DG_FAN_IN", 300.0) as usize;
            let decay = env("CA3_DECAY", 0.7);
            let mut ca3 = Ca3Memory::new(BITS, cells, k, decay, settle);
            ca3.readout_fraction = env("CA3_READOUT", 0.5);
            (Some(DentateGyrus::new(BITS, cells, fan_in, k, seed + 100)), Some(ca3))
        }
        _ => (None, None),
    };
    let mut sentence = BitVector::new(BITS, Some(0)); // bag of the current sentence so far
    let mut bg = BasalGanglia::new(BITS);
    bg.trace_len = std::env::var("BG_TRACE").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut bg_pending: Option<BitVector> = None; // hop-2 content of the latest choice, awaiting reward
    // CA1-style comparator (NOVELTY=prediction): store, cue and read out only what the
    // predictor failed to predict, instead of frequency habituation.
    let predictive_novelty = std::env::var("NOVELTY").map_or(false, |v| v == "prediction");
    let mut surprising = BitVector::new(BITS, Some(0)); // unpredicted bits of the sentence so far
    let mut last_out = BitVector::new(BITS, Some(0)); // the predictor's last prediction
    let mut last_confidence = 0f32; // reliability of the kernel that made it
    let predicted_share: f32 = std::env::var("PREDICTED_SHARE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
    // recalled content: drop bits in more than 40% of episodes (all kept with prediction novelty)
    let habituation = if predictive_novelty { 1.0 } else { 0.4 };
    // which recalled item to cue with: the rarest among stored items (an IDF-like
    // specificity); RARITY=all cues with everything unpredicted
    let rarity_ratio = if std::env::var("RARITY").map_or(false, |v| v == "all") { f32::INFINITY } else { 1.5 };
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
    let mut recall_has_answer = 0usize;
    let full_stop = index["."];
    for s_i in 0..TRAIN + TEST {
        let testing = s_i >= TRAIN;
        let s = story(&mut rng, task, max_facts, testing && s_i % 2 == 1);
        let ids: Vec<usize> = s.words.iter().map(|w| index[w]).collect();
        for t in 0..ids.len() {
            let code = &enc.codes[ids[t]];
            th.observe(code);
            sentence.or_mut(code);
            // Comparator: was this word predicted? Graded, at the word level: the share of
            // the prediction that this word accounts for. A prediction that superimposes a
            // whole class ("some name", "some place") gives each member a small share, so
            // the actual member still counts as unpredicted. Bitwise mismatch would not.
            let predicted_bits = last_out.count_ones();
            let share = if predicted_bits == 0 {
                0.0
            } else {
                code.as_words().iter().zip(last_out.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() as f32 / predicted_bits as f32
            };
            // probability the predictor gave this word: its share of the prediction times
            // the predicting kernel's reliability (a lucky guess is still a surprise)
            let share = share * last_confidence;
            if share < predicted_share {
                surprising.or_mut(code);
            }
            if std::env::var("TRACE_SHARE").is_ok() && testing && s_i < TRAIN + 3 {
                eprint!("{}:{share:.2}/{predicted_bits} ", s.words[t]);
                if t + 1 == ids.len() {
                    eprintln!();
                }
            }
            let cue_source = if predictive_novelty { &surprising } else { &sentence };

            if t + 1 < ids.len() {
                let mut words = code.as_words().to_vec();
                match policy {
                    Policy::NoMemory => words.extend(std::iter::repeat(0).take(BITS / 64)),
                    Policy::FixedRelay => words.extend_from_slice(th.relay().as_words()),
                    Policy::Loop(hops) => {
                        let cue = memory.rarest(cue_source, 0.1, rarity_ratio);
                        let chain = memory.recall_chain(&cue, hops, habituation, 0.1, rarity_ratio);
                        if std::env::var("TRACE").is_ok() && testing && t + 1 == s.answer_at && s_i < TRAIN + 3 {
                            let names = |bv: &BitVector| -> Vec<&str> {
                                (0..vocab.len())
                                    .filter(|&i| enc.codes[i].as_words().iter().zip(bv.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24)
                                    .map(|i| vocab[i])
                                    .collect()
                            };
                            let hops_named: Vec<Vec<&str>> = chain.iter().map(|h| names(h)).collect();
                            eprintln!("{:?}\n  cue {:?} -> hops {:?}", s.words, names(&cue), hops_named);
                            let freqs: Vec<String> = ["picked", "up", "the", "went", "where", "mary", "john", "kitchen", "ball"]
                                .iter()
                                .map(|w| format!("{w}:{:.2}", memory.frequency(&enc.codes[index[w]])))
                                .collect();
                            eprintln!("  frequencies: {}", freqs.join(" "));
                        }
                        for h in 0..hops {
                            match chain.get(h) {
                                Some(frame) => words.extend_from_slice(frame.as_words()),
                                None => words.extend(std::iter::repeat(0).take(BITS / 64)),
                            }
                        }
                    }
                    Policy::Select => {
                        let cue = memory.rarest(cue_source, 0.1, rarity_ratio);
                        let (mut hop1, mut hop2) = (BitVector::new(BITS, Some(0)), BitVector::new(BITS, Some(0)));
                        let need = ((cue.count_ones() as f32 * 0.7).ceil() as u32).max(1);
                        if cue.count_ones() > 0 {
                            if let Some((id, ep)) = memory.recall_excluding(&cue, need, &[]) {
                                hop1 = memory.novel(ep, habituation);
                                for (c, &u) in hop1.as_words_mut().iter_mut().zip(cue.as_words()) {
                                    *c &= !u;
                                }
                                let items = memory.items(&hop1, 16);
                                let explore = if testing { None } else { Some(&mut rng) };
                                if let Some(i) = bg.select(&items, explore) {
                                    let item = &items[i];
                                    let need = ((item.count_ones() as f32 * 0.7).ceil() as u32).max(1);
                                    if let Some((_, ep2)) = memory.recall_excluding(item, need, &[id]) {
                                        hop2 = memory.novel(ep2, habituation);
                                        for (c, (&u, &it)) in hop2.as_words_mut().iter_mut().zip(cue.as_words().iter().zip(item.as_words())) {
                                            *c &= !(u | it);
                                        }
                                    }
                                    bg_pending = Some(hop2.clone());
                                }
                                if std::env::var("TRACE").is_ok() && testing && t + 1 == s.answer_at && s_i < TRAIN + 5 {
                                    let names = |bv: &BitVector| -> Vec<&str> {
                                        (0..vocab.len())
                                            .filter(|&i| enc.codes[i].as_words().iter().zip(bv.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24)
                                            .map(|i| vocab[i])
                                            .collect()
                                    };
                                    let vals: Vec<String> = items.iter().map(|it| format!("{:?}={:.2}", names(it), bg.value(it))).collect();
                                    eprintln!("{:?}\n  cue {:?} -> hop1 {:?}; items {}; hop2 {:?}", s.words, names(&cue), names(&hop1), vals.join(" "), names(&hop2));
                                }
                            }
                        }
                        words.extend_from_slice(hop1.as_words());
                        words.extend_from_slice(hop2.as_words());
                    }
                    Policy::Branch(b) => {
                        let cue = memory.rarest(cue_source, 0.1, rarity_ratio);
                        for frame in memory.recall_branches(&cue, b, habituation, 16) {
                            words.extend_from_slice(frame.as_words());
                        }
                    }
                    Policy::Ca3 { .. } => {
                        let cue = memory.rarest(cue_source, 0.1, rarity_ratio);
                        let mut recalled = BitVector::new(BITS, Some(0));
                        let cue_bits = set_bits(&cue);
                        if !cue_bits.is_empty() {
                            let (bits, strength) = ca3.as_ref().unwrap().recall(&cue_bits, BITS);
                            if std::env::var("TRACE").is_ok() && testing && t + 1 == s.answer_at && s_i < TRAIN + 3 {
                                let raw = BitVector::from_bits(&bits, BITS);
                                let names = |bv: &BitVector| -> Vec<&str> {
                                    (0..vocab.len())
                                        .filter(|&i| enc.codes[i].as_words().iter().zip(bv.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24)
                                        .map(|i| vocab[i])
                                        .collect()
                                };
                                eprintln!("{:?}\n  CA3 cue {:?} -> raw recall {:?} ({} bits, strength {strength:.2})", s.words, names(&cue), names(&raw), bits.len());
                            }
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
                        let cue = memory.rarest(cue_source, 0.1, rarity_ratio);
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
                if testing && t + 1 == s.answer_at {
                    // memory diagnostic: does any recalled frame contain the answer word?
                    let frame = BITS / 64;
                    let answer = enc.codes[ids[t + 1]].as_words();
                    let found = (0..mid_frames).any(|f| {
                        words[frame * (1 + f)..frame * (2 + f)].iter().zip(answer).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24
                    });
                    recall_has_answer += found as usize;
                }
                let input = BitVector::from_words(words);

                let mut out = BitVector::new(BITS, Some(0));
                class.process_predictive(&input, &mut out);
                last_out = out.clone();
                last_confidence = class.confidence().unwrap_or(0.0);
                let next = ids[t + 1];
                if testing && t + 1 == s.answer_at {
                    let right = enc.decode(&out) == Some(next);
                    let r = if s.held_out { &mut held } else { &mut seen };
                    r.0 += right as usize;
                    r.1 += 1;
                }
                if !testing {
                    class.feedback(&input, &enc.codes[next], &mut rng);
                    // dopamine: did the followed item's recall contain what came next?
                    if let Some(hop2) = bg_pending.take() {
                        let hit = hop2.as_words().iter().zip(enc.codes[next].as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24;
                        bg.reward(hit as u32 as f32, &mut rng);
                    }
                }
                bg_pending = None;
            }

            if ids[t] == full_stop {
                // one-shot: the whole sentence (or its unpredicted part) is one episode
                memory.store(if predictive_novelty { &surprising } else { &sentence });
                if let (Some(dg), Some(ca3)) = (&dg, &mut ca3) {
                    // Encode the novel part: content shared by most episodes ("went to the")
                    // would otherwise dominate the dentate gyrus, give every episode the same
                    // code, and swamp recall.
                    let x = set_bits(&if predictive_novelty { surprising.clone() } else { memory.novel(&sentence, 0.4) });
                    if !x.is_empty() {
                        ca3.store(&x, &dg.separate(&x));
                    }
                }
                sentence = BitVector::new(BITS, Some(0));
                surprising = BitVector::new(BITS, Some(0));
            }
            prev = Some(ids[t]);
        }
    }
    let pct = |r: (usize, usize)| 100.0 * r.0 as f64 / r.1.max(1) as f64;
    Outcome { seen: pct(seen), held_out: pct(held), recall: 100.0 * recall_has_answer as f64 / TEST as f64 }
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
        Ok("twohop") => vec![Task::TwoHop],
        _ => vec![Task::Short, Task::Long, Task::Varied],
    };
    println!("answer accuracy on {TEST} test stories after {TRAIN} training stories (predictor learning off at test); chance 1/6");
    if std::env::var("NOVELTY").map_or(false, |v| v == "prediction") {
        println!("NOVELTY=prediction: CA1-style comparator; store, cue and read out only what the predictor did not predict");
    }
    for task in tasks {
        let fact_settings: &[usize] = if task == Task::TwoHop { &[0] } else { &[2, 3] };
        for &max_facts in fact_settings {
            println!();
            if task == Task::TwoHop {
                println!("TwoHop stories: 2-3 people, 1-2 moves each, 1-2 objects picked up; \"where is the O ?\"");
            } else {
                println!("{task:?} stories, 1-{max_facts} facts, question about a random one");
            }
            let policies: Vec<Policy> = match std::env::var("POLICIES").as_deref() {
                Ok("loop") => vec![Policy::NoMemory, Policy::Episodic, Policy::Loop(1), Policy::Loop(2), Policy::Branch(3)],
                Ok("branch") => vec![Policy::Branch(3)],
                Ok("select") => vec![Policy::Select],
                Ok("bg") => vec![Policy::Loop(2), Policy::Branch(3), Policy::Select],
                Ok("episodic") => vec![Policy::Episodic],
                Ok("ca1") => vec![Policy::Episodic, Policy::Loop(2), Policy::Branch(3), Policy::Ca3 { cells: 16384, k: 32, settle: 2 }],
                Ok("ca3_high") => vec![Policy::Ca3 { cells: 16384, k: 32, settle: 2 }],
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
                    "  {:<40} seen pairs {:5.1}%   held-out pairs {:5.1}%   answer in recall {:5.1}%   held-out runs [{}]",
                    format!("{policy:?}"),
                    mean(|o| o.seen),
                    mean(|o| o.held_out),
                    mean(|o| o.recall),
                    runs.iter().map(|o| format!("{:.0}", o.held_out)).collect::<Vec<_>>().join(" ")
                );
            }
        }
    }
}
