//! First probes of meaning.
//!
//! Run:  ./scripts/fetch_corpora.sh   (once)
//!       cargo run --release --example meaning
//!
//! (A) Word meaning from topical context (Brown corpus). Same mechanism as the
//!     syntax example (one SimpleKernel per word, existing Hebbian strengthen
//!     rule) but the context is the content words within +-5 tokens instead of
//!     the immediate neighbours. Scored on hand-made semantic groups: is a
//!     word's nearest neighbour (among the group words) in its own group?
//!
//! (B) Sentence meaning as fact binding (synthetic bAbI-style stories):
//!       "mary went to the kitchen . [john went to the garden .] where is mary ? kitchen"
//!     The predictive network reads the stories word by word and is scored on
//!     the answer word. Some name/place pairs never occur in training; answering
//!     those needs binding "who" to "where", not recall of a memorized pair.

mod common;

use std::collections::{HashMap, HashSet};

use common::{load_brown, Encoder, Predictor};
use neurocomp::bitvec::BitVector;
use neurocomp::kernel::{KernelContext, KernelOp, KernelTrait, SimpleKernel};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

const GROUPS: &[(&str, &[&str])] = &[
    ("numbers", &["two", "three", "four", "five", "six", "seven", "eight", "ten"]),
    ("colors", &["red", "white", "black", "blue", "green", "yellow", "brown", "gray"]),
    ("family", &["father", "mother", "son", "daughter", "wife", "husband", "brother", "sister"]),
    ("time", &["day", "week", "month", "year", "hour", "minute", "morning", "night"]),
    ("weekdays", &["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]),
    ("body", &["hand", "head", "eyes", "face", "arm", "feet", "hair", "body"]),
    ("politics", &["president", "congress", "senate", "government", "party", "election", "vote", "state"]),
    ("money", &["money", "tax", "price", "cost", "income", "budget", "pay", "dollars"]),
    ("religion", &["god", "church", "faith", "christ", "religion", "religious", "spirit", "lord"]),
    ("music/art", &["music", "art", "artist", "painting", "dance", "song", "theater", "poetry"]),
    ("science", &["scientific", "theory", "research", "data", "analysis", "experiment", "method", "physical"]),
    ("food", &["food", "bread", "meat", "coffee", "milk", "dinner", "eat", "sugar"]),
];

// ---------------------------------------------------------------------------
// (A)

fn word_meaning(rng: &mut StdRng) {
    let tokens = load_brown("data/brown_tagged.txt");
    let n_targets = 3000;
    let stop = 60; // the most frequent words carry no topic
    let window = 5;
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for (w, t) in &tokens {
        if *t != "PUNCT" {
            *freq.entry(w).or_default() += 1;
        }
    }
    let mut by_freq: Vec<(&str, usize)> = freq.into_iter().collect();
    by_freq.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let words: Vec<&str> = by_freq.iter().take(n_targets).map(|(w, _)| *w).collect();
    let index: HashMap<&str, usize> = words.iter().enumerate().map(|(i, w)| (*w, i)).collect();
    // token -> word id, None for punctuation / rare words
    let ids: Vec<Option<usize>> = tokens.iter().map(|(w, _)| index.get(w.as_str()).copied()).collect();
    let content = |i: Option<usize>| i.filter(|&i| i >= stop);

    println!("(A) word meaning from topical context: {} target words, context = content words within +-{window}", n_targets);
    let mut test: Vec<(usize, usize)> = Vec::new(); // (word id, group)
    let mut missing = Vec::new();
    for (g, (_, ws)) in GROUPS.iter().enumerate() {
        for w in *ws {
            match index.get(w) {
                Some(&i) => test.push((i, g)),
                None => missing.push(*w),
            }
        }
    }
    if !missing.is_empty() {
        println!("    (not in the {n_targets} target words, skipped: {})", missing.join(", "));
    }
    let closed_chance = {
        let mut sizes = vec![0usize; GROUPS.len()];
        for &(_, g) in &test {
            sizes[g] += 1;
        }
        sizes.iter().map(|&s| (s * s.saturating_sub(1)) as f64).sum::<f64>() / (test.len() * (test.len() - 1)) as f64 * 100.0
    };
    println!("    {} test words in {} groups; chance {:.1}%", test.len(), GROUPS.len(), closed_chance);

    let score = |sim: &dyn Fn(usize, usize) -> f64| -> f64 {
        let mut ok = 0;
        for &(a, ga) in &test {
            let best = test.iter().filter(|(b, _)| *b != a).max_by(|x, y| sim(a, x.0).partial_cmp(&sim(a, y.0)).unwrap()).unwrap();
            ok += (best.1 == ga) as usize;
        }
        100.0 * ok as f64 / test.len() as f64
    };

    // Sparse context codes (see syntax example), one Hebbian kernel per word.
    // Each occurrence applies the strengthen rule `moves` times (one connection
    // moved per application); rarer words need more moves per occurrence for
    // their mask to leave its random starting point.
    let bits = 32768;
    let enc = Encoder::new(n_targets, bits, 4, rng);
    let positions: Vec<usize> = (0..bits).collect();
    let inhibit = BitVector::new(64, Some(0));
    let mut sink = BitVector::new(64, Some(0));
    let mut counts: Vec<HashMap<usize, f64>> = vec![HashMap::new(); n_targets];
    let mut best_masks: Option<(f64, Vec<BitVector>)> = None;
    for (mask_bits, moves) in [(512usize, 1usize), (128, 8), (256, 16), (256, 32), (512, 32)] {
        let mut kernels: Vec<SimpleKernel> = (0..n_targets)
            .map(|_| {
                let mask = BitVector::from_bits(&positions.choose_multiple(rng, mask_bits).cloned().collect::<Vec<_>>(), bits);
                SimpleKernel::new(mask, 0, BitVector::from_words(vec![1]), 0, 0, KernelOp::Or)
            })
            .collect();
        let first = best_masks.is_none();
        for t in 0..ids.len() {
            let Some(w) = content(ids[t]) else { continue };
            let mut context = BitVector::new(bits, Some(0));
            let lo = t.saturating_sub(window);
            let hi = (t + window).min(ids.len() - 1);
            for u in lo..=hi {
                if u != t {
                    if let Some(c) = content(ids[u]) {
                        context.or_mut(&enc.codes[c]);
                        if first {
                            *counts[w].entry(c).or_default() += 1.0;
                        }
                    }
                }
            }
            if context.count_ones() == 0 {
                continue;
            }
            let ctx = KernelContext { input: &context, inhibit: &inhibit, temperature: 0, phase: 0 };
            for _ in 0..moves {
                kernels[w].try_fire(&ctx, &mut sink, rng);
            }
        }
        let masks: Vec<BitVector> = kernels.into_iter().map(|k| k.input_mask).collect();
        let ov = |a: usize, b: usize| -> f64 {
            masks[a].as_words().iter().zip(masks[b].as_words()).map(|(x, y)| (x & y).count_ones()).sum::<u32>() as f64
        };
        let acc = score(&ov);
        println!("    Hebbian topical masks: mask {mask_bits:>3} bits, {moves} move(s)/occurrence   {acc:5.1}%");
        if best_masks.as_ref().map_or(true, |(b, _)| acc > *b) {
            best_masks = Some((acc, masks));
        }
    }
    let masks = best_masks.unwrap().1;
    let overlap = |a: usize, b: usize| -> f64 {
        masks[a].as_words().iter().zip(masks[b].as_words()).map(|(x, y)| (x & y).count_ones()).sum::<u32>() as f64
    };

    let norms: Vec<f64> = counts.iter().map(|m| m.values().map(|v| v * v).sum::<f64>().sqrt()).collect();
    let cos = |a: usize, b: usize| -> f64 {
        let (s, l) = if counts[a].len() < counts[b].len() { (&counts[a], &counts[b]) } else { (&counts[b], &counts[a]) };
        s.iter().map(|(k, v)| v * l.get(k).unwrap_or(&0.0)).sum::<f64>() / (norms[a] * norms[b]).max(1e-9)
    };
    println!("    count vectors + cosine (reference) {:5.1}%", score(&cos));

    println!("    nearest neighbours among all {n_targets} words (best network setting):");
    for probe in ["mother", "red", "monday", "church", "money", "music", "eyes", "president", "coffee", "research"] {
        if let Some(&a) = index.get(probe) {
            let mut sims: Vec<(f64, usize)> = (stop..n_targets).filter(|&b| b != a).map(|b| (overlap(a, b), b)).collect();
            sims.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
            println!("      {probe:<10} -> {}", sims.iter().take(7).map(|(_, b)| words[*b]).collect::<Vec<_>>().join(", "));
        }
    }
    println!();
}

// ---------------------------------------------------------------------------
// (B)

const NAMES: &[&str] = &["mary", "john", "sandra", "daniel", "anna", "peter"];
const PLACES: &[&str] = &["kitchen", "garden", "office", "hallway", "bathroom", "bedroom"];

struct Story {
    words: Vec<String>,
    answer_at: usize, // index of the answer word
    held_out: bool,   // asked about a (name, place) pair never seen in training
}

fn story(rng: &mut StdRng, facts: usize, allowed: &dyn Fn(usize, usize) -> bool, want_held_out: bool) -> Story {
    loop {
        let mut words = Vec::new();
        let mut loc: HashMap<usize, usize> = HashMap::new();
        let mut order = Vec::new();
        let mut ok = true;
        for _ in 0..facts {
            let n = rng.gen_range(0..NAMES.len());
            let p = rng.gen_range(0..PLACES.len());
            ok &= allowed(n, p) || want_held_out;
            loc.insert(n, p);
            order.push(n);
            for w in [NAMES[n], "went", "to", "the", PLACES[p], "."] {
                words.push(w.to_string());
            }
        }
        // ask about the first-mentioned person, so other facts are distractors
        let q = order[0];
        let a = loc[&q];
        let held_out = !allowed(q, a);
        if !ok || held_out != want_held_out {
            continue;
        }
        for w in ["where", "is", NAMES[q], "?"] {
            words.push(w.to_string());
        }
        let answer_at = words.len();
        words.push(PLACES[a].to_string());
        words.push(".".to_string());
        return Story { words, answer_at, held_out };
    }
}

fn fact_binding(rng: &mut StdRng) {
    println!("(B) fact binding: \"<name> went to the <place> . (distractor facts) where is <name> ? <place>\"");
    // Hold out one place per name (a diagonal), never seen together in training.
    let allowed = |n: usize, p: usize| p != n % PLACES.len();
    let mut vocab: Vec<&str> = vec!["went", "to", "the", ".", "where", "is", "?"];
    vocab.extend(NAMES);
    vocab.extend(PLACES);
    let index: HashMap<&str, usize> = vocab.iter().enumerate().map(|(i, w)| (*w, i)).collect();
    let enc = Encoder::new(vocab.len(), 512, 32, rng);
    let places: HashSet<usize> = PLACES.iter().map(|p| index[p]).collect();

    for facts in [1usize, 2] {
        let window = 6 * facts + 4; // the whole story fits in the context window
        let mut p = Predictor::new(512, window, 1_000_000);
        let run = |s: &Story, p: &mut Predictor, learn: bool| -> (bool, bool) {
            let ids: Vec<usize> = s.words.iter().map(|w| index[w.as_str()]).collect();
            let mut correct = false;
            let mut place_like = false;
            for t in 0..ids.len() - 1 {
                p.step(&enc.codes[ids[t]]);
                if t + 1 == s.answer_at {
                    let guess = enc.decode(p.prediction());
                    correct = guess == Some(ids[t + 1]);
                    place_like = guess.map_or(false, |g| places.contains(&g));
                }
                if learn {
                    p.learn(&enc.codes[ids[t + 1]]);
                }
            }
            (correct, place_like)
        };
        for _ in 0..3000 {
            let s = story(rng, facts, &allowed, false);
            run(&s, &mut p, true);
        }
        let mut res = [(0usize, 0usize, 0usize); 2];
        for i in 0..1000 {
            let s = story(rng, facts, &allowed, i % 2 == 1);
            let (c, pl) = run(&s, &mut p, false);
            let r = &mut res[s.held_out as usize];
            r.0 += 1;
            r.1 += c as usize;
            r.2 += pl as usize;
        }
        println!(
            "    {facts} fact(s), {window}-word window, 3000 training stories: seen pairs {:5.1}% correct | held-out pairs {:5.1}% correct (answered with some place {:.0}%; chance 1/{})   [{} kernels]",
            100.0 * res[0].1 as f64 / res[0].0 as f64,
            100.0 * res[1].1 as f64 / res[1].0 as f64,
            100.0 * res[1].2 as f64 / res[1].0 as f64,
            PLACES.len(),
            p.class().len()
        );
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(3);
    word_meaning(&mut rng);
    fact_binding(&mut rng);
}
