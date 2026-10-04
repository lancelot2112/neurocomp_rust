//! Syntax from word sequences (Brown corpus).
//!
//! Run:  ./scripts/fetch_corpora.sh   (once)
//!       cargo run --release --example syntax [max_tokens_for_prediction]
//!
//! Input is the real word tokens (lowercased, punctuation kept as tokens). The
//! part-of-speech tags are never shown to the network; they are only used to score.
//!
//! (a) Next-word prediction: a predictive network reads one word code per tick
//!     with a 3-word context window; scored online against word n-grams.
//!
//! (b) Category induction: each of the most frequent words owns one
//!     SimpleKernel whose input mask spans [previous word code | next word code].
//!     Every time the word occurs, the kernel fires on its context and the
//!     repo's local Hebbian rule (`strengthen`: move a connected-but-silent bit
//!     onto an active-but-unconnected one) pulls the mask toward the contexts the
//!     word appears in. Words used the same way end up with overlapping masks.
//!     Scored by whether a word's nearest neighbour (largest mask overlap) has
//!     the same part of speech, against chance, untrained masks, and classic
//!     count-based context vectors. Context codes are much sparser than the
//!     prediction codes: with dense codes each bit is shared by many words and
//!     the masks can't tell contexts apart.

mod common;

use std::collections::HashMap;
use std::time::Instant;

use common::{load_brown, Encoder, Predictor};
use neurocomp::bitvec::BitVector;
use neurocomp::kernel::{KernelContext, KernelOp, KernelTrait, SimpleKernel};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

const WORD_BITS: usize = 1024;
const WORD_ACTIVE: usize = 32;
const VOCAB: usize = 5000; // + UNK
const TARGETS: usize = 1000; // words that get a syntax kernel

fn vocab(tokens: &[(String, &str)], size: usize) -> (Vec<String>, HashMap<String, usize>) {
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for (w, _) in tokens {
        *freq.entry(w).or_default() += 1;
    }
    let mut words: Vec<(&str, usize)> = freq.into_iter().collect();
    words.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let mut list: Vec<String> = words.into_iter().take(size).map(|(w, _)| w.to_string()).collect();
    list.push("<unk>".into());
    let index = list.iter().enumerate().map(|(i, w)| (w.clone(), i)).collect();
    (list, index)
}

// ---------------------------------------------------------------------------
// (a) next-word prediction

fn next_word(ids: &[usize], enc: &Encoder, n_words: usize) {
    let unk = n_words - 1;
    println!("(a) next-word prediction, {} tokens read once, scored online (predicting <unk> counts as wrong)", ids.len());
    let max_order = 3;
    let mut tables: Vec<HashMap<Vec<usize>, HashMap<usize, u32>>> = vec![HashMap::new(); max_order];
    let mut fixed = vec![(0usize, 0usize); max_order];
    let mut backoff = (0usize, 0usize);
    let best = |row: &HashMap<usize, u32>| row.iter().max_by_key(|(w, k)| (**k, std::cmp::Reverse(**w))).map(|(w, _)| *w);
    for t in 0..ids.len() - 1 {
        let next = ids[t + 1];
        let mut bo = None;
        for o in (1..=max_order).rev() {
            if t + 1 < o {
                continue;
            }
            let pred = tables[o - 1].get(&ids[t + 1 - o..=t]).and_then(best);
            fixed[o - 1].1 += 1;
            fixed[o - 1].0 += (pred == Some(next) && next != unk) as usize;
            if bo.is_none() {
                bo = pred;
            }
        }
        backoff.1 += 1;
        backoff.0 += (bo == Some(next) && next != unk) as usize;
        for o in 1..=max_order.min(t + 1) {
            *tables[o - 1].entry(ids[t + 1 - o..=t].to_vec()).or_default().entry(next).or_default() += 1;
        }
    }
    for o in 1..=max_order {
        println!("    {o}-word n-gram           {:5.1}%", 100.0 * fixed[o - 1].0 as f64 / fixed[o - 1].1 as f64);
    }
    println!("    back-off n-gram (<=3)   {:5.1}%", 100.0 * backoff.0 as f64 / backoff.1 as f64);

    let start = Instant::now();
    let mut p = Predictor::new(WORD_BITS, 3, 1_000_000);
    let (mut hits, mut total) = (0usize, 0usize);
    for t in 0..ids.len() - 1 {
        p.step(&enc.codes[ids[t]]);
        let next = ids[t + 1];
        total += 1;
        hits += (enc.decode(p.prediction()) == Some(next) && next != unk) as usize;
        p.learn(&enc.codes[next]);
    }
    println!(
        "    network, 3-word context  {:5.1}%   [{} kernels, {:.0}s]",
        100.0 * hits as f64 / total as f64,
        p.class().len(),
        start.elapsed().as_secs_f64()
    );
    println!();
}

// ---------------------------------------------------------------------------
// (b) category induction

/// Nearest-neighbour part-of-speech agreement for a similarity function.
fn nn_agreement(n: usize, gold: &[&str], sim: impl Fn(usize, usize) -> f64) -> (f64, Vec<usize>) {
    let mut agree = 0;
    let mut nn = vec![0; n];
    for a in 0..n {
        let mut best = (f64::MIN, a);
        for b in 0..n {
            if a != b {
                let s = sim(a, b);
                if s > best.0 {
                    best = (s, b);
                }
            }
        }
        nn[a] = best.1;
        agree += (gold[a] == gold[best.1]) as usize;
    }
    (100.0 * agree as f64 / n as f64, nn)
}

fn overlap(a: &BitVector, b: &BitVector) -> f64 {
    a.as_words().iter().zip(b.as_words()).map(|(x, y)| (x & y).count_ones()).sum::<u32>() as f64
}

fn categories(ids: &[usize], tags: &[&'static str], enc: &Encoder, words: &[String], mask_bits: usize, verbose: bool, rng: &mut StdRng) {
    // gold = most frequent coarse tag of each target word
    let mut tag_counts: Vec<HashMap<&str, usize>> = vec![HashMap::new(); TARGETS];
    for (&w, &t) in ids.iter().zip(tags) {
        if w < TARGETS {
            *tag_counts[w].entry(t).or_default() += 1;
        }
    }
    let gold: Vec<&str> = tag_counts.iter().map(|m| *m.iter().max_by_key(|(t, k)| (**k, **t)).unwrap().0).collect();
    let mut by_tag: HashMap<&str, usize> = HashMap::new();
    for g in &gold {
        *by_tag.entry(g).or_default() += 1;
    }
    let chance: f64 = by_tag.values().map(|&k| (k as f64 / TARGETS as f64).powi(2)).sum::<f64>() * 100.0;
    let mut dist: Vec<(&str, usize)> = by_tag.into_iter().collect();
    dist.sort_by(|a, b| b.1.cmp(&a.1));
    if verbose {
        println!("    gold tags: {}", dist.iter().map(|(t, k)| format!("{t} {k}")).collect::<Vec<_>>().join(", "));
        println!("    chance (random neighbour)          {chance:5.1}%");
    }

    // One Hebbian kernel per target word over [prev code | next code].
    let ctx_bits = 2 * enc.bits;
    let positions: Vec<usize> = (0..ctx_bits).collect();
    let mut kernels: Vec<SimpleKernel> = (0..TARGETS)
        .map(|_| {
            let mask = BitVector::from_bits(&positions.choose_multiple(rng, mask_bits).cloned().collect::<Vec<_>>(), ctx_bits);
            // threshold 0: fire (and so learn) on every occurrence
            SimpleKernel::new(mask, 0, BitVector::from_words(vec![1]), 0, 0, KernelOp::Or)
        })
        .collect();
    let untrained: Vec<BitVector> = kernels.iter().map(|k| k.input_mask.clone()).collect();
    if verbose {
        let (u, _) = nn_agreement(TARGETS, &gold, |a, b| overlap(&untrained[a], &untrained[b]));
        println!("    untrained random masks             {u:5.1}%");
    }

    let inhibit = BitVector::new(64, Some(0));
    let mut sink = BitVector::new(64, Some(0));
    let start = Instant::now();
    let mut ctx_words = Vec::with_capacity(ctx_bits / 64);
    for t in 1..ids.len() - 1 {
        let w = ids[t];
        if w >= TARGETS {
            continue;
        }
        ctx_words.clear();
        ctx_words.extend_from_slice(enc.codes[ids[t - 1]].as_words());
        ctx_words.extend_from_slice(enc.codes[ids[t + 1]].as_words());
        let context = BitVector::from_words(ctx_words.clone());
        let ctx = KernelContext { input: &context, inhibit: &inhibit, temperature: 0, phase: 0 };
        kernels[w].try_fire(&ctx, &mut sink, rng);
    }
    let masks: Vec<BitVector> = kernels.iter().map(|k| k.input_mask.clone()).collect();
    let (h, nn) = nn_agreement(TARGETS, &gold, |a, b| overlap(&masks[a], &masks[b]));
    println!(
        "    Hebbian masks: codes {:>5} bits/{:>2} active, mask {mask_bits:>3} bits   nearest-neighbour POS agreement {h:5.1}%   [{:.1}s]",
        enc.bits,
        enc.codes[0].count_ones(),
        start.elapsed().as_secs_f64()
    );
    if !verbose {
        return;
    }

    // Reference: count vectors over (prev word, next word) ids, cosine similarity.
    let mut counts: Vec<HashMap<usize, f64>> = vec![HashMap::new(); TARGETS];
    for t in 1..ids.len() - 1 {
        let w = ids[t];
        if w < TARGETS {
            *counts[w].entry(ids[t - 1]).or_default() += 1.0;
            *counts[w].entry(ids[t + 1] + words.len()).or_default() += 1.0;
        }
    }
    let norms: Vec<f64> = counts.iter().map(|m| m.values().map(|v| v * v).sum::<f64>().sqrt()).collect();
    let cos = |a: usize, b: usize| {
        let (small, large) = if counts[a].len() < counts[b].len() { (&counts[a], &counts[b]) } else { (&counts[b], &counts[a]) };
        small.iter().map(|(k, v)| v * large.get(k).unwrap_or(&0.0)).sum::<f64>() / (norms[a] * norms[b]).max(1e-9)
    };
    let (c, _) = nn_agreement(TARGETS, &gold, cos);
    println!("    count vectors + cosine (reference) {c:5.1}%");

    // Per-tag agreement for the network.
    let mut per: HashMap<&str, (usize, usize)> = HashMap::new();
    for a in 0..TARGETS {
        let e = per.entry(gold[a]).or_default();
        e.1 += 1;
        e.0 += (gold[a] == gold[nn[a]]) as usize;
    }
    let mut per: Vec<(&str, (usize, usize))> = per.into_iter().collect();
    per.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
    println!(
        "    network by tag: {}",
        per.iter().map(|(t, (a, n))| format!("{t} {:.0}%", 100.0 * *a as f64 / *n as f64)).collect::<Vec<_>>().join(", ")
    );

    // A few neighbourhoods.
    let index: HashMap<&str, usize> = words.iter().take(TARGETS).enumerate().map(|(i, w)| (w.as_str(), i)).collect();
    println!("    nearest neighbours (network):");
    for probe in ["the", "he", "said", "was", "man", "house", "good", "very", "in", "and", "three", "quickly"] {
        if let Some(&a) = index.get(probe) {
            let mut sims: Vec<(f64, usize)> = (0..TARGETS).filter(|&b| b != a).map(|b| (overlap(&masks[a], &masks[b]), b)).collect();
            sims.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
            println!(
                "      {probe:<8} [{}] -> {}",
                gold[a],
                sims.iter().take(6).map(|(_, b)| words[*b].as_str()).collect::<Vec<_>>().join(", ")
            );
        }
    }
}

fn main() {
    let max_tokens: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let tokens = load_brown("data/brown_tagged.txt");
    let (words, index) = vocab(&tokens, VOCAB);
    let unk = words.len() - 1;
    let ids: Vec<usize> = tokens.iter().map(|(w, _)| *index.get(w).unwrap_or(&unk)).collect();
    let tags: Vec<&'static str> = tokens.iter().map(|(_, t)| *t).collect();
    let coverage = ids.iter().filter(|&&i| i != unk).count() as f64 / ids.len() as f64;
    println!("Brown corpus: {} tokens, vocabulary {} + <unk> ({:.1}% of tokens covered)", ids.len(), VOCAB, 100.0 * coverage);
    println!();

    let mut rng = StdRng::seed_from_u64(5);
    let enc = Encoder::new(words.len(), WORD_BITS, WORD_ACTIVE, &mut rng);
    if max_tokens > 0 {
        next_word(&ids[..max_tokens.min(ids.len())], &enc, words.len());
    }
    println!("(b) category induction for the {TARGETS} most frequent words ({} tokens)", ids.len());
    // Sparser context codes collide less (each bit names fewer words); the last
    // setting is reported in detail.
    let sweep = [(1024usize, 32usize, 32usize), (4096, 16, 128), (8192, 8, 256), (16384, 4, 256), (32768, 4, 512)];
    for (i, &(bits, active, mask_bits)) in sweep.iter().enumerate() {
        let ctx_enc = Encoder::new(words.len(), bits, active, &mut rng);
        categories(&ids, &tags, &ctx_enc, &words, mask_bits, i + 1 == sweep.len(), &mut rng);
    }
}
