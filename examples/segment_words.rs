//! Word recognition from an unsegmented letter stream.
//!
//! Run:  ./scripts/fetch_corpora.sh   (once)
//!       cargo run --release --example segment_words [path]
//!
//! The book is reduced to letters only ("aliceswasbeginningtoget..."), so word
//! boundaries are never shown to the network. It has to find them itself:
//!
//! 1. Segmentation. A predictive network reads the letters and predicts the next
//!    one. Inside a word the next letter is predictable; after a word ends it is
//!    not. We cut where the network's confidence (the hit rate of its winning
//!    kernel) dips, and compare with cutting on the classic transitional
//!    probability P(next | current) used in infant word-segmentation studies.
//!
//! 2. Recognition. Each segmented chunk is presented to a word layer (a
//!    KernelClass) as its letters plus a boundary marker. If a word kernel fires,
//!    the chunk is recognized as that word; otherwise a new word kernel is grown
//!    with a fresh word code (the lexicon grows by one).
//!
//! Scores compare against the real word boundaries (spaces/punctuation in the
//! original text) on the second half of the book, after the first half has been read.

mod common;

use std::collections::HashSet;

use common::{normalize, read_corpus, Encoder, Predictor, LETTERS};
use neurocomp::bitvec::BitVector;
use neurocomp::kernel::{GrowthConfig, KernelClass, SimpleKernel};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

const CHAR_BITS: usize = 512;
const CHAR_ACTIVE: usize = 32;
const WORD_BITS: usize = 1024;
const WORD_ACTIVE: usize = 32;
const MAX_WORD: usize = 15; // longer chunks are not stored in the lexicon

/// Letters only, plus where the real words end (`ends[i]` = a word ends at letter i).
fn letters_and_boundaries(text: &str) -> (Vec<usize>, Vec<bool>) {
    let mut letters = Vec::new();
    let mut ends = Vec::new();
    for c in text.chars() {
        if c == '\'' {
            continue; // "alice's" -> "alices"
        }
        if let Some(i) = LETTERS.find(c) {
            letters.push(i);
            ends.push(false);
        } else if let Some(last) = ends.last_mut() {
            *last = true;
        }
    }
    if let Some(last) = ends.last_mut() {
        *last = true;
    }
    (letters, ends)
}

#[derive(Default, Clone, Copy)]
struct Prf {
    tp: usize,
    fp: usize,
    fn_: usize,
}

impl Prf {
    fn p(&self) -> f64 {
        self.tp as f64 / (self.tp + self.fp).max(1) as f64
    }
    fn r(&self) -> f64 {
        self.tp as f64 / (self.tp + self.fn_).max(1) as f64
    }
    fn f1(&self) -> f64 {
        let (p, r) = (self.p(), self.r());
        if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
    }
    fn show(&self) -> String {
        format!("P {:4.1} R {:4.1} F1 {:4.1}", 100.0 * self.p(), 100.0 * self.r(), 100.0 * self.f1())
    }
}

/// Boundary and word-token scores of predicted cuts vs the truth over [from, to).
fn score(cuts: &[bool], truth: &[bool], from: usize, to: usize) -> (Prf, Prf) {
    let mut b = Prf::default();
    for i in from..to - 1 {
        match (cuts[i], truth[i]) {
            (true, true) => b.tp += 1,
            (true, false) => b.fp += 1,
            (false, true) => b.fn_ += 1,
            _ => {}
        }
    }
    let spans = |c: &[bool]| -> HashSet<(usize, usize)> {
        let mut out = HashSet::new();
        let mut start = from;
        for i in from..to {
            if c[i] || i == to - 1 {
                out.insert((start, i));
                start = i + 1;
            }
        }
        out
    };
    // Align both segmentations to start at a real word start.
    let (p, t) = (spans(cuts), spans(truth));
    let hit = p.intersection(&t).count();
    let w = Prf { tp: hit, fp: p.len() - hit, fn_: t.len() - hit };
    (b, w)
}

/// Cut where `s` has a local minimum (s[i] < s[i-1] and s[i] <= s[i+1]).
fn local_minima(s: &[f32]) -> Vec<bool> {
    (0..s.len()).map(|i| i > 0 && i + 1 < s.len() && s[i] < s[i - 1] && s[i] <= s[i + 1]).collect()
}

fn below(s: &[f32], theta: f32) -> Vec<bool> {
    s.iter().map(|&x| x < theta).collect()
}

/// The word layer: chunk -> word id, growing a word kernel for unseen chunks.
struct Lexicon {
    class: KernelClass<SimpleKernel>,
    words: Encoder,
    spelling: Vec<String>,
    seen: Vec<usize>,
    boundary: BitVector,
}

impl Lexicon {
    fn new(rng: &mut StdRng) -> Self {
        let class = KernelClass::predictive(GrowthConfig {
            max_kernels: 1_000_000,
            frame_words: CHAR_BITS / 64,
            max_frames: MAX_WORD + 1,
            sample_bits: 16,
            match_fraction: 0.8,
            surprise_fraction: 0.5,
        });
        let positions: Vec<usize> = (0..CHAR_BITS).collect();
        let boundary =
            BitVector::from_bits(&positions.choose_multiple(rng, CHAR_ACTIVE).cloned().collect::<Vec<_>>(), CHAR_BITS);
        Self { class, words: Encoder::new(0, WORD_BITS, WORD_ACTIVE, rng), spelling: Vec::new(), seen: Vec::new(), boundary }
    }

    /// Frames most recent first: last letter, ..., first letter, boundary marker.
    fn chunk_input(&self, chunk: &[usize], letters: &Encoder) -> BitVector {
        let mut words = Vec::with_capacity((MAX_WORD + 1) * CHAR_BITS / 64);
        for &c in chunk.iter().rev() {
            words.extend_from_slice(letters.codes[c].as_words());
        }
        words.extend_from_slice(self.boundary.as_words());
        words.resize((MAX_WORD + 1) * CHAR_BITS / 64, 0);
        BitVector::from_words(words)
    }

    /// Word id if a word kernel recognizes the chunk (no learning).
    fn lookup(&mut self, chunk: &[usize], letters: &Encoder) -> Option<usize> {
        if chunk.is_empty() || chunk.len() > MAX_WORD {
            return None;
        }
        let input = self.chunk_input(chunk, letters);
        let mut out = BitVector::new(WORD_BITS, Some(0));
        if self.class.process_predictive(&input, &mut out) > 0 { self.words.decode(&out) } else { None }
    }

    /// Recognize (or learn) a chunk. Returns (word id, recognized-before?).
    fn read(&mut self, chunk: &[usize], letters: &Encoder, rng: &mut StdRng) -> Option<(usize, bool)> {
        if let Some(id) = self.lookup(chunk, letters) {
            self.seen[id] += 1;
            return Some((id, true));
        }
        if chunk.is_empty() || chunk.len() > MAX_WORD {
            return None;
        }
        let input = self.chunk_input(chunk, letters);
        let id = self.words.push(rng, WORD_ACTIVE);
        let code = self.words.codes[id].clone();
        self.class.grow(&input, &code, chunk.len() + 1, rng);
        self.spelling.push(chunk.iter().map(|&c| LETTERS.as_bytes()[c] as char).collect());
        self.seen.push(1);
        Some((id, false))
    }
}

/// Segment online, growing the lexicon as chunks are cut. Cuts at local minima
/// of P(next letter); with `assist = Some((min_seen, p_max))` also cuts when the
/// chunk so far (3+ letters) is a word recognized at least `min_seen` times and the next
/// letter is not strongly expected (P < p_max). Returns cuts, lexicon, and how
/// many second-half chunks were recognized.
fn segment(
    stream: &[usize],
    prob: &[f32],
    assist: Option<(usize, f32)>,
    letters: &Encoder,
    rng: &mut StdRng,
    half: usize,
) -> (Vec<bool>, Lexicon, usize, usize, Vec<String>) {
    let n = stream.len();
    let minima = local_minima(prob);
    let mut lex = Lexicon::new(rng);
    let mut cuts = vec![false; n];
    let (mut start, mut tokens, mut recognized) = (0usize, 0usize, 0usize);
    let mut sample = Vec::new();
    for i in 0..n {
        let mut cut = minima[i] || i == n - 1;
        if !cut {
            if let Some((min_seen, p_max)) = assist {
                if prob[i] < p_max {
                    // only words of 3+ letters: shorter chunks snowball into letter-by-letter cuts
                    let chunk = &stream[start..=i];
                    if let Some(id) = if chunk.len() >= 3 { lex.lookup(chunk, letters) } else { None } {
                        cut = lex.seen[id] >= min_seen;
                    }
                }
            }
        }
        if cut {
            cuts[i] = true;
            let chunk = &stream[start..=i];
            let res = lex.read(chunk, letters, rng);
            let known = matches!(res, Some((_, true)));
            if start >= half {
                tokens += 1;
                recognized += known as usize;
            }
            if start >= n - 400 && sample.len() < 40 {
                let w: String = chunk.iter().map(|&c| LETTERS.as_bytes()[c] as char).collect();
                sample.push(if known { w } else { format!("{w}*") });
            }
            start = i + 1;
        }
    }
    (cuts, lex, tokens, recognized, sample)
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "data/alice.txt".into());
    let text = normalize(&read_corpus(&path));
    let (stream, truth) = letters_and_boundaries(&text);
    let n = stream.len();
    let half = n / 2;
    let true_words = truth.iter().filter(|&&b| b).count();
    println!("{path}: {n} letters, {true_words} words; spaces and punctuation removed");
    println!("scored on the second half (after reading the first half)");
    println!();

    let mut rng = StdRng::seed_from_u64(11);
    let letters = Encoder::new(LETTERS.len(), CHAR_BITS, CHAR_ACTIVE, &mut rng);

    // --- 1. per-position signals ------------------------------------------
    // tp[i]   = P(letter i+1 | letter i), online counts (baseline)
    // conf[i] = network confidence in its guess for letter i+1 (0 if no guess)
    let mut counts = vec![vec![0u32; 26]; 26];
    let mut tp = vec![0f32; n];
    for i in 0..n - 1 {
        let (a, b) = (stream[i], stream[i + 1]);
        let total: u32 = counts[a].iter().sum();
        tp[i] = if total == 0 { 0.0 } else { counts[a][b] as f32 / total as f32 };
        counts[a][b] += 1;
    }

    let frames = 8;
    let mut p = Predictor::new(CHAR_BITS, frames, 1_000_000);
    let mut conf = vec![0f32; n];
    let mut prob = vec![0f32; n]; // network's P(actual next letter)
    let mut correct = 0usize;
    for i in 0..n - 1 {
        p.step(&letters.codes[stream[i]]);
        conf[i] = p.class().confidence().unwrap_or(0.0);
        if i >= half && letters.decode(p.prediction()) == Some(stream[i + 1]) {
            correct += 1;
        }
        p.learn(&letters.codes[stream[i + 1]]);
        prob[i] = p.class().target_probability();
    }
    println!(
        "next-letter accuracy without spaces (2nd half): {:.1}%   [{} kernels, {frames}-letter context]",
        100.0 * correct as f64 / (n - 1 - half) as f64,
        p.class().len()
    );
    println!();

    // --- 2. segmentation ----------------------------------------------------
    let rate = true_words as f64 / n as f64;
    let random: Vec<bool> = (0..n).map(|_| rng.gen_bool(rate)).collect();
    let mut candidates: Vec<(String, Vec<bool>)> = vec![
        ("random cuts (true word rate)".into(), random),
        ("transitional prob: local minima".into(), local_minima(&tp)),
    ];
    for th in [0.1f32, 0.2] {
        candidates.push((format!("transitional prob: < {th}"), below(&tp, th)));
    }
    candidates.push(("network confidence: local minima".into(), local_minima(&conf)));
    candidates.push(("network P(next): local minima".into(), local_minima(&prob)));
    for th in [0.2f32, 0.3] {
        candidates.push((format!("network P(next): < {th}"), below(&prob, th)));
    }
    candidates.push(("network confidence: < 0.5".into(), below(&conf, 0.5)));

    println!("segmentation (second half)          boundaries               word tokens");
    let mut best: Option<(f64, usize)> = None;
    for (i, (name, cuts)) in candidates.iter().enumerate() {
        let (b, w) = score(cuts, &truth, half, n);
        println!("  {name:<34} {}   {}", b.show(), w.show());
        if name.starts_with("network") && best.map_or(true, |(f, _)| w.f1() > f) {
            best = Some((w.f1(), i));
        }
    }
    println!();

    // --- 3. recognition, and recognition feeding back into segmentation ------
    let true_types: HashSet<String> = {
        let mut set = HashSet::new();
        let mut w = String::new();
        for (i, &c) in stream.iter().enumerate() {
            w.push(LETTERS.as_bytes()[c] as char);
            if truth[i] {
                set.insert(std::mem::take(&mut w));
            }
        }
        set
    };
    println!("segmentation + word recognition (online, lexicon grown from the cuts):");
    let mut variants: Vec<(String, Option<(usize, f32)>)> = vec![("P(next) minima only".into(), None)];
    for (min_seen, p_max) in [(10usize, 0.3f32), (10, 0.5), (30, 0.5)] {
        variants.push((format!("+ known word (seen>={min_seen}) & P<{p_max}"), Some((min_seen, p_max))));
    }
    for (name, assist) in variants {
        let (cuts, lex, tokens, recognized, sample) = segment(&stream, &prob, assist, &letters, &mut rng, half);
        let (b, w) = score(&cuts, &truth, half, n);
        let entries = lex.spelling.len();
        let real = lex.spelling.iter().filter(|w| true_types.contains(*w)).count();
        let frequent: Vec<usize> = (0..entries).filter(|&i| lex.seen[i] >= 5).collect();
        let frequent_real = frequent.iter().filter(|&&i| true_types.contains(&lex.spelling[i])).count();
        let tokens_real: usize = (0..entries).filter(|&i| true_types.contains(&lex.spelling[i])).map(|i| lex.seen[i]).sum();
        let tokens_all: usize = lex.seen.iter().sum();
        println!("  {name}");
        println!("    boundaries {}   word tokens {}", b.show(), w.show());
        println!(
            "    lexicon {entries} entries ({:.1}% real words); seen>=5: {} ({:.1}% real); {:.1}% of chunk tokens are real words; 2nd half {:.1}% recognized",
            100.0 * real as f64 / entries.max(1) as f64,
            frequent.len(),
            100.0 * frequent_real as f64 / frequent.len().max(1) as f64,
            100.0 * tokens_real as f64 / tokens_all.max(1) as f64,
            100.0 * recognized as f64 / tokens.max(1) as f64
        );
        let mut top: Vec<(usize, &String)> = frequent.iter().map(|&i| (lex.seen[i], &lex.spelling[i])).collect();
        top.sort_by(|a, b| b.0.cmp(&a.0));
        println!("    top: {}", top.iter().take(20).map(|(k, w)| format!("{w}({k})")).collect::<Vec<_>>().join(" "));
        println!("    end of book (* = new word): {}", sample.join(" | "));
    }
}
