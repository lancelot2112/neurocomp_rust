//! Does top-down bias from a higher layer help the character predictor?
//!
//! Run:  ./scripts/fetch_corpora.sh   (once)
//!       cargo run --release --example topdown [max_chars]
//!
//! Two layers read Alice (with spaces) online:
//! - a character layer: predictive KernelClass over the last 6 characters;
//! - a word layer: predictive KernelClass over the last 2 words, run at each word
//!   end to predict the next word.
//! Top-down signal: while a word is being read, if the predicted next word still
//! fits the letters read so far, its next letter (or a space once it is complete)
//! is the word layer's expected next character. That expectation is handed to the
//! character layer as a bias (`process_predictive_biased`) in one of four modes.
//! Learning is unchanged in all conditions; only the choice of prediction differs.

mod common;

use std::collections::HashMap;
use std::time::Instant;

use common::{normalize, read_corpus, Encoder, KEEP};
use neurocomp::bitvec::BitVector;
use neurocomp::kernel::{BiasMode, GrowthConfig, KernelClass, SimpleKernel};
use rand::rngs::StdRng;
use rand::SeedableRng;

const CHAR_BITS: usize = 512;
const WORD_BITS: usize = 1024;
const CHAR_FRAMES: usize = 6;
const WORD_FRAMES: usize = 2;

fn predictive(bits: usize, frames: usize) -> KernelClass<SimpleKernel> {
    KernelClass::predictive(GrowthConfig {
        max_kernels: 1_000_000,
        frame_words: bits / 64,
        max_frames: frames,
        sample_bits: 16,
        match_fraction: 0.8,
        surprise_fraction: 0.5,
        generalize: None,
    })
}

/// Concatenate the codes of `hist` (most recent last) into `frames` frames, most recent first.
fn frames_input(hist: &[usize], codes: &[BitVector], frames: usize, bits: usize) -> BitVector {
    let mut words = Vec::with_capacity(frames * bits / 64);
    for f in 0..frames {
        match hist.len().checked_sub(f + 1) {
            Some(i) => words.extend_from_slice(codes[hist[i]].as_words()),
            None => words.extend(std::iter::repeat(0).take(bits / 64)),
        }
    }
    BitVector::from_words(words)
}

/// The word layer: predicts the next word at each word boundary.
struct WordLayer {
    class: KernelClass<SimpleKernel>,
    codes: Encoder,
    ids: HashMap<String, usize>,
    spelling: Vec<String>,
    hist: Vec<usize>,
    last_input: Option<BitVector>,
    predicted: Option<usize>,
    expected_word: Option<String>, // what `expect` spells from: the prediction, or the oracle's word
    hits: usize,
    total: usize,
}

impl WordLayer {
    fn new(rng: &mut StdRng) -> Self {
        Self {
            class: predictive(WORD_BITS, WORD_FRAMES),
            codes: Encoder::new(0, WORD_BITS, 32, rng),
            ids: HashMap::new(),
            spelling: Vec::new(),
            hist: Vec::new(),
            last_input: None,
            predicted: None,
            expected_word: None,
            hits: 0,
            total: 0,
        }
    }

    /// A word just ended: learn from it, then predict the next one.
    fn word_done(&mut self, word: &str, rng: &mut StdRng) {
        let id = match self.ids.get(word) {
            Some(&i) => i,
            None => {
                let i = self.codes.push(rng, 32);
                self.ids.insert(word.to_string(), i);
                self.spelling.push(word.to_string());
                i
            }
        };
        self.total += 1;
        self.hits += (self.predicted == Some(id)) as usize;
        if let Some(input) = &self.last_input {
            self.class.feedback(input, &self.codes.codes[id].clone(), rng);
        }
        self.hist.push(id);
        let input = frames_input(&self.hist, &self.codes.codes, WORD_FRAMES, WORD_BITS);
        let mut out = BitVector::new(WORD_BITS, Some(0));
        self.predicted = if self.class.process_predictive(&input, &mut out) > 0 { self.codes.decode(&out) } else { None };
        self.expected_word = self.predicted.map(|i| self.spelling[i].clone());
        self.last_input = Some(input);
    }

    /// Expected next character given the letters of the current word so far.
    fn expect(&self, prefix: &str, current: char) -> Option<char> {
        let w = self.expected_word.as_ref()?;
        if prefix.is_empty() {
            // between words: after a space the predicted word should start
            return if current == ' ' { w.chars().next() } else { None };
        }
        if !w.starts_with(prefix) {
            return None;
        }
        Some(w[prefix.len()..].chars().next().unwrap_or(' '))
    }
}

#[derive(Default, Clone, Copy)]
struct Acc {
    hit: usize,
    n: usize,
}
impl Acc {
    fn add(&mut self, ok: bool) {
        self.n += 1;
        self.hit += ok as usize;
    }
    fn pct(&self) -> f64 {
        100.0 * self.hit as f64 / self.n.max(1) as f64
    }
}

struct Outcome {
    all: Acc,
    by_pos: [Acc; 3], // next char is: first letter of a word, later letter, space/punctuation
    topdown: Acc,     // was the word layer's expectation right (when it had one)
    coverage: Acc,    // how often it had one
    words: Acc,       // word layer next-word accuracy
    secs: f64,
}

/// `gate`: pass the bias down only when the word layer's confidence in its
/// prediction is at least this (precision weighting). `oracle`: the word layer
/// is told the true next word instead of predicting it (an upper bound).
fn run(text: &[char], alphabet: &[char], mode: Option<BiasMode>, gate: Option<f32>, oracle: bool, seed: u64) -> Outcome {
    let mut rng = StdRng::seed_from_u64(seed);
    let chars = Encoder::new(alphabet.len(), CHAR_BITS, 32, &mut rng);
    let idx = |c: char| alphabet.iter().position(|&a| a == c).unwrap();
    let mut class = predictive(CHAR_BITS, CHAR_FRAMES);
    let mut words = WordLayer::new(&mut rng);

    // the true word sequence, for the oracle
    let true_words: Vec<String> = text
        .iter()
        .collect::<String>()
        .split(|c: char| !c.is_ascii_lowercase())
        .filter(|w| !w.is_empty())
        .map(String::from)
        .collect();
    let mut words_done = 0usize;
    let mut hist: Vec<usize> = Vec::new();
    let mut prefix = String::new();
    let (mut all, mut by_pos, mut td, mut td_cover) = (Acc::default(), [Acc::default(); 3], Acc::default(), Acc::default());
    let start = Instant::now();
    for t in 0..text.len() - 1 {
        let c = text[t];
        if c.is_ascii_lowercase() {
            prefix.push(c);
        } else if !prefix.is_empty() {
            let w = std::mem::take(&mut prefix);
            words.word_done(&w, &mut rng);
            words_done += 1;
            if oracle {
                words.expected_word = true_words.get(words_done).cloned();
            }
        }
        hist.push(idx(c));
        let input = frames_input(&hist, &chars.codes, CHAR_FRAMES, CHAR_BITS);

        let expected = words.expect(&prefix, c);
        let confident = oracle || gate.map_or(true, |g| words.class.confidence().unwrap_or(0.0) >= g);
        let bias_code = expected.map(|e| chars.codes[idx(e)].clone());
        let mut out = BitVector::new(CHAR_BITS, Some(0));
        let bias = match (mode, &bias_code) {
            (Some(m), Some(b)) if confident => Some((b, m)),
            _ => None,
        };
        class.process_predictive_biased(&input, &mut out, bias);

        let next = text[t + 1];
        let ok = chars.decode(&out) == Some(idx(next));
        all.add(ok);
        // 0 = first letter of a word, 1 = later letter, 2 = space/punctuation
        let pos = if next.is_ascii_lowercase() { if prefix.is_empty() { 0 } else { 1 } } else { 2 };
        by_pos[pos].add(ok);
        td_cover.add(expected.is_some());
        if let Some(e) = expected {
            td.add(e == next);
        }
        class.feedback(&input, &chars.codes[idx(next)], &mut rng);
    }
    Outcome {
        all,
        by_pos,
        topdown: td,
        coverage: td_cover,
        words: Acc { hit: words.hits, n: words.total },
        secs: start.elapsed().as_secs_f64(),
    }
}

fn main() {
    let max: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(60_000);
    let mut text = normalize(&read_corpus("data/alice.txt"));
    text.truncate(max);
    let text: Vec<char> = text.chars().collect();
    let alphabet: Vec<char> = KEEP.chars().collect();
    println!("Alice, first {} chars, read once online; {CHAR_FRAMES}-char layer + {WORD_FRAMES}-word layer", text.len());
    println!();
    let conditions: [(&str, Option<BiasMode>, Option<f32>, bool); 8] = [
        ("no top-down bias", None, None, false),
        ("SameDepth", Some(BiasMode::SameDepth), None, false),
        ("Prefer", Some(BiasMode::Prefer), None, false),
        ("Prefer, only if word layer confidence >= 0.5", Some(BiasMode::Prefer), Some(0.5), false),
        ("SameDepth, only if word layer confidence >= 0.5", Some(BiasMode::SameDepth), Some(0.5), false),
        ("oracle word layer + SameDepth", Some(BiasMode::SameDepth), None, true),
        ("oracle word layer + Prefer", Some(BiasMode::Prefer), None, true),
        ("oracle word layer + PreferAndFallback", Some(BiasMode::PreferAndFallback), None, true),
    ];
    for (name, mode, gate, oracle) in conditions {
        println!("  {name}");
        let o = run(&text, &alphabet, mode, gate, oracle, 1);
        println!(
            "      char accuracy {:5.1}%   word-initial {:5.1}%   inside word {:5.1}%   space/punct {:5.1}%   [{:.0}s]",
            o.all.pct(),
            o.by_pos[0].pct(),
            o.by_pos[1].pct(),
            o.by_pos[2].pct(),
            o.secs
        );
        if mode.is_none() {
            println!(
                "      word layer: next word {:.1}%; its expected next char is available at {:.1}% of positions and right {:.1}% of those",
                o.words.pct(),
                o.coverage.pct(),
                o.topdown.pct()
            );
        }
    }
}
