//! Next-character prediction on a real book, read once.
//!
//! Run:  ./scripts/fetch_corpora.sh   (once)
//!       cargo run --release --example read_corpus [path] [max_chars]
//!
//! The text (default data/alice.txt) is lowercased and reduced to letters, space
//! and basic punctuation, then streamed one character per tick through a
//! predictive RuntimeNetwork (input history -> output). Every prediction is made
//! before the character is seen and learning is online, so all scores are on text
//! the model has not seen yet ("prequential" evaluation). The same protocol is
//! used for the n-gram baselines.

mod common;

use std::collections::HashMap;
use std::time::Instant;

use common::{argmax, normalize, read_corpus, Encoder, Predictor, KEEP};
use rand::rngs::StdRng;
use rand::SeedableRng;

const INPUT_BITS: usize = 512;
const CHAR_BITS: usize = 32;

/// Accuracy per tenth of the stream (a learning curve).
struct Curve {
    hits: [usize; 10],
    total: [usize; 10],
    len: usize,
}

impl Curve {
    fn new(len: usize) -> Self {
        Self { hits: [0; 10], total: [0; 10], len }
    }
    fn add(&mut self, t: usize, ok: bool) {
        let d = (t * 10 / self.len).min(9);
        self.total[d] += 1;
        self.hits[d] += ok as usize;
    }
    fn pct(&self, d: usize) -> f64 {
        100.0 * self.hits[d] as f64 / self.total[d].max(1) as f64
    }
    fn overall(&self) -> f64 {
        100.0 * self.hits.iter().sum::<usize>() as f64 / self.total.iter().sum::<usize>().max(1) as f64
    }
    fn row(&self) -> String {
        (0..10).map(|d| format!("{:5.1}", self.pct(d))).collect::<Vec<_>>().join(" ")
    }
}

/// Counts of next char per context of each order 1..=max_order.
struct NGrams {
    tables: Vec<HashMap<Vec<usize>, Vec<u32>>>, // tables[o-1] for order o
    n_chars: usize,
}

impl NGrams {
    fn new(max_order: usize, n_chars: usize) -> Self {
        Self { tables: vec![HashMap::new(); max_order], n_chars }
    }
    fn predict_order(&self, hist: &[usize], o: usize) -> Option<usize> {
        if hist.len() < o {
            return None;
        }
        self.tables[o - 1].get(&hist[hist.len() - o..]).and_then(|row| argmax(row))
    }
    /// Longest context seen before wins (back-off).
    fn predict_backoff(&self, hist: &[usize]) -> Option<usize> {
        (1..=self.tables.len()).rev().find_map(|o| self.predict_order(hist, o))
    }
    fn learn(&mut self, hist: &[usize], next: usize) {
        for o in 1..=self.tables.len().min(hist.len()) {
            let n = self.n_chars;
            self.tables[o - 1].entry(hist[hist.len() - o..].to_vec()).or_insert_with(|| vec![0; n])[next] += 1;
        }
    }
    fn entries(&self) -> usize {
        self.tables.iter().map(|t| t.values().map(|row| row.iter().filter(|&&k| k > 0).count()).sum::<usize>()).sum()
    }
}

fn run_network(enc: &Encoder, stream: &[usize], frames: usize, budget: usize) -> (Curve, usize, usize, Vec<usize>, f64) {
    let mut p = Predictor::new(INPUT_BITS, frames, budget);
    let start = Instant::now();
    let mut curve = Curve::new(stream.len() - 1);
    for t in 0..stream.len() - 1 {
        p.step(&enc.codes[stream[t]]);
        let next = stream[t + 1];
        curve.add(t, enc.decode(p.prediction()) == Some(next));
        p.learn(&enc.codes[next]);
    }
    let secs = start.elapsed().as_secs_f64();
    let kc = p.class();
    let mut by_depth = vec![0; frames];
    for k in kc.kernels() {
        by_depth[k.context_frames - 1] += 1;
    }
    (curve, kc.len(), kc.recycled(), by_depth, secs)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(String::as_str).unwrap_or("data/alice.txt");
    let mut text = normalize(&read_corpus(path));
    if let Some(max) = args.get(2).and_then(|s| s.parse::<usize>().ok()) {
        text.truncate(max);
    }

    let mut rng = StdRng::seed_from_u64(7);
    let alphabet: Vec<char> = KEEP.chars().collect();
    let enc = Encoder::new(alphabet.len(), INPUT_BITS, CHAR_BITS, &mut rng);
    let stream: Vec<usize> = text.chars().map(|c| alphabet.iter().position(|&a| a == c).unwrap()).collect();
    println!("{path}: {} chars after normalizing, alphabet {}, read once, scored online", stream.len(), alphabet.len());
    println!("accuracy per tenth of the text, then overall:");
    println!();

    let max_order = 8;
    let mut ng = NGrams::new(max_order, alphabet.len());
    let mut fixed: Vec<Curve> = (0..max_order).map(|_| Curve::new(stream.len() - 1)).collect();
    let mut backoff = Curve::new(stream.len() - 1);
    for t in 0..stream.len() - 1 {
        let hist = &stream[..=t];
        let next = stream[t + 1];
        for o in 1..=max_order {
            fixed[o - 1].add(t, ng.predict_order(hist, o) == Some(next));
        }
        backoff.add(t, ng.predict_backoff(hist) == Some(next));
        ng.learn(hist, next);
    }
    for o in [1, 2, 3, 4, 6, 8] {
        println!("  {o}-char n-gram            {}  | {:5.1}%", fixed[o - 1].row(), fixed[o - 1].overall());
    }
    println!(
        "  back-off n-gram (<=8)     {}  | {:5.1}%   ({} context->char entries)",
        backoff.row(),
        backoff.overall(),
        ng.entries()
    );
    println!();

    for (frames, budget) in [(4usize, 1_000_000usize), (6, 1_000_000), (8, 1_000_000), (8, 50_000), (8, 10_000)] {
        let (curve, live, recycled, by_depth, secs) = run_network(&enc, &stream, frames, budget);
        println!("  network {frames} frames, budget {budget:>7} {}  | {:5.1}%", curve.row(), curve.overall());
        println!("      {live} kernels ({recycled} recycled) by depth {by_depth:?}, {secs:.1}s");
    }
}
