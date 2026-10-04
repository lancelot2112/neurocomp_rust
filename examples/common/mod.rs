//! Helpers shared by the corpus examples.
#![allow(dead_code)]

use neurocomp::bitvec::{AdvanceMode, BitVector};
use neurocomp::kernel::{GrowthConfig, KernelClass, KernelGroup, SimpleKernel};
use neurocomp::program::{GraphProgram, ProgramDefaults, RuntimeNetwork};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

pub const KEEP: &str = "abcdefghijklmnopqrstuvwxyz .,;:!?'-\"";
pub const LETTERS: &str = "abcdefghijklmnopqrstuvwxyz";

pub fn read_corpus(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e} (run ./scripts/fetch_corpora.sh)"))
}

/// Lowercase, keep letters/space/basic punctuation, collapse whitespace.
pub fn normalize(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut last_space = true;
    for c in raw.chars().flat_map(|c| c.to_lowercase()) {
        let c = if c.is_whitespace() { ' ' } else { c };
        if !KEEP.contains(c) || (c == ' ' && last_space) {
            continue;
        }
        last_space = c == ' ';
        out.push(c);
    }
    out
}

/// Fixed random sparse codes, one per symbol.
pub struct Encoder {
    pub bits: usize,
    pub codes: Vec<BitVector>,
}

impl Encoder {
    pub fn new(n_symbols: usize, bits: usize, active: usize, rng: &mut StdRng) -> Self {
        let positions: Vec<usize> = (0..bits).collect();
        let codes = (0..n_symbols)
            .map(|_| BitVector::from_bits(&positions.choose_multiple(rng, active).cloned().collect::<Vec<_>>(), bits))
            .collect();
        Self { bits, codes }
    }

    pub fn push(&mut self, rng: &mut StdRng, active: usize) -> usize {
        let positions: Vec<usize> = (0..self.bits).collect();
        self.codes.push(BitVector::from_bits(&positions.choose_multiple(rng, active).cloned().collect::<Vec<_>>(), self.bits));
        self.codes.len() - 1
    }

    /// Symbol whose code overlaps `out` the most (None if nothing overlaps).
    pub fn decode(&self, out: &BitVector) -> Option<usize> {
        let words = out.as_words();
        let overlaps: Vec<u32> =
            self.codes.iter().map(|c| c.as_words().iter().zip(words).map(|(a, b)| (a & b).count_ones()).sum()).collect();
        argmax(&overlaps)
    }
}

pub fn argmax(v: &[u32]) -> Option<usize> {
    if v.iter().all(|&k| k == 0) {
        return None;
    }
    v.iter().enumerate().max_by_key(|(_, k)| **k).map(|(i, _)| i)
}

/// input -> output network whose single edge reads the last `frames` input frames
/// through one predictive kernel class.
pub struct Predictor {
    pub net: RuntimeNetwork,
    pub out: usize,
}

impl Predictor {
    pub fn new(bits: usize, frames: usize, budget: usize) -> Self {
        let defaults = ProgramDefaults { default_bits: bits, ..ProgramDefaults::default() };
        let mut g = GraphProgram::new().build_graph(&defaults);
        for e in g.edge_weights_mut() {
            e.input_frames = frames;
        }
        let mut net = RuntimeNetwork::from_graph(&g);
        let mut group = KernelGroup::new();
        group.add_class(KernelClass::predictive(GrowthConfig {
            max_kernels: budget,
            frame_words: bits / 64,
            max_frames: frames,
            sample_bits: 16,
            match_fraction: 0.8,
            surprise_fraction: 0.5,
            generalize: None,
            generalize_after: 1,
        }));
        net.edges[0].group = group;
        let out = net.nodes.len() - 1;
        Self { net, out }
    }

    /// Feed one symbol; afterwards `prediction()` holds the guess for the next one.
    pub fn step(&mut self, code: &BitVector) {
        self.net.write_input(code, 0);
        self.net.tick(AdvanceMode::Clear, 0);
    }

    pub fn prediction(&self) -> &BitVector {
        self.net.read_node(self.out)
    }

    /// Learn from what actually came next.
    pub fn learn(&mut self, next: &BitVector) {
        self.net.learn(self.out, next);
    }

    pub fn class(&self) -> &KernelClass<SimpleKernel> {
        &self.net.edges[0].group.classes()[0]
    }
}

/// Collapse Brown tags into 12 coarse parts of speech.
pub fn coarse_tag(tag: &str) -> &'static str {
    if tag.starts_with("fw-") {
        return "X";
    }
    let t = tag.split('-').next().unwrap_or(tag).trim_end_matches(['*', '$']);
    let any = |ps: &[&str]| ps.iter().any(|p| t.starts_with(p));
    if t.is_empty() || t == "*" || tag == "*" {
        return "ADV"; // "not", "n't"
    }
    if any(&["wdt"]) || any(&["at", "dt", "ap", "abn", "abx", "abl"]) {
        "DET"
    } else if any(&["nn", "np", "nr"]) {
        "NOUN"
    } else if any(&["vb", "be", "hv", "do", "md"]) {
        "VERB"
    } else if any(&["jj"]) {
        "ADJ"
    } else if any(&["rb", "ql", "wrb", "rn"]) {
        "ADV"
    } else if any(&["pp", "pn", "wp", "ex"]) {
        "PRON"
    } else if any(&["in"]) {
        "ADP"
    } else if any(&["cc", "cs"]) {
        "CONJ"
    } else if any(&["cd", "od"]) {
        "NUM"
    } else if any(&["rp", "to"]) {
        "PRT"
    } else if t.chars().all(|c| !c.is_ascii_alphanumeric()) {
        "PUNCT"
    } else {
        "X"
    }
}

/// (word, coarse tag) per token, sentences concatenated.
pub fn load_brown(path: &str) -> Vec<(String, &'static str)> {
    let text = read_corpus(path);
    let mut out = Vec::new();
    for tok in text.split_whitespace() {
        if let Some((w, t)) = tok.rsplit_once('/') {
            out.push((w.to_lowercase(), coarse_tag(t)));
        }
    }
    out
}
