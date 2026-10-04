//! Reading experiment: can the current network learn next-character prediction?
//!
//! Run with: cargo run --release --example read_text
//!
//! Characters are encoded as fixed random sparse bit patterns and fed in one per
//! tick. Networks A-C only learn with the local Hebbian kernel rules, so they have
//! no output that names a character; we measure how much next-character
//! information their hidden state carries with an external "probe": a vote table
//! counts[bit][next_char] built online from the same stream. The probe is NOT part
//! of the network; it only reads it.
//!
//! Network D is predictive: its output node is trained (`RuntimeNetwork::learn`)
//! to hold the next character's code, with surprise-driven kernel growth and
//! recycling. It is scored directly by decoding its own output.
//!
//! Two scores per model: the last training epoch (memorization of seen text) and a
//! held-out sentence made of the same words in a new order (generalization).
//!
//! Baselines use the same online protocol (predict, then update counts):
//! - unigram / n-gram tables over the raw characters
//! - the same probe applied to the raw input encoding (a bigram-like ceiling
//!   for a network with no context)

use std::collections::HashMap;

use neurocomp::bitvec::{AdvanceMode, BitVector};
use neurocomp::kernel::{GrowthConfig, KernelClass, KernelGroup, KernelOp, SimpleKernel};
use neurocomp::program::{GraphProgram, ProgramDefaults, RuntimeNetwork};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

const CORPUS: &str = "the cat sat on the mat. the dog sat on the log. \
the cat saw the dog and the dog saw the cat. a bird sang in the tree \
and the cat looked up at the bird. the dog ran to the tree and the bird \
flew away. then the cat and the dog sat in the sun on the mat by the log. ";

/// Seen words, new order. Fed once after training; learning stays on (online).
const HELDOUT: &str = "the bird sat on the log and the dog saw the sun. the cat ran to the mat. ";

const EPOCHS: usize = 40;
const INPUT_BITS: usize = 512; // 8 words
const CHAR_BITS: usize = 32; // active bits per character (~6% density)
const HIDDEN_BITS: usize = 512; // 8 words
const KERNELS_PER_EDGE: usize = 256;
const MASK_BITS: usize = 24; // connected bits per kernel (of a 64-bit window)

// ---------------------------------------------------------------------------
// Encoding

struct Encoder {
    alphabet: Vec<char>,
    codes: Vec<BitVector>,
}

impl Encoder {
    fn new(text: &str, rng: &mut StdRng) -> Self {
        let mut alphabet: Vec<char> = text.chars().collect();
        alphabet.sort();
        alphabet.dedup();
        let positions: Vec<usize> = (0..INPUT_BITS).collect();
        let codes = alphabet
            .iter()
            .map(|_| {
                let bits: Vec<usize> = positions.choose_multiple(rng, CHAR_BITS).cloned().collect();
                BitVector::from_bits(&bits, INPUT_BITS)
            })
            .collect();
        Self { alphabet, codes }
    }

    fn index(&self, c: char) -> usize {
        self.alphabet.iter().position(|&a| a == c).unwrap()
    }
}

// ---------------------------------------------------------------------------
// Probes and baselines (all online: predict first, then learn)

/// Votes over active bits: counts[bit][next_char].
struct BitVoteProbe {
    counts: Vec<Vec<u32>>,
}

impl BitVoteProbe {
    fn new(bits: usize, n_chars: usize) -> Self {
        Self { counts: vec![vec![0; n_chars]; bits] }
    }

    fn predict(&self, state: &BitVector) -> Option<usize> {
        let n = self.counts[0].len();
        let mut score = vec![0f64; n];
        let mut any = false;
        for b in active_bits(state) {
            let row = &self.counts[b];
            let total: u32 = row.iter().sum();
            if total == 0 {
                continue;
            }
            any = true;
            for (c, &k) in row.iter().enumerate() {
                score[c] += k as f64 / total as f64;
            }
        }
        if !any {
            return None;
        }
        argmax_f(&score)
    }

    fn learn(&mut self, state: &BitVector, next: usize) {
        for b in active_bits(state) {
            self.counts[b][next] += 1;
        }
    }
}

/// Exact-state lookup: how well does the *whole* state identify the context?
#[derive(Default)]
struct StateLookupProbe {
    table: HashMap<Vec<u64>, Vec<u32>>,
}

impl StateLookupProbe {
    fn predict(&self, state: &BitVector) -> Option<usize> {
        self.table.get(state.as_words()).and_then(|row| argmax_u(row))
    }

    fn learn(&mut self, state: &BitVector, next: usize, n_chars: usize) {
        self.table.entry(state.as_words().to_vec()).or_insert_with(|| vec![0; n_chars])[next] += 1;
    }
}

/// Character n-gram (context of `order` previous chars incl. the current one).
struct NGram {
    order: usize,
    table: HashMap<Vec<usize>, Vec<u32>>,
}

impl NGram {
    fn predict(&self, ctx: &[usize]) -> Option<usize> {
        if ctx.len() < self.order {
            return None;
        }
        self.table.get(&ctx[ctx.len() - self.order..]).and_then(|row| argmax_u(row))
    }

    fn learn(&mut self, ctx: &[usize], next: usize, n_chars: usize) {
        if ctx.len() < self.order {
            return;
        }
        let key = ctx[ctx.len() - self.order..].to_vec();
        self.table.entry(key).or_insert_with(|| vec![0; n_chars])[next] += 1;
    }
}

fn active_bits(bv: &BitVector) -> impl Iterator<Item = usize> + '_ {
    bv.as_words().iter().enumerate().flat_map(|(wi, &w)| {
        let mut w = w;
        std::iter::from_fn(move || {
            if w == 0 {
                return None;
            }
            let b = w.trailing_zeros() as usize;
            w &= w - 1;
            Some(wi * 64 + b)
        })
    })
}

fn argmax_f(v: &[f64]) -> Option<usize> {
    v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i)
}

fn argmax_u(v: &[u32]) -> Option<usize> {
    if v.iter().all(|&k| k == 0) {
        return None;
    }
    v.iter().enumerate().max_by_key(|(_, k)| **k).map(|(i, _)| i)
}

// ---------------------------------------------------------------------------
// Scoring

#[derive(Default, Clone, Copy)]
struct Score {
    hits: usize,
    total: usize,
}

impl Score {
    fn add(&mut self, pred: Option<usize>, actual: usize) {
        self.total += 1;
        if pred == Some(actual) {
            self.hits += 1;
        }
    }
    fn pct(&self) -> f64 {
        if self.total == 0 { 0.0 } else { 100.0 * self.hits as f64 / self.total as f64 }
    }
}

// ---------------------------------------------------------------------------
// Networks

/// Random kernels reading `src_words` of a source node and writing one bit each
/// into a destination node. Uses SimpleKernel and its built-in learning rules.
fn random_kernel_class(
    rng: &mut StdRng,
    src_words: usize,
    dst_bits: usize,
    threshold: usize,
) -> KernelClass<SimpleKernel> {
    let mut out_bits: Vec<usize> = (0..dst_bits).collect();
    out_bits.shuffle(rng);
    let kernels = (0..KERNELS_PER_EDGE)
        .map(|i| {
            let positions: Vec<usize> = (0..64).collect();
            let conn: Vec<usize> = positions.choose_multiple(rng, MASK_BITS).cloned().collect();
            let input_mask = BitVector::from_bits(&conn, 64);
            let input_idx = rng.gen_range(0..src_words);
            let out_bit = out_bits[i % out_bits.len()];
            let output_mask = BitVector::from_bits(&[out_bit & 63], 64);
            SimpleKernel::new(input_mask, input_idx, output_mask, out_bit >> 6, threshold, KernelOp::Or)
        })
        .collect();
    KernelClass::with_kernels(kernels)
}

/// Scores for the last training epoch and for the held-out text.
#[derive(Default, Clone, Copy)]
struct Split {
    train: Score,
    heldout: Score,
}

impl Split {
    fn add(&mut self, seg: Option<Seg>, pred: Option<usize>, actual: usize) {
        match seg {
            Some(Seg::Train) => self.train.add(pred, actual),
            Some(Seg::Heldout) => self.heldout.add(pred, actual),
            None => {}
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Seg {
    Train,
    Heldout,
}

/// Which scoring segment (if any) a prediction made at tick `t` belongs to.
struct Segments {
    eval_from: usize,
    heldout_from: usize,
}

impl Segments {
    fn of(&self, t: usize) -> Option<Seg> {
        if t >= self.heldout_from {
            Some(Seg::Heldout)
        } else if t >= self.eval_from {
            Some(Seg::Train)
        } else {
            None
        }
    }
}

struct Stats {
    probe: Split,
    lookup: Split,
    density: f64,
    distinct_states: usize,
    ever_fired: usize,
    always_on: usize,
}

/// Run the stream through `net`, probing `probe_node` after every tick.
fn run_network(
    net: &mut RuntimeNetwork,
    probe_node: usize,
    enc: &Encoder,
    stream: &[usize],
    segs: &Segments,
) -> Stats {
    let n_chars = enc.alphabet.len();
    let hidden_bits = net.read_node(probe_node).bit_len();
    let mut probe = BitVoteProbe::new(hidden_bits, n_chars);
    let mut lookup = StateLookupProbe::default();
    let mut s = Stats {
        probe: Split::default(),
        lookup: Split::default(),
        density: 0.0,
        distinct_states: 0,
        ever_fired: 0,
        always_on: 0,
    };
    let mut fire_counts = vec![0usize; hidden_bits];
    let mut states = std::collections::HashSet::new();
    let mut eval_ticks = 0usize;

    for t in 0..stream.len() - 1 {
        net.write_input(&enc.codes[stream[t]], 0);
        net.tick(AdvanceMode::Clear, 0);
        let state = net.read_node(probe_node).clone();
        let next = stream[t + 1];

        let seg = segs.of(t);
        s.probe.add(seg, probe.predict(&state), next);
        s.lookup.add(seg, lookup.predict(&state), next);
        if seg == Some(Seg::Train) {
            s.density += state.count_ones() as f64;
            for b in active_bits(&state) {
                fire_counts[b] += 1;
            }
            states.insert(state.as_words().to_vec());
            eval_ticks += 1;
        }
        probe.learn(&state, next);
        lookup.learn(&state, next, n_chars);
    }
    s.density /= eval_ticks as f64;
    s.distinct_states = states.len();
    s.ever_fired = fire_counts.iter().filter(|&&k| k > 0).count();
    s.always_on = fire_counts.iter().filter(|&&k| k == eval_ticks).count();
    s
}

fn print_stats(name: &str, s: &Stats, bits: usize) {
    println!("  {name}");
    println!("    bit-vote probe accuracy : {:5.1}%   held-out {:5.1}%", s.probe.train.pct(), s.probe.heldout.pct());
    println!("    state-lookup accuracy   : {:5.1}%   held-out {:5.1}%", s.lookup.train.pct(), s.lookup.heldout.pct());
    println!(
        "    active bits/tick        : {:5.1} of {bits}   ever active: {}   always on: {}",
        s.density, s.ever_fired, s.always_on
    );
    println!("    distinct states (eval)  : {}", s.distinct_states);
}

fn main() {
    let mut rng = StdRng::seed_from_u64(7);
    let enc = Encoder::new(CORPUS, &mut rng);
    let n_chars = enc.alphabet.len();
    let one_pass: Vec<usize> = CORPUS.chars().map(|c| enc.index(c)).collect();
    let mut stream: Vec<usize> = one_pass.iter().cloned().cycle().take(one_pass.len() * EPOCHS).collect();
    stream.extend(HELDOUT.chars().map(|c| enc.index(c)));
    // Score the final training epoch (after EPOCHS-1 passes), then the held-out text.
    let segs = Segments {
        eval_from: one_pass.len() * (EPOCHS - 1),
        heldout_from: one_pass.len() * EPOCHS,
    };

    println!(
        "corpus: {} chars, alphabet {}, {} epochs, scoring last epoch; held-out: {} chars",
        one_pass.len(),
        n_chars,
        EPOCHS,
        HELDOUT.chars().count()
    );
    println!();

    // ---- Baselines --------------------------------------------------------
    println!("Baselines (online counting, same protocol):");
    let mut uni = Split::default();
    let mut uni_counts = vec![0u32; n_chars];
    let mut ngrams: Vec<(NGram, Split)> =
        (1..=6).map(|o| (NGram { order: o, table: HashMap::new() }, Split::default())).collect();
    let mut raw_probe = BitVoteProbe::new(INPUT_BITS, n_chars);
    let mut raw_score = Split::default();
    for t in 0..stream.len() - 1 {
        let ctx = &stream[..=t];
        let next = stream[t + 1];
        let seg = segs.of(t);
        uni.add(seg, argmax_u(&uni_counts), next);
        for (g, sc) in ngrams.iter_mut() {
            sc.add(seg, g.predict(ctx), next);
        }
        raw_score.add(seg, raw_probe.predict(&enc.codes[stream[t]]), next);
        uni_counts[next] += 1;
        for (g, _) in ngrams.iter_mut() {
            g.learn(ctx, next, n_chars);
        }
        raw_probe.learn(&enc.codes[stream[t]], next);
    }
    println!("  most frequent char         : {:5.1}%   held-out {:5.1}%", uni.train.pct(), uni.heldout.pct());
    for (g, sc) in &ngrams {
        println!("  {}-char context n-gram      : {:5.1}%   held-out {:5.1}%", g.order, sc.train.pct(), sc.heldout.pct());
    }
    println!(
        "  bit-vote probe on raw input: {:5.1}%   held-out {:5.1}%  (no-context ceiling for this probe)",
        raw_score.train.pct(),
        raw_score.heldout.pct()
    );
    println!();

    // ---- Network A: stock defaults ---------------------------------------
    // GraphProgram::new() + ProgramDefaults::default(): input -> output with the
    // built-in KernelGroup (one SimpleKernel, threshold 16, writes output bit 0).
    println!("Network A: stock RuntimeNetwork (empty program, default kernels)");
    {
        let defaults = ProgramDefaults { default_bits: INPUT_BITS, ..ProgramDefaults::default() };
        let mut net = RuntimeNetwork::from_program(&GraphProgram::new(), &defaults);
        let out = net.nodes.len() - 1;
        let s = run_network(&mut net, out, &enc, &stream, &segs);
        print_stats("output node", &s, INPUT_BITS);
    }
    println!();

    // ---- Network B: scaled up, feed-forward only --------------------------
    // input -> N0 -> output, with KERNELS_PER_EDGE random SimpleKernels on the
    // input -> N0 edge, learning with the local Hebbian rules.
    println!("Network B: input -> hidden, {KERNELS_PER_EDGE} SimpleKernels (no recurrence)");
    for (threshold, target) in [(3usize, None), (5, None), (8, None), (8, Some(16usize))] {
        let defaults = ProgramDefaults { default_bits: HIDDEN_BITS, ..ProgramDefaults::default() };
        let prog = GraphProgram::new().nop(); // non-empty -> creates hidden node N0 (index 1)
        let mut net = RuntimeNetwork::from_program(&prog, &defaults);
        for eg in net.edges.iter_mut() {
            if eg.src.idx == 0 && eg.dst == 1 {
                eg.group = hebbian_group(&mut rng, INPUT_BITS / 64, threshold, target);
            }
        }
        let s = run_network(&mut net, 1, &enc, &stream, &segs);
        print_stats(&format!("hidden node, threshold {threshold}{}", homeo_label(target)), &s, HIDDEN_BITS);
    }
    println!();

    // ---- Network C: with recurrence (Stay -> self loop on N0) -------------
    // GraphProgram::stay() adds an N0 -> N0 edge reading N0's previous frame.
    println!("Network C: input -> hidden + hidden -> hidden (recurrent context)");
    for (threshold, target) in [(3usize, None), (5, None), (8, Some(16usize))] {
        let defaults = ProgramDefaults { default_bits: HIDDEN_BITS, ..ProgramDefaults::default() };
        let prog = GraphProgram::new().stay();
        let mut net = RuntimeNetwork::from_program(&prog, &defaults);
        for eg in net.edges.iter_mut() {
            let src_words = if eg.src.idx == 0 { INPUT_BITS / 64 } else { HIDDEN_BITS / 64 };
            if eg.dst == 1 {
                eg.group = hebbian_group(&mut rng, src_words, threshold, target);
            }
        }
        let s = run_network(&mut net, 1, &enc, &stream, &segs);
        print_stats(&format!("hidden node, threshold {threshold}{}", homeo_label(target)), &s, HIDDEN_BITS);
    }
    println!();

    // ---- Network D: predictive, surprise-driven growth --------------------
    // input -> output where the edge reads the last `frames` input frames and the
    // output is trained to hold the next character's code. Kernels start at zero
    // and are grown (and recycled at the budget) by KernelClass::feedback.
    println!("Network D: predictive input history -> output (native prediction, no probe)");
    for (frames, budget) in [(1usize, 4096usize), (3, 4096), (6, 4096), (6, 256), (6, 64)] {
        let s = run_predictive(&enc, &stream, &segs, frames, budget);
        println!("  context window {frames} frame(s), kernel budget {budget}");
        println!("    prediction accuracy     : {:5.1}%   held-out {:5.1}%", s.acc.train.pct(), s.acc.heldout.pct());
        println!(
            "    kernels: {} live, {} grown, {} recycled; by context depth {:?}",
            s.live, s.grown, s.recycled, s.by_depth
        );
    }
}

fn hebbian_group(rng: &mut StdRng, src_words: usize, threshold: usize, target: Option<usize>) -> KernelGroup {
    let mut kc = random_kernel_class(rng, src_words, HIDDEN_BITS, threshold);
    if let Some(t) = target {
        kc = kc.with_target_active(t);
    }
    let mut g = KernelGroup::new();
    g.add_class(kc);
    g
}

fn homeo_label(target: Option<usize>) -> String {
    target.map_or(String::new(), |t| format!(" + homeostasis (target {t} active)"))
}

struct PredictiveStats {
    acc: Split,
    live: usize,
    grown: usize,
    recycled: usize,
    by_depth: Vec<usize>,
}

fn run_predictive(enc: &Encoder, stream: &[usize], segs: &Segments, frames: usize, budget: usize) -> PredictiveStats {
    let defaults = ProgramDefaults { default_bits: INPUT_BITS, ..ProgramDefaults::default() };
    let mut g = GraphProgram::new().build_graph(&defaults); // input -> output
    for e in g.edge_weights_mut() {
        e.input_frames = frames;
    }
    let mut net = RuntimeNetwork::from_graph(&g);
    let cfg = GrowthConfig {
        max_kernels: budget,
        frame_words: INPUT_BITS / 64,
        max_frames: frames,
        sample_bits: 16,
        match_fraction: 0.8,
        surprise_fraction: 0.5,
        generalize: None,
    };
    let mut group = KernelGroup::new();
    group.add_class(KernelClass::predictive(cfg));
    net.edges[0].group = group;
    let out = net.nodes.len() - 1;

    let mut acc = Split::default();
    for t in 0..stream.len() - 1 {
        net.write_input(&enc.codes[stream[t]], 0);
        net.tick(AdvanceMode::Clear, 0);
        let next = stream[t + 1];
        acc.add(segs.of(t), decode(enc, net.read_node(out)), next);
        net.learn(out, &enc.codes[next]);
    }

    let kc = &net.edges[0].group.classes()[0];
    let mut by_depth = vec![0; frames];
    for k in kc.kernels() {
        by_depth[k.context_frames - 1] += 1;
    }
    PredictiveStats { acc, live: kc.len(), grown: kc.grown(), recycled: kc.recycled(), by_depth }
}

/// The character whose code overlaps the output the most (None if nothing is active).
fn decode(enc: &Encoder, out: &BitVector) -> Option<usize> {
    let words = out.as_words();
    let overlaps: Vec<u32> = enc
        .codes
        .iter()
        .map(|c| c.as_words().iter().zip(words).map(|(a, b)| (a & b).count_ones()).sum())
        .collect();
    argmax_u(&overlaps)
}
