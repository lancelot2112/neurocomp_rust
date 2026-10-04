//! Reading experiment: can the current network learn next-character prediction?
//!
//! Run with: cargo run --release --example read_text
//!
//! The library is used exactly as it is today (no changes to kernels or learning
//! rules). Characters are encoded as fixed random sparse bit patterns and fed in
//! one per tick. Because the network has no supervised output, we measure how
//! much next-character information its hidden state carries with an external
//! "probe": a vote table counts[bit][next_char] built online from the same
//! stream. The probe is NOT part of the network; it only reads it.
//!
//! Baselines use the same online protocol (predict, then update counts):
//! - unigram / n-gram tables over the raw characters
//! - the same probe applied to the raw input encoding (a bigram-like ceiling
//!   for a network with no context)

use std::collections::HashMap;

use neurocomp::bitvec::{AdvanceMode, BitVector};
use neurocomp::kernel::{KernelClass, KernelGroup, KernelOp, SimpleKernel};
use neurocomp::program::{GraphProgram, ProgramDefaults, RuntimeNetwork};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

const CORPUS: &str = "the cat sat on the mat. the dog sat on the log. \
the cat saw the dog and the dog saw the cat. a bird sang in the tree \
and the cat looked up at the bird. the dog ran to the tree and the bird \
flew away. then the cat and the dog sat in the sun on the mat by the log. ";

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
    let mut out_bits: Vec<usize> = (1..dst_bits).collect(); // bit 0 is used by KernelGroup's built-in default kernel
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

struct Stats {
    probe: Score,
    lookup: Score,
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
    eval_from: usize,
) -> Stats {
    let n_chars = enc.alphabet.len();
    let hidden_bits = net.read_node(probe_node).bit_len();
    let mut probe = BitVoteProbe::new(hidden_bits, n_chars);
    let mut lookup = StateLookupProbe::default();
    let mut s = Stats {
        probe: Score::default(),
        lookup: Score::default(),
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

        if t >= eval_from {
            s.probe.add(probe.predict(&state), next);
            s.lookup.add(lookup.predict(&state), next);
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
    println!("    bit-vote probe accuracy : {:5.1}%", s.probe.pct());
    println!("    state-lookup accuracy   : {:5.1}%", s.lookup.pct());
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
    let stream: Vec<usize> = one_pass.iter().cloned().cycle().take(one_pass.len() * EPOCHS).collect();
    // Score only the final epoch (after the network has seen the text EPOCHS-1 times).
    let eval_from = one_pass.len() * (EPOCHS - 1);

    println!("corpus: {} chars, alphabet {}, {} epochs, scoring last epoch", one_pass.len(), n_chars, EPOCHS);
    println!();

    // ---- Baselines --------------------------------------------------------
    println!("Baselines (online counting, same protocol):");
    let mut uni = Score::default();
    let mut uni_counts = vec![0u32; n_chars];
    let mut ngrams: Vec<(NGram, Score)> =
        (1..=6).map(|o| (NGram { order: o, table: HashMap::new() }, Score::default())).collect();
    let mut raw_probe = BitVoteProbe::new(INPUT_BITS, n_chars);
    let mut raw_score = Score::default();
    for t in 0..stream.len() - 1 {
        let ctx = &stream[..=t];
        let next = stream[t + 1];
        if t >= eval_from {
            uni.add(argmax_u(&uni_counts), next);
            for (g, sc) in ngrams.iter_mut() {
                sc.add(g.predict(ctx), next);
            }
            raw_score.add(raw_probe.predict(&enc.codes[stream[t]]), next);
        }
        uni_counts[next] += 1;
        for (g, _) in ngrams.iter_mut() {
            g.learn(ctx, next, n_chars);
        }
        raw_probe.learn(&enc.codes[stream[t]], next);
    }
    println!("  most frequent char        : {:5.1}%", uni.pct());
    for (g, sc) in &ngrams {
        println!("  {}-char context n-gram     : {:5.1}%", g.order, sc.pct());
    }
    println!("  bit-vote probe on raw input: {:5.1}%  (no-context ceiling for this probe)", raw_score.pct());
    println!();

    // ---- Network A: stock defaults ---------------------------------------
    // GraphProgram::new() + ProgramDefaults::default(): input -> output with the
    // built-in KernelGroup (one SimpleKernel, threshold 16, writes output bit 0).
    println!("Network A: stock RuntimeNetwork (empty program, default kernels)");
    {
        let defaults = ProgramDefaults { default_bits: INPUT_BITS, ..ProgramDefaults::default() };
        let mut net = RuntimeNetwork::from_program(&GraphProgram::new(), &defaults);
        let out = net.nodes.len() - 1;
        let s = run_network(&mut net, out, &enc, &stream, eval_from);
        print_stats("output node", &s, INPUT_BITS);
    }
    println!();

    // ---- Network B: scaled up, feed-forward only --------------------------
    // input -> N0 -> output, with KERNELS_PER_EDGE random SimpleKernels on the
    // input -> N0 edge. Learning rules unchanged.
    println!("Network B: input -> hidden, {KERNELS_PER_EDGE} SimpleKernels (no recurrence)");
    for threshold in [3usize, 5, 8] {
        let defaults = ProgramDefaults { default_bits: HIDDEN_BITS, ..ProgramDefaults::default() };
        let prog = GraphProgram::new().nop(); // non-empty -> creates hidden node N0 (index 1)
        let mut net = RuntimeNetwork::from_program(&prog, &defaults);
        for eg in net.edges.iter_mut() {
            if eg.src.idx == 0 && eg.dst == 1 {
                let mut g = KernelGroup::default();
                g.add_class(random_kernel_class(&mut rng, INPUT_BITS / 64, HIDDEN_BITS, threshold));
                eg.group = g;
            }
        }
        let s = run_network(&mut net, 1, &enc, &stream, eval_from);
        print_stats(&format!("hidden node, threshold {threshold}"), &s, HIDDEN_BITS);
    }
    println!();

    // ---- Network C: with recurrence (Stay -> self loop on N0) -------------
    // GraphProgram::stay() adds an N0 -> N0 edge with read_back = history_depth.
    println!("Network C: input -> hidden + hidden -> hidden (recurrent context)");
    for (label, fix_read_back) in [("as built (read_back=2)", false), ("read_back forced to 1", true)] {
        for threshold in [3usize, 5, 8] {
            let defaults = ProgramDefaults { default_bits: HIDDEN_BITS, ..ProgramDefaults::default() };
            let prog = GraphProgram::new().stay();
            let mut net = RuntimeNetwork::from_program(&prog, &defaults);
            for eg in net.edges.iter_mut() {
                let src_words = if eg.src.idx == 0 { INPUT_BITS / 64 } else { HIDDEN_BITS / 64 };
                if eg.dst == 1 {
                    let mut g = KernelGroup::default();
                    g.add_class(random_kernel_class(&mut rng, src_words, HIDDEN_BITS, threshold));
                    eg.group = g;
                }
                if fix_read_back && eg.src.idx == 1 && eg.dst == 1 {
                    eg.src.read_back = 1;
                }
            }
            let s = run_network(&mut net, 1, &enc, &stream, eval_from);
            print_stats(&format!("hidden node, {label}, threshold {threshold}"), &s, HIDDEN_BITS);
        }
    }
}
