//! Modules: networks built from base kernels, recursively.
//!
//! Everything here is a `Module`: something with bit-vector input ports and output
//! ports that ticks once per step. There are two kinds:
//! - **Leaves**, the base kernels:
//!   - `Predictor`, a predictive `KernelClass` that learns from a teaching port;
//!   - `BitOp`, a `KernelOp` (or, and, xor, clear) over two inputs;
//!   - `Delay`, one frame of history;
//!   - `Concat`, frames side by side (an L4-style assembly);
//!   - `Separate`, a hashed expansion with k winners (dentate-gyrus-style pattern
//!     separation).
//! - **`Network`**, a module made of modules. Its children read its inputs and each
//!   other's outputs by wires, and it exposes some of their outputs as its own. A
//!   network is a module, so networks nest: a column is a network of kernels, and a
//!   hierarchy is a network of columns.
//!
//! **Learning is local and travels on wires.** A `Predictor` has a teaching input. At
//! each tick it first learns that what arrives there is what should have followed its
//! previous input, then predicts from its current input. There is no backpropagation and
//! no outside call to `learn`: what a kernel learns from is part of the wiring.
//!
//! **Timing.** Children tick in the order they were placed. A wire from an earlier
//! child (or the network's input) carries this tick's value. A wire from the same or a
//! later child carries that child's previous-tick value. So a feedback loop always has
//! a one-tick delay and the network is deterministic.
//!
//! **The grammar.** `NetOp` is a stack language that builds networks: signals are pushed
//! on a stack, `Place` pops a module's inputs and pushes its outputs, and
//! `Feedback`/`Close` make loops. A `Genome` is a library of definitions. `Sub(k)`
//! places an earlier definition as a module, which is how architectures recurse. Any
//! instruction sequence builds a valid network: missing signals become unconnected
//! inputs (all zeros), and out-of-range references are no-ops. That makes genomes safe
//! to mutate.

use rand::RngCore;

use crate::bitvec::BitVector;
use crate::kernel::class::KernelOp;
use crate::kernel::{GrowthConfig, KernelClass, SimpleKernel};
use crate::fixed::{div_round, phasors, ratio, recip, recip32, Q16, ONE};
use crate::program::hippocampus::{mix64, top_k, write_amounts, DentateGyrus, Pathway};

/// What a module needs from outside for a tick: randomness for learning, and whether
/// slow learning is on (it is off at test, when only fast inhibition runs).
pub struct Ctx<'a> {
    pub rng: &'a mut dyn RngCore,
    pub learn: bool,
}

/// A unit with bit-vector ports that ticks once per step.
pub trait Module {
    /// A short name, for `describe`.
    fn name(&self) -> String;
    fn n_inputs(&self) -> usize;
    fn n_outputs(&self) -> usize;
    /// One step: read `inputs` (one per input port; an unconnected port is empty) and
    /// update the outputs.
    fn tick(&mut self, inputs: &[&BitVector], ctx: &mut Ctx);
    /// The value on output `port` after the latest tick.
    fn output(&self, port: usize) -> &BitVector;
    /// Offline consolidation (kernel sleep). Default: nothing.
    fn sleep(&mut self) {}
    /// Clear activity (a story boundary). Learned state is kept. Default: nothing.
    fn reset(&mut self) {}
    /// The number of kernels this module holds, summed over its parts.
    fn kernels(&self) -> usize {
        0
    }
    /// An indented tree of the module and its parts.
    fn describe(&self, depth: usize) -> String {
        format!("{}{}\n", "  ".repeat(depth), self.name())
    }
}

static EMPTY: BitVector = BitVector::EMPTY;

fn zeros(bits: usize) -> BitVector {
    if bits == 0 {
        return BitVector::EMPTY;
    }
    BitVector::new(bits, Some(0))
}

// ---------------------------------------------------------------------------
// Leaves

/// A predictive kernel class with a teaching port.
///
/// Inputs: `[input, teach]`. Outputs: `[prediction, surprise]`.
/// - At each tick, if `teach` has bits, the class learns that `teach` should have
///   followed the previous tick's input (`feedback`), or only runs fast inhibition
///   when slow learning is off.
/// - `surprise` is the part of `teach` the previous prediction missed
///   (`teach & !prediction`), the bitwise prediction error.
/// - Then it predicts from `input`.
pub struct Predictor {
    pub class: KernelClass<SimpleKernel>,
    bits: usize,
    last_input: Option<BitVector>,
    outs: [BitVector; 3],
    /// Expose a third output: the winning kernel's reliability (`scalar`).
    pub(crate) confidence_port: bool,
    /// Take no part in sleep.
    pub(crate) no_sleep: bool,
    /// Its own random stream (else the network's shared one): with one stream per module,
    /// adding or removing a module does not shift every other module's draws.
    pub own_rng: Option<rand::rngs::StdRng>,
}

impl Predictor {
    pub fn new(bits: usize, class: KernelClass<SimpleKernel>) -> Self {
        Self { class, bits, last_input: None, outs: [zeros(bits), zeros(bits), scalar(0)], confidence_port: false, no_sleep: false, own_rng: None }
    }
}

/// A graded signal on a wire (a reliability, a share): one `Q16` in a one-word vector.
pub fn scalar(v: Q16) -> BitVector {
    BitVector::from_words(vec![v as u64])
}

/// The value of a `scalar` wire (0 if unconnected).
pub fn scalar_of(x: &BitVector) -> Q16 {
    x.as_words().first().map_or(0, |&w| w.min(ONE as u64) as Q16)
}

impl Module for Predictor {
    fn name(&self) -> String {
        format!("Predictor({} bits, {} kernels)", self.bits, self.class.len())
    }
    fn n_inputs(&self) -> usize {
        2
    }
    fn n_outputs(&self) -> usize {
        2 + self.confidence_port as usize
    }
    fn tick(&mut self, inputs: &[&BitVector], ctx: &mut Ctx) {
        let (input, teach) = (inputs[0], inputs[1]);
        let mut surprise = zeros(self.bits);
        if teach.count_ones() > 0 {
            for (i, w) in surprise.as_words_mut().iter_mut().enumerate() {
                let t = teach.as_words().get(i).copied().unwrap_or(0);
                *w = t & !self.outs[0].as_words()[i];
            }
            if let Some(last) = &self.last_input {
                if ctx.learn {
                    match self.own_rng.as_mut() {
                        Some(r) => self.class.feedback(last, teach, r),
                        None => self.class.feedback(last, teach, &mut *ctx.rng),
                    }
                } else {
                    self.class.fast_inhibit(teach);
                }
            }
        }
        self.outs[1] = surprise;
        let mut out = zeros(self.bits);
        if input.bit_len() > 0 {
            self.class.process_predictive(input, &mut out);
            self.last_input = Some(input.clone());
        } else {
            self.last_input = None;
        }
        self.outs[0] = out;
        self.outs[2] = scalar(self.class.confidence().unwrap_or(0));
    }
    fn output(&self, port: usize) -> &BitVector {
        self.outs.get(port).unwrap_or(&EMPTY)
    }
    fn sleep(&mut self) {
        if !self.no_sleep {
            self.class.sleep();
        }
    }
    fn reset(&mut self) {
        self.last_input = None;
        self.outs = [zeros(self.bits), zeros(self.bits), scalar(0)];
    }
    fn kernels(&self) -> usize {
        self.class.len()
    }
}

/// A bitwise kernel op over two inputs: `a op b` (`Clear` is `a & !b`). Inputs of
/// different lengths are zero-padded to the longer.
pub struct BitOp {
    op: KernelOp,
    out: BitVector,
}

impl BitOp {
    pub fn new(op: KernelOp) -> Self {
        Self { op, out: BitVector::EMPTY }
    }
}

impl Module for BitOp {
    fn name(&self) -> String {
        format!("BitOp({:?})", self.op)
    }
    fn n_inputs(&self) -> usize {
        2
    }
    fn n_outputs(&self) -> usize {
        1
    }
    fn tick(&mut self, inputs: &[&BitVector], _ctx: &mut Ctx) {
        let (a, b) = (inputs[0].as_words(), inputs[1].as_words());
        let n = a.len().max(b.len());
        let f: fn(u64, u64) -> u64 = match self.op {
            KernelOp::Or => |x, y| x | y,
            KernelOp::And => |x, y| x & y,
            KernelOp::Xor => |x, y| x ^ y,
            KernelOp::Clear => |x, y| x & !y,
        };
        let words = (0..n).map(|i| f(a.get(i).copied().unwrap_or(0), b.get(i).copied().unwrap_or(0))).collect();
        self.out = BitVector::from_words(words);
    }
    fn output(&self, port: usize) -> &BitVector {
        if port == 0 { &self.out } else { &EMPTY }
    }
    fn reset(&mut self) {
        self.out = BitVector::EMPTY;
    }
}

/// One frame of history: outputs the previous tick's input (empty at first).
pub struct Delay {
    held: BitVector,
    out: BitVector,
}

impl Default for Delay {
    fn default() -> Self {
        Self { held: BitVector::EMPTY, out: BitVector::EMPTY }
    }
}

impl Module for Delay {
    fn name(&self) -> String {
        "Delay".into()
    }
    fn n_inputs(&self) -> usize {
        1
    }
    fn n_outputs(&self) -> usize {
        1
    }
    fn tick(&mut self, inputs: &[&BitVector], _ctx: &mut Ctx) {
        self.out = std::mem::replace(&mut self.held, inputs[0].clone());
        if self.out.bit_len() == 0 {
            self.out = zeros(inputs[0].bit_len());
        }
    }
    fn output(&self, port: usize) -> &BitVector {
        if port == 0 { &self.out } else { &EMPTY }
    }
    fn reset(&mut self) {
        self.held = BitVector::EMPTY;
        self.out = BitVector::EMPTY;
    }
}

/// `n` inputs side by side, as frames of equal width (each padded to the widest), so a
/// `Predictor` can read them as `frame_words`-wide frames.
pub struct Concat {
    n: usize,
    out: BitVector,
}

impl Concat {
    pub fn new(n: usize) -> Self {
        Self { n, out: BitVector::EMPTY }
    }
}

impl Module for Concat {
    fn name(&self) -> String {
        format!("Concat({})", self.n)
    }
    fn n_inputs(&self) -> usize {
        self.n
    }
    fn n_outputs(&self) -> usize {
        1
    }
    fn tick(&mut self, inputs: &[&BitVector], _ctx: &mut Ctx) {
        let width = inputs.iter().map(|x| x.as_words().len()).max().unwrap_or(0);
        let mut words = Vec::with_capacity(width * self.n);
        for x in inputs {
            let w = x.as_words();
            words.extend_from_slice(w);
            words.extend(std::iter::repeat(0).take(width - w.len()));
        }
        self.out = BitVector::from_words(words);
    }
    fn output(&self, port: usize) -> &BitVector {
        if port == 0 { &self.out } else { &EMPTY }
    }
    fn reset(&mut self) {
        self.out = BitVector::EMPTY;
    }
}

/// Pattern separation: each active input bit drives `fan_out` of `cells` cells by a
/// fixed hash, and the `k` most driven cells fire (`DentateGyrus::hashed`).
pub struct Separate {
    dg: DentateGyrus,
    out: BitVector,
}

impl Separate {
    pub fn new(cells: usize, fan_out: usize, k: usize, seed: u64) -> Self {
        Self { dg: DentateGyrus::hashed(cells, fan_out, k, seed), out: zeros(cells) }
    }
}

impl Module for Separate {
    fn name(&self) -> String {
        format!("Separate({} cells, k={})", self.dg.cells, self.dg.k)
    }
    fn n_inputs(&self) -> usize {
        1
    }
    fn n_outputs(&self) -> usize {
        1
    }
    fn tick(&mut self, inputs: &[&BitVector], _ctx: &mut Ctx) {
        let x: Vec<usize> = (0..inputs[0].bit_len()).filter(|&b| inputs[0].bit_get(b)).collect();
        let mut out = zeros(self.dg.cells);
        if !x.is_empty() {
            for c in self.dg.separate(&x) {
                out.bit_set(c as usize);
            }
        }
        self.out = out;
    }
    fn output(&self, port: usize) -> &BitVector {
        if port == 0 { &self.out } else { &EMPTY }
    }
    fn reset(&mut self) {
        self.out = zeros(self.dg.cells);
    }
}

impl Separate {
    /// A separation with a fixed random fan-in table: each of `cells` cells samples
    /// `fan_in` of `inputs` input bits (`DentateGyrus::new`).
    pub fn table(inputs: usize, cells: usize, fan_in: usize, k: usize, seed: u64) -> Self {
        Self { dg: DentateGyrus::new(inputs, cells, fan_in, k, seed), out: zeros(cells) }
    }
}

/// A fixed one-to-one projection: input bit i drives one output bit, drawn at random
/// once (mossy fibres: each granule cell has one "detonator" target in CA3).
pub struct Scatter {
    map: Vec<u32>,
    outputs: usize,
    out: BitVector,
}

impl Scatter {
    pub fn new(inputs: usize, outputs: usize, seed: u64) -> Self {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self { map: (0..inputs).map(|_| rng.gen_range(0..outputs) as u32).collect(), outputs, out: zeros(outputs) }
    }
}

impl Module for Scatter {
    fn name(&self) -> String {
        format!("Scatter({} -> {})", self.map.len(), self.outputs)
    }
    fn n_inputs(&self) -> usize {
        1
    }
    fn n_outputs(&self) -> usize {
        1
    }
    fn tick(&mut self, inputs: &[&BitVector], _ctx: &mut Ctx) {
        let mut out = zeros(self.outputs);
        for b in active(inputs[0]) {
            if let Some(&t) = self.map.get(b) {
                out.bit_set(t as usize);
            }
        }
        self.out = out;
    }
    fn output(&self, port: usize) -> &BitVector {
        if port == 0 { &self.out } else { &EMPTY }
    }
    fn reset(&mut self) {
        self.out = zeros(self.outputs);
    }
}

/// The sentence so far: inputs `[x, clear]`; while `clear` has a bit the bag empties first,
/// then `x` joins it. Output: the bag.
pub struct Bag {
    bag: BitVector,
}

impl Default for Bag {
    fn default() -> Self {
        Self { bag: BitVector::EMPTY }
    }
}

impl Module for Bag {
    fn name(&self) -> String {
        "Bag".into()
    }
    fn n_inputs(&self) -> usize {
        2
    }
    fn n_outputs(&self) -> usize {
        1
    }
    fn tick(&mut self, inputs: &[&BitVector], _ctx: &mut Ctx) {
        if inputs[1].count_ones() > 0 || self.bag.bit_len() != inputs[0].bit_len() {
            self.bag = zeros(inputs[0].bit_len());
        }
        self.bag.or_mut(inputs[0]);
    }
    fn output(&self, port: usize) -> &BitVector {
        if port == 0 { &self.bag } else { &EMPTY }
    }
    fn reset(&mut self) {
        self.bag = BitVector::EMPTY;
    }
}

/// A slow state over sentences: inputs `[x, advance]`. `x` joins the current sentence's
/// content; while `advance` has a bit, the current content (if any) is first kept as one
/// of the last `span` sentences. Output: the current content OR the kept ones (the higher
/// area's slow state). With `keep`, a reset (a story's end) clears nothing.
pub struct Window {
    span: usize,
    keep: bool,
    current: BitVector,
    kept: std::collections::VecDeque<BitVector>,
    out: BitVector,
}

impl Window {
    pub fn new(span: usize, keep: bool) -> Self {
        Self { span: span.max(1), keep, current: BitVector::EMPTY, kept: Default::default(), out: BitVector::EMPTY }
    }
}

impl Module for Window {
    fn name(&self) -> String {
        format!("Window({}{})", self.span, if self.keep { ", kept across stories" } else { "" })
    }
    fn n_inputs(&self) -> usize {
        2
    }
    fn n_outputs(&self) -> usize {
        1
    }
    fn tick(&mut self, inputs: &[&BitVector], _ctx: &mut Ctx) {
        let bits = inputs[0].bit_len();
        if self.current.bit_len() != bits {
            self.current = zeros(bits);
        }
        if inputs[1].count_ones() > 0 && self.current.count_ones() > 0 {
            self.kept.push_back(std::mem::replace(&mut self.current, zeros(bits)));
            while self.kept.len() > self.span {
                self.kept.pop_front();
            }
        }
        self.current.or_mut(inputs[0]);
        let mut out = self.current.clone();
        for k in &self.kept {
            out.or_mut(k);
        }
        self.out = out;
    }
    fn output(&self, port: usize) -> &BitVector {
        if port == 0 { &self.out } else { &EMPTY }
    }
    fn reset(&mut self) {
        if !self.keep {
            self.current = BitVector::EMPTY;
            self.kept.clear();
            self.out = BitVector::EMPTY;
        }
    }
}

/// The word-level comparator (L5): inputs `[x, prediction, confidence]`. Passes `x` when
/// the prediction gave it less than `threshold` (`Q16`): its share of the prediction times
/// the predicting kernel's reliability. No prediction: everything is surprising.
pub struct Surprise {
    threshold: Q16,
    out: BitVector,
}

impl Module for Surprise {
    fn name(&self) -> String {
        "Surprise".into()
    }
    fn n_inputs(&self) -> usize {
        3
    }
    fn n_outputs(&self) -> usize {
        1
    }
    fn tick(&mut self, inputs: &[&BitVector], _ctx: &mut Ctx) {
        let (x, p) = (inputs[0], inputs[1]);
        let predicted = p.count_ones();
        let share = if predicted == 0 {
            0
        } else {
            let hit: u32 = x.as_words().iter().zip(p.as_words()).map(|(a, b)| (a & b).count_ones()).sum();
            ((ratio(hit as u64, predicted as u64) as u64 * scalar_of(inputs[2]) as u64) >> 16) as Q16
        };
        self.out = if share < self.threshold { x.clone() } else { zeros(x.bit_len()) };
    }
    fn output(&self, port: usize) -> &BitVector {
        if port == 0 { &self.out } else { &EMPTY }
    }
    fn reset(&mut self) {
        self.out = BitVector::EMPTY;
    }
}

/// Graded wiring: several sources into one input, each passing a share of its bits (its
/// gain, `Q16`; the same fixed subset of bit positions for a given gain, as a synapse's
/// strength sets how much of a pathway gets through), ORed together.
pub struct Blend {
    pub gains: Vec<Q16>,
    /// The output's width in words (that of the input's main connection): a source of
    /// another width is cut or padded to it, so adding a connection never reshapes the
    /// input it joins.
    pub words: usize,
    out: BitVector,
}

impl Blend {
    pub fn new(gains: Vec<Q16>, words: usize) -> Self {
        Self { gains, words, out: BitVector::EMPTY }
    }
}

/// Keep the bits of `x` at positions whose hash falls under `gain` (`Q16`; `ONE` keeps all).
pub fn pass_share(x: &BitVector, gain: Q16) -> BitVector {
    if gain >= ONE {
        return x.clone();
    }
    let mut w = x.as_words().to_vec();
    for (wi, d) in w.iter_mut().enumerate() {
        let mut keep = 0u64;
        for b in 0..64 {
            let h = ((wi * 64 + b) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 48;
            if h < gain as u64 {
                keep |= 1 << b;
            }
        }
        *d &= keep;
    }
    BitVector::from_words(w)
}

impl Module for Blend {
    fn name(&self) -> String {
        let g: Vec<String> = self.gains.iter().map(|&g| format!("{}/16", (g as u64 * 16 + ONE as u64 / 2) / ONE as u64)).collect();
        format!("Blend({})", g.join(", "))
    }
    fn n_inputs(&self) -> usize {
        self.gains.len()
    }
    fn n_outputs(&self) -> usize {
        1
    }
    fn tick(&mut self, inputs: &[&BitVector], _ctx: &mut Ctx) {
        let mut words = vec![0u64; self.words];
        for (x, &g) in inputs.iter().zip(&self.gains) {
            for (d, s) in words.iter_mut().zip(pass_share(x, g).as_words()) {
                *d |= s;
            }
        }
        self.out = BitVector::from_words(words);
    }
    fn output(&self, port: usize) -> &BitVector {
        if port == 0 { &self.out } else { &EMPTY }
    }
    fn reset(&mut self) {
        self.out = BitVector::EMPTY;
    }
}

/// A mode switch: passes input 0 while input 1 has any bit, else nothing (a
/// neuromodulatory gate, e.g. acetylcholine switching the hippocampus into encoding).
pub struct Gate {
    out: BitVector,
}

impl Module for Gate {
    fn name(&self) -> String {
        "Gate".into()
    }
    fn n_inputs(&self) -> usize {
        2
    }
    fn n_outputs(&self) -> usize {
        1
    }
    fn tick(&mut self, inputs: &[&BitVector], _ctx: &mut Ctx) {
        self.out = if inputs[1].count_ones() > 0 { inputs[0].clone() } else { BitVector::EMPTY };
    }
    fn output(&self, port: usize) -> &BitVector {
        if port == 0 { &self.out } else { &EMPTY }
    }
    fn reset(&mut self) {
        self.out = BitVector::EMPTY;
    }
}

/// How an `Associate` population turns summed drive into active cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Readout {
    /// The `k` most driven cells (k-winners-take-all).
    TopK(usize),
    /// Every cell driven at least this fraction (`Q16`) of the most driven one.
    Fraction(Q16),
}

/// How an `Associate` pathway weighs a presynaptic row written by n events.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scale {
    /// Every row counts in full.
    None,
    /// Each synapse's counter is shifted right by ⌊log2 n⌋ before summing (as in
    /// `Hippocampus`; single writes round to zero once n is large).
    Shift,
    /// The row's summed drive is multiplied by 1/n in `Q16` (no rounding per synapse;
    /// 1/n from a reciprocal table, no division): an input's vote is divided among the
    /// events that wrote it, like inverse document frequency.
    Inverse,
}

/// How an `Associate` population removes chance drive before its readout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Center {
    Off,
    /// Subtract (cue rows' weighted writes) × (the cell's writes / all writes) × the
    /// write amount: exact, one division per cell.
    Exact,
    /// The same with the cell's write rate kept as a running mean (a moving average
    /// over the last 2^s writes once there are more), updated at each write with 1/n from
    /// the reciprocal table: a homeostatic threshold that rises with the cell's use. No
    /// division.
    Homeostatic(u8),
}

/// One learned input pathway of an `Associate` population.
struct Path {
    w: Pathway,
    /// Writes per source row (for presynaptic scaling).
    writes: Vec<u32>,
    /// Writes per target cell, and writes in all (for centering).
    used: Vec<u32>,
    total: u32,
    /// Each cell's running write rate (32 fractional bits, for `Center::Homeostatic`).
    rate: Vec<u64>,
    scale: Scale,
}

impl Path {
    fn new(cells: usize, scale: Scale) -> Self {
        Self { w: Pathway::new(0, cells), writes: Vec::new(), used: vec![0; cells], total: 0, rate: vec![0; cells], scale }
    }

    fn n(&self, i: usize) -> u32 {
        self.writes.get(i).copied().unwrap_or(0)
    }

    /// Each cell's drive from the active `sources`, in `Q16` units of write amount.
    fn drive(&self, sources: &[usize], epoch: u32) -> Vec<u64> {
        match self.scale {
            Scale::None => self.w.drive(sources, epoch).into_iter().map(|v| (v as u64) << 16).collect(),
            Scale::Shift => self.w.drive_scaled(sources, epoch, |i| log2_floor(self.n(i))).into_iter().map(|v| (v as u64) << 16).collect(),
            Scale::Inverse => {
                let mut out = vec![0u64; self.w_targets()];
                for &i in sources {
                    let per = recip32(self.n(i).max(1) as u64);
                    for (o, v) in out.iter_mut().zip(self.w.drive(&[i], epoch)) {
                        *o += (v as u64 * per) >> 16;
                    }
                }
                out
            }
        }
    }

    fn w_targets(&self) -> usize {
        self.used.len()
    }

    /// A row's weight per write, in `Q16`.
    fn weight(&self, n: u32) -> u64 {
        match self.scale {
            Scale::None => ONE as u64,
            Scale::Shift => (ONE as u64) >> log2_floor(n),
            Scale::Inverse => recip32(n.max(1) as u64) >> 16,
        }
    }

    fn write(&mut self, sources: &[usize], mask: &BitVector, amount: u32, planes: usize, epoch: u32, skip_self: bool) {
        for &i in sources {
            if skip_self {
                let mut m = mask.clone();
                m.bit_clear(i);
                self.w.strengthen(i, &m, amount, planes, epoch);
            } else {
                self.w.strengthen(i, mask, amount, planes, epoch);
            }
            if i >= self.writes.len() {
                self.writes.resize(i + 1, 0);
            }
            self.writes[i] = self.writes[i].saturating_add(1);
        }
        if !sources.is_empty() {
            self.total += 1;
            for j in active(mask) {
                self.used[j] = self.used[j].saturating_add(1);
            }
        }
    }

    /// Update the running write rates after a write to `mask`: rate += (written − rate)
    /// / min(writes, 2^max_s), with 1/n from the reciprocal table (no division). Below
    /// 2^max_s writes this is the exact running mean; after that a moving average.
    fn track_rate(&mut self, mask: &BitVector, max_s: u8) {
        let step = recip32((self.total as u64).min(1u64 << max_s)) as i128;
        for (j, r) in self.rate.iter_mut().enumerate() {
            let x: i128 = if mask.bit_get(j) { 1 << 32 } else { 0 };
            *r = (*r as i128 + (((x - *r as i128) * step) >> 32)).clamp(0, 1 << 32) as u64;
        }
    }

    /// The drive each cell would get by chance: if the stored events were unrelated to
    /// the cue, cell j would receive (cue rows' writes) × (j's writes) / (all writes)
    /// write amounts. Returned as (Σ over the cue of row writes, total) for each cell's
    /// `used` count to scale. A row's writes count as its `Scale` weighs them (`Q16`).
    fn chance(&self, sources: &[usize]) -> (u64, u64) {
        let rows: u64 = sources.iter().map(|&i| self.n(i) as u64 * self.weight(self.n(i))).sum();
        (rows, self.total.max(1) as u64)
    }
}

fn log2_floor(n: u32) -> u32 {
    31 - n.max(1).leading_zeros()
}

fn active(x: &BitVector) -> Vec<usize> {
    let mut v = Vec::new();
    for (wi, &w) in x.as_words().iter().enumerate() {
        let mut w = w;
        while w != 0 {
            v.push(wi * 64 + w.trailing_zeros() as usize);
            w &= w - 1;
        }
    }
    v
}

/// The second base kernel: a population of `cells` cells with Hebbian input pathways.
///
/// Inputs: `[teach, gain, pre_0, …, pre_{n-1}]`. Output: `[activity]`.
/// - **Weights:** each pathway is a matrix of small counters (bit-sliced, 7 planes),
///   one row per presynaptic bit, halved every `half_life` writes (lazy plane shifts).
/// - **Read:** drive = Σ over pathways of the rows of the active presynaptic bits (with
///   presynaptic scaling, a row written n times counts 1/2^⌊log2 n⌋). The readout keeps
///   the top k cells, or every cell within a fraction of the best. With `settle` > 0 the
///   population also has a recurrent pathway from itself: it then iterates
///   activity ← readout(drive + recurrent(activity)), settling into an attractor.
/// - **Write (encoding):** when `teach` has bits, the population is clamped to `teach`
///   (it is the activity that is stored and passed on), and every pathway's active rows
///   gain `amount` on the teach cells (the recurrent pathway: from each teach cell to
///   the others).
/// - **Gain:** `amount` = base × (1 + `gain` × novelty), where novelty is `gain`'s
///   popcount over `gain_k` (a population code for a scalar: e.g. the CA1 cells the
///   comparator found unmatched).
pub struct Associate {
    cells: usize,
    readout: Readout,
    settle: usize,
    paths: Vec<Path>,
    recurrent: Option<Path>,
    half_life: u32,
    amounts: Vec<u32>,
    planes: usize,
    stores: u32,
    gain: Q16,
    gain_k: usize,
    /// How chance drive is removed before the readout (see `set_center`).
    center: Center,
    out: BitVector,
}

impl Associate {
    /// Centering (the covariance rule): before the readout, each cell's drive has
    /// subtracted what it would get by chance, (cue rows' writes × the cell's writes /
    /// all writes) × the mean write amount. Count-based crosstalk is all positive and
    /// grows with a cell's use, so much-written cells win recall by bulk; centered,
    /// crosstalk has mean zero. Approximate under decay (uses undecayed counts).
    pub fn set_center(&mut self, c: Center) {
        self.center = c;
    }

    pub fn new(cells: usize, pathways: &[Scale], settle: usize, readout: Readout, half_life: u32, gain: Q16, gain_k: usize) -> Self {
        let half_life = half_life.max(1);
        Self {
            cells,
            readout,
            settle,
            paths: pathways.iter().map(|&scale| Path::new(cells, scale)).collect(),
            recurrent: if settle > 0 { Some(Path::new(cells, Scale::None)) } else { None },
            half_life,
            amounts: write_amounts(half_life),
            planes: 7,
            stores: 0,
            gain,
            gain_k: gain_k.max(1),
            center: Center::Off,
            out: zeros(cells),
        }
    }

    fn epoch(&self) -> u32 {
        self.stores / self.half_life
    }

    fn select(&self, drive: &[u64]) -> Vec<u32> {
        match self.readout {
            Readout::TopK(k) => top_k(drive.iter().copied(), k),
            Readout::Fraction(f) => {
                let best = drive.iter().copied().max().unwrap_or(0);
                if best == 0 {
                    return Vec::new();
                }
                let floor = best as u128 * f as u128;
                (0..drive.len()).filter(|&b| (drive[b] as u128) << 16 >= floor).map(|b| b as u32).collect()
            }
        }
    }

    /// Writes so far.
    pub fn stores(&self) -> u32 {
        self.stores
    }
}

impl Module for Associate {
    fn name(&self) -> String {
        format!("Associate({} cells, {} pathways{}, {:?})", self.cells, self.paths.len(), if self.settle > 0 { " + recurrent" } else { "" }, self.readout)
    }
    fn n_inputs(&self) -> usize {
        2 + self.paths.len()
    }
    fn n_outputs(&self) -> usize {
        1
    }
    fn tick(&mut self, inputs: &[&BitVector], _ctx: &mut Ctx) {
        let (teach, gain) = (inputs[0], inputs[1]);
        let pres: Vec<Vec<usize>> = inputs[2..].iter().map(|x| active(x)).collect();
        if teach.count_ones() > 0 {
            // encoding: clamp to teach and write
            let novelty = ratio(gain.count_ones() as u64, self.gain_k as u64).min(ONE);
            self.stores += 1;
            let epoch = self.epoch();
            let base = self.amounts[(self.stores % self.half_life) as usize] as u64;
            let factor = ONE as u64 + ((self.gain as u64 * novelty as u64) >> 16);
            let amount = div_round(base * factor, ONE as u64).min((1u64 << self.planes) - 1) as u32;
            let mut mask = zeros(self.cells);
            let t: Vec<usize> = active(teach).into_iter().filter(|&b| b < self.cells).collect();
            for &b in &t {
                mask.bit_set(b);
            }
            for (p, pre) in self.paths.iter_mut().zip(&pres) {
                p.write(pre, &mask, amount, self.planes, epoch, false);
                if let Center::Homeostatic(s) = self.center {
                    if !pre.is_empty() {
                        p.track_rate(&mask, s);
                    }
                }
            }
            if let Some(r) = &mut self.recurrent {
                r.write(&t, &mask, amount, self.planes, epoch, true);
            }
            self.out = mask;
            return;
        }
        // recall
        let epoch = self.epoch();
        let mut drive = vec![0u64; self.cells];
        for (p, pre) in self.paths.iter().zip(&pres) {
            let d_p = p.drive(pre, epoch);
            if self.center != Center::Off {
                // mean write amount: the base amount at gain 0 (16..31 within a halving period)
                let (rows, total) = p.chance(pre);
                let amount = self.amounts[(self.stores % self.half_life) as usize] as u64;
                for (j, (d, v)) in drive.iter_mut().zip(d_p).enumerate() {
                    let expect = match self.center {
                        Center::Homeostatic(_) => ((rows as u128 * p.rate[j] as u128 * amount as u128) >> 32) as u64,
                        _ => div_round(rows * p.used[j] as u64 * amount, total),
                    };
                    *d += v.saturating_sub(expect);
                }
            } else {
                for (d, v) in drive.iter_mut().zip(d_p) {
                    *d += v;
                }
            }
        }
        let mut c = self.select(&drive);
        if let Some(r) = &self.recurrent {
            for _ in 0..self.settle {
                let a: Vec<usize> = c.iter().map(|&j| j as usize).collect();
                let rec = r.drive(&a, epoch);
                let total: Vec<u64> = rec.iter().zip(&drive).map(|(r, f)| r + f).collect();
                c = self.select(&total);
            }
        }
        let mut out = zeros(self.cells);
        for j in c {
            out.bit_set(j as usize);
        }
        self.out = out;
    }
    fn output(&self, port: usize) -> &BitVector {
        if port == 0 { &self.out } else { &EMPTY }
    }
    fn reset(&mut self) {
        self.out = zeros(self.cells);
    }
}

// ---------------------------------------------------------------------------
// Phase codes

/// A phase code in bits: `cells` blocks of `phases` bits; an active cell j with phase
/// p sets bit j·`phases` + p (one bit per active block).
pub fn phase_code(cells: &[(usize, usize)], n_cells: usize, phases: usize) -> BitVector {
    let mut v = zeros(n_cells * phases);
    for &(j, p) in cells {
        v.bit_set(j * phases + p % phases);
    }
    v
}

/// The (cell, phase) pairs of a phase code (the first phase of each block).
pub fn phase_cells(x: &BitVector, phases: usize) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for b in active(x) {
        let j = b / phases;
        if out.last().map(|l| l.0) != Some(j) {
            out.push((j, b % phases));
        }
    }
    out
}

/// Gives a plain pattern phases: each active bit j becomes cell j with a phase from a
/// fixed hash of (the whole pattern, j), so every distinct event gets its own random
/// phases (as spikes of an assembly fall at event-specific phases of a rhythm).
pub struct Phase {
    phases: usize,
    seed: u64,
    out: BitVector,
}

impl Phase {
    pub fn new(phases: usize, seed: u64) -> Self {
        Self { phases: phases.max(1), seed, out: BitVector::EMPTY }
    }
}

impl Module for Phase {
    fn name(&self) -> String {
        format!("Phase({})", self.phases)
    }
    fn n_inputs(&self) -> usize {
        1
    }
    fn n_outputs(&self) -> usize {
        1
    }
    fn tick(&mut self, inputs: &[&BitVector], _ctx: &mut Ctx) {
        let bits = active(inputs[0]);
        let key = bits.iter().fold(self.seed, |h, &b| mix64(h ^ b as u64));
        let cells: Vec<(usize, usize)> = bits.iter().map(|&j| (j, (mix64(key ^ j as u64) % self.phases as u64) as usize)).collect();
        self.out = phase_code(&cells, inputs[0].bit_len().max(1), self.phases);
    }
    fn output(&self, port: usize) -> &BitVector {
        if port == 0 { &self.out } else { &EMPTY }
    }
    fn reset(&mut self) {
        self.out = BitVector::EMPTY;
    }
}

/// Drops the phases of a phase code: the active cells as a plain pattern.
pub struct Cells {
    phases: usize,
    out: BitVector,
}

impl Module for Cells {
    fn name(&self) -> String {
        format!("Cells({})", self.phases)
    }
    fn n_inputs(&self) -> usize {
        1
    }
    fn n_outputs(&self) -> usize {
        1
    }
    fn tick(&mut self, inputs: &[&BitVector], _ctx: &mut Ctx) {
        let n = (inputs[0].bit_len() / self.phases).max(1);
        let mut out = zeros(n);
        for (j, _) in phase_cells(inputs[0], self.phases) {
            out.bit_set(j);
        }
        self.out = out;
    }
    fn output(&self, port: usize) -> &BitVector {
        if port == 0 { &self.out } else { &EMPTY }
    }
    fn reset(&mut self) {
        self.out = BitVector::EMPTY;
    }
}

/// A Hebbian population with phase-coded (integer complex) weights: a phasor
/// associative memory.
///
/// Inputs `[teach, pre]`, output `[activity]`, all phase codes (`pre` may be a plain
/// pattern if `in_phases` = 1: every input at phase 0).
/// - **Write:** for each active input i (phase φ_i) and teach cell j (phase θ_j),
///   w_ij += e^{i(θ_j − φ_i)}, as integer (cos, sin) pairs from a table (scale 2^14).
/// - **Read:** h_j = Σ_i w_ij · e^{iφ_i}. A cell's score is |h_j|²; the k best cells fire,
///   each at the table phase nearest to arg h_j.
/// - **Why phases:** the stored event's terms all arrive at the same phase and add up
///   (|signal| = cue size); other events' terms arrive at unrelated phases and largely
///   cancel (their sum grows like √n, not n). Crosstalk has mean zero with nothing
///   subtracted: the complex sum is the projection of each cell's phase histogram onto
///   its first harmonic, which has no constant term.
/// - `inverse`: weight each input row by 1/n (its writes), as `Scale::Inverse`.
pub struct PhaseAssociate {
    cells: usize,
    k: usize,
    phases: usize,
    in_phases: usize,
    inverse: bool,
    table: Vec<(i32, i32)>,
    in_table: Vec<(i32, i32)>,
    rows: Vec<Option<Vec<(i32, i32)>>>,
    writes: Vec<u32>,
    out: BitVector,
}

impl PhaseAssociate {
    pub fn new(cells: usize, k: usize, phases: usize, in_phases: usize, inverse: bool) -> Self {
        let phases = phases.max(1);
        let in_phases = in_phases.max(1);
        Self {
            cells,
            k,
            phases,
            in_phases,
            inverse,
            table: phasors(phases),
            in_table: phasors(in_phases),
            rows: Vec::new(),
            writes: Vec::new(),
            out: zeros(cells * phases),
        }
    }

    fn inputs(&self, x: &BitVector) -> Vec<(usize, usize)> {
        if self.in_phases == 1 { active(x).into_iter().map(|i| (i, 0)).collect() } else { phase_cells(x, self.in_phases) }
    }
}

/// (a + ib)(c + id) / 2^14.
fn cmul(a: (i32, i32), b: (i32, i32)) -> (i32, i32) {
    let (a0, a1, b0, b1) = (a.0 as i64, a.1 as i64, b.0 as i64, b.1 as i64);
    (((a0 * b0 - a1 * b1) >> 14) as i32, ((a0 * b1 + a1 * b0) >> 14) as i32)
}

impl Module for PhaseAssociate {
    fn name(&self) -> String {
        format!("PhaseAssociate({} cells, k={}, {} phases)", self.cells, self.k, self.phases)
    }
    fn n_inputs(&self) -> usize {
        2
    }
    fn n_outputs(&self) -> usize {
        1
    }
    fn tick(&mut self, inputs: &[&BitVector], _ctx: &mut Ctx) {
        let pre = self.inputs(inputs[1]);
        let teach = phase_cells(inputs[0], self.phases);
        if !teach.is_empty() {
            for &(i, phi) in &pre {
                if i >= self.rows.len() {
                    self.rows.resize_with(i + 1, || None);
                    self.writes.resize(i + 1, 0);
                }
                let (c, sn) = self.in_table[phi];
                let conj = (c, -sn);
                let row = self.rows[i].get_or_insert_with(|| vec![(0, 0); self.cells]);
                for &(j, theta) in &teach {
                    if j < self.cells {
                        let z = cmul(self.table[theta], conj);
                        row[j].0 += z.0;
                        row[j].1 += z.1;
                    }
                }
                self.writes[i] = self.writes[i].saturating_add(1);
            }
            self.out = inputs[0].clone();
            return;
        }
        let mut h = vec![(0i64, 0i64); self.cells];
        for &(i, phi) in &pre {
            let Some(Some(row)) = self.rows.get(i) else { continue };
            let per = if self.inverse { recip(self.writes[i].max(1) as u64) as i64 } else { ONE as i64 };
            let rot = self.in_table[phi];
            for (hj, &w) in h.iter_mut().zip(row.iter()) {
                if w != (0, 0) {
                    let z = cmul(w, rot);
                    hj.0 += z.0 as i64 * per >> 4;
                    hj.1 += z.1 as i64 * per >> 4;
                }
            }
        }
        // scores |h|² (scaled down to stay in u64), then the k best cells and their phases
        let score: Vec<u64> = h.iter().map(|&(a, b)| ((a as i128 * a as i128 + b as i128 * b as i128) >> 32) as u64).collect();
        let winners = top_k(score.into_iter(), self.k);
        let cells: Vec<(usize, usize)> = winners
            .iter()
            .map(|&j| {
                let (a, b) = h[j as usize];
                let best = (0..self.phases).max_by_key(|&p| a * self.table[p].0 as i64 + b * self.table[p].1 as i64).unwrap_or(0);
                (j as usize, best)
            })
            .collect();
        self.out = phase_code(&cells, self.cells, self.phases);
    }
    fn output(&self, port: usize) -> &BitVector {
        if port == 0 { &self.out } else { &EMPTY }
    }
    fn reset(&mut self) {
        self.out = zeros(self.cells * self.phases);
    }
}

// ---------------------------------------------------------------------------
// Networks

/// Where a wire reads from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Src {
    /// The network's input port.
    In(usize),
    /// Child `.0`'s output port `.1`.
    Child(usize, usize),
}

/// A module made of modules (see the module docs for timing).
pub struct Network {
    pub label: String,
    n_in: usize,
    pub children: Vec<Box<dyn Module>>,
    /// For each child, the source of each input port (`None` = unconnected).
    pub wires: Vec<Vec<Option<Src>>>,
    /// The source of each output port.
    pub exports: Vec<Option<Src>>,
    ins: Vec<BitVector>,
    outs: Vec<BitVector>,
}

impl Network {
    pub fn new(label: &str, n_in: usize) -> Self {
        Self { label: label.into(), n_in, children: Vec::new(), wires: Vec::new(), exports: Vec::new(), ins: Vec::new(), outs: Vec::new() }
    }

    /// Place `m` with its input ports wired from `inputs` (missing ones unconnected).
    /// Returns its child index.
    pub fn place(&mut self, m: Box<dyn Module>, inputs: &[Option<Src>]) -> usize {
        let mut w = inputs.to_vec();
        w.resize(m.n_inputs(), None);
        self.children.push(m);
        self.wires.push(w);
        self.children.len() - 1
    }

    /// Rewire one input port of a placed child (e.g. to close a loop).
    pub fn connect(&mut self, child: usize, port: usize, src: Src) {
        if let Some(slot) = self.wires.get_mut(child).and_then(|w| w.get_mut(port)) {
            *slot = Some(src);
        }
    }

    /// Expose `src` as the next output port.
    pub fn export(&mut self, src: Option<Src>) {
        self.exports.push(src);
    }

    fn read(&self, src: Option<Src>) -> &BitVector {
        match src {
            Some(Src::In(i)) => self.ins.get(i).unwrap_or(&EMPTY),
            Some(Src::Child(c, p)) => self.children.get(c).map(|m| m.output(p)).unwrap_or(&EMPTY),
            None => &EMPTY,
        }
    }
}

impl Module for Network {
    fn name(&self) -> String {
        format!("{} [{} in, {} out]", self.label, self.n_in, self.exports.len())
    }
    fn n_inputs(&self) -> usize {
        self.n_in
    }
    fn n_outputs(&self) -> usize {
        self.exports.len()
    }
    fn tick(&mut self, inputs: &[&BitVector], ctx: &mut Ctx) {
        self.ins = inputs.iter().map(|x| (*x).clone()).collect();
        for c in 0..self.children.len() {
            let vals: Vec<BitVector> = self.wires[c].iter().map(|&s| self.read(s).clone()).collect();
            let refs: Vec<&BitVector> = vals.iter().collect();
            self.children[c].tick(&refs, ctx);
        }
        self.outs = self.exports.iter().map(|&s| self.read(s).clone()).collect();
    }
    fn output(&self, port: usize) -> &BitVector {
        self.outs.get(port).unwrap_or(&EMPTY)
    }
    fn sleep(&mut self) {
        for c in &mut self.children {
            c.sleep();
        }
    }
    fn reset(&mut self) {
        for c in &mut self.children {
            c.reset();
        }
        self.outs.clear();
    }
    fn kernels(&self) -> usize {
        self.children.iter().map(|c| c.kernels()).sum()
    }
    fn describe(&self, depth: usize) -> String {
        let mut s = format!("{}{}\n", "  ".repeat(depth), self.name());
        for c in &self.children {
            s += &c.describe(depth + 1);
        }
        s
    }
}

// ---------------------------------------------------------------------------
// The grammar

/// The learning rule of a predictive kernel class, as genes: what the harness used to set
/// from environment variables per run. `Default` is the plain class.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct KernelSpec {
    /// Near-miss generalisation: drop silent inputs of near-matching kernels that would
    /// have been right (fraction kept), after `generalize_after` misses.
    pub generalize: Option<f32>, // float: config (the kernel class's own type)
    pub generalize_after: u8,
    /// Learn fully only on surprise; an expected target confirms the winner.
    pub surprise_gate: bool,
    /// Canonical kernels (deterministic sampling, hash-consing at growth).
    pub canonical: bool,
    /// Memoised interpretation.
    pub memo: bool,
    /// Uncertainty-gated growth `(k, min)` (see `KernelClass::set_growth_gate`).
    pub growth_gate: Option<(u32, u16)>,
    /// Slow learning: a miss grows a kernel with this probability (`Q16`).
    pub growth_prob: Option<Q16>,
    /// Credit-tagged synapses need this many times the confirmations before pruning.
    pub sticky: Option<u8>,
    /// New kernels sample only the target's bits in a frame that contains them.
    pub copy_growth: bool,
    /// A lucky unreliable kernel (below this rate) does not block growth.
    pub grow_trust: Option<(u16, u16)>,
    /// Depth only outranks reliability among kernels at least this reliable.
    pub trust_floor: Option<(u16, u16)>,
    /// Recent inputs kept for sleep replay (0 = none).
    pub replay: usize,
    /// Take no part in sleep (no consolidation of this class).
    pub no_sleep: bool,
    /// Most kernels the class may hold (0: the class's default).
    pub max_kernels: usize,
    /// Share of a kernel's sampled bits that must be present for it to match (0: default).
    pub match_fraction: Option<f32>, // float: config (the kernel class's own type)
    /// Share of the target a prediction may miss and still count as right (0: default).
    pub surprise_fraction: Option<f32>, // float: config (the kernel class's own type)
}

impl KernelSpec {
    /// A predictive kernel class with this rule.
    pub fn class(&self, bits: usize, frames: usize, sample_bits: usize) -> KernelClass<SimpleKernel> {
        let d = GrowthConfig::default();
        let cfg = GrowthConfig {
            max_kernels: if self.max_kernels > 0 { self.max_kernels } else { d.max_kernels },
            frame_words: bits.div_ceil(64),
            max_frames: frames.max(1),
            sample_bits,
            match_fraction: self.match_fraction.unwrap_or(d.match_fraction),
            surprise_fraction: self.surprise_fraction.unwrap_or(d.surprise_fraction),
            generalize: self.generalize,
            generalize_after: self.generalize_after.max(1),
        };
        let mut c = KernelClass::predictive(cfg);
        c.set_surprise_gate(self.surprise_gate);
        c.set_canonical(self.canonical);
        c.set_memo(self.memo);
        c.set_growth_gate(self.growth_gate);
        c.set_growth_probability(self.growth_prob);
        if let Some(f) = self.sticky {
            c.set_sticky(f, None);
        }
        c.set_copy_growth(self.copy_growth);
        c.set_growth_trust(self.grow_trust);
        c.set_trust_floor(self.trust_floor);
        if self.replay > 0 {
            c.set_replay(self.replay);
        }
        c
    }
}

/// A base kernel the grammar can place.
#[derive(Clone, Debug, PartialEq)]
pub enum Prim {
    /// A `Predictor` whose kernel class follows `spec` (its learning-rule genes).
    PredictWith { bits: usize, frames: usize, sample_bits: usize, spec: KernelSpec },
    /// A `Predictor` with `bits` outputs, reading inputs of `frames` frames, each `bits`
    /// wide, growing kernels that sample `sample_bits` bits per frame.
    Predict { bits: usize, frames: usize, sample_bits: usize },
    Op(KernelOp),
    Delay,
    Concat(usize),
    Separate { cells: usize, fan_out: usize, k: usize, seed: u64 },
    /// `Separate` with a fixed random fan-in table over `inputs` bits.
    SeparateTable { inputs: usize, cells: usize, fan_in: usize, k: usize, seed: u64 },
    Scatter { inputs: usize, outputs: usize, seed: u64 },
    Gate,
    /// An `Associate` population: one input pathway per entry of `pathways` (its
    /// presynaptic scaling), recurrent settling steps (0 = no recurrent pathway).
    /// Plain pattern → phase code (`Phase`); phase code → plain cells (`Cells`).
    Phase { phases: usize, seed: u64 },
    Cells { phases: usize },
    /// A `PhaseAssociate` population.
    PhaseAssociate { cells: usize, k: usize, phases: usize, in_phases: usize, inverse: bool },
    Associate { cells: usize, pathways: Vec<Scale>, settle: usize, readout: Readout, half_life: u32, gain: Q16, gain_k: usize },
}

impl Prim {
    pub fn build(&self) -> Box<dyn Module> {
        match self {
            &Prim::Predict { bits, frames, sample_bits } => {
                let cfg = GrowthConfig { frame_words: bits.div_ceil(64), max_frames: frames.max(1), sample_bits, ..GrowthConfig::default() };
                Box::new(Predictor::new(bits, KernelClass::predictive(cfg)))
            }
            Prim::PredictWith { bits, frames, sample_bits, spec } => Box::new(Predictor::new(*bits, spec.class(*bits, *frames, *sample_bits))),
            &Prim::Op(op) => Box::new(BitOp::new(op)),
            Prim::Delay => Box::<Delay>::default(),
            &Prim::Concat(n) => Box::new(Concat::new(n)),
            &Prim::Separate { cells, fan_out, k, seed } => Box::new(Separate::new(cells, fan_out, k, seed)),
            &Prim::SeparateTable { inputs, cells, fan_in, k, seed } => Box::new(Separate::table(inputs, cells, fan_in, k, seed)),
            &Prim::Scatter { inputs, outputs, seed } => Box::new(Scatter::new(inputs, outputs, seed)),
            Prim::Gate => Box::new(Gate { out: BitVector::EMPTY }),
            &Prim::Phase { phases, seed } => Box::new(Phase::new(phases, seed)),
            &Prim::Cells { phases } => Box::new(Cells { phases: phases.max(1), out: BitVector::EMPTY }),
            &Prim::PhaseAssociate { cells, k, phases, in_phases, inverse } => Box::new(PhaseAssociate::new(cells, k, phases, in_phases, inverse)),
            Prim::Associate { cells, pathways, settle, readout, half_life, gain, gain_k } => {
                Box::new(Associate::new(*cells, pathways, *settle, *readout, *half_life, *gain, *gain_k))
            }
        }
    }
}

/// One instruction of the network grammar. The builder keeps a stack of signals.
#[derive(Clone, Debug, PartialEq)]
pub enum NetOp {
    /// Push the network's input port `i` (no-op if out of range).
    In(usize),
    /// Place a base kernel: pop its inputs (the last input on top), push its outputs
    /// (the last output on top).
    Place(Prim),
    /// Place definition `k` of the genome (only earlier definitions; else a no-op), the
    /// same way as `Place`.
    Sub(usize),
    Dup,
    Swap,
    Drop,
    /// Push a copy of the second signal.
    Over,
    /// Push a copy of the signal `n` below the top (`Pick(0)` = `Dup`, `Pick(1)` = `Over`).
    Pick(usize),
    /// Push a forward reference: a signal that will be bound later by `Close`. Reading it
    /// gives the bound signal's previous-tick value, if it is later in the network.
    Feedback,
    /// Pop a signal and bind the most recent open `Feedback` to it.
    Close,
    /// Pop a signal and expose it as the next output port.
    Out,
    Nop,
    /// Push an unconnected signal: an empty frame (a slot reserved in a `Concat`).
    Zero,
    /// Push a number on the number stack.
    Num(i64),
    /// Pop a number into a gene of this definition (see `Gene`).
    Set(Gene),
    /// Place a base kernel whose parameters are popped from the number stack (see `Kind`),
    /// then its signal inputs, as `Place`.
    Make(Kind),
}

/// A gene: one number of a definition's learning rule or (in the top definition) its
/// update loop. Fractions are `Q16` (65536 = 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gene {
    Generalize,
    GeneralizeAfter,
    SurpriseGate,
    Canonical,
    Memo,
    GrowthGate,
    GrowthGateMin,
    GrowthProb,
    Sticky,
    CopyGrowth,
    GrowTrust,
    TrustFloor,
    Replay,
    SleepEvery,
    ResetAtStory,
    ResetAtSentence,
    /// 1: this definition's predictors consolidate at sleep (the default); 0: they do not.
    Sleep,
    MaxKernels,
    MatchFraction,
    SurpriseFraction,
}

impl Gene {
    pub const ALL: [(&'static str, Gene); 20] = [
        ("max_kernels", Gene::MaxKernels),
        ("match_fraction", Gene::MatchFraction),
        ("surprise_fraction", Gene::SurpriseFraction),
        ("sleep", Gene::Sleep),
        ("generalize", Gene::Generalize),
        ("generalize_after", Gene::GeneralizeAfter),
        ("surprise_gate", Gene::SurpriseGate),
        ("canonical", Gene::Canonical),
        ("memo", Gene::Memo),
        ("growth_gate", Gene::GrowthGate),
        ("growth_gate_min", Gene::GrowthGateMin),
        ("growth_prob", Gene::GrowthProb),
        ("sticky", Gene::Sticky),
        ("copy_growth", Gene::CopyGrowth),
        ("grow_trust", Gene::GrowTrust),
        ("trust_floor", Gene::TrustFloor),
        ("replay", Gene::Replay),
        ("sleep_every", Gene::SleepEvery),
        ("reset_at_story", Gene::ResetAtStory),
        ("reset_at_sentence", Gene::ResetAtSentence),
    ];
}

/// A base kernel placed by `Make`, with the numbers it pops (in the order pushed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `bits frames sample_bits` → a `Predictor` with this definition's genes; inputs
    /// `[input, teach]`, outputs `[prediction, surprise, confidence]`.
    Predict,
    /// `n` → `Concat(n)`.
    Concat,
    Delay,
    Gate,
    And,
    Or,
    Xor,
    Clear,
    /// → `Bag`; inputs `[x, clear]`.
    Bag,
    /// `span keep` → `Window`; inputs `[x, advance]`.
    Window,
    /// `threshold` → `Surprise`; inputs `[x, prediction, confidence]`.
    Surprise,
}

impl Kind {
    pub const ALL: [(&'static str, Kind); 11] = [
        ("predict", Kind::Predict),
        ("concat", Kind::Concat),
        ("delay", Kind::Delay),
        ("gate", Kind::Gate),
        ("and", Kind::And),
        ("or", Kind::Or),
        ("xor", Kind::Xor),
        ("clear", Kind::Clear),
        ("bag", Kind::Bag),
        ("window", Kind::Window),
        ("surprise", Kind::Surprise),
    ];
}

/// The genes a definition has set so far.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Genes {
    pub(crate) spec: KernelSpec,
    pub(crate) schedule: Schedule,
}

impl Genes {
    pub(crate) fn set(&mut self, g: Gene, v: i64) {
        let q = |v: i64| v.clamp(0, ONE as i64) as Q16;
        // a Q16 fraction as a ratio the kernel class compares by cross-multiplication
        let r = |v: i64| (((v.clamp(0, ONE as i64) as u64 * 1000) >> 16) as u16, 1000u16);
        let s = &mut self.spec;
        match g {
            Gene::Generalize => s.generalize = (v > 0).then(|| q(v) as f32 / ONE as f32), // float: config (the class's own type)
            Gene::GeneralizeAfter => s.generalize_after = v.clamp(1, 255) as u8,
            Gene::SurpriseGate => s.surprise_gate = v != 0,
            Gene::Canonical => s.canonical = v != 0,
            Gene::Memo => s.memo = v != 0,
            Gene::GrowthGate => s.growth_gate = (v > 0).then(|| (v as u32, s.growth_gate.map_or(16, |g| g.1))),
            Gene::GrowthGateMin => s.growth_gate = s.growth_gate.map(|g| (g.0, v.clamp(0, u16::MAX as i64) as u16)),
            Gene::GrowthProb => s.growth_prob = (v > 0).then(|| q(v)),
            Gene::Sticky => s.sticky = (v > 0).then(|| v.clamp(1, 255) as u8),
            Gene::CopyGrowth => s.copy_growth = v != 0,
            Gene::GrowTrust => s.grow_trust = (v > 0).then(|| r(v)),
            Gene::TrustFloor => s.trust_floor = (v > 0).then(|| r(v)),
            Gene::Replay => s.replay = v.max(0) as usize,
            Gene::SleepEvery => self.schedule.sleep_every = v.max(0) as usize,
            Gene::ResetAtStory => self.schedule.reset_at_story = v != 0,
            Gene::ResetAtSentence => self.schedule.reset_at_sentence = v != 0,
            Gene::Sleep => s.no_sleep = v == 0,
            Gene::MaxKernels => s.max_kernels = v.max(0) as usize,
            Gene::MatchFraction => s.match_fraction = (v > 0).then(|| q(v) as f32 / ONE as f32), // float: config (the class's own type)
            Gene::SurpriseFraction => s.surprise_fraction = (v > 0).then(|| q(v) as f32 / ONE as f32), // float: config
        }
    }

    /// Build a `Make` kernel from the numbers it pops.
    pub(crate) fn make(&self, k: Kind, nums: &mut Vec<i64>) -> Box<dyn Module> {
        let mut pop = |n: usize| -> Vec<i64> {
            let mut v: Vec<i64> = (0..n).map(|_| nums.pop().unwrap_or(0)).collect();
            v.reverse();
            v
        };
        match k {
            Kind::Predict => {
                let a = pop(3);
                let (bits, frames, sample) = (a[0].max(64) as usize, a[1].max(1) as usize, a[2].max(1) as usize);
                let mut p = Predictor::new(bits, self.spec.class(bits, frames, sample));
                p.confidence_port = true;
                p.no_sleep = self.spec.no_sleep;
                Box::new(p)
            }
            Kind::Concat => Box::new(Concat::new(pop(1)[0].max(1) as usize)),
            Kind::Delay => Box::<Delay>::default(),
            Kind::Gate => Box::new(Gate { out: BitVector::EMPTY }),
            Kind::And => Box::new(BitOp::new(KernelOp::And)),
            Kind::Or => Box::new(BitOp::new(KernelOp::Or)),
            Kind::Xor => Box::new(BitOp::new(KernelOp::Xor)),
            Kind::Clear => Box::new(BitOp::new(KernelOp::Clear)),
            Kind::Bag => Box::<Bag>::default(),
            Kind::Window => {
                let a = pop(2);
                Box::new(Window::new(a[0].max(1) as usize, a[1] != 0))
            }
            Kind::Surprise => Box::new(Surprise { threshold: pop(1)[0].clamp(0, ONE as i64) as Q16, out: BitVector::EMPTY }),
        }
    }
}


/// One module definition: its number of inputs and the code that builds it.
#[derive(Clone, Debug, PartialEq)]
pub struct NetDef {
    pub label: String,
    pub inputs: usize,
    pub code: Vec<NetOp>,
}

/// The update loop: what the network does besides ticking once per word.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Schedule {
    /// Clear activity at the end of each sentence.
    pub reset_at_sentence: bool,
    /// Clear activity at the end of each story (what was learned is kept).
    pub reset_at_story: bool,
    /// Sleep (offline consolidation) every this many stories while learning; 0 = never.
    pub sleep_every: usize,
}

/// A library of definitions; later ones may place earlier ones (`Sub`). The last is the
/// whole architecture (the initial configuration); the genes its code sets include its
/// update loop (`Schedule`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Genome {
    pub defs: Vec<NetDef>,
}

/// A signal on the builder's stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sig {
    Real(Src),
    Pending(usize),
}

impl Genome {
    /// Add a definition and return its index.
    pub fn define(&mut self, label: &str, inputs: usize, code: Vec<NetOp>) -> usize {
        self.defs.push(NetDef { label: label.into(), inputs, code });
        self.defs.len() - 1
    }

    /// Build definition `k` as a network.
    pub fn build(&self, k: usize) -> Network {
        self.build_with_genes(k).0
    }

    /// Build the last definition (the whole architecture) and its update loop.
    pub fn build_top_and_schedule(&self) -> (Network, Schedule) {
        let (net, genes) = self.build_with_genes(self.defs.len() - 1);
        (net, genes.schedule)
    }

    fn build_with_genes(&self, k: usize) -> (Network, Genes) {
        let def = &self.defs[k];
        let mut genes = Genes::default();
        let mut nums: Vec<i64> = Vec::new();
        let mut net = Network::new(&def.label, def.inputs);
        let mut stack: Vec<Option<Sig>> = Vec::new();
        let mut bound: Vec<Option<Option<Sig>>> = Vec::new(); // per Feedback: its binding
        let mut open: Vec<usize> = Vec::new();
        let mut pending_wires: Vec<(usize, usize, usize)> = Vec::new(); // (child, port, feedback)
        let mut exports: Vec<Option<Sig>> = Vec::new();

        let place = |net: &mut Network, stack: &mut Vec<Option<Sig>>, pending_wires: &mut Vec<(usize, usize, usize)>, m: Box<dyn Module>| {
            let n = m.n_inputs();
            let mut args: Vec<Option<Sig>> = (0..n).map(|_| stack.pop().flatten()).collect();
            args.reverse();
            let srcs: Vec<Option<Src>> = args.iter().map(|a| match a { Some(Sig::Real(s)) => Some(*s), _ => None }).collect();
            let outs = m.n_outputs();
            let c = net.place(m, &srcs);
            for (port, a) in args.iter().enumerate() {
                if let Some(Sig::Pending(f)) = a {
                    pending_wires.push((c, port, *f));
                }
            }
            for p in 0..outs {
                stack.push(Some(Sig::Real(Src::Child(c, p))));
            }
        };

        for op in &def.code {
            match op {
                NetOp::In(i) => {
                    if *i < def.inputs {
                        stack.push(Some(Sig::Real(Src::In(*i))));
                    }
                }
                NetOp::Place(p) => place(&mut net, &mut stack, &mut pending_wires, p.build()),
                NetOp::Sub(j) => {
                    if *j < k {
                        place(&mut net, &mut stack, &mut pending_wires, Box::new(self.build(*j)));
                    }
                }
                NetOp::Dup => {
                    if let Some(&top) = stack.last() {
                        stack.push(top);
                    }
                }
                NetOp::Swap => {
                    let n = stack.len();
                    if n >= 2 {
                        stack.swap(n - 1, n - 2);
                    }
                }
                NetOp::Drop => {
                    stack.pop();
                }
                NetOp::Over => {
                    let n = stack.len();
                    if n >= 2 {
                        stack.push(stack[n - 2]);
                    }
                }
                NetOp::Pick(n) => {
                    let len = stack.len();
                    if *n < len {
                        stack.push(stack[len - 1 - n]);
                    }
                }
                NetOp::Feedback => {
                    bound.push(None);
                    open.push(bound.len() - 1);
                    stack.push(Some(Sig::Pending(bound.len() - 1)));
                }
                NetOp::Close => {
                    let s = stack.pop().flatten();
                    if let Some(f) = open.pop() {
                        bound[f] = Some(s);
                    }
                }
                NetOp::Out => exports.push(stack.pop().flatten()),
                NetOp::Nop => {}
                NetOp::Zero => stack.push(None),
                NetOp::Num(n) => nums.push(*n),
                NetOp::Set(g) => {
                    let v = nums.pop().unwrap_or(0);
                    genes.set(*g, v);
                }
                NetOp::Make(kind) => {
                    let m = genes.make(*kind, &mut nums);
                    place(&mut net, &mut stack, &mut pending_wires, m);
                }
            }
        }

        // Resolve forward references (a chain of them ends at a real source or nothing).
        let resolve = |mut s: Option<Sig>| -> Option<Src> {
            for _ in 0..=bound.len() {
                match s {
                    Some(Sig::Real(src)) => return Some(src),
                    Some(Sig::Pending(f)) => s = bound[f].flatten(),
                    None => return None,
                }
            }
            None // a cycle of references
        };
        for (c, port, f) in pending_wires {
            if let Some(src) = resolve(Some(Sig::Pending(f))) {
                net.connect(c, port, src);
            }
        }
        for e in exports {
            net.export(resolve(e));
        }
        (net, genes)
    }

    /// Read a genome from text. Each definition is `def <name> <inputs>` … `end`. Inside:
    /// a number pushes it on the number stack; `in:i`, `sub:<name>`, `pick:n`,
    /// `set:<gene>`, `make:<kind>`; and `dup swap drop over feedback close out zero nop`.
    /// `#` starts a comment.
    pub fn parse(text: &str) -> Result<Genome, String> {
        let mut g = Genome::default();
        let mut names: Vec<String> = Vec::new();
        let mut cur: Option<(String, usize, Vec<NetOp>)> = None;
        let toks: Vec<&str> = text.lines().flat_map(|l| l.split('#').next().unwrap_or("").split_whitespace()).collect();
        let mut i = 0;
        while i < toks.len() {
            let t = toks[i];
            i += 1;
            if t == "def" {
                let name = toks.get(i).ok_or("def without a name")?.to_string();
                let n: usize = toks.get(i + 1).and_then(|x| x.parse().ok()).ok_or(format!("def {name}: inputs"))?;
                i += 2;
                cur = Some((name, n, Vec::new()));
                continue;
            }
            let Some((name, n, code)) = cur.as_mut() else { return Err(format!("`{t}` outside a def")) };
            if t == "end" {
                g.define(name, *n, std::mem::take(code));
                names.push(name.clone());
                cur = None;
                continue;
            }
            let arg = |p: &str| t.strip_prefix(p);
            let op = if let Ok(v) = t.parse::<i64>() {
                NetOp::Num(v)
            } else if let Some(a) = arg("in:") {
                NetOp::In(a.parse().map_err(|_| format!("bad {t}"))?)
            } else if let Some(a) = arg("pick:") {
                NetOp::Pick(a.parse().map_err(|_| format!("bad {t}"))?)
            } else if let Some(a) = arg("sub:") {
                NetOp::Sub(names.iter().position(|x| x == a).ok_or(format!("{t}: no earlier def named {a}"))?)
            } else if let Some(a) = arg("set:") {
                NetOp::Set(Gene::ALL.iter().find(|x| x.0 == a).ok_or(format!("{t}: no such gene"))?.1)
            } else if let Some(a) = arg("make:") {
                NetOp::Make(Kind::ALL.iter().find(|x| x.0 == a).ok_or(format!("{t}: no such kernel"))?.1)
            } else {
                match t {
                    "dup" => NetOp::Dup,
                    "swap" => NetOp::Swap,
                    "drop" => NetOp::Drop,
                    "over" => NetOp::Over,
                    "feedback" => NetOp::Feedback,
                    "close" => NetOp::Close,
                    "out" => NetOp::Out,
                    "zero" => NetOp::Zero,
                    "nop" => NetOp::Nop,
                    _ => return Err(format!("unknown word `{t}`")),
                }
            };
            code.push(op);
        }
        if cur.is_some() {
            return Err("a def without end".into());
        }
        Ok(g)
    }

    /// Build the last definition (the whole architecture).
    pub fn build_top(&self) -> Network {
        self.build(self.defs.len() - 1)
    }
}

// ---------------------------------------------------------------------------
// Architectures written in the grammar

/// A cortical column (`CorticalColumn` with no relayed frames), in the grammar.
///
/// Inputs: `[x]` (the current input). Outputs: `[prediction, surprise]`.
/// L6 = `Delay` (the previous input), L4 = `Concat(x, previous)`, L2/3 = a `Predictor`
/// taught by the next input. The surprise is the predictor's bitwise error.
pub fn column_code(bits: usize, sample_bits: usize) -> Vec<NetOp> {
    use NetOp::*;
    vec![
        In(0),
        Dup,
        Place(Prim::Delay),
        Place(Prim::Concat(2)),
        In(0),
        Place(Prim::Predict { bits, frames: 2, sample_bits }),
        Swap,
        Out,
        Out,
    ]
}

/// A column with a top-down frame: inputs `[x, topdown]`, L4 = `Concat(x, topdown,
/// previous)`. Outputs `[prediction, surprise]`.
pub fn column_td_code(bits: usize, sample_bits: usize) -> Vec<NetOp> {
    use NetOp::*;
    vec![
        In(0),
        In(1),
        In(0),
        Place(Prim::Delay),
        Place(Prim::Concat(3)),
        In(0),
        Place(Prim::Predict { bits, frames: 3, sample_bits }),
        Swap,
        Out,
        Out,
    ]
}

/// A two-level hierarchy, built from the column definitions:
/// - **state**: the lower column's surprise, OR-accumulated over the story (a loop
///   through `Delay`), the slow context of `HigherArea`;
/// - **higher area**: a column whose input is `Concat(x, state)`… written as a
///   `Predictor` reading `[x, state]` and taught by `x`;
/// - **lower column**: reads `x` and the higher area's prediction as its top-down frame.
///
/// Defines: 0 = column with top-down, 1 = the hierarchy. Inputs `[x]`, outputs
/// `[lower prediction, higher prediction]`.
pub fn hierarchy_genome(bits: usize, sample_bits: usize) -> Genome {
    use NetOp::*;
    let mut g = Genome::default();
    let col = g.define("column+td", 2, column_td_code(bits, sample_bits));
    g.define(
        "hierarchy",
        1,
        vec![
            // state = Delay(state | surprise): the lower column's surprise (a feedback
            // signal, bound below) OR its own previous value.
            Feedback,                        // [s?]            the lower surprise
            Feedback,                        // [s? st?]        the state itself
            Place(Prim::Op(KernelOp::Or)),   // [s|st]
            Place(Prim::Delay),              // [state]
            Dup,                             // [state state]
            Close,                           // bind st? to state; [state]
            // higher area: predict x(t+1) from [x, state], taught by x.
            In(0),                           // [state x]
            Swap,                            // [x state]
            Place(Prim::Concat(2)),          // [hi_in]
            In(0),                           // [hi_in x]
            Place(Prim::Predict { bits, frames: 2, sample_bits }), // [hi_pred hi_surp]
            Drop,                            // [hi_pred]
            // lower column with the higher prediction as top-down.
            In(0),                           // [hi_pred x]
            Over,                            // [hi_pred x hi_pred]
            Sub(col),                        // [hi_pred lo_pred lo_surp]
            Close,                           // bind s? to lo_surp; [hi_pred lo_pred]
            Out,                             // out 0 = lower prediction
            Out,                             // out 1 = higher prediction
        ],
    );
    g
}

/// The hippocampal circuit of `Hippocampus` (default settings: table projections,
/// presynaptic scaling on the perforant path, novelty-gated encoding; no CA2, tags or
/// hashed input), as a genome of base kernels.
///
/// Inputs: `[x (EC II/III), out (what EC V should give back), encode]`.
/// Outputs: `[EC V readout, novelty (CA1 cells the comparator found unmatched), CA3]`.
///
/// - DG = `SeparateTable`, mossy fibres = `Scatter`;
/// - CA3 = `Associate` with the perforant path (scaled) and a recurrent pathway, taught
///   by the mossy-fibre code;
/// - EC III → CA1 = `SeparateTable`; CA1 = `Associate` over the Schaffer collaterals,
///   taught by the EC III code;
/// - subiculum → EC V = `Associate` with a fraction readout, taught by `out`;
/// - the comparator = `BitOp(Clear)`: the EC III code's cells that recall did not
///   reproduce. Its popcount is the novelty, fed back as every population's gain.
///
/// One event is two ticks, as in a theta cycle (`theta_store`): a retrieval half
/// (`encode` empty: recall, the comparator measures novelty) and an encoding half
/// (`encode` set: every population is clamped to its teaching code and writes, with
/// the previous half's novelty as gain, through the comparator's feedback wire).
pub fn hippocampus_genome(cfg: &crate::program::HippocampusConfig) -> Genome {
    use NetOp::*;
    let half_life = crate::program::hippocampus::half_life_of(cfg.decay);
    let out_bits = if cfg.out_bits == 0 { cfg.ec_bits } else { cfg.out_bits };
    let pop = |cells: usize, scale: Scale, settle: usize, readout: Readout| Prim::Associate {
        cells,
        pathways: vec![scale],
        settle,
        readout,
        half_life,
        gain: cfg.novelty_gain,
        gain_k: cfg.ca1_k,
    };
    let mut g = Genome::default();
    g.define(
        "hippocampus",
        3,
        vec![
            Feedback, // [G]: novelty, bound to the comparator below
            In(0),
            Place(Prim::SeparateTable { inputs: cfg.ec_bits, cells: cfg.dg_cells, fan_in: cfg.dg_fan_in, k: cfg.dg_k, seed: cfg.seed.wrapping_add(1) }),
            Place(Prim::Scatter { inputs: cfg.dg_cells, outputs: cfg.ca3_cells, seed: cfg.seed }),
            In(2),
            Place(Prim::Gate), // [G cT]: CA3's teaching code (mossy fibres), only when encoding
            Pick(1),
            In(0), // [G cT G x]
            Place(pop(cfg.ca3_cells, Scale::Shift, cfg.settle, Readout::TopK(cfg.dg_k))), // [G ca3]
            In(0),
            Place(Prim::SeparateTable { inputs: cfg.ec_bits, cells: cfg.ca1_cells, fan_in: cfg.ca1_fan_in, k: cfg.ca1_k, seed: cfg.seed.wrapping_add(2) }), // [G ca3 a_cue]
            Dup,
            In(2),
            Place(Prim::Gate), // [G ca3 a_cue aT]
            Pick(3),
            Pick(3), // [G ca3 a_cue aT G ca3]
            Place(pop(cfg.ca1_cells, if cfg.scale_all { Scale::Shift } else { Scale::None }, 0, Readout::TopK(cfg.ca1_k))), // [G ca3 a_cue ca1]
            In(1),
            In(2),
            Place(Prim::Gate), // [G ca3 a_cue ca1 oT]
            Pick(4),
            Pick(2), // [G ca3 a_cue ca1 oT G ca1]
            Place(pop(out_bits, Scale::None, 0, Readout::Fraction(cfg.readout_fraction))), // [G ca3 a_cue ca1 ecv]
            Out, // out 0: EC V
            Place(Prim::Op(KernelOp::Clear)), // [G ca3 mismatch]: a_cue & !ca1
            Dup,
            Close, // G = the mismatch (read next tick)
            Out,   // out 1: novelty bits
            Out,   // out 2: CA3
        ],
    );
    g
}

/// One bit: the `encode` signal.
pub fn on() -> BitVector {
    BitVector::from_words(vec![1])
}

/// Store one event in a `hippocampus_genome` network: a retrieval tick, then an encoding
/// tick. Returns the novelty (`Q16`) the retrieval tick measured.
pub fn theta_store(net: &mut Network, x: &BitVector, out: &BitVector, ctx: &mut Ctx, gain_k: usize) -> Q16 {
    net.tick(&[x, &BitVector::EMPTY, &BitVector::EMPTY], ctx);
    let novelty = ratio(net.output(1).count_ones() as u64, gain_k as u64).min(ONE);
    net.tick(&[x, out, &on()], ctx);
    novelty
}

/// Recall from a cue in a `hippocampus_genome` network (one retrieval tick): EC V.
pub fn theta_recall(net: &mut Network, cue: &BitVector, ctx: &mut Ctx) -> BitVector {
    net.tick(&[cue, &BitVector::EMPTY, &BitVector::EMPTY], ctx);
    net.output(0).clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::{ContextBuffer, CorticalColumn};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn code(rng: &mut StdRng, bits: usize, active: usize) -> BitVector {
        let mut v = zeros(bits);
        while v.count_ones() < active {
            v.bit_set(rng.gen_range(0..bits));
        }
        v
    }

    fn overlap(a: &BitVector, b: &BitVector) -> usize {
        a.as_words().iter().zip(b.as_words()).map(|(x, y)| (x & y).count_ones() as usize).sum()
    }

    #[test]
    fn column_in_the_grammar_matches_cortical_column() {
        let bits = 256;
        let mut vocab_rng = StdRng::seed_from_u64(7);
        let words: Vec<BitVector> = (0..6).map(|_| code(&mut vocab_rng, bits, 16)).collect();
        let seq: Vec<usize> = (0..300).map(|t| [0, 1, 2, 3, 1, 4, 5][t % 7]).collect();

        // Reference: CorticalColumn driven by hand.
        let cfg = GrowthConfig { frame_words: bits / 64, max_frames: 2, sample_bits: 8, ..GrowthConfig::default() };
        let mut col = CorticalColumn::new(bits, KernelClass::predictive(cfg), ContextBuffer::new(bits, 4, vec![]));
        let mut rng_a = StdRng::seed_from_u64(1);
        let mut reference = Vec::new();
        let mut last_l4: Option<BitVector> = None;
        for &w in &seq {
            let x = &words[w];
            if let Some(l4) = &last_l4 {
                col.learn(l4, x, &mut rng_a);
            }
            col.observe(x);
            let l4 = col.assemble(x, &[]);
            reference.push(col.predict(&l4).clone());
            last_l4 = Some(l4);
        }

        // The same column, written in the grammar.
        let mut g = Genome::default();
        g.define("column", 1, column_code(bits, 8));
        let mut net = g.build_top();
        let mut rng_b = StdRng::seed_from_u64(1);
        let mut ctx = Ctx { rng: &mut rng_b, learn: true };
        for (t, &w) in seq.iter().enumerate() {
            net.tick(&[&words[w]], &mut ctx);
            assert_eq!(net.output(0).as_words(), reference[t].as_words(), "step {t}");
        }
        assert!(net.kernels() > 0);
        // And it learned: the last cycle is predicted (except the ambiguous step after 1).
        assert!(overlap(net.output(0), &words[seq[300 % 7]]) >= 12);
    }

    /// Sequences A X Y B and C X Y D: at Y, the next word depends on the first word,
    /// three steps back. A column sees only [Y, X]; the hierarchy also sees the story's
    /// surprising words (A or C) through the higher area.
    #[test]
    fn hierarchy_of_columns_resolves_long_context() {
        let bits = 256;
        let mut vocab_rng = StdRng::seed_from_u64(3);
        let w: Vec<BitVector> = (0..6).map(|_| code(&mut vocab_rng, bits, 16)).collect();
        let stories = [[0usize, 2, 3, 1], [4, 2, 3, 5]];

        let run = |net: &mut Network, seed: u64| -> usize {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut pick = StdRng::seed_from_u64(seed + 100);
            let mut right = 0;
            for trial in 0..400 {
                let s = &stories[pick.gen_range(0..2)];
                net.reset();
                let mut ctx = Ctx { rng: &mut rng, learn: true };
                for (i, &word) in s.iter().enumerate() {
                    net.tick(&[&w[word]], &mut ctx);
                    if i == 2 && trial >= 300 && overlap(net.output(0), &w[s[3]]) >= 12 && overlap(net.output(0), &w[s[3] ^ 4]) < 12 {
                        right += 1;
                    }
                }
            }
            right
        };

        let mut g = Genome::default();
        g.define("column", 1, column_code(bits, 8));
        let flat = run(&mut g.build_top(), 1);
        let mut h = hierarchy_genome(bits, 8).build_top();
        let deep = run(&mut h, 1);
        assert!(deep >= 95, "hierarchy right {deep}/100, column alone {flat}/100");
        assert!(flat <= 70, "column alone should be ambiguous at Y: {flat}/100");
        assert!(h.describe(0).contains("column+td"));
    }

    #[test]
    fn random_genomes_always_build_and_run() {
        let mut rng = StdRng::seed_from_u64(9);
        let prims = [
            Prim::Predict { bits: 128, frames: 2, sample_bits: 4 },
            Prim::Op(KernelOp::Or),
            Prim::Op(KernelOp::Clear),
            Prim::Delay,
            Prim::Concat(2),
            Prim::Separate { cells: 128, fan_out: 4, k: 8, seed: 1 },
            Prim::Scatter { inputs: 128, outputs: 64, seed: 2 },
            Prim::Gate,
            Prim::Phase { phases: 8, seed: 3 },
            Prim::Cells { phases: 8 },
            Prim::PhaseAssociate { cells: 64, k: 4, phases: 8, in_phases: 8, inverse: true },
            Prim::Associate { cells: 128, pathways: vec![Scale::Inverse, Scale::Shift], settle: 1, readout: Readout::TopK(8), half_life: 50, gain: ONE, gain_k: 8 },
        ];
        for _ in 0..50 {
            let mut g = Genome::default();
            for d in 0..3 {
                let ops = (0..rng.gen_range(0..20))
                    .map(|_| match rng.gen_range(0..11) {
                        0 => NetOp::In(rng.gen_range(0..3)),
                        1 => NetOp::Place(prims[rng.gen_range(0..prims.len())].clone()),
                        2 => NetOp::Sub(rng.gen_range(0..4)),
                        3 => NetOp::Dup,
                        4 => NetOp::Swap,
                        5 => NetOp::Drop,
                        6 => NetOp::Over,
                        7 => NetOp::Feedback,
                        8 => NetOp::Close,
                        9 => NetOp::Out,
                        _ => NetOp::Nop,
                    })
                    .collect();
                g.define(&format!("d{d}"), 2, ops);
            }
            let mut net = g.build_top();
            let mut r2 = StdRng::seed_from_u64(2);
            let mut ctx = Ctx { rng: &mut r2, learn: true };
            for t in 0..5 {
                let x = code(&mut rng, 128, 8);
                let y = if t % 2 == 0 { x.clone() } else { BitVector::EMPTY };
                net.tick(&[&x, &y], &mut ctx);
            }
            net.sleep();
            net.reset();
        }
    }

    /// Sequence memory from base kernels: separate each pattern into a sparse code and
    /// let a predictor, taught by the next code, link each to the next. After one
    /// exposure, feeding the predictions back as input (a closed loop) replays the
    /// sequence from its first element.
    #[test]
    fn one_shot_sequence_replay_from_kernels() {
        use NetOp::*;
        let bits = 512;
        let mut g = Genome::default();
        // inputs [x, replaying]: in the loop, the input is x OR the fed-back prediction.
        g.define(
            "sequence memory",
            1,
            vec![
                In(0),
                Place(Prim::Separate { cells: bits, fan_out: 8, k: 16, seed: 5 }), // [code]
                Feedback,                                                         // [code pred?]
                Place(Prim::Op(KernelOp::Or)),                                    // [code|pred]
                Dup,
                Dup, // [c c c]: the predictor's input and teach, and one kept
                Place(Prim::Predict { bits, frames: 1, sample_bits: 12 }),        // [c pred surp]
                Drop,                                                             // [c pred]
                Dup,
                Close, // bind pred? to pred
                Out,   // out 0 = prediction
                Out,   // out 1 = the current code
            ],
        );
        let mut net = g.build_top();
        let mut rng = StdRng::seed_from_u64(4);
        let items: Vec<BitVector> = (0..8).map(|_| code(&mut rng, 1024, 24)).collect();
        let mut ctx = Ctx { rng: &mut rng, learn: true };
        // Store: one pass. The predictor's teaching input is wired to its own input, so
        // each code teaches the previous one's prediction.
        let mut codes = Vec::new();
        for x in &items {
            net.tick(&[x], &mut ctx);
            codes.push(net.output(1).clone());
        }
        net.reset();
        // Replay: cue with the first item, then run with no input.
        ctx.learn = false;
        net.tick(&[&items[0]], &mut ctx);
        let mut recalled = 0;
        for t in 1..items.len() {
            let pred = net.output(0).clone();
            if overlap(&pred, &codes[t]) >= 14 {
                recalled += 1;
            }
            net.tick(&[&BitVector::EMPTY], &mut ctx);
        }
        assert_eq!(recalled, 7, "replayed {recalled}/7 steps");
    }
    /// The genome-built hippocampus against `Hippocampus`: the same events stored, the
    /// same cues recalled, with novelty and recall compared exactly.
    #[test]
    fn hippocampus_genome_matches_the_circuit() {
        use crate::program::{Hippocampus, HippocampusConfig};
        use rand::seq::SliceRandom;
        const BITS: usize = 2048;
        let word = |i: usize| -> Vec<usize> {
            let mut rng = StdRng::seed_from_u64(i as u64 + 77);
            let all: Vec<usize> = (0..BITS).collect();
            all.choose_multiple(&mut rng, 16).copied().collect()
        };
        let episode = |ws: &[usize]| -> Vec<usize> {
            let mut x: Vec<usize> = ws.iter().flat_map(|&w| word(w)).collect();
            x.sort_unstable();
            x.dedup();
            x
        };
        let mut cfg = HippocampusConfig::new(BITS, 5);
        cfg.dg_cells = 4096;
        cfg.ca3_cells = 2048;
        cfg.ca1_cells = 2048;
        cfg.dg_fan_in = 100;
        cfg.ca1_fan_in = 100;
        let mut h = Hippocampus::new(cfg.clone());
        let mut net = hippocampus_genome(&cfg).build_top();
        let mut rng = StdRng::seed_from_u64(9);
        let mut r = StdRng::seed_from_u64(0);
        let mut ctx = Ctx { rng: &mut r, learn: true };
        let bv = |x: &[usize]| BitVector::from_bits(x, BITS);

        // the one-shot test's events: common episodes, one rare one, more common ones
        let mut events: Vec<Vec<usize>> = Vec::new();
        for i in 0..351 {
            if i == 300 {
                events.push(episode(&[0, 1, 50, 51]));
                continue;
            }
            let mut ws: Vec<usize> = (0..6).collect::<Vec<_>>().choose_multiple(&mut rng, 3).copied().collect();
            ws.push(10 + rng.gen_range(0..6));
            events.push(episode(&ws));
        }
        for (i, e) in events.iter().enumerate() {
            let a = h.store(e);
            let b = theta_store(&mut net, &bv(e), &bv(e), &mut ctx, cfg.ca1_k);
            assert_eq!(a, b, "novelty of event {i}");
            if i % 50 == 0 {
                let cue = &e[..e.len() / 2];
                assert_eq!(active(&theta_recall(&mut net, &bv(cue), &mut ctx)), h.recall(cue).ec, "recall after event {i}");
            }
        }
        let cue = episode(&[0, 1, 50]);
        let ec = active(&theta_recall(&mut net, &bv(&cue), &mut ctx));
        assert_eq!(ec, h.recall(&cue).ec);
        assert!(word(51).iter().filter(|b| ec.contains(b)).count() >= 12, "the one-shot family comes back");
    }
    fn random_events(n: usize, seed: u64, phases: usize) -> Vec<(Vec<(usize, usize)>, Vec<(usize, usize)>)> {
        use rand::seq::SliceRandom;
        let mut rng = StdRng::seed_from_u64(seed);
        let all: Vec<usize> = (0..1024).collect();
        (0..n)
            .map(|_| {
                let mut x: Vec<(usize, usize)> = all.choose_multiple(&mut rng, 32).map(|&b| (b, rng.gen_range(0..phases))).collect();
                let mut y: Vec<(usize, usize)> = all.choose_multiple(&mut rng, 16).map(|&b| (b, rng.gen_range(0..phases))).collect();
                x.sort_unstable();
                y.sort_unstable();
                (x, y)
            })
            .collect()
    }

    /// Centering with a running rate (no division) recalls as well as exact centering.
    #[test]
    fn homeostatic_centering_matches_exact() {
        let data = random_events(1500, 3, 1);
        let mut r = StdRng::seed_from_u64(0);
        let mut ctx = Ctx { rng: &mut r, learn: true };
        let mut recalled = Vec::new();
        for c in [Center::Off, Center::Exact, Center::Homeostatic(16)] {
            let mut a = Associate::new(1024, &[Scale::None], 0, Readout::TopK(16), 100_000, 0, 16);
            a.set_center(c);
            let bits = |v: &[(usize, usize)]| BitVector::from_bits(&v.iter().map(|p| p.0).collect::<Vec<_>>(), 1024);
            for (x, y) in &data {
                a.tick(&[&bits(y), &BitVector::EMPTY, &bits(x)], &mut ctx);
            }
            let mut right = 0;
            for (x, y) in &data {
                a.tick(&[&BitVector::EMPTY, &BitVector::EMPTY, &bits(x)], &mut ctx);
                if y.iter().filter(|p| a.output(0).bit_get(p.0)).count() >= 13 {
                    right += 1;
                }
            }
            recalled.push(right);
        }
        assert!(recalled[1] > recalled[0], "centering helps: {recalled:?}");
        assert!(recalled[2] * 100 >= recalled[1] * 99, "homeostatic ≈ exact: {recalled:?}");
    }

    /// A phasor memory recalls cells and their phases, and phase-coded inputs let it
    /// hold more than plain inputs (crosstalk cancels).
    #[test]
    fn phase_population_recalls_cells_and_phases() {
        let run = |n: usize, phased_in: bool| {
            let data = random_events(n, 4, 16);
            let mut a = PhaseAssociate::new(1024, 16, 16, if phased_in { 16 } else { 1 }, false);
            let mut r = StdRng::seed_from_u64(0);
            let mut ctx = Ctx { rng: &mut r, learn: true };
            let input = |x: &[(usize, usize)]| {
                if phased_in { phase_code(x, 1024, 16) } else { BitVector::from_bits(&x.iter().map(|p| p.0).collect::<Vec<_>>(), 1024) }
            };
            for (x, y) in &data {
                a.tick(&[&phase_code(y, 1024, 16), &input(x)], &mut ctx);
            }
            let mut right = 0;
            for (x, y) in &data {
                a.tick(&[&BitVector::EMPTY, &input(x)], &mut ctx);
                let got = phase_cells(a.output(0), 16);
                if y.iter().filter(|p| got.contains(p)).count() >= 13 {
                    right += 1;
                }
            }
            right
        };
        assert_eq!(run(200, true), 200);
        let (plain, phased) = (run(1500, false), run(1500, true));
        assert!(phased > plain + 300, "phase-coded inputs cancel crosstalk: {phased} vs {plain} of 1500");
    }
}
