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
use crate::program::hippocampus::DentateGyrus;

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
    outs: [BitVector; 2],
}

impl Predictor {
    pub fn new(bits: usize, class: KernelClass<SimpleKernel>) -> Self {
        Self { class, bits, last_input: None, outs: [zeros(bits), zeros(bits)] }
    }
}

impl Module for Predictor {
    fn name(&self) -> String {
        format!("Predictor({} bits, {} kernels)", self.bits, self.class.len())
    }
    fn n_inputs(&self) -> usize {
        2
    }
    fn n_outputs(&self) -> usize {
        2
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
                    self.class.feedback(last, teach, &mut *ctx.rng);
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
    }
    fn output(&self, port: usize) -> &BitVector {
        self.outs.get(port).unwrap_or(&EMPTY)
    }
    fn sleep(&mut self) {
        self.class.sleep();
    }
    fn reset(&mut self) {
        self.last_input = None;
        self.outs = [zeros(self.bits), zeros(self.bits)];
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

/// A base kernel the grammar can place.
#[derive(Clone, Debug, PartialEq)]
pub enum Prim {
    /// A `Predictor` with `bits` outputs, reading inputs of `frames` frames, each `bits`
    /// wide, growing kernels that sample `sample_bits` bits per frame.
    Predict { bits: usize, frames: usize, sample_bits: usize },
    Op(KernelOp),
    Delay,
    Concat(usize),
    Separate { cells: usize, fan_out: usize, k: usize, seed: u64 },
}

impl Prim {
    pub fn build(&self) -> Box<dyn Module> {
        match *self {
            Prim::Predict { bits, frames, sample_bits } => {
                let cfg = GrowthConfig { frame_words: bits.div_ceil(64), max_frames: frames.max(1), sample_bits, ..GrowthConfig::default() };
                Box::new(Predictor::new(bits, KernelClass::predictive(cfg)))
            }
            Prim::Op(op) => Box::new(BitOp::new(op)),
            Prim::Delay => Box::<Delay>::default(),
            Prim::Concat(n) => Box::new(Concat::new(n)),
            Prim::Separate { cells, fan_out, k, seed } => Box::new(Separate::new(cells, fan_out, k, seed)),
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
    /// Push a forward reference: a signal that will be bound later by `Close`. Reading it
    /// gives the bound signal's previous-tick value, if it is later in the network.
    Feedback,
    /// Pop a signal and bind the most recent open `Feedback` to it.
    Close,
    /// Pop a signal and expose it as the next output port.
    Out,
    Nop,
}

/// One module definition: its number of inputs and the code that builds it.
#[derive(Clone, Debug, PartialEq)]
pub struct NetDef {
    pub label: String,
    pub inputs: usize,
    pub code: Vec<NetOp>,
}

/// A library of definitions; later ones may place earlier ones (`Sub`).
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
        let def = &self.defs[k];
        let mut net = Network::new(&def.label, def.inputs);
        let mut stack: Vec<Option<Sig>> = Vec::new();
        let mut bound: Vec<Option<Option<Sig>>> = Vec::new(); // per Feedback: its binding
        let mut open: Vec<usize> = Vec::new();
        let mut pending_wires: Vec<(usize, usize, usize)> = Vec::new(); // (child, port, feedback)
        let mut exports: Vec<Option<Sig>> = Vec::new();

        let mut place = |net: &mut Network, stack: &mut Vec<Option<Sig>>, pending_wires: &mut Vec<(usize, usize, usize)>, m: Box<dyn Module>| {
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
        net
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
}
