use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Directed;

use crate::bitvec::{ones_of, select_nth_bit, BitVector};
use crate::bitvec::bithistory::{AdvanceMode, BitVecHistory};
use crate::kernel::class::KernelOp;
use crate::kernel::group::KernelGroup;
use crate::kernel::KernelClass;
use crate::kernel::SimpleKernel;
use crate::program::graph::{EdgeKind, EdgeSpec, GraphProgram, NodeSpec, ProgramDefaults};
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::Rng;
use std::sync::atomic::{AtomicBool, Ordering};

/// The process's default for new worlds (`World::in_place`): declared channels held in place.
static IN_PLACE: AtomicBool = AtomicBool::new(false);

/// Sets the process's default for worlds made from now on (an experiment's binary sets it once,
/// before any world is made).
pub fn set_in_place(on: bool) {
    IN_PLACE.store(on, Ordering::Relaxed);
}

pub fn in_place() -> bool {
    IN_PLACE.load(Ordering::Relaxed)
}

/// The process's default for new worlds (`World::explore`): a stuck population grows a trial
/// connection (`RuntimeNetwork::explore`).
static EXPLORE: AtomicBool = AtomicBool::new(false);

pub fn set_explore(on: bool) {
    EXPLORE.store(on, Ordering::Relaxed);
}

/// The attractant (`set_attract`): each input port (eye, ear, touch, affect, the self channel)
/// that no population reads counts the steps its lines changed, a `u8`; when it reaches a
/// population's own threshold (2^(its accident gene - 10), 4 to 256 steps) that population, the
/// one whose threshold is lowest (ties in genome order), grows a trial connection from the port
/// into its own destination, as exploration's trials are grown, and the count starts again; a
/// trial whose predictions come true is kept by the genome (`tags::take_notes`, kinds 8 to 12).
/// An unread input that keeps changing asks for a connection, as a growth cone follows a
/// chemical. Off by default.
static ATTRACT: AtomicBool = AtomicBool::new(false);

pub fn set_attract(on: bool) {
    ATTRACT.store(on, Ordering::Relaxed);
}

pub fn attract() -> bool {
    ATTRACT.load(Ordering::Relaxed)
}

/// The attractant on the output side (`set_attract_out`): each word of the act that holds a body
/// code (a move, a token, a key) counts the steps nothing wrote into it, a `u8`; when the count
/// reaches a population's own threshold (as `set_attract`'s, the lowest answering), that
/// population grows a trial population reading its own source and writing that word, whose
/// newborn kernels write the body's codes there (a babble into the unused part of the body); a
/// trial is kept by the genome (`tags::take_notes`, kinds 13 voice, 14 hand) if, in the lifetime,
/// its word of the act held a code the body made (with the muscle's response, what was done). An
/// unused muscle asks for a nerve. Off by default.
static ATTRACT_OUT: AtomicBool = AtomicBool::new(false);

pub fn set_attract_out(on: bool) {
    ATTRACT_OUT.store(on, Ordering::Relaxed);
}

thread_local! {
    /// (tests) `set_attract_out` for this thread only
    static ATTRACT_OUT_HERE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

pub fn attract_out() -> bool {
    ATTRACT_OUT_HERE.with(|c| c.get()).unwrap_or_else(|| ATTRACT_OUT.load(Ordering::Relaxed))
}

/// (tests) `set_attract_out` for this thread only.
pub fn set_attract_out_here(on: Option<bool>) {
    ATTRACT_OUT_HERE.with(|c| c.set(on));
}

pub fn explore() -> bool {
    EXPLORE.load(Ordering::Relaxed)
}

/// Competition between populations for their destination (`set_write_compete`): at each
/// destination, the kernels of the populations writing it compete for its words as kernels of
/// one population compete for their input. Each step the populations into a destination run
/// strongest hold first: with pruning their grip (the input that already drives a muscle fibre
/// most keeps taking it), without it how well their best kernel matched the step before; ties,
/// and wires and noise, keep the genome's order. Each word an earlier population's kernels wrote
/// this step is claimed: a kernel more than its overlap rule allows (half, by default) of whose
/// output words are claimed is held down and learns anti-Hebbian. A word is a slot (a move or a
/// token is a code inside one word), so one population writes each: the act is written by the
/// winners, not the OR of every reflex.
static COMPETE: AtomicBool = AtomicBool::new(false);

pub fn set_write_compete(on: bool) {
    COMPETE.store(on, Ordering::Relaxed);
}

pub fn write_compete() -> bool {
    COMPETE.load(Ordering::Relaxed)
}

/// Pruning (`set_prune`, with competition): each population has a hold on its destination (its
/// gene `grip`) at the start of a lifetime, up one each step one of its kernels writes there, down
/// one each step its kernels only lose to other populations'; at 0 it lets go of its destination
/// for the rest of the lifetime (it runs no more, as a paused population). Every lifetime starts
/// wired broadly and keeps the populations that keep winning; the genome is not changed.
static PRUNE: AtomicBool = AtomicBool::new(false);

/// The contest's counts since last read (`take_contest`), over every network on every thread:
/// steps run with competition, kernels held down by another population's claim, and populations
/// that let go of their destination. Only a record: nothing reads it inside a network.
static CONTEST_STEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CONTEST_LOST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CONTEST_LET_GO: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Learning frozen (`set_frozen`), for measuring what learning buys a lifetime (`ISL_LEARN_BATTERY`
/// in examples/islands.rs): the same network with no mask moves, births, replication, output moves,
/// teachers, accidents or exploration. Never on in a run. Off by default.
static FROZEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_frozen(on: bool) {
    FROZEN.store(on, std::sync::atomic::Ordering::Relaxed);
}

thread_local! {
    /// (`set_frozen_here`) learning frozen for this thread's lifetimes only
    static FROZEN_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Learning frozen for this thread only (`evolve::set_learn_gain`: the frozen twin of a lifetime,
/// scored beside it while other threads score other genomes).
pub fn set_frozen_here(on: bool) {
    FROZEN_HERE.with(|c| c.set(on));
}

thread_local! {
    /// (`set_frozen_pop_here`) one population's learning frozen, by its genome id
    static FROZEN_POP_HERE: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

/// One population's learning frozen, by its genome id (`spec.salt`), on this thread only: what that
/// population's learning buys (`ISL_LEARN_LESION`, a learning lesion). Measurement only.
pub fn set_frozen_pop_here(id: Option<u32>) {
    FROZEN_POP_HERE.with(|c| c.set(id));
}

pub fn frozen() -> bool {
    if FROZEN_HERE.with(|c| c.get()) {
        return true;
    }
    FROZEN.load(std::sync::atomic::Ordering::Relaxed)
}

/// The clock drawn (`set_clock_drawn`): the self channel's step count starts each lifetime at a
/// number drawn from the lifetime's seed, so it still counts steps but no longer says when in the
/// lifetime a step is (a reversal at trial 100, a block's end), which a network could time from it.
/// Off by default.
static CLOCK_DRAWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_clock_drawn(on: bool) {
    CLOCK_DRAWN.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn clock_drawn() -> bool {
    CLOCK_DRAWN.load(std::sync::atomic::Ordering::Relaxed)
}

/// (steps, kernels held down by another population, populations that let go) since last read.
pub fn take_contest() -> (u64, u64, u64) {
    (CONTEST_STEPS.swap(0, Ordering::Relaxed), CONTEST_LOST.swap(0, Ordering::Relaxed), CONTEST_LET_GO.swap(0, Ordering::Relaxed))
}

pub fn set_prune(on: bool) {
    PRUNE.store(on, Ordering::Relaxed);
}

pub fn prune() -> bool {
    PRUNE.load(Ordering::Relaxed)
}

/// The explorer's kind (`set_shadow`, with exploration): a stuck (or accident-prone) population
/// grows a shadow predictor instead of a trial XOR loop. The shadow reads what the population
/// reads and predicts, a step ahead, one vector after another (a sense, the self channel, an
/// output port, a channel), writing nothing into the network; what it predicts well is noted
/// for the genome (`notes`).
static SHADOW: AtomicBool = AtomicBool::new(false);

pub fn set_shadow(on: bool) {
    SHADOW.store(on, Ordering::Relaxed);
}

pub fn shadow() -> bool {
    SHADOW.load(Ordering::Relaxed)
}

/// Steps a shadow predicts one target before it moves to the next, unless its predictions came
/// true at least `KEEP_SHADOW` more often than they failed.
pub const RETARGET: u32 = 128;

/// Where a shadow's kernel stands against a hint band (`EdgeSpec::hint_low`): 1 when it is partly
/// right (more than `low` quarters of its `lines` came on, fewer than `high`), -1 when at or under
/// `low`, 0 when fully right (or it wrote nothing).
/// A moving channel's next frame (`NodeSpec::conveyor`, `NodeSpec::wave`), from its last: a
/// conveyor's words one place along (word i gets word i-1, word 0 empty) when the step count is a
/// multiple of 2^(conveyor-1), else as they were; then, with a wave, each line the XOR of its two
/// neighbours (cellular rule 90, across the whole vector; lines past the ends read as off).
pub fn moved(prev: &[u64], conveyor: u8, wave: bool, steps: u64) -> Vec<u64> {
    let mut v = prev.to_vec();
    if conveyor > 0 && steps % (1u64 << (conveyor - 1).min(30)) == 0 {
        v.rotate_right(1);
        if let Some(w) = v.first_mut() {
            *w = 0;
        }
    }
    if wave {
        let old = v.clone();
        let k = old.len();
        for i in 0..k {
            let lo = if i > 0 { old[i - 1] >> 63 } else { 0 };
            let hi = if i + 1 < k { old[i + 1] << 63 } else { 0 };
            v[i] = ((old[i] << 1) | lo) ^ ((old[i] >> 1) | hi);
        }
    }
    v
}

pub fn in_band(lines: u32, residue: u32, low: u8, high: u8) -> i32 {
    if lines == 0 {
        return 0;
    }
    let agree = 4 * (lines - residue);
    if agree <= low as u32 * lines {
        -1
    } else if agree < high.clamp(1, 4) as u32 * lines {
        1
    } else {
        0
    }
}
pub const KEEP_SHADOW: i32 = 16;

/// Steps a population's kernels must all go without firing before it is stuck and grows a trial
/// connection (the starting value; kernels count their own age, saturating at 255).
pub const STUCK: u8 = 64;

/// The process's default for new worlds (`World::count_reads`): the words a kernel population
/// reads count as its work.
static COUNT_READS: AtomicBool = AtomicBool::new(false);

pub fn set_count_reads(on: bool) {
    COUNT_READS.store(on, Ordering::Relaxed);
}

pub fn count_reads() -> bool {
    COUNT_READS.load(Ordering::Relaxed)
}

/// A runnable network: node histories plus per-edge kernel groups.
/// For now, each edge also carries a simple default kernel used for execution
/// until KernelClass execution is wired in.
pub struct RuntimeNetwork {
    pub nodes: Vec<BitVecHistory>,
    pub edges: Vec<RuntimeEdgeGroup>,
    /// each node's own random stream (its spontaneous activity)
    pub node_rngs: Vec<StdRng>,
    /// the act node
    pub output: usize,
    /// spontaneous lines per step, per node
    pub noise: Vec<usize>,
    /// per node: it holds its lines from step to step instead of starting empty
    pub hold: Vec<bool>,
    /// (motions) each node's conveyor stride and wave (`NodeSpec::conveyor`, `NodeSpec::wave`)
    pub motion: Vec<(u8, bool)>,
    /// per node: held in place (a declared channel, when built so): one frame, never cleared or
    /// copied, read as it is now (writes earlier in this step included) whatever the lag: no
    /// history
    pub live: Vec<bool>,
    /// the words a kernel population reads count as work (so thrift sees a wide read)
    pub count_reads: bool,
    /// exploration: a population whose kernels have all gone `STUCK` steps without firing grows,
    /// once a lifetime, a trial connection: a predicting (XOR) population reading its destination
    /// and writing back into its source (a loop), or into its destination itself where the
    /// source is a port. The trial's kernels are born and replicate by the usual rules, and each
    /// of their predictions that comes true is counted (`confirms`), for the genome to keep it
    pub explore: bool,
    /// (`set_attract`) each input port's attractant: steps its lines changed while unread (eye,
    /// ear, touch, affect, the self channel)
    pub attracted: [u8; 5],
    /// (`set_attract_out`) each output word's attractant: steps nothing wrote into it
    pub attracted_out: Vec<u8>,
    /// the body's codes in the act (a word and its lines), for what writes the act to be born
    /// with (set by the genome's build)
    pub motor: Vec<(usize, u64)>,
    /// (with `explore`) explorers are shadow predictors (`set_shadow`)
    pub shadow: bool,
    /// (with `shadow`) shadows start only by accident, at each population's own rate, not when
    /// it is stuck (`tags::set_shadow_hints`)
    pub shadow_random: bool,
    /// work done since it was built: kernels matched plus words a wire or noise edge wrote (a
    /// paused or idle population adds nothing)
    pub work: u64,
    /// the self node (named "self"), if the program has one: what the network senses of itself
    pub self_node: Option<usize>,
    /// last step's work and how many edges ran (for the self node), and the steps so far
    last_work: u64,
    last_ran: u64,
    steps: u64,
    /// (`set_clock_drawn`) where this lifetime's clock (the self channel's word 2) starts
    clock_from: u64,
}

/// The frame a read `back` steps reaches: a node held in place has no history, and is read as
/// it is now whatever the lag (a network that wants the past copies it out and keeps it).
fn at(live: bool, back: usize) -> usize {
    if live { 0 } else { back }
}

/// What an edge reads: frames `read_back .. read_back + span` of a node ORed together, only the
/// words of its window (the offset wraps; at least one word). Costs the window, not the vector.
fn view(node: &BitVecHistory, live: bool, read_back: usize, span: usize, window: Option<(usize, usize)>) -> BitVector {
    let first = node.get_frame(at(live, read_back));
    let n = first.word_len().max(1);
    let (off, len) = match window {
        Some((o, k)) => { let o = o % n; (o, k.max(1).min(n - o)) }
        None => (0, n),
    };
    let mut w: Vec<u64> = first.as_words()[off..off + len].to_vec();
    for f in 1..span.max(1) {
        for (a, b) in w.iter_mut().zip(&node.get_frame(at(live, read_back + f)).as_words()[off..off + len]) {
            *a |= b;
        }
    }
    BitVector::from_words(w)
}

/// One node to read and another to write, borrowed together.
fn pair(nodes: &mut [BitVecHistory], a: usize, b: usize) -> (&BitVecHistory, &mut BitVecHistory) {
    assert_ne!(a, b);
    if a < b {
        let (l, r) = nodes.split_at_mut(b);
        (&l[a], &mut r[0])
    } else {
        let (l, r) = nodes.split_at_mut(a);
        (&r[0], &mut l[b])
    }
}

/// What a kernel edge read this step, for its learning after the tick: its own copy, or a frame
/// of a node, read again where it lies (the same lines: frames behind the current one are not
/// written during a step; a node held in place is read as the step left it).
enum Seen {
    Owned(BitVector),
    Frame(usize, usize),
}

impl Seen {
    fn of(owned: Option<BitVector>, node: usize, frame: usize) -> Self {
        match owned {
            Some(v) => Seen::Owned(v),
            None => Seen::Frame(node, frame),
        }
    }

    fn get<'a>(&'a self, nodes: &'a [BitVecHistory]) -> &'a BitVector {
        match self {
            Seen::Owned(v) => v,
            Seen::Frame(n, f) => nodes[*n].get_frame(*f),
        }
    }
}

/// The word with the most lines on, and how many.
fn busiest_word(words: &[u64]) -> (usize, u32) {
    words.iter().enumerate().map(|(i, w)| (i, w.count_ones())).max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))).unwrap_or((0, 0))
}

fn apply(d: &mut [u64], t: &[u64], op: KernelOp) {
    for (a, b) in d.iter_mut().zip(t) {
        *a = op.word(*a, *b);
    }
}

/// What an edge's kernel factory is told about the edge it fills.
pub struct EdgeInfo<'a> {
    pub index: usize,
    pub src: usize,
    pub dst: usize,
    pub src_bits: usize,
    pub dst_bits: usize,
    /// the destination is the output node
    pub to_output: bool,
    pub spec: &'a EdgeSpec,
    /// the lifetime's seed (so starting cells can differ from lifetime to lifetime)
    pub seed: u64,
}

/// Per-edge runtime unit: group of kernel classes plus a default kernel.
pub struct RuntimeEdgeTarget {
    pub idx: usize, // target node index
    pub read_back: usize, // how many frames back to read from
    pub span: usize,      // frames read (ORed), from read_back back
}
pub struct RuntimeEdgeGroup {
    pub src: RuntimeEdgeTarget,
    pub dst: usize,
    pub group: KernelGroup,
    pub kind: EdgeKind,
    pub op: KernelOp,
    pub rotate: u16,
    pub grow: bool,
    /// kernels hold each other down where their masks overlap
    pub inhibit: bool,
    /// it learns only while this node had lines on (true) or none (false) a frame ago
    pub gate: Option<(usize, bool)>,
    /// births into an empty destination write a fresh random code
    pub new_code: bool,
    /// births sample synapses across the whole source
    pub wide: bool,
    /// read only these source words
    pub window: Option<(usize, usize)>,
    /// a wire lands its copy from this destination word
    pub dst_word: usize,
    /// births write what the destination held a frame ago
    pub repeat: bool,
    pub repeat_lag: usize,
    /// all of this edge's own parameters (births, rates, its gate's lag, ...)
    pub spec: EdgeSpec,
    /// this edge's own random stream
    pub rng: StdRng,
    /// births are allowed (on edges into the act node only when they repeat acts or write into
    /// a port of it, where they copy the move being made)
    pub can_grow: bool,
    /// it writes the act node
    pub into_act: bool,
    /// its kernels' predictions that came true (all lines they wrote cancelled), this lifetime
    pub confirms: u32,
    /// a trial connection grown by exploration (0: a genome's own; 1: back into the stuck
    /// population's source; 2: on its destination), its `spec.salt` the stuck population's
    pub explorer: u8,
    /// (`set_attract_out`) an output trial: the word of the act it writes
    pub out_word: Option<(usize, u64)>,
    /// it has grown its trial connection this lifetime
    pub spawned: bool,
    /// a shadow: it fires into `scratch`, not its destination (it writes nothing into the network)
    pub silent: bool,
    pub scratch: BitVector,
    /// a predictor's kernels that fired last step, and what they read then
    pub teach_fired: Vec<usize>,
    pub teach_src: Option<BitVector>,
    /// a shadow's targets in turn (node, words, kind), the one it is on, steps on it, its score
    /// there (predictions that came true less those that failed), and each target's best score
    pub targets: Vec<(usize, Option<(usize, usize)>, u8)>,
    pub target_at: usize,
    pub on_target: u32,
    pub score: i32,
    pub scores: Vec<(u8, u32)>,
    /// (competition) its best kernel's share last step, which orders it among the populations
    /// into its destination; (pruning) its hold on the destination (None: not started, taken
    /// from `spec.grip`) and whether it has let go
    pub share: usize,
    pub grip: Option<u8>,
    pub detached: bool,
    /// (the three-factor rule, `EdgeSpec::elig`) its kernels that died this step, removed before
    /// the next
    pub dying: Vec<usize>,
}

impl RuntimeEdgeGroup {
    /// Kernels on this edge.
    pub fn group_len(&self) -> usize {
        self.group.kernel_count()
    }
}

impl RuntimeNetwork {
    /// Build a runnable network from a GraphProgram.
    pub fn from_program(prog: &GraphProgram, defaults: &ProgramDefaults) -> Self {
        let g = prog.build_graph(defaults);
        Self::from_graph(&g)
    }

    /// Build a runnable network from a Graph<NodeSpec, EdgeSpec>.
    pub fn from_graph(g: &Graph<NodeSpec, EdgeSpec, Directed>) -> Self {
        // Materialize node histories
        let mut nodes: Vec<BitVecHistory> = Vec::with_capacity(g.node_count());
        for ni in g.node_indices() {
            let ns = g.node_weight(ni).expect("node weight");
            let bits = ns.bits;
            // one more frame than the deepest read, so a read never lands on the frame being
            // written
            let ring_len = ns.history_depth.max(1) + 1;
            let init = BitVector::new(bits, Some(0));
            nodes.push(BitVecHistory::new(init, ring_len));
        }

        // Compile edges to runtime edge groups
        let mut edges: Vec<RuntimeEdgeGroup> = Vec::with_capacity(g.edge_count());
        for e in g.edge_references() {
            let src = e.source().index();
            let dst = e.target().index();
            let es = e.weight();

            //let src_bits = g.node_weight(NodeIndex::new(src)).unwrap();
            //let dst_bits = g.node_weight(NodeIndex::new(dst)).unwrap();

            //let input_mask = ones_mask(src_bits);
            //let output_mask = ones_mask(dst_bits);

            edges.push(RuntimeEdgeGroup {
                src: RuntimeEdgeTarget {
                    idx: src,
                    read_back: es.input_read_back,
                    span: es.span.max(1),
                },
                dst,
                group: KernelGroup::default(), // placeholder; later: pick group per edge
                kind: es.kind,
                op: es.op,
                rotate: es.rotate,
                grow: es.grow,
                inhibit: es.inhibit,
                gate: es.gate,
                new_code: es.new_code,
                wide: es.wide,
                window: es.window,
                dst_word: es.dst_word,
                repeat: es.repeat,
                repeat_lag: es.repeat_lag,
                spec: es.clone(),
                rng: StdRng::seed_from_u64((es.salt as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                can_grow: true,
                into_act: false,
                confirms: 0,
                explorer: 0,
                spawned: false,
                silent: false,
                scratch: BitVector::new(64, Some(0)),
                share: 0, grip: None, detached: false, dying: Vec::new(),
                teach_fired: Vec::new(),
                teach_src: None,
                targets: Vec::new(),
                target_at: 0,
                on_target: 0,
                score: 0,
                scores: Vec::new(),
                out_word: None,
            });
        }

        let output = g.node_indices().find(|n| g.node_weight(*n).unwrap().name.as_deref() == Some("output")).map_or(g.node_count() - 1, |n| n.index());
        let noise = g.node_indices().map(|n| g.node_weight(n).unwrap().noise).collect();
        let node_rngs = (0..g.node_count()).map(|i| StdRng::seed_from_u64(i as u64 ^ 0xA0A0)).collect();
        let hold = g.node_indices().map(|n| g.node_weight(n).unwrap().hold).collect();
        let motion = g.node_indices().map(|n| { let w = g.node_weight(n).unwrap(); (w.conveyor, w.wave) }).collect();
        let self_node = g.node_indices().find(|n| g.node_weight(*n).unwrap().name.as_deref() == Some("self")).map(|n| n.index());
        let live = vec![false; g.node_count()];
        Self { nodes, edges, node_rngs, output, noise, hold, motion, live, count_reads: false, explore: false, attracted: [0; 5], attracted_out: Vec::new(), motor: Vec::new(), shadow: false, shadow_random: false, work: 0, self_node, last_work: 0, last_ran: 0, steps: 0, clock_from: 0 }
    }

    /// Build from a graph with the input and output nodes resized (to a world's senses and
    /// acts) and each edge's kernels made by `factory`; the network's random stream is seeded
    /// with `seed`.
    pub fn from_graph_with<F>(g: &Graph<NodeSpec, EdgeSpec, Directed>, in_bits: usize, out_bits: usize, seed: u64, factory: F) -> Self
    where
        F: FnMut(&EdgeInfo) -> KernelGroup,
    {
        Self::from_graph_held(g, in_bits, out_bits, seed, false, factory)
    }

    /// `from_graph_with`, and with `in_place` every node but the senses (the first), the act and
    /// the self node is held in place (`live`): one frame, no history (half the memory or less,
    /// and no clearing or copying each step).
    pub fn from_graph_held<F>(g: &Graph<NodeSpec, EdgeSpec, Directed>, in_bits: usize, out_bits: usize, seed: u64, in_place: bool, mut factory: F) -> Self
    where
        F: FnMut(&EdgeInfo) -> KernelGroup,
    {
        // the act node: the one named "output" (else the last)
        let last = g.node_indices().find(|n| g.node_weight(*n).unwrap().name.as_deref() == Some("output")).map_or(g.node_count() - 1, |n| n.index());
        let self_node = g.node_indices().find(|n| g.node_weight(*n).unwrap().name.as_deref() == Some("self")).map(|n| n.index());
        let live: Vec<bool> = (0..g.node_count()).map(|i| in_place && i != 0 && i != last && Some(i) != self_node).collect();
        let bits_of = |i: usize| if i == 0 { in_bits } else if i == last { out_bits } else { g.node_weight(NodeIndex::new(i)).unwrap().bits };
        let mut nodes: Vec<BitVecHistory> = Vec::with_capacity(g.node_count());
        for ni in g.node_indices() {
            let ns = g.node_weight(ni).expect("node weight");
            let d = ns.history_depth.max(1);
            let ring_len = if live[ni.index()] { 1 } else { d + 1 };
            nodes.push(BitVecHistory::new(BitVector::new(bits_of(ni.index()), Some(0)), ring_len));
        }
        let mut edges: Vec<RuntimeEdgeGroup> = Vec::with_capacity(g.edge_count());
        for (index, e) in g.edge_references().enumerate() {
            let (src, dst) = (e.source().index(), e.target().index());
            let info = EdgeInfo { index, src, dst, src_bits: bits_of(src), dst_bits: bits_of(dst), to_output: dst == last, spec: e.weight(), seed };
            edges.push(RuntimeEdgeGroup {
                src: RuntimeEdgeTarget { idx: src, read_back: e.weight().input_read_back.max(1), span: e.weight().span.max(1) },
                dst,
                group: factory(&info),
                kind: e.weight().kind,
                op: e.weight().op,
                rotate: e.weight().rotate,
                grow: e.weight().grow,
                inhibit: e.weight().inhibit,
                gate: e.weight().gate,
                new_code: e.weight().new_code,
                wide: e.weight().wide,
                window: e.weight().window,
                dst_word: e.weight().dst_word,
                repeat: e.weight().repeat,
                repeat_lag: e.weight().repeat_lag.max(1),
                spec: e.weight().clone(),
                rng: StdRng::seed_from_u64(seed ^ (e.weight().salt as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                can_grow: dst != last || e.weight().repeat || e.weight().dst_len > 0,
                into_act: dst == last,
                confirms: 0,
                explorer: 0,
                spawned: false,
                silent: false,
                scratch: BitVector::new(64, Some(0)),
                share: 0, grip: None, detached: false, dying: Vec::new(),
                teach_fired: Vec::new(),
                teach_src: None,
                targets: Vec::new(),
                target_at: 0,
                on_target: 0,
                score: 0,
                scores: Vec::new(),
                out_word: None,
            });
        }
        let noise = g.node_indices().map(|n| g.node_weight(n).unwrap().noise).collect();
        let node_rngs = (0..g.node_count()).map(|i| StdRng::seed_from_u64(seed ^ (i as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03))).collect();
        let hold = g.node_indices().map(|n| g.node_weight(n).unwrap().hold).collect();
        let motion = g.node_indices().map(|n| { let w = g.node_weight(n).unwrap(); (w.conveyor, w.wave) }).collect();
        Self { nodes, edges, node_rngs, output: last, noise, hold, motion, live, count_reads: false, explore: false, attracted: [0; 5], attracted_out: Vec::new(), motor: Vec::new(), shadow: false, shadow_random: false, work: 0, self_node, last_work: 0, last_ran: 0, steps: 0, clock_from: if clock_drawn() { (seed ^ 0xC10C_C10C_C10C_C10C).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29) ^ seed.wrapping_mul(0xBF58_476D_1CE4_E5B9) } else { 0 } }
    }

    /// Advance histories and process all edges once.
    pub fn tick(&mut self, mode: AdvanceMode, adj_temperature: i16) {
        // (`set_attract_out`) an output trial is confirmed when its word of the act holds a code
        // (with the muscle's response, one the body made last step)
        if attract_out() {
            let out = self.nodes[self.output].current().as_words().to_vec();
            for e in self.edges.iter_mut() {
                if let Some((w, _)) = e.out_word {
                    if out.get(w).is_some_and(|&x| x != 0) {
                        e.confirms = e.confirms.saturating_add(1);
                    }
                }
            }
        }
        // (`set_frozen`, measurement only) no learning in the lifetime: kernels keep their masks and
        // outputs, and nothing is born, copied or explored
        let one = FROZEN_POP_HERE.with(|c| c.get());
        if frozen() || one.is_some() {
            if frozen() {
                self.explore = false;
                self.shadow = false;
            }
            for eg in self.edges.iter_mut().filter(|eg| frozen() || Some(eg.spec.salt) == one) {
                eg.spec.grow = false;
                eg.grow = false;
                eg.spec.replicate = 0;
                eg.spec.elig = 0;
                eg.spec.teacher = None;
                eg.spec.accident = 0;
                for class in eg.group.classes_mut() {
                    for k in class.kernels_mut() {
                        k.learn_fire = false;
                        k.learn_held = false;
                    }
                }
            }
        }
        // Advance all nodes first so writes go into a fresh current frame
        // (a node that holds starts from its last frame: only clearing writes take lines away; one
        // held in place stays where it is, its ring, if it has one, keeping the frames before)
        for ((n, &hold), &live) in self.nodes.iter_mut().zip(&self.hold).zip(&self.live) {
            if live {
                if n.frames() > 1 {
                    n.advance(AdvanceMode::CopyPrev);
                }
                continue;
            }
            n.advance(if hold && matches!(mode, AdvanceMode::Clear) { AdvanceMode::CopyPrev } else { mode });
        }

        // (motions) a conveyor's words move one place along (every 2^(stride-1) steps; else it keeps
        // what it held), and a wave's lines each become the XOR of their neighbours', from the
        // frame before, ahead of any write
        let steps = self.steps;
        for (n, &(conveyor, wave)) in self.nodes.iter_mut().zip(&self.motion) {
            if (conveyor == 0 && !wave) || n.frames() < 2 {
                continue;
            }
            let v = moved(n.get_frame(1).as_words(), conveyor, wave, steps);
            n.current_mut().as_words_mut().copy_from_slice(&v);
        }

        // Execute edges in their order (the genome's): writes into a vector land in that order,
        // whatever their ops. Each edge reads its source as it was `read_back` (at least 1)
        // frames ago, never the frame being written, so edges into different vectors do not
        // depend on each other.
        self.write_self();
        let Self { nodes, edges, node_rngs, noise, live, count_reads, .. } = self;
        // spontaneous activity: random lines on, in the nodes that have it (each its own stream)
        for (i, &k) in noise.iter().enumerate() {
            let bits = nodes[i].current().bit_len();
            for _ in 0..k {
                let b = node_rngs[i].gen_range(0..bits.max(1));
                nodes[i].current_mut().bit_set(b);
            }
        }
        // (the three-factor rule) the kernels that died last step go before any edge runs; (a
        // rule table) its kernels take their firing and held-down moves from it
        for eg in edges.iter_mut() {
            // (a trigger) its kernels match the way its gene says
            if eg.spec.trigger != 0 {
                let (t, lane) = (eg.spec.trigger, eg.spec.lane);
                if let Some(class) = eg.group.classes_mut().first_mut() {
                    for k in class.kernels_mut() {
                        k.trigger = t;
                        k.lane = lane;
                    }
                }
            }
            if eg.spec.rule != 0 {
                let (f, h) = (crate::program::graph::rule_entry(eg.spec.rule, 0), crate::program::graph::rule_entry(eg.spec.rule, 2));
                if let Some(class) = eg.group.classes_mut().first_mut() {
                    for k in class.kernels_mut() {
                        if f == 0 {
                            k.learn_fire = false;
                        }
                        k.fire_anti = f == 2;
                        if h == 0 {
                            k.learn_held = false;
                        }
                        k.held_hebb = h == 2;
                    }
                }
            }
            if !eg.dying.is_empty() {
                let dead = std::mem::take(&mut eg.dying);
                if let Some(class) = eg.group.classes_mut().first_mut() {
                    class.remove_kernels(&dead);
                }
            }
        }
        let mut fired: Vec<Vec<usize>> = vec![Vec::new(); edges.len()];
        let mut seen: Vec<Option<Seen>> = (0..edges.len()).map(|_| None).collect();
        let mut paused: Vec<bool> = vec![false; edges.len()];
        let mut work = 0u64;
        // (competition) the order edges run in: the genome's, but the kernel populations into one
        // destination take that destination's kernel slots strongest hold first (with pruning: its
        // grip, as the input that already drives a muscle fibre most takes it; else its best match
        // last step), ties in the genome's order; and each destination's words claimed so far
        let compete = write_compete();
        let mut run: Vec<usize> = (0..edges.len()).collect();
        let mut taken: Vec<Vec<u64>> = Vec::new();
        if compete {
            let mut by_dst: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
            for (i, eg) in edges.iter().enumerate() {
                if eg.kind == EdgeKind::Kernels && !eg.silent {
                    by_dst.entry(eg.dst).or_default().push(i);
                }
            }
            for slots in by_dst.values() {
                let mut who = slots.clone();
                let hold = |e: &RuntimeEdgeGroup| if e.spec.grip > 0 { e.grip.unwrap_or(e.spec.grip) as usize } else { e.share };
                who.sort_by(|&a, &b| hold(&edges[b]).cmp(&hold(&edges[a])).then(a.cmp(&b)));
                for (k, &slot) in slots.iter().enumerate() {
                    run[slot] = who[k];
                }
            }
            taken = nodes.iter().map(|n| vec![0u64; n.current().word_len()]).collect();
            CONTEST_STEPS.fetch_add(1, Ordering::Relaxed);
        }
        {
            for &i in &run {
                let eg = &mut edges[i];
                // (pruning) a population that let go of its destination runs no more this lifetime
                if eg.detached {
                    paused[i] = true;
                    continue;
                }
                // an execution gate shut: the edge does nothing this step
                if let Some((g, on)) = eg.spec.pause {
                    let f = nodes[g].get_frame(at(live[g], 1)).as_words();
                    let f = eg.spec.pause_window.map_or(f, |(o, n)| &f[o.min(f.len())..(o + n).min(f.len())]);
                    if f.iter().any(|&w| w != 0) != on {
                        paused[i] = true;
                        continue;
                    }
                }
                // a heat input: how full its vector was a frame ago moves every threshold
                if let Some((hn, hw, warm)) = eg.spec.heat {
                    let f = nodes[hn].get_frame(at(live[hn], 1)).as_words();
                    let f = hw.map_or(f, |(o, n)| &f[o.min(f.len())..(o + n).min(f.len())]);
                    let on: u64 = f.iter().map(|w| w.count_ones() as u64).sum();
                    let h = (on * 256 / (f.len() as u64 * 64).max(1)) as i16;
                    eg.group.set_heat(if warm { h } else { -h });
                }
                eg.group.set_inhibit_quarters(eg.spec.inhibit_quarters);
                // an inhibit input: the lines its vector had on a frame ago are taken first
                if let Some((vn, vw)) = eg.spec.inhibitor {
                    let f = nodes[vn].get_frame(at(live[vn], 1)).as_words();
                    let f = vw.map_or(f, |(o, n)| &f[o.min(f.len())..(o + n).min(f.len())]);
                    eg.group.set_veto(f);
                }
                // what it reads: a whole vector one frame is borrowed where it lies (no copy); a
                // window, a span of frames, a rotation or its own destination is copied
                let frame = at(live[eg.src.idx], eg.src.read_back);
                let borrow = eg.src.span <= 1 && eg.window.is_none() && eg.rotate == 0 && eg.src.idx != eg.dst;
                let owned = (!borrow).then(|| {
                    let mut v = view(&nodes[eg.src.idx], live[eg.src.idx], eg.src.read_back, eg.src.span, eg.window);
                    if eg.kind == EdgeKind::Wire && eg.rotate > 0 {
                        let n = v.bit_len();
                        v.rotr_mut(eg.rotate as usize % n.max(1));
                    }
                    v
                });
                let (src_view, dst_curr): (&BitVector, &mut BitVector) = match &owned {
                    Some(v) => (v, nodes[eg.dst].current_mut()),
                    None => {
                        let (s, d) = pair(nodes, eg.src.idx, eg.dst);
                        (s.get_frame(frame), d.current_mut())
                    }
                };
                match eg.kind {
                    EdgeKind::Noise => {
                        // spontaneous lines in the destination's span, each touching one word
                        let dw = dst_curr.word_len().max(1);
                        let from = eg.spec.dst_word % dw;
                        let n = if eg.spec.out_words == 0 { dw - from } else { (eg.spec.out_words as usize).min(dw - from) };
                        work += eg.spec.noise as u64;
                        for _ in 0..eg.spec.noise {
                            let b = from * 64 + eg.rng.gen_range(0..n * 64);
                            let d = &mut dst_curr.as_words_mut()[b / 64];
                            *d = eg.op.word(*d, 1u64 << (b % 64));
                        }
                    }
                    EdgeKind::Wire => {
                        // the copy lands from `dst_word`, wrapping round the destination; the op
                        // touches only the words it lands on (a wider source folds in by OR)
                        let dw = dst_curr.word_len().max(1);
                        let src = src_view.as_words();
                        work += src.len() as u64;
                        // into a port (part of the node), it wraps inside the port
                        let (base, len) = if eg.spec.dst_len > 0 { (eg.dst_word % dw, eg.spec.dst_len.min(dw - eg.dst_word % dw)) } else { (0, dw) };
                        let land = |j: usize| if eg.spec.dst_len > 0 { base + j % len } else { (j + eg.dst_word) % dw };
                        if let Some((left, n)) = eg.op.long() {
                            // (a long shift) the whole source shifted as one bit string, landing
                            // as a copy does, wrapping round its destination, ORed in
                            let (s, off) = crate::kernel::class::long_shift(src, left, n);
                            let span = if eg.spec.dst_len > 0 { len } else { dw } as isize;
                            let d = dst_curr.as_words_mut();
                            for (j, w) in s.into_iter().enumerate() {
                                if w != 0 {
                                    d[land((j as isize + off).rem_euclid(span) as usize)] |= w;
                                }
                            }
                        } else if src.len() <= len {
                            let d = dst_curr.as_words_mut();
                            for (j, w) in src.iter().enumerate() {
                                let k = land(j);
                                d[k] = eg.op.word(d[k], *w);
                            }
                        } else {
                            let mut t = vec![0u64; dw];
                            let mut touched = vec![false; dw];
                            for (j, w) in src.iter().enumerate() {
                                t[land(j)] |= w;
                                touched[land(j)] = true;
                            }
                            let d = dst_curr.as_words_mut();
                            for k in 0..dw {
                                if touched[k] {
                                    d[k] = eg.op.word(d[k], t[k]);
                                }
                            }
                        }
                    }
                    EdgeKind::Kernels => {
                        // what it reads is all off (a population upstream cleared it): none of
                        // its kernels can fire or learn, so it is paused, at no cost (exact,
                        // unless a kernel acts on nothing)
                        let read = src_view.word_len() as u64;
                        if *count_reads {
                            work += read;
                        }
                        if src_view.as_words().iter().all(|&w| w == 0) && !eg.group.acts_on_nothing() {
                            seen[i] = Some(Seen::of(owned, eg.src.idx, frame));
                            continue;
                        }
                        work += eg.group.kernel_count() as u64;
                        if eg.silent {
                            // a shadow fires into its own scratch vector: nothing reaches the network
                            let mut scratch = std::mem::replace(&mut eg.scratch, BitVector::new(64, Some(0)));
                            scratch.as_words_mut().fill(0);
                            fired[i] = eg.group.process(src_view, &mut scratch, 0, adj_temperature, eg.inhibit, &mut eg.rng).into_iter().flatten().collect();
                            eg.scratch = scratch;
                        } else {
                            // (competition) it competes with the populations before it for its
                            // destination's words
                            if compete {
                                eg.group.set_taken(&taken[eg.dst]);
                            }
                            let by_class = eg.group.process(src_view, dst_curr, 0, adj_temperature, eg.inhibit, &mut eg.rng);
                            if compete {
                                eg.group.mark_outputs(&by_class, &mut taken[eg.dst]);
                                eg.share = eg.group.best_share();
                                let lost = eg.group.take_lost();
                                CONTEST_LOST.fetch_add(lost as u64, Ordering::Relaxed);
                                let wrote = by_class.iter().any(|c| !c.is_empty());
                                // (pruning) its hold: up when it wrote, down when it only lost
                                if eg.spec.grip > 0 {
                                    let g = eg.grip.get_or_insert(eg.spec.grip);
                                    if wrote {
                                        *g = g.saturating_add(1).min(eg.spec.grip.saturating_mul(2));
                                    } else if lost > 0 {
                                        *g -= 1;
                                        if *g == 0 {
                                            eg.detached = true;
                                            CONTEST_LET_GO.fetch_add(1, Ordering::Relaxed);
                                        }
                                    }
                                }
                            }
                            // (a branch) the kernels that fired write their outputs into a second
                            // vector too, at the same place (wrapping), by OR; with competition only
                            // into words no other population claimed this step, and they claim them
                            if let Some((bn, bw)) = eg.spec.branch {
                                let words = nodes[bn].current_mut().as_words_mut();
                                let (base, len) = bw.unwrap_or((0, words.len()));
                                let claim = if compete { taken.get_mut(bn).map(|t| t.as_mut_slice()) } else { None };
                                work += eg.group.write_branch(&by_class, words, base, len, claim) as u64;
                            }
                            // (more branches) each its own trigger, op and rotation: while its gate
                            // holds, what the fired kernels write, rotated, goes into its vector
                            // with its op
                            for &(bn, bw, op, rot, gate, gw) in &eg.spec.more {
                                if let Some((g, on)) = gate {
                                    let f = nodes[g].get_frame(at(live[g], 1)).as_words();
                                    let f = gw.map_or(f, |(o, n)| &f[o.min(f.len())..(o + n).min(f.len())]);
                                    if f.iter().any(|&w| w != 0) != on {
                                        continue;
                                    }
                                }
                                let n = nodes[bn].current().as_words().len();
                                let (base, len) = bw.unwrap_or((0, n));
                                let mut buf = vec![0u64; n];
                                work += eg.group.write_branch(&by_class, &mut buf, base, len, None) as u64;
                                let words = nodes[bn].current_mut().as_words_mut();
                                for (k, &m) in buf.iter().enumerate() {
                                    if m != 0 {
                                        words[k] = op.word(words[k], m.rotate_left(rot as u32 & 63));
                                    }
                                }
                            }
                            fired[i] = by_class.into_iter().flatten().collect();
                        }
                        seen[i] = Some(Seen::of(owned, eg.src.idx, frame));
                    }
                }
            }
        }
        self.work += work;
        self.last_work = work;
        self.last_ran = paused.iter().zip(&seen).filter(|(p, _)| !**p).count() as u64;
        self.steps += 1;
        let Self { nodes, edges, live, motor, .. } = self;
        // After the tick: cancelling kernels learn their outputs, and growing edges give birth
        // where their destination was left uncovered (a paused edge does not learn)
        for (i, eg) in edges.iter_mut().enumerate() {
            if paused[i] {
                continue;
            }
            let sp = eg.spec.clone();
            let rng = &mut eg.rng;
            if eg.kind != EdgeKind::Kernels {
                continue;
            }
            // kernels count the steps since they fired
            if let Some(class) = eg.group.classes_mut().first_mut() {
                for k in class.kernels_mut() {
                    if !fired[i].is_empty() || k.age < u8::MAX {
                        k.age = k.age.saturating_add(1);
                    }
                }
                for &k in &fired[i] {
                    class.kernels_mut()[k].age = 0;
                }
            }
            // a gate: it learns only while the gate node had lines on (or none) a frame ago
            if let Some((g, on)) = eg.gate {
                let f = nodes[g].get_frame(at(live[g], sp.gate_lag.max(1))).as_words();
                let f = sp.gate_window.map_or(f, |(o, n)| &f[o.min(f.len())..(o + n).min(f.len())]);
                let open = f.iter().any(|&w| w != 0) == on;
                // (replication) a repeating kernel that acted `repeat_lag` steps ago is confirmed
                // if the gate opened (what should follow its act did), else it counts down
                if sp.replicate > 0 && eg.repeat {
                    if let Some(class) = eg.group.classes_mut().first_mut() {
                        for k in class.kernels_mut() {
                            if k.age as usize == eg.repeat_lag {
                                k.confirm(open, None);
                            }
                        }
                    }
                }
                // extinction on a repeat edge: a kernel that acted `repeat_lag` steps ago, with
                // the gate still shut, fades (chance 1/4)
                if !open && eg.repeat {
                    let then = view(&nodes[eg.src.idx], live[eg.src.idx], eg.src.read_back + eg.repeat_lag, 1, eg.window);
                    if let Some(class) = eg.group.classes_mut().first_mut() {
                        for k in class.kernels_mut() {
                            if k.age as usize == eg.repeat_lag && rng.gen_range(0..1u32 << sp.fade.min(31)) == 0 {
                                k.fade(&then, rng);
                            }
                        }
                    }
                }
                if !open {
                    continue;
                }
            }
            // (the three-factor rule, `EdgeSpec::elig`) the modulator is on (the gate is open):
            // each kernel that fired `elig` steps ago is checked against its destination now, less
            // what this population writes this step; right, it counts up; wrong, an output line
            // moves toward what is there and it counts down, and wrong at 0 it dies
            // (a rule table) what each eligible check moves; without one, the runtime's own
            let rule = if sp.rule != 0 { sp.rule } else { crate::program::graph::RULE_SEED };
            let entry = |at: u8| crate::program::graph::rule_entry(rule, at);
            let (right_in, wrong_in, wrong_out, dies) = (entry(6), entry(8), entry(10), rule & (1 << 12) != 0);
            if (sp.elig > 0 || entry(4) == 1) && sp.teacher.is_none() {
                // what it read when it fired `elig` steps ago (for the input moves)
                let then = if sp.elig > 0 && (right_in > 0 || wrong_in > 0) { Some(view(&nodes[eg.src.idx], live[eg.src.idx], eg.src.read_back + sp.elig as usize, 1, eg.window)) } else { None };
                if let Some(class) = eg.group.classes_mut().first_mut() {
                    let mut target: Vec<u64> = nodes[eg.dst].current().as_words().to_vec();
                    for &k in &fired[i] {
                        let kk = &class.kernels()[k];
                        for (j, &m) in kk.output_mask.as_words().iter().enumerate() {
                            if let Some(t) = target.get_mut(kk.output_idx + j) {
                                *t &= !m;
                            }
                        }
                    }
                    // (a rule table) on firing, the output moves toward what is there
                    if entry(4) == 1 {
                        for &k in &fired[i] {
                            class.kernels_mut()[k].learn_toward(&target, sp.out_learn, rng);
                        }
                    }
                    for (k, kk) in class.kernels_mut().iter_mut().enumerate() {
                        if sp.elig == 0 || kk.age as usize != sp.elig as usize {
                            continue;
                        }
                        let o = kk.output_mask.as_words();
                        let lines: u32 = o.iter().map(|w| w.count_ones()).sum();
                        let held: u32 = o.iter().enumerate().map(|(j, &m)| (m & target.get(kk.output_idx + j).copied().unwrap_or(0)).count_ones()).sum();
                        let right = lines > 0 && 2 * held >= lines;
                        let input = if right { right_in } else { wrong_in };
                        if let Some(then) = then.as_ref().filter(|_| input > 0) {
                            if input == 1 { kk.reinforce(then, rng) } else { kk.fade(then, rng) }
                        }
                        if right {
                            kk.confirmed = kk.confirmed.saturating_add(1).min(8);
                        } else if kk.confirmed == 0 && dies {
                            eg.dying.push(k);
                        } else {
                            kk.confirmed = kk.confirmed.saturating_sub(1);
                            if wrong_out == 1 {
                                kk.learn_toward(&target, sp.out_learn, rng);
                            }
                        }
                    }
                }
            }
            // a predictor learns against its target a frame later, not against its destination
            let mut predictor = false;
            if let Some((tn, tw, _)) = eg.spec.teacher {
                predictor = true;
                let f = nodes[tn].get_frame(at(live[tn], 1)).as_words();
                let t: Vec<u64> = match tw { Some((o, n)) => f[o.min(f.len())..(o + n).min(f.len())].to_vec(), None => f.to_vec() };
                let then = eg.teach_src.take();
                let fired_then = std::mem::take(&mut eg.teach_fired);
                let mut predicted = vec![0u64; t.len()];
                let (mut hits, mut misses) = (0i32, 0i32);
                // (a shadow with a hint band) kernels between its bounds, and under them
                let (mut within, mut under) = (0i32, 0i32);
                let band = eg.silent && sp.hint_low > 0;
                if let Some(class) = eg.group.classes_mut().first_mut() {
                    for &k in &fired_then {
                        if k >= class.len() {
                            continue;
                        }
                        let kk = &mut class.kernels_mut()[k];
                        let o = kk.output_idx;
                        let m: Vec<u64> = kk.output_mask.as_words().to_vec();
                        // the residue on the lines it wrote: its prediction XOR what came on there;
                        // right when that popcount is under its threshold (a fraction of its lines)
                        let lines: u32 = m.iter().map(|w| w.count_ones()).sum();
                        let residue: u32 = m.iter().enumerate().map(|(j, &w)| (w & !t.get(o + j).copied().unwrap_or(0)).count_ones()).sum();
                        let hit = lines > 0 && 4 * (lines - residue) >= sp.hit_quarters.clamp(1, 4) as u32 * lines;
                        if band {
                            match in_band(lines, residue, sp.hint_low, sp.hit_quarters) {
                                -1 => under += 1,
                                1 => within += 1,
                                _ => {}
                            }
                        }
                        for (j, &w) in m.iter().enumerate() {
                            if let Some(p) = predicted.get_mut(o + j) { *p |= w; }
                        }
                        if hit {
                            hits += 1;
                            if let Some(th) = &then { kk.reinforce(th, rng); }
                        } else {
                            misses += 1;
                            if let Some(th) = &then { kk.fade(th, rng); }
                            kk.learn_toward(&t, sp.out_learn, rng);
                        }
                        if sp.replicate > 0 {
                            kk.confirm(hit, None);
                        }
                    }
                    // lines of the target that nobody predicted recruit a kernel on what was read
                    // then, writing those lines (one birth a step, under the cap)
                    if eg.grow && class.len() < sp.cap as usize {
                        if let Some(th) = &then {
                            // (only where it can write: its scratch, or its destination, which may be
                            // narrower than the target)
                            let room = if eg.silent { eg.scratch.word_len() } else { nodes[eg.dst].current().word_len() };
                            let un: Vec<u64> = t.iter().zip(&predicted).take(room).map(|(a, b)| a & !b).collect();
                            let (dw, dn) = busiest_word(&un);
                            let (sw, sn) = busiest_word(th.as_words());
                            if dn as usize >= sp.out_min.max(1) as usize && sn > 0 {
                                let lines = ones_of(un[dw]);
                                let out: Vec<usize> = lines.choose_multiple(rng, (sp.out_max as usize).min(lines.len())).copied().collect();
                                let active = ones_of(th.as_words()[sw]);
                                let syn: Vec<usize> = active.choose_multiple(rng, (sp.birth_syn as usize).min(active.len())).copied().collect();
                                let threshold = (sp.birth_quarters as usize * syn.len()).div_ceil(4).max(1);
                                let mut k = SimpleKernel::new(BitVector::from_bits(&syn, 64), sw, BitVector::from_bits(&out, 64), dw, threshold, KernelOp::Or);
                                k.learn_fire = false;
                                class.add(k);
                            }
                        }
                    }
                }
                eg.confirms = eg.confirms.saturating_add(hits.max(0) as u32);
                eg.score += if band { within - under } else { hits - misses };
                eg.teach_fired = fired[i].clone();
                eg.teach_src = seen[i].as_ref().map(|v| v.get(nodes).clone());
                // a shadow moves to its next target after `RETARGET` steps, unless it predicts this one
                if eg.silent {
                    eg.on_target += 1;
                    if eg.on_target >= RETARGET {
                        let kind = eg.targets.get(eg.target_at).map_or(0, |t| t.2);
                        let sc = eg.score.max(0) as u32;
                        match eg.scores.iter_mut().find(|x| x.0 == kind) {
                            Some(x) => x.1 = x.1.max(sc),
                            None => eg.scores.push((kind, sc)),
                        }
                        if eg.score < KEEP_SHADOW && eg.targets.len() > 1 {
                            eg.target_at = (eg.target_at + 1) % eg.targets.len();
                            let nt = eg.targets[eg.target_at];
                            eg.spec.teacher = Some(nt);
                            let len = nt.1.map_or(nodes[nt.0].current().word_len(), |w| w.1);
                            eg.scratch = BitVector::new(len.max(1) * 64, Some(0));
                            if let Some(class) = eg.group.classes_mut().first_mut() { class.clear(); }
                            eg.teach_fired.clear();
                            eg.teach_src = None;
                        }
                        eg.on_target = 0;
                        eg.score = 0;
                    }
                }
            }
            let Some(class) = eg.group.classes_mut().first_mut() else { continue };
            if !predictor && matches!(eg.op, KernelOp::Xor | KernelOp::Clear) {
                let dest = nodes[eg.dst].current();
                let mut came_true = 0u32;
                for &k in &fired[i] {
                    // its prediction came true if every line it wrote is now off (counted; and,
                    // with replication, confirmed)
                    let kk = &class.kernels()[k];
                    let left = kk.output_mask.as_words().iter().enumerate().any(|(j, &m)| dest.as_words().get(kk.output_idx + j).is_some_and(|&d| d & m != 0));
                    came_true += !left as u32;
                    if sp.replicate > 0 {
                        class.kernels_mut()[k].confirm(!left, None);
                    }
                    class.kernels_mut()[k].learn_output(dest, sp.out_learn, rng);
                }
                eg.confirms = eg.confirms.saturating_add(came_true);
            }
            // replication: the first kernel confirmed 2^replicate times (net) is copied, once a
            // step, while there is room under the cap
            if sp.replicate > 0 && class.len() < sp.cap as usize {
                let need = (1u32 << sp.replicate.min(7)).min(255) as u8;
                if let Some(k) = class.kernels().iter().position(|k| k.confirmed >= need) {
                    class.kernels_mut()[k].confirmed = 0;
                    let parent = class.kernels()[k].clone();
                    let child = match (sp.replicate_shift, parent.tile_at, seen[i].as_ref().map(|s| s.get(nodes))) {
                        (true, Some((start, bits)), Some(view)) if view.bit_len() > bits => {
                            // the neighbouring tile, one way or the other (wrapping), output at the
                            // same relative place (not into the act, whose outputs are moves)
                            let span = view.bit_len() - bits + 1;
                            let to = if rng.gen_bool(0.5) { (start + bits) % span } else { (start + span - bits % span) % span };
                            let dwords = nodes[eg.dst].current().word_len().max(1);
                            let (base, len) = if sp.dst_len > 0 { (sp.dst_word % dwords, sp.dst_len.min(dwords - sp.dst_word % dwords)) } else { (0, dwords) };
                            let out = if eg.can_grow { base + (to * len / view.word_len().max(1)).min(len - 1) } else { parent.output_idx };
                            parent.moved_to(to, out)
                        }
                        _ => {
                            let mut c = parent.clone();
                            c.confirmed = 0;
                            c.age = u8::MAX;
                            c
                        }
                    };
                    class.add(child);
                }
            }
            if predictor || !(eg.grow && eg.can_grow) || class.len() >= sp.cap as usize {
                continue;
            }
            // a repeat birth reads its source as it was when the repeated act was made
            let older;
            let src = if eg.repeat && eg.repeat_lag > 1 {
                older = view(&nodes[eg.src.idx], live[eg.src.idx], eg.src.read_back + eg.repeat_lag, 1, eg.window);
                &older
            } else {
                seen[i].as_ref().unwrap().get(nodes)
            };
            let ones = ones_of;
            let dest = nodes[eg.dst].current();
            if sp.tile > 0 {
                // full coverage: a kernel is born in the tile (2^tile bits, every half tile) with
                // the most active input lines no firing kernel reads, on those lines, writing at
                // the same relative place in the destination
                let sw = src.as_words();
                let mut covered = vec![0u64; sw.len()];
                for &k in &fired[i] {
                    let kk = &class.kernels()[k];
                    for (j, m) in kk.input_mask.as_words().iter().enumerate() {
                        if let Some(c) = covered.get_mut(kk.input_idx + j) { *c |= m; }
                    }
                }
                let bits = src.bit_len();
                let t = (1usize << sp.tile.min(20)).min(bits).max(1);
                let stride = (t / 2).max(1);
                let tiles = if t >= bits { 1 } else { (bits - t) / stride + 1 };
                let count = |s: usize| -> u32 { (s / 64..=(s + t - 1) / 64).map(|w| (sw[w] & !covered[w] & SimpleKernel::tile_word(s, t, w)).count_ones()).sum() };
                // only a tile over a word with uncovered lines can have any, so only those are
                // scored (the same winner: the most lines, the first on a tie; none: no birth)
                let mut cand: Vec<usize> = Vec::new();
                let mut next = 0usize;
                for w in 0..sw.len() {
                    if sw[w] & !covered[w] == 0 {
                        continue;
                    }
                    let lo = (w * 64 + 1).saturating_sub(t).div_ceil(stride).max(next);
                    let hi = ((w * 64 + 63) / stride).min(tiles - 1);
                    if lo <= hi {
                        cand.extend(lo..=hi);
                        next = hi + 1;
                    }
                }
                let (start, n) = cand.iter().map(|&j| (j * stride, count(j * stride))).max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))).unwrap_or((0, 0));
                if n == 0 {
                    continue;
                }
                let lines: Vec<usize> = (start / 64..=(start + t - 1) / 64)
                    .flat_map(|w| ones(sw[w] & !covered[w] & SimpleKernel::tile_word(start, t, w)).into_iter().map(move |b| w * 64 + b))
                    .collect();
                let syn: Vec<usize> = if sp.expect {
                    // two-mask: half its lines on uncovered input, half on lines of the tile that
                    // are off (it will expect each as it is now)
                    let on = (sp.birth_syn as usize).div_ceil(2).min(lines.len());
                    let mut syn: Vec<usize> = lines.choose_multiple(rng, on).copied().collect();
                    let off: Vec<usize> = (start / 64..=(start + t - 1) / 64)
                        .flat_map(|w| ones(!sw[w] & SimpleKernel::tile_word(start, t, w)).into_iter().map(move |b| w * 64 + b))
                        .collect();
                    syn.extend(off.choose_multiple(rng, (sp.birth_syn as usize).saturating_sub(on).min(off.len())).copied());
                    syn
                } else {
                    lines.choose_multiple(rng, (sp.birth_syn as usize).min(lines.len())).copied().collect()
                };
                let threshold = (sp.birth_quarters as usize * syn.len()).div_ceil(4).max(1);
                let (out, dw) = if eg.repeat {
                    let prev = nodes[eg.dst].get_frame(at(live[eg.dst], eg.repeat_lag));
                    let (dw, dn) = busiest_word(prev.as_words());
                    if (dn as usize) < sp.out_min as usize {
                        continue;
                    }
                    (ones(prev.as_words()[dw]), dw)
                } else if let Some((w, _)) = eg.out_word {
                    // (`set_attract_out`) a body code in its word
                    let codes: Vec<u64> = motor.iter().filter(|m| m.0 == w).map(|m| m.1).collect();
                    let Some(&c) = codes.choose(rng) else { continue };
                    (ones(c), w)
                } else if eg.into_act {
                    // into a port of the act: the move being made now (the busiest word of the
                    // port), copied onto what is seen: a reflex from this input to that move; or
                    // (`EdgeSpec::birth_lag`) the move made that many frames ago, onto what is seen
                    // now: what came of that move (the sound one's own voice made), so the kernel
                    // makes the move that brings what it sees
                    let then = if sp.birth_lag > 0 { nodes[eg.dst].get_frame(at(live[eg.dst], sp.birth_lag as usize)) } else { dest };
                    let dwords = then.word_len().max(1);
                    let (base, len) = if sp.dst_len > 0 { (sp.dst_word % dwords, sp.dst_len.min(dwords - sp.dst_word % dwords)) } else { (0, dwords) };
                    let (w, n) = busiest_word(&then.as_words()[base..base + len]);
                    if (n as usize) < sp.out_min as usize {
                        continue;
                    }
                    (ones(then.as_words()[base + w]), base + w)
                } else {
                    let dwords = dest.word_len().max(1);
                    let (base, len) = if sp.dst_len > 0 { (sp.dst_word % dwords, sp.dst_len.min(dwords - sp.dst_word % dwords)) } else { (0, dwords) };
                    let dw = base + (start * len / sw.len().max(1)).min(len - 1);
                    let wrote = fired[i].iter().map(|&k| &class.kernels()[k]).filter(|kk| kk.output_idx == dw).fold(0u64, |a, kk| a | kk.output_mask.as_words()[0]);
                    let left = ones(dest.as_words()[dw] & !wrote);
                    if left.len() >= sp.out_min as usize {
                        (left.choose_multiple(rng, (sp.out_max as usize).min(left.len())).copied().collect(), dw)
                    } else {
                        let all: Vec<usize> = (0..64).collect();
                        (all.choose_multiple(rng, sp.code_lines as usize).copied().collect(), dw)
                    }
                };
                let out_mask = BitVector::from_bits(&out, 64);
                let mut k = SimpleKernel::tile(start, t, &syn, out_mask.clone(), dw, threshold, eg.op);
                if sp.expect {
                    k.expecting(Some(src));
                }
                // (the three-factor rule) a newborn starts with two chances
                if sp.elig > 0 {
                    k.confirmed = 2;
                }
                if eg.new_code {
                    let d = nodes[eg.dst].current_mut();
                    apply(&mut d.as_words_mut()[dw..dw + 1], out_mask.as_words(), eg.op);
                }
                class.add(k);
                continue;
            }
            // what it writes: a fresh random code into an empty destination (`new_code`), or the
            // destination word with the most lines its firing kernels left uncovered
            let (out, dw) = if eg.repeat {
                // what the destination held a frame ago (the act just made)
                let prev = nodes[eg.dst].get_frame(at(live[eg.dst], eg.repeat_lag));
                let (dw, dn) = busiest_word(prev.as_words());
                if (dn as usize) < sp.out_min as usize {
                    continue;
                }
                (ones(prev.as_words()[dw]), dw)
            } else if let Some((w, _)) = eg.out_word {
                // (`set_attract_out`) a body code in its word
                let codes: Vec<u64> = motor.iter().filter(|m| m.0 == w).map(|m| m.1).collect();
                let Some(&c) = codes.choose(rng) else { continue };
                (ones(c), w)
            } else if eg.new_code {
                // (not into the act: a random code there is no move)
                if eg.into_act || (dest.count_ones() as usize) >= sp.out_min as usize {
                    continue;
                }
                let dw = rng.gen_range(0..dest.word_len().max(1));
                let lines: Vec<usize> = (0..64).collect();
                (lines.choose_multiple(rng, sp.code_lines as usize).copied().collect::<Vec<usize>>(), dw)
            } else {
                // the destination word with the most lines no firing kernel wrote (word by word, no
                // copy of the destination)
                let mut covered: Vec<(usize, u64)> = Vec::new();
                for &k in &fired[i] {
                    let kk = &class.kernels()[k];
                    for (j, m) in kk.output_mask.as_words().iter().enumerate() {
                        covered.push((kk.output_idx + j, *m));
                    }
                }
                let cover = |w: usize| covered.iter().filter(|c| c.0 == w).fold(0u64, |a, c| a | c.1);
                let (dw, dn) = dest.as_words().iter().enumerate().filter(|(_, x)| **x != 0)
                    .map(|(w, x)| (w, (x & !cover(w)).count_ones()))
                    .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))).unwrap_or((0, 0));
                if (dn as usize) < sp.out_min as usize {
                    continue;
                }
                let lines = ones(dest.as_words()[dw] & !cover(dw));
                (lines.choose_multiple(rng, (sp.out_max as usize).min(lines.len())).copied().collect(), dw)
            };
            // what it reads: the source word with the most active lines, or (`wide`) all of the
            // source's active lines
            let (syn_mask, sw) = if eg.wide {
                // k of its n active lines, drawn by rank (the same draw as choosing from the list
                // of active lines, without making the list): each rank is found by walking word
                // popcounts and selecting the bit within the word
                let words = src.as_words();
                let n: usize = words.iter().map(|x| x.count_ones() as usize).sum();
                if n == 0 {
                    continue;
                }
                let mut syn: Vec<usize> = Vec::with_capacity(sp.birth_syn as usize);
                for mut r in rand::seq::index::sample(rng, n, (sp.birth_syn as usize).min(n)).into_iter() {
                    for (w, x) in words.iter().enumerate() {
                        let c = x.count_ones() as usize;
                        if r < c {
                            syn.push(w * 64 + select_nth_bit(*x, r));
                            break;
                        }
                        r -= c;
                    }
                }
                (BitVector::from_bits(&syn, src.bit_len()), 0)
            } else {
                let (sw, sn) = busiest_word(src.as_words());
                if sn == 0 {
                    continue;
                }
                let active = ones(src.as_words()[sw]);
                let syn: Vec<usize> = active.choose_multiple(rng, (sp.birth_syn as usize).min(active.len())).copied().collect();
                (BitVector::from_bits(&syn, 64), sw)
            };
            let threshold = (sp.birth_quarters as usize * syn_mask.count_ones()).div_ceil(4).max(1);
            let out_mask = BitVector::from_bits(&out, 64);
            let _ = dest;
            let mut k = SimpleKernel::new(syn_mask, sw, out_mask.clone(), dw, threshold, eg.op);
            if sp.sparse {
                k.sparsify();
            }
            if sp.expect {
                k.expecting(Some(src));
            }
            // (the three-factor rule) a newborn starts with two chances
            if sp.elig > 0 {
                k.confirmed = 2;
            }
            if eg.new_code {
                // a new code fires as it is born (the input that recruited it is here now)
                let d = nodes[eg.dst].current_mut();
                apply(&mut d.as_words_mut()[dw..dw + 1], out_mask.as_words(), eg.op);
            }
            class.add(k);
        }
        if self.explore {
            self.grow_trials();
        }
        if attract() {
            self.grow_attracted();
        }
        if attract_out() && !frozen() {
            self.grow_attracted_out();
        }
    }

    /// The attractant (`set_attract`): each unread input port whose lines changed this step
    /// counts it; at the lowest threshold of the populations that may answer, that population
    /// grows a trial connection from the port into its destination.
    fn grow_attracted(&mut self) {
        let cw = crate::world::CH_BITS / 64;
        let ports: [(usize, Option<(usize, usize)>); 5] = [
            (0, Some((crate::world::EYE * cw, cw))),
            (0, Some((crate::world::EAR * cw, cw))),
            (0, Some((crate::world::TOUCH * cw, cw))),
            (0, Some((crate::world::AFFECT * cw, cw))),
            (self.self_node.unwrap_or(usize::MAX), None),
        ];
        let internal = |i: usize, s: &Self| i != 0 && i != s.output && Some(i) != s.self_node;
        for (j, &(node, window)) in ports.iter().enumerate() {
            if node >= self.nodes.len() || self.live[node] {
                continue;
            }
            // read by any population (all of the senses, or the port's own words)?
            let read = self.edges.iter().any(|e| e.src.idx == node && match (e.window, window) {
                (None, _) | (_, None) => true,
                (Some((o, l)), Some((po, pl))) => o < po + pl && po < o + l,
            });
            if read {
                self.attracted[j] = 0;
                continue;
            }
            let now = view(&self.nodes[node], false, 0, 1, window);
            let before = view(&self.nodes[node], false, 1, 1, window);
            if now.as_words() != before.as_words() {
                self.attracted[j] = self.attracted[j].saturating_add(1);
            }
            // the population that answers: a genome's own, of kernels, on a channel, not grown
            // yet this lifetime, with the lowest threshold
            let answer = (0..self.edges.len()).filter(|&i| {
                let e = &self.edges[i];
                e.explorer == 0 && !e.spawned && e.kind == EdgeKind::Kernels && internal(e.dst, self) && e.spec.accident > 0
            }).min_by_key(|&i| (self.edges[i].spec.accident, i));
            let Some(i) = answer else { continue };
            let threshold = 1u32 << (self.edges[i].spec.accident as u32).saturating_sub(10).min(8);
            if (self.attracted[j] as u32) < threshold {
                continue;
            }
            self.attracted[j] = 0;
            let eg = &self.edges[i];
            let mut spec = eg.spec.clone();
            spec.op = KernelOp::Xor;
            spec.replicate = spec.replicate.max(3);
            spec.cap = spec.cap.max(16);
            spec.gate = None;
            spec.pause = None;
            spec.repeat = false;
            spec.dst_len = 0;
            spec.dst_word = 0;
            let rng = StdRng::seed_from_u64((spec.salt as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ self.steps ^ 0xA77A);
            let trial = RuntimeEdgeGroup {
                src: RuntimeEdgeTarget { idx: node, read_back: 1, span: 1 },
                dst: eg.dst,
                group: KernelGroup::from_classes(vec![KernelClass::with_kernels(vec![])]),
                kind: EdgeKind::Kernels,
                op: KernelOp::Xor,
                rotate: 0,
                grow: true,
                inhibit: eg.inhibit,
                gate: None,
                new_code: false,
                wide: eg.wide,
                window,
                dst_word: 0,
                repeat: false,
                repeat_lag: 1,
                spec,
                rng,
                can_grow: true,
                into_act: false,
                confirms: 0,
                explorer: 8 + j as u8,
                spawned: true,
                silent: false,
                scratch: BitVector::new(64, Some(0)),
                share: 0, grip: None, detached: false, dying: Vec::new(),
                teach_fired: Vec::new(),
                teach_src: None,
                targets: Vec::new(),
                target_at: 0,
                on_target: 0,
                score: 0,
                scores: Vec::new(),
                out_word: None,
            };
            self.edges[i].spawned = true;
            self.edges.push(trial);
        }
    }

    /// The attractant on the output side (`set_attract_out`): each word of the act holding a body
    /// code that nothing wrote this step counts it; at the lowest threshold of the populations that
    /// may answer, that population grows a trial reading its own source and writing that word.
    fn grow_attracted_out(&mut self) {
        let out = self.nodes[self.output].current().as_words().to_vec();
        if self.attracted_out.len() != out.len() {
            self.attracted_out = vec![0; out.len()];
        }
        let half = out.len() / 2;
        for w in 0..out.len() {
            if !self.motor.iter().any(|m| m.0 == w) {
                continue;
            }
            if out[w] != 0 || self.edges.iter().any(|e| e.out_word.is_some_and(|o| o.0 == w)) {
                self.attracted_out[w] = 0;
                continue;
            }
            self.attracted_out[w] = self.attracted_out[w].saturating_add(1);
            let answer = (0..self.edges.len()).filter(|&i| {
                let e = &self.edges[i];
                e.explorer == 0 && !e.spawned && e.kind == EdgeKind::Kernels && e.spec.accident > 0
            }).min_by_key(|&i| (self.edges[i].spec.accident, i));
            let Some(i) = answer else { continue };
            let threshold = 1u32 << (self.edges[i].spec.accident as u32).saturating_sub(10).min(8);
            if (self.attracted_out[w] as u32) < threshold {
                continue;
            }
            self.attracted_out[w] = 0;
            let eg = &self.edges[i];
            let mut spec = eg.spec.clone();
            spec.op = KernelOp::Or;
            spec.replicate = spec.replicate.max(3);
            spec.cap = spec.cap.max(16);
            spec.gate = None;
            spec.pause = None;
            spec.repeat = false;
            spec.dst_word = w;
            spec.dst_len = 1;
            let rng = StdRng::seed_from_u64((spec.salt as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ self.steps ^ 0x0A77);
            let trial = RuntimeEdgeGroup {
                src: RuntimeEdgeTarget { idx: eg.src.idx, read_back: eg.src.read_back, span: eg.src.span },
                dst: self.output,
                group: KernelGroup::from_classes(vec![KernelClass::with_kernels(vec![])]),
                kind: EdgeKind::Kernels,
                op: KernelOp::Or,
                rotate: 0,
                grow: true,
                inhibit: eg.inhibit,
                gate: None,
                new_code: false,
                wide: eg.wide,
                window: eg.window,
                dst_word: w,
                repeat: false,
                repeat_lag: 1,
                spec,
                rng,
                can_grow: true,
                into_act: false,
                confirms: 0,
                explorer: if w < half { 13 } else { 14 },
                spawned: true,
                silent: false,
                scratch: BitVector::new(64, Some(0)),
                share: 0, grip: None, detached: false, dying: Vec::new(),
                teach_fired: Vec::new(),
                teach_src: None,
                targets: Vec::new(),
                target_at: 0,
                on_target: 0,
                score: 0,
                scores: Vec::new(),
                out_word: Some((w, 0)),
            };
            self.edges[i].spawned = true;
            self.edges.push(trial);
        }
    }

    /// Exploration: each population of kernels (a genome's own) whose kernels have all gone
    /// `STUCK` steps without firing, and whose destination is a vector it may read (not the act),
    /// grows once a lifetime a trial connection: a predicting (XOR) population, replicating, that
    /// reads that destination and writes back into the population's source if the source is a
    /// channel (a loop), else into the destination itself. It starts with no kernels: they are
    /// born where its input is uncovered, as any growing population's are.
    fn grow_trials(&mut self) {
        let n = self.edges.len();
        let internal = |i: usize, s: &Self| i != 0 && i != s.output && Some(i) != s.self_node;
        let mut grown = Vec::new();
        for i in 0..n {
            let eg = &self.edges[i];
            if eg.explorer > 0 || eg.spawned || eg.kind != EdgeKind::Kernels || (!self.shadow && !internal(eg.dst, self)) {
                continue;
            }
            let Some(class) = eg.group.classes().first() else { continue };
            // a freak accident: with chance 2^-accident a step (its own gene), drawn from its id
            // and the step, it grows the trial whether or not it is stuck
            let k = eg.spec.accident.min(30) as u32;
            let accident = k > 0 && {
                let h = (eg.spec.salt as u64 ^ 0xA5A5_5A5A).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ self.steps.wrapping_mul(0xD6E8_FEB8_6659_FD93);
                (h ^ (h >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9) >> (64 - k) == 0
            };
            let stuck = !class.kernels().is_empty() && class.kernels().iter().all(|k| k.age >= STUCK);
            if !accident && (!stuck || (self.shadow && self.shadow_random)) {
                continue;
            }
            if self.shadow {
                // a shadow predictor: what this population reads, predicting each candidate in turn
                let targets = self.targets(eg.src.idx, eg.window, eg.spec.salt);
                let Some(&first) = targets.first() else { continue };
                let mut spec = eg.spec.clone();
                spec.op = KernelOp::Or;
                spec.teacher = Some(first);
                spec.replicate = spec.replicate.max(3);
                spec.cap = spec.cap.max(16);
                spec.gate = None;
                spec.pause = None;
                spec.repeat = false;
                spec.dst_len = 0;
                spec.dst_word = 0;
                let len = first.1.map_or(self.nodes[first.0].current().word_len(), |w| w.1);
                let rng = StdRng::seed_from_u64((spec.salt as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ self.steps ^ 0x5AD0);
                grown.push((i, RuntimeEdgeGroup {
                    src: RuntimeEdgeTarget { idx: eg.src.idx, read_back: 1, span: 1 },
                    dst: eg.dst,
                    group: KernelGroup::from_classes(vec![KernelClass::with_kernels(vec![])]),
                    kind: EdgeKind::Kernels,
                    op: KernelOp::Or,
                    rotate: 0,
                    grow: true,
                    inhibit: eg.inhibit,
                    gate: None,
                    new_code: false,
                    wide: false,
                    window: eg.window,
                    dst_word: 0,
                    repeat: false,
                    repeat_lag: 1,
                    spec,
                    rng,
                    can_grow: true,
                    into_act: false,
                    confirms: 0,
                    explorer: 3,
                    spawned: true,
                    silent: true,
                    scratch: BitVector::new(len.max(1) * 64, Some(0)),
                    share: 0, grip: None, detached: false, dying: Vec::new(),
                    teach_fired: Vec::new(),
                    teach_src: None,
                    targets,
                    target_at: 0,
                    on_target: 0,
                    score: 0,
                    scores: Vec::new(),
                    out_word: None,
                }));
                continue;
            }
            let back = internal(eg.src.idx, self) && eg.src.idx != eg.dst;
            let mut spec = eg.spec.clone();
            spec.op = KernelOp::Xor;
            spec.replicate = spec.replicate.max(3);
            spec.cap = spec.cap.max(16);
            spec.gate = None;
            spec.pause = None;
            spec.repeat = false;
            spec.dst_len = 0;
            spec.dst_word = 0;
            let rng = StdRng::seed_from_u64((spec.salt as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ self.steps);
            grown.push((i, RuntimeEdgeGroup {
                src: RuntimeEdgeTarget { idx: eg.dst, read_back: 1, span: 1 },
                dst: if back { eg.src.idx } else { eg.dst },
                group: KernelGroup::from_classes(vec![KernelClass::with_kernels(vec![])]),
                kind: EdgeKind::Kernels,
                op: KernelOp::Xor,
                rotate: 0,
                grow: true,
                inhibit: eg.inhibit,
                gate: None,
                new_code: false,
                wide: eg.wide,
                window: None,
                dst_word: 0,
                repeat: false,
                repeat_lag: 1,
                spec,
                rng,
                can_grow: true,
                into_act: false,
                confirms: 0,
                explorer: if back { 1 } else { 2 },
                spawned: true,
                silent: false,
                scratch: BitVector::new(64, Some(0)),
                share: 0, grip: None, detached: false, dying: Vec::new(),
                teach_fired: Vec::new(),
                teach_src: None,
                targets: Vec::new(),
                target_at: 0,
                on_target: 0,
                score: 0,
                scores: Vec::new(),
                out_word: None,
            }));
        }
        for (i, trial) in grown {
            self.edges[i].spawned = true;
            self.edges.push(trial);
        }
    }

    /// What this lifetime found, for the genome: per population (by id, its `salt`), the
    /// predictions that came true, and each trial connection grown from it (kind 1: back into
    /// its source, 2: on its destination; 8 to 12: from an unread input port, `set_attract`) with
    /// its own.
    pub fn notes(&self) -> Vec<(u32, u32, u8)> {
        // (`set_attract`: kinds 8 to 12, a trial grown from an unread port, eye to the self channel)
        let mut out: Vec<(u32, u32, u8)> = self.edges.iter().filter(|e| (e.explorer < 3 || (8..=14).contains(&e.explorer)) && (e.confirms > 0 || e.explorer > 0)).map(|e| (e.spec.salt, e.confirms, e.explorer)).collect();
        // a shadow's targets, each with its best score (the one it is on included)
        for e in self.edges.iter().filter(|e| e.explorer == 3) {
            let mut sc = e.scores.clone();
            if let Some(t) = e.targets.get(e.target_at) {
                let now = e.score.max(0) as u32;
                match sc.iter_mut().find(|x| x.0 == t.2) {
                    Some(x) => x.1 = x.1.max(now),
                    None => sc.push((t.2, now)),
                }
            }
            out.extend(sc.into_iter().filter(|x| x.1 > 0).map(|(kind, c)| (e.spec.salt, c, kind)));
        }
        out
    }

    /// What a shadow reading node `src` (words `window`) may predict, in turn from a place
    /// drawn from its population's id: the eye, ear, touch and affect (kinds 17-20), the self
    /// channel (21), the voice and hand (23, 24), and every declared channel (32 + its index - 1),
    /// less what it reads.
    pub fn targets(&self, src: usize, window: Option<(usize, usize)>, salt: u32) -> Vec<(usize, Option<(usize, usize)>, u8)> {
        let cw = crate::world::CH_BITS / 64;
        let mut t: Vec<(usize, Option<(usize, usize)>, u8)> = Vec::new();
        for (j, s) in [crate::world::EYE, crate::world::EAR, crate::world::TOUCH, crate::world::AFFECT].into_iter().enumerate() {
            t.push((0, Some((s * cw, cw)), 17 + j as u8));
        }
        if let Some(sn) = self.self_node {
            t.push((sn, None, 21));
        }
        for (j, s) in [crate::world::VOICE, crate::world::HAND].into_iter().enumerate() {
            t.push((self.output, Some((s * cw, cw)), 23 + j as u8));
        }
        for node in 1..self.nodes.len() {
            if node != self.output && Some(node) != self.self_node && node - 1 < 200 {
                t.push((node, None, 32 + (node - 1) as u8));
            }
        }
        // not what it reads (the same words, or a sense inside all of the senses it reads)
        t.retain(|x| !(x.0 == src && (window.is_none() || x.1 == window)));
        if !t.is_empty() {
            let k = (salt as usize).wrapping_mul(0x9E37_79B9) % t.len();
            t.rotate_left(k);
        }
        t
    }

    /// The self node's frame this step, written by the runtime before any edge runs (bits only,
    /// for kernels to read like any other vector):
    /// - word 0: last step's work (kernels matched plus words wired), as a thermometer: the low
    ///   `bit length of the count` lines on, so popcount reads the order of magnitude;
    /// - word 1: how many kernels it has, the same way;
    /// - word 2: the steps so far in binary (line k flips every 2^k steps: a clock), from a start
    ///   drawn each lifetime under `set_clock_drawn`;
    /// - word 3: how many edges ran last step (not paused), the same way as word 0.
    fn write_self(&mut self) {
        let Some(m) = self.self_node else { return };
        let therm = |x: u64| -> u64 { let b = 64 - x.leading_zeros(); if b >= 64 { u64::MAX } else { (1u64 << b) - 1 } };
        let kernels: u64 = self.edges.iter().map(|e| e.group.kernel_count() as u64).sum();
        let words = [therm(self.last_work), therm(kernels), self.steps.wrapping_add(self.clock_from), therm(self.last_ran)];
        let cur = self.nodes[m].current_mut().as_words_mut();
        for (d, w) in cur.iter_mut().zip(words) {
            *d = w;
        }
    }

    /// Read-only access to current frame of a node (for inspection).
    pub fn node_current(&self, idx: usize) -> &BitVector {
        self.nodes[idx].current()
    }

    /// Mutable access to current frame of a node (e.g., feed inputs).
    pub fn node_current_mut(&mut self, idx: usize) -> &mut BitVector {
        self.nodes[idx].current_mut()
    }

    pub fn write_input(&mut self, input: &BitVector, word_offset: usize) {
        let node = self.node_current_mut(0);
        node.mask_mut(word_offset, input, |a, b| a | b);
    }

    pub fn read_output(&self) -> &BitVector {
        self.node_current(self.output)
    }

    /// Replace the act node's current frame (`evolve::set_muscle`: what the body did, after the
    /// world's step), so the frames after it hold what was done, not what was written.
    pub fn set_output(&mut self, done: &BitVector) {
        let out = self.output;
        let d = self.node_current_mut(out).as_words_mut();
        for (k, w) in d.iter_mut().enumerate() {
            *w = done.as_words().get(k).copied().unwrap_or(0);
        }
    }

    pub fn read_node(&self, node_idx: usize) -> &BitVector {
        self.node_current(node_idx)
    }

    pub fn write_node(&mut self, input: &BitVector, node_idx: usize, word_offset: usize) {
        let node = self.node_current_mut(node_idx);
        node.mask_mut(word_offset, input, |a, b| a | b);
    }

    pub fn overwrite_node(&mut self, input: &BitVector, node_idx: usize, word_offset: usize) {
        let node = self.node_current_mut(node_idx);
        node.mask_mut(word_offset, input, |_, b| b);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_hint_band_counts_kernels_partly_right() {
        // 16 lines, a band over 1/4 and under 3/4 of them
        assert_eq!(super::in_band(16, 16, 1, 3), -1); // none came on
        assert_eq!(super::in_band(16, 12, 1, 3), -1); // 4 of 16: at the lower bound
        assert_eq!(super::in_band(16, 8, 1, 3), 1); // half
        assert_eq!(super::in_band(16, 5, 1, 3), 1); // 11 of 16
        assert_eq!(super::in_band(16, 4, 1, 3), 0); // 12 of 16: fully right, counts for nothing
        assert_eq!(super::in_band(16, 0, 1, 3), 0);
        assert_eq!(super::in_band(0, 0, 1, 3), 0);
    }

    use super::*;
    use crate::kernel::KernelClass;
    use crate::program::graph::GraphInstruction as I;

    /// The residue circuit, written in the graph language: the input wired (OR) into R, and a
    /// second edge from the input one frame back whose kernels cancel (XOR) and are born where R
    /// is left on. On a repeating sequence R goes to nothing.
    #[test]
    fn residue_circuit_cancels_a_repeating_sequence() {
        let prog = GraphProgram { code: vec![I::EdgeWire(), I::ReadInput(), I::EdgeLag(), I::EdgeXor(), I::EdgeGrow(), I::EdgeShare()] };
        let g = prog.build_graph(&ProgramDefaults { default_bits: 512, ..ProgramDefaults::default() });
        let mut net = RuntimeNetwork::from_graph_with(&g, 512, 512, 1, |_| KernelGroup::from_classes(vec![KernelClass::with_kernels(vec![])]));
        let mut rng = StdRng::seed_from_u64(2);
        let codes: Vec<BitVector> = (0..5).map(|_| crate::world::clustered_code(512, 16, &mut rng)).collect();
        let (mut early, mut late) = (0, 0);
        for t in 0..200 {
            net.tick(AdvanceMode::Clear, 0);
            net.write_input(&codes[t % 5], 0);
            let r = net.read_node(1).count_ones();
            if t < 10 { early += r } else if t >= 150 { late += r }
        }
        assert!(early >= 40, "the first steps are unpredicted ({early})");
        assert_eq!(late, 0, "a learned cycle leaves no residue");
        assert_eq!(net.edges.iter().map(|e| e.group_len()).sum::<usize>(), 5, "one kernel per transition");
    }
    use crate::program::graph::{GraphProgram};

    #[test]
    fn materialize_with_groups_and_tick() {
        // Build: input -> N0 (implicit in program)->output, then tick once
        let prog = GraphProgram::new();
        let mut defaults = ProgramDefaults::default();
        defaults.default_threshold = 1; // make sure kernel fires easily for test

        let mut net = RuntimeNetwork::from_program(&prog, &defaults);

        // Seed input: set 8 low bits
        net.write_input(
            &BitVector::from_words(vec![0xFF]),
            0,
        );
        // One tick: CopyPrev then process edges with default kernel
        net.tick(AdvanceMode::CopyPrev, 16);

        // N0 (node 1) should have fired, with high temperature
        let output = net.read_output();
        assert_eq!(output.count_ones(), 1);
    }

    #[test]
    fn simple_network_learns_static_pattern() {
        // Build: input -> N0 (implicit in program)->output, then tick once
        let prog = GraphProgram::new();
        let defaults = ProgramDefaults::default();
        
        let mut net = RuntimeNetwork::from_program(&prog, &defaults);

        //build vocab
        //need at least 16 bits for the mask to fire (threshold is 16)
        let a = BitVector::from_words(vec![0b01010101010101010101010101010101]);
        net.write_input(&a, 0);
        
        //show that it doesn't respond yet
        net.tick(AdvanceMode::CopyPrev, 0);
        assert_eq!(net.read_output().count_ones(),0);

        //start at a temperature of 16 that forces it to fire
        //and then gradually reduce temperature down
        for i in 0..16 {
            net.tick(AdvanceMode::CopyPrev, 16 - i);
            assert_eq!(net.read_output().count_ones(),1, "Failed at temperature adjustment {}",16 - i);
            net.overwrite_node(&BitVector::new(64, Some(0)), 1, 0); //clear output after first tick
            assert_eq!(net.read_output().count_ones(),0);
        }
        //fire with 0 temperature and show that the pattern has learned
        net.tick(AdvanceMode::CopyPrev, 0);

        //show that the input is preserved
        let input = net.read_node(0);
        assert_eq!(input.count_ones(), 16);
        //show that the kernel has fired to output
        let output = net.read_output();
        assert_eq!(output.count_ones(), 1);

    }
}

#[cfg(test)]
mod motion_tests {
    use super::moved;

    #[test]
    fn a_conveyor_carries_a_word_one_place_a_move() {
        // stride 1: every step
        let a = moved(&[7, 0, 0], 1, false, 0);
        assert_eq!(a, vec![0, 7, 0]);
        assert_eq!(moved(&a, 1, false, 1), vec![0, 0, 7]);
        // off the end it is gone
        assert_eq!(moved(&[0, 0, 7], 1, false, 2), vec![0, 0, 0]);
        // stride 2: every second step, held in between
        assert_eq!(moved(&[7, 0], 2, false, 1), vec![7, 0]);
        assert_eq!(moved(&[7, 0], 2, false, 2), vec![0, 7]);
    }

    #[test]
    fn a_wave_is_rule_90() {
        // one line becomes its two neighbours, across a word boundary too
        assert_eq!(moved(&[1 << 5], 0, true, 0), vec![(1 << 4) | (1 << 6)]);
        assert_eq!(moved(&[1 << 63, 0], 0, true, 0), vec![1 << 62, 1]);
        // lines 0 and 2: line 1 sees both and cancels (XOR), line 3 sees line 2, line -1 is off
        assert_eq!(moved(&[0b101], 0, true, 0), vec![0b1000]);
    }
}
