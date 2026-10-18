//! A tag-addressed grammar: the genome is a string of tokens, and every string builds a network.
//!
//! - `Channel { tag, width }`: a bit vector of 2^width words.
//! - `Pop { from, to, .. }`: a population of kernels (or a wire) reading the channel whose tag is
//!   nearest `from` and writing the one nearest `to`, with every property inline.
//!
//! Tags name the genome's own channels only: a tag resolves to the nearest declared channel
//! (Hamming distance; ties to the earlier one). The world's side has ports with a tag space of
//! their own: a population with `from_port` reads the input port nearest its `from` tag (all the
//! senses, the eye, ear, touch or affect, or the self channel), and one with `to_port` writes the
//! output port nearest its `to` (the whole act, the voice or the hand). A population with no
//! declared channel to resolve to is left out until one exists. Inserting, deleting, moving or
//! copying any part of the string still makes a network. Wiring changes gradually (a flipped tag bit may move a population to a neighbouring
//! channel, or change nothing), and dead ends (channels nobody reads) are kept as raw material.
//!
//! Two density knobs per population:
//! - **connectivity**: `fan_in` lines per kernel, drawn from its `field` (2^field source words, from
//!   word `at`);
//! - **activity**: `tonic` 0 fires only on a match (responsive); k > 0 also fires spontaneously with
//!   chance 2^-k a step (tonic: babbling is a tonic population on the act).

use petgraph::graph::Graph;
use petgraph::Directed;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::bitvec::BitVector;
use crate::kernel::{KernelClass, KernelGroup, KernelOp, SimpleKernel};
use crate::program::graph::{EdgeKind, EdgeSpec, NodeSpec};
use crate::program::neurocomp::{EdgeInfo, RuntimeNetwork};
use crate::world::{self, World};

/// The widest channel: 2^11 words (131,072 bits). Fields and spreads go as wide.
pub const MAX_WIDTH: u8 = 11;
/// The widest tile: 2^17 bits (a whole widest channel).
pub const MAX_TILE: u8 = 17;

/// How far a mutation moves an offset: one word, or a whole field (so a wide population can step
/// to its neighbour's lines in one move).
fn hop(scale: u8, rng: &mut StdRng) -> u16 {
    if rng.gen_bool(0.5) { 1 } else { 1u16 << scale.min(MAX_WIDTH) }
}

use super::Evolvable;

/// The senses' tag.
pub const SENSES: u8 = 0x00;
/// The act's tag.
pub const ACT: u8 = 0xFF;
/// The self channel's tag: what the network senses of itself (read-only; see
/// `RuntimeNetwork::write_self`).
pub const SELF: u8 = 0x80;
/// The self channel's width: 4 words.
pub const SELF_BITS: usize = 256;

/// Input ports: where a population reading from outside (`from_port`) can read, each with its own
/// tag in a tag space of its own (a population's `from` tag picks the nearest): all the senses,
/// one sense channel, or the self channel.
pub const EYE: u8 = 0x07;
pub const EAR: u8 = 0x38;
pub const TOUCH: u8 = 0xC1;
pub const AFFECT: u8 = 0xE0;
const INPUTS: [(u8, &str); 6] = [(SENSES, "S"), (EYE, "eye"), (EAR, "ear"), (TOUCH, "touch"), (AFFECT, "affect"), (SELF, "M")];
/// Output ports: where a population writing outward (`to_port`) can write: the whole act, the
/// voice or the hand.
pub const VOICE: u8 = 0x1F;
pub const HAND: u8 = 0xF8;
const OUTPUTS: [(u8, &str); 3] = [(ACT, "A"), (VOICE, "voice"), (HAND, "hand")];

/// Where an endpoint lands: a node, its first word and how many words (a port may be part of a
/// node, as the eye is part of the senses).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Place {
    node: usize,
    off: usize,
    len: usize,
    /// the whole node
    whole: bool,
}

/// A population: kernels (or a wire) from one channel to another, with all its properties.
#[derive(Clone, Debug, PartialEq)]
pub struct Pop {
    /// a permanent id: keys its random stream (a copy gets a new one)
    pub id: u32,
    /// where it reads: with `from_port`, the input port nearest `from` (senses, one sense, self);
    /// otherwise the declared channel nearest `from` (tags name the genome's own channels only)
    pub from: u8,
    pub from_port: bool,
    /// where it writes: with `to_port`, the output port nearest `to` (act, voice, hand); otherwise
    /// the declared channel nearest `to`
    pub to: u8,
    pub to_port: bool,
    /// a wire copies what it reads (no kernels)
    pub wire: bool,
    pub op: KernelOp,
    /// frames back it reads, and how many frames it ORs together
    pub lag: u8,
    pub span: u8,
    /// its field: 2^field source words from word `at` (wrapping)
    pub at: u16,
    pub field: u8,
    /// where it writes: from destination word `place` (wrapping), over 2^spread words
    pub place: u16,
    pub spread: u8,
    /// starting kernels; lines each reads (fan-in) and writes (fan-out); threshold in quarters
    pub count: u8,
    pub fan_in: u8,
    pub fan_out: u8,
    pub quarters: u8,
    /// activity: 0 responsive; k > 0 tonic, firing spontaneously with chance 2^-k
    pub tonic: u8,
    /// its kernels do not hold each other down
    pub share: bool,
    /// births where its destination is left uncovered; with a fresh code into an empty one
    pub grow: bool,
    pub new_code: bool,
    /// births repeating the act `repeat_lag` frames back, read on what came before it
    pub repeat: bool,
    pub repeat_lag: u8,
    /// learns only while the channel nearest this tag had lines on (true) or none, `gate_lag` back
    pub gate: Option<(u8, bool)>,
    pub gate_lag: u8,
    /// births: synapses, threshold quarters, fewest and most output lines, cap; fresh code size
    pub birth_syn: u8,
    pub birth_quarters: u8,
    pub out_min: u8,
    pub out_max: u8,
    pub cap: u16,
    pub code_lines: u8,
    /// output learning chance 2^-out_learn; extinction chance 2^-fade
    pub out_learn: u8,
    pub fade: u8,
    /// a wire's rotation, in bits
    pub rotate: u16,
    /// sparse kernels: each visits only the words its lines are in
    pub sparse: bool,
    /// full coverage: it reads its whole source (no `at`/`field`) in tiles of 2^tile bits
    /// every half tile, its kernels are born where input is not yet read, and each writes at the
    /// same relative place in its destination (no `place`/`spread`). A wire copies the whole
    /// source onto the destination.
    pub cover: bool,
    pub tile: u8,
    /// an execution gate: it runs only while the channel nearest this tag had lines on (true)
    /// or none (false) a frame ago; paused, it reads, writes and learns nothing and costs nothing
    pub pause: Option<(u8, bool)>,
    /// kernel replication: 0 off; k > 0, a kernel confirmed 2^k times (net) is copied, to a
    /// neighbouring tile (`replicate_shift`) or in place
    pub replicate: u8,
    pub replicate_shift: bool,
    /// the epigenetic mark: evidence from its lifetimes that it is useful (its predictions came
    /// true), halved each generation and added to; mutation duplicates a marked population more
    /// often and deletes it less often (`set_epigenetics`)
    pub mark: u8,
    /// grown in a lifetime by exploration and kept: a trial connection whose predictions came
    /// true often enough was added to the genome (`inherit`)
    pub explored: bool,
    /// two-mask kernels (`set_zeros`): each expects a value on each of its lines, ones and zeros
    /// alike, so it can match the absence of input (born expecting what its lines held, half on
    /// and half off; starting kernels expect every line off)
    pub expect: bool,
    /// a freak accident (`set_accident`, with exploration): each step it grows its trial
    /// connection with chance 2^-accident even when it is not stuck (0: only when stuck)
    pub accident: u8,
    /// (`set_port_gates`) with `to_port`, it writes an input port (all the senses, the eye, ear,
    /// touch or affect, nearest `to`) instead of an output port: the network adds to what it senses
    /// next step (the world never reads it). The self channel stays read-only.
    pub to_input: bool,
    /// (`set_hint_use`, `set_read_outputs`) with `from_port`, it reads an output port (the act, voice or hand, nearest
    /// `from`) instead of an input port: what the network did a step ago (an efference copy)
    pub from_output: bool,
    /// (`set_port_gates`) what its gate and its pause name: 0 a declared channel, 1 an input port
    /// (a sense or the self channel), 2 an output port (the act, voice or hand)
    pub gate_at: u8,
    pub pause_at: u8,
    /// a predictor (kept from a shadow explorer): it writes its prediction into its destination
    /// and learns against this tag a frame ahead (at: 0 a channel, 1 an input port, 2 an output
    /// port), not against what it writes
    pub teacher: Option<(u8, u8)>,
    /// a predictor's kernel was right when at least this many quarters of the lines it wrote came
    /// on (the residue, prediction XOR actual on its lines, under a quarter of them by default)
    pub hit_quarters: u8,
    /// (`set_hint_band`) a shadow grown from it scores a target by its kernels that are partly
    /// right: more than `hint_low` quarters of their lines came on, fewer than `hit_quarters`
    pub hint_low: u8,
    /// (`set_heat_inputs`) a heat input: a tag (with `heat_at`, what it names: 0 a declared
    /// channel, 1 an input port, 2 an output port) and warm. Each step every kernel's threshold
    /// moves by its own threshold times how full that vector was a frame ago: down when warm (to
    /// one line when full), up when cool (to twice). A wire has no threshold, so none for it.
    pub heat: Option<(u8, bool)>,
    pub heat_at: u8,
    /// (`set_inhibit_inputs`) an inhibit input: a tag (with `inhibitor_at`, what it names: 0 a
    /// declared channel, 1 an input port, 2 an output port). Each step the lines that vector had on
    /// a frame ago are taken before any of its kernels fires, so a kernel whose mask lies more
    /// than half on them is held down and learns anti-Hebbian: inhibition from another vector,
    /// only on the patterns it is active on. A wire has no kernels, so none for it.
    pub inhibitor: Option<u8>,
    pub inhibitor_at: u8,
    /// (`set_branches`) a branch: a tag (with `branch_at`, what it names: 0 a declared channel, 2 an
    /// output port; 1 reads as 0) of a second vector its fired kernels also write their outputs
    /// into, at the same place, by OR, as an axon's collateral: an efference copy, a line to an
    /// inhibitory partner, a second target driven by the same decision. A wire has none.
    pub branch: Option<u8>,
    pub branch_at: u8,
    /// (`set_birth_lags`) births into a port of the act copy the move made this many frames ago
    /// (`EdgeSpec::birth_lag`, 0 to 4); 0 the move being made
    pub birth_lag: u8,
    /// (`set_eligibility`) the three-factor output rule (`EdgeSpec::elig`): 0 none, k the steps
    /// after firing a kernel is checked (1 to 4). A wire has none.
    pub elig: u8,
    /// (`set_rule_tables`) the learning rule as a table (`EdgeSpec::rule`): 0 the runtime's own.
    /// A wire has none.
    pub rule: u16,
    /// (`set_triggers`) its kernels' trigger (`EdgeSpec::trigger`, 0 to 9; 4 to 9 integer). A wire
    /// has none.
    pub trigger: u8,
    /// (`set_triggers`) the lane width of an integer trigger, 2^lane bits (3 to 6)
    pub lane: u8,
    /// (`set_overlap_genes`) how much of a kernel's mask taken lines (a better-matched kernel's,
    /// or an inhibit input's) must cover to hold it down, in quarters: more than this many (1 to
    /// 4; 2, more than half, as without the gene; 4: never held down by overlap)
    pub inhibit_quarters: u8,
    /// (pruning, `program::neurocomp::set_prune`) its hold on its destination at the start of each
    /// lifetime (`EdgeSpec::grip`); 16 to start, halved or doubled by mutation (1 to 128)
    pub grip: u8,
    /// (`set_shadow_hints`) what a shadow grown from this population found its input predicts
    /// (a tag, and what it names: 0 a channel, 1 an input port, 2 an output port): an indicator
    /// of shared information, built into nothing; its gate or pause, when they mutate, take it
    /// half the time
    pub hint: Option<(u8, u8)>,
}

impl Pop {
    /// A population with the grammar's starting values.
    pub fn new(id: u32, from: u8, to: u8) -> Self {
        let p = EdgeSpec::plain();
        Self {
            id, from, from_port: from == SENSES || from == SELF, to, to_port: to == ACT, wire: false, op: KernelOp::Or, lag: 1, span: 1, at: 0, field: 0, place: 0, spread: 0,
            count: 0, fan_in: p.start_syn, fan_out: p.start_out, quarters: p.start_quarters, tonic: 0, share: false,
            grow: false, new_code: false, repeat: false, repeat_lag: 1, gate: None, gate_lag: 1, birth_syn: p.birth_syn,
            birth_quarters: p.birth_quarters, out_min: p.out_min, out_max: p.out_max, cap: p.cap, code_lines: p.code_lines,
            out_learn: p.out_learn, fade: p.fade, rotate: 0, sparse: false, cover: false, tile: 6, pause: None, replicate: 0, replicate_shift: false, mark: 0, explored: false, expect: false, accident: 0, to_input: false, from_output: false, gate_at: 0, pause_at: 0, teacher: None, hit_quarters: 3, hint_low: 1, heat: None, heat_at: 0, inhibitor: None, inhibitor_at: 0, branch: None, branch_at: 0, birth_lag: 0, elig: 0, rule: 0, trigger: 0, lane: 6, inhibit_quarters: 2, grip: 16, hint: None,
        }
    }

    pub fn wire(id: u32, from: u8, to: u8) -> Self {
        Self { wire: true, ..Self::new(id, from, to) }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    /// a channel: 2^width words. `hold`: it starts each step with what it held the step before
    /// (only writes that clear take lines away); otherwise it starts each step empty.
    Channel { tag: u8, width: u8, hold: bool },
    Pop(Pop),
    /// (`set_motions`) the declared channel nearest `tag` moves by itself each step, before any
    /// population writes: a conveyor (`conveyor` k > 0: its words move one place along every
    /// 2^(k-1) steps, so word j holds what was written j moves ago: a delay line, slower for larger
    /// k), and/or a travelling wave (`wave`: each line becomes the XOR of its neighbours', cellular
    /// rule 90). The last motion naming a channel wins.
    Motion { tag: u8, conveyor: u8, wave: bool },
}

#[derive(Clone, Debug, PartialEq)]
pub struct TagGenome {
    pub tokens: Vec<Token>,
    pub next_id: u32,
}


/// The predictions a trial connection's kernels must have had come true, over a genome's
/// lifetimes in a generation, for the genome to keep it (`inherit`).
pub const KEEP: u32 = 16;

/// Two-mask kernels in the grammar (`set_zeros`): a population's `expect` is drawn for a new
/// population (one in four) and flipped as one of its properties. Off by default, so earlier
/// runs draw as before.
static ZEROS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_zeros(on: bool) {
    ZEROS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn zeros() -> bool {
    ZEROS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Freak accidents in the grammar (`set_accident`): a new population draws its own `accident`
/// rate (one every 2^12 to 2^18 steps) and mutation moves it one step up or down. With
/// exploration on, a population then grows its trial connection by accident, rarely, even when it
/// is not stuck. Off by default, so earlier runs draw as before.
static ACCIDENT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_accident(on: bool) {
    ACCIDENT.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn accident() -> bool {
    ACCIDENT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Ports for gates, pauses and writes (`set_port_gates`): a population's gate and pause may name
/// an input port (a sense, or the self channel) or an output port (the act, voice or hand) as well
/// as a declared channel, and a population writing outward may write a sense instead. A new
/// population draws its gate's kind (a third each) and writes a sense one time in eight when it
/// writes outward; mutation changes each. Off by default, so earlier runs draw as before.
static PORT_GATES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_port_gates(on: bool) {
    PORT_GATES.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn port_gates() -> bool {
    PORT_GATES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Shadows as indicators (`set_shadow_hints`, with shadow explorers): a shadow starts only by
/// accident, at its population's own rate, not when the population is stuck; and what it predicts
/// well is not built into the genome but noted on its population as a hint, which that
/// population's gate and pause take, half the time, when they mutate. Off by default.
static SHADOW_HINTS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_shadow_hints(on: bool) {
    SHADOW_HINTS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn shadow_hints() -> bool {
    SHADOW_HINTS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Edges built on hints (`set_hint_use`, with hints): a mutation of its own, one in sixteen while
/// any population carries a hint, adds a new population of any kind (its op, kind, gate, pause and
/// the rest drawn as any new population's) on a hinted path: from the hinted population's input
/// to the hinted vector, or from the hinted vector to the hinted population's destination, the
/// way that feeds back three times in four when only one does (`TagGenome::hint_feedback`). A hint
/// makes an edge likelier there; what the edge is, and whether it stays, is evolution's. Off by
/// default.
static HINT_USE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_hint_use(on: bool) {
    HINT_USE.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn hint_use() -> bool {
    HINT_USE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Output ports read by any population (`set_read_outputs`): a new population reading from
/// outside reads an output port (the act, voice or hand, as it was a step ago) one time in four,
/// and a mutation of its own (one of a population's properties) switches a population between
/// reading an input and an output port. Off by default; without it only edges built on hints read
/// an output port.
static READ_OUTPUTS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_read_outputs(on: bool) {
    READ_OUTPUTS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn read_outputs() -> bool {
    READ_OUTPUTS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Hint bands (`set_hint_band`, with shadows as hints): a shadow scores a target by its kernels
/// that are partly right (more than its population's `hint_low` quarters of their lines came on,
/// fewer than `hit_quarters`), less those at or under `hint_low`; kernels fully right count for
/// nothing. A target the source determines (a copy of it, or what the network computes from it)
/// stops scoring once the shadow has learned it; one correlated with the source keeps scoring.
/// `hint_low` is a gene (1 to 3, starting at 1). Off by default.
static HINT_BAND: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_hint_band(on: bool) {
    HINT_BAND.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn hint_band() -> bool {
    HINT_BAND.load(std::sync::atomic::Ordering::Relaxed)
}

/// Writes into the self channel (`set_self_writes`): a population writing an input port may write
/// the self channel like a sense, its writes landing on what the runtime wrote there this step.
/// Off by default (the self channel read-only).
static SELF_WRITES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_self_writes(on: bool) {
    SELF_WRITES.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn self_writes() -> bool {
    SELF_WRITES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Loops closed (`set_close_loops`): a mutation of its own, one in sixteen, walks a population's
/// path forward 1 to 4 populations (ports included) and adds a new population of any kind from
/// where the walk ends back to where it starts: a loop of any size in one step, where two
/// mutations whose tags line up were needed (only self-loops came in one). Off by default.
static CLOSE_LOOPS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_close_loops(on: bool) {
    CLOSE_LOOPS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn close_loops() -> bool {
    CLOSE_LOOPS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Heat inputs (`set_heat_inputs`): a population may carry a heat input, a channel or port whose
/// fullness a frame ago moves every one of its kernels' thresholds each step (down when warm, up
/// when cool, by the threshold times popcount over bits): a population's excitability set by
/// another vector (arousal, attention, a loop damping itself). A new population has one one time
/// in four; two properties of its own change it and what it names, and with hints it takes the
/// hint half the time. Off by default.
static HEAT_INPUTS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_heat_inputs(on: bool) {
    HEAT_INPUTS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn heat_inputs() -> bool {
    HEAT_INPUTS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Inhibit inputs (`set_inhibit_inputs`): a population may carry an inhibit input, a channel or
/// port whose lines a frame ago are taken before any of its kernels fires, so a kernel whose mask
/// lies more than half on them is held down and learns anti-Hebbian: inhibition between
/// populations, on the patterns the other vector is active on (where `share` turns off only the
/// inhibition among a population's own kernels). A new population has one one time in four; two
/// properties change it and what it names, and with hints it takes the hint half the time. Off
/// by default.
static INHIBIT_INPUTS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_inhibit_inputs(on: bool) {
    INHIBIT_INPUTS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn inhibit_inputs() -> bool {
    INHIBIT_INPUTS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Branches (`Pop::branch`): a population's fired kernels may write their outputs into a second
/// vector too (a declared channel or an output port), as an axon's collaterals reach several
/// targets: a copy of what it did (an efference copy) for others to read, a line to an inhibitory
/// partner (another population's inhibit input), two targets driven by one decision. A new
/// population has one one time in four, on a channel or an output port, half each; a property
/// adds, removes or retags it, another turns it between the two. Off by default.
static BRANCHES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_branches(on: bool) {
    BRANCHES.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn branches() -> bool {
    BRANCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Birth lags (`Pop::birth_lag`): a population's births into a port of the act may copy the move
/// made some frames ago instead of the one being made, pairing what is sensed now with what was
/// done before it: one's own voice heard after it is made, so a kernel born on the sound makes the
/// move that makes it. A property steps it (0 to 4). Off by default (always the move being made).
static BIRTH_LAGS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_birth_lags(on: bool) {
    BIRTH_LAGS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn birth_lags() -> bool {
    BIRTH_LAGS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Motions (`Token::Motion`): a channel that moves by itself each step, before any population
/// writes: a conveyor (its words move one place along every 2^(k-1) steps, a delay line whose word
/// j holds what was written j moves ago; conveyors of different strides give delays on several
/// time scales), or a travelling wave (cellular rule 90: each line the XOR of its neighbours').
/// New tokens are motions one time in about twenty. Off by default.
static MOTIONS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_motions(on: bool) {
    MOTIONS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn motions() -> bool {
    MOTIONS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Triggers (`Pop::trigger`, `EdgeSpec::trigger`): how a population's kernels match, still one
/// popcount and a compare: at least their threshold of lines agreeing (the default), at most (enough
/// of their lines silent), a band (too dense a match is no match), or the whole tile (lines on
/// outside the mask count against it). A property draws it again. Off by default.
static TRIGGERS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_triggers(on: bool) {
    TRIGGERS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn triggers() -> bool {
    TRIGGERS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Rule tables (`Pop::rule`, `EdgeSpec::rule`): a population's learning rule as a gene, a table of
/// what each event (firing, held down, eligible and right, eligible and wrong) moves on the input
/// or the output. A new population starts with the runtime's own rules (`RULE_SEED`), so nothing
/// changes until a mutation flips one bit of its table. Off by default.
static RULE_TABLES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_rule_tables(on: bool) {
    RULE_TABLES.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn rule_tables() -> bool {
    RULE_TABLES.load(std::sync::atomic::Ordering::Relaxed)
}

/// The three-factor output rule (`Pop::elig`, `EdgeSpec::elig`): a kernel that fired some steps
/// ago (eligibility: its own count of steps since firing) is checked, while its gate is open (the
/// modulator), against what its destination holds now; right, it counts up; wrong, one of its
/// output lines moves toward what is there and it counts down, and wrong at 0 it dies, freeing its
/// input for a new birth. So a kernel can learn what it writes, and a wrong pairing can die. A
/// property steps its lag (0 off, 1 to 4). Off by default.
static ELIGIBILITY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_eligibility(on: bool) {
    ELIGIBILITY.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn eligibility() -> bool {
    ELIGIBILITY.load(std::sync::atomic::Ordering::Relaxed)
}

/// The overlap rule as a gene (`set_overlap_genes`): each population's `inhibit_quarters` sets
/// how much of a kernel's mask taken lines must cover to hold it down (more than 1 to 4 quarters;
/// 2, more than half, is the rule without the gene), so a population can let kernels on the same
/// spot fire together, or hold them apart harder; a property of its own steps it. Off by default
/// (always more than half).
static OVERLAP_GENES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_overlap_genes(on: bool) {
    OVERLAP_GENES.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn overlap_genes() -> bool {
    OVERLAP_GENES.load(std::sync::atomic::Ordering::Relaxed)
}

/// A new population `q` placed on `h`'s hint: from `h`'s input to the hinted vector (an output
/// port, a sense, which it then writes, or a channel), or from the hinted vector to `h`'s
/// destination (a sense, the self channel, an output port, read as the act a step ago, or a
/// channel; the self channel cannot be written, so there it always goes back).
/// `feedback` is `TagGenome::hint_feedback`'s: when one way feeds back and the other does not, it
/// goes the way that feeds back three times in four, else either way half the time. Only the
/// endpoints are set; everything else is `q`'s own.
fn on_hint(q: &mut Pop, h: &Pop, feedback: [Option<bool>; 2], rng: &mut StdRng) {
    let Some((t, at)) = h.hint else { return };
    let forward = match feedback {
        [Some(_), None] => true,
        [None, Some(_)] => false,
        [None, None] => return,
        [Some(f), Some(b)] if f != b => (rng.gen_range(0..4) != 0) == f,
        _ => rng.gen_bool(0.5),
    };
    if forward {
        q.from = h.from;
        q.from_port = h.from_port;
        q.from_output = h.from_output;
        q.to = t;
        q.to_port = at != 0;
        q.to_input = at == 1;
    } else {
        q.from = t;
        q.from_port = at != 0;
        q.from_output = at == 2;
        q.to = h.to;
        q.to_port = h.to_port;
        q.to_input = h.to_input;
    }
}

/// Pauses drawn as gates are (`set_pause_draws`): a new population runs only while some channel
/// (or, with port gates, a port) had lines on, or none, one time in four; and a pause made by
/// mutation draws what it names (with port gates: a channel, an input or an output port, a third
/// each) in the same step. Without it a pause comes only from one mutation and a port pause needs
/// a second. Off by default, so earlier runs draw as before.
static PAUSE_DRAWS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_pause_draws(on: bool) {
    PAUSE_DRAWS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn pause_draws() -> bool {
    PAUSE_DRAWS.load(std::sync::atomic::Ordering::Relaxed)
}

/// A new population's pause (`set_pause_draws`): one time in four, on a random tag, and with
/// port gates naming a channel, an input or an output port, a third each. Nothing is drawn off.
fn draw_pause(p: &mut Pop, rng: &mut StdRng, draws: bool, ports: bool) {
    if draws && rng.gen_range(0..4) == 0 {
        p.pause = Some((rng.r#gen(), rng.gen_bool(0.5)));
        if ports {
            p.pause_at = rng.gen_range(0..3);
        }
    }
}

/// What the string's channels resolve to.
struct Channels {
    /// (tag, words) per node: the senses first, then the declared channels, the self channel, the
    /// act last
    list: Vec<(u8, usize)>,
    /// whether each holds its lines from step to step
    hold: Vec<bool>,
    /// (motions) each node's conveyor and wave
    motion: Vec<(u8, bool)>,
}

impl Channels {
    fn of(tokens: &[Token]) -> Self {
        let mut list = vec![(SENSES, world::SENSE_BITS / 64)];
        let mut hold = vec![false];
        for t in tokens {
            if let Token::Channel { tag, width, hold: h } = t {
                list.push((*tag, 1usize << (*width).min(MAX_WIDTH)));
                hold.push(*h);
            }
        }
        list.push((SELF, SELF_BITS / 64));
        hold.push(false);
        list.push((ACT, world::ACT_BITS / 64));
        hold.push(false);
        let mut ch = Self { motion: vec![(0, false); list.len()], list, hold };
        if motions() {
            for t in tokens {
                if let Token::Motion { tag, conveyor, wave } = t {
                    if let Some(i) = ch.internal(*tag) {
                        ch.motion[i] = ((*conveyor).min(8), *wave);
                    }
                }
            }
        }
        ch
    }

    /// The self channel's index (just before the act).
    fn self_index(&self) -> usize {
        self.list.len() - 2
    }

    fn act_index(&self) -> usize {
        self.list.len() - 1
    }

    /// The declared channel nearest a tag (ties: the earlier), if any is declared.
    fn internal(&self, tag: u8) -> Option<usize> {
        (1..self.self_index()).min_by_key(|&i| ((self.list[i].0 ^ tag).count_ones(), i))
    }

    fn whole(&self, node: usize) -> Place {
        Place { node, off: 0, len: self.list[node].1, whole: true }
    }

    /// The input port nearest a tag.
    fn input(&self, tag: u8) -> Place {
        let k = (0..INPUTS.len()).min_by_key(|&i| ((INPUTS[i].0 ^ tag).count_ones(), i)).unwrap();
        match k {
            0 => self.whole(0),
            5 => self.whole(self.self_index()),
            c => {
                let ch = world::CH_BITS / 64;
                let sense = [world::EYE, world::EAR, world::TOUCH, world::AFFECT][c - 1];
                Place { node: 0, off: sense * ch, len: ch, whole: false }
            }
        }
    }

    /// The output port nearest a tag.
    fn output(&self, tag: u8) -> Place {
        let k = (0..OUTPUTS.len()).min_by_key(|&i| ((OUTPUTS[i].0 ^ tag).count_ones(), i)).unwrap();
        let a = self.act_index();
        let ch = world::CH_BITS / 64;
        match k {
            0 => self.whole(a),
            1 => Place { node: a, off: world::VOICE * ch, len: ch, whole: false },
            _ => Place { node: a, off: world::HAND * ch, len: ch, whole: false },
        }
    }

    /// Where a population reads, if anywhere.
    fn from(&self, p: &Pop) -> Option<Place> {
        if p.from_port && p.from_output { Some(self.output(p.from)) } else if p.from_port { Some(self.input(p.from)) } else { self.internal(p.from).map(|i| self.whole(i)) }
    }

    /// Where a population writes, if anywhere.
    fn to(&self, p: &Pop) -> Option<Place> {
        if p.to_port && p.to_input { Some(self.writable_input(p.to)) } else if p.to_port { Some(self.output(p.to)) } else { self.internal(p.to).map(|i| self.whole(i)) }
    }

    /// The input port nearest a tag that a population may write: all the senses or one of them
    /// (the self channel is the runtime's, read-only, unless `set_self_writes`).
    fn writable_input(&self, tag: u8) -> Place {
        let ports = if self_writes() { INPUTS.len() } else { INPUTS.len() - 1 };
        let k = (0..ports).min_by_key(|&i| ((INPUTS[i].0 ^ tag).count_ones(), i)).unwrap();
        self.input(INPUTS[k].0)
    }

    /// What a gate or a pause names: a declared channel (`at` 0), an input port (1) or an output
    /// port (2).
    fn gate_place(&self, tag: u8, at: u8) -> Option<Place> {
        match at {
            1 => Some(self.input(tag)),
            2 => Some(self.output(tag)),
            _ => self.internal(tag).map(|i| self.whole(i)),
        }
    }

    /// What names a place: its tag and what the tag names (0 a declared channel, 1 an input port,
    /// 2 an output port).
    fn tag_of(&self, at: Place) -> (u8, u8) {
        let ch = world::CH_BITS / 64;
        if at.node == 0 {
            if at.whole { return (SENSES, 1) }
            return ([EYE, EAR, TOUCH, AFFECT][(at.off / ch).min(3)], 1);
        }
        if at.node == self.self_index() { return (SELF, 1) }
        if at.node == self.act_index() {
            if at.whole { return (ACT, 2) }
            return (if at.off == world::VOICE * ch { VOICE } else { HAND }, 2);
        }
        (self.list[at.node].0, 0)
    }

    /// A short name for a place.
    fn name(&self, at: Place) -> String {
        if at.node == 0 {
            if at.whole { return "S".into() }
            let ch = world::CH_BITS / 64;
            return ["eye", "ear", "touch", "affect"][at.off / ch].into();
        }
        if at.node == self.self_index() { return "M".into() }
        if at.node == self.act_index() {
            if at.whole { return "A".into() }
            return if at.off == world::VOICE * world::CH_BITS / 64 { "voice".into() } else { "hand".into() };
        }
        format!("C{}", at.node)
    }
}

/// The fewest edges from any of `from` to each node (usize::MAX where none reaches).
fn distances(next: &[Vec<usize>], from: &[usize]) -> Vec<usize> {
    let mut d = vec![usize::MAX; next.len()];
    let mut queue = std::collections::VecDeque::new();
    for &s in from {
        d[s] = 0;
        queue.push_back(s);
    }
    while let Some(x) = queue.pop_front() {
        for &y in &next[x] {
            if d[y] == usize::MAX {
                d[y] = d[x] + 1;
                queue.push_back(y);
            }
        }
    }
    d
}

fn sparse(bits: usize, k: usize, rng: &mut StdRng) -> BitVector {
    let all: Vec<usize> = (0..bits).collect();
    BitVector::from_bits(&all.choose_multiple(rng, k.min(bits)).copied().collect::<Vec<_>>(), bits)
}

impl TagGenome {
    fn pops(&self) -> impl Iterator<Item = &Pop> {
        self.tokens.iter().filter_map(|t| if let Token::Pop(p) = t { Some(p) } else { None })
    }

    /// The genome as text, one token a line: each channel with its width, and each population
    /// with the channels its tags resolve to (S the senses, A the act, Cn the n-th channel) and
    /// its properties.
/// The network as a graph (`wiring`): its bit vectors (name, bits, held), its populations (id,
/// from, to, op, wire or kernels) and their modulatory inputs (population id, kind: gate, pause,
/// heat warm or cool, inhibit; the vector's name; whether it acts while that vector had lines on).
    pub fn wiring(&self) -> (Vec<(String, usize, bool)>, Vec<(u32, String, String, String, bool)>, Vec<(u32, String, String, bool)>) {
        let ch = Channels::of(&self.tokens);
        let mut nodes: Vec<(String, usize, bool)> = Vec::new();
        let seen = |name: String, bits: usize, hold: bool, nodes: &mut Vec<(String, usize, bool)>| {
            if !nodes.iter().any(|n| n.0 == name) {
                nodes.push((name, bits, hold));
            }
        };
        for i in 1..ch.self_index() {
            seen(ch.name(ch.whole(i)), ch.list[i].1 * 64, ch.hold[i], &mut nodes);
        }
        let mut edges = Vec::new();
        let mut mods = Vec::new();
        for p in self.pops() {
            let (Some(a), Some(b)) = (ch.from(p), ch.to(p)) else { continue };
            for pl in [a, b] {
                seen(ch.name(pl), pl.len * 64, pl.node > 0 && pl.node < ch.self_index() && ch.hold[pl.node], &mut nodes);
            }
            edges.push((p.id, ch.name(a), ch.name(b), p.op.text(), p.wire));
            let mut m = |kind: &str, t: u8, at: u8, on: bool, nodes: &mut Vec<(String, usize, bool)>| {
                if let Some(g) = ch.gate_place(t, at) {
                    seen(ch.name(g), g.len * 64, false, nodes);
                    mods.push((p.id, kind.to_string(), ch.name(g), on));
                }
            };
            if let Some((t, on)) = p.gate { m("gate", t, p.gate_at, on, &mut nodes) }
            if let Some((t, on)) = p.pause { m("pause", t, p.pause_at, on, &mut nodes) }
            if !p.wire {
                if let Some((t, w)) = p.heat { m(if w { "warm" } else { "cool" }, t, p.heat_at, true, &mut nodes) }
                if let Some(t) = p.inhibitor { m("inhibit", t, p.inhibitor_at, true, &mut nodes) }
                if let Some(t) = p.branch.filter(|_| branches()) {
                    if let Some(g) = ch.gate_place(t, if p.branch_at == 2 { 2 } else { 0 }) {
                        seen(ch.name(g), g.len * 64, false, &mut nodes);
                        edges.push((p.id, ch.name(a), ch.name(g), format!("{} branch", p.op.text()), false));
                    }
                }
            }
        }
        (nodes, edges, mods)
    }

    pub fn describe(&self) -> String {
        let ch = Channels::of(&self.tokens);
        let gname = |t: u8, at: u8| ch.gate_place(t, at).map_or("-".to_string(), |p| ch.name(p));
        let mut out = String::new();
        let mut c = 0;
        for t in &self.tokens {
            match t {
                Token::Channel { tag, width, hold } => {
                    c += 1;
                    out += &format!("C{c} channel {tag:08b}  {} bits{}\n", 64usize << (*width).min(MAX_WIDTH), if *hold { "  holds" } else { "" });
                }
                Token::Motion { tag, conveyor, wave } => {
                    out += &format!("   motion {tag:08b}: conveyor {conveyor}{}\n", if *wave { ", wave" } else { "" });
                }
                Token::Pop(p) => {
                    let (Some(a), Some(b)) = (ch.from(p), ch.to(p)) else {
                        out += "   (a population with no channel to read or write)\n";
                        continue;
                    };
                    out += &format!(
                        "   {} -> {} {} {:?} lag {}{} {}{}{}{}{}{}{}{}\n",
                        ch.name(a), ch.name(b), if p.wire { "wire" } else { "kernels" }, p.op, p.lag,
                        if p.span > 1 { format!(" span {}", p.span) } else { String::new() },
                        if p.cover {
                            if p.wire { "whole".to_string() } else { format!("cover, tiles of {} bits", 1usize << p.tile.clamp(3, MAX_TILE)) }
                        } else {
                            format!("field {}w@{} -> {}w@{}", 1usize << p.field.min(MAX_WIDTH), p.at, 1usize << p.spread.min(MAX_WIDTH), p.place)
                        },
                        if p.wire { String::new() } else { format!(" n{} in{} out{} q{}/4", p.count, p.fan_in, p.fan_out, p.quarters) },
                        if p.tonic > 0 { format!(" tonic 1/{}", 1u32 << p.tonic) } else { String::new() },
                        if p.share { " share" } else { "" },
                        if p.grow { format!(" grow(syn {}, out {}-{}{}{})", p.birth_syn, p.out_min, p.out_max, if p.new_code { ", new" } else { "" }, if p.repeat { format!(", repeat {}", p.repeat_lag) } else { String::new() }) } else { String::new() },
                        p.gate.map_or(String::new(), |(g, on)| format!(" gate {}{}", if on { "" } else { "!" }, gname(g, p.gate_at)))
                            + &p.pause.map_or(String::new(), |(g, on)| format!(" runs while {}{}", if on { "" } else { "!" }, gname(g, p.pause_at)))
                            + &p.heat.map_or(String::new(), |(g, w)| format!(" {} by {}", if w { "warmed" } else { "cooled" }, gname(g, p.heat_at)))
                            + &p.inhibitor.map_or(String::new(), |g| format!(" inhibited by {}", gname(g, p.inhibitor_at))),
                        if p.sparse { " sparse".to_string() } else { String::new() }
                            + &if p.replicate > 0 { format!(" replicate at {} ({})", 1u32 << p.replicate.min(7), if p.replicate_shift { "to a neighbour tile" } else { "in place" }) } else { String::new() },
                        if p.rotate > 0 { format!(" rot {}", p.rotate) } else { String::new() }
                            + &if p.mark > 0 { format!(" mark {}", p.mark) } else { String::new() }
                            + if p.explored { " explored" } else { "" }
                            + if p.expect { " expect" } else { "" }
                            + &if p.accident > 0 { format!(" accident 1/2^{}", p.accident) } else { String::new() }
                            + &p.teacher.map_or(String::new(), |(t, at)| format!(" predicts {}", ch.gate_place(t, at).map_or("-".to_string(), |g| ch.name(g))))
                            + &p.hint.map_or(String::new(), |(t, at)| format!(" hint {}", ch.gate_place(t, at).map_or("-".to_string(), |g| ch.name(g)))),
                    );
                }
            }
        }
        out
    }

    /// Give every population an id of its own: a population whose id an earlier one already has
    /// (two parents of one lineage joined by crossover) takes the next free id.
    pub fn unique_ids(&mut self) {
        let mut seen = std::collections::HashSet::new();
        let mut next = self.next_id.max(self.pops().map(|p| p.id + 1).max().unwrap_or(0));
        for t in self.tokens.iter_mut() {
            if let Token::Pop(p) = t {
                if !seen.insert(p.id) {
                    p.id = next;
                    seen.insert(next);
                    next += 1;
                }
            }
        }
        self.next_id = next;
    }

    /// A population's ends and op as the runtime resolves them (`from->to op`), or why it has none.
    pub fn label(&self, p: &Pop) -> String {
        let ch = Channels::of(&self.tokens);
        match (ch.from(p), ch.to(p)) {
            (Some(a), Some(b)) => format!("{}->{} {}{}", ch.name(a), ch.name(b), p.op.text(), if p.wire { " wire" } else { "" }),
            _ => "(no channel to read or write)".to_string(),
        }
    }

    /// The runtime graph: one node per channel (senses first, act last), one edge per population.
    pub fn graph(&self) -> Graph<NodeSpec, EdgeSpec, Directed> {
        let ch = Channels::of(&self.tokens);
        let n = ch.list.len();
        // history depth: the deepest frame read from each channel
        let mut depth = vec![1usize; n];
        let mut edges: Vec<(usize, usize, EdgeSpec)> = Vec::new();
        for p in self.pops() {
            let (Some(a), Some(b)) = (ch.from(p), ch.to(p)) else { continue };
            let lag = (p.lag as usize).max(1);
            let span = (p.span as usize).max(1);
            depth[a.node] = depth[a.node].max(lag + span - 1);
            // (birth lags) its destination keeps the frames its births copy
            if birth_lags() && p.birth_lag > 0 && !p.wire {
                depth[b.node] = depth[b.node].max(p.birth_lag as usize);
            }
            if p.repeat {
                depth[a.node] = depth[a.node].max(lag + p.repeat_lag as usize + 1);
                depth[b.node] = depth[b.node].max(p.repeat_lag as usize);
            }
            let gate_at = p.gate.and_then(|(t, _)| ch.gate_place(t, p.gate_at));
            let gate = p.gate.zip(gate_at).map(|((_, on), g)| (g.node, on));
            if let Some((g, _)) = gate {
                depth[g] = depth[g].max((p.gate_lag as usize).max(1));
            }
            let pause_at = p.pause.and_then(|(t, _)| ch.gate_place(t, p.pause_at));
            let pause = p.pause.zip(pause_at).map(|((_, on), g)| (g.node, on));
            let heat_at = p.heat.filter(|_| !p.wire).and_then(|(t, _)| ch.gate_place(t, p.heat_at));
            if let Some(h) = heat_at {
                depth[h.node] = depth[h.node].max(1);
            }
            let inhibitor_at = p.inhibitor.filter(|_| !p.wire).and_then(|t| ch.gate_place(t, p.inhibitor_at));
            let branch_at = p.branch.filter(|_| !p.wire && branches()).and_then(|t| ch.gate_place(t, if p.branch_at == 2 { 2 } else { 0 }));
            if let Some(h) = inhibitor_at {
                depth[h.node] = depth[h.node].max(1);
            }
            let window_of = |g: Option<Place>| g.filter(|g| !g.whole).map(|g| (g.off, g.len));
            let field = 1usize << p.field.min(MAX_WIDTH);
            // what it reads: all of its port or channel (cover), or a field placed inside it
            let window = if p.cover {
                if a.whole { None } else { Some((a.off, a.len)) }
            } else {
                let at = p.at as usize % a.len.max(1);
                Some((a.off + at, field.min(a.len - at)))
            };
            // where it writes: inside its port (a part of the act), or its channel
            let place = if p.cover { 0 } else { p.place as usize % b.len.max(1) };
            edges.push((a.node, b.node, EdgeSpec {
                input_read_back: lag,
                op: p.op,
                kind: if p.wire { EdgeKind::Wire } else { EdgeKind::Kernels },
                rotate: p.rotate,
                grow: p.grow,
                inhibit: !p.share,
                gate,
                new_code: p.new_code,
                // births sample across the whole field when it is wider than a word
                wide: field > 1 && !p.cover,
                span,
                window,
                dst_word: b.off + place,
                dst_len: if b.whole { 0 } else { b.len },
                repeat: p.repeat,
                repeat_lag: (p.repeat_lag as usize).max(1),
                start_kernels: p.count,
                start_syn: p.fan_in,
                start_quarters: p.quarters,
                start_out: p.fan_out,
                birth_syn: p.birth_syn,
                birth_quarters: p.birth_quarters,
                out_min: p.out_min,
                out_max: p.out_max,
                cap: p.cap,
                code_lines: p.code_lines,
                out_learn: p.out_learn,
                fade: p.fade,
                gate_lag: (p.gate_lag as usize).max(1),
                salt: p.id,
                out_words: if p.cover { 0 } else { (1usize << p.spread.min(MAX_WIDTH)).min(if b.whole { usize::MAX } else { b.len - place }) as u16 },
                tile: if p.cover && !p.wire { p.tile.clamp(3, MAX_TILE) } else { 0 },
                tonic: p.tonic,
                sparse: p.sparse,
                pause,
                replicate: p.replicate,
                replicate_shift: p.replicate_shift,
                expect: p.expect && !p.wire,
                accident: if p.wire { 0 } else { p.accident },
                gate_window: window_of(gate_at),
                pause_window: window_of(pause_at),
                teacher: p.teacher.and_then(|(t, at)| ch.gate_place(t, at)).filter(|_| !p.wire).map(|g| (g.node, window_of(Some(g)), 0)),
                hit_quarters: p.hit_quarters,
                heat: p.heat.zip(heat_at).map(|((_, warm), h)| (h.node, window_of(Some(h)), warm)),
                inhibitor: inhibitor_at.map(|h| (h.node, window_of(Some(h)))),
                branch: branch_at.map(|h| (h.node, window_of(Some(h)))),
                birth_lag: if birth_lags() && !p.wire { p.birth_lag.min(4) } else { 0 },
                elig: if eligibility() && !p.wire { p.elig.min(4) } else { 0 },
                rule: if rule_tables() && !p.wire { p.rule } else { 0 },
                trigger: if triggers() && !p.wire { p.trigger.min(9) } else { 0 },
                lane: p.lane.clamp(3, 6),
                inhibit_quarters: if overlap_genes() { p.inhibit_quarters.clamp(1, 4) } else { 2 },
                grip: if crate::program::neurocomp::prune() { p.grip.max(1) } else { 0 },
                hint_low: if hint_band() { p.hint_low.min(p.hit_quarters.saturating_sub(1)).max(1) } else { 0 },
                ..EdgeSpec::plain()
            }));
        }
        let mut g: Graph<NodeSpec, EdgeSpec, Directed> = Graph::new();
        let idx: Vec<_> = (0..n)
            .map(|i| {
                let name = if i == 0 { Some("input".to_string()) } else if i == n - 1 { Some("output".to_string()) } else if i == n - 2 { Some("self".to_string()) } else { None };
                // (a motion) it reads its own last frame to move it
                let d = if ch.motion[i] != (0, false) { depth[i].max(1) } else { depth[i] };
                g.add_node(NodeSpec { name, bits: ch.list[i].1 * 64, history_depth: d, noise: 0, hold: ch.hold[i], conveyor: ch.motion[i].0, wave: ch.motion[i].1 })
            })
            .collect();
        for (a, b, spec) in edges {
            g.add_edge(idx[a], idx[b], spec);
        }
        g
    }

    /// Split: the population at token `at` writes into a new channel one tag bit (`bit`) from
    /// its destination, as wide, and a wire carries that channel on to the old destination one
    /// frame later (a stage inserted on the path).
    pub fn split(&mut self, at: usize, bit: u8) {
        let Some(Token::Pop(p)) = self.tokens.get(at) else { return };
        let ch = Channels::of(&self.tokens);
        let Some(b) = ch.to(p) else { return };
        let width = b.len.max(1).trailing_zeros() as u8;
        let (old, old_port) = (p.to, p.to_port);
        let new = old ^ (1u8 << (bit & 7));
        // the carry copies the whole new channel on to the old destination (inside its port)
        let carry = Pop { cover: true, from_port: false, to: old, to_port: old_port, ..Pop::wire(self.next_id, new, old) };
        self.next_id += 1;
        if let Token::Pop(p) = &mut self.tokens[at] {
            p.to = new;
            p.to_port = false;
        }
        self.tokens.insert(at, Token::Channel { tag: new, width, hold: false });
        self.tokens.insert(at + 2, Token::Pop(carry));
    }

    /// A circuit: the declared channel at `channel` (its index among the channels, 1-based as
    /// `Channels` numbers them) and every population that reads it, writes it, or is gated or
    /// paused on it. Returns token indices (the channel's first).
    pub fn circuit(&self, channel: usize) -> Vec<usize> {
        let ch = Channels::of(&self.tokens);
        let mut c = 0;
        let mut out = Vec::new();
        for (i, t) in self.tokens.iter().enumerate() {
            if let Token::Channel { .. } = t {
                c += 1;
                if c == channel {
                    out.insert(0, i);
                }
            }
        }
        if out.is_empty() {
            return out;
        }
        for (i, t) in self.tokens.iter().enumerate() {
            if let Token::Pop(p) = t {
                let here = |t: u8| ch.internal(t) == Some(channel);
                let touches = (!p.from_port && here(p.from))
                    || (!p.to_port && here(p.to))
                    || p.gate.is_some_and(|(g, _)| here(g))
                    || p.pause.is_some_and(|(g, _)| here(g))
                    || (p.heat_at == 0 && p.heat.is_some_and(|(g, _)| here(g)))
                    || (p.inhibitor_at == 0 && p.inhibitor.is_some_and(here))
                    || (p.branch_at != 2 && p.branch.is_some_and(here));
                if touches {
                    out.push(i);
                }
            }
        }
        out
    }

    /// The tokens of a circuit (`circuit(channel)`) made private: the channel gets the fresh tag
    /// `fresh`, every reference that resolved to it now names `fresh` exactly, and populations
    /// get new ids from `next_id`. What the circuit reads and writes outside stays as it was.
    fn private_circuit(&self, channel: usize, fresh: u8, next_id: &mut u32) -> Vec<Token> {
        self.private_tokens(&self.circuit(channel), channel, fresh, next_id)
    }

    /// `private_circuit` on some of the circuit's tokens (`at`: indices, the channel's first).
    fn private_tokens(&self, at: &[usize], channel: usize, fresh: u8, next_id: &mut u32) -> Vec<Token> {
        let ch = Channels::of(&self.tokens);
        at.iter()
            .copied()
            .map(|i| match &self.tokens[i] {
                Token::Channel { width, hold, .. } => Token::Channel { tag: fresh, width: *width, hold: *hold },
                Token::Motion { conveyor, wave, .. } => Token::Motion { tag: fresh, conveyor: *conveyor, wave: *wave },
                Token::Pop(p) => {
                    let mut q = p.clone();
                    let here = |t: u8| ch.internal(t) == Some(channel);
                    if !p.from_port && here(p.from) { q.from = fresh }
                    if !p.to_port && here(p.to) { q.to = fresh }
                    if let Some((g, on)) = p.gate { if here(g) { q.gate = Some((fresh, on)) } }
                    if let Some((g, on)) = p.pause { if here(g) { q.pause = Some((fresh, on)) } }
                    if let (Some((g, w)), 0) = (p.heat, p.heat_at) { if here(g) { q.heat = Some((fresh, w)) } }
                    if let (Some(g), 0) = (p.inhibitor, p.inhibitor_at) { if here(g) { q.inhibitor = Some(fresh) } }
                    if let Some(g) = p.branch.filter(|_| p.branch_at != 2) { if here(g) { q.branch = Some(fresh) } }
                    q.id = *next_id;
                    *next_id += 1;
                    Token::Pop(q)
                }
            })
            .collect()
    }

    /// A tag no declared channel (nor the senses, act or self channel) has.
    fn fresh_tag(&self, rng: &mut StdRng) -> u8 {
        let ch = Channels::of(&self.tokens);
        loop {
            let t: u8 = rng.r#gen();
            if ch.list.iter().all(|c| c.0 != t) {
                return t;
            }
        }
    }

    /// Circuit copy (gene duplication): the circuit around a declared channel, copied whole with
    /// a private channel, appended. With `rewire`, the copy's populations that read from outside
    /// read another channel instead (the same machinery on other data).
    pub fn copy_circuit(&mut self, channel: usize, rewire: Option<(u8, bool)>, rng: &mut StdRng) {
        let fresh = self.fresh_tag(rng);
        let mut next = self.next_id;
        let mut copy = self.private_circuit(channel, fresh, &mut next);
        if copy.is_empty() {
            return;
        }
        if let Some((src, port)) = rewire {
            for t in &mut copy {
                if let Token::Pop(p) = t {
                    if p.to == fresh && !p.to_port && (p.from != fresh || p.from_port) {
                        p.from = src;
                        p.from_port = port;
                    }
                }
            }
        }
        self.next_id = next;
        self.tokens.extend(copy);
    }

    /// Plasmid transfer: the circuit around one of `donor`'s declared channels, made private,
    /// appended to this genome. What it reads and writes outside resolves among this genome's
    /// channels (the senses and the act always match).
    pub fn take_plasmid(&mut self, donor: &TagGenome, channel: usize, rng: &mut StdRng) {
        let fresh = self.fresh_tag(rng);
        let mut next = self.next_id.max(donor.next_id);
        let plasmid = donor.private_circuit(channel, fresh, &mut next);
        if plasmid.is_empty() {
            return;
        }
        self.next_id = next;
        self.tokens.extend(plasmid);
    }

    /// The ports its populations read and write (each named once): "eye", "hand", ...
    pub fn ports(&self) -> (Vec<String>, Vec<String>) {
        let ch = Channels::of(&self.tokens);
        let (mut r, mut w) = (Vec::new(), Vec::new());
        for p in self.pops() {
            if let (Some(a), Some(b)) = (ch.from(p), ch.to(p)) {
                if p.from_port && !r.contains(&ch.name(a)) {
                    r.push(ch.name(a));
                }
                if p.to_port && !w.contains(&ch.name(b)) {
                    w.push(ch.name(b));
                }
            }
        }
        (r, w)
    }

    /// The two ways an edge may run on `h`'s hint (from `h`'s input to the hinted vector, then from
    /// the hinted vector to `h`'s destination): None where it cannot (the self channel is not
    /// written), else whether it feeds back: it closes a loop (its
    /// destination already reaches its source, on the channel graph, where every channel reaches
    /// the self channel) or runs from a higher level to a lower one (a level: the fewest
    /// populations from the senses; one nothing reaches is the highest).
    pub fn hint_feedback(&self, h: &Pop) -> [Option<bool>; 2] {
        let Some((t, at)) = h.hint else { return [None, None] };
        let ch = Channels::of(&self.tokens);
        let mut next = self.channel_next(&ch);
        // the runtime writes the self channel from all the network's work: every node reaches it
        let me = ch.self_index();
        for (x, out) in next.iter_mut().enumerate() {
            if x != me {
                out.push(me);
            }
        }
        let dist = |from: &[usize]| distances(&next, from);
        let level = dist(&[0]);
        let feeds_back = |x: Option<usize>, y: Option<usize>| match (x, y) {
            (Some(x), Some(y)) => dist(&[y])[x] != usize::MAX || level[x] > level[y],
            _ => false,
        };
        let hinted = if at == 1 { Some(ch.input(t).node) } else { ch.gate_place(t, at).map(|p| p.node) };
        let (src, dst) = (ch.from(h).map(|p| p.node), ch.to(h).map(|p| p.node));
        let writable = self_writes() || !(at == 1 && t == SELF);
        [writable.then(|| feeds_back(src, hinted)), Some(feeds_back(hinted, dst))]
    }

    /// For each node of the channel graph, the nodes its populations write.
    fn channel_next(&self, ch: &Channels) -> Vec<Vec<usize>> {
        let mut next = vec![Vec::new(); ch.list.len()];
        for p in self.pops() {
            if let (Some(a), Some(b)) = (ch.from(p), ch.to(p)) {
                next[a.node].push(b.node);
            }
        }
        next
    }

    /// How recursive it is, on its channel graph (channels the nodes, populations the edges):
    /// (populations on a loop, of them self-loops, the largest loop's channels, populations). A
    /// population is on a loop if it writes the channel it reads, or both its channels lie on one
    /// cycle; the senses and the act are never on one (nothing writes the senses, nothing reads
    /// the act).
    pub fn loops(&self) -> (usize, usize, usize, usize) {
        let g = self.graph();
        let scc = petgraph::algo::tarjan_scc(&g);
        let mut comp = vec![0usize; g.node_count()];
        let mut size = vec![0usize; scc.len()];
        for (c, nodes) in scc.iter().enumerate() {
            size[c] = nodes.len();
            for n in nodes {
                comp[n.index()] = c;
            }
        }
        let (mut on, mut selfs, mut largest) = (0, 0, 0);
        for e in g.edge_indices() {
            let (a, b) = g.edge_endpoints(e).unwrap();
            if a == b {
                on += 1;
                selfs += 1;
                largest = largest.max(1);
            } else if comp[a.index()] == comp[b.index()] {
                on += 1;
                largest = largest.max(size[comp[a.index()]]);
            }
        }
        (on, selfs, largest, g.edge_count())
    }

    /// Its blind control: every population that reads the senses (or one sense) reads the self
    /// channel instead, so the same circuit runs with nothing from the world.
    pub fn blind(&self) -> TagGenome {
        let ch = Channels::of(&self.tokens);
        let mut g = self.clone();
        for t in &mut g.tokens {
            if let Token::Pop(p) = t {
                if p.from_port && !p.from_output && ch.input(p.from).node == 0 {
                    p.from = SELF;
                }
            }
        }
        g
    }

    /// A plasmid around one of `donor`'s populations (the `k`-th): the circuit of a declared
    /// channel it reads, writes or is gated by, made private; or, when it touches none (it
    /// reads and writes ports), the population alone, with a new id. Appended to this genome;
    /// returns the plasmid's tokens as moved (empty if the donor has no population).
    pub fn take_population_plasmid(&mut self, donor: &TagGenome, k: usize, rng: &mut StdRng) -> Vec<Token> {
        self.take_population_plasmid_upto(donor, k, 0, rng)
    }

    /// `take_population_plasmid`, with a circuit of more than `max` populations (0: no limit)
    /// cut to the chosen population and its nearest neighbours in the circuit (genome order),
    /// `max` of them with the channel: a small mobile unit, not a busy channel's whole tangle.
    pub fn take_population_plasmid_upto(&mut self, donor: &TagGenome, k: usize, max: usize, rng: &mut StdRng) -> Vec<Token> {
        let pops: Vec<usize> = (0..donor.tokens.len()).filter(|&i| matches!(donor.tokens[i], Token::Pop(_))).collect();
        if pops.is_empty() {
            return vec![];
        }
        let chosen = pops[k % pops.len()];
        let Token::Pop(p) = &donor.tokens[chosen] else { unreachable!() };
        let ch = Channels::of(&donor.tokens);
        let touched = [(!p.from_port).then_some(p.from), (!p.to_port).then_some(p.to), p.gate.map(|g| g.0), p.pause.map(|g| g.0), p.heat.filter(|_| p.heat_at == 0).map(|g| g.0), p.inhibitor.filter(|_| p.inhibitor_at == 0), p.branch.filter(|_| p.branch_at != 2)]
            .into_iter()
            .flatten()
            .find_map(|t| ch.internal(t));
        let before = self.tokens.len();
        match touched {
            Some(c) if max > 0 && donor.circuit(c).len() > max + 1 => {
                let circ = donor.circuit(c);
                let members = &circ[1..];
                let at = members.iter().position(|&i| i == chosen).unwrap_or(0);
                let lo = at.saturating_sub((max - 1) / 2).min(members.len() - max);
                let keep: Vec<usize> = std::iter::once(circ[0]).chain(members[lo..lo + max].iter().copied()).collect();
                let fresh = self.fresh_tag(rng);
                let mut next = self.next_id.max(donor.next_id);
                let moved = donor.private_tokens(&keep, c, fresh, &mut next);
                self.next_id = next;
                self.tokens.extend(moved);
            }
            Some(c) => self.take_plasmid(donor, c, rng),
            None => {
                let mut q = p.clone();
                q.id = self.next_id.max(donor.next_id);
                self.next_id = q.id + 1;
                self.tokens.push(Token::Pop(q));
            }
        }
        self.tokens[before..].to_vec()
    }

    /// The genome as text, one token a line (`next_id` first), read back by `from_text`: a
    /// checkpoint can be saved and an evolution resumed from it.
    pub fn to_text(&self) -> String {
        let b = |x: bool| x as u8;
        let opt = |o: Option<(u8, bool)>| o.map_or("-".to_string(), |(t, on)| format!("{t}:{}", b(on)));
        let mut out = format!("genome {}\n", self.next_id);
        for t in &self.tokens {
            match t {
                Token::Channel { tag, width, hold } => out += &format!("channel {tag} {width} {}\n", b(*hold)),
                Token::Motion { tag, conveyor, wave } => out += &format!("motion {tag} {conveyor} {}\n", b(*wave)),
                Token::Pop(p) => {
                    out += &format!(
                        "pop id={} from={} from_port={} to={} to_port={} wire={} op={} lag={} span={} at={} field={} place={} spread={} count={} fan_in={} fan_out={} quarters={} tonic={} share={} grow={} new_code={} repeat={} repeat_lag={} gate={} gate_lag={} birth_syn={} birth_quarters={} out_min={} out_max={} cap={} code_lines={} out_learn={} fade={} rotate={} sparse={} cover={} tile={} pause={} replicate={} replicate_shift={} mark={} explored={}\n",
                        p.id, p.from, b(p.from_port), p.to, b(p.to_port), b(p.wire), p.op.text(), p.lag, p.span, p.at, p.field, p.place, p.spread,
                        p.count, p.fan_in, p.fan_out, p.quarters, p.tonic, b(p.share), b(p.grow), b(p.new_code), b(p.repeat), p.repeat_lag,
                        opt(p.gate), p.gate_lag, p.birth_syn, p.birth_quarters, p.out_min, p.out_max, p.cap, p.code_lines, p.out_learn, p.fade,
                        p.rotate, b(p.sparse), b(p.cover), p.tile, opt(p.pause), p.replicate, b(p.replicate_shift), p.mark, b(p.explored)
                    );
                    // (written only when set, so genomes without it read and write as before)
                    if p.expect {
                        out.insert_str(out.len() - 1, " expect=1");
                    }
                    if p.accident > 0 {
                        out.insert_str(out.len() - 1, &format!(" accident={}", p.accident));
                    }
                    if p.to_input {
                        out.insert_str(out.len() - 1, " to_input=1");
                    }
                    if p.from_output {
                        out.insert_str(out.len() - 1, " from_output=1");
                    }
                    if p.hint_low != 1 {
                        out.insert_str(out.len() - 1, &format!(" hint_low={}", p.hint_low));
                    }
                    if let Some((t, w)) = p.heat {
                        out.insert_str(out.len() - 1, &format!(" heat={t}:{} heat_at={}", w as u8, p.heat_at));
                    }
                    if let Some(t) = p.branch {
                        out.insert_str(out.len() - 1, &format!(" branch={t} branch_at={}", p.branch_at));
                    }
                    if p.elig > 0 {
                        out.insert_str(out.len() - 1, &format!(" elig={}", p.elig));
                    }
                    if p.rule != 0 {
                        out.insert_str(out.len() - 1, &format!(" rule={:x}", p.rule));
                    }
                    if p.trigger != 0 {
                        out.insert_str(out.len() - 1, &format!(" trigger={}", p.trigger));
                    }
                    if p.lane != 6 {
                        out.insert_str(out.len() - 1, &format!(" lane={}", p.lane));
                    }
                    if p.birth_lag > 0 {
                        out.insert_str(out.len() - 1, &format!(" birth_lag={}", p.birth_lag));
                    }
                    if let Some(t) = p.inhibitor {
                        out.insert_str(out.len() - 1, &format!(" inhibitor={t} inhibitor_at={}", p.inhibitor_at));
                    }
                    if p.grip != 16 {
                        out.insert_str(out.len() - 1, &format!(" grip={}", p.grip));
                    }
                    if p.inhibit_quarters != 2 {
                        out.insert_str(out.len() - 1, &format!(" inhibit_quarters={}", p.inhibit_quarters));
                    }
                    if p.gate_at > 0 {
                        out.insert_str(out.len() - 1, &format!(" gate_at={}", p.gate_at));
                    }
                    if p.pause_at > 0 {
                        out.insert_str(out.len() - 1, &format!(" pause_at={}", p.pause_at));
                    }
                    if let Some((t, at)) = p.teacher {
                        out.insert_str(out.len() - 1, &format!(" teacher={t}:{at} hit_quarters={}", p.hit_quarters));
                    }
                    if let Some((t, at)) = p.hint {
                        out.insert_str(out.len() - 1, &format!(" hint={t}:{at}"));
                    }
                }
            }
        }
        out
    }

    /// The genomes in `text` (as `to_text` wrote them, one after another).
    pub fn from_text(text: &str) -> Result<Vec<TagGenome>, String> {
        let mut out: Vec<TagGenome> = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            let bad = |what: &str| format!("line {}: {what}: {line}", n + 1);
            let mut w = line.split_whitespace();
            match w.next() {
                None => continue,
                Some("genome") => out.push(TagGenome { tokens: vec![], next_id: w.next().and_then(|x| x.parse().ok()).ok_or_else(|| bad("next id"))? }),
                Some("channel") => {
                    let v: Vec<u8> = w.map(|x| x.parse().map_err(|_| bad("channel"))).collect::<Result<_, _>>()?;
                    let [tag, width, hold] = v[..] else { return Err(bad("channel")) };
                    out.last_mut().ok_or_else(|| bad("before any genome"))?.tokens.push(Token::Channel { tag, width, hold: hold == 1 });
                }
                Some("motion") => {
                    let v: Vec<u8> = w.map(|x| x.parse().map_err(|_| bad("motion"))).collect::<Result<_, _>>()?;
                    let [tag, conveyor, wave] = v[..] else { return Err(bad("motion")) };
                    out.last_mut().ok_or_else(|| bad("before any genome"))?.tokens.push(Token::Motion { tag, conveyor, wave: wave == 1 });
                }
                Some("pop") => {
                    let kv: std::collections::HashMap<&str, &str> = w.filter_map(|x| x.split_once('=')).collect();
                    let get = |k: &str| kv.get(k).copied().ok_or_else(|| bad(k));
                    macro_rules! num { ($k:expr) => { get($k)?.parse().map_err(|_| bad($k))? }; }
                    let flag = |k: &str| -> Result<bool, String> { Ok(get(k)? == "1") };
                    let opt = |k: &str| -> Result<Option<(u8, bool)>, String> {
                        let v = get(k)?;
                        if v == "-" { return Ok(None) }
                        let (t, on) = v.split_once(':').ok_or_else(|| bad(k))?;
                        Ok(Some((t.parse().map_err(|_| bad(k))?, on == "1")))
                    };
                    let p = Pop {
                        id: num!("id"), from: num!("from"), from_port: flag("from_port")?, to: num!("to"), to_port: flag("to_port")?,
                        wire: flag("wire")?, op: KernelOp::parse(get("op")?).ok_or_else(|| bad("op"))?, lag: num!("lag"), span: num!("span"),
                        at: num!("at"), field: num!("field"), place: num!("place"), spread: num!("spread"), count: num!("count"),
                        fan_in: num!("fan_in"), fan_out: num!("fan_out"), quarters: num!("quarters"), tonic: num!("tonic"),
                        share: flag("share")?, grow: flag("grow")?, new_code: flag("new_code")?, repeat: flag("repeat")?,
                        repeat_lag: num!("repeat_lag"), gate: opt("gate")?, gate_lag: num!("gate_lag"), birth_syn: num!("birth_syn"),
                        birth_quarters: num!("birth_quarters"), out_min: num!("out_min"), out_max: num!("out_max"), cap: num!("cap"),
                        code_lines: num!("code_lines"), out_learn: num!("out_learn"), fade: num!("fade"), rotate: num!("rotate"),
                        sparse: flag("sparse")?, cover: flag("cover")?, tile: num!("tile"), pause: opt("pause")?,
                        replicate: num!("replicate"), replicate_shift: flag("replicate_shift")?,
                        // (absent in genomes written before marks)
                        mark: kv.get("mark").and_then(|x| x.parse().ok()).unwrap_or(0),
                        explored: kv.get("explored") == Some(&"1"),
                        expect: kv.get("expect") == Some(&"1"),
                        accident: kv.get("accident").and_then(|x| x.parse().ok()).unwrap_or(0),
                        to_input: kv.get("to_input") == Some(&"1"),
                        from_output: kv.get("from_output") == Some(&"1"),
                        hint_low: kv.get("hint_low").and_then(|x| x.parse().ok()).unwrap_or(1),
                        heat: kv.get("heat").and_then(|x| x.split_once(':')).and_then(|(a, b)| Some((a.parse().ok()?, b == "1"))),
                        heat_at: kv.get("heat_at").and_then(|x| x.parse().ok()).unwrap_or(0),
                        inhibitor: kv.get("inhibitor").and_then(|x| x.parse().ok()),
                        inhibitor_at: kv.get("inhibitor_at").and_then(|x| x.parse().ok()).unwrap_or(0),
                        branch: kv.get("branch").and_then(|x| x.parse().ok()),
                        branch_at: kv.get("branch_at").and_then(|x| x.parse().ok()).unwrap_or(0),
                        birth_lag: kv.get("birth_lag").and_then(|x| x.parse().ok()).unwrap_or(0),
                        elig: kv.get("elig").and_then(|x| x.parse().ok()).unwrap_or(0),
                        rule: kv.get("rule").and_then(|x| u16::from_str_radix(x, 16).ok()).unwrap_or(0),
                        trigger: kv.get("trigger").and_then(|x| x.parse().ok()).unwrap_or(0),
                        lane: kv.get("lane").and_then(|x| x.parse().ok()).unwrap_or(6),
                        inhibit_quarters: kv.get("inhibit_quarters").and_then(|x| x.parse().ok()).unwrap_or(2),
                        grip: kv.get("grip").and_then(|x| x.parse().ok()).unwrap_or(16),
                        gate_at: kv.get("gate_at").and_then(|x| x.parse().ok()).unwrap_or(0),
                        pause_at: kv.get("pause_at").and_then(|x| x.parse().ok()).unwrap_or(0),
                        teacher: kv.get("teacher").and_then(|x| x.split_once(':')).and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?))),
                        hit_quarters: kv.get("hit_quarters").and_then(|x| x.parse().ok()).unwrap_or(3),
                        hint: kv.get("hint").and_then(|x| x.split_once(':')).and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?))),
                    };
                    out.last_mut().ok_or_else(|| bad("before any genome"))?.tokens.push(Token::Pop(p));
                }
                Some(_) => return Err(bad("unknown line")),
            }
        }
        Ok(out)
    }

    /// How many populations it has.
    pub fn populations(&self) -> usize {
        self.pops().count()
    }

    /// How many populations carry a branch (`Pop::branch`, with `set_branches`; wires have none).
    pub fn pops_branched(&self) -> usize {
        self.pops().filter(|p| p.branch.is_some() && !p.wire).count()
    }

    /// The wiring length (`evolve::set_wire_cost`): for each population, the Hamming distance
    /// between the tags of what its two ends resolve to: the nearest declared channel, or for an end
    /// on a port the nearest port's own tag (the senses at 0x00 and the act at 0xFF, the eye, ear,
    /// voice, hand ... between them), summed. Ports and channels share one tag space for this, so
    /// the ports sit at fixed places in it and a channel's tag says where it lies among them; a
    /// population from a channel into itself adds nothing.
    pub fn wire_length(&self) -> usize {
        let tags: Vec<u8> = self.tokens.iter().filter_map(|t| if let Token::Channel { tag, .. } = t { Some(*tag) } else { None }).collect();
        let near = |set: &[u8], x: u8| set.iter().copied().min_by_key(|&t| (t ^ x).count_ones()).unwrap_or(x);
        let inputs: Vec<u8> = INPUTS.iter().map(|x| x.0).collect();
        let outputs: Vec<u8> = OUTPUTS.iter().map(|x| x.0).collect();
        self.pops()
            .map(|p| {
                let a = if p.from_port { if p.from_output { near(&outputs, p.from) } else { near(&inputs, p.from) } } else { near(&tags, p.from) };
                let b = if p.to_port { if p.to_input { near(&inputs, p.to) } else { near(&outputs, p.to) } } else { near(&tags, p.to) };
                // (a branch) its collateral from the same source end
                let c = p.branch.filter(|_| branches() && !p.wire).map_or(0, |t| (a ^ if p.branch_at == 2 { near(&outputs, t) } else { near(&tags, t) }).count_ones() as usize);
                (a ^ b).count_ones() as usize + c
            })
            .sum()
    }

    /// How many channels are declared.
    pub fn channels(&self) -> usize {
        self.tokens.iter().filter(|t| matches!(t, Token::Channel { .. })).count()
    }

    /// Self-loop: the population at token `at` writes back into the channel it reads (a carry,
    /// a clock, a counter, held state). Not on the senses, which nothing writes.
    pub fn self_loop(&mut self, at: usize) {
        let ch = Channels::of(&self.tokens);
        if let Some(Token::Pop(p)) = self.tokens.get_mut(at) {
            if p.from_port {
                return;
            }
            if let Some(i) = ch.internal(p.from) {
                p.to = ch.list[i].0;
                p.to_port = false;
            }
        }
    }

    /// Closing a loop (`set_close_loops`): from the population at token `at`, walk forward 0 to 3
    /// more populations (each reading where the last wrote, ports included: a population reading
    /// the act, or one reading a sense another writes), then add a new population of any kind
    /// (`q`, its endpoints set here) from where the walk ended back to where it started, after the
    /// first. A loop of 1 to 4 populations, through channels, the senses or the act; nothing if
    /// the start cannot be written (the self channel, without self writes).
    pub fn close_loop(&mut self, at: usize, steps: usize, mut q: Pop, rng: &mut StdRng) -> bool {
        let ch = Channels::of(&self.tokens);
        let Some(Token::Pop(p)) = self.tokens.get(at) else { return false };
        let (Some(start), Some(mut end)) = (ch.from(p), ch.to(p)) else { return false };
        for _ in 0..steps {
            let next: Vec<Place> = self.pops().filter(|r| ch.from(r).is_some_and(|f| f.node == end.node)).filter_map(|r| ch.to(r)).collect();
            let Some(&e) = next.choose(rng) else { break };
            end = e;
        }
        let (st, sat) = ch.tag_of(start);
        if sat == 1 && st == SELF && !self_writes() {
            return false;
        }
        let (et, eat) = ch.tag_of(end);
        q.from = et;
        q.from_port = eat != 0;
        q.from_output = eat == 2;
        q.to = st;
        q.to_port = sat != 0;
        q.to_input = sat == 1;
        self.tokens.insert(at + 1, Token::Pop(q));
        true
    }

    fn random_pop(&mut self, rng: &mut StdRng) -> Pop {
        let id = self.next_id;
        self.next_id += 1;
        // a new population is born active: some kernels, a random field and fan-in, sometimes
        // tonic, sometimes growing (an inert gene would need several more mutations to matter)
        let mut p = Pop::new(id, rng.r#gen(), rng.r#gen());
        // reading from outside (a sense or the self channel) or a channel; writing outward (the
        // act, voice or hand) or a channel
        p.from_port = rng.gen_bool(0.4);
        p.to_port = rng.gen_bool(0.4);
        p.wire = rng.gen_bool(0.25);
        p.sparse = rng.gen_bool(0.5);
        // full coverage, at a random scale from a byte to two words
        p.cover = true;
        p.tile = rng.gen_range(3..=7);
        // sometimes replicating its confirmed kernels
        if rng.gen_range(0..3) == 0 {
            p.replicate = rng.gen_range(2..=5);
            p.replicate_shift = rng.gen_bool(0.5);
        }
        // any of the ops, evenly (OR is one of twelve)
        p.op = KernelOp::random(rng);
        p.count = rng.gen_range(1..=8);
        p.field = rng.gen_range(0..4);
        p.at = rng.r#gen();
        p.fan_in = 1 << rng.gen_range(2..5);
        p.tonic = if rng.gen_range(0..3) == 0 { rng.gen_range(1..5) } else { 0 };
        p.grow = rng.gen_bool(0.3);
        p.repeat = p.grow && rng.gen_bool(0.3);
        p.repeat_lag = rng.gen_range(1..6);
        if rng.gen_range(0..3) == 0 {
            p.gate = Some((rng.r#gen(), rng.gen_bool(0.5)));
        }
        // (with `set_zeros`) sometimes expecting zeros as well as ones
        if zeros() {
            p.expect = rng.gen_range(0..4) == 0;
        }
        // (with `set_accident`) an accident rate of its own, one every 2^12 to 2^18 steps
        if accident() {
            p.accident = rng.gen_range(12..=18);
        }
        // (with `set_port_gates`) its gate on a channel, a sense or an output port, a third each;
        // and now and then writing into a sense instead of an output port
        if port_gates() {
            p.gate_at = rng.gen_range(0..3);
            p.to_input = p.to_port && rng.gen_range(0..8) == 0;
        }
        // (with `set_read_outputs`) reading from outside, it reads an output port one time in four
        if read_outputs() {
            p.from_output = p.from_port && rng.gen_range(0..4) == 0;
        }
        // (with `set_inhibit_inputs`) an inhibit input one time in four, on a channel or (with
        // port gates) a port, a third each
        // (with `set_rule_tables`) it learns by the runtime's own rules, as a table to vary
        if rule_tables() {
            p.rule = crate::program::graph::RULE_SEED;
        }
        // (with `set_eligibility`) the three-factor output rule one time in four, checked 1 or 2
        // steps after firing
        if eligibility() && rng.gen_range(0..4) == 0 {
            p.elig = rng.gen_range(1..=2);
        }
        // (with `set_branches`) a branch one time in four, on a channel or an output port, half each
        if branches() && rng.gen_range(0..4) == 0 {
            p.branch = Some(rng.r#gen());
            p.branch_at = if rng.gen_bool(0.5) { 2 } else { 0 };
        }
        if inhibit_inputs() && rng.gen_range(0..4) == 0 {
            p.inhibitor = Some(rng.r#gen());
            if port_gates() {
                p.inhibitor_at = rng.gen_range(0..3);
            }
        }
        // (with `set_heat_inputs`) a heat input one time in four, on a channel or (with port
        // gates) a port, a third each
        if heat_inputs() && rng.gen_range(0..4) == 0 {
            p.heat = Some((rng.r#gen(), rng.gen_bool(0.5)));
            if port_gates() {
                p.heat_at = rng.gen_range(0..3);
            }
        }
        // (with `set_pause_draws`) sometimes paused on a channel or port from the start
        draw_pause(&mut p, rng, pause_draws(), port_gates());
        p
    }

    /// The act path (experiment 06 starting point): a tonic population babbling into the act, and
    /// a growing population repeating acts a stroke followed, on what the eye saw before them.
    pub fn act_seed() -> Self {
        let ch = (world::CH_BITS / 64) as u16;
        let (eye, aff) = (0x0F, 0xF0);
        let mut t = vec![Token::Channel { tag: eye, width: 3, hold: false }, Token::Channel { tag: aff, width: 3, hold: false }];
        let mut w = Pop::wire(1, SENSES, eye);
        w.at = world::EYE as u16 * ch;
        w.field = 3;
        t.push(Token::Pop(w));
        let mut w = Pop::wire(2, SENSES, aff);
        w.at = world::AFFECT as u16 * ch;
        w.field = 3;
        t.push(Token::Pop(w));
        let mut b = Pop::new(3, eye, ACT);
        b.count = 8;
        b.tonic = 3;
        t.push(Token::Pop(b));
        let mut r = Pop::new(4, eye, ACT);
        r.field = 3;
        r.grow = true;
        r.repeat = true;
        r.repeat_lag = 4;
        r.gate = Some((aff, true));
        t.push(Token::Pop(r));
        Self { tokens: t, next_id: 5 }
    }

    /// The context circuit as tokens: the start that learned with the edge genome (experiment 06).
    /// Channels (tags far apart, so they resolve exactly):
    /// - Y: eye and affect of the last three steps, side by side;
    /// - Z: conjunction codes on Y (fresh codes born where nothing matches);
    /// - R: Z less what the context C expects (expectations learned while young);
    /// - C: the context, held by a self-wire, cleared by surprises when settled, restored by
    ///   what a context expected, set to a fresh code when empty;
    /// - N, YG: C's onset, held 32 frames ("young");
    /// - AFF: the affect channel; A: the eye beside C.
    ///
    /// Acts: a tonic population babbling into the act, and acts a stroke followed repeated on what
    /// A held before them (fading when no stroke follows).
    pub fn context_seed() -> Self {
        let ch = (world::CH_BITS / 64) as u16;
        let (y, z, r, c, n, yg, aff, a) = (0x33u8, 0x55, 0x66, 0x99, 0xAA, 0xCC, 0xF0, 0x0F);
        let mut t = vec![
            Token::Channel { tag: y, width: 6, hold: false },
            Token::Channel { tag: z, width: 3, hold: false },
            Token::Channel { tag: r, width: 3, hold: false },
            Token::Channel { tag: c, width: 3, hold: false },
            Token::Channel { tag: n, width: 3, hold: false },
            Token::Channel { tag: yg, width: 3, hold: false },
            Token::Channel { tag: aff, width: 3, hold: false },
            Token::Channel { tag: a, width: 4, hold: false },
        ];
        let mut id = 1u32;
        let mut pop = |p: Pop, t: &mut Vec<Token>| { t.push(Token::Pop(Pop { id, ..p })); id += 1; };
        for lag in 1..=3u8 {
            for (k, chan) in [world::EYE, world::AFFECT].into_iter().enumerate() {
                pop(Pop { lag, at: chan as u16 * ch, field: 3, place: ((lag - 1) as u16 * 2 + k as u16) * ch, ..Pop::wire(0, SENSES, y) }, &mut t);
            }
        }
        pop(Pop { field: 6, spread: 3, grow: true, new_code: true, ..Pop::new(0, y, z) }, &mut t);
        pop(Pop { field: 3, ..Pop::wire(0, z, r) }, &mut t);
        pop(Pop { field: 3, op: KernelOp::Clear, grow: true, share: true, gate: Some((yg, true)), ..Pop::new(0, c, r) }, &mut t);
        pop(Pop { field: 3, spread: 3, op: KernelOp::Clear, grow: true, gate: Some((yg, false)), ..Pop::new(0, r, c) }, &mut t);
        pop(Pop { field: 3, grow: true, gate: Some((yg, true)), ..Pop::new(0, r, c) }, &mut t);
        pop(Pop { field: 3, grow: true, new_code: true, ..Pop::new(0, r, c) }, &mut t);
        pop(Pop { field: 3, ..Pop::wire(0, c, c) }, &mut t);
        pop(Pop { field: 3, ..Pop::wire(0, c, n) }, &mut t);
        pop(Pop { field: 3, lag: 2, op: KernelOp::Clear, ..Pop::wire(0, c, n) }, &mut t);
        pop(Pop { field: 3, span: 32, ..Pop::wire(0, n, yg) }, &mut t);
        pop(Pop { at: world::AFFECT as u16 * ch, field: 3, ..Pop::wire(0, SENSES, aff) }, &mut t);
        pop(Pop { at: world::EYE as u16 * ch, field: 3, ..Pop::wire(0, SENSES, a) }, &mut t);
        pop(Pop { field: 3, place: ch, ..Pop::wire(0, c, a) }, &mut t);
        pop(Pop { count: 8, tonic: 3, ..Pop::new(0, a, ACT) }, &mut t);
        pop(Pop { field: 4, grow: true, repeat: true, repeat_lag: 4, gate: Some((aff, true)), ..Pop::new(0, a, ACT) }, &mut t);
        Self { tokens: t, next_id: id }
    }

    /// A bare start: one tonic population on the act, nothing else.
    pub fn bare_seed() -> Self {
        let mut b = Pop::new(1, SENSES, ACT);
        b.count = 8;
        b.tonic = 3;
        Self { tokens: vec![Token::Pop(b)], next_id: 2 }
    }
}

/// A population's starting kernels, from its own properties and its own random stream: each reads
/// `fan_in` lines across the field (the window) and writes `fan_out` lines in one word of its
/// spread (into the act: one of the body's motor primitives); tonic ones also fire spontaneously.
pub fn start_kernels(e: &EdgeInfo, primitives: &[BitVector]) -> KernelGroup {
    let sp = e.spec;
    // drawn fresh each lifetime: the genome sets the kind of cell, not the cell
    let mut r = StdRng::seed_from_u64((sp.salt as u64 + 1).wrapping_mul(0x2545_F491_4F6C_DD1D) ^ e.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let src_words = (e.src_bits / 64).max(1);
    let field = sp.window.map_or(src_words, |(o, n)| { let o = o % src_words; n.max(1).min(src_words - o) });
    let dst_words = (e.dst_bits / 64).max(1);
    let from = sp.dst_word % dst_words;
    let span = (sp.out_words as usize).max(1).min(dst_words - from);
    // the body's moves and tokens inside its port (the hand's, the voice's, or all of them)
    let (lo, hi) = if sp.dst_len > 0 { (from, from + sp.dst_len) } else { (0, dst_words) };
    let mine: Vec<&BitVector> = primitives.iter().filter(|p| p.as_words().iter().position(|&x| x != 0).is_some_and(|w| (lo..hi).contains(&w))).collect();
    let motor = |r: &mut StdRng| {
        let p = mine.choose(r).unwrap();
        let w = p.as_words().iter().position(|&x| x != 0).unwrap_or(0);
        (BitVector::from_words(vec![p.as_words()[w]]), w, KernelOp::Or)
    };
    let kernels = (0..sp.start_kernels)
        .map(|_| {
            let mut k = if sp.tile > 0 {
                // full coverage: a random tile, lines inside it, output at the same relative place
                let bits = field * 64;
                let t = (1usize << sp.tile.min(20)).min(bits);
                let stride = (t / 2).max(1);
                let tiles = if t >= bits { 1 } else { (bits - t) / stride + 1 };
                let start = r.gen_range(0..tiles) * stride;
                let all: Vec<usize> = (start..start + t).collect();
                let syn: Vec<usize> = all.choose_multiple(&mut r, (sp.start_syn as usize).min(t)).copied().collect();
                let (out, out_word, op) = if e.to_output && !mine.is_empty() {
                    motor(&mut r)
                } else {
                    let len = if sp.dst_len > 0 { sp.dst_len.min(dst_words - from) } else { dst_words };
                    let base = if sp.dst_len > 0 { from } else { 0 };
                    (sparse(64, sp.start_out as usize, &mut r), base + (start * len / field.max(1)).min(len - 1), sp.op)
                };
                let threshold = (sp.start_quarters as usize * syn.len()).div_ceil(4).max(1);
                SimpleKernel::tile(start, t, &syn, out, out_word, threshold, op)
            } else {
                let syn = sparse(field * 64, sp.start_syn as usize, &mut r);
                let (out, out_word, op) = if e.to_output && !mine.is_empty() {
                    motor(&mut r)
                } else {
                    (sparse(64, sp.start_out as usize, &mut r), from + r.gen_range(0..span), sp.op)
                };
                let threshold = (sp.start_quarters as usize * syn.count_ones()).div_ceil(4).max(1);
                let mut k = SimpleKernel::new(syn, 0, out, out_word, threshold, op);
                if sp.sparse {
                    k.sparsify();
                }
                k
            };
            if sp.expect {
                // no input yet: it expects every line off
                k.expecting(None);
            }
            if sp.teacher.is_some() {
                k.learn_fire = false;
            }
            k.tonic = sp.tonic;
            k
        })
        .collect();
    KernelGroup::from_classes(vec![KernelClass::with_kernels(kernels)])
}

/// A shadow's target kind as a tag and what it names (0 a channel, 1 an input port, 2 an output
/// port), and its width in words (as a channel width: 2^width words).
fn target_of(g: &TagGenome, kind: u8) -> Option<(u8, u8, u8)> {
    let cw = (world::CH_BITS / 64).trailing_zeros() as u8;
    match kind {
        17 => Some((EYE, 1, cw)),
        18 => Some((EAR, 1, cw)),
        19 => Some((TOUCH, 1, cw)),
        20 => Some((AFFECT, 1, cw)),
        21 => Some((SELF, 1, (SELF_BITS / 64).trailing_zeros() as u8)),
        23 => Some((VOICE, 2, cw)),
        24 => Some((HAND, 2, cw)),
        k if k >= 32 => {
            let ch = Channels::of(&g.tokens);
            let i = (k - 32) as usize + 1;
            (i < ch.self_index()).then(|| (ch.list[i].0, 0, (ch.list[i].1 as u32).trailing_zeros() as u8))
        }
        _ => None,
    }
}

/// A channel tag no existing tag would move to (each stays nearer, or as near, the channel it
/// resolves to now), as far as possible from every channel; None if there is none.
fn safe_tag(g: &TagGenome) -> Option<u8> {
    let ch = Channels::of(&g.tokens);
    let chans: Vec<u8> = g.tokens.iter().filter_map(|t| if let Token::Channel { tag, .. } = t { Some(*tag) } else { None }).collect();
    let mut tags: Vec<u8> = Vec::new();
    for p in g.pops() {
        if !p.from_port { tags.push(p.from) }
        if !p.to_port { tags.push(p.to) }
        if let (Some((t, _)), 0) = (p.gate, p.gate_at) { tags.push(t) }
        if let (Some((t, _)), 0) = (p.pause, p.pause_at) { tags.push(t) }
        if let (Some((t, _)), 0) = (p.heat, p.heat_at) { tags.push(t) }
        if let (Some(t), 0) = (p.inhibitor, p.inhibitor_at) { tags.push(t) }
        if let Some(t) = p.branch.filter(|_| p.branch_at != 2) { tags.push(t) }
        if let Some((t, 0)) = p.teacher { tags.push(t) }
    }
    let near = |t: u8| ch.internal(t).map(|i| (ch.list[i].0 ^ t).count_ones());
    if chans.is_empty() && !tags.is_empty() {
        return None;
    }
    (0..=255u8)
        .filter(|&c| tags.iter().all(|&t| near(t).is_some_and(|d| (c ^ t).count_ones() >= d)))
        .max_by_key(|&c| chans.iter().map(|&x| (c ^ x).count_ones()).min().unwrap_or(8))
}

/// A shadow's prediction kept: a channel of its own (as wide as the target) and a predictor into
/// it, reading what the population `id` reads, learning against the target; tagged explored,
/// marked, after the population. Once per (population source, target).
fn keep_predictor(g: &mut TagGenome, id: u32, kind: u8, _c: u32) {
    let Some(at) = g.tokens.iter().position(|t| matches!(t, Token::Pop(p) if p.id == id)) else { return };
    let Token::Pop(parent) = g.tokens[at].clone() else { return };
    let Some((tag, tat, width)) = target_of(g, kind) else { return };
    if g.pops().any(|p| p.explored && p.teacher == Some((tag, tat)) && p.from == parent.from && p.from_port == parent.from_port) {
        return;
    }
    let Some(c) = safe_tag(g) else { return };
    let q = Pop {
        id: g.next_id, to: c, to_port: false, to_input: false, from_output: false, wire: false, op: KernelOp::Or, count: 0, grow: true, cover: false,
        replicate: parent.replicate.max(3), cap: parent.cap.max(16), gate: None, pause: None, repeat: false, mark: 128, explored: true,
        teacher: Some((tag, tat)), ..parent
    };
    g.next_id += 1;
    g.tokens.insert(at + 1, Token::Pop(q));
    g.tokens.push(Token::Channel { tag: c, width, hold: false });
}

/// `Evolvable::inherit` for the tag grammar (with `set_epigenetics`; here without, for tests).
pub fn take_notes(g: &mut TagGenome, notes: &[super::Note]) {
    take_notes_with(g, notes, shadow_hints())
}

/// `take_notes`, with shadows' finds noted as hints (`hints`) or built as predictors.
fn take_notes_with(g: &mut TagGenome, notes: &[super::Note], hints: bool) {
    for t in &mut g.tokens {
        if let Token::Pop(p) = t {
            let ev: u32 = notes.iter().filter(|n| n.0 == p.id && n.2 == 0).map(|n| n.1).sum();
            p.mark = (p.mark / 2).saturating_add(((32 - ev.leading_zeros()) * 16).min(255) as u8);
        }
    }
    // a shadow's best target over the generation: kept as a predictor into a channel of its
    // own (one a genome a generation)
    if hints {
        // (hints) each population's best-predicted target becomes its hint; nothing is built
        let mut best: Vec<(u32, u32, u8)> = Vec::new();
        for &(id, c, kind) in notes.iter().filter(|n| n.2 >= 16 && n.1 >= KEEP * 4) {
            match best.iter_mut().find(|b| b.0 == id) {
                Some(b) if c > b.1 => *b = (id, c, kind),
                Some(_) => {}
                None => best.push((id, c, kind)),
            }
        }
        for (id, _, kind) in best {
            let Some((tag, at, _)) = target_of(g, kind) else { continue };
            for t in &mut g.tokens {
                if let Token::Pop(p) = t {
                    if p.id == id {
                        p.hint = Some((tag, at));
                    }
                }
            }
        }
    } else if let Some(&(id, c, kind)) = notes.iter().filter(|n| n.2 >= 16 && n.1 >= KEEP * 4).max_by_key(|n| n.1) {
        keep_predictor(g, id, kind, c);
    }
    for &(id, c, kind) in notes {
        if kind == 0 || kind >= 16 || c < KEEP {
            continue;
        }
        let Some(at) = g.tokens.iter().position(|t| matches!(t, Token::Pop(p) if p.id == id)) else { continue };
        let Token::Pop(parent) = &g.tokens[at] else { continue };
        if parent.to_port || (kind == 1 && parent.from_port) {
            continue;
        }
        let to = if kind == 1 { parent.from } else { parent.to };
        if g.pops().any(|p| p.explored && p.from == parent.to && p.to == to && !p.from_port && !p.to_port) {
            continue;
        }
        let q = Pop {
            id: g.next_id, from: parent.to, from_port: false, to, to_port: false, wire: false, op: KernelOp::Xor, count: 0,
            grow: true, cover: true, replicate: parent.replicate.max(3), gate: None, pause: None, repeat: false, mark: 128, explored: true,
            ..parent.clone()
        };
        g.next_id += 1;
        g.tokens.insert(at + 1, Token::Pop(q));
    }
}

impl Evolvable for TagGenome {
    fn build(&self, w: &World, seed: u64) -> RuntimeNetwork {
        let g = self.graph();
        let body = w.body();
        let primitives: Vec<BitVector> = (0..body.voice.len())
            .map(|i| w.act(Some(i), None))
            .chain((0..body.hand.len()).map(|i| w.act(None, Some(i))))
            .collect();
        let mut net = RuntimeNetwork::from_graph_held(&g, world::SENSE_BITS, world::ACT_BITS, seed, w.in_place, |e: &EdgeInfo| start_kernels(e, &primitives));
        net.count_reads = w.count_reads;
        net.explore = w.explore;
        net.shadow = w.shadow;
        net.shadow_random = shadow_hints();
        net
    }

    /// With `set_epigenetics`: each population's mark becomes half its old mark plus 16 for each
    /// doubling of its predictions that came true this generation (over all its lifetimes), and
    /// each trial connection whose predictions came true `KEEP` times or more is added as a
    /// population, tagged `explored`, after the one it grew from, with a starting mark: a
    /// predicting (XOR), replicating population with no starting kernels, reading that
    /// population's destination and writing back into its source (or into the destination).
    fn inherit(&mut self, notes: &[super::Note]) {
        if super::epigenetics() {
            take_notes(self, notes);
        }
    }


    fn mutate(&self, rng: &mut StdRng) -> Self {
        let mut g = self.clone();
        let n = g.tokens.len();
        // (epigenetics: a marked population is spared deletion and chosen for duplication more
        // often, in proportion to its mark)
        let epi = super::epigenetics();
        let marked = |g: &TagGenome, rng: &mut StdRng| -> Option<usize> {
            let w: Vec<(usize, u32)> = g.tokens.iter().enumerate().filter_map(|(i, t)| if let Token::Pop(p) = t { (p.mark > 0).then_some((i, p.mark as u32)) } else { None }).collect();
            let total: u32 = w.iter().map(|x| x.1).sum();
            if total == 0 { return None }
            let mut r = rng.gen_range(0..total);
            w.into_iter().find(|&(_, m)| if r < m { true } else { r -= m; false }).map(|x| x.0)
        };
        match rng.gen_range(0..16) {
            // a new token (two chances in sixteen: new populations are the raw material)
            0 | 15 => {
                let t = if rng.gen_bool(0.3) { Token::Channel { tag: rng.r#gen(), width: rng.gen_range(0..=MAX_WIDTH), hold: rng.gen_bool(0.25) } } else if motions() && rng.gen_bool(0.15) {
                    // (with `set_motions`) a motion for one of its channels, a conveyor of stride
                    // 1 to 4 or a wave
                    let tags: Vec<u8> = g.tokens.iter().filter_map(|t| if let Token::Channel { tag, .. } = t { Some(*tag) } else { None }).collect();
                    let tag = tags.choose(rng).copied().unwrap_or_else(|| rng.r#gen());
                    if rng.gen_bool(0.75) { Token::Motion { tag, conveyor: rng.gen_range(1..=4), wave: false } } else { Token::Motion { tag, conveyor: 0, wave: true } }
                } else { Token::Pop(g.random_pop(rng)) };
                let at = rng.gen_range(0..=n);
                g.tokens.insert(at, t);
            }
            // a token removed
            1 if n > 0 => {
                let at = rng.gen_range(0..n);
                let spared = epi && matches!(&g.tokens[at], Token::Pop(p) if p.mark > 0 && rng.gen_range(0..256u32) < p.mark as u32);
                if !spared {
                    g.tokens.remove(at);
                }
            }
            // a segment copied elsewhere; sometimes privately (channels declared inside get fresh
            // tags, and the copy's references to them follow, so its inner wiring is its own)
            2 | 3 if n > 0 => {
                let a = if epi && rng.gen_bool(0.5) { marked(&g, rng).unwrap_or_else(|| rng.gen_range(0..n)) } else { rng.gen_range(0..n) };
                let len = rng.gen_range(1..=4.min(n - a));
                let mut seg: Vec<Token> = g.tokens[a..a + len].to_vec();
                if rng.gen_bool(0.5) {
                    let mut map: Vec<(u8, u8)> = Vec::new();
                    for t in &mut seg {
                        if let Token::Channel { tag, .. } = t {
                            let new: u8 = rng.r#gen();
                            map.push((*tag, new));
                            *tag = new;
                        }
                    }
                    for t in &mut seg {
                        if let Token::Pop(p) = t {
                            for (old, new) in &map {
                                if p.from == *old { p.from = *new }
                                if p.to == *old { p.to = *new }
                                if let Some((gt, on)) = p.gate { if gt == *old { p.gate = Some((*new, on)) } }
                            }
                        }
                    }
                }
                for t in &mut seg {
                    if let Token::Pop(p) = t {
                        p.id = g.next_id;
                        g.next_id += 1;
                    }
                }
                let at = rng.gen_range(0..=n);
                for (k, t) in seg.into_iter().enumerate() {
                    g.tokens.insert(at + k, t);
                }
            }
            // a segment moved
            4 if n > 1 => {
                let a = rng.gen_range(0..n);
                let len = rng.gen_range(1..=4.min(n - a));
                let seg: Vec<Token> = g.tokens.drain(a..a + len).collect();
                let at = rng.gen_range(0..=g.tokens.len());
                for (k, t) in seg.into_iter().enumerate() {
                    g.tokens.insert(at + k, t);
                }
            }
            // a circuit copied whole (gene duplication), half the time reading another channel
            14 if g.channels() > 0 => {
                let c = rng.gen_range(1..=g.channels());
                let rewire = if rng.gen_bool(0.5) { Some((rng.r#gen(), rng.gen_bool(0.5))) } else { None };
                g.copy_circuit(c, rewire, rng);
            }
            // (with `set_hint_use`) an edge built on a hint: one mutation in sixteen, while any
            // population carries one, adds a new population of any kind on a hinted path, after
            // the hinted population
            10 if hint_use() && g.pops().any(|p| p.hint.is_some()) => {
                let hinted: Vec<usize> = (0..n).filter(|&i| matches!(&g.tokens[i], Token::Pop(p) if p.hint.is_some())).collect();
                let at = *hinted.choose(rng).unwrap();
                let Token::Pop(h) = g.tokens[at].clone() else { unreachable!() };
                let feedback = g.hint_feedback(&h);
                let mut q = g.random_pop(rng);
                on_hint(&mut q, &h, feedback, rng);
                g.tokens.insert(at + 1, Token::Pop(q));
            }
            // (with the op choice) one population's op drawn again: one mutation in sixteen, where as
            // one of a population's 38 properties it is about one in a hundred
            11 if crate::kernel::class::op_choice() && g.pops().count() > 0 => {
                let pops: Vec<usize> = (0..n).filter(|&i| matches!(g.tokens[i], Token::Pop(_))).collect();
                let at = *pops.choose(rng).unwrap();
                if let Token::Pop(p) = &mut g.tokens[at] {
                    p.op = KernelOp::random(rng);
                }
            }
            // (with `set_close_loops`) a loop closed: one mutation in sixteen, a population's path
            // walked forward 1 to 4 populations and a new population of any kind added from where
            // it ends back to where it starts
            9 if close_loops() && g.pops().count() > 0 => {
                let pops: Vec<usize> = (0..n).filter(|&i| matches!(g.tokens[i], Token::Pop(_))).collect();
                let at = *pops.choose(rng).unwrap();
                let steps = rng.gen_range(0..4);
                let q = g.random_pop(rng);
                g.close_loop(at, steps, q, rng);
            }
            // a stage split into a population's path, or a population turned back on its source
            12 | 13 if g.pops().count() > 0 => {
                let pops: Vec<usize> = (0..n).filter(|&i| matches!(g.tokens[i], Token::Pop(_))).collect();
                let at = *pops.choose(rng).unwrap();
                if rng.gen_bool(0.5) { g.split(at, rng.gen_range(0..8)) } else { g.self_loop(at) }
            }
            // one property of one token
            _ if n > 0 => {
                let at = rng.gen_range(0..n);
                let up = rng.gen_bool(0.5);
                let step = |v: u8, lo: u8, hi: u8| if up { v.saturating_add(1).min(hi) } else { v.saturating_sub(1).max(lo) };
                let bit = 1u8 << rng.gen_range(0..8);
                match &mut g.tokens[at] {
                    Token::Channel { tag, width, hold } => match rng.gen_range(0..3) {
                        0 => *tag ^= bit,
                        1 => *width = step(*width, 0, MAX_WIDTH),
                        _ => *hold = !*hold,
                    },
                    Token::Motion { tag, conveyor, wave } => match rng.gen_range(0..3) {
                        0 => *tag ^= bit,
                        1 => *conveyor = step(*conveyor, 0, 8),
                        _ => *wave = !*wave,
                    },
                    Token::Pop(p) => match rng.gen_range(0..if triggers() { 58 } else if rule_tables() { 57 } else if eligibility() { 56 } else if birth_lags() { 55 } else if branches() { 54 } else if crate::program::neurocomp::prune() { 52 } else if overlap_genes() { 51 } else if inhibit_inputs() { 50 } else if heat_inputs() { 48 } else if hint_band() { 46 } else if read_outputs() { 45 } else if crate::program::neurocomp::shadow() { 44 } else if port_gates() { 43 } else if accident() { 40 } else if zeros() { 39 } else { 38 }) {
                        0 => p.from ^= bit,
                        1 => p.to ^= bit,
                        2 => p.wire = !p.wire,
                        3 => p.op = if rng.gen_bool(0.5) { KernelOp::random(rng) } else { p.op.nudge(up) },
                        4 => p.lag = step(p.lag, 1, 8),
                        5 => p.span = if up { (p.span * 2).min(64) } else { (p.span / 2).max(1) },
                        6 => { let d = hop(p.field, rng); p.at = if up { p.at.wrapping_add(d) } else { p.at.wrapping_sub(d) } }
                        7 => p.field = step(p.field, 0, MAX_WIDTH),
                        8 => { let d = hop(p.spread, rng); p.place = if up { p.place.wrapping_add(d) } else { p.place.wrapping_sub(d) } }
                        9 => p.spread = step(p.spread, 0, MAX_WIDTH),
                        10 => p.count = step(p.count, 0, 32),
                        11 => p.fan_in = step(p.fan_in, 1, 64),
                        12 => p.fan_out = step(p.fan_out, 1, 64),
                        13 => p.quarters = step(p.quarters, 1, 4),
                        14 => p.tonic = step(p.tonic, 0, 12),
                        15 => p.share = !p.share,
                        16 => p.grow = !p.grow,
                        17 => p.new_code = !p.new_code,
                        18 => p.repeat = !p.repeat,
                        19 => p.repeat_lag = step(p.repeat_lag, 1, 8),
                        // (with hints) half the time a gate takes what this population's shadow found
                        20 if shadow_hints() && p.hint.is_some() && rng.gen_bool(0.5) => {
                            let (t, at) = p.hint.unwrap();
                            p.gate = Some((t, rng.gen_bool(0.5)));
                            p.gate_at = at;
                        }
                        20 => p.gate = match p.gate { Some((t, on)) => if rng.gen_bool(0.3) { None } else if rng.gen_bool(0.5) { Some((t ^ bit, on)) } else { Some((t, !on)) }, None => Some((rng.r#gen(), rng.gen_bool(0.5))) },
                        21 => p.gate_lag = step(p.gate_lag, 1, 8),
                        22 => p.birth_syn = step(p.birth_syn, 1, 64),
                        23 => p.birth_quarters = step(p.birth_quarters, 1, 4),
                        24 => p.out_min = step(p.out_min, 1, 64),
                        25 => p.out_max = step(p.out_max, 1, 64),
                        26 => p.code_lines = step(p.code_lines, 1, 64),
                        27 => p.out_learn = step(p.out_learn, 0, 8),
                        28 => p.fade = step(p.fade, 0, 8),
                        29 => p.rotate = if up { p.rotate.wrapping_add(61) } else { p.rotate.wrapping_sub(61) },
                        30 => p.sparse = !p.sparse,
                        31 => p.cover = !p.cover,
                        32 => p.tile = step(p.tile, 3, MAX_TILE),
                        33 => p.replicate = step(p.replicate, 0, 7),
                        34 => p.replicate_shift = !p.replicate_shift,
                        36 => p.from_port = !p.from_port,
                        37 => p.to_port = !p.to_port,
                        38 if zeros() => p.expect = !p.expect,
                        39 if accident() => p.accident = step(p.accident, 0, 20),
                        40 if port_gates() => p.to_input = !p.to_input,
                        41 if port_gates() => p.gate_at = (p.gate_at + if up { 1 } else { 2 }) % 3,
                        42 if port_gates() => p.pause_at = (p.pause_at + if up { 1 } else { 2 }) % 3,
                        43 if p.teacher.is_some() => p.hit_quarters = step(p.hit_quarters, 1, 4),
                        44 if read_outputs() => p.from_output = !p.from_output,
                        45 if hint_band() => p.hint_low = step(p.hint_low, 1, 3),
                        // (with heat inputs) a heat input on the hint half the time, if there is one
                        46 if heat_inputs() && shadow_hints() && p.hint.is_some() && rng.gen_bool(0.5) => {
                            let (t, at) = p.hint.unwrap();
                            p.heat = Some((t, rng.gen_bool(0.5)));
                            p.heat_at = at;
                        }
                        46 if heat_inputs() => {
                            p.heat = match p.heat { Some((t, w)) => if rng.gen_bool(0.3) { None } else if rng.gen_bool(0.5) { Some((t ^ bit, w)) } else { Some((t, !w)) }, None => Some((rng.r#gen(), rng.gen_bool(0.5))) };
                            if port_gates() && p.heat.is_some() && rng.gen_bool(0.5) {
                                p.heat_at = rng.gen_range(0..3);
                            }
                        }
                        47 if heat_inputs() => p.heat_at = (p.heat_at + if up { 1 } else { 2 }) % 3,
                        // (with inhibit inputs) an inhibit input on the hint half the time, if there is one
                        48 if inhibit_inputs() && shadow_hints() && p.hint.is_some() && rng.gen_bool(0.5) => {
                            let (t, at) = p.hint.unwrap();
                            p.inhibitor = Some(t);
                            p.inhibitor_at = at;
                        }
                        48 if inhibit_inputs() => {
                            p.inhibitor = match p.inhibitor { Some(t) => if rng.gen_bool(0.3) { None } else { Some(t ^ bit) }, None => Some(rng.r#gen()) };
                            if port_gates() && p.inhibitor.is_some() && rng.gen_bool(0.5) {
                                p.inhibitor_at = rng.gen_range(0..3);
                            }
                        }
                        49 if inhibit_inputs() => p.inhibitor_at = (p.inhibitor_at + if up { 1 } else { 2 }) % 3,
                        50 if overlap_genes() => p.inhibit_quarters = step(p.inhibit_quarters, 1, 4),
                        51 if crate::program::neurocomp::prune() => p.grip = if up { p.grip.saturating_mul(2).min(128) } else { (p.grip / 2).max(1) },
                        52 if branches() => p.branch = match p.branch { Some(t) => if rng.gen_bool(0.3) { None } else { Some(t ^ bit) }, None => Some(rng.r#gen()) },
                        53 if branches() => p.branch_at = if p.branch_at == 2 { 0 } else { 2 },
                        54 if birth_lags() => p.birth_lag = step(p.birth_lag, 0, 4),
                        55 if eligibility() => p.elig = step(p.elig, 0, 4),
                        // (a rule table) one bit of it flips (a population without one starts
                        // from the runtime's own rules)
                        57 if triggers() => if rng.gen_bool(0.25) { p.lane = rng.gen_range(3..=6) } else { p.trigger = rng.gen_range(0..10) },
                        56 if rule_tables() => p.rule = (if p.rule == 0 { crate::program::graph::RULE_SEED } else { p.rule }) ^ (1 << rng.gen_range(0..13)),
                        // (with hints) and so, half the time, does a pause
                        _ if shadow_hints() && p.hint.is_some() && rng.gen_bool(0.5) => {
                            let (t, at) = p.hint.unwrap();
                            p.pause = Some((t, rng.gen_bool(0.5)));
                            p.pause_at = at;
                        }
                        _ => {
                            let fresh = p.pause.is_none();
                            p.pause = match p.pause { Some((t, on)) => if rng.gen_bool(0.3) { None } else if rng.gen_bool(0.5) { Some((t ^ bit, on)) } else { Some((t, !on)) }, None => Some((rng.r#gen(), rng.gen_bool(0.5))) };
                            // (with `set_pause_draws` and port gates) a new pause draws what it names
                            if fresh && pause_draws() && port_gates() {
                                p.pause_at = rng.gen_range(0..3);
                            }
                        }
                    },
                }
            }
            _ => g.tokens.push(Token::Pop(g.clone().random_pop(rng))),
        }
        g
    }

    fn size(&self) -> usize {
        self.tokens.len()
    }

    fn wire_length(&self) -> usize {
        TagGenome::wire_length(self)
    }

    /// Half the time a plasmid: one whole circuit of the other genome, made private and appended
    /// (so a working circuit moves intact); otherwise one point: this string's front, the
    /// other's back (tags carry the wiring across).
    fn crossover(&self, other: &Self, rng: &mut StdRng) -> Self {
        if other.populations() > 0 && rng.gen_bool(0.5) {
            let mut g = self.clone();
            let k = rng.gen_range(0..other.populations());
            g.take_population_plasmid(other, k, rng);
            return g;
        }
        let a = rng.gen_range(0..=self.tokens.len());
        let b = rng.gen_range(0..=other.tokens.len());
        let mut g = TagGenome { tokens: self.tokens[..a].iter().chain(other.tokens[b..].iter()).cloned().collect(), next_id: self.next_id.max(other.next_id) };
        g.unique_ids();
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolve::{live, Physics, Task};
    use crate::world::Conditioning;

    #[test]
    fn a_crossover_child_never_repeats_an_id() {
        let mut rng = <StdRng as rand::SeedableRng>::seed_from_u64(4);
        let c = 0x5A;
        let pop = |id: u32| { let mut p = Pop::new(id, c, c); p.id = id; Token::Pop(p) };
        let a = TagGenome { tokens: vec![Token::Channel { tag: c, width: 3, hold: false }, pop(1), pop(2), pop(3)], next_id: 4 };
        for _ in 0..200 {
            let child = a.crossover(&a, &mut rng);
            let ids: Vec<u32> = child.pops().map(|p| p.id).collect();
            let set: std::collections::HashSet<u32> = ids.iter().copied().collect();
            assert_eq!(ids.len(), set.len(), "{ids:?}");
            assert!(child.next_id > ids.iter().copied().max().unwrap_or(0));
        }
    }

    #[test]
    fn tags_name_channels_and_ports_have_their_own_tags() {
        let ch = Channels::of(&[Token::Channel { tag: 0b1010_0000, width: 0, hold: false }, Token::Channel { tag: 0b0000_0101, width: 1, hold: false }]);
        // an ordinary tag resolves among the declared channels only
        assert_eq!(ch.internal(0b1010_0001), Some(1));
        assert_eq!(ch.internal(0b0000_0111), Some(2));
        // 0x00 is two bits from both channels (ties go to the earlier): just a channel tag here
        assert_eq!(ch.internal(SENSES), Some(1), "the senses are not in the channels' tag space");
        assert_eq!(Channels::of(&[]).internal(0x42), None);
        // ports: each sense, the self channel, and parts of the act, by their own tags
        let w = world::CH_BITS / 64;
        assert_eq!(ch.input(SENSES), Place { node: 0, off: 0, len: 4 * w, whole: true });
        assert_eq!(ch.input(EYE), Place { node: 0, off: world::EYE * w, len: w, whole: false });
        assert_eq!(ch.input(AFFECT), Place { node: 0, off: world::AFFECT * w, len: w, whole: false });
        assert_eq!(ch.input(SELF).node, ch.self_index());
        assert_eq!(ch.output(ACT), Place { node: ch.act_index(), off: 0, len: 2 * w, whole: true });
        assert_eq!(ch.output(HAND), Place { node: ch.act_index(), off: world::HAND * w, len: w, whole: false });
        // every input port is reachable from some tag, and so is every output port
        let mut seen_in = std::collections::HashSet::new();
        let mut seen_out = std::collections::HashSet::new();
        for t in 0..=255u8 {
            seen_in.insert(ch.name(ch.input(t)));
            seen_out.insert(ch.name(ch.output(t)));
        }
        assert_eq!((seen_in.len(), seen_out.len()), (6, 3));
    }

    #[test]
    fn any_string_builds_a_network_that_lives() {
        let mut rng = StdRng::seed_from_u64(9);
        let mut g = TagGenome::act_seed();
        for i in 0..400u64 {
            g = g.mutate(&mut rng);
            let _ = g.graph();
            if i % 20 == 0 {
                let _ = live(&g, Task::Conditioning(Conditioning::Reversal, 6), Physics::Varied, i);
            }
        }
        // and an empty string is a network too
        let _ = live(&TagGenome { tokens: vec![], next_id: 0 }, Task::Conditioning(Conditioning::Reversal, 4), Physics::Varied, 1);
    }

    /// A sliding window from the grammar alone: a channel carrying its previous frame shifted
    /// left a byte, and the newest byte written in. No window is built into the runtime.
    #[test]
    fn a_sliding_window_is_two_populations() {
        use crate::bitvec::bithistory::AdvanceMode;
        let w = 0x5A;
        let mut into = Pop::wire(1, SENSES, w);
        into.field = 0;
        let mut carry = Pop::wire(2, w, w);
        carry.op = KernelOp::Shl(8);
        let g = TagGenome { tokens: vec![Token::Channel { tag: w, width: 0, hold: false }, Token::Pop(into), Token::Pop(carry)], next_id: 3 };
        let world = World::new(1, 4);
        let mut net = g.build(&world, 1);
        let mut last = 0u64;
        for b in [0x11u64, 0x22, 0x33, 0x44] {
            net.tick(AdvanceMode::Clear, 0);
            net.write_input(&BitVector::from_words(vec![b]), 0);
            last = net.read_node(1).as_words()[0];
        }
        // the window holds the bytes before the newest, newest lowest (it reads the senses a frame back)
        assert_eq!(last, 0x11_22_33);
    }

    #[test]
    fn the_context_seed_resolves_every_tag_exactly() {
        let g = TagGenome::context_seed();
        let ch = Channels::of(&g.tokens);
        for p in g.pops() {
            let (a, b) = (ch.from(p).unwrap(), ch.to(p).unwrap());
            assert_eq!(ch.list[a.node].0, p.from);
            assert_eq!(ch.list[b.node].0, p.to);
        }
        let _ = live(&g, Task::Conditioning(Conditioning::Reversal, 8), Physics::Varied, 1);
    }

    #[test]
    fn split_inserts_a_stage_and_self_loop_closes_one() {
        let (a, b) = (0x0Fu8, 0xF0u8);
        let p = Pop { count: 2, ..Pop::new(1, a, b) };
        let mut g = TagGenome { tokens: vec![Token::Channel { tag: a, width: 2, hold: false }, Token::Channel { tag: b, width: 5, hold: false }, Token::Pop(p)], next_id: 2 };
        g.split(2, 0);
        // a new channel 0xF1 as wide as B; the population writes it; a wire carries it on to B
        assert_eq!(g.tokens[2], Token::Channel { tag: 0xF1, width: 5, hold: false });
        let Token::Pop(p) = &g.tokens[3] else { panic!() };
        assert_eq!((p.from, p.to), (a, 0xF1));
        let Token::Pop(w) = &g.tokens[4] else { panic!() };
        assert!(w.wire && w.from == 0xF1 && w.to == b && w.cover && !w.from_port && !w.to_port);
        assert_eq!(g.graph().edge_count(), 2);
        g.self_loop(3);
        let Token::Pop(p) = &g.tokens[3] else { panic!() };
        assert_eq!(p.to, a);
        // a population reading the senses is left alone
        let mut s = TagGenome { tokens: vec![Token::Pop(Pop::new(1, SENSES, a))], next_id: 2 };
        s.self_loop(0);
        assert_eq!(s.tokens[0], Token::Pop(Pop::new(1, SENSES, a)));
    }

    #[test]
    fn a_held_channel_keeps_its_lines_until_something_clears_them() {
        // H copies the senses' word 0 (OR); a CLEAR wire takes off whatever is on in the senses'
        // word 1. Shown a code once, a held H keeps it with no carry wire; an all-on word 1
        // clears it. A plain H loses it the next step.
        let run = |hold: bool| -> Vec<u32> {
            let h = 0x3C;
            let set = Pop { field: 0, ..Pop::wire(1, SENSES, h) };
            let clear = Pop { at: 1, field: 0, op: KernelOp::Clear, ..Pop::wire(2, SENSES, h) };
            let g = TagGenome { tokens: vec![Token::Channel { tag: h, width: 0, hold }, Token::Pop(set), Token::Pop(clear)], next_id: 3 };
            let mut net = g.build(&World::new(1, 4), 1);
            let mut seen = Vec::new();
            for t in 0..6 {
                let mut words = vec![0u64; world::SENSE_BITS / 64];
                if t == 0 { words[0] = 0xF0F0 }
                if t == 4 { words[1] = u64::MAX }
                net.tick(crate::bitvec::bithistory::AdvanceMode::Clear, 0);
                net.write_input(&BitVector::from_words(words), 0);
                seen.push(net.read_node(1).count_ones() as u32);
            }
            seen
        };
        // wires read one frame back, so H sees step t's input at step t+1
        assert_eq!(run(true), vec![0, 8, 8, 8, 8, 0]);
        assert_eq!(run(false), vec![0, 8, 0, 0, 0, 0]);
    }

    #[test]
    fn in_place_a_network_builds_its_own_delay_and_latch() {
        use crate::bitvec::bithistory::AdvanceMode;
        let (a, b, e, c) = (0x3C, 0xC3, 0x5A, 0xA5);
        // a delay line: B takes A before A takes the senses' word 0 (genome order), so B is A a
        // step late; both overwrite (Set), so a value passes through and is gone
        let a_to_b = Pop { cover: true, op: KernelOp::Set, ..Pop::wire(1, a, b) };
        let in_a = Pop { field: 0, op: KernelOp::Set, ..Pop::wire(2, SENSES, a) };
        // a latch: E is the senses' word 1 now; C takes word 0 only while E is on, else keeps it
        let in_e = Pop { at: 1, field: 0, op: KernelOp::Set, ..Pop::wire(3, SENSES, e) };
        let latch = Pop { field: 0, op: KernelOp::Set, pause: Some((e, true)), ..Pop::wire(4, SENSES, c) };
        let ch = |t: u8| Token::Channel { tag: t, width: 0, hold: false };
        let g = TagGenome { tokens: vec![ch(a), ch(b), ch(e), ch(c), Token::Pop(a_to_b), Token::Pop(in_a), Token::Pop(in_e), Token::Pop(latch)], next_id: 5 };
        let mut w = World::new(1, 4);
        w.in_place = true;
        let mut net = g.build(&w, 1);
        let (x1, x2) = (0xF0F0u64, 0x0F0Fu64);
        let mut rows = Vec::new();
        for t in 0..6 {
            let mut words = vec![0u64; world::SENSE_BITS / 64];
            words[0] = [x1, 0, x2, 0, 0, 0][t];
            words[1] = [u64::MAX, 0, 0, 0, 0, 0][t];
            net.tick(AdvanceMode::Clear, 0);
            net.write_input(&BitVector::from_words(words), 0);
            rows.push([1, 2, 4].map(|n| net.read_node(n).as_words()[0]));
        }
        // (the senses arrive a step after they are shown)
        let a_seen: Vec<u64> = rows.iter().map(|r| r[0]).collect();
        let b_seen: Vec<u64> = rows.iter().map(|r| r[1]).collect();
        let c_seen: Vec<u64> = rows.iter().map(|r| r[2]).collect();
        assert_eq!(a_seen, vec![0, x1, 0, x2, 0, 0]);
        assert_eq!(b_seen, vec![0, 0, x1, 0, x2, 0]);
        // the latch took x1 while enabled and kept it through x2
        assert_eq!(c_seen, vec![0, x1, x1, x1, x1, x1]);
        // no history: every held channel is one frame
        assert!((1..=4).all(|n| net.nodes[n].frames() == 1));
    }

    #[test]
    fn in_place_every_channel_holds_in_one_frame_and_order_matters() {
        use crate::bitvec::bithistory::AdvanceMode;
        let (h, k) = (0x3C, 0xC3);
        let set = Pop { field: 0, ..Pop::wire(1, SENSES, h) };
        let clear = Pop { at: 1, field: 0, op: KernelOp::Clear, ..Pop::wire(2, SENSES, h) };
        // K copies H (lag 1): held in place, it sees what H holds now, written earlier this step
        let copy = Pop { cover: true, ..Pop::wire(3, h, k) };
        let g = TagGenome { tokens: vec![Token::Channel { tag: h, width: 0, hold: false }, Token::Channel { tag: k, width: 0, hold: false }, Token::Pop(set), Token::Pop(clear), Token::Pop(copy)], next_id: 4 };
        let run = |in_place: bool| -> (Vec<u32>, Vec<u32>, usize) {
            let mut w = World::new(1, 4);
            w.in_place = in_place;
            let mut net = g.build(&w, 1);
            let (mut hs, mut ks) = (Vec::new(), Vec::new());
            for t in 0..6 {
                let mut words = vec![0u64; world::SENSE_BITS / 64];
                if t == 0 { words[0] = 0xF0F0 }
                if t == 4 { words[1] = u64::MAX }
                net.tick(AdvanceMode::Clear, 0);
                net.write_input(&BitVector::from_words(words), 0);
                hs.push(net.read_node(1).count_ones() as u32);
                ks.push(net.read_node(2).count_ones() as u32);
            }
            (hs, ks, net.nodes[1].frames())
        };
        // held in place: H keeps the code (no hold flag, no carry) until the clearing write, in
        // one frame; K has it the same step H gets it, and keeps it after H is cleared
        let (hs, ks, frames) = run(true);
        assert_eq!(hs, vec![0, 8, 8, 8, 8, 0]);
        assert_eq!(ks, vec![0, 8, 8, 8, 8, 8]);
        assert_eq!(frames, 1);
        // the old runtime: H loses it the next step, K sees it a step late, two frames each
        let (hs, ks, frames) = run(false);
        assert_eq!(hs, vec![0, 8, 0, 0, 0, 0]);
        assert_eq!(ks, vec![0, 0, 8, 0, 0, 0]);
        assert_eq!(frames, 2);
    }

    #[test]
    fn a_covering_population_finds_input_wherever_it_lands() {
        // one growing population over the whole senses, tiles of 64 bits: shown a code in a
        // different word each lifetime, it is born on that code and fires on it again
        for word in [0usize, 5, 17, 30] {
            let c = 0x3C;
            let p = Pop { cover: true, tile: 6, grow: true, new_code: true, ..Pop::new(1, SENSES, c) };
            let g = TagGenome { tokens: vec![Token::Channel { tag: c, width: 3, hold: false }, Token::Pop(p)], next_id: 2 };
            let mut net = g.build(&World::new(1, 4), word as u64);
            let mut on = vec![0u64; world::SENSE_BITS / 64];
            on[word] = 0x0F0F_0000_0000_F0F0;
            let mut seen = Vec::new();
            for t in 0..8 {
                let x = if t % 2 == 0 { on.clone() } else { vec![0u64; world::SENSE_BITS / 64] };
                net.tick(crate::bitvec::bithistory::AdvanceMode::Clear, 0);
                net.write_input(&BitVector::from_words(x), 0);
                seen.push(net.read_node(1).count_ones());
            }
            // the code shown at even steps is read at odd ones (lag 1): born at step 1, then the
            // kernel fires at each odd step
            assert!(seen[3] > 0 && seen[5] > 0 && seen[7] > 0, "word {word}: {seen:?}");
            assert_eq!((seen[2], seen[4], seen[6]), (0, 0, 0), "word {word}: {seen:?}");
            let n = net.edges[0].group.kernel_count();
            assert!(n >= 1 && n <= 3, "word {word}: {n} kernels");
        }
    }

    #[test]
    fn only_a_genome_silent_in_every_lifetime_dies() {
        use crate::evolve::{fitness_detail, Physics, Task};
        use crate::world::{observe, Conditioning};
        let task = Task::Conditioning(Conditioning::Instrumental, 40);
        let empty = TagGenome { tokens: vec![], next_id: 1 };
        let (f, _, silent) = fitness_detail(&empty, task, Physics::Varied, &[1, 2, 3]);
        assert_eq!((f, silent), (observe::SILENT, 3));
        // a tonic population on the act clicks in some lifetimes: no death, silent ones count 0
        let babble = TagGenome { tokens: vec![Token::Pop(Pop { count: 8, tonic: 1, ..Pop::new(1, SELF, ACT) })], next_id: 2 };
        let (f, _, silent) = fitness_detail(&babble, task, Physics::Varied, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(silent < 8 && f > -1024, "{f} {silent}");
    }

    #[test]
    fn a_paused_population_does_nothing_and_costs_nothing() {
        // C copies the senses, with an execution gate on G (which nothing writes, so G is empty):
        // "runs while G" never runs, "runs while !G" always does
        let run = |on: bool| -> (usize, u64) {
            let (c, gt) = (0x3C, 0xC3);
            let copy = Pop { pause: Some((gt, on)), count: 4, ..Pop::new(1, SENSES, c) };
            let g = TagGenome { tokens: vec![Token::Channel { tag: c, width: 3, hold: false }, Token::Channel { tag: gt, width: 0, hold: false }, Token::Pop(copy)], next_id: 2 };
            let mut net = g.build(&World::new(1, 4), 1);
            let mut lines = 0;
            for _ in 0..6 {
                net.tick(crate::bitvec::bithistory::AdvanceMode::Clear, 0);
                net.write_input(&BitVector::new(world::SENSE_BITS, Some(u64::MAX)), 0);
                lines += net.read_node(1).count_ones();
            }
            (lines, net.work)
        };
        assert_eq!(run(true), (0, 0));
        let (lines, work) = run(false);
        assert!(lines > 0 && work >= 4 * 5, "{lines} {work}");
    }

    #[test]
    fn the_self_channel_reports_work_size_and_time() {
        // C copies the self channel (a wire reads it a frame back); a population of 4 kernels on
        // the senses gives it some work to report
        let c = 0x3C;
        let look = Pop { cover: true, ..Pop::wire(1, SELF, c) };
        let busy = Pop { count: 4, ..Pop::new(2, SENSES, ACT) };
        let g = TagGenome { tokens: vec![Token::Channel { tag: c, width: 2, hold: false }, Token::Pop(look), Token::Pop(busy)], next_id: 3 };
        let ch = Channels::of(&g.tokens);
        assert_eq!(ch.input(SELF).node, ch.self_index());
        assert!((0..=255u8).all(|t| ch.output(t).node != ch.self_index()), "nothing writes the self channel");
        let mut net = g.build(&World::new(1, 4), 1);
        for _ in 0..6 {
            net.tick(crate::bitvec::bithistory::AdvanceMode::Clear, 0);
        }
        let w = net.read_node(1).as_words().to_vec();
        // the frame read was written at step 4 (0-based), about step 3: work 4 words wired (the
        // senses are empty, so the population is idle and costs nothing) -> 3 lines; 4 kernels
        // -> 3 lines; 4 steps so far; 2 edges ran -> 2 lines
        assert_eq!(w, vec![0b111, 0b111, 4, 0b11]);
    }

    #[test]
    fn a_circuit_copies_whole_and_moves_between_genomes() {
        // channel R (0x3C): the eye copied in, predictions XORed out of it, and R wired to the act
        let r = 0x3C;
        let copy = Pop { field: 3, ..Pop::wire(1, SENSES, r) };
        let predict = Pop { lag: 2, op: KernelOp::Xor, grow: true, ..Pop::new(2, SENSES, r) };
        let out = Pop::wire(3, r, ACT);
        let other = Pop::new(4, SENSES, ACT);
        let g = TagGenome { tokens: vec![Token::Channel { tag: r, width: 3, hold: false }, Token::Pop(copy), Token::Pop(predict), Token::Pop(out), Token::Pop(other)], next_id: 5 };
        assert_eq!(g.circuit(1), vec![0, 1, 2, 3], "R and the three populations that touch it");
        let mut rng = StdRng::seed_from_u64(1);
        let mut d = g.clone();
        d.copy_circuit(1, None, &mut rng);
        assert_eq!((d.channels(), d.tokens.len(), d.graph().edge_count()), (2, 9, 7));
        // the copy is private: its populations name its own channel, with fresh ids
        let Token::Channel { tag: fresh, .. } = d.tokens[5] else { panic!() };
        assert_ne!(fresh, r);
        let pops: Vec<&Pop> = d.tokens[6..].iter().map(|t| if let Token::Pop(p) = t { p } else { panic!() }).collect();
        assert!(pops[0].to == fresh && pops[1].to == fresh && pops[2].from == fresh);
        assert!(pops.iter().all(|p| p.id >= 5));
        // a plasmid: the circuit moves into a genome that has none of it
        let mut h = TagGenome { tokens: vec![Token::Pop(Pop::new(1, SENSES, ACT))], next_id: 2 };
        h.take_plasmid(&g, 1, &mut rng);
        assert_eq!((h.channels(), h.tokens.len(), h.graph().edge_count()), (1, 5, 4));
    }

    #[test]
    fn confirmed_predicting_kernels_replicate() {
        // the residue circuit (an eye copy ORed into R, XOR predictions from two frames back) on a
        // repeating cycle: its kernels come true, and with replication they copy themselves
        let run = |replicate: u8| -> usize {
            let r = 0x3C;
            let copy = Pop { field: 3, ..Pop::wire(1, SENSES, r) };
            let predict = Pop { lag: 2, field: 3, op: KernelOp::Xor, grow: true, share: true, replicate, cap: 64, ..Pop::new(2, SENSES, r) };
            let g = TagGenome { tokens: vec![Token::Channel { tag: r, width: 3, hold: false }, Token::Pop(copy), Token::Pop(predict)], next_id: 3 };
            let mut net = g.build(&World::new(1, 4), 1);
            let mut rng = StdRng::seed_from_u64(2);
            let codes: Vec<BitVector> = (0..3).map(|_| world::clustered_code(world::CH_BITS, 16, &mut rng)).collect();
            for t in 0..300 {
                net.tick(crate::bitvec::bithistory::AdvanceMode::Clear, 0);
                net.write_input(&codes[t % 3], 0);
            }
            net.edges[1].group.kernel_count()
        };
        let (plain, copied) = (run(0), run(2));
        assert!(copied > plain, "{plain} -> {copied}");
    }

    #[test]
    fn a_port_reads_one_sense_and_writes_one_effector() {
        // C gets the eye only (a covering wire from the eye port); the hand gets C (a covering wire
        // into the hand port); the voice is never touched
        let c = 0x3C;
        let see = Pop { cover: true, from: EYE, from_port: true, ..Pop::wire(1, EYE, c) };
        let act = Pop { cover: true, to: HAND, to_port: true, ..Pop::wire(2, c, HAND) };
        let g = TagGenome { tokens: vec![Token::Channel { tag: c, width: 3, hold: false }, Token::Pop(see), Token::Pop(act)], next_id: 3 };
        let mut net = g.build(&World::new(1, 4), 1);
        let w = world::CH_BITS / 64;
        let mut senses = vec![u64::MAX; world::SENSE_BITS / 64];
        for x in &mut senses[world::EYE * w..(world::EYE + 1) * w] {
            *x = 0;
        }
        senses[world::EYE * w] = 0b1011;
        for _ in 0..3 {
            net.tick(crate::bitvec::bithistory::AdvanceMode::Clear, 0);
            net.write_input(&BitVector::from_words(senses.clone()), 0);
        }
        // C holds the eye's word (nothing from the other senses, which were all on)
        assert_eq!(net.read_node(1).count_ones(), 3);
        let a = net.read_output().as_words().to_vec();
        assert_eq!(a[world::HAND * w], 0b1011);
        assert!(a[world::VOICE * w..(world::VOICE + 1) * w].iter().all(|&x| x == 0));
    }

    #[test]
    fn a_population_plasmid_moves_its_circuit_or_itself() {
        let r = 0x3C;
        let copy = Pop { field: 3, ..Pop::wire(1, SENSES, r) };
        let out = Pop::wire(2, r, ACT);
        let reflex = Pop { count: 4, cover: true, from: EYE, from_port: true, to: HAND, to_port: true, ..Pop::new(3, EYE, HAND) };
        let donor = TagGenome { tokens: vec![Token::Channel { tag: r, width: 3, hold: false }, Token::Pop(copy), Token::Pop(out), Token::Pop(reflex)], next_id: 4 };
        let mut rng = StdRng::seed_from_u64(1);
        // a population touching R brings R's circuit (R and both its populations)
        let mut g = TagGenome { tokens: vec![], next_id: 1 };
        let moved = g.take_population_plasmid(&donor, 0, &mut rng);
        assert_eq!((moved.len(), g.channels(), g.populations()), (3, 1, 2));
        // a population on ports only moves alone, with a new id
        let mut h = TagGenome { tokens: vec![], next_id: 1 };
        let moved = h.take_population_plasmid(&donor, 2, &mut rng);
        assert_eq!((moved.len(), h.channels(), h.populations()), (1, 0, 1));
        let Token::Pop(p) = &h.tokens[0] else { panic!() };
        assert!(p.from_port && p.to_port && p.id >= 4);
    }

    #[test]
    fn a_capped_plasmid_takes_the_chosen_population_and_its_neighbours() {
        let r = 0x3C;
        // five populations touch R; the third (k = 2) is chosen
        let mut t = vec![Token::Channel { tag: r, width: 3, hold: false }];
        for id in 1..=5 {
            t.push(Token::Pop(Pop { count: id as u8, ..Pop::wire(id, SENSES, r) }));
        }
        let donor = TagGenome { tokens: t, next_id: 6 };
        let mut rng = StdRng::seed_from_u64(1);
        let mut g = TagGenome { tokens: vec![], next_id: 1 };
        let moved = g.take_population_plasmid_upto(&donor, 2, 3, &mut rng);
        // the channel and three populations: the chosen one (count 3) and one either side
        let counts: Vec<u8> = moved.iter().filter_map(|t| if let Token::Pop(p) = t { Some(p.count) } else { None }).collect();
        assert_eq!((g.channels(), counts), (1, vec![2, 3, 4]));
        // no limit: the whole circuit
        let mut h = TagGenome { tokens: vec![], next_id: 1 };
        assert_eq!(h.take_population_plasmid_upto(&donor, 2, 0, &mut rng).len(), 6);
    }

    #[test]
    fn energy_makes_a_network_rest_when_its_work_outruns_its_ration() {
        use crate::evolve::{live_lesson, Lesson};
        // a wire copying all the senses (32 words a step) into a channel
        let h = 0x3C;
        let copy = Pop { cover: true, ..Pop::wire(1, SENSES, h) };
        let g = TagGenome { tokens: vec![Token::Channel { tag: h, width: 5, hold: false }, Token::Pop(copy.clone())], next_id: 2 };
        let task = Task::Conditioning(Conditioning::Delayed, 40);
        let run = |ration: u64, food: u64| live_lesson(&g, task, Physics::Varied, 1, 0, Lesson { ration, food, ..Lesson::default() });
        let (ev, free) = run(0, 0);
        let steps = ev.len() as u64;
        assert_eq!(free, 32 * steps);
        // a ration of 4 pays an eighth of it: it rests about seven steps in eight (after its
        // starting store of 64 rations)
        let (ev2, fed) = run(4, 0);
        assert_eq!(ev2.len() as u64, steps);
        assert!(fed <= 4 * steps + 4 * 64 + 32, "{fed} of {free}");
        // a network that costs less than its ration never rests
        let cheap = TagGenome { tokens: vec![], next_id: 1 };
        assert_eq!(live_lesson(&cheap, task, Physics::Varied, 1, 0, Lesson { ration: 4, food: 0, ..Lesson::default() }).1, 0);
        // it eats what it scores: the same costly network, babbling as well (a varied voice
        // scores babble in a reading lifetime), runs more of its lifetime with food than without
        let babble = Pop { count: 8, tonic: 1, ..Pop::new(2, SELF, ACT) };
        let g = TagGenome { tokens: vec![Token::Channel { tag: h, width: 5, hold: false }, Token::Pop(copy), Token::Pop(babble)], next_id: 3 };
        let read = Task::Reading(200);
        let fed = |food: u64| live_lesson(&g, read, Physics::Varied, 1, 0, Lesson { ration: 4, food, ..Lesson::default() }).1;
        let (hungry, eating) = (fed(0), fed(4 * 256));
        assert!(eating > 2 * hungry, "eating {eating}, hungry {hungry}");
    }

    #[test]
    fn a_stuck_population_grows_a_trial_connection_and_a_kept_one_joins_the_genome() {
        use crate::bitvec::bithistory::AdvanceMode;
        let c = 0x3C;
        // kernels from the senses into C that never fire (the senses stay empty)
        let p = Pop { count: 2, cover: true, from_port: true, ..Pop::new(1, SENSES, c) };
        let g = TagGenome { tokens: vec![Token::Channel { tag: c, width: 0, hold: false }, Token::Pop(p)], next_id: 2 };
        let run = |explore: bool| {
            let mut w = World::new(1, 4);
            w.explore = explore;
            let mut net = g.build(&w, 1);
            let before = net.edges.len();
            for _ in 0..80 {
                net.tick(AdvanceMode::Clear, 0);
                net.write_input(&BitVector::new(world::SENSE_BITS, Some(0)), 0);
            }
            (before, net.edges.len(), net.notes())
        };
        // without exploration nothing grows; with it the stuck population grows one trial, on its
        // destination (its source is a port), noted with the population's id
        assert_eq!(run(false).1, run(false).0);
        let (before, after, notes) = run(true);
        assert_eq!(after, before + 1);
        assert!(notes.iter().any(|n| n.0 == 1 && n.2 == 2), "{notes:?}");
        // a trial whose predictions came true often enough is added, tagged and marked; marks take
        // half the old one plus 16 a doubling of the population's own predictions that came true
        let mut h = g.clone();
        take_notes(&mut h, &[(1, 100, 0), (1, KEEP, 2)]);
        let pops: Vec<&Pop> = h.pops().collect();
        assert_eq!(pops.len(), 2);
        assert_eq!(pops[0].mark, 7 * 16);
        assert!(pops[1].explored && pops[1].from == c && pops[1].to == c && !pops[1].from_port && pops[1].op == KernelOp::Xor);
        // once kept it is not added again, and marks and the tag survive the text form
        take_notes(&mut h, &[(1, KEEP, 2)]);
        assert_eq!(h.pops().count(), 2);
        let back = TagGenome::from_text(&h.to_text()).unwrap().remove(0);
        assert_eq!(back, h);
        // a genome written before marks reads back unmarked
        let old: String = h.to_text().lines().map(|l| l.split(" mark=").next().unwrap().to_string() + "\n").collect();
        assert!(TagGenome::from_text(&old).unwrap()[0].pops().all(|p| p.mark == 0 && !p.explored));
    }

    #[test]
    fn a_shadow_predicts_a_sense_writes_nothing_and_a_kept_predictor_gets_a_channel_of_its_own() {
        use crate::bitvec::bithistory::AdvanceMode;
        let c = 0x3C;
        let cw = world::CH_BITS / 64;
        // a population from the eye into C, made to explore at once (an accident a step in two);
        // the touch sense follows the eye a step later, so the eye predicts it
        let p = Pop { from_port: true, cover: true, count: 2, accident: 1, ..Pop::new(1, EYE, c) };
        let g = TagGenome { tokens: vec![Token::Channel { tag: c, width: 3, hold: false }, Token::Pop(p)], next_id: 2 };
        let mut w = World::new(1, 4);
        w.explore = true;
        w.shadow = true;
        let mut net = g.build(&w, 1);
        let mut c_before_shadow = 0u32;
        let mut last_eye = 0u64;
        for step in 0..600u64 {
            net.tick(AdvanceMode::Clear, 0);
            if step == 1 {
                c_before_shadow = net.edges.len() as u32;
            }
            let mut s = BitVector::new(world::SENSE_BITS, Some(0));
            let eye = if step % 2 == 0 { 0x0F0Fu64 } else { 0xF0F0u64 };
            s.as_words_mut()[world::EYE * cw] = eye;
            s.as_words_mut()[world::TOUCH * cw] = last_eye;
            last_eye = eye;
            net.write_input(&s, 0);
        }
        let sh: Vec<&crate::program::neurocomp::RuntimeEdgeGroup> = net.edges.iter().filter(|e| e.explorer == 3).collect();
        assert_eq!(sh.len(), 1, "one shadow (grown at step {c_before_shadow})");
        assert!(sh[0].silent);
        let notes = net.notes();
        let touch = notes.iter().find(|n| n.0 == 1 && n.2 == 19).copied();
        assert!(touch.is_some_and(|n| n.1 >= KEEP * 4), "{notes:?}");
        // kept: a predictor from the eye into a new channel, learning against the touch, and the
        // first population still writes C
        let mut h = g.clone();
        take_notes(&mut h, &[touch.unwrap()]);
        let pops: Vec<&Pop> = h.pops().collect();
        assert_eq!(pops.len(), 2);
        assert!(pops[1].explored && pops[1].teacher == Some((TOUCH, 1)) && pops[1].from_port && pops[1].from == EYE);
        assert_eq!(h.channels(), 2);
        let ch = Channels::of(&h.tokens);
        assert_eq!(ch.to(pops[0]).map(|x| x.node), Some(1));
        assert_eq!(ch.to(pops[1]).map(|x| x.node), Some(2));
        assert_eq!(TagGenome::from_text(&h.to_text()).unwrap().remove(0), h);
        // and the kept predictor writes its prediction of the touch into its channel
        let mut net = h.build(&World::new(1, 4), 1);
        let mut last_eye = 0u64;
        let mut wrote = 0;
        for step in 0..400u64 {
            net.tick(AdvanceMode::Clear, 0);
            wrote += (net.read_node(2).count_ones() > 0) as u32;
            let mut s = BitVector::new(world::SENSE_BITS, Some(0));
            let eye = if step % 2 == 0 { 0x0F0Fu64 } else { 0xF0F0u64 };
            s.as_words_mut()[world::EYE * cw] = eye;
            s.as_words_mut()[world::TOUCH * cw] = last_eye;
            last_eye = eye;
            net.write_input(&s, 0);
        }
        assert!(wrote > 100, "{wrote}");
    }

    #[test]
    fn with_pause_draws_new_populations_are_sometimes_paused_on_a_port() {
        let mut rng = StdRng::seed_from_u64(9);
        let pops: Vec<Pop> = (0..400u32).map(|i| { let mut p = Pop::new(i, 0, 0); draw_pause(&mut p, &mut rng, true, true); p }).collect();
        // off, nothing is drawn (and nothing is taken from the stream)
        let mut a = StdRng::seed_from_u64(3);
        let mut q = Pop::new(1, 0, 0);
        draw_pause(&mut q, &mut a, false, true);
        assert!(q.pause.is_none() && a.r#gen::<u64>() == StdRng::seed_from_u64(3).r#gen::<u64>());
        let paused = pops.iter().filter(|p| p.pause.is_some()).count();
        let on_port = pops.iter().filter(|p| p.pause.is_some() && p.pause_at > 0).count();
        assert!((60..140).contains(&paused), "{paused}");
        assert!(on_port * 3 > paused, "{on_port} of {paused}");
    }

    #[test]
    fn an_edge_built_on_a_hint_runs_along_the_hinted_path() {
        let mut rng = StdRng::seed_from_u64(4);
        let c = 0x3C;
        // a population from the eye into C whose input predicts the affect sense: new edges go eye
        // -> affect (written) or affect -> C, half and half; their kind is their own
        let h = Pop { from_port: true, hint: Some((AFFECT, 1)), ..Pop::new(1, EYE, c) };
        let (mut fwd, mut back) = (0, 0);
        for i in 0..200 {
            let mut q = Pop { op: KernelOp::Xor, wire: i % 2 == 0, ..Pop::new(2, 0, 0) };
            on_hint(&mut q, &h, [Some(false), Some(false)], &mut rng);
            assert_eq!(q.op, KernelOp::Xor);
            if (q.from, q.from_port, q.to, q.to_port, q.to_input) == (EYE, true, AFFECT, true, true) {
                fwd += 1;
            } else if (q.from, q.from_port, q.to, q.to_port) == (AFFECT, true, c, false) {
                back += 1;
            } else {
                panic!("{q:?}");
            }
        }
        assert!(fwd > 60 && back > 60, "{fwd} {back}");
        // a hint at the hand (an output): eye -> hand, or the hand (as it was a step ago) -> C,
        // which feeds back (nothing here writes the hand, so it is the highest level)
        let h = Pop { from_port: true, hint: Some((HAND, 2)), ..Pop::new(1, EYE, c) };
        let g = TagGenome { tokens: vec![Token::Channel { tag: c, width: 3, hold: false }, Token::Pop(h.clone())], next_id: 2 };
        assert_eq!(g.hint_feedback(&h), [Some(false), Some(true)]);
        let (mut fwd, mut back) = (0, 0);
        for _ in 0..100 {
            let mut q = Pop::new(2, 0, 0);
            on_hint(&mut q, &h, g.hint_feedback(&h), &mut rng);
            match (q.from, q.from_port, q.from_output, q.to, q.to_port, q.to_input) {
                (EYE, true, false, HAND, true, false) => fwd += 1,
                (HAND, true, true, t, false, false) if t == c => back += 1,
                x => panic!("{x:?}"),
            }
        }
        assert!(fwd > 10 && back > 60, "{fwd} {back}");
        // read, it is the act's hand channel
        let mut g = g;
        let mut q = Pop::new(2, HAND, c);
        q.from_port = true;
        q.from_output = true;
        g.tokens.push(Token::Pop(q));
        let ch = Channels::of(&g.tokens);
        let Token::Pop(q) = &g.tokens[2] else { unreachable!() };
        assert_eq!(ch.from(q).map(|p| ch.name(p)), Some("hand".to_string()));
        assert_eq!(TagGenome::from_text(&g.to_text()).unwrap().remove(0), g);
    }

    #[test]
    fn a_hint_edge_goes_the_way_that_feeds_back_three_times_in_four() {
        let mut rng = StdRng::seed_from_u64(5);
        let (a, b) = (0x3C, 0xC3);
        // eye -> A -> B -> hand, and the population eye -> A predicts B: forward (eye -> B) runs
        // up, from level 0 to level 2; back (B -> A) closes the loop A -> B -> A
        let h = Pop { from_port: true, hint: Some((b, 0)), ..Pop::new(1, EYE, a) };
        let g = TagGenome {
            tokens: vec![
                Token::Channel { tag: a, width: 3, hold: false },
                Token::Channel { tag: b, width: 3, hold: false },
                Token::Pop(h.clone()),
                Token::Pop(Pop::new(2, a, b)),
                Token::Pop(Pop { to_port: true, ..Pop::new(3, b, HAND) }),
            ],
            next_id: 4,
        };
        assert_eq!(g.hint_feedback(&h), [Some(false), Some(true)]);
        let back = (0..400)
            .filter(|_| {
                let mut q = Pop::new(9, 0, 0);
                on_hint(&mut q, &h, g.hint_feedback(&h), &mut rng);
                (q.from, q.from_port, q.to, q.to_port) == (b, false, a, false)
            })
            .count();
        assert!((260..340).contains(&back), "{back}");
        // B -> hand hinted at the eye: forward (B -> eye, written) runs down to the senses; back
        // (eye -> hand) does not
        let h = Pop { to_port: true, hint: Some((EYE, 1)), ..Pop::new(3, b, HAND) };
        assert_eq!(g.hint_feedback(&h), [Some(true), Some(false)]);
    }

    #[test]
    fn an_inhibit_input_names_a_vector_and_survives_the_text_form() {
        let c = 0x3C;
        let held = Pop { from_port: true, inhibitor: Some(TOUCH), inhibitor_at: 1, ..Pop::new(1, EYE, c) };
        let g = TagGenome { tokens: vec![Token::Channel { tag: c, width: 3, hold: false }, Token::Pop(held)], next_id: 2 };
        assert_eq!(TagGenome::from_text(&g.to_text()).unwrap().remove(0), g);
        let graph = g.graph();
        let spec = graph.edge_indices().map(|e| graph[e].clone()).next().unwrap();
        let cw = world::CH_BITS / 64;
        assert_eq!(spec.inhibitor, Some((0, Some((world::TOUCH * cw, cw)))));
    }

    #[test]
    fn a_heat_input_names_a_vector_and_survives_the_text_form() {
        let c = 0x3C;
        // a population of eye -> C warmed by the affect sense, another cooled by C itself
        let warm = Pop { from_port: true, heat: Some((AFFECT, true)), heat_at: 1, ..Pop::new(1, EYE, c) };
        let cool = Pop { heat: Some((c, false)), ..Pop::new(2, c, c) };
        let g = TagGenome { tokens: vec![Token::Channel { tag: c, width: 3, hold: false }, Token::Pop(warm), Token::Pop(cool)], next_id: 3 };
        assert_eq!(TagGenome::from_text(&g.to_text()).unwrap().remove(0), g);
        let graph = g.graph();
        let specs: Vec<_> = graph.edge_indices().map(|e| graph[e].clone()).collect();
        let cw = world::CH_BITS / 64;
        assert_eq!(specs[0].heat, Some((0, Some((world::AFFECT * cw, cw)), true)));
        assert_eq!(specs[1].heat, Some((1, None, false)));
        // a population without one has none
        let plain = TagGenome { tokens: vec![Token::Pop(Pop::new(3, c, c))], next_id: 4 };
        assert!(!plain.to_text().contains("heat"));
    }

    #[test]
    fn closing_a_loop_adds_an_edge_from_the_end_of_a_path_back_to_its_start() {
        let (a, b) = (0x3C, 0xC3);
        // eye -> A -> B -> hand: no loop
        let g = TagGenome {
            tokens: vec![
                Token::Channel { tag: a, width: 3, hold: false },
                Token::Channel { tag: b, width: 3, hold: false },
                Token::Pop(Pop { from_port: true, ..Pop::new(1, EYE, a) }),
                Token::Pop(Pop::new(2, a, b)),
                Token::Pop(Pop { to_port: true, ..Pop::new(3, b, HAND) }),
            ],
            next_id: 4,
        };
        assert_eq!(g.loops().0, 0);
        let mut rng = StdRng::seed_from_u64(6);
        // from A -> B, one step on: B -> hand, so the new edge reads the hand and writes A
        let mut h = g.clone();
        assert!(h.close_loop(3, 1, Pop { wire: true, ..Pop::new(9, 0, 0) }, &mut rng));
        let Token::Pop(q) = &h.tokens[4] else { panic!() };
        assert_eq!((q.from, q.from_port, q.from_output, q.to, q.to_port, q.to_input, q.wire), (HAND, true, true, a, false, false, true));
        assert_eq!(h.loops().0, 3);
        // from eye -> A, the whole way: hand -> eye (the network senses its own move)
        let mut h = g.clone();
        assert!(h.close_loop(2, 3, Pop::new(9, 0, 0), &mut rng));
        let Token::Pop(q) = &h.tokens[3] else { panic!() };
        assert_eq!((q.from, q.from_output, q.to, q.to_port, q.to_input), (HAND, true, EYE, true, true));
        assert_eq!(h.loops().0, 4);
        // no steps: B back to A
        let mut h = g.clone();
        assert!(h.close_loop(3, 0, Pop::new(9, 0, 0), &mut rng));
        let Token::Pop(q) = &h.tokens[4] else { panic!() };
        assert_eq!((q.from, q.from_port, q.to, q.to_port), (b, false, a, false));
        assert_eq!(h.loops().0, 2);
        assert_eq!(TagGenome::from_text(&h.to_text()).unwrap().remove(0), h);
    }

    #[test]
    fn a_hint_is_noted_not_built_and_a_mutated_gate_takes_it() {
        // the hint step on its own (the switch is process-wide, so the parts are tested directly)
        let c = 0x3C;
        let p = Pop { from_port: true, ..Pop::new(1, EYE, c) };
        let mut g = TagGenome { tokens: vec![Token::Channel { tag: c, width: 3, hold: false }, Token::Pop(p)], next_id: 2 };
        let before = g.tokens.len();
        take_notes_with(&mut g, &[(1, KEEP * 4, 19), (1, KEEP * 8, 20)], true);
        // the best of the population's targets (the affect, kind 20) is its hint; no token is added
        assert_eq!(g.tokens.len(), before);
        let Token::Pop(p) = &g.tokens[1] else { unreachable!() };
        assert_eq!(p.hint, Some((AFFECT, 1)));
        assert_eq!(TagGenome::from_text(&g.to_text()).unwrap().remove(0), g);
    }

    #[test]
    fn gates_and_pauses_name_ports_and_a_population_can_write_a_sense() {
        let c = 0x3C;
        let ch = Token::Channel { tag: c, width: 3, hold: false };
        // a learner gated on the affect port, a population paused on the hand, and one writing the eye
        let learner = Pop { from_port: true, to_port: true, gate: Some((AFFECT, true)), gate_at: 1, ..Pop::new(1, EYE, HAND) };
        let paused = Pop { pause: Some((HAND, false)), pause_at: 2, ..Pop::new(2, c, c) };
        let writer = Pop { to_port: true, to_input: true, ..Pop::wire(3, c, EYE) };
        let g = TagGenome { tokens: vec![ch, Token::Pop(learner), Token::Pop(paused), Token::Pop(writer)], next_id: 4 };
        let graph = g.graph();
        let specs: Vec<&EdgeSpec> = graph.edge_weights().collect();
        let cw = world::CH_BITS / 64;
        // the gate is the senses node, on the affect's words only
        assert_eq!(specs[0].gate.map(|x| x.0), Some(0));
        assert_eq!(specs[0].gate_window, Some((world::AFFECT * cw, cw)));
        // the pause is the act node, on the hand's words
        let act = graph.node_count() - 1;
        assert_eq!(specs[1].pause.map(|x| x.0), Some(act));
        assert_eq!(specs[1].pause_window, Some((world::HAND * cw, cw)));
        // the writer's destination is the senses node, at the eye
        let e = graph.edge_indices().nth(2).unwrap();
        assert_eq!(graph.edge_endpoints(e).map(|(_, b)| b.index()), Some(0));
        assert_eq!((specs[2].dst_word, specs[2].dst_len), (world::EYE * cw, cw));
        // the properties survive the text form, and are written only when set
        assert_eq!(TagGenome::from_text(&g.to_text()).unwrap().remove(0), g);
        let plain = TagGenome { tokens: vec![Token::Channel { tag: c, width: 3, hold: false }, Token::Pop(Pop::new(1, c, c))], next_id: 2 };
        assert!(!plain.to_text().contains("gate_at") && !plain.to_text().contains("to_input"));
    }

    #[test]
    fn a_learner_gated_on_the_affect_port_learns_only_after_a_stroke() {
        use crate::bitvec::bithistory::AdvanceMode;
        // the eye on every step; a stroke on some: births (repeat ones, of the act just made) only
        // while the affect port had lines on a frame ago
        let run = |stroke: bool| {
            let babble = Pop { from_port: true, to_port: true, count: 4, tonic: 1, ..Pop::new(1, EYE, HAND) };
            let learner = Pop { from_port: true, to_port: true, cover: true, grow: true, repeat: true, repeat_lag: 1, gate: Some((AFFECT, true)), gate_at: 1, ..Pop::new(2, EYE, HAND) };
            let g = TagGenome { tokens: vec![Token::Pop(babble), Token::Pop(learner)], next_id: 3 };
            let w = World::new(1, 4);
            let mut net = g.build(&w, 1);
            let cw = world::CH_BITS / 64;
            for _ in 0..40 {
                net.tick(AdvanceMode::Clear, 0);
                let mut s = BitVector::new(world::SENSE_BITS, Some(0));
                s.as_words_mut()[world::EYE * cw] = 0xFF;
                if stroke {
                    s.as_words_mut()[world::AFFECT * cw] = 0xF0;
                }
                net.write_input(&s, 0);
            }
            net.edges[1].group.kernel_count()
        };
        assert_eq!(run(false), 0);
        assert!(run(true) > 0);
    }

    #[test]
    fn a_busy_population_grows_a_trial_only_by_accident() {
        use crate::bitvec::bithistory::AdvanceMode;
        let c = 0x3C;
        // tonic kernels that fire every other step on average: never stuck
        let run = |accident: u8| {
            let p = Pop { count: 2, cover: true, from_port: true, tonic: 1, accident, ..Pop::new(1, SENSES, c) };
            let g = TagGenome { tokens: vec![Token::Channel { tag: c, width: 0, hold: false }, Token::Pop(p)], next_id: 2 };
            let mut w = World::new(1, 4);
            w.explore = true;
            let mut net = g.build(&w, 1);
            let before = net.edges.len();
            for _ in 0..80 {
                net.tick(AdvanceMode::Clear, 0);
                net.write_input(&BitVector::new(world::SENSE_BITS, Some(0)), 0);
            }
            net.edges.len() - before
        };
        assert_eq!(run(0), 0);
        // an accident a step in two: one trial, grown once a lifetime
        assert_eq!(run(1), 1);
        // and the rate is a gene, written only when set
        let p = Pop { accident: 12, ..Pop::new(1, SENSES, c) };
        let g = TagGenome { tokens: vec![Token::Channel { tag: c, width: 0, hold: false }, Token::Pop(p)], next_id: 2 };
        assert!(g.to_text().contains(" accident=12"));
        assert_eq!(TagGenome::from_text(&g.to_text()).unwrap().remove(0), g);
    }

    #[test]
    fn expecting_zeros_is_a_gene_written_only_when_set() {
        let r = 0x3C;
        let look = Pop { from_port: true, expect: true, ..Pop::new(1, AFFECT, r) };
        let out = Pop::wire(2, r, ACT);
        let g = TagGenome { tokens: vec![Token::Channel { tag: r, width: 3, hold: false }, Token::Pop(look), Token::Pop(out)], next_id: 3 };
        let text = g.to_text();
        assert_eq!(text.matches("expect=1").count(), 1);
        assert_eq!(TagGenome::from_text(&text).unwrap().remove(0), g);
        // the graph's edge expects; a wire never does
        let graph = g.graph();
        let specs: Vec<bool> = graph.edge_weights().map(|e| e.expect).collect();
        assert_eq!(specs.iter().filter(|&&e| e).count(), 1);
        // a genome without it reads and writes as before
        let mut h = g.clone();
        if let Token::Pop(p) = &mut h.tokens[1] { p.expect = false; }
        assert!(!h.to_text().contains("expect"));
    }

    #[test]
    fn loops_count_self_loops_and_cycles_through_channels() {
        let (a, b, c) = (0x3C, 0xC3, 0x5A);
        let ch = |t: u8| Token::Channel { tag: t, width: 0, hold: false };
        // senses -> A, A -> B, B -> A (a loop of two), C -> C (a self-loop), B -> act
        let pops = [Pop::wire(1, SENSES, a), Pop::wire(2, a, b), Pop::wire(3, b, a), Pop::wire(4, c, c), Pop::wire(5, b, ACT)];
        let mut tokens = vec![ch(a), ch(b), ch(c)];
        tokens.extend(pops.into_iter().map(Token::Pop));
        let g = TagGenome { tokens, next_id: 6 };
        assert_eq!(g.loops(), (3, 1, 2, 5));
    }

    #[test]
    fn a_blind_control_reads_the_self_channel_for_the_senses() {
        let r = 0x3C;
        let look = Pop { from_port: true, ..Pop::wire(1, EYE, r) };
        let out = Pop::wire(2, r, ACT);
        let g = TagGenome { tokens: vec![Token::Channel { tag: r, width: 3, hold: false }, Token::Pop(look), Token::Pop(out)], next_id: 3 };
        assert_eq!(g.ports().0, vec!["eye".to_string()]);
        let b = g.blind();
        assert_eq!(b.ports().0, vec!["M".to_string()]);
        // the channel's own reader is untouched
        assert_eq!(b.tokens[2], g.tokens[2]);
    }

    #[test]
    fn a_genome_survives_text_and_back() {
        let mut rng = StdRng::seed_from_u64(11);
        let mut gs = vec![TagGenome::context_seed(), TagGenome::act_seed()];
        let mut g = TagGenome { tokens: vec![], next_id: 1 };
        for _ in 0..300 {
            g = g.mutate(&mut rng);
        }
        gs.push(g);
        let text: String = gs.iter().map(|g| g.to_text()).collect();
        let back = TagGenome::from_text(&text).unwrap();
        assert_eq!(back, gs);
    }

    #[test]
    fn a_tonic_population_babbles_and_a_responsive_one_does_not() {
        let t = Task::Conditioning(Conditioning::Instrumental, 30);
        let mut g = TagGenome::bare_seed();
        let acts = |g: &TagGenome| live(g, t, Physics::Varied, 3).iter().filter(|e| e.moved.is_some() || e.said.is_some()).count();
        assert!(acts(&g) > 0);
        if let Token::Pop(p) = &mut g.tokens[0] {
            p.tonic = 0;
        }
        assert_eq!(acts(&g), 0);
    }
}
