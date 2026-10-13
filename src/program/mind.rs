//! A mind: senses in, acts out, and every reward its own.
//!
//! `Mind::step(senses) -> acts` is the whole interface. Inside, one pipeline runs every step,
//! whatever is going on outside (a caregiver babbling back, a lesson, nothing); nothing outside
//! says what phase it is, what to attend to, when to speak, or what was right.
//!
//! 1. **Learning what came.** The cortex's columns learn from their last state what was sensed
//!    now (self-supervised: what the effectors' senses received next). The motor areas learn
//!    each act's result (inverse and forward models: no babbling phase handed the world's
//!    answers).
//! 2. **The hippocampus,** one step over the whole cortical state: every area's cells, sensed or
//!    imagined. Nobody tells it what was heard; attention outward (an input changed), the
//!    voice's efference copy (one's own sound) and the cortex's surprise set what it does.
//! 3. **Sensing:** every area's column, the motion stream (what changed in view), the mirror
//!    (seen acts evoke how they would feel), the association columns grown by co-activity, the
//!    cerebellum, and the thalamic relay's mix (the cortex's expectation of what comes next).
//! 4. **Valuing what came** (innate valuations, the only source of dopamine):
//!    - *its own effect:* after its own act, as surprising as the senses find what came
//!      (contingent change, fading as it comes to be expected; Redgrave & Gurney);
//!    - *a response:* someone there (presence) and a sound from outside soon after its own
//!      sound made aloud (contingent replies, Goldstein et al. 2003);
//!    - *affective touch:* a stroke on the skin, a sense of its own whose receptor line carries its
//!      value (as sweet taste's or pain's does), going through its own cortical area like any sense;
//! 5. **Loops that grow with the cortex** (Alexander, DeLong & Strick 1986): every column with
//!    a layer 5, sensory or association, has its own loop through the basal ganglia, created
//!    when the column is. A column's layer 5 offers its goal (of the cells that fired, the one
//!    whose output has been worth most when followed: dopamine tags goal cells); its loop takes
//!    it in or not. An association column's loop can also gate its content into working memory
//!    (PBWM's stripes). Every motor area has a loop too: its candidates are the goals taken in
//!    that it can plan, and its spontaneously active primitive (pattern generators); it takes
//!    one, and its motor gate releases it to the body or withholds it, and then it is imagined
//!    (the forward model's result, fed back top-down at the next step).

use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};

use crate::bitvec::BitVector;
use crate::fixed::ONE;
use crate::program::amygdala::Amygdala;
use crate::program::claustrum::Claustrum;
use crate::program::ofc::{self as ofc_area, Ofc};
use crate::program::area::fit;
use crate::program::area::{Brain, ColumnId, Kind, MAX_PAIRS};
use crate::program::cerebellar_circuit::CerebellarCircuit;
use crate::program::index_hc::IndexHc;
use crate::program::mask_striatum::MaskStriatum;
use crate::program::prefrontal::WorkingMemory;
use crate::program::thalamic_gate::{Driver, ThalamicRelay};
use crate::program::world::{Acts, Body, Senses, ACTIVE, BITS};

pub const EYE: usize = 0;
pub const EAR: usize = 1;
pub const HAND: usize = 2;
pub const VOICE: usize = 3;
/// seen acts: what changed in view (the motion stream)
pub const ACTS: usize = 4;
pub const PRESENCE: usize = 5;
/// affective touch: a gentle stroke on the skin (C-tactile afferents), its valence a property of
/// the receptor line itself, as sweet taste's or pain's
pub const AFFECT: usize = 6;
/// the sensing areas, each a slot of BITS cells in the hippocampus's pointers
pub const SLOTS: [usize; 6] = [EYE, EAR, HAND, ACTS, PRESENCE, AFFECT];

/// where goal candidates sit in the striatum's input ids (each effector's primitives after them)
const ACT_BASE: usize = 1 << 25;
/// steps back a followed goal's cell still takes dopamine
const GOAL_TRACE: usize = 4;

/// A column's loop channel (taking its goal in); an association column's working-memory gate
/// is the next channel.
fn loop_channel(id: ColumnId) -> usize {
    match id {
        ColumnId::Sense(a) => 20 + 2 * a,
        ColumnId::Assoc(i) => 60 + 2 * i,
    }
}

/// A motor area's loop channel (which candidate) and its motor gate (release or withhold).
fn motor_channels(a: usize) -> (usize, usize) {
    (200 + 2 * a, 201 + 2 * a)
}

pub fn overlap(a: &BitVector, b: &BitVector) -> u32 {
    a.as_words().iter().zip(b.as_words()).map(|(x, y)| (x & y).count_ones()).sum()
}

pub fn ones(v: &BitVector) -> Vec<usize> {
    let mut out = Vec::new();
    for (wi, &x) in v.as_words().iter().enumerate() {
        let mut x = x;
        while x != 0 {
            out.push(wi * 64 + x.trailing_zeros() as usize);
            x &= x - 1;
        }
    }
    out
}

fn zero() -> BitVector {
    BitVector::new(BITS, Some(0))
}

fn frames(fs: &[&BitVector]) -> BitVector {
    let mut w = Vec::new();
    for f in fs {
        w.extend_from_slice(f.as_words());
    }
    BitVector::from_words(w)
}

fn or(x: &BitVector, y: &BitVector) -> BitVector {
    let mut v = x.clone();
    for (w, z) in v.as_words_mut().iter_mut().zip(y.as_words()) {
        *w |= z;
    }
    v
}

fn same(a: &BitVector, b: &BitVector) -> bool {
    a.as_words() == b.as_words()
}

/// A fixed projection of cells into one frame (8 bits per cell).
fn project(cells: &[usize]) -> BitVector {
    let mut v = zero();
    for &c in cells {
        for i in 0..8u64 {
            let mut h = (c as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            h ^= h >> 31;
            v.bit_set((h % BITS as u64) as usize);
        }
    }
    v
}

/// Cells of several areas as one pointer vector: area `a`'s cells at its slot.
pub fn multi(parts: &[(usize, &BitVector)]) -> BitVector {
    let mut v = BitVector::new(SLOTS.len() * BITS, Some(0));
    for &(a, c) in parts {
        if let Some(slot) = SLOTS.iter().position(|&x| x == a) {
            for b in ones(c) {
                v.bit_set(slot * BITS + b);
            }
        }
    }
    v
}

/// Area `a`'s part of a pointer vector.
pub fn part(v: &BitVector, a: usize) -> BitVector {
    let mut out = zero();
    if let Some(slot) = SLOTS.iter().position(|&x| x == a) {
        for b in ones(v).into_iter().filter(|&b| b / BITS == slot) {
            out.bit_set(b % BITS);
        }
    }
    out
}

/// Association column `i`'s slot in the association-level index (pairs at their index, the full
/// area at the last slot, so a slot never moves as pairs grow).
fn assoc_slot(i: usize, pairs: usize) -> usize {
    if i < pairs {
        i
    } else {
        MAX_PAIRS
    }
}

/// Where each kind of input sits in the striatum's input ids (stable across loops and the critic).
fn column_offset(id: ColumnId) -> usize {
    match id {
        ColumnId::Sense(a) => a * BITS,
        ColumnId::Assoc(i) => (8 + i) * BITS,
    }
}
const WM_OFFSET: usize = 60 * BITS;
const RECALL_OFFSET: usize = 41 * BITS;
const AMY_OFFSET: usize = 80 * BITS;
const AMY_VALUE_OFFSET: usize = 120 * BITS;
const EFFERENCE_OFFSET: usize = 130 * BITS;
const OFC_VALUE_OFFSET: usize = 140 * BITS;
const SUSTAINED_OFFSET: usize = 150 * BITS;
/// the orbitofrontal area's state Z, and what it holds
const OFC_Z_OFFSET: usize = 162 * BITS;
const OFC_HELD_OFFSET: usize = 164 * BITS;
/// the orbitofrontal area's hold loop (hold the state or not)
const OFC_HOLD_CHANNEL: usize = 300;
/// the value of an imagined outcome, as the motor gate sees it (16 bits for good, 16 for bad)
const VALUE_BASE: usize = ACT_BASE + 8 * BITS;

/// What the mind is built with (each switch is a part of the biology, to be ablated).
#[derive(Clone, Debug)]
pub struct Config {
    /// the hippocampus's input and output at the top of the association hierarchy (the full and
    /// pairwise association columns, as the entorhinal cortex's inputs are); recall reaches the
    /// senses only by the columns' feedback. false: it indexes every sensing area directly
    pub hc_assoc: bool,
    /// each loop sees its own cortical territory (corticostriatal topography); false: the whole state
    pub topographic: bool,
    /// the amygdala: fast cue value, attention to predictive cues, the onset trace to the striatum
    pub amygdala: bool,
    /// the cholinergic interneurons learn to pause to cues that came before large dopamine errors
    pub learned_pause: bool,
    /// a reply pays only as far as replies are contingent on its own sounds (a contingency
    /// detector); false: any sound from outside soon after its own sound
    pub contingency: bool,
    /// the motor gate sees the amygdala's value of the act's imagined outcome
    pub imagine_value: bool,
    /// the valuation of its own effect (how far an act's result was from what its forward model
    /// expected: fading per act and outcome as the forward model learns them)
    pub own_effect: bool,
    /// the own-effect signal trains only the motor loops' choices (agency); false: it goes to the
    /// critic with the outcomes (MIND_OWN=critic)
    pub agency: bool,
    /// Pavlovian–instrumental transfer: what the cues now are worth (the amygdala) lowers the motor
    /// loops' brake (or raises it, if they are worth less than nothing)
    pub pit: bool,
    /// the valuation of a response
    pub response: bool,
    /// the critic sees the amygdala's value cells firing (MIND_AMY_CRITIC=0 off)
    pub amy_critic: bool,
    /// an orbitofrontal / prelimbic learner of what an act leads to (cues and the efference copy
    /// of the act just done); its values reach the critic and the motor loops, and the efference
    /// copy itself the motor loops (MIND_EFFERENCE=0 off)
    pub efference: bool,
    /// a tonic level from the average outcomes: exploration rises when reward is scarce or
    /// falling, and acting costs `effort` (MIND_TONIC=0 off; MIND_EFFORT, `Q16`, default 1/32)
    pub tonic: bool,
    pub effort: i32,
    /// withholding is recorded as a choice, so the no-go side learns (MIND_NOGO=0 off)
    pub nogo: bool,
    /// the motor loops see the senses' sustained state, and valued groups take a larger share of
    /// the striatum's context (MIND_SALIENCE=0 off)
    pub salience: bool,
    /// the senses' layer 4 cover every part of their input (MIND_COVERAGE=0 off)
    pub coverage: bool,
    /// the critic also sees recall and working memory (MIND_CRITIC_ALL=1; default off: they
    /// change every step whatever happens, and made errors where nothing happened)
    pub critic_all: bool,
    /// no-go plasticity symmetric (a burst removes as many no-go cells as a dip adds, new ones
    /// only where none fired), and withholds recorded only at the motor gates (MIND_NOGO_SYM=0 off)
    pub nogo_symmetric: bool,
    /// homeostasis: a loop's go and no-go pools kept within twice each other (MIND_BALANCE=0 off)
    pub balance: bool,
    /// opportunity cost: an act costs what acting usually brings, if more than the effort
    /// (MIND_OPPORTUNITY=0 off)
    pub opportunity: bool,
    /// the orbitofrontal learner keeps a baseline value per act, never extinguished, its other
    /// cells learning what the cues add (MIND_ACT_BASELINE=0 off)
    pub act_baseline: bool,
    /// the opportunity cost is each act's own usual worth (its orbitofrontal baseline), not one
    /// average for every act (MIND_PER_ACT=0 off)
    pub per_act: bool,
    /// separate credit per act: its outcome against its usual worth goes to its own loop's and
    /// gate's choices (MIND_ACT_CREDIT=0 off)
    pub act_credit: bool,
    /// the amygdala's and orbitofrontal learner's value cells form and fire only on attended
    /// cues (and one's own acts) (MIND_ATTEND_GATE=1 on; off by default: it broke tone learning)
    pub attend_gate: bool,
    /// the amygdala's new value cells each sample one cue (one area's onsets at one age), not the
    /// whole trace; the orbitofrontal learner keeps its conjunctions of act and cues
    /// (MIND_ELEMENTAL=0 off)
    pub elemental: bool,
    /// focused cells: the striatum's contexts keep each input group in a region of its own, and
    /// its new cells take two groups (the candidate act and one other); the orbitofrontal
    /// learner's new cells pair one act with one cue (MIND_FOCUSED=0 off)
    pub focused: bool,
    /// per-act credit against what the loops expected of the act there (their q), not the act's
    /// average worth (MIND_CREDIT_OWN=0 off)
    pub credit_own: bool,
    /// steps back the dopamine error reaches the striatum's choices (its eligibility trace;
    /// MIND_WINDOW, default 2)
    pub window: usize,
    /// associability (Pearce–Hall): the amygdala's per-cue surprise sets a motor choice's
    /// learning rate and where its new cells draw from (MIND_ALPHA=0 off)
    pub alpha: bool,
    /// the orbitofrontal cortex as a cortical area: a column over the cues, the act, the outcome
    /// and the hippocampus's context, its layer 2/3's activity the state Z (to the striatum and
    /// the critic), an outcome layer predicting the next touch, and a loop that chooses to hold
    /// the state (MIND_OFC_COLUMN=0 off)
    pub ofc_column: bool,
    /// the orbitofrontal value cells read the area's state Z (and the act), not the raw cue
    /// trace (MIND_OFC_VALUE_Z=0 off; needs the column)
    pub ofc_value_z: bool,
    /// the orbitofrontal area splits on outcome surprise: an outcome badly mispredicted after a
    /// familiar input, with the cues associable, makes it hold its state now (a new context)
    /// (MIND_OFC_SPLIT=0 off)
    pub ofc_split: bool,
    /// frustration (Amsel): an act that brings less than its loops expected adds the shortfall
    /// (one that brings more takes it away), fading by 1/16 a step; it lowers the motor loops'
    /// brake (the extinction burst) and widens their noise (extinction-induced variability), so
    /// when what paid stops paying the mind tries harder and more variously before it gives up
    /// (MIND_FRUSTRATION=0 off)
    pub frustration: bool,
    /// persistence in every column (intrinsic timescales grow up the hierarchy): the pairwise
    /// association columns' activity outlasts their input a little (τ 1), the full one's and the
    /// orbitofrontal area's longer (τ 3, 4); a loop's "take in" refreshes its column's own trace.
    /// This replaces the separate working-memory store and the orbitofrontal held copy
    /// (MIND_PERSIST=1 on; off by default, recurrence instead)
    pub persist: bool,
    /// a held population beside the driven code (`Column::set_recurrence`) in the full association
    /// column and the orbitofrontal area (none in the senses or the pairs): loaded from the driven
    /// code when the column is surprised by its input or its loop opens the gate, otherwise
    /// maintained by autoassociative completion. Working memory is the full association column's
    /// held population, the orbitofrontal hold its own; the separate store and copy are not used
    /// (MIND_RECUR=0 off)
    pub recur: bool,
    /// superposed expectation in every column (`Column::superpose`): layer 2/3 expects every next
    /// input its context supports, so a surprise is something not expected at all
    /// (MIND_SUPERPOSE=0 off)
    pub superpose: bool,
    /// a claustrum-like selector decides what the cortex holds: the full association column and
    /// the orbitofrontal area propose what surprised them, and one item a step at most is held,
    /// something known that came unexpectedly and stands out, weighed by the cues'
    /// associability (MIND_CLAUSTRUM=0: every surprise is held)
    pub claustrum: bool,
    /// the top areas (the full association column and the orbitofrontal area) twice as wide as
    /// the rest (4096 bits): room for their layer 4 to recruit cells as they learn to tell more
    /// apart (MIND_WIDE=0 off)
    pub wide: bool,
    /// sleep generalizes as well as compacts: in every cortical layer, cells that predict the
    /// same thing from mostly the same input become one cell on what they share (a more general,
    /// compressed representation) (MIND_GENERALIZE=0 off)
    pub generalize: bool,
    /// recruitment by need in the top areas' layer 4 (the full association column and the
    /// orbitofrontal area): what they do not cover gets cells only when it keeps coming or the
    /// moment is salient (a dopamine error the step before), and sleep recycles
    /// the cells that went unused (MIND_NEED=0 off)
    pub need: bool,
    /// keep the order: every column sends its layer 2/3 activity up in layer 4's coordinates,
    /// each cell at a fixed home and slot, so an output bit keeps its meaning when the cell
    /// behind it is freed and reused (`Column::topo`) (MIND_KEEP_ORDER=0 off)
    pub keep_order: bool,
    /// with `keep_order`, input a column's layer 2/3 did not predict bursts: every slot of each
    /// unpredicted layer 4 bit fires, so what came unexpected weighs more in what the column
    /// sends up (`Column::burst`) (MIND_BURST=0 off)
    pub burst: bool,
    /// adaptation in the association columns' and the orbitofrontal area's layer 4: a cell that
    /// has fired several steps in a row loses ground in the competition for the winners' slots,
    /// so a new input (a light coming on in a familiar scene) wins slots from what has been there
    /// all along (`Pool::adapt`) (MIND_ADAPT=0 off)
    pub adapt: bool,
    /// bounded striatal learning (`MaskStriatum::bounded`): an act's go or no-go cells are
    /// recruited only while its summed prediction in the context (go less no-go firing) does not
    /// already say what came (Rescorla–Wagner's asymptote) (MIND_BOUNDED=0 off)
    pub bounded: bool,
    /// local inhibition in the association columns' and the orbitofrontal area's layer 4: winners
    /// compete within their own input source, each normalized by its own active bits, so many
    /// inputs keep cells side by side in a wide code rather than the busiest crowding the rest
    /// out (`Pool::local`) (MIND_LOCAL=0 off)
    pub local: bool,
    /// sleep compacts the cortical layers (duplicates merged, stale or wrong cells pruned)
    /// (MIND_COMPACT=0 off)
    pub compact: bool,
    /// sleep's replay teaches the cortex (the next replayed row's sound as its goals' target),
    /// and rows the cortex now completes fade faster (MIND_CONSOLIDATE=0 off)
    pub consolidate: bool,
    /// predictive transmission: each sense passes up only what its column did not predict
    /// (MIND_PREDICTIVE=1 on; default off)
    pub predictive: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self { hc_assoc: true, topographic: true, amygdala: true, learned_pause: true, contingency: true, imagine_value: true, own_effect: true, agency: true, pit: true, response: true, amy_critic: true, efference: true, tonic: true, effort: (ONE / 32) as i32, nogo: true, salience: true, coverage: true, critic_all: false, nogo_symmetric: true, balance: true, opportunity: true, act_baseline: true, per_act: true, act_credit: true, attend_gate: false, elemental: true, focused: true, credit_own: true, window: 2, alpha: true, ofc_column: true, ofc_value_z: true, ofc_split: true, frustration: true, persist: false, recur: true, superpose: true, claustrum: true, wide: true, generalize: true, need: true, keep_order: true, burst: true, adapt: true, bounded: true, local: true, compact: true, consolidate: true, predictive: false }
    }
}

impl Config {
    /// The defaults, with any switch turned off by its variable set to 0 (MIND_HC=senses for the
    /// sensory-level index): MIND_TOPO, MIND_AMY, MIND_PAUSE, MIND_CONTINGENCY, MIND_IMAGINE_VALUE,
    /// MIND_OWN (or =critic: into the critic), MIND_PIT, MIND_RESPONSE.
    pub fn from_env() -> Self {
        let on = |k: &str| std::env::var(k).map_or(true, |v| v != "0");
        Self {
            hc_assoc: std::env::var("MIND_HC").map_or(true, |v| v != "senses"),
            topographic: on("MIND_TOPO"),
            amygdala: on("MIND_AMY"),
            learned_pause: on("MIND_PAUSE"),
            contingency: on("MIND_CONTINGENCY"),
            imagine_value: on("MIND_IMAGINE_VALUE"),
            own_effect: on("MIND_OWN"),
            agency: std::env::var("MIND_OWN").map_or(true, |v| v != "critic"),
            pit: on("MIND_PIT"),
            response: on("MIND_RESPONSE"),
            amy_critic: on("MIND_AMY_CRITIC"),
            efference: on("MIND_EFFERENCE"),
            tonic: on("MIND_TONIC"),
            nogo: on("MIND_NOGO"),
            salience: on("MIND_SALIENCE"),
            coverage: on("MIND_COVERAGE"),
            critic_all: std::env::var("MIND_CRITIC_ALL").map_or(false, |v| v == "1"),
            nogo_symmetric: on("MIND_NOGO_SYM"),
            balance: on("MIND_BALANCE"),
            opportunity: on("MIND_OPPORTUNITY"),
            act_baseline: on("MIND_ACT_BASELINE"),
            per_act: on("MIND_PER_ACT"),
            act_credit: on("MIND_ACT_CREDIT"),
            attend_gate: std::env::var("MIND_ATTEND_GATE").map_or(false, |v| v == "1"),
            elemental: on("MIND_ELEMENTAL"),
            focused: on("MIND_FOCUSED"),
            credit_own: on("MIND_CREDIT_OWN"),
            alpha: on("MIND_ALPHA"),
            ofc_column: on("MIND_OFC_COLUMN"),
            ofc_value_z: on("MIND_OFC_VALUE_Z"),
            ofc_split: on("MIND_OFC_SPLIT"),
            frustration: on("MIND_FRUSTRATION"),
            persist: std::env::var("MIND_PERSIST").map_or(false, |v| v == "1"),
            recur: on("MIND_RECUR"),
            superpose: on("MIND_SUPERPOSE"),
            claustrum: on("MIND_CLAUSTRUM"),
            wide: on("MIND_WIDE"),
            generalize: on("MIND_GENERALIZE"),
            need: on("MIND_NEED"),
            keep_order: on("MIND_KEEP_ORDER"),
            burst: on("MIND_BURST"),
            adapt: on("MIND_ADAPT"),
            bounded: on("MIND_BOUNDED"),
            local: on("MIND_LOCAL"),
            window: std::env::var("MIND_WINDOW").ok().and_then(|v| v.parse().ok()).unwrap_or(2),
            compact: on("MIND_COMPACT"),
            consolidate: on("MIND_CONSOLIDATE"),
            predictive: std::env::var("MIND_PREDICTIVE").map_or(false, |v| v == "1"),
            effort: std::env::var("MIND_EFFORT").ok().and_then(|v| v.parse().ok()).unwrap_or((ONE / 32) as i32),
        }
    }
}

/// What the cortex did this step, kept to learn from what comes next.
struct Step {
    /// per association column, its input and expectation; the tuft; per sense, (area, code, expectation)
    ins: Vec<(BitVector, BitVector)>,
    tuft: BitVector,
    sense_ins: Vec<(usize, BitVector, BitVector)>,
    mossy: BitVector,
    ctx: BitVector,
    props: Vec<(u8, Driver, BitVector)>,
    /// the relay's mix: the cortex's expectation of what comes next
    expect: BitVector,
}

/// What the mind did, for an observer (never fed back).
#[derive(Clone, Default, Debug)]
pub struct Report {
    pub spoke: bool,
    pub moved: bool,
    /// a candidate taken into a motor loop and withheld (imagined)
    pub inward: bool,
    /// goals taken into their columns' loops this step
    pub goals_taken: usize,
    /// an act released came from a column's goal (not a primitive)
    pub from_goal: bool,
    /// the dopamine-producing valuations this step, `Q16`: (own effect, response, affective touch, effort)
    pub valued: [i32; 4],
    /// the hippocampus's offer reached the cortex
    pub recalled: bool,
    /// columns with a loop
    pub loops: usize,
    /// the cholinergic interneurons paused (a learned cue)
    pub paused: bool,
    /// the critic's value of this step's state, and the dopamine error closing the last step
    pub value: i32,
    pub delta: i32,
    /// what the amygdala predicted for this step before it came (`Q16`)
    pub amy_predicted: i32,
    /// the tonic exploration noise added this step
    pub tonic: i32,
    /// what an act usually brings (`Q16`)
    pub rate_act: i32,
    /// frustration now (`Q16`)
    pub frustration: i32,
}

pub struct Mind {
    rng: StdRng,
    pub cfg: Config,
    pub brain: Brain,
    cb: CerebellarCircuit,
    relay: ThalamicRelay,
    wm: WorkingMemory,
    pub bg: MaskStriatum,
    pub hc: IndexHc,
    pub amy: Amygdala,
    /// the orbitofrontal / prelimbic outcome learner: the same bit-cell value learner, over the
    /// cues and the efference copy of the act just done ("this act, here, leads to that")
    pub ofc: Amygdala,
    /// the orbitofrontal cortex as an area (its state Z, what it holds, its outcome expectation)
    pub ofc_area: Ofc,
    /// the claustrum-like selector of what the cortex holds
    pub claustrum: Claustrum,
    /// the efference copy of the act sensed this step (features), for the motor loops' view
    efference_now: Vec<u32>,
    /// each motor area, its primitives (the genome's pattern generators), and the sensing area
    /// its acts are sensed by (its result)
    motors: Vec<(usize, Vec<BitVector>, usize)>,
    pons: BitVector,
    /// top-down input for the next step, per area: what is imagined
    topdown: Vec<(usize, BitVector)>,
    /// each sensing area's raw input and layer 4 code at the last step
    prev_in: Vec<BitVector>,
    prev_codes: Vec<BitVector>,
    last_recall: BitVector,
    last: Option<Step>,
    last_expect: BitVector,
    /// the command issued last step per motor area (the efference copy)
    issued: Vec<(usize, BitVector)>,
    acted: usize,
    /// goals followed recently: (column, cell), newest last
    followed: Vec<Vec<(ColumnId, usize)>>,
    /// steps since its own sound was last heard aloud
    since_said: usize,
    /// fast and slow averages of the outcomes (`Q16`): the tonic level
    rate_fast: i32,
    /// frustration (`Q16`): expected outcomes of acts that did not come
    frustration: i32,
    /// the held populations' salience: the cues' associability when each was last loaded (the
    /// full association column's, the orbitofrontal area's), fading by 1/16 a step, and the
    /// loads seen so far
    held_tag: [u32; 2],
    held_loads: [u64; 2],
    rate_slow: i32,
    /// what an act usually brings (`Q16`), and the acts whose outcome window is still open
    /// (steps left, outcome so far)
    rate_act: i32,
    /// (steps left, outcome so far, motor area, the act's usual worth when done)
    act_windows: Vec<(u8, i32, usize, i32, Option<(i32, i32)>, i32)>,
    /// per motor area, the loop's and gate's q (cells) for the act released last: what the
    /// loops expected of it there
    chosen_q: Vec<(usize, i32, i32)>,
    /// the contingency detector (`Q16`): how often a sound from outside comes within two steps
    /// after its own sound, and otherwise
    p_after: i32,
    p_else: i32,
    /// a primitive of each effector is spontaneously active 1 step in this many
    pub spontaneous: u32,
    pub report: Report,
    /// sleep's compaction so far: (merged, pruned) cortical cells
    pub compacted: (usize, usize),
    /// the step before was salient (`Config::need`)
    salient: bool,
    /// cells recycled by sleep in the orbitofrontal area's layer 4
    pub ofc_recycled: usize,
}

/// A cortical cell unfired for this many of its layer's predictions is pruned at sleep.
const COMPACT_IDLE: u64 = 20_000;

impl Mind {
    pub fn new(body: Body, seed: u64) -> Self {
        Self::with_config(body, seed, Config::default())
    }

    pub fn with_config(body: Body, seed: u64, cfg: Config) -> Self {
        let mut b = Brain::new(BITS, ACTIVE, seed.wrapping_add(500));
        b.columns = true;
        b.selforg = true;
        // coverage in the association areas too (a cue joining a familiar scene changes their
        // code as it changes the senses')
        b.assoc_coverage = cfg.coverage;
        b.superpose = cfg.superpose;
        b.select = cfg.claustrum && cfg.recur;
        if cfg.wide {
            b.full_bits = 2 * BITS;
        }
        b.generalize = cfg.generalize && cfg.compact;
        b.need = cfg.need && cfg.coverage;
        b.topo = cfg.keep_order;
        b.burst = cfg.keep_order && cfg.burst;
        b.adapt = cfg.adapt;
        b.local = cfg.local;
        if cfg.recur {
            b.pair_gain = 0;
            b.full_gain = 3;
        } else if cfg.persist {
            b.pair_tau = 1;
            b.full_tau = 3;
        }
        b.attach("eye", Kind::Sensor);
        b.attach("ear", Kind::Sensor);
        b.attach("hand", Kind::SensorMotor);
        b.attach("voice", Kind::Motor);
        b.attach("seen acts", Kind::Sensor);
        b.attach("presence", Kind::Sensor);
        b.attach("affect", Kind::Sensor);
        // coverage: every part of what each sense receives gets cells of its own
        for a in SLOTS {
            if let Some(l4) = b.areas[a].l4.as_mut() {
                l4.coverage = cfg.coverage;
            }
        }
        let mut cb = CerebellarCircuit::new(131_072, BITS, ONE / 256, seed.wrapping_add(77));
        cb.set_rates(2, 0);
        cb.set_cross_frames(true);
        // with focused cells each input group has a region (a 64-bit word) of its own: 63 of them
        let mut bg = MaskStriatum::new(if cfg.focused { 4096 } else { 2048 }, 64, seed.wrapping_add(31));
        bg.hold = false;
        bg.sticky = true;
        bg.conj = true;
        bg.push_pull = false;
        bg.release = 2;
        bg.grow = true;
        bg.learn_nogo = cfg.nogo;
        bg.nogo_symmetric = cfg.nogo_symmetric;
        bg.bounded = cfg.bounded;
        bg.balance = cfg.balance;
        bg.focused = cfg.focused;
        bg.window = cfg.window;
        // the prefrontal gates (each association column's "take in", the orbitofrontal hold)
        // keep their eligibility longer: holding pays only when what is held is used later
        if cfg.recur {
            bg.long_channels = (0..=MAX_PAIRS).map(|i| loop_channel(ColumnId::Assoc(i)) + 1).chain([OFC_HOLD_CHANNEL]).collect();
            bg.long_window = 8;
            // and their default is closed: opening (an update) needs more than the motor loops
            bg.brakes = bg.long_channels.iter().map(|&c| (c, 2)).collect();
        }
        if cfg.nogo_symmetric {
            bg.nogo_channels = [VOICE, HAND].iter().map(|&m| motor_channels(m).1).collect();
        }
        let act_baseline = cfg.act_baseline;
        let attend_gate = cfg.attend_gate;
        let elemental = cfg.elemental;
        let focused = cfg.focused;
        let value_z = cfg.ofc_column && cfg.ofc_value_z;
        let ofc_w = if cfg.wide { 2 * BITS } else { BITS };
        let persist = cfg.persist;
        let recur = cfg.recur;
        let coverage = cfg.coverage;
        let need = cfg.need;
        let keep_order = cfg.keep_order;
        let adapt = cfg.adapt;
        let local = cfg.local;
        let burst = cfg.keep_order && cfg.burst;
        let superpose = cfg.superpose;
        let claustrum = cfg.claustrum;
        let hc_bits = if cfg.hc_assoc { (MAX_PAIRS + 2) * BITS } else { SLOTS.len() * BITS };
        Self {
            rng: StdRng::seed_from_u64(seed ^ 0x313D),
            cfg,
            brain: b,
            cb,
            relay: ThalamicRelay::new(),
            wm: WorkingMemory::new(BITS, 1),
            bg,
            hc: {
                let mut h = IndexHc::new(hc_bits, seed);
                h.superpose = superpose;
                h
            },
            amy: {
                let mut a = Amygdala::new(SLOTS.len() * BITS, seed);
                a.gated = attend_gate;
                a.elemental = elemental;
                a
            },
            ofc: {
                // on Z: the area's state (BITS features) then the acts; else the senses' cues
                // then the acts
                let z = value_z;
                let first = if z { ofc_w } else { SLOTS.len() * BITS };
                let mut o = Amygdala::new(first + 2 * BITS, seed ^ 0x0FC);
                o.gated = attend_gate && !z;
                o.elemental = z;
                o.paired = focused;
                if act_baseline {
                    o.with_act_baseline(first as u32)
                } else {
                    o
                }
            },
            ofc_area: {
                let mut o = Ofc::new(ofc_w, ACTIVE, 8192, seed ^ 0x0FCA);
                o.set_coverage(coverage);
                o.set_need(need && coverage);
                o.col.topo = keep_order;
                o.col.burst = burst;
                o.set_adapt(adapt);
                o.set_local(local);
                o.col.superpose = superpose;
                o.col.select = claustrum && recur;
                if recur {
                    o.set_recurrence(3);
                } else if persist {
                    o.set_persistence(4);
                }
                o
            },
            claustrum: Claustrum::new(),
            efference_now: Vec::new(),
            motors: vec![(VOICE, body.voice, EAR), (HAND, body.hand, HAND)],
            pons: zero(),
            topdown: Vec::new(),
            prev_in: vec![zero(); SLOTS.len()],
            prev_codes: vec![zero(); SLOTS.len()],
            last_recall: zero(),
            last: None,
            last_expect: zero(),
            issued: Vec::new(),
            acted: 0,
            followed: Vec::new(),
            since_said: usize::MAX,
            rate_fast: 0,
            frustration: 0,
            held_tag: [0; 2],
            held_loads: [0; 2],
            rate_slow: 0,
            rate_act: 0,
            act_windows: Vec::new(),
            chosen_q: Vec::new(),
            p_after: 0,
            p_else: 0,
            spontaneous: 1,
            report: Report::default(),
            compacted: (0, 0),
            salient: false,
            ofc_recycled: 0,
        }
    }

    /// Each sensing area's raw input this step (the seen-acts area: what changed in view).
    fn inputs(&self, s: &Senses) -> Vec<BitVector> {
        let mut onset = s.eye.clone();
        for (w, x) in onset.as_words_mut().iter_mut().zip(self.prev_in[0].as_words()) {
            *w &= !x;
        }
        vec![s.eye.clone(), s.ear.clone(), s.touch.clone(), onset, s.presence.clone(), s.affect.clone()]
    }

    fn slot(a: usize) -> usize {
        SLOTS.iter().position(|&x| x == a).unwrap()
    }

    /// The association columns' codes for the areas' codes (by area), packed by slot: the
    /// association-level index's input.
    fn assoc_state(&self, codes: &[BitVector]) -> BitVector {
        let n = self.brain.pairs.len();
        // (the full column's slot is followed by a spare, for a wider top area)
        let mut v = BitVector::new((MAX_PAIRS + 2) * BITS, Some(0));
        for (i, c) in self.brain.assoc_inputs(codes).iter().enumerate() {
            let slot = assoc_slot(i, n);
            for b in ones(c) {
                v.bit_set(slot * BITS + b);
            }
        }
        v
    }

    /// One step: senses in, acts out.
    pub fn step(&mut self, s: &Senses) -> Acts {
        self.report = Report::default();
        if self.cfg.need {
            self.brain.set_salient(self.salient);
            self.ofc_area.set_salient(self.salient);
        }
        let raw = self.inputs(s);
        // its own voice: the sound sensed right after a command to the voice (the efference copy)
        let own = self.issued.iter().any(|(a, _)| *a == VOICE);
        // 1. learning what came: the columns, toward what the effectors' senses received (the
        // hand's when it acted, else the ear's); the motor areas, each act's result
        if let Some(st) = self.last.take() {
            let target = if self.issued.iter().any(|(a, _)| *a == HAND) { &s.touch } else { &s.ear };
            self.learn(&st, target);
        }
        self.acted = self.issued.len();
        // its own effect: how far each act's result was from what the forward model expected
        let mut effect = 0u32;
        let mut effect_on: Vec<usize> = Vec::new();
        // the efference copy of what was just done (the motor area's command), as features after
        // the senses' (each motor area a slot of BITS): the amygdala and the critic see the act
        let mut efference: Vec<u32> = Vec::new();
        // each act sensed now: (motor area, its efference features)
        let mut acts_now: Vec<(usize, Vec<u32>)> = Vec::new();
        for (a, c) in std::mem::take(&mut self.issued) {
            if let Some(k) = self.motors.iter().position(|m| m.0 == a) {
                let f: Vec<u32> = ones(&c).into_iter().map(|b| ((SLOTS.len() + k) * BITS + b) as u32).collect();
                if self.cfg.efference {
                    efference.extend(f.iter().copied());
                }
                acts_now.push((a, f));
            }
            let sense = self.motors.iter().find(|m| m.0 == a).map_or(EAR, |m| m.2);
            let result = raw[Self::slot(sense)].clone();
            let area = &mut self.brain.areas[a];
            area.sense(&result, false);
            if let Some(m) = area.motor.as_mut() {
                let e = m.observe(&c, &result, &mut self.rng);
                effect = effect.max(e);
                let (l, g) = motor_channels(a);
                effect_on.extend([l, g]);
            }
            // what the act produced, sensed with the command that made it (the dorsal stream)
            let r4 = self.brain.encode(sense, &result, true);
            let mut codes = vec![zero(); self.brain.areas.len()];
            codes[sense] = r4;
            codes[a] = c;
            self.brain.associate_codes(&codes, true);
        }
        // 2. what the cortex holds of what came (each area's code, with what is imagined there),
        // the onsets (cues) for the amygdala, attention outward where an input changed
        let td = std::mem::take(&mut self.topdown);
        let mut codes = vec![zero(); self.brain.areas.len()];
        let mut outward = false;
        let mut onsets: Vec<u32> = Vec::new();
        for (k, &a) in SLOTS.iter().enumerate() {
            let c = self.brain.encode(a, &raw[k], false);
            if !(a == EAR && own) && raw[k].count_ones() > 0 && !same(&raw[k], &self.prev_in[k]) {
                outward = true;
            }
            for b in ones(&c) {
                if !self.prev_codes[k].bit_get(b) {
                    onsets.push((k * BITS + b) as u32);
                }
            }
            self.prev_codes[k] = c.clone();
            let mut with = c;
            for (_, t) in td.iter().filter(|x| x.0 == a) {
                with = or(&with, t);
            }
            codes[a] = with;
        }
        // the amygdala: cues alone (what a stimulus leads to); the orbitofrontal learner: cues and
        // the act just done (what an act leads to, here)
        if self.cfg.efference && !self.value_z() {
            let mut with_act = onsets.clone();
            with_act.extend(efference.iter().copied());
            self.ofc.onsets(with_act);
        }
        if self.cfg.amygdala || self.cfg.learned_pause || self.cfg.imagine_value {
            self.amy.onsets(onsets);
        }
        self.efference_now = efference.clone();
        // the hippocampus: one step over the cortical state at its level (the association
        // columns', or every sensing area's)
        let state = if self.cfg.hc_assoc { self.assoc_state(&codes) } else { multi(&SLOTS.iter().map(|&a| (a, &codes[a])).collect::<Vec<_>>()) };
        let surprise = ONE - overlap(&self.last_expect, &s.ear).min(ACTIVE as u32) * ONE / ACTIVE as u32;
        let recall = self.hc.step(&state, outward, own, surprise);
        self.report.recalled = recall.count_ones() > 0;
        self.last_recall = recall.clone();
        // 3. the cortex
        let st = self.cortex(&raw, &recall, td);
        self.last_expect = st.expect.clone();
        // the orbitofrontal area: the cues (the association areas' code), the act just made, the
        // outcome as sensed (affective touch) and the hippocampus's context
        if self.cfg.ofc_column {
            let mut z = self.ofc_step(&codes, &acts_now, &recall, raw[Self::slot(AFFECT)].count_ones() as u32);
            // the gate: the cues' associability (with it on), else open
            let gate = !self.cfg.alpha || self.amy.alpha_now() >= 64;
            if self.cfg.ofc_split && self.ofc_area.split(gate) {
                z = self.ofc_area.z();
            }
            if self.value_z() {
                let mut feats: Vec<u32> = ones(&z).into_iter().map(|b| b as u32).collect();
                for (_, f) in &acts_now {
                    feats.extend(self.ofc_act(f));
                }
                self.ofc.onsets(feats);
            }
        }
        // the claustrum: of what surprised the full association column and the orbitofrontal area,
        // one item at most is held, something known that came unexpectedly, weighed by the cues'
        // associability, if it stands out
        if self.cfg.claustrum && self.cfg.recur {
            let full = ColumnId::Assoc(self.brain.pairs.len());
            let a = 16 + self.amy.alpha_now();
            let sal = |p: Option<(u32, bool)>| p.map_or(0, |(n, known)| if known { n * a / 16 } else { 0 });
            let offers = [sal(if self.brain.full.is_some() { self.brain.proposal_of(full) } else { None }), sal(self.ofc_area.col.proposal().map(|_| (self.ofc_area.col.surprise, self.ofc_area.col.recognized)))];
            match self.claustrum.choose(&offers) {
                Some(0) => self.brain.commit(full),
                Some(_) => {
                    let mut rng = StdRng::seed_from_u64(self.rng.next_u64());
                    self.ofc_area.col.commit(true, &mut rng);
                }
                None => {}
            }
        }
        // the held populations' salience: a held state loaded now is tagged with the cues'
        // associability (what made it worth holding), and the tag fades
        if self.cfg.recur {
            let n = self.brain.pairs.len();
            let loads = [self.brain.loads_of(ColumnId::Assoc(n)), self.ofc_area.col.loads];
            for k in 0..2 {
                let l = loads[k].0 + loads[k].1;
                if l != self.held_loads[k] {
                    self.held_loads[k] = l;
                    self.held_tag[k] = self.amy.alpha_now();
                } else {
                    self.held_tag[k] -= self.held_tag[k] / 16;
                }
            }
        }
        // 4. valuing what came
        let mut r = [0i32; 4];
        if self.acted > 0 {
            if self.cfg.own_effect {
                r[0] += effect as i32 / 4;
            }
            // with a tonic level to keep exploring, acting costs a little effort again; with
            // opportunity cost, what acting usually brings, if that is more (Niv et al. 2007)
            // (per act: each act costs its own usual worth, the orbitofrontal baseline cells that
            // fire on its efference copy, which grow as new acts are done)
            if self.cfg.tonic {
                for (_, f) in &acts_now {
                    let worth = if self.cfg.per_act { self.ofc.act_worth(&self.ofc_act(f)) } else { self.rate_act };
                    let cost = if self.cfg.opportunity { self.cfg.effort.max(worth) } else { self.cfg.effort };
                    r[3] -= cost;
                }
            }
        }
        if own {
            self.since_said = 0;
        } else {
            self.since_said = self.since_said.saturating_add(1);
        }
        let ear_onset = !same(&s.ear, &self.prev_in[Self::slot(EAR)]) && !own;
        let after = (1..=2).contains(&self.since_said);
        // the contingency detector: how much likelier a sound from outside is soon after its own
        // sound than otherwise (Watson's contingency detection in infants)
        let heard = if ear_onset { ONE as i32 } else { 0 };
        if after {
            self.p_after += (heard - self.p_after) / 32;
        } else {
            self.p_else += (heard - self.p_else) / 32;
        }
        if self.cfg.response && s.presence.count_ones() > 0 && ear_onset && after {
            r[1] += if self.cfg.contingency { (self.p_after - self.p_else).max(0) / 2 } else { ONE as i32 / 2 };
        }
        // affective touch: the receptor line's own activity is its value (a labelled line), as
        // strong as the stroke is wide on the skin
        r[2] += ((raw[Self::slot(AFFECT)].count_ones() as i32 * ONE as i32) / ACTIVE as i32).min(ONE as i32);
        // what acting usually brings: the outcomes in the two steps after each act (the act's
        // result is sensed now; an outcome it earns comes now or next step), slowly averaged
        let outcome = r[1] + r[2];
        for w in self.act_windows.iter_mut() {
            w.1 += outcome;
            w.0 -= 1;
        }
        for (a, f) in &acts_now {
            let worth = self.ofc.act_worth(&self.ofc_act(f));
            let q = self.chosen_q.iter().find(|x| x.0 == *a).map(|x| (x.1, x.2));
            // what the orbitofrontal value cells expect now, with the act in its trace: what
            // this act, here, should bring
            let here = self.ofc.cue_value().max(0);
            self.act_windows.push((1, outcome, *a, worth, q, here));
        }
        let done: Vec<(i32, usize, i32, Option<(i32, i32)>, i32)> = self.act_windows.iter().filter(|w| w.0 == 0).map(|w| (w.1, w.2, w.3, w.4, w.5)).collect();
        self.act_windows.retain(|w| w.0 > 0);
        for (got, motor, worth, q, here) in done {
            self.rate_act += (got - self.rate_act) / 32;
            // separate credit per act, straight to its own loop's and gate's choices of two steps
            // back: what it brought against what each expected of it there (its q, a cell an
            // eighth: Rescorla–Wagner's error for the act in its context), or against what the act
            // is usually worth
            if self.cfg.act_credit {
                let (l, g) = motor_channels(motor);
                match q {
                    Some((ql, qg)) if self.cfg.credit_own => {
                        let expect = |q: i32| (q * ONE as i32 / 8).clamp(0, ONE as i32);
                        self.bg.reinforce_at(&[l], got - expect(ql), 1);
                        self.bg.reinforce_at(&[g], got - expect(qg), 1);
                    }
                    _ => self.bg.reinforce_at(&[l, g], got - worth, 1),
                }
            }
            // frustration: what the act, here, was expected to bring (the orbitofrontal value) and
            // did not; relief when more came
            if self.cfg.frustration {
                self.frustration = (self.frustration + here.min(ONE as i32) - got.min(ONE as i32)).clamp(0, 2 * ONE as i32);
            }
        }
        // outcomes go to the critic; its own effect, with agency, trains the motor
        // loops' last choices directly
        for (k, &x) in r.iter().enumerate() {
            if x == 0 {
                continue;
            }
            if k == 0 && self.cfg.agency {
                self.bg.reinforce_last(&effect_on, x);
            } else {
                self.bg.reward(x);
            }
        }
        self.report.valued = r;
        if self.cfg.amygdala || self.cfg.imagine_value || self.cfg.learned_pause {
            // the amygdala learns what outcomes are worth (a response, a touch), not the surprise of
            // its own acts' effects (agency, not value)
            self.report.amy_predicted = self.amy.predicted();
            self.amy.learn(r[1] + r[2]);
        }
        if self.cfg.efference {
            self.ofc.learn(r[1] + r[2]);
        }
        // 5. the striatum: the critic sees everything; the dopamine error closing the last step
        // reaches the goal cells followed recently; the interneurons pause to learned cues
        let ids = self.brain.column_ids();
        self.report.loops = ids.len();
        // the critic (ventral striatum) sees its own territory with topography: the limbic
        // inputs (the amygdala's attended cues and value cells, the orbitofrontal learner's value
        // cells, someone's presence, affective touch); without, everything
        // (with a quiet critic, MIND_CRITIC_ALL=0 by default: not recall or working memory, which
        // change every step whatever happens, so a step with nothing in it has no error)
        let wide = self.cfg.critic_all;
        let mut all = if self.cfg.topographic { self.state_groups(&[ColumnId::Sense(PRESENCE), ColumnId::Sense(AFFECT)], &[], wide, wide) } else { self.state_groups(&ids, &ids, true, true) };
        // and the amygdala's value cells firing on the cues now: the basolateral amygdala drives
        // the ventral striatum (and through it the dopamine neurons) at the cue
        if self.cfg.amygdala && self.cfg.amy_critic {
            all.push(self.amy.value_cells().into_iter().map(|c| c + AMY_VALUE_OFFSET).collect());
        }
        // and the orbitofrontal learner's value cells: what the act just done, here, is worth
        // (the critic gets act values only through it; motor efference itself goes dorsally)
        if self.cfg.efference {
            all.push(self.ofc.value_cells().into_iter().map(|c| c + OFC_VALUE_OFFSET).collect());
        }
        // and the orbitofrontal area's state and what it holds (the ventral striatum's cortical
        // input is orbitofrontal and medial prefrontal)
        if self.cfg.ofc_column {
            all.extend(self.ofc_groups());
        }
        if self.cfg.learned_pause {
            self.bg.pause_open = self.amy.pausing();
            self.report.paused = self.bg.pause_open;
        }
        self.bg.begin_groups(&all, true);
        let delta = self.bg.last_delta;
        self.report.delta = delta;
        // a neuromodulatory burst (a dopamine error, phasic, not a value held): the next step's top
        // areas recruit at once for what they do not cover
        self.salient = delta.unsigned_abs() >= ONE / 4;
        self.report.value = self.bg.value();
        if self.cfg.learned_pause {
            self.amy.learn_pause(delta, self.bg.pause_min * (ONE as i32 / 8));
        }
        if delta != 0 {
            for (age, step) in self.followed.iter().rev().enumerate() {
                for &(id, cell) in step {
                    self.brain.reinforce(id, cell, delta >> age);
                }
            }
        }
        // the tonic level (noradrenaline's tonic mode, tonic dopamine): a fast and a slow average of
        // the outcomes; where the recent ones fall below the long-run ones, or nothing has paid
        // lately, choices loosen (exploration rises when utility falls: Aston-Jones & Cohen)
        let outcome = r[1] + r[2];
        self.frustration -= self.frustration / 16;
        self.report.frustration = self.frustration;
        self.rate_fast += (outcome - self.rate_fast) / 16;
        self.rate_slow += (outcome - self.rate_slow) / 256;
        let tonic = if self.cfg.tonic {
            let falling = ((self.rate_slow - self.rate_fast) * 64 / ONE as i32).clamp(0, 2);
            let barren = (self.rate_fast < ONE as i32 / 64) as i32;
            falling + barren
        } else {
            0
        };
        self.report.tonic = tonic;
        self.report.rate_act = self.rate_act;
        // curiosity: where the cortex's goals are getting better, choices loosen a little
        // frustration widens the choices (extinction-induced variability)
        let varied = if self.cfg.frustration { (self.frustration / (ONE as i32 / 8)).clamp(0, 6) } else { 0 };
        self.bg.extra_noise = self.brain.goal_progress().clamp(0, 2) + tonic + varied;
        let acts = self.choose(&ids);
        self.bg.extra_noise = 0;
        self.bg.set_view(None);
        self.prev_in = raw;
        self.last = Some(st);
        acts
    }

    /// The striatal input groups of some columns (`cols`), with working memory and recall if
    /// asked, and the amygdala's attended trace: each at its stable place among the input ids.
    fn state_groups(&self, cols: &[ColumnId], _all: &[ColumnId], wm: bool, recall: bool) -> Vec<Vec<usize>> {
        let mut groups: Vec<Vec<usize>> = cols.iter().map(|&id| ones(&self.brain.activity_of(id)).into_iter().map(|b| b + column_offset(id)).collect()).collect();
        if wm {
            groups.push(ones(&self.wm_content()).into_iter().map(|b| b + WM_OFFSET).collect());
        }
        if recall {
            groups.push(ones(&self.last_recall).into_iter().map(|b| b + RECALL_OFFSET).collect());
        }
        if self.cfg.amygdala {
            groups.push(self.amy.attended().into_iter().map(|f| f + AMY_OFFSET).collect());
        }
        groups
    }

    /// The association columns over area `a` (pairs that include it).
    fn pairs_with(&self, a: usize) -> Vec<ColumnId> {
        self.brain.pairs.iter().enumerate().filter(|(_, p)| p.0 .0 == a || p.0 .1 == a).map(|(i, _)| ColumnId::Assoc(i)).collect()
    }

    /// A loop's view (its cortical territory): a sensory column with the association columns over
    /// it; a pairwise one with its two areas and the full area above, and working memory; the
    /// full area with working memory and recall.
    fn loop_view(&self, id: ColumnId) -> Vec<Vec<usize>> {
        let n = self.brain.pairs.len();
        match id {
            ColumnId::Sense(a) => {
                let mut cols = vec![id];
                cols.extend(self.pairs_with(a));
                self.state_groups(&cols, &[], false, false)
            }
            ColumnId::Assoc(i) if i < n => {
                let (a, b) = self.brain.pairs[i].0;
                let mut cols = vec![id];
                for x in [a, b] {
                    if self.brain.areas[x].col.is_some() {
                        cols.push(ColumnId::Sense(x));
                    }
                }
                if self.brain.full.is_some() {
                    cols.push(ColumnId::Assoc(n));
                }
                self.state_groups(&cols, &[], true, false)
            }
            ColumnId::Assoc(_) => self.state_groups(&[id], &[], true, true),
        }
    }

    /// A motor loop's view: the column of the sense its acts are sensed by, the association
    /// columns over it or over the motor area, and working memory.
    fn motor_view(&self, motor: usize, sense: usize) -> Vec<Vec<usize>> {
        let mut cols = vec![ColumnId::Sense(sense)];
        for c in self.pairs_with(sense).into_iter().chain(self.pairs_with(motor)) {
            if !cols.contains(&c) {
                cols.push(c);
            }
        }
        let mut groups = self.state_groups(&cols, &[], true, false);
        // the senses' sustained state (what is in view, heard, felt now, not only what began):
        // a cue that stays on is a context while it is on
        if self.cfg.salience {
            for k in 0..SLOTS.len() {
                if SLOTS[k] != sense {
                    groups.push(ones(&self.prev_codes[k]).into_iter().map(|b| b + SUSTAINED_OFFSET + k * BITS).collect());
                }
            }
        }
        // salience gain: a group the learners value takes a larger share of the striatum's
        // context (it is repeated, each copy hashed apart), as attention raises a predictive
        // cue's gain
        let gain = |v: i32| if self.cfg.salience { 1 + (v.abs() / (ONE as i32 / 4)).clamp(0, 3) as usize } else { 1 };
        let amy_gain = gain(self.amy.cue_value());
        if self.cfg.salience && self.cfg.amygdala {
            if let Some(att) = groups.iter().position(|g| g.first().map_or(false, |&b| (AMY_OFFSET..AMY_VALUE_OFFSET).contains(&b))) {
                let g = groups[att].clone();
                for _ in 1..amy_gain {
                    groups.push(g.clone());
                }
            }
        }
        // the amygdala's value cells firing on the cues now (the basolateral amygdala reaches the
        // striatum's motor territory too): "a valued cue is here" is a context the loop can learn
        if self.cfg.amygdala {
            let g: Vec<usize> = self.amy.value_cells().into_iter().map(|c| c + AMY_VALUE_OFFSET).collect();
            for _ in 0..amy_gain {
                groups.push(g.clone());
            }
        }
        // the act just done (motor cortex's collaterals to the dorsal striatum: the efference
        // copy) and what the orbitofrontal learner says it is worth here
        if self.cfg.ofc_column {
            groups.extend(self.ofc_groups());
        }
        if self.cfg.efference {
            groups.push(self.efference_now.iter().map(|&f| f as usize + EFFERENCE_OFFSET).collect());
            let g: Vec<usize> = self.ofc.value_cells().into_iter().map(|c| c + OFC_VALUE_OFFSET).collect();
            for _ in 0..gain(self.ofc.cue_value()) {
                groups.push(g.clone());
            }
        }
        groups
    }

    /// For a viewer: what is active now, by name, as bit indices (each a 2048-bit area unless
    /// noted): the senses' layer 4 codes, working memory, the orbitofrontal state and what it
    /// holds, the amygdala's onsets now (features, 6 × 2048) and its value cells firing.
    pub fn snapshot(&self) -> Vec<(&'static str, Vec<usize>)> {
        let names = ["eye", "ear", "hand", "acts", "presence", "affect"];
        let mut v: Vec<(&'static str, Vec<usize>)> = names.iter().enumerate().map(|(k, n)| (*n, ones(&self.prev_codes[k]))).collect();
        v.push(("wm", ones(&self.wm_content())));
        v.push(("ofc_z", ones(&self.ofc_area.z())));
        v.push(("ofc_held", ones(&self.ofc_area.held_state())));
        v.push(("amy_onsets", self.amy.trace_at(0).iter().map(|&f| f as usize).collect()));
        v.push(("amy_value", self.amy.value_cells().into_iter().map(|c| c % 8192).collect()));
        let (l4, l23) = self.brain.full.as_ref().map_or((zero(), zero()), |f| (f.code.clone(), f.activity()));
        v.push(("top_l4", ones(&l4)));
        v.push(("top_l23", ones(&l23)));
        v.push(("top_driven", self.brain.full.as_ref().map_or(Vec::new(), |f| ones(&BitVector::from_words(f.code.as_words().iter().zip(f.held().as_words()).map(|(a, b)| a & !b).collect())))));
        v.push(("top_in", self.brain.full.as_ref().and_then(|f| f.l4.as_ref()).map_or(Vec::new(), |p| p.last_input().iter().map(|&b| b as usize).collect())));
        v.push(("top_burst", self.brain.full.as_ref().map_or(Vec::new(), |f| ones(f.unexpected()))));
        v
    }

    /// For a viewer: the hand loop's latest choice (its context, and the masks of the go and
    /// no-go cells that fired there), and its pools' sizes.
    pub fn hand_view(&self) -> (Option<(Vec<usize>, Vec<Vec<usize>>, Vec<Vec<usize>>)>, (usize, usize)) {
        let (l, _) = motor_channels(HAND);
        (self.bg.choice_view(l), self.bg.pool_sizes(l))
    }

    /// What working memory holds: with persistence, the full association column's trace (else
    /// the pairs' traces merged); else the separate store.
    fn wm_content(&self) -> BitVector {
        if self.cfg.recur {
            // the full association column's held population
            let n = self.brain.pairs.len();
            return if self.brain.full.is_some() { self.brain.held_of(ColumnId::Assoc(n)) } else { zero() };
        }
        if !self.cfg.persist {
            return self.wm.content();
        }
        let n = self.brain.pairs.len();
        if self.brain.full.is_some() {
            return self.brain.trace_of(ColumnId::Assoc(n));
        }
        let mut v = zero();
        for i in 0..n {
            v = or(&v, &self.brain.trace_of(ColumnId::Assoc(i)));
        }
        v
    }

    /// Whether the orbitofrontal value cells read the area's state.
    fn value_z(&self) -> bool {
        self.cfg.ofc_column && self.cfg.ofc_value_z
    }

    /// An act's efference features as the orbitofrontal value cells take them (after the state's
    /// features when they read the state, else after the senses').
    fn ofc_act(&self, f: &[u32]) -> Vec<u32> {
        if !self.value_z() {
            return f.to_vec();
        }
        let from = (SLOTS.len() * BITS) as u32;
        let w = self.ofc_area.bits() as u32;
        f.iter().map(|&x| x - from + w).collect()
    }

    /// The orbitofrontal area's step: its four inputs from this step's codes (by area), the acts
    /// sensed now and the hippocampus's recall. Returns the state Z.
    fn ofc_step(&mut self, codes: &[BitVector], acts_now: &[(usize, Vec<u32>)], recall: &BitVector, raw_affect: u32) -> BitVector {
        let assoc = self.brain.assoc_inputs(codes);
        // the cues: the full association area's code, else the pairs' merged
        let mut cues = zero();
        match (self.brain.full.is_some(), assoc.last()) {
            (true, Some(f)) if f.count_ones() > 0 => cues = fit(f, BITS),
            _ => {
                for c in &assoc {
                    cues = or(&cues, &fit(c, BITS));
                }
            }
        }
        let mut act = zero();
        for (_, f) in acts_now {
            for &x in f {
                act.bit_set(x as usize % BITS);
            }
        }
        let outcome = codes[AFFECT].clone();
        let mut context = zero();
        for x in ones(recall).into_iter().filter(|&x| x >= MAX_PAIRS * BITS) {
            context.bit_set((x - MAX_PAIRS * BITS) % BITS);
        }
        let mut inputs: [BitVector; 4] = [zero(), zero(), zero(), zero()];
        inputs[ofc_area::CUES] = cues;
        inputs[ofc_area::ACT] = act;
        inputs[ofc_area::OUTCOME] = outcome;
        inputs[ofc_area::CONTEXT] = context;
        // the outcome's level on the labelled line (affective touch: as wide as the stroke)
        let level = (raw_affect * 3 / ACTIVE as u32).min(3) as usize;
        self.ofc_area.step(&inputs, level, true)
    }

    /// The orbitofrontal area's state and what it holds, as input groups for the striatum.
    fn ofc_groups(&self) -> Vec<Vec<usize>> {
        vec![ones(&self.ofc_area.z()).into_iter().map(|b| b + OFC_Z_OFFSET).collect(), ones(&self.ofc_area.held_state()).into_iter().map(|b| b + OFC_HELD_OFFSET).collect()]
    }

    /// A motor choice's learning rate from its cues' associability: a gain from the trace's
    /// (¼× when nothing has been surprising, up to about 2¼×), and a weight per region of the
    /// striatum's focused context (one per live input group, in order) from the mean
    /// associability of the group's features where they are features (the senses' sustained
    /// codes, the amygdala's attended trace), else a middling weight.
    fn choice_alpha(&self, groups: &[Vec<usize>]) -> (u32, Vec<u16>) {
        let gain = 64 + 2 * self.amy.alpha_now();
        let regions = if self.cfg.focused { 4096 / 64 - 1 } else { 2048 / 64 - 1 };
        let mut w = vec![0u16; regions];
        for (gi, g) in groups.iter().filter(|g| !g.is_empty()).enumerate() {
            let feats: Vec<usize> = g
                .iter()
                .filter_map(|&b| {
                    if (SUSTAINED_OFFSET..SUSTAINED_OFFSET + SLOTS.len() * BITS).contains(&b) {
                        Some(b - SUSTAINED_OFFSET)
                    } else if (AMY_OFFSET..AMY_VALUE_OFFSET).contains(&b) {
                        Some((b - AMY_OFFSET) / crate::program::amygdala::TRACE)
                    } else {
                        None
                    }
                })
                .collect();
            // a held population's region: the salience it was loaded with
            let first = g.first().copied().unwrap_or(0);
            let held = if (WM_OFFSET..WM_OFFSET + 2 * BITS).contains(&first) {
                Some(self.held_tag[0])
            } else if (OFC_HELD_OFFSET..OFC_HELD_OFFSET + 2 * BITS).contains(&first) {
                Some(self.held_tag[1])
            } else {
                None
            };
            let a = match held {
                Some(t) if self.cfg.recur => t,
                _ if feats.is_empty() => 32,
                _ => feats.iter().map(|&f| self.amy.alpha_of(f) as u32).sum::<u32>() / feats.len() as u32,
            };
            let r = gi % regions;
            w[r] = w[r].max(8 + a as u16);
        }
        (gain, w)
    }

    /// The loops: each column's takes its goal in or not (and an association column's may load
    /// working memory); each motor area's takes one candidate and releases or withholds it.
    fn choose(&mut self, ids: &[ColumnId]) -> Acts {
        // the orbitofrontal area's loop (through the mediodorsal thalamus): hold its state now,
        // or not; it sees the state, what is held, the cues' and the state's values, and the
        // outcome as sensed
        if self.cfg.ofc_column {
            if self.cfg.topographic {
                let mut v = self.ofc_groups();
                v.push(self.amy.value_cells().into_iter().map(|c| c + AMY_VALUE_OFFSET).collect());
                v.push(self.ofc.value_cells().into_iter().map(|c| c + OFC_VALUE_OFFSET).collect());
                v.push(ones(&self.prev_codes[Self::slot(AFFECT)]).into_iter().map(|b| b + SUSTAINED_OFFSET + Self::slot(AFFECT) * BITS).collect());
                self.bg.set_view(Some(v));
            }
            let hold = self.bg.choose(OFC_HOLD_CHANNEL, 2, true) == 1;
            self.ofc_area.hold(hold);
        }
        let mut taken: Vec<(ColumnId, usize, BitVector)> = Vec::new();
        for &id in ids {
            if self.cfg.topographic {
                let v = self.loop_view(id);
                self.bg.set_view(Some(v));
            }
            if let Some((cell, g, _)) = self.brain.goal_of(id) {
                let offer = vec![vec![ones(&g).into_iter().map(|b| ACT_BASE + b).collect::<Vec<usize>>()]];
                if self.bg.choose_among(loop_channel(id), &offer, true).is_some() {
                    taken.push((id, cell, g));
                }
            }
            if let ColumnId::Assoc(_) = id {
                if self.bg.choose(loop_channel(id) + 1, 2, true) == 1 {
                    if self.cfg.persist || self.cfg.recur {
                        // the loop keeps what the column holds (its own trace), or with
                        // recurrence opens its gate (the input becomes its state)
                        self.brain.refresh(id, false);
                    } else {
                        let c = self.brain.code_of_column(id);
                        if c.count_ones() > 0 {
                            self.wm.load(0, &c);
                        }
                    }
                }
            }
        }
        self.report.goals_taken = taken.len();
        // each goal taken in goes to the motor area that can produce it best (its inverse model
        // has a command, or for the voice the imitation route; its forward model's result
        // closest to the goal)
        type Cand = (BitVector, Vec<usize>, Option<(ColumnId, usize)>);
        let mut per_motor: Vec<Vec<Cand>> = vec![Vec::new(); self.motors.len()];
        let motor_ids: Vec<usize> = self.motors.iter().map(|m| m.0).collect();
        for (id, cell, g) in &taken {
            let mut best: Option<(u32, usize, BitVector)> = None;
            for (k, &m) in motor_ids.iter().enumerate() {
                if let Some(c) = self.command(m, g) {
                    let fit = self.brain.areas[m].motor.as_mut().and_then(|x| x.predict(&c)).map_or(0, |p| overlap(&p, g));
                    if best.as_ref().map_or(true, |b| fit > b.0) {
                        best = Some((fit, k, c));
                    }
                }
            }
            if let Some((_, k, c)) = best {
                per_motor[k].push((c, ones(g).into_iter().map(|b| ACT_BASE + b).collect(), Some((*id, *cell))));
            }
        }
        // spontaneous motor primitives (pattern generators), one per effector now and then
        for k in 0..self.motors.len() {
            if self.rng.gen_range(0..self.spontaneous) == 0 {
                let n = self.motors[k].1.len();
                let c = self.motors[k].1[self.rng.gen_range(0..n)].clone();
                let bits = ones(&c).into_iter().map(|b| ACT_BASE + (k + 1) * BITS + b).collect();
                per_motor[k].push((c, bits, None));
            }
        }
        let mut acts = Acts::default();
        let mut followed_now = Vec::new();
        for (k, cands) in per_motor.into_iter().enumerate() {
            if cands.is_empty() {
                continue;
            }
            let (motor, sense) = (self.motors[k].0, self.motors[k].2);
            if self.cfg.topographic {
                let v = self.motor_view(motor, sense);
                self.bg.set_view(Some(v));
            } else {
                self.bg.set_view(None);
            }
            let (loop_ch, gate) = motor_channels(motor);
            // associability (Pearce–Hall): how much this choice learns, and from which of its
            // inputs its new cells draw, follow how surprising its cues have been
            if self.cfg.alpha && self.cfg.topographic {
                let v = self.motor_view(motor, sense);
                self.bg.alpha = self.choice_alpha(&v);
            }
            // Pavlovian–instrumental transfer: cues worth something lower the brake
            let brake = self.bg.release;
            if self.cfg.pit {
                let v = self.amy.cue_value();
                self.bg.release = brake - (v / (ONE as i32 / 4)).clamp(-2, 2);
            }
            // frustration lowers it too (the extinction burst)
            if self.cfg.frustration {
                self.bg.release -= (self.frustration / (ONE as i32 / 4)).clamp(0, 2);
            }
            let offer: Vec<Vec<Vec<usize>>> = cands.iter().map(|c| vec![c.1.clone()]).collect();
            let Some(i) = self.bg.choose_among(loop_ch, &offer, true) else {
                self.bg.release = brake;
                self.bg.alpha = (256, Vec::new());
                continue;
            };
            let q_loop = self.bg.last_q;
            let (cmd, _, from) = cands[i].clone();
            if let Some(f) = from {
                followed_now.push(f);
            }
            // the motor gate sees the candidate and, with imagined value, what its imagined
            // outcome has been worth (the forward model's result, valued by the amygdala)
            let mut gate_offer = offer[i].clone();
            if self.cfg.imagine_value {
                if let Some(result) = self.brain.areas[motor].motor.as_mut().and_then(|m| m.predict(&cmd)) {
                    let code = self.brain.encode_only(sense, &result);
                    let feats: Vec<u32> = ones(&code).into_iter().map(|b| (Self::slot(sense) * BITS + b) as u32).collect();
                    let v = self.amy.value_of(&feats);
                    if v.abs() >= ONE as i32 / 8 {
                        let base = VALUE_BASE + if v > 0 { 0 } else { 16 };
                        gate_offer.push((base..base + 16).collect());
                    }
                }
            }
            let released = self.bg.choose_among(gate, &[gate_offer], true).is_some();
            self.bg.release = brake;
            self.bg.alpha = (256, Vec::new());
            if released {
                let q_gate = self.bg.last_q;
                self.chosen_q.retain(|x| x.0 != motor);
                self.chosen_q.push((motor, q_loop, q_gate));
                self.brain.areas[motor].issue(&cmd);
                self.issued.push((motor, cmd.clone()));
                self.report.from_goal |= from.is_some();
                if motor == VOICE {
                    acts.voice = Some(cmd);
                    self.report.spoke = true;
                } else {
                    acts.hand = Some(cmd);
                    self.report.moved = true;
                }
            } else {
                // withheld: imagined, fed back top-down at the next step
                self.imagine(motor, &cmd);
                self.report.inward = true;
            }
        }
        self.followed.push(followed_now);
        if self.followed.len() > GOAL_TRACE {
            self.followed.remove(0);
        }
        acts
    }

    /// The command motor area `a` would use for goal `g`: for the voice, the imitation route first
    /// (the meant sound evokes, through the ear–voice association, the command that says it),
    /// else the inverse model.
    fn command(&mut self, a: usize, g: &BitVector) -> Option<BitVector> {
        if a == VOICE {
            let e = self.brain.encode_only(EAR, g);
            let m = self.brain.mirror(EAR, &e, VOICE);
            if m.count_ones() > 0 {
                return Some(m);
            }
        }
        self.brain.areas[a].command_for(g)
    }

    /// A command held back from the body, run through its forward model: what it would produce,
    /// top-down at the next step (the hand: how it would feel, and through the mirror how it
    /// would look; the voice: the sound).
    fn imagine(&mut self, motor: usize, cmd: &BitVector) {
        let b = &mut self.brain;
        let Some(result) = b.areas[motor].motor.as_mut().and_then(|m| m.predict(cmd)) else { return };
        if motor == HAND {
            let h = b.encode_only(HAND, &result);
            let seen = b.mirror(HAND, &h, ACTS);
            let e = b.encode_only(ACTS, &seen);
            self.topdown.push((HAND, h));
            self.topdown.push((ACTS, e));
        } else {
            let e = b.encode_only(EAR, &result);
            self.topdown.push((EAR, e));
        }
    }

    /// The cortex's step: every column senses, the association columns grow and predict, every
    /// layer 5 offers a goal, and the relay mixes them into the expectation. Recall arrives where
    /// the index points: in the association columns (and down the hierarchy by their feedback),
    /// or in the senses directly.
    fn cortex(&mut self, raw: &[BitVector], recall: &BitVector, td: Vec<(usize, BitVector)>) -> Step {
        let hc_assoc = self.cfg.hc_assoc;
        let pfc = fit(&self.wm_content(), BITS);
        let b = &mut self.brain;
        let mut ff: Vec<BitVector> = SLOTS.iter().enumerate().map(|(k, &a)| b.encode(a, &raw[k], true)).collect();
        let ear = &raw[Self::slot(EAR)];
        let mossy = frames(&[ear, &self.pons]);
        let cb_out = self.cb.predict(&mossy).clone();
        let mut props: Vec<(u8, Driver, BitVector)> = Vec::new();
        if cb_out.count_ones() > 0 {
            props.push((2, Driver::Tonic(0), cb_out.clone()));
        }
        // the mirror: an act seen evokes how it would feel, which joins what the hand represents
        let (hk, ak) = (Self::slot(HAND), Self::slot(ACTS));
        if ff[ak].count_ones() > 0 {
            let res = b.mirror(ACTS, &ff[ak], HAND);
            if res.count_ones() > 0 {
                ff[hk] = b.encode(HAND, &or(&raw[hk], &res), true);
            }
        }
        let (fb, _) = b.columns_activity();
        // top-down (recall and what is imagined) arrives as feedback and weak input, never
        // through layer 4: a column learns only from what it sensed
        let mut down: Vec<BitVector> = vec![zero(); SLOTS.len()];
        let mut extra: Vec<(usize, BitVector)> = Vec::new();
        if hc_assoc {
            let n = b.pairs.len();
            let cols = n + b.full.is_some() as usize;
            for i in 0..cols {
                let slot = assoc_slot(i, n);
                let width = b.width_of(i);
                let mut cells = BitVector::new(width, Some(0));
                for x in ones(recall).into_iter().filter(|&x| x >= slot * BITS && x < slot * BITS + width) {
                    cells.bit_set(x - slot * BITS);
                }
                if cells.count_ones() == 0 {
                    continue;
                }
                for (a, c) in b.cascade_down(i, &cells) {
                    if let Some(k) = SLOTS.iter().position(|&x| x == a) {
                        down[k] = or(&down[k], &c);
                    }
                }
                extra.push((i, cells));
            }
        } else {
            for (k, &a) in SLOTS.iter().enumerate() {
                down[k] = part(recall, a);
            }
        }
        for (a, c) in td {
            down[Self::slot(a)] = or(&down[Self::slot(a)], &c);
        }
        let mut codes = vec![zero(); b.areas.len()];
        let mut with_down = vec![zero(); b.areas.len()];
        let predictive = self.cfg.predictive;
        for (k, &a) in SLOTS.iter().enumerate() {
            let sensed = ff[k].count_ones() > 0;
            let input = or(&ff[k], &down[k]);
            // predictive transmission: what the column expected (from the step before) is
            // cancelled, and only the unpredicted part goes up (Rao & Ballard)
            let expected = if predictive { b.sense_state(a).map(|x| x.1) } else { None };
            b.sense_step(a, &input, Some(&or(&fb, &down[k])), sensed);
            let up = |x: &BitVector| match expected.as_ref() {
                Some(e) => {
                    let mut v = x.clone();
                    for (w, y) in v.as_words_mut().iter_mut().zip(e.as_words()) {
                        *w &= !y;
                    }
                    v
                }
                None => x.clone(),
            };
            codes[a] = up(&ff[k]);
            with_down[a] = up(&input);
        }
        // association columns grow from what is sensed together
        b.associate_codes(&codes, true);
        let tuft = or(&pfc, &cb_out);
        let outs = b.step_columns(&with_down, &extra, &tuft, true);
        let mut sense_ins = Vec::new();
        for a in SLOTS {
            if let Some((code, exp)) = b.sense_state(a) {
                if code.count_ones() > 0 {
                    if let Some((o, burst)) = b.sense_goal(a, &tuft) {
                        props.push((30 + a as u8, if burst { Driver::Burst } else { Driver::Spike }, o));
                    }
                    sense_ins.push((a, code, exp));
                }
            }
        }
        let (_, fired) = b.columns_activity();
        self.pons = project(&fired);
        let mut spiked = false;
        for (i, (goal, _, _)) in outs.iter().enumerate() {
            if let Some((o, burst)) = goal {
                props.push((10 + i.min(19) as u8, if *burst { Driver::Burst } else { Driver::Spike }, o.clone()));
                spiked |= !burst;
            }
        }
        self.relay.step(spiked);
        let ctx = frames(&[&pfc, ear]);
        let expect = self.mix(&props, &ctx, &cb_out);
        let ins = outs.into_iter().map(|(_, x, e)| (x, e)).collect();
        Step { ins, tuft, sense_ins, mossy, ctx, props, expect }
    }

    /// The relay mixes the columns' predictions, each weighted by its learned reliability in
    /// this context; the strongest 32 bits.
    fn mix(&mut self, props: &[(u8, Driver, BitVector)], ctx: &BitVector, cb_out: &BitVector) -> BitVector {
        let mut score = vec![0u32; BITS];
        for (src, drv, pat) in props {
            let agrees = *src != 2 && overlap(pat, cb_out) >= 16;
            let w = self.relay.weight(*src, *drv, ctx, agrees) as u64;
            for b in ones(pat) {
                score[b] += (w >> 4) as u32 + 1;
            }
        }
        let mut order: Vec<usize> = (0..BITS).filter(|&b| score[b] > 0).collect();
        order.sort_unstable_by(|a, b| score[*b].cmp(&score[*a]).then(a.cmp(b)));
        let mut v = zero();
        for &b in order.iter().take(ACTIVE) {
            v.bit_set(b);
        }
        v
    }

    fn learn(&mut self, st: &Step, target: &BitVector) {
        for (i, (x, e)) in st.ins.iter().enumerate() {
            self.brain.learn_goal(i, x, &st.tuft, e, target);
        }
        for (a, code, exp) in &st.sense_ins {
            self.brain.learn_sense_goal(*a, code, &st.tuft, exp, target);
        }
        self.cb.learn(&st.mossy, target, &mut self.rng);
        for (src, drv, pat) in &st.props {
            self.relay.record(*src, *drv, &st.ctx, overlap(pat, target) >= 16, &mut self.rng);
        }
    }

    /// Sleep: the hippocampus replays `chains` chains; the cortex completes each pointer, and the
    /// row follows what it completes now.
    pub fn sleep(&mut self, chains: usize) {
        let mut rng = StdRng::seed_from_u64(self.rng.next_u64());
        let bits = if self.cfg.hc_assoc { (MAX_PAIRS + 2) * BITS } else { SLOTS.len() * BITS };
        let mut hc = std::mem::replace(&mut self.hc, IndexHc::new(bits, 0));
        hc.consolidate = self.cfg.consolidate;
        // consolidation: each replayed row is presented to the cortex, and the cortex's goals
        // learn the next replayed row's sound as their target, interleaved over chains of rows
        // (the hippocampus teaching the slow cortex what episodes share: McClelland, McNaughton
        // & O'Reilly 1995)
        let mut prev: Option<Step> = None;
        let consolidate = self.cfg.consolidate && self.cfg.hc_assoc;
        hc.replay(&mut rng, chains, |frame| {
            let blank = vec![zero(); SLOTS.len()];
            let st = self.cortex(&blank, frame, Vec::new());
            if consolidate {
                let target = self.recalled_raw(frame, EAR);
                if let Some(p) = prev.take() {
                    if target.count_ones() > 0 {
                        self.learn(&p, &target);
                    }
                }
            }
            let out = if self.cfg.hc_assoc {
                // what the senses' columns now hold, as the association columns code it
                let mut codes = vec![zero(); self.brain.areas.len()];
                for a in SLOTS {
                    if let Some((c, _)) = self.brain.sense_state(a) {
                        codes[a] = c;
                    }
                }
                ones(&self.assoc_state(&codes))
            } else {
                let e4 = self.brain.encode_only(EAR, &st.expect);
                ones(&multi(&[(EAR, &e4)]))
            };
            prev = Some(st);
            out
        });
        self.hc = hc;
        // compaction: duplicate and useless cortical cells go
        if self.cfg.compact {
            let (m, p) = self.brain.compact(COMPACT_IDLE);
            self.compacted.0 += m;
            self.compacted.1 += p;
            if self.cfg.need {
                self.ofc_recycled += self.ofc_area.recycle();
            }
        }
        self.last = None;
    }

    /// What a recalled frame (the association columns' slots) expects area `a` to sense, as raw
    /// input: each slot's cells cascade down to the area's layer 4 code, and that back to its
    /// input.
    fn recalled_raw(&self, frame: &BitVector, a: usize) -> BitVector {
        let n = self.brain.pairs.len();
        let cols = n + self.brain.full.is_some() as usize;
        let mut code = zero();
        for i in 0..cols {
            let slot = assoc_slot(i, n);
            let width = self.brain.width_of(i);
            let mut cells = BitVector::new(width, Some(0));
            for x in ones(frame).into_iter().filter(|&x| x >= slot * BITS && x < slot * BITS + width) {
                cells.bit_set(x - slot * BITS);
            }
            if cells.count_ones() == 0 {
                continue;
            }
            for (area, c) in self.brain.cascade_down(i, &cells) {
                if area == a {
                    code = or(&code, &c);
                }
            }
        }
        if code.count_ones() == 0 {
            return code;
        }
        self.brain.raw_of(a, &code)
    }

    /// A motor area's loop and gate: (go cells, no-go cells) each, for diagnosis.
    pub fn motor_cells(&self, motor: usize) -> [(usize, usize); 2] {
        let (l, g) = motor_channels(motor);
        [self.bg.pool_sizes(l), self.bg.pool_sizes(g)]
    }

    /// The critic's value now (`Q16`).
    pub fn value(&self) -> i32 {
        self.bg.value()
    }
}
