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
//!    - *a smile:* an innately recognised pattern in the presence sense;
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
/// the sensing areas, each a slot of BITS cells in the hippocampus's pointers
pub const SLOTS: [usize; 5] = [EYE, EAR, HAND, ACTS, PRESENCE];

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
const WM_OFFSET: usize = 40 * BITS;
const RECALL_OFFSET: usize = 41 * BITS;
const AMY_OFFSET: usize = 80 * BITS;
const AMY_VALUE_OFFSET: usize = 120 * BITS;
const EFFERENCE_OFFSET: usize = 130 * BITS;
const OFC_VALUE_OFFSET: usize = 140 * BITS;
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
}

impl Default for Config {
    fn default() -> Self {
        Self { hc_assoc: true, topographic: true, amygdala: true, learned_pause: true, contingency: true, imagine_value: true, own_effect: true, agency: true, pit: true, response: true, amy_critic: true, efference: true }
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
    /// the dopamine-producing valuations this step, `Q16`: (own effect, response, smile, unused: acting costs nothing)
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
    /// the efference copy of the act sensed this step (features), for the motor loops' view
    efference_now: Vec<u32>,
    /// each motor area, its primitives (the genome's pattern generators), and the sensing area
    /// its acts are sensed by (its result)
    motors: Vec<(usize, Vec<BitVector>, usize)>,
    smile: BitVector,
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
    /// the contingency detector (`Q16`): how often a sound from outside comes within two steps
    /// after its own sound, and otherwise
    p_after: i32,
    p_else: i32,
    /// a primitive of each effector is spontaneously active 1 step in this many
    pub spontaneous: u32,
    pub report: Report,
}

impl Mind {
    pub fn new(body: Body, seed: u64) -> Self {
        Self::with_config(body, seed, Config::default())
    }

    pub fn with_config(body: Body, seed: u64, cfg: Config) -> Self {
        let mut b = Brain::new(BITS, ACTIVE, seed.wrapping_add(500));
        b.columns = true;
        b.selforg = true;
        b.attach("eye", Kind::Sensor);
        b.attach("ear", Kind::Sensor);
        b.attach("hand", Kind::SensorMotor);
        b.attach("voice", Kind::Motor);
        b.attach("seen acts", Kind::Sensor);
        b.attach("presence", Kind::Sensor);
        let mut cb = CerebellarCircuit::new(131_072, BITS, ONE / 256, seed.wrapping_add(77));
        cb.set_rates(2, 0);
        cb.set_cross_frames(true);
        let mut bg = MaskStriatum::new(2048, 64, seed.wrapping_add(31));
        bg.hold = false;
        bg.sticky = true;
        bg.conj = true;
        bg.push_pull = false;
        bg.release = 2;
        bg.grow = true;
        let hc_bits = if cfg.hc_assoc { (MAX_PAIRS + 1) * BITS } else { SLOTS.len() * BITS };
        Self {
            rng: StdRng::seed_from_u64(seed ^ 0x313D),
            cfg,
            brain: b,
            cb,
            relay: ThalamicRelay::new(),
            wm: WorkingMemory::new(BITS, 1),
            bg,
            hc: IndexHc::new(hc_bits, seed),
            amy: Amygdala::new(SLOTS.len() * BITS, seed),
            ofc: Amygdala::new((SLOTS.len() + 2) * BITS, seed ^ 0x0FC),
            efference_now: Vec::new(),
            motors: vec![(VOICE, body.voice, EAR), (HAND, body.hand, HAND)],
            smile: body.smile,
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
            p_after: 0,
            p_else: 0,
            spontaneous: 1,
            report: Report::default(),
        }
    }

    /// Each sensing area's raw input this step (the seen-acts area: what changed in view).
    fn inputs(&self, s: &Senses) -> Vec<BitVector> {
        let mut onset = s.eye.clone();
        for (w, x) in onset.as_words_mut().iter_mut().zip(self.prev_in[0].as_words()) {
            *w &= !x;
        }
        vec![s.eye.clone(), s.ear.clone(), s.touch.clone(), onset, s.presence.clone()]
    }

    fn slot(a: usize) -> usize {
        SLOTS.iter().position(|&x| x == a).unwrap()
    }

    /// The association columns' codes for the areas' codes (by area), packed by slot: the
    /// association-level index's input.
    fn assoc_state(&self, codes: &[BitVector]) -> BitVector {
        let n = self.brain.pairs.len();
        let mut v = BitVector::new((MAX_PAIRS + 1) * BITS, Some(0));
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
        for (a, c) in std::mem::take(&mut self.issued) {
            if self.cfg.efference {
                if let Some(k) = self.motors.iter().position(|m| m.0 == a) {
                    efference.extend(ones(&c).into_iter().map(|b| ((SLOTS.len() + k) * BITS + b) as u32));
                }
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
        if self.cfg.efference {
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
        // 4. valuing what came
        let mut r = [0i32; 4];
        if self.acted > 0 {
            if self.cfg.own_effect {
                r[0] += effect as i32 / 4;
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
        if overlap(&s.presence, &self.smile) * 2 >= self.smile.count_ones() as u32 {
            r[2] += ONE as i32;
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
            // the amygdala learns what outcomes are worth (a response, a smile), not the surprise of
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
        // inputs (the amygdala's attended cues, the hippocampus's recall, working memory, and
        // someone's presence); without, everything
        let mut all = if self.cfg.topographic { self.state_groups(&[ColumnId::Sense(PRESENCE)], &[], true, true) } else { self.state_groups(&ids, &ids, true, true) };
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
        if self.cfg.learned_pause {
            self.bg.pause_open = self.amy.pausing();
            self.report.paused = self.bg.pause_open;
        }
        self.bg.begin_groups(&all, true);
        let delta = self.bg.last_delta;
        self.report.delta = delta;
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
        // curiosity: where the cortex's goals are getting better, choices loosen a little
        self.bg.extra_noise = self.brain.goal_progress().clamp(0, 2);
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
            groups.push(ones(&self.wm.content()).into_iter().map(|b| b + WM_OFFSET).collect());
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
        // the amygdala's value cells firing on the cues now (the basolateral amygdala reaches the
        // striatum's motor territory too): "a valued cue is here" is a context the loop can learn
        if self.cfg.amygdala {
            groups.push(self.amy.value_cells().into_iter().map(|c| c + AMY_VALUE_OFFSET).collect());
        }
        // the act just done (motor cortex's collaterals to the dorsal striatum: the efference
        // copy) and what the orbitofrontal learner says it is worth here
        if self.cfg.efference {
            groups.push(self.efference_now.iter().map(|&f| f as usize + EFFERENCE_OFFSET).collect());
            groups.push(self.ofc.value_cells().into_iter().map(|c| c + OFC_VALUE_OFFSET).collect());
        }
        groups
    }

    /// The loops: each column's takes its goal in or not (and an association column's may load
    /// working memory); each motor area's takes one candidate and releases or withholds it.
    fn choose(&mut self, ids: &[ColumnId]) -> Acts {
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
                    let c = self.brain.code_of_column(id);
                    if c.count_ones() > 0 {
                        self.wm.load(0, &c);
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
            // Pavlovian–instrumental transfer: cues worth something lower the brake
            let brake = self.bg.release;
            if self.cfg.pit {
                let v = self.amy.cue_value();
                self.bg.release = brake - (v / (ONE as i32 / 4)).clamp(-2, 2);
            }
            let offer: Vec<Vec<Vec<usize>>> = cands.iter().map(|c| vec![c.1.clone()]).collect();
            let Some(i) = self.bg.choose_among(loop_ch, &offer, true) else {
                self.bg.release = brake;
                continue;
            };
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
            if released {
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
        let b = &mut self.brain;
        let mut ff: Vec<BitVector> = SLOTS.iter().enumerate().map(|(k, &a)| b.encode(a, &raw[k], true)).collect();
        let ear = &raw[Self::slot(EAR)];
        let pfc = self.wm.content();
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
                let mut cells = zero();
                for x in ones(recall).into_iter().filter(|&x| x / BITS == slot) {
                    cells.bit_set(x % BITS);
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
        for (k, &a) in SLOTS.iter().enumerate() {
            let sensed = ff[k].count_ones() > 0;
            let input = or(&ff[k], &down[k]);
            b.sense_step(a, &input, Some(&or(&fb, &down[k])), sensed);
            codes[a] = ff[k].clone();
            with_down[a] = input;
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
        let bits = if self.cfg.hc_assoc { (MAX_PAIRS + 1) * BITS } else { SLOTS.len() * BITS };
        let mut hc = std::mem::replace(&mut self.hc, IndexHc::new(bits, 0));
        hc.replay(&mut rng, chains, |frame| {
            let blank = vec![zero(); SLOTS.len()];
            let st = self.cortex(&blank, frame, Vec::new());
            if self.cfg.hc_assoc {
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
            }
        });
        self.hc = hc;
        self.last = None;
    }

    /// The critic's value now (`Q16`).
    pub fn value(&self) -> i32 {
        self.bg.value()
    }
}
