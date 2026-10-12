//! A mind: senses in, acts out, and every reward its own.
//!
//! `Mind::step(senses) -> acts` is the whole interface. Inside, one pipeline runs every step,
//! whatever is going on outside (a caregiver babbling back, a lesson, nothing); nothing outside
//! says what phase it is, what to attend to, when to speak, or what was right.
//!
//! 1. **Learning what came.** The cortex learns from the last step's state what was sensed now
//!    (self-supervised: the next sound heard, or after a hand act, what the hand felt). The
//!    motor areas learn each act's result (the inverse and forward models: no babbling phase
//!    with the world's answers handed over).
//! 2. **Valuing what came** (innate valuations, the only source of dopamine):
//!    - *its own effect:* after its own act, as surprising as the sensory column finds the
//!      result (sound for the voice, sight for the hand): contingent change, fading as it comes
//!      to be expected (the colliculus's short-latency route, Redgrave & Gurney);
//!    - *a response:* someone there (presence) and a sound from outside soon after its own sound
//!      made aloud (contingent replies, Goldstein et al. 2003);
//!    - *a smile:* an innately recognised pattern in the presence sense;
//!    - *effort:* every act costs a little.
//!    Nothing else: no reward for predicting a word, for saying the word someone wanted, for
//!    opening a book.
//! 3. **Sensing:** each area's column, the association columns grown by co-activity, the motion
//!    stream (what changed in view), the mirror (seen acts evoke how they would feel), the
//!    hippocampus (always completing; acetylcholine follows attention to the outside), the
//!    cerebellum, the thalamic relay's mix of every column's goal into a plan.
//! 4. **Choosing** (basal ganglia, one critic, TD): candidates are the cortex's goals and the
//!    plan, and the motor primitives that are spontaneously active (as brainstem pattern
//!    generators make an infant babble and flail). The associative loop's gate takes one into
//!    the loop (or none); the motor loop's gate of the effector it belongs to releases it to the
//!    body or withholds it, and then it runs inwardly (imagined: the forward model's result,
//!    top-down, completed by the hippocampus). Working memory's gate loads what is heard.

use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};

use crate::bitvec::BitVector;
use crate::fixed::ONE;
use crate::program::area::{Brain, Kind};
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
/// the areas the hippocampus's pointers reach, each a slot of BITS cells
pub const SLOTS: [usize; 4] = [EYE, EAR, HAND, ACTS];

/// The striatum's channels.
const CH_LOAD: usize = 3;
const CH_ACT: usize = 5;
const CH_IMAGINE: usize = 7;
const CH_SPEAK: usize = 9;
/// where candidates sit in the striatum's input ids: goals, then each effector's primitives
const ACT_BASE: usize = 1 << 25;
const EFFORT: i32 = (ONE / 16) as i32;

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

/// What the cortex did this step, kept to learn from what comes next.
struct Step {
    /// per association column, its input and expectation; the tuft; per sense, (area, code, expectation)
    ins: Vec<(BitVector, BitVector)>,
    tuft: BitVector,
    sense_ins: Vec<(usize, BitVector, BitVector)>,
    mossy: BitVector,
    ctx: BitVector,
    props: Vec<(u8, Driver, BitVector)>,
    plan: BitVector,
}

/// A candidate for the gates.
#[derive(Clone)]
enum Cand {
    /// a goal of the cortex (a sensory pattern meant), for whichever effector can produce it
    Goal(BitVector),
    /// a motor primitive spontaneously active: (effector area, command)
    Primitive(usize, BitVector),
}

/// What the mind did, for an observer (never fed back).
#[derive(Clone, Default, Debug)]
pub struct Report {
    pub spoke: bool,
    pub moved: bool,
    /// a candidate taken into the loop and withheld (run inwardly)
    pub inward: bool,
    /// the dopamine-producing valuations this step, `Q16`: (own effect, response, smile, effort)
    pub valued: [i32; 4],
    /// the hippocampus's offer reached the cortex
    pub recalled: bool,
}

pub struct Mind {
    rng: StdRng,
    pub brain: Brain,
    cb: CerebellarCircuit,
    relay: ThalamicRelay,
    wm: WorkingMemory,
    pub bg: MaskStriatum,
    pub hc: IndexHc,
    body: Body,
    rec23: BitVector,
    pons: BitVector,
    /// top-down input for the next step, per area: what is imagined
    topdown: Vec<(usize, BitVector)>,
    prev_eye: BitVector,
    prev_ear: BitVector,
    last_assoc: BitVector,
    prev_assoc: BitVector,
    last_recall: BitVector,
    last: Option<Step>,
    last_plan: BitVector,
    /// the commands issued last step (efference copies), per effector
    voice_cmd: Option<BitVector>,
    hand_cmd: Option<BitVector>,
    /// steps since its own sound was last heard aloud (a reply soon after is contingent)
    since_said: usize,
    /// the share of steps a primitive of each effector is spontaneously active (1 in this many)
    pub spontaneous: u32,
    pub report: Report,
}

impl Mind {
    pub fn new(body: Body, seed: u64) -> Self {
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
        Self {
            rng: StdRng::seed_from_u64(seed ^ 0x313D),
            brain: b,
            cb,
            relay: ThalamicRelay::new(),
            wm: WorkingMemory::new(BITS, 1),
            bg,
            hc: IndexHc::new(SLOTS.len() * BITS, seed),
            body,
            rec23: zero(),
            pons: zero(),
            topdown: Vec::new(),
            prev_eye: zero(),
            prev_ear: zero(),
            last_assoc: zero(),
            prev_assoc: zero(),
            last_recall: zero(),
            last: None,
            last_plan: zero(),
            voice_cmd: None,
            hand_cmd: None,
            since_said: usize::MAX,
            spontaneous: 2,
            report: Report::default(),
        }
    }

    /// One step: senses in, acts out.
    pub fn step(&mut self, s: &Senses) -> Acts {
        self.report = Report::default();
        let ear_onset = !same(&s.ear, &self.prev_ear);
        self.prev_ear = s.ear.clone();
        // its own voice: a sound heard right after a command to the voice (corollary discharge)
        let own = self.voice_cmd.is_some();
        // 1. learning what came: the cortex, from its last state; the motor areas, each act's result
        if let Some(st) = self.last.take() {
            if self.hand_cmd.is_some() {
                self.learn(&st, &s.touch);
            } else {
                self.learn(&st, &s.ear);
            }
        }
        if let Some(c) = self.voice_cmd.as_ref() {
            let v = &mut self.brain.areas[VOICE];
            v.sense(&s.ear, false);
            if let Some(m) = v.motor.as_mut() {
                m.observe(c, &s.ear, &mut self.rng);
            }
            // the dorsal speech stream: the sound heard with the command that made it
            let a4 = self.brain.encode(EAR, &s.ear, true);
            self.brain.associate_codes(&[zero(), a4, zero(), c.clone()], true);
        }
        if let Some(c) = self.hand_cmd.as_ref() {
            if let Some(m) = self.brain.areas[HAND].motor.as_mut() {
                m.observe(c, &s.touch, &mut self.rng);
            }
        }
        // 2. sensing: the hippocampus first (what the cortex represents of what came), then the
        // cortex with what the hippocampus offers
        let e4 = self.brain.encode(EYE, &s.eye, false);
        let a4 = self.brain.encode(EAR, &s.ear, false);
        let cells = multi(&[(EYE, &e4), (EAR, &a4)]);
        let external = (ear_onset && !own) || !same(&s.eye, &self.prev_eye) && s.eye.count_ones() > 0;
        // how unexpected the sound is to the cortex (its plan was what it expected next)
        let surprise = ONE - overlap(&self.last_plan, &s.ear).min(ACTIVE as u32) * ONE / ACTIVE as u32;
        if external || own {
            self.hc.sense(&cells, external, own, surprise);
            self.prev_assoc = std::mem::replace(&mut self.last_assoc, cells);
        }
        let recall = self.hc.complete();
        self.report.recalled = recall.count_ones() > 0;
        self.last_recall = recall.clone();
        let st = self.cortex(s, &recall);
        self.last_plan = st.plan.clone();
        // 3. valuing what came (after the senses' columns have compared it with what they expected)
        let mut r = [0i32; 4];
        if own {
            r[0] += self.brain.surprise(EAR) as i32 / 4;
            self.since_said = 0;
        } else {
            self.since_said = self.since_said.saturating_add(1);
        }
        if self.hand_cmd.is_some() {
            r[0] += self.brain.surprise(EYE) as i32 / 4;
        }
        let someone = s.presence.count_ones() > 0;
        if someone && ear_onset && !own && (1..=2).contains(&self.since_said) {
            r[1] += ONE as i32 / 2;
        }
        if overlap(&s.presence, &self.body.smile) * 2 >= self.body.smile.count_ones() as u32 {
            r[2] += ONE as i32;
        }
        if self.voice_cmd.is_some() || self.hand_cmd.is_some() {
            r[3] -= EFFORT;
        }
        for &x in &r {
            if x != 0 {
                self.bg.reward(x);
            }
        }
        self.report.valued = r;
        // 4. choosing: the state the striatum sees, then the gates
        let eye4 = self.brain.encode(EYE, &s.eye, false);
        let presence4 = self.brain.encode(PRESENCE, &s.presence, s.presence.count_ones() > 0);
        let state = state_groups(&self.wm.content(), &self.last_assoc, &self.prev_assoc, &self.rec23, &self.pons, &eye4, &self.last_recall, &presence4);
        self.bg.begin_groups(&state, true);
        if ear_onset && !own && self.bg.choose(CH_LOAD, 2, true) == 1 {
            self.wm.load(0, &s.ear);
        }
        // curiosity: where the cortex's goals are getting better, choices loosen a little
        self.bg.extra_noise = self.brain.goal_progress().clamp(0, 2);
        let acts = self.choose(&st);
        self.bg.extra_noise = 0;
        self.prev_eye = s.eye.clone();
        self.last = Some(st);
        acts
    }

    /// The candidates, the two gates, and the act (or the inward run).
    fn choose(&mut self, st: &Step) -> Acts {
        let mut cands: Vec<Cand> = Vec::new();
        if st.plan.count_ones() > 0 {
            cands.push(Cand::Goal(st.plan.clone()));
        }
        for (_, p, _) in self.brain.goal_proposals() {
            if cands.len() < 8 && !cands.iter().any(|c| matches!(c, Cand::Goal(g) if overlap(g, &p) * 2 >= ACTIVE as u32)) {
                cands.push(Cand::Goal(p));
            }
        }
        // spontaneous motor primitives (pattern generators), one per effector now and then
        if self.rng.gen_range(0..self.spontaneous) == 0 {
            let k = self.rng.gen_range(0..self.body.voice.len());
            cands.push(Cand::Primitive(VOICE, self.body.voice[k].clone()));
        }
        if self.rng.gen_range(0..self.spontaneous) == 0 {
            let k = self.rng.gen_range(0..self.body.hand.len());
            cands.push(Cand::Primitive(HAND, self.body.hand[k].clone()));
        }
        let offer: Vec<Vec<Vec<usize>>> = cands
            .iter()
            .map(|c| {
                let (base, v) = match c {
                    Cand::Goal(g) => (ACT_BASE, g),
                    Cand::Primitive(a, cmd) => (ACT_BASE + if *a == VOICE { BITS } else { 2 * BITS }, cmd),
                };
                vec![ones(v).into_iter().map(|b| base + b).collect()]
            })
            .collect();
        self.voice_cmd = None;
        self.hand_cmd = None;
        let mut acts = Acts::default();
        let Some(i) = self.bg.choose_among(CH_IMAGINE, &offer, true) else {
            // nothing taken into the loop: the cortex's own state moves the hippocampus's context
            if st.plan.count_ones() > 0 {
                let e4 = self.brain.encode_only(EAR, &st.plan);
                self.hc.represent(&ones(&multi(&[(EAR, &e4)])));
            }
            return acts;
        };
        // which effector, and its command
        let (motor, cmd) = match &cands[i] {
            Cand::Primitive(a, c) => (*a, Some(c.clone())),
            Cand::Goal(g) => {
                let v = self.command(VOICE, g);
                let h = self.command(HAND, g);
                let fit = |m: &mut Mind, a: usize, c: &Option<BitVector>| c.as_ref().and_then(|c| m.brain.areas[a].motor.as_mut().and_then(|x| x.predict(c))).map_or(0, |p| overlap(&p, g));
                let (fv, fh) = (fit(self, VOICE, &v), fit(self, HAND, &h));
                if fv == 0 && fh == 0 {
                    (VOICE, v.or(h))
                } else if fv >= fh {
                    (VOICE, v)
                } else {
                    (HAND, h)
                }
            }
        };
        let Some(cmd) = cmd else { return acts };
        let gate = if motor == VOICE { CH_SPEAK } else { CH_ACT };
        if self.bg.choose_among(gate, &[offer[i].clone()], true).is_some() {
            if motor == VOICE {
                self.brain.areas[VOICE].issue(&cmd);
                self.voice_cmd = Some(cmd.clone());
                acts.voice = Some(cmd);
                self.report.spoke = true;
            } else {
                self.brain.areas[HAND].issue(&cmd);
                self.hand_cmd = Some(cmd.clone());
                acts.hand = Some(cmd);
                self.report.moved = true;
            }
        } else {
            // withheld: it runs inwardly
            let cue = self.imagine(motor, &cmd);
            self.hc.represent(&ones(&cue));
            self.last_recall = self.hc.complete();
            self.report.inward = true;
        }
        acts
    }

    /// The command effector `a` would use for goal `g`: for the voice, the imitation route first
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
    /// would look; the voice: the sound). Returns the cells the cortex then represents.
    fn imagine(&mut self, motor: usize, cmd: &BitVector) -> BitVector {
        let b = &mut self.brain;
        let Some(result) = b.areas[motor].motor.as_mut().and_then(|m| m.predict(cmd)) else { return BitVector::new(SLOTS.len() * BITS, Some(0)) };
        let parts: Vec<(usize, BitVector)> = if motor == HAND {
            let h = b.encode_only(HAND, &result);
            let seen = b.mirror(HAND, &h, ACTS);
            let e = b.encode_only(ACTS, &seen);
            vec![(HAND, h), (ACTS, e)]
        } else {
            vec![(EAR, b.encode_only(EAR, &result))]
        };
        let out = multi(&parts.iter().map(|(a, c)| (*a, c)).collect::<Vec<_>>());
        self.topdown.extend(parts);
        out
    }

    /// The cortex's step: every column senses, the association columns grow and predict, every
    /// layer 5 offers a goal, and the relay mixes them into the plan.
    fn cortex(&mut self, s: &Senses, recall: &BitVector) -> Step {
        let b = &mut self.brain;
        let e4 = b.encode(EYE, &s.eye, true);
        let a4 = b.encode(EAR, &s.ear, true);
        let mut h4 = b.encode(HAND, &s.touch, true);
        let pfc = self.wm.content();
        let mossy = frames(&[&s.ear, &self.pons]);
        let cb_out = self.cb.predict(&mossy).clone();
        let mut props: Vec<(u8, Driver, BitVector)> = Vec::new();
        if cb_out.count_ones() > 0 {
            props.push((2, Driver::Tonic(0), cb_out.clone()));
        }
        // the motion stream: what changed in view; through the mirror, how it would feel
        let mut onset = s.eye.clone();
        for (w, x) in onset.as_words_mut().iter_mut().zip(self.prev_eye.as_words()) {
            *w &= !x;
        }
        let acts4 = b.encode(ACTS, &onset, true);
        if acts4.count_ones() > 0 {
            let res = b.mirror(ACTS, &acts4, HAND);
            if res.count_ones() > 0 {
                h4 = b.encode(HAND, &or(&s.touch, &res), true);
            }
        }
        let (fb, _) = b.columns_activity();
        let mut td: Vec<BitVector> = SLOTS.iter().map(|&a| part(recall, a)).collect();
        for (a, c) in std::mem::take(&mut self.topdown) {
            if let Some(slot) = SLOTS.iter().position(|&x| x == a) {
                td[slot] = or(&td[slot], &c);
            }
        }
        let ff = [e4.clone(), a4.clone(), h4.clone(), acts4.clone()];
        let mut both: Vec<BitVector> = Vec::new();
        for (slot, &a) in SLOTS.iter().enumerate() {
            let sensed = ff[slot].count_ones() > 0;
            let input = or(&ff[slot], &td[slot]);
            b.sense_step(a, &input, Some(&or(&fb, &td[slot])), sensed);
            both.push(input);
        }
        b.associate_codes(&[e4.clone(), a4.clone(), h4.clone(), zero(), acts4.clone()], true);
        let tuft = or(&pfc, &cb_out);
        let outs = b.step_columns(&[both[0].clone(), both[1].clone(), both[2].clone(), zero(), both[3].clone()], &[], &tuft, true);
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
        let (act, fired) = b.columns_activity();
        self.rec23 = act;
        self.pons = project(&fired);
        let mut spiked = false;
        for (i, (goal, _, _)) in outs.iter().enumerate() {
            if let Some((o, burst)) = goal {
                props.push((10 + i.min(19) as u8, if *burst { Driver::Burst } else { Driver::Spike }, o.clone()));
                spiked |= !burst;
            }
        }
        self.relay.step(spiked);
        let ctx = frames(&[&pfc, &s.ear]);
        let plan = self.mix(&props, &ctx, &cb_out);
        let ins = outs.into_iter().map(|(_, x, e)| (x, e)).collect();
        Step { ins, tuft, sense_ins, mossy, ctx, props, plan }
    }

    /// The relay mixes the sources' proposals, each weighted by its learned reliability in this
    /// context; the strongest 32 bits.
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
        let mut plan = zero();
        for &b in order.iter().take(ACTIVE) {
            plan.bit_set(b);
        }
        plan
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
        let mut hc = std::mem::replace(&mut self.hc, IndexHc::new(SLOTS.len() * BITS, 0));
        hc.replay(&mut rng, chains, |frame| {
            let blank = Senses { eye: zero(), ear: zero(), touch: zero(), presence: zero() };
            let st = self.cortex(&blank, frame);
            let e4 = self.brain.encode_only(EAR, &st.plan);
            ones(&multi(&[(EAR, &e4)]))
        });
        self.hc = hc;
        self.last = None;
    }

    /// The critic's value now (`Q16`).
    pub fn value(&self) -> i32 {
        self.bg.value()
    }
}

/// What the striatum sees, as groups of active bits, one per input pathway.
fn state_groups(pfc: &BitVector, heard: &BitVector, before: &BitVector, l23: &BitVector, l5: &BitVector, eye4: &BitVector, recall: &BitVector, presence: &BitVector) -> Vec<Vec<usize>> {
    let off = |v: &BitVector, k: usize| ones(v).into_iter().map(|b| b + k * 4 * BITS).collect::<Vec<usize>>();
    vec![off(pfc, 0), off(heard, 1), off(before, 2), off(l23, 3), off(l5, 4), off(eye4, 5), off(recall, 6), off(presence, 7)]
}
