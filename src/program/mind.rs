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
//!    - *effort:* every act costs a little.
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
use crate::program::area::{Brain, ColumnId, Kind};
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
const EFFORT: i32 = (ONE / 16) as i32;
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
    /// the dopamine-producing valuations this step, `Q16`: (own effect, response, smile, effort)
    pub valued: [i32; 4],
    /// the hippocampus's offer reached the cortex
    pub recalled: bool,
    /// columns with a loop
    pub loops: usize,
}

pub struct Mind {
    rng: StdRng,
    pub brain: Brain,
    cb: CerebellarCircuit,
    relay: ThalamicRelay,
    wm: WorkingMemory,
    pub bg: MaskStriatum,
    pub hc: IndexHc,
    /// each motor area, its primitives (the genome's pattern generators), and the sensing area
    /// its acts are sensed by (its result)
    motors: Vec<(usize, Vec<BitVector>, usize)>,
    smile: BitVector,
    pons: BitVector,
    /// top-down input for the next step, per area: what is imagined
    topdown: Vec<(usize, BitVector)>,
    /// each sensing area's input at the last step (its changes are attention outward)
    prev_in: Vec<BitVector>,
    last_recall: BitVector,
    last: Option<Step>,
    last_expect: BitVector,
    /// the command issued last step per motor area (the efference copy)
    issued: Vec<(usize, BitVector)>,
    /// motor areas whose act's result is sensed this step
    acted: usize,
    /// goals followed recently: (column, cell), newest last
    followed: Vec<Vec<(ColumnId, usize)>>,
    /// steps since its own sound was last heard aloud (a reply soon after is contingent)
    since_said: usize,
    /// a primitive of each effector is spontaneously active 1 step in this many
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
            motors: vec![(VOICE, body.voice, EAR), (HAND, body.hand, HAND)],
            smile: body.smile,
            pons: zero(),
            topdown: Vec::new(),
            prev_in: vec![zero(); SLOTS.len()],
            last_recall: zero(),
            last: None,
            last_expect: zero(),
            issued: Vec::new(),
            acted: 0,
            followed: Vec::new(),
            since_said: usize::MAX,
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
        for (a, c) in std::mem::take(&mut self.issued) {
            let sense = self.motors.iter().find(|m| m.0 == a).map_or(EAR, |m| m.2);
            let result = raw[Self::slot(sense)].clone();
            let area = &mut self.brain.areas[a];
            area.sense(&result, false);
            if let Some(m) = area.motor.as_mut() {
                m.observe(&c, &result, &mut self.rng);
            }
            // what the act produced, sensed with the command that made it (the dorsal stream)
            let r4 = self.brain.encode(sense, &result, true);
            let mut codes = vec![zero(); self.brain.areas.len()];
            codes[sense] = r4;
            codes[a] = c;
            self.brain.associate_codes(&codes, true);
        }
        // 2. the hippocampus: the whole cortical state (each area's code for what it senses, or
        // what is imagined there), attention outward where an input changed, one's own sound
        // marked by the efference copy
        let td = std::mem::take(&mut self.topdown);
        let mut parts: Vec<(usize, BitVector)> = Vec::new();
        let mut outward = false;
        for (k, &a) in SLOTS.iter().enumerate() {
            let mut c = self.brain.encode(a, &raw[k], false);
            if !(a == EAR && own) && raw[k].count_ones() > 0 && !same(&raw[k], &self.prev_in[k]) {
                outward = true;
            }
            for (_, t) in td.iter().filter(|x| x.0 == a) {
                c = or(&c, t);
            }
            parts.push((a, c));
        }
        let state = multi(&parts.iter().map(|(a, c)| (*a, c)).collect::<Vec<_>>());
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
            let sensed = SLOTS.iter().map(|&a| self.brain.surprise(a)).max().unwrap_or(0);
            r[0] += sensed as i32 / 4;
            r[3] -= EFFORT * self.acted as i32;
        }
        if own {
            self.since_said = 0;
        } else {
            self.since_said = self.since_said.saturating_add(1);
        }
        let ear_onset = !same(&s.ear, &self.prev_in[Self::slot(EAR)]);
        if s.presence.count_ones() > 0 && ear_onset && !own && (1..=2).contains(&self.since_said) {
            r[1] += ONE as i32 / 2;
        }
        if overlap(&s.presence, &self.smile) * 2 >= self.smile.count_ones() as u32 {
            r[2] += ONE as i32;
        }
        for &x in &r {
            if x != 0 {
                self.bg.reward(x);
            }
        }
        self.report.valued = r;
        // 5. the striatum sees every column's activity, working memory, and recall; the TD
        // error that closes the last step reaches the goal cells followed recently
        let ids = self.brain.column_ids();
        self.report.loops = ids.len();
        let mut groups: Vec<BitVector> = ids.iter().map(|&id| self.brain.activity_of(id)).collect();
        groups.push(self.wm.content());
        groups.push(self.last_recall.clone());
        let state: Vec<Vec<usize>> = groups.iter().enumerate().map(|(k, v)| ones(v).into_iter().map(|b| b + k * SLOTS.len() * BITS).collect()).collect();
        self.bg.begin_groups(&state, true);
        let delta = self.bg.last_delta;
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
        self.prev_in = raw;
        self.last = Some(st);
        acts
    }

    /// The loops: each column's takes its goal in or not (and an association column's may load
    /// working memory); each motor area's takes one candidate and releases or withholds it.
    fn choose(&mut self, ids: &[ColumnId]) -> Acts {
        let mut taken: Vec<(ColumnId, usize, BitVector)> = Vec::new();
        for &id in ids {
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
            let motor = self.motors[k].0;
            let (loop_ch, gate) = motor_channels(motor);
            let offer: Vec<Vec<Vec<usize>>> = cands.iter().map(|c| vec![c.1.clone()]).collect();
            let Some(i) = self.bg.choose_among(loop_ch, &offer, true) else { continue };
            let (cmd, _, from) = cands[i].clone();
            if let Some(f) = from {
                followed_now.push(f);
            }
            if self.bg.choose_among(gate, &[offer[i].clone()], true).is_some() {
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
    /// layer 5 offers a goal, and the relay mixes them into the expectation.
    fn cortex(&mut self, raw: &[BitVector], recall: &BitVector, td: Vec<(usize, BitVector)>) -> Step {
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
        let mut down: Vec<BitVector> = SLOTS.iter().map(|&a| part(recall, a)).collect();
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
        let outs = b.step_columns(&with_down, &[], &tuft, true);
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
        let mut hc = std::mem::replace(&mut self.hc, IndexHc::new(SLOTS.len() * BITS, 0));
        hc.replay(&mut rng, chains, |frame| {
            let blank = vec![zero(); SLOTS.len()];
            let st = self.cortex(&blank, frame, Vec::new());
            let e4 = self.brain.encode_only(EAR, &st.expect);
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
