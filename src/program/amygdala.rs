//! A fast learner of what cues are worth, and of which cues deserve attention (after the
//! basolateral and central amygdala), with the striatum's cholinergic interneurons' learned pause.
//! Bitwise throughout: populations of mask cells (AND and popcount), as the striatum's.
//!
//! - **Cues are onsets:** a feature (a cell of some sensing area's layer 4 code, `slot * bits +
//!   cell`) that was not active the step before. Each onset stays in a trace for a few steps,
//!   tagged by its age (time-locked "microstimuli": the cue 1, 2, 3 steps back are different
//!   inputs), so value can attach to a cue across a delay. The tagged trace is hashed into a
//!   context bit vector of `D` bits.
//! - **Value** (the basolateral amygdala, which gets sensory input directly and learns in a few
//!   pairings): two pools of value cells, appetitive and aversive, each cell a mask of 16 context
//!   bits that fires when 12 are present. The value predicted is the appetitive cells firing minus
//!   the aversive, a unit each. Learning follows Rescorla–Wagner's shared error: the outcome came,
//!   the error is what came minus what the trace predicted. A positive error removes aversive
//!   cells that fired, recruits appetitive cells on the trace's context for the rest of the error
//!   (one per unit) and moves the firing ones toward it; a negative error removes appetitive cells
//!   that fired (extinction), or recruits aversive ones. The cues present add up, so a cue that
//!   predicts nothing does not dilute one that does.
//! - **Attention** (Pearce–Hall's surprise, Mackintosh's good predictors; the central amygdala and
//!   cholinergic basal forebrain): a bit vector over features. Features in the trace when the
//!   error or the predicted value is large are set (they stay attended while their value is being
//!   learned, and while it holds); each step a random 1/32 of the bits clear; a feature never seen
//!   before is attended (novelty). Only attended features reach the striatum.
//! - **The cholinergic pause** (TANs: Aosaki et al. 1994; Morris et al. 2004): pause cells, masks
//!   over the features (untagged) of the cues one or two steps before a large dopamine error,
//!   recruited then; a pause cell that fires when no large error follows is removed now and then.
//!   When one fires on the cues now, the interneurons pause and striatal plasticity opens.

use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};

use crate::bitvec::BitVector;
use crate::fixed::ONE;

/// Steps an onset stays in the trace.
pub const TRACE: usize = 4;
/// Features attended at most (reach the striatum).
const ATTEND: usize = 64;
/// Context bits (the tagged trace, hashed).
const D: usize = 4096;
/// Bits per cell, and how many must be present for it to fire.
const CELL_BITS: usize = 16;
const THETA: u32 = 12;
/// Cells per pool at most (the stalest goes).
const CAP: usize = 2048;
/// The value of one cell (`Q16`).
const UNIT: i32 = (ONE / 16) as i32;

fn mix(x: u64) -> u64 {
    let mut y = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    y ^= y >> 29;
    y = y.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    y ^ (y >> 31)
}

fn bits_of(v: &[u64]) -> Vec<usize> {
    let mut out = Vec::new();
    for (w, &x) in v.iter().enumerate() {
        let mut x = x;
        while x != 0 {
            out.push(w * 64 + x.trailing_zeros() as usize);
            x &= x - 1;
        }
    }
    out
}

/// A pool of mask cells: (mask, the step it last fired or was made, a stable id).
#[derive(Default)]
struct Pool {
    cells: Vec<(Vec<u64>, u64, u32)>,
    next_id: u32,
}

impl Pool {
    fn firing(&self, ctx: &[u64]) -> Vec<usize> {
        self.cells.iter().enumerate().filter(|(_, (m, _, _))| m.iter().zip(ctx).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= THETA).map(|(i, _)| i).collect()
    }

    /// A new cell on `CELL_BITS` of the context's bits (none if it has too few).
    fn recruit(&mut self, ctx: &[u64], now: u64, rng: &mut StdRng) {
        let mut on = bits_of(ctx);
        if on.len() < CELL_BITS {
            return;
        }
        if self.cells.len() >= CAP {
            let i = (0..8).map(|_| rng.gen_range(0..self.cells.len())).min_by_key(|&i| self.cells[i].1).unwrap();
            self.cells.swap_remove(i);
        }
        let mut m = vec![0u64; ctx.len()];
        for i in 0..CELL_BITS {
            let j = rng.gen_range(i..on.len());
            on.swap(i, j);
            m[on[i] / 64] |= 1 << (on[i] % 64);
        }
        self.cells.push((m, now, self.next_id));
        self.next_id = self.next_id.wrapping_add(1);
    }

    /// Cell `i` moves one bit toward the context: a bit it has that is absent goes, a present one
    /// it lacks comes (constant density).
    fn toward(&mut self, i: usize, ctx: &[u64], now: u64, rng: &mut StdRng) {
        let (m, t, _) = &mut self.cells[i];
        *t = now;
        let off = bits_of(&m.iter().zip(ctx).map(|(a, c)| a & !c).collect::<Vec<u64>>());
        let add = bits_of(&m.iter().zip(ctx).map(|(a, c)| c & !a).collect::<Vec<u64>>());
        if off.is_empty() || add.is_empty() {
            return;
        }
        let (o, n) = (off[rng.gen_range(0..off.len())], add[rng.gen_range(0..add.len())]);
        m[o / 64] &= !(1 << (o % 64));
        m[n / 64] |= 1 << (n % 64);
    }

    /// Remove up to `k` of the cells `among`, chosen at random.
    fn remove(&mut self, among: &[usize], k: usize, rng: &mut StdRng) {
        let mut ids = among.to_vec();
        let mut gone = Vec::new();
        for _ in 0..k.min(ids.len()) {
            let j = rng.gen_range(0..ids.len());
            gone.push(ids.swap_remove(j));
        }
        gone.sort_unstable_by(|a, b| b.cmp(a));
        for i in gone {
            self.cells.swap_remove(i);
        }
    }
}

pub struct Amygdala {
    /// features in all (every sensing area's cells, area by area)
    feat: usize,
    /// onsets of the last steps, newest first
    trace: Vec<Vec<u32>>,
    appetitive: Pool,
    aversive: Pool,
    pause: Pool,
    /// with `act_from`: baseline cells on the act's features alone (what the act is worth on
    /// average, whatever the cues), recruited and removed one at a time toward the outcomes, so
    /// they track the average and are never extinguished; the other cells learn what the cues
    /// add to it
    baseline: Pool,
    /// features from this index on are acts (efference copies); None: no baseline
    act_from: Option<u32>,
    /// attended features, and features seen before (bit vectors over the features)
    attended: BitVector,
    seen: BitVector,
    now: u64,
    rng: StdRng,
    /// (updates, sum of |error|)
    pub stats: (u64, u64),
}

impl Amygdala {
    /// `feat`: how many features there are (every sensing area's cells).
    pub fn new(feat: usize, seed: u64) -> Self {
        Self {
            feat,
            trace: Vec::new(),
            appetitive: Pool::default(),
            aversive: Pool::default(),
            pause: Pool::default(),
            baseline: Pool::default(),
            act_from: None,
            attended: BitVector::new(feat, Some(0)),
            seen: BitVector::new(feat, Some(0)),
            now: 0,
            rng: StdRng::seed_from_u64(seed ^ 0xA316),
            stats: (0, 0),
        }
    }

    /// Features from `from` on are acts: keep a baseline value per act (see `baseline`).
    pub fn with_act_baseline(mut self, from: u32) -> Self {
        self.act_from = Some(from);
        self
    }

    /// The context of the act features among some items (empty without a baseline).
    fn act_context(&self, items: &[(usize, u32)]) -> Vec<u64> {
        match self.act_from {
            Some(t) => Self::context(items.iter().filter(|x| x.1 >= t).map(|(a, f)| (*a, f)), true),
            None => vec![0u64; D / 64],
        }
    }

    /// The value of some items: appetitive minus aversive cells firing on their context, plus the
    /// baseline cells firing on their acts' (`Q16`, a unit per cell).
    fn value_items(&self, items: &[(usize, u32)]) -> i32 {
        let ctx = Self::context(items.iter().map(|(a, f)| (*a, f)), true);
        let base = self.baseline.firing(&self.act_context(items)).len() as i32;
        let n = self.appetitive.firing(&ctx).len() as i32 - self.aversive.firing(&ctx).len() as i32 + base;
        (n * UNIT).clamp(-2 * ONE as i32, 2 * ONE as i32)
    }

    /// The trace's items from earlier steps (ages 1..), as owned (age, feature).
    fn past_items(&self) -> Vec<(usize, u32)> {
        self.items(1, TRACE).map(|(a, &f)| (a, f)).collect()
    }

    /// The trace's items now, each a step older (what they predict for the next step).
    fn now_items(&self) -> Vec<(usize, u32)> {
        self.items(0, TRACE - 1).map(|(a, &f)| (a + 1, f)).collect()
    }

    /// The context of some traced (age, feature) items: each hashed to one of `D` bits
    /// (`tagged`: the age is part of it; else the feature alone).
    fn context<'a>(items: impl Iterator<Item = (usize, &'a u32)>, tagged: bool) -> Vec<u64> {
        let mut c = vec![0u64; D / 64];
        for (age, &f) in items {
            let id = if tagged { f as u64 * TRACE as u64 + age as u64 } else { f as u64 ^ 0xF00D_0000_0000 };
            let b = (mix(id) % D as u64) as usize;
            c[b / 64] |= 1 << (b % 64);
        }
        c
    }

    /// The trace's items at ages `ages`.
    fn items(&self, skip: usize, take: usize) -> impl Iterator<Item = (usize, &u32)> {
        self.trace.iter().enumerate().skip(skip).take(take).flat_map(|(a, fs)| fs.iter().map(move |f| (a, f)))
    }

    /// The context of the trace from earlier steps (ages 1..): what predicts this step.
    fn past(&self) -> Vec<u64> {
        Self::context(self.items(1, TRACE), true)
    }

    /// This step's onsets (features newly active). Shifts the trace; a feature never seen before
    /// is attended (novelty); attention fades (a random 1/32 of the bits clear).
    pub fn onsets(&mut self, onsets: Vec<u32>) {
        self.now += 1;
        let onsets: Vec<u32> = onsets.into_iter().filter(|&f| (f as usize) < self.feat).collect();
        for &f in &onsets {
            if !self.seen.bit_get(f as usize) {
                self.seen.bit_set(f as usize);
                self.attended.bit_set(f as usize);
            }
        }
        self.trace.insert(0, onsets);
        self.trace.truncate(TRACE);
        let rng = &mut self.rng;
        for w in self.attended.as_words_mut().iter_mut() {
            // a bit clears where all five random words are 0: 1/32
            let keep = (0..5).fold(0u64, |k, _| k | rng.next_u64());
            *w &= keep;
        }
    }

    /// What the trace predicts for this step (`Q16`).
    pub fn predicted(&self) -> i32 {
        self.value_items(&self.past_items())
    }

    /// The innate valuations of outcomes `r` (`Q16`) came this step: the value cells learn the
    /// shared error, and the features traced when the error or the prediction is large are
    /// attended. Returns the error.
    pub fn learn(&mut self, r: i32) -> i32 {
        let ctx = self.past();
        if ctx.iter().all(|&w| w == 0) {
            return 0;
        }
        let past = self.past_items();
        let v = self.value_items(&past);
        let err = r - v;
        // the baseline: one cell at a time toward the outcome on its own (an average per act)
        let actx = self.act_context(&past);
        if actx.iter().any(|&w| w != 0) {
            let firing = self.baseline.firing(&actx);
            let eb = r - firing.len() as i32 * UNIT;
            if eb >= UNIT {
                self.baseline.recruit(&actx, self.now, &mut self.rng);
            } else if eb <= -UNIT {
                self.baseline.remove(&firing, 1, &mut self.rng);
            }
        }
        self.stats.0 += 1;
        self.stats.1 += err.unsigned_abs() as u64;
        let k = (err.abs() / UNIT).min(4) as usize;
        let (app, avr) = (self.appetitive.firing(&ctx), self.aversive.firing(&ctx));
        if err >= UNIT {
            for &i in &app {
                self.appetitive.toward(i, &ctx, self.now, &mut self.rng);
            }
            let gone = avr.len().min(k);
            self.aversive.remove(&avr, gone, &mut self.rng);
            for _ in gone..k {
                self.appetitive.recruit(&ctx, self.now, &mut self.rng);
            }
        } else if err <= -UNIT {
            for &i in &avr {
                self.aversive.toward(i, &ctx, self.now, &mut self.rng);
            }
            let gone = app.len().min(k);
            self.appetitive.remove(&app, gone, &mut self.rng);
            if r < 0 {
                for _ in gone..k {
                    self.aversive.recruit(&ctx, self.now, &mut self.rng);
                }
            }
        }
        // attention: surprise (Pearce–Hall) or a cue worth something (Mackintosh)
        if err.abs() >= 2 * UNIT || v.abs() >= 2 * UNIT {
            let fs: Vec<u32> = self.items(1, TRACE).map(|(_, &f)| f).collect();
            for f in fs {
                self.attended.bit_set(f as usize);
            }
        }
        err
    }

    /// The attended features in the trace, tagged by age (`feature * TRACE + age`): what the
    /// amygdala and the time-locked trace send the striatum (at most `ATTEND`, newest first).
    pub fn attended(&self) -> Vec<usize> {
        self.items(0, TRACE).filter(|(_, f)| self.attended.bit_get(**f as usize)).take(ATTEND).map(|(a, &f)| f as usize * TRACE + a).collect()
    }

    /// The value now of the cues in the trace (`Q16`): what the newest cues (one step old next
    /// step) and the older ones predict.
    pub fn cue_value(&self) -> i32 {
        self.value_items(&self.now_items())
    }

    /// The value cells firing on the cues now (as `cue_value` reads them): appetitive cells' ids
    /// below 4096, aversive ones' from 4096, baseline ones' from 8192, each modulo 4096 (an input group for the striatum).
    pub fn value_cells(&self) -> Vec<usize> {
        let ctx = Self::context(self.items(0, TRACE - 1).map(|(a, f)| (a + 1, f)), true);
        let app = self.appetitive.firing(&ctx).into_iter().map(|i| self.appetitive.cells[i].2 as usize % 4096);
        let avr = self.aversive.firing(&ctx).into_iter().map(|i| 4096 + self.aversive.cells[i].2 as usize % 4096);
        let actx = self.act_context(&self.now_items());
        let base = self.baseline.firing(&actx).into_iter().map(|i| 8192 + self.baseline.cells[i].2 as usize % 4096);
        app.chain(avr).chain(base).collect()
    }

    /// An act's usual worth (`Q16`): the baseline cells firing on its features (its efference
    /// copy) as one step old; 0 without a baseline.
    pub fn act_worth(&self, features: &[u32]) -> i32 {
        let items: Vec<(usize, u32)> = features.iter().map(|&f| (1usize, f)).collect();
        self.baseline.firing(&self.act_context(&items)).len() as i32 * UNIT
    }

    /// What some features would be worth if they came now (an imagined outcome: they predict
    /// the next step as one step old), `Q16`.
    pub fn value_of(&self, features: &[u32]) -> i32 {
        self.value_items(&features.iter().map(|&f| (1usize, f)).collect::<Vec<_>>())
    }

    /// The dopamine error `delta` came (`Q16`): if it is large, a pause cell is recruited on the
    /// cues of one and two steps before (unless one fires on them already); a pause cell that
    /// fired on those cues with no large error is removed now and then. Returns whether the
    /// interneurons pause now.
    pub fn learn_pause(&mut self, delta: i32, large: i32) -> bool {
        let before = Self::context(self.items(1, 2), false);
        if before.iter().any(|&w| w != 0) {
            let firing = self.pause.firing(&before);
            if delta.abs() >= large {
                if firing.is_empty() {
                    self.pause.recruit(&before, self.now, &mut self.rng);
                }
            } else if !firing.is_empty() && self.rng.gen_range(0..4) == 0 {
                self.pause.remove(&firing, 1, &mut self.rng);
            }
        }
        self.pausing()
    }

    /// The interneurons pause: a pause cell fires on the cues now or one step before.
    pub fn pausing(&self) -> bool {
        !self.pause.firing(&Self::context(self.items(0, 2), false)).is_empty()
    }

    /// (this step's onsets, attended features in the trace), for diagnosis.
    pub fn trace_sizes(&self) -> (usize, usize) {
        (self.trace.first().map_or(0, |t| t.len()), self.attended().len())
    }

    /// (appetitive, aversive, pause, baseline) cells.
    pub fn cells(&self) -> (usize, usize, usize, usize) {
        (self.appetitive.cells.len(), self.aversive.cells.len(), self.pause.cells.len(), self.baseline.cells.len())
    }
}
