//! Curiosity: which open questions are worth the effort of a search.
//!
//! An open question is a key whose answer is not settled (a family two sources dispute, a
//! proposal neither confirmed nor contradicted). A search (asking, looking back, replaying
//! what bears on it) costs something and brings evidence. Curiosity ranks the open
//! questions by the *value of information*: how much a search is expected to settle.
//!
//! - **What it sees.** Each question comes with a context: a small integer that describes
//!   its state (here: the band of the believed value's belief × the band of its lead over
//!   the runner-up, as for the answer-or-unknown go/no-go).
//! - **What it learns.** For each context, the information a search there has actually
//!   gained: the rise in the believed value's lead once the new evidence is weighed.
//! - **The prior is the question's own uncertainty.** Before any search in its context, a
//!   question is expected to gain what it lacks: one minus its lead (a split question up to
//!   a full unit, a settled one nothing), counted as one search's worth of evidence. The
//!   gains realised in its context then correct it: contexts where searching settles
//!   nothing (a decided question; an undecidable one whose search brings no evidence) lose
//!   their pull, and those where it pays keep it. (An optimistic prior of a full unit for
//!   every untried context explored contexts instead of asking about what was uncertain:
//!   with ten questions to spend, it spent them on settled names.)
//! - **What it does.** `pick` returns the questions with the highest expected gain, within
//!   a budget. The gain realised is the intrinsic reward (the dopamine signal for the value
//!   of information): `learn` takes it.
//! - **Or the network's own policy** (`pick_learned`): no ranking rule and no prior. The
//!   open questions are candidates to a basal-ganglia selector, each coded by its state
//!   (its belief band's bits joined with its lead band's, so what is learned about one
//!   state carries to its neighbours); the selector releases the question to ask, with
//!   exploration, and the gain realised is its reward. It learns which states of
//!   uncertainty a search pays off in, from what searching brought.
//! - **Cost, balanced by the network** (`decide`): a search uses energy (the compute it
//!   took), from a reserve that refills by a fixed power budget each sleep. At each step the
//!   selector chooses one of the open questions or *stop*; every choice is coded with the
//!   reserve's band and how far into the sleep it is (a sense of time). Stopping is worth a
//!   neutral one half; asking, one half plus half the information gained, minus the energy
//!   spent, weighed up to twice as heavily as the reserve empties. Nothing says when to stop:
//!   the selector learns per state, energy and time whether a question is worth its cost.
//!
//! Brain: the anterior cingulate and lateral habenula track uncertainty and when information
//! will arrive; midbrain dopamine neurons signal the value of information itself
//! (Bromberg-Martin & Hikosaka 2009); frontopolar cortex chooses to explore an uncertain
//! option (directed exploration). Integers only (`Q16`).

use std::hash::Hash;

use rand::Rng;

use crate::bitvec::BitVector;
use crate::det::HashMap;
use crate::fixed::ONE;
use crate::program::BasalGanglia;

/// Bits per band in a question's code for the learned policy.
const BAND_BITS: usize = 8;
/// Bands of the energy reserve and of time within a sleep, in the codes of `decide`.
pub const ENERGY_BANDS: usize = 4;
pub const TIME_BANDS: usize = 4;

pub struct Curiosity<K> {
    /// Per context: (gain realised, summed, `Q16`; searches).
    gain: Vec<(u64, u64)>,
    /// Open questions: key → (context, uncertainty `Q16`).
    open: HashMap<K, (usize, u32)>,
    searches: u64,
    gained: u64,
    stops: u64,
    spent: u64,
    /// The learned policy: a selector over question states, and the (belief, lead) bands
    /// a context splits into.
    bg: BasalGanglia,
    bands: (usize, usize),
}

impl<K: Hash + Eq + Clone + Ord> Curiosity<K> {
    /// Question states: `belief_bands` × `lead_bands` contexts (context = belief band ×
    /// `lead_bands` + lead band).
    pub fn new(belief_bands: usize, lead_bands: usize) -> Self {
        Self {
            gain: vec![(0, 0); belief_bands * lead_bands],
            open: HashMap::default(),
            searches: 0,
            gained: 0,
            stops: 0,
            spent: 0,
            bg: BasalGanglia::new((belief_bands + lead_bands + 1 + 2 * (ENERGY_BANDS + TIME_BANDS)) * BAND_BITS),
            bands: (belief_bands, lead_bands),
        }
    }

    fn width(&self) -> usize {
        (self.bands.0 + self.bands.1 + 1 + 2 * (ENERGY_BANDS + TIME_BANDS)) * BAND_BITS
    }

    /// A context's code for the selector: its belief band's bits and its lead band's.
    fn code(&self, context: usize) -> BitVector {
        BitVector::from_bits(&self.state_bits(context), self.width())
    }

    fn state_bits(&self, context: usize) -> Vec<usize> {
        let (bb, lb) = self.bands;
        let (b, l) = ((context / lb).min(bb - 1), context % lb);
        let mut bits: Vec<usize> = (b * BAND_BITS..(b + 1) * BAND_BITS).collect();
        bits.extend((bb + l) * BAND_BITS..(bb + l + 1) * BAND_BITS);
        bits
    }

    /// Block `i` (in bands) after the question-state blocks.
    fn block(&self, i: usize) -> std::ops::Range<usize> {
        let at = (self.bands.0 + self.bands.1 + i) * BAND_BITS;
        at..at + BAND_BITS
    }

    /// Asking about a question in `context`, at energy band `e` and time band `t`.
    fn ask_code(&self, context: usize, e: usize, t: usize) -> BitVector {
        let mut bits = self.state_bits(context);
        bits.extend(self.block(1 + e.min(ENERGY_BANDS - 1)));
        bits.extend(self.block(1 + ENERGY_BANDS + t.min(TIME_BANDS - 1)));
        BitVector::from_bits(&bits, self.width())
    }

    /// Stopping, at energy band `e` and time band `t`.
    fn stop_code(&self, e: usize, t: usize) -> BitVector {
        let mut bits: Vec<usize> = self.block(0).collect();
        bits.extend(self.block(1 + ENERGY_BANDS + TIME_BANDS + e.min(ENERGY_BANDS - 1)));
        bits.extend(self.block(1 + 2 * ENERGY_BANDS + TIME_BANDS + t.min(TIME_BANDS - 1)));
        BitVector::from_bits(&bits, self.width())
    }

    /// The energy band of a reserve (`Q16`).
    pub fn energy_band(energy: u32) -> usize {
        ((energy.min(ONE) as u64 * ENERGY_BANDS as u64) >> 16).min(ENERGY_BANDS as u64 - 1) as usize
    }

    /// One step of the network's own cost-aware policy: a question to ask (removed from the
    /// open ones until noted again), or None to stop. `energy` is the reserve (`Q16`),
    /// `time` the step's band within the sleep.
    pub fn decide<R: Rng>(&mut self, energy: u32, time: usize, rng: &mut R) -> Option<(K, usize)> {
        let e = Self::energy_band(energy);
        let mut qs: Vec<(K, usize)> = self.open.iter().map(|(k, &(c, _))| (k.clone(), c)).collect();
        qs.sort_by(|a, b| a.0.cmp(&b.0));
        let mut codes: Vec<BitVector> = qs.iter().map(|q| self.ask_code(q.1, e, time)).collect();
        codes.push(self.stop_code(e, time));
        let i = self.bg.select(&codes, Some(&mut *rng))?;
        if i == qs.len() {
            self.stops += 1;
            return None;
        }
        let (k, c) = qs.swap_remove(i);
        self.open.remove(&k);
        Some((k, c))
    }

    /// Reward a question asked by `decide`: it gained `gain` and cost `cost` (both `Q16`),
    /// asked at reserve `energy` before the cost and time band `time`.
    pub fn reward_ask<R: Rng>(&mut self, context: usize, energy: u32, time: usize, gain: u32, cost: u32, rng: &mut R) {
        // energy weighs more as the reserve empties: once at full, twice at empty
        let scarcity = 2 * ONE as u64 - energy.min(ONE) as u64;
        let price = (cost as u64 * scarcity) >> 16;
        let r = (ONE as i64 / 2 + gain.min(ONE) as i64 / 2 - price as i64).clamp(0, ONE as i64);
        let code = self.ask_code(context, Self::energy_band(energy), time);
        self.bg.reward_candidate(&code, r as i32, rng);
        let c = context.min(self.gain.len() - 1);
        self.gain[c].0 += gain as u64;
        self.gain[c].1 += 1;
        self.searches += 1;
        self.gained += gain as u64;
        self.spent += cost as u64;
    }

    /// Reward a stop (worth a neutral one half).
    pub fn reward_stop<R: Rng>(&mut self, energy: u32, time: usize, rng: &mut R) {
        let code = self.stop_code(Self::energy_band(energy), time);
        self.bg.reward_candidate(&code, ONE as i32 / 2, rng);
    }

    /// The learned values of asking about `context` and of stopping, at energy band `e`
    /// and time band `t` (`Q16`).
    pub fn values_at(&self, context: usize, e: usize, t: usize) -> (u32, u32) {
        (self.bg.value(&self.ask_code(context, e, t)), self.bg.value(&self.stop_code(e, t)))
    }

    /// (stops chosen, energy spent `Q16`).
    pub fn spending(&self) -> (u64, u64) {
        (self.stops, self.spent)
    }

    /// The network's own choice of `budget` questions: one at a time, the selector releases
    /// one of the open questions by its state (exploring with its own probability).
    pub fn pick_learned<R: Rng>(&mut self, budget: usize, rng: &mut R) -> Vec<K> {
        let mut qs: Vec<(K, usize)> = self.open.iter().map(|(k, &(c, _))| (k.clone(), c)).collect();
        qs.sort_by(|a, b| a.0.cmp(&b.0));
        let mut out = Vec::new();
        while out.len() < budget && !qs.is_empty() {
            let codes: Vec<BitVector> = qs.iter().map(|q| self.code(q.1)).collect();
            let Some(i) = self.bg.select(&codes, Some(&mut *rng)) else { break };
            out.push(qs.remove(i).0);
        }
        out
    }

    /// The selector's learned value of asking in `context` (`Q16`).
    pub fn learned_value(&self, context: usize) -> u32 {
        self.bg.value(&self.code(context))
    }

    /// Note an open question (or update its state): its context, and its uncertainty (`Q16`,
    /// one minus its answer's lead). A settled one is closed with `close`.
    pub fn note(&mut self, key: K, context: usize, uncertainty: u32) {
        self.open.insert(key, (context.min(self.gain.len() - 1), uncertainty.min(ONE)));
    }

    pub fn close(&mut self, key: &K) {
        self.open.remove(key);
    }

    /// The expected gain of a search on a question in `context` with `uncertainty` (`Q16`):
    /// the gains realised in that context, with the question's uncertainty as one search's
    /// worth of prior.
    pub fn expected_gain(&self, context: usize, uncertainty: u32) -> u32 {
        let (sum, n) = self.gain[context.min(self.gain.len() - 1)];
        ((sum + uncertainty as u64) / (n + 1)) as u32
    }

    /// The `budget` open questions most worth a search, best first (ties to the lower key).
    pub fn pick(&self, budget: usize) -> Vec<K> {
        let mut qs: Vec<(u32, &K)> = self.open.iter().map(|(k, &(c, u))| (self.expected_gain(c, u), k)).collect();
        qs.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(b.1)));
        qs.into_iter().take(budget).map(|x| x.1.clone()).collect()
    }

    /// A search in `context` gained `gain` (`Q16`, the rise in the answer's lead): the
    /// context's record, and the selector's reward.
    pub fn learn<R: Rng>(&mut self, context: usize, gain: u32, rng: &mut R) {
        let code = self.code(context);
        self.bg.reward_candidate(&code, gain.min(ONE) as i32, rng);
        let c = context.min(self.gain.len() - 1);
        let g = &mut self.gain[c];
        g.0 += gain as u64;
        g.1 += 1;
        self.searches += 1;
        self.gained += gain as u64;
    }

    /// (open questions, searches made, mean gain per search, `Q16`).
    pub fn stats(&self) -> (usize, u64, u32) {
        (self.open.len(), self.searches, (self.gained / self.searches.max(1)) as u32)
    }

    /// The contexts searched so far: (context, searches, mean gain `Q16`).
    pub fn learned(&self) -> Vec<(usize, u64, u32)> {
        self.gain.iter().enumerate().filter(|g| g.1 .1 > 0).map(|(c, g)| (c, g.1, (g.0 / g.1) as u32)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn searches_go_where_they_have_paid_off() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut c: Curiosity<u32> = Curiosity::new(2, 2);
        // keys 0..4 in context 0 (searching settles them), 10..14 in context 1 (it does not),
        // all equally uncertain; key 20 settled (no uncertainty)
        for k in 0..4 {
            c.note(k, 0, ONE / 2);
            c.note(10 + k, 1, ONE / 2);
        }
        c.note(20, 2, 0);
        // before any search: by uncertainty alone, the settled question last
        assert_eq!(c.expected_gain(0, ONE / 2), c.expected_gain(1, ONE / 2));
        assert!(!c.pick(8).contains(&20));
        for _ in 0..5 {
            c.learn(0, ONE / 2, &mut rng);
            c.learn(1, 0, &mut rng);
        }
        assert!(c.expected_gain(0, ONE / 2) > c.expected_gain(1, ONE / 2));
        assert_eq!(c.pick(2), vec![0, 1]);
        // closed questions are no longer picked
        c.close(&0);
        assert_eq!(c.pick(1), vec![1]);
        assert_eq!(c.stats().1, 10);
    }

    #[test]
    fn the_learned_policy_asks_where_asking_has_paid() {
        let mut rng = StdRng::seed_from_u64(2);
        // contexts: belief band 0..2 × lead band 0..2; asking pays only at lead band 0
        let mut c: Curiosity<u32> = Curiosity::new(2, 2);
        for k in 0..10 {
            c.note(k, (k % 2) as usize, ONE / 2); // contexts 0 (lead 0) and 1 (lead 1)
        }
        for _ in 0..300 {
            for k in c.pick_learned(1, &mut rng) {
                let ctx = (k % 2) as usize;
                c.learn(ctx, if ctx == 0 { ONE } else { 0 }, &mut rng);
            }
        }
        assert!(c.learned_value(0) > c.learned_value(1), "{} vs {}", c.learned_value(0), c.learned_value(1));
        // greedy enough: most of the last picks in the paying context
        let mut paying = 0;
        for _ in 0..50 {
            paying += c.pick_learned(1, &mut rng).iter().filter(|&&k| k % 2 == 0).count();
        }
        assert!(paying >= 40, "{paying} of 50");
    }

    #[test]
    fn the_network_learns_when_a_question_is_worth_its_cost() {
        let mut rng = StdRng::seed_from_u64(3);
        // context 0: a search gains 0.8; context 1: it gains nothing. Each costs 0.25.
        let mut c: Curiosity<u32> = Curiosity::new(2, 2);
        let cost = ONE / 4;
        for _ in 0..400 {
            for k in 0..6 {
                c.note(k, (k % 2) as usize, ONE / 2);
            }
            let energy = ONE;
            if let Some((_, ctx)) = c.decide(energy, 0, &mut rng) {
                c.reward_ask(ctx, energy, 0, if ctx == 0 { ONE * 4 / 5 } else { 0 }, cost, &mut rng);
            } else {
                c.reward_stop(energy, 0, &mut rng);
            }
        }
        let (ask0, stop) = c.values_at(0, ENERGY_BANDS - 1, 0);
        let (ask1, _) = c.values_at(1, ENERGY_BANDS - 1, 0);
        assert!(ask0 > stop, "worth asking: {ask0} vs stop {stop}");
        assert!(ask1 < stop, "not worth its cost: {ask1} vs stop {stop}");
    }
}
