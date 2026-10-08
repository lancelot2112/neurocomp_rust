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

pub struct Curiosity<K> {
    /// Per context: (gain realised, summed, `Q16`; searches).
    gain: Vec<(u64, u64)>,
    /// Open questions: key → (context, uncertainty `Q16`).
    open: HashMap<K, (usize, u32)>,
    searches: u64,
    gained: u64,
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
            bg: BasalGanglia::new((belief_bands + lead_bands) * BAND_BITS),
            bands: (belief_bands, lead_bands),
        }
    }

    /// A context's code for the selector: its belief band's bits and its lead band's.
    fn code(&self, context: usize) -> BitVector {
        let (bb, lb) = self.bands;
        let (b, l) = ((context / lb).min(bb - 1), context % lb);
        let mut bits: Vec<usize> = (b * BAND_BITS..(b + 1) * BAND_BITS).collect();
        bits.extend((bb + l) * BAND_BITS..(bb + l + 1) * BAND_BITS);
        BitVector::from_bits(&bits, (bb + lb) * BAND_BITS)
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
}
