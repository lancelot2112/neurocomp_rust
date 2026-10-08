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
//!   gained: the rise in the believed value's lead once the new evidence is weighed. The
//!   expected gain starts optimistic (a context never searched is assumed worth one full
//!   unit of lead, as one search's worth of prior), so every kind of question gets tried,
//!   and contexts where searching settles nothing (a decided question, an undecidable one
//!   whose search brings no evidence) lose their pull.
//! - **What it does.** `pick` returns the questions with the highest expected gain, within
//!   a budget. The gain realised is the intrinsic reward (the dopamine signal for the value
//!   of information): `learn` takes it.
//!
//! Brain: the anterior cingulate and lateral habenula track uncertainty and when information
//! will arrive; midbrain dopamine neurons signal the value of information itself
//! (Bromberg-Martin & Hikosaka 2009); frontopolar cortex chooses to explore an uncertain
//! option (directed exploration). Integers only (`Q16`).

use std::hash::Hash;

use crate::det::HashMap;
use crate::fixed::ONE;

pub struct Curiosity<K> {
    /// Per context: (gain realised, summed, `Q16`; searches).
    gain: Vec<(u64, u64)>,
    /// Open questions: key → context.
    open: HashMap<K, usize>,
    searches: u64,
    gained: u64,
}

impl<K: Hash + Eq + Clone + Ord> Curiosity<K> {
    /// `contexts` distinct question states.
    pub fn new(contexts: usize) -> Self {
        Self { gain: vec![(0, 0); contexts], open: HashMap::default(), searches: 0, gained: 0 }
    }

    /// Note an open question (or update its state); a settled one is closed with `close`.
    pub fn note(&mut self, key: K, context: usize) {
        self.open.insert(key, context.min(self.gain.len() - 1));
    }

    pub fn close(&mut self, key: &K) {
        self.open.remove(key);
    }

    /// The expected gain of a search in `context` (`Q16`): the gains realised there, with
    /// one optimistic search (a full unit) as the prior.
    pub fn expected_gain(&self, context: usize) -> u32 {
        let (sum, n) = self.gain[context.min(self.gain.len() - 1)];
        ((sum + ONE as u64) / (n + 1)) as u32
    }

    /// The `budget` open questions most worth a search, best first (ties to the lower key).
    pub fn pick(&self, budget: usize) -> Vec<K> {
        let mut qs: Vec<(u32, &K)> = self.open.iter().map(|(k, &c)| (self.expected_gain(c), k)).collect();
        qs.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(b.1)));
        qs.into_iter().take(budget).map(|x| x.1.clone()).collect()
    }

    /// A search in `context` gained `gain` (`Q16`, the rise in the answer's lead).
    pub fn learn(&mut self, context: usize, gain: u32) {
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

    #[test]
    fn searches_go_where_they_have_paid_off() {
        let mut c: Curiosity<u32> = Curiosity::new(4);
        // keys 0..4 in context 0 (searching settles them), 10..14 in context 1 (it does not)
        for k in 0..4 {
            c.note(k, 0);
            c.note(10 + k, 1);
        }
        // untried contexts are equally attractive
        assert_eq!(c.expected_gain(0), c.expected_gain(1));
        for _ in 0..5 {
            c.learn(0, ONE / 2);
            c.learn(1, 0);
        }
        assert!(c.expected_gain(0) > c.expected_gain(1));
        assert_eq!(c.pick(2), vec![0, 1]);
        // closed questions are no longer picked
        c.close(&0);
        assert_eq!(c.pick(1), vec![1]);
        assert_eq!(c.stats().1, 10);
    }
}
