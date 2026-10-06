//! Basal-ganglia-style action selection learned from reward, in bits.
//!
//! Candidates (e.g. recalled items a memory hop could follow, or thalamic routes)
//! arrive as sparse bit patterns. The striatum holds a small "go" counter per input
//! bit, stored as bit-sliced counters (`SlicedCounter`); a candidate's value is its
//! counters' sum over its active bits, popcount(candidate ∧ plane_i) · 2^i, divided by
//! its size, so similar candidates share value. Selection releases the single best
//! candidate (disinhibition of one thalamic channel), compared by integer
//! cross-multiplication, with occasional random exploration during learning.
//!
//! Learning is a three-factor rule (Frémaux & Gerstner 2016): the chosen candidate's
//! bits are its eligibility trace, a bit mask kept for `trace_len` steps. When a reward
//! arrives, its sign relative to the chosen value (the reward-prediction error, the
//! "dopamine" signal) says whether to increment or decrement the counters under the
//! trace, and its size sets the probability that each eligible bit actually steps
//! (`gain` × |error|; older masks scaled by `trace_decay` per step). The learning rate
//! is a flip probability, as in stochastic binary synapses (Amit & Fusi 1994).
//!
//! All of it is integer: values, rewards and errors are fixed-point `Q16` (`fixed::ONE` =
//! 1; rewards and errors signed), and every probability is an integer draw (`chance`).

use std::collections::VecDeque;

use rand::Rng;

use crate::bitvec::{BitVector, SlicedCounter};
use crate::fixed::{chance, ratio, Q16, ONE};

/// A signed `Q16` value (rewards and reward-prediction errors).
pub type SQ16 = i32;

pub struct BasalGanglia {
    go: SlicedCounter,
    bits: usize,
    /// Eligibility traces: the chosen candidates' bit masks, newest first.
    trace: VecDeque<BitVector>,
    pub trace_len: usize,
    pub trace_decay: Q16,
    /// Step probability per unit of reward-prediction error.
    pub gain: Q16,
    /// Probability of choosing a random candidate when exploring.
    pub explore: Q16,
    last_value: Option<(u64, u64)>, // (counter sum, max possible sum) of the latest choice
    /// Reward baseline: with `Some(rate)`, the reward-prediction error is reward minus a
    /// running average of rewards (updated at `rate`), not minus the latest choice's value.
    /// Needed when one reward credits a long trace of choices: subtracting the last
    /// choice's value makes almost every error negative when rewards are rare, which
    /// drags down whichever action is chosen most.
    pub baseline_rate: Option<Q16>,
    baseline: i64,
}

impl BasalGanglia {
    /// `planes`-bit counters (4 → 16 levels), all starting halfway.
    pub fn new(bits: usize) -> Self {
        let planes = 4;
        Self {
            go: SlicedCounter::new(bits, planes, 1 << (planes - 1)),
            bits,
            trace: VecDeque::new(),
            trace_len: 1,
            trace_decay: ONE / 2,
            gain: ONE * 3 / 2,
            explore: 6554, // 0.1
            last_value: None,
            baseline_rate: None,
            baseline: 0,
        }
    }

    /// (counter sum over the candidate's bits, the maximum that sum could be).
    fn score(&self, candidate: &BitVector) -> (u64, u64) {
        let n = candidate.count_ones() as u64;
        if n == 0 {
            return (1, 2); // unknown: halfway
        }
        (self.go.sum(candidate), n * self.go.max() as u64)
    }

    /// Learned value of a candidate in [0, 1] (for inspection).
    /// In `Q16`.
    pub fn value(&self, candidate: &BitVector) -> Q16 {
        let (s, m) = self.score(candidate);
        ratio(s, m)
    }

    /// Index of the candidate to release, or None if there are none. With `rng`, a
    /// random candidate is chosen with probability `self.explore`. The choice becomes
    /// the newest eligibility trace.
    pub fn select<R: Rng>(&mut self, candidates: &[BitVector], rng: Option<&mut R>) -> Option<usize> {
        if candidates.is_empty() {
            return None;
        }
        let scores: Vec<(u64, u64)> = candidates.iter().map(|c| self.score(c)).collect();
        let mut best = 0;
        for i in 1..scores.len() {
            // s_i / m_i > s_best / m_best, in integers
            if scores[i].0 * scores[best].1 > scores[best].0 * scores[i].1 {
                best = i;
            }
        }
        if let Some(rng) = rng {
            if chance(rng, self.explore) {
                best = rng.gen_range(0..candidates.len());
            }
        }
        self.trace.push_front(candidates[best].clone());
        self.trace.truncate(self.trace_len.max(1));
        self.last_value = Some(scores[best]);
        Some(best)
    }

    /// Reward one specific `candidate` (not the trace). Without a baseline its counters step
    /// toward `reward` by the error reward − its own value (a per-candidate, bandit-style
    /// estimate). With `baseline_rate`, the error is reward − the running average reward (an
    /// advantage): the candidate gains when it did better than usual, so a graded reward
    /// that starts low (e.g. the cortex's L5 outcome before it has learned to use the
    /// choice) still ranks candidates instead of dragging them all below the default.
    /// Returns the error.
    /// `reward` is a signed `Q16`.
    pub fn reward_candidate<R: Rng>(&mut self, candidate: &BitVector, reward: SQ16, rng: &mut R) -> SQ16 {
        let (s, m) = self.score(candidate);
        let delta = self.error(reward, s, m);
        let p = self.step_probability(delta);
        let mut step = BitVector::new(self.bits, Some(0));
        for (wi, &w) in candidate.as_words().iter().enumerate() {
            let mut w = w;
            while w != 0 {
                let b = w.trailing_zeros() as usize;
                w &= w - 1;
                if chance(rng, p) {
                    step.bit_set(wi * 64 + b);
                }
            }
        }
        if delta > 0 {
            self.go.increment(&step);
        } else {
            self.go.decrement(&step);
        }
        delta as SQ16
    }

    /// The reward-prediction error (signed `Q16`): reward − the running-average reward
    /// (with a baseline, which is updated), else reward − the chosen value s / m.
    fn error(&mut self, reward: SQ16, s: u64, m: u64) -> i64 {
        match self.baseline_rate {
            Some(rate) => {
                let d = reward as i64 - self.baseline;
                self.baseline += (rate as i64 * d) >> 16;
                d
            }
            None => reward as i64 - ratio(s, m) as i64,
        }
    }

    /// The per-bit step probability `gain` × |error|, capped at 1 (`Q16`).
    fn step_probability(&self, delta: i64) -> Q16 {
        ((self.gain as u64 * delta.unsigned_abs()) >> 16).min(ONE as u64) as Q16
    }

    /// Reward (0..=1) for the latest choice and, through the trace, earlier ones.
    /// Returns the reward-prediction error.
    /// `reward` is a signed `Q16`.
    pub fn reward<R: Rng>(&mut self, reward: SQ16, rng: &mut R) -> SQ16 {
        let Some((s, m)) = self.last_value else { return 0 };
        let delta = self.error(reward, s, m);
        let mut p = self.step_probability(delta);
        for mask in &self.trace {
            // stochastic step: each eligible bit moves with probability p
            let mut step = BitVector::new(self.bits, Some(0));
            for (wi, &w) in mask.as_words().iter().enumerate() {
                let mut w = w;
                while w != 0 {
                    let b = w.trailing_zeros() as usize;
                    w &= w - 1;
                    if chance(rng, p) {
                        step.bit_set(wi * 64 + b);
                    }
                }
            }
            if delta > 0 {
                self.go.increment(&step);
            } else {
                self.go.decrement(&step);
            }
            p = ((p as u64 * self.trace_decay as u64) >> 16) as Q16;
        }
        delta as SQ16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn item(i: usize) -> BitVector {
        BitVector::from_bits(&(i * 8..i * 8 + 8).collect::<Vec<_>>(), 256)
    }

    #[test]
    fn learns_to_release_the_rewarded_kind_of_candidate() {
        // items 0-4 are "names" (following them pays), 10-14 are "verbs" (never pays);
        // each trial offers one of each in random order
        let mut bg = BasalGanglia::new(256);
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..500 {
            let (n, v) = (item(rng.gen_range(0..5)), item(rng.gen_range(10..15)));
            let swap = rng.gen_bool(0.5);
            let cands = if swap { vec![v, n] } else { vec![n, v] };
            let chosen = bg.select(&cands, Some(&mut rng)).unwrap();
            let picked_name = (chosen == 0) != swap;
            bg.reward(if picked_name { ONE as i32 } else { 0 }, &mut rng);
        }
        let mut right = 0;
        for i in 0..5 {
            let chosen = bg.select::<StdRng>(&[item(10 + i), item(i)], None).unwrap();
            right += (chosen == 1) as usize;
        }
        assert_eq!(right, 5);
    }

    #[test]
    fn eligibility_trace_credits_an_earlier_choice() {
        let mut bg = BasalGanglia::new(256);
        let mut rng = StdRng::seed_from_u64(2);
        bg.trace_len = 2;
        bg.explore = 0;
        for _ in 0..5 {
            bg.select::<StdRng>(&[item(1)], None);
            bg.select::<StdRng>(&[item(2)], None);
            bg.reward(ONE as i32, &mut rng);
        }
        // both raised, the more recent more
        assert!(bg.value(&item(2)) > bg.value(&item(1)));
        assert!(bg.value(&item(1)) > ONE / 2);
    }
}
