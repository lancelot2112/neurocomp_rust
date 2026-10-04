//! Basal-ganglia-style action selection learned from reward.
//!
//! Candidates (e.g. recalled items a memory hop could follow, or thalamic routes)
//! arrive as sparse bit patterns. The striatum holds a "go" weight per input bit; a
//! candidate's value is the mean weight over its active bits, so similar candidates
//! share value. Selection releases the single best candidate (disinhibition of one
//! thalamic channel), with occasional random exploration during learning.
//!
//! Learning is a three-factor rule (Frémaux & Gerstner 2016): chosen candidates leave
//! an eligibility trace on their bits that decays by `trace_decay` per step; when a
//! reward arrives, the reward-prediction error (reward minus the chosen candidate's
//! value, the "dopamine" signal) times the trace changes the weights. With
//! `trace_decay > 0` a reward can credit choices made several steps earlier.

use rand::Rng;

use crate::bitvec::BitVector;

pub struct BasalGanglia {
    go: Vec<f32>,
    trace: Vec<f32>,
    traced: Vec<usize>, // bits with a nonzero trace
    pub learning_rate: f32,
    pub trace_decay: f32,
    /// Probability of choosing a random candidate when exploring.
    pub explore: f64,
    /// Initial (and unseen-bit) value; optimistic values make untried candidates attractive.
    pub initial: f32,
    last_value: Option<f32>,
}

impl BasalGanglia {
    pub fn new(bits: usize) -> Self {
        Self {
            go: vec![0.5; bits],
            trace: vec![0.0; bits],
            traced: Vec::new(),
            learning_rate: 0.1,
            trace_decay: 0.0,
            explore: 0.1,
            initial: 0.5,
            last_value: None,
        }
    }

    /// Learned value of a candidate: mean "go" weight over its active bits.
    pub fn value(&self, candidate: &BitVector) -> f32 {
        let (mut sum, mut n) = (0.0, 0usize);
        for b in set_bits(candidate) {
            if let Some(&w) = self.go.get(b) {
                sum += w;
                n += 1;
            }
        }
        if n == 0 {
            self.initial
        } else {
            sum / n as f32
        }
    }

    /// Index of the candidate to release, or None if there are none. With `explore`
    /// set, a random candidate is chosen with probability `self.explore`. The choice is
    /// marked eligible for the next `reward`.
    pub fn select<R: Rng>(&mut self, candidates: &[BitVector], rng: Option<&mut R>) -> Option<usize> {
        if candidates.is_empty() {
            return None;
        }
        let values: Vec<f32> = candidates.iter().map(|c| self.value(c)).collect();
        let mut best = 0;
        for i in 1..values.len() {
            if values[i] > values[best] {
                best = i;
            }
        }
        if let Some(rng) = rng {
            if rng.gen_bool(self.explore) {
                best = rng.gen_range(0..candidates.len());
            }
        }
        self.decay_trace();
        for b in set_bits(&candidates[best]) {
            if b < self.trace.len() {
                if self.trace[b] == 0.0 {
                    self.traced.push(b);
                }
                self.trace[b] = 1.0;
            }
        }
        self.last_value = Some(values[best]);
        Some(best)
    }

    fn decay_trace(&mut self) {
        let d = self.trace_decay;
        let trace = &mut self.trace;
        self.traced.retain(|&b| {
            trace[b] *= d;
            if trace[b] < 1e-3 {
                trace[b] = 0.0;
                false
            } else {
                true
            }
        });
    }

    /// Reward for the latest choice (and, through the trace, earlier ones):
    /// weights move by learning_rate × (reward − predicted value) × eligibility.
    /// Returns the reward-prediction error.
    pub fn reward(&mut self, reward: f32) -> f32 {
        let Some(v) = self.last_value else { return 0.0 };
        let delta = reward - v;
        for &b in &self.traced {
            self.go[b] = (self.go[b] + self.learning_rate * delta * self.trace[b]).clamp(0.0, 1.0);
        }
        delta
    }
}

fn set_bits(bv: &BitVector) -> Vec<usize> {
    let mut out = Vec::new();
    for (wi, &w) in bv.as_words().iter().enumerate() {
        let mut w = w;
        while w != 0 {
            out.push(wi * 64 + w.trailing_zeros() as usize);
            w &= w - 1;
        }
    }
    out
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
            bg.reward(if picked_name { 1.0 } else { 0.0 });
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
        bg.trace_decay = 0.5;
        bg.explore = 0.0;
        bg.select::<StdRng>(&[item(1)], None);
        bg.select::<StdRng>(&[item(2)], None);
        bg.reward(1.0);
        // both raised, the more recent more
        assert!(bg.value(&item(2)) > bg.value(&item(1)));
        assert!(bg.value(&item(1)) > 0.5);
    }
}
