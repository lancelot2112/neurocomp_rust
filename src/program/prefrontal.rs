//! Prefrontal working memory with basal-ganglia gating (after PBWM: O'Reilly & Frank 2006).
//!
//! `WorkingMemory` holds a few items (bit patterns) in slots, the prefrontal "stripes", and
//! maintains them across words until they are overwritten. `PfcGate` decides, at each
//! input, whether to **load** it into a slot or **keep** what is held: a basal-ganglia
//! go / no-go choice. Its candidates are the two action codes bound to the current input
//! (by bit rotation), so the striatum learns a value per (action, input). Reward arrives
//! later, when what working memory held turns out to be useful (e.g. it cued a recall
//! that contained the answer); the eligibility trace carries that credit back to the
//! gating decisions made several inputs earlier.

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::bitvec::BitVector;
use crate::program::BasalGanglia;

pub struct WorkingMemory {
    bits: usize,
    slots: Vec<BitVector>,
}

impl WorkingMemory {
    pub fn new(bits: usize, slots: usize) -> Self {
        Self { bits, slots: vec![BitVector::new(bits, Some(0)); slots.max(1)] }
    }

    /// Replace slot `i` with `pattern`.
    pub fn load(&mut self, i: usize, pattern: &BitVector) {
        if let Some(s) = self.slots.get_mut(i) {
            *s = pattern.clone();
        }
    }

    pub fn slot(&self, i: usize) -> Option<&BitVector> {
        self.slots.get(i)
    }

    /// Everything held, superimposed (OR of the slots).
    pub fn content(&self) -> BitVector {
        let mut out = BitVector::new(self.bits, Some(0));
        for s in &self.slots {
            out.or_mut(s);
        }
        out
    }

    pub fn clear(&mut self) {
        for s in &mut self.slots {
            *s = BitVector::new(self.bits, Some(0));
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    Load,
    Keep,
}

/// The basal-ganglia gate on working memory: load or keep, valued per input.
pub struct PfcGate {
    pub bg: BasalGanglia,
    bits: usize,
    load_code: BitVector,
    keep_code: BitVector,
}

impl PfcGate {
    /// `trace_len` inputs of eligibility (how far back a reward can reach), each older one
    /// credited with probability × `trace_decay`.
    pub fn new(bits: usize, trace_len: usize, trace_decay: f64, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let all: Vec<usize> = (0..bits).collect();
        let mut code = || BitVector::from_bits(&all.choose_multiple(&mut rng, 32).copied().collect::<Vec<_>>(), bits);
        let (load_code, keep_code) = (code(), code());
        let mut bg = BasalGanglia::new(bits);
        bg.trace_len = trace_len;
        bg.trace_decay = trace_decay;
        Self { bg, bits, load_code, keep_code }
    }

    fn bound(&self, code: &BitVector, key: usize) -> BitVector {
        let mut c = code.clone();
        c.rotl_mut((key * 131) % self.bits);
        c
    }

    /// Decide for the input identified by `key` (e.g. its word index). With `rng`, explore.
    pub fn decide<R: Rng>(&mut self, key: usize, rng: Option<&mut R>) -> Gate {
        let cands = [self.bound(&self.load_code, key), self.bound(&self.keep_code, key)];
        match self.bg.select(&cands, rng) {
            Some(0) => Gate::Load,
            _ => Gate::Keep,
        }
    }

    /// Learned value of an action for an input (for inspection).
    pub fn value(&self, gate: Gate, key: usize) -> f32 {
        let code = if gate == Gate::Load { &self.load_code } else { &self.keep_code };
        self.bg.value(&self.bound(code, key))
    }

    /// Reward the recent gating decisions (through the eligibility trace).
    pub fn reward<R: Rng>(&mut self, reward: f32, rng: &mut R) -> f32 {
        self.bg.reward(reward, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;

    #[test]
    fn working_memory_holds_until_overwritten() {
        let mut wm = WorkingMemory::new(128, 1);
        let a = BitVector::from_bits(&[1, 2, 3], 128);
        let b = BitVector::from_bits(&[7, 8], 128);
        wm.load(0, &a);
        assert_eq!(wm.content().count_ones(), 3);
        wm.load(0, &b);
        assert_eq!(wm.content().as_words(), b.as_words());
    }

    #[test]
    fn gate_learns_to_load_the_item_that_pays_off_later() {
        // episodes: [name, distractor, distractor, question]; reward at the question if
        // the slot still holds the name. Key 1 = name, 2..=3 = distractors.
        let mut gate = PfcGate::new(1024, 4, 0.9, 3);
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..600 {
            let mut held = 0usize;
            for key in [1usize, 2, 3] {
                if gate.decide(key, Some(&mut rng)) == Gate::Load {
                    held = key;
                }
            }
            gate.reward(if held == 1 { 1.0 } else { 0.0 }, &mut rng);
        }
        assert!(gate.value(Gate::Load, 1) > gate.value(Gate::Keep, 1), "should load the name");
        assert!(gate.value(Gate::Keep, 2) > gate.value(Gate::Load, 2), "should keep over a distractor");
        assert!(gate.value(Gate::Keep, 3) > gate.value(Gate::Load, 3), "should keep over a distractor");
    }
}
