//! A bitwise thalamic gate (`ThalamicGate`): each source's reliability learned from the
//! activity pattern, not from a table keyed by word ids.
//!
//! Each source reaching the thalamus (the column, memory, a higher area, the cerebellum, …)
//! has, per confidence band of its own, one gate cell with a binary synapse on every context
//! bit (the cortex's layer 6 context: here the current and the previous input). The share
//! of the active context bits whose synapse is potent is the source's reliability in this
//! context, so similar contexts share what was learned.
//!
//! **Learning** (stochastic binary synapses, Fusi): when the source was right, each depressed
//! synapse on an active context bit is restored with probability 1/4; when it was wrong, each
//! potent one is depressed with probability 1/4. Over many outcomes the potent share in a
//! context tracks the rate at which the source is right there. A new gate cell starts with
//! half its synapses potent: a reliability of one half.

use crate::bitvec::BitVector;
use crate::fixed::{Q16, ONE};

pub struct ThalamicGate {
    masks: crate::det::HashMap<(u8, u8), Vec<u64>>,
}

impl Default for ThalamicGate {
    fn default() -> Self {
        Self::new()
    }
}

impl ThalamicGate {
    pub fn new() -> Self {
        Self { masks: crate::det::HashMap::default() }
    }

    /// The reliability (`Q16`) of `source` in confidence band `band`, in context `ctx`.
    pub fn rate(&self, source: u8, band: u8, ctx: &BitVector) -> Q16 {
        let n = ctx.count_ones() as u64;
        if n == 0 {
            return ONE / 2;
        }
        let potent: u64 = match self.masks.get(&(source, band)) {
            Some(m) => ctx.as_words().iter().zip(m).map(|(c, m)| (c & m).count_ones() as u64).sum(),
            None => ctx.as_words().iter().map(|c| (c & 0x5555_5555_5555_5555).count_ones() as u64).sum(),
        };
        ((potent << 16) / n) as Q16
    }

    /// Learn whether `source` (in band `band`) was right in context `ctx`.
    pub fn record<R: rand::Rng + ?Sized>(&mut self, source: u8, band: u8, ctx: &BitVector, right: bool, rng: &mut R) {
        let words = ctx.as_words().len();
        let m = self.masks.entry((source, band)).or_insert_with(|| vec![0x5555_5555_5555_5555; words]);
        if m.len() < words {
            m.resize(words, 0x5555_5555_5555_5555);
        }
        for (w, &c) in m.iter_mut().zip(ctx.as_words()) {
            if c == 0 {
                continue;
            }
            let r = rng.next_u64() & rng.next_u64() & c;
            if right {
                *w |= r;
            } else {
                *w &= !r;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.masks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.masks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn learns_reliability_per_context() {
        let mut g = ThalamicGate::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let a = BitVector::from_bits(&(0..32).collect::<Vec<_>>(), 256);
        let b = BitVector::from_bits(&(128..160).collect::<Vec<_>>(), 256);
        // source 0: right in context a, wrong in context b
        for _ in 0..200 {
            g.record(0, 0, &a, true, &mut rng);
            g.record(0, 0, &b, false, &mut rng);
        }
        assert!(g.rate(0, 0, &a) > ONE * 9 / 10);
        assert!(g.rate(0, 0, &b) < ONE / 10);
        // right 3 times in 4: the rate tracks it
        let c = BitVector::from_bits(&(64..96).collect::<Vec<_>>(), 256);
        for i in 0..2000 {
            g.record(1, 0, &c, i % 4 != 0, &mut rng);
        }
        let r = g.rate(1, 0, &c);
        assert!(r > ONE / 2 && r < ONE * 19 / 20, "{}", r as f64 / ONE as f64);
    }
}
