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

/// What drives a thalamic relay cell from a source.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Driver {
    /// a layer 5 burst (input and context together): passes fully
    Burst,
    /// a layer 5 single spike (input alone): a depressing driver synapse, weak
    Spike,
    /// another source (memory, a higher area, the cerebellum) in its own confidence band
    Tonic(u8),
}

/// The thalamic relay (`ThalamicRelay`): each source's weight in the vote is the product of
/// - **driver strength:** a burst passes fully; a single spike passes at one half, and
///   consecutive spikes depress the synapse further (1/2, 1/4, 1/8, …), as depressing driver
///   synapses do; a tonic source passes fully;
/// - **the learned context gate** (`ThalamicGate`, layer 6 context): the source's reliability
///   in this context, per driver type;
/// - **modulation by the cerebellum:** a proposal the cerebellum's deep nuclei also make is
///   strengthened (a quarter of the remaining way to certainty).
pub struct ThalamicRelay {
    pub gate: ThalamicGate,
    spike_run: u32,
}

impl Default for ThalamicRelay {
    fn default() -> Self {
        Self::new()
    }
}

impl ThalamicRelay {
    pub fn new() -> Self {
        Self { gate: ThalamicGate::new(), spike_run: 0 }
    }

    fn band(driver: Driver) -> u8 {
        match driver {
            Driver::Burst => 8,
            Driver::Spike => 9,
            Driver::Tonic(b) => b.min(7),
        }
    }

    /// One step of layer 5's output: did it spike (without bursting)? Consecutive spikes
    /// depress the driver synapse; a silent step or a burst lets it recover.
    pub fn step(&mut self, spiked: bool) {
        self.spike_run = if spiked { self.spike_run + 1 } else { 0 };
    }

    /// The weight (`Q16`) of a proposal from `source` with `driver`, in context `ctx`.
    pub fn weight(&self, source: u8, driver: Driver, ctx: &BitVector, cerebellum_agrees: bool) -> Q16 {
        let rate = self.gate.rate(source, Self::band(driver), ctx);
        let drive: u64 = match driver {
            Driver::Burst | Driver::Tonic(_) => ONE as u64,
            Driver::Spike => (ONE as u64 / 2) >> self.spike_run.saturating_sub(1).min(3),
        };
        let mut w = ((rate as u64 * drive) >> 16) as Q16;
        if cerebellum_agrees {
            w += (ONE - w.min(ONE)) / 4;
        }
        w
    }

    /// Learn whether `source` (with `driver`) was right in context `ctx`.
    pub fn record<R: rand::Rng + ?Sized>(&mut self, source: u8, driver: Driver, ctx: &BitVector, right: bool, rng: &mut R) {
        self.gate.record(source, Self::band(driver), ctx, right, rng);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn spikes_are_weak_and_depress() {
        let r = ThalamicRelay::new();
        let ctx = BitVector::from_bits(&(0..32).collect::<Vec<_>>(), 256);
        let burst = r.weight(0, Driver::Burst, &ctx, false);
        let spike = r.weight(0, Driver::Spike, &ctx, false);
        assert!(spike < burst);
        let mut r2 = ThalamicRelay::new();
        r2.step(true);
        r2.step(true);
        r2.step(true);
        assert!(r2.weight(0, Driver::Spike, &ctx, false) < spike);
        assert!(r.weight(0, Driver::Spike, &ctx, true) > spike);
    }

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
