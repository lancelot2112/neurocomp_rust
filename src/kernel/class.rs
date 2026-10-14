use crate::bitvec::BitVector;
use crate::common::config;

/// How to combine the output mask into the output BitVector (word-aligned).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelOp {
    Or,
    And,
    Xor,
    Clear, // a & !b
}

pub struct KernelClassTemperature {
    pub max: i16,
    pub min: i16,
    pub current: i16,
}

impl KernelClassTemperature {
    pub fn default() -> Self {
        let kdef = &config().kernel;
        Self {
            min: kdef.temperature_min,
            max: kdef.temperature_max,
            current: kdef.temperature_initial,
        }
    }

    pub fn new(min: i16, max: i16, initial: i16) -> Self {
        assert!(min <= initial && initial <= max);
        Self { min, max, current: initial }
    }

    pub fn incr(&mut self) {
        if self.current < self.max {
            self.current += 1;
        }
    }

    pub fn decr(&mut self) {
        if self.current > self.min {
            self.current -= 1;
        }
    }
}

// Light, borrow-based context passed to each spawned kernel
pub struct KernelContext<'a> {
    pub input: &'a BitVector,
    pub inhibit: &'a BitVector,
    pub temperature: i16,
    pub phase: u16,
}

pub trait KernelTrait {
    fn try_fire<R: rand::Rng + ?Sized>(&mut self, ctx: &KernelContext, out: &mut BitVector, rng: &mut R) -> bool;
    fn word_range(&self) -> (usize, usize);
    /// Mark the input lines this kernel takes when it fires (for overlap inhibition).
    fn mark_inhibit(&self, inhibit: &mut BitVector);
    /// How many of its lines agree with the input (the better match goes first).
    fn strength(&self, input: &BitVector) -> usize;
}

pub struct KernelClass<K: KernelTrait> {
    active_kernels: Vec<K>,
    temperature: KernelClassTemperature,
}

impl<K: KernelTrait> KernelClass<K> {
    pub fn default() -> Self {
        Self { active_kernels: Vec::new(), temperature: KernelClassTemperature::new(0, 0, 0) }
    }

    pub fn with_kernels(kernels: Vec<K>) -> Self {
        Self { active_kernels: kernels, temperature: KernelClassTemperature::new(0, 0, 0) }
    }

    pub fn add(&mut self, k: K) {
        self.active_kernels.push(k);
    }

    /// Run all kernels; kernels may OR/XOR/AND/CLEAR into `output`. A kernel that fires marks
    /// its input mask as taken: a later kernel whose mask overlaps the taken lines by more than
    /// half is held down (and, if it matched, learns anti-Hebbian).
    pub fn process_all<R: rand::Rng + ?Sized>(&mut self, input: &BitVector, output: &mut BitVector, phase: u16, adj_temperature: i16, rng: &mut R) -> Vec<usize> {
        self.process(input, output, phase, adj_temperature, true, rng)
    }

    /// As `process_all`; with `inhibit` off, kernels do not hold each other down.
    pub fn process<R: rand::Rng + ?Sized>(&mut self, input: &BitVector, output: &mut BitVector, phase: u16, adj_temperature: i16, inhibit_on: bool, rng: &mut R) -> Vec<usize> {
        let mut inhibit = BitVector::new(input.bit_len(), Some(0));
        let mut fired = Vec::new();
        // the better-matched kernels go first, so where masks overlap the best match fires and
        // the others are held down (ties: the older kernel)
        let mut order: Vec<(usize, usize)> = self.active_kernels.iter().enumerate().map(|(i, k)| (k.strength(input), i)).collect();
        order.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        for (_, i) in order {
            let k = &mut self.active_kernels[i];
            let ctx = KernelContext {
                input,
                inhibit: &inhibit,
                temperature: self.temperature.current + adj_temperature,
                phase,
            };
            if k.try_fire(&ctx, output, rng) {
                if inhibit_on {
                    k.mark_inhibit(&mut inhibit);
                }
                fired.push(i);
            }
        }
        fired
    }

    pub fn kernels(&self) -> &[K] { &self.active_kernels }
    pub fn kernels_mut(&mut self) -> &mut [K] { &mut self.active_kernels }

    pub fn len(&self) -> usize { self.active_kernels.len() }
    pub fn is_empty(&self) -> bool { self.active_kernels.is_empty() }
}