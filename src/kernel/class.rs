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
}

pub struct KernelClass<K: KernelTrait> {
    active_kernels: Vec<K>,
    temperature: KernelClassTemperature,
}

impl<K: KernelTrait> KernelClass<K> {
    pub fn default() -> Self {
        Self { active_kernels: Vec::new(), temperature: KernelClassTemperature::default() }
    }

    pub fn with_kernels(kernels: Vec<K>) -> Self {
        Self { active_kernels: kernels, temperature: KernelClassTemperature::default() }
    }

    pub fn add(&mut self, k: K) {
        self.active_kernels.push(k);
    }

    /// Run all kernels; kernels may OR/XOR/AND/CLEAR into `output`.
    pub fn process_all(&mut self, input: &BitVector, output: &mut BitVector, phase: u16) {
        let mut inhibit = BitVector::new(input.word_len(),Some(0));
        let mut rng = rand::thread_rng();
        for k in &mut self.active_kernels {
            let ctx = KernelContext {
                input,
                inhibit: &inhibit,
                temperature: self.temperature.current,
                phase,
            };

            if k.try_fire(&ctx, output, &mut rng) {
                // Kernel fired inhibit overlapping kernels
                let (start,end) = k.word_range();
                for i in start..end {
                    inhibit.bit_set(i);
                }

            }
        }
    }

    pub fn len(&self) -> usize { self.active_kernels.len() }
    pub fn is_empty(&self) -> bool { self.active_kernels.is_empty() }
}