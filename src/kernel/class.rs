use crate::bitvec::BitVector;
use crate::common::config;
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

pub trait KernelTrait {
    fn process(&self, input: &BitVector, output: &mut BitVector, temperature: &i16, phase: &u16);
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
    pub fn process_all(&self, input: &BitVector, output: &mut BitVector, phase: u16) {
        for k in &self.active_kernels {
            k.process(input, output, &self.temperature.current, &phase);
        }
    }

    pub fn len(&self) -> usize { self.active_kernels.len() }
    pub fn is_empty(&self) -> bool { self.active_kernels.is_empty() }
}