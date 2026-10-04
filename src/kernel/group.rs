use crate::kernel::class::KernelClass;
use crate::kernel::simple::SimpleKernel;
use crate::bitvec::BitVector;

/// A collection of different KernelClasses that can operate together on an edge.
pub struct KernelGroup {
    available_classes: Vec<KernelClass<SimpleKernel>>,
}

impl KernelGroup {
    pub fn default() -> Self {
        let mut available_classes = Vec::new();
        available_classes.push(KernelClass::with_kernels(vec![
            SimpleKernel::default(0),
        ]));

        Self { available_classes }
    }

    /// A group with no classes (no built-in default kernel).
    pub fn new() -> Self {
        Self { available_classes: Vec::new() }
    }

    pub fn add_class(&mut self, kc: KernelClass<SimpleKernel>) {
        self.available_classes.push(kc);
    }

    pub fn len(&self) -> usize {
        self.available_classes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.available_classes.is_empty()
    }

    pub fn classes(&self) -> &[KernelClass<SimpleKernel>] {
        &self.available_classes
    }

    /// Returns the number of kernels that fired across all classes.
    pub fn process_all(&mut self, input: &BitVector, output: &mut BitVector, phase: u16, adj_temperature: i16) -> usize {
        let mut fired = 0;
        for kclass in &mut self.available_classes {
            fired += kclass.process_all(input, output, phase, adj_temperature);
        }
        fired
    }

    /// Predictive learning for every class; see `KernelClass::feedback`.
    pub fn feedback<R: rand::Rng + ?Sized>(&mut self, input: &BitVector, target: &BitVector, rng: &mut R) {
        for kclass in &mut self.available_classes {
            kclass.feedback(input, target, rng);
        }
    }
}

impl Default for KernelGroup {
    fn default() -> Self {
        Self::default()
    }
}
