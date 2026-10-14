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

    pub fn add_class(&mut self, kc: KernelClass<SimpleKernel>) {
        self.available_classes.push(kc);
    }

    pub fn len(&self) -> usize {
        self.available_classes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.available_classes.is_empty()
    }

    /// A group of the given classes.
    pub fn from_classes(available_classes: Vec<KernelClass<SimpleKernel>>) -> Self {
        Self { available_classes }
    }

    /// Run every class; returns, per class, the kernels that fired.
    pub fn process_all<R: rand::Rng + ?Sized>(&mut self, input: &BitVector, output: &mut BitVector, phase: u16, adj_temperature: i16, rng: &mut R) -> Vec<Vec<usize>> {
        self.process(input, output, phase, adj_temperature, true, rng)
    }

    /// As `process_all`; with `inhibit` off, kernels do not hold each other down.
    pub fn process<R: rand::Rng + ?Sized>(&mut self, input: &BitVector, output: &mut BitVector, phase: u16, adj_temperature: i16, inhibit: bool, rng: &mut R) -> Vec<Vec<usize>> {
        let mut fired = Vec::new();
        for kclass in &mut self.available_classes {
            fired.push(kclass.process(input, output, phase, adj_temperature, inhibit, rng));
        }
        fired
    }

    /// Whether any kernel can do anything on an input with no lines on (see
    /// `SimpleKernel::acts_on_nothing`).
    pub fn acts_on_nothing(&self) -> bool {
        self.available_classes.iter().any(|c| c.kernels().iter().any(|k| k.acts_on_nothing()))
    }

    pub fn kernel_count(&self) -> usize {
        self.available_classes.iter().map(|c| c.len()).sum()
    }

    pub fn classes_mut(&mut self) -> &mut [KernelClass<SimpleKernel>] {
        &mut self.available_classes
    }
}

impl Default for KernelGroup {
    fn default() -> Self {
        Self::default()
    }
}
