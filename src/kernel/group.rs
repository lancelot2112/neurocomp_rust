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

    pub fn process_all<R: rand::Rng + ?Sized>(&mut self, input: &BitVector, output: &mut BitVector, phase: u16, adj_temperature: i16, rng: &mut R) {
        for kclass in &mut self.available_classes {
            kclass.process_all(input, output, phase, adj_temperature, rng);
        }
    }
}

impl Default for KernelGroup {
    fn default() -> Self {
        Self::default()
    }
}
