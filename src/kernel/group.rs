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

    pub fn classes(&self) -> &[KernelClass<SimpleKernel>] {
        &self.available_classes
    }

    /// The overlap rule on every class (`KernelContext::inhibit_quarters`).
    pub fn set_inhibit_quarters(&mut self, q: u8) {
        for c in self.classes_mut() {
            c.set_inhibit_quarters(q);
        }
    }

    /// This step's inhibit input on every class (`KernelClass::set_veto`).
    pub fn set_veto(&mut self, words: &[u64]) {
        for c in self.classes_mut() {
            c.set_veto(words);
        }
    }

    /// This step's heat on every class (`KernelContext::heat`).
    pub fn set_heat(&mut self, heat: i16) {
        for c in self.classes_mut() {
            c.set_heat(heat);
        }
    }

    /// This step's taken destination lines on every class (`KernelContext::taken_out`).
    pub fn set_taken(&mut self, words: &[u64]) {
        for c in self.classes_mut() {
            c.set_taken(words);
        }
    }

    /// Kernels held down this step by other populations' outputs (cleared on reading).
    pub fn take_lost(&mut self) -> u32 {
        self.classes_mut().iter_mut().map(|c| c.take_lost()).sum()
    }

    /// The best share (in 1/4096 of its own lines) among the kernels that fired this step.
    pub fn best_share(&self) -> usize {
        self.available_classes.iter().map(|c| c.best_share()).max().unwrap_or(0)
    }

    /// Mark the destination lines that the kernels that fired (per class) write.
    pub fn mark_outputs(&self, fired: &[Vec<usize>], taken: &mut [u64]) {
        for (c, f) in self.available_classes.iter().zip(fired) {
            c.mark_outputs(f, taken);
        }
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
