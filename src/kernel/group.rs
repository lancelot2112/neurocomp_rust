use crate::kernel::class::KernelClass;

/// A collection of different KernelClasses that can operate together on an edge.
pub struct KernelGroup {
    available_classes: Vec<KernelClass>,
}

impl KernelGroup {
    pub fn default() -> Self {
        Self { available_classes: Vec::new() }
    }

    pub fn add_class(&mut self, kc: KernelClass) {
        self.available_classes.push(kc);
    }

    pub fn len(&self) -> usize {
        self.available_classes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.available_classes.is_empty()
    }
}

impl Default for KernelGroup {
    fn default() -> Self {
        Self::default()
    }
}
