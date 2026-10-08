//! The cerebellum: the fast, error-driven learner.
//!
//! The brain combines learners of different speeds and teachers (Doya 2000; McClelland,
//! McNaughton & O'Reilly 1995): the hippocampus stores single events fast; the cortex
//! learns slowly and interleaved, so it generalises; the cerebellum learns from errors,
//! each output with its own teacher; the basal ganglia learn from reward.
//!
//! **Circuit.** Mossy fibres (here: a copy of the cortex's input, as cortex → pontine
//! nuclei → mossy fibres) fan out into a large granule-cell expansion; Purkinje cells learn
//! from a climbing-fibre error per output (Marr 1969; Albus 1971; Ito). Its output leaves
//! through the deep nuclei to the thalamus and back to the cortex.
//!
//! **Here.** A predictive `KernelClass`: a sparse expansion of conjunctions of the input
//! (the granule layer), each kernel predicting output bits (Purkinje cells), corrected by
//! the actual next input (the climbing fibre) and grown in one shot on a miss. This is the
//! rule the cortical column's L2/3 used until the three learning systems were split
//! (`LEARNING=three` in the episodic example): fast and precise, but it memorises.

use crate::bitvec::BitVector;
use crate::fixed::Q16;
use crate::kernel::{KernelClass, SimpleKernel};

pub struct Cerebellum {
    /// Granule expansion and Purkinje cells: the fast predictive kernels.
    pub kernels: KernelClass<SimpleKernel>,
    prediction: BitVector,
    confidence: Q16,
}

impl Cerebellum {
    pub fn new(bits: usize, kernels: KernelClass<SimpleKernel>) -> Self {
        Self { kernels, prediction: BitVector::new(bits, Some(0)), confidence: 0 }
    }

    /// Predict the next input from the mossy-fibre input (a copy of the cortex's input).
    pub fn predict(&mut self, input: &BitVector) -> &BitVector {
        let mut out = BitVector::new(self.prediction.bit_len(), Some(0));
        self.kernels.process_predictive(input, &mut out);
        self.prediction = out;
        self.confidence = self.kernels.confidence().unwrap_or(0);
        &self.prediction
    }

    /// Learn from the climbing fibre: what actually came next. Fast: one-shot growth on a
    /// miss (unless the kernel class was told otherwise).
    pub fn learn<R: rand::Rng + ?Sized>(&mut self, input: &BitVector, target: &BitVector, rng: &mut R) {
        self.kernels.feedback(input, target, rng);
    }

    /// The latest prediction (the deep nuclei's output).
    pub fn prediction(&self) -> &BitVector {
        &self.prediction
    }

    /// The winning kernel's reliability for the latest prediction (`Q16`).
    pub fn confidence(&self) -> Q16 {
        self.confidence
    }

    /// Live kernels.
    pub fn live(&self) -> usize {
        self.kernels.live()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::GrowthConfig;
    use rand::SeedableRng;

    #[test]
    fn learns_a_sequence_in_one_exposure() {
        let bits = 256;
        let cfg = GrowthConfig { max_kernels: 1000, frame_words: bits / 64, max_frames: 1, sample_bits: 8, match_fraction: 0.8, surprise_fraction: 0.5, generalize: None, generalize_after: 1 };
        let mut cb = Cerebellum::new(bits, KernelClass::predictive(cfg));
        let a = BitVector::from_bits(&[1, 5, 9, 13, 17, 21, 25, 29], bits);
        let b = BitVector::from_bits(&[100, 104, 108, 112, 116, 120, 124, 128], bits);
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        cb.predict(&a);
        cb.learn(&a, &b, &mut rng);
        let p = cb.predict(&a).clone();
        assert!(p.as_words().iter().zip(b.as_words()).map(|(x, y)| (x & y).count_ones()).sum::<u32>() >= 6, "one exposure is enough for the fast learner");
    }
}
