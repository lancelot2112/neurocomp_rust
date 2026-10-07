//! Speech as an action: a motor area, the vocal tract it drives, and the forward model
//! whose prediction is the efference copy.
//!
//! - **The vocal tract is the world.** Each word has an articulatory (motor) code of its
//!   own, unrelated to the code it is heard as. Driving the tract with a motor code
//!   produces the word whose articulation it matches; nothing if it matches none.
//! - **The motor area learns two maps by babbling** (random motor commands, each heard):
//!   - the *inverse model*, heard code → motor code: how to say what is meant;
//!   - the *forward model*, motor code → heard code: what a command will sound like. Its
//!     prediction, sent to the sensory side while speaking, is the efference copy
//!     (corollary discharge): the heard word is compared with it.
//!   Both are predictive kernel classes, as the rest of the cortex.
//! - **Planning is cortical, selection is not.** The plan is the cortex's evidence for the
//!   word (the source mix's choice, a sensory code); the motor area turns it into a
//!   command. Whether to speak at all is a basal-ganglia go/no-go, decided elsewhere.

use rand::seq::SliceRandom;
use rand::Rng;

use crate::bitvec::BitVector;
use crate::kernel::{GrowthConfig, KernelClass, SimpleKernel};

/// The articulators: word ↔ motor code.
pub struct VocalTract {
    motor: Vec<BitVector>,
}

impl VocalTract {
    /// `n` words, each with a random sparse motor code (`active` of `bits` bits).
    pub fn new<R: Rng>(n: usize, bits: usize, active: usize, rng: &mut R) -> Self {
        let all: Vec<usize> = (0..bits).collect();
        Self { motor: (0..n).map(|_| BitVector::from_bits(&all.choose_multiple(rng, active).copied().collect::<Vec<_>>(), bits)).collect() }
    }

    /// The command that says word `w` (what babbling tries).
    pub fn command(&self, w: usize) -> &BitVector {
        &self.motor[w]
    }

    /// Drive the tract with a motor code: the word whose articulation it holds at least
    /// half of, best first; None if it matches none (a garbled command is silence).
    pub fn articulate(&self, m: &BitVector) -> Option<usize> {
        let ov = |c: &BitVector| c.as_words().iter().zip(m.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>();
        (0..self.motor.len()).map(|w| (ov(&self.motor[w]), w)).filter(|&(o, w)| o as usize * 2 >= self.motor[w].count_ones() as usize).max().map(|x| x.1)
    }

    pub fn len(&self) -> usize {
        self.motor.len()
    }

    pub fn is_empty(&self) -> bool {
        self.motor.is_empty()
    }
}

pub struct MotorArea {
    bits: usize,
    inverse: KernelClass<SimpleKernel>,
    forward: KernelClass<SimpleKernel>,
    babbles: usize,
}

fn class(bits: usize) -> KernelClass<SimpleKernel> {
    KernelClass::predictive(GrowthConfig {
        max_kernels: 20_000,
        frame_words: bits / 64,
        max_frames: 1,
        sample_bits: 16,
        match_fraction: 0.8,
        surprise_fraction: 0.5,
        generalize: None,
        generalize_after: 1,
    })
}

impl MotorArea {
    pub fn new(bits: usize) -> Self {
        Self { bits, inverse: class(bits), forward: class(bits), babbles: 0 }
    }

    /// Babble: `rounds` times through the tract's repertoire in random order, each command
    /// issued and its sound heard (`heard[w]`, the sensory code of the word it says); both
    /// models learn from each (command, sound) pair.
    pub fn babble<R: Rng>(&mut self, tract: &VocalTract, heard: &[BitVector], rounds: usize, rng: &mut R) {
        let mut order: Vec<usize> = (0..tract.len().min(heard.len())).collect();
        for _ in 0..rounds {
            order.shuffle(rng);
            for &w in &order {
                let m = tract.command(w).clone();
                let Some(said) = tract.articulate(&m) else { continue };
                let sound = &heard[said];
                let mut out = BitVector::new(self.bits, Some(0));
                self.inverse.process_predictive(sound, &mut out);
                self.inverse.feedback(sound, &m, rng);
                let mut out = BitVector::new(self.bits, Some(0));
                self.forward.process_predictive(&m, &mut out);
                self.forward.feedback(&m, sound, rng);
                self.babbles += 1;
            }
        }
    }

    /// How to say what the cortex means (a heard-word code): a motor command, or None.
    pub fn plan(&self, meant: &BitVector) -> Option<BitVector> {
        self.inverse.peek(meant)
    }

    /// What a command will sound like: the efference copy's prediction.
    pub fn predict(&self, command: &BitVector) -> Option<BitVector> {
        self.forward.peek(command)
    }

    /// (inverse kernels, forward kernels, babbled commands)
    pub fn stats(&self) -> (usize, usize, usize) {
        (self.inverse.len(), self.forward.len(), self.babbles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    const BITS: usize = 2048;

    #[test]
    fn babbling_learns_to_say_and_to_predict() {
        let mut rng = StdRng::seed_from_u64(1);
        let n = 60;
        let all: Vec<usize> = (0..BITS).collect();
        let heard: Vec<BitVector> = (0..n).map(|_| BitVector::from_bits(&all.choose_multiple(&mut rng, 32).copied().collect::<Vec<_>>(), BITS)).collect();
        let tract = VocalTract::new(n, BITS, 32, &mut rng);
        let mut m = MotorArea::new(BITS);
        // before babbling it cannot say anything
        assert!(m.plan(&heard[0]).is_none());
        m.babble(&tract, &heard, 10, &mut rng);
        let ov = |a: &BitVector, b: &BitVector| a.as_words().iter().zip(b.as_words()).map(|(x, y)| (x & y).count_ones()).sum::<u32>();
        let mut said_right = 0;
        let mut predicted = 0;
        for w in 0..n {
            if let Some(cmd) = m.plan(&heard[w]) {
                said_right += (tract.articulate(&cmd) == Some(w)) as usize;
                predicted += m.predict(&cmd).map_or(false, |p| ov(&p, &heard[w]) >= 24) as usize;
            }
        }
        assert!(said_right * 100 >= n * 95, "said right {said_right} of {n}");
        assert!(predicted * 100 >= n * 95, "predicted {predicted} of {n}");
    }

    #[test]
    fn a_garbled_command_says_nothing() {
        let mut rng = StdRng::seed_from_u64(2);
        let tract = VocalTract::new(10, BITS, 32, &mut rng);
        assert_eq!(tract.articulate(&BitVector::new(BITS, Some(0))), None);
        assert_eq!(tract.articulate(tract.command(3)), Some(3));
    }
}
