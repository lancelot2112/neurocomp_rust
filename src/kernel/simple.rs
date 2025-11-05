use crate::bitvec::BitVector;
use crate::kernel::class::{KernelTrait, KernelOp};



/// A simple “dendrite” kernel:
/// - Reads input ∧ input_mask, counts ones
/// - If count >= threshold, applies `op` of output_mask at `dest_word_idx` into `output`
pub struct SimpleKernel {
    pub input_mask: BitVector,
    pub input_idx: usize,  // word index in input where input_mask[0] lands
    pub output_mask: BitVector,
    pub output_idx: usize, // word index in output where output_mask[0] lands
    pub threshold: usize,     // fire when (input & input_mask).ones() >= threshold
    pub op: KernelOp,
}

impl SimpleKernel {
    pub fn new(input_mask: BitVector, input_idx: usize, output_mask: BitVector, output_idx: usize, threshold: usize, op: KernelOp) -> Self {
        Self { input_mask, input_idx, output_mask, output_idx, threshold, op }
    }

    #[inline]
    fn op_fn(op: KernelOp) -> fn(u64, u64) -> u64 {
        match op {
            KernelOp::Or => |a, b| a | b,
            KernelOp::And => |a, b| a & b,
            KernelOp::Xor => |a, b| a ^ b,
            KernelOp::Clear => |a, b| a & !b,
        }
    }
}

impl KernelTrait for SimpleKernel {
    fn process(&self, input: &BitVector, output: &mut BitVector, _temperature: &i16, _phase: &u16) {
        let count = input.mask_and_count(self.input_idx, &self.input_mask, |a, m| a & m);
        if count >= self.threshold {
            let f = Self::op_fn(self.op);
            output.mask_mut(self.output_idx, &self.output_mask, f);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitvec::BitVector;

    #[test]
    fn fires_when_threshold_met() {
        // input has ones at bits 0..=7 (first byte)
        let input = BitVector::from_words(vec![0xFF]);
        // input mask checks those same bits
        let in_mask = BitVector::from_words(vec![0xFF]);

        // output starts zero
        let mut output = BitVector::new(64, Some(0));
        // output mask sets upper nibble of first byte
        let out_mask = BitVector::from_words(vec![0xF0]);

        let k = SimpleKernel::new(in_mask, 0, out_mask, 0, 8, KernelOp::Or);
        k.process(&input, &mut output, &0, &0);

        // Expect upper nibble set
        assert_eq!(output.as_words()[0] & 0xFF, 0xF0);
    }

    #[test]
    fn does_not_fire_below_threshold() {
        let input = BitVector::from_words(vec![0x0F]); // 4 ones
        let in_mask = BitVector::from_words(vec![0xFF]);

        let mut output = BitVector::new(64, Some(0));
        let out_mask = BitVector::from_words(vec![0x0F]);

        let k = SimpleKernel::new(in_mask, 0, out_mask, 0, 5, KernelOp::Or);
        k.process(&input, &mut output, &0, &0);

        assert_eq!(output.as_words()[0] & 0xFF, 0x00);
    }

    #[test]
    fn clear_mode_works() {
        let input = BitVector::from_words(vec![0xFFFF_FFFF_FFFF_FFFF]);
        let in_mask = BitVector::from_words(vec![0xFFFF_FFFF_FFFF_FFFF]);

        let mut output = BitVector::from_words(vec![0xFFFF]);
        let out_mask = BitVector::from_words(vec![0x0F0F]);

        let k = SimpleKernel::new(in_mask, 0, out_mask, 0, 1, KernelOp::Clear);
        k.process(&input, &mut output, &0, &0);

        assert_eq!(output.as_words()[0] & 0xFFFF, 0xF0F0);
    }

    #[test]
    fn reacts_to_input_update() {
        let mut input = BitVector::from_words(vec![0x0F]); // 4 ones
        let in_mask = BitVector::from_words(vec![0xFF]);

        let mut output = BitVector::new(64, Some(0));
        let out_mask = BitVector::from_words(vec![0x0F]);

        let k = SimpleKernel::new(in_mask, 0, out_mask, 0, 5, KernelOp::Or);
        k.process(&input, &mut output, &0, &0);

        assert_eq!(output.as_words()[0] & 0xFF, 0x00);

        input.bit_set(4); // now 5 ones (fires >= threshold of 5)
        assert_eq!(input.count_ones(), 5);
        k.process(&input, &mut output, &0, &0);
        assert_eq!(output.as_words()[0] & 0xFF, 0x0F);

        output.bit_clear_all();
        assert_eq!(output.count_ones(), 0);
        input.bit_set(5); // now 6 ones
        assert_eq!(input.count_ones(), 6);
        k.process(&input, &mut output, &0, &0);
        assert_eq!(output.as_words()[0] & 0xFF, 0x0F);
    }
}