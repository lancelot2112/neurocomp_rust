use crate::bitvec::BitVector;
use crate::kernel::class::{KernelTrait, KernelOp, KernelContext};



/// Per-kernel bookkeeping used by predictive learning and recycling.
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelStats {
    pub fired_last: bool, // fired on the most recent tick
    pub fires: u32,
    /// Fired and the target confirmed its output (8-bit, halved with `misses`, see
    /// `record_hit`).
    pub hits: u8,
    /// Fired and the target did not confirm its output.
    pub misses: u8,
    pub last_useful: u64, // class tick of creation or last hit (least-recently-useful recycling)
}

impl KernelStats {
    /// Count a hit. The counters are 8 bits: when either would pass 255, both are shifted
    /// right by one first, which keeps their ratio and makes the hit rate weight recent
    /// evidence (as the bit-sliced counters of the basal ganglia and CA3 do).
    pub fn record_hit(&mut self) {
        if self.hits == u8::MAX {
            self.halve();
        }
        self.hits += 1;
    }

    /// Count a miss (see `record_hit`).
    pub fn record_miss(&mut self) {
        if self.misses == u8::MAX {
            self.halve();
        }
        self.misses += 1;
    }

    /// Shift both counters right by one (ratio kept).
    pub fn halve(&mut self) {
        self.hits >>= 1;
        self.misses >>= 1;
    }
}

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
    pub plastic: bool,        // apply the local Hebbian rules (strengthen / search_remap) while firing
    pub context_frames: usize, // history frames the input mask spans (set by predictive growth)
    pub stats: KernelStats,
    /// Cached `input_mask.count_ones()`; code that clears or sets mask bits must call
    /// `sync_input_bits`.
    pub input_bits: usize,
    /// The input mask's set bits, relative to the mask (as `set_bits(&input_mask)`), kept
    /// with `input_bits`: learning walks these few positions instead of scanning the whole
    /// input width.
    pub input_set: Vec<u32>,
    /// The output mask as absolute bit positions (the mask is fixed after construction):
    /// checking a prediction against a target touches only these ~32 bits, not the
    /// whole output width.
    pub output_set: Vec<u32>,
}

impl SimpleKernel {
    pub fn new(input_mask: BitVector, input_idx: usize, output_mask: BitVector, output_idx: usize, threshold: usize, op: KernelOp) -> Self {
        let input_set = Self::positions(&input_mask, 0);
        let input_bits = input_set.len();
        let output_set = Self::positions(&output_mask, output_idx);
        Self {
            input_mask,
            input_idx,
            output_mask,
            output_idx,
            threshold,
            op,
            plastic: true,
            context_frames: 1,
            stats: KernelStats::default(),
            input_bits,
            input_set,
            output_set,
        }
    }

    fn positions(mask: &BitVector, word_offset: usize) -> Vec<u32> {
        let mut out = Vec::new();
        for (wi, &w) in mask.as_words().iter().enumerate() {
            let mut w = w;
            while w != 0 {
                out.push(((word_offset + wi) * 64 + w.trailing_zeros() as usize) as u32);
                w &= w - 1;
            }
        }
        out
    }

    /// Recompute `input_bits` after changing `input_mask` directly.
    pub fn sync_input_bits(&mut self) {
        self.input_set = Self::positions(&self.input_mask, 0);
        self.input_bits = self.input_set.len();
    }

    pub fn default(output_bit: usize) -> Self {
        let mut k = Self::new(
            BitVector::new(64, Some(0xF0F0_F0F0_F0F0_F0F0)),
            0,
            BitVector::from_bits(&[output_bit & 63], 1),
            output_bit >> 6,
            16,
            KernelOp::Or,
        );
        k.plastic = true;
        k
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

    #[inline]
    fn input_stats(&self, ctx: &KernelContext) -> InputStats {
        // Single pass popcount (reuse existing API)
        let count = ctx.input.mask_and_count(self.input_idx, &self.input_mask, |a,m| a & m);
        
        //The firing threshold is a function of "temperature" (as in simulated annealing)
        // - Higher temperature means we are more likely to fire (lower threshold)
        // It's also a function of the phase (to introduce oscillations)
        let firing_thresh = if ctx.temperature > 0 {
            self.threshold.saturating_sub(ctx.temperature as usize) + (ctx.phase & 0x10) as usize
        } else {
            self.threshold + (ctx.phase & 0x10) as usize
        };
        InputStats { threshold:firing_thresh, count }
    }

    #[inline]
    fn is_inhibited(&self, ctx: &KernelContext, _stats: &InputStats) -> bool {
        // The class sets one inhibit bit per OUTPUT word a winning kernel wrote,
        // so check our own output words (not our input window).
        let (start, end) = self.word_range();
        (start..end).any(|w| w < ctx.inhibit.bit_len() && ctx.inhibit.bit_get(w))
    }

    #[inline]
    fn search_remap<R: rand::Rng + ?Sized>(&mut self, ctx: &KernelContext, _stats: &InputStats, rng: &mut R) {
        // Example placeholder: move one connected+active to inactive+unconnected
        // (use your mask_move_random_* helpers)
        // If inhibited we SEARCH for a new pattern to connect to
        //  eg. [SEARCH] CLEAR a CONNECTED + ACTIVE bit and choose a random INACTIVE + UNCONNECTED bit to set.
        ctx.input.mask_move_random_connected_and_set(self.input_idx, &mut self.input_mask, rng);
        self.sync_input_bits();
    }

    #[inline]
    fn strengthen<R: rand::Rng + ?Sized>(&mut self, ctx: &KernelContext, _stats: &InputStats, rng: &mut R,) {
        // Eg: move one connected+inactive to active+unconnected
        //If not inhibited we STRENGTHEN our active pattern
        // eg. [STRENGTHEN] CLEAR a CONNECTED + INACTIVE bit and choose a random ACTIVE + UNCONNECTED bit to SET.
        ctx.input.mask_move_random_connected_and_not_set(self.input_idx, &mut self.input_mask, rng);
        self.sync_input_bits();
    }

    #[inline]
    fn apply_output(&self, out: &mut BitVector) {
        let f = Self::op_fn(self.op);
        out.mask_mut(self.output_idx, &self.output_mask, f);
    }
}

pub struct InputStats {
    pub threshold: usize, 
    pub count: usize,
}

impl KernelTrait for SimpleKernel {
    fn try_fire<R: rand::Rng + ?Sized>(
        &mut self,
        ctx: &KernelContext,
        out: &mut BitVector,
        rng: &mut R,
    ) -> bool {
        self.stats.fired_last = false;
        let stats = self.input_stats(ctx);
        if stats.count < stats.threshold {
            return false;
        }
        if self.is_inhibited(ctx, &stats) {
            if self.plastic {
                self.search_remap(ctx, &stats, rng);
            }
            return false;
        }
        if self.plastic {
            self.strengthen(ctx, &stats, rng);
        }
        self.apply_output(out);
        self.stats.fired_last = true;
        self.stats.fires += 1;
        true
    }

    fn excitation(&self, ctx: &KernelContext) -> isize {
        let stats = self.input_stats(ctx);
        stats.count as isize - stats.threshold as isize
    }

    fn word_range(&self) -> (usize, usize) {
        let start = self.output_idx;
        let end = start + self.output_mask.word_len();
        (start, end)
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

        let mut rng = rand::thread_rng();
        let ctx = KernelContext {
            input: &input,
            inhibit: &BitVector::new(64, Some(0)),
            temperature: 0,
            phase: 0,
        };

        let mut k = SimpleKernel::new(in_mask, 0, out_mask, 0, 8, KernelOp::Or);
        k.try_fire(&ctx, &mut output, &mut rng);

        // Expect upper nibble set
        assert_eq!(output.as_words()[0] & 0xFF, 0xF0);
    }

    #[test]
    fn does_not_fire_below_threshold() {
        let input = BitVector::from_words(vec![0x0F]); // 4 ones
        let in_mask = BitVector::from_words(vec![0xFF]);

        let mut output = BitVector::new(64, Some(0));
        let out_mask = BitVector::from_words(vec![0x0F]);

        let mut rng = rand::thread_rng();
        let ctx = KernelContext {
            input: &input,
            inhibit: &BitVector::new(64, Some(0)),
            temperature: 0,
            phase: 0,
        };

        let mut k = SimpleKernel::new(in_mask, 0, out_mask, 0, 5, KernelOp::Or);
        k.try_fire(&ctx, &mut output, &mut rng);

        assert_eq!(output.as_words()[0] & 0xFF, 0x00);
    }

    #[test]
    fn clear_mode_works() {
        let input = BitVector::from_words(vec![0xFFFF_FFFF_FFFF_FFFF]);
        let in_mask = BitVector::from_words(vec![0xFFFF_FFFF_FFFF_FFFF]);

        let mut output = BitVector::from_words(vec![0xFFFF]);
        let out_mask = BitVector::from_words(vec![0x0F0F]);

        let mut rng = rand::thread_rng();
        let ctx = KernelContext {
            input: &input,
            inhibit: &BitVector::new(64, Some(0)),
            temperature: 0,
            phase: 0,
        };

        let mut k = SimpleKernel::new(in_mask, 0, out_mask, 0, 1, KernelOp::Clear);
        k.try_fire(&ctx, &mut output, &mut rng);

        assert_eq!(output.as_words()[0] & 0xFFFF, 0xF0F0);
    }

    #[test]
    fn reacts_to_input_update() {
        let mut input = BitVector::from_words(vec![0x0F]); // 4 ones
        let in_mask = BitVector::from_words(vec![0xFF]);

        let mut output = BitVector::new(64, Some(0));
        let out_mask = BitVector::from_words(vec![0x0F]);

        let mut rng = rand::thread_rng();

        let mut k = SimpleKernel::new(in_mask, 0, out_mask, 0, 5, KernelOp::Or);
        let inhibit = BitVector::new(64, Some(0));
        k.try_fire(&KernelContext {
            input: &input,
            inhibit: &inhibit,
            temperature: 0,
            phase: 0,
        }, &mut output, &mut rng);

        assert_eq!(output.as_words()[0] & 0xFF, 0x00);

        input.bit_set(4); // now 5 ones (fires >= threshold of 5)
        assert_eq!(input.count_ones(), 5);

        k.try_fire(
            &KernelContext {
                input: &input,
                inhibit: &inhibit,
                temperature: 0,
                phase: 0,
            }, 
            &mut output,
            &mut rng);
        assert_eq!(output.as_words()[0] & 0xFF, 0x0F);

        output.bit_clear_all();
        assert_eq!(output.count_ones(), 0);
        input.bit_set(5); // now 6 ones
        assert_eq!(input.count_ones(), 6);
        k.try_fire(
            &KernelContext {
                input: &input,
                inhibit: &inhibit,
                temperature: 0,
                phase: 0,
            }, 
            &mut output,
            &mut rng);
        assert_eq!(output.as_words()[0] & 0xFF, 0x0F);
    }

    #[test]
    fn inhibition_is_checked_on_output_words() {
        let input = BitVector::from_words(vec![0xFF]);
        let mut output = BitVector::new(128, Some(0));
        let mut rng = rand::thread_rng();
        // reads input word 0, writes output word 1
        let mut k = SimpleKernel::new(BitVector::from_words(vec![0xFF]), 0, BitVector::from_words(vec![0x1]), 1, 8, KernelOp::Or);

        // a winner on output word 0 must not block a kernel writing word 1
        let inhibit = BitVector::from_bits(&[0], 64);
        let ctx = KernelContext { input: &input, inhibit: &inhibit, temperature: 0, phase: 0 };
        assert!(k.try_fire(&ctx, &mut output, &mut rng));

        // a winner on output word 1 does block it
        let inhibit = BitVector::from_bits(&[1], 64);
        let ctx = KernelContext { input: &input, inhibit: &inhibit, temperature: 0, phase: 0 };
        assert!(!k.try_fire(&ctx, &mut output, &mut rng));
        assert!(!k.stats.fired_last);
    }
}