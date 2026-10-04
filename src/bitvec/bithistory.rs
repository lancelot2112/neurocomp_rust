use crate::bitvec::BitVector;

pub struct BitVecId {
    pub index: usize, // index of the BitVector in a frame store
    pub frame: Option<usize>, //Optional historical step [0=current, 1=prev1, 2=prev2]
}

#[derive(Clone, Copy, Debug)]
pub enum AdvanceMode {
    /// Advance frame, copy Prev1 into Current before writes (read-after-write isolation).
    CopyPrev,
    /// Advance frame, clear Current before writes.
    Clear,
    /// Do not advance; write into current (not recommended for multi-edge steps).
    Stay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistError {
    TooOld { requested: usize, max_back: usize },
}

/// A ring buffer of BitVectors to maintain a history of frames.
/// the HEAD always points to the "current" frame or frame 0. 
/// The previous frames are at increasing indices modulo the ring size.
pub struct BitVecHistory {
    frames: Vec<BitVector>, // frame
    head: usize,            // index of "current" frame
}

impl BitVecHistory {
    pub fn new(init: BitVector, hist_len: usize) -> Self {
        assert!(hist_len >= 1, "hist_len must be >= 1");
        let mut frames = Vec::with_capacity(hist_len);
        // Seed the ring with copies so Prev1/Prev2 exist at t0.
        for _ in 0..hist_len {
            frames.push(init.clone());
        }
        // Keep the original to avoid a clone on last push
        frames[hist_len - 1] = init;
        Self { frames, head: hist_len - 1 }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.frames[self.head].bit_len()
    }



    pub fn current(&self) -> &BitVector {
        &self.frames[self.head]
    }

    pub fn current_mut(&mut self) -> &mut BitVector {
        &mut self.frames[self.head]
    }

    pub fn get_frame(&self, steps: usize) -> &BitVector {
        assert!(steps < self.frames.len(), "frame {steps} is older than the history ring ({} frames)", self.frames.len());
        &self.frames[self.frame_idx(steps)]
    }

    pub fn snapshot(&self, steps: usize) -> BitVector {
        self.get_frame(steps).clone()
    }

    pub fn advance(&mut self, mode: AdvanceMode) {
        if let AdvanceMode::Stay = mode {
            return;
        }
        let prev = self.head;
        // Move head back by one (modulo ring size)
        self.head = if self.head == 0 { self.frames.len() - 1 } else { self.head - 1 };

        //Now handle the advance mode
        match mode {
            AdvanceMode::CopyPrev => {
                self.copy_frame_from(self.head, prev);
            }
            AdvanceMode::Clear => {
                // Reinitialize to zeros, preserving bit length.
                self.frames[self.head].bit_clear_all();
            }
            AdvanceMode::Stay => {}
        }
    }

    #[inline]
    fn frame_idx(&self, n: usize) -> usize {
        (self.head + n) % self.frames.len()
    }

        /// Copy content from src_idx into dst_idx using disjoint borrows.
    fn copy_frame_from(&mut self, dst_idx: usize, src_idx: usize) {
        if dst_idx == src_idx {
            return;
        }
        assert!(dst_idx < self.frames.len() && src_idx < self.frames.len());

        // Split the slice so dst and src live in disjoint slices.
        if dst_idx < src_idx {
            let (left, right) = self.frames.split_at_mut(src_idx);
            let dst = &mut left[dst_idx];
            let src = &right[0]; // element at src_idx
            copy_bitvec(dst, src);
        } else {
            let (left, right) = self.frames.split_at_mut(dst_idx);
            let src = &left[src_idx];
            let dst = &mut right[0]; // element at dst_idx
            copy_bitvec(dst, src);
        }
    }
}

/// Prefer a no-alloc copy. If BitVector exposes word slices, use copy_from_slice.
/// Fallback to clone_from (which you can specialize in BitVector to reuse buffers).
#[inline]
fn copy_bitvec(dst: &mut BitVector, src: &BitVector) {
    // Fast path if available in your BitVector:
    // dst.as_words_mut().copy_from_slice(src.as_words());

    // Generic fallback:
    dst.clone_from(src);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitvec::BitVector;

    #[inline]
    fn first_word(bv: &BitVector) -> u64 {
        // Rely on LS-word iteration to read the first 64-bit word
        bv.as_lswords().next().unwrap_or(0)
    }

    #[inline]
    fn set_current_exact(hist: &mut BitVecHistory, word: u64) {
        // Clear then OR a mask so current frame equals `word`
        let mask = BitVector::from_words(vec![word]);
        let cur = hist.current_mut();
        cur.bit_clear_all();
        cur.mask_mut(0, &mask, |a, b| a | b);
    }

    #[test]
    fn init_ring_frames_are_copies() {
        let init_word = 0xF0F0_F0F0_F0F0_F0F0u64;
        let init_bv = BitVector::from_words(vec![init_word]);
        let hist = BitVecHistory::new(init_bv, 3);

        assert_eq!(first_word(hist.get_frame(0)), init_word);
        assert_eq!(first_word(hist.get_frame(1)), init_word);
        assert_eq!(first_word(hist.get_frame(2)), init_word);
    }

    #[test]
    fn advance_copyprev_copies_previous_frame() {
        let init = BitVector::from_words(vec![0u64]);
        let mut hist = BitVecHistory::new(init, 4);

        // Write A, then CopyPrev -> current should be A
        let a = 0x00FF_00FF_00FF_00FFu64;
        set_current_exact(&mut hist, a);
        hist.advance(AdvanceMode::CopyPrev);
        assert_eq!(first_word(hist.current()), a);

        // Write B, then CopyPrev -> current should be B
        let b = 0xAA55_AA55_AA55_AA55u64;
        set_current_exact(&mut hist, b);
        hist.advance(AdvanceMode::CopyPrev);
        assert_eq!(first_word(hist.current()), b);

        // Write C, advance enough to force ring wrap (exercise other split_at_mut branch)
        let c = 0xDEAD_BEEF_CAFE_BABEu64;
        set_current_exact(&mut hist, c);
        // Head will wrap only when old head == 0; do two more advances to be safe.
        hist.advance(AdvanceMode::CopyPrev);
        assert_eq!(first_word(hist.current()), c);

        // One more distinct value after wrap
        let d = 0x0123_4567_89AB_CDEFu64;
        set_current_exact(&mut hist, d);
        hist.advance(AdvanceMode::CopyPrev);
        assert_eq!(first_word(hist.current()), d);
    }

    #[test]
    fn advance_clear_zeros_current_frame() {
        let init = BitVector::from_words(vec![0u64]);
        let mut hist = BitVecHistory::new(init, 3);

        // Make current non-zero
        set_current_exact(&mut hist, 0xFFFF_FFFF_FFFF_FFFF);
        // Clear
        hist.advance(AdvanceMode::Clear);
        assert_eq!(first_word(hist.current()), 0);
    }

    #[test]
    fn advance_stay_leaves_current_untouched() {
        let init = BitVector::from_words(vec![0u64]);
        let mut hist = BitVecHistory::new(init, 3);

        let v = 0x0F0F_0F0F_0F0F_0F0Fu64;
        set_current_exact(&mut hist, v);
        hist.advance(AdvanceMode::Stay);
        assert_eq!(first_word(hist.current()), v);
    }

    #[test]
    fn snapshot_returns_independent_clone() {
        let init = BitVector::from_words(vec![0u64]);
        let mut hist = BitVecHistory::new(init, 3);

        let a = 0x1111_2222_3333_4444u64;
        set_current_exact(&mut hist, a);
        let snap = hist.snapshot(0); // clone of current

        // Change current, snapshot should remain A
        let b = 0xFFFF_0000_F0F0_0F0Fu64;
        set_current_exact(&mut hist, b);

        assert_eq!(first_word(&snap), a);
        assert_eq!(first_word(hist.current()), b);
    }

    #[test]
    fn get_frame_steps_observe_history() {
        let init = BitVector::from_words(vec![0u64]);
        let mut hist = BitVecHistory::new(init, 3);

        // Step 1: write A and carry forward
        let a = 0xAAAA_AAAA_AAAA_AAAAu64;
        set_current_exact(&mut hist, a);
        hist.advance(AdvanceMode::CopyPrev);
        // Now: current=A, prev1=A, prev2=0

        // Step 2: write B and carry forward
        let b = 0x5555_5555_5555_5555u64;
        set_current_exact(&mut hist, b);
        hist.advance(AdvanceMode::CopyPrev);
        // Now: current=B, prev1=B (copied from previous frame), prev2=A

        assert_eq!(first_word(hist.get_frame(0)), b);
        assert_eq!(first_word(hist.get_frame(1)), b); // CopyPrev duplicates last frame
        assert_eq!(first_word(hist.get_frame(2)), a);
    }
}