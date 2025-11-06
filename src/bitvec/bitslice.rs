use crate::bitvec::bitvector::BitVector;

/// Metadata describing a contiguous range of bits inside a `BitVector`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitSlice {
    vector_index: usize,
    start_bit: usize,
    bit_len: usize,
}

impl BitSlice {
    /// Create a new slice that points at `bit_len` bits starting at `start_bit`
    /// in the `vector_index`th `BitVector` within some pool.
    pub fn from_bit_len(vector_index: usize, start_bit: usize, bit_len: usize) -> Self {
        assert!(bit_len > 0, "BitSlice length must be > 0");
        Self { vector_index, start_bit, bit_len }
    }

    pub fn from_bit_range(vector_index: usize, start_bit: usize, end_bit: Option<usize>) -> Self {
        let unwrapped_end_bit = end_bit.unwrap_or(start_bit + 1);
        assert!(unwrapped_end_bit > start_bit, "BitSlice must include at least 1 bit");

        let bit_len = unwrapped_end_bit - start_bit;
        Self { vector_index, start_bit, bit_len}
    }

    /// Index of the backing `BitVector` inside a pool.
    #[inline]
    pub fn vector_index(&self) -> usize { self.vector_index }

    /// First bit covered by this slice (inclusive).
    #[inline]
    pub fn start_bit(&self) -> usize { self.start_bit }

    /// Number of bits covered by this slice.
    #[inline]
    pub fn bit_len(&self) -> usize { self.bit_len }

    /// Bit immediately after the slice (exclusive).
    #[inline]
    pub fn end_bit(&self) -> usize { self.start_bit + self.bit_len }

    /// Index of the first word touched by this slice.
    #[inline]
    pub fn start_word(&self) -> usize { self.start_bit >> 6 }

    /// Index of the first word *after* this slice.
    #[inline]
    pub fn end_word(&self) -> usize { (self.end_bit() + 63) >> 6 }

    /// Offset of the first bit within the starting word.
    #[inline]
    pub fn bit_offset(&self) -> u32 { (self.start_bit & 0x3F) as u32 }

    /// Returns `true` if this slice touches the same vector and overlaps `other`.
    #[inline]
    pub fn overlaps(&self, other: &BitSlice) -> bool {
        self.vector_index == other.vector_index
            && self.start_bit < other.end_bit()
            && other.start_bit < self.end_bit()
    }

    /// Compute the bit-mask for `word_idx` restricted to the slice.
    fn word_mask(&self, word_idx: usize) -> u64 {
        //Sanity check in debug mode to check that we are only 
        //calling this function for words in range.
        assert!(
            (self.start_word()..self.end_word()).contains(&word_idx),
            "word index outside slice"
        );

        //Early return if we are FULLY inside the bitrange
        let slice_end = self.end_bit();
        if (self.start_bit >> 6) < word_idx && (slice_end >> 6) > word_idx {
            return u64::MAX
        }

        let slice_start = self.start_bit;
        let word_start = word_idx << 6;

        let start_in_word = slice_start.saturating_sub(word_start).max(0);
        let end_in_word = slice_end.saturating_sub(word_start).min(64);

        let mut mask = (1u64 << start_in_word)-1;
        mask |= u64::MAX - (1u64.unbounded_shl(end_in_word as u32).wrapping_sub(1));
        mask = !mask;
        mask
    }

    /// Iterate over word indices touched by this slice.
    pub fn iter_words(&self) -> BitSliceIter {
        BitSliceIter { slice: self.clone(), current_word: self.start_word() }
    }

    /// Perform the provided operation on the bits indicated by this slice
    ///
    /// `combine` receives `(current_bits, src_bits)` where both operands contain
    /// only the bits that fall inside the slice. The callback must return a value
    /// that only sets bits within the mask.
    pub fn mask_mut<F>(&self, dst: &mut BitVector, src: &BitVector, mut op: F)
    where
        F: FnMut(u64, u64) -> u64,
    {
        assert!(
            self.end_bit() <= dst.bit_len() && self.end_bit() <= src.bit_len(),
            "BitSlice out of bounds"
        );

        let dst_words = dst.as_words_mut();
        let src_words = src.as_words();

        for word_idx in self.start_word()..self.end_word() {
            let mask = self.word_mask(word_idx);
            if mask == 0 {
                continue;
            }

            let current = dst_words[word_idx];
            let src_bits = src_words[word_idx];
            let updated = op(current, src_bits) & mask;

            // Preserve bits outside the mask.
            dst_words[word_idx] = (dst_words[word_idx] & !mask) | updated;
        }
    }
}

/// Iterator over the word indices covered by a `BitSlice`.
pub struct BitSliceIter {
    slice: BitSlice,
    current_word: usize,
}

impl Iterator for BitSliceIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_word >= self.slice.end_word() {
            None
        } else {
            let word = self.current_word;
            self.current_word += 1;
            Some(word)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_masks_single_word() {
        let slice = BitSlice::from_bit_len(0, 5, 10);
        assert_eq!(slice.start_word(), 0);
        assert_eq!(slice.end_word(), 1);
        assert_eq!(slice.word_mask(0), ((1u64 << 10) - 1) << 5);
    }

    #[test]
    fn test_word_masks_crossing_words() {
        let slice = BitSlice::from_bit_len(0, 60, 20);
        let first_mask = slice.word_mask(0);
        let second_mask = slice.word_mask(1);
        assert_eq!(first_mask, (!0u64) << 60);
        assert_eq!(second_mask, (1u64 << 16) - 1);
    }

    #[test]
    fn test_overlaps() {
        let a = BitSlice::from_bit_len(0, 0, 32);
        let b = BitSlice::from_bit_len(0, 16, 32);
        let c = BitSlice::from_bit_len(1, 16, 32);
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_combine_with_or() {
        let slice = BitSlice::from_bit_len(0, 40, 40); // spans bits 40..79
        let mut dst = BitVector::from_bits(&[39, 85], 128); // bits just outside slice
        let src = BitVector::from_bits(&[45, 63, 78, 90], 128);

        slice.mask_mut(&mut dst, &src, |current, src_bits| current | src_bits);

        // Bits inside the slice should be OR-ed in.
        assert!(dst.bit_get(45));
        assert!(dst.bit_get(63));
        assert!(dst.bit_get(78));
        // Bit outside the slice should remain untouched.
        assert!(dst.bit_get(39));
        assert!(dst.bit_get(85));
        assert!(!dst.bit_get(90));
    }

    #[test]
    fn test_combine_with_xor_crossing_words() {
        let slice = BitSlice::from_bit_len(0, 60, 12); // crosses into second word
        let mut dst = BitVector::from_bits(&[60, 61, 70], 128);
        let src = BitVector::from_bits(&[60, 62, 70], 128);

        slice.mask_mut(&mut dst, &src, |current, src_bits| current ^ src_bits);

        assert!(!dst.bit_get(60)); // xor toggled off
        assert!(dst.bit_get(61));  // untouched by xor
        assert!(dst.bit_get(62));  // toggled on
        assert!(!dst.bit_get(70)); // toggled off
    }

    #[test]
    fn test_combine_with_and_clears_only_slice_bits() {
        let slice = BitSlice::from_bit_len(0, 10, 20);
        let mut dst = BitVector::from_bits(&[5, 10, 15, 25, 35], 128);
        let src = BitVector::from_bits(&[10, 15, 18], 128);

        slice.mask_mut(&mut dst, &src, |current, src_bits| current & src_bits);

        assert!(dst.bit_get(5));   // untouched
        assert!(!dst.bit_get(25));  // cleared by AND
        assert!(dst.bit_get(35));  // untouched
        assert!(dst.bit_get(10));  // kept (present in src and dest)
        assert!(dst.bit_get(15)); // kept (present in src and dest)
        assert!(!dst.bit_get(18)); // was 0 before, still 0
    }
}
