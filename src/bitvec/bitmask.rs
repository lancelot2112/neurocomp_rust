use crate::bitvec::BitVector;

impl BitVector {
    pub fn mask<F>(&self, idx: usize, mask: &BitVector, mut op: F) -> BitVector
    where
        F: FnMut(u64, u64) -> u64,
    {
        let words = self.as_slice(idx, mask.word_len()).iter().zip(mask.as_words().iter())
            .map(|(&a, &m)| op(a, m))
            .collect();
        BitVector::from_words(words)
    }

    pub fn mask_and_count<F>(&self, idx: usize, mask: &BitVector, mut op: F) -> usize
    where
        F: FnMut(u64, u64) -> u64,
    {
        self.as_slice(idx, mask.word_len()).iter().zip(mask.as_words().iter())
            .map(|(&a, &m)| op(a, m).count_ones() as usize)
            .sum()
    }

    /// Apply a word-aligned mask slice at `idx` using `op` per word.
    /// `idx` is a word index (bit_offset >> 6).
    pub fn mask_mut<F>(&mut self, idx: usize, mask: &BitVector, mut op: F)
    where
        F: FnMut(u64, u64) -> u64,
    {
        for (a, &m) in self.as_slice_mut(idx,mask.word_len()).iter_mut().zip(mask.words()) {
            *a = op(*a, m);
        }
    }

    /// Convenience bitwise assign helpers (word aligned).
    pub fn mask_mut_and(&mut self, dest_word_idx: usize, mask: &BitVector) {
        self.mask_mut(dest_word_idx, mask, |a, b| a & b);
    }
    pub fn mask_mut_or(&mut self, dest_word_idx: usize, mask: &BitVector) {
        self.mask_mut(dest_word_idx, mask, |a, b| a | b);
    }
    pub fn mask_mut_xor(&mut self, dest_word_idx: usize, mask: &BitVector) {
        self.mask_mut(dest_word_idx, mask, |a, b| a ^ b);
    }
    pub fn mask_mut_clear(&mut self, dest_word_idx: usize, mask: &BitVector) {
        self.mask_mut(dest_word_idx, mask, |a, b| a & !b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_mut() {
        let mut bv = BitVector::new(128, Some(0));

        assert!(bv.count_ones() == 0);
        let mask = BitVector::from_words(vec![0xFFFFFFFFFFFFFFFF; 2]);
        bv.mask_mut(0, &mask, |a, b| a | b);

        assert!(bv.get(5));
        assert!(bv.get(70));
        assert!(bv.count_ones() == 128);

        let mask = BitVector::from_words(vec![0x0F0F0F0F0F0F0F0F; 2]);
        bv.mask_mut(0, &mask, |a, b| a ^ b);

        assert!(!bv.get(0));
        assert!(bv.get(4));
        assert!(bv.count_ones() == 64);
    }
}