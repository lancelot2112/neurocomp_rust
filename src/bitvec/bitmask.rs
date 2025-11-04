use crate::bitvec::BitVector;

impl BitVector {
    /// Apply a word-aligned mask slice at `dest_word_idx` using `op` per word.
    /// `dest_word_idx` is a word index (bit_offset >> 6).
    pub fn mask_apply<F>(&mut self, dest_word_idx: usize, mask_words: &[u64], mut op: F)
    where
        F: FnMut(u64, u64) -> u64,
    {
        let words = self.as_words_mut();
        let n = words.len();
        assert!(dest_word_idx <= n, "dest_word_idx out of range");

        let max = std::cmp::min(n - dest_word_idx, mask_words.len());
        let dst = &mut words[dest_word_idx..dest_word_idx + max];

        for (a, &m) in dst.iter_mut().zip(mask_words.iter()) {
            *a = op(*a, m);
        }
    }

    /// Apply a single 64-bit mask at the given word index.
    pub fn mask_apply_word<F>(&mut self, dest_word_idx: usize, mask: u64, mut op: F)
    where
        F: FnMut(u64, u64) -> u64,
    {
        let words = self.as_words_mut();
        assert!(dest_word_idx < words.len());
        let w = &mut words[dest_word_idx];
        *w = op(*w, mask);
    }

    /// Convenience bitwise assign helpers (word aligned).
    pub fn mask_and(&mut self, dest_word_idx: usize, mask_words: &[u64]) {
        self.mask_apply(dest_word_idx, mask_words, |a, b| a & b);
    }
    pub fn mask_or(&mut self, dest_word_idx: usize, mask_words: &[u64]) {
        self.mask_apply(dest_word_idx, mask_words, |a, b| a | b);
    }
    pub fn mask_xor(&mut self, dest_word_idx: usize, mask_words: &[u64]) {
        self.mask_apply(dest_word_idx, mask_words, |a, b| a ^ b);
    }
    pub fn mask_clear(&mut self, dest_word_idx: usize, mask_words: &[u64]) {
        self.mask_apply(dest_word_idx, mask_words, |a, b| a & !b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_apply() {
        let mut bv = BitVector::new(128, Some(0));

        assert!(bv.count_ones() == 0);
        let mask = vec![0xFFFFFFFFFFFFFFFF; 2];
        bv.mask_apply(0, &mask, |a, b| a | b);

        assert!(bv.get(5));
        assert!(bv.get(70));
        assert!(bv.count_ones() == 128);

        let mask = vec![0x0F0F0F0F0F0F0F0F; 2];
        bv.mask_apply(0, &mask, |a, b| a ^ b);

        assert!(!bv.get(0));
        assert!(bv.get(4));
        assert!(bv.count_ones() == 64);
    }
}