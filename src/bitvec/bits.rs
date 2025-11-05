use crate::bitvec::BitVector;

impl BitVector {
        /// Returns the number of bits in the vectror
    #[inline]
    pub fn len(&self) -> usize {
        self.as_words().len() << 6
    }

    /// Gets the value of the bit at the given index.
    #[inline]
    pub fn get(&self, index: usize) -> bool {
        let word_index = index >> 6;
        let bit_index = index & 0x3F;
        (self.as_words()[word_index] & (1u64 << bit_index)) != 0
    }

    /// Sets the bit at the given index to 1.
    #[inline]
    pub fn set(&mut self, index: usize) {
        let word_index = index >> 6;
        let bit_index = index & 0x3F;
        self.as_words_mut()[word_index] |= 1u64 << bit_index;
    }

    /// Clears the bit at the given index to 0.
    #[inline]
    pub fn clear(&mut self, index: usize) {
        let word_index = index >> 6;
        let bit_index = index & 0x3F;
        self.as_words_mut()[word_index] &= !(1u64 << bit_index);
    }

    #[inline]
    pub fn toggle(&mut self, index: usize) {
        let word_index = index >> 6;
        let bit_index = index & 0x3F;
        self.as_words_mut()[word_index] ^= 1u64 << bit_index;
    }
    /// Clears all bits in the vector to 0.
    /// This is more efficient than clearing bits one by one.
    pub fn clear_all(&mut self) -> &mut Self {
        for word in self.words_mut() {
            *word = 0;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitvector_clear_all() {
        let mut bv = BitVector::new(128, Some(u64::MAX));
        assert!(bv.get(0));
        assert!(bv.get(64));
        bv.clear_all();
        for i in 0..128 {
            assert!(!bv.get(i));
        }

        bv.set(5);
        bv.set(70);
        bv.set(127);
        bv.clear_all();
        assert!(!bv.get(5));
        assert!(!bv.get(70));
        assert!(!bv.get(127));
        for i in 0..128 {
            assert!(!bv.get(i));
        }
    }

    #[test]
    fn test_bitvector_get_set_clear() {
        let mut bv = BitVector::new(128, Some(0));
        assert_eq!(bv.len(), 128);
        assert!(!bv.get(10));
        bv.set(10);
        assert!(bv.get(10));
        bv.clear(10);
        assert!(!bv.get(10));
    }
}