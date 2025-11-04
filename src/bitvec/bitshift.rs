use std::ops::{Shl, ShlAssign, Shr, ShrAssign};
use crate::bitvec::BitVector;

impl BitVector {
    pub fn shl_mut(&mut self, shift: usize) -> &mut Self {
        *self <<= shift;
        self
    }

    pub fn shr_mut(&mut self, shift: usize) -> &mut Self {
        *self >>= shift;
        self
    }
}

impl Shl<usize> for &BitVector {
    type Output = BitVector;

    fn shl(self, rhs: usize) -> Self::Output {
        let num_bits = self.len();
        assert!(rhs < num_bits);
        let num_words = self.as_words().len();
        let mut words = vec![0u64; num_words];

        let word_shift = rhs >> 6;
        let bit_shift = rhs & 0x3F;

        if bit_shift == 0 {
            // Pure whole-word shift: copy words into destination offset
            for i in 0..num_words - word_shift {
                words[i + word_shift] = self.as_words()[i];
            }
        } else {
            // Word + intra-word shift with carry to next word
            for i in 0..num_words {
                let v = self.as_words()[i];
                let dest = i + word_shift;
                if dest < num_words {
                    words[dest] |= v << bit_shift;
                }
                if dest + 1 < num_words {
                    words[dest + 1] |= v >> (64 - bit_shift);
                }
            }
        }

        BitVector::from_words(words)
    }
}

impl ShlAssign<usize> for BitVector {
    fn shl_assign(&mut self, rhs: usize) {
        let num_bits = self.len();
        assert!(rhs < num_bits);
        let num_words = self.as_words().len();
        let word_shift = rhs >> 6;
        let bit_shift = rhs & 0x3F;

        if bit_shift == 0 {
            // move whole words up; iterate high->low to avoid clobber
            for i in (0..num_words).rev() {
                if i >= word_shift {
                    self.as_words_mut()[i] = self.as_words()[i - word_shift];
                } else {
                    self.as_words_mut()[i] = 0;
                }
            }
        } else {
            // combine parts from lower source words; iterate high->low
            let inv_shift = 64 - bit_shift;
            for i in (0..num_words).rev() {
                let src = i as isize - word_shift as isize;
                if src < 0 {
                    self.as_words_mut()[i] = 0;
                } else {
                    let lo = self.as_words()[src as usize];
                    let hi = if src > 0 { self.as_words()[(src - 1) as usize] } else { 0 };
                    self.as_words_mut()[i] = (lo << bit_shift) | (hi >> inv_shift);
                }
            }
        }
    }
}

impl Shr<usize> for &BitVector {
    type Output = BitVector;

    fn shr(self, rhs: usize) -> Self::Output {
        let num_bits = self.len();
        assert!(rhs < num_bits);
        let num_words = self.as_words().len();
        let mut words = vec![0u64; num_words];

        let word_shift = rhs >> 6;
        let bit_shift = rhs & 0x3F;

        if bit_shift == 0 {
            // Pure whole-word shift: copy words into lower indices
            for i in word_shift..num_words {
                words[i - word_shift] = self.as_words()[i];
            }
        } else {
            // Word + intra-word shift with carry from next word
            for i in 0..num_words {
                let v = self.as_words()[i];
                if i >= word_shift {
                    let dest = i - word_shift;
                    words[dest] |= v >> bit_shift;
                    if dest > 0 {
                        words[dest - 1] |= v << (64 - bit_shift);
                    }
                }
            }
        }

        BitVector::from_words(words)
    }
}

impl ShrAssign<usize> for BitVector {
    fn shr_assign(&mut self, rhs: usize) {
        let num_bits = self.len();
        assert!(rhs < num_bits);
        let num_words = self.as_words().len();
        let word_shift = rhs >> 6;
        let bit_shift = rhs & 0x3F;

        if bit_shift == 0 {
            // move whole words down; iterate low->high to avoid clobber
            for i in 0..num_words {
                if i + word_shift < num_words {
                    self.as_words_mut()[i] = self.as_words()[i + word_shift];
                } else {
                    self.as_words_mut()[i] = 0;
                }
            }
        } else {
            // combine parts from higher source words; iterate low->high
            let inv_shift = 64 - bit_shift;
            for i in 0..num_words {
                let src = i + word_shift;
                if src >= num_words {
                    self.as_words_mut()[i] = 0;
                } else {
                    let lo = self.as_words()[src];
                    let hi = if src + 1 < num_words { self.as_words()[src + 1] } else { 0 };
                    self.as_words_mut()[i] = (lo >> bit_shift) | (hi << inv_shift);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::bitvec::BitVector;

    #[test]
    fn test_bitvector_shl_shr() {
        let mut bv1 = BitVector::new(128, Some(0));
        bv1.set(1);
        bv1.set(5);
        bv1.set(63);
        assert!(bv1.count_ones() == 3);

        let mut bv_shl = bv1.clone();
        bv_shl.shl_mut(3);
        assert!(bv_shl.get(8));
        assert!(!bv_shl.get(5));
        assert!(!bv_shl.get(63));
        assert!(!bv_shl.get(1));
        assert!(!bv_shl.get(2));
        assert!(bv_shl.get(4));
        assert!(bv_shl.count_ones() == 3);

        let mut bv_shr = bv1.clone();
        bv_shr.shr_mut(3);
        assert!(bv_shr.get(2));
        assert!(!bv_shr.get(5));
        assert!(!bv_shr.get(63));
        assert!(bv_shr.get(60));
        assert!(!bv_shr.get(1));
        assert!(bv_shr.count_ones() == 2);

        //Didn't mutate original
        assert!(bv1.get(1));
        assert!(bv1.get(5));
        assert!(bv1.get(63));
    }

    #[test]
    fn test_bitvector_shifts() {
        let mut bv = BitVector::new(128, Some(0));
        bv.set(5);
        bv.set(70);

        let bv_shl = &bv << 3;
        assert!(bv_shl.get(8));
        assert!(bv_shl.get(73));
        assert!(!bv_shl.get(5));

        let bv_shr = &bv >> 3;
        assert!(bv_shr.get(2));
        assert!(bv_shr.get(67));
        assert!(!bv_shr.get(70));

        // In-place shifts
        bv <<= 2;
        assert!(bv.get(7));
        assert!(bv.get(72));
        assert!(!bv.get(5));

        bv >>= 2;
        assert!(bv.get(5));
        assert!(bv.get(70));
        assert!(!bv.get(7));

        //large shifts
        bv <<= 122;
        assert!(bv.get(127));
        assert!(!bv.get(70));

        bv >>= 122;
        assert!(bv.get(5));
        assert!(!bv.get(127));
    }

    #[test]
    fn test_shift_time() 
    {
        use std::time::Instant;
        let mut bv = BitVector::new(1_000_000, Some(123456));
        let start = Instant::now();
        for _ in 0..1000 {
            bv <<= 13;
        }
        let duration = start.elapsed();
        println!("Time taken for <<= 13 on 1,000,000 bits: {:?}", duration);

        let start = Instant::now();
        for _ in 0..1000 {
            let _ = &bv << 13;
        }
        let duration = start.elapsed();
        println!("Time taken for << 13 on 1,000,000 bits: {:?}", duration);
    }
}