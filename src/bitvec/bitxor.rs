use std::ops::{BitXor, BitXorAssign};
use crate::bitvec::BitVector;

impl BitXor for &BitVector {
    type Output = BitVector;

    fn bitxor(self, rhs: Self) -> Self::Output {
        assert!(self.as_words().len() == rhs.as_words().len());
        let words = self.words().zip(rhs.words())
            .map(|(a, b)| *a ^ *b)
            .collect();
        BitVector::from_words(words)
    }
}

impl BitXorAssign<&BitVector> for BitVector {
    fn bitxor_assign(&mut self, rhs: &BitVector) {
        assert!(self.as_words().len() == rhs.as_words().len());
        for (a, b) in self.words_mut().zip(rhs.words()) {
            *a ^= *b;
        }
    }
}

impl BitVector {
    pub fn xor_mut(&mut self, other: &BitVector) -> &mut Self {
        *self ^= other;
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::bitvec::BitVector;

    #[test]
    fn test_bitvector_bitxor() {
        let mut bv1 = BitVector::new(64, Some(0));
        let mut bv2 = BitVector::new(64, Some(0));
        bv1.bit_set(5);
        bv1.bit_set(10);
        bv1.bit_set(1);
        bv1.bit_set(63);
        bv2.bit_set(10);
        bv2.bit_set(15);

        let bv_xor = &bv1 ^ &bv2;
        assert!(bv_xor.bit_get(5));
        assert!(!bv_xor.bit_get(10));
        assert!(bv_xor.bit_get(15));

        let mut bv_xor = bv1.clone();
        bv_xor.xor_mut(&bv2);
        assert!(bv_xor.bit_get(5));
        assert!(!bv_xor.bit_get(10));
        assert!(bv_xor.bit_get(15));

        //Didn't mutate original
        assert!(bv1.bit_get(5));
        assert!(bv1.bit_get(10));
        assert!(!bv1.bit_get(15));

        bv1 ^= &bv2;
        assert!(bv1.bit_get(1));
        assert!(bv1.bit_get(5));
        assert!(!bv1.bit_get(10));
        assert!(bv1.bit_get(15));
        assert!(bv1.bit_get(63));
        assert!(bv1.count_ones() == 4);
    }
}