use std::ops::{BitOr, BitOrAssign};
use crate::bitvec::BitVector;

impl BitOr for &BitVector {
    type Output = BitVector;

    fn bitor(self, rhs: Self) -> Self::Output {
        assert!(self.as_words().len() == rhs.as_words().len());
        let words = self.words().zip(rhs.words())
            .map(|(a, b)| *a | *b)
            .collect();
        BitVector::from_words(words)
    }
}

impl BitOrAssign<&BitVector> for BitVector {
    fn bitor_assign(&mut self, rhs: &BitVector) {
        assert!(self.as_words().len() == rhs.as_words().len());
        for (a, b) in self.words_mut().zip(rhs.words()) {
            *a |= *b;
        }
    }
}

impl BitVector {
    pub fn or_mut(&mut self, other: &BitVector) -> &mut Self {
        *self |= other;
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::bitvec::BitVector;

    #[test]
    fn test_bitor() {
        let mut bv1 = BitVector::new(64, Some(0));
        let mut bv2 = BitVector::new(64, Some(0));
        bv1.bit_set(5);
        bv1.bit_set(10);
        bv1.bit_set(1);
        bv1.bit_set(63);
        bv2.bit_set(10);
        bv2.bit_set(15);

        let bv_or = &bv1 | &bv2;
        assert!(bv_or.bit_get(5));
        assert!(bv_or.bit_get(10));
        assert!(bv_or.bit_get(15));

        let mut bv_or = bv1.clone();
        bv_or.or_mut(&bv2);
        assert!(bv_or.bit_get(5));
        assert!(bv_or.bit_get(10));
        assert!(bv_or.bit_get(15));

        //Didn't mutate original
        assert!(bv1.bit_get(5));
        assert!(bv1.bit_get(10));
        assert!(!bv1.bit_get(15));

        bv1 |= &bv2;
        assert!(bv1.bit_get(1));
        assert!(bv1.bit_get(5));
        assert!(bv1.bit_get(10));
        assert!(bv1.bit_get(15));
        assert!(bv1.bit_get(63));
        assert!(bv1.count_ones() == 5);

    }
}