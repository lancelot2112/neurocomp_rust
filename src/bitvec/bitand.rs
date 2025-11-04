use std::ops::{BitAnd, BitAndAssign};
use crate::bitvec::BitVector;

impl BitAnd for &BitVector {
    type Output = BitVector;

    fn bitand(self, rhs: Self) -> Self::Output {
        assert!(self.as_words().len() == rhs.as_words().len());
        let words = self.words().zip(rhs.words())
            .map(|(a, b)| *a & *b)
            .collect();
        BitVector::from_words(words)
    }
}

impl BitAndAssign<&BitVector> for BitVector {
    fn bitand_assign(&mut self, rhs: &BitVector) {
        assert!(self.as_words().len() == rhs.as_words().len());
        for (a, b) in self.words_mut().zip(rhs.words()) {
            *a &= *b;
        }
    }
}

impl BitVector {
    pub fn and_mut(&mut self, other: &BitVector) -> &mut Self {
        *self &= other;
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::bitvec::BitVector;

    #[test]
    fn test_bitand() {
        let mut bv1 = BitVector::new(64, Some(0));
        let mut bv2 = BitVector::new(64, Some(0));
        bv1.set(5);
        bv1.set(10);
        bv1.set(1);
        bv1.set(63);
        bv2.set(10);
        bv2.set(15);

        let bv_and = &bv1 & &bv2;
        assert!(!bv_and.get(5));
        assert!(bv_and.get(10));
        assert!(!bv_and.get(15));

        let mut bv_and = bv1.clone();
        bv_and.and_mut(&bv2);
        assert!(!bv_and.get(5));
        assert!(bv_and.get(10));
        assert!(!bv_and.get(15));

        //Didn't mutate original
        assert!(bv1.get(5));
        assert!(bv1.get(10));
        assert!(!bv1.get(15));

        bv1 &= &bv2;
        assert!(!bv1.get(1));
        assert!(!bv1.get(5));
        assert!(bv1.get(10));
        assert!(!bv1.get(15));
        assert!(!bv1.get(63));
        assert!(bv1.count_ones() == 1);
    }
}