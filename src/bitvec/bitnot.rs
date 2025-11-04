use std::ops::Not;
use crate::bitvec::BitVector;

impl BitVector {
    pub fn not_mut(&mut self) -> &mut Self {
        for word in self.words_mut() {
            *word = !*word;
        }
        self
    }
}

impl Not for &BitVector {
    type Output = BitVector;

    fn not(self) -> Self::Output {
        let words = self.words()
            .map(|a| !*a)
            .collect();
        BitVector::from_words(words)
    }
}

#[cfg(test)]
mod tests {
    use crate::bitvec::BitVector;
    use crate::common::ByteVector;
    use std::ops::Not;

    #[test]
    fn test_bitnot() {
        //TODO: Plan what to do with extra bits? Currently they are just flipped as well.
        let mut bv = BitVector::from_bytes(&[0b10101010, 0b11110000], 16);
        bv.not_mut();
        assert_eq!(bv.as_bytes()[0..2], vec![0b01010101, 0b00001111]);

        let bv2 = BitVector::from_bytes(&[0b00001111, 0b11000000], 16);
        let bv3 = !&bv2;
        assert_eq!(bv3.as_bytes()[0..2], vec![0b11110000, 0b00111111]);

        let bv4 = bv2.not();
        assert_eq!(bv4.as_bytes()[0..2], vec![0b11110000, 0b00111111]);
    }
}