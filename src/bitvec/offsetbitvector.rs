use crate::bitvec::BitVector;

pub struct OffsetBitVector{
    mask: BitVector,
    offset: usize,
}

impl OffsetBitVector {
    /// Creates a new OffsetBitVector with the given number of bits and offset.
    pub fn new(num_bits: usize, offset: usize) -> Self {
        OffsetBitVector {
            mask: BitVector::new(num_bits, Some(0)),
            offset,
        }
    }

    /// Gets the value of the bit at the given index, adjusted by the offset.
    #[inline]
    pub fn get_bit(&self, index: usize) -> bool {
        self.mask.get_bit(index + self.offset)
    }

    /// Sets the bit at the given index to 1, adjusted by the offset.
    #[inline]
    pub fn set_bit(&mut self, index: usize) {
        self.mask.set_bit(index + self.offset);
    }

    /// Clears the bit at the given index to 0, adjusted by the offset.
    #[inline]
    pub fn clear_bit(&mut self, index: usize) {
        self.mask.clear_bit(index + self.offset);
    }
}