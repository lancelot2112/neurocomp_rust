use std::vec;
use crate::common::{ByteVector};


 /// A BitVector is a vector of bits stored in little-endian u64 words.
 /// The length of the vector is always a multiple of 64 bits (i.e., whole u64 words).
 /// 
 /// Within each u64 word: Bit 0 is the least significant bit (LSB), and bit 63 is the most significant bit (MSB). 
 /// For example, when loading from bytes in from_bytes, byte 0 (bits 0-7) is placed at the LSB end of the word, 
 /// with bit 0 of the byte becoming bit 0 of the word.
 /// 
 /// Across words: Word 0 contains bits 0-63, word 1 contains bits 64-127, and so on. The overall bit 0 of the vector 
 /// is the LSB of word 0, and the highest bit (e.g., bit 127 for a 128-bit vector) is the MSB of the last word.
 /// 
 /// This means the vector's bit 0 is the least significant bit of the entire data, and bit (len-1) is the most 
 /// significant bit. Your as_msbits() iterator yields bits from MSB to LSB (high index to low), while as_lsbits() 
 /// yields LSB to MSB (low index to high). If you need big-endian storage, you could reverse the words or 
 /// bits during construction.
pub struct BitVector {
    words: Vec<u64>,
}

impl BitVector {
    /// Creates a new BitVector with the given number of bits.
    /// The number of bits must be a multiple of 64.
    /// If init_val is provided, all bits are initialized to that value.
    pub fn new(num_bits: usize, init_val: Option<u64>) -> Self {
        assert!(num_bits > 0);
        //assert!(num_bits % 64 == 0, "num_bits must be a multiple of 64");
        let num_words = (num_bits + 63) >> 6; // divide by 64, rounding up

        BitVector {
            words: vec![init_val.unwrap_or(0); num_words],
        }
    }

    pub fn from_bits(indices: &[usize], len: usize) -> BitVector {
        let mut bv = BitVector::new(len, Some(0));
        for &idx in indices {
            bv.bit_set(idx);
        }
        bv
    }

    pub fn from_bytes(bytes: &[u8], num_bits: usize) -> Self {
        assert!(num_bits > 0);
        //assert!(num_bits % 8 == 0, "num_bits must be a multiple of 8");
        let num_bytes = (num_bits + 7) >> 3; // divide by 8, rounding up
        //assert!(bytes.len() >= num_bytes, "not enough bytes to fill BitVector");

        let num_words = (num_bits + 63) >> 6; // divide by 64, rounding up
        let mut words = vec![0u64; num_words];

        for i in 0..num_bytes {
            //Tile the bytes if not enough provided
            let byte = bytes[i % bytes.len()] as u64;
            let bit_index = i << 3; // i * 8
            let word_index = bit_index >> 6;
            let bit_offset = bit_index & 0x3F;

            words[word_index] |= byte << bit_offset;
        }

        BitVector { words }
    }

    pub fn from_words(words: Vec<u64>) -> Self {
        BitVector {
            words,
        }
    }
    
    /// Clone a slice of `count` bits starting at bit index `start`.
    /// `start + count` must be <= self.len(). The returned BitVector's
    /// length is rounded up to a multiple of 64 (unused high bits in the
    /// final word are zeroed).
    pub fn clone_slice(&self, start: usize, count: usize) -> BitVector {
        assert!(count > 0, "count must be > 0");
        assert!(
            start.checked_add(count).map_or(false, |end| end <= self.bit_len()),
            "slice out of bounds"
        );

        let res_words = (count + 63) >> 6;
        let mut bits = vec![0u64; res_words];
        let num_words = self.words.len();

        let mut src_bit = start;
        for dst_w in 0..res_words {
            let src_word = src_bit >> 6;
            let bit_shift = (src_bit & 0x3F) as u32;

            let lo = if src_word < num_words {
                self.words[src_word]
            } else {
                0
            };

            let word_val = if bit_shift == 0 {
                lo
            } else {
                let hi = if src_word + 1 < num_words {
                    self.words[src_word + 1]
                } else {
                    0
                };
                // bit_shift in 1..63 -> 64 - bit_shift in 1..63 safe
                (lo >> bit_shift) | (hi << (64 - bit_shift))
            };

            bits[dst_w] = word_val;
            src_bit = src_bit.saturating_add(64);
        }

        // Mask off high bits in last word if count is not a multiple of 64
        let rem = count & 0x3F;
        if rem != 0 {
            let mask = (1u64 << rem) - 1;
            bits[res_words - 1] &= mask;
        }

        BitVector { words: bits }
    }

    #[inline]
    pub fn word_len(&self) -> usize {
        self.words.len()
    }

    #[inline]
    pub fn word(&self, index: usize) -> u64 {
        self.words[index]
    }

    #[inline]
    pub fn as_words(&self) -> &[u64] {
        &self.words
    }

    #[inline]
    pub fn as_words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }

    #[inline]
    pub fn words(&self) -> std::slice::Iter<'_,u64> {
        self.words.iter()
    }

    #[inline]
    pub fn words_mut(&mut self) -> std::slice::IterMut<'_,u64> {
        self.words.iter_mut()
    }

    fn get_overlap_range(&self, idx: usize, len: usize) -> (usize, usize){
        let n = self.word_len();
        assert!(idx <= n, "mask idx out of range");
        let max = std::cmp::min(n - idx, len);
        assert!(max > 0, "mask length is zero at given index");
        (idx, idx+max)
    }

    pub fn as_slice<'a>(&'a self, idx: usize, len: usize) -> &'a [u64] {
        let (start, end) = self.get_overlap_range(idx, len);
        &self.as_words()[start..end]
    }
    pub fn as_slice_mut<'a>(&'a mut self, idx: usize, len: usize) -> &'a mut [u64] {
        let (start, end) = self.get_overlap_range(idx, len);
        &mut self.as_words_mut()[start..end]
    }
}


impl Clone for BitVector {
    fn clone(&self) -> Self {
        BitVector {
            words: self.words.clone(),
        }
    }
}

impl ByteVector for BitVector {
    fn as_bytes(&self) -> &[u8] {
        // Safety: u64 is guaranteed to be aligned and packed
        let byte_len = self.words.len() * 8;
        unsafe {
            std::slice::from_raw_parts(
                self.words.as_ptr() as *const u8,
                byte_len,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::BitVector;
    //use super::ByteRender;

    #[test]
    fn test_bitvector_clone_slice() {
        let mut bv = BitVector::new(128, Some(0));
        bv.bit_set(5);
        bv.bit_set(70);
        bv.bit_set(127);

        let bv_slice = bv.clone_slice(64, 64);
        assert!(!bv_slice.bit_get(5));
        assert!(bv_slice.bit_get(6));
        assert!(bv_slice.bit_get(63));

        let bv_slice2 = bv.clone_slice(0, 128);
        assert!(bv_slice2.bit_get(5));
        assert!(bv_slice2.bit_get(70));
        assert!(bv_slice2.bit_get(127));

        let bv_slice3 = bv.clone_slice(65, 10);
        assert!(!bv_slice3.bit_get(0));
        assert!(bv_slice3.bit_get(5));
        assert!(!bv_slice3.bit_get(9));
    }

    #[test]
    fn test_bitvector_clone() {
        let mut bv = BitVector::new(128, Some(0));
        bv.bit_set(5);
        bv.bit_set(70);

        let bv_clone = bv.clone();
        assert!(bv_clone.bit_get(5));
        assert!(bv_clone.bit_get(70));
        assert!(!bv_clone.bit_get(10));
    }
}