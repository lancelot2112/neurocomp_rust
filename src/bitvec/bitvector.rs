use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, ShlAssign, Shr, ShrAssign};

use crate::common::{ByteVector, ByteRender};

pub struct BitVector {
    bits: Vec<u64>,
}

impl BitVector {
    /// Creates a new BitVector with the given number of bits.
    /// The number of bits must be a multiple of 64.
    /// If init_val is provided, all bits are initialized to that value.
    pub fn new(num_bits: usize, init_val: Option<u64>) -> Self {
        assert!(num_bits > 0);
        assert!(num_bits % 64 == 0, "num_bits must be a multiple of 64");
        let num_words = (num_bits + 63) >> 6; // divide by 64, rounding up

        BitVector {
            bits: vec![init_val.unwrap_or(0); num_words],
        }
    }

    /// Returns the number of bits in the vectror
    #[inline]
    pub fn len(&self) -> usize {
        self.bits.len() << 6
    }

    /// Gets the value of the bit at the given index.
    #[inline]
    pub fn get(&self, index: usize) -> bool {
        let word_index = index >> 6;
        let bit_index = index & 0x3F;
        (self.bits[word_index] & (1u64 << bit_index)) != 0
    }

    /// Sets the bit at the given index to 1.
    #[inline]
    pub fn set(&mut self, index: usize) {
        let word_index = index >> 6;
        let bit_index = index & 0x3F;
        self.bits[word_index] |= 1u64 << bit_index;
    }

    /// Clears the bit at the given index to 0.
    #[inline]
    pub fn clear(&mut self, index: usize) {
        let word_index = index >> 6;
        let bit_index = index & 0x3F;
        self.bits[word_index] &= !(1u64 << bit_index);
    }

    /// Clears all bits in the vector to 0.
    /// This is more efficient than clearing bits one by one.
    pub fn clear_all(&mut self) -> &mut Self {
        for word in self.bits.iter_mut() {
            *word = 0;
        }
        self
    }

    pub fn not(&mut self) -> &mut Self {
        for word in self.bits.iter_mut() {
            *word = !*word;
        }
        self
    }

    
    /// Clone a slice of `count` bits starting at bit index `start`.
    /// `start + count` must be <= self.len(). The returned BitVector's
    /// length is rounded up to a multiple of 64 (unused high bits in the
    /// final word are zeroed).
    pub fn clone_slice(&self, start: usize, count: usize) -> BitVector {
        assert!(count > 0, "count must be > 0");
        assert!(
            start.checked_add(count).map_or(false, |end| end <= self.len()),
            "slice out of bounds"
        );

        let res_words = (count + 63) >> 6;
        let mut bits = vec![0u64; res_words];
        let num_words = self.bits.len();

        let mut src_bit = start;
        for dst_w in 0..res_words {
            let src_word = src_bit >> 6;
            let bit_shift = (src_bit & 0x3F) as u32;

            let lo = if src_word < num_words {
                self.bits[src_word]
            } else {
                0
            };

            let word_val = if bit_shift == 0 {
                lo
            } else {
                let hi = if src_word + 1 < num_words {
                    self.bits[src_word + 1]
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

        BitVector { bits }
    }

    pub fn rotr(&mut self, shift: usize) -> &mut Self {
                let num_bits = self.len();
        if num_bits == 0 { return self; }

        // normalize shift
        let k = shift % num_bits;
        if k == 0 { return self; }

        let num_words = self.bits.len();
        let word_shift = k >> 6;
        let bit_shift = (k & 0x3F) as u32;

        // perform whole-word rotation in-place (no full clone)
        // CORRECTION: rotate left by word_shift for a right-rotate of bits
        if word_shift != 0 {
            self.bits.rotate_left(word_shift);
        }

        // if we only rotated whole words, we're done
        if bit_shift == 0 {
            return self;
        }

        // do the intra-word rotate using a single u64 temp (save first element for wrap)
        let inv = 64 - bit_shift;
        let first = self.bits[0];
        for i in 0..num_words {
            let curr = self.bits[i];
            let next = if i + 1 < num_words { self.bits[i + 1] } else { first };
            self.bits[i] = (curr >> bit_shift) | (next << inv);
        }

        self
    }

    pub fn rotl(&mut self, shift: usize) -> &mut Self {
        let num_bits = self.len();
        if num_bits == 0 { return self; }

        let k = shift % num_bits;
        if k == 0 { return self; }

        // rotate-left by k is rotate-right by (num_bits - k)
        let complement = (num_bits - k) % num_bits;
        self.rotr(complement);
        self
    }

    pub fn shl(&mut self, shift: usize) -> &mut Self {
        *self <<= shift;
        self
    }

    pub fn shr(&mut self, shift: usize) -> &mut Self {
        *self >>= shift;
        self
    }

    pub fn and(&mut self, other: &BitVector) -> &mut Self {
        *self &= other;
        self
    }

    pub fn or(&mut self, other: &BitVector) -> &mut Self {
        *self |= other;
        self
    }

    pub fn xor(&mut self, other: &BitVector) -> &mut Self {
        *self ^= other;
        self
    }

        /// Count number of set bits in the whole vector.
    #[inline]
    pub fn popcount(&self) -> usize {
        // fast: use hardware popcount via count_ones on u64
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Count number of set bits in the range [start, start+count).
    /// `start + count` must be <= self.len().
    pub fn popcount_range(&self, start: usize, count: usize) -> usize {
        if count == 0 {
            return 0;
        }
        assert!(
            start.checked_add(count).map_or(false, |end| end <= self.len()),
            "range out of bounds"
        );

        let mut remaining = count;
        let mut pos = start;
        let mut sum = 0usize;

        while remaining > 0 {
            let word_idx = pos >> 6;
            let bit_off = pos & 0x3F;
            // number of bits we can take from this word
            let take = std::cmp::min(remaining, 64 - bit_off);

            // build mask for bits [bit_off, bit_off + take)
            let mask = if take == 64 {
                u64::MAX
            } else {
                ((1u64 << take) - 1) << bit_off
            };

            let v = self.bits[word_idx] & mask;
            sum += v.count_ones() as usize;

            pos += take;
            remaining -= take;
        }

        sum
    }
}

impl BitAnd for &BitVector {
    type Output = BitVector;

    fn bitand(self, rhs: Self) -> Self::Output {
        assert!(self.bits.len() == rhs.bits.len());
        let bits = self.bits.iter().zip(rhs.bits.iter())
            .map(|(a, b)| *a & *b)
            .collect();
        BitVector { bits }
    }
}

impl BitAndAssign<&BitVector> for BitVector {
    fn bitand_assign(&mut self, rhs: &BitVector) {
        assert!(self.bits.len() == rhs.bits.len());
        for (a, b) in self.bits.iter_mut().zip(rhs.bits.iter()) {
            *a &= *b;
        }
    }
}

impl BitOr for &BitVector {
    type Output = BitVector;

    fn bitor(self, rhs: Self) -> Self::Output {
        assert!(self.bits.len() == rhs.bits.len());
        let bits = self.bits.iter().zip(rhs.bits.iter())
            .map(|(a, b)| *a | *b)
            .collect();
        BitVector { bits }
    }
}

impl BitOrAssign<&BitVector> for BitVector {
    fn bitor_assign(&mut self, rhs: &BitVector) {
        assert!(self.bits.len() == rhs.bits.len());
        for (a, b) in self.bits.iter_mut().zip(rhs.bits.iter()) {
            *a |= *b;
        }
    }
}

impl BitXor for &BitVector {
    type Output = BitVector;

    fn bitxor(self, rhs: Self) -> Self::Output {
        assert!(self.bits.len() == rhs.bits.len());
        let bits = self.bits.iter().zip(rhs.bits.iter())
            .map(|(a, b)| *a ^ *b)
            .collect();
        BitVector { bits }
    }
}

impl BitXorAssign<&BitVector> for BitVector {
    fn bitxor_assign(&mut self, rhs: &BitVector) {
        assert!(self.bits.len() == rhs.bits.len());
        for (a, b) in self.bits.iter_mut().zip(rhs.bits.iter()) {
            *a ^= *b;
        }
    }
}

impl Not for &BitVector {
    type Output = BitVector;

    fn not(self) -> Self::Output {
        let bits = self.bits.iter()
            .map(|a| !*a)
            .collect();
        BitVector { bits }
    }
}

impl Shl<usize> for &BitVector {
    type Output = BitVector;

    fn shl(self, rhs: usize) -> Self::Output {
        let num_bits = self.len();
        assert!(rhs < num_bits);
        let num_words = self.bits.len();
        let mut bits = vec![0u64; num_words];

        let word_shift = rhs >> 6;
        let bit_shift = rhs & 0x3F;

        if bit_shift == 0 {
            // Pure whole-word shift: copy words into destination offset
            for i in 0..num_words - word_shift {
                bits[i + word_shift] = self.bits[i];
            }
        } else {
            // Word + intra-word shift with carry to next word
            for i in 0..num_words {
                let v = self.bits[i];
                let dest = i + word_shift;
                if dest < num_words {
                    bits[dest] |= v << bit_shift;
                }
                if dest + 1 < num_words {
                    bits[dest + 1] |= v >> (64 - bit_shift);
                }
            }
        }

        BitVector { bits }
    }
}

impl ShlAssign<usize> for BitVector {
    fn shl_assign(&mut self, rhs: usize) {
        let num_bits = self.len();
        assert!(rhs < num_bits);
        let num_words = self.bits.len();
        let word_shift = rhs >> 6;
        let bit_shift = rhs & 0x3F;

        if bit_shift == 0 {
            // move whole words up; iterate high->low to avoid clobber
            for i in (0..num_words).rev() {
                if i >= word_shift {
                    self.bits[i] = self.bits[i - word_shift];
                } else {
                    self.bits[i] = 0;
                }
            }
        } else {
            // combine parts from lower source words; iterate high->low
            let inv_shift = 64 - bit_shift;
            for i in (0..num_words).rev() {
                let src = i as isize - word_shift as isize;
                if src < 0 {
                    self.bits[i] = 0;
                } else {
                    let lo = self.bits[src as usize];
                    let hi = if src > 0 { self.bits[(src - 1) as usize] } else { 0 };
                    self.bits[i] = (lo << bit_shift) | (hi >> inv_shift);
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
        let num_words = self.bits.len();
        let mut bits = vec![0u64; num_words];

        let word_shift = rhs >> 6;
        let bit_shift = rhs & 0x3F;

        if bit_shift == 0 {
            // Pure whole-word shift: copy words into lower indices
            for i in word_shift..num_words {
                bits[i - word_shift] = self.bits[i];
            }
        } else {
            // Word + intra-word shift with carry from next word
            for i in 0..num_words {
                let v = self.bits[i];
                if i >= word_shift {
                    let dest = i - word_shift;
                    bits[dest] |= v >> bit_shift;
                    if dest > 0 {
                        bits[dest - 1] |= v << (64 - bit_shift);
                    }
                }
            }
        }

        BitVector { bits }
    }
}

impl ShrAssign<usize> for BitVector {
    fn shr_assign(&mut self, rhs: usize) {
        let num_bits = self.len();
        assert!(rhs < num_bits);
        let num_words = self.bits.len();
        let word_shift = rhs >> 6;
        let bit_shift = rhs & 0x3F;

        if bit_shift == 0 {
            // move whole words down; iterate low->high to avoid clobber
            for i in 0..num_words {
                if i + word_shift < num_words {
                    self.bits[i] = self.bits[i + word_shift];
                } else {
                    self.bits[i] = 0;
                }
            }
        } else {
            // combine parts from higher source words; iterate low->high
            let inv_shift = 64 - bit_shift;
            for i in 0..num_words {
                let src = i + word_shift;
                if src >= num_words {
                    self.bits[i] = 0;
                } else {
                    let lo = self.bits[src];
                    let hi = if src + 1 < num_words { self.bits[src + 1] } else { 0 };
                    self.bits[i] = (lo >> bit_shift) | (hi << inv_shift);
                }
            }
        }
    }
}

impl Clone for BitVector {
    fn clone(&self) -> Self {
        BitVector {
            bits: self.bits.clone(),
        }
    }
}

impl ByteVector for BitVector {
    fn as_bytes(&self) -> &[u8] {
        // Safety: u64 is guaranteed to be aligned and packed
        let byte_len = self.bits.len() * 8;
        unsafe {
            std::slice::from_raw_parts(
                self.bits.as_ptr() as *const u8,
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
    fn test_bitvector_basic() {
        let mut bv = BitVector::new(128, Some(0));
        assert_eq!(bv.len(), 128);
        assert!(!bv.get(10));
        bv.set(10);
        assert!(bv.get(10));
        bv.clear(10);
        assert!(!bv.get(10));
    }

    #[test]
    fn test_popcount() {
        let mut bv = BitVector::new(128, Some(0));
        assert_eq!(bv.popcount(), 0);
        bv.set(0);
        bv.set(63);
        bv.set(64);
        bv.set(127);
        assert_eq!(bv.popcount(), 4);
        assert_eq!(bv.popcount_range(0, 64), 2);
        assert_eq!(bv.popcount_range(64, 64), 2);
        assert_eq!(bv.popcount_range(32, 64), 2);
    }

    #[test]
    fn test_bitvector_ops() {
        let mut bv1 = BitVector::new(64, Some(0));
        let mut bv2 = BitVector::new(64, Some(0));
        bv1.set(5);
        bv1.set(10);
        bv1.set(1);
        bv1.set(63);
        bv2.set(10);
        bv2.set(15);

        let mut bv_and = bv1.clone();
        bv_and.and(&bv2);
        assert!(!bv_and.get(5));
        assert!(bv_and.get(10));
        assert!(!bv_and.get(15));


        let mut bv_or = bv1.clone();
        bv_or.or(&bv2);
        assert!(bv_or.get(5));
        assert!(bv_or.get(10));
        assert!(bv_or.get(15));

        let mut bv_not = bv1.clone();
        bv_not.not();
        assert!(!bv_not.get(5));
        assert!(!bv_not.get(10));
        assert!(bv_not.get(15));

        let mut bv_xor = bv1.clone();
        bv_xor.xor(&bv2);
        assert!(bv_xor.get(5));
        assert!(!bv_xor.get(10));
        assert!(bv_xor.get(15));

        let mut bv_shl = bv1.clone();
        bv_shl.shl(3);
        assert!(bv_shl.get(8));
        assert!(!bv_shl.get(5));
        assert!(!bv_shl.get(63));
        assert!(!bv_shl.get(1));
        assert!(bv_shl.get(4));

        let mut bv_shr = bv1.clone();
        bv_shr.shr(3);
        assert!(bv_shr.get(2));
        assert!(!bv_shr.get(5));
        assert!(!bv_shr.get(63));
        assert!(bv_shr.get(60));
        assert!(!bv_shr.get(1));

        let mut bv_rotr = bv1.clone();
        bv_rotr.rotr(3);
        assert!(bv_rotr.get(2));
        assert!(!bv_rotr.get(5));
        assert!(!bv_rotr.get(63));
        assert!(bv_rotr.get(60));
        assert!(!bv_rotr.get(1));
        assert!(bv_rotr.get(62));

        let mut bv_rotl = bv1.clone();
        bv_rotl.rotl(3);
        assert!(bv_rotl.get(8));
        assert!(!bv_rotl.get(5));
        assert!(!bv_rotl.get(63));
        assert!(bv_rotl.get(2));
        assert!(!bv_rotl.get(1));
        assert!(bv_rotl.get(4));

    }

    #[test]
    fn test_bitvector_clone_slice() {
        let mut bv = BitVector::new(128, Some(0));
        bv.set(5);
        bv.set(70);
        bv.set(127);

        let bv_slice = bv.clone_slice(64, 64);
        assert!(!bv_slice.get(5));
        assert!(bv_slice.get(6));
        assert!(bv_slice.get(63));

        let bv_slice2 = bv.clone_slice(0, 128);
        assert!(bv_slice2.get(5));
        assert!(bv_slice2.get(70));
        assert!(bv_slice2.get(127));

        let bv_slice3 = bv.clone_slice(65, 10);
        assert!(!bv_slice3.get(0));
        assert!(bv_slice3.get(5));
        assert!(!bv_slice3.get(9));
    }

    #[test]
    fn test_bitvector_clear_all() {
        let mut bv = BitVector::new(128, Some(u64::MAX));
        assert!(bv.get(0));
        assert!(bv.get(64));
        bv.clear_all();
        for i in 0..128 {
            assert!(!bv.get(i));
        }
    }

    #[test]
    fn test_bitvector_clone() {
        let mut bv = BitVector::new(128, Some(0));
        bv.set(5);
        bv.set(70);

        let bv_clone = bv.clone();
        assert!(bv_clone.get(5));
        assert!(bv_clone.get(70));
        assert!(!bv_clone.get(10));
    }

    #[test]
    fn test_large_rotate()
    {
        let mut bv = BitVector::new(256, Some(0));
        bv.set(0);
        bv.set(128);
        bv.set(255);
        //bv.set(9995);

        //bv.write_raw(128).unwrap();
        bv.rotl(130);
        //bv.write_raw(128).unwrap();
        assert!(!bv.get(0));
        assert!(bv.get(130)); // 0 + 130 = 130
        assert!(bv.get(2)); // 128 + 130 - 256 = 2
        assert!(bv.get(129)); // 255 + 130 - 256 = 129
        //assert!(bv.get(78)); // 9995 + 130 - 10048 = 77
        assert!(!bv.get(128));
        assert!(!bv.get(255));

        bv.rotr(130); //Undoes left rotate
        assert!(bv.get(0));
        assert!(bv.get(128));
        assert!(bv.get(255));
        //assert!(bv.get(9995)); 
    }

    #[test]
    fn test_bitvector_and_or_not() {
        let mut bv1 = BitVector::new(128, Some(0));
        let mut bv2 = BitVector::new(128, Some(0));
        bv1.set(5);
        bv1.set(10);
        bv2.set(10);
        bv2.set(15);

        let bv_and = &bv1 & &bv2;
        assert!(!bv_and.get(5));
        assert!(bv_and.get(10));
        assert!(!bv_and.get(15));

        let bv_or = &bv1 | &bv2;
        assert!(bv_or.get(5));
        assert!(bv_or.get(10));
        assert!(bv_or.get(15));

        let bv_not = !&bv1;
        assert!(!bv_not.get(5));
        assert!(!bv_not.get(10));
        assert!(bv_not.get(15));

        let bv_xor = &bv1 ^ &bv2;
        assert!(bv_xor.get(5));
        assert!(!bv_xor.get(10));
        assert!(bv_xor.get(15));

        //Didn't mutate original
        assert!(bv1.get(5));
        assert!(bv1.get(10));
        assert!(!bv1.get(15));

        // Mutate in place
        bv1.not();
        assert!(!bv1.get(5));
        assert!(!bv1.get(10));
        assert!(bv1.get(15));

        bv1 |= &bv2;
        assert!(bv1.get(1));
        assert!(!bv1.get(5));
        assert!(bv1.get(10));
        assert!(bv1.get(15));

        bv1 &= &bv2;
        assert!(!bv1.get(1));
        assert!(!bv1.get(5));
        assert!(bv1.get(10));
        assert!(bv1.get(15));

        bv1 ^= &bv2;
        assert!(!bv1.get(1));
        assert!(!bv1.get(5));
        assert!(!bv1.get(10));
        assert!(!bv1.get(15));
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