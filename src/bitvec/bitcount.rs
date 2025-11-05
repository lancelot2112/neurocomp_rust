use crate::bitvec::BitVector;

impl BitVector {
        /// Count number of set bits in the whole vector.
    #[inline]
    pub fn count_ones(&self) -> usize {
        // fast: use hardware popcount via count_ones on u64
        self.words().map(|w| w.count_ones() as usize).sum()
    }

    /// Count number of set bits in the range [start, start+count).
    /// `start + count` must be <= self.len().
    pub fn count_ones_in_range(&self, start: usize, count: usize) -> usize {
        if count == 0 {
            return 0;
        }

        assert!(start < self.bit_len(), "start out of bounds");
        /*
        assert!(
            start.checked_add(count).map_or(false, |end| end <= self.len()),
            "range out of bounds"
        );
        */

        

        let mut remaining = count.min(self.bit_len() - start);
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

            let v = self.as_words()[word_idx] & mask;
            sum += v.count_ones() as usize;

            pos += take;
            remaining -= take;
        }

        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitcount() {
        let mut bv = BitVector::new(128, Some(0));
        assert_eq!(bv.count_ones(), 0);
        bv.bit_set(0);
        bv.bit_set(63);
        bv.bit_set(64);
        bv.bit_set(127);
        assert_eq!(bv.count_ones(), 4);
        assert_eq!(bv.count_ones_in_range(0, 64), 2);
        assert_eq!(bv.count_ones_in_range(64, 64), 2);
        assert_eq!(bv.count_ones_in_range(32, 64), 2);
    }
}