use crate::bitvec::BitVector;

impl BitVector {
    pub fn mask<F>(&self, idx: usize, mask: &BitVector, mut op: F) -> BitVector
    where
        F: FnMut(u64, u64) -> u64,
    {
        let words = self.as_slice(idx, mask.word_len()).iter().zip(mask.as_words().iter())
            .map(|(&a, &m)| op(a, m))
            .collect();
        BitVector::from_words(words)
    }

    pub fn mask_and_count<F>(&self, idx: usize, mask: &BitVector, mut op: F) -> usize
    where
        F: FnMut(u64, u64) -> u64,
    {
        self.as_slice(idx, mask.word_len()).iter().zip(mask.words())
            .map(|(&a, &m)| op(a, m).count_ones() as usize)
            .sum()
    }

    pub fn mask_move_random_connected_and_set<R: rand::Rng + ?Sized>(&self, idx: usize, mask: &mut BitVector, rng: &mut R) {
        // Find a bit that is CONNECTED (in mask) + SET in self
        // clear that bit in the mask then set a random bit that is
        // UNCONNECTED (not in mask) + NOT SET in self.
        self.mask_move_random_internal(
            idx,
            mask,
            rng,
            |a, m, _valid| a & m,
            |a, m, valid| (!a) & (!m) & valid,
        );
    }

    pub fn mask_move_random_connected_and_not_set<R: rand::Rng + ?Sized>(&self, idx: usize, mask: &mut BitVector, rng: &mut R) {
        // Find a bit that is CONNECTED (in mask) + NOT SET in self 
        // clear that bit in the mask then set a random bit that is 
        // UNCONNECTED (not in mask) + SET in self.
        self.mask_move_random_internal(
            idx,
            mask,
            rng,
            |a, m, valid| (!a) & m & valid,
            |a, m, _valid| a & (!m),
        );
    }

    /// Shared fast path: pick a random source bit from src_candidates(a,m,valid),
    /// and a random destination bit from dst_candidates(a,m,valid), then clear/set in mask.
    fn mask_move_random_internal<R, FS, FD>(
        &self,
        idx: usize,
        mask: &mut BitVector,
        rng: &mut R,
        mut src_candidates: FS,
        mut dst_candidates: FD,
    )
    where
        R: rand::Rng + ?Sized,
        FS: FnMut(u64, u64, u64) -> u64,
        FD: FnMut(u64, u64, u64) -> u64,
    {
        let nwords = mask.word_len();
        if nwords == 0 {
            return;
        }

        let a_words = self.as_slice(idx, nwords);
        let m_words = mask.as_words();

        // Valid bits in last word
        let tail_bits = mask.bit_len() & 63;
        let last_valid = if tail_bits == 0 { u64::MAX } else { (1u64 << tail_bits) - 1 };

        // Pass 1: count candidates
        let mut src_total = 0usize;
        let mut dst_total = 0usize;
        for wi in 0..nwords {
            let a = a_words[wi];
            let m = m_words[wi];
            let valid = if wi + 1 == nwords { last_valid } else { u64::MAX };
            src_total += src_candidates(a, m, valid).count_ones() as usize;
            dst_total += dst_candidates(a, m, valid).count_ones() as usize;
        }
        if src_total == 0 || dst_total == 0 {
            return;
        }

        // Choose random ranks
        let mut k_src = rng.gen_range(0..src_total);
        let mut k_dst = rng.gen_range(0..dst_total);

        // Pass 2: locate k-th bits
        let (mut src_wi, mut src_bi) = (0usize, 0usize);
        let (mut dst_wi, mut dst_bi) = (0usize, 0usize);

        for wi in 0..nwords {
            let a = a_words[wi];
            let m = m_words[wi];
            let valid = if wi + 1 == nwords { last_valid } else { u64::MAX };

            let w_src = src_candidates(a, m, valid);
            let c_src = w_src.count_ones() as usize;
            if c_src != 0 {
                if k_src < c_src {
                    src_bi = select_nth_bit(w_src, k_src);
                    src_wi = wi;
                } else {
                    k_src -= c_src;
                }
            }

            let w_dst = dst_candidates(a, m, valid);
            let c_dst = w_dst.count_ones() as usize;
            if c_dst != 0 {
                if k_dst < c_dst {
                    dst_bi = select_nth_bit(w_dst, k_dst);
                    dst_wi = wi;
                } else {
                    k_dst -= c_dst;
                }
            }
        }

        // Flip bits in mask
        mask.as_words_mut()[src_wi] &= !(1u64 << src_bi);
        mask.as_words_mut()[dst_wi] |= 1u64 << dst_bi;
    }

    /// Apply a word-aligned mask slice at `idx` using `op` per word.
    /// `idx` is a word index (bit_offset >> 6).
    pub fn mask_mut<F>(&mut self, idx: usize, mask: &BitVector, mut op: F)
    where
        F: FnMut(u64, u64) -> u64,
    {
        for (a, &m) in self.as_slice_mut(idx,mask.word_len()).iter_mut().zip(mask.words()) {
            *a = op(*a, m);
        }
    }

    /// Convenience bitwise assign helpers (word aligned).
    pub fn mask_mut_and(&mut self, dest_word_idx: usize, mask: &BitVector) {
        self.mask_mut(dest_word_idx, mask, |a, b| a & b);
    }
    pub fn mask_mut_or(&mut self, dest_word_idx: usize, mask: &BitVector) {
        self.mask_mut(dest_word_idx, mask, |a, b| a | b);
    }
    pub fn mask_mut_xor(&mut self, dest_word_idx: usize, mask: &BitVector) {
        self.mask_mut(dest_word_idx, mask, |a, b| a ^ b);
    }
    pub fn mask_mut_clear(&mut self, dest_word_idx: usize, mask: &BitVector) {
        self.mask_mut(dest_word_idx, mask, |a, b| a & !b);
    }
}

// Helper: return the position (0..63) of the n-th set bit in w (0-based).
#[inline]
fn select_nth_bit(mut w: u64, mut n: usize) -> usize {
    loop {
        let tz = w.trailing_zeros() as usize;
        if n == 0 {
            return tz;
        }
        n -= 1;
        // clear lowest set bit
        w &= w - 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_mask_and_count_matches_mask_and_count_ones() {
        // 128 bits (2 words) to avoid tail-bit complications here
        let a = BitVector::from_words(vec![
            0xF0F0_F0F0_F0F0_F0F0,
            0x00FF_00FF_00FF_00FF,
        ]);
        let m = BitVector::from_words(vec![
            0xFFFF_0000_FFFF_0000,
            0x0F0F_0F0F_F0F0_F0F0,
        ]);

        // a & m via mask()
        let and_bv = a.mask(0, &m, |x, y| x & y);
        let and_cnt = and_bv.count_ones();

        // a & m via mask_and_count()
        let and_cnt_fast = a.mask_and_count(0, &m, |x, y| x & y);
        assert_eq!(and_cnt_fast, and_cnt);

        // a ^ m
        let xor_bv = a.mask(0, &m, |x, y| x ^ y);
        let xor_cnt = xor_bv.count_ones();
        let xor_cnt_fast = a.mask_and_count(0, &m, |x, y| x ^ y);
        assert_eq!(xor_cnt_fast, xor_cnt);

        // a & !m
        let clear_bv = a.mask(0, &m, |x, y| x & !y);
        let clear_cnt = clear_bv.count_ones();
        let clear_cnt_fast = a.mask_and_count(0, &m, |x, y| x & !y);
        assert_eq!(clear_cnt_fast, clear_cnt);
    }

    #[test]
    fn test_move_connected_and_set_noop_when_no_candidates() {
        // No connected+set candidates if a = 0 (a & m == 0)
        let a = BitVector::from_words(vec![0, 0]);
        let mut m = BitVector::from_words(vec![0xFFFF_FFFF_FFFF_FFFF, 0x1234_5678_9ABC_DEF0]);
        let before = m.clone();

        let mut rng = StdRng::seed_from_u64(1);
        a.mask_move_random_connected_and_set(0, &mut m, &mut rng);

        assert_eq!(m.as_words(), before.as_words(), "mask should be unchanged");
    }

    #[test]
    fn test_move_connected_and_set_flips_two_bits_and_preserves_count() {
        // Ensure both src (a&m) and dst ((!a)&(!m)) candidates exist
        let a = BitVector::from_words(vec![
            0x0000_0000_0000_00F0, // bits 4..7 set
            0x0000_0000_0000_0F00, // bits 8..11 set
        ]);
        let mut m = BitVector::from_words(vec![
            0x0000_0000_0000_00F3, // overlap with a on 4..7 plus two LSBs
            0x0000_0000_0000_F000, // no overlap here (a has 8..11, m has 12..15)
        ]);

        let ones_before = m.count_ones();
        let overlap_before = a.mask_and_count(0, &m, |x, y| x & y);
        let m_before = m.clone();

        let mut rng = StdRng::seed_from_u64(42);
        a.mask_move_random_connected_and_set(0, &mut m, &mut rng);

        // Popcount preserved
        assert_eq!(m.count_ones(), ones_before, "mask ones should be preserved");

        // Exactly two bits flipped
        let delta = m.mask(0, &m_before, |x, y| x ^ y);
        assert_eq!(delta.count_ones(), 2, "exactly two bits should change");

        // Overlap with 'a' decreases by 1 (moved from a&m to !a&!m)
        let overlap_after = a.mask_and_count(0, &m, |x, y| x & y);
        assert_eq!(overlap_after + 1, overlap_before, "overlap with a should decrease by 1");
    }

    #[test]
    fn test_move_connected_and_not_set_flips_two_bits_and_preserves_count() {
        // Ensure src ((!a)&m) and dst (a&!m) both have candidates
        let a = BitVector::from_words(vec![
            0x0000_0000_0000_0F00, // bits 8..11 set
            0x0000_0000_0000_000F, // bits 0..3 set
        ]);
        let mut m = BitVector::from_words(vec![
            0x0000_0000_0000_F0F0, // has some 1s where a has 0s at 4..7,12..15
            0x0000_0000_0000_00F0, // bit 4..7 set (a has 0..3 set -> destination exists)
        ]);

        let ones_before = m.count_ones();
        let overlap_a_before = a.mask_and_count(0, &m, |x, y| x & y);
        // count of (!a) & m
        let not_a_and_m_before = a.mask_and_count(0, &m, |x, y| (!x) & y);

        let m_before = m.clone();
        let mut rng = StdRng::seed_from_u64(99);
        a.mask_move_random_connected_and_not_set(0, &mut m, &mut rng);

        // Popcount preserved
        assert_eq!(m.count_ones(), ones_before, "mask ones should be preserved");

        // Exactly two bits flipped
        let delta = m.mask(0, &m_before, |x, y| x ^ y);
        assert_eq!(delta.count_ones(), 2, "exactly two bits should change");

        // Overlap with 'a' increases by 1 (moved from !a&m to a&!m)
        let overlap_a_after = a.mask_and_count(0, &m, |x, y| x & y);
        assert_eq!(overlap_a_after, overlap_a_before + 1);

        // And (!a)&m decreases by 1
        let not_a_and_m_after = a.mask_and_count(0, &m, |x, y| (!x) & y);
        assert_eq!(not_a_and_m_after + 1, not_a_and_m_before);
    }

    #[test]
    fn test_mask_mut() {
        let mut bv = BitVector::new(128, Some(0));

        assert!(bv.count_ones() == 0);
        let mask = BitVector::from_words(vec![0xFFFFFFFFFFFFFFFF; 2]);
        bv.mask_mut(0, &mask, |a, b| a | b);

        assert!(bv.bit_get(5));
        assert!(bv.bit_get(70));
        assert!(bv.count_ones() == 128);

        let mask = BitVector::from_words(vec![0x0F0F0F0F0F0F0F0F; 2]);
        bv.mask_mut(0, &mask, |a, b| a ^ b);

        assert!(!bv.bit_get(0));
        assert!(bv.bit_get(4));
        assert!(bv.count_ones() == 64);
    }
}