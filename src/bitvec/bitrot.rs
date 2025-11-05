use crate::bitvec::BitVector;

impl BitVector {
    pub fn rotr_mut(&mut self, shift: usize) -> &mut Self {
        let num_bits = self.len();
        if num_bits == 0 { 
            return self; 
        }

        // normalize shift
        let k = shift % num_bits;
        if k == 0 { 
            return self; 
        }

        let num_words = self.as_words().len();
        let word_shift = k >> 6;
        let bit_shift = (k & 0x3F) as u32;

        // perform whole-word rotation in-place (no full clone)
        // CORRECTION: rotate left by word_shift for a right-rotate of bits
        if word_shift != 0 {
            self.as_words_mut().rotate_left(word_shift);
        }

        // if we only rotated whole words, we're done
        if bit_shift == 0 {
            return self;
        }

        // do the intra-word rotate using a single u64 temp (save first element for wrap)
        let inv = 64 - bit_shift;
        let first = self.as_words()[0];
        for i in 0..num_words {
            let curr = self.as_words()[i];
            let next = if i + 1 < num_words { self.as_words()[i + 1] } else { first };
            self.as_words_mut()[i] = (curr >> bit_shift) | (next << inv);
        }

        self
    }

    pub fn rotl_mut(&mut self, shift: usize) -> &mut Self {
        let num_bits = self.len();
        if num_bits == 0 
        { 
            return self; 
        }

        let k = shift % num_bits;
        if k == 0 
        { 
            return self; 
        }

        // rotate-left by k is rotate-right by (num_bits - k)
        let complement = (num_bits - k) % num_bits;
        self.rotr_mut(complement);
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::bitvec::BitVector;

    #[test]
    fn test_bitrotate_mut() {
        let mut bv1 = BitVector::new(64, Some(0));
        bv1.set(5);
        bv1.set(10);
        bv1.set(1);
        bv1.set(63);

        let mut bv_rotr = bv1.clone();
        bv_rotr.rotr_mut(3);
        assert!(bv_rotr.get(2));
        assert!(!bv_rotr.get(5));
        assert!(!bv_rotr.get(63));
        assert!(bv_rotr.get(60));
        assert!(!bv_rotr.get(1));
        assert!(bv_rotr.get(62));
        assert!(bv_rotr.count_ones() == 4);

        let mut bv_rotl = bv1.clone();
        bv_rotl.rotl_mut(3);
        assert!(bv_rotl.get(8));
        assert!(!bv_rotl.get(5));
        assert!(!bv_rotl.get(63));
        assert!(bv_rotl.get(2));
        assert!(!bv_rotl.get(1));
        assert!(bv_rotl.get(4));
        assert!(bv_rotl.count_ones() == 4);
    }

    #[test]
    fn test_bitrotate_large()
    {
        let mut bv = BitVector::new(256, Some(0));
        bv.set(0);
        bv.set(128);
        bv.set(255);
        //bv.set(9995);

        //bv.write_raw(128).unwrap();
        bv.rotl_mut(130);
        //bv.write_raw(128).unwrap();
        assert!(!bv.get(0));
        assert!(bv.get(130)); // 0 + 130 = 130
        assert!(bv.get(2)); // 128 + 130 - 256 = 2
        assert!(bv.get(129)); // 255 + 130 - 256 = 129
        //assert!(bv.get(78)); // 9995 + 130 - 10048 = 77
        assert!(!bv.get(128));
        assert!(!bv.get(255));

        bv.rotr_mut(130); //Undoes left rotate
        assert!(bv.get(0));
        assert!(bv.get(128));
        assert!(bv.get(255));
        //assert!(bv.get(9995)); 
    }

    #[test]
    #[ignore="Performance test, not functional"]
    fn test_bitrot_exec_time() 
    {
        use std::time::Instant;
        let mut bv = BitVector::new(1_000_000, Some(123456));
        let start = Instant::now();
        for _ in 0..1000 {
            bv.rotl_mut(13);
        }
        let duration = start.elapsed();
        println!("Time taken for 1000x rotl_mut(13) on 1,000,000 bits: {:?}", duration);

        let start = Instant::now();
        for _ in 0..1000 {
            bv.rotl_mut(10_000);
        }
        let duration = start.elapsed();
        println!("Time taken for 1000x rotl_mut(10,000) on 1,000,000 bits: {:?}", duration);
    }
}