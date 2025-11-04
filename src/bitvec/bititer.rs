use crate::bitvec::BitVector;

pub struct MsWordIter<'a> {
    bv: &'a BitVector,
    pos: isize,
}

impl<'a> MsWordIter<'a> {
    pub fn new(bv: &'a BitVector) -> Self {
        Self { bv, pos: bv.word_len() as isize - 1 }
    }
}

impl<'a> Iterator for MsWordIter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < 0 {
            return None;
        }
        let idx = self.pos as usize;
        self.pos -= 1;
        Some(self.bv.word(idx).reverse_bits())
    }
}

pub struct LsWordIter<'a> {
    bv: &'a BitVector,
    pos: isize,
}

impl<'a> LsWordIter<'a> {
    pub fn new(bv: &'a BitVector) -> Self {
        Self { bv, pos: 0 }
    }
}

impl<'a> Iterator for LsWordIter<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.bv.word_len() as isize {
            return None;
        }
        let idx = self.pos as usize;
        self.pos += 1;
        Some(self.bv.word(idx))
    }
}

pub struct MsBitIter<'a> {
    bv: &'a BitVector,
    pos: isize,
}

impl<'a> MsBitIter<'a> {
    pub fn new(bv: &'a BitVector) -> Self {
        let total = bv.len();
        Self { bv, pos: total as isize - 1 }
    }
}

impl<'a> Iterator for MsBitIter<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < 0 {
            return None;
        }
        let idx = self.pos as usize;
        self.pos -= 1;
        Some(self.bv.get(idx))
    }
}

pub struct LsBitIter<'a> {
    bv: &'a BitVector,
    pos: usize,
}

impl<'a> LsBitIter<'a> {
    pub fn new(bv: &'a BitVector) -> Self {
        Self { bv, pos: 0 }
    }
}

impl<'a> Iterator for LsBitIter<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.bv.len() {
            return None;
        }
        let idx = self.pos;
        self.pos += 1;
        Some(self.bv.get(idx))
    }
}

impl BitVector {

    pub fn as_mswords(&self) -> impl Iterator<Item = u64> + '_ {
        MsWordIter::new(self)
    }

    pub fn as_lswords(&self) -> impl Iterator<Item = u64> + '_ {
        LsWordIter::new(self)
    }

    pub fn as_msbits(&self) -> impl Iterator<Item=bool> + '_ {
        MsBitIter::new(self)
    }

    pub fn as_lsbits(&self) -> impl Iterator<Item=bool> + '_ {
        LsBitIter::new(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::bitvec::BitVector;

    #[test]
    fn test_as_mswords() {
        let bv = BitVector::from_words(vec![0x8F0F_F0F0_FFFF_0000, 0xFFFF_FFFF_0000_0000]);
        let collected: Vec<u64> = bv.as_mswords().collect();
        assert_eq!(
            collected,
            vec![0x0000_0000_FFFF_FFFF, 0x0000_FFFF_0F0F_F0F1]
        );
    }

    #[test]
    fn test_as_lswords() {
        let bv = BitVector::from_words(vec![0x8F0F_F0F0_FFFF_0000, 0xFFFF_FFFF_0000_0000]);
        let collected: Vec<u64> = bv.as_lswords().collect();
        assert_eq!(
            collected,
            vec![0x8F0F_F0F0_FFFF_0000, 0xFFFF_FFFF_0000_0000]
        );
    }

    #[test]
    fn test_as_msbits() {
        let mut bv = BitVector::new(128, Some(0));
        bv.set(5);
        bv.set(63);
        bv.set(126);

        let bits: Vec<bool> = bv.as_msbits().collect();
        assert_eq!(bits.len(), 128);
        assert_eq!(bits[1], true);           // index 126
        assert_eq!(bits[64], true);          // index 63
        assert_eq!(bits[122], true);         // index 5
    }

    #[test]
    fn test_as_lsbits() {
        let mut bv = BitVector::new(128, Some(0));
        bv.set(5);
        bv.set(63);
        bv.set(126);

        let bits: Vec<bool> = bv.as_lsbits().collect();
        assert_eq!(bits.len(), 128);
        assert_eq!(bits[5], true);           // index 5
        assert_eq!(bits[63], true);          // index 63
        assert_eq!(bits[126], true);         // index 126
    }
}