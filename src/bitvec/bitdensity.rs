use std::iter::FusedIterator;

use crate::bitvec::BitVector;

#[derive(Debug, PartialEq, Eq)]
pub struct BitDensity {
    pub ones: usize,
    pub bits_checked: usize,
    pub density: u8, // 0-100% [0..255]
}
pub struct LsDensityIter<'a> {
    bv: &'a BitVector,
    pos: usize,
    bits_in_wndw: usize,
    total_bits: usize,
}

impl<'a> LsDensityIter<'a> {
    pub fn new(bv: &'a BitVector, bits_in_wndw: usize) -> Self {
        assert!(bits_in_wndw > 0, "bits_in_wndw must be > 0");
        Self {
            bv,
            pos: 0,
            bits_in_wndw,
            total_bits: bv.bit_len(),
        }
    }
}

impl<'a> Iterator for LsDensityIter<'a> {
    type Item = BitDensity;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.total_bits {
            return None;
        }

        let bits_left = self.total_bits - self.pos;
        let wndw_size = bits_left.min(self.bits_in_wndw);
        let count = self.bv.count_ones_in_range(self.pos, wndw_size);
        self.pos = self.pos.saturating_add(self.bits_in_wndw);

        Some(BitDensity {
            ones: count,
            bits_checked: wndw_size,
            density: ((count << 8) / wndw_size).min(255) as u8,
        })
    }
}

impl<'a> FusedIterator for LsDensityIter<'a> {}

pub struct MsDensityIter<'a> {
    bv: &'a BitVector,
    pos: isize,
    bits_in_wndw: usize,
}

impl<'a> MsDensityIter<'a> {
    pub fn new(bv: &'a BitVector, bits_in_wndw: usize) -> Self {
        assert!(bits_in_wndw > 0, "bits_in_wndw must be > 0");
        Self {
            bv,
            pos: bv.bit_len() as isize,
            bits_in_wndw,
        }
    }
}

impl<'a> Iterator for MsDensityIter<'a> {
    type Item = BitDensity;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos <= 0 {
            return None;
        }

        let start = self.pos - self.bits_in_wndw as isize;
        let wndw_size = if start < 0 { self.bits_in_wndw as isize + start } else { self.bits_in_wndw as isize };

        let start = start.max(0) as usize;
        let count = self.bv.count_ones_in_range(start, wndw_size as usize);
        self.pos = start as isize;

        Some(BitDensity {
            ones: count,
            bits_checked: wndw_size as usize,
            density: ((count << 8) / wndw_size as usize).min(255) as u8,
        })
    }
}

impl<'a> FusedIterator for MsDensityIter<'a> {}

impl BitVector {
    /// Create a density iterator over least-significant bits (left to right, 0..len).
    /// Each item yields the count of set bits in a window of `bits_in_wndw` bits.
    pub fn as_lsdensity(&self, bits_in_wndw: usize) -> impl Iterator<Item = BitDensity> + '_ {
        LsDensityIter::new(self, bits_in_wndw)
    }

    /// Create a density iterator over most-significant bits (right to left, len..0).
    /// Each item yields the count of set bits in a window of `bits_in_wndw` bits.
    pub fn as_msdensity(&self, bits_in_wndw: usize) -> impl Iterator<Item = BitDensity> + '_ {
        MsDensityIter::new(self, bits_in_wndw)
    }
}

#[cfg(test)]
mod tests {
    use crate::bitvec::BitVector;
    use super::BitDensity;

    #[test]
    fn test_ls_density_iter_basic() {
        let bv = BitVector::from_words(vec![u64::MAX, 0]);
        let densities: Vec<BitDensity> = bv.as_lsdensity(16).collect();
        assert_eq!(densities.iter().map(|c| c.ones).collect::<Vec<_>>(), 
                    vec![16,16,16,16,0,0,0,0]);
        assert_eq!(densities.iter().map(|c| c.bits_checked).collect::<Vec<_>>(), 
                    vec![16, 16, 16, 16, 16, 16, 16, 16]);
        assert_eq!(densities.iter().map(|c| c.density).collect::<Vec<_>>(), 
                    vec![255, 255, 255, 255, 0, 0, 0, 0]);
    }

    #[test]
    fn test_ms_density_iter_basic() {
        let bv = BitVector::from_words(vec![u64::MAX, 0]);
        let densities: Vec<BitDensity> = bv.as_msdensity(16).collect();
        assert_eq!(densities.iter().map(|c| c.ones).collect::<Vec<_>>(),
                    vec![0,0,0,0,16,16,16,16]);
        assert_eq!(densities.iter().map(|c| c.bits_checked).collect::<Vec<_>>(), 
                    vec![16, 16, 16, 16, 16, 16, 16, 16]);
        assert_eq!(densities.iter().map(|c| c.density).collect::<Vec<_>>(), 
                    vec![0, 0, 0, 0, 255, 255, 255, 255]);
    }

    #[test]
    fn test_ls_density_iter_partial_window() {
        // 0xFFFF_FFFF_FFFF_0000 has 48 set bits in positions 16-63
        let bv = BitVector::from_words(vec![0xFFFF_FFFF_FFFF_0000]);
        let densities: Vec<BitDensity> = bv.as_lsdensity(10).collect();
        // Window 0: bits 0-9 (all zero) → 0
        // Window 1: bits 10-19 (bits 16-19 set) → 4
        // Window 2: bits 20-29 (all set) → 10
        // Window 3: bits 30-39 (all set) → 10
        // Window 4: bits 40-49 (all set) → 10
        // Window 5: bits 50-59 (all set) → 10
        // Window 6: bits 60-63 (4 bits, all set) → 4
        assert_eq!(densities.iter().map(|c| c.ones).collect::<Vec<_>>(), 
                    vec![0, 4, 10, 10, 10, 10, 4]);
        assert_eq!(densities.iter().map(|c| c.bits_checked).collect::<Vec<_>>(), 
                    vec![10, 10, 10, 10, 10, 10, 4]);
        assert_eq!(densities.iter().map(|c| c.density).collect::<Vec<_>>(), 
                    vec![0, 102, 255, 255, 255, 255, 255]);
    }

    #[test]
    fn test_ms_density_iter_partial_window() {
        let bv = BitVector::from_words(vec![0xFFFF_FFFF_FFFF_0000]);
        let densities: Vec<BitDensity> = bv.as_msdensity(10).collect();
        // Starting from bit 63 going down
        // Window 0: bits 54-63 (all set) → 10
        // Window 1: bits 44-53 (all set) → 10
        // Window 2: bits 34-43 (all set) → 10
        // Window 3: bits 24-33 (all set) → 10
        // Window 4: bits 14-23 (bits 16-23 set) → 8
        // Window 5: bits 4-13 (all zero) → 0
        // Window 6: bits 0-3 (4 bits, all zero) → 0
        assert_eq!(densities.iter().map(|d| d.ones).collect::<Vec<_>>(),
                    vec![10, 10, 10, 10, 8, 0, 0]);
        assert_eq!(densities.iter().map(|d| d.bits_checked).collect::<Vec<_>>(),
                    vec![10, 10, 10, 10, 10, 10, 4]);
        assert_eq!(densities.iter().map(|d| d.density).collect::<Vec<_>>(),
                    vec![255, 255, 255, 255, 204, 0, 0]);
    }
}