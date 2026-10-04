//! Bit-sliced (vertical) saturating counters: one small counter per bit position,
//! stored as `planes` bit vectors where plane `i` holds bit `i` of every counter.
//! Updating every counter under a mask is a word-wide ripple of XOR/AND; halving all
//! counters (exponential decay) is a plane shift; a weighted sum over a pattern is
//! Σ 2^i · popcount(pattern ∧ plane_i). This is the "count" of hyperdimensional
//! bundling, held in bits.

use crate::bitvec::BitVector;

#[derive(Clone)]
pub struct SlicedCounter {
    planes: Vec<BitVector>,
    bits: usize,
}

impl SlicedCounter {
    /// `bits` counters of `planes` bits each (values 0..2^planes − 1), all set to `initial`.
    pub fn new(bits: usize, planes: usize, initial: u32) -> Self {
        assert!(planes > 0 && planes < 32);
        let planes = (0..planes)
            .map(|i| BitVector::new(bits, Some(if initial >> i & 1 == 1 { u64::MAX } else { 0 })))
            .collect();
        Self { planes, bits }
    }

    pub fn max(&self) -> u32 {
        (1u32 << self.planes.len()) - 1
    }

    /// Add 1 to every counter under `mask`, saturating at `max`.
    pub fn increment(&mut self, mask: &BitVector) {
        let mut carry = mask.as_words().to_vec();
        for plane in &mut self.planes {
            for (p, c) in plane.as_words_mut().iter_mut().zip(carry.iter_mut()) {
                let next = *p & *c;
                *p ^= *c;
                *c = next;
            }
        }
        // overflowed counters wrapped to 0: set them back to max
        for plane in &mut self.planes {
            for (p, &c) in plane.as_words_mut().iter_mut().zip(&carry) {
                *p |= c;
            }
        }
    }

    /// Subtract 1 from every counter under `mask`, saturating at 0.
    pub fn decrement(&mut self, mask: &BitVector) {
        let mut borrow = mask.as_words().to_vec();
        for plane in &mut self.planes {
            for (p, b) in plane.as_words_mut().iter_mut().zip(borrow.iter_mut()) {
                let next = !*p & *b;
                *p ^= *b;
                *b = next;
            }
        }
        // underflowed counters wrapped to max: clear them back to 0
        for plane in &mut self.planes {
            for (p, &b) in plane.as_words_mut().iter_mut().zip(&borrow) {
                *p &= !b;
            }
        }
    }

    /// Halve every counter (drop the lowest plane, shift the rest down).
    pub fn halve(&mut self) {
        let top = self.planes.len() - 1;
        for i in 0..top {
            let (lo, hi) = self.planes.split_at_mut(i + 1);
            lo[i] = hi[0].clone();
        }
        self.planes[top] = BitVector::new(self.bits, Some(0));
    }

    /// Σ over the set bits of `pattern` of their counters.
    pub fn sum(&self, pattern: &BitVector) -> u64 {
        self.planes
            .iter()
            .enumerate()
            .map(|(i, plane)| {
                let n: u32 = plane.as_words().iter().zip(pattern.as_words()).map(|(a, b)| (a & b).count_ones()).sum();
                (n as u64) << i
            })
            .sum()
    }

    /// The counter at one position.
    pub fn get(&self, bit: usize) -> u32 {
        let (w, m) = (bit / 64, 1u64 << (bit % 64));
        self.planes.iter().enumerate().map(|(i, p)| ((p.as_words()[w] & m != 0) as u32) << i).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_saturate_and_halve() {
        let mut c = SlicedCounter::new(128, 3, 5);
        let a = BitVector::from_bits(&[1, 70], 128);
        for _ in 0..4 {
            c.increment(&a);
        }
        assert_eq!((c.get(1), c.get(70), c.get(2)), (7, 7, 5)); // saturated at 7
        let b = BitVector::from_bits(&[2], 128);
        for _ in 0..9 {
            c.decrement(&b);
        }
        assert_eq!(c.get(2), 0); // saturated at 0
        assert_eq!(c.sum(&BitVector::from_bits(&[1, 2, 3], 128)), 7 + 0 + 5);
        c.halve();
        assert_eq!((c.get(1), c.get(3)), (3, 2));
    }
}
