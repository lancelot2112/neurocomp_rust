//! Layer 4 (`Layer4`): the cortex's input layer, a learned sparse recoding of the thalamic
//! input, bitwise.
//!
//! In primary sensory cortex, layer 4's spiny stellate cells receive the core thalamus's input
//! and pass it to layer 2/3, recoded: each cell responds to a feature of the input, and lateral
//! inhibition (PV basket cells) lets only the most driven cells fire.
//!
//! - **Cells:** each has `sample` binary synapses on input bits (a random projection to begin
//!   with).
//! - **Firing:** popcount(input & synapses); the `k` most driven cells fire (k-winners-take-all).
//!   The output is a bit vector over the cells.
//! - **Learning** (competitive Hebbian, one synapse per event): each winner moves one synapse
//!   from an input bit that was off to one that was on, with probability 1/4. Cells come to
//!   specialise on recurring features of the input, and different cells on different features,
//!   because only the winners learn.

use crate::bitvec::BitVector;

pub struct Layer4 {
    in_bits: usize,
    cells: usize,
    k: usize,
    /// per cell: its synapses (input bits), sorted
    syn: Vec<Vec<u32>>,
    /// per input bit: the cells with a synapse on it
    index: Vec<Vec<u32>>,
    /// settling plasticity (`set_settling`): per cell, how often it has won
    settle: bool,
    wins: Vec<u32>,
}

impl Layer4 {
    /// `cells` cells over `in_bits` input bits, `sample` synapses each, `k` winners; wired by a
    /// fixed pseudo-random projection from `seed`.
    pub fn new(in_bits: usize, cells: usize, sample: usize, k: usize, seed: u64) -> Self {
        let mut x = seed | 1;
        let mut syn = Vec::with_capacity(cells);
        let mut index = vec![Vec::new(); in_bits];
        for c in 0..cells {
            let mut s: Vec<u32> = Vec::with_capacity(sample);
            while s.len() < sample {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let b = (x % in_bits as u64) as u32;
                if !s.contains(&b) {
                    s.push(b);
                }
            }
            s.sort_unstable();
            for &b in &s {
                index[b as usize].push(c as u32);
            }
            syn.push(s);
        }
        Self { in_bits, cells, k, syn, index, settle: false, wins: vec![0; cells] }
    }

    /// Settling plasticity: a cell's chance to move a synapse halves with each doubling of its
    /// wins past 64 (1/4, then 1/8 after 128 wins, 1/16 after 256, ...), as sensory tuning
    /// settles with experience, so what other areas point to in layer 4 stays put.
    pub fn set_settling(&mut self, on: bool) {
        self.settle = on;
    }

    /// The winners for `input` (at most `k`, most driven first; ties to the lower cell).
    fn winners(&self, input: &BitVector) -> Vec<u32> {
        let mut count = vec![0u16; self.cells];
        let mut touched = Vec::new();
        for (wi, &w) in input.as_words().iter().enumerate() {
            let mut w = w;
            while w != 0 {
                let b = wi * 64 + w.trailing_zeros() as usize;
                if let Some(cs) = self.index.get(b) {
                    for &c in cs {
                        if count[c as usize] == 0 {
                            touched.push(c);
                        }
                        count[c as usize] += 1;
                    }
                }
                w &= w - 1;
            }
        }
        touched.sort_unstable_by(|a, b| count[*b as usize].cmp(&count[*a as usize]).then(a.cmp(b)));
        touched.truncate(self.k);
        touched
    }

    /// The recoded input: the winners' bits.
    pub fn encode(&self, input: &BitVector) -> BitVector {
        let mut out = BitVector::new(self.cells, Some(0));
        for c in self.winners(input) {
            out.bit_set(c as usize);
        }
        out
    }

    /// Encode and learn: each winner moves one synapse toward the input (probability 1/4).
    pub fn encode_learn<R: rand::Rng + ?Sized>(&mut self, input: &BitVector, rng: &mut R) -> BitVector {
        let winners = self.winners(input);
        let mut out = BitVector::new(self.cells, Some(0));
        let on: Vec<u32> = (0..self.in_bits).filter(|&b| input.bit_get(b)).map(|b| b as u32).collect();
        for &c in &winners {
            out.bit_set(c as usize);
            let r = rng.next_u64();
            // the chance to learn: 1/4, or with settling 2^-(2 + log2(wins / 64))
            let k = if self.settle {
                let w = &mut self.wins[c as usize];
                *w = w.saturating_add(1);
                2 + (32 - (*w / 64).leading_zeros()).min(14)
            } else {
                2
            };
            if r & ((1u64 << k) - 1) != 0 || on.is_empty() {
                continue;
            }
            let s = &self.syn[c as usize];
            let off: Vec<usize> = (0..s.len()).filter(|&i| !input.bit_get(s[i] as usize)).collect();
            let gain: Vec<u32> = on.iter().copied().filter(|b| s.binary_search(b).is_err()).collect();
            if off.is_empty() || gain.is_empty() {
                continue;
            }
            let i = off[((r >> 8) as usize) % off.len()];
            let b_new = gain[((r >> 32) as usize) % gain.len()];
            let b_old = self.syn[c as usize][i];
            self.index[b_old as usize].retain(|&x| x != c);
            self.index[b_new as usize].push(c);
            let s = &mut self.syn[c as usize];
            s[i] = b_new;
            s.sort_unstable();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn code(i: usize) -> BitVector {
        BitVector::from_bits(&(0..16).map(|k| (i * 97 + k * 31) % 1024).collect::<Vec<_>>(), 1024)
    }

    fn overlap(a: &BitVector, b: &BitVector) -> u32 {
        a.as_words().iter().zip(b.as_words()).map(|(x, y)| (x & y).count_ones()).sum()
    }

    #[test]
    fn recodes_sparsely_and_stably() {
        let mut l4 = Layer4::new(1024, 1024, 16, 16, 9);
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        for _ in 0..200 {
            for i in 0..8 {
                l4.encode_learn(&code(i), &mut rng);
            }
        }
        let a = l4.encode(&code(1));
        let b = l4.encode(&code(2));
        assert_eq!(a.count_ones(), 16);
        // the same input gives the same code; different inputs mostly different codes
        assert_eq!(overlap(&a, &l4.encode(&code(1))), 16);
        assert!(overlap(&a, &b) < 8, "{}", overlap(&a, &b));
        // learning concentrated the winners' synapses on their inputs
        let w = l4.winners(&code(1));
        let on: u32 = w.iter().map(|&c| l4.syn[c as usize].iter().filter(|&&s| code(1).bit_get(s as usize)).count() as u32).sum();
        assert!(on as usize > w.len() * 4, "{on}");
    }
}
