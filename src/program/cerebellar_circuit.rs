//! The cerebellar circuit, bitwise (`CerebellarCircuit`): the fast, error-driven learner as
//! the cerebellum builds it, not as a cortical kernel class.
//!
//! - **Mossy fibres** carry a copy of the cortex's input (cortex → pontine nuclei → mossy
//!   fibres), here the column's input row.
//! - **Granule cells:** a fixed random expansion, not learned. Each granule cell has four
//!   dendrites, each on a random mossy fibre (as real granule cells have about four), and fires
//!   when at least `k` of them are active. Golgi cells adjust `k` by feedback inhibition so that
//!   about `target` of the granule layer is active: the expansion stays sparse whatever the
//!   input density (Marr 1969; Albus 1971).
//! - **Purkinje cells:** one per output bit. Every parallel fibre (granule axon) starts with a
//!   potent synapse on every Purkinje cell. A Purkinje cell's drive is the number of active
//!   granule cells whose synapse on it is still potent.
//! - **Deep cerebellar nuclei:** Purkinje cells inhibit them. An output bit fires where its
//!   Purkinje cell has gone quiet for this granule pattern: fewer than a quarter of the active
//!   granule cells still drive it.
//! - **Learning by the climbing fibre** (one per output bit, from the inferior olive, carrying
//!   the error):
//!   - an output bit that should have fired and did not: **LTD**, the synapses from the active
//!     granule cells onto that Purkinje cell are depressed (one shot: a bit cleared);
//!   - an output bit that fired and should not have: **LTP** restores each depressed synapse
//!     from an active granule cell with probability 1/4 (slower).
//!
//! Depression is the main plasticity (Ito), so the circuit stores what it has learned as the
//! depressed synapses: per granule cell, the set of Purkinje cells it no longer drives (sparse,
//! allocated on first use).

use crate::bitvec::BitVector;
use crate::fixed::{Q16, ONE};

pub struct CerebellarCircuit {
    out_bits: usize,
    out_words: usize,
    /// per granule cell: its four mossy fibres (row bits)
    dendrites: Vec<[u32; 4]>,
    /// per mossy fibre (row bit): the granule cells it contacts
    fanout: Vec<Vec<u32>>,
    row_bits: usize,
    /// Golgi inhibition: a granule cell fires with at least `k` active dendrites
    k: u32,
    target: Q16,
    /// running share of granule cells active (`Q16`)
    density: Q16,
    /// per granule cell: depressed synapses, as a bitset over Purkinje cells (empty: none)
    ltd: Vec<Vec<u64>>,
    prediction: BitVector,
    confidence: Q16,
    seed: u64,
    /// LTP: a wrongly depressed synapse is restored with probability 2^-ltp_k
    ltp_k: u32,
    /// a Purkinje cell is quiet (its output bit fires) when fewer than 1/quiet of the active
    /// granule cells still drive it; 0: relative readout (see `set_rates`)
    quiet: u32,
    /// running mean size of the targets (bits), for the relative readout
    target_bits: u32,
    /// each granule cell's dendrites on different sources (frames of `out_bits` bits)
    cross_frames: bool,
}

impl CerebellarCircuit {
    /// `granules` granule cells, `out_bits` Purkinje cells; about `target` (`Q16`) of the
    /// granule layer active.
    pub fn new(granules: usize, out_bits: usize, target: Q16, seed: u64) -> Self {
        Self {
            out_bits,
            out_words: out_bits.div_ceil(64),
            dendrites: vec![[0; 4]; granules],
            fanout: Vec::new(),
            row_bits: 0,
            k: 2,
            target,
            density: target,
            ltd: vec![Vec::new(); granules],
            prediction: BitVector::new(out_bits, Some(0)),
            confidence: 0,
            seed,
            ltp_k: 2,
            quiet: 4,
            target_bits: 0,
            cross_frames: false,
        }
    }

    /// LTP rate (2^-k per wrongly depressed synapse; 0: always) and the quiet threshold
    /// (fewer than 1/quiet of the active granule cells still driving a Purkinje cell).
    /// `quiet` 0: **relative readout**: the deep nuclei compare Purkinje cells with each other,
    /// and the output bits are the most depressed ones, as many as a target usually has (a
    /// running mean of the targets learned from), among those depressed above the median.
    /// Granule cells depressed for every output (common inputs) then cancel out.
    pub fn set_rates(&mut self, ltp_k: u32, quiet: u32) {
        self.ltp_k = ltp_k;
        self.quiet = quiet;
    }

    /// Granule cells integrate different sources: each cell's four dendrites go to four
    /// different frames of the row (frames of `out_bits` bits: the input, its context frames,
    /// the previous input), as a granule cell's mossy fibres come from different origins.
    /// Rows with fewer frames than dendrites reuse frames in turn.
    pub fn set_cross_frames(&mut self, on: bool) {
        self.cross_frames = on;
        self.row_bits = 0; // rewire at the next input
    }

    /// Wire the granule layer to a row of `bits` mossy fibres (once, at the first input).
    fn wire(&mut self, bits: usize) {
        if self.row_bits == bits {
            return;
        }
        self.row_bits = bits;
        self.fanout = vec![Vec::new(); bits];
        let frames = (bits / self.out_bits.max(1)).max(1);
        let mut x = self.seed | 1;
        for g in 0..self.dendrites.len() {
            for d in 0..4 {
                // xorshift: a fixed random wiring
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let b = if self.cross_frames && frames > 1 {
                    let f = (g + d) % frames;
                    (f * self.out_bits + (x % self.out_bits as u64) as usize) as u32
                } else {
                    (x % bits as u64) as u32
                };
                self.dendrites[g][d] = b;
                self.fanout[b as usize].push(g as u32);
            }
        }
    }

    /// The active granule cells for `input`.
    fn granules(&self, input: &BitVector) -> Vec<u32> {
        // event-driven: each active mossy fibre reaches only the granule cells it contacts
        let mut count = vec![0u8; self.dendrites.len()];
        let mut active = Vec::new();
        for (wi, &w) in input.as_words().iter().enumerate() {
            let mut w = w;
            while w != 0 {
                let b = wi * 64 + w.trailing_zeros() as usize;
                if let Some(gs) = self.fanout.get(b) {
                    for &g in gs {
                        let c = &mut count[g as usize];
                        *c += 1;
                        if *c as u32 == self.k {
                            active.push(g);
                        }
                    }
                }
                w &= w - 1;
            }
        }
        active.sort_unstable();
        active
    }

    /// Deep-nuclei output for the active granule cells: (output, confidence).
    fn output(&self, active: &[u32]) -> (BitVector, Q16) {
        let mut out = BitVector::new(self.out_bits, Some(0));
        let n = active.len() as u32;
        if n == 0 {
            return (out, 0);
        }
        // depressions per Purkinje cell: the active granule cells' depressed-synapse bitsets
        // summed by bit-sliced counters (64 Purkinje cells per word operation)
        let planes_n = (32 - n.leading_zeros()) as usize + 1;
        let words = self.out_words;
        let mut planes = vec![vec![0u64; words]; planes_n];
        let mut carry = vec![0u64; words];
        for &g in active {
            let l = &self.ltd[g as usize];
            if l.is_empty() {
                continue;
            }
            carry.copy_from_slice(l);
            for plane in planes.iter_mut() {
                let mut any = 0;
                for (p, c) in plane.iter_mut().zip(carry.iter_mut()) {
                    let t = *p & *c;
                    *p ^= *c;
                    *c = t;
                    any |= t;
                }
                if any == 0 {
                    break;
                }
            }
        }
        let dep: Vec<u32> = (0..self.out_bits).map(|j| (0..planes_n).map(|i| ((planes[i][j / 64] >> (j % 64)) & 1) as u32 * (1 << i)).sum()).collect();
        if self.quiet == 0 {
            // relative readout: the `target_bits` most depressed Purkinje cells, above the median
            let mut sorted = dep.clone();
            sorted.sort_unstable_by(|a, b| b.cmp(a));
            let t = (self.target_bits.max(1) as usize).min(sorted.len());
            let cut = sorted[t - 1];
            let median = sorted[sorted.len() / 2];
            if cut <= median {
                return (out, 0);
            }
            for (j, &d) in dep.iter().enumerate() {
                if d >= cut {
                    out.bit_set(j);
                }
            }
            // confidence: the cut's margin over the median, as a share of the active cells
            return (out, (((cut - median) as u64) << 16).checked_div(n as u64).unwrap_or(0).min(ONE as u64) as Q16);
        }
        // a Purkinje cell is quiet when fewer than a quarter of the active granule cells
        // still drive it; confidence: how quiet, over the bits that fire
        let (mut fired, mut quiet) = (0u64, 0u64);
        for (j, &d) in dep.iter().enumerate() {
            let drive = n - d;
            if drive * self.quiet < n {
                out.bit_set(j);
                fired += 1;
                quiet += ((d as u64) << 16) / n as u64;
            }
        }
        let conf = if fired == 0 { 0 } else { (quiet / fired) as Q16 };
        (out, conf)
    }

    /// Predict the next input from the mossy-fibre input (and adjust Golgi inhibition).
    pub fn predict(&mut self, input: &BitVector) -> &BitVector {
        self.wire(input.bit_len());
        let active = self.granules(input);
        // Golgi feedback inhibition: keep the granule layer near its target density
        let share = ((active.len() as u64) << 16) / self.dendrites.len().max(1) as u64;
        self.density = ((self.density as u64 * 15 + share) / 16) as Q16;
        if self.density > self.target * 2 && self.k < 4 {
            self.k += 1;
            self.density = self.target;
        } else if self.density < self.target / 2 && self.k > 1 {
            self.k -= 1;
            self.density = self.target;
        }
        let (out, conf) = self.output(&active);
        self.prediction = out;
        self.confidence = conf;
        &self.prediction
    }

    /// The output for `input` without any state change.
    pub fn peek(&self, input: &BitVector) -> BitVector {
        if self.row_bits != input.bit_len() {
            return BitVector::new(self.out_bits, Some(0));
        }
        self.output(&self.granules(input)).0
    }

    /// The climbing fibre: learn from what actually came next.
    pub fn learn<R: rand::Rng + ?Sized>(&mut self, input: &BitVector, target: &BitVector, rng: &mut R) {
        self.wire(input.bit_len());
        let active = self.granules(input);
        if active.is_empty() {
            return;
        }
        let tb = target.count_ones() as u32;
        self.target_bits = if self.target_bits == 0 { tb } else { (self.target_bits * 15 + tb) / 16 };
        let (out, _) = self.output(&active);
        let words = self.out_words;
        for wi in 0..words {
            let t = target.as_words().get(wi).copied().unwrap_or(0);
            let o = out.as_words().get(wi).copied().unwrap_or(0);
            let missed = t & !o; // should have fired: LTD
            let wrong = o & !t; // fired wrongly: LTP (slower)
            if missed == 0 && wrong == 0 {
                continue;
            }
            for &g in &active {
                let l = &mut self.ltd[g as usize];
                if l.is_empty() {
                    if missed == 0 {
                        continue;
                    }
                    l.resize(words, 0);
                }
                l[wi] |= missed;
                if wrong != 0 {
                    // restore each depressed synapse with probability 1/4
                    let mut r = u64::MAX;
                    for _ in 0..self.ltp_k {
                        r &= rng.next_u64();
                    }
                    l[wi] &= !(wrong & r);
                }
            }
        }
    }

    pub fn prediction(&self) -> &BitVector {
        &self.prediction
    }

    pub fn confidence(&self) -> Q16 {
        self.confidence
    }

    /// Granule cells carrying at least one depressed synapse.
    pub fn live(&self) -> usize {
        self.ltd.iter().filter(|l| l.iter().any(|&w| w != 0)).count()
    }

    /// The running share of granule cells active (`Q16`).
    pub fn density(&self) -> Q16 {
        self.density
    }

    /// The Golgi threshold (active dendrites a granule cell needs).
    pub fn golgi_k(&self) -> u32 {
        self.k
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn code(i: usize, bits: usize) -> BitVector {
        BitVector::from_bits(&(0..16).map(|k| (i * 97 + k * 31) % bits).collect::<Vec<_>>(), bits)
    }

    fn overlap(a: &BitVector, b: &BitVector) -> u32 {
        a.as_words().iter().zip(b.as_words()).map(|(x, y)| (x & y).count_ones()).sum()
    }

    #[test]
    fn learns_a_sequence_from_its_errors() {
        let bits = 1024;
        let mut cb = CerebellarCircuit::new(8192, bits, ONE / 64, 7);
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        // a → b, c → d: a few exposures each
        for _ in 0..3 {
            for (x, y) in [(1, 2), (3, 4)] {
                cb.predict(&code(x, bits));
                cb.learn(&code(x, bits), &code(y, bits), &mut rng);
            }
        }
        let p = cb.predict(&code(1, bits)).clone();
        assert!(overlap(&p, &code(2, bits)) >= 12, "{}", overlap(&p, &code(2, bits)));
        assert!(overlap(&p, &code(4, bits)) <= 4);
        let p = cb.predict(&code(3, bits)).clone();
        assert!(overlap(&p, &code(4, bits)) >= 12);
    }

    #[test]
    fn relative_readout_learns_a_sequence() {
        let bits = 1024;
        let mut cb = CerebellarCircuit::new(8192, bits, ONE / 64, 7);
        cb.set_rates(2, 0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        for _ in 0..3 {
            for (x, y) in [(1, 2), (3, 4)] {
                cb.predict(&code(x, bits));
                cb.learn(&code(x, bits), &code(y, bits), &mut rng);
            }
        }
        let p = cb.predict(&code(1, bits)).clone();
        assert!(overlap(&p, &code(2, bits)) >= 12, "{}", overlap(&p, &code(2, bits)));
        let p = cb.predict(&code(3, bits)).clone();
        assert!(overlap(&p, &code(4, bits)) >= 12);
    }

    #[test]
    fn golgi_keeps_the_granule_layer_sparse() {
        let bits = 1024;
        let mut cb = CerebellarCircuit::new(8192, bits, ONE / 64, 7);
        let dense = BitVector::from_bits(&(0..300).map(|k| k * 3).collect::<Vec<_>>(), bits);
        for _ in 0..200 {
            cb.predict(&dense);
        }
        let active = cb.granules(&dense).len();
        assert!(active < 8192 / 16, "{active} granule cells active");
    }
}
