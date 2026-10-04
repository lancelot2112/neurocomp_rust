//! Dentate gyrus and CA3: pattern separation by sparse expansion, and a
//! Hebbian autoassociative store with recurrent pattern completion.
//!
//! Follows the classic hippocampal memory model (Marr 1971; Treves & Rolls
//! 1994). An episode arrives as an entorhinal (EC) pattern `x`:
//! - **Dentate gyrus** (`DentateGyrus`): a fixed random expansion. Each of many
//!   granule cells samples `fan_in` random EC bits; the `k` cells with the most
//!   active inputs win (k-winner-take-all). The code is sparse and conjunctive,
//!   so episodes that share most of their words still get largely different codes
//!   (pattern separation).
//! - **CA3** (`Ca3Memory`): storing links EC→CA3 (`x`→`c`), CA3→CA3 (`c`↔`c`,
//!   recurrent collaterals) and CA3→EC (`c`→`x`) with Hebbian weights. Recall
//!   drives CA3 from a partial EC cue directly (perforant path; the dentate gyrus is used for
//!   storage, not retrieval), lets the recurrent weights settle the CA3 code for a
//!   few steps (pattern completion, a self-reference loop), and reads EC back out.
//!   Weights are bit-sliced counters (`SlicedCounter`, one row per source cell) and
//!   decay by halving, a plane shift, every few stores (a palimpsest), so recent
//!   episodes dominate recall. All arithmetic is on integers and bit planes.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::bitvec::{BitVector, SlicedCounter};

pub struct DentateGyrus {
    pub cells: usize,
    pub k: usize,
    inputs_to_cells: Vec<Vec<u32>>, // EC bit -> granule cells sampling it
}

impl DentateGyrus {
    pub fn new(input_bits: usize, cells: usize, fan_in: usize, k: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut inputs_to_cells = vec![Vec::new(); input_bits];
        let all: Vec<usize> = (0..input_bits).collect();
        for cell in 0..cells {
            for &b in all.choose_multiple(&mut rng, fan_in.min(input_bits)) {
                inputs_to_cells[b].push(cell as u32);
            }
        }
        Self { cells, k, inputs_to_cells }
    }

    /// The sparse code for active input bits `x`: the `k` granule cells with the
    /// most active inputs (ties broken by cell index).
    pub fn separate(&self, x: &[usize]) -> Vec<u32> {
        let mut drive = vec![0u16; self.cells];
        for &b in x {
            if let Some(cells) = self.inputs_to_cells.get(b) {
                for &c in cells {
                    drive[c as usize] += 1;
                }
            }
        }
        top_k(drive.iter().map(|&d| d as f32), self.k)
    }
}

/// One pathway's weights in bits: for each source cell, a row of bit-sliced counters
/// (one small counter per target), allocated when first written. Decay is a plane
/// shift (halving) every `half_life` stores, applied lazily: each row remembers the
/// epoch it was last normalized, and reads skip the planes that would have been shifted
/// out (floor(v / 2^s) = Σ_{p ≥ s} bit_p · 2^(p − s)).
struct Pathway {
    rows: Vec<Option<(SlicedCounter, u32)>>,
    targets: usize,
}

impl Pathway {
    fn new(sources: usize, targets: usize) -> Self {
        Self { rows: (0..sources).map(|_| None).collect(), targets }
    }

    /// Add `amount` to row `i`'s counters under `mask` (after bringing the row up to `epoch`).
    fn strengthen(&mut self, i: usize, mask: &BitVector, amount: u32, planes: usize, epoch: u32) {
        let targets = self.targets;
        let (row, at) = self.rows[i].get_or_insert_with(|| (SlicedCounter::new(targets, planes, 0), epoch));
        row.shift_down((epoch - *at) as usize);
        *at = epoch;
        for p in 0..planes {
            if amount >> p & 1 == 1 {
                row.add_power(mask, p);
            }
        }
    }

    /// Summed (decayed) weights from the active `sources` onto every target.
    fn drive(&self, sources: &[usize], epoch: u32) -> Vec<u32> {
        let mut out = vec![0u32; self.targets];
        for &i in sources {
            let Some(Some((row, at))) = self.rows.get(i) else { continue };
            let shift = (epoch - at) as usize;
            for p in shift..row.planes() {
                let w = 1u32 << (p - shift);
                for (wi, &word) in row.plane(p).as_words().iter().enumerate() {
                    let mut word = word;
                    while word != 0 {
                        out[wi * 64 + word.trailing_zeros() as usize] += w;
                        word &= word - 1;
                    }
                }
            }
        }
        out
    }

    /// Number of nonzero counters (stored synapses), as of their last normalization.
    fn synapses(&self) -> usize {
        self.rows
            .iter()
            .flatten()
            .map(|(row, _)| {
                let mut any = BitVector::new(self.targets, Some(0));
                for p in 0..row.planes() {
                    any.or_mut(row.plane(p));
                }
                any.count_ones()
            })
            .sum()
    }
}

pub struct Ca3Memory {
    pub k: usize,
    /// Stores per halving of every weight (decay ≈ 0.5^(1/half_life) per store).
    pub half_life: u32,
    pub settle_steps: usize,
    /// Readout keeps EC bits scoring at least this fraction of the best score
    /// (higher = sharper clean-up toward the single strongest memory).
    pub readout_fraction: f32,
    planes: usize,
    ec_to_ca3: Pathway,
    ca3_to_ca3: Pathway,
    ca3_to_ec: Pathway,
    stores: u32,
}

impl Ca3Memory {
    /// `decay` per stored episode is approximated by halving every
    /// round(ln 0.5 / ln decay) stores. Weights are 7-plane bit-sliced counters
    /// (0..127). Within a halving period, the n-th store adds 16·2^(n/half_life)
    /// (16 up to 31), so later stores outweigh earlier ones exactly as continuous decay
    /// would; each write survives about 5 halvings.
    pub fn new(ec_bits: usize, ca3_cells: usize, k: usize, decay: f32, settle_steps: usize) -> Self {
        let half_life = ((0.5f32.ln() / decay.clamp(0.01, 0.999).ln()).round() as u32).max(1);
        Self {
            k,
            half_life,
            settle_steps,
            readout_fraction: 0.5,
            planes: 7,
            ec_to_ca3: Pathway::new(ec_bits, ca3_cells),
            ca3_to_ca3: Pathway::new(ca3_cells, ca3_cells),
            ca3_to_ec: Pathway::new(ca3_cells, ec_bits),
            stores: 0,
        }
    }

    fn epoch(&self) -> u32 {
        self.stores / self.half_life
    }

    /// Store one episode: EC pattern `x` (active bits) with CA3 code `c`
    /// (e.g. from the dentate gyrus).
    pub fn store(&mut self, x: &[usize], c: &[u32]) {
        self.stores += 1;
        let (epoch, planes) = (self.epoch(), self.planes);
        let phase = (self.stores % self.half_life) as f32 / self.half_life as f32;
        let amount = (16.0 * 2f32.powf(phase)).round() as u32;
        let c_idx: Vec<usize> = c.iter().map(|&j| j as usize).collect();
        let c_mask = BitVector::from_bits(&c_idx, self.ca3_to_ca3.targets);
        let x_mask = BitVector::from_bits(x, self.ca3_to_ec.targets);
        for &i in x {
            self.ec_to_ca3.strengthen(i, &c_mask, amount, planes, epoch);
        }
        for &i in &c_idx {
            let mut others = c_mask.clone();
            others.bit_clear(i);
            self.ca3_to_ca3.strengthen(i, &others, amount, planes, epoch);
            self.ca3_to_ec.strengthen(i, &x_mask, amount, planes, epoch);
        }
    }

    /// Recall from a partial EC cue: drive CA3 from the cue, settle through the
    /// recurrent weights, read EC out. Returns the EC bits scoring at least
    /// `readout_fraction` of the best score, and that best score (0 if nothing was recalled).
    pub fn recall(&self, cue: &[usize], ec_bits: usize) -> (Vec<usize>, f32) {
        let epoch = self.epoch();
        let from_cue = self.ec_to_ca3.drive(cue, epoch);
        let mut c = top_k(from_cue.iter().map(|&v| v as f32), self.k);
        for _ in 0..self.settle_steps {
            let active: Vec<usize> = c.iter().map(|&j| j as usize).collect();
            let rec = self.ca3_to_ca3.drive(&active, epoch);
            c = top_k(rec.iter().zip(&from_cue).map(|(r, f)| (r + f) as f32), self.k);
        }
        let active: Vec<usize> = c.iter().map(|&j| j as usize).collect();
        let out = self.ca3_to_ec.drive(&active, epoch);
        let best = out.iter().copied().max().unwrap_or(0);
        if best == 0 {
            return (Vec::new(), 0.0);
        }
        let floor = self.readout_fraction * best as f32;
        ((0..ec_bits.min(out.len())).filter(|&b| out[b] as f32 >= floor).collect(), best as f32)
    }

    /// Number of stored synapses (all three pathways).
    pub fn synapses(&self) -> usize {
        self.ec_to_ca3.synapses() + self.ca3_to_ca3.synapses() + self.ca3_to_ec.synapses()
    }
}

/// Indices of the `k` largest positive values (ties by index).
fn top_k(values: impl Iterator<Item = f32>, k: usize) -> Vec<u32> {
    let mut v: Vec<(f32, u32)> = values.enumerate().filter(|(_, x)| *x > 0.0).map(|(i, x)| (x, i as u32)).collect();
    v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    let mut out: Vec<u32> = v.into_iter().take(k).map(|(_, i)| i).collect();
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(i: usize) -> Vec<usize> {
        (i * 8..i * 8 + 8).collect()
    }

    fn episode(words: &[usize]) -> Vec<usize> {
        words.iter().flat_map(|&w| word(w)).collect()
    }

    #[test]
    fn dentate_gyrus_separates_overlapping_episodes() {
        let dg = DentateGyrus::new(512, 4096, 64, 20, 1);
        // two episodes sharing 4 of 5 words
        let a = dg.separate(&episode(&[1, 2, 3, 4, 5]));
        let b = dg.separate(&episode(&[1, 2, 3, 4, 6]));
        let shared = a.iter().filter(|c| b.contains(c)).count();
        // inputs overlap 80%, codes much less
        assert_eq!(a.len(), 20);
        assert!(shared < 16, "codes too similar: {shared}/20 shared");
    }

    #[test]
    fn ca3_completes_the_most_recent_episode_from_a_partial_cue() {
        let dg = DentateGyrus::new(512, 4096, 64, 20, 1);
        let mut m = Ca3Memory::new(512, 4096, 20, 0.9, 2);
        // mary(1) went(10) kitchen(20); john(2) went(10) garden(21); mary(1) went(10) office(22)
        for ep in [episode(&[1, 10, 20]), episode(&[2, 10, 21]), episode(&[1, 10, 22])] {
            m.store(&ep, &dg.separate(&ep));
        }
        let (out, strength) = m.recall(&word(1), 512);
        assert!(strength > 0.0);
        let has = |w: usize| word(w).iter().all(|b| out.contains(b));
        assert!(has(22) && !has(20), "should recall the recent mary episode (office)");
        let (out, _) = m.recall(&word(2), 512);
        assert!(word(21).iter().all(|b| out.contains(b)));
    }
}
