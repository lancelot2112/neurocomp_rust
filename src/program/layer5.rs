//! Layer 5: two-compartment pyramidal cells (Larkum 2013).
//!
//! Each cell has two synapse sets with their own thresholds:
//! - **basal** (perisomatic): bits of the input stream, the column row's first frame (the
//!   current input) and last frame (the previous input);
//! - **apical** (the tuft in layer 1): bits of the context frames between them (top-down
//!   feedback, memory, other thalamic channels).
//!
//! A cell whose basal set matches fires; if its apical set matches too, it **bursts**. Only
//! bursts count here: a basal match alone is the habit L2/3 already makes. Among bursting
//! cells, the one whose recent bursts were most often confirmed wins (a trace at rate 1/2:
//! the moment, not the long run). With no burst, the layer has nothing to say and the
//! caller falls back (to L2/3, and from there to the cerebellum).
//!
//! Learning is gated by bursts:
//! - a burst confirmed by the next input raises the cell's trace, a contradicted one lowers it;
//!   a cell whose trace falls low after repeated misses is recycled;
//! - **recruitment:** where no burst predicted the next input and both streams are active, a
//!   new cell is grown on that coincidence (sampled basal and apical bits, the input as output),
//!   so each cell stands for "this input, in this context, is followed by that".
//!
//! Integer and bitwise only; an inverted index from row bits to cells.

use crate::bitvec::BitVector;
use crate::fixed::{Q16, ONE};
use rand::seq::SliceRandom;

struct Cell {
    basal: Vec<u32>,
    apical: Vec<u32>,
    out: Vec<u32>,
    basal_thr: u16,
    apical_thr: u16,
    trace: u32,
    hits: u16,
    misses: u16,
    last_used: u64,
    live: bool,
}

pub struct Layer5 {
    frame_words: usize,
    sample: usize,
    match_fraction_q: u32,
    max_cells: usize,
    cells: Vec<Cell>,
    /// row bit → (cell, apical?)
    index: Vec<Vec<(u32, bool)>>,
    tick: u64,
    /// predictions: (bursts won, no burst); learning: (cells grown, recycled)
    pub stats: [u64; 4],
}

impl Layer5 {
    /// `frame_words`: words per frame of the row; `sample`: synapses per compartment;
    /// `match_fraction`: share of a compartment's synapses that must be active.
    pub fn new(frame_words: usize, sample: usize, match_fraction: f64, max_cells: usize) -> Self {
        Self {
            frame_words,
            sample,
            match_fraction_q: (match_fraction * ONE as f64) as u32, // float: config
            max_cells,
            cells: Vec::new(),
            index: Vec::new(),
            tick: 0,
            stats: [0; 4],
        }
    }

    pub fn cells(&self) -> usize {
        self.cells.iter().filter(|c| c.live).count()
    }

    /// Active row bits split into (basal, apical).
    fn streams(&self, row: &BitVector) -> (Vec<u32>, Vec<u32>) {
        let fw = self.frame_words.max(1);
        let frames = row.as_words().len() / fw;
        let (mut basal, mut apical) = (Vec::new(), Vec::new());
        for (wi, &w) in row.as_words().iter().enumerate() {
            let f = wi / fw;
            let mut w = w;
            while w != 0 {
                let b = (wi * 64 + w.trailing_zeros() as usize) as u32;
                if f == 0 || f + 1 >= frames {
                    basal.push(b);
                } else {
                    apical.push(b);
                }
                w &= w - 1;
            }
        }
        (basal, apical)
    }

    /// The bursting cells for `row`.
    fn bursting(&self, row: &BitVector) -> Vec<usize> {
        let mut counts: crate::det::HashMap<u32, (u16, u16)> = crate::det::HashMap::default();
        for (wi, &w) in row.as_words().iter().enumerate() {
            let mut w = w;
            while w != 0 {
                let b = wi * 64 + w.trailing_zeros() as usize;
                if let Some(list) = self.index.get(b) {
                    for &(c, ap) in list {
                        let e = counts.entry(c).or_insert((0, 0));
                        if ap {
                            e.1 += 1;
                        } else {
                            e.0 += 1;
                        }
                    }
                }
                w &= w - 1;
            }
        }
        let mut out: Vec<usize> = counts
            .into_iter()
            .filter(|&(c, (b, a))| {
                let cell = &self.cells[c as usize];
                cell.live && b >= cell.basal_thr && a >= cell.apical_thr
            })
            .map(|(c, _)| c as usize)
            .collect();
        out.sort_unstable();
        out
    }

    fn rank(&self, c: usize) -> (u32, u32, std::cmp::Reverse<usize>) {
        let cell = &self.cells[c];
        let rate = ((cell.hits as u32 + 1) << 16) / (cell.hits as u32 + cell.misses as u32 + 2);
        (cell.trace, rate, std::cmp::Reverse(c))
    }

    fn output(&self, c: usize, bits: usize) -> BitVector {
        let mut v = BitVector::new(bits, Some(0));
        for &b in &self.cells[c].out {
            if (b as usize) < bits {
                v.bit_set(b as usize);
            }
        }
        v
    }

    /// The winning burst's output and trace (`Q16`), or None: nothing bursts.
    pub fn predict(&mut self, row: &BitVector, out_bits: usize) -> Option<(BitVector, Q16)> {
        let b = self.bursting(row);
        match b.iter().copied().max_by_key(|&c| self.rank(c)) {
            Some(c) => {
                self.stats[0] += 1;
                Some((self.output(c, out_bits), self.cells[c].trace.min(ONE)))
            }
            None => {
                self.stats[1] += 1;
                None
            }
        }
    }

    /// Learn from what came next (`target`).
    pub fn learn<R: rand::Rng + ?Sized>(&mut self, row: &BitVector, target: &BitVector, rng: &mut R) {
        self.tick += 1;
        if target.count_ones() == 0 {
            return;
        }
        let predicts = |cell: &Cell| cell.out.iter().filter(|&&b| (b as usize) < target.bit_len() && target.bit_get(b as usize)).count() * 2 >= cell.out.len();
        let mut confirmed = false;
        let mut to_recycle = Vec::new();
        for c in self.bursting(row) {
            let hit = predicts(&self.cells[c]);
            let tick = self.tick;
            let cell = &mut self.cells[c];
            if hit {
                cell.trace += (ONE - cell.trace.min(ONE)) / 2;
                cell.hits = cell.hits.saturating_add(1);
                cell.last_used = tick;
                confirmed = true;
            } else {
                cell.trace -= cell.trace / 2;
                cell.misses = cell.misses.saturating_add(1);
                if cell.misses >= 2 && cell.trace < ONE / 8 {
                    to_recycle.push(c);
                }
            }
        }
        for c in to_recycle {
            self.remove(c);
            self.stats[3] += 1;
        }
        if confirmed {
            return;
        }
        // recruitment on this coincidence
        let (basal, apical) = self.streams(row);
        if basal.is_empty() || apical.is_empty() {
            return;
        }
        let pick = |v: &[u32], rng: &mut R| -> Vec<u32> {
            let mut s: Vec<u32> = v.choose_multiple(rng, self.sample.min(v.len())).copied().collect();
            s.sort_unstable();
            s
        };
        let b = pick(&basal, rng);
        let a = pick(&apical, rng);
        let thr = |n: usize| -> u16 { (((n as u64 * self.match_fraction_q as u64) + (ONE as u64 - 1)) >> 16).max(1) as u16 };
        let mut out = Vec::new();
        for (wi, &w) in target.as_words().iter().enumerate() {
            let mut w = w;
            while w != 0 {
                out.push((wi * 64 + w.trailing_zeros() as usize) as u32);
                w &= w - 1;
            }
        }
        let cell = Cell { basal_thr: thr(b.len()), apical_thr: thr(a.len()), basal: b, apical: a, out, trace: ONE / 2, hits: 0, misses: 0, last_used: self.tick, live: true };
        let slot = if self.cells.iter().filter(|c| c.live).count() >= self.max_cells {
            // recycle the least recently useful cell
            let v = (0..self.cells.len()).filter(|&c| self.cells[c].live).min_by_key(|&c| self.cells[c].last_used).unwrap();
            self.remove(v);
            self.stats[3] += 1;
            Some(v)
        } else {
            self.cells.iter().position(|c| !c.live)
        };
        let id = match slot {
            Some(s) => {
                self.cells[s] = cell;
                s
            }
            None => {
                self.cells.push(cell);
                self.cells.len() - 1
            }
        };
        let (bs, as_) = (self.cells[id].basal.clone(), self.cells[id].apical.clone());
        for (bits, ap) in [(bs, false), (as_, true)] {
            for b in bits {
                let b = b as usize;
                if self.index.len() <= b {
                    self.index.resize(b + 1, Vec::new());
                }
                self.index[b].push((id as u32, ap));
            }
        }
        self.stats[2] += 1;
    }

    fn remove(&mut self, c: usize) {
        let cell = &mut self.cells[c];
        cell.live = false;
        let bits: Vec<u32> = cell.basal.iter().chain(&cell.apical).copied().collect();
        for b in bits {
            if let Some(list) = self.index.get_mut(b as usize) {
                list.retain(|&(x, _)| x as usize != c);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn bursts_only_on_input_with_its_context() {
        // rows of 3 frames of 64 bits: [input | context | previous]
        let row = |input: usize, ctx: usize| {
            let mut v = BitVector::new(192, Some(0));
            for i in 0..8 {
                v.bit_set(input * 8 + i);
                v.bit_set(64 + ctx * 8 + i);
                v.bit_set(128 + i);
            }
            v
        };
        let target = |w: usize| {
            let mut v = BitVector::new(64, Some(0));
            for i in 0..8 {
                v.bit_set(w * 8 + i);
            }
            v
        };
        let mut l5 = Layer5::new(1, 8, 0.8, 100);
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        // the same input is followed by 2 in context 0 and by 3 in context 1
        for _ in 0..4 {
            l5.learn(&row(1, 0), &target(2), &mut rng);
            l5.learn(&row(1, 1), &target(3), &mut rng);
        }
        let (o, _) = l5.predict(&row(1, 0), 64).unwrap();
        assert_eq!(o.as_words(), target(2).as_words());
        let (o, _) = l5.predict(&row(1, 1), 64).unwrap();
        assert_eq!(o.as_words(), target(3).as_words());
        // a context never seen with this input: no burst
        assert!(l5.predict(&row(1, 5), 64).is_none());
    }
}
