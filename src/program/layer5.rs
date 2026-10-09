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

/// Layer 5 with priming (`L5=primed`): the tuft primes, the input triggers.
///
/// - **A fixed pool of cells.** A free cell is wired to a random context: its apical synapses
///   are sampled from the context of a random moment of experience. It has no basal synapses
///   and no output yet.
/// - **Two popcounts.** Apical: the share of the tuft's connected synapses that are active is
///   the cell's *priming* (graded, persisting over steps, decaying by 1/4 per step). Basal:
///   the input. Priming lowers the basal threshold (from 80% of the basal synapses to 50% at
///   full priming): a primed cell fires on partial input.
/// - **Burst.** A firing cell whose priming is at or above the area's context threshold
///   bursts. Bursts beat single spikes; then higher priming, then the burst trace.
/// - **Matrix gain.** The context threshold is one value for the area (diffuse matrix input,
///   acetylcholine-like): it falls while nothing bursts and rises while bursts happen, between
///   0.3 and 0.9. Tufts stay selective; the matrix sets how selective the area is now.
/// - **Hebbian learning** (permanences, connected at 128):
///   - a confirmed burst strengthens the winner's active basal and apical synapses (+12) and
///     weakens its inactive ones (−6);
///   - a contradicted burst weakens the winner's active apical synapses (−24, burst-gated
///     depression: this context does not make it right) and halves its trace;
///   - **lateral inhibition:** the other primed cells that fired with a different output lose
///     on their active apical synapses (−12, anti-Hebbian), so they separate from this pattern.
/// - **Commitment.** Where no burst predicted the next input, the most primed free cell takes
///   the input (basal synapses sampled from it) and the next input as its output. A committed
///   cell that keeps failing is freed and rewired to another random context.
pub struct PrimedLayer5 {
    frame_words: usize,
    sample: usize,
    cells: Vec<PCell>,
    /// row bit → (cell, apical?)
    index: Vec<Vec<(u32, bool)>>,
    tick: u64,
    theta: Q16,
    /// (bursts won, single spikes won, nothing fired, commitments, freed)
    pub stats: [u64; 5],
}

#[derive(Clone)]
struct PCell {
    basal: Vec<(u32, u8)>,
    apical: Vec<(u32, u8)>,
    out: Vec<u32>,
    trace: u32,
    misses: u16,
    prime: Q16,
    prime_tick: u64,
    wired_tick: u64,
}

const CONNECTED: u8 = 128;

struct Eval {
    /// (cell, priming now, fired, burst)
    fired: Vec<(usize, Q16, bool)>,
    /// free cells primed at or above the threshold: (cell, priming)
    free_primed: Vec<(usize, Q16)>,
}

impl PrimedLayer5 {
    pub fn new(frame_words: usize, sample: usize, cells: usize) -> Self {
        let empty = PCell { basal: Vec::new(), apical: Vec::new(), out: Vec::new(), trace: ONE / 2, misses: 0, prime: 0, prime_tick: 0, wired_tick: 0 };
        Self { frame_words, sample, cells: vec![empty; cells], index: Vec::new(), tick: 0, theta: ONE * 4 / 5, stats: [0; 5] }
    }

    pub fn committed(&self) -> usize {
        self.cells.iter().filter(|c| !c.out.is_empty()).count()
    }

    pub fn threshold(&self) -> Q16 {
        self.theta
    }

    fn split(&self, row: &BitVector) -> (Vec<u32>, Vec<u32>) {
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

    fn index_add(&mut self, cell: usize, bit: u32, apical: bool) {
        let b = bit as usize;
        if self.index.len() <= b {
            self.index.resize(b + 1, Vec::new());
        }
        self.index[b].push((cell as u32, apical));
    }

    fn index_remove(&mut self, cell: usize, bit: u32) {
        if let Some(l) = self.index.get_mut(bit as usize) {
            l.retain(|&(c, _)| c as usize != cell);
        }
    }

    /// Priming carried from earlier steps, decayed.
    fn carried(&self, c: usize) -> Q16 {
        let cell = &self.cells[c];
        let dt = self.tick.saturating_sub(cell.prime_tick) as u32;
        cell.prime.saturating_sub(dt.saturating_mul(ONE / 4))
    }

    fn evaluate(&self, row: &BitVector) -> Eval {
        // connected active synapses per cell and compartment
        let mut counts: crate::det::HashMap<u32, (u16, u16)> = crate::det::HashMap::default();
        for (wi, &w) in row.as_words().iter().enumerate() {
            let mut w = w;
            while w != 0 {
                let b = wi * 64 + w.trailing_zeros() as usize;
                if let Some(list) = self.index.get(b) {
                    for &(c, ap) in list {
                        let cell = &self.cells[c as usize];
                        let syn = if ap { &cell.apical } else { &cell.basal };
                        if syn.iter().any(|&(x, p)| x as usize == b && p >= CONNECTED) {
                            let e = counts.entry(c).or_insert((0, 0));
                            if ap {
                                e.1 += 1;
                            } else {
                                e.0 += 1;
                            }
                        }
                    }
                }
                w &= w - 1;
            }
        }
        let mut fired = Vec::new();
        let mut free_primed = Vec::new();
        let mut keys: Vec<u32> = counts.keys().copied().collect();
        keys.sort_unstable();
        for c in keys {
            let (b, a) = counts[&c];
            let c = c as usize;
            let cell = &self.cells[c];
            let na = cell.apical.iter().filter(|s| s.1 >= CONNECTED).count().max(1) as u64;
            let p_now = ((a as u64) << 16) / na;
            let p = (p_now as Q16).max(self.carried(c)).min(ONE);
            if cell.out.is_empty() {
                if p >= self.theta {
                    free_primed.push((c, p));
                }
                continue;
            }
            let nb = cell.basal.iter().filter(|s| s.1 >= CONNECTED).count() as u64;
            if nb == 0 {
                continue;
            }
            // need: 80% of the basal synapses, down to 50% at full priming
            let frac = (ONE as u64 * 4 / 5) - (p as u64 * 3 / 10);
            let need = ((nb * frac) + (ONE as u64 - 1)) >> 16;
            if b as u64 >= need.max(1) {
                fired.push((c, p, p >= self.theta));
            }
        }
        Eval { fired, free_primed }
    }

    fn winner(&self, e: &Eval) -> Option<(usize, Q16, bool)> {
        e.fired.iter().copied().max_by_key(|&(c, p, burst)| (burst, p, self.cells[c].trace, std::cmp::Reverse(c)))
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

    /// The winner's output and confidence (its priming), with whether it burst; None:
    /// nothing fired. Updates the cells' carried priming and the matrix gain.
    pub fn predict(&mut self, row: &BitVector, out_bits: usize) -> Option<(BitVector, Q16, bool)> {
        self.tick += 1;
        let e = self.evaluate(row);
        // priming persists
        for &(c, p, _) in &e.fired {
            self.cells[c].prime = p;
            self.cells[c].prime_tick = self.tick;
        }
        let w = self.winner(&e);
        // the matrix gain
        if w.map_or(false, |x| x.2) {
            self.theta += (ONE * 9 / 10).saturating_sub(self.theta) / 64;
        } else {
            self.theta = (self.theta - self.theta / 64).max(ONE * 3 / 10);
        }
        match w {
            Some((c, p, burst)) => {
                self.stats[if burst { 0 } else { 1 }] += 1;
                Some((self.output(c, out_bits), p, burst))
            }
            None => {
                self.stats[2] += 1;
                None
            }
        }
    }

    fn hebb(syn: &mut [(u32, u8)], active: &[u32], up: u8, down: u8) {
        for s in syn.iter_mut() {
            if active.binary_search(&s.0).is_ok() {
                s.1 = s.1.saturating_add(up);
            } else {
                s.1 = s.1.saturating_sub(down);
            }
        }
    }

    fn depress(syn: &mut [(u32, u8)], active: &[u32], down: u8) {
        for s in syn.iter_mut() {
            if active.binary_search(&s.0).is_ok() {
                s.1 = s.1.saturating_sub(down);
            }
        }
    }

    fn free(&mut self, c: usize) {
        let bits: Vec<u32> = self.cells[c].basal.iter().chain(&self.cells[c].apical).map(|s| s.0).collect();
        for b in bits {
            self.index_remove(c, b);
        }
        let cell = &mut self.cells[c];
        cell.basal.clear();
        cell.apical.clear();
        cell.out.clear();
        cell.trace = ONE / 2;
        cell.misses = 0;
        cell.prime = 0;
        self.stats[4] += 1;
    }

    /// Wire free cell `c` to the context `apical` (a random sample of it).
    fn wire<R: rand::Rng + ?Sized>(&mut self, c: usize, apical: &[u32], rng: &mut R) {
        let old: Vec<u32> = self.cells[c].apical.iter().map(|s| s.0).collect();
        for b in old {
            self.index_remove(c, b);
        }
        let mut pick: Vec<u32> = apical.choose_multiple(rng, self.sample.min(apical.len())).copied().collect();
        pick.sort_unstable();
        self.cells[c].apical = pick.iter().map(|&b| (b, CONNECTED + 32)).collect();
        self.cells[c].wired_tick = self.tick;
        for b in pick {
            self.index_add(c, b, true);
        }
    }

    /// Learn from what came next (`target`).
    pub fn learn<R: rand::Rng + ?Sized>(&mut self, row: &BitVector, target: &BitVector, rng: &mut R) {
        if target.count_ones() == 0 {
            return;
        }
        let (basal, apical) = self.split(row);
        let e = self.evaluate(row);
        let predicts = |cell: &PCell| !cell.out.is_empty() && cell.out.iter().filter(|&&b| (b as usize) < target.bit_len() && target.bit_get(b as usize)).count() * 2 >= cell.out.len();
        let w = self.winner(&e);
        let mut confirmed = false;
        if let Some((wc, _, burst)) = w {
            let hit = predicts(&self.cells[wc]);
            let cell = &mut self.cells[wc];
            if hit {
                confirmed = true;
                cell.trace += (ONE - cell.trace.min(ONE)) / 2;
                cell.misses = 0;
                Self::hebb(&mut cell.basal, &basal, 12, 6);
                if burst {
                    Self::hebb(&mut cell.apical, &apical, 12, 6);
                }
                // lateral inhibition: the other primed cells that fired, predicting otherwise
                let out = cell.out.clone();
                for &(c, p, _) in &e.fired {
                    if c != wc && p > 0 && self.cells[c].out != out {
                        Self::depress(&mut self.cells[c].apical, &apical, 12);
                    }
                }
            } else {
                cell.trace -= cell.trace / 2;
                cell.misses = cell.misses.saturating_add(1);
                if burst {
                    Self::depress(&mut cell.apical, &apical, 24);
                }
                if cell.misses >= 4 && cell.trace < ONE / 8 {
                    self.free(wc);
                }
            }
        }
        // other fired cells: those right are confirmed in their trace, those wrong decay
        for &(c, _, _) in &e.fired {
            if Some(c) == w.map(|x| x.0) || self.cells[c].out.is_empty() {
                continue;
            }
            let hit = predicts(&self.cells[c]);
            let cell = &mut self.cells[c];
            cell.trace = if hit { cell.trace + (ONE - cell.trace.min(ONE)) / 4 } else { cell.trace - cell.trace / 4 };
        }
        // commitment: the most primed free cell takes this input
        if !confirmed && !basal.is_empty() {
            if let Some(&(c, _)) = e.free_primed.iter().max_by_key(|x| (x.1, std::cmp::Reverse(x.0))) {
                let mut pick: Vec<u32> = basal.choose_multiple(rng, self.sample.min(basal.len())).copied().collect();
                pick.sort_unstable();
                for &b in &pick {
                    self.index_add(c, b, false);
                }
                let mut out = Vec::new();
                for (wi, &w) in target.as_words().iter().enumerate() {
                    let mut w = w;
                    while w != 0 {
                        out.push((wi * 64 + w.trailing_zeros() as usize) as u32);
                        w &= w - 1;
                    }
                }
                let cell = &mut self.cells[c];
                cell.basal = pick.iter().map(|&b| (b, CONNECTED + 32)).collect();
                cell.out = out;
                cell.trace = ONE / 2;
                self.stats[3] += 1;
            }
        }
        // drop dead synapses (permanence 0)
        if let Some((wc, _, _)) = w {
            for ap in [false, true] {
                let dead: Vec<u32> = {
                    let syn = if ap { &self.cells[wc].apical } else { &self.cells[wc].basal };
                    syn.iter().filter(|s| s.1 == 0).map(|s| s.0).collect()
                };
                for b in &dead {
                    self.index_remove(wc, *b);
                }
                let syn = if ap { &mut self.cells[wc].apical } else { &mut self.cells[wc].basal };
                syn.retain(|s| s.1 > 0);
            }
        }
        // random context: one free cell is (re)wired to this moment's context; one wired
        // long ago and never committed is rewired
        if !apical.is_empty() {
            let n = self.cells.len();
            let c = rng.gen_range(0..n);
            if self.cells[c].out.is_empty() && (self.cells[c].apical.is_empty() || self.tick.saturating_sub(self.cells[c].wired_tick) > 4 * n as u64) {
                self.wire(c, &apical, rng);
            }
        }
    }
}

#[cfg(test)]
mod primed_tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn primed_cells_learn_input_in_context() {
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
        let mut l5 = PrimedLayer5::new(1, 8, 64);
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        for _ in 0..400 {
            for (ctx, out) in [(0, 2), (1, 3)] {
                let r = row(1, ctx);
                l5.predict(&r, 64);
                l5.learn(&r, &target(out), &mut rng);
            }
        }
        let (o, _, burst) = l5.predict(&row(1, 0), 64).unwrap();
        assert!(burst);
        assert_eq!(o.as_words(), target(2).as_words());
        let (o, _, _) = l5.predict(&row(1, 1), 64).unwrap();
        assert_eq!(o.as_words(), target(3).as_words());
    }
}
