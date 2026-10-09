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
    /// row bit → (cell, synapse position, with the apical flag in bit 15)
    index: Vec<Vec<(u32, u16)>>,
    /// the evaluation of the last predicted row, reused by `learn` on the same row
    cached: Option<(Vec<u64>, Eval)>,
    /// Interneurons instead of the gain rule (`set_interneurons`): SST activity (tuft
    /// inhibition, driven by the area's bursts) and VIP activity (disinhibition, driven by
    /// the matrix's error signal: the layer's prediction failed).
    interneurons: bool,
    sst: Q16,
    vip: Q16,
    /// Vectorized counting (`set_vectorized`): per row bit, bitsets over cells of its
    /// connected synapses, summed into bit-sliced counters.
    masks: Option<BitMasks>,
    /// synapses as bits (`set_bit_synapses`): active, silent and sticky flags, no strengths
    bit_synapses: bool,
    /// cells whose synapses changed this step (re-synced into the masks)
    dirty: Vec<usize>,
    tick: u64,
    theta: Q16,
    /// Grown on demand (`grown`): at most this many cells; None = a fixed pool wired to
    /// random contexts.
    grow_cap: Option<usize>,
    /// the previous step's context bits (a grown cell's tuft samples the recent context)
    recent_apical: Vec<u32>,
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
    /// how much this cell has burst recently (`Q16`): its Martinotti (SST) cells' drive
    bursty: u32,
    /// connected synapses (basal, apical), kept current as learning changes them
    conn: (u16, u16, u16),
    prime: Q16,
    prime_tick: u64,
    wired_tick: u64,
    used_tick: u64,
}

const CONNECTED: u8 = 128;

// Bit synapses (`set_bit_synapses`): a synapse's state is its flags, no strength.
/// active (AMPA): always counts (≥ CONNECTED, so it counts as connected everywhere)
const AMPA: u8 = 0x80;
/// silent (NMDA only): counts only when the cell is depolarized (primed)
const SILENT: u8 = 0x40;
/// consolidated: resists pruning
const STICKY: u8 = 0x01;

/// Bit planes of the counters: counts up to 31 per compartment.
const PLANES: usize = 5;

/// Connected synapses as bitsets: for each row bit, one bitset over cells per compartment.
/// Counting adds the bitsets of the active row bits into bit-sliced counters (a ripple of
/// AND/XOR over whole words), so 64 cells are counted per instruction and the loops
/// vectorize; no per-synapse branches.
struct BitMasks {
    words: usize,
    basal: Vec<Vec<u64>>,
    apical: Vec<Vec<u64>>,
    silent: Vec<Vec<u64>>,
}

impl BitMasks {
    fn new(cells: usize) -> Self {
        Self { words: cells.div_ceil(64).max(1), basal: Vec::new(), apical: Vec::new(), silent: Vec::new() }
    }

    fn grow_to(&mut self, cells: usize) {
        let need = cells.div_ceil(64).max(1);
        if need > self.words {
            let w = need.max(self.words * 2);
            for m in self.basal.iter_mut().chain(self.apical.iter_mut()).chain(self.silent.iter_mut()) {
                if !m.is_empty() {
                    m.resize(w, 0);
                }
            }
            self.words = w;
        }
    }

    /// table: 0 basal (active), 1 apical (active), 2 basal silent
    fn set(&mut self, bit: u32, cell: usize, table: usize, on: bool) {
        let words = self.words;
        let table = match table {
            0 => &mut self.basal,
            1 => &mut self.apical,
            _ => &mut self.silent,
        };
        let b = bit as usize;
        if table.len() <= b {
            table.resize(b + 1, Vec::new());
        }
        let m = &mut table[b];
        if m.is_empty() {
            if !on {
                return;
            }
            m.resize(words, 0);
        }
        let (w, k) = (cell / 64, cell % 64);
        if on {
            m[w] |= 1 << k;
        } else {
            m[w] &= !(1 << k);
        }
    }

    fn add(planes: &mut [Vec<u64>; PLANES], mask: &[u64], carry: &mut [u64]) {
        carry.copy_from_slice(mask);
        for plane in planes.iter_mut() {
            for (p, c) in plane.iter_mut().zip(carry.iter_mut()) {
                let t = *p & *c;
                *p ^= *c;
                *c = t;
            }
        }
    }

    /// Per-cell (basal, apical, basal silent) counts of active synapses, and the touched cells.
    fn count(&self, row: &BitVector, cells: usize) -> (Vec<(u16, u16, u16)>, Vec<u32>) {
        let w = self.words;
        let mut pb: [Vec<u64>; PLANES] = std::array::from_fn(|_| vec![0u64; w]);
        let mut pa: [Vec<u64>; PLANES] = std::array::from_fn(|_| vec![0u64; w]);
        let mut ps: [Vec<u64>; PLANES] = std::array::from_fn(|_| vec![0u64; w]);
        let mut touched = vec![0u64; w];
        let mut carry = vec![0u64; w];
        for (wi, &x) in row.as_words().iter().enumerate() {
            let mut x = x;
            while x != 0 {
                let b = wi * 64 + x.trailing_zeros() as usize;
                for (table, planes) in [(&self.basal, &mut pb), (&self.apical, &mut pa), (&self.silent, &mut ps)] {
                    if let Some(m) = table.get(b).filter(|m| !m.is_empty()) {
                        Self::add(planes, m, &mut carry);
                        for (t, &v) in touched.iter_mut().zip(m.iter()) {
                            *t |= v;
                        }
                    }
                }
                x &= x - 1;
            }
        }
        let mut counts = vec![(0u16, 0u16, 0u16); cells];
        let mut list = Vec::new();
        for (wi, &t) in touched.iter().enumerate() {
            let mut t = t;
            while t != 0 {
                let k = t.trailing_zeros() as usize;
                let c = wi * 64 + k;
                let get = |planes: &[Vec<u64>; PLANES]| -> u16 { (0..PLANES).map(|i| (((planes[i][wi] >> k) & 1) as u16) << i).sum() };
                if c < cells {
                    let v = (get(&pb), get(&pa), get(&ps));
                    if v != (0, 0, 0) {
                        counts[c] = v;
                        list.push(c as u32);
                    }
                }
                t &= t - 1;
            }
        }
        (counts, list)
    }
}

struct Eval {
    /// (cell, priming now, fired, burst)
    fired: Vec<(usize, Q16, bool)>,
    /// free cells primed at or above the threshold: (cell, priming)
    free_primed: Vec<(usize, Q16)>,
}

impl PrimedLayer5 {
    pub fn new(frame_words: usize, sample: usize, cells: usize) -> Self {
        let empty = PCell { basal: Vec::new(), apical: Vec::new(), out: Vec::new(), trace: ONE / 2, misses: 0, bursty: 0, conn: (0, 0, 0), prime: 0, prime_tick: 0, wired_tick: 0, used_tick: 0 };
        Self { frame_words, sample, cells: vec![empty; cells], index: Vec::new(), cached: None, masks: None, bit_synapses: false, dirty: Vec::new(), interneurons: false, sst: ONE / 2, vip: 0, tick: 0, theta: ONE * 4 / 5, grow_cap: None, recent_apical: Vec::new(), stats: [0; 5] }
    }

    /// Cells grown as needed instead of a pool (the limit of a large reserve of silent cells
    /// with random synapses: the one recruited is the one wired closest to the moment). A
    /// cell is created where nothing predicted the next input, its tuft sampled from the
    /// recent context (this step's and the previous step's), its basal synapses from the
    /// input. Past `cap` cells the least recently useful is recycled.
    pub fn grown(frame_words: usize, sample: usize, cap: usize) -> Self {
        let mut l = Self::new(frame_words, sample, 0);
        l.grow_cap = Some(cap);
        l
    }

    pub fn committed(&self) -> usize {
        self.cells.iter().filter(|c| !c.out.is_empty()).count()
    }

    pub fn threshold(&self) -> Q16 {
        if self.interneurons {
            self.area_theta()
        } else {
            self.theta
        }
    }

    /// Interneurons set the context threshold instead of the gain rule:
    /// - **SST (Martinotti)** cells inhibit the tufts. Their area activity follows how much the
    ///   area bursts (rate 1/16), and each cell's own SST input grows with its own recent
    ///   bursting (facilitating synapses from the cells they inhibit): a cell that bursts a
    ///   lot becomes harder to prime, so others get their turn.
    /// - **VIP** cells inhibit the SST cells. The matrix drives them with an error signal:
    ///   they rise when the layer's prediction failed or nothing fired, and decay when it
    ///   was confirmed (rate 1/8).
    /// - The tuft threshold is 0.3 + 0.6 × SST × (1 − VIP), plus a quarter of the cell's own
    ///   burstiness, at most 0.95. (PV cells are the winner-take-all among the fired cells.)
    pub fn set_interneurons(&mut self, on: bool) {
        self.interneurons = on;
    }

    pub fn interneuron_state(&self) -> (Q16, Q16) {
        (self.sst, self.vip)
    }

    fn area_theta(&self) -> Q16 {
        let inhib = (self.sst as u64 * (ONE - self.vip.min(ONE)) as u64) >> 16;
        ONE * 3 / 10 + ((inhib * 6 / 10) as Q16)
    }

    fn theta_for(&self, c: usize) -> Q16 {
        if !self.interneurons {
            return self.theta;
        }
        (self.area_theta() + self.cells[c].bursty.min(ONE) / 4).min(ONE * 95 / 100)
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

    fn index_add(&mut self, cell: usize, bit: u32, apical: bool, pos: usize) {
        let b = bit as usize;
        if self.index.len() <= b {
            self.index.resize(b + 1, Vec::new());
        }
        self.index[b].push((cell as u32, pos as u16 | if apical { 1 << 15 } else { 0 }));
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
        let (counts, touched) = match &self.masks {
            Some(m) => m.count(row, self.cells.len()),
            None => self.count_sparse(row),
        };
        self.threshold_counts(counts, touched)
    }

    /// Event-driven counting: each active row bit visits the synapses it contacts.
    fn count_sparse(&self, row: &BitVector) -> (Vec<(u16, u16, u16)>, Vec<u32>) {
        let mut counts: Vec<(u16, u16, u16)> = vec![(0, 0, 0); self.cells.len()];
        let mut touched: Vec<u32> = Vec::new();
        for (wi, &w) in row.as_words().iter().enumerate() {
            let mut w = w;
            while w != 0 {
                let b = wi * 64 + w.trailing_zeros() as usize;
                if let Some(list) = self.index.get(b) {
                    for &(c, code) in list {
                        let cell = &self.cells[c as usize];
                        let ap = code >> 15 == 1;
                        let pos = (code & 0x7fff) as usize;
                        let syn = if ap { &cell.apical } else { &cell.basal };
                        let f = syn.get(pos).map_or(0, |s| s.1);
                        let silent = !ap && f < CONNECTED && f & SILENT != 0;
                        if f >= CONNECTED || silent {
                            let e = &mut counts[c as usize];
                            if *e == (0, 0, 0) {
                                touched.push(c);
                            }
                            if ap {
                                e.1 += 1;
                            } else if silent {
                                e.2 += 1;
                            } else {
                                e.0 += 1;
                            }
                        }
                    }
                }
                w &= w - 1;
            }
        }
        (counts, touched)
    }

    fn threshold_counts(&self, counts: Vec<(u16, u16, u16)>, mut touched: Vec<u32>) -> Eval {
        let mut fired = Vec::new();
        let mut free_primed = Vec::new();
        touched.sort_unstable();
        for c in touched {
            let (b, a, bs) = counts[c as usize];
            let c = c as usize;
            let cell = &self.cells[c];
            let na = cell.conn.1.max(1) as u64;
            let p_now = ((a as u64) << 16) / na;
            let p = (p_now as Q16).max(self.carried(c)).min(ONE);
            if cell.out.is_empty() {
                if p >= self.theta_for(c) {
                    free_primed.push((c, p));
                }
                continue;
            }
            let theta = self.theta_for(c);
            if self.bit_synapses {
                // silent synapses count only when the cell is depolarized (primed); the
                // threshold is fixed: 80% of all basal synapses
                let nb = (cell.conn.0 + cell.conn.2) as u64;
                if nb == 0 {
                    continue;
                }
                let need = (nb * 4).div_ceil(5);
                let eff = b as u64 + if p >= theta { bs as u64 } else { 0 };
                if eff >= need.max(1) {
                    fired.push((c, p, p >= theta));
                }
                continue;
            }
            let nb = cell.conn.0 as u64;
            if nb == 0 {
                continue;
            }
            // need: 80% of the basal synapses, down to 50% at full priming
            let frac = (ONE as u64 * 4 / 5) - (p as u64 * 3 / 10);
            let need = ((nb * frac) + (ONE as u64 - 1)) >> 16;
            if b as u64 >= need.max(1) {
                fired.push((c, p, p >= theta));
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
        let e2 = Eval { fired: e.fired.clone(), free_primed: e.free_primed.clone() };
        self.cached = Some((row.as_words().to_vec(), e2));
        // priming persists
        for &(c, p, _) in &e.fired {
            self.cells[c].prime = p;
            self.cells[c].prime_tick = self.tick;
        }
        let w = self.winner(&e);
        if self.interneurons {
            // SST follows the area's bursting; each winner's own burstiness drives its SST
            let burst = w.map_or(false, |x| x.2);
            self.sst = if burst { self.sst + (ONE - self.sst.min(ONE)) / 16 } else { self.sst - self.sst / 16 };
            if let Some((c, _, b)) = w {
                let cell = &mut self.cells[c];
                cell.bursty = if b { cell.bursty + (ONE - cell.bursty.min(ONE)) / 8 } else { cell.bursty - cell.bursty / 8 };
            }
        } else if w.map_or(false, |x| x.2) {
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

    /// Count with bitsets and bit-sliced counters instead of the event-driven index (same
    /// counts, a different layout).
    pub fn set_vectorized(&mut self, on: bool) {
        if !on {
            self.masks = None;
            return;
        }
        self.masks = Some(BitMasks::new(self.cells.len()));
        for c in 0..self.cells.len() {
            self.sync(c);
        }
    }

    /// Synapses as bits instead of strengths: each synapse is absent, **silent** (NMDA
    /// only: it counts only when the cell is primed, the precondition), **active** (AMPA:
    /// always counts) or active and **sticky** (consolidated: resists pruning). Learning moves
    /// bits between these states with fixed probabilities; nothing is counted per synapse.
    /// - a new cell's basal synapses are silent, so it fires only when its context primes it;
    /// - a confirmed fire: active silent synapses become active (1/2), active synapses
    ///   become sticky (1/8), unused non-sticky ones are pruned (1/16), and one silent
    ///   synapse may grow onto an active input (1/4); after a burst the same on the tuft
    ///   (new tuft synapses are active);
    /// - a contradicted burst prunes active non-sticky tuft synapses (1/4) and unsticks
    ///   active sticky ones (1/8);
    /// - lateral inhibition: the losing primed cells prune active non-sticky tuft synapses (1/8).
    /// The fire threshold is fixed (80% of all basal synapses): priming lowers it in effect
    /// by letting silent synapses count.
    pub fn set_bit_synapses(&mut self, on: bool) {
        self.bit_synapses = on;
    }

    /// Learning on one compartment's bit synapses (see `set_bit_synapses`).
    /// `confirm`: confirmed fire; else `prune_active` (probability as 1/n) of active
    /// non-sticky synapses, `unstick` (1/n) of active sticky ones.
    fn bit_learn<R: rand::Rng + ?Sized>(syn: &mut [(u32, u8)], active: &[u32], confirm: bool, prune_active: u32, unstick: u32, rng: &mut R) {
        for s in syn.iter_mut() {
            let on = active.binary_search(&s.0).is_ok();
            let f = s.1;
            if f & (AMPA | SILENT) == 0 {
                continue; // absent (a pruned slot)
            }
            if confirm {
                if on && f & SILENT != 0 && f < CONNECTED {
                    if rng.gen_range(0..2) == 0 {
                        s.1 = AMPA | (f & STICKY);
                    }
                } else if on && f >= CONNECTED {
                    if rng.gen_range(0..8) == 0 {
                        s.1 |= STICKY;
                    }
                } else if !on && f & STICKY == 0 && rng.gen_range(0..16) == 0 {
                    s.1 = 0;
                }
            } else if on && f >= CONNECTED {
                if f & STICKY == 0 {
                    if prune_active > 0 && rng.gen_range(0..prune_active) == 0 {
                        s.1 = 0;
                    }
                } else if unstick > 0 && rng.gen_range(0..unstick) == 0 {
                    s.1 &= !STICKY;
                }
            }
        }
    }

    /// Grow one synapse of cell `c` (basal or apical) onto an active row bit it lacks,
    /// reusing a pruned slot if there is one (at most 2 × sample synapses).
    fn bit_grow<R: rand::Rng + ?Sized>(&mut self, c: usize, apical_side: bool, active: &[u32], rng: &mut R) {
        let cap = 2 * self.sample;
        let syn = if apical_side { &self.cells[c].apical } else { &self.cells[c].basal };
        let have: Vec<u32> = syn.iter().filter(|s| s.1 & (AMPA | SILENT) != 0).map(|s| s.0).collect();
        let cands: Vec<u32> = active.iter().copied().filter(|b| !have.contains(b)).collect();
        let Some(&b) = cands.choose(rng) else { return };
        let flags = if apical_side { AMPA } else { SILENT };
        let slot = syn.iter().position(|s| s.1 & (AMPA | SILENT) == 0);
        let pos = match slot {
            Some(pos) => {
                let old = syn[pos].0;
                self.index_remove(c, old);
                if let Some(m) = self.masks.as_mut() {
                    m.set(old, c, if apical_side { 1 } else { 0 }, false);
                    m.set(old, c, 2, false);
                }
                // other synapses of this cell on the old bit lost their index entry too
                let (bs, aps): (Vec<(usize, u32)>, Vec<(usize, u32)>) = (
                    self.cells[c].basal.iter().enumerate().filter(|(i, s)| s.0 == old && !(!apical_side && *i == pos)).map(|(i, s)| (i, s.0)).collect(),
                    self.cells[c].apical.iter().enumerate().filter(|(i, s)| s.0 == old && !(apical_side && *i == pos)).map(|(i, s)| (i, s.0)).collect(),
                );
                for (i, bit) in bs {
                    self.index_add(c, bit, false, i);
                }
                for (i, bit) in aps {
                    self.index_add(c, bit, true, i);
                }
                pos
            }
            None if syn.len() < cap => syn.len(),
            None => return,
        };
        let syn = if apical_side { &mut self.cells[c].apical } else { &mut self.cells[c].basal };
        if pos == syn.len() {
            syn.push((b, flags));
        } else {
            syn[pos] = (b, flags);
        }
        self.index_add(c, b, apical_side, pos);
        self.dirty.push(c);
    }

    /// Recount cell `c`'s connected synapses.
    fn refresh(&mut self, c: usize) {
        let cell = &mut self.cells[c];
        cell.conn = (
            cell.basal.iter().filter(|s| s.1 >= CONNECTED).count() as u16,
            cell.apical.iter().filter(|s| s.1 >= CONNECTED).count() as u16,
            cell.basal.iter().filter(|s| s.1 < CONNECTED && s.1 & SILENT != 0).count() as u16,
        );
    }

    /// Write cell `c`'s connected synapses into the masks.
    fn sync(&mut self, c: usize) {
        if let Some(m) = self.masks.as_mut() {
            m.grow_to(self.cells.len());
            for &(b, p) in &self.cells[c].basal {
                m.set(b, c, 0, p >= CONNECTED);
                m.set(b, c, 2, p < CONNECTED && p & SILENT != 0);
            }
            for &(b, p) in &self.cells[c].apical {
                m.set(b, c, 1, p >= CONNECTED);
            }
        }
    }

    /// Clear cell `c` from the masks (before its synapses are replaced).
    fn unsync(&mut self, c: usize, basal: bool, apical: bool) {
        if let Some(m) = self.masks.as_mut() {
            if basal {
                for &(b, _) in &self.cells[c].basal {
                    m.set(b, c, 0, false);
                    m.set(b, c, 2, false);
                }
            }
            if apical {
                for &(b, _) in &self.cells[c].apical {
                    m.set(b, c, 1, false);
                }
            }
        }
    }

    fn free(&mut self, c: usize) {
        self.unsync(c, true, true);
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
        cell.conn = (0, 0, 0);
        self.stats[4] += 1;
    }

    /// Wire free cell `c` to the context `apical` (a random sample of it).
    fn wire<R: rand::Rng + ?Sized>(&mut self, c: usize, apical: &[u32], rng: &mut R) {
        self.unsync(c, false, true);
        let old: Vec<u32> = self.cells[c].apical.iter().map(|s| s.0).collect();
        for b in old {
            self.index_remove(c, b);
        }
        let mut pick: Vec<u32> = apical.choose_multiple(rng, self.sample.min(apical.len())).copied().collect();
        pick.sort_unstable();
        self.cells[c].apical = pick.iter().map(|&b| (b, if self.bit_synapses { AMPA } else { CONNECTED + 32 })).collect();
        self.cells[c].wired_tick = self.tick;
        for (i, b) in pick.into_iter().enumerate() {
            self.index_add(c, b, true, i);
        }
        self.refresh(c);
        self.sync(c);
    }

    /// Learn from what came next (`target`).
    pub fn learn<R: rand::Rng + ?Sized>(&mut self, row: &BitVector, target: &BitVector, rng: &mut R) {
        if target.count_ones() == 0 {
            return;
        }
        let (basal, apical) = self.split(row);
        let bits = self.bit_synapses;
        let e = match self.cached.take() {
            Some((words, e)) if words == row.as_words() => e,
            _ => self.evaluate(row),
        };
        let predicts = |cell: &PCell| !cell.out.is_empty() && cell.out.iter().filter(|&&b| (b as usize) < target.bit_len() && target.bit_get(b as usize)).count() * 2 >= cell.out.len();
        let w = self.winner(&e);
        let mut confirmed = false;
        if let Some((wc, _, burst)) = w {
            let hit = predicts(&self.cells[wc]);
            let cell = &mut self.cells[wc];
            if hit {
                confirmed = true;
                cell.used_tick = self.tick;
                cell.trace += (ONE - cell.trace.min(ONE)) / 2;
                cell.misses = 0;
                if self.bit_synapses {
                    Self::bit_learn(&mut cell.basal, &basal, true, 0, 0, rng);
                    if burst {
                        Self::bit_learn(&mut cell.apical, &apical, true, 0, 0, rng);
                    }
                } else {
                    Self::hebb(&mut cell.basal, &basal, 12, 6);
                    if burst {
                        Self::hebb(&mut cell.apical, &apical, 12, 6);
                    }
                }
                // lateral inhibition: the other primed cells that fired, predicting otherwise
                let out = cell.out.clone();
                for &(c, p, _) in &e.fired {
                    if c != wc && p > 0 && self.cells[c].out != out {
                        if self.bit_synapses {
                            Self::bit_learn(&mut self.cells[c].apical, &apical, false, 8, 0, rng);
                        } else {
                            Self::depress(&mut self.cells[c].apical, &apical, 12);
                        }
                    }
                }
                if self.bit_synapses {
                    if rng.gen_range(0..4) == 0 {
                        self.bit_grow(wc, false, &basal, rng);
                    }
                    if burst && rng.gen_range(0..4) == 0 {
                        self.bit_grow(wc, true, &apical, rng);
                    }
                }
            } else {
                cell.trace -= cell.trace / 2;
                cell.misses = cell.misses.saturating_add(1);
                if burst {
                    if self.bit_synapses {
                        Self::bit_learn(&mut cell.apical, &apical, false, 4, 8, rng);
                    } else {
                        Self::depress(&mut cell.apical, &apical, 24);
                    }
                }
                if cell.misses >= 4 && cell.trace < ONE / 8 {
                    self.free(wc);
                }
            }
        }
        if self.interneurons {
            // the matrix's error signal drives VIP
            self.vip = if confirmed { self.vip - self.vip / 8 } else { self.vip + (ONE - self.vip.min(ONE)) / 8 };
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
        // grown: a new cell on this coincidence (tuft from the recent context)
        if let (Some(cap), false, false) = (self.grow_cap, confirmed, basal.is_empty()) {
            let mut ctx: Vec<u32> = apical.iter().chain(&self.recent_apical).copied().collect();
            ctx.sort_unstable();
            ctx.dedup();
            if !ctx.is_empty() {
                let live = self.cells.iter().filter(|c| !c.out.is_empty()).count();
                let slot = if live >= cap {
                    let v = (0..self.cells.len()).filter(|&c| !self.cells[c].out.is_empty()).min_by_key(|&c| (self.cells[c].used_tick, c)).unwrap();
                    self.free(v);
                    v
                } else if let Some(f) = self.cells.iter().position(|c| c.out.is_empty()) {
                    f
                } else {
                    self.cells.push(PCell { basal: Vec::new(), apical: Vec::new(), out: Vec::new(), trace: ONE / 2, misses: 0, bursty: 0, conn: (0, 0, 0), prime: 0, prime_tick: 0, wired_tick: 0, used_tick: 0 });
                    self.cells.len() - 1
                };
                self.wire(slot, &ctx, rng);
                let mut pick: Vec<u32> = basal.choose_multiple(rng, self.sample.min(basal.len())).copied().collect();
                pick.sort_unstable();
                for (i, &b) in pick.iter().enumerate() {
                    self.index_add(slot, b, false, i);
                }
                let mut out = Vec::new();
                for (wi, &w) in target.as_words().iter().enumerate() {
                    let mut w = w;
                    while w != 0 {
                        out.push((wi * 64 + w.trailing_zeros() as usize) as u32);
                        w &= w - 1;
                    }
                }
                let tick = self.tick;
                let cell = &mut self.cells[slot];
                cell.basal = pick.iter().map(|&b| (b, if bits { SILENT } else { CONNECTED + 32 })).collect();
                cell.out = out;
                cell.trace = ONE / 2;
                cell.used_tick = tick;
                self.stats[3] += 1;
                self.dirty.push(slot);
            }
        }
        // commitment: the most primed free cell takes this input
        if self.grow_cap.is_none() && !confirmed && !basal.is_empty() {
            if let Some(&(c, _)) = e.free_primed.iter().max_by_key(|x| (x.1, std::cmp::Reverse(x.0))) {
                let mut pick: Vec<u32> = basal.choose_multiple(rng, self.sample.min(basal.len())).copied().collect();
                pick.sort_unstable();
                for (i, &b) in pick.iter().enumerate() {
                    self.index_add(c, b, false, i);
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
                cell.basal = pick.iter().map(|&b| (b, if bits { SILENT } else { CONNECTED + 32 })).collect();
                cell.out = out;
                cell.trace = ONE / 2;
                self.stats[3] += 1;
                self.dirty.push(c);
            }
        }
        // the cells whose synapses changed: their connected counts (and masks) follow
        let mut d: Vec<usize> = e.fired.iter().map(|x| x.0).collect();
        d.append(&mut self.dirty);
        for c in d {
            self.refresh(c);
            if self.masks.is_some() && (!self.cells[c].out.is_empty() || !self.cells[c].apical.is_empty()) {
                self.sync(c);
            }
        }
        self.recent_apical = apical.clone();
        // random context: one free cell is (re)wired to this moment's context; one wired
        // long ago and never committed is rewired
        if self.grow_cap.is_none() && !apical.is_empty() && !self.cells.is_empty() {
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

    #[test]
    fn vectorized_counting_matches_event_driven() {
        use rand::Rng;
        let mut a = PrimedLayer5::new(4, 8, 256);
        let mut b = PrimedLayer5::new(4, 8, 256);
        b.set_vectorized(true);
        let (mut ra, mut rb) = (rand::rngs::StdRng::seed_from_u64(9), rand::rngs::StdRng::seed_from_u64(9));
        let mut g = rand::rngs::StdRng::seed_from_u64(5);
        for step in 0..3000 {
            // rows of 3 frames × 256 bits, ~12 active bits per frame from a small vocabulary
            let mut row = BitVector::new(768, Some(0));
            for f in 0..3 {
                let w = g.gen_range(0..12usize);
                for i in 0..12 {
                    row.bit_set(f * 256 + (w * 37 + i * 13) % 256);
                }
            }
            let mut target = BitVector::new(256, Some(0));
            let w = g.gen_range(0..12usize);
            for i in 0..12 {
                target.bit_set((w * 37 + i * 13) % 256);
            }
            let pa = a.predict(&row, 256).map(|(o, c, x)| (o.as_words().to_vec(), c, x));
            let pb = b.predict(&row, 256).map(|(o, c, x)| (o.as_words().to_vec(), c, x));
            assert_eq!(pa, pb, "step {step}");
            a.learn(&row, &target, &mut ra);
            b.learn(&row, &target, &mut rb);
        }
        assert!(a.committed() > 0);
    }

    #[test]
    fn bit_synapses_learn_input_in_context() {
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
        for vec in [false, true] {
            let mut l5 = PrimedLayer5::new(1, 8, 64);
            l5.set_bit_synapses(true);
            l5.set_vectorized(vec);
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

    #[test]
    fn bit_synapses_vectorized_matches_event_driven() {
        use rand::Rng;
        let mut a = PrimedLayer5::new(4, 8, 256);
        let mut b = PrimedLayer5::new(4, 8, 256);
        a.set_bit_synapses(true);
        b.set_bit_synapses(true);
        b.set_vectorized(true);
        let (mut ra, mut rb) = (rand::rngs::StdRng::seed_from_u64(9), rand::rngs::StdRng::seed_from_u64(9));
        let mut g = rand::rngs::StdRng::seed_from_u64(5);
        for step in 0..3000 {
            let mut row = BitVector::new(768, Some(0));
            for f in 0..3 {
                let w = g.gen_range(0..12usize);
                for i in 0..12 {
                    row.bit_set(f * 256 + (w * 37 + i * 13) % 256);
                }
            }
            let mut target = BitVector::new(256, Some(0));
            let w = g.gen_range(0..12usize);
            for i in 0..12 {
                target.bit_set((w * 37 + i * 13) % 256);
            }
            let pa = a.predict(&row, 256).map(|(o, c, x)| (o.as_words().to_vec(), c, x));
            let pb = b.predict(&row, 256).map(|(o, c, x)| (o.as_words().to_vec(), c, x));
            assert_eq!(pa, pb, "step {step}");
            a.learn(&row, &target, &mut ra);
            b.learn(&row, &target, &mut rb);
        }
        assert!(a.committed() > 0);
    }

    #[test]
    fn grown_cells_learn_input_in_context() {
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
        let mut l5 = PrimedLayer5::grown(1, 8, 64);
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        for _ in 0..50 {
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
        assert!(l5.committed() <= 64);
    }
}
