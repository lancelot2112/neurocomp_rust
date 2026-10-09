//! Primed two-compartment cells, fully bitwise (`BitCells`).
//!
//! A cell is masks and nothing else. For each word of the input row it touches, it holds
//! three 64-bit masks over that word's bits:
//! - **active** (AMPA): the synapse always counts;
//! - **silent** (NMDA only): it counts only when the cell is depolarized (primed), the
//!   precondition;
//! - **sticky** (consolidated): it resists pruning.
//!
//! A bit set in neither active nor silent is no synapse. Basal synapses sit on the input
//! frames (the row's first and last frame), apical ones on the context frames between.
//!
//! **Firing.** popcount(input & active) on the tuft over its synapses is the cell's
//! priming (carried over to the next steps, decaying by 1/4 per step). The basal drive is
//! popcount(input & active), plus popcount(input & silent) when primed, against a fixed 80%
//! of the basal synapses. A cell that fires while primed above the context threshold bursts;
//! bursts beat spikes, then priming, then the burst trace.
//!
//! **Learning is mask arithmetic.** A random mask with probability 2^-k per bit is the AND of
//! k random words. On a confirmed fire (per touched word, `r_k` such masks):
//! - promote: `silent & input & r_1` become active;
//! - consolidate: `sticky |= active & input & r_3`;
//! - prune: `(active | silent) & !input & !sticky & r_4` are removed;
//! - grow: with probability 1/4, one silent synapse onto an active input the cell lacks
//!   (after a burst also one active synapse on the tuft).
//!
//! A contradicted burst prunes `active & input & !sticky & r_2` on the tuft and unsticks
//! `sticky & input & r_3`; the losing primed cells (lateral inhibition) prune
//! `active & input & !sticky & r_3` on theirs.
//!
//! **The context threshold** is set by interneurons (SST from the area's bursting and each
//! cell's own, VIP from the matrix's error signal) or by a gain rule.
//!
//! **Cells** come from a pool wired to random contexts: a free cell's tuft is sampled from
//! the context of a random moment; where no burst predicted the next input, the most primed
//! free cell commits (silent basal synapses on the input, the next input as output). A
//! committed cell that keeps failing is freed.
//!
//! **Finding the cells to count:** per row bit, a bitset over cells marks which cells have a
//! synapse there; the active bits' bitsets ORed give the touched cells.

use crate::bitvec::BitVector;
use crate::fixed::{Q16, ONE};
use crate::program::layer5::BitMasks;
use rand::seq::SliceRandom;

#[derive(Clone, Copy, Default)]
struct Seg {
    w: u32,
    active: u64,
    silent: u64,
    sticky: u64,
}

impl Seg {
    fn present(&self) -> u64 {
        self.active | self.silent
    }
}

#[derive(Clone, Default)]
struct Cell {
    basal: Vec<Seg>,
    apical: Vec<Seg>,
    out: Vec<u32>,
    trace: u32,
    misses: u16,
    bursty: u32,
    /// present basal synapses, active apical synapses (kept as the masks change)
    nb: u32,
    na: u32,
    /// metaplasticity (`set_meta`): the learning rate as the number of random bits a change
    /// needs (k: probability 2^-k), and the firing threshold as a share of the basal
    /// synapses (`Q16`), both set by the cell's own outcomes; 0 = not yet set
    k: u8,
    thr: u32,
    prime: Q16,
    prime_tick: u64,
    wired_tick: u64,
}

#[derive(Clone)]
struct Eval {
    /// (cell, priming, burst)
    fired: Vec<(usize, Q16, bool)>,
    free_primed: Vec<(usize, Q16)>,
}

pub struct BitCells {
    frame_words: usize,
    sample: usize,
    cells: Vec<Cell>,
    /// The axons' view, kept in step with the cells' masks one bit at a time: per row bit, a
    /// bitset over cells for each synapse state (basal active, apical active, basal silent),
    /// summed by bit-sliced counters to count every cell at once.
    tables: BitMasks,
    tick: u64,
    theta: Q16,
    interneurons: bool,
    /// the plasticity index from the cell's id and the step (`set_id_index`), no generator
    id_index: bool,
    /// self-calibrating rates and thresholds (`set_meta`), and the area's error rate
    meta: bool,
    err: Q16,
    /// consolidation into sticky synapses (`set_sticky`)
    sticky_on: bool,
    /// how many trailing frames of the row are basal (input side): 1 (the previous input),
    /// 2 when a further input frame (L2/3's prediction) is appended (`set_basal_tail`)
    basal_tail: usize,
    sst: Q16,
    vip: Q16,
    cached: Option<(Vec<u64>, Eval)>,
    /// (bursts won, spikes won, nothing fired, commitments, freed)
    pub stats: [u64; 5],
}

impl BitCells {
    pub fn new(frame_words: usize, sample: usize, cells: usize) -> Self {
        let c = Cell { trace: ONE / 2, ..Default::default() };
        Self {
            frame_words,
            sample,
            cells: vec![c; cells],
            tables: BitMasks::new(cells),
            tick: 0,
            theta: ONE * 4 / 5,
            interneurons: false,
            id_index: false,
            meta: false,
            err: 0,
            sticky_on: true,
            basal_tail: 1,
            sst: ONE / 2,
            vip: 0,
            cached: None,
            stats: [0; 5],
        }
    }

    /// SST/VIP interneurons set the context threshold (see `PrimedLayer5::set_interneurons`).
    /// Take each plasticity event's index and gates from the cell's id plus the step
    /// counter instead of a random word: (id + step) mod 256 in every byte, so each cell
    /// sweeps through its synapses and cells on the same step change different ones; the
    /// gates fire on fixed residues of the same sum. Deterministic, no generator.
    pub fn set_id_index(&mut self, on: bool) {
        self.id_index = on;
    }

    /// Self-calibration (metaplasticity and intrinsic plasticity), from each cell's outcomes:
    /// - **rate:** a change needs k random bits (probability 2^-k; consolidation and growth
    ///   need k, pruning k − 1). A confirmed fire raises the cell's k by one (at most 6: it
    ///   settles), a contradicted one lowers it (at least 1: it learns faster). While the
    ///   area's error rate is above one half (surprise), every cell's k counts one less.
    /// - **threshold:** a contradicted fire raises the cell's threshold by 1/32 of its basal
    ///   synapses (at most 95%: it fires more selectively), a confirmed one lowers it by
    ///   1/128 (at least 50%).
    /// Starting at k = 2 and 80%, the fixed rates and threshold are where it begins.
    /// Consolidation on or off: off, no synapse ever becomes sticky.
    pub fn set_sticky(&mut self, on: bool) {
        self.sticky_on = on;
    }

    /// The number of trailing frames on the input (basal) side (see the field).
    pub fn set_basal_tail(&mut self, n: usize) {
        self.basal_tail = n.max(1);
    }

    pub fn set_meta(&mut self, on: bool) {
        self.meta = on;
    }

    fn cell_k(&self, c: usize) -> u32 {
        if !self.meta {
            return 2;
        }
        let k = if self.cells[c].k == 0 { 2 } else { self.cells[c].k as u32 };
        if self.err > ONE / 2 { k.saturating_sub(1).max(1) } else { k }
    }

    fn cell_thr(&self, c: usize) -> u32 {
        if !self.meta || self.cells[c].thr == 0 { ONE * 4 / 5 } else { self.cells[c].thr }
    }

    pub fn set_interneurons(&mut self, on: bool) {
        self.interneurons = on;
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

    fn frames(&self, row: &BitVector) -> usize {
        row.as_words().len() / self.frame_words.max(1)
    }

    fn is_basal_word(&self, w: usize, frames: usize) -> bool {
        let f = w / self.frame_words.max(1);
        f == 0 || f + self.basal_tail >= frames
    }

    /// Active row bits split into (basal, apical).
    fn split(&self, row: &BitVector) -> (Vec<u32>, Vec<u32>) {
        let frames = self.frames(row);
        let (mut b, mut a) = (Vec::new(), Vec::new());
        for (wi, &x) in row.as_words().iter().enumerate() {
            let mut x = x;
            while x != 0 {
                let bit = (wi * 64 + x.trailing_zeros() as usize) as u32;
                if self.is_basal_word(wi, frames) {
                    b.push(bit);
                } else {
                    a.push(bit);
                }
                x &= x - 1;
            }
        }
        (b, a)
    }

    /// A compartment's segment of cell `c` changed from `before` to `after` (same word):
    /// the tables and the cell's synapse counts follow, bit by bit.
    fn update_tables(&mut self, c: usize, apical: bool, before: Seg, after: Seg) {
        let w = before.w.max(after.w);
        let states: [(usize, u64, u64); 2] = if apical {
            [(1, before.active, after.active), (1, 0, 0)]
        } else {
            [(0, before.active, after.active), (2, before.silent & !before.active, after.silent & !after.active)]
        };
        for (table, b, a) in states {
            let mut d = b ^ a;
            while d != 0 {
                let k = d.trailing_zeros();
                self.tables.set(w * 64 + k, c, table, a >> k & 1 == 1);
                d &= d - 1;
            }
        }
        let cell = &mut self.cells[c];
        if apical {
            cell.na = cell.na + after.active.count_ones() - before.active.count_ones();
        } else {
            cell.nb = cell.nb + after.present().count_ones() - before.present().count_ones();
        }
    }

    fn carried(&self, c: usize) -> Q16 {
        let cell = &self.cells[c];
        let dt = self.tick.saturating_sub(cell.prime_tick) as u32;
        cell.prime.saturating_sub(dt.saturating_mul(ONE / 4))
    }

    fn evaluate(&self, row: &BitVector) -> Eval {
        let (counts, list) = self.tables.count(row, self.cells.len());
        let (mut fired, mut free_primed) = (Vec::new(), Vec::new());
        for c in list {
            let c = c as usize;
            let (b, a, bs) = counts[c];
            let cell = &self.cells[c];
            let p = if cell.na == 0 { 0 } else { (((a as u64) << 16) / cell.na as u64) as Q16 }.max(self.carried(c)).min(ONE);
            let theta = self.theta_for(c);
            if cell.out.is_empty() {
                if p >= theta && cell.na > 0 {
                    free_primed.push((c, p));
                }
                continue;
            }
            // a committed cell fires only on its input side
            if cell.nb == 0 || b + bs == 0 {
                continue;
            }
            let need = (((cell.nb as u64 * self.cell_thr(c) as u64) + (ONE as u64 - 1)) >> 16).max(1) as u32;
            let eff = b as u32 + if p >= theta { bs as u32 } else { 0 };
            if eff >= need {
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

    /// The winner's output, its priming and whether it burst; None: nothing fired.
    pub fn predict(&mut self, row: &BitVector, out_bits: usize) -> Option<(BitVector, Q16, bool)> {
        self.tick += 1;
        let e = self.evaluate(row);
        for &(c, p, _) in &e.fired {
            self.cells[c].prime = p;
            self.cells[c].prime_tick = self.tick;
        }
        let w = self.winner(&e);
        self.cached = Some((row.as_words().to_vec(), e));
        if self.interneurons {
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

    /// One synapse event: among the bits `cand(seg, input word)` of cell `c`'s compartment,
    /// the `n`-th (modulo their number) is changed by `f(seg, its bit)`.
    fn change<C: Fn(&Seg, u64) -> u64, F: FnOnce(&mut Seg, u64)>(&mut self, c: usize, apical: bool, row: &[u64], n: u32, cand: C, f: F) {
        let segs = if apical { &self.cells[c].apical } else { &self.cells[c].basal };
        let x = |s: &Seg| row.get(s.w as usize).copied().unwrap_or(0);
        let total: u32 = segs.iter().map(|s| cand(s, x(s)).count_ones()).sum();
        if total == 0 {
            return;
        }
        let mut n = n % total;
        let mut at = None;
        for (i, s) in segs.iter().enumerate() {
            let mut m = cand(s, x(s));
            let k = m.count_ones();
            if n < k {
                for _ in 0..n {
                    m &= m - 1;
                }
                at = Some((i, m & m.wrapping_neg()));
                break;
            }
            n -= k;
        }
        let Some((i, bit)) = at else { return };
        let segs = if apical { &mut self.cells[c].apical } else { &mut self.cells[c].basal };
        let before = segs[i];
        f(&mut segs[i], bit);
        segs[i].sticky &= segs[i].present();
        let after = segs[i];
        if after.present() == 0 {
            segs.remove(i);
        }
        self.update_tables(c, apical, before, after);
    }

    /// Grow a synapse on `bit` if the cell lacks it (at most 2 × sample): silent on the
    /// basal side, active on the tuft.
    fn grow_one(&mut self, c: usize, apical: bool, bit: u32) {
        if (self.n_present(c, apical) as usize) < 2 * self.sample && !self.has(c, apical, bit) {
            self.add_synapse(c, apical, bit, apical);
        }
    }

    fn n_present(&self, c: usize, apical: bool) -> u32 {
        let segs = if apical { &self.cells[c].apical } else { &self.cells[c].basal };
        segs.iter().map(|s| s.present().count_ones()).sum()
    }

    /// Add synapse `bit` to cell `c` (silent or active).
    fn add_synapse(&mut self, c: usize, apical: bool, bit: u32, active: bool) {
        let (w, k) = (bit / 64, bit % 64);
        let segs = if apical { &mut self.cells[c].apical } else { &mut self.cells[c].basal };
        let i = match segs.binary_search_by_key(&w, |s| s.w) {
            Ok(i) => i,
            Err(i) => {
                segs.insert(i, Seg { w, ..Default::default() });
                i
            }
        };
        if segs[i].present() >> k & 1 == 1 {
            return;
        }
        let before = segs[i];
        if active {
            segs[i].active |= 1 << k;
        } else {
            segs[i].silent |= 1 << k;
        }
        let after = segs[i];
        self.update_tables(c, apical, before, after);
    }

    fn has(&self, c: usize, apical: bool, bit: u32) -> bool {
        let segs = if apical { &self.cells[c].apical } else { &self.cells[c].basal };
        segs.binary_search_by_key(&(bit / 64), |s| s.w).map_or(false, |i| segs[i].present() >> (bit % 64) & 1 == 1)
    }

    fn free(&mut self, c: usize) {
        for apical in [false, true] {
            let segs = if apical { std::mem::take(&mut self.cells[c].apical) } else { std::mem::take(&mut self.cells[c].basal) };
            for s in segs {
                self.update_tables(c, apical, s, Seg { w: s.w, ..Default::default() });
            }
        }
        let cell = &mut self.cells[c];
        cell.out.clear();
        cell.k = 0;
        cell.thr = 0;
        cell.trace = ONE / 2;
        cell.misses = 0;
        cell.prime = 0;
        cell.bursty = 0;
        self.stats[4] += 1;
    }

    fn wire<R: rand::Rng + ?Sized>(&mut self, c: usize, apical: &[u32], rng: &mut R) {
        let old = std::mem::take(&mut self.cells[c].apical);
        for s in old {
            self.update_tables(c, true, s, Seg { w: s.w, ..Default::default() });
        }
        let pick: Vec<u32> = apical.choose_multiple(rng, self.sample.min(apical.len())).copied().collect();
        for b in pick {
            self.add_synapse(c, true, b, true);
        }
        self.cells[c].wired_tick = self.tick;
    }

    /// Learn from what came next (`target`).
    pub fn learn<R: rand::Rng + ?Sized>(&mut self, row: &BitVector, target: &BitVector, rng: &mut R) {
        if target.count_ones() == 0 {
            return;
        }
        let (basal, apical) = self.split(row);
        let e = match self.cached.take() {
            Some((words, e)) if words == row.as_words() => e,
            _ => self.evaluate(row),
        };
        let words = row.as_words().to_vec();
        let predicts = |cell: &Cell| !cell.out.is_empty() && cell.out.iter().filter(|&&b| (b as usize) < target.bit_len() && target.bit_get(b as usize)).count() * 2 >= cell.out.len();
        let w = self.winner(&e);
        let mut confirmed = false;
        if let Some((wc, _, burst)) = w {
            // one plasticity event: one random word; its bytes index the synapses changed
            // and its high bits gate the rarer changes
            let r = if self.id_index { ((wc as u64).wrapping_add(self.tick) & 0xff).wrapping_mul(0x0101_0101_0101_0101) } else { rng.next_u64() };
            let byte = |i: u32| ((r >> (8 * i)) & 0xff) as u32;
            // a change gated by k random bits (all zero: probability 2^-k)
            let k = self.cell_k(wc);
            let gate = |pos: u32, k: u32| k == 0 || (r >> pos) & ((1u64 << k) - 1) == 0;
            if predicts(&self.cells[wc]) {
                confirmed = true;
                let cell = &mut self.cells[wc];
                cell.trace += (ONE - cell.trace.min(ONE)) / 2;
                cell.misses = 0;
                if self.meta {
                    cell.k = (if cell.k == 0 { 2 } else { cell.k } + 1).min(6);
                    let t = if cell.thr == 0 { ONE * 4 / 5 } else { cell.thr };
                    cell.thr = (t - ONE / 128).max(ONE / 2);
                }
                for (apical, base) in [(false, 0u32), (true, 3u32)] {
                    if apical && !burst {
                        continue;
                    }
                    // promote one active silent synapse
                    self.change(wc, apical, &words, byte(base), |s, x| s.silent & x, |s, m| {
                        s.silent &= !m;
                        s.active |= m;
                    });
                    // consolidate one active synapse (1/4)
                    if gate(48 + base, k) {
                        if self.sticky_on {
                            self.change(wc, apical, &words, byte(base + 1), |s, x| s.active & x & !s.sticky, |s, m| s.sticky |= m);
                        }
                    }
                    // prune one unused synapse (1/2)
                    if gate(52 + base, k - 1) {
                        self.change(wc, apical, &words, byte(base + 2), |s, x| s.present() & !x & !s.sticky, |s, m| {
                            s.active &= !m;
                            s.silent &= !m;
                        });
                    }
                }
                // grow one synapse onto an active input the cell lacks (1/4)
                if gate(58, k) && !basal.is_empty() {
                    let b = basal[((r >> 16) & 0xffff) as usize % basal.len()];
                    self.grow_one(wc, false, b);
                }
                if burst && gate(60, k) && !apical.is_empty() {
                    let b = apical[((r >> 32) & 0xffff) as usize % apical.len()];
                    self.grow_one(wc, true, b);
                }
                // lateral inhibition: each losing primed cell loses one active tuft synapse
                let out = self.cells[wc].out.clone();
                let losers: Vec<usize> = e.fired.iter().filter(|&&(c, p, _)| c != wc && p > 0 && self.cells[c].out != out).map(|x| x.0).collect();
                for (j, c) in losers.into_iter().enumerate() {
                    self.change(c, true, &words, byte(j as u32 % 8), |s, x| s.active & x & !s.sticky, |s, m| s.active &= !m);
                }
            } else {
                let cell = &mut self.cells[wc];
                cell.trace -= cell.trace / 2;
                cell.misses = cell.misses.saturating_add(1);
                if self.meta {
                    cell.k = (if cell.k == 0 { 2 } else { cell.k }).saturating_sub(1).max(1);
                    let t = if cell.thr == 0 { ONE * 4 / 5 } else { cell.thr };
                    cell.thr = (t + ONE / 32).min(ONE * 95 / 100);
                }
                let fail = cell.misses >= 4 && cell.trace < ONE / 8;
                if burst {
                    // this context did not make it right: one active tuft synapse pruned,
                    // and (1/2) one sticky one unstuck
                    self.change(wc, true, &words, byte(0), |s, x| s.active & x & !s.sticky, |s, m| s.active &= !m);
                    if gate(48, k - 1) {
                        self.change(wc, true, &words, byte(1), |s, x| s.sticky & x, |s, m| s.sticky &= !m);
                    }
                }
                if fail {
                    self.free(wc);
                }
            }
        }
        if self.meta && w.is_some() {
            self.err = if confirmed { self.err - self.err / 16 } else { self.err + (ONE - self.err.min(ONE)) / 16 };
        }
        if self.interneurons {
            self.vip = if confirmed { self.vip - self.vip / 8 } else { self.vip + (ONE - self.vip.min(ONE)) / 8 };
        }
        for &(c, _, _) in &e.fired {
            if Some(c) == w.map(|x| x.0) || self.cells[c].out.is_empty() {
                continue;
            }
            let hit = predicts(&self.cells[c]);
            let cell = &mut self.cells[c];
            cell.trace = if hit { cell.trace + (ONE - cell.trace.min(ONE)) / 4 } else { cell.trace - cell.trace / 4 };
        }
        // commitment: the most primed free cell takes this input (silent basal synapses)
        if !confirmed && !basal.is_empty() {
            if let Some(&(c, _)) = e.free_primed.iter().max_by_key(|x| (x.1, std::cmp::Reverse(x.0))) {
                let pick: Vec<u32> = basal.choose_multiple(rng, self.sample.min(basal.len())).copied().collect();
                for b in pick {
                    self.add_synapse(c, false, b, false);
                }
                let mut out = Vec::new();
                for (wi, &x) in target.as_words().iter().enumerate() {
                    let mut x = x;
                    while x != 0 {
                        out.push((wi * 64 + x.trailing_zeros() as usize) as u32);
                        x &= x - 1;
                    }
                }
                let cell = &mut self.cells[c];
                cell.out = out;
                cell.trace = ONE / 2;
                self.stats[3] += 1;
            }
        }
        // random context: a free cell is (re)wired to this moment's context
        if !apical.is_empty() && !self.cells.is_empty() {
            let n = self.cells.len();
            let c = rng.gen_range(0..n);
            if self.cells[c].out.is_empty() && (self.cells[c].apical.is_empty() || self.tick.saturating_sub(self.cells[c].wired_tick) > 4 * n as u64) {
                self.wire(c, &apical, rng);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn row(input: usize, ctx: usize) -> BitVector {
        let mut v = BitVector::new(192, Some(0));
        for i in 0..8 {
            v.bit_set(input * 8 + i);
            v.bit_set(64 + ctx * 8 + i);
            v.bit_set(128 + i);
        }
        v
    }

    fn target(w: usize) -> BitVector {
        let mut v = BitVector::new(64, Some(0));
        for i in 0..8 {
            v.bit_set(w * 8 + i);
        }
        v
    }

    #[test]
    fn bitwise_cells_learn_input_in_context() {
        for inter in [false, true] {
            let mut l = BitCells::new(1, 8, 64);
            l.set_interneurons(inter);
            let mut rng = rand::rngs::StdRng::seed_from_u64(3);
            for _ in 0..400 {
                for (ctx, out) in [(0, 2), (1, 3)] {
                    let r = row(1, ctx);
                    l.predict(&r, 64);
                    l.learn(&r, &target(out), &mut rng);
                }
            }
            let (o, _, _) = l.predict(&row(1, 0), 64).unwrap();
            assert_eq!(o.as_words(), target(2).as_words());
            let (o, _, _) = l.predict(&row(1, 1), 64).unwrap();
            assert_eq!(o.as_words(), target(3).as_words());
        }
    }

    #[test]
    fn self_calibrating_cells_learn_input_in_context() {
        let mut l = BitCells::new(1, 8, 64);
        l.set_meta(true);
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        for _ in 0..400 {
            for (ctx, out) in [(0, 2), (1, 3)] {
                let r = row(1, ctx);
                l.predict(&r, 64);
                l.learn(&r, &target(out), &mut rng);
            }
        }
        let (o, _, _) = l.predict(&row(1, 0), 64).unwrap();
        assert_eq!(o.as_words(), target(2).as_words());
        let (o, _, _) = l.predict(&row(1, 1), 64).unwrap();
        assert_eq!(o.as_words(), target(3).as_words());
        // confirmed cells settled: their rate exponent rose
        assert!(l.cells.iter().any(|c| c.k > 2));
    }

    #[test]
    fn tables_match_synapses() {
        // after much learning, the tables' counts equal counts taken from the cells' masks
        let mut l = BitCells::new(1, 8, 64);
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for step in 0..2000usize {
            let r = row(step % 4, (step / 4) % 3);
            l.predict(&r, 64);
            l.learn(&r, &target((step * 7) % 5), &mut rng);
        }
        for input in 0..4 {
            for ctx in 0..3 {
                let r = row(input, ctx);
                let (counts, _) = l.tables.count(&r, l.cells.len());
                for (c, cell) in l.cells.iter().enumerate() {
                    let x = |s: &Seg| r.as_words()[s.w as usize];
                    let b: u32 = cell.basal.iter().map(|s| (x(s) & s.active).count_ones()).sum();
                    let bs: u32 = cell.basal.iter().map(|s| (x(s) & s.silent & !s.active).count_ones()).sum();
                    let a: u32 = cell.apical.iter().map(|s| (x(s) & s.active).count_ones()).sum();
                    assert_eq!((counts[c].0 as u32, counts[c].1 as u32, counts[c].2 as u32), (b, a, bs), "cell {c}");
                    let nb: u32 = cell.basal.iter().map(|s| s.present().count_ones()).sum();
                    let na: u32 = cell.apical.iter().map(|s| s.active.count_ones()).sum();
                    assert_eq!((cell.nb, cell.na), (nb, na), "cell {c} counts");
                }
            }
        }
        assert!(l.committed() > 0);
    }
}
