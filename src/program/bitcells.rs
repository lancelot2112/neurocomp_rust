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
    /// per row bit: bitset over cells with a synapse there
    presence: Vec<Vec<u64>>,
    cell_words: usize,
    tick: u64,
    theta: Q16,
    interneurons: bool,
    sst: Q16,
    vip: Q16,
    cached: Option<(Vec<u64>, Eval)>,
    /// (bursts won, spikes won, nothing fired, commitments, freed)
    pub stats: [u64; 5],
}

/// A random mask with each bit set with probability 2^-k.
fn rmask<R: rand::Rng + ?Sized>(rng: &mut R, k: u32) -> u64 {
    let mut m = u64::MAX;
    for _ in 0..k {
        m &= rng.next_u64();
    }
    m
}

impl BitCells {
    pub fn new(frame_words: usize, sample: usize, cells: usize) -> Self {
        let c = Cell { trace: ONE / 2, ..Default::default() };
        Self {
            frame_words,
            sample,
            cells: vec![c; cells],
            presence: Vec::new(),
            cell_words: cells.div_ceil(64).max(1),
            tick: 0,
            theta: ONE * 4 / 5,
            interneurons: false,
            sst: ONE / 2,
            vip: 0,
            cached: None,
            stats: [0; 5],
        }
    }

    /// SST/VIP interneurons set the context threshold (see `PrimedLayer5::set_interneurons`).
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
        f == 0 || f + 1 >= frames
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

    fn set_presence(&mut self, bit: u32, c: usize, on: bool) {
        let b = bit as usize;
        if self.presence.len() <= b {
            self.presence.resize(b + 1, Vec::new());
        }
        let m = &mut self.presence[b];
        if m.is_empty() {
            if !on {
                return;
            }
            m.resize(self.cell_words, 0);
        }
        if on {
            m[c / 64] |= 1 << (c % 64);
        } else {
            m[c / 64] &= !(1 << (c % 64));
        }
    }

    /// Clear presence for bits of `before` that `after` no longer has (one word).
    fn update_presence(&mut self, c: usize, w: u32, before: u64, after: u64) {
        let mut gone = before & !after;
        while gone != 0 {
            let k = gone.trailing_zeros();
            self.set_presence(w * 64 + k, c, false);
            gone &= gone - 1;
        }
        let mut new = after & !before;
        while new != 0 {
            let k = new.trailing_zeros();
            self.set_presence(w * 64 + k, c, true);
            new &= new - 1;
        }
    }

    fn carried(&self, c: usize) -> Q16 {
        let cell = &self.cells[c];
        let dt = self.tick.saturating_sub(cell.prime_tick) as u32;
        cell.prime.saturating_sub(dt.saturating_mul(ONE / 4))
    }

    fn evaluate(&self, row: &BitVector) -> Eval {
        let words = row.as_words();
        let mut touched = vec![0u64; self.cell_words];
        for (wi, &x) in words.iter().enumerate() {
            let mut x = x;
            while x != 0 {
                let b = wi * 64 + x.trailing_zeros() as usize;
                if let Some(m) = self.presence.get(b).filter(|m| !m.is_empty()) {
                    for (t, &v) in touched.iter_mut().zip(m.iter()) {
                        *t |= v;
                    }
                }
                x &= x - 1;
            }
        }
        let (mut fired, mut free_primed) = (Vec::new(), Vec::new());
        for (tw, &t) in touched.iter().enumerate() {
            let mut t = t;
            while t != 0 {
                let c = tw * 64 + t.trailing_zeros() as usize;
                t &= t - 1;
                if c >= self.cells.len() {
                    continue;
                }
                let cell = &self.cells[c];
                let (mut a, mut na) = (0u32, 0u32);
                for s in &cell.apical {
                    a += (words.get(s.w as usize).copied().unwrap_or(0) & s.active).count_ones();
                    na += s.active.count_ones();
                }
                let p = if na == 0 { 0 } else { (((a as u64) << 16) / na as u64) as Q16 }.max(self.carried(c)).min(ONE);
                let theta = self.theta_for(c);
                if cell.out.is_empty() {
                    if p >= theta && na > 0 {
                        free_primed.push((c, p));
                    }
                    continue;
                }
                let (mut b, mut bs, mut nb) = (0u32, 0u32, 0u32);
                for s in &cell.basal {
                    let x = words.get(s.w as usize).copied().unwrap_or(0);
                    b += (x & s.active).count_ones();
                    bs += (x & s.silent & !s.active).count_ones();
                    nb += s.present().count_ones();
                }
                if nb == 0 {
                    continue;
                }
                let need = (nb * 4).div_ceil(5).max(1);
                let eff = b + if p >= theta { bs } else { 0 };
                if eff >= need {
                    fired.push((c, p, p >= theta));
                }
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

    /// Apply a mask operation to every segment of one compartment of cell `c`:
    /// `op(seg, input word)` returns the new seg; presence follows.
    fn apply<F: FnMut(Seg, u64) -> Seg>(&mut self, c: usize, apical: bool, row: &[u64], mut op: F) {
        let n = if apical { self.cells[c].apical.len() } else { self.cells[c].basal.len() };
        for i in 0..n {
            let s = if apical { self.cells[c].apical[i] } else { self.cells[c].basal[i] };
            let x = row.get(s.w as usize).copied().unwrap_or(0);
            let t = op(s, x);
            let t = Seg { sticky: t.sticky & t.present(), ..t };
            if apical {
                self.cells[c].apical[i] = t;
            } else {
                self.cells[c].basal[i] = t;
            }
            if s.present() != t.present() {
                self.update_presence(c, s.w, s.present(), t.present());
            }
        }
        let segs = if apical { &mut self.cells[c].apical } else { &mut self.cells[c].basal };
        segs.retain(|s| s.present() != 0);
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
        if active {
            segs[i].active |= 1 << k;
        } else {
            segs[i].silent |= 1 << k;
        }
        self.set_presence(bit, c, true);
    }

    fn has(&self, c: usize, apical: bool, bit: u32) -> bool {
        let segs = if apical { &self.cells[c].apical } else { &self.cells[c].basal };
        segs.binary_search_by_key(&(bit / 64), |s| s.w).map_or(false, |i| segs[i].present() >> (bit % 64) & 1 == 1)
    }

    /// Grow one synapse onto an active input the cell lacks (at most 2 × sample).
    fn grow<R: rand::Rng + ?Sized>(&mut self, c: usize, apical: bool, active: &[u32], rng: &mut R) {
        if self.n_present(c, apical) as usize >= 2 * self.sample {
            return;
        }
        for _ in 0..4 {
            let Some(&b) = active.choose(rng) else { return };
            if !self.has(c, apical, b) {
                self.add_synapse(c, apical, b, apical);
                return;
            }
        }
    }

    fn free(&mut self, c: usize) {
        for apical in [false, true] {
            let segs = if apical { std::mem::take(&mut self.cells[c].apical) } else { std::mem::take(&mut self.cells[c].basal) };
            for s in segs {
                self.update_presence(c, s.w, s.present(), 0);
            }
        }
        let cell = &mut self.cells[c];
        cell.out.clear();
        cell.trace = ONE / 2;
        cell.misses = 0;
        cell.prime = 0;
        cell.bursty = 0;
        self.stats[4] += 1;
    }

    fn wire<R: rand::Rng + ?Sized>(&mut self, c: usize, apical: &[u32], rng: &mut R) {
        let old = std::mem::take(&mut self.cells[c].apical);
        for s in old {
            self.update_presence(c, s.w, s.present(), 0);
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
            if predicts(&self.cells[wc]) {
                confirmed = true;
                let cell = &mut self.cells[wc];
                cell.trace += (ONE - cell.trace.min(ONE)) / 2;
                cell.misses = 0;
                let confirm = |s: Seg, x: u64, rng: &mut R| -> Seg {
                    let promote = s.silent & x & rmask(rng, 1);
                    let active = s.active | promote;
                    let silent = s.silent & !promote;
                    let sticky = s.sticky | (active & x & rmask(rng, 3));
                    let prune = (active | silent) & !x & !sticky & rmask(rng, 4);
                    Seg { w: s.w, active: active & !prune, silent: silent & !prune, sticky }
                };
                self.apply(wc, false, &words, |s, x| confirm(s, x, rng));
                if burst {
                    self.apply(wc, true, &words, |s, x| confirm(s, x, rng));
                }
                if rmask(rng, 2) & 1 == 1 {
                    self.grow(wc, false, &basal, rng);
                }
                if burst && rmask(rng, 2) & 1 == 1 {
                    self.grow(wc, true, &apical, rng);
                }
                // lateral inhibition on the losing primed cells
                let out = self.cells[wc].out.clone();
                let losers: Vec<usize> = e.fired.iter().filter(|&&(c, p, _)| c != wc && p > 0 && self.cells[c].out != out).map(|x| x.0).collect();
                for c in losers {
                    self.apply(c, true, &words, |s, x| {
                        let prune = s.active & x & !s.sticky & rmask(rng, 3);
                        Seg { active: s.active & !prune, ..s }
                    });
                }
            } else {
                let cell = &mut self.cells[wc];
                cell.trace -= cell.trace / 2;
                cell.misses = cell.misses.saturating_add(1);
                let fail = cell.misses >= 4 && cell.trace < ONE / 8;
                if burst {
                    self.apply(wc, true, &words, |s, x| {
                        let prune = s.active & x & !s.sticky & rmask(rng, 2);
                        let unstick = s.sticky & x & rmask(rng, 3);
                        Seg { active: s.active & !prune, sticky: s.sticky & !unstick, ..s }
                    });
                }
                if fail {
                    self.free(wc);
                }
            }
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
    fn presence_matches_synapses() {
        // after much learning, the presence bitsets equal the cells' synapses exactly
        let mut l = BitCells::new(1, 8, 64);
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for step in 0..2000usize {
            let r = row(step % 4, (step / 4) % 3);
            l.predict(&r, 64);
            l.learn(&r, &target((step * 7) % 5), &mut rng);
        }
        for (b, m) in l.presence.iter().enumerate() {
            for c in 0..l.cells.len() {
                let marked = !m.is_empty() && m[c / 64] >> (c % 64) & 1 == 1;
                let has = l.has(c, false, b as u32) || l.has(c, true, b as u32);
                assert_eq!(marked, has, "bit {b} cell {c}");
            }
        }
        assert!(l.committed() > 0);
    }
}
