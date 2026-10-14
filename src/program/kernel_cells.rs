//! The kernel: cells that are binary masks over a high-dimensional bit vector, born as needed,
//! separated by Hebbian learning and inhibition between the cells an input matches, writing a
//! sparse distributed code through their own output masks (`Kernel`).
//!
//! - **A cell** is a mask over the input (its synapses, one bit each) and a mask over the output
//!   (the bits it writes when it fires), with the operation it writes them by (`Write`: OR by
//!   default; XOR, AND-NOT or an add to the output, so that codes can do work as they are
//!   written).
//! - **Firing** is local logic: a cell fires when at least three quarters of its own synapses are
//!   on the input. Its threshold is a share of its own mask, so a cell that has come to take in
//!   several inputs (more synapses) needs more of them together and fires for none alone: there is
//!   no rate to keep (homeostasis without a hidden variable).
//! - **Inhibition between cells whose masks overlap**: of the cells that fire, a cell is held down
//!   by a better-matched one whose mask overlaps its own (more than half the smaller), as cells
//!   competing for the same lines share their interneurons; cells on different parts of the
//!   input both fire. The held-down cells, which matched, unlearn the input (anti-Hebbian: their
//!   synapses on its lines go, each at random), so overlapping inputs come apart.
//! - **Hebbian**: a winner's synapses on lines that are off move to lines of the input it has
//!   none on yet, each at random; **anti-Hebbian** moves a held-down cell's synapses on the
//!   input's lines to lines that were not active. Learning only moves synapses: every cell keeps
//!   as many as it was born with (its density is kept), so none shrinks into firing for
//!   anything.
//! - **Births**: the part of an input no winner covers (the input AND NOT their masks), when it
//!   meets one cell's popcount (all of it when no cell fires), gets `k` new cells,
//!   its lines dealt among them (so cells born together share no lines and do not inhibit each
//!   other), each writing its own random output bits. So a new
//!   pattern gets cells of its own, and what it shares with an old one does not absorb it.
//!
//! - **Conjunctions in time** (`with_history`): the kernel keeps its last few inputs (a
//!   history), and a cell may also look back: it fires only when its synapses on the input hold
//!   and three quarters of its `then` synapses are on the input from `lag` inputs back. Cells
//!   born for an input start at different lags. A cell that has not fired yet, whose input holds
//!   but whose look back does not, moves to another lag and wires to what was there (it searches
//!   for the conjunction that recurs). A cell that has fired, whose input holds but whose look
//!   back does not, has met its familiar now after an unfamiliar then: a new sequence, which gets
//!   a cell of its own at the same lag (and is new to the area).
//!
//! All bitwise: masks, popcounts, compares.

use crate::bitvec::BitVector;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

/// How a cell writes its output bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Write {
    Or,
    Xor,
    AndNot,
    /// add the output mask to the output as an integer (bit 0 of word 0 least significant), with
    /// carry
    Add,
}

#[derive(Clone)]
pub struct Cell {
    pub syn: Vec<u64>,
    pub out: Vec<u64>,
    pub write: Write,
    /// a label the user may give it (an assembly's, say)
    pub label: Option<usize>,
    /// a conjunction cell also needs a pattern from `lag` inputs back (`then`, over the input as
    /// it was then; lag 0: none), and whether it has ever fired
    pub lag: usize,
    pub then: Vec<u64>,
    pub fired: bool,
}

#[derive(Clone)]
pub struct Kernel {
    pub bits: usize,
    out_bits: usize,
    pub k: usize,
    /// output bits each cell writes
    out_k: usize,
    pub cells: Vec<Cell>,
    pub rng: StdRng,
    /// cells born so far
    pub births: u64,
    /// conjunctions: how many inputs back cells may look (0: none), and the inputs that came,
    /// newest first
    depth: usize,
    hist: std::collections::VecDeque<Vec<u64>>,
    /// only conjunction cells write the output (what came after what), plain cells code and
    /// learn but do not send
    pub conj_only: bool,
}

/// The bits one coding cell writes (the popcount a new part must meet to be a cell of its own).
const ONE_CELL: u32 = 8;

fn words(bits: usize) -> usize {
    bits.div_ceil(64)
}

fn pop(a: &[u64]) -> u32 {
    a.iter().map(|w| w.count_ones()).sum()
}

fn and_pop(a: &[u64], b: &[u64]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x & y).count_ones()).sum()
}

impl Kernel {
    /// Cells over `bits` input lines writing into `out_bits` output bits, `out_k` bits each, `k`
    /// winning an input.
    pub fn new(bits: usize, out_bits: usize, k: usize, out_k: usize, seed: u64) -> Self {
        Self { bits, out_bits, k, out_k, cells: Vec::new(), rng: StdRng::seed_from_u64(seed), births: 0, depth: 0, hist: Default::default(), conj_only: false }
    }

    pub fn rand_mask(&mut self, p: u32) -> u64 {
        let mut m = !0u64;
        for _ in 0..p {
            m &= self.rng.next_u64();
        }
        m
    }

    /// Whether cell `c` fires for the input (three quarters of its own synapses on it), and how
    /// many of its synapses are.
    pub fn match_of(&self, c: usize, input: &[u64]) -> (bool, u32) {
        let cell = &self.cells[c];
        let n = pop(&cell.syn);
        let on = and_pop(&cell.syn, input);
        (n > 0 && 4 * on >= 3 * n, on)
    }

    /// Give birth to a cell wired to a random half of the input's lines (at least one), writing
    /// `out_k` random output bits; returns its index.
    pub fn birth(&mut self, input: &[u64], label: Option<usize>) -> usize {
        self.birth_on(input.to_vec(), label)
    }

    /// Give birth to `n` cells for an input, its lines dealt among them at random (each line to
    /// one cell), so cells born together share no lines and do not inhibit each other; returns
    /// their indices.
    pub fn births_for(&mut self, input: &[u64], n: usize, label: Option<usize>) -> Vec<usize> {
        let lines = Self::ones(input);
        let mut parts: Vec<Vec<u64>> = vec![vec![0u64; input.len()]; n.max(1)];
        for b in lines {
            let j = (self.rng.next_u64() % parts.len() as u64) as usize;
            parts[j][b / 64] |= 1u64 << (b % 64);
        }
        parts.into_iter().filter(|m| pop(m) > 0).map(|m| self.birth_on(m, label)).collect()
    }

    /// A cell wired to exactly `syn`, writing `out_k` random output bits.
    fn birth_on(&mut self, syn: Vec<u64>, label: Option<usize>) -> usize {
        let mut out = vec![0u64; words(self.out_bits)];
        let mut n = 0;
        while n < self.out_k {
            let b = (self.rng.next_u64() % self.out_bits as u64) as usize;
            if out[b / 64] >> (b % 64) & 1 == 0 {
                out[b / 64] |= 1 << (b % 64);
                n += 1;
            }
        }
        self.cells.push(Cell { syn, out, write: Write::Or, label, lag: 0, then: Vec::new(), fired: false });
        self.births += 1;
        self.cells.len() - 1
    }

    /// Conjunction cells: born cells also look `depth` inputs back at most (see `code`).
    pub fn with_history(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }

    /// `n` lines of `frame` at random (all of them if fewer).
    fn sample(&mut self, frame: &[u64], n: usize) -> Vec<u64> {
        let mut lines = Self::ones(frame);
        let mut v = vec![0u64; frame.len()];
        for i in 0..n.min(lines.len()) {
            let j = i + (self.rng.next_u64() % (lines.len() - i) as u64) as usize;
            lines.swap(i, j);
            v[lines[i] / 64] |= 1u64 << (lines[i] % 64);
        }
        v
    }

    /// Cell `c` looks `lag` inputs back, wired to as many of that input's lines as it has
    /// synapses now (none if that input is not in the history: it is then a plain cell).
    fn wire_then(&mut self, c: usize, lag: usize) {
        match self.hist.get(lag.wrapping_sub(1)).cloned() {
            Some(frame) if lag > 0 && pop(&frame) > 0 => {
                let n = pop(&self.cells[c].syn) as usize;
                let then = self.sample(&frame, n);
                self.cells[c].lag = lag;
                self.cells[c].then = then;
            }
            _ => {
                self.cells[c].lag = 0;
                self.cells[c].then.clear();
            }
        }
    }

    /// Whether cell `c`'s look back holds (three quarters of its `then` synapses on the input
    /// that came `lag` inputs back; always, for a plain cell).
    fn then_holds(&self, c: usize) -> bool {
        let cell = &self.cells[c];
        if cell.lag == 0 {
            return true;
        }
        match self.hist.get(cell.lag - 1) {
            Some(frame) => {
                let n = pop(&cell.then);
                n > 0 && 4 * and_pop(&cell.then, frame) >= 3 * n
            }
            None => false,
        }
    }

    /// The bit positions set in `mask`.
    pub fn ones(mask: &[u64]) -> Vec<usize> {
        let mut v = Vec::new();
        for (w, &x) in mask.iter().enumerate() {
            let mut x = x;
            while x != 0 {
                v.push(w * 64 + x.trailing_zeros() as usize);
                x &= x - 1;
            }
        }
        v
    }

    /// Move synapses of cell `c`: each synapse in `from` (a mask) moves, with probability 2^-p,
    /// to a random line in `to` (a mask) where the cell has none yet. The cell keeps as many
    /// synapses as it had (its density is kept): learning only moves them.
    fn move_synapses(&mut self, c: usize, from: &[u64], to: &[u64], p: u32) {
        let syn = self.cells[c].syn.clone();
        let movable: Vec<u64> = syn.iter().zip(from).map(|(s, f)| s & f).collect();
        let free: Vec<u64> = syn.iter().zip(to).map(|(s, t)| !s & t).collect();
        let mut dest = Self::ones(&free);
        if dest.is_empty() {
            return;
        }
        for b in Self::ones(&movable) {
            if self.rand_mask(p) & 1 == 0 || dest.is_empty() {
                continue;
            }
            let i = (self.rng.next_u64() % dest.len() as u64) as usize;
            let d = dest.swap_remove(i);
            let syn = &mut self.cells[c].syn;
            syn[b / 64] &= !(1u64 << (b % 64));
            syn[d / 64] |= 1u64 << (d % 64);
        }
    }

    /// Hebbian: cell `c`'s synapses on lines that are off move to lines of the input it has
    /// none on yet (each with probability 2^-p): its mask moves toward the input, as many
    /// synapses as before.
    pub fn hebbian(&mut self, c: usize, input: &[u64], p: u32) {
        let off: Vec<u64> = input.iter().map(|x| !x).collect();
        self.move_synapses(c, &off, input, p);
    }

    /// Anti-Hebbian: cell `c`'s synapses on the input's lines move to lines that were not active
    /// (each with probability 2^-p): it stops answering to this input, as many synapses as
    /// before, so it does not come to fire for everything.
    pub fn anti_hebbian(&mut self, c: usize, input: &[u64], p: u32) {
        let mut off: Vec<u64> = input.iter().map(|x| !x).collect();
        // (only lines inside the input's width)
        let bits = self.bits;
        if bits % 64 != 0 {
            if let Some(last) = off.last_mut() {
                *last &= (1u64 << (bits % 64)) - 1;
            }
        }
        self.move_synapses(c, input, &off, p);
    }

    /// Write cell `c`'s output into `out` by its operation.
    pub fn write_out(&self, c: usize, out: &mut [u64]) {
        let cell = &self.cells[c];
        match cell.write {
            Write::Or => out.iter_mut().zip(&cell.out).for_each(|(o, m)| *o |= m),
            Write::Xor => out.iter_mut().zip(&cell.out).for_each(|(o, m)| *o ^= m),
            Write::AndNot => out.iter_mut().zip(&cell.out).for_each(|(o, m)| *o &= !m),
            Write::Add => {
                let mut carry = 0u64;
                for (o, m) in out.iter_mut().zip(&cell.out) {
                    let (s1, c1) = o.overflowing_add(*m);
                    let (s2, c2) = s1.overflowing_add(carry);
                    *o = s2;
                    carry = (c1 || c2) as u64;
                }
            }
        }
    }

    /// Code an input: the cells that fire compete with those whose masks overlap theirs (the
    /// better-matched win), the winners learn it and the held-down ones unlearn it (with
    /// `learn`), an input no cell fires for gets `k` new cells (with `learn`), and the winners
    /// write the output. Returns the
    /// output and the winners.
    pub fn code(&mut self, input: &BitVector, learn: bool) -> (BitVector, Vec<usize>) {
        let mut inw = input.as_words().to_vec();
        inw.resize(words(self.bits), 0);
        // a cell fires when its synapses on the input hold and, a conjunction cell, its look back
        // holds too; one whose input holds but whose look back does not is noted
        let mut fired: Vec<(u32, usize)> = Vec::new();
        let mut unmet: Vec<usize> = Vec::new();
        for c in 0..self.cells.len() {
            let (f, on) = self.match_of(c, &inw);
            if f {
                if self.then_holds(c) {
                    fired.push((on, c));
                } else {
                    unmet.push(c);
                }
            }
        }
        fired.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        // inhibition is local: a fired cell is held down by a better-matched one whose mask
        // overlaps its own (more than half the smaller), as cells competing for the same lines
        // share their inhibition; cells on different parts of the input both fire
        let mut winners: Vec<usize> = Vec::new();
        let mut held: Vec<usize> = Vec::new();
        for &(_, c) in &fired {
            let mine = &self.cells[c].syn;
            let n = pop(mine);
            let rivals = winners.iter().any(|&w| {
                let theirs = &self.cells[w].syn;
                2 * and_pop(mine, theirs) > n.min(pop(theirs))
            });
            if rivals {
                held.push(c);
            } else {
                winners.push(c);
            }
        }
        // the cells that fired by matching (newborn cells have not fired yet)
        let matched: Vec<usize> = winners.clone();
        if learn && self.depth > 0 {
            let mut spawned_lags: Vec<usize> = Vec::new();
            for c in unmet {
                if !self.cells[c].fired {
                    // not yet fired: it moves its look back to another lag and wires to what was
                    // there (searching for the conjunction that recurs)
                    let lag = 1 + (self.rng.next_u64() % self.depth as u64) as usize;
                    self.wire_then(c, lag);
                } else {
                    // a conjunction it knows, with something else before it: the familiar now
                    // after an unfamiliar then is a new conjunction, and gets a cell of its own at
                    // the same lag (counted as a birth: the area has not seen this sequence) —
                    // unless a cell at that lag that fired already covers it, and one a lag
                    let lag = self.cells[c].lag;
                    let covered = matched.iter().any(|&w| self.cells[w].lag == lag && 2 * and_pop(&self.cells[w].syn, &self.cells[c].syn) > pop(&self.cells[c].syn));
                    if covered || spawned_lags.contains(&lag) {
                        continue;
                    }
                    spawned_lags.push(lag);
                    let now: Vec<u64> = self.cells[c].syn.iter().zip(&inw).map(|(m, x)| m & x).collect();
                    let n = self.birth_on(now, None);
                    self.wire_then(n, lag);
                    winners.push(n);
                }
            }
        }
        if learn && pop(&inw) > 0 {
            // what of the input no winner covers (the input AND NOT their masks): if it meets the
            // popcount of one cell's output, that part gets cells of its own (every part of an
            // input is coded)
            let mut uncovered = inw.clone();
            for &c in &winners {
                for (u, m) in uncovered.iter_mut().zip(&self.cells[c].syn) {
                    *u &= !m;
                }
            }
            if pop(&uncovered) >= ONE_CELL.min(pop(&inw)) && !winners.is_empty() {
                for &c in &winners {
                    self.hebbian(c, &inw, 3);
                }
                let born = self.births_for(&uncovered, self.k, None);
                for (i, &c) in born.iter().enumerate() {
                    if self.depth > 0 {
                        self.wire_then(c, 1 + i % self.depth);
                    }
                }
                winners.extend(born);
            } else if winners.is_empty() {
                winners = self.births_for(&inw, self.k, None);
                for (i, &c) in winners.clone().iter().enumerate() {
                    if self.depth > 0 {
                        self.wire_then(c, 1 + i % self.depth);
                    }
                }
            } else {
                for &c in &winners {
                    self.hebbian(c, &inw, 3);
                }
                for &c in &held {
                    self.anti_hebbian(c, &inw, 1);
                }
            }
        }
        for &c in &matched {
            self.cells[c].fired = true;
        }
        if self.depth > 0 {
            self.hist.push_front(inw.clone());
            self.hist.truncate(self.depth);
        }
        let mut out = vec![0u64; words(self.out_bits)];
        for &c in &winners {
            if !self.conj_only || self.cells[c].lag > 0 {
                self.write_out(c, &mut out);
            }
        }
        (BitVector::from_words(out), winners)
    }
}

/// Holding with kernel cells: an assembly is the cells born into it (one set per kind of input
/// it has explained) and holds itself (its label stays active) with no input. An input is
/// explained by an assembly when one of its cells fires for it (local: three quarters of that
/// cell's own synapses on the input).
///
/// - While the active assembly is young (its first 16 explained inputs: a context is learned at
///   its start), an input it does not explain gets cells of its own born into it, and no rival
///   takes over from it.
/// - Once settled, only rivals contradict it. A stored assembly that explains an input the active
///   one does not gains drive against it; an input both explain carries no context and moves
///   nothing. Every rival's drive leaks, one at random in sixteen inputs, so only what comes
///   often under one assembly and not under the other builds up (a distractor that comes in every
///   context now and then never does).
/// - Something new while settled (its kernel cells were born for it), when its new part (the
///   input AND NOT what the active assembly covers) meets the popcount (one cell's output: a
///   whole new cell firing), seeds a candidate assembly wired to that part, beside the active one, a possible
///   new cause, with that input as its first drive (a smaller novelty is learned into the active
///   one): if it
///   keeps coming it takes over (the first reversal); if not, the leak keeps it down. Something
///   familiar that no assembly explains is background, learned into the active one.
/// - A rival whose drive reaches the active assembly's strength (the bit length of how many
///   inputs it has explained: what has held longer needs more) takes over.
pub struct Holder {
    pub kernel: Kernel,
    /// per assembly, the inputs it has explained (saturating)
    explained: Vec<u8>,
    rival: Vec<u8>,
    active: Option<usize>,
    pub now: usize,
    pub switches: u32,
    pub resets: u32,
    pub debug: bool,
    /// matched cells of assemblies held down by the active one learn anti-Hebbian
    pub separate: bool,
}

impl Holder {
    pub fn new(bits: usize, seed: u64) -> Self {
        Self { kernel: Kernel::new(bits, 64, 1, 1, seed), explained: Vec::new(), rival: Vec::new(), active: None, now: 0, switches: 0, resets: 0, debug: false, separate: false }
    }

    pub fn contexts(&self) -> usize {
        self.explained.len()
    }

    pub fn unsettled(&self) -> bool {
        self.active.is_none()
    }

    /// The cells of the active assembly (none while silent).
    pub fn active_cells(&self) -> Vec<usize> {
        match self.active {
            Some(a) => (0..self.kernel.cells.len()).filter(|&c| self.kernel.cells[c].label == Some(a)).collect(),
            None => Vec::new(),
        }
    }

    /// The assemblies whose cells fire for the input.
    /// The assemblies that explain the input: one of their cells fires for it.
    fn explainers(&self, inw: &[u64]) -> Vec<bool> {
        let mut e = vec![false; self.explained.len()];
        for c in 0..self.kernel.cells.len() {
            if let Some(l) = self.kernel.cells[c].label {
                if !e[l] && self.kernel.match_of(c, inw).0 {
                    e[l] = true;
                }
            }
        }
        e
    }

    fn enter(&mut self, l: usize) {
        self.active = Some(l);
        self.rival.iter_mut().for_each(|r| *r = 0);
        if l != self.now {
            self.switches += 1;
            self.now = l;
        }
    }

    /// One input, and whether it is novel (new to the area: the cells that code it were born
    /// for it); returns the assembly now active (or last active). A novel input is learned into
    /// the active assembly (acetylcholine's encoding mode: what is new is learned, not taken as
    /// a contradiction); only a familiar input the active assembly does not explain counts
    /// against it.
    pub fn hold(&mut self, input: &BitVector, novel: bool) -> usize {
        let mut inw = input.as_words().to_vec();
        inw.resize(words(self.kernel.bits), 0);
        if pop(&inw) == 0 {
            return self.now;
        }
        let e = self.explainers(&inw);
        self.rival.resize(self.explained.len(), 0);
        match self.active {
            Some(a) => {
                let strength = 32 - (self.explained[a] as u32).leading_zeros() as u8;
                let young = self.explained[a] < 16;
                if e[a] {
                    self.explained[a] = self.explained[a].saturating_add(1);
                    // (`separate`: the other assemblies' cells that matched it were held down by
                    // the active one, and learn anti-Hebbian: pattern separation, so an input
                    // stays with one assembly)
                    if self.separate {
                        for c in 0..self.kernel.cells.len() {
                            if self.kernel.cells[c].label.map_or(false, |l| l != a) && self.kernel.match_of(c, &inw).0 {
                                self.kernel.anti_hebbian(c, &inw, 1);
                            }
                        }
                    }
                } else if young {
                    // young: it learns what comes
                    self.kernel.births_for(&inw, 4, Some(a));
                } else if novel {
                    // settled, and something new. What is new about it is the input AND NOT the
                    // lines the active assembly's cells cover. When that meets the popcount (one
                    // cell's output: a whole new cell is firing, not a shift of a few bits), it
                    // seeds a candidate assembly wired to it, a possible new cause, stored beside
                    // the active one with that input as its first drive; else it is a small
                    // novelty, learned into the active one
                    let mut covered = vec![0u64; inw.len()];
                    for c in 0..self.kernel.cells.len() {
                        if self.kernel.cells[c].label == Some(a) {
                            for (w, m) in covered.iter_mut().zip(&self.kernel.cells[c].syn) {
                                *w |= m;
                            }
                        }
                    }
                    let own: Vec<u64> = inw.iter().zip(&covered).map(|(x, c)| x & !c).collect();
                    if pop(&own) >= ONE_CELL {
                        let l = self.explained.len();
                        self.explained.push(0);
                        self.rival.push(1);
                        self.kernel.births_for(&own, 4, Some(l));
                    } else {
                        self.kernel.births_for(&inw, 4, Some(a));
                    }
                } else if !(0..e.len()).any(|l| l != a && e[l]) {
                    // familiar, and tied to no context: background, learned into the active one
                    self.kernel.births_for(&inw, 4, Some(a));
                }
                // only rivals contradict: a stored assembly that explains an input the active one
                // does not gains drive against it (an input both explain carries no context);
                // every rival's drive leaks, one at random in sixteen inputs, so only what comes
                // often in one context and not in the other builds up
                let leak = self.kernel.rand_mask(4) & 1 == 1;
                let mut winner: Option<usize> = None;
                for l in 0..self.explained.len() {
                    if l == a {
                        continue;
                    }
                    if e.get(l).copied().unwrap_or(false) && !e[a] {
                        self.rival[l] = self.rival[l].saturating_add(1);
                    }
                    if leak {
                        self.rival[l] = self.rival[l].saturating_sub(1);
                    }
                    // (a young assembly is still learning what comes: no rival takes over from it)
                    if !young && self.rival[l] >= strength && winner.map_or(true, |w| self.rival[l] > self.rival[w]) {
                        winner = Some(l);
                    }
                }
                if let Some(l) = winner {
                    self.resets += 1;
                    self.enter(l);
                }
            }
            None => {
                // nothing held yet: an assembly that explains the input comes on; none does, a
                // new one
                if let Some(l) = (0..e.len()).find(|&l| e[l]) {
                    self.enter(l);
                } else {
                    let l = self.explained.len();
                    self.explained.push(0);
                    self.rival.push(0);
                    self.kernel.births_for(&inw, 4, Some(l));
                    self.enter(l);
                }
            }
        }
        if self.debug {
            eprintln!("holder: explained {:?} by {:?} rivals {:?} -> {:?}", self.explained, e, self.rival, self.active);
        }
        self.now
    }
}

/// Prediction by kernel cells, learned by local rules: each cell reads the context (what came
/// before: a trace) through its synapses and, firing, writes its prediction of the next input
/// through its output mask. The prediction is what the firing cells write (ORed); the error is
/// the input XOR the prediction: what came unpredicted and what was predicted but did not come.
///
/// The firing cells do not hold each other down: a context followed by either of two things
/// predicts both, and the error is then the one that did not come.
///
/// Learning is local to each cell, and its output is fixed at birth:
/// - a cell that fired and whose output came (most of it) moves its synapses toward the context
///   it fired on (Hebbian); one whose output did not come moves them off it (anti-Hebbian), so a
///   cell for what no longer follows its context stops firing on it;
/// - when the unpredicted part meets one cell's popcount, `k` cells are born on the context (its
///   lines dealt among them), each writing that part.
pub struct Predictor {
    pub kernel: Kernel,
    in_bits: usize,
}

impl Predictor {
    /// Cells over `ctx_bits` context lines predicting `in_bits` input lines.
    pub fn new(ctx_bits: usize, in_bits: usize, seed: u64) -> Self {
        Self { kernel: Kernel::new(ctx_bits, in_bits, 4, 8, seed), in_bits }
    }

    /// One step: the context before and the input that came; returns the error (input XOR
    /// prediction), learning.
    pub fn step(&mut self, context: &BitVector, input: &BitVector) -> BitVector {
        let mut cw = context.as_words().to_vec();
        cw.resize(words(self.kernel.bits), 0);
        let mut iw = input.as_words().to_vec();
        iw.resize(words(self.in_bits), 0);
        // the cells the context fires: each predicts what it writes, and they are not held down
        // by each other (a context followed by either of two things predicts both)
        let winners: Vec<usize> = (0..self.kernel.cells.len()).filter(|&c| self.kernel.match_of(c, &cw).0).collect();
        let mut pred = vec![0u64; iw.len()];
        for &c in &winners {
            for (p, o) in pred.iter_mut().zip(&self.kernel.cells[c].out) {
                *p |= o;
            }
        }
        let error: Vec<u64> = iw.iter().zip(&pred).map(|(x, p)| x ^ p).collect();
        // local learning: a cell whose output came (most of it) moves its synapses toward the
        // context it fired on; one whose output did not come moves them off it
        for &c in &winners {
            let out = &self.kernel.cells[c].out;
            let came = and_pop(out, &iw);
            if 2 * came >= pop(out) {
                self.kernel.hebbian(c, &cw, 3);
            } else {
                self.kernel.anti_hebbian(c, &cw, 3);
            }
        }
        let unpredicted: Vec<u64> = iw.iter().zip(&pred).map(|(x, p)| x & !p).collect();
        if pop(&unpredicted) >= ONE_CELL.min(pop(&iw).max(1)) && pop(&cw) > 0 {
            let born = self.kernel.births_for(&cw, self.kernel.k, None);
            for c in born {
                self.kernel.cells[c].out = unpredicted.clone();
            }
        }
        BitVector::from_words(error)
    }
}
