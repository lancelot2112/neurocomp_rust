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
//!   is more than a quarter of the input (all of it when no cell fires), gets `k` new cells,
//!   its lines dealt among them (so cells born together share no lines and do not inhibit each
//!   other), each writing its own random output bits. So a new
//!   pattern gets cells of its own, and what it shares with an old one does not absorb it.
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
}

#[derive(Clone)]
pub struct Kernel {
    bits: usize,
    out_bits: usize,
    k: usize,
    /// output bits each cell writes
    out_k: usize,
    pub cells: Vec<Cell>,
    rng: StdRng,
    /// cells born so far
    pub births: u64,
}

/// The fewest new lines a candidate assembly's cells are wired to (else the whole input).
const MIN_SYN: u32 = 8;

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
        Self { bits, out_bits, k, out_k, cells: Vec::new(), rng: StdRng::seed_from_u64(seed), births: 0 }
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
        self.cells.push(Cell { syn, out, write: Write::Or, label });
        self.births += 1;
        self.cells.len() - 1
    }

    /// The bit positions set in `mask`.
    fn ones(mask: &[u64]) -> Vec<usize> {
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
        let mut fired: Vec<(u32, usize)> = (0..self.cells.len()).filter_map(|c| {
            let (f, on) = self.match_of(c, &inw);
            f.then_some((on, c))
        }).collect();
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
        if learn && pop(&inw) > 0 {
            // what of the input no winner covers (the input AND NOT their masks): if more than a
            // quarter of it, that part gets cells of its own (every part of an input is coded)
            let mut uncovered = inw.clone();
            for &c in &winners {
                for (u, m) in uncovered.iter_mut().zip(&self.cells[c].syn) {
                    *u &= !m;
                }
            }
            if 4 * pop(&uncovered) > pop(&inw) {
                for &c in &winners {
                    self.hebbian(c, &inw, 3);
                }
                let born = self.births_for(&uncovered, self.k, None);
                winners.extend(born);
            } else if winners.is_empty() {
                winners = self.births_for(&inw, self.k, None);
            } else {
                for &c in &winners {
                    self.hebbian(c, &inw, 3);
                }
                for &c in &held {
                    self.anti_hebbian(c, &inw, 1);
                }
            }
        }
        let mut out = vec![0u64; words(self.out_bits)];
        for &c in &winners {
            self.write_out(c, &mut out);
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
/// - Something new while settled (its kernel cells were born for it) seeds a candidate assembly
///   beside the active one, a possible new cause, with that input as its first drive: if it
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
}

impl Holder {
    pub fn new(bits: usize, seed: u64) -> Self {
        Self { kernel: Kernel::new(bits, 64, 1, 1, seed), explained: Vec::new(), rival: Vec::new(), active: None, now: 0, switches: 0, resets: 0, debug: false }
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
                } else if young {
                    // young: it learns what comes
                    self.kernel.births_for(&inw, 4, Some(a));
                } else if novel {
                    // settled, and something new: it seeds a candidate assembly of its own (a
                    // possible new cause), stored beside the active one, with that input as its
                    // first drive
                    // its cells are wired to what is new about the input: the input AND NOT the
                    // lines the active assembly's cells cover (the distinctive part)
                    let mut covered = vec![0u64; inw.len()];
                    for c in 0..self.kernel.cells.len() {
                        if self.kernel.cells[c].label == Some(a) {
                            for (w, m) in covered.iter_mut().zip(&self.kernel.cells[c].syn) {
                                *w |= m;
                            }
                        }
                    }
                    let own: Vec<u64> = inw.iter().zip(&covered).map(|(x, c)| x & !c).collect();
                    let wire = if pop(&own) >= MIN_SYN { own } else { inw.clone() };
                    let l = self.explained.len();
                    self.explained.push(0);
                    self.rival.push(1);
                    self.kernel.births_for(&wire, 4, Some(l));
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
