//! One kind of population for coding and for holding: cells with binary synapses, learned
//! Hebbian and anti-Hebbian at random, under inhibition (`Assembly`). The same cells and rules
//! serve two uses, as the cortex uses one circuit for both:
//!
//! - **Coding** (`Assembly::coding`; layer 4, the dentate gyrus): each input fires the `k` cells
//!   it drives most (feedback inhibition), and learning separates overlapping inputs; the code
//!   is the cells that fired.
//! - **Holding** (`Assembly::holding`; layer 2/3's held population): `k` cells recruited
//!   together form an assembly that sustains itself through recurrent synapses with no input
//!   (persistent activity), and falls silent when inhibition reaches its recurrent strength;
//!   which assembly holds is the area's state (its context).
//!
//! **No counters in the synapses.** Every synapse is one bit, as a single synapse is close to
//! all-or-none (Petersen et al. 1998), and learning flips it at random: a random mask with
//! probability 2^-p per bit is the AND of p random words. How strongly an input reaches a
//! population is how many of its cells have the synapse (connectivity): a synapse set with
//! probability q whenever its line comes as the cell fires, and cleared with the same
//! probability whenever the cell fires without it, is on about as often as the line comes with
//! the cell (Fusi's binary synapses), so over an assembly's cells the count of synapses from a
//! line follows how often the line comes while the assembly holds. How often a cell has fired
//! lately is a window into a bitfield: each cell keeps its last 64 inputs as a shift register
//! (fired or not), and its rate is that register's popcount.
//!
//! **Coding.** A cell has potential synapses (silent: a random fan, as a young cell's
//! connections) and learned ones (active). Its drive is twice its active synapses on the input
//! plus its silent ones, less how far its window says it fires above its share
//! (homeostasis: a cell that would fire for everything is held back until cells for one input
//! alone win). The winners learn the input (Hebbian: set) and let go of what they fire without
//! (cleared); cells that matched and were held down are depressed on it (anti-Hebbian). Its
//! plasticity closes as its experience doubles (a critical period); a novel input, more than a
//! quarter of whose lines none of its winners has learned, recruits cells of its own for the new
//! part and depresses the winners there (so what it shares does not absorb it).
//!
//! **Holding.** An input's net drive on a set of cells, in bit lengths: per input line, the bit
//! length of how many of the cells have the synapse, averaged over the input's lines (the log
//! before the mean: a few familiar lines do not vouch for the rest), less the bit length of the
//! mean over every line seen (a log-likelihood against a flat expectation: feedforward
//! inhibition). The dendrite-targeting interneurons sum the active assembly's negative net drive,
//! positive drive taking it back off (an opponent count, at least zero, less one bit each
//! input: only an input at least twice as unlikely as expected counts); each stored rival
//! assembly the input drives more, net, gathers drive against the active one the same way. When
//! either reaches the active assembly's recurrent strength (the bit length of how many of its cells'
//! pairs are connected: a young assembly falls easily, a long-held one needs more) it falls
//! silent and its cells adapt for as many inputs; a rival that won takes over at once. Silent,
//! the stored assembly the input drives most, net, over its cells seeds a settle and
//! recurrence completes it (an old state comes back on one input that only it predicts); when nothing seeds by
//! the time the silenced cells recover, the least recently used cells form a new assembly. The
//! active assembly learns an input that drives it (within the allowance), not one that
//! inhibits it; a rival the input drove while the active one held it down is depressed.
//!
//! All integer and bitwise: masks, popcounts, bit lengths, compares.

use crate::bitvec::BitVector;
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

fn bitlen(x: u32) -> i32 {
    32 - x.leading_zeros() as i32
}

/// What holding adds: recurrent synapses and the assemblies they make.
#[derive(Clone)]
struct Hold {
    /// recurrent synapses: per cell, a mask over the cells it is connected from
    rec: Vec<u64>,
    /// the assembly each cell was recruited into
    label: Vec<Option<usize>>,
    /// when each cell was last active (the input count), and inputs until its adaptation wears off
    last: Vec<u64>,
    adapted: Vec<u32>,
    /// the inhibition on the active assembly from the unknown, and each rival's drive against it
    sst: i32,
    rival: Vec<i32>,
    /// inputs silent with nothing to seed a settle, and how long the assembly silenced last
    /// stays adapted
    silent: u32,
    last_strength: u32,
    now: usize,
    contexts: usize,
    switches: u32,
    resets: u32,
}

#[derive(Clone)]
pub struct Assembly {
    n: usize,
    k: usize,
    bits: usize,
    words: usize,
    /// learned synapses (active) and potential ones (silent), per cell a mask over the input
    syn: Vec<u64>,
    pot: Vec<u64>,
    /// the input lines seen so far, and how many
    seen: Vec<u64>,
    n_seen: u32,
    /// per cell, its last 64 inputs (fired or not), newest in bit 0
    fired: Vec<u64>,
    /// the cells active now
    active: Vec<usize>,
    /// inputs so far (coding: plasticity closes as they double)
    clock: u64,
    hold: Option<Hold>,
    rng: StdRng,
    pub debug: bool,
}

impl Assembly {
    fn base(n: usize, k: usize, bits: usize, seed: u64) -> Self {
        let words = bits.div_ceil(64);
        Self { n, k, bits, words, syn: vec![0; n * words], pot: vec![0; n * words], seen: vec![0; words], n_seen: 0, fired: vec![0; n], active: Vec::new(), clock: 0, hold: None, rng: StdRng::seed_from_u64(seed), debug: false }
    }

    /// For coding: `n` cells over `bits` input lines, `k` firing; each cell starts with potential
    /// synapses from a random `fan` of the lines.
    pub fn coding(n: usize, k: usize, bits: usize, fan: usize, seed: u64) -> Self {
        let mut a = Self::base(n, k, bits, seed);
        for i in 0..n {
            for _ in 0..fan {
                let b = (a.rng.next_u64() % bits as u64) as usize;
                a.pot[i * a.words + b / 64] |= 1u64 << (b % 64);
            }
        }
        a
    }

    /// For holding: `n` cells over `bits` input lines, assemblies of `k`.
    pub fn holding(n: usize, k: usize, bits: usize, seed: u64) -> Self {
        let mut a = Self::base(n, k, bits, seed);
        let rw = n.div_ceil(64);
        a.hold = Some(Hold { rec: vec![0; n * rw], label: vec![None; n], last: vec![0; n], adapted: vec![0; n], sst: 0, rival: Vec::new(), silent: 0, last_strength: 0, now: 0, contexts: 0, switches: 0, resets: 0 });
        a
    }

    /// The cells active now.
    pub fn active(&self) -> &[usize] {
        &self.active
    }

    /// Holding: the assembly active (or last active), how many there are, changes of assembly
    /// and silencings so far, and whether none holds now.
    pub fn now(&self) -> usize {
        self.hold.as_ref().map_or(0, |h| h.now)
    }
    pub fn contexts(&self) -> usize {
        self.hold.as_ref().map_or(0, |h| h.contexts)
    }
    pub fn switches(&self) -> u32 {
        self.hold.as_ref().map_or(0, |h| h.switches)
    }
    pub fn resets(&self) -> u32 {
        self.hold.as_ref().map_or(0, |h| h.resets)
    }
    pub fn unsettled(&self) -> bool {
        self.active.is_empty()
    }

    /// A random mask, each bit set with probability 2^-p.
    fn rand_mask(&mut self, p: u32) -> u64 {
        let mut m = !0u64;
        for _ in 0..p {
            m &= self.rng.next_u64();
        }
        m
    }

    fn lines(&mut self, input: &BitVector) -> (Vec<usize>, Vec<u64>) {
        let mut w: Vec<u64> = input.as_words().to_vec();
        w.resize(self.words, 0);
        let on: Vec<usize> = (0..self.bits).filter(|&b| w[b / 64] >> (b % 64) & 1 == 1).collect();
        for (s, x) in self.seen.iter_mut().zip(&w) {
            self.n_seen += (x & !*s).count_ones();
            *s |= x;
        }
        (on, w)
    }

    fn has(&self, i: usize, b: usize) -> bool {
        self.syn[i * self.words + b / 64] >> (b % 64) & 1 == 1
    }

    /// Hebbian: each cell's synapses on the lines of `mask` are set, each with probability 2^-p.
    fn set(&mut self, cells: &[usize], mask: &[u64], p: u32) {
        for &i in cells {
            for w in 0..self.words {
                if mask[w] != 0 {
                    let r = self.rand_mask(p);
                    self.syn[i * self.words + w] |= mask[w] & r;
                }
            }
        }
    }

    /// Anti-Hebbian: each cell's synapses on the lines of `mask` are cleared, each with
    /// probability 2^-p.
    fn clear(&mut self, cells: &[usize], mask: &[u64], p: u32) {
        for &i in cells {
            for w in 0..self.words {
                if mask[w] & self.syn[i * self.words + w] != 0 {
                    let r = self.rand_mask(p);
                    self.syn[i * self.words + w] &= !(mask[w] & r);
                }
            }
        }
    }

    /// The lines seen that are not in the input.
    fn absent(&self, input: &[u64]) -> Vec<u64> {
        self.seen.iter().zip(input).map(|(s, x)| s & !x).collect()
    }

    /// The input's net drive on a set of cells (see the module notes).
    fn net(&self, cells: &[usize], on: &[usize]) -> i32 {
        if cells.is_empty() || on.is_empty() {
            return 0;
        }
        let from: i32 = on.iter().map(|&b| bitlen(cells.iter().filter(|&&i| self.has(i, b)).count() as u32)).sum::<i32>() / on.len() as i32;
        let total: u32 = cells.iter().map(|&i| (0..self.words).map(|w| (self.syn[i * self.words + w] & self.seen[w]).count_ones()).sum::<u32>()).sum();
        from - bitlen(total / self.n_seen.max(1))
    }

    /// Coding: the cells `input` fires (as bits of an `n`-bit vector), learning with `learn`
    /// while its plasticity is open (see the module notes).
    pub fn code(&mut self, input: &BitVector, learn: bool) -> BitVector {
        let (on, inw) = self.lines(input);
        self.clock += 1;
        // the drive: twice the active synapses on the input plus the silent ones, less how far
        // the cell's window says it fires above its share
        let share = (64 * self.k / self.n).max(1) as u64;
        let drive_of = |a: &Self, i: usize, mask: &[u64]| -> u32 {
            let mut d = 0u64;
            for w in 0..a.words {
                d += 2 * (mask[w] & a.syn[i * a.words + w]).count_ones() as u64 + (mask[w] & a.pot[i * a.words + w] & !a.syn[i * a.words + w]).count_ones() as u64;
            }
            d.saturating_sub((a.fired[i].count_ones() as u64).saturating_sub(share)) as u32
        };
        let mut drive: Vec<(u32, usize)> = (0..self.n).map(|i| (drive_of(self, i, &inw), i)).collect();
        drive.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut winners: Vec<usize> = drive.iter().take(self.k).filter(|d| d.0 > 0).map(|d| d.1).collect();
        let weakest = drive.get(self.k.saturating_sub(1)).map_or(0, |d| d.0);
        // novel: more than a quarter of the input's lines learned by none of the winners
        let new_lines: Vec<u64> = (0..self.words).map(|w| winners.iter().fold(inw[w], |m, &i| m & !self.syn[i * self.words + w])).collect();
        let uncoded: u32 = new_lines.iter().map(|w| w.count_ones()).sum();
        let novel = 4 * uncoded as usize > on.len();
        let every = 1u64 << (64 - (self.clock / 2048).leading_zeros()).min(20);
        let open = learn && self.clock % every == 0;
        if learn && novel && !on.is_empty() {
            // the new part recruits cells of its own (other than the winners), which learn the
            // whole input and are its code; the winners are depressed on the new part
            let mut fresh: Vec<(u32, usize)> = (0..self.n).filter(|i| !winners.contains(i)).map(|i| (drive_of(self, i, &new_lines), i)).collect();
            fresh.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            let recruits: Vec<usize> = fresh.iter().take(self.k).map(|d| d.1).collect();
            self.set(&recruits, &inw, 1);
            self.clear(&winners, &new_lines, 1);
            winners = recruits;
        } else if open && !on.is_empty() {
            // Hebbian for the winners (and they let go of what they fire without); anti-Hebbian
            // for the cells that matched (at least half the weakest winner's drive) and lost
            self.set(&winners, &inw, 1);
            let absent = self.absent(&inw);
            self.clear(&winners, &absent, 3);
            let losers: Vec<usize> = drive.iter().skip(self.k).take_while(|&&(d, _)| d > 0 && d * 2 >= weakest).map(|&(_, i)| i).collect();
            self.clear(&losers, &inw, 2);
        }
        // each cell's window shifts in whether it fired
        for (i, f) in self.fired.iter_mut().enumerate() {
            *f = *f << 1 | winners.contains(&i) as u64;
        }
        self.active = winners;
        let mut v = BitVector::new(self.n, Some(0));
        for &i in &self.active {
            v.bit_set(i);
        }
        v
    }

    /// The active assembly's recurrent strength, in bits as the evidence against it is: the bit
    /// length of how many of its cells' ordered pairs are connected.
    fn strength(&self) -> i32 {
        let h = self.hold.as_ref().unwrap();
        let k = self.active.len();
        if k < 2 {
            return 0;
        }
        let rw = self.n.div_ceil(64);
        let mut pairs = 0u32;
        for &i in &self.active {
            for &j in &self.active {
                if i != j && h.rec[i * rw + j / 64] >> (j % 64) & 1 == 1 {
                    pairs += 1;
                }
            }
        }
        bitlen(pairs)
    }

    /// Holding, Hebbian: the active cells learn the input (set at 1/8, and let go of lines seen
    /// without them at 1/8, so each synapse is on about as often as its line comes with them)
    /// and connect to each other (each pair at 1/4).
    fn learn_held(&mut self, inw: &[u64]) {
        let act = self.active.clone();
        self.set(&act, inw, 3);
        let absent = self.absent(inw);
        self.clear(&act, &absent, 3);
        let rw = self.n.div_ceil(64);
        let clock = self.clock;
        let mut am = vec![0u64; rw];
        for &j in &act {
            am[j / 64] |= 1u64 << (j % 64);
        }
        for &i in &act {
            for w in 0..rw {
                let r = self.rand_mask(2);
                let own = if w == i / 64 { 1u64 << (i % 64) } else { 0 };
                let h = self.hold.as_mut().unwrap();
                h.rec[i * rw + w] |= am[w] & r & !own;
            }
            self.hold.as_mut().unwrap().last[i] = clock;
        }
    }

    /// The free cells of assembly `l` (at most k).
    fn cells_of(&self, l: usize) -> Vec<usize> {
        let h = self.hold.as_ref().unwrap();
        (0..self.n).filter(|&i| h.label[i] == Some(l) && h.adapted[i] == 0).take(self.k).collect()
    }

    fn silence(&mut self, strength: i32) {
        let h = self.hold.as_mut().unwrap();
        for &i in &self.active {
            h.adapted[i] = strength as u32;
        }
        h.last_strength = strength as u32;
        h.silent = 0;
        h.resets += 1;
        self.active.clear();
    }

    fn enter(&mut self, to: usize) {
        let h = self.hold.as_mut().unwrap();
        h.silent = 0;
        h.sst = 0;
        h.rival.iter_mut().for_each(|r| *r = 0);
        if to != h.now {
            h.switches += 1;
            h.now = to;
        }
    }

    /// Holding: one input (its active lines); returns the assembly now active (or last active).
    pub fn hold(&mut self, input: &BitVector) -> usize {
        let (on, inw) = self.lines(input);
        self.clock += 1;
        for a in self.hold.as_mut().unwrap().adapted.iter_mut() {
            *a = a.saturating_sub(1);
        }
        if !self.active.is_empty() {
            let strength = self.strength();
            let net = self.net(&self.active, &on);
            let (now, contexts) = (self.now(), self.contexts());
            {
                let h = self.hold.as_mut().unwrap();
                h.sst = (h.sst - net - 1).max(0);
                h.rival.resize(contexts, 0);
            }
            let mut winner: Option<(i32, usize)> = None;
            let mut matched: Vec<usize> = Vec::new();
            for l in 0..contexts {
                if l == now {
                    self.hold.as_mut().unwrap().rival[l] = 0;
                    continue;
                }
                let cells = self.cells_of(l);
                if cells.is_empty() {
                    continue;
                }
                let drive = self.net(&cells, &on);
                if drive > 0 {
                    matched.push(l);
                }
                let h = self.hold.as_mut().unwrap();
                h.rival[l] = (h.rival[l] + drive - net - 1).max(0);
                if strength > 0 && h.rival[l] >= strength && winner.map_or(true, |w| h.rival[l] > w.0) {
                    winner = Some((h.rival[l], l));
                }
            }
            let sst = self.hold.as_ref().unwrap().sst;
            if let Some((_, l)) = winner {
                self.silence(strength);
                self.active = self.cells_of(l);
                self.enter(l);
                self.learn_held(&inw);
            } else if strength > 0 && sst >= strength {
                self.silence(strength);
            } else {
                if net >= -1 {
                    self.learn_held(&inw);
                }
                for l in matched {
                    let cells = self.cells_of(l);
                    self.clear(&cells, &inw, 3);
                }
            }
        }
        if self.active.is_empty() {
            self.settle(&on, &inw);
        }
        if self.debug {
            let h = self.hold.as_ref().unwrap();
            eprintln!("assembly: active {:?} sst {} rivals {:?} strength {} -> {}", self.active, h.sst, h.rival, self.strength(), h.now);
        }
        self.now()
    }

    fn settle(&mut self, on: &[usize], inw: &[u64]) {
        let n = self.n;
        let rw = n.div_ceil(64);
        // the stored assembly this input drives most over its free cells, if it drives it net (a
        // single binary cell is too coarse a reading: the input reaches the assembly's cells)
        let seed = (0..self.contexts()).map(|l| (self.net(&self.cells_of(l), on), l)).filter(|&(r, _)| r > 0).max().and_then(|(_, l)| self.cells_of(l).first().copied());
        if let Some(s) = seed {
            // recurrence completes the seed's assembly: the free cells of its assembly it is
            // connected with
            let (cells, to) = {
                let h = self.hold.as_ref().unwrap();
                let l = h.label[s];
                let mut c: Vec<usize> = vec![s];
                c.extend((0..n).filter(|&i| i != s && h.adapted[i] == 0 && h.label[i] == l && (h.rec[i * rw + s / 64] >> (s % 64) & 1 == 1 || h.rec[s * rw + i / 64] >> (i % 64) & 1 == 1)).take(self.k - 1));
                (c, l.unwrap_or(0))
            };
            self.active = cells;
            self.enter(to);
            self.learn_held(inw);
            return;
        }
        let h = self.hold.as_mut().unwrap();
        h.silent += 1;
        // nothing seeds: once the silenced assembly's adaptation has worn off, the least recently
        // used free cells form a new assembly
        if h.silent >= h.last_strength.max(1) {
            let mut by: Vec<(u64, usize)> = (0..n).filter(|&i| h.adapted[i] == 0).map(|i| (if h.label[i].is_some() { h.last[i] + 1 } else { 0 }, i)).collect();
            by.sort_unstable();
            let cells: Vec<usize> = by.into_iter().take(self.k).map(|(_, i)| i).collect();
            let to = h.contexts;
            h.contexts += 1;
            for &i in &cells {
                // a recycled cell leaves its old assembly
                for w in 0..rw {
                    h.rec[i * rw + w] = 0;
                }
                for j in 0..n {
                    h.rec[j * rw + i / 64] &= !(1u64 << (i % 64));
                }
                h.label[i] = Some(to);
            }
            for &i in &cells {
                for w in 0..self.words {
                    self.syn[i * self.words + w] = 0;
                }
            }
            self.active = cells;
            self.enter(to);
            self.learn_held(inw);
        }
    }
}
