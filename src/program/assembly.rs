//! One kind of population for coding and for holding: cells with counting synapses, learned
//! Hebbian and anti-Hebbian, under inhibition (`Assembly`). The same cells and rules serve two
//! uses, as the cortex uses one circuit for both:
//!
//! - **Coding** (`Assembly::coding`; layer 4, the dentate gyrus): each input fires the `k` cells
//!   it drives most (feedback inhibition), and learning separates overlapping inputs (below);
//!   the code is the cells that fired.
//! - **Holding** (`Assembly::holding`; layer 2/3's held population): `k` cells recruited
//!   together form an assembly that sustains itself through recurrent synapses with no input
//!   (persistent activity), and falls silent when inhibition reaches its recurrent strength;
//!   which assembly holds is the area's state (its context).
//!
//! **Synapses.** Input synapses are small counts, from each input line to each cell: Hebbian
//! learning adds one (a cell whose synapse saturates halves all of them, keeping their
//! proportions), anti-Hebbian learning takes one off. So the counts follow how often each input
//! comes while a cell fires: graded synapses carry the frequencies a binary cell cannot.
//! Recurrent synapses (holding) count up to 15 between cells active together.
//!
//! **The two learning rules.** Hebbian: a cell that fires strengthens its synapses from the
//! input's lines. Anti-Hebbian: a cell the input drove (it matched and would fire) but
//! inhibition held down weakens them (long-term depression). Inputs that overlap drive the same
//! cells at first; each time one wins and the others are held down, the losers lose exactly
//! what the inputs share, and cells come to answer to what is distinctive in each (pattern
//! separation). The inhibition is local in the input's own terms: only cells that matched are
//! depressed.
//!
//! **Homeostasis** (coding). A cell's drive is divided by how far it fires above its share of
//! inputs, so a cell that would fire for everything (as one for what overlapping inputs share
//! would) is held back until cells for one input alone win.
//!
//! **Net drive** (holding). Every input line also drives the interneurons (feedforward
//! inhibition): an input excites a set of cells through their synapses from its lines and
//! inhibits them as their synapses from all lines seen do on average. In bit lengths: the bit
//! length of their mean synapse from each of its lines, averaged, less that of their mean
//! synapse over every line seen (a log-likelihood against a flat expectation; the log before the
//! mean, so a few familiar lines do not vouch for the rest).
//!
//! **Release** (holding). The dendrite-targeting interneurons sum the active assembly's negative
//! net drive through facilitating synapses, and positive net drive takes it back off (an
//! opponent count, at least zero, less one bit each input: only an input at least twice as
//! unlikely as expected counts, as a change detector allows half the change it is to find).
//! Each stored rival assembly gets the same input; one it drives more, net, gathers drive against
//! the active one through the shared interneurons (an opponent count per rival, with the same
//! allowance). When either count reaches the active assembly's recurrent strength (its mean
//! recurrent synapse: a young assembly falls easily, a long-held one needs more), it falls
//! silent and its cells adapt for as many inputs; a rival that won takes over at once. Silent,
//! the free cell the input drives most, net, seeds a settle and recurrence completes its
//! assembly (an old state comes back on one input that only it predicts); when nothing seeds by
//! the time the silenced cells recover, the least-used cells form a new assembly. The active
//! assembly learns an input that drives it (within the allowance) and not one that inhibits it;
//! a rival the input drove while the active one held it down is depressed (anti-Hebbian).
//!
//! All integer: counts, bit lengths, compares.

use crate::bitvec::BitVector;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// A coding cell's first, weak synapses (before it has learned anything).
const FIRST: u8 = 4;

fn bitlen(x: u32) -> i32 {
    32 - x.leading_zeros() as i32
}

/// What holding adds: recurrent synapses and the assemblies they make.
#[derive(Clone)]
struct Hold {
    /// recurrent synapses, `rec[i * n + j]`: from j onto i (0..=15)
    rec: Vec<u8>,
    /// the assembly each cell was recruited into
    label: Vec<Option<usize>>,
    /// inputs each cell has been active for, and inputs until its adaptation wears off
    used: Vec<u32>,
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
    /// input synapses, `w[i * bits + b]`, and per cell their sum
    w: Vec<u8>,
    sum: Vec<u32>,
    /// the input lines seen so far, and how many
    seen: BitVector,
    n_seen: u32,
    /// each cell's recent firing (0..=256: its share of inputs lately, times 256): homeostasis
    rate: Vec<u32>,
    /// the cells active now
    active: Vec<usize>,
    /// coding: inputs coded so far (plasticity closes as they double: a critical period; a
    /// novel input reopens it, as acetylcholine does)
    coded: u64,
    hold: Option<Hold>,
    pub debug: bool,
}

impl Assembly {
    fn base(n: usize, k: usize, bits: usize) -> Self {
        Self { n, k, bits, w: vec![0; n * bits], sum: vec![0; n], seen: BitVector::new(bits, Some(0)), n_seen: 0, rate: vec![0; n], active: Vec::new(), coded: 0, hold: None, debug: false }
    }

    /// For coding: `n` cells over `bits` input lines, `k` firing; each cell starts with weak
    /// synapses from a random `fan` of the lines.
    pub fn coding(n: usize, k: usize, bits: usize, fan: usize, seed: u64) -> Self {
        let mut a = Self::base(n, k, bits);
        let mut rng = StdRng::seed_from_u64(seed);
        for i in 0..n {
            for _ in 0..fan {
                let b = rng.gen_range(0..bits);
                if a.w[i * bits + b] == 0 {
                    a.w[i * bits + b] = FIRST;
                    a.sum[i] += FIRST as u32;
                }
            }
        }
        a
    }

    /// For holding: `n` cells over `bits` input lines, assemblies of `k`.
    pub fn holding(n: usize, k: usize, bits: usize) -> Self {
        let mut a = Self::base(n, k, bits);
        a.hold = Some(Hold { rec: vec![0; n * n], label: vec![None; n], used: vec![0; n], adapted: vec![0; n], sst: 0, rival: Vec::new(), silent: 0, last_strength: 0, now: 0, contexts: 0, switches: 0, resets: 0 });
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

    fn lines(&mut self, input: &BitVector) -> Vec<usize> {
        let on: Vec<usize> = (0..self.bits).filter(|&b| input.bit_get(b)).collect();
        for &b in &on {
            if !self.seen.bit_get(b) {
                self.seen.bit_set(b);
                self.n_seen += 1;
            }
        }
        on
    }

    /// Hebbian: the cells' synapses from the lines strengthen by one (a saturated cell halves
    /// all of its synapses first).
    fn potentiate(&mut self, cells: &[usize], on: &[usize]) {
        for &i in cells {
            for &b in on {
                if self.w[i * self.bits + b] == u8::MAX {
                    let row = &mut self.w[i * self.bits..(i + 1) * self.bits];
                    for w in row.iter_mut() {
                        *w /= 2;
                    }
                    self.sum[i] = row.iter().map(|&w| w as u32).sum();
                }
                self.w[i * self.bits + b] += 1;
                self.sum[i] += 1;
            }
        }
    }

    /// Anti-Hebbian: the cells' synapses from the lines weaken by one.
    fn depress(&mut self, cells: &[usize], on: &[usize]) {
        for &i in cells {
            for &b in on {
                let w = &mut self.w[i * self.bits + b];
                if *w > 0 {
                    *w -= 1;
                    self.sum[i] -= 1;
                }
            }
        }
    }

    /// The input's net drive on a set of cells (see the module notes).
    fn net(&self, cells: &[usize], on: &[usize]) -> i32 {
        if cells.is_empty() || on.is_empty() {
            return 0;
        }
        let k = cells.len() as u32;
        let from: i32 = on.iter().map(|&b| bitlen(cells.iter().map(|&i| self.w[i * self.bits + b] as u32).sum::<u32>() / k)).sum::<i32>() / on.len() as i32;
        let mean: u32 = cells.iter().map(|&i| self.sum[i]).sum::<u32>() / (k * self.n_seen.max(1));
        from - bitlen(mean)
    }

    /// Coding: the cells `input` fires (as bits of an `n`-bit vector). With `learn`, it learns
    /// while its plasticity is open: every input for the first 2048, then every 2nd, 4th, ... as
    /// its experience doubles (a critical period). A novel input (more than a quarter of its lines
    /// learned by none of its winners: part of it nothing codes yet) always learns, by recruiting
    /// cells of its own for the new part (as the dentate gyrus recruits), so it is not absorbed by
    /// the cells of what it shares.
    pub fn code(&mut self, input: &BitVector, learn: bool) -> BitVector {
        let on = self.lines(input);
        // the drive, divided by how far the cell fires above its share (homeostasis)
        let share = (256 * self.k / self.n).max(1) as u64;
        let mut drive: Vec<(u32, usize)> = (0..self.n)
            .map(|i| {
                let d = on.iter().map(|&b| self.w[i * self.bits + b] as u32).sum::<u32>() as u64;
                ((d * share / (share + self.rate[i] as u64)) as u32, i)
            })
            .collect();
        // the most driven fire (ties to the lower index, a fixed order)
        drive.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let winners: Vec<usize> = drive.iter().take(self.k).filter(|d| d.0 > 0).map(|d| d.1).collect();
        let weakest = drive.get(self.k.saturating_sub(1)).map_or(0, |d| d.0);
        // novel: more than a quarter of the input's lines are learned by none of the winners
        // (no winner's synapse there beyond its first weak one): part of the input nothing codes
        let uncoded = on.iter().filter(|&&b| winners.iter().all(|&i| self.w[i * self.bits + b] <= FIRST)).count();
        let novel = 4 * uncoded > on.len();
        self.coded += 1;
        let every = 1u64 << (64 - (self.coded / 2048).leading_zeros()).min(20);
        let open = learn && self.coded % every == 0;
        // firing-rate adaptation follows every input (it is not plasticity)
        for (i, r) in self.rate.iter_mut().enumerate() {
            *r = if winners.contains(&i) { *r + (256 - (*r).min(256)) / 16 } else { *r - *r / 16 };
        }
        let mut winners = winners;
        if learn && novel && !on.is_empty() {
            // a novel input: the part nothing codes recruits its own cells (the cells its
            // uncoded lines drive most, homeostasis applied, other than the winners), which learn
            // the whole input and are its code; the winners, which matched only the familiar
            // part, are depressed on the new part (they do not absorb it)
            let new_lines: Vec<usize> = on.iter().copied().filter(|&b| winners.iter().all(|&i| self.w[i * self.bits + b] <= FIRST)).collect();
            let mut fresh: Vec<(u32, usize)> = (0..self.n)
                .filter(|i| !winners.contains(i))
                .map(|i| {
                    let d = new_lines.iter().map(|&b| self.w[i * self.bits + b] as u32).sum::<u32>() as u64;
                    ((d * share / (share + self.rate[i] as u64)) as u32, i)
                })
                .collect();
            fresh.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
            let recruits: Vec<usize> = fresh.iter().take(self.k).map(|d| d.1).collect();
            self.potentiate(&recruits, &on);
            self.depress(&winners, &new_lines);
            winners = recruits;
        } else if open && !on.is_empty() {
            self.potentiate(&winners, &on);
            // the cells that matched (at least half the weakest winner's drive) and were held down
            let losers: Vec<usize> = drive.iter().skip(self.k).take_while(|&&(d, _)| d > 0 && d * 2 >= weakest).map(|&(_, i)| i).collect();
            self.depress(&losers, &on);
        }
        self.active = winners;
        let mut v = BitVector::new(self.n, Some(0));
        for &i in &self.active {
            v.bit_set(i);
        }
        v
    }

    /// The active assembly's recurrent strength: its mean recurrent synapse (0..=15).
    fn strength(&self) -> i32 {
        let h = self.hold.as_ref().unwrap();
        let k = self.active.len();
        if k < 2 {
            return 0;
        }
        let mut sum = 0u32;
        for &i in &self.active {
            for &j in &self.active {
                if i != j {
                    sum += h.rec[i * self.n + j] as u32;
                }
            }
        }
        (sum / (k * (k - 1)) as u32) as i32
    }

    /// Holding, Hebbian: the active cells learn the input and each other.
    fn learn_held(&mut self, on: &[usize]) {
        let act = self.active.clone();
        self.potentiate(&act, on);
        let n = self.n;
        let h = self.hold.as_mut().unwrap();
        for &i in &act {
            h.used[i] += 1;
            for &j in &act {
                if i != j {
                    let r = &mut h.rec[i * n + j];
                    *r = (*r + 1).min(15);
                }
            }
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
        let on = self.lines(input);
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
            // the rivals this input drives (they matched it and would fire)
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
                // the rival wins: it comes on as the active one falls silent
                self.silence(strength);
                self.active = self.cells_of(l);
                self.enter(l);
                self.learn_held(&on);
            } else if strength > 0 && sst >= strength {
                self.silence(strength);
            } else {
                if net >= -1 {
                    self.learn_held(&on);
                }
                for l in matched {
                    let cells = self.cells_of(l);
                    self.depress(&cells, &on);
                }
            }
        }
        if self.active.is_empty() {
            self.settle(&on);
        }
        if self.debug {
            let h = self.hold.as_ref().unwrap();
            eprintln!("assembly: active {:?} sst {} rivals {:?} strength {} -> {}", self.active, h.sst, h.rival, self.strength(), h.now);
        }
        self.now()
    }

    fn settle(&mut self, on: &[usize]) {
        let n = self.n;
        // the free cell this input drives most, if it drives it net
        let seed = {
            let h = self.hold.as_ref().unwrap();
            (0..n).filter(|&i| h.adapted[i] == 0 && h.label[i].is_some()).map(|i| (self.net(&[i], on), i)).filter(|&(r, _)| r > 0).max()
        };
        if let Some((_, s)) = seed {
            // recurrence completes the seed's assembly: the k free cells it excites most
            let (cells, to) = {
                let h = self.hold.as_ref().unwrap();
                let mut by: Vec<(u8, usize)> = (0..n).filter(|&i| h.adapted[i] == 0).map(|i| (if i == s { 16 } else { h.rec[i * n + s] }, i)).collect();
                by.sort_unstable_by(|a, b| b.cmp(a));
                (by.into_iter().take(self.k).filter(|&(r, _)| r > 0).map(|(_, i)| i).collect::<Vec<_>>(), h.label[s].unwrap_or(0))
            };
            self.active = cells;
            self.enter(to);
            self.learn_held(on);
            return;
        }
        let h = self.hold.as_mut().unwrap();
        h.silent += 1;
        // nothing seeds: once the silenced assembly's adaptation has worn off (so it too could
        // have come back), the least-used free cells form a new assembly
        if h.silent >= h.last_strength.max(1) {
            let mut by: Vec<(u32, usize)> = (0..n).filter(|&i| h.adapted[i] == 0).map(|i| (h.used[i], i)).collect();
            by.sort_unstable();
            let cells: Vec<usize> = by.into_iter().take(self.k).map(|(_, i)| i).collect();
            let to = h.contexts;
            h.contexts += 1;
            for &i in &cells {
                // a recycled cell leaves its old assembly
                for j in 0..n {
                    h.rec[i * n + j] = 0;
                    h.rec[j * n + i] = 0;
                }
                h.label[i] = Some(to);
            }
            for &i in &cells {
                for w in self.w[i * self.bits..(i + 1) * self.bits].iter_mut() {
                    *w = 0;
                }
                self.sum[i] = 0;
            }
            self.active = cells;
            self.enter(to);
            self.learn_held(on);
        }
    }
}
