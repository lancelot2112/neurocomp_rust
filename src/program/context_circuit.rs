//! The context as a self-sustaining assembly in a recurrent bit population, learned Hebbian,
//! stopped by inhibition.
//!
//! Cells: `n` context cells (layer 2/3 of the orbitofrontal cortex), `k` active at a time (feedback
//! inhibition from basket cells keeps the count). Two kinds of learned synapse, small counts:
//!
//! - recurrent, cell to cell: cells active together strengthen their links (saturating at 15), so
//!   an assembly re-excites itself and stays on with no input (persistent activity);
//! - input, from the outcome lines (light shown × paid or not) to cells: while an assembly is on,
//!   the outcomes that arrive strengthen their synapses onto it, so the counts come to follow how
//!   often each outcome happens in that context.
//!
//! Competition with rivals: the outcome reaches every stored assembly too. A rival assembly the
//! outcome drives more, net, than the active one (each against its own feedforward inhibition,
//! below, so it is the outcome's share at that cue that counts, not how much each has seen)
//! gathers drive against it, through its own facilitating synapses onto the shared interneurons;
//! an outcome that drives the active one more takes it back off (an opponent count per rival, at
//! least zero). A rival whose count reaches the active assembly's recurrent strength takes over:
//! the active one falls silent and the rival's cells come on.
//!
//! Inhibition from the unknown: the outcome lines come in pairs per cue (paid, not paid), and the cue's lines all
//! drive the same interneurons (feedforward inhibition): a cue's outcome inhibits a cell by the
//! mean of its synapses from that cue's two lines, and excites it by its synapse from the line
//! that fired. Net, in bit lengths, the outcome is support when the cell's synapse from it beats
//! that mean (the outcome is the more usual one for that cue in this context), and inhibition
//! when it falls short; an outcome the context never saw at that cue inhibits by the whole bit
//! length of the mean (a log-likelihood ratio against a flat expectation, in bit lengths). The dendrite-targeting (somatostatin) interneurons sum
//! that inhibition through facilitating synapses, and support takes it back off (the assembly's
//! excitation relieves it): an opponent count that never goes below zero. When it reaches the
//! assembly's own recurrent strength (its mean recurrent synapse), recurrence can no longer
//! hold it and it falls silent. A young assembly falls easily; a long-held one needs more.
//! Dendritic inhibition also blocks plasticity: the assembly learns its input only while that
//! inhibition is low, so it does not learn the outcomes that are taking it down.
//!
//! Silent, the outcome lines drive the cells directly: the cell an informative outcome (one that
//! reaches it more than the mean) drives most seeds the settle, and recurrence completes its
//! assembly (pattern completion: an old context comes back on one outcome that only it
//! predicts). Cells that just fell silent are adapted for as many acts as their recurrent
//! strength. When nothing seeds by the time that adaptation wears off, the least-used cells form
//! a new assembly.
//!
//! All integer: counts, bit lengths, compares.

#[derive(Clone, Debug)]
pub struct ContextCircuit {
    n: usize,
    k: usize,
    inputs: usize,
    /// recurrent synapses, `rec[i * n + j]`: from j onto i (0..=15)
    rec: Vec<u8>,
    /// input synapses, `inp[i * inputs + x]`: from outcome line x onto i
    inp: Vec<u8>,
    active: Vec<usize>,
    /// the assembly each cell was recruited into (its context label)
    label: Vec<Option<usize>>,
    /// acts each cell has been active, and acts until its adaptation wears off
    used: Vec<u32>,
    adapted: Vec<u32>,
    /// the dendritic inhibition on the active assembly (against the flat expectation: a new
    /// cause), and each rival assembly's drive against it, by label
    sst: i32,
    rival: Vec<i32>,
    /// acts silent with nothing to seed a settle, and how long the assembly silenced last stays
    /// adapted (a new one is recruited only once it could have come back and did not)
    silent: u32,
    last_strength: u32,
    pub now: usize,
    pub contexts: usize,
    pub switches: u32,
    pub resets: u32,
    pub debug: bool,
}

fn bitlen(x: u32) -> i32 {
    32 - x.leading_zeros() as i32
}

impl ContextCircuit {
    pub fn new(n: usize, k: usize, inputs: usize) -> Self {
        Self {
            n,
            k,
            inputs,
            rec: vec![0; n * n],
            inp: vec![0; n * inputs],
            active: Vec::new(),
            label: vec![None; n],
            used: vec![0; n],
            adapted: vec![0; n],
            sst: 0,
            rival: Vec::new(),
            silent: 0,
            last_strength: 0,
            now: 0,
            contexts: 0,
            switches: 0,
            resets: 0,
            debug: false,
        }
    }

    pub fn unsettled(&self) -> bool {
        self.active.is_empty()
    }

    /// The active assembly's recurrent strength: its mean recurrent synapse (0..=15).
    fn strength(&self) -> i32 {
        let k = self.active.len();
        if k < 2 {
            return 0;
        }
        let mut sum = 0u32;
        for &i in &self.active {
            for &j in &self.active {
                if i != j {
                    sum += self.rec[i * self.n + j] as u32;
                }
            }
        }
        (sum / (k * (k - 1)) as u32) as i32
    }

    /// Outcome line `x`'s net drive on a set of cells: the bit length of their mean synapse from
    /// it, less that of their mean synapse from its cue's two lines (`x` and `x ^ 1`).
    fn net(&self, cells: &[usize], x: usize) -> i32 {
        if cells.is_empty() {
            return 0;
        }
        let k = cells.len() as u32;
        let from = |y: usize| cells.iter().map(|&i| self.inp[i * self.inputs + y] as u32).sum::<u32>() / k;
        let (on, other) = (from(x), from(x ^ 1));
        bitlen(on) - bitlen((on + other) / 2)
    }

    fn learn(&mut self, x: usize) {
        let act = self.active.clone();
        for &i in &act {
            self.used[i] += 1;
            let w = &mut self.inp[i * self.inputs + x];
            if *w == u8::MAX {
                // saturated: halve the cell's input synapses (the counts keep their proportions)
                for y in 0..self.inputs {
                    self.inp[i * self.inputs + y] /= 2;
                }
            }
            self.inp[i * self.inputs + x] += 1;
            for &j in &act {
                if i != j {
                    let r = &mut self.rec[i * self.n + j];
                    *r = (*r + 1).min(15);
                }
            }
        }
    }

    /// One act's outcome on line `x`; returns the context (assembly label) now active.
    pub fn step(&mut self, x: usize) -> usize {
        for a in self.adapted.iter_mut() {
            *a = a.saturating_sub(1);
        }
        if !self.active.is_empty() {
            let strength = self.strength();
            let net = self.net(&self.active, x);
            self.sst = (self.sst - net).max(0);
            // each rival: its net drive from this outcome (against its own feedforward inhibition,
            // so proportions count, not experience) against the active one's
            self.rival.resize(self.contexts, 0);
            let mut winner: Option<(i32, usize)> = None;
            for l in 0..self.contexts {
                if l == self.now {
                    self.rival[l] = 0;
                    continue;
                }
                let cells = self.cells_of(l);
                if cells.is_empty() {
                    continue;
                }
                self.rival[l] = (self.rival[l] + self.net(&cells, x) - net).max(0);
                if strength > 0 && self.rival[l] >= strength && winner.map_or(true, |w| self.rival[l] > w.0) {
                    winner = Some((self.rival[l], l));
                }
            }
            if let Some((_, l)) = winner {
                // the rival wins the competition: the active assembly falls silent, the rival's
                // cells come on
                for &i in &self.active {
                    self.adapted[i] = strength as u32;
                }
                self.last_strength = strength as u32;
                self.resets += 1;
                self.active = self.cells_of(l);
                self.rival.iter_mut().for_each(|r| *r = 0);
                self.enter(l);
                self.learn(x);
            } else if strength > 0 && self.sst >= strength {
                // recurrence can no longer hold it
                for &i in &self.active {
                    self.adapted[i] = strength as u32;
                }
                self.active.clear();
                self.sst = 0;
                self.silent = 0;
                self.last_strength = strength as u32;
                self.resets += 1;
            } else if 2 * self.sst < strength.max(1) {
                self.learn(x);
            }
        }
        if self.active.is_empty() {
            self.settle(x);
        }
        if self.debug {
            eprintln!("circuit: line {x} active {:?} sst {} strength {} -> {}", self.active, self.sst, self.strength(), self.now);
        }
        self.now
    }

    fn settle(&mut self, x: usize) {
        // the free cell this outcome drives most, if it drives it net (the outcome is the usual
        // one for its cue there)
        let seed = (0..self.n)
            .filter(|&i| self.adapted[i] == 0 && self.label[i].is_some())
            .map(|i| (self.net(&[i], x), self.inp[i * self.inputs + x], i))
            .filter(|&(r, _, _)| r > 0)
            .max()
            .map(|(r, _, i)| (r, i));
        if let Some((_, s)) = seed {
            // recurrence completes the seed's assembly: the k free cells it excites most
            let mut by: Vec<(u8, usize)> = (0..self.n).filter(|&i| self.adapted[i] == 0).map(|i| (if i == s { 16 } else { self.rec[i * self.n + s] }, i)).collect();
            by.sort_unstable_by(|a, b| b.cmp(a));
            self.active = by.into_iter().take(self.k).filter(|&(r, _)| r > 0).map(|(_, i)| i).collect();
            let to = self.label[s].unwrap_or(0);
            self.enter(to);
            self.learn(x);
            return;
        }
        self.silent += 1;
        // nothing seeds: once the silenced assembly's adaptation has worn off (so it too could
        // have come back), the least-used free cells form a new assembly
        let wait = self.last_strength.max(1);
        if self.silent >= wait {
            let mut by: Vec<(u32, usize)> = (0..self.n).filter(|&i| self.adapted[i] == 0).map(|i| (self.used[i], i)).collect();
            by.sort_unstable();
            let cells: Vec<usize> = by.into_iter().take(self.k).map(|(_, i)| i).collect();
            let to = self.contexts;
            self.contexts += 1;
            for &i in &cells {
                // a recycled cell leaves its old assembly
                for j in 0..self.n {
                    self.rec[i * self.n + j] = 0;
                    self.rec[j * self.n + i] = 0;
                }
                for y in 0..self.inputs {
                    self.inp[i * self.inputs + y] = 0;
                }
                self.label[i] = Some(to);
            }
            self.active = cells;
            self.enter(to);
            self.learn(x);
        }
    }

    /// The free cells of assembly `l` (at most k).
    fn cells_of(&self, l: usize) -> Vec<usize> {
        (0..self.n).filter(|&i| self.label[i] == Some(l) && self.adapted[i] == 0).take(self.k).collect()
    }

    fn enter(&mut self, to: usize) {
        self.silent = 0;
        self.sst = 0;
        self.rival.iter_mut().for_each(|r| *r = 0);
        if to != self.now {
            self.switches += 1;
            self.now = to;
        }
    }
}
