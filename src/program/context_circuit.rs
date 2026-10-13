//! A held population as a self-sustaining assembly with counting synapses, learned Hebbian and
//! stopped by inhibition: generic, for any area, on whatever sparse input the area receives
//! (its layer 4 code: a conjunction of what came together, as layer 4 recruits a cell for each
//! new input pattern).
//!
//! Cells: `n` cells, `k` active at a time (feedback inhibition from basket cells keeps the
//! count). Two kinds of learned synapse, small counts:
//!
//! - recurrent, cell to cell: cells active together strengthen their links (saturating at 15), so
//!   an assembly re-excites itself and stays on with no input (persistent activity);
//! - input, from each input line to each cell: while an assembly is on, the lines that fire
//!   strengthen their synapses onto it (a cell whose synapse saturates halves all of its input
//!   synapses), so the counts come to follow how often each input comes while that assembly
//!   holds: graded synapses carry the frequencies a binary cell cannot.
//!
//! Inhibition. Every input line also drives the interneurons (feedforward inhibition), so an
//! input excites a cell through its own synapses and inhibits it as the cell's synapses from all
//! the lines seen so far do on average. The net drive of an input on a set of cells, in bit
//! lengths, is the bit length of their mean synapse from its lines less that of their mean
//! synapse over every line seen (a log-likelihood against a flat expectation): an input usual
//! for the assembly drives it, one it did not learn inhibits it by the whole bit length.
//!
//! - Inhibition from the unknown: the dendrite-targeting (somatostatin) interneurons sum the
//!   active assembly's negative net drive through facilitating synapses, and positive net
//!   drive takes it back off (an opponent count, at least zero), less one bit each input (an
//!   allowance: only an input at least twice as unlikely as expected counts, as a change
//!   detector allows half the change it is to find, so ordinary variation does not walk it up).
//! - Competition with rivals: each stored assembly gets the same input; one the input drives
//!   more, net, than the active one gathers drive against it through the shared interneurons,
//!   and an input that drives the active one more takes it back off (an opponent count per
//!   rival, at least zero, with the same allowance).
//!
//! When either count reaches the active assembly's recurrent strength (its mean recurrent
//! synapse: a young assembly falls easily, a long-held one needs more), recurrence can no longer
//! hold it: it falls silent and its cells adapt for as many inputs. A rival that won takes over
//! at once. Silent, the input drives the cells directly: the free cell it drives most, net, seeds
//! a settle and recurrence completes its assembly (pattern completion: an old state comes back on
//! one input that only it predicts); when nothing seeds by the time the silenced cells recover,
//! the least-used cells form a new assembly. Learning is Hebbian: the active assembly learns an
//! input that drives it (within the one-bit allowance of what it expects); an input that
//! inhibits it does not fire it and is not learned, so it does not come to expect what is taking
//! it down. Anti-Hebbian: a rival assembly the input drives (it matched and would fire) but the
//! active one holds down weakens its synapses from that input (long-term depression), until the
//! input no longer favours it: what comes in every context stops driving the rivals, what is
//! theirs alone is kept (pattern separation).
//!
//! All integer: counts, bit lengths, compares.

use crate::bitvec::BitVector;

#[derive(Clone)]
pub struct ContextCircuit {
    n: usize,
    k: usize,
    bits: usize,
    /// recurrent synapses, `rec[i * n + j]`: from j onto i (0..=15)
    rec: Vec<u8>,
    /// input synapses, `inp[i * bits + b]`: from input line b onto i; per cell their sum
    inp: Vec<u8>,
    sum: Vec<u32>,
    /// the input lines seen so far, and how many
    seen: BitVector,
    n_seen: u32,
    active: Vec<usize>,
    /// the assembly each cell was recruited into
    label: Vec<Option<usize>>,
    /// inputs each cell has been active for, and inputs until its adaptation wears off
    used: Vec<u32>,
    adapted: Vec<u32>,
    /// the inhibition on the active assembly from the unknown, and each rival's drive against
    /// it (by assembly)
    sst: i32,
    rival: Vec<i32>,
    /// inputs silent with nothing to seed a settle, and how long the assembly silenced last
    /// stays adapted
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
    pub fn new(n: usize, k: usize, bits: usize) -> Self {
        Self {
            n,
            k,
            bits,
            rec: vec![0; n * n],
            inp: vec![0; n * bits],
            sum: vec![0; n],
            seen: BitVector::new(bits, Some(0)),
            n_seen: 0,
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

    /// The active cells.
    pub fn active(&self) -> &[usize] {
        &self.active
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

    /// The input's net drive on a set of cells: the bit length of their mean synapse from each of
    /// its lines, averaged over them, less that of their mean synapse over every line seen.
    fn net(&self, cells: &[usize], on: &[usize]) -> i32 {
        if cells.is_empty() || on.is_empty() {
            return 0;
        }
        let k = cells.len() as u32;
        // per line, the bit length of the cells' mean synapse from it, averaged over the input's
        // lines (the log before the mean: a few familiar lines do not vouch for the rest)
        let from: i32 = on.iter().map(|&b| bitlen(cells.iter().map(|&i| self.inp[i * self.bits + b] as u32).sum::<u32>() / k)).sum::<i32>() / on.len() as i32;
        let mean: u32 = cells.iter().map(|&i| self.sum[i]).sum::<u32>() / (k * self.n_seen.max(1));
        from - bitlen(mean)
    }

    fn learn(&mut self, on: &[usize]) {
        let act = self.active.clone();
        for &i in &act {
            self.used[i] += 1;
            for &b in on {
                if self.inp[i * self.bits + b] == u8::MAX {
                    // saturated: halve the cell's input synapses (the counts keep their proportions)
                    let row = &mut self.inp[i * self.bits..(i + 1) * self.bits];
                    for w in row.iter_mut() {
                        *w /= 2;
                    }
                    self.sum[i] = row.iter().map(|&w| w as u32).sum();
                }
                self.inp[i * self.bits + b] += 1;
                self.sum[i] += 1;
            }
            for &j in &act {
                if i != j {
                    let r = &mut self.rec[i * self.n + j];
                    *r = (*r + 1).min(15);
                }
            }
        }
    }

    fn unlearn(&mut self, cells: &[usize], on: &[usize]) {
        for &i in cells {
            for &b in on {
                let w = &mut self.inp[i * self.bits + b];
                if *w > 0 {
                    *w -= 1;
                    self.sum[i] -= 1;
                }
            }
        }
    }

    /// The free cells of assembly `l` (at most k).
    fn cells_of(&self, l: usize) -> Vec<usize> {
        (0..self.n).filter(|&i| self.label[i] == Some(l) && self.adapted[i] == 0).take(self.k).collect()
    }

    fn silence(&mut self, strength: i32) {
        for &i in &self.active {
            self.adapted[i] = strength as u32;
        }
        self.active.clear();
        self.last_strength = strength as u32;
        self.silent = 0;
        self.resets += 1;
    }

    /// One input (its active lines); returns the assembly now active.
    pub fn step(&mut self, input: &BitVector) -> usize {
        let all: Vec<usize> = (0..self.bits).filter(|&b| input.bit_get(b)).collect();
        let on = all.clone();
        for &b in &all {
            if !self.seen.bit_get(b) {
                self.seen.bit_set(b);
                self.n_seen += 1;
            }
        }
        for a in self.adapted.iter_mut() {
            *a = a.saturating_sub(1);
        }
        if !self.active.is_empty() {
            let strength = self.strength();
            let net = self.net(&self.active, &on);
            // (less one bit: only an input at least twice as unlikely as the flat expectation
            // counts, as a change detector allows half the change it is to find; ordinary
            // variation, a fraction of a bit, does not walk it up)
            self.sst = (self.sst - net - 1).max(0);
            self.rival.resize(self.contexts, 0);
            let mut winner: Option<(i32, usize)> = None;
            // the rivals this input drives (they matched it and would fire)
            let mut matched: Vec<usize> = Vec::new();
            for l in 0..self.contexts {
                if l == self.now {
                    self.rival[l] = 0;
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
                self.rival[l] = (self.rival[l] + drive - net - 1).max(0);
                if strength > 0 && self.rival[l] >= strength && winner.map_or(true, |w| self.rival[l] > w.0) {
                    winner = Some((self.rival[l], l));
                }
            }
            if let Some((_, l)) = winner {
                // the rival wins the competition: it comes on as the active one falls silent
                self.silence(strength);
                self.active = self.cells_of(l);
                self.enter(l);
                self.learn(&all);
            } else if strength > 0 && self.sst >= strength {
                self.silence(strength);
            } else {
                if net >= -1 {
                    // Hebbian: an input that drives the assembly (within the allowance of what it
                    // expects) strengthens its synapses (a young assembly, expecting little,
                    // learns what comes)
                    self.learn(&all);
                }
                // anti-Hebbian: a rival the input drove, held down by the active assembly (it
                // matched and would have fired, and was inhibited), weakens its synapses from
                // that input (long-term depression): what comes in every context stops driving
                // the rivals, what is theirs alone is kept (pattern separation)
                for l in matched {
                    let cells = self.cells_of(l);
                    self.unlearn(&cells, &all);
                }
            }
        }
        if self.active.is_empty() {
            self.settle(&on, &all);
        }
        if self.debug {
            eprintln!("circuit: active {:?} sst {} rivals {:?} strength {} mean {} -> {}", self.active, self.sst, self.rival, self.strength(), self.active.iter().map(|&i| self.sum[i]).sum::<u32>() / (self.active.len().max(1) as u32 * self.n_seen.max(1)), self.now);
        }
        self.now
    }

    fn settle(&mut self, on: &[usize], all: &[usize]) {
        // the free cell this input drives most, if it drives it net
        let seed = (0..self.n)
            .filter(|&i| self.adapted[i] == 0 && self.label[i].is_some())
            .map(|i| (self.net(&[i], on), i))
            .filter(|&(r, _)| r > 0)
            .max();
        if let Some((_, s)) = seed {
            // recurrence completes the seed's assembly: the k free cells it excites most
            let mut by: Vec<(u8, usize)> = (0..self.n).filter(|&i| self.adapted[i] == 0).map(|i| (if i == s { 16 } else { self.rec[i * self.n + s] }, i)).collect();
            by.sort_unstable_by(|a, b| b.cmp(a));
            self.active = by.into_iter().take(self.k).filter(|&(r, _)| r > 0).map(|(_, i)| i).collect();
            let to = self.label[s].unwrap_or(0);
            self.enter(to);
            self.learn(all);
            return;
        }
        self.silent += 1;
        // nothing seeds: once the silenced assembly's adaptation has worn off (so it too could
        // have come back), the least-used free cells form a new assembly
        if self.silent >= self.last_strength.max(1) {
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
                for w in self.inp[i * self.bits..(i + 1) * self.bits].iter_mut() {
                    *w = 0;
                }
                self.sum[i] = 0;
                self.label[i] = Some(to);
            }
            self.active = cells;
            self.enter(to);
            self.learn(all);
        }
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
