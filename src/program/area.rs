//! Cortical areas, one per device, and association areas that grow as conjunctions are needed.
//!
//! - **An area per device.** A sensor (an eye, an ear), a motor (a vocal tract) or both (a hand:
//!   it moves, and it feels its own motion) is attached with `Brain::attach`. Each gets its own
//!   cortical area:
//!   - a sensory side: its own layer 4, a pool that recruits cells as new inputs need them;
//!   - a motor side: the inverse and forward models (`MotorArea`). Issuing a command keeps the
//!     forward model's prediction (the efference copy); what is then sensed is compared with
//!     it, and the mismatch is the act's *effect error*: "did it do what I expected?".
//! - **Association areas grow by conjunction.** When two areas are active together often
//!   enough, an association area for the pair is created; its cells are recruited from their
//!   joint activity, each with synapses on both. Either one alone then evokes most of the
//!   pair's code (a word heard evokes what it looks like).
//! - **A full association area grows** once there are two pairwise areas: its cells conjoin
//!   the pairwise areas' codes, so it represents the whole scene. It is what an index (the
//!   hippocampus) reads.
//!
//! Everything is binary: a cell is a set of input bits, it fires when at least `theta` of
//! them are on (popcount), and the `k` most driven cells win.

use std::collections::HashMap;

use crate::bitvec::BitVector;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::program::column::Column;
use crate::program::motor::{MotorArea, VocalTract};

/// A motor side: the predictive kernels (`MotorArea`), or a column (agranular: layer 5 the
/// inverse model, layer 2/3 the forward model).
pub enum Motor {
    Kernels(MotorArea),
    Column(Box<Column>),
}

impl Motor {
    /// The command for what is meant (the inverse model).
    pub fn plan(&mut self, meant: &BitVector) -> Option<BitVector> {
        match self {
            Motor::Kernels(m) => m.plan(meant),
            Motor::Column(c) => c.command_for(meant),
        }
    }

    /// What a command will be sensed as (the forward model: the efference copy).
    pub fn predict(&mut self, command: &BitVector) -> Option<BitVector> {
        match self {
            Motor::Kernels(m) => m.predict(command),
            Motor::Column(c) => c.predict_result(command),
        }
    }

    /// A command and what it produced; returns the forward model's error before learning.
    pub fn observe<R: rand::Rng>(&mut self, command: &BitVector, sensed: &BitVector, rng: &mut R) -> u32 {
        match self {
            Motor::Kernels(m) => m.observe(command, sensed, rng),
            Motor::Column(c) => c.observe(command, sensed, rng),
        }
    }

    /// Babble: `rounds` times through the effector's repertoire in random order, each command
    /// issued and its result sensed (`heard[w]`).
    pub fn babble<R: rand::Rng>(&mut self, tract: &VocalTract, heard: &[BitVector], rounds: usize, rng: &mut R) {
        if let Motor::Kernels(m) = self {
            m.babble(tract, heard, rounds, rng);
            return;
        }
        let mut order: Vec<usize> = (0..tract.len().min(heard.len())).collect();
        for _ in 0..rounds {
            order.shuffle(rng);
            for &w in &order {
                let m = tract.command(w).clone();
                if let Some(said) = tract.articulate(&m) {
                    self.observe(&m, &heard[said], rng);
                }
            }
        }
    }

    /// (inverse cells, forward cells, -) for a column; the kernels' own stats otherwise
    pub fn stats(&self) -> (usize, usize, usize) {
        match self {
            Motor::Kernels(m) => m.stats(),
            Motor::Column(c) => (c.l5.committed(), c.l23.committed(), 0),
        }
    }
}

/// A pool of cells over several sources (each `src_bits` bits), recruited as needed.
pub struct Pool {
    src_bits: usize,
    sources: usize,
    /// at most this many cells (the output's width)
    pub out_bits: usize,
    sample: usize,
    theta: usize,
    k: usize,
    /// a new cell samples at least this many active sources (2: a conjunction)
    min_sources: usize,
    /// an input not yet covered recruits on its `repeat`-th occurrence
    repeat: u8,
    syn: Vec<Vec<u32>>,
    last: Vec<u64>,
    index: Vec<Vec<u32>>,
    seen: HashMap<u64, u8>,
    tick: u64,
    rng: u64,
    /// cells recruited so far (including those that replaced old ones)
    pub recruited: u64,
    /// coverage: every part of the input gets cells. Input left uncovered by the synapses of the
    /// cells that fired (a quarter of it or more) recruits cells on that part alone, and the
    /// winners grow with the input (k up to 4k, as divisive normalization lets more input drive
    /// more cells), so adding a thing to what is sensed adds cells rather than displacing them
    pub coverage: bool,
}

impl Pool {
    pub fn new(sources: usize, src_bits: usize, out_bits: usize, k: usize, min_sources: usize, seed: u64) -> Self {
        let sample = 16;
        // a conjunction cell fires on one of its sources fully (completion from either part)
        let theta = if min_sources >= 2 { sample / 2 } else { sample * 5 / 8 };
        Self {
            src_bits,
            sources,
            out_bits,
            sample,
            theta,
            k,
            min_sources,
            repeat: 1,
            syn: Vec::new(),
            last: Vec::new(),
            index: vec![Vec::new(); sources * src_bits],
            seen: HashMap::new(),
            tick: 0,
            rng: seed | 1,
            recruited: 0,
            coverage: false,
        }
    }

    /// An uncovered input recruits only on its `n`-th occurrence (default 1).
    pub fn set_repeat(&mut self, n: u8) {
        self.repeat = n.max(1);
    }

    pub fn cells(&self) -> usize {
        self.syn.len()
    }

    fn next(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }

    /// The input's active bits, in the pool's input space (source `s` at offset `s·src_bits`).
    fn active(&self, input: &[(usize, &BitVector)]) -> Vec<u32> {
        let mut on = Vec::new();
        for &(s, v) in input {
            if s >= self.sources {
                continue;
            }
            for (wi, &w) in v.as_words().iter().enumerate() {
                let mut x = w;
                while x != 0 {
                    let b = wi * 64 + x.trailing_zeros() as usize;
                    if b < self.src_bits {
                        on.push((s * self.src_bits + b) as u32);
                    }
                    x &= x - 1;
                }
            }
        }
        on.sort_unstable();
        on
    }

    /// The cells that fire on these active bits (at least `theta`), most driven first, at most `k`.
    fn fire(&self, on: &[u32]) -> Vec<(u32, usize)> {
        let mut count: HashMap<u32, usize> = HashMap::new();
        for &b in on {
            for &c in &self.index[b as usize] {
                *count.entry(c).or_default() += 1;
            }
        }
        let mut f: Vec<(u32, usize)> = count.into_iter().filter(|&(_, n)| n >= self.theta).collect();
        f.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        f.truncate(self.winners(on.len()));
        f
    }

    /// How many cells may win: `k`, or with coverage as many as the input has bits (up to 4k).
    fn winners(&self, active: usize) -> usize {
        if self.coverage {
            active.clamp(self.k, 4 * self.k)
        } else {
            self.k
        }
    }

    /// Coverage: recruit cells on the input bits that no fired cell has a synapse on, when they
    /// are a quarter of the input or more (and enough for a cell to fire on). Returns the new
    /// cells.
    fn cover(&mut self, on: &[u32], fired: &[(u32, usize)]) -> Vec<usize> {
        let mut covered: Vec<u32> = fired.iter().flat_map(|&(c, _)| self.syn[c as usize].iter().copied()).filter(|b| on.binary_search(b).is_ok()).collect();
        covered.sort_unstable();
        covered.dedup();
        let open: Vec<u32> = on.iter().copied().filter(|b| covered.binary_search(b).is_err()).collect();
        if open.len() * 4 < on.len() || open.len() < self.theta.max(8) {
            return Vec::new();
        }
        let n = open.len().min(self.k);
        let mut new = Vec::new();
        for _ in 0..n {
            let mut s: Vec<u32> = Vec::with_capacity(self.sample);
            let mut tries = 0;
            while s.len() < self.sample.min(open.len()) && tries < self.sample * 8 {
                tries += 1;
                let b = open[(self.next() as usize) % open.len()];
                if !s.contains(&b) {
                    s.push(b);
                }
            }
            s.sort_unstable();
            let c = self.place();
            for &b in &s {
                self.index[b as usize].push(c as u32);
            }
            self.syn[c] = s;
            self.last[c] = self.tick;
            self.recruited += 1;
            new.push(c);
        }
        new
    }

    fn code(&self, fired: &[(u32, usize)]) -> BitVector {
        let mut out = BitVector::new(self.out_bits, Some(0));
        for &(c, _) in fired {
            out.bit_set(c as usize);
        }
        out
    }

    /// The pool's code for this input, without learning.
    pub fn encode(&self, input: &[(usize, &BitVector)]) -> BitVector {
        let on = self.active(input);
        self.code(&self.fire(&on))
    }

    /// The code, learning: the winners tighten toward the input (one synapse, 1/4); if fewer
    /// than half of `k` fire, the input is not yet covered, and (on its `repeat`-th occurrence)
    /// new cells are recruited from it, up to `k` in all, each sampling the active sources in
    /// turn (at least `min_sources` of them must be active). A full pool replaces, of 8 cells
    /// sampled, the one unused longest.
    pub fn learn(&mut self, input: &[(usize, &BitVector)]) -> BitVector {
        self.tick += 1;
        let on = self.active(input);
        let fired = self.fire(&on);
        // the sources present: a synapse is moved away only if its own source was active (an
        // absent sense is no evidence against it, so a pair's cell keeps both halves when one
        // comes alone)
        let present: Vec<bool> = (0..self.sources).map(|s| on.iter().any(|&b| b as usize / self.src_bits == s)).collect();
        for &(c, _) in &fired {
            self.last[c as usize] = self.tick;
            let r = self.next();
            if r & 3 != 0 {
                continue;
            }
            let s = &self.syn[c as usize];
            let off: Vec<usize> = (0..s.len()).filter(|&i| on.binary_search(&s[i]).is_err() && present[s[i] as usize / self.src_bits]).collect();
            // and it moves within its source (each half stays a half)
            let gain_in = |src: usize| on.iter().copied().filter(|b| s.binary_search(b).is_err() && *b as usize / self.src_bits == src).collect::<Vec<u32>>();
            let Some(&i) = off.get(((r >> 8) as usize) % off.len().max(1)) else { continue };
            let gain = gain_in(s[i] as usize / self.src_bits);
            if gain.is_empty() {
                continue;
            }
            let nb = gain[((r >> 32) as usize) % gain.len()];
            self.rewire(c as usize, i, nb);
        }
        let mut code = self.code(&fired);
        if self.coverage && fired.len() * 2 >= self.k {
            for c in self.cover(&on, &fired) {
                code.bit_set(c);
            }
        }
        let srcs: Vec<usize> = {
            let mut v: Vec<usize> = on.iter().map(|&b| b as usize / self.src_bits).collect();
            v.dedup();
            v
        };
        if fired.len() * 2 < self.k && srcs.len() >= self.min_sources && !on.is_empty() {
            // the input's signature: a min-hash of its bits
            let sig = on.iter().map(|&b| (b as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)).min().unwrap_or(0) ^ (on.len() as u64);
            let n = self.seen.entry(sig).or_default();
            *n = n.saturating_add(1);
            if *n >= self.repeat {
                self.seen.remove(&sig);
                if self.seen.len() > 4096 {
                    self.seen.clear();
                }
                let by_src: Vec<Vec<u32>> = srcs.iter().map(|&s| on.iter().copied().filter(|&b| b as usize / self.src_bits == s).collect()).collect();
                for j in fired.len()..self.k {
                    let mut s: Vec<u32> = Vec::with_capacity(self.sample);
                    let mut tries = 0;
                    while s.len() < self.sample && tries < self.sample * 8 {
                        tries += 1;
                        let pick = &by_src[(s.len() + j) % by_src.len()];
                        let b = pick[(self.next() as usize) % pick.len()];
                        if !s.contains(&b) {
                            s.push(b);
                        }
                    }
                    s.sort_unstable();
                    let c = self.place();
                    for &b in &s {
                        self.index[b as usize].push(c as u32);
                    }
                    self.syn[c] = s;
                    self.last[c] = self.tick;
                    self.recruited += 1;
                    code.bit_set(c);
                }
            }
        }
        code
    }

    /// Feedback: what the cells in `code` expect on source `source`, as that source's bits
    /// with the most votes from those cells' synapses (the top `k`, at least two votes, or one
    /// when a single cell is active). The back projection that lets part of a conjunction
    /// evoke the rest.
    pub fn backproject(&self, code: &BitVector, source: usize) -> BitVector {
        let mut votes: HashMap<u32, u32> = HashMap::new();
        let mut cells = 0;
        for c in 0..self.syn.len().min(code.bit_len()) {
            if !code.bit_get(c) {
                continue;
            }
            cells += 1;
            for &b in &self.syn[c] {
                if b as usize / self.src_bits == source {
                    *votes.entry(b % self.src_bits as u32).or_default() += 1;
                }
            }
        }
        let need = if cells > 1 { 2 } else { 1 };
        let mut v: Vec<(u32, u32)> = votes.into_iter().filter(|&(_, n)| n >= need).collect();
        v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut out = BitVector::new(self.src_bits, Some(0));
        for &(b, _) in v.iter().take(self.k) {
            out.bit_set(b as usize);
        }
        out
    }

    /// A slot for a new cell: a new one while there is room, else the stalest of 8 sampled.
    fn place(&mut self) -> usize {
        if self.syn.len() < self.out_bits {
            self.syn.push(Vec::new());
            self.last.push(0);
            return self.syn.len() - 1;
        }
        let mut best = (u64::MAX, 0usize);
        for _ in 0..8 {
            let c = (self.next() as usize) % self.syn.len();
            if self.last[c] < best.0 {
                best = (self.last[c], c);
            }
        }
        let c = best.1;
        for &b in &std::mem::take(&mut self.syn[c]) {
            self.index[b as usize].retain(|&x| x as usize != c);
        }
        c
    }

    fn rewire(&mut self, c: usize, i: usize, nb: u32) {
        let old = self.syn[c][i];
        self.index[old as usize].retain(|&x| x as usize != c);
        self.index[nb as usize].push(c as u32);
        self.syn[c][i] = nb;
        self.syn[c].sort_unstable();
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Sensor,
    Motor,
    SensorMotor,
}

/// A column with a layer 5: a sensory area's, or association column `i` (pairs first, then the
/// full area).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ColumnId {
    Sense(usize),
    Assoc(usize),
}

/// A device's own cortical area.
pub struct Area {
    pub name: String,
    pub kind: Kind,
    /// layer 4 (sensory side)
    pub l4: Option<Pool>,
    /// inverse and forward models (motor side)
    pub motor: Option<Motor>,
    /// the latest sensory code (layer 4's winners)
    pub code: BitVector,
    /// with columns, the rest of the sensory column over layer 4's code (layer 2/3 predicting
    /// the next code, with feedback from above on its tuft)
    pub col: Option<Column>,
    /// the efference copy of the latest command (what it should be sensed as)
    expect: Option<BitVector>,
    /// the latest act's effect error (`Q16`): the share of the expected and sensed bits that
    /// do not match; None until an act is sensed
    pub effect_error: Option<u32>,
}

impl Area {
    /// Sense `input` (the device's raw code): layer 4's code, learned when `learn`. If a
    /// command was issued, what is sensed is compared with its efference copy.
    pub fn sense(&mut self, input: &BitVector, learn: bool) -> BitVector {
        if let Some(e) = self.expect.take() {
            let hit: u32 = e.as_words().iter().zip(input.as_words()).map(|(a, b)| (a & b).count_ones()).sum();
            let total = (e.count_ones() + input.count_ones()) as u32;
            let one = crate::fixed::ONE;
            self.effect_error = Some(if total == 0 { 0 } else { one - 2 * hit * one / total });
        }
        if let Some(p) = self.l4.as_mut() {
            self.code = if input.count_ones() == 0 {
                BitVector::new(p.out_bits, Some(0))
            } else if learn {
                p.learn(&[(0, input)])
            } else {
                p.encode(&[(0, input)])
            };
        }
        self.code.clone()
    }

    /// The command for what is meant (a sensory code), from the inverse model.
    pub fn command_for(&mut self, meant: &BitVector) -> Option<BitVector> {
        self.motor.as_mut().and_then(|m| m.plan(meant))
    }

    /// A command is issued: its efference copy (the forward model's prediction) is kept, to be
    /// compared with what is sensed next. Returns the prediction.
    pub fn issue(&mut self, command: &BitVector) -> Option<BitVector> {
        let p = self.motor.as_mut().and_then(|m| m.predict(command));
        self.expect = Some(p.clone().unwrap_or_else(|| BitVector::new(command.bit_len(), Some(0))));
        p
    }

    /// Both models learn from a command and what it produced (babbling, or any act).
    pub fn learn_act<R: rand::Rng>(&mut self, command: &BitVector, sensed: &BitVector, rng: &mut R) {
        if let Some(m) = self.motor.as_mut() {
            m.observe(command, sensed, rng);
        }
    }
}

/// Areas, the pairwise association areas grown between them, and the full association area.
pub struct Brain {
    bits: usize,
    k: usize,
    pub areas: Vec<Area>,
    /// ((area a, area b), its column: layer 4 the conjunction pool)
    pub pairs: Vec<((usize, usize), Column)>,
    /// co-activity counts of pairs without an association area yet
    together: HashMap<(usize, usize), u32>,
    /// a pair gets an association area after this many co-active steps
    pub pair_after: u32,
    pub full: Option<Column>,
    seed: u64,
    /// columns: motor sides are agranular columns, and association areas predict their next
    /// code (layer 2/3); else motor sides are kernels and association areas only conjoin
    pub columns: bool,
    /// self-organizing (with columns): the association columns' layers 2/3 and 5 are stepped by
    /// `step_columns`, every column's layer 5 offering goals to every motor area
    pub selforg: bool,
    /// persistence (each column's trace decay, `Column::set_persistence`) of the pairwise
    /// association columns and of the full one
    pub pair_tau: u32,
    pub full_tau: u32,
    rng: StdRng,
}

/// Cells a column's layers may grow to.
const CAP: usize = 8192;

/// Room for this many pairwise association areas as sources of the full one.
pub const MAX_PAIRS: usize = 16;

impl Brain {
    /// Areas of `bits` cells each, `k` active.
    pub fn new(bits: usize, k: usize, seed: u64) -> Self {
        Self { bits, k, areas: Vec::new(), pairs: Vec::new(), together: HashMap::new(), pair_after: 8, full: None, seed, columns: false, selforg: false, pair_tau: 0, full_tau: 0, rng: StdRng::seed_from_u64(seed ^ 0xC01) }
    }

    /// Attach a device: its own area (with a layer 4 if it senses, motor models if it acts).
    /// Returns the area's index.
    pub fn attach(&mut self, name: &str, kind: Kind) -> usize {
        let i = self.areas.len();
        let l4 = (kind != Kind::Motor).then(|| Pool::new(1, self.bits, self.bits, self.k, 1, self.seed.wrapping_add(17 * i as u64 + 1)));
        let motor = (kind != Kind::Sensor).then(|| if self.columns { Motor::Column(Box::new(Column::agranular(self.bits, CAP))) } else { Motor::Kernels(MotorArea::new(self.bits)) });
        let col = (self.columns && kind != Kind::Motor).then(|| Column::agranular(self.bits, CAP));
        self.areas.push(Area { name: name.to_string(), kind, l4, motor, code: BitVector::new(self.bits, Some(0)), col, expect: None, effect_error: None });
        i
    }

    /// One step's association: the areas' current codes; pairs active together count toward an
    /// association area of their own (created after `pair_after` steps); every pairwise area
    /// codes its pair; once there are two, the full area conjoins them. Returns the full area's
    /// code if it has one, else the pairwise codes merged (or nothing).
    pub fn associate(&mut self, learn: bool) -> BitVector {
        let codes: Vec<BitVector> = self.areas.iter().map(|a| a.code.clone()).collect();
        self.associate_codes(&codes, learn)
    }

    /// Layer 4's code for `input` in area `a`, learned when `learn` (the area's current code is
    /// left as it is: for what is imagined or merely considered).
    pub fn encode(&mut self, a: usize, input: &BitVector, learn: bool) -> BitVector {
        match self.areas[a].l4.as_mut() {
            Some(p) if input.count_ones() > 0 => {
                if learn {
                    p.learn(&[(0, input)])
                } else {
                    p.encode(&[(0, input)])
                }
            }
            _ => BitVector::new(self.bits, Some(0)),
        }
    }

    /// An association column's code for this input. Learning, with columns, it is sensed: layer
    /// 2/3 learns that this code followed what it predicted, and predicts the next. Otherwise
    /// (and for what is only considered) layer 4 alone.
    fn code_of(col: &mut Column, input: &[(usize, &BitVector)], learn: bool, columns: bool, rng: &mut StdRng) -> BitVector {
        if learn && columns {
            col.sense(input, None, true, rng)
        } else if learn {
            col.l4.as_mut().unwrap().learn(input)
        } else {
            col.l4.as_ref().unwrap().encode(input)
        }
    }

    /// The index of the association column of the pair (`a`, `b`), if it has grown.
    pub fn pair_index(&self, a: usize, b: usize) -> Option<usize> {
        self.pairs.iter().position(|p| p.0 == (a, b) || p.0 == (b, a))
    }

    /// The association columns, pairs then the full one.
    fn assoc_columns(&mut self) -> Vec<&mut Column> {
        self.pairs.iter_mut().map(|p| &mut p.1).chain(self.full.iter_mut()).collect()
    }

    /// Every column with a layer 5, sensory and association: each has a loop of its own through
    /// the basal ganglia, so loops grow as association columns grow.
    pub fn column_ids(&self) -> Vec<ColumnId> {
        let mut v: Vec<ColumnId> = (0..self.areas.len()).filter(|&a| self.areas[a].col.is_some()).map(ColumnId::Sense).collect();
        v.extend((0..self.pairs.len() + self.full.is_some() as usize).map(ColumnId::Assoc));
        v
    }

    fn column_ref(&self, id: ColumnId) -> Option<&Column> {
        match id {
            ColumnId::Sense(a) => self.areas.get(a).and_then(|x| x.col.as_ref()),
            ColumnId::Assoc(i) => self.pairs.get(i).map(|p| &p.1).or_else(|| if i == self.pairs.len() { self.full.as_ref() } else { None }),
        }
    }

    fn column_mut(&mut self, id: ColumnId) -> Option<&mut Column> {
        match id {
            ColumnId::Sense(a) => self.areas.get_mut(a).and_then(|x| x.col.as_mut()),
            ColumnId::Assoc(i) => {
                let n = self.pairs.len();
                if i < n {
                    self.pairs.get_mut(i).map(|p| &mut p.1)
                } else if i == n {
                    self.full.as_mut()
                } else {
                    None
                }
            }
        }
    }

    /// A column's goal now (after its layer 5 predicted this step): (cell, output, burst).
    pub fn goal_of(&self, id: ColumnId) -> Option<(usize, BitVector, bool)> {
        self.column_ref(id).and_then(|c| c.l5.goal())
    }

    /// Dopamine for a column's layer 5 cell whose output was followed as a goal.
    pub fn reinforce(&mut self, id: ColumnId, cell: usize, delta: i32) {
        if let Some(c) = self.column_mut(id) {
            c.l5.reinforce(cell, delta);
        }
    }

    /// A column's layer 2/3 activity (what its projection to the striatum carries).
    pub fn activity_of(&self, id: ColumnId) -> BitVector {
        self.column_ref(id).map_or_else(|| BitVector::new(self.bits, Some(0)), |c| c.activity())
    }

    /// A column's loop chose to keep what the column holds (its trace refreshed; `replace`: the
    /// trace becomes its activity now).
    pub fn refresh(&mut self, id: ColumnId, replace: bool) {
        if let Some(c) = self.column_mut(id) {
            c.refresh(replace);
        }
    }

    /// A column's persistent trace (empty if it has none).
    pub fn trace_of(&self, id: ColumnId) -> BitVector {
        self.column_ref(id).map_or_else(|| BitVector::new(self.bits, Some(0)), |c| c.trace().clone())
    }

    /// A column's current input code (what holding it in working memory would hold).
    pub fn code_of_column(&self, id: ColumnId) -> BitVector {
        self.column_ref(id).map_or_else(|| BitVector::new(self.bits, Some(0)), |c| c.code.clone())
    }

    /// Each association column's layer 4 code for the areas' codes (`codes`, by area; a missing
    /// one is silent), pairs first, then the full area (over the pairs' codes). Not learned.
    pub fn assoc_inputs(&self, codes: &[BitVector]) -> Vec<BitVector> {
        let silent = BitVector::new(self.bits, Some(0));
        let mut v: Vec<BitVector> = self.pairs.iter().map(|((a, b), col)| col.l4.as_ref().unwrap().encode(&[(0, codes.get(*a).unwrap_or(&silent)), (1, codes.get(*b).unwrap_or(&silent))])).collect();
        if let Some(full) = self.full.as_ref() {
            let input: Vec<(usize, &BitVector)> = v.iter().enumerate().filter(|(_, c)| c.count_ones() > 0).collect();
            let f = full.l4.as_ref().unwrap().encode(&input);
            v.push(f);
        }
        v
    }

    /// Feedback down the hierarchy: association column `i`'s cells `cells` project back through
    /// its layer 4's synapses to its sources' codes (the full area to the pairs, a pair to its two
    /// areas), and on down, as far as the areas. Returns (area, its layer 4 code expected).
    pub fn cascade_down(&self, i: usize, cells: &BitVector) -> Vec<(usize, BitVector)> {
        let mut out: Vec<(usize, BitVector)> = Vec::new();
        let n = self.pairs.len();
        let pair_down = |p: usize, code: &BitVector, out: &mut Vec<(usize, BitVector)>| {
            let ((a, b), col) = &self.pairs[p];
            let l4 = col.l4.as_ref().unwrap();
            for (side, area) in [(0usize, *a), (1, *b)] {
                let c = l4.backproject(code, side);
                if c.count_ones() > 0 {
                    out.push((area, c));
                }
            }
        };
        if i < n {
            pair_down(i, cells, &mut out);
        } else if let (true, Some(full)) = (i == n, self.full.as_ref()) {
            let l4 = full.l4.as_ref().unwrap();
            for p in 0..n {
                let c = l4.backproject(cells, p);
                if c.count_ones() > 0 {
                    pair_down(p, &c, &mut out);
                }
            }
        }
        out
    }

    /// Self-organizing: every association column steps. Its input is its layer 4's code for the
    /// areas' codes (`codes`), with any cells `extra` reinstates there; its layer 2/3 predicts its
    /// next input with `tuft` on its tuft; its layer 5 offers a goal (what a motor area should
    /// produce) from its input, tuft and expectation. Returns, per column: (its layer 5's goal
    /// and whether it burst, its input, its expectation), to learn from and to mix.
    pub fn step_columns(&mut self, codes: &[BitVector], extra: &[(usize, BitVector)], tuft: &BitVector, learn: bool) -> Vec<(Option<(BitVector, bool)>, BitVector, BitVector)> {
        let inputs = self.assoc_inputs(codes);
        let mut rng = std::mem::replace(&mut self.rng, StdRng::seed_from_u64(0));
        let mut out = Vec::new();
        for (i, (col, inp)) in self.assoc_columns().into_iter().zip(inputs).enumerate() {
            // `extra`: (column, cells) reinstated in that column (the hippocampus's return to the
            // cells its pointers name, whether or not a sense is active)
            let mut x = inp;
            for (_, e) in extra.iter().filter(|(j, _)| *j == i) {
                for (w, y) in x.as_words_mut().iter_mut().zip(e.as_words()) {
                    *w |= y;
                }
            }
            col.sense_code(&x, Some(tuft), learn, &mut rng);
            let exp = col.expected.clone().unwrap_or_else(|| BitVector::new(x.bit_len(), Some(0)));
            let goal = col.l5.predict_full(&x, Some(tuft), Some(&exp));
            out.push((goal, x, exp));
        }
        self.rng = rng;
        out
    }

    /// Self-organizing: association column `i`'s layer 5 learns that `target` was the goal for
    /// this input (it learns only where its input was active: what it sees is what it learns).
    pub fn learn_goal(&mut self, i: usize, input: &BitVector, tuft: &BitVector, exp: &BitVector, target: &BitVector) {
        let mut rng = std::mem::replace(&mut self.rng, StdRng::seed_from_u64(0));
        if let Some(col) = self.assoc_columns().into_iter().nth(i) {
            col.l5.learn(input, Some(tuft), Some(exp), target, &mut rng);
        }
        self.rng = rng;
    }

    /// Self-organizing: every association column's layer 5 proposals (the motor areas' options),
    /// with the column's index.
    pub fn goal_proposals(&mut self) -> Vec<(usize, BitVector, bool)> {
        let mut v = Vec::new();
        for (i, col) in self.assoc_columns().into_iter().enumerate() {
            for (p, b) in col.l5.proposals() {
                v.push((i, p, b));
            }
        }
        v
    }

    /// The association columns' activity (layer 2/3, projected) and their layer 5's (the cells
    /// that fired), merged: feedback for the areas below, and what the basal ganglia and pons see.
    pub fn columns_activity(&mut self) -> (BitVector, Vec<usize>) {
        let mut act = BitVector::new(self.bits, Some(0));
        let mut fired = Vec::new();
        for (i, col) in self.assoc_columns().into_iter().enumerate() {
            for (w, y) in act.as_words_mut().iter_mut().zip(col.activity().as_words()) {
                *w |= y;
            }
            fired.extend(col.l5.last_fired().into_iter().map(|c| c * 64 + i));
        }
        (act, fired)
    }

    /// Self-organizing: the largest learning progress among the columns' layer 5 winners.
    pub fn goal_progress(&mut self) -> i32 {
        self.assoc_columns().into_iter().map(|c| c.l5.progress()).max().unwrap_or(0)
    }

    /// A sensory column's step (with columns): layer 2/3 learns that `code` (layer 4's, this
    /// step) followed what it predicted, and predicts the next, with `feedback` (the activity of
    /// the area above) on its tuft.
    pub fn sense_step(&mut self, a: usize, code: &BitVector, feedback: Option<&BitVector>, learn: bool) {
        if let Some(col) = self.areas[a].col.as_mut() {
            col.sense(&[(0, code)], feedback, learn, &mut self.rng);
        }
    }

    /// A sensory column's goal (every area's layer 5 offers one): from its latest code, the
    /// tuft and its expectation. With its input the area's own (what is seen, alone), it can
    /// learn what the association areas cannot tell apart (say what is seen, not what comes next).
    pub fn sense_goal(&mut self, a: usize, tuft: &BitVector) -> Option<(BitVector, bool)> {
        let col = self.areas[a].col.as_mut()?;
        let code = col.code.clone();
        let exp = col.expected.clone().unwrap_or_else(|| BitVector::new(code.bit_len(), Some(0)));
        col.l5.predict_full(&code, Some(tuft), Some(&exp))
    }

    /// A sensory column's layer 5 learns that `target` was the goal (only where its input is active).
    pub fn learn_sense_goal(&mut self, a: usize, code: &BitVector, tuft: &BitVector, exp: &BitVector, target: &BitVector) {
        let mut rng = std::mem::replace(&mut self.rng, StdRng::seed_from_u64(0));
        if let Some(col) = self.areas[a].col.as_mut() {
            col.l5.learn(code, Some(tuft), Some(exp), target, &mut rng);
        }
        self.rng = rng;
    }

    /// A sensory column's latest code and expectation (to learn from).
    pub fn sense_state(&self, a: usize) -> Option<(BitVector, BitVector)> {
        let col = self.areas[a].col.as_ref()?;
        Some((col.code.clone(), col.expected.clone().unwrap_or_else(|| BitVector::new(col.code.bit_len(), Some(0)))))
    }

    /// A sensory column's surprise at its latest input (`Q16`): the share of its code that its
    /// layer 2/3 did not expect (all of it if nothing was expected; 0 if nothing came).
    pub fn surprise(&self, a: usize) -> u32 {
        let one = crate::fixed::ONE;
        let Some(col) = self.areas[a].col.as_ref() else { return 0 };
        let n = col.code.count_ones() as u32;
        if n == 0 {
            return 0;
        }
        let hit = col.last_hit;
        one - hit.min(n) * one / n
    }

    /// The sensory columns' predictions: (checked, right).
    /// Sleep's compaction of every column's layers 2/3 and 5 (senses, motor sides, association
    /// columns): (merged, pruned).
    pub fn compact(&mut self, idle: u64) -> (usize, usize) {
        let mut cols: Vec<&mut Column> = Vec::new();
        for a in self.areas.iter_mut() {
            if let Some(c) = a.col.as_mut() {
                cols.push(c);
            }
            if let Some(Motor::Column(c)) = a.motor.as_mut() {
                cols.push(c);
            }
        }
        for p in self.pairs.iter_mut() {
            cols.push(&mut p.1);
        }
        if let Some(f) = self.full.as_mut() {
            cols.push(f);
        }
        let mut t = (0, 0);
        for c in cols {
            for (m, p) in [c.l23.compact(idle), c.l5.compact(idle)] {
                t.0 += m;
                t.1 += p;
            }
        }
        t
    }

    /// What area `a`'s layer 4 code `code` expects of its raw input (layer 4's back-projection).
    pub fn raw_of(&self, a: usize, code: &BitVector) -> BitVector {
        match self.areas[a].l4.as_ref() {
            Some(l4) => l4.backproject(code, 0),
            None => code.clone(),
        }
    }

    /// Committed cells in every column's layers 2/3 and 5.
    pub fn cortical_cells(&self) -> usize {
        let mut n = 0;
        for a in &self.areas {
            if let Some(c) = a.col.as_ref() {
                n += c.l23.committed() + c.l5.committed();
            }
            if let Some(Motor::Column(c)) = a.motor.as_ref() {
                n += c.l23.committed() + c.l5.committed();
            }
        }
        for p in &self.pairs {
            n += p.1.l23.committed() + p.1.l5.committed();
        }
        if let Some(f) = self.full.as_ref() {
            n += f.l23.committed() + f.l5.committed();
        }
        n
    }

    pub fn sense_hits(&self) -> (u64, u64) {
        self.areas.iter().filter_map(|a| a.col.as_ref()).fold((0, 0), |a, c| (a.0 + c.hits.0, a.1 + c.hits.1))
    }

    /// The association columns' predictions: (checked, right), over pairs and the full area.
    pub fn assoc_hits(&self) -> (u64, u64) {
        self.pairs.iter().map(|p| &p.1).chain(self.full.iter()).fold((0, 0), |a, c| (a.0 + c.hits.0, a.1 + c.hits.1))
    }

    /// Layer 4's code for `input` in area `a`, without learning (from a shared reference).
    pub fn encode_only(&self, a: usize, input: &BitVector) -> BitVector {
        match self.areas[a].l4.as_ref() {
            Some(p) if input.count_ones() > 0 => p.encode(&[(0, input)]),
            _ => BitVector::new(self.bits, Some(0)),
        }
    }

    /// As `associate`, for given codes (one per area, in order; empty for a silent area).
    pub fn associate_codes(&mut self, codes: &[BitVector], learn: bool) -> BitVector {
        let active: Vec<usize> = (0..codes.len()).filter(|&i| codes[i].count_ones() > 0).collect();
        if learn {
            for x in 0..active.len() {
                for y in x + 1..active.len() {
                    let key = (active[x], active[y]);
                    if self.pairs.iter().any(|p| p.0 == key) || self.pairs.len() >= MAX_PAIRS {
                        continue;
                    }
                    let n = self.together.entry(key).or_default();
                    *n += 1;
                    if *n >= self.pair_after {
                        self.together.remove(&key);
                        let seed = self.seed.wrapping_add(1000 + 31 * self.pairs.len() as u64);
                        let mut col = Column::granular(self.bits, Pool::new(2, self.bits, self.bits, self.k, 2, seed), CAP);
                        col.set_persistence(self.pair_tau);
                        self.pairs.push((key, col));
                    }
                }
            }
        }
        let mut pair_codes: Vec<BitVector> = Vec::with_capacity(self.pairs.len());
        let silent = BitVector::new(self.bits, Some(0));
        // (self-organizing: their layers 2/3 and 5 are stepped by `step_columns`)
        let columns = self.columns && !self.selforg;
        for ((a, b), col) in self.pairs.iter_mut() {
            // an area not given is silent
            let input = [(0usize, codes.get(*a).unwrap_or(&silent)), (1usize, codes.get(*b).unwrap_or(&silent))];
            pair_codes.push(Self::code_of(col, &input, learn, columns, &mut self.rng));
        }
        if self.full.is_none() && self.pairs.len() >= 2 {
            let mut col = Column::granular(self.bits, Pool::new(MAX_PAIRS, self.bits, self.bits, self.k, 2, self.seed.wrapping_add(9999)), CAP);
            col.set_persistence(self.full_tau);
            self.full = Some(col);
        }
        // the highest level active: the full area when it has a code, else the pairwise codes
        if let Some(full) = self.full.as_mut() {
            let input: Vec<(usize, &BitVector)> = pair_codes.iter().enumerate().filter(|(_, c)| c.count_ones() > 0).collect();
            let c = Self::code_of(full, &input, learn, columns, &mut self.rng);
            if c.count_ones() > 0 {
                return c;
            }
        }
        let mut merged = BitVector::new(self.bits, Some(0));
        for c in &pair_codes {
            for (w, x) in merged.as_words_mut().iter_mut().zip(c.as_words()) {
                *w |= x;
            }
        }
        merged
    }
}

impl Brain {
    /// The mirror: area `from`'s code alone (an act seen) evokes, through the association area
    /// of the pair (`from`, `to`), the cells that conjoined it with `to` when both were active
    /// (own acts, seen and felt); their synapses on `to` project back to `to`'s layer 4 code, and
    /// that code back to `to`'s input: what `to` would sense (the act as if felt). Nothing if
    /// the pair has no association area yet.
    pub fn mirror(&self, from: usize, code: &BitVector, to: usize) -> BitVector {
        let none = BitVector::new(self.bits, Some(0));
        let Some(((a, _), col)) = self.pairs.iter().find(|p| p.0 == (from, to) || p.0 == (to, from)) else {
            return none;
        };
        let pool = col.l4.as_ref().unwrap();
        let (side_from, side_to) = if *a == from { (0, 1) } else { (1, 0) };
        let cells = pool.encode(&[(side_from, code)]);
        if cells.count_ones() == 0 {
            return none;
        }
        let to_code = pool.backproject(&cells, side_to);
        match self.areas[to].l4.as_ref() {
            Some(l4) => l4.backproject(&to_code, 0),
            None => to_code,
        }
    }

    /// Top-down: the cells `cells` of the association area of the pair (`a`, `b`) project back
    /// to area `to` (one of the pair), and on to its input: what that area should sense if the
    /// association area's pattern is right (an expectation sent down).
    pub fn down(&self, a: usize, b: usize, cells: &BitVector, to: usize) -> BitVector {
        let none = BitVector::new(self.bits, Some(0));
        let Some(((x, _), col)) = self.pairs.iter().find(|p| p.0 == (a, b) || p.0 == (b, a)) else {
            return none;
        };
        let side = if *x == to { 0 } else { 1 };
        let to_code = col.l4.as_ref().unwrap().backproject(cells, side);
        match self.areas[to].l4.as_ref() {
            Some(l4) => l4.backproject(&to_code, 0),
            None => to_code,
        }
    }

    /// What area `a` sends up (feedforward, from layer 2/3): its layer 4 code `code`, with its
    /// sensory column's layer 2/3 activity (the input in its context) when it has one.
    pub fn feedforward(&self, a: usize, code: &BitVector) -> BitVector {
        let mut v = code.clone();
        if code.count_ones() > 0 {
            if let Some(c) = self.areas[a].col.as_ref() {
                for (w, x) in v.as_words_mut().iter_mut().zip(c.activity().as_words()) {
                    *w |= x;
                }
            }
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    const BITS: usize = 2048;
    const K: usize = 32;

    fn rand_code(rng: &mut StdRng) -> BitVector {
        let all: Vec<usize> = (0..BITS).collect();
        BitVector::from_bits(&all.choose_multiple(rng, K).copied().collect::<Vec<_>>(), BITS)
    }

    fn ov(a: &BitVector, b: &BitVector) -> u32 {
        a.as_words().iter().zip(b.as_words()).map(|(x, y)| (x & y).count_ones()).sum()
    }

    #[test]
    fn a_sensory_area_grows_for_new_inputs_and_keeps_its_codes() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut brain = Brain::new(BITS, K, 1);
        let eye = brain.attach("eye", Kind::Sensor);
        let inputs: Vec<BitVector> = (0..20).map(|_| rand_code(&mut rng)).collect();
        let first: Vec<BitVector> = inputs.iter().map(|x| brain.areas[eye].sense(x, true)).collect();
        let grown = brain.areas[eye].l4.as_ref().unwrap().cells();
        assert_eq!(grown, 20 * K, "one code per new input");
        for _ in 0..5 {
            for x in &inputs {
                brain.areas[eye].sense(x, true);
            }
        }
        assert_eq!(brain.areas[eye].l4.as_ref().unwrap().cells(), grown, "no growth on familiar inputs");
        for (x, f) in inputs.iter().zip(&first) {
            let now = brain.areas[eye].sense(x, false);
            assert!(ov(&now, f) * 2 >= K as u32, "the code stays put");
        }
    }

    #[test]
    fn association_areas_grow_for_pairs_then_a_full_one() {
        let mut rng = StdRng::seed_from_u64(2);
        let mut brain = Brain::new(BITS, K, 2);
        let eye = brain.attach("eye", Kind::Sensor);
        let ear = brain.attach("ear", Kind::Sensor);
        let hand = brain.attach("hand", Kind::SensorMotor);
        let words: Vec<(BitVector, BitVector)> = (0..10).map(|_| (rand_code(&mut rng), rand_code(&mut rng))).collect();
        // seen and heard together: one association area, eye–ear
        let mut pair_code = Vec::new();
        for round in 0..3 {
            for (seen, heard) in &words {
                brain.areas[eye].sense(seen, true);
                brain.areas[ear].sense(heard, true);
                brain.areas[hand].sense(&BitVector::new(BITS, Some(0)), true);
                let c = brain.associate(true);
                if round == 2 {
                    pair_code.push(c);
                }
            }
        }
        assert_eq!(brain.pairs.len(), 1);
        assert!(brain.full.is_none());
        // the word heard alone evokes most of what it evoked seen and heard
        let mut evoked = 0;
        for ((_, heard), pc) in words.iter().zip(&pair_code) {
            brain.areas[eye].sense(&BitVector::new(BITS, Some(0)), false);
            brain.areas[ear].sense(heard, false);
            let c = brain.associate(false);
            evoked += (ov(&c, pc) * 2 >= K as u32) as usize;
        }
        assert!(evoked >= 9, "heard alone evoked the pair's code for {evoked} of 10");
        // heard alone (learning) many times: the cells keep their eye half, so seeing alone
        // still evokes the pair's code
        for _ in 0..20 {
            for (_, heard) in &words {
                brain.areas[eye].sense(&BitVector::new(BITS, Some(0)), true);
                brain.areas[ear].sense(heard, true);
                brain.associate(true);
            }
        }
        let mut kept = 0;
        for ((seen, _), pc) in words.iter().zip(&pair_code) {
            brain.areas[eye].sense(seen, false);
            brain.areas[ear].sense(&BitVector::new(BITS, Some(0)), false);
            kept += (ov(&brain.associate(false), pc) * 2 >= K as u32) as usize;
        }
        assert!(kept >= 9, "seen alone evoked the pair's code for {kept} of 10 after hearing alone");
        // top-down: the pair's code for a word projects back to the sound it should be heard as
        let mut down = 0;
        for ((_, heard), pc) in words.iter().zip(&pair_code) {
            down += (ov(&brain.down(eye, ear, pc, ear), heard) * 2 >= K as u32) as usize;
        }
        assert!(down >= 9, "the pair's code projected down to the right sound for {down} of 10");
        // now the hand moves while words are heard: a second pair, and the full area
        let motions: Vec<BitVector> = (0..3).map(|_| rand_code(&mut rng)).collect();
        for i in 0..30 {
            brain.areas[eye].sense(&BitVector::new(BITS, Some(0)), true);
            brain.areas[ear].sense(&words[i % 10].1, true);
            brain.areas[hand].sense(&motions[i % 3], true);
            brain.associate(true);
        }
        assert_eq!(brain.pairs.len(), 2);
        assert!(brain.full.is_some());
    }

    #[test]
    fn a_seen_act_evokes_how_it_feels() {
        let mut rng = StdRng::seed_from_u64(4);
        let mut brain = Brain::new(BITS, K, 4);
        let eye = brain.attach("eye", Kind::Sensor);
        let hand = brain.attach("hand", Kind::SensorMotor);
        let seen: Vec<BitVector> = (0..3).map(|_| rand_code(&mut rng)).collect();
        let felt: Vec<BitVector> = (0..3).map(|_| rand_code(&mut rng)).collect();
        // before any own act, seeing evokes nothing
        let e = brain.encode(eye, &seen[0], false);
        assert_eq!(brain.mirror(eye, &e, hand).count_ones(), 0);
        // own acts: each seen and felt together
        for i in 0..30 {
            let e = brain.areas[eye].sense(&seen[i % 3], true);
            let h = brain.areas[hand].sense(&felt[i % 3], true);
            brain.associate_codes(&[e, h], true);
        }
        // someone else's act, seen only: it evokes how it would feel
        for i in 0..3 {
            let e = brain.encode(eye, &seen[i], false);
            let m = brain.mirror(eye, &e, hand);
            assert!(ov(&m, &felt[i]) * 2 >= K as u32, "act {i}: {} of {K} felt bits evoked", ov(&m, &felt[i]));
            for j in (0..3).filter(|&j| j != i) {
                assert!(ov(&m, &felt[j]) * 4 < K as u32, "act {i} evokes act {j}");
            }
        }
    }

    #[test]
    fn association_columns_predict_the_next_scene() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut brain = Brain::new(BITS, K, 5);
        brain.columns = true;
        let eye = brain.attach("eye", Kind::Sensor);
        let ear = brain.attach("ear", Kind::Sensor);
        let voice = brain.attach("voice", Kind::Motor);
        assert!(matches!(brain.areas[voice].motor, Some(Motor::Column(_))));
        // a story read aloud, over and over: word i seen and heard, then word i+1
        let words: Vec<(BitVector, BitVector)> = (0..8).map(|_| (rand_code(&mut rng), rand_code(&mut rng))).collect();
        for _ in 0..40 {
            for (seen, heard) in &words {
                brain.areas[eye].sense(seen, true);
                brain.areas[ear].sense(heard, true);
                brain.associate(true);
            }
        }
        let before = brain.assoc_hits();
        for (seen, heard) in &words {
            brain.areas[eye].sense(seen, true);
            brain.areas[ear].sense(heard, true);
            brain.associate(true);
        }
        let after = brain.assoc_hits();
        let (n, right) = (after.0 - before.0, after.1 - before.1);
        assert!(right * 8 >= n * 7, "the association column predicted the next scene {right} of {n} times");
    }

    #[test]
    fn an_act_that_does_nothing_is_an_effect_error() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut brain = Brain::new(BITS, K, 3);
        let hand = brain.attach("hand", Kind::SensorMotor);
        let cmds: Vec<BitVector> = (0..3).map(|_| rand_code(&mut rng)).collect();
        let felt: Vec<BitVector> = (0..3).map(|_| rand_code(&mut rng)).collect();
        // babbling: each command and what it is felt as
        for _ in 0..10 {
            for (c, f) in cmds.iter().zip(&felt) {
                brain.areas[hand].learn_act(c, f, &mut rng);
            }
        }
        brain.areas[hand].issue(&cmds[0]);
        brain.areas[hand].sense(&felt[0], false);
        assert!(brain.areas[hand].effect_error.unwrap() < crate::fixed::ONE / 4, "as expected");
        brain.areas[hand].issue(&cmds[1]);
        brain.areas[hand].sense(&BitVector::new(BITS, Some(0)), false);
        assert_eq!(brain.areas[hand].effect_error, Some(crate::fixed::ONE), "nothing happened");
    }
}
