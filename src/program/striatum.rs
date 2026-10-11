//! The basal ganglia loop as bit cells: the striatum (the input nucleus), the external and
//! internal pallidum, the subthalamic nucleus, the dopamine neurons and the striatum's
//! interneurons; an actor–critic made of populations, with three-factor learning.
//!
//! - **Inputs** are the active bits of the pathways the striatum sees (layer 5 and layer 2/3 of
//!   the cortex, working memory, the hippocampus's output, the senses), given as groups, one per
//!   pathway. Nothing else: no hand-made features.
//! - **Spiny cells are popcount units.** Each has `SYN` synapses on input bits and fires when at
//!   least `THETA` of them are active (the up state needs many coincident inputs). A cell is
//!   recruited for a state by sampling its synapses from one active pathway (corticostriatal
//!   projections are topographic), so it becomes a detector for states like that one.
//! - **Channels and actions:** each action of each channel has a population of D1 cells (the
//!   direct, "go" pathway) and of D2 cells (the indirect, "no-go" pathway).
//! - **Fast-spiking interneurons** normalise the striatum's output per channel (feedforward
//!   inhibition): the D1 and D2 counts are scaled so their total is at most `POP`.
//! - **The rest of the loop is populations of `POP` cells,** each cell with its own threshold, so
//!   a population's response is the number of cells whose threshold the net input reaches:
//!   - **GPe** (per action): tonically active, inhibited by its action's D2 cells;
//!   - **STN** (per channel): a tonic baseline, excited by the cortex when several options are
//!     driven about as strongly as the strongest (the hyperdirect path: conflict) and inhibited
//!     by GPe;
//!   - **GPi / SNr** (per action): tonically active, inhibited by its action's D1 cells (direct
//!     path) and by GPe, excited by STN, with noisy tonic firing (exploration).
//!   
//!   An action is released (its thalamus disinhibited) when its GPi falls at least `RELEASE`
//!   below tonic; the most released wins; if none is, nothing is released and the default
//!   (action 0: no action, never learned) holds. Conflict raises STN and so every GPi: the
//!   loop waits.
//! - **The critic:** striosome cells. A state's value is the number of striosome cells firing,
//!   each worth `UNIT` of reward.
//! - **Dopamine neurons** (`POP` cells, tonic baseline `DA_BASE`): reward and the next state's
//!   value excite them, the current state's striosome cells inhibit them. Their count above or
//!   below baseline is the temporal-difference error: bursts can rise far, dips only to silence.
//! - **Cholinergic interneurons** (tonically active) pause when dopamine fires phasically; only
//!   during a pause do eligibility tags become lasting changes.
//! - **Learning is three-factor and structural.** A state and the choices made in it stay
//!   eligible for `WINDOW` steps, their credit falling to four-fifths per step back (a synaptic
//!   tag). A burst
//!   (δ > 0) recruits striosome cells for the eligible states (their value was too low) and D1
//!   cells for the chosen actions, and removes D2 cells that fired for them; a dip (δ < 0) does
//!   the reverse. The number of cells changed follows |δ| / `UNIT`, rounded at random.

use rand::Rng;

use crate::fixed::ONE;

/// Synapses per cell, and how many must be active for it to fire.
const SYN: usize = 12;
const THETA: usize = 8;
/// Reward per striosome cell firing (an eighth of a reward), and per dopamine cell above or
/// below baseline.
const UNIT: i32 = (ONE / 8) as i32;
/// The learning rate: each change happens with probability 1 / `RATE_DIV`.
const RATE_DIV: u64 = 4;
/// Steps a state and its choices stay eligible.
const WINDOW: usize = 8;
/// Cells per population at most.
const CAP: usize = 4096;
/// Cells in each pallidal, subthalamic and dopamine population.
const POP: i32 = 32;
/// Tonic levels: GPe, STN, GPi, dopamine.
const GPE_TONIC: i32 = 24;
const STN_TONIC: i32 = 8;
const GPI_TONIC: i32 = 24;
const DA_BASE: i32 = 8;
/// How far below tonic a GPi population must fall to release its action.
const RELEASE: i32 = 3;
/// Discounting: the next state's value counts 1 − 1/`GAMMA_DIV`.
const GAMMA_DIV: i32 = 8;
/// Options whose go drive is within this many cells of the strongest compete (conflict).
const CLOSE: i32 = 2;
/// Hyperdirect excitation of STN per extra option driven at once.
const HYPER: i32 = 4;

#[derive(Clone)]
struct Spiny {
    syn: [u32; SYN],
    /// when it last fired (the striatum's step count)
    last: std::cell::Cell<u32>,
}

#[derive(Default)]
struct Pool {
    cells: Vec<Spiny>,
}

impl Pool {
    /// The cells that fire for `active`; each one firing is marked as used at `now`.
    fn firing(&self, active: &dyn Fn(u32) -> bool, now: u32) -> Vec<usize> {
        let f: Vec<usize> = self.cells.iter().enumerate().filter(|(_, c)| c.syn.iter().filter(|&&s| active(s)).count() >= THETA).map(|(i, _)| i).collect();
        for &i in &f {
            self.cells[i].last.set(now);
        }
        f
    }

    /// Add a cell. A full pool makes room first: of a few cells sampled, the one unused longest
    /// is pruned (spines that are not used are lost).
    fn add<R: Rng>(&mut self, c: Spiny, prune: bool, rng: &mut R, changes: &mut (u64, u64)) {
        if self.cells.len() >= CAP {
            if !prune {
                return;
            }
            let i = (0..8).map(|_| rng.gen_range(0..self.cells.len())).min_by_key(|&i| self.cells[i].last.get()).unwrap();
            self.cells.swap_remove(i);
            changes.1 += 1;
        }
        self.cells.push(c);
        changes.0 += 1;
    }
}

/// One eligible step: the state's groups and the choices made (channel, action, and for a
/// candidate chosen with `choose_among`, the candidate's own groups).
#[derive(Clone)]
struct Step {
    groups: Vec<Vec<u32>>,
    choices: Vec<(usize, usize, Vec<Vec<u32>>)>,
    /// false for a re-armed tag (an old state): the critic does not learn from it
    critic: bool,
}

pub struct Striatum {
    /// striosome cells (the critic)
    striosome: Pool,
    /// channel → action → (D1 pool, D2 pool)
    actors: Vec<Vec<(Pool, Pool)>>,
    /// the input bitset (input bits hashed into `space` bits) of the current state
    space: usize,
    active: Vec<u64>,
    /// recent steps, newest last
    recent: Vec<Step>,
    value: i32,
    reward: i32,
    started: bool,
    /// exploration: GPi's tonic firing varies by up to ±`noise` cells per action
    pub noise: u32,
    /// (TD errors applied, sum of |δ|) for reports
    pub stats: (u64, u64),
    /// the latest TD error (`Q16`), for reports
    pub last_delta: i32,
    /// cells recruited and removed
    pub changes: (u64, u64),
    /// cholinergic pauses (plasticity windows)
    pub pauses: u64,
    /// steps so far (for when cells were last used)
    clock: u32,
    /// a full pool prunes its least used cell to take a new one (else it takes none)
    pub prune: bool,
    /// the same, for the striosome (critic) pool only
    pub prune_critic: bool,
    /// a recruited cell takes its synapses from two pathways (a conjunction), else from one
    pub mixed: bool,
    /// the smallest dopamine change (in units) that makes the cholinergic interneurons pause
    pub pause_min: i32,
    /// the next value counts 1 − 1/`gamma_div` (0: no discounting)
    pub gamma_div: i32,
    /// the critic learns only for the newest state (TD(0)); else for every eligible state
    pub critic_td0: bool,
    /// dips are read by high-affinity D2 receptors and reach as far as bursts; else they stop
    /// at silence
    pub d2_affinity: bool,
    /// per channel: the gating choice whose content is being held, with its state (its tag,
    /// kept alive by the held activity)
    maintained: Vec<Option<Step>>,
}

fn mix(x: u64) -> u64 {
    let mut y = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    y ^= y >> 29;
    y = y.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    y ^ (y >> 31)
}

impl Striatum {
    /// The input space has `space` bits (a multiple of 64); input bits are hashed into it.
    pub fn new(space: usize) -> Self {
        Self {
            striosome: Pool::default(),
            actors: Vec::new(),
            space,
            active: vec![0; space / 64],
            recent: Vec::new(),
            value: 0,
            reward: 0,
            started: false,
            noise: 4,
            stats: (0, 0),
            last_delta: 0,
            changes: (0, 0),
            pauses: 0,
            clock: 0,
            prune: false,
            prune_critic: true,
            mixed: false,
            pause_min: 2,
            gamma_div: GAMMA_DIV,
            critic_td0: true,
            d2_affinity: true,
            maintained: Vec::new(),
        }
    }

    fn hash(&self, b: usize) -> u32 {
        (mix(b as u64 ^ 0x57A7) % self.space as u64) as u32
    }

    fn is_active(&self, s: u32) -> bool {
        self.active[s as usize / 64] >> (s % 64) & 1 == 1
    }

    fn ensure(&mut self, channel: usize, n: usize) {
        while self.actors.len() <= channel {
            self.actors.push(Vec::new());
        }
        let ch = &mut self.actors[channel];
        while ch.len() < n {
            ch.push((Pool::default(), Pool::default()));
        }
    }

    /// A cell for a state: synapses sampled from one of its pathways (one with enough bits), or
    /// with `mixed`, half from each of two (a conjunction of two pathways converging).
    fn recruit<R: Rng>(groups: &[Vec<u32>], mixed: bool, now: u32, rng: &mut R) -> Option<Spiny> {
        let live: Vec<&Vec<u32>> = groups.iter().filter(|g| g.len() >= THETA).collect();
        if live.is_empty() {
            return None;
        }
        let g = live[rng.gen_range(0..live.len())];
        let h = if mixed && live.len() > 1 {
            loop {
                let h = live[rng.gen_range(0..live.len())];
                if !std::ptr::eq(h, g) {
                    break h;
                }
            }
        } else {
            g
        };
        let mut syn = [0u32; SYN];
        for (i, s) in syn.iter_mut().enumerate() {
            let src = if i % 2 == 0 { g } else { h };
            *s = src[rng.gen_range(0..src.len())];
        }
        Some(Spiny { syn, last: std::cell::Cell::new(now) })
    }

    /// A cell for a candidate in a state: half its synapses from one of the state's pathways,
    /// half from one of the candidate's (a conjunction: "in states like this, candidates like
    /// that").
    fn recruit_pair<R: Rng>(groups: &[Vec<u32>], extra: &[Vec<u32>], now: u32, rng: &mut R) -> Option<Spiny> {
        let live = |gs: &[Vec<u32>]| -> Vec<Vec<u32>> { gs.iter().filter(|g| g.len() >= THETA / 2).cloned().collect() };
        let (a, b) = (live(groups), live(extra));
        if b.is_empty() {
            return Self::recruit(groups, false, now, rng);
        }
        let h = &b[rng.gen_range(0..b.len())];
        let g = if a.is_empty() { h } else { &a[rng.gen_range(0..a.len())] };
        let mut syn = [0u32; SYN];
        for (i, s) in syn.iter_mut().enumerate() {
            let src = if i % 2 == 0 { g } else { h };
            *s = src[rng.gen_range(0..src.len())];
        }
        Some(Spiny { syn, last: std::cell::Cell::new(now) })
    }

    /// Apply one dopamine signal to the eligible steps (newest first, credit fading per step).
    fn learn<R: Rng>(&mut self, delta: i32, rng: &mut R) {
        self.stats.0 += 1;
        self.stats.1 += delta.unsigned_abs() as u64;
        self.last_delta = delta;
        // the cholinergic interneurons pause only at phasic dopamine (at least `pause_min`
        // units); the actors' tags become lasting changes only during a pause. The critic
        // learns from every error.
        if delta == 0 {
            return;
        }
        let pause = delta.abs() >= self.pause_min * UNIT;
        self.pauses += pause as u64;
        let mut weight = ONE as u64;
        for k in (0..self.recent.len()).rev() {
            // cells to change for this step: |δ| / UNIT × its credit × the rate (a change happens
            // with probability RATE: plasticity is probabilistic), rounded at random
            let amount = delta.unsigned_abs() as u64 * weight / UNIT as u64 / RATE_DIV;
            let mut n = (amount >> 16) as usize;
            if rng.gen_range(0..ONE as u64) < (amount & 0xFFFF) {
                n += 1;
            }
            // the tag fades: credit four-fifths per step back
            weight = weight * 4 / 5;
            if n == 0 {
                continue;
            }
            let st = self.recent[k].clone();
            // the cells that fired in that state (counted now: the pools change in between)
            let mut bits: Vec<u32> = st.groups.iter().flatten().copied().collect();
            bits.sort_unstable();
            let was = |s: u32| bits.binary_search(&s).is_ok();
            // the critic: a burst adds striosome cells for the state the error belongs to (the
            // newest; earlier states' values are corrected in turn, step by step), a dip removes
            // some that fired
            if !st.critic || (self.critic_td0 && k + 1 != self.recent.len()) {
            } else if delta > 0 {
                for _ in 0..n {
                    if let Some(c) = Self::recruit(&st.groups, self.mixed, self.clock, rng) {
                        self.striosome.add(c, self.prune || self.prune_critic, rng, &mut self.changes);
                    }
                }
            } else {
                let fired = self.striosome.firing(&was, self.clock);
                remove(&mut self.striosome, &fired, n, rng, &mut self.changes.1);
            }
            // the actors: a burst adds go cells and removes no-go cells for the chosen actions;
            // a dip adds no-go cells and removes go cells
            if !pause {
                continue;
            }
            for (ch, a, extra) in st.choices {
                // a candidate's cells see the state and the candidate together
                let mut cbits = bits.clone();
                cbits.extend(extra.iter().flatten().copied());
                cbits.sort_unstable();
                let was_c = |s: u32| cbits.binary_search(&s).is_ok();
                let (go, nogo) = &mut self.actors[ch][a];
                let (grow, shrink) = if delta > 0 { (go, nogo) } else { (nogo, go) };
                let fired = shrink.firing(&was_c, self.clock);
                for _ in 0..n {
                    let c = if extra.is_empty() { Self::recruit(&st.groups, self.mixed, self.clock, rng) } else { Self::recruit_pair(&st.groups, &extra, self.clock, rng) };
                    if let Some(c) = c {
                        grow.add(c, self.prune, rng, &mut self.changes);
                    }
                }
                remove(shrink, &fired, n, rng, &mut self.changes.1);
            }
        }
    }

    /// The dopamine neurons: reward and the next value excite them, the current value (the
    /// striosome cells) inhibits them; `drive` is that sum (`Q16`). Their count above or below
    /// baseline, in units, is the error: a burst can rise to `POP` − `DA_BASE`. A dip silences
    /// at most `DA_BASE` cells, but its depth is in how long the pause lasts, which the
    /// high-affinity D2 receptors read (`d2_affinity`): it then reaches as far as a burst.
    /// Returned in `Q16`.
    fn dopamine(&self, drive: i32) -> i32 {
        if drive < 0 && self.d2_affinity {
            // a dip, read by high-affinity D2 receptors, which sense how long the pause lasts:
            // it reaches as far as a burst, at the same resolution
            return (drive / UNIT).max(DA_BASE - POP) * UNIT;
        }
        let count = (DA_BASE + drive / UNIT).clamp(0, POP);
        (count - DA_BASE) * UNIT
    }

    /// The value of the current state (`Q16`): striosome cells firing × `UNIT`.
    fn value_now(&self) -> i32 {
        self.striosome.firing(&|s| self.is_active(s), self.clock).len() as i32 * UNIT
    }

    /// A new state, as groups of active input bits (one per pathway); `learn`: update. The
    /// dopamine for the step that ends here is the reward gathered + ⅞ V(this state) − V(the
    /// previous one).
    pub fn begin_groups<R: Rng>(&mut self, state: &[Vec<usize>], learn: bool, rng: &mut R) {
        self.clock += 1;
        self.active.iter_mut().for_each(|w| *w = 0);
        let groups: Vec<Vec<u32>> = state.iter().map(|g| g.iter().map(|&b| self.hash(b)).collect()).collect();
        for g in &groups {
            for &s in g {
                self.active[s as usize / 64] |= 1 << (s % 64);
            }
        }
        let v = self.value_now();
        if self.started && learn {
            // the next value counts seven-eighths (discounting): a steady reward rate is then
            // predicted by a steady value, without needing to know how long it will last
            let delta = self.dopamine(self.reward + v - v / self.gamma_div.max(1) * (self.gamma_div > 0) as i32 - self.value);
            self.learn(delta, rng);
        }
        self.recent.push(Step { groups, choices: Vec::new(), critic: true });
        if self.recent.len() > WINDOW {
            self.recent.remove(0);
        }
        self.value = v;
        self.reward = 0;
        self.started = true;
    }

    /// Reward (`Q16`) for the current step.
    pub fn reward(&mut self, r: i32) {
        self.reward += r;
    }

    /// The striatum's output on `channel`: each action's (D1, D2) cells firing, normalised by
    /// the fast-spiking interneurons so the channel's total is at most `POP`.
    fn striatal(&mut self, channel: usize, n: usize) -> Vec<(i32, i32)> {
        self.ensure(channel, n);
        let ch = &self.actors[channel];
        let act = |s: u32| self.is_active(s);
        let raw: Vec<(i32, i32)> = (0..n).map(|a| (ch[a].0.firing(&act, self.clock).len() as i32, ch[a].1.firing(&act, self.clock).len() as i32)).collect();
        let total: i32 = raw.iter().map(|x| x.0 + x.1).sum();
        if total <= POP {
            raw
        } else {
            raw.iter().map(|&(g, s)| (g * POP / total, s * POP / total)).collect()
        }
    }

    /// The loop's GPi / SNr populations on `channel` (cells firing per action), with `noise`
    /// added to each one's tonic drive (0 without exploration).
    fn pallidum(&mut self, channel: usize, n: usize, noise: &[i32]) -> Vec<i32> {
        let st = self.striatal(channel, n);
        Self::pallidal(&st, noise)
    }

    /// GPe, STN and GPi / SNr from the striatum's (D1, D2) counts per option.
    fn pallidal(st: &[(i32, i32)], noise: &[i32]) -> Vec<i32> {
        let n = st.len();
        let pop = |x: i32| x.clamp(0, POP);
        // GPe: tonic, inhibited by D2
        let gpe: Vec<i32> = st.iter().map(|&(_, d2)| pop(GPE_TONIC - d2)).collect();
        // STN: tonic, excited by the cortex when options compete closely (hyperdirect: conflict
        // is several options driven about as strongly as the strongest, not many driven at all),
        // inhibited by GPe (less GPe, more STN)
        let top = st.iter().map(|x| x.0).max().unwrap_or(0);
        let driven = st.iter().filter(|x| x.0 > 0 && x.0 + CLOSE >= top).count() as i32;
        let gpe_mean = gpe.iter().sum::<i32>() / n.max(1) as i32;
        let stn = pop(STN_TONIC + HYPER * (driven - 1).max(0) + (GPE_TONIC - gpe_mean) / 2);
        // GPi / SNr: tonic, excited by STN, inhibited by D1 and by GPe
        (0..n).map(|a| pop(GPI_TONIC + (stn - STN_TONIC) / 2 - st[a].0 + (GPE_TONIC - gpe[a]) + noise[a])).collect()
    }

    /// Each action's release on `channel` now: how far its GPi is below tonic (for reports).
    pub fn drives(&mut self, channel: usize, n: usize) -> Vec<i32> {
        self.pallidum(channel, n, &vec![0; n]).iter().map(|&g| GPI_TONIC - g).collect()
    }

    /// Choose one of `n` actions on `channel`: the action (1..n) whose GPi falls furthest below
    /// tonic, at least `RELEASE`, is released; if none is, nothing happens: action 0, the default,
    /// is not an action but the absence of one, and is not learned. With `rng`, GPi's tonic
    /// firing is noisy (±`noise` per action): exploration. A released action becomes eligible.
    pub fn choose<R: Rng>(&mut self, channel: usize, n: usize, rng: Option<&mut R>) -> usize {
        let noise: Vec<i32> = match rng {
            Some(rng) => (0..n).map(|_| rng.gen_range(-(self.noise as i32)..=self.noise as i32)).collect(),
            None => vec![0; n],
        };
        let gpi = self.pallidum(channel, n, &noise);
        let mut best = 0;
        let mut lowest = GPI_TONIC - RELEASE + 1;
        for (a, &g) in gpi.iter().enumerate().skip(1) {
            if g < lowest {
                lowest = g;
                best = a;
            }
        }
        // nothing released: the thalamus stays shut, there is no efference copy, nothing to tag
        if best == 0 {
            return 0;
        }
        if let Some(st) = self.recent.last_mut() {
            st.choices.push((channel, best, Vec::new()));
        }
        best
    }

    /// The cortex proposes candidates, the loop picks one or none: each candidate is a list of
    /// input groups (its own activity: a bookmark's age and onset, a motor command's bits),
    /// scored by `channel`'s one pair of go / no-go pools on the current state together with
    /// it. The pallidal competition runs across the candidates; if none is released, nothing
    /// is done (`None`, not learned). With `rng`, exploration noise. The chosen candidate
    /// becomes eligible with its groups, so learning generalises to candidates like it.
    pub fn choose_among<R: Rng>(&mut self, channel: usize, candidates: &[Vec<Vec<usize>>], rng: Option<&mut R>) -> Option<usize> {
        if candidates.is_empty() {
            return None;
        }
        self.ensure(channel, 2);
        let hashed: Vec<Vec<Vec<u32>>> = candidates.iter().map(|c| c.iter().map(|g| g.iter().map(|&b| self.hash(b)).collect()).collect()).collect();
        let raw: Vec<(i32, i32)> = hashed
            .iter()
            .map(|c| {
                let mut extra: Vec<u32> = c.iter().flatten().copied().collect();
                extra.sort_unstable();
                let act = |s: u32| self.is_active(s) || extra.binary_search(&s).is_ok();
                let (go, nogo) = &self.actors[channel][1];
                (go.firing(&act, self.clock).len() as i32, nogo.firing(&act, self.clock).len() as i32)
            })
            .collect();
        // the fast-spiking interneurons normalise across the candidates
        let total: i32 = raw.iter().map(|x| x.0 + x.1).sum();
        let st: Vec<(i32, i32)> = if total <= POP { raw } else { raw.iter().map(|&(g, s)| (g * POP / total, s * POP / total)).collect() };
        let noise: Vec<i32> = match rng {
            Some(rng) => (0..st.len()).map(|_| rng.gen_range(-(self.noise as i32)..=self.noise as i32)).collect(),
            None => vec![0; st.len()],
        };
        let gpi = Self::pallidal(&st, &noise);
        let mut best = None;
        let mut lowest = GPI_TONIC - RELEASE + 1;
        for (i, &g) in gpi.iter().enumerate() {
            if g < lowest {
                lowest = g;
                best = Some(i);
            }
        }
        if let (Some(i), Some(st)) = (best, self.recent.last_mut()) {
            st.choices.push((channel, 1, hashed[i].clone()));
        }
        best
    }

    /// Candidate `c` was done on `channel` without the loop choosing it (the cortex initiated
    /// it): it is tagged as a choice would be.
    pub fn did_among(&mut self, channel: usize, candidate: &[Vec<usize>]) {
        self.ensure(channel, 2);
        let h: Vec<Vec<u32>> = candidate.iter().map(|g| g.iter().map(|&b| self.hash(b)).collect()).collect();
        if let Some(st) = self.recent.last_mut() {
            st.choices.push((channel, 1, h));
        }
    }

    /// Action `a` of `n` on `channel` was done without the loop choosing it (the motor cortex
    /// initiated it): its efference copy tags the striatal cells as a choice would.
    pub fn did(&mut self, channel: usize, n: usize, a: usize) {
        self.ensure(channel, n);
        if let Some(st) = self.recent.last_mut() {
            st.choices.push((channel, a, Vec::new()));
        }
    }

    /// Action `a` on `channel` stands for something new now: its go and no-go cells are cleared.
    pub fn reset_action(&mut self, channel: usize, a: usize) {
        if let Some(p) = self.actors.get_mut(channel).and_then(|c| c.get_mut(a)) {
            *p = (Pool::default(), Pool::default());
        }
    }

    /// The action just released on `channel` loaded something that is now held (in the
    /// prefrontal cortex): its tag is kept with the held activity, replacing the tag of what was
    /// held before. Call right after `choose` released it.
    pub fn maintain(&mut self, channel: usize) {
        while self.maintained.len() <= channel {
            self.maintained.push(None);
        }
        self.maintained[channel] = self.recent.last().and_then(|st| {
            let c = st.choices.iter().rev().find(|c| c.0 == channel)?.clone();
            Some(Step { groups: st.groups.clone(), choices: vec![c], critic: false })
        });
    }

    /// What is held on `channel` is being used: the held activity re-arms the tag of the
    /// choice that loaded it, so the dopamine that follows its use also reaches that choice
    /// (credit for gating is given when what was gated is used: O'Reilly & Frank 2006).
    pub fn retag(&mut self, channel: usize) {
        if let Some(Some(st)) = self.maintained.get(channel) {
            let k = self.recent.len().saturating_sub(1);
            self.recent.insert(k, st.clone());
            if self.recent.len() > WINDOW {
                self.recent.remove(0);
            }
        }
    }

    /// (the current state's value, action 1's release − action 0's on `channel`, both `Q16`),
    /// for reports.
    pub fn debug_now(&mut self, channel: usize) -> (i32, i32) {
        let d = self.drives(channel, 2);
        (self.value, (d[1] - d[0]) * ONE as i32)
    }

    /// The episode ends: the last dopamine is the reward − V(state), and eligibility clears.
    pub fn end<R: Rng>(&mut self, learn: bool, rng: &mut R) {
        if self.started && learn {
            let delta = self.dopamine(self.reward - self.value);
            self.learn(delta, rng);
        }
        self.recent.clear();
        self.started = false;
        self.reward = 0;
    }

    /// (striosome cells, go cells, no-go cells), for reports.
    pub fn sizes(&self) -> (usize, usize, usize) {
        let go = self.actors.iter().flatten().map(|p| p.0.cells.len()).sum();
        let nogo = self.actors.iter().flatten().map(|p| p.1.cells.len()).sum();
        (self.striosome.cells.len(), go, nogo)
    }
}

/// Remove up to `n` of the cells that fired (`fired`, by index) from a pool, at random.
fn remove<R: Rng>(pool: &mut Pool, fired: &[usize], n: usize, rng: &mut R, count: &mut u64) {
    let mut f: Vec<usize> = fired.iter().copied().filter(|&i| i < pool.cells.len()).collect();
    for _ in 0..n {
        if f.is_empty() {
            break;
        }
        let i = f.swap_remove(rng.gen_range(0..f.len()));
        // the last cell moves into the removed one's place: fix its index
        let last = pool.cells.len() - 1;
        pool.cells.swap_remove(i);
        for x in f.iter_mut() {
            if *x == last {
                *x = i;
            }
        }
        *count += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    /// A delayed payoff: in state A the agent picks 0 or 1; three neutral steps follow; then a
    /// reward comes only if it picked 1. The credit must reach the choice.
    #[test]
    fn learns_a_delayed_payoff() {
        let mut st = Striatum::new(1 << 14);
        let mut rng = StdRng::seed_from_u64(1);
        let a: Vec<Vec<usize>> = vec![(0..16).collect()];
        let mid = |k: usize| -> Vec<Vec<usize>> { vec![(100 * (k + 1)..100 * (k + 1) + 16).collect()] };
        for _ in 0..400 {
            st.begin_groups(&a, true, &mut rng);
            let c = st.choose(0, 2, Some(&mut rng));
            for k in 0..3 {
                st.begin_groups(&mid(k), true, &mut rng);
            }
            st.reward(if c == 1 { ONE as i32 } else { 0 });
            st.end(true, &mut rng);
        }
        st.begin_groups(&a, false, &mut rng);
        assert_eq!(st.choose::<StdRng>(0, 2, None), 1);
    }

    /// Candidates proposed fresh each trial (a feature and identity bits never seen again): the
    /// right feature depends on the context, and the choice must generalise to new candidates.
    #[test]
    fn chooses_among_new_candidates_by_their_features() {
        let mut st = Striatum::new(1 << 16);
        let mut rng = StdRng::seed_from_u64(5);
        let ctx = |k: usize| -> Vec<Vec<usize>> { vec![(1000 * k + 50..1000 * k + 66).collect()] };
        let feature = |f: usize| -> Vec<usize> { (20000 + 100 * f..20000 + 100 * f + 16).collect() };
        let mut fresh = |rng: &mut StdRng| -> Vec<Vec<Vec<usize>>> {
            let mut fs = vec![0, 1, 2];
            fs.shuffle(rng);
            fs.iter().map(|&f| vec![feature(f), (0..16).map(|_| rng.gen_range(100_000..1_000_000)).collect()]).collect()
        };
        let mut feats = Vec::new();
        for i in 0..1500 {
            let k = i % 2;
            let cands = fresh(&mut rng);
            let fs: Vec<usize> = cands.iter().map(|c| (c[0][0] - 20000) / 100).collect();
            st.begin_groups(&ctx(k), true, &mut rng);
            let c = st.choose_among(0, &cands, Some(&mut rng));
            st.reward(if c.map(|c| fs[c]) == Some(k) { ONE as i32 } else { 0 });
            st.end(true, &mut rng);
            feats.push(fs);
        }
        let mut right = 0;
        for i in 0..100 {
            let k = i % 2;
            let cands = fresh(&mut rng);
            let fs: Vec<usize> = cands.iter().map(|c| (c[0][0] - 20000) / 100).collect();
            st.begin_groups(&ctx(k), false, &mut rng);
            right += (st.choose_among::<StdRng>(0, &cands, None).map(|c| fs[c]) == Some(k)) as usize;
            st.end(false, &mut rng);
        }
        assert!(right >= 90, "{right} of 100");
    }

    /// Two contexts, opposite best actions: the cells must learn each from its own pathway bits.
    #[test]
    fn learns_by_context() {
        let mut st = Striatum::new(1 << 14);
        let mut rng = StdRng::seed_from_u64(2);
        let ctx = |k: usize| -> Vec<Vec<usize>> { vec![(1000 * k..1000 * k + 16).collect(), (5000..5016).collect()] };
        for i in 0..800 {
            let k = i % 2;
            st.begin_groups(&ctx(k), true, &mut rng);
            let c = st.choose(0, 2, Some(&mut rng));
            st.reward(if c == k { ONE as i32 } else { 0 });
            st.end(true, &mut rng);
        }
        for k in 0..2 {
            st.begin_groups(&ctx(k), false, &mut rng);
            assert_eq!(st.choose::<StdRng>(0, 2, None), k, "context {k}");
            st.end(false, &mut rng);
        }
    }
}
