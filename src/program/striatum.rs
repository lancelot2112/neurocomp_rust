//! The striatum, the basal ganglia's input nucleus, as bit cells (an actor–critic made of
//! spiny projection neurons; three-factor learning).
//!
//! - **Inputs** are the active bits of the pathways the striatum sees (layer 5 and layer 2/3 of
//!   the cortex, working memory, the hippocampus's output, the senses), given as groups, one per
//!   pathway. Nothing else: no hand-made features.
//! - **Spiny cells are popcount units.** Each has `SYN` synapses on input bits and fires when at
//!   least `THETA` of them are active (the up state needs many coincident inputs). A cell is
//!   recruited for a state by sampling its synapses from one active pathway (corticostriatal
//!   projections are topographic), so it becomes a detector for states like that one.
//! - **Channels and actions:** each action of each channel has a population of D1 cells (the
//!   direct, "go" pathway) and of D2 cells (the indirect, "no-go" pathway). An action's drive is
//!   the number of its D1 cells firing minus its D2 cells firing, plus a little noise; the
//!   output nuclei (GPi / SNr) release the strongest (action 0, the default, wins ties).
//! - **The critic:** striosome cells, which project to the dopamine neurons. A state's value is
//!   the number of striosome cells firing, each worth `UNIT` of reward.
//! - **Dopamine is the temporal-difference error,** δ = reward + V(next) − V(state), in integers.
//! - **Learning is three-factor and structural.** A state and the choices made in it stay
//!   eligible for `WINDOW` steps, their credit halving per step back (a synaptic tag). A burst
//!   (δ > 0) recruits striosome cells for the eligible states (their value was too low) and D1
//!   cells for the chosen actions, and removes D2 cells that fired for them; a dip (δ < 0) does
//!   the reverse. The number of cells changed follows |δ| / `UNIT`, rounded at random.

use rand::Rng;

use crate::fixed::ONE;

/// Synapses per cell, and how many must be active for it to fire.
const SYN: usize = 12;
const THETA: usize = 8;
/// Reward per striosome cell firing.
const UNIT: i32 = (ONE / 2) as i32;
/// Steps a state and its choices stay eligible.
const WINDOW: usize = 8;
/// Cells per population at most.
const CAP: usize = 4096;

#[derive(Clone)]
struct Cell {
    syn: [u32; SYN],
}

#[derive(Default)]
struct Pool {
    cells: Vec<Cell>,
}

impl Pool {
    fn firing(&self, active: &dyn Fn(u32) -> bool) -> Vec<usize> {
        self.cells.iter().enumerate().filter(|(_, c)| c.syn.iter().filter(|&&s| active(s)).count() >= THETA).map(|(i, _)| i).collect()
    }
}

/// One eligible step: the state's groups, the striosome cells that fired, and the choices made
/// (channel, action, D1 and D2 cells that fired for it).
#[derive(Clone)]
struct Step {
    groups: Vec<Vec<u32>>,
    striosome: Vec<usize>,
    choices: Vec<(usize, usize, Vec<usize>, Vec<usize>)>,
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
    /// exploration noise: each action's drive gets 0..=`noise` added
    pub noise: u32,
    /// (TD errors applied, sum of |δ|) for reports
    pub stats: (u64, u64),
    /// the latest TD error (`Q16`), for reports
    pub last_delta: i32,
    /// cells recruited and removed
    pub changes: (u64, u64),
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
            noise: 1,
            stats: (0, 0),
            last_delta: 0,
            changes: (0, 0),
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

    /// A cell for a state: synapses sampled from one of its pathways (one with enough bits).
    fn recruit<R: Rng>(groups: &[Vec<u32>], rng: &mut R) -> Option<Cell> {
        let live: Vec<&Vec<u32>> = groups.iter().filter(|g| g.len() >= THETA).collect();
        if live.is_empty() {
            return None;
        }
        let g = live[rng.gen_range(0..live.len())];
        let mut syn = [0u32; SYN];
        for s in syn.iter_mut() {
            *s = g[rng.gen_range(0..g.len())];
        }
        Some(Cell { syn })
    }

    /// Apply one dopamine signal to the eligible steps (newest first, credit halving per step).
    fn learn<R: Rng>(&mut self, delta: i32, rng: &mut R) {
        self.stats.0 += 1;
        self.stats.1 += delta.unsigned_abs() as u64;
        self.last_delta = delta;
        let mut weight = ONE as u64;
        for k in (0..self.recent.len()).rev() {
            // cells to change for this step: |δ| / UNIT × its credit, rounded at random
            let amount = delta.unsigned_abs() as u64 * weight / UNIT as u64;
            let mut n = (amount >> 16) as usize;
            if rng.gen_range(0..ONE as u64) < (amount & 0xFFFF) {
                n += 1;
            }
            weight /= 2;
            if n == 0 {
                continue;
            }
            let st = self.recent[k].clone();
            // the critic: a burst adds striosome cells for this state, a dip removes some that fired
            if delta > 0 {
                for _ in 0..n {
                    if self.striosome.cells.len() < CAP {
                        if let Some(c) = Self::recruit(&st.groups, rng) {
                            self.striosome.cells.push(c);
                            self.changes.0 += 1;
                        }
                    }
                }
            } else {
                remove(&mut self.striosome, &st.striosome, n, rng, &mut self.changes.1);
            }
            // the actors: a burst adds go cells and removes no-go cells for the chosen actions;
            // a dip adds no-go cells and removes go cells
            for (ch, a, d1, d2) in st.choices {
                let (go, nogo) = &mut self.actors[ch][a];
                let (grow, shrink, fired) = if delta > 0 { (go, nogo, d2) } else { (nogo, go, d1) };
                for _ in 0..n {
                    if grow.cells.len() < CAP {
                        if let Some(c) = Self::recruit(&st.groups, rng) {
                            grow.cells.push(c);
                            self.changes.0 += 1;
                        }
                    }
                }
                remove(shrink, &fired, n, rng, &mut self.changes.1);
            }
        }
    }

    /// The value of the current state (`Q16`): striosome cells firing × `UNIT`.
    fn value_now(&self) -> i32 {
        self.striosome.firing(&|s| self.is_active(s)).len() as i32 * UNIT
    }

    /// A new state, as groups of active input bits (one per pathway); `learn`: update. The
    /// dopamine for the step that ends here is the reward gathered + V(this state) − V(the
    /// previous one).
    pub fn begin_groups<R: Rng>(&mut self, state: &[Vec<usize>], learn: bool, rng: &mut R) {
        self.active.iter_mut().for_each(|w| *w = 0);
        let groups: Vec<Vec<u32>> = state.iter().map(|g| g.iter().map(|&b| self.hash(b)).collect()).collect();
        for g in &groups {
            for &s in g {
                self.active[s as usize / 64] |= 1 << (s % 64);
            }
        }
        let v = self.value_now();
        if self.started && learn {
            let delta = self.reward + v - self.value;
            self.learn(delta, rng);
        }
        let striosome = self.striosome.firing(&|s| self.is_active(s));
        self.recent.push(Step { groups, striosome, choices: Vec::new() });
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

    /// Each action's drive on `channel` now: D1 cells firing − D2 cells firing.
    pub fn drives(&mut self, channel: usize, n: usize) -> Vec<i32> {
        self.ensure(channel, n);
        let ch = &self.actors[channel];
        let act = |s: u32| self.is_active(s);
        (0..n).map(|a| ch[a].0.firing(&act).len() as i32 - ch[a].1.firing(&act).len() as i32).collect()
    }

    /// Choose one of `n` actions on `channel` (action 0, the default, wins ties). With `rng`,
    /// each drive gets a little noise (exploration). The choice becomes eligible.
    pub fn choose<R: Rng>(&mut self, channel: usize, n: usize, rng: Option<&mut R>) -> usize {
        let mut d = self.drives(channel, n);
        if let Some(rng) = rng {
            for x in d.iter_mut() {
                *x += rng.gen_range(0..=self.noise) as i32;
            }
        }
        let mut best = 0;
        for a in 1..n {
            if d[a] > d[best] {
                best = a;
            }
        }
        let act = |s: u32| self.is_active(s);
        let (go, nogo) = &self.actors[channel][best];
        let (d1, d2) = (go.firing(&act), nogo.firing(&act));
        if let Some(st) = self.recent.last_mut() {
            st.choices.push((channel, best, d1, d2));
        }
        best
    }

    /// (the current state's value, action 1's drive − action 0's on `channel`, both `Q16`), for
    /// reports.
    pub fn debug_now(&mut self, channel: usize) -> (i32, i32) {
        let d = self.drives(channel, 2);
        (self.value, (d[1] - d[0]) * ONE as i32)
    }

    /// The episode ends: the last dopamine is the reward − V(state), and eligibility clears.
    pub fn end<R: Rng>(&mut self, learn: bool, rng: &mut R) {
        if self.started && learn {
            let delta = self.reward - self.value;
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
