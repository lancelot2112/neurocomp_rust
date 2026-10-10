//! The striatum as an actor–critic that learns when to give credit (temporal-difference
//! learning; Sutton & Barto; Schultz, Dayan & Montague 1997).
//!
//! - **State is a pattern:** the active bits of whatever the striatum sees (the prefrontal
//!   content, an association-area assembly, mode signals). Each bit is a cortical input with
//!   its own synapses, so states that share bits share what they learned.
//! - **The critic** (patch / ventral striatum) learns the state's value: the mean weight of its
//!   active inputs.
//! - **Actors** (matrix / dorsal striatum): one channel per gate (e.g. reach, hold, reinstate,
//!   load into working memory), each with a few actions. An action's preference in a state is
//!   the mean weight of the state's inputs on that action's cells (a fixed hash of input bit,
//!   channel and action picks the synapse). The best action is released; with exploration, a
//!   random one at a small rate.
//! - **Dopamine is the temporal-difference error:** δ = reward + γ·V(next state) − V(state).
//!   Credit arrives when the outlook changes, not when a rule says so: a choice whose payoff
//!   comes many steps later is credited because the critic learns that it raises the value of
//!   what follows.
//! - **Eligibility traces** (λ): the inputs of recent states and the synapses of recent choices
//!   stay eligible, decaying by λ each step, so one dopamine signal reaches them all.
//!
//! Integer arithmetic: weights and values in `Q16`.

use rand::Rng;

use crate::fixed::{Q16, ONE};

pub struct Striatum {
    size: usize,
    critic: Vec<i32>,
    actor: Vec<i32>,
    /// γ and λ, in `Q16`
    pub gamma: Q16,
    pub lambda: Q16,
    /// learning rate: a step of δ >> `shift` for each eligible synapse
    pub shift: u32,
    /// exploration rate, in `Q16`
    pub explore: Q16,
    /// the current state's inputs (synapse indices) and value
    state: Vec<usize>,
    value: i32,
    /// reward gathered since the current state began
    reward: i32,
    /// eligible synapses with their eligibility (`Q16`): (critic or actor, index, weight)
    trace: Vec<(bool, usize, u32)>,
    started: bool,
    /// (TD errors applied, sum of |δ|) for reports
    pub stats: (u64, u64),
}

fn mix(x: u64) -> u64 {
    let mut y = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    y ^= y >> 29;
    y = y.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    y ^ (y >> 31)
}

impl Striatum {
    /// `size` synapses per table (critic, actors).
    pub fn new(size: usize) -> Self {
        Self {
            size,
            critic: vec![0; size],
            actor: vec![0; size],
            gamma: ONE * 97 / 100,
            lambda: ONE * 8 / 10,
            shift: 4,
            explore: ONE / 10,
            state: Vec::new(),
            value: 0,
            reward: 0,
            trace: Vec::new(),
            started: false,
            stats: (0, 0),
        }
    }

    fn critic_syn(&self, b: usize) -> usize {
        (mix(b as u64 ^ 0xC717) % self.size as u64) as usize
    }

    fn actor_syn(&self, b: usize, channel: usize, action: usize) -> usize {
        (mix(b as u64 ^ ((channel as u64 + 1) << 40) ^ ((action as u64 + 1) << 52)) % self.size as u64) as usize
    }

    fn mean(&self, table: &[i32], syns: &[usize]) -> i32 {
        if syns.is_empty() {
            return 0;
        }
        (syns.iter().map(|&s| table[s] as i64).sum::<i64>() / syns.len() as i64) as i32
    }

    /// The value the critic gives a state (its active bits), in `Q16`.
    pub fn value_of(&self, state: &[usize]) -> i32 {
        let syns: Vec<usize> = state.iter().map(|&b| self.critic_syn(b)).collect();
        self.mean(&self.critic, &syns)
    }

    /// An action's preference in a state, in `Q16`.
    pub fn preference(&self, state: &[usize], channel: usize, action: usize) -> i32 {
        let syns: Vec<usize> = state.iter().map(|&b| self.actor_syn(b, channel, action)).collect();
        self.mean(&self.actor, &syns)
    }

    /// Apply one dopamine signal `delta` to every eligible synapse.
    fn learn(&mut self, delta: i32) {
        self.stats.0 += 1;
        self.stats.1 += delta.unsigned_abs() as u64;
        for &(is_critic, i, e) in &self.trace {
            let step = ((delta as i64 * e as i64) >> 16) >> self.shift;
            let w = if is_critic { &mut self.critic[i] } else { &mut self.actor[i] };
            *w = (*w as i64 + step).clamp(-(64 * ONE as i64), 64 * ONE as i64) as i32;
        }
    }

    /// A new state (its active bits, `learn`: update): the dopamine signal for the step that
    /// ends here is the reward gathered + γ·V(this state) − V(the previous state).
    pub fn begin(&mut self, state: &[usize], learn: bool) {
        let v = self.value_of(state);
        if self.started && learn {
            let delta = self.reward + ((self.gamma as i64 * v as i64) >> 16) as i32 - self.value;
            self.learn(delta);
        }
        // decay the traces, then make this state's inputs eligible
        let l = self.lambda as u64;
        let g = self.gamma as u64;
        for t in self.trace.iter_mut() {
            t.2 = ((t.2 as u64 * l * g) >> 32) as u32;
        }
        self.trace.retain(|t| t.2 > ONE / 64);
        // a value is the mean weight of the active inputs, so each eligible synapse moves by the
        // whole step to move the value by it
        for b in state {
            let s = self.critic_syn(*b);
            self.trace.push((true, s, ONE));
        }
        self.state = state.to_vec();
        self.value = v;
        self.reward = 0;
        self.started = true;
    }

    /// (the current state's value, the preference for action 1 over 0 on `channel`), for reports.
    pub fn debug_now(&self, channel: usize) -> (i32, i32) {
        (self.value, self.preference(&self.state, channel, 1) - self.preference(&self.state, channel, 0))
    }

    /// Reward (`Q16`) for the current step.
    pub fn reward(&mut self, r: i32) {
        self.reward += r;
    }

    /// Choose one of `n` actions on `channel` in the current state (action 0 is the default:
    /// it wins ties). With `rng`, explore. The choice becomes eligible.
    pub fn choose<R: Rng>(&mut self, channel: usize, n: usize, rng: Option<&mut R>) -> usize {
        let prefs: Vec<i32> = (0..n).map(|a| self.preference(&self.state, channel, a)).collect();
        let mut best = 0;
        for a in 1..n {
            if prefs[a] > prefs[best] {
                best = a;
            }
        }
        if let Some(rng) = rng {
            if rng.gen_range(0..ONE) < self.explore {
                best = rng.gen_range(0..n);
            }
        }
        let syns: Vec<usize> = self.state.iter().map(|&b| self.actor_syn(b, channel, best)).collect();
        for s in syns {
            self.trace.push((false, s, ONE));
        }
        best
    }

    /// The episode ends: the last step's dopamine is the reward − V(state) (nothing follows),
    /// and the traces are cleared.
    pub fn end(&mut self, learn: bool) {
        if self.started && learn {
            let delta = self.reward - self.value;
            self.learn(delta);
        }
        self.trace.clear();
        self.started = false;
        self.reward = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// A delayed payoff: in state A the agent picks 0 or 1; three neutral steps follow; then a
    /// reward comes only if it picked 1. TD learning must carry the credit back to the choice.
    #[test]
    fn learns_a_delayed_payoff() {
        let mut st = Striatum::new(1 << 14);
        let mut rng = StdRng::seed_from_u64(1);
        let a: Vec<usize> = (0..16).collect();
        let mid = |k: usize| -> Vec<usize> { (100 * (k + 1)..100 * (k + 1) + 16).collect() };
        for _ in 0..2000 {
            st.begin(&a, true);
            let c = st.choose(0, 2, Some(&mut rng));
            for k in 0..3 {
                st.begin(&mid(k), true);
            }
            st.reward(if c == 1 { ONE as i32 } else { 0 });
            st.end(true);
        }
        st.begin(&a, false);
        assert!(st.preference(&a, 0, 1) > st.preference(&a, 0, 0));
        assert_eq!(st.choose::<StdRng>(0, 2, None), 1);
    }
}
