//! The basal ganglia as masks: each action's go (D1) and no-go (D2) pathway is one bit mask over
//! the context, its value the difference of two overlaps, learned by swapping bits at constant
//! density. The same loop as [`Striatum`](super::Striatum), in popcount and masking only.
//!
//! - **Context.** The input groups (one per pathway) are hashed into `d` bits, and each group
//!   gives an equal share of the `k` bits kept: the share of its bits with the smallest hashes
//!   (min-hash, so similar inputs keep similar bits). The context C has a fixed popcount, so
//!   overlaps do not scale with how much input is active.
//! - **Values.** For an action a, `go = popcount(C & Mgo[a] & R1)`, `nogo = popcount(C & Mno[a]
//!   & R2)`, `q = go − nogo`. Go and no-go start as the same random mask, so chance overlap
//!   cancels exactly and q starts at 0; swaps keep both at the same density. Similar contexts
//!   share bits, so an unseen state inherits its neighbours' values.
//! - **Selection.** The default (action 0) is not an action. Among the others, the highest `q`
//!   is released if it reaches the tonic brake `RELEASE`. Conflict (the runner-up within
//!   `MARGIN` of the winner) raises the brake by `HOLD` for that step (the STN's hyperdirect
//!   pause).
//! - **Exploration is a mask.** R1 and R2 (drawn apart, so the noise does not cancel) keep each
//!   bit with probability 1 − τ; τ is a power of two
//!   (`drop_shift`: τ = 2^−drop_shift). τ doubles (at most to ½) when tonic dopamine is low,
//!   that is where the state's value is at or below zero (`low_value`: nothing good expected,
//!   so look further), and with `novelty` when most of the context has not been seen before (a
//!   "seen" mask S, the OR of past contexts).
//! - **Critic.** `V = popcount(C & Mv+) − popcount(C & Mv−)`, each bit worth `UNIT`. The error
//!   is δ = r + ⅞ V(C_t) − V(C_{t−1}); dips are read as far as bursts (D2 affinity).
//! - **Learning is three-factor bit swaps** (k bits, k from |δ|, rounded at random):
//!   - δ > 0: in `Mgo` of the chosen action, set k bits of `C_prev & ~Mgo` and clear k of
//!     `Mgo & ~C_prev`; δ < 0: the same on `Mno`. The critic does the same on `Mv+` or `Mv−`
//!     for the newest state only (TD(0)).
//!   - Eligibility: the states and choices of the last steps, credit fading ⅘ per step; or with
//!     `taps`, the states 1, 2, 4, 8 and 16 steps back (log-spaced, several timescales).
//!   - Actors change only when the cholinergic interneurons pause (|δ| ≥ `pause_min` units).
//!   - With `sticky`: a bit that bursts keep confirming becomes sticky (a habit) and is never
//!     cleared. It passes two tags first, each step with probability 1/8 per confirming burst
//!     (about 24 bursts in all), and loses its tags if it is cleared before. A dip loosens a
//!     contradicted habit: its sticky bits in the context go back to the second tag, each with
//!     probability 1/8.

use rand::rngs::StdRng;
use rand::{Rng, RngCore};

use crate::fixed::ONE;

const UNIT: i32 = (ONE / 8) as i32;
const RATE_DIV: u64 = 4;
const WINDOW: usize = 8;
const TAPS: [usize; 5] = [1, 2, 4, 8, 16];
const HISTORY: usize = 17;
const RELEASE: i32 = 2;
const MARGIN: i32 = 1;
const HOLD: i32 = 2;
const POP: i32 = 32;
const DA_BASE: i32 = 8;

#[derive(Clone)]
struct Mask {
    bits: Vec<u64>,
    sticky: Vec<u64>,
    /// the two tags a bit passes on its way to sticky (synaptic tagging and capture)
    tag1: Vec<u64>,
    tag2: Vec<u64>,
}

impl Mask {
    /// A random mask of density 1/8 (the AND of three random words).
    fn random(words: usize, rng: &mut StdRng) -> Self {
        Self { bits: (0..words).map(|_| rng.next_u64() & rng.next_u64() & rng.next_u64()).collect(), sticky: vec![0; words], tag1: vec![0; words], tag2: vec![0; words] }
    }

    fn overlap(&self, c: &[u64], keep: Option<&[u64]>) -> i32 {
        match keep {
            Some(r) => self.bits.iter().zip(c).zip(r).map(|((m, x), k)| (m & x & k).count_ones()).sum::<u32>() as i32,
            None => self.bits.iter().zip(c).map(|(m, x)| (m & x).count_ones()).sum::<u32>() as i32,
        }
    }

    /// Move toward `c` by `k` bits at constant density: set k bits of c that the mask lacks,
    /// clear k (non-sticky) bits it has outside c. With `stick`, confirmed bits become sticky
    /// with probability 1/8.
    fn swap(&mut self, c: &[u64], k: usize, stick: bool, rng: &mut StdRng) {
        let pick = |cand: Vec<usize>, k: usize, rng: &mut StdRng| -> Vec<usize> {
            let mut cand = cand;
            let k = k.min(cand.len());
            for i in 0..k {
                let j = rng.gen_range(i..cand.len());
                cand.swap(i, j);
            }
            cand.truncate(k);
            cand
        };
        let list = |f: &dyn Fn(usize) -> u64| -> Vec<usize> {
            let mut v = Vec::new();
            for w in 0..c.len() {
                let mut x = f(w);
                while x != 0 {
                    v.push(w * 64 + x.trailing_zeros() as usize);
                    x &= x - 1;
                }
            }
            v
        };
        // constant density: as many bits added as removed
        let del = pick(list(&|w| self.bits[w] & !c[w] & !self.sticky[w]), k, rng);
        let add = pick(list(&|w| c[w] & !self.bits[w]), del.len(), rng);
        for b in add {
            self.bits[b / 64] |= 1 << (b % 64);
        }
        for b in del {
            self.bits[b / 64] &= !(1 << (b % 64));
        }
        self.untag();
        if stick {
            self.consolidate(c, rng);
        }
    }
}

impl Mask {
    /// A burst confirms the mask's bits in `c`: each, with probability 1/8, moves one stage
    /// toward a habit (untagged → first tag → second tag → sticky), so a bit becomes sticky
    /// only after about 24 confirming bursts.
    fn consolidate(&mut self, c: &[u64], rng: &mut StdRng) {
        for w in 0..c.len() {
            let r = self.bits[w] & c[w] & rng.next_u64() & rng.next_u64() & rng.next_u64();
            let (t1, t2) = (self.tag1[w], self.tag2[w]);
            self.sticky[w] |= t2 & r;
            self.tag2[w] |= t1 & r;
            self.tag1[w] |= r;
        }
    }

    /// A dip contradicts the habit: each sticky bit in `c`, with probability 1/8, is no longer
    /// sticky (it keeps its second tag, so it can be relearned faster).
    fn loosen(&mut self, c: &[u64], rng: &mut StdRng) {
        for w in 0..c.len() {
            let r = self.sticky[w] & c[w] & rng.next_u64() & rng.next_u64() & rng.next_u64();
            self.sticky[w] &= !r;
            self.tag2[w] |= r;
        }
    }

    /// Move away from `c` by `k` bits at constant density: clear `k` (non-sticky) bits the mask
    /// has inside `c`, set as many it lacks outside `c`.
    fn away(&mut self, c: &[u64], k: usize, rng: &mut StdRng) {
        let mut inside = Vec::new();
        let mut outside = Vec::new();
        for w in 0..c.len() {
            let (mut x, mut y) = (self.bits[w] & c[w] & !self.sticky[w], !self.bits[w] & !c[w]);
            while x != 0 {
                inside.push(w * 64 + x.trailing_zeros() as usize);
                x &= x - 1;
            }
            // outside the context, a random sample is enough (most bits are free)
            y &= rng.next_u64() & rng.next_u64() & rng.next_u64();
            while y != 0 {
                outside.push(w * 64 + y.trailing_zeros() as usize);
                y &= y - 1;
            }
        }
        let k = k.min(inside.len()).min(outside.len());
        for i in 0..k {
            let j = rng.gen_range(i..inside.len());
            inside.swap(i, j);
            self.bits[inside[i] / 64] &= !(1 << (inside[i] % 64));
            let j = rng.gen_range(i..outside.len());
            outside.swap(i, j);
            self.bits[outside[i] / 64] |= 1 << (outside[i] % 64);
        }
        self.untag();
    }

    /// Forget the tags of bits no longer in the mask.
    fn untag(&mut self) {
        for w in 0..self.bits.len() {
            self.tag1[w] &= self.bits[w];
            self.tag2[w] &= self.bits[w];
        }
    }

    /// Clear `k` (non-sticky) bits of the mask inside `c` (long-term depression of the synapses
    /// the context drove). Returns how many were cleared.
    fn pull(&mut self, c: &[u64], k: usize, rng: &mut StdRng) -> usize {
        let mut v = Vec::new();
        for w in 0..c.len() {
            let mut x = self.bits[w] & c[w] & !self.sticky[w];
            while x != 0 {
                v.push(w * 64 + x.trailing_zeros() as usize);
                x &= x - 1;
            }
        }
        let k = k.min(v.len());
        for i in 0..k {
            let j = rng.gen_range(i..v.len());
            v.swap(i, j);
            self.bits[v[i] / 64] &= !(1 << (v[i] % 64));
        }
        self.untag();
        k
    }

    /// Set `k` bits of `c` the mask lacks (potentiation). Returns how many were set.
    fn push(&mut self, c: &[u64], k: usize, stick: bool, rng: &mut StdRng) -> usize {
        let mut v = Vec::new();
        for w in 0..c.len() {
            let mut x = c[w] & !self.bits[w];
            while x != 0 {
                v.push(w * 64 + x.trailing_zeros() as usize);
                x &= x - 1;
            }
        }
        let k = k.min(v.len());
        for i in 0..k {
            let j = rng.gen_range(i..v.len());
            v.swap(i, j);
            self.bits[v[i] / 64] |= 1 << (v[i] % 64);
        }
        if stick {
            self.consolidate(c, rng);
        }
        k
    }
}

/// Bits in a mask-cell, how many must be in the context for it to fire, and a pool's size.
const CELL_BITS: usize = 16;
const CELL_THETA: u32 = 10;
const POOL_CAP: usize = 2048;
/// Grow mode: GPi's tonic noise per option when exploring.
const JITTER: i32 = 3;

/// A pool of mask-cells (`grow`): each a sparse mask over the context that fires when at least
/// `CELL_THETA` of its `CELL_BITS` bits are in it (a striatal cell with a large receptive
/// field), with when it last fired.
#[derive(Clone, Default)]
struct CellPool {
    cells: Vec<(Mask, u64)>,
}

impl CellPool {
    fn firing(&self, x: &[u64], keep: Option<&[u64]>) -> Vec<usize> {
        (0..self.cells.len()).filter(|&i| self.cells[i].0.overlap(x, keep) as u32 >= CELL_THETA).collect()
    }

    fn count(&self, x: &[u64], keep: Option<&[u64]>) -> i32 {
        self.cells.iter().filter(|c| c.0.overlap(x, keep) as u32 >= CELL_THETA).count() as i32
    }

    /// A new cell for context `x`: `CELL_BITS` of its bits. A full pool first drops, of a few
    /// sampled, the cell unused longest.
    fn recruit(&mut self, x: &[u64], now: u64, rng: &mut StdRng) {
        let mut on: Vec<usize> = Vec::new();
        for (w, &v) in x.iter().enumerate() {
            let mut v = v;
            while v != 0 {
                on.push(w * 64 + v.trailing_zeros() as usize);
                v &= v - 1;
            }
        }
        if on.len() < CELL_BITS {
            return;
        }
        if self.cells.len() >= POOL_CAP {
            let i = (0..8).map(|_| rng.gen_range(0..self.cells.len())).min_by_key(|&i| self.cells[i].1).unwrap();
            self.cells.swap_remove(i);
        }
        let words = x.len();
        let mut m = Mask { bits: vec![0; words], sticky: vec![0; words], tag1: vec![0; words], tag2: vec![0; words] };
        for i in 0..CELL_BITS {
            let j = rng.gen_range(i..on.len());
            on.swap(i, j);
            m.bits[on[i] / 64] |= 1 << (on[i] % 64);
        }
        self.cells.push((m, now));
    }
}

#[derive(Clone)]
struct Step {
    c: Vec<u64>,
    /// (channel, action, the context the choice saw: C, or C with the candidate)
    choices: Vec<(usize, usize, Vec<u64>)>,
    critic: bool,
}

pub struct MaskStriatum {
    d: usize,
    k: usize,
    critic: (Mask, Mask),
    actors: Vec<Vec<(Mask, Mask)>>,
    c: Vec<u64>,
    /// the current state's input groups (candidates are scored on them with their own)
    state: Vec<Vec<usize>>,
    seen: Vec<u64>,
    recent: Vec<Step>,
    maintained: Vec<Option<Step>>,
    value: i32,
    reward: i32,
    started: bool,
    rng: StdRng,
    /// τ = 2^−drop_shift
    pub drop_shift: u32,
    /// τ doubles when most of the context is new
    pub novelty: bool,
    /// the STN's hold on close competitors
    pub hold: bool,
    /// the tonic brake: q must reach this to release
    pub release: i32,
    /// τ doubles where the state's value is at or below zero (tonic dopamine low)
    pub low_value: bool,
    /// eligibility at log-spaced delays instead of a fading window
    pub taps: bool,
    /// habit bits
    pub sticky: bool,
    /// half the context bits are conjunctions of two pathways' inputs
    pub conj: bool,
    /// push-pull learning: a burst potentiates the go mask and depresses the no-go mask on the
    /// context's bits (a dip the reverse), so nothing outside the context changes and the pair
    /// keeps its total density; else the swap rule (each mask at constant density, bits outside
    /// the context cleared)
    pub push_pull: bool,
    /// with the swap rule, a dip also moves the go mask away from the context (go LTD)
    pub go_ltd: bool,
    /// grow: each pathway is a pool of mask-cells recruited as needed, not one mask
    pub grow: bool,
    /// grow mode: extra GPi noise (±this on top of ±`JITTER`) for the next choices, e.g.
    /// curiosity: where predictions are improving, try other things a little more
    pub extra_noise: i32,
    pools: Vec<Vec<(CellPool, CellPool)>>,
    vpool: (CellPool, CellPool),
    clock: u64,
    pub gamma_div: i32,
    pub pause_min: i32,
    /// (dopamine events, sum of |δ|)
    pub stats: (u64, u64),
    pub last_delta: i32,
    pub swaps: u64,
    /// cholinergic pauses (actor plasticity)
    pub pauses: u64,
}

fn mix(x: u64) -> u64 {
    let mut y = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    y ^= y >> 29;
    y = y.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    y ^ (y >> 31)
}

impl MaskStriatum {
    /// A context of `k` bits among `d` (a multiple of 64).
    pub fn new(d: usize, k: usize, seed: u64) -> Self {
        let mut rng = <StdRng as rand::SeedableRng>::seed_from_u64(seed);
        let words = d / 64;
        let m = Mask::random(words, &mut rng);
        let critic = (m.clone(), m);
        Self {
            d,
            k,
            critic,
            actors: Vec::new(),
            c: vec![0; words],
            state: Vec::new(),
            seen: vec![0; words],
            recent: Vec::new(),
            maintained: Vec::new(),
            value: 0,
            reward: 0,
            started: false,
            rng,
            drop_shift: 2,
            novelty: true,
            hold: true,
            release: RELEASE,
            low_value: true,
            taps: false,
            sticky: false,
            conj: false,
            push_pull: true,
            go_ltd: true,
            grow: false,
            extra_noise: 0,
            pools: Vec::new(),
            vpool: (CellPool::default(), CellPool::default()),
            clock: 0,
            gamma_div: 8,
            pause_min: 2,
            stats: (0, 0),
            last_delta: 0,
            swaps: 0,
            pauses: 0,
        }
    }

    /// The fixed-popcount context of some input groups: each group gives an equal share of the
    /// `k` bits, its bits with the smallest hashes.
    /// The context a candidate is scored on: half the bits are the state's own (its single
    /// inputs, an equal share per pathway), half are pairs of one state input and one candidate
    /// input. The candidate never counts alone, so "candidates like this" are valued only "in
    /// states like that" (as a spiny cell with synapses from both).
    fn candidate_context(&self, candidate: &[Vec<usize>]) -> Vec<u64> {
        let mut c = self.context_k(&self.state, self.k / 2);
        let picks = |groups: &[Vec<usize>], n: usize| -> Vec<u64> {
            let live: Vec<&Vec<usize>> = groups.iter().filter(|g| !g.is_empty()).collect();
            let mut out = Vec::new();
            if live.is_empty() {
                return out;
            }
            let share = (n / live.len()).max(1);
            for (gi, g) in live.into_iter().enumerate() {
                let salt = 0x5EED ^ (gi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let mut h: Vec<u64> = g.iter().map(|&b| mix(b as u64 ^ salt)).collect();
                h.sort_unstable();
                out.extend(h.into_iter().take(share));
            }
            out
        };
        let (a, b) = (picks(&self.state, 16), picks(candidate, 8));
        if a.is_empty() || b.is_empty() {
            return c;
        }
        for i in 0..self.k - self.k / 2 {
            let x = mix(a[i % a.len()] ^ b[(i / a.len() + i) % b.len()].rotate_left(29) ^ 0xBEEF);
            let bit = (x % self.d as u64) as usize;
            c[bit / 64] |= 1 << (bit % 64);
        }
        c
    }

    /// The fixed-popcount context of some input groups with `k` bits (singles only).
    fn context_k(&self, groups: &[Vec<usize>], k: usize) -> Vec<u64> {
        let live: Vec<&Vec<usize>> = groups.iter().filter(|g| !g.is_empty()).collect();
        let mut c = vec![0u64; self.d / 64];
        if live.is_empty() {
            return c;
        }
        let share = (k / live.len()).max(1);
        for (gi, g) in live.into_iter().enumerate() {
            // each pathway its own permutation, so their picks are not correlated
            let salt = 0x5EED ^ (gi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut h: Vec<u64> = g.iter().map(|&b| mix(b as u64 ^ salt)).collect();
            h.sort_unstable();
            for &x in h.iter().take(share) {
                let b = (x % self.d as u64) as usize;
                c[b / 64] |= 1 << (b % 64);
            }
        }
        c
    }

    fn context(&self, groups: &[Vec<usize>]) -> Vec<u64> {
        let live: Vec<&Vec<usize>> = groups.iter().filter(|g| !g.is_empty()).collect();
        let mut c = vec![0u64; self.d / 64];
        if live.is_empty() {
            return c;
        }
        // with conjunctions, half the bits are single inputs and half are pairs of inputs from
        // two pathways (as granule-like cells mixing them), so a mask can learn "this with that"
        let singles = if self.conj { self.k / 2 } else { self.k };
        let share = (singles / live.len()).max(1);
        let mut picks: Vec<Vec<u64>> = Vec::new();
        for (gi, g) in live.iter().enumerate() {
            let salt = 0x5EED ^ (gi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut h: Vec<u64> = g.iter().map(|&b| mix(b as u64 ^ salt)).collect();
            h.sort_unstable();
            for &x in h.iter().take(share) {
                let b = (x % self.d as u64) as usize;
                c[b / 64] |= 1 << (b % 64);
            }
            h.truncate(share.max(4));
            picks.push(h);
        }
        if self.conj && picks.len() > 1 {
            let pairs = self.k - singles;
            let n = picks.len();
            for i in 0..pairs {
                let (ga, gb) = (i % n, (i / n + 1 + i % n) % n);
                if ga == gb {
                    continue;
                }
                let a = picks[ga][(i / n) % picks[ga].len()];
                let b = picks[gb][(i / (n * n) + i) % picks[gb].len()];
                let x = mix(a ^ b.rotate_left(17) ^ 0xC0FFEE);
                let bit = (x % self.d as u64) as usize;
                c[bit / 64] |= 1 << (bit % 64);
            }
        }
        c
    }

    fn ensure(&mut self, channel: usize, n: usize) {
        while self.pools.len() <= channel {
            self.pools.push(Vec::new());
        }
        while self.pools[channel].len() < n {
            self.pools[channel].push((CellPool::default(), CellPool::default()));
        }
        let words = self.d / 64;
        while self.actors.len() <= channel {
            self.actors.push(Vec::new());
        }
        while self.actors[channel].len() < n {
            // go and no-go start as the same mask: q is exactly 0 until something is learned
            let m = Mask::random(words, &mut self.rng);
            self.actors[channel].push((m.clone(), m));
        }
    }

    fn v(&self, c: &[u64]) -> i32 {
        if self.grow {
            return (self.vpool.0.count(c, None) - self.vpool.1.count(c, None)) * UNIT;
        }
        (self.critic.0.overlap(c, None) - self.critic.1.overlap(c, None)) * UNIT
    }

    /// An action's q on context `x`: go minus no-go (cells firing, or mask overlaps).
    fn q(&self, channel: usize, a: usize, x: &[u64], keep: &Option<(Vec<u64>, Vec<u64>)>) -> i32 {
        let (k0, k1) = (keep.as_ref().map(|k| k.0.as_slice()), keep.as_ref().map(|k| k.1.as_slice()));
        if self.grow {
            let (g, o) = &self.pools[channel][a];
            return g.count(x, k0) - o.count(x, k1);
        }
        let (g, o) = &self.actors[channel][a];
        g.overlap(x, k0) - o.overlap(x, k1)
    }

    /// Grow mode, exploring: GPi's tonic firing is noisy (±`JITTER` per option), so options
    /// with no cells yet are tried (dropout only removes what cells there are).
    fn jitter(&mut self, q: &mut [i32], explore: bool) {
        if self.grow && explore {
            let j = JITTER + self.extra_noise.max(0);
            for x in q.iter_mut() {
                *x += self.rng.gen_range(-j..=j);
            }
        }
    }

    /// Grow mode: a burst (`up`) or dip in pools `(more, fewer)` for context `x`, `n` changes:
    /// cells of `more` that fired tighten toward `x` (and with `stick` consolidate) and `n` new
    /// ones are recruited (the error says the state is worth `n` cells more or less); up to `n`
    /// cells of `fewer` that fired move away from `x` (with `loosen`, their habits first loosen).
    #[allow(clippy::too_many_arguments)]
    fn grow_learn(more: &mut CellPool, fewer: &mut CellPool, x: &[u64], n: usize, stick: bool, loosen: bool, now: u64, rng: &mut StdRng) -> u64 {
        let mut changed = 0u64;
        let fired = more.firing(x, None);
        for &i in &fired {
            more.cells[i].0.swap(x, 1, stick, rng);
            more.cells[i].1 = now;
            changed += 1;
        }
        for _ in 0..n {
            more.recruit(x, now, rng);
            changed += 1;
        }
        let mut against = fewer.firing(x, None);
        for _ in 0..n.min(against.len()) {
            let j = rng.gen_range(0..against.len());
            let i = against.swap_remove(j);
            if loosen {
                fewer.cells[i].0.loosen(x, rng);
            }
            fewer.cells[i].0.away(x, 2, rng);
            changed += 1;
        }
        changed
    }

    /// The dropout keep-masks R, one for each pathway (independent noise in D1 and D2; None
    /// without exploration).
    fn keep(&mut self, explore: bool) -> Option<(Vec<u64>, Vec<u64>)> {
        // grow mode explores with GPi's tonic noise: dropout would silence threshold cells
        if !explore || self.grow {
            return None;
        }
        let mut shift = self.drop_shift.max(1);
        // tonic dopamine: where nothing good is expected (the state's value at or below zero),
        // exploration doubles
        if self.low_value && self.value <= 0 {
            shift = shift.saturating_sub(1).max(1);
        }
        if self.novelty {
            let new: u32 = self.c.iter().zip(&self.seen).map(|(x, s)| (x & !s).count_ones()).sum();
            let all: u32 = self.c.iter().map(|x| x.count_ones()).sum();
            if 2 * new > all {
                shift = shift.saturating_sub(1).max(1);
            }
        }
        let mut one = || -> Vec<u64> {
            (0..self.d / 64)
                .map(|_| {
                    let mut drop = u64::MAX;
                    for _ in 0..shift {
                        drop &= self.rng.next_u64();
                    }
                    !drop
                })
                .collect()
        };
        Some((one(), one()))
    }

    /// Release among `q` (index 0 = the first candidate): the highest, if it reaches the brake
    /// (raised when the runner-up is close).
    fn release(&self, q: &[i32]) -> Option<usize> {
        let (best, &top) = q.iter().enumerate().max_by_key(|x| (*x.1, std::cmp::Reverse(x.0)))?;
        let second = q.iter().enumerate().filter(|x| x.0 != best).map(|x| *x.1).max();
        let brake = self.release + if self.hold && second.map_or(false, |s| s + MARGIN >= top) { HOLD } else { 0 };
        (top >= brake).then_some(best)
    }

    fn learn(&mut self, delta: i32) {
        self.stats.0 += 1;
        self.stats.1 += delta.unsigned_abs() as u64;
        self.last_delta = delta;
        if delta == 0 {
            return;
        }
        let pause = delta.abs() >= self.pause_min * UNIT;
        self.pauses += pause as u64;
        let n = self.recent.len();
        // (index into recent, credit weight in Q16)
        let elig: Vec<(usize, u64)> = if self.taps {
            TAPS.iter().zip([ONE as u64, ONE as u64 * 3 / 4, ONE as u64 / 2, ONE as u64 * 3 / 8, ONE as u64 / 4]).filter(|(d, _)| **d <= n).map(|(d, w)| (n - d, w)).collect()
        } else {
            let mut w = ONE as u64;
            (0..n.min(WINDOW))
                .map(|i| {
                    let x = (n - 1 - i, w);
                    w = w * 4 / 5;
                    x
                })
                .collect()
        };
        for (k, weight) in elig {
            let amount = delta.unsigned_abs() as u64 * weight / UNIT as u64 / RATE_DIV;
            let mut bits = (amount >> 16) as usize;
            if self.rng.gen_range(0..ONE as u64) < (amount & 0xFFFF) {
                bits += 1;
            }
            if bits == 0 {
                continue;
            }
            let st = self.recent[k].clone();
            if st.critic && k + 1 == n && self.grow {
                let (more, fewer) = if delta > 0 { (&mut self.vpool.0, &mut self.vpool.1) } else { (&mut self.vpool.1, &mut self.vpool.0) };
                self.swaps += Self::grow_learn(more, fewer, &st.c, bits, false, false, self.clock, &mut self.rng);
            } else if st.critic && k + 1 == n {
                let (up, down) = if delta > 0 { (&mut self.critic.0, &mut self.critic.1) } else { (&mut self.critic.1, &mut self.critic.0) };
                if self.push_pull {
                    let a = up.push(&st.c, bits, false, &mut self.rng);
                    let b = down.pull(&st.c, bits, &mut self.rng);
                    self.swaps += (a + b) as u64;
                } else {
                    up.swap(&st.c, bits, false, &mut self.rng);
                    self.swaps += bits as u64;
                }
            }
            if !pause {
                continue;
            }
            for (ch, a, x) in &st.choices {
                let stick = self.sticky && delta > 0;
                if self.grow {
                    let pair = &mut self.pools[*ch][*a];
                    let (more, fewer) = if delta > 0 { (&mut pair.0, &mut pair.1) } else { (&mut pair.1, &mut pair.0) };
                    // a dip moves the go cells that fired away (go LTD) and loosens their habits
                    self.swaps += Self::grow_learn(more, fewer, x, bits, stick, delta < 0 && self.sticky, self.clock, &mut self.rng);
                    continue;
                }
                let pair = &mut self.actors[*ch][*a];
                // a dip loosens the habit it contradicts: sticky go bits in this context go back
                // to their second tag now and then
                if delta < 0 && self.sticky {
                    pair.0.loosen(x, &mut self.rng);
                }
                let (up, down) = if delta > 0 { (&mut pair.0, &mut pair.1) } else { (&mut pair.1, &mut pair.0) };
                if self.push_pull {
                    let a = up.push(x, bits, stick, &mut self.rng);
                    let b = down.pull(x, bits, &mut self.rng);
                    self.swaps += (a + b) as u64;
                } else {
                    up.swap(x, bits, stick, &mut self.rng);
                    self.swaps += bits as u64;
                    // the other side of the three-factor rule: a dip also depresses the go
                    // pathway (LTD), moving the go mask away from the context at constant density
                    if delta < 0 && self.go_ltd {
                        pair.0.away(x, bits, &mut self.rng);
                        self.swaps += bits as u64;
                    }
                }
            }
        }
    }

    fn dopamine(&self, drive: i32) -> i32 {
        if drive < 0 {
            return (drive / UNIT).max(DA_BASE - POP) * UNIT;
        }
        ((DA_BASE + drive / UNIT).min(POP) - DA_BASE) * UNIT
    }

    pub fn begin_groups(&mut self, state: &[Vec<usize>], learn: bool) {
        self.clock += 1;
        self.c = self.context(state);
        self.state = state.to_vec();
        let v = self.v(&self.c);
        if self.started && learn {
            let disc = if self.gamma_div > 0 { v / self.gamma_div } else { 0 };
            let delta = self.dopamine(self.reward + v - disc - self.value);
            self.learn(delta);
        }
        self.recent.push(Step { c: self.c.clone(), choices: Vec::new(), critic: true });
        let cap = if self.taps { HISTORY } else { WINDOW };
        if self.recent.len() > cap {
            self.recent.remove(0);
        }
        for (s, x) in self.seen.iter_mut().zip(&self.c) {
            *s |= x;
        }
        self.value = v;
        self.reward = 0;
        self.started = true;
    }

    pub fn reward(&mut self, r: i32) {
        self.reward += r;
    }

    /// One of `n` actions on `channel` (0: the default, nothing released).
    pub fn choose(&mut self, channel: usize, n: usize, explore: bool) -> usize {
        self.ensure(channel, n);
        let keep = self.keep(explore);
        let c = self.c.clone();
        let mut q: Vec<i32> = (1..n).map(|a| self.q(channel, a, &c, &keep)).collect();
        self.jitter(&mut q, explore);
        match self.release(&q) {
            Some(i) => {
                let c = self.c.clone();
                if let Some(st) = self.recent.last_mut() {
                    st.choices.push((channel, i + 1, c));
                }
                i + 1
            }
            None => 0,
        }
    }

    /// The cortex proposes candidates (each its own input groups), the loop picks one or none:
    /// each is scored by `channel`'s one mask pair on the state's bits and pairs of state and
    /// candidate bits (`candidate_context`).
    pub fn choose_among(&mut self, channel: usize, candidates: &[Vec<Vec<usize>>], explore: bool) -> Option<usize> {
        if candidates.is_empty() {
            return None;
        }
        self.ensure(channel, 2);
        let keep = self.keep(explore);
        let xs: Vec<Vec<u64>> = candidates.iter().map(|c| self.candidate_context(c)).collect();
        let mut q: Vec<i32> = xs.iter().map(|x| self.q(channel, 1, x, &keep)).collect();
        self.jitter(&mut q, explore);
        let i = self.release(&q)?;
        if let Some(st) = self.recent.last_mut() {
            st.choices.push((channel, 1, xs[i].clone()));
        }
        Some(i)
    }

    /// A candidate done without the loop choosing it (the cortex initiated it).
    pub fn did_among(&mut self, channel: usize, candidate: &[Vec<usize>]) {
        self.ensure(channel, 2);
        let x = self.candidate_context(candidate);
        if let Some(st) = self.recent.last_mut() {
            st.choices.push((channel, 1, x));
        }
    }

    pub fn maintain(&mut self, channel: usize) {
        while self.maintained.len() <= channel {
            self.maintained.push(None);
        }
        self.maintained[channel] = self.recent.last().and_then(|st| {
            let c = st.choices.iter().rev().find(|c| c.0 == channel)?.clone();
            Some(Step { c: st.c.clone(), choices: vec![c], critic: false })
        });
    }

    pub fn retag(&mut self, channel: usize) {
        if let Some(Some(st)) = self.maintained.get(channel) {
            let k = self.recent.len().saturating_sub(1);
            self.recent.insert(k, st.clone());
        }
    }

    pub fn end(&mut self, learn: bool) {
        if self.started && learn {
            let delta = self.dopamine(self.reward - self.value);
            self.learn(delta);
        }
        self.recent.clear();
        self.started = false;
        self.reward = 0;
    }

    /// The current state's value (`Q16`).
    pub fn value(&self) -> i32 {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A delayed payoff: the choice in state A pays three steps later.
    #[test]
    fn learns_a_delayed_payoff() {
        let mut st = MaskStriatum::new(2048, 64, 1);
        let a: Vec<Vec<usize>> = vec![(0..64).collect()];
        let mid = |k: usize| -> Vec<Vec<usize>> { vec![(1000 * (k + 1)..1000 * (k + 1) + 64).collect()] };
        let mut rng = <StdRng as rand::SeedableRng>::seed_from_u64(9);
        for _ in 0..600 {
            st.begin_groups(&a, true);
            let c = st.choose(0, 3, true);
            for k in 0..3 {
                st.begin_groups(&mid(k), true);
            }
            st.reward(if c == 2 { ONE as i32 } else { 0 });
            let _ = rng.next_u32();
            st.end(true);
        }
        st.begin_groups(&a, false);
        assert_eq!(st.choose(0, 3, false), 2);
    }

    /// Two contexts, opposite best actions.
    #[test]
    fn learns_by_context() {
        let mut st = MaskStriatum::new(2048, 64, 2);
        let ctx = |k: usize| -> Vec<Vec<usize>> { vec![(1000 * k..1000 * k + 64).collect(), (5000..5064).collect()] };
        for i in 0..800 {
            let k = i % 2;
            st.begin_groups(&ctx(k), true);
            let c = st.choose(0, 3, true);
            st.reward(if c == k + 1 { ONE as i32 } else { 0 });
            st.end(true);
        }
        for k in 0..2 {
            st.begin_groups(&ctx(k), false);
            assert_eq!(st.choose(0, 3, false), k + 1, "context {k}");
            st.end(false);
        }
    }

    /// Grow mode: the same two tasks, learned by recruiting mask-cells.
    #[test]
    fn grows_cells_for_a_delayed_payoff_and_by_context() {
        let mut st = MaskStriatum::new(2048, 64, 3);
        st.grow = true;
        let a: Vec<Vec<usize>> = vec![(0..64).collect()];
        let mid = |k: usize| -> Vec<Vec<usize>> { vec![(1000 * (k + 1)..1000 * (k + 1) + 64).collect()] };
        for _ in 0..600 {
            st.begin_groups(&a, true);
            let c = st.choose(0, 3, true);
            for k in 0..3 {
                st.begin_groups(&mid(k), true);
            }
            st.reward(if c == 2 { ONE as i32 } else { 0 });
            st.end(true);
        }
        st.begin_groups(&a, false);
        assert_eq!(st.choose(0, 3, false), 2);
        st.end(false);
        let mut st = MaskStriatum::new(2048, 64, 4);
        st.grow = true;
        let ctx = |k: usize| -> Vec<Vec<usize>> { vec![(1000 * k..1000 * k + 64).collect(), (5000..5064).collect()] };
        for i in 0..1600 {
            let k = i % 2;
            st.begin_groups(&ctx(k), true);
            let c = st.choose(0, 3, true);
            st.reward(if c == k + 1 { ONE as i32 } else { 0 });
            st.end(true);
        }
        for k in 0..2 {
            st.begin_groups(&ctx(k), false);
            assert_eq!(st.choose(0, 3, false), k + 1, "context {k}");
            st.end(false);
        }
    }
}


