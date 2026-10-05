//! Thalamus: gating which pathway's content reaches cortex.
//!
//! Biologically the thalamus relays between cortical areas (higher-order nuclei such as
//! pulvinar and mediodorsal), and the inhibitory reticular nucleus and the basal
//! ganglia decide which channel gets through. Here the candidate pathways are the
//! cortical match rules of `cortex` (`RelayChannel` over a `ContextBuffer`), and the
//! gates below learn which to open: `RouteGate` by per-context precision tables,
//! `KernelGate` by bit-pattern matching. The basal-ganglia-driven gate, with memory
//! recall as one more channel, is `Policy::ThalamicGate` / `LearnedGate` in the
//! episodic example.

use std::collections::HashMap;

use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::bitvec::BitVector;
use crate::kernel::{GrowthConfig, KernelClass, SimpleKernel};

pub use super::cortex::{ContextBuffer, RelayChannel};
use super::cortex::overlap;

/// The context buffer was called `Thalamus` before the cortex/thalamus split.
pub type Thalamus = ContextBuffer;

/// Attention by inhibition: every route in a pool computes its relay, and a
/// gate passes only routes that have proven reliable *in the current context*.
///
/// Reliability is per-context precision: of the times a route relayed something
/// in this context, how often was it the next frame (`learn`). The context is the
/// current frame, optionally together with what the route relays
/// (`condition_on_value`): "this route is trustworthy here when it brings back
/// something like *this*". Routes are inhibited by default: a route opens only
/// with at least `min_tries` observations and precision >= `threshold`. Among open
/// routes, lateral inhibition keeps the `winners` most reliable (winner-take-k,
/// a hard stand-in for softmax). The surviving relays are OR'ed into one frame.
pub struct RouteGate {
    pub threshold: f64,
    pub min_tries: f64,
    pub winners: usize,
    pub condition_on_value: bool,
    stats: HashMap<(RelayChannel, u64, u64), (f64, f64)>, // (route, context, value) -> (hits, tries)
}

impl RouteGate {
    pub fn new(threshold: f64, min_tries: f64, winners: usize, condition_on_value: bool) -> Self {
        Self { threshold, min_tries, winners, condition_on_value, stats: HashMap::new() }
    }

    fn key(&self, r: RelayChannel, ctx: &BitVector, value: &BitVector) -> (RelayChannel, u64, u64) {
        (r, frame_hash(ctx), if self.condition_on_value { frame_hash(value) } else { 0 })
    }

    /// Smoothed precision of route `r` relaying `value` in context `ctx`, and its tries.
    fn reliability(&self, r: RelayChannel, ctx: &BitVector, value: &BitVector) -> (f64, f64) {
        self.stats.get(&self.key(r, ctx, value)).map_or((0.0, 0.0), |&(h, t)| ((h + 0.5) / (t + 1.0), t))
    }

    /// Routes from `pool` that pass the gate now, most reliable first, with their values.
    pub fn select<'a>(&self, th: &'a Thalamus, pool: &[RelayChannel]) -> Vec<(RelayChannel, &'a BitVector)> {
        let Some(ctx) = th.current() else { return Vec::new() };
        let mut open: Vec<(f64, RelayChannel, &BitVector)> = pool
            .iter()
            .filter_map(|&r| th.relay_channel(r).map(|v| (r, v)))
            .filter_map(|(r, v)| {
                let (p, tries) = self.reliability(r, ctx, v);
                (tries >= self.min_tries && p >= self.threshold).then_some((p, r, v))
            })
            .collect();
        open.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        open.into_iter().take(self.winners).map(|(_, r, v)| (r, v)).collect()
    }

    /// The gated relay: the passing routes' values OR'ed into one frame of `bits`.
    pub fn gated_relay(&self, th: &Thalamus, pool: &[RelayChannel], bits: usize) -> BitVector {
        let mut out = BitVector::new(bits, Some(0));
        for (_, v) in self.select(th, pool) {
            for (o, &w) in out.as_words_mut().iter_mut().zip(v.as_words()) {
                *o |= w;
            }
        }
        out
    }

    /// Learn from what came next (call before observing `next`): every pool route
    /// that relays something now is scored right or wrong in the current context.
    pub fn learn(&mut self, th: &Thalamus, pool: &[RelayChannel], next: &BitVector) {
        let Some(ctx) = th.current() else { return };
        let need = (next.count_ones() as f32 * th.match_fraction).ceil() as u32;
        for &r in pool {
            if let Some(v) = th.relay_channel(r) {
                let right = overlap(v, next) >= need;
                let key = self.key(r, ctx, v);
                let e = self.stats.entry(key).or_insert((0.0, 0.0));
                e.1 += 1.0;
                if right {
                    e.0 += 1.0;
                }
            }
        }
    }
}

/// `RouteGate` built from the network's own machinery: a predictive
/// `KernelClass` learns, by bit-pattern matching, whether a route is right.
///
/// Each candidate is presented as three frames `[route code | context | value]`
/// (the route's own sparse code, the current frame, what the route relays) with
/// target RIGHT or WRONG (two sparse codes). Because predictive kernels grow
/// one frame deeper only when a shallower one is wrong, the class backs off
/// from "this route is usually right" (route frame only) to "right in this
/// context" to "right in this context when it brings back this" without any of
/// those levels being chosen by hand, and contexts match by bit overlap rather
/// than exact identity. A route opens when the winning kernel says RIGHT with
/// reliability >= `threshold`; the `winners` most reliable open routes pass.
pub struct KernelGate {
    pub threshold: f32,
    pub winners: usize,
    bits: usize,
    class: KernelClass<SimpleKernel>,
    route_codes: HashMap<RelayChannel, BitVector>,
    right: BitVector,
    wrong: BitVector,
    rng: rand::rngs::StdRng,
}

impl KernelGate {
    pub fn new(bits: usize, threshold: f32, winners: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let positions: Vec<usize> = (0..bits).collect();
        let mut code = |rng: &mut rand::rngs::StdRng| {
            BitVector::from_bits(&positions.choose_multiple(rng, 32).cloned().collect::<Vec<_>>(), bits)
        };
        let right = code(&mut rng);
        let wrong = code(&mut rng);
        let class = KernelClass::predictive(GrowthConfig {
            max_kernels: 200_000,
            frame_words: bits.div_ceil(64),
            max_frames: 3,
            sample_bits: 16,
            match_fraction: 0.8,
            surprise_fraction: 0.5,
            generalize: None,
            generalize_after: 1,
        });
        Self { threshold, winners, bits, class, route_codes: HashMap::new(), right, wrong, rng }
    }

    fn input(&mut self, r: RelayChannel, ctx: &BitVector, value: &BitVector) -> BitVector {
        let bits = self.bits;
        let positions: Vec<usize> = (0..bits).collect();
        let rng = &mut self.rng;
        let code = self
            .route_codes
            .entry(r)
            .or_insert_with(|| BitVector::from_bits(&positions.choose_multiple(rng, 32).cloned().collect::<Vec<_>>(), bits));
        let mut words = code.as_words().to_vec();
        words.extend_from_slice(ctx.as_words());
        words.extend_from_slice(value.as_words());
        BitVector::from_words(words)
    }

    fn says_right(&self, out: &BitVector) -> bool {
        overlap(out, &self.right) > overlap(out, &self.wrong)
    }

    /// Routes from `pool` that pass the gate now, most reliable first, with their values.
    pub fn select<'a>(&mut self, th: &'a Thalamus, pool: &[RelayChannel]) -> Vec<(RelayChannel, &'a BitVector)> {
        let Some(ctx) = th.current() else { return Vec::new() };
        let ctx = ctx.clone();
        let mut open: Vec<(f32, RelayChannel, &BitVector)> = Vec::new();
        for &r in pool {
            if let Some(v) = th.relay_channel(r) {
                let input = self.input(r, &ctx, v);
                if let Some((out, rel)) = self.class.peek_scored(&input) {
                    if rel >= self.threshold && self.says_right(out) {
                        open.push((rel, r, v));
                    }
                }
            }
        }
        open.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        open.into_iter().take(self.winners).map(|(_, r, v)| (r, v)).collect()
    }

    /// The gated relay: the passing routes' values OR'ed into one frame of `bits`.
    pub fn gated_relay(&mut self, th: &Thalamus, pool: &[RelayChannel], bits: usize) -> BitVector {
        let mut out = BitVector::new(bits, Some(0));
        for (_, v) in self.select(th, pool) {
            for (o, &w) in out.as_words_mut().iter_mut().zip(v.as_words()) {
                *o |= w;
            }
        }
        out
    }

    /// Learn from what came next (call before observing `next`): every pool route
    /// that relays something is a training example, target RIGHT or WRONG.
    pub fn learn(&mut self, th: &Thalamus, pool: &[RelayChannel], next: &BitVector) {
        let Some(ctx) = th.current() else { return };
        let ctx = ctx.clone();
        let need = (next.count_ones() as f32 * th.match_fraction).ceil() as u32;
        for &r in pool {
            if let Some(v) = th.relay_channel(r) {
                let target = if overlap(v, next) >= need { self.right.clone() } else { self.wrong.clone() };
                let input = self.input(r, &ctx, v);
                let mut out = BitVector::new(self.bits, Some(0));
                self.class.process_predictive(&input, &mut out);
                self.class.feedback(&input, &target, &mut self.rng);
            }
        }
    }

    /// Number of gate kernels grown so far.
    pub fn kernels(&self) -> usize {
        self.class.len()
    }
}

fn frame_hash(frame: &BitVector) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    frame.as_words().hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(i: usize) -> BitVector {
        BitVector::from_bits(&[i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3], 64)
    }

    #[test]
    fn gate_opens_only_routes_reliable_in_this_context() {
        // Episodes "1 2 3 | 1 ?": at "?", route (1,1) (what followed the last "1")
        // relays 2, which is what comes next. Elsewhere it is unreliable.
        let good = RelayChannel { query_lag: 1, value_offset: 1 };
        let bad = RelayChannel { query_lag: 0, value_offset: 2 };
        let pool = [good, bad];
        let mut gate = RouteGate::new(0.6, 2.0, 1, false);
        let mut th = Thalamus::new(64, 40, vec![]);
        for _ in 0..4 {
            for s in [1, 2, 3, 1, 9] {
                gate.learn(&th, &pool, &sym(s)); // score routes against what comes next
                th.observe(&sym(s));
            }
            gate.learn(&th, &pool, &sym(2));
            th.observe(&sym(2));
        }
        // a new episode, at "9": only the reliable route passes
        for s in [1, 2, 3, 1, 9] {
            th.observe(&sym(s));
        }
        let open = gate.select(&th, &pool);
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].0, good);
        assert_eq!(gate.gated_relay(&th, &pool, 64).as_words(), sym(2).as_words());
    }

    #[test]
    fn kernel_gate_learns_which_route_to_open_by_pattern_matching() {
        let good = RelayChannel { query_lag: 1, value_offset: 1 };
        let bad = RelayChannel { query_lag: 0, value_offset: 2 };
        let pool = [good, bad];
        let mut gate = KernelGate::new(64, 0.6, 1, 7);
        let mut th = Thalamus::new(64, 40, vec![]);
        for _ in 0..8 {
            for s in [1, 2, 3, 1, 9] {
                gate.learn(&th, &pool, &sym(s));
                th.observe(&sym(s));
            }
            gate.learn(&th, &pool, &sym(2));
            th.observe(&sym(2));
        }
        for s in [1, 2, 3, 1, 9] {
            th.observe(&sym(s));
        }
        let open = gate.select(&th, &pool);
        assert_eq!(open.iter().map(|(r, _)| *r).collect::<Vec<_>>(), vec![good]);
        assert_eq!(gate.gated_relay(&th, &pool, 64).as_words(), sym(2).as_words());
    }
}
