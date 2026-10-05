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
                    if rel >= self.threshold && self.says_right(&out) {
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

/// Layer-6 corticothalamic gating: cortex sets the gain of each thalamic relay channel.
///
/// Unlike the basal-ganglia gate (one channel released, learned from dopamine), every
/// channel has its own relay frame and any number can be open at once, as L6 feedback
/// (directly and through the reticular nucleus) enhances some relays and suppresses
/// others. The gain is a facilitation value per (channel, cortical context): the
/// channel's sparse code bound to the context by rotation, read through bit-sliced
/// counters, as in `BasalGanglia`. A channel is open when its facilitation is at least
/// `threshold`; every channel starts there (relays pass until cortex learns to suppress
/// them).
///
/// Learning is Hebbian and needs no reward: after a prediction, each open channel that
/// relayed something is strengthened if the column's prediction read its frame and came
/// true (L5 attribution, `CorticalColumn::outcome_via`), and weakened if not. A closed
/// channel relays nothing, so it can never be read; with `explore` probability it opens
/// anyway during learning, so a suppressed channel can recover.
pub struct CorticothalamicGate {
    gain: crate::bitvec::SlicedCounter,
    bits: usize,
    codes: Vec<BitVector>,
    /// Open if facilitation >= threshold (0..=1).
    pub threshold: f32,
    /// Probability that a closed channel opens anyway while learning.
    pub explore: f64,
    /// Probability that each bit of a channel steps per update.
    pub rate: f64,
    /// Weakening is `rate × weaken` (strengthening is `rate`): with weaken w < 1, a
    /// channel stays open in a context if it is used in at least about w / (1 + w) of
    /// its uses there, so a relay that matters in only some of a context's cases is kept.
    pub weaken: f64,
}

impl CorticothalamicGate {
    pub fn new(bits: usize, channels: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let all: Vec<usize> = (0..bits).collect();
        let codes = (0..channels)
            .map(|_| BitVector::from_bits(&all.choose_multiple(&mut rng, 32).copied().collect::<Vec<_>>(), bits))
            .collect();
        let planes = 4;
        Self { gain: crate::bitvec::SlicedCounter::new(bits, planes, 1 << (planes - 1)), bits, codes, threshold: 0.5, explore: 0.05, rate: 0.3, weaken: 1.0 }
    }

    fn bound(&self, channel: usize, context: &BitVector) -> BitVector {
        let mut c = self.codes[channel].clone();
        c.rotl_mut((frame_hash(context) % self.bits as u64) as usize);
        c
    }

    /// Facilitation of `channel` in `context`, in 0..=1.
    pub fn facilitation(&self, channel: usize, context: &BitVector) -> f32 {
        let c = self.bound(channel, context);
        self.gain.sum(&c) as f32 / (c.count_ones() as f32 * self.gain.max() as f32)
    }

    /// Which channels are open in `context`. With `rng`, closed channels open with
    /// probability `explore`.
    pub fn open<R: rand::Rng>(&self, context: &BitVector, mut rng: Option<&mut R>) -> Vec<bool> {
        (0..self.codes.len())
            .map(|ch| {
                self.facilitation(ch, context) >= self.threshold
                    || rng.as_mut().map_or(false, |r| r.gen_bool(self.explore))
            })
            .collect()
    }

    /// Hebbian update for one open channel: `used` = the prediction read its frame and
    /// came true.
    pub fn learn<R: rand::Rng>(&mut self, channel: usize, context: &BitVector, used: bool, rng: &mut R) {
        let c = self.bound(channel, context);
        let mut step = BitVector::new(self.bits, Some(0));
        for (wi, &w) in c.as_words().iter().enumerate() {
            let mut w = w;
            while w != 0 {
                let b = w.trailing_zeros() as usize;
                w &= w - 1;
                if rng.gen_bool(if used { self.rate } else { self.rate * self.weaken }) {
                    step.bit_set(wi * 64 + b);
                }
            }
        }
        if used {
            self.gain.increment(&step);
        } else {
            self.gain.decrement(&step);
        }
    }
}

fn frame_hash(frame: &BitVector) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    frame.as_words().hash(&mut h);
    h.finish()
}

/// Precision-weighted mixing of sources: a pulvinar-like gain per source, not a switch.
///
/// Several sources (the column's own prediction, hippocampal recall, a higher area's
/// top-down prediction) each propose candidates. Instead of gating one through, every
/// candidate collects a vote from each source that proposes it, weighted by that
/// source's learned reliability in the current context, and the candidate with the most
/// evidence wins. Weights add, as precisions do in a Kalman update: two sources that agree
/// give more evidence than either alone, and a reliable source outweighs an unreliable one
/// without silencing it.
///
/// Reliability is kept per (source, key), where the key is the caller's context (e.g. the
/// previous and current word) plus a bucket of the source's own per-question confidence
/// (the higher area's L5 confidence, how many places a recall offers). It is the smoothed
/// rate p = (hits + 1) / (hits + misses + 2) that a candidate the source proposed was
/// right, held in 8-bit counters halved together at saturation (as `KernelStats`).
///
/// A vote's weight is −log2(1 − p) in 1/16 bits: the evidence that the source is not
/// wrong. It is computed in integers from a log table, −log2(1 − p) = log2(h + m + 2) −
/// log2(m + 1). Summed over the sources proposing a candidate, it is −log2 of the chance
/// that all of them are wrong (if independent); `confidence` turns a total back into that
/// probability's complement. A source never seen in a context has p = 1/2: one bit.
pub struct SourceMix {
    stats: HashMap<(u8, u64), (u8, u8)>,
    /// log2(i) × 16, rounded, for i in 0..=512 (index 0 unused)
    log2: Vec<u16>,
}

impl Default for SourceMix {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceMix {
    pub fn new() -> Self {
        let log2 = (0..=512usize).map(|i| if i == 0 { 0 } else { ((i as f64).log2() * 16.0).round() as u16 }).collect();
        Self { stats: HashMap::new(), log2 }
    }

    /// Weight (1/16 bits) of a vote by `source` under `key`.
    pub fn weight(&self, source: u8, key: u64) -> u32 {
        let (h, m) = self.stats.get(&(source, key)).copied().unwrap_or((0, 0));
        let den = h as usize + m as usize + 2;
        (self.log2[den] - self.log2[m as usize + 1]) as u32
    }

    /// Smoothed rate that a candidate `source` proposed under `key` is right.
    pub fn rate(&self, source: u8, key: u64) -> f32 {
        let (h, m) = self.stats.get(&(source, key)).copied().unwrap_or((0, 0));
        (h as f32 + 1.0) / (h as f32 + m as f32 + 2.0)
    }

    /// Record whether a candidate `source` proposed under `key` was right.
    pub fn record(&mut self, source: u8, key: u64, right: bool) {
        let e = self.stats.entry((source, key)).or_insert((0, 0));
        if (right && e.0 == u8::MAX) || (!right && e.1 == u8::MAX) {
            *e = (e.0 / 2, e.1 / 2);
        }
        if right {
            e.0 += 1;
        } else {
            e.1 += 1;
        }
    }

    /// The candidate with the most summed evidence among `votes` (candidate, weight);
    /// ties go to the earliest vote's candidate. Returns (candidate, total weight).
    pub fn combine(votes: &[(usize, u32)]) -> Option<(usize, u32)> {
        let mut totals: Vec<(usize, u32)> = Vec::new();
        for &(c, w) in votes {
            match totals.iter_mut().find(|t| t.0 == c) {
                Some(t) => t.1 += w,
                None => totals.push((c, w)),
            }
        }
        let mut best: Option<(usize, u32)> = None;
        for t in totals {
            if best.map_or(true, |b| t.1 > b.1) {
                best = Some(t);
            }
        }
        best
    }

    /// Probability that not all of the sources behind `total` are wrong: 1 − 2^(−total/16).
    pub fn confidence(total: u32) -> f32 {
        1.0 - (-(total as f32) / 16.0).exp2()
    }

    /// Number of (source, key) entries learned.
    pub fn len(&self) -> usize {
        self.stats.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stats.is_empty()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn source_mix_weights_by_reliability_and_adds_agreement() {
        let mut mix = SourceMix::new();
        assert_eq!(mix.weight(0, 1), 16); // unknown: p = 1/2, one bit
        for i in 0..20 {
            mix.record(0, 1, i % 10 != 0); // source 0: right 90%
            mix.record(1, 1, i % 2 == 0); // source 1: right 50%
            mix.record(2, 1, i % 4 == 0); // source 2: right 25%
        }
        let (w0, w1, w2) = (mix.weight(0, 1), mix.weight(1, 1), mix.weight(2, 1));
        assert!(w0 > w1 && w1 > w2);
        // the reliable source beats the unreliable one when they disagree...
        assert_eq!(SourceMix::combine(&[(7, w2), (3, w0)]).unwrap().0, 3);
        // ...but weaker sources that agree can outvote it: 90% right is ~2.9 bits, three
        // 50% sources are 3
        assert_eq!(SourceMix::combine(&[(3, w0), (5, w1), (5, w1)]).unwrap().0, 3);
        let (c, total) = SourceMix::combine(&[(3, w0), (5, w1), (5, w1), (5, w1)]).unwrap();
        assert_eq!(c, 5);
        // agreement raises confidence above either source alone
        let alone = SourceMix::confidence(w1);
        assert!(SourceMix::confidence(total) > alone && (alone - mix.rate(1, 1)).abs() < 0.03);
        // other keys are independent
        assert_eq!(mix.weight(0, 2), 16);
    }

    #[test]
    fn source_mix_counters_halve_together_at_saturation() {
        let mut mix = SourceMix::new();
        for _ in 0..300 {
            mix.record(0, 0, true);
        }
        mix.record(0, 0, false);
        let r = mix.rate(0, 0);
        assert!(r > 0.98 && r < 1.0);
    }

    #[test]
    fn corticothalamic_gate_opens_what_cortex_uses_per_context() {
        let mut g = CorticothalamicGate::new(1024, 2, 1);
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let (ctx_a, ctx_b) = (BitVector::from_bits(&[1, 2, 3], 1024), BitVector::from_bits(&[9, 10, 11], 1024));
        let none: Option<&mut rand::rngs::StdRng> = None;
        assert_eq!(g.open(&ctx_a, none), vec![true, true]); // relays pass until suppressed
        for _ in 0..40 {
            // context a uses channel 0, context b uses channel 1
            g.learn(0, &ctx_a, true, &mut rng);
            g.learn(1, &ctx_a, false, &mut rng);
            g.learn(0, &ctx_b, false, &mut rng);
            g.learn(1, &ctx_b, true, &mut rng);
        }
        let none: Option<&mut rand::rngs::StdRng> = None;
        assert_eq!(g.open(&ctx_a, none), vec![true, false]);
        let none: Option<&mut rand::rngs::StdRng> = None;
        assert_eq!(g.open(&ctx_b, none), vec![false, true]);
    }

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
