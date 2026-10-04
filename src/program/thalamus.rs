//! A thalamus-like relay: content-addressed routing of earlier cortical states.
//!
//! The relay keeps a short history of the frames a cortical node held. Each relay
//! *channel* is a fixed routing rule:
//! - a **query**: the frame `query_lag` steps back from now (0 = the current frame);
//! - a **key match**: the most recent earlier frame that overlaps the query by at
//!   least `match_fraction` of the query's active bits (content addressing);
//! - a **value**: the frame `value_offset` steps after that match, relayed to the
//!   channel's output.
//!
//! This is hard attention with one fixed query/key/value pattern per channel, the
//! operation an induction head performs ("find where this happened before and copy
//! what followed"). Biologically it plays the role of higher-order thalamic relay
//! between cortical areas (pulvinar / mediodorsal), with the limited number of
//! channels standing in for reticular-nucleus gating. Which channels are worth
//! keeping is left to the caller (e.g. ablation credit, see research/experiments/08).

use std::collections::HashMap;

use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::bitvec::BitVector;
use crate::kernel::{GrowthConfig, KernelClass, SimpleKernel};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RelayChannel {
    pub query_lag: usize,
    pub value_offset: usize,
}

pub struct Thalamus {
    bits: usize,
    capacity: usize,
    history: Vec<BitVector>, // oldest first
    pub channels: Vec<RelayChannel>,
    pub match_fraction: f32,
}

impl Thalamus {
    pub fn new(bits: usize, capacity: usize, channels: Vec<RelayChannel>) -> Self {
        Self { bits, capacity, history: Vec::new(), channels, match_fraction: 0.8 }
    }

    /// Record the cortical frame for this tick (call once per tick, before `relay`).
    pub fn observe(&mut self, frame: &BitVector) {
        self.history.push(frame.clone());
        if self.history.len() > self.capacity {
            self.history.remove(0);
        }
    }

    /// The most recently observed frame.
    pub fn current(&self) -> Option<&BitVector> {
        self.history.last()
    }

    /// Forget everything observed so far (e.g. between independent episodes).
    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// The value frame relayed by one channel now, if its query found a match.
    pub fn relay_channel(&self, c: RelayChannel) -> Option<&BitVector> {
        let n = self.history.len();
        let q_pos = n.checked_sub(1 + c.query_lag)?;
        let query = &self.history[q_pos];
        let q_bits = query.count_ones();
        if q_bits == 0 {
            return None;
        }
        let need = (q_bits as f32 * self.match_fraction).ceil() as u32;
        // most recent earlier frame matching the query whose value is already in the past
        (0..q_pos)
            .rev()
            .filter(|&p| p + c.value_offset < n)
            .find(|&p| overlap(query, &self.history[p]) >= need)
            .map(|p| &self.history[p + c.value_offset])
    }

    /// Hindsight route proposal: of the `candidates`, the routes that would
    /// relay `target` right now (overlap >= `match_fraction` of its bits). Call
    /// it when the cortex was surprised by `target`, before observing it, to
    /// find which routes would have carried the missing information.
    pub fn routes_that_would_relay(&self, target: &BitVector, candidates: &[RelayChannel]) -> Vec<RelayChannel> {
        let need = (target.count_ones() as f32 * self.match_fraction).ceil() as u32;
        if need == 0 {
            return Vec::new();
        }
        candidates
            .iter()
            .copied()
            .filter(|&c| self.relay_channel(c).map_or(false, |v| overlap(v, target) >= need))
            .collect()
    }

    /// Open-ended route discovery: on a surprise by `target` (call before
    /// observing it), find routes from the data instead of a fixed menu. For each
    /// earlier frame matching `target`, look at the frames up to `max_value_offset`
    /// before it (candidate keys); wherever one of them matches a frame in the
    /// current context, up to `max_query_lag` back, that pair defines a route
    /// (query lag, value offset). Routes are kept only if the relay, with its
    /// "most recent match" rule, would actually deliver `target` now.
    pub fn discover_routes(&self, target: &BitVector, max_query_lag: usize, max_value_offset: usize) -> Vec<RelayChannel> {
        let n = self.history.len();
        let need = |a: &BitVector| (a.count_ones() as f32 * self.match_fraction).ceil() as u32;
        let t_need = need(target);
        if t_need == 0 || n == 0 {
            return Vec::new();
        }
        let mut found: Vec<RelayChannel> = Vec::new();
        for p in 0..n {
            if overlap(&self.history[p], target) < t_need {
                continue;
            }
            for v in 1..=max_value_offset.min(p) {
                let key = &self.history[p - v];
                let k_need = need(key);
                if k_need == 0 {
                    continue;
                }
                for q in 0..=max_query_lag {
                    let Some(q_pos) = n.checked_sub(1 + q) else { break };
                    if q_pos <= p - v {
                        break; // the query must come after the key it matches
                    }
                    let c = RelayChannel { query_lag: q, value_offset: v };
                    if !found.contains(&c) && overlap(&self.history[q_pos], key) >= k_need {
                        found.push(c);
                    }
                }
            }
        }
        found.retain(|&c| self.relay_channel(c).map_or(false, |v| overlap(v, target) >= t_need));
        found
    }

    /// All channels' relayed frames concatenated, channel 0 first; an empty
    /// frame where a channel found nothing.
    pub fn relay(&self) -> BitVector {
        let mut words = Vec::with_capacity(self.channels.len() * self.bits.div_ceil(64));
        for &c in &self.channels {
            match self.relay_channel(c) {
                Some(v) => words.extend_from_slice(v.as_words()),
                None => words.extend(std::iter::repeat(0).take(self.bits.div_ceil(64))),
            }
        }
        BitVector::from_words(words)
    }
}

/// Consistency scores for candidate routes, for learning routes from surprises.
///
/// Raw "would have relayed it" votes favour routes that are right by coincidence
/// often (e.g. "the word after the last *the*" is some place 1 time in 6). Here
/// every candidate discovered so far is re-checked on each surprise: `tries`
/// counts surprises where it relayed something, `hits` those where it relayed the
/// right thing. Routes are ranked by smoothed precision `hits / (tries + 2)`.
#[derive(Default)]
pub struct RouteScores {
    stats: HashMap<RelayChannel, (f64, f64)>, // (hits, tries)
}

impl RouteScores {
    /// Record a surprise by `target` (before observing it): discover new
    /// candidates, then score every known candidate on this event.
    pub fn observe_surprise(&mut self, th: &Thalamus, target: &BitVector, max_query_lag: usize, max_value_offset: usize) {
        for r in th.discover_routes(target, max_query_lag, max_value_offset) {
            self.stats.entry(r).or_insert((0.0, 0.0));
        }
        let need = (target.count_ones() as f32 * th.match_fraction).ceil() as u32;
        for (r, (hits, tries)) in self.stats.iter_mut() {
            if let Some(v) = th.relay_channel(*r) {
                *tries += 1.0;
                if overlap(v, target) >= need {
                    *hits += 1.0;
                }
            }
        }
    }

    /// Smoothed precision of a route (0 if never seen).
    pub fn precision(&self, r: RelayChannel) -> f64 {
        self.stats.get(&r).map_or(0.0, |&(h, t)| h / (t + 2.0))
    }

    /// The most consistent route not in `exclude`, with at least `min_hits` hits.
    pub fn best_unused(&self, exclude: &[RelayChannel], min_hits: f64) -> Option<RelayChannel> {
        self.stats
            .iter()
            .filter(|(r, (h, _))| *h >= min_hits && !exclude.contains(r))
            .max_by(|a, b| {
                let pa = a.1 .0 / (a.1 .1 + 2.0);
                let pb = b.1 .0 / (b.1 .1 + 2.0);
                pa.partial_cmp(&pb).unwrap().then(b.0.query_lag.cmp(&a.0.query_lag)).then(b.0.value_offset.cmp(&a.0.value_offset))
            })
            .map(|(r, _)| *r)
    }

    /// Up to `n` routes with at least `min_hits` hits, most consistent first.
    pub fn top(&self, n: usize, min_hits: f64) -> Vec<RelayChannel> {
        let mut v: Vec<(f64, RelayChannel)> = self
            .stats
            .iter()
            .filter(|(_, (h, _))| *h >= min_hits)
            .map(|(r, (h, t))| (h / (t + 2.0), *r))
            .collect();
        v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.query_lag.cmp(&b.1.query_lag)).then(a.1.value_offset.cmp(&b.1.value_offset)));
        v.into_iter().take(n).map(|(_, r)| r).collect()
    }

    /// Multiply all counts by `factor` (forget slowly).
    pub fn decay(&mut self, factor: f64) {
        for (h, t) in self.stats.values_mut() {
            *h *= factor;
            *t *= factor;
        }
    }

    pub fn len(&self) -> usize {
        self.stats.len()
    }
}

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

fn overlap(a: &BitVector, b: &BitVector) -> u32 {
    a.as_words().iter().zip(b.as_words()).map(|(x, y)| (x & y).count_ones()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(i: usize) -> BitVector {
        BitVector::from_bits(&[i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3], 64)
    }

    #[test]
    fn induction_channel_copies_what_followed_the_earlier_match() {
        // mary(1) went(2) to(3) the(4) kitchen(5) .(6) where(7) is(8) mary(1) ?(9)
        let ch = RelayChannel { query_lag: 1, value_offset: 4 };
        let mut th = Thalamus::new(64, 32, vec![ch]);
        for s in [1, 2, 3, 4, 5, 6, 7, 8, 1, 9] {
            th.observe(&sym(s));
        }
        // at "?": query = "mary" one step back; earlier "mary" + 4 = "kitchen"
        assert_eq!(th.relay_channel(ch).unwrap().as_words(), sym(5).as_words());
        assert_eq!(th.relay().as_words(), sym(5).as_words());
    }

    #[test]
    fn no_match_relays_an_empty_frame() {
        let ch = RelayChannel { query_lag: 0, value_offset: 1 };
        let mut th = Thalamus::new(64, 8, vec![ch]);
        for s in [1, 2, 3] {
            th.observe(&sym(s));
        }
        assert!(th.relay_channel(ch).is_none());
        assert_eq!(th.relay().count_ones(), 0);
    }

    #[test]
    fn most_recent_match_wins_and_history_is_bounded() {
        let ch = RelayChannel { query_lag: 0, value_offset: 1 };
        let mut th = Thalamus::new(64, 5, vec![ch]);
        // 1 2 | 1 3 | 1  -> most recent earlier "1" was followed by 3
        for s in [1, 2, 1, 3, 1] {
            th.observe(&sym(s));
        }
        assert_eq!(th.relay_channel(ch).unwrap().as_words(), sym(3).as_words());
        th.observe(&sym(7));
        th.observe(&sym(7)); // capacity 5: the first "1 2" pair is gone
        assert_eq!(th.history.len(), 5);
    }

    #[test]
    fn hindsight_finds_the_route_that_would_have_relayed_the_answer() {
        let mut th = Thalamus::new(64, 32, vec![]);
        for s in [1, 2, 3, 4, 5, 6, 7, 8, 1, 9] {
            th.observe(&sym(s));
        }
        let all: Vec<RelayChannel> = (0..=2)
            .flat_map(|q| (1..=6).map(move |v| RelayChannel { query_lag: q, value_offset: v }))
            .collect();
        // the answer after "?" is "kitchen" (5)
        let routes = th.routes_that_would_relay(&sym(5), &all);
        assert_eq!(routes, vec![RelayChannel { query_lag: 1, value_offset: 4 }]);
    }

    #[test]
    fn discovery_finds_routes_outside_any_menu() {
        // mary(1) went(2) all(3) the(4) way(5) over(6) to(7) the(4) kitchen(9) .(10)
        // where(11) is(12) mary(1) right(13) now(14) ?(15)  -> answer kitchen
        let mut th = Thalamus::new(64, 40, vec![]);
        for s in [1, 2, 3, 4, 5, 6, 7, 4, 9, 10, 11, 12, 1, 13, 14, 15] {
            th.observe(&sym(s));
        }
        let routes = th.discover_routes(&sym(9), 8, 12);
        // query "mary" 3 back, kitchen 8 after the earlier "mary"
        assert!(routes.contains(&RelayChannel { query_lag: 3, value_offset: 8 }));
        // and every proposed route really relays the answer
        for r in &routes {
            assert_eq!(th.relay_channel(*r).unwrap().as_words(), sym(9).as_words());
        }
    }

    #[test]
    fn consistent_route_beats_coincidental_one() {
        // Two episodes "a b c X" ... then a surprise by the word after the query.
        // Route (0,1) "what followed the last copy of the current word" is right
        // both times; we check that precision ranks it first.
        let mut th = Thalamus::new(64, 40, vec![]);
        let mut scores = RouteScores::default();
        for s in [1, 5, 2, 6, 1] {
            th.observe(&sym(s));
        }
        scores.observe_surprise(&th, &sym(5), 4, 4); // after "1" came 5 before
        th.observe(&sym(5));
        for s in [3, 7, 1] {
            th.observe(&sym(s));
        }
        scores.observe_surprise(&th, &sym(5), 4, 4);
        let best = scores.best_unused(&[], 1.0).unwrap();
        assert_eq!(best, RelayChannel { query_lag: 0, value_offset: 1 });
        assert!(scores.precision(best) > 0.4);
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