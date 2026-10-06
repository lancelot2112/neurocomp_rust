//! Cortex: the context a cortical area holds, and content-addressed matching over it.
//!
//! `ContextBuffer` keeps a short history of the frames an area held (working memory,
//! the role of layer-6 / recurrent cortical context). A `RelayChannel` is a match rule
//! over that history:
//! - a **query**: the frame `query_lag` steps back from now (0 = the current frame);
//! - a **key match**: the most recent earlier frame that overlaps the query by at
//!   least `match_fraction` of the query's active bits (content addressing);
//! - a **value**: the frame `value_offset` steps after that match.
//!
//! This is hard attention with one query/key/value pattern per rule, the operation an
//! induction head performs ("find where this happened before and copy what followed").
//! Matching over stored context is cortical (and hippocampal), not thalamic: thalamic
//! cells do not store a history or search it by content. What *is* thalamic is
//! choosing which pathway gets through: see `thalamus` (`RouteGate`, `KernelGate`) and
//! the basal-ganglia gate in the episodic example. `RouteScores` learns which match
//! rules are worth having, from the predictor's surprises.
//!
//! (Until this split, `ContextBuffer` was called `Thalamus`; that name remains as an
//! alias in `thalamus`.)
//!
//! `CorticalColumn` puts the pieces of one area together as layers:
//! - **L4** (`assemble`): the input frames: the current input, frames relayed in through the
//!   thalamus (relay routes, memory recall), and the previous input from L6.
//! - **L2/3** (`l23`): the predictive `KernelClass`, which learns to predict the next input
//!   from L4 (`predict`, `learn`).
//! - **L5** (`prediction`, `confidence`, `surprise`): what the column sends to subcortex:
//!   the prediction, how reliable it is, and how surprising the actual input was (the
//!   probability-weighted comparator of experiment 14, as a column output).
//! - **L6** (`l6`): context held over time, the previous-input frame, and the match rules
//!   over it that the thalamus gates.

use crate::det::HashMap;

use crate::bitvec::BitVector;
use crate::fixed::{chance, isqrt, mul_ceil, ratio, Q16, ONE};
use crate::kernel::{KernelClass, SimpleKernel};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RelayChannel {
    pub query_lag: usize,
    pub value_offset: usize,
}

pub struct ContextBuffer {
    bits: usize,
    capacity: usize,
    history: Vec<BitVector>, // oldest first
    pub channels: Vec<RelayChannel>,
    /// In `Q16` (0.8 by default).
    pub match_fraction: Q16,
}

impl ContextBuffer {
    pub fn new(bits: usize, capacity: usize, channels: Vec<RelayChannel>) -> Self {
        Self { bits, capacity, history: Vec::new(), channels, match_fraction: 52429 }
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

    /// The frame `lag` steps before the most recent one (0 = the most recent).
    pub fn back(&self, lag: usize) -> Option<&BitVector> {
        self.history.len().checked_sub(1 + lag).map(|i| &self.history[i])
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
        let need = mul_ceil(q_bits as u64, self.match_fraction) as u32;
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
        let need = mul_ceil(target.count_ones() as u64, self.match_fraction) as u32;
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
        let need = |a: &BitVector| mul_ceil(a.count_ones() as u64, self.match_fraction) as u32;
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
/// right thing. Routes are ranked by smoothed precision `hits / (tries + 2)`. Counts are
/// kept in `Q16` units (one event = `ONE`), so `decay` can scale them in integers.
#[derive(Default)]
pub struct RouteScores {
    stats: HashMap<RelayChannel, (u64, u64)>, // (hits, tries), in Q16 units
}

impl RouteScores {
    /// Record a surprise by `target` (before observing it): discover new
    /// candidates, then score every known candidate on this event.
    pub fn observe_surprise(&mut self, th: &ContextBuffer, target: &BitVector, max_query_lag: usize, max_value_offset: usize) {
        for r in th.discover_routes(target, max_query_lag, max_value_offset) {
            self.stats.entry(r).or_insert((0, 0));
        }
        let need = mul_ceil(target.count_ones() as u64, th.match_fraction) as u32;
        for (r, (hits, tries)) in self.stats.iter_mut() {
            if let Some(v) = th.relay_channel(*r) {
                *tries += ONE as u64;
                if overlap(v, target) >= need {
                    *hits += ONE as u64;
                }
            }
        }
    }

    /// `hits / (tries + 2)` as an exact pair, for cross-multiplied comparison.
    fn smoothed(h: u64, t: u64) -> (u64, u64) {
        (h, t + 2 * ONE as u64)
    }

    /// Smoothed precision of a route in `Q16` (0 if never seen).
    pub fn precision(&self, r: RelayChannel) -> Q16 {
        self.stats.get(&r).map_or(0, |&(h, t)| {
            let (n, d) = Self::smoothed(h, t);
            ratio(n, d)
        })
    }

    /// The most consistent route not in `exclude`, with at least `min_hits` hits (`min_hits`
    /// in `Q16` units of hits).
    pub fn best_unused(&self, exclude: &[RelayChannel], min_hits: u64) -> Option<RelayChannel> {
        self.stats
            .iter()
            .filter(|(r, (h, _))| *h >= min_hits && !exclude.contains(r))
            .max_by(|a, b| {
                let (na, da) = Self::smoothed(a.1 .0, a.1 .1);
                let (nb, db) = Self::smoothed(b.1 .0, b.1 .1);
                ((na as u128) * (db as u128)).cmp(&((nb as u128) * (da as u128))).then(b.0.query_lag.cmp(&a.0.query_lag)).then(b.0.value_offset.cmp(&a.0.value_offset))
            })
            .map(|(r, _)| *r)
    }

    /// Up to `n` routes with at least `min_hits` hits (`Q16` units), most consistent first.
    pub fn top(&self, n: usize, min_hits: u64) -> Vec<RelayChannel> {
        let mut v: Vec<((u64, u64), RelayChannel)> = self
            .stats
            .iter()
            .filter(|(_, (h, _))| *h >= min_hits)
            .map(|(r, (h, t))| (Self::smoothed(*h, *t), *r))
            .collect();
        v.sort_by(|a, b| {
            ((b.0 .0 as u128) * (a.0 .1 as u128)).cmp(&((a.0 .0 as u128) * (b.0 .1 as u128))).then(a.1.query_lag.cmp(&b.1.query_lag)).then(a.1.value_offset.cmp(&b.1.value_offset))
        });
        v.into_iter().take(n).map(|(_, r)| r).collect()
    }

    /// Multiply all counts by `factor` (`Q16`; forget slowly).
    pub fn decay(&mut self, factor: Q16) {
        for (h, t) in self.stats.values_mut() {
            *h = (*h * factor as u64) >> 16;
            *t = (*t * factor as u64) >> 16;
        }
    }

    pub fn len(&self) -> usize {
        self.stats.len()
    }
}

pub(crate) fn overlap(a: &BitVector, b: &BitVector) -> u32 {
    a.as_words().iter().zip(b.as_words()).map(|(x, y)| (x & y).count_ones()).sum()
}

/// One cortical area as layers (see the module docs).
pub struct CorticalColumn {
    /// L2/3: the predictor.
    pub l23: KernelClass<SimpleKernel>,
    /// L6: context over time (and the match rules over it).
    pub l6: ContextBuffer,
    bits: usize,
    prediction: BitVector,
    confidence: Q16,
}

impl CorticalColumn {
    pub fn new(bits: usize, l23: KernelClass<SimpleKernel>, l6: ContextBuffer) -> Self {
        Self { l23, l6, bits, prediction: BitVector::new(bits, Some(0)), confidence: 0 }
    }

    /// L6: record this step's input in context.
    pub fn observe(&mut self, input: &BitVector) {
        self.l6.observe(input);
    }

    /// L6: the input before the current one (an empty frame if none).
    pub fn previous(&self) -> BitVector {
        self.l6.back(1).cloned().unwrap_or_else(|| BitVector::new(self.bits, Some(0)))
    }

    /// L4: `[current | frames… | previous]`, the input L2/3 predicts from.
    pub fn assemble(&self, current: &BitVector, frames: &[BitVector]) -> BitVector {
        let mut words = current.as_words().to_vec();
        for f in frames {
            words.extend_from_slice(f.as_words());
        }
        words.extend_from_slice(self.previous().as_words());
        BitVector::from_words(words)
    }

    /// L2/3 → L5: predict the next input from an assembled L4 input.
    pub fn predict(&mut self, l4: &BitVector) -> &BitVector {
        let mut out = BitVector::new(self.bits, Some(0));
        self.l23.process_predictive(l4, &mut out);
        self.prediction = out;
        self.confidence = self.l23.confidence().unwrap_or(0);
        &self.prediction
    }

    /// L2/3: learn from what actually came next.
    pub fn learn<R: rand::Rng + ?Sized>(&mut self, l4: &BitVector, target: &BitVector, rng: &mut R) {
        self.l23.feedback(l4, target, rng);
    }

    /// Without slow learning (e.g. at test): run only the L2/3 fast inhibitory loop on
    /// what actually came next. `learn` includes it.
    pub fn fast_inhibit(&mut self, target: &BitVector) {
        self.l23.fast_inhibit(target);
    }

    /// L5: the latest prediction.
    pub fn prediction(&self) -> &BitVector {
        &self.prediction
    }

    /// L5: reliability of the kernel that made the latest prediction (0 if none), in `Q16`.
    pub fn confidence(&self) -> Q16 {
        self.confidence
    }

    /// L5 → basal ganglia: the outcome of the latest prediction, the column's shared
    /// reward signal (dopamine-like): `actual`'s share of the prediction × the predicting
    /// kernel's reliability, in 0..=1 (= 1 − surprise). Any selector whose choice fed this
    /// prediction (a route, a recalled item, what working memory held) can learn from it,
    /// instead of each one checking its own content against the target.
    pub fn outcome(&self, actual: &BitVector) -> Q16 {
        ONE - self.surprise(actual)
    }

    /// L5: did the latest prediction read L4 frame `frame` (0 = current input, then the
    /// frames passed to `assemble`, then the previous input)? True if the winning kernel's
    /// input mask covers any bit of that frame. This is the column's attribution: a
    /// selector whose choice filled a frame the prediction never read had no effect on it.
    pub fn winner_reads(&self, frame: usize) -> bool {
        let Some(k) = self.l23.winner() else { return false };
        let words = self.bits / 64;
        k.input_set.iter().any(|&b| (k.input_idx + b as usize / 64) / words == frame)
    }

    /// L5 → basal ganglia, attributed: the outcome if the prediction read `frame`, else 0.
    /// The reward for a choice that fed `frame` (a route, a recalled item, a held cue).
    pub fn outcome_via(&self, frame: usize, actual: &BitVector) -> Q16 {
        if self.winner_reads(frame) {
            self.outcome(actual)
        } else {
            0
        }
    }

    /// L5: how surprising `actual` is given the latest prediction: 1 − (its share of the
    /// prediction × the predicting kernel's reliability).
    /// In `Q16`.
    pub fn surprise(&self, actual: &BitVector) -> Q16 {
        let predicted = self.prediction.count_ones();
        if predicted == 0 {
            return ONE;
        }
        let share = ratio(overlap(actual, &self.prediction) as u64, predicted as u64);
        ONE - ((share as u64 * self.confidence as u64) >> 16) as Q16
    }
}

/// A higher cortical area: a column one level up a hierarchy, working on a slower
/// timescale than the column below it.
///
/// - **Feedforward (L5 → higher-order thalamus → L4 above):** it receives two frames, the
///   bag of the current sentence's words and a slow state, the words the lower column
///   found surprising over the last `span` sentences (the lower column's L5 surprise,
///   integrated over time). Predictable filler does not enter the state, so a fact stated
///   several sentences back stays in view.
/// - **L2/3:** a predictive `CorticalColumn` that predicts the lower column's next input
///   from that sentence- and story-level context.
/// - **Feedback (L6 / apical → lower column):** its prediction is the lower column's
///   top-down frame, which the lower column learns to use (or ignore) like any other frame.
/// Role cells: an association area that learns, by competitive Hebbian plasticity, stable
/// codes for recurring kinds of input. Fed with the column's expectation for the next slot
/// (what its matching kernels predict, superposed), its cells come to stand for kinds of
/// slot ("a place goes here", "a person goes here") without anyone defining them.
///
/// - Each cell has a prototype (a bit pattern) and a fixed random output code.
/// - The cell whose prototype best matches the input (cosine of the bit sets) wins
///   (lateral inhibition, one winner); its output code is the area's output.
/// - Learning: the winner's prototype moves toward the input. Each input synapse it lacks
///   is gained with probability `rate`, each one the input lacks is lost with probability
///   `rate` / 4 (stochastic binary synapses), so a prototype settles on the bits that are
///   frequent in its kind of input.
/// - If no cell matches at least `vigilance`, an unused cell is recruited with the input
///   as its prototype (adaptive resonance: new categories only for new kinds of input).
pub struct RoleArea {
    protos: Vec<BitVector>,
    codes: Vec<BitVector>,
    used: usize,
    bits: usize,
    /// Minimum cosine to join a cell, in `Q16`.
    pub vigilance: Q16,
    /// Synapse gain probability per learning step, in `Q16` (loss: a quarter of it).
    pub rate: Q16,
}

impl RoleArea {
    pub fn new<R: rand::Rng>(bits: usize, cells: usize, rng: &mut R) -> Self {
        let all: Vec<usize> = (0..bits).collect();
        let codes = (0..cells)
            .map(|_| BitVector::from_bits(&rand::seq::SliceRandom::choose_multiple(all.as_slice(), rng, 32).copied().collect::<Vec<_>>(), bits))
            .collect();
        Self { protos: vec![BitVector::new(bits, Some(0)); cells], codes, used: 0, bits, vigilance: ONE / 2, rate: 6554 }
    }

    /// The cosine of two bit sets as an exact fraction of its square: (|a∧b|², |a|·|b|).
    /// Compared by cross-multiplication, so no square root is needed.
    fn similarity(a: &BitVector, b: &BitVector) -> (u64, u64) {
        let (mut inter, mut na, mut nb) = (0u64, 0u64, 0u64);
        for (x, y) in a.as_words().iter().zip(b.as_words()) {
            inter += (x & y).count_ones() as u64;
            na += x.count_ones() as u64;
            nb += y.count_ones() as u64;
        }
        if na == 0 || nb == 0 { (0, 1) } else { (inter * inter, na * nb) }
    }

    /// The winning cell for `x` and its cosine in `Q16` (None if `x` is empty or no cell is
    /// in use).
    pub fn winner(&self, x: &BitVector) -> Option<(usize, Q16)> {
        if x.count_ones() == 0 {
            return None;
        }
        (0..self.used)
            .map(|c| (c, Self::similarity(x, &self.protos[c])))
            .max_by(|a, b| ((a.1 .0 as u128) * (b.1 .1 as u128)).cmp(&((b.1 .0 as u128) * (a.1 .1 as u128))))
            .map(|(c, (n, d))| (c, isqrt((ratio(n, d) as u64) << 16) as Q16))
    }

    /// Present `x`: returns the active cell (if any); with `learn`, the winner moves toward
    /// `x`, or a new cell is recruited when nothing matches well enough.
    pub fn observe<R: rand::Rng>(&mut self, x: &BitVector, learn: bool, rng: &mut R) -> Option<usize> {
        if x.count_ones() == 0 {
            return None;
        }
        let best = self.winner(x);
        if learn {
            match best {
                Some((c, sim)) if sim >= self.vigilance || self.used == self.protos.len() => {
                    let p = &mut self.protos[c];
                    for (pw, &xw) in p.as_words_mut().iter_mut().zip(x.as_words()) {
                        let mut diff = *pw ^ xw;
                        while diff != 0 {
                            let b = diff.trailing_zeros();
                            diff &= diff - 1;
                            let gain = xw >> b & 1 == 1;
                            if chance(rng, if gain { self.rate } else { self.rate / 4 }) {
                                *pw ^= 1u64 << b;
                            }
                        }
                    }
                    return Some(c);
                }
                _ => {
                    let c = self.used;
                    self.protos[c] = x.clone();
                    self.used += 1;
                    return Some(c);
                }
            }
        }
        best.map(|b| b.0)
    }

    /// Output code of `cell`.
    pub fn code(&self, cell: usize) -> &BitVector {
        &self.codes[cell]
    }

    /// Cells recruited so far.
    pub fn used(&self) -> usize {
        self.used
    }

    pub fn bits(&self) -> usize {
        self.bits
    }
}

/// A higher area's saved context (see `HigherArea::save_context`).
#[derive(Clone)]
pub struct AreaContext {
    window: std::collections::VecDeque<BitVector>,
    window_words: std::collections::VecDeque<Vec<BitVector>>,
    faded: BitVector,
}

pub struct HigherArea {
    pub column: CorticalColumn,
    /// Slow state as one frame per recent sentence (newest first) instead of one frame
    /// for all of them: a fact's sentence stays apart from the filler around it.
    pub separate: bool,
    bits: usize,
    span: usize,
    /// Surprising content of recent sentences, newest last.
    window: std::collections::VecDeque<BitVector>,
    /// Attention at growth: a new kernel samples the slow state from one remembered word
    /// only (chosen at random), plus the whole sentence bag, so it keys on (sentence, one
    /// earlier fact) and generalizes across the other words that happen to be in the state.
    pub focus: bool,
    /// The surprising words, one code each: of the current sentence, and of recent ones.
    pending_words: Vec<BitVector>,
    window_words: std::collections::VecDeque<Vec<BitVector>>,
    /// Fading state (see `set_fade`): the per-sentence survival probability of a bit.
    fade: Option<Q16>,
    faded: BitVector,
    fade_rng: rand::rngs::StdRng,
}

impl HigherArea {
    pub fn new(bits: usize, l23: KernelClass<SimpleKernel>, span: usize) -> Self {
        Self {
            column: CorticalColumn::new(bits, l23, ContextBuffer::new(bits, 4, Vec::new())),
            separate: false,
            bits,
            span: span.max(1),
            window: std::collections::VecDeque::new(),
            focus: false,
            pending_words: Vec::new(),
            window_words: std::collections::VecDeque::new(),
            fade: None,
            faded: BitVector::new(bits, Some(0)),
            fade_rng: rand::SeedableRng::seed_from_u64(span as u64 * 7919 + 13),
        }
    }

    /// A fading state instead of a window (a drifting temporal context, after Howard &
    /// Kahana's temporal context model and the time-varying codes of lateral entorhinal
    /// cortex). Every sentence end, each bit of the state survives with probability
    /// 0.5^(1 / half_life); the sentence's surprising words then enter with all their
    /// bits. Newer words are stronger, older ones fade, and a word's strength is how many
    /// of its bits remain, so recency is in the code itself. None: the fixed window.
    pub fn set_fade(&mut self, half_life: Option<f64>) { // float: config (converted once)
        self.fade = half_life.map(|h| crate::fixed::q16(0.5f64.powf(1.0 / h.max(0.1)))); // float: config
    }

    /// The slow state: everything the lower column found surprising in the last `span`
    /// sentences (including the current one's surprises so far, `current`).
    pub fn state(&self, current: &BitVector) -> BitVector {
        if self.fade.is_some() {
            let mut s = current.clone();
            s.or_mut(&self.faded);
            return s;
        }
        let mut s = current.clone();
        for w in &self.window {
            s.or_mut(w);
        }
        s
    }

    /// The higher area's input: `[sentence bag | slow state]`, or with `separate`,
    /// `[sentence bag | last sentence's surprises | the one before | ...]` (`span` frames,
    /// empty where there is no such sentence yet).
    pub fn input(&self, sentence: &BitVector, surprising: &BitVector) -> BitVector {
        let mut words = sentence.as_words().to_vec();
        if self.separate {
            let empty = BitVector::new(self.bits, Some(0));
            for i in 0..self.span {
                let w = self.window.len().checked_sub(i + 1).map(|j| &self.window[j]).unwrap_or(&empty);
                words.extend_from_slice(w.as_words());
            }
        } else {
            words.extend_from_slice(self.state(surprising).as_words());
        }
        BitVector::from_words(words)
    }

    /// `input` plus, for an area inside a chain, the prediction of the area above it as one
    /// more frame: `[sentence bag | slow state | from above]`. The area learns to copy it
    /// where it carries the target, as the column copies its own top-down frame.
    pub fn input_with(&self, sentence: &BitVector, surprising: &BitVector, above: Option<&BitVector>) -> BitVector {
        let mut x = self.input(sentence, surprising);
        if let Some(a) = above {
            let mut words = x.as_words().to_vec();
            words.extend_from_slice(a.as_words());
            x = BitVector::from_words(words);
        }
        x
    }

    /// The surprising words of the window, newest sentence first (as noted).
    pub fn recent_words(&self) -> impl Iterator<Item = &BitVector> {
        self.window_words.iter().rev().flat_map(|ws| ws.iter().rev())
    }

    /// A context boundary just after the newest window word matching `held`: forget the
    /// sentence holding it and every older one, keep the newer ones. Returns whether
    /// anything was forgotten.
    pub fn forget_through(&mut self, held: impl Fn(&BitVector) -> bool) -> bool {
        let Some(i) = self.window_words.iter().rposition(|ws| ws.iter().any(|w| held(w))) else { return false };
        let drop = i + 1;
        // the OR window has one entry per sentence with surprising content, as the word
        // lists do; if they ever differ, forget it all rather than misalign them
        if self.window.len() == self.window_words.len() {
            self.window.drain(..drop);
        } else {
            self.window.clear();
        }
        self.window_words.drain(..drop);
        true
    }

    /// The area's context (its window), to be saved and later reinstated.
    pub fn save_context(&self) -> AreaContext {
        AreaContext { window: self.window.clone(), window_words: self.window_words.clone(), faded: self.faded.clone() }
    }

    /// Reinstate a saved context: the window becomes what it was when saved.
    pub fn restore_context(&mut self, c: &AreaContext) {
        self.window = c.window.clone();
        self.window_words = c.window_words.clone();
        self.faded = c.faded.clone();
        self.pending_words.clear();
    }

    /// Context boundary (a new story): forget the window.
    pub fn clear(&mut self) {
        self.faded = BitVector::new(self.bits, Some(0));
        self.window.clear();
        self.window_words.clear();
        self.pending_words.clear();
    }

    /// The input with a leading frame (e.g. role cells): `[lead | slow state | sentence bag]`.
    /// Growth deepens frame by frame, so kernels can key on (lead, state) before they need
    /// the sentence's words.
    pub fn input_lead(&self, lead: &BitVector, sentence: &BitVector, surprising: &BitVector) -> BitVector {
        let mut words = lead.as_words().to_vec();
        words.extend_from_slice(self.state(surprising).as_words());
        words.extend_from_slice(sentence.as_words());
        BitVector::from_words(words)
    }

    /// The area's window, in sentences.
    pub fn span(&self) -> usize {
        self.span
    }

    /// Top-down prediction for the lower column's next input.
    pub fn predict(&mut self, input: &BitVector) -> BitVector {
        self.column.predict(input).clone()
    }

    pub fn learn<R: rand::Rng + ?Sized>(&mut self, input: &BitVector, target: &BitVector, rng: &mut R) {
        if self.focus && !self.separate {
            // growth may sample all of frame 0 (the sentence) and one state word in frame 1
            // earlier sentences only: the current one is already in the bag
            let words: Vec<&BitVector> = self.window_words.iter().flatten().collect();
            if !words.is_empty() {
                let pick = words[rng.gen_range(0..words.len())];
                let mut mask = BitVector::new(2 * self.bits, Some(0));
                for (i, w) in mask.as_words_mut().iter_mut().enumerate().take(self.bits / 64) {
                    *w = !0;
                    let _ = i;
                }
                for (i, &w) in pick.as_words().iter().enumerate() {
                    mask.as_words_mut()[self.bits / 64 + i] = w;
                }
                self.column.l23.set_growth_mask(Some(mask));
                self.column.learn(input, target, rng);
                self.column.l23.set_growth_mask(None);
                return;
            }
        }
        self.column.learn(input, target, rng);
    }

    /// A word the lower column found surprising (kept as its own item for `focus`).
    pub fn note_word(&mut self, code: &BitVector) {
        self.pending_words.push(code.clone());
    }

    /// End of a sentence: its surprising content joins the slow state.
    pub fn end_sentence(&mut self, surprising: &BitVector) {
        if let Some(p) = self.fade {
            for w in self.faded.as_words_mut() {
                let mut x = *w;
                while x != 0 {
                    let b = x.trailing_zeros();
                    x &= x - 1;
                    if !chance(&mut self.fade_rng, p) {
                        *w &= !(1u64 << b);
                    }
                }
            }
            self.faded.or_mut(surprising);
        }
        let words = std::mem::take(&mut self.pending_words);
        if !words.is_empty() {
            self.window_words.push_back(words);
            while self.window_words.len() > self.span {
                self.window_words.pop_front();
            }
        }
        if self.separate || surprising.count_ones() > 0 {
            self.window.push_back(surprising.clone());
            while self.window.len() > self.span {
                self.window.pop_front();
            }
        }
    }

    pub fn bits(&self) -> usize {
        self.bits
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn fading_state_keeps_recent_words_strongest() {
        use crate::kernel::GrowthConfig;
        let bits = 512;
        let cfg = GrowthConfig { frame_words: bits / 64, max_frames: 2, sample_bits: 8, ..GrowthConfig::default() };
        let mut area = HigherArea::new(bits, KernelClass::predictive(cfg), 4);
        area.set_fade(Some(4.0));
        let word = |i: usize| BitVector::from_bits(&(i * 64..i * 64 + 32).collect::<Vec<_>>(), bits);
        let empty = BitVector::new(bits, Some(0));
        area.end_sentence(&word(0)); // an old word
        for _ in 0..8 {
            area.end_sentence(&empty); // 8 sentences pass
        }
        area.end_sentence(&word(1)); // a recent word
        let st = area.state(&empty);
        let strength = |w: &BitVector| w.as_words().iter().zip(st.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>();
        // half-life 4 sentences: after 9 sentences about 32 * 0.5^(9/4) = 7 bits remain
        assert_eq!(strength(&word(1)), 32);
        assert!(strength(&word(0)) < 16, "old word kept {} bits", strength(&word(0)));
        area.clear();
        assert_eq!(area.state(&empty).count_ones(), 0);
    }

    #[test]
    fn role_cells_learn_two_kinds_of_input_unsupervised() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let bits = 256;
        let mut roles = RoleArea::new(bits, 8, &mut rng);
        // two kinds: bits 0..40 or 100..140, each presentation a random 30 of its 40
        let sample = |base: usize, rng: &mut rand::rngs::StdRng| {
            let mut v: Vec<usize> = (base..base + 40).collect();
            while v.len() > 30 {
                let i = rng.gen_range(0..v.len());
                v.swap_remove(i);
            }
            BitVector::from_bits(&v, bits)
        };
        for i in 0..400 {
            let x = sample(if i % 2 == 0 { 0 } else { 100 }, &mut rng);
            roles.observe(&x, true, &mut rng);
        }
        let a: Vec<usize> = (0..20).filter_map(|_| roles.observe(&sample(0, &mut rng), false, &mut rng)).collect();
        let b: Vec<usize> = (0..20).filter_map(|_| roles.observe(&sample(100, &mut rng), false, &mut rng)).collect();
        // each kind maps to one cell, and the two kinds to different cells
        assert!(a.iter().all(|&c| c == a[0]) && b.iter().all(|&c| c == b[0]));
        assert_ne!(a[0], b[0]);
        assert!(roles.used() <= 4);
    }

    use super::*;
    use crate::kernel::GrowthConfig;

    #[test]
    fn column_learns_a_sequence_and_reports_surprise() {
        let bits = 64;
        let cfg = GrowthConfig { frame_words: 1, max_frames: 2, sample_bits: 4, ..GrowthConfig::default() };
        let mut col = CorticalColumn::new(bits, KernelClass::predictive(cfg), ContextBuffer::new(bits, 8, Vec::new()));
        let (a, b) = (sym(1), sym(2));
        let mut rng = rand::thread_rng();
        for _ in 0..5 {
            col.observe(&a);
            let l4 = col.assemble(&a, &[]);
            col.predict(&l4);
            col.learn(&l4, &b, &mut rng);
            col.observe(&b);
        }
        col.observe(&a);
        let l4 = col.assemble(&a, &[]);
        assert_eq!(col.predict(&l4).as_words(), b.as_words()); // L5: a is followed by b
        assert!(col.surprise(&b) < crate::fixed::q16(0.5));
        assert!(col.surprise(&sym(3)) > crate::fixed::q16(0.9));
        assert!(col.outcome(&b) > crate::fixed::q16(0.5) && col.outcome(&sym(3)) < crate::fixed::q16(0.1)); // L5 → BG
        assert!(col.winner_reads(0) && !col.winner_reads(1)); // read the current input only
        assert_eq!(col.previous().as_words(), b.as_words()); // L6: the input before a
    }

    #[test]
    fn fast_inhibition_makes_the_column_try_an_alternative() {
        // x is followed by a and by b equally often; with adaptation, once x → a has
        // happened, the next x predicts b (inhibition of return), and the tag fades
        let bits = 64;
        let cfg = GrowthConfig { frame_words: 1, max_frames: 1, sample_bits: 4, ..GrowthConfig::default() };
        let mut col = CorticalColumn::new(bits, KernelClass::predictive(cfg), ContextBuffer::new(bits, 8, Vec::new()));
        let mut rng = rand::thread_rng();
        let (x, a, b) = (sym(1), sym(2), sym(3));
        let mut step = |col: &mut CorticalColumn, next: &BitVector| -> BitVector {
            let l4 = BitVector::from_words(x.as_words().to_vec());
            let p = col.predict(&l4).clone();
            col.learn(&l4, next, &mut rng);
            p
        };
        for i in 0..20 {
            step(&mut col, if i % 2 == 0 { &a } else { &b });
        }
        col.l23.set_fast_inhibition(Some(crate::kernel::FastInhibition::new(2, true, false)));
        step(&mut col, &a); // x → a happens: the a-kernel is inhibited
        assert_eq!(col.l23.inhibited(), 1);
        assert_eq!(step(&mut col, &b).as_words(), b.as_words()); // so x now predicts b
        assert_eq!(step(&mut col, &a).as_words(), a.as_words()); // b inhibited in turn, a's 2-step tag has faded

        // gated by reliability, in integers: k = 0 tags only kernels right less than half
        // the time ((misses + 1) << 0 > hits + 1); x → a has been right about half the
        // time and was just confirmed, so it is not inhibited
        let mut fast = crate::kernel::FastInhibition::new(2, true, false);
        fast.reliable_shift = Some(0);
        col.l23.set_fast_inhibition(Some(fast));
        step(&mut col, &a);
        assert_eq!(col.l23.inhibited(), 0);
    }

    #[test]
    fn higher_area_carries_a_story_level_fact_across_filler_and_predicts_with_it() {
        use rand::SeedableRng;
        // story: a time-of-day sentence (surprising), filler (predictable, nothing
        // surprising), then "name went to the" -> place, where place depends on the name
        // AND the time; the higher area sees the name in the sentence bag and the time in
        // its slow state
        let bits = 64;
        let cfg = GrowthConfig { frame_words: 1, max_frames: 2, sample_bits: 4, ..GrowthConfig::default() };
        let mut area = HigherArea::new(bits, KernelClass::predictive(cfg), 3);
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        let (times, names, places) = ([sym(1), sym(2)], [sym(3), sym(4), sym(5)], [sym(6), sym(7), sym(8), sym(9), sym(10), sym(11)]);
        let place = |n: usize, t: usize| (2 * n + t) % 6;
        let mut right = 0;
        for story in 0..400 {
            let (t, n) = (story % 2, (story / 2) % 3);
            area.end_sentence(&times[t]); // "in the morning ." was surprising
            area.end_sentence(&BitVector::new(bits, Some(0))); // filler: nothing surprising
            let mut bag = names[n].clone();
            bag.or_mut(&sym(12)); // "went to the"
            let input = area.input(&bag, &names[n]);
            let pred = area.predict(&input);
            if story >= 300 && pred.as_words() == places[place(n, t)].as_words() {
                right += 1;
            }
            area.learn(&input, &places[place(n, t)], &mut rng);
            area.end_sentence(&names[n]);
        }
        assert_eq!(right, 100);
    }

    fn sym(i: usize) -> BitVector {
        BitVector::from_bits(&[i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3], 64)
    }

    #[test]
    fn induction_channel_copies_what_followed_the_earlier_match() {
        // mary(1) went(2) to(3) the(4) kitchen(5) .(6) where(7) is(8) mary(1) ?(9)
        let ch = RelayChannel { query_lag: 1, value_offset: 4 };
        let mut th = ContextBuffer::new(64, 32, vec![ch]);
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
        let mut th = ContextBuffer::new(64, 8, vec![ch]);
        for s in [1, 2, 3] {
            th.observe(&sym(s));
        }
        assert!(th.relay_channel(ch).is_none());
        assert_eq!(th.relay().count_ones(), 0);
    }

    #[test]
    fn most_recent_match_wins_and_history_is_bounded() {
        let ch = RelayChannel { query_lag: 0, value_offset: 1 };
        let mut th = ContextBuffer::new(64, 5, vec![ch]);
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
        let mut th = ContextBuffer::new(64, 32, vec![]);
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
        let mut th = ContextBuffer::new(64, 40, vec![]);
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
        let mut th = ContextBuffer::new(64, 40, vec![]);
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
        let best = scores.best_unused(&[], crate::fixed::ONE as u64).unwrap();
        assert_eq!(best, RelayChannel { query_lag: 0, value_offset: 1 });
        assert!(scores.precision(best) > crate::fixed::q16(0.4));
    }

}
