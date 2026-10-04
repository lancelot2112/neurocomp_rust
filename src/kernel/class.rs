use crate::bitvec::BitVector;
use crate::common::config;
use crate::kernel::simple::{KernelStats, SimpleKernel};
use rand::seq::SliceRandom;

/// How to combine the output mask into the output BitVector (word-aligned).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelOp {
    Or,
    And,
    Xor,
    Clear, // a & !b
}

pub struct KernelClassTemperature {
    pub max: i16,
    pub min: i16,
    pub current: i16,
}

impl KernelClassTemperature {
    pub fn default() -> Self {
        let kdef = &config().kernel;
        Self {
            min: kdef.temperature_min,
            max: kdef.temperature_max,
            current: kdef.temperature_initial,
        }
    }

    pub fn new(min: i16, max: i16, initial: i16) -> Self {
        assert!(min <= initial && initial <= max);
        Self { min, max, current: initial }
    }

    pub fn incr(&mut self) {
        if self.current < self.max {
            self.current += 1;
        }
    }

    pub fn decr(&mut self) {
        if self.current > self.min {
            self.current -= 1;
        }
    }
}

// Light, borrow-based context passed to each spawned kernel
pub struct KernelContext<'a> {
    pub input: &'a BitVector,
    pub inhibit: &'a BitVector,
    pub temperature: i16,
    pub phase: u16,
}

pub trait KernelTrait {
    fn try_fire<R: rand::Rng + ?Sized>(&mut self, ctx: &KernelContext, out: &mut BitVector, rng: &mut R) -> bool;
    fn word_range(&self) -> (usize, usize);
    /// How far above (positive) or below (negative) its firing threshold the kernel is.
    fn excitation(&self, ctx: &KernelContext) -> isize;
}

/// Surprise-driven growth for a predictive class (see `KernelClass::feedback`).
///
/// The class input is expected to be `frames` history frames concatenated
/// most-recent first, each `frame_words` words wide.
#[derive(Clone, Copy, Debug)]
pub struct GrowthConfig {
    pub max_kernels: usize,      // budget; past this the least-recently-useful kernel is recycled
    pub frame_words: usize,      // words per history frame in the class input
    pub max_frames: usize,       // deepest context a grown kernel may span
    pub sample_bits: usize,      // active input bits sampled per frame for a new kernel
    pub match_fraction: f32,     // per-frame match needed: threshold = sampled - floor(sample_bits * (1 - match_fraction))
    pub surprise_fraction: f32,  // grow when more than this fraction of target bits went unpredicted
    /// Synapse-level credit: a kernel that matched at least this fraction of its
    /// connections (but not its threshold) and whose prediction the target confirms
    /// drops the connections that were silent, becoming more general. None = off.
    pub generalize: Option<f32>,
    /// How many such confirmations a silent connection needs before it is dropped
    /// (a connection's count resets whenever it is active while its kernel is
    /// right). 1 = drop on the first near miss.
    pub generalize_after: u8,
}

impl Default for GrowthConfig {
    fn default() -> Self {
        Self {
            max_kernels: 1024,
            frame_words: 1,
            max_frames: 1,
            sample_bits: 16,
            match_fraction: 0.8,
            surprise_fraction: 0.5,
            generalize: None,
            generalize_after: 1,
        }
    }
}

pub struct KernelClass<K: KernelTrait> {
    active_kernels: Vec<K>,
    temperature: KernelClassTemperature,
    target_active: Option<usize>, // homeostasis: steer temperature toward this many firing kernels per tick
    predictive: Option<PredictiveState>,
    tick: u64,
    grown: usize,
    recycled: usize,
}

/// Bookkeeping for predictive classes.
struct PredictiveState {
    cfg: GrowthConfig,
    index: Vec<Vec<u32>>,    // input bit -> kernels connected to it
    counts: Vec<u32>,        // scratch: matched bits per kernel this tick
    touched: Vec<u32>,       // scratch: kernels with a nonzero count this tick
    last_matches: Vec<usize>, // kernels at/above threshold on the last tick
    last_winner: Option<usize>,
    last_target_prob: f32,    // see `target_probability`
    last_near: Vec<usize>,    // kernels that nearly matched on the last step (see `generalize`)
    growth_mask: Option<BitVector>, // if set, new kernels may only sample these input bits
    silent_counts: std::collections::HashMap<usize, std::collections::HashMap<usize, u8>>, // kernel -> bit -> confirmations it was irrelevant
    last_hits: Vec<usize>,    // matching kernels confirmed by the last target (credit)
    last_misses: Vec<usize>,  // matching kernels contradicted by the last target (blame)
}

/// How a higher layer's expectation (`bias`) steers a predictive step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiasMode {
    /// Among candidates at the deepest matching depth, ones agreeing with the bias
    /// win (over hit rate); a deeper non-agreeing candidate still wins.
    SameDepth,
    /// Any candidate agreeing with the bias beats every candidate that doesn't.
    Prefer,
    /// Only when no kernel matches: output the bias itself.
    Fallback,
    /// Prefer, and Fallback when nothing matches.
    PreferAndFallback,
}

impl<K: KernelTrait> KernelClass<K> {
    pub fn default() -> Self {
        Self::with_kernels(Vec::new())
    }

    pub fn with_kernels(kernels: Vec<K>) -> Self {
        Self {
            active_kernels: kernels,
            temperature: KernelClassTemperature::default(),
            target_active: None,
            predictive: None,
            tick: 0,
            grown: 0,
            recycled: 0,
        }
    }

    /// Keep roughly `target` kernels firing per tick by nudging the class temperature.
    pub fn with_target_active(mut self, target: usize) -> Self {
        self.target_active = Some(target);
        self
    }

    pub fn add(&mut self, k: K) {
        self.active_kernels.push(k);
    }

    /// Run all kernels; kernels may OR/XOR/AND/CLEAR into `output`.
    /// Returns the number of kernels that fired.
    pub fn process_all(&mut self, input: &BitVector, output: &mut BitVector, phase: u16, adj_temperature: i16) -> usize {
        // One inhibit bit per output word.
        let mut inhibit = BitVector::new(output.word_len(), Some(0));
        let mut rng = rand::thread_rng();
        let temperature = self.temperature.current + adj_temperature;

        // Visit the most excited kernels first so inhibition picks the best match,
        // not whichever kernel happens to sit first in the list.
        let mut order: Vec<(isize, usize)> = {
            let ctx = KernelContext { input, inhibit: &inhibit, temperature, phase };
            self.active_kernels.iter().enumerate().map(|(i, k)| (k.excitation(&ctx), i)).collect()
        };
        order.sort_by(|a, b| b.0.cmp(&a.0));

        let mut fired = 0;
        for (_, i) in order {
            let k = &mut self.active_kernels[i];
            let ctx = KernelContext {
                input,
                inhibit: &inhibit,
                temperature,
                phase,
            };

            if k.try_fire(&ctx, output, &mut rng) {
                fired += 1;
                // Kernel fired inhibit overlapping kernels
                let (start,end) = k.word_range();
                for i in start..end {
                    inhibit.bit_set(i);
                }

            }
        }

        if let Some(target) = self.target_active {
            if fired < target {
                self.temperature.incr();
            } else if fired > target {
                self.temperature.decr();
            }
        }
        fired
    }

    pub fn kernels(&self) -> &[K] { &self.active_kernels }
    pub fn temperature(&self) -> i16 { self.temperature.current }
    pub fn grown(&self) -> usize { self.grown }
    pub fn recycled(&self) -> usize { self.recycled }
    pub fn len(&self) -> usize { self.active_kernels.len() }
    pub fn is_empty(&self) -> bool { self.active_kernels.is_empty() }
}

impl KernelClass<SimpleKernel> {
    /// An initially empty class that learns to predict its target by growing kernels.
    pub fn predictive(cfg: GrowthConfig) -> Self {
        let mut kc = Self::default();
        kc.predictive = Some(PredictiveState {
            cfg,
            index: Vec::new(),
            counts: Vec::new(),
            touched: Vec::new(),
            last_matches: Vec::new(),
            last_winner: None,
            last_target_prob: 0.0,
            last_near: Vec::new(),
            growth_mask: None,
            silent_counts: std::collections::HashMap::new(),
            last_hits: Vec::new(),
            last_misses: Vec::new(),
        });
        kc
    }

    pub fn is_predictive(&self) -> bool {
        self.predictive.is_some()
    }

    /// Run the class: predictive classes use `process_predictive`, others `process_all`.
    pub fn process(&mut self, input: &BitVector, output: &mut BitVector, phase: u16, adj_temperature: i16) -> usize {
        if self.is_predictive() {
            self.process_predictive(input, output)
        } else {
            self.process_all(input, output, phase, adj_temperature)
        }
    }

    /// Predictive step: every kernel whose matched input bits reach its threshold
    /// is a candidate; the winner is the one with the longest context, then the best
    /// hit rate, then the most matched bits. Only the winner writes its prediction.
    /// Uses an inverted index so cost scales with active input bits, not kernels.
    pub fn process_predictive(&mut self, input: &BitVector, output: &mut BitVector) -> usize {
        self.process_predictive_biased(input, output, None)
    }

    /// `process_predictive` with an optional top-down `bias`: a pattern a higher
    /// layer expects the output to be, applied according to `BiasMode`.
    pub fn process_predictive_biased(
        &mut self,
        input: &BitVector,
        output: &mut BitVector,
        bias: Option<(&BitVector, BiasMode)>,
    ) -> usize {
        let st = self.predictive.as_mut().expect("process_predictive on a non-predictive class");
        if let Some(w) = st.last_winner.take() {
            self.active_kernels[w].stats.fired_last = false;
        }
        st.last_matches.clear();
        st.last_near.clear();
        if st.counts.len() < self.active_kernels.len() {
            st.counts.resize(self.active_kernels.len(), 0);
        }

        for (wi, &w) in input.as_words().iter().enumerate() {
            let mut w = w;
            while w != 0 {
                let b = wi * 64 + w.trailing_zeros() as usize;
                w &= w - 1;
                if let Some(ks) = st.index.get(b) {
                    for &k in ks {
                        if st.counts[k as usize] == 0 {
                            st.touched.push(k);
                        }
                        st.counts[k as usize] += 1;
                    }
                }
            }
        }

        let mode = bias.map(|(_, m)| m);
        let agrees = |k: &SimpleKernel| bias.map_or(false, |(b, _)| predicts(k, b));
        // (agree-first, depth, agree-within-depth, reliability, count)
        let mut best: Option<((bool, usize, bool, f32, u32), usize)> = None;
        for &k in &st.touched {
            let k = k as usize;
            let count = st.counts[k];
            st.counts[k] = 0;
            let kern = &self.active_kernels[k];
            if (count as usize) < kern.threshold {
                if let Some(frac) = st.cfg.generalize {
                    if count as f32 >= frac * kern.input_mask.count_ones() as f32 {
                        st.last_near.push(k);
                    }
                }
                continue;
            }
            st.last_matches.push(k);
            let a = agrees(kern);
            let prefer = matches!(mode, Some(BiasMode::Prefer | BiasMode::PreferAndFallback)) && a;
            let tie = mode == Some(BiasMode::SameDepth) && a;
            let key = (prefer, kern.context_frames, tie, reliability(&kern.stats), count);
            if best.map_or(true, |(b, _)| key > b) {
                best = Some((key, k));
            }
        }
        st.touched.clear();

        let Some((_, w)) = best else {
            if let Some((b, BiasMode::Fallback | BiasMode::PreferAndFallback)) = bias {
                output.mask_mut(0, b, |a, b| a | b);
            }
            return 0;
        };
        let k = &mut self.active_kernels[w];
        output.mask_mut(k.output_idx, &k.output_mask, |a, b| a | b);
        k.stats.fired_last = true;
        k.stats.fires += 1;
        st.last_winner = Some(w);
        1
    }

    /// What the class would predict for `input`, without changing any state
    /// (no stats, no winner bookkeeping). Returns the winning kernel's output
    /// pattern, or None if no kernel matches. Used for counterfactual
    /// (ablation) credit: compare the prediction with and without some inputs.
    pub fn peek(&self, input: &BitVector) -> Option<&BitVector> {
        self.peek_scored(input).map(|(out, _)| out)
    }

    /// Like `peek`, also returning the winning kernel's smoothed hit rate.
    pub fn peek_scored(&self, input: &BitVector) -> Option<(&BitVector, f32)> {
        let st = self.predictive.as_ref()?;
        let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for b in set_bits(input) {
            if let Some(ks) = st.index.get(b) {
                for &k in ks {
                    *counts.entry(k).or_default() += 1;
                }
            }
        }
        counts
            .into_iter()
            .filter(|&(k, c)| c as usize >= self.active_kernels[k as usize].threshold)
            .map(|(k, c)| {
                let kern = &self.active_kernels[k as usize];
                ((kern.context_frames, reliability(&kern.stats), c), k as usize)
            })
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)))
            .map(|((_, rel, _), k)| (&self.active_kernels[k].output_mask, rel))
    }

    /// Restrict which input bits newly grown kernels may sample (None = all).
    /// A caller can use it for credit-guided growth: e.g. let new kernels depend
    /// on one memory unit at a time, chosen by that unit's credit.
    pub fn set_growth_mask(&mut self, mask: Option<BitVector>) {
        if let Some(st) = self.predictive.as_mut() {
            st.growth_mask = mask;
        }
    }

    /// Credit assignment readout: the input bits that matching kernels relied on
    /// when they predicted the last target correctly (`bits` = input width).
    /// A layer feeding this class can use it to learn which of its outputs are useful.
    pub fn credited_inputs(&self, bits: usize) -> BitVector {
        self.union_of_masks(bits, |st| &st.last_hits)
    }

    /// Like `credited_inputs`, for kernels whose prediction the last target contradicted.
    pub fn blamed_inputs(&self, bits: usize) -> BitVector {
        self.union_of_masks(bits, |st| &st.last_misses)
    }

    fn union_of_masks(&self, bits: usize, pick: impl Fn(&PredictiveState) -> &Vec<usize>) -> BitVector {
        let mut out = BitVector::new(bits, Some(0));
        if let Some(st) = self.predictive.as_ref() {
            for &k in pick(st) {
                let m = &self.active_kernels[k].input_mask;
                let n = out.word_len().min(m.word_len());
                for (o, &w) in out.as_words_mut()[..n].iter_mut().zip(&m.as_words()[..n]) {
                    *o |= w;
                }
            }
        }
        out
    }

    /// Estimated probability that the current prediction is right (the winning
    /// kernel's smoothed hit rate), or None if nothing was predicted.
    pub fn confidence(&self) -> Option<f32> {
        let st = self.predictive.as_ref()?;
        st.last_winner.map(|w| reliability(&self.active_kernels[w].stats))
    }

    /// How expected the last target was: the hit rate (before this update) of the
    /// longest-context matching kernel that predicted it, or 0 if none did.
    /// Low values mark surprising transitions (e.g. the start of a new word).
    pub fn target_probability(&self) -> f32 {
        self.predictive.as_ref().map_or(0.0, |st| st.last_target_prob)
    }

    /// The kernel that made the last prediction, if any.
    pub fn winner(&self) -> Option<&SimpleKernel> {
        let st = self.predictive.as_ref()?;
        st.last_winner.map(|w| &self.active_kernels[w])
    }

    /// Context depth (frames) of the current winning kernel, if any.
    pub fn winner_depth(&self) -> Option<usize> {
        let st = self.predictive.as_ref()?;
        st.last_winner.map(|w| self.active_kernels[w].context_frames)
    }

    /// Local learning from what actually happened next.
    ///
    /// `input` is the input the class saw on its last step, and `target` is the
    /// pattern its output should have predicted. Only predictive classes learn here:
    /// - every kernel that matched is scored a hit if the target confirms at least
    ///   half of its output bits, otherwise a miss (so each context keeps a hit rate
    ///   per continuation, and the most reliable continuation wins);
    /// - when the winner was wrong (or nothing matched) and too much of the target
    ///   went unpredicted, grow a kernel at the winner's depth that outputs the
    ///   target (unless a matching one already does), and if a kernel fired wrongly,
    ///   another spanning one more history frame, so longer contexts can override
    ///   shorter ones that keep being wrong;
    /// - at the kernel budget, the least-recently-useful kernel is recycled.
    pub fn feedback<R: rand::Rng + ?Sized>(&mut self, input: &BitVector, target: &BitVector, rng: &mut R) {
        let Some(st) = self.predictive.as_ref() else { return };
        let cfg = st.cfg;
        self.tick += 1;
        let target_bits = target.count_ones();
        if target_bits == 0 {
            return;
        }

        let winner = st.last_winner;
        let matches = st.last_matches.clone();
        let mut depth_has_target = vec![false; cfg.max_frames + 2];
        let mut expected: Option<(usize, f32)> = None;
        for &m in &matches {
            let k = &self.active_kernels[m];
            if predicts(k, target) {
                let key = (k.context_frames, reliability(&k.stats));
                if expected.map_or(true, |e| key > e) {
                    expected = Some(key);
                }
            }
        }
        let mut hits = Vec::new();
        let mut misses = Vec::new();
        for &m in &matches {
            let k = &mut self.active_kernels[m];
            if predicts(k, target) {
                k.stats.hits += 1;
                k.stats.last_useful = self.tick;
                depth_has_target[k.context_frames] = true;
                hits.push(m);
            } else {
                k.stats.misses += 1;
                misses.push(m);
            }
        }
        if let Some(st) = self.predictive.as_mut() {
            st.last_target_prob = expected.map_or(0.0, |e| e.1);
            st.last_hits = hits;
            st.last_misses = misses;
        }
        if cfg.generalize.is_some() {
            self.generalize_near_misses(input, target, &cfg);
        }

        let unpredicted = match winner {
            Some(w) => {
                let k = &self.active_kernels[w];
                target_bits - target.mask_and_count(k.output_idx, &k.output_mask, |a, m| a & m)
            }
            None => target_bits,
        };
        if (unpredicted as f32) <= cfg.surprise_fraction * target_bits as f32 {
            return;
        }

        let depth = winner.map_or(1, |w| self.active_kernels[w].context_frames);
        if !depth_has_target[depth] {
            self.grow(input, target, depth, rng);
        }
        // Grow one frame deeper only if that frame carries something new.
        if winner.is_some() && depth < cfg.max_frames && frame_has_bits(input, depth, cfg.frame_words) {
            self.grow(input, target, depth + 1, rng);
        }
    }

    /// Synapse-level credit (see `GrowthConfig::generalize`): near-matching kernels
    /// whose prediction the target confirms keep only the connections that were
    /// active, so inputs that didn't matter stop being required.
    fn generalize_near_misses(&mut self, input: &BitVector, target: &BitVector, cfg: &GrowthConfig) {
        let Some(st) = self.predictive.as_mut() else { return };
        let near = std::mem::take(&mut st.last_near);
        let min_bits = cfg.sample_bits.max(2);
        let need = cfg.generalize_after.max(1);
        // A connection that is active while its kernel is right has proven compatible.
        for &k in &st.last_hits {
            if let Some(counts) = st.silent_counts.get_mut(&k) {
                counts.retain(|&b, _| !input.bit_get(b));
            }
        }
        for k in near {
            if !predicts(&self.active_kernels[k], target) {
                continue;
            }
            let old = set_bits(&self.active_kernels[k].input_mask);
            let counts = st.silent_counts.entry(k).or_default();
            let mut drop = Vec::new();
            for &b in &old {
                if input.bit_get(b) {
                    counts.remove(&b);
                } else {
                    let c = counts.entry(b).or_insert(0);
                    *c = c.saturating_add(1);
                    if *c >= need {
                        drop.push(b);
                    }
                }
            }
            if drop.is_empty() || old.len() - drop.len() < min_bits {
                continue;
            }
            for &b in &drop {
                self.active_kernels[k].input_mask.bit_clear(b);
                st.index[b].retain(|&x| x as usize != k);
                counts.remove(&b);
            }
            let tolerance = (cfg.sample_bits as f32 * (1.0 - cfg.match_fraction)).floor() as usize;
            let kern = &mut self.active_kernels[k];
            kern.threshold = (old.len() - drop.len()).saturating_sub(tolerance).max(1);
            kern.stats.hits += 1;
            kern.stats.last_useful = self.tick;
        }
    }

    /// Grow (or, at the budget, recycle) a kernel that recognises a sample of the
    /// active bits in the `depth` most recent frames of `input` and outputs `target`.
    pub fn grow<R: rand::Rng + ?Sized>(&mut self, input: &BitVector, target: &BitVector, depth: usize, rng: &mut R) {
        let tick = self.tick;
        let Some(st) = self.predictive.as_mut() else { return };
        let cfg = st.cfg;

        let mut input_mask = BitVector::new(input.bit_len(), Some(0));
        let mut sampled = 0;
        let mut reach = 0; // frames back the kernel actually connects to
        for f in 0..depth {
            let first = f * cfg.frame_words * 64;
            let last = ((f + 1) * cfg.frame_words * 64).min(input.bit_len());
            let allowed = |b: usize| st.growth_mask.as_ref().map_or(true, |m| b < m.bit_len() && m.bit_get(b));
            let active: Vec<usize> = (first..last).filter(|&b| input.bit_get(b) && allowed(b)).collect();
            for &b in active.choose_multiple(rng, cfg.sample_bits) {
                input_mask.bit_set(b);
                sampled += 1;
                reach = f + 1;
            }
        }
        if sampled == 0 {
            return;
        }
        // Tolerate one frame's worth of noise (not a fraction of all sampled bits),
        // so a long-context kernel can't fire when a whole frame is different.
        let tolerance = (sampled.min(cfg.sample_bits) as f32 * (1.0 - cfg.match_fraction)).floor() as usize;
        let threshold = sampled - tolerance;

        let mut k = SimpleKernel::new(
            input_mask,
            0,
            target.clone(),
            0,
            threshold,
            KernelOp::Or,
        );
        k.plastic = false;
        // Depth is how far back the connections really reach: a kernel grown over
        // empty frames is no more specific than a shallower one and must not
        // outrank (or block the growth of) kernels that use those frames later.
        k.context_frames = reach;
        k.stats.last_useful = tick;

        let slot = if self.active_kernels.len() < cfg.max_kernels {
            self.active_kernels.push(k);
            self.grown += 1;
            self.active_kernels.len() - 1
        } else {
            let victim = (0..self.active_kernels.len())
                .filter(|&i| Some(i) != st.last_winner)
                .min_by_key(|&i| self.active_kernels[i].stats.last_useful)
                .expect("budget must allow at least two kernels");
            for b in set_bits(&self.active_kernels[victim].input_mask) {
                st.index[b].retain(|&x| x as usize != victim);
            }
            st.last_matches.retain(|&x| x != victim);
            st.last_hits.retain(|&x| x != victim);
            st.silent_counts.remove(&victim);
            st.last_misses.retain(|&x| x != victim);
            self.active_kernels[victim] = k;
            self.recycled += 1;
            victim
        };

        if st.index.len() < input.bit_len() {
            st.index.resize(input.bit_len(), Vec::new());
        }
        for b in set_bits(&self.active_kernels[slot].input_mask) {
            st.index[b].push(slot as u32);
        }
    }
}

/// Whether frame `f` (0 = most recent) of a concatenated input has any active bit.
fn frame_has_bits(input: &BitVector, f: usize, frame_words: usize) -> bool {
    input.as_words().iter().skip(f * frame_words).take(frame_words).any(|&w| w != 0)
}

/// Smoothed hit rate of a predictive kernel.
fn reliability(s: &KernelStats) -> f32 {
    (s.hits as f32 + 1.0) / ((s.hits + s.misses) as f32 + 2.0)
}

/// True if `target` confirms at least half of `k`'s output bits.
fn predicts(k: &SimpleKernel, target: &BitVector) -> bool {
    target.mask_and_count(k.output_idx, &k.output_mask, |a, m| a & m) * 2 >= k.output_mask.count_ones()
}

fn set_bits(bv: &BitVector) -> Vec<usize> {
    let mut out = Vec::new();
    for (wi, &w) in bv.as_words().iter().enumerate() {
        let mut w = w;
        while w != 0 {
            out.push(wi * 64 + w.trailing_zeros() as usize);
            w &= w - 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel(in_bits: &[usize], out_bit: usize, threshold: usize) -> SimpleKernel {
        SimpleKernel::new(BitVector::from_bits(in_bits, 64), 0, BitVector::from_bits(&[out_bit], 64), 0, threshold, KernelOp::Or)
    }

    #[test]
    fn strongest_kernel_wins_inhibition_regardless_of_order() {
        // both write output word 0; the second matches 8 bits vs the first's 4
        let weak = kernel(&[0, 1, 2, 3, 40, 41, 42, 43], 1, 4);
        let strong = kernel(&[0, 1, 2, 3, 4, 5, 6, 7], 2, 4);
        let mut kc = KernelClass::with_kernels(vec![weak, strong]);
        let input = BitVector::from_words(vec![0xFF]);
        let mut out = BitVector::new(64, Some(0));
        assert_eq!(kc.process_all(&input, &mut out, 0, 0), 1);
        assert!(out.bit_get(2) && !out.bit_get(1));
    }

    #[test]
    fn homeostasis_raises_temperature_when_quiet() {
        let mut kc = KernelClass::with_kernels(vec![kernel(&[0, 1, 2, 3, 4, 5, 6, 7], 0, 8)]).with_target_active(1);
        let input = BitVector::from_words(vec![0x0F]); // only 4 of 8 bits
        let mut out = BitVector::new(64, Some(0));
        let t0 = kc.temperature();
        for _ in 0..4 {
            kc.process_all(&input, &mut out, 0, 0);
        }
        assert!(kc.temperature() > t0);
        // threshold 8 - temperature 4 = 4 matched bits -> now fires
        out.bit_clear_all();
        assert_eq!(kc.process_all(&input, &mut out, 0, 0), 1);
    }

    fn frames(words: &[u64]) -> BitVector {
        BitVector::from_words(words.to_vec())
    }

    /// One predictive step: run, read the prediction, learn from `target`.
    fn step(kc: &mut KernelClass<SimpleKernel>, ctx: &BitVector, target: &BitVector) -> u64 {
        let mut rng = rand::thread_rng();
        let mut out = BitVector::new(64, Some(0));
        kc.process(ctx, &mut out, 0, 0);
        kc.feedback(ctx, target, &mut rng);
        out.as_words()[0]
    }

    #[test]
    fn predictive_class_grows_then_predicts() {
        let cfg = GrowthConfig { frame_words: 1, max_frames: 2, sample_bits: 8, ..GrowthConfig::default() };
        let mut kc = KernelClass::predictive(cfg);
        let ctx = frames(&[0xFF, 0]); // A at t, nothing at t-1
        let target = BitVector::from_words(vec![0xFF00]); // B

        assert_eq!(step(&mut kc, &ctx, &target), 0); // surprise -> grow
        assert_eq!((kc.len(), kc.grown()), (1, 1));
        assert_eq!(kc.kernels()[0].context_frames, 1);

        assert_eq!(step(&mut kc, &ctx, &target), 0xFF00); // correct -> no growth
        assert_eq!(kc.len(), 1);
        assert_eq!(kc.kernels()[0].stats.hits, 1);
        assert!(kc.confidence().unwrap() > 0.5);
    }

    #[test]
    fn longer_context_overrides_shorter_one() {
        let cfg = GrowthConfig { frame_words: 1, max_frames: 2, sample_bits: 8, ..GrowthConfig::default() };
        let mut kc = KernelClass::predictive(cfg);
        let b = BitVector::from_words(vec![0xFF00]);
        let c = BitVector::from_words(vec![0xFF0000]);
        // "xA -> B" and "yA -> C": same current frame, different previous frame
        let xa = frames(&[0xFF, 0xF000_0000_0000_0000]);
        let ya = frames(&[0xFF, 0x0F00_0000_0000_0000]);

        for _ in 0..3 {
            step(&mut kc, &xa, &b);
            step(&mut kc, &ya, &c);
        }
        assert!(kc.kernels().iter().any(|k| k.context_frames == 2));
        assert_eq!(step(&mut kc, &ya, &c), 0xFF0000);
        assert_eq!(step(&mut kc, &xa, &b), 0xFF00);
        assert_eq!(kc.winner_depth(), Some(2));
    }

    #[test]
    fn most_reliable_continuation_wins_at_same_depth() {
        let cfg = GrowthConfig { frame_words: 1, max_frames: 1, sample_bits: 8, ..GrowthConfig::default() };
        let mut kc = KernelClass::predictive(cfg);
        let a = frames(&[0xFF]);
        let b = BitVector::from_words(vec![0xFF00]);
        let c = BitVector::from_words(vec![0xFF0000]);
        step(&mut kc, &a, &b); // grows A->B
        assert_eq!(step(&mut kc, &a, &c), 0xFF00); // wrong -> grows sibling A->C
        assert_eq!(kc.len(), 2);
        step(&mut kc, &a, &c);
        // A was followed by C twice and B once: C wins, B is kept as an alternative
        assert_eq!(step(&mut kc, &a, &b), 0xFF0000);
        assert_eq!(kc.len(), 2);
    }

    #[test]
    fn budget_recycles_least_recently_useful() {
        let cfg = GrowthConfig { max_kernels: 2, frame_words: 1, max_frames: 1, sample_bits: 8, ..GrowthConfig::default() };
        let mut kc = KernelClass::predictive(cfg);
        for i in 0..3u64 {
            step(&mut kc, &frames(&[0xFF << (8 * i)]), &BitVector::from_words(vec![1 << i]));
        }
        assert_eq!((kc.len(), kc.grown(), kc.recycled()), (2, 2, 1));
        // the first kernel (oldest, never useful since) was replaced, and the
        // index no longer routes its old input to the new occupant
        let outs: Vec<u64> = kc.kernels().iter().map(|k| k.output_mask.as_words()[0]).collect();
        assert!(!outs.contains(&1) && outs.contains(&2) && outs.contains(&4));
        assert_eq!(step(&mut kc, &frames(&[0xFF0000]), &BitVector::from_words(vec![4])), 4);
        assert_eq!(step(&mut kc, &frames(&[0xFF]), &BitVector::from_words(vec![8])), 0);
    }

    #[test]
    fn target_probability_tracks_how_expected_the_target_was() {
        let cfg = GrowthConfig { frame_words: 1, max_frames: 1, sample_bits: 8, ..GrowthConfig::default() };
        let mut kc = KernelClass::predictive(cfg);
        let a = frames(&[0xFF]);
        let b = BitVector::from_words(vec![0xFF00]);
        let c = BitVector::from_words(vec![0xFF0000]);
        step(&mut kc, &a, &b);
        assert_eq!(kc.target_probability(), 0.0); // nothing predicted B yet
        for _ in 0..4 {
            step(&mut kc, &a, &b);
        }
        assert!(kc.target_probability() > 0.7); // A->B is well established
        step(&mut kc, &a, &c);
        assert_eq!(kc.target_probability(), 0.0); // C after A is a surprise
    }

    #[test]
    fn bias_modes_steer_or_fill_in_the_prediction() {
        let cfg = GrowthConfig { frame_words: 1, max_frames: 1, sample_bits: 8, ..GrowthConfig::default() };
        let mut kc = KernelClass::predictive(cfg);
        let a = frames(&[0xFF]);
        let b = BitVector::from_words(vec![0xFF00]);
        let c = BitVector::from_words(vec![0xFF0000]);
        // A->B twice, A->C once: unbiased, B wins
        for t in [&b, &b, &c, &b] {
            step(&mut kc, &a, t);
        }
        let run = |kc: &mut KernelClass<SimpleKernel>, input: &BitVector, mode: BiasMode| {
            let mut out = BitVector::new(64, Some(0));
            kc.process_predictive_biased(input, &mut out, Some((&c, mode)));
            out.as_words()[0]
        };
        assert_eq!(run(&mut kc, &a, BiasMode::SameDepth), 0xFF0000); // C agrees, same depth as B
        assert_eq!(run(&mut kc, &a, BiasMode::Prefer), 0xFF0000); // a candidate agrees with C
        assert_eq!(run(&mut kc, &a, BiasMode::Fallback), 0xFF00); // something matched: no fallback
        let unseen = frames(&[0xFF00_0000]);
        assert_eq!(run(&mut kc, &unseen, BiasMode::Fallback), 0xFF0000); // nothing matched: use the bias
        assert_eq!(run(&mut kc, &unseen, BiasMode::Prefer), 0);
    }

    #[test]
    fn credit_points_at_the_inputs_correct_kernels_used() {
        let cfg = GrowthConfig { frame_words: 1, max_frames: 1, sample_bits: 8, ..GrowthConfig::default() };
        let mut kc = KernelClass::predictive(cfg);
        let a = frames(&[0xFF]);
        let b = BitVector::from_words(vec![0xFF00]);
        let c = BitVector::from_words(vec![0xFF0000]);
        step(&mut kc, &a, &b); // grow A->B
        step(&mut kc, &a, &b); // hit
        assert_eq!(kc.credited_inputs(64).as_words()[0], 0xFF);
        assert_eq!(kc.blamed_inputs(64).count_ones(), 0);
        step(&mut kc, &a, &c); // miss
        assert_eq!(kc.credited_inputs(64).count_ones(), 0);
        assert_eq!(kc.blamed_inputs(64).as_words()[0], 0xFF);
    }

    #[test]
    fn near_miss_that_would_be_right_drops_irrelevant_connections() {
        let cfg = GrowthConfig { frame_words: 1, max_frames: 2, sample_bits: 8, generalize: Some(0.5), ..GrowthConfig::default() };
        let b = BitVector::from_words(vec![0xFF00]);
        // frame 0 = cue A (relevant), frame 1 = a filler that varies (irrelevant)
        let mut kc2 = KernelClass::predictive(cfg);
        let mut rng = rand::thread_rng();
        kc2.grow(&frames(&[0xFF, 0x0F]), &b, 2, &mut rng); // A + filler1 -> B
        let a_f2 = frames(&[0xFF, 0xF0]); // A + filler2: only frame 0 matches
        let mut out = BitVector::new(64, Some(0));
        kc2.process(&a_f2, &mut out, 0, 0);
        assert_eq!(out.count_ones(), 0); // too specific: no match
        kc2.feedback(&a_f2, &b, &mut rng); // but it would have been right -> generalize
        assert_eq!(kc2.kernels()[0].input_mask.as_words(), &[0xFF, 0]);
        out.bit_clear_all();
        kc2.process(&frames(&[0xFF, 0xF000]), &mut out, 0, 0); // any filler now
        assert_eq!(out.as_words()[0], 0xFF00);
    }

    #[test]
    fn peek_predicts_without_changing_state() {
        let cfg = GrowthConfig { frame_words: 1, max_frames: 1, sample_bits: 8, ..GrowthConfig::default() };
        let mut kc = KernelClass::predictive(cfg);
        let a = frames(&[0xFF]);
        let b = BitVector::from_words(vec![0xFF00]);
        step(&mut kc, &a, &b);
        step(&mut kc, &a, &b);
        let hits = kc.kernels()[0].stats.hits;
        assert_eq!(kc.peek(&a).map(|o| o.as_words()[0]), Some(0xFF00));
        assert!(kc.peek(&frames(&[0xF0])).is_none()); // only half the bits: no match
        assert_eq!(kc.kernels()[0].stats.hits, hits);
        assert_eq!(kc.kernels()[0].stats.fires, 1); // only the real step counted
    }

    #[test]
    fn growth_mask_limits_what_new_kernels_sample() {
        let cfg = GrowthConfig { frame_words: 1, max_frames: 1, sample_bits: 8, ..GrowthConfig::default() };
        let mut kc = KernelClass::predictive(cfg);
        kc.set_growth_mask(Some(BitVector::from_words(vec![0x0F])));
        step(&mut kc, &frames(&[0xFF]), &BitVector::from_words(vec![0xFF00]));
        assert_eq!(kc.kernels()[0].input_mask.as_words()[0], 0x0F);
    }

    #[test]
    fn gradual_generalization_needs_repeated_evidence() {
        let cfg = GrowthConfig {
            frame_words: 1,
            max_frames: 2,
            sample_bits: 8,
            generalize: Some(0.5),
            generalize_after: 2,
            ..GrowthConfig::default()
        };
        let b = BitVector::from_words(vec![0xFF00]);
        let mut kc = KernelClass::predictive(cfg);
        let mut rng = rand::thread_rng();
        kc.grow(&frames(&[0xFF, 0x0F]), &b, 2, &mut rng); // A + filler1 -> B
        let a_f2 = frames(&[0xFF, 0xF0]);
        step(&mut kc, &a_f2, &b); // first confirmation: not yet dropped
        assert_eq!(kc.kernels()[0].input_mask.as_words(), &[0xFF, 0x0F]);
        step(&mut kc, &a_f2, &b); // second: filler connections dropped
        assert_eq!(kc.kernels()[0].input_mask.as_words(), &[0xFF, 0]);
    }

    #[test]
    fn kernels_grown_over_empty_frames_do_not_block_later_context() {
        // frame 0 = cue "?", frame 1 = a relay that starts out empty
        let cfg = GrowthConfig { frame_words: 1, max_frames: 2, sample_bits: 8, ..GrowthConfig::default() };
        let mut kc = KernelClass::predictive(cfg);
        let b = BitVector::from_words(vec![0xFF00]);
        let c = BitVector::from_words(vec![0xFF0000]);
        // relay empty: answers vary, so only "?"-only kernels can be grown
        for t in [&b, &c, &b, &c] {
            step(&mut kc, &frames(&[0xFF, 0]), t);
        }
        assert!(kc.kernels().iter().all(|k| k.context_frames == 1));
        // relay now carries the answer: relay-specific kernels must be able to grow and win
        for _ in 0..3 {
            step(&mut kc, &frames(&[0xFF, 0x0F]), &b);
            step(&mut kc, &frames(&[0xFF, 0xF0]), &c);
        }
        assert_eq!(step(&mut kc, &frames(&[0xFF, 0x0F]), &b), 0xFF00);
        assert_eq!(step(&mut kc, &frames(&[0xFF, 0xF0]), &c), 0xFF0000);
    }
}