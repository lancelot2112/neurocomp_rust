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
        let st = self.predictive.as_mut().expect("process_predictive on a non-predictive class");
        if let Some(w) = st.last_winner.take() {
            self.active_kernels[w].stats.fired_last = false;
        }
        st.last_matches.clear();
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

        let mut best: Option<(usize, f32, u32, usize)> = None; // (depth, reliability, count, kernel)
        for &k in &st.touched {
            let k = k as usize;
            let count = st.counts[k];
            st.counts[k] = 0;
            let kern = &self.active_kernels[k];
            if (count as usize) < kern.threshold {
                continue;
            }
            st.last_matches.push(k);
            let key = (kern.context_frames, reliability(&kern.stats), count, k);
            let better = match best {
                None => true,
                Some(b) => (key.0, key.1, key.2) > (b.0, b.1, b.2),
            };
            if better {
                best = Some(key);
            }
        }
        st.touched.clear();

        let Some((_, _, _, w)) = best else { return 0 };
        let k = &mut self.active_kernels[w];
        output.mask_mut(k.output_idx, &k.output_mask, |a, b| a | b);
        k.stats.fired_last = true;
        k.stats.fires += 1;
        st.last_winner = Some(w);
        1
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
        if let Some(st) = self.predictive.as_mut() {
            st.last_target_prob = expected.map_or(0.0, |e| e.1);
        }
        for &m in &matches {
            let k = &mut self.active_kernels[m];
            if predicts(k, target) {
                k.stats.hits += 1;
                k.stats.last_useful = self.tick;
                depth_has_target[k.context_frames] = true;
            } else {
                k.stats.misses += 1;
            }
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
        if winner.is_some() && depth < cfg.max_frames {
            self.grow(input, target, depth + 1, rng);
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
        for f in 0..depth {
            let first = f * cfg.frame_words * 64;
            let last = ((f + 1) * cfg.frame_words * 64).min(input.bit_len());
            let active: Vec<usize> = (first..last).filter(|&b| input.bit_get(b)).collect();
            for &b in active.choose_multiple(rng, cfg.sample_bits) {
                input_mask.bit_set(b);
                sampled += 1;
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
        k.context_frames = depth;
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
}