use crate::bitvec::BitVector;
use crate::common::config;
use crate::kernel::simple::SimpleKernel;
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
    pub match_fraction: f32,     // new kernel threshold = ceil(sampled * match_fraction)
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
    growth: Option<GrowthConfig>,
    tick: u64,
    grown: usize,
    recycled: usize,
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
            growth: None,
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
        kc.growth = Some(cfg);
        kc
    }

    /// Local learning from what actually happened next.
    ///
    /// `input` is the input the class saw on its last `process_all`, and `target`
    /// is the pattern that output should have predicted. Only predictive classes
    /// (built with `predictive`) learn here:
    /// - a kernel that fired is scored a hit if the target confirmed at least half
    ///   of its output bits, otherwise a miss;
    /// - when too much of the target went unpredicted (surprise), a kernel is grown
    ///   from a sample of the active input bits, outputting the target. If a kernel
    ///   fired wrongly, the new one spans one more history frame than it did, so a
    ///   longer context can override a shorter one that keeps being wrong;
    /// - at the deepest allowed context, a kernel that misses more than it hits
    ///   (by two or more) is retargeted to the latest target instead;
    /// - at the kernel budget, the least-recently-useful kernel is recycled.
    pub fn feedback<R: rand::Rng + ?Sized>(&mut self, input: &BitVector, target: &BitVector, rng: &mut R) {
        let Some(cfg) = self.growth else { return };
        self.tick += 1;
        let target_bits = target.count_ones();
        if target_bits == 0 {
            return;
        }

        let mut predicted = BitVector::new(target.bit_len(), Some(0));
        let mut deepest_wrong: Option<usize> = None;
        for k in self.active_kernels.iter_mut().filter(|k| k.stats.fired_last) {
            let confirmed = target.mask_and_count(k.output_idx, &k.output_mask, |a, m| a & m);
            if confirmed * 2 >= k.output_mask.count_ones() {
                k.stats.hits += 1;
                k.stats.last_useful = self.tick;
            } else {
                k.stats.misses += 1;
                deepest_wrong = Some(deepest_wrong.map_or(k.context_frames, |d| d.max(k.context_frames)));
            }
            predicted.mask_mut(k.output_idx, &k.output_mask, |a, b| a | b);
        }

        let unpredicted = target_bits - target.mask_and_count(0, &predicted, |a, m| a & m);
        if (unpredicted as f32) <= cfg.surprise_fraction * target_bits as f32 {
            return;
        }

        let depth = deepest_wrong.map_or(1, |d| d + 1);
        if depth > cfg.max_frames {
            // Context window exhausted: nothing longer to grow. A kernel at the
            // limit that is wrong clearly more often than right switches its prediction to
            // the latest target instead, so it tracks the common continuation.
            for k in self.active_kernels.iter_mut() {
                if k.stats.fired_last && k.context_frames == cfg.max_frames && k.stats.misses >= k.stats.hits + 2 {
                    k.output_mask = target.clone();
                    k.output_idx = 0;
                    k.stats.hits = 0;
                    k.stats.misses = 0;
                    k.stats.last_useful = self.tick;
                }
            }
            return;
        }

        // Sample active input bits from the `depth` most recent frames.
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

        let mut k = SimpleKernel::new(
            input_mask,
            0,
            target.clone(),
            0,
            (sampled as f32 * cfg.match_fraction).ceil() as usize,
            KernelOp::Or,
        );
        k.plastic = false;
        k.context_frames = depth;
        k.stats.last_useful = self.tick;

        if self.active_kernels.len() < cfg.max_kernels {
            self.active_kernels.push(k);
            self.grown += 1;
        } else if let Some(victim) = self
            .active_kernels
            .iter()
            .enumerate()
            .min_by_key(|(_, k)| k.stats.last_useful)
            .map(|(i, _)| i)
        {
            self.active_kernels[victim] = k;
            self.recycled += 1;
        }
    }
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

    #[test]
    fn predictive_class_grows_then_predicts() {
        let cfg = GrowthConfig { frame_words: 1, max_frames: 2, sample_bits: 8, ..GrowthConfig::default() };
        let mut kc = KernelClass::predictive(cfg);
        let mut rng = rand::thread_rng();
        let ctx = frames(&[0xFF, 0]); // A at t, nothing at t-1
        let target = BitVector::from_words(vec![0xFF00]); // B

        let mut out = BitVector::new(64, Some(0));
        kc.process_all(&ctx, &mut out, 0, 0);
        assert_eq!(out.count_ones(), 0);
        kc.feedback(&ctx, &target, &mut rng); // surprise -> grow
        assert_eq!((kc.len(), kc.grown()), (1, 1));
        assert_eq!(kc.kernels()[0].context_frames, 1);

        out.bit_clear_all();
        kc.process_all(&ctx, &mut out, 0, 0);
        assert_eq!(out.as_words()[0], 0xFF00);
        kc.feedback(&ctx, &target, &mut rng); // correct -> no growth
        assert_eq!(kc.len(), 1);
        assert_eq!(kc.kernels()[0].stats.hits, 1);
    }

    #[test]
    fn wrong_prediction_grows_deeper_context_that_wins() {
        let cfg = GrowthConfig { frame_words: 1, max_frames: 2, sample_bits: 8, ..GrowthConfig::default() };
        let mut kc = KernelClass::predictive(cfg);
        let mut rng = rand::thread_rng();
        let b = BitVector::from_words(vec![0xFF00]);
        let c = BitVector::from_words(vec![0xFF0000]);
        // "xA -> B" and "yA -> C": same current frame, different previous frame
        let xa = frames(&[0xFF, 0xF000_0000_0000_0000]);
        let ya = frames(&[0xFF, 0x0F00_0000_0000_0000]);

        let mut run = |kc: &mut KernelClass<SimpleKernel>, ctx: &BitVector, target: &BitVector| {
            let mut out = BitVector::new(64, Some(0));
            kc.process_all(ctx, &mut out, 0, 0);
            kc.feedback(ctx, target, &mut rng);
            out
        };
        run(&mut kc, &xa, &b); // grows A->B (1 frame)
        run(&mut kc, &ya, &c); // A->B fires wrongly -> grows yA->C (2 frames)
        assert_eq!(kc.kernels()[1].context_frames, 2);

        assert_eq!(run(&mut kc, &ya, &c).as_words()[0], 0xFF0000);
        assert_eq!(run(&mut kc, &xa, &b).as_words()[0], 0xFF00);
    }

    #[test]
    fn budget_recycles_least_recently_useful() {
        let cfg = GrowthConfig { max_kernels: 2, frame_words: 1, max_frames: 1, sample_bits: 8, ..GrowthConfig::default() };
        let mut kc = KernelClass::predictive(cfg);
        let mut rng = rand::thread_rng();
        for i in 0..3u64 {
            let ctx = frames(&[0xFF << (8 * i)]);
            let mut out = BitVector::new(64, Some(0));
            kc.process_all(&ctx, &mut out, 0, 0);
            kc.feedback(&ctx, &BitVector::from_words(vec![1 << i]), &mut rng);
        }
        assert_eq!((kc.len(), kc.grown(), kc.recycled()), (2, 2, 1));
        // the first kernel (oldest, never useful since) was replaced
        let outs: Vec<u64> = kc.kernels().iter().map(|k| k.output_mask.as_words()[0]).collect();
        assert!(!outs.contains(&1) && outs.contains(&2) && outs.contains(&4));
    }

    #[test]
    fn kernel_at_context_limit_retargets_after_mostly_missing() {
        let cfg = GrowthConfig { frame_words: 1, max_frames: 1, sample_bits: 8, ..GrowthConfig::default() };
        let mut kc = KernelClass::predictive(cfg);
        let mut rng = rand::thread_rng();
        let a = frames(&[0xFF]);
        let b = BitVector::from_words(vec![0xFF00]);
        let c = BitVector::from_words(vec![0xFF0000]);
        let mut step = |kc: &mut KernelClass<SimpleKernel>, target: &BitVector| {
            let mut out = BitVector::new(64, Some(0));
            kc.process_all(&a, &mut out, 0, 0);
            kc.feedback(&a, target, &mut rng);
            out.as_words()[0]
        };
        step(&mut kc, &b); // grows A->B
        assert_eq!(step(&mut kc, &c), 0xFF00); // miss 1: one miss is not enough
        assert_eq!(step(&mut kc, &c), 0xFF00); // miss 2 (misses 2 >= hits 0 + 2) -> retarget to C
        assert_eq!(step(&mut kc, &c), 0xFF0000);
        assert_eq!(kc.len(), 1);
    }
}