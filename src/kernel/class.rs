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
    silent_counts: crate::det::HashMap<usize, crate::det::HashMap<usize, u8>>, // kernel -> bit -> confirmations it was irrelevant
    /// Sticky synapses (credit tags): kernel -> input bits that earned credit; pruning
    /// them takes `sticky_factor` times as many silent confirmations. 0 = off.
    sticky_factor: u8,
    sticky_tags: crate::det::HashMap<usize, crate::det::HashSet<usize>>,
    /// Caller-provided credit: input bits that are always sticky.
    sticky_mask: Option<BitVector>,
    /// Credit-guided growth (see `grow`).
    copy_growth: bool,
    /// Bad credit releases tags: a tagged bit that was active when its kernel fired and
    /// the target did not contain the bit it copies loses its tag (then prunes normally).
    sticky_blame: bool,
    /// Winner ranking: a kernel's depth only counts if its reliability is at least this
    /// (deeper-but-unreliable kernels no longer outrank reliable shallower ones). None = off.
    trust_floor: Option<Rate>,
    /// Growth: only kernels at least this reliable count as "this depth already
    /// predicts the target" (None = any kernel does).
    growth_trust: Option<Rate>,
    last_hits: Vec<usize>,    // matching kernels confirmed by the last target (credit)
    last_misses: Vec<usize>,  // matching kernels contradicted by the last target (blame)
    /// Fast inhibitory loop (see `set_fast_inhibition`). None = off.
    fast: Option<FastInhibition>,
    /// Uncertainty-gated growth (see `set_growth_gate`): (shift k, minimum evidence).
    growth_gate: Option<(u32, u16)>,
    /// Growth events the gate suppressed.
    gated: usize,
    /// Slots of kernels removed by `sleep`, reused by growth.
    free: Vec<usize>,
    /// Per-frame memo of match counts (see `set_frame_memo`).
    frame_memo: Option<FrameMemo>,
    /// Canonical kernels (see `set_canonical`): connection-set hash -> kernel.
    canon: Option<crate::det::HashMap<u64, u32>>,
    /// Growth events that found an identical kernel already present.
    canon_reused: usize,
    /// Near-miss generalisation spawns a general copy instead of pruning in place
    /// (see `set_generalize_spawn`).
    generalize_spawn: bool,
    /// General kernels spawned so far.
    spawned: usize,
    /// Spawning needs the dropped inputs absent in this many confirmed near misses.
    spawn_after: u8,
    /// Copies not spawned because a more general kernel already covered them.
    spawn_subsumed: usize,
    /// Generalisation during sleep (see `set_sleep_generalize`): Some(n) = an input is
    /// dropped when absent in n confirmed near misses over the replay.
    sleep_generalize: Option<u8>,
    /// Replay with targets, for sleep generalisation: (input bits, target bits).
    replay_pairs: std::collections::VecDeque<(Vec<u32>, Vec<u32>)>,
    /// General kernels formed during sleep, and candidates rejected by the replay test.
    slept_general: (usize, usize),
    /// Memoised interpretation (see `set_memo`): input hash -> (prior version, winner).
    memo: Option<crate::det::HashMap<u64, (u64, Option<u32>)>>,
    /// Bumped whenever the prior changes in a way that could change a winner.
    version: u64,
    /// The last step's matches were not computed (memo hit); `feedback` recounts them.
    matches_stale: bool,
    memo_lookups: usize,
    memo_hits: usize,
    /// Surprise-gated learning (see `set_surprise_gate`).
    surprise_gate: bool,
    /// Feedback steps that took the expected (no-surprise) path.
    expected_steps: usize,
    /// Recent inputs (active bits only), replayed by `sleep` to find kernels that respond
    /// to exactly the same inputs. Capacity `replay_len` (0 = off).
    replay: std::collections::VecDeque<Vec<u32>>,
    replay_len: usize,
}

/// Per-frame memo of match counts, after Hashlife's memoised sub-nodes: the input is a
/// stack of frames (current word, relayed / recalled content, previous word), and each
/// frame's content recurs far more often than the whole input does. An entry keyed by
/// (frame, content) holds that content's contribution to every kernel's match count, so
/// matching sums a few cached lists instead of fanning every active bit out through the
/// index. Entries are invalidated per kernel, not globally: every change to a kernel's
/// connections (growth, pruning, removal) is logged, and a stale entry is patched by
/// recounting only the changed kernels against its stored bits.
struct FrameMemo {
    entries: crate::det::HashMap<(u32, u64), FrameEntry>,
    /// (change number, kernel) for recent connection changes, oldest first.
    log: std::collections::VecDeque<(u64, u32)>,
    /// Connection changes so far.
    changes: u64,
    lookups: usize,
    hits: usize,
    patched: usize,
}

struct FrameEntry {
    /// The frame's active bits (absolute input positions, sorted).
    bits: Vec<u32>,
    /// `changes` when the entry was last brought up to date.
    seen: u64,
    /// (kernel, matched bits), sorted by kernel.
    contrib: Vec<(u32, u16)>,
}

const FRAME_LOG: usize = 4096; // changes kept for patching
const FRAME_PATCH: u64 = 512; // patch at most this many changes; rebuild beyond
const FRAME_ENTRIES: usize = 200_000; // clear the memo beyond this

impl FrameMemo {
    fn new() -> Self {
        Self { entries: crate::det::HashMap::default(), log: std::collections::VecDeque::new(), changes: 0, lookups: 0, hits: 0, patched: 0 }
    }

    fn note(&mut self, k: usize) {
        self.changes += 1;
        self.log.push_back((self.changes, k as u32));
        if self.log.len() > FRAME_LOG {
            self.log.pop_front();
        }
    }
}

/// Bits shared by two sorted position lists.
fn sorted_overlap(a: &[u32], b: &[u32]) -> u16 {
    let (mut i, mut j, mut n) = (0, 0, 0u16);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                n += 1;
                i += 1;
                j += 1;
            }
        }
    }
    n
}

/// A fast-learning inhibitory loop on the winner competition: one-shot inhibitory tags on
/// kernels, set by the latest target and decaying over `ttl` steps. A tagged kernel loses
/// to every untagged candidate, and still wins if nothing else matches.
#[derive(Clone, Debug)]
pub struct FastInhibition {
    /// Steps a tag lasts.
    pub ttl: u8,
    /// Tag the kernels the target confirmed (adaptation: what just happened in this
    /// context is suppressed next time, so an alternative continuation wins).
    pub on_hits: bool,
    /// Tag the kernels the target contradicted (error-driven: a prediction that just
    /// failed is not repeated while the tag lasts).
    pub on_misses: bool,
    /// Gate, in integers: with `Some(k)`, only kernels whose smoothed hit rate is below
    /// 2^k / (2^k + 1) are tagged (k = 3: 8/9 ≈ 0.89; k = 4: 16/17 ≈ 0.94), tested as
    /// `(misses + 1) << k > hits + 1` (one shift and a compare; see `unreliable`). None = all.
    /// Inhibition then acts where the context is unpredictable (several continuations,
    /// none dominant) and leaves reliable predictions ("went → to", memory copies) alone.
    pub reliable_shift: Option<u32>,
    /// Per kernel, the step until which it is inhibited (a flat array: tagging is one
    /// write, checking one compare, and nothing has to be counted down).
    until: Vec<u64>,
    /// Predictive steps so far.
    now: u64,
}

impl FastInhibition {
    pub fn new(ttl: u8, on_hits: bool, on_misses: bool) -> Self {
        Self { ttl, on_hits, on_misses, reliable_shift: None, until: Vec::new(), now: 0 }
    }

    // a tag set after step n is seen by steps n+1 .. n+ttl-1 (as a counter set to ttl and
    // decremented before each step would be)
    fn set(&mut self, k: usize) {
        if self.until.len() <= k {
            self.until.resize(k + 1, 0);
        }
        self.until[k] = self.now + self.ttl as u64;
    }

    fn inhibits(&self, k: usize) -> bool {
        self.until.get(k).map_or(false, |&u| u > self.now)
    }

    fn tag(&mut self, hits: &[usize], misses: &[usize]) {
        if self.on_hits {
            for &k in hits {
                self.set(k);
            }
        }
        if self.on_misses {
            for &k in misses {
                self.set(k);
            }
        }
    }
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
            silent_counts: crate::det::HashMap::default(),
            sticky_factor: 0,
            sticky_tags: crate::det::HashMap::default(),
            sticky_mask: None,
            sticky_blame: false,
            copy_growth: false,
            trust_floor: None,
            growth_trust: None,
            last_hits: Vec::new(),
            last_misses: Vec::new(),
            fast: None,
            growth_gate: None,
            gated: 0,
            free: Vec::new(),
            replay: std::collections::VecDeque::new(),
            replay_len: 0,
            surprise_gate: false,
            expected_steps: 0,
            memo: None,
            frame_memo: None,
            canon: None,
            canon_reused: 0,
            generalize_spawn: false,
            spawned: 0,
            spawn_after: 1,
            spawn_subsumed: 0,
            sleep_generalize: None,
            replay_pairs: std::collections::VecDeque::new(),
            slept_general: (0, 0),
            version: 0,
            matches_stale: false,
            memo_lookups: 0,
            memo_hits: 0,
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
        if st.replay_len > 0 {
            let mut bits = Vec::new();
            for (wi, &w) in input.as_words().iter().enumerate() {
                let mut w = w;
                while w != 0 {
                    bits.push((wi * 64 + w.trailing_zeros() as usize) as u32);
                    w &= w - 1;
                }
            }
            if st.replay.len() == st.replay_len {
                st.replay.pop_front();
            }
            st.replay.push_back(bits);
        }
        if st.counts.len() < self.active_kernels.len() {
            st.counts.resize(self.active_kernels.len(), 0);
        }
        st.matches_stale = false;
        // Memoised interpretation (Hashlife-style): the same input under the same prior has
        // the same winner, so a repeated input skips matching. Off with fast inhibition or
        // a top-down bias, which change the winner step by step.
        let memo_key = if st.memo.is_some() && bias.is_none() && st.fast.is_none() { Some(input_hash(input)) } else { None };
        if let (Some(key), Some(memo)) = (memo_key, st.memo.as_ref()) {
            st.memo_lookups += 1;
            if let Some(&(v, w)) = memo.get(&key) {
                if v == st.version {
                    st.memo_hits += 1;
                    st.matches_stale = true;
                    let Some(w) = w else { return 0 };
                    let w = w as usize;
                    let k = &mut self.active_kernels[w];
                    for &b in &k.output_set {
                        if (b as usize) < output.bit_len() {
                            output.bit_set(b as usize);
                        }
                    }
                    k.stats.fired_last = true;
                    k.stats.fires += 1;
                    st.last_winner = Some(w);
                    return 1;
                }
            }
        }

        self.accumulate(input);
        let st = self.predictive.as_mut().expect("predictive");

        // fast inhibition: one step later (tags expire by comparison, not by counting down)
        if let Some(f) = st.fast.as_mut() {
            f.now += 1;
        }
        let mode = bias.map(|(_, m)| m);
        let agrees = |k: &SimpleKernel| bias.map_or(false, |(b, _)| predicts(k, b));
        // (not inhibited, agree-first, trusted, depth, agree-within-depth, reliability, count)
        let trust_floor = st.trust_floor;
        // ... then the older kernel (lower id): ties must not depend on the order kernels
        // are visited, which differs between the index fan-out and the frame memo
        let mut best: Option<((bool, bool, bool, usize, bool, Rate, u32, std::cmp::Reverse<usize>), usize)> = None;
        for &k in &st.touched {
            let k = k as usize;
            let count = st.counts[k];
            st.counts[k] = 0;
            let kern = &self.active_kernels[k];
            if (count as usize) < kern.threshold {
                if let Some(frac) = st.cfg.generalize {
                    if count as f32 >= frac * kern.input_bits as f32 {
                        st.last_near.push(k);
                    }
                }
                continue;
            }
            st.last_matches.push(k);
            let a = agrees(kern);
            let prefer = matches!(mode, Some(BiasMode::Prefer | BiasMode::PreferAndFallback)) && a;
            let tie = mode == Some(BiasMode::SameDepth) && a;
            let r = Rate::of(&kern.stats);
            let trusted = trust_floor.map_or(true, |f| r >= f);
            let free = st.fast.as_ref().map_or(true, |f| !f.inhibits(k));
            let key = (free, prefer, trusted, kern.context_frames, tie, r, count, std::cmp::Reverse(k));
            if best.map_or(true, |(b, _)| key > b) {
                best = Some((key, k));
            }
        }
        st.touched.clear();

        if let (Some(key), Some(memo)) = (memo_key, st.memo.as_mut()) {
            if memo.len() > 500_000 {
                memo.clear();
            }
            memo.insert(key, (st.version, best.map(|(_, w)| w as u32)));
        }
        let Some((_, w)) = best else {
            if let Some((b, BiasMode::Fallback | BiasMode::PreferAndFallback)) = bias {
                output.mask_mut(0, b, |a, b| a | b);
            }
            return 0;
        };
        let k = &mut self.active_kernels[w];
        for &b in &k.output_set {
            if (b as usize) < output.bit_len() {
                output.bit_set(b as usize);
            }
        }
        k.stats.fired_last = true;
        k.stats.fires += 1;
        st.last_winner = Some(w);
        1
    }

    /// What the class would predict for `input`, without changing any state
    /// (no stats, no winner bookkeeping). Returns the winning kernel's output
    /// pattern, or None if no kernel matches. Used for counterfactual
    /// (ablation) credit: compare the prediction with and without some inputs.
    pub fn peek(&self, input: &BitVector) -> Option<BitVector> {
        self.peek_scored(input).map(|(out, _)| out)
    }

    /// Like `peek`, also returning the winning kernel's smoothed hit rate.
    pub fn peek_scored(&self, input: &BitVector) -> Option<(BitVector, f32)> {
        let st = self.predictive.as_ref()?;
        let mut counts: crate::det::HashMap<u32, u32> = crate::det::HashMap::default();
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
                ((kern.context_frames, Rate::of(&kern.stats), c), k as usize)
            })
            .max_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)))
            .map(|((_, rel, _), k)| (self.active_kernels[k].output_vector(), rel.value()))
    }

    /// The union of what every matching kernel predicts for `input` (the column's possible
    /// continuations, superposed), without changing any state. A confident context gives
    /// one word's code; an uncertain one superposes a whole class (e.g. every place), so
    /// similar uncertain states share bits.
    pub fn peek_union(&self, input: &BitVector, bits: usize) -> BitVector {
        let mut out = BitVector::new(bits, Some(0));
        let Some(st) = self.predictive.as_ref() else { return out };
        let mut counts: crate::det::HashMap<u32, u32> = crate::det::HashMap::default();
        for b in set_bits(input) {
            if let Some(ks) = st.index.get(b) {
                for &k in ks {
                    *counts.entry(k).or_default() += 1;
                }
            }
        }
        for (k, c) in counts {
            let kern = &self.active_kernels[k as usize];
            if c as usize >= kern.threshold {
                for &b in &kern.output_set {
                    if (b as usize) < bits {
                        out.bit_set(b as usize);
                    }
                }
            }
        }
        out
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
                for &b in &self.active_kernels[k].input_set {
                    if (b as usize) < bits {
                        out.bit_set(b as usize);
                    }
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

    /// Sticky synapses for gradual pruning (`GrowthConfig::generalize`): input bits that
    /// earned copy credit on a hit (they carry the bit the kernel correctly predicted),
    /// or that are in `mask`, need `factor` times as many silent confirmations before
    /// they are dropped. `factor` 0 turns stickiness off; `u8::MAX` makes tagged bits
    /// permanent (never pruned).
    pub fn set_sticky(&mut self, factor: u8, mask: Option<BitVector>) {
        if let Some(st) = self.predictive.as_mut() {
            st.sticky_factor = factor;
            st.sticky_mask = mask;
        }
    }

    /// Winner ranking: with `Some((p, q))`, kernels whose hit rate is below p/q rank below
    /// every trusted kernel, whatever their depth (longest context wins only among kernels
    /// that have earned trust). None restores pure longest-context ranking. The floor is a
    /// ratio of integers and is compared by cross-multiplication (see `Rate`).
    pub fn set_trust_floor(&mut self, floor: Option<(u16, u16)>) {
        if let Some(st) = self.predictive.as_mut() {
            st.trust_floor = floor.map(|(p, q)| Rate::new(p, q));
            st.version += 1;
        }
    }

    /// Growth: a matching kernel that predicted the target blocks same-depth growth only
    /// if its reliability is at least `floor` (None: any kernel blocks it).
    /// Fast inhibitory loop on the winner competition (None = off). See `FastInhibition`.
    pub fn set_fast_inhibition(&mut self, fast: Option<FastInhibition>) {
        if let Some(st) = self.predictive.as_mut() {
            st.fast = fast;
        }
    }

    /// The fast loop alone, without slow learning: tag the kernels that matched on the last
    /// step by whether `target` confirmed them. Use it when `feedback` is off (e.g. at
    /// test); `feedback` already does this.
    pub fn fast_inhibit(&mut self, target: &BitVector) {
        let Some(st) = self.predictive.as_ref() else { return };
        if st.fast.is_none() {
            return;
        }
        let shift = st.fast.as_ref().and_then(|f| f.reliable_shift);
        let (mut hits, mut misses) = (Vec::new(), Vec::new());
        for &m in &st.last_matches {
            if shift.map_or(false, |sh| !unreliable(&self.active_kernels[m].stats, sh)) {
                continue;
            }
            if predicts(&self.active_kernels[m], target) {
                hits.push(m);
            } else {
                misses.push(m);
            }
        }
        if let Some(f) = self.predictive.as_mut().and_then(|st| st.fast.as_mut()) {
            f.tag(&hits, &misses);
        }
    }

    /// The no-surprise path of `feedback` (see `set_surprise_gate`).
    fn confirm_expected(&mut self, w: usize, input: &BitVector, target: &BitVector, cfg: &GrowthConfig) {
        let tick = self.tick;
        let before = Rate::of(&self.active_kernels[w].stats);
        let k = &mut self.active_kernels[w];
        let halves = k.stats.hits == u8::MAX;
        k.stats.record_hit();
        k.stats.last_useful = tick;
        let Some(st) = self.predictive.as_mut() else { return };
        if halves {
            st.version += 1; // halving can shift a rate by rounding: re-rank
        }
        st.last_target_prob = before.value();
        // fast inhibition still sees what happened (adaptation tags every kernel that
        // predicted it; error tags every kernel that did not), as on the full path
        if let Some(f) = st.fast.as_mut() {
            let gate = |m: usize| f.reliable_shift.map_or(true, |sh| unreliable(&self.active_kernels[m].stats, sh));
            let (mut h, mut m) = (Vec::new(), Vec::new());
            if f.on_hits || f.on_misses {
                for &x in &st.last_matches {
                    if !gate(x) {
                        continue;
                    }
                    if predicts(&self.active_kernels[x], target) {
                        h.push(x);
                    } else {
                        m.push(x);
                    }
                }
            }
            f.tag(&h, &m);
        }
        st.last_hits = vec![w];
        st.last_misses.clear();
        st.last_near.clear();
        st.expected_steps += 1;
        // the winner's active connections proved compatible; copy credit for its bits
        // that carry the target
        if let Some(counts) = st.silent_counts.get_mut(&w) {
            counts.retain(|&b, _| !input.bit_get(b));
        }
        if st.sticky_factor > 0 {
            let fb = cfg.frame_words * 64;
            for &b in &self.active_kernels[w].input_set {
                let b = b as usize;
                let t = b % fb;
                if input.bit_get(b) && t < target.bit_len() && target.bit_get(t) {
                    st.sticky_tags.entry(w).or_default().insert(b);
                }
            }
        }
    }

    /// Memoised interpretation (after Hashlife's memoised node results): a hash of the input
    /// maps to the winner it produced, valid while the prior is unchanged (the version is
    /// bumped by full-path learning, growth, sleep, trust-floor changes and counter
    /// halving). A repeated input under the same prior skips matching. Results are exactly
    /// those without the memo.
    pub fn set_memo(&mut self, on: bool) {
        if let Some(st) = self.predictive.as_mut() {
            st.memo = if on { Some(crate::det::HashMap::default()) } else { None };
        }
    }

    /// (lookups, hits) of the memo so far.
    pub fn memo_stats(&self) -> (usize, usize) {
        self.predictive.as_ref().map_or((0, 0), |st| (st.memo_lookups, st.memo_hits))
    }

    /// Canonical kernels (after Hashlife's hash-consed nodes): growth samples each frame's
    /// active bits deterministically (smallest fixed hash first), so the same context
    /// always yields the same kernel, and a table of connection sets stops an identical
    /// kernel from being grown twice.
    pub fn set_canonical(&mut self, on: bool) {
        if let Some(st) = self.predictive.as_mut() {
            st.canon = if on { Some(crate::det::HashMap::default()) } else { None };
        }
    }

    /// Near-miss generalisation (`GrowthConfig::generalize`): with `on`, a confirmed near
    /// miss does not prune the kernel's unused inputs in place. It spawns a general copy
    /// without them and keeps the specific original. Both compete as usual (depth, then
    /// reliability): the general kernel serves where the pruned inputs never mattered (a new
    /// filler in a known slot), the specific one where they do.
    pub fn set_generalize_spawn(&mut self, on: bool) {
        if let Some(st) = self.predictive.as_mut() {
            st.generalize_spawn = on;
        }
    }

    /// General kernels spawned by near-miss generalisation.
    pub fn spawned(&self) -> usize {
        self.predictive.as_ref().map_or(0, |st| st.spawned)
    }

    /// Spawning needs repeated evidence: an input is dropped from a copy only after it was
    /// absent in `n` confirmed near misses of the kernel (default 1), so a copy stands for a
    /// regularity rather than one coincidence.
    pub fn set_spawn_after(&mut self, n: u8) {
        if let Some(st) = self.predictive.as_mut() {
            st.spawn_after = n.max(1);
        }
    }

    /// Generalisation during sleep: with `Some(n)`, every sleep first replays the last
    /// `replay_len` (input, target) pairs against each kernel. Inputs absent in at least n of
    /// a kernel's confirmed near misses (the target was its output, some inputs missing) are
    /// dropped to form a candidate general rule. The candidate is tested on the whole replay
    /// and installed only if it matches more replayed inputs than its source and is at least
    /// as reliable on them. Sleep's merge can then absorb specific kernels into it. Needs
    /// `set_replay`.
    pub fn set_sleep_generalize(&mut self, n: Option<u8>) {
        if let Some(st) = self.predictive.as_mut() {
            st.sleep_generalize = n.map(|n| n.max(1));
        }
    }

    /// (General kernels formed during sleep, candidates the replay test rejected.)
    pub fn slept_general(&self) -> (usize, usize) {
        self.predictive.as_ref().map_or((0, 0), |st| st.slept_general)
    }

    /// Add an (input, target) pair to the replay that sleep generalisation reads (e.g. a
    /// hippocampal replay of an episode). Needs `set_sleep_generalize` and `set_replay`.
    pub fn add_replay(&mut self, input: &BitVector, target: &BitVector) {
        let Some(st) = self.predictive.as_mut() else { return };
        if st.sleep_generalize.is_none() || st.replay_len == 0 {
            return;
        }
        let bits = |v: &BitVector| -> Vec<u32> {
            let mut out = Vec::new();
            for (wi, &w) in v.as_words().iter().enumerate() {
                let mut w = w;
                while w != 0 {
                    out.push((wi * 64 + w.trailing_zeros() as usize) as u32);
                    w &= w - 1;
                }
            }
            out
        };
        if st.replay_pairs.len() >= st.replay_len {
            st.replay_pairs.pop_front();
        }
        st.replay_pairs.push_back((bits(input), bits(target)));
    }

    /// Generalisation from the replay alone, without sleep's downscaling, pruning and merging:
    /// new general rules are added, nothing is removed.
    pub fn generalize_from_replay(&mut self) {
        self.generalize_offline();
    }

    /// Form general rules offline from the replay (see `set_sleep_generalize`).
    fn generalize_offline(&mut self) {
        let Some(st) = self.predictive.as_ref() else { return };
        let Some(n) = st.sleep_generalize else { return };
        if st.replay_pairs.is_empty() {
            return;
        }
        let cfg = st.cfg;
        let frac = cfg.generalize.unwrap_or(0.5);
        let frame_bits = cfg.frame_words * 64;
        let in_len = st.index.len().max(1);
        let out_len = self.active_kernels.iter().map(|k| k.output_len as usize).max().unwrap_or(1).max(1);
        let pairs: Vec<(BitVector, BitVector)> = st
            .replay_pairs
            .iter()
            .map(|(i, t)| {
                let mut iv = BitVector::new(in_len, Some(0));
                for &b in i {
                    if (b as usize) < in_len {
                        iv.bit_set(b as usize);
                    }
                }
                let mut tv = BitVector::new(out_len, Some(0));
                for &b in t {
                    if (b as usize) < out_len {
                        tv.bit_set(b as usize);
                    }
                }
                (iv, tv)
            })
            .collect();
        // per kernel: confirmed near misses, and how often each input was absent in them
        let mut absent: crate::det::HashMap<usize, (u32, crate::det::HashMap<u32, u32>)> = crate::det::HashMap::default();
        let mut counts = vec![0u32; self.active_kernels.len()];
        for (iv, tv) in &pairs {
            let mut touched: Vec<usize> = Vec::new();
            for wi in 0..iv.as_words().len() {
                let mut w = iv.as_words()[wi];
                while w != 0 {
                    let b = wi * 64 + w.trailing_zeros() as usize;
                    w &= w - 1;
                    if let Some(ks) = st.index.get(b) {
                        for &k in ks {
                            let k = k as usize;
                            if counts[k] == 0 {
                                touched.push(k);
                            }
                            counts[k] += 1;
                        }
                    }
                }
            }
            for k in touched {
                let c = counts[k] as usize;
                counts[k] = 0;
                let kern = &self.active_kernels[k];
                if c < kern.threshold && c as f32 >= frac * kern.input_set.len() as f32 && predicts(kern, tv) {
                    let e = absent.entry(k).or_default();
                    e.0 += 1;
                    for &b in &kern.input_set {
                        if !iv.bit_get(b as usize) {
                            *e.1.entry(b).or_default() += 1;
                        }
                    }
                }
            }
        }
        // replay score of a rule: (matched, right)
        let score = |inputs: &[u32], threshold: usize, output: &SimpleKernel| -> (u32, u32) {
            let mut m = (0, 0);
            for (iv, tv) in &pairs {
                if inputs.iter().filter(|&&b| iv.bit_get(b as usize)).count() >= threshold {
                    m.0 += 1;
                    m.1 += predicts(output, tv) as u32;
                }
            }
            m
        };
        let min_bits = cfg.sample_bits.max(2);
        let mut installs: Vec<SimpleKernel> = Vec::new();
        let mut rejected = 0usize;
        for (k, (confirmed, bits)) in absent {
            if confirmed < n as u32 {
                continue;
            }
            let src = &self.active_kernels[k];
            let kept: Vec<u32> = src.input_set.iter().copied().filter(|b| bits.get(b).map_or(true, |&c| c < n as u32)).collect();
            if kept.len() == src.input_set.len() || kept.len() < min_bits {
                continue;
            }
            if let Some(canon) = st.canon.as_ref() {
                if canon.contains_key(&connection_key(&kept, &src.output_set)) {
                    continue;
                }
            }
            let mut per_frame: crate::det::HashMap<usize, usize> = crate::det::HashMap::default();
            for &b in &kept {
                *per_frame.entry(b as usize / frame_bits).or_default() += 1;
            }
            let smallest = per_frame.values().copied().min().unwrap_or(cfg.sample_bits).min(cfg.sample_bits);
            let tolerance = (smallest as f32 * (1.0 - cfg.match_fraction)).floor() as usize;
            let threshold = kept.len().saturating_sub(tolerance).max(1);
            // the replay test: more general in fact, and at least as reliable
            let (gm, gr) = score(&kept, threshold, src);
            let (sm, sr) = score(&src.input_set, src.threshold, src);
            let as_reliable = (gr as u64 + 1) * (sm as u64 + 2) >= (sr as u64 + 1) * (gm as u64 + 2);
            if gm <= sm || !as_reliable {
                rejected += 1;
                continue;
            }
            let reach = kept.iter().map(|&b| b as usize / frame_bits + 1).max().unwrap_or(1);
            let mut g = SimpleKernel::sparse(kept, src.output_set.clone(), src.output_len as usize, threshold, KernelOp::Or);
            g.context_frames = reach;
            // its replay record, scaled into the 8-bit counters
            let scale = ((gm as f32) / 64.0).max(1.0);
            g.stats.hits = ((gr as f32 / scale).round() as u32).min(255) as u8;
            g.stats.misses = (((gm - gr) as f32 / scale).round() as u32).min(255) as u8;
            g.stats.last_useful = self.tick;
            installs.push(g);
        }
        let (iv, tv) = pairs[0].clone();
        let made = installs.len();
        for g in installs {
            self.install(g, &iv, &tv);
        }
        if let Some(st) = self.predictive.as_mut() {
            st.slept_general.0 += made;
            st.slept_general.1 += rejected;
        }
    }

    /// Copies not spawned because a kernel with the same output reading a subset of the kept
    /// inputs already existed.
    pub fn spawn_subsumed(&self) -> usize {
        self.predictive.as_ref().map_or(0, |st| st.spawn_subsumed)
    }

    /// Growth events that found an identical kernel already present.
    pub fn canon_reused(&self) -> usize {
        self.predictive.as_ref().map_or(0, |st| st.canon_reused)
    }

    /// Per-frame memo of match counts (see `FrameMemo`). Exact: the counts are those of the
    /// index fan-out.
    pub fn set_frame_memo(&mut self, on: bool) {
        if let Some(st) = self.predictive.as_mut() {
            st.frame_memo = if on { Some(FrameMemo::new()) } else { None };
        }
    }

    /// (lookups, hits, patched) of the frame memo.
    pub fn frame_memo_stats(&self) -> (usize, usize, usize) {
        self.predictive.as_ref().and_then(|st| st.frame_memo.as_ref()).map_or((0, 0, 0), |f| (f.lookups, f.hits, f.patched))
    }

    /// Record that kernel `k`'s connections changed (for the frame memo).
    fn note_connections(&mut self, k: usize) {
        if let Some(f) = self.predictive.as_mut().and_then(|st| st.frame_memo.as_mut()) {
            f.note(k);
        }
    }

    /// Match counts for `input` into `counts` / `touched`: through the frame memo if on,
    /// else by fanning every active bit out through the index.
    fn accumulate(&mut self, input: &BitVector) {
        let kernels = &self.active_kernels;
        let Some(st) = self.predictive.as_mut() else { return };
        if st.counts.len() < kernels.len() {
            st.counts.resize(kernels.len(), 0);
        }
        let Some(fm) = st.frame_memo.as_mut() else {
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
            return;
        };
        if fm.entries.len() > FRAME_ENTRIES {
            fm.entries.clear();
        }
        let fw = st.cfg.frame_words.max(1);
        let words = input.as_words();
        for f in 0..words.len().div_ceil(fw) {
            let span = &words[f * fw..((f + 1) * fw).min(words.len())];
            if span.iter().all(|&w| w == 0) {
                continue;
            }
            let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ f as u64;
            for (i, &w) in span.iter().enumerate() {
                if w != 0 {
                    h = (h ^ i as u64).wrapping_mul(0x0100_0000_01b3);
                    h = (h ^ w).wrapping_mul(0x0100_0000_01b3).rotate_left(29);
                }
            }
            fm.lookups += 1;
            let key = (f as u32, h);
            let mut bits: Vec<u32> = Vec::new();
            for (i, &w) in span.iter().enumerate() {
                let mut w = w;
                while w != 0 {
                    bits.push(((f * fw + i) * 64 + w.trailing_zeros() as usize) as u32);
                    w &= w - 1;
                }
            }
            let usable = fm.entries.get(&key).map_or(false, |e| e.bits == bits);
            let oldest = fm.log.front().map_or(fm.changes + 1, |&(c, _)| c);
            let mut rebuild = !usable;
            if usable {
                let e = fm.entries.get_mut(&key).expect("entry");
                if e.seen == fm.changes {
                    fm.hits += 1;
                } else if fm.changes - e.seen <= FRAME_PATCH && oldest <= e.seen + 1 {
                    // patch: recount only the kernels whose connections changed since
                    let mut changed: Vec<u32> = fm.log.iter().filter(|&&(c, _)| c > e.seen).map(|&(_, k)| k).collect();
                    changed.sort_unstable();
                    changed.dedup();
                    for k in changed {
                        let n = kernels.get(k as usize).map_or(0, |kern| sorted_overlap(&kern.input_set, &e.bits));
                        match e.contrib.binary_search_by_key(&k, |&(x, _)| x) {
                            Ok(i) if n == 0 => {
                                e.contrib.remove(i);
                            }
                            Ok(i) => e.contrib[i].1 = n,
                            Err(i) if n > 0 => e.contrib.insert(i, (k, n)),
                            Err(_) => {}
                        }
                    }
                    e.seen = fm.changes;
                    fm.patched += 1;
                } else {
                    rebuild = true;
                }
            }
            if rebuild {
                let mut tally: crate::det::HashMap<u32, u16> = crate::det::HashMap::default();
                for &b in &bits {
                    if let Some(ks) = st.index.get(b as usize) {
                        for &k in ks {
                            *tally.entry(k).or_default() += 1;
                        }
                    }
                }
                let mut contrib: Vec<(u32, u16)> = tally.into_iter().collect();
                contrib.sort_unstable();
                fm.entries.insert(key, FrameEntry { bits, seen: fm.changes, contrib });
            }
            let e = &fm.entries[&key];
            for &(k, n) in &e.contrib {
                if st.counts[k as usize] == 0 {
                    st.touched.push(k);
                }
                st.counts[k as usize] += n as u32;
            }
        }
    }

    /// Recompute the last step's matched and near-matched kernels for `input` (after a memo
    /// hit skipped matching), without changing the winner or any statistics.
    fn recount(&mut self, input: &BitVector) {
        let Some(st) = self.predictive.as_mut() else { return };
        st.matches_stale = false;
        st.last_matches.clear();
        st.last_near.clear();
        if st.counts.len() < self.active_kernels.len() {
            st.counts.resize(self.active_kernels.len(), 0);
        }
        self.accumulate(input);
        let st = self.predictive.as_mut().expect("predictive");
        for &k in &st.touched {
            let k = k as usize;
            let count = st.counts[k];
            st.counts[k] = 0;
            let kern = &self.active_kernels[k];
            if (count as usize) < kern.threshold {
                if let Some(frac) = st.cfg.generalize {
                    if count as f32 >= frac * kern.input_bits as f32 {
                        st.last_near.push(k);
                    }
                }
                continue;
            }
            st.last_matches.push(k);
        }
        st.touched.clear();
    }

    /// Surprise-gated learning: when the winner predicted the target, confirm only the
    /// winner (no re-scoring of other kernels, no pruning, no growth); learn fully only
    /// on surprise.
    pub fn set_surprise_gate(&mut self, on: bool) {
        if let Some(st) = self.predictive.as_mut() {
            st.surprise_gate = on;
        }
    }

    /// Feedback steps that took the expected (no-surprise) path.
    pub fn expected_steps(&self) -> usize {
        self.predictive.as_ref().map_or(0, |st| st.expected_steps)
    }

    /// Uncertainty-gated growth (None = off). With `Some((k, min))`, a miss grows nothing
    /// when the winning kernel has at least `min` observations, its hit rate is below
    /// 2^k / (2^k + 1) (integer test, see `unreliable`), and no input frame carries the
    /// target: the context is known to be noisy and nothing in the input could explain the
    /// outcome, so a new kernel would only be another guess.
    pub fn set_growth_gate(&mut self, gate: Option<(u32, u16)>) {
        if let Some(st) = self.predictive.as_mut() {
            st.growth_gate = gate;
        }
    }

    /// Growth events suppressed by the gate so far.
    pub fn gated_growth(&self) -> usize {
        self.predictive.as_ref().map_or(0, |st| st.gated)
    }

    /// Keep the last `n` inputs for sleep replay (0 = off).
    pub fn set_replay(&mut self, n: usize) {
        if let Some(st) = self.predictive.as_mut() {
            st.replay_len = n;
            st.replay.clear();
        }
    }

    /// Kernels in use (removed ones excluded).
    pub fn live(&self) -> usize {
        self.active_kernels.len() - self.predictive.as_ref().map_or(0, |st| st.free.len())
    }

    /// Sleep (synaptic homeostasis, after Tononi & Cirelli): an offline pass that keeps the
    /// prior compact. Returns (pruned, merged).
    /// 1. Downscale: every kernel's hit and miss counts shift right by one (relative
    ///    strengths kept; old evidence weighs less).
    /// 2. Prune: kernels that have only failed (no hits left, some misses) are removed.
    /// 3. Merge: a kernel is redundant if another kernel with the same output reads a
    ///    strict subset of its inputs (so it matches whenever this one does) and is at
    ///    least as reliable; the specific one is removed and the general one predicts for
    ///    both.
    /// Removed kernels' slots go to a free list that growth reuses; the inverted index is
    /// rebuilt once.
    pub fn sleep(&mut self) -> (usize, usize) {
        // 0. generalise from the replay (if on), before downscaling and merging
        self.generalize_offline();
        let Some(st) = self.predictive.as_mut() else { return (0, 0) };
        st.version += 1;
        st.matches_stale = false;
        if let Some(w) = st.last_winner.take() {
            self.active_kernels[w].stats.fired_last = false;
        }
        st.last_matches.clear();
        st.last_hits.clear();
        st.last_misses.clear();
        st.last_near.clear();
        let live: Vec<usize> = (0..self.active_kernels.len()).filter(|&k| !self.active_kernels[k].input_set.is_empty()).collect();
        // 1. downscale
        for &k in &live {
            self.active_kernels[k].stats.halve();
        }
        let mut remove = vec![false; self.active_kernels.len()];
        // 2. prune what only fails
        let mut pruned = 0;
        for &k in &live {
            let s = &self.active_kernels[k].stats;
            if s.hits == 0 && s.misses > 0 {
                remove[k] = true;
                pruned += 1;
            }
        }
        // 3. merge: within each output, a kernel whose inputs contain a more general,
        // at-least-as-reliable kernel's inputs is redundant. Candidates are found through
        // the general kernel's lowest input bit, which the specific kernel must contain.
        let mut merged = 0;
        let mut by_output: crate::det::HashMap<&[u32], Vec<usize>> = crate::det::HashMap::default();
        for &k in &live {
            if !remove[k] {
                by_output.entry(self.active_kernels[k].output_set.as_slice()).or_default().push(k);
            }
        }
        let mut redundant = Vec::new();
        for group in by_output.values() {
            if group.len() < 2 {
                continue;
            }
            let mut by_first: crate::det::HashMap<u32, Vec<usize>> = crate::det::HashMap::default();
            for &k in group {
                by_first.entry(self.active_kernels[k].input_set[0]).or_default().push(k);
            }
            for &a in group {
                let ka = &self.active_kernels[a];
                let ra = Rate::of(&ka.stats);
                let general = ka.input_set.iter().filter_map(|b| by_first.get(b)).flatten().any(|&b| {
                    let kb = &self.active_kernels[b];
                    b != a
                        && kb.input_set.len() < ka.input_set.len()
                        && Rate::of(&kb.stats) >= ra
                        && is_subset(&kb.input_set, &ka.input_set)
                });
                if general {
                    redundant.push(a);
                }
            }
        }
        for a in redundant {
            if !remove[a] {
                remove[a] = true;
                merged += 1;
            }
        }
        // 3b. merge by replay: kernels with the same output that match exactly the same
        // replayed inputs carry the same information (e.g. two random bit samples of the
        // same words); keep the most reliable (then most used), drop the rest. Kernels
        // that match nothing in the replay are left alone.
        if !st.replay.is_empty() {
            let n = self.active_kernels.len();
            let mut signature = vec![0u64; n];
            let mut fired = vec![0u32; n];
            let mut counts = vec![0u32; n];
            let mut touched: Vec<usize> = Vec::new();
            for (i, bits) in st.replay.iter().enumerate() {
                for &b in bits {
                    if let Some(ks) = st.index.get(b as usize) {
                        for &k in ks {
                            let k = k as usize;
                            if counts[k] == 0 {
                                touched.push(k);
                            }
                            counts[k] += 1;
                        }
                    }
                }
                // a replay step's identity, mixed into every kernel that matched it
                let tag = (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                for &k in &touched {
                    if counts[k] as usize >= self.active_kernels[k].threshold && !remove[k] {
                        signature[k] = signature[k].rotate_left(5) ^ tag;
                        fired[k] += 1;
                    }
                    counts[k] = 0;
                }
                touched.clear();
            }
            let mut best: crate::det::HashMap<(&[u32], u64, u32), usize> = crate::det::HashMap::default();
            for &k in &live {
                if remove[k] || fired[k] == 0 {
                    continue;
                }
                let key = (self.active_kernels[k].output_set.as_slice(), signature[k], fired[k]);
                match best.get(&key).copied() {
                    None => {
                        best.insert(key, k);
                    }
                    Some(j) => {
                        let (rk, rj) = (Rate::of(&self.active_kernels[k].stats), Rate::of(&self.active_kernels[j].stats));
                        let uses = |x: usize| self.active_kernels[x].stats.hits as u16 + self.active_kernels[x].stats.misses as u16;
                        let (keep, drop) = if (rk, uses(k)) > (rj, uses(j)) { (k, j) } else { (j, k) };
                        best.insert(key, keep);
                        remove[drop] = true;
                        merged += 1;
                    }
                }
            }
        }
        // free the removed kernels and rebuild the index from the survivors
        for k in 0..remove.len() {
            if remove[k] {
                if let Some(canon) = st.canon.as_mut() {
                    let key = connection_key(&self.active_kernels[k].input_set, &self.active_kernels[k].output_set);
                    if canon.get(&key) == Some(&(k as u32)) {
                        canon.remove(&key);
                    }
                }
                let kern = &mut self.active_kernels[k];
                kern.input_mask = BitVector::new(64, Some(0));
                kern.output_mask = BitVector::new(64, Some(0));
                kern.input_set.clear();
                kern.output_set.clear();
                kern.input_bits = 0;
                kern.threshold = usize::MAX;
                st.silent_counts.remove(&k);
                st.sticky_tags.remove(&k);
                if let Some(f) = st.fast.as_mut() {
                    if let Some(u) = f.until.get_mut(k) {
                        *u = 0;
                    }
                }
                st.free.push(k);
                if let Some(f) = st.frame_memo.as_mut() {
                    f.note(k);
                }
            }
        }
        for list in st.index.iter_mut() {
            list.clear();
        }
        for (k, kern) in self.active_kernels.iter().enumerate() {
            for &b in &kern.input_set {
                if let Some(list) = st.index.get_mut(b as usize) {
                    list.push(k as u32);
                }
            }
        }
        (pruned, merged)
    }

    /// Kernels that matched on the last predictive step.
    pub fn matched(&self) -> usize {
        self.predictive.as_ref().map_or(0, |st| st.last_matches.len())
    }

    /// Distinct outputs among the kernels that matched on the last step.
    pub fn matched_outputs(&self) -> usize {
        let Some(st) = self.predictive.as_ref() else { return 0 };
        let mut outs: Vec<&[u32]> = st.last_matches.iter().map(|&k| self.active_kernels[k].output_set.as_slice()).collect();
        outs.sort();
        outs.dedup();
        outs.len()
    }

    /// Kernels currently inhibited by the fast loop.
    pub fn inhibited(&self) -> usize {
        self.predictive.as_ref().and_then(|st| st.fast.as_ref()).map_or(0, |f| (0..f.until.len()).filter(|&k| f.inhibits(k)).count())
    }

    /// Growth trust as an integer ratio p/q (see `set_trust_floor`).
    pub fn set_growth_trust(&mut self, floor: Option<(u16, u16)>) {
        if let Some(st) = self.predictive.as_mut() {
            st.growth_trust = floor.map(|(p, q)| Rate::new(p, q));
        }
    }

    /// Credit-guided growth: a new kernel samples, in any frame that contains the target's
    /// bits, only those bits (it is born as a copy kernel for the whole word).
    pub fn set_copy_growth(&mut self, on: bool) {
        if let Some(st) = self.predictive.as_mut() {
            st.copy_growth = on;
        }
    }

    /// Let bad credit release sticky tags (see `sticky_blame`).
    pub fn set_sticky_blame(&mut self, on: bool) {
        if let Some(st) = self.predictive.as_mut() {
            st.sticky_blame = on;
        }
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
        if st.sleep_generalize.is_some() && st.replay_len > 0 {
            let bits = |v: &BitVector| -> Vec<u32> {
                let mut out = Vec::new();
                for (wi, &w) in v.as_words().iter().enumerate() {
                    let mut w = w;
                    while w != 0 {
                        out.push((wi * 64 + w.trailing_zeros() as usize) as u32);
                        w &= w - 1;
                    }
                }
                out
            };
            let pair = (bits(input), bits(target));
            let st = self.predictive.as_mut().expect("predictive");
            if st.replay_pairs.len() >= st.replay_len {
                st.replay_pairs.pop_front();
            }
            st.replay_pairs.push_back(pair);
        }
        let Some(st) = self.predictive.as_ref() else { return };

        let winner = st.last_winner;
        // Surprise-gated learning: an expected input (the winner predicted it) carries no
        // error, so only the winner is confirmed: its hit, its recency and its copy-credit
        // tags. The other matched kernels are not re-scored, nothing is pruned or grown.
        // A surprise (wrong winner, or none) takes the full path below.
        if st.surprise_gate {
            if let Some(w) = winner.filter(|&w| predicts(&self.active_kernels[w], target)) {
                self.confirm_expected(w, input, target, &cfg);
                return;
            }
        }
        // the full path changes the prior: memoised winners are no longer valid, and if
        // matching was skipped (memo hit) the matched kernels are recounted from `input`
        if st.matches_stale {
            self.recount(input);
        }
        let Some(st) = self.predictive.as_mut() else { return };
        st.version += 1;
        let matches = st.last_matches.clone();
        let trust_floor = st.growth_trust;
        let mut depth_has_target = vec![false; cfg.max_frames + 2];
        let mut expected: Option<(usize, Rate)> = None;
        for &m in &matches {
            let k = &self.active_kernels[m];
            if predicts(k, target) {
                let key = (k.context_frames, Rate::of(&k.stats));
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
                // A kernel that was right blocks same-depth growth only if it is trusted:
                // a lucky guess by an unreliable kernel must not stop a better kernel
                // from being grown for this case.
                let trusted = trust_floor.map_or(true, |f| Rate::of(&k.stats) >= f);
                k.stats.record_hit();
                k.stats.last_useful = self.tick;
                if trusted {
                    depth_has_target[k.context_frames] = true;
                }
                hits.push(m);
            } else {
                k.stats.record_miss();
                misses.push(m);
            }
        }
        if let Some(st) = self.predictive.as_mut() {
            st.last_target_prob = expected.map_or(0.0, |e| e.1.value());
            if let Some(f) = st.fast.as_mut() {
                let gate = |v: &Vec<usize>| -> Vec<usize> {
                    v.iter().copied().filter(|&k| f.reliable_shift.map_or(true, |sh| unreliable(&self.active_kernels[k].stats, sh))).collect()
                };
                let (h, m) = (gate(&hits), gate(&misses));
                f.tag(&h, &m);
            }
            st.last_hits = hits;
            st.last_misses = misses;
        }
        if cfg.generalize.is_some() {
            self.generalize_near_misses(input, target, &cfg);
        }

        let unpredicted = match winner {
            Some(w) => {
                let k = &self.active_kernels[w];
                target_bits - k.output_set.iter().filter(|&&b| (b as usize) < target.bit_len() && target.bit_get(b as usize)).count()
            }
            None => target_bits,
        };
        if (unpredicted as f32) <= cfg.surprise_fraction * target_bits as f32 {
            return;
        }

        // Expected uncertainty (acetylcholine-like): if the winner's context is known to be
        // unpredictable (enough evidence, hit rate below the gate) and no input frame
        // carries the target (nothing a new kernel could copy or key on), the miss is the
        // context's normal noise: count it, but grow nothing.
        if let (Some((shift, min)), Some(w)) = (self.predictive.as_ref().and_then(|st| st.growth_gate), winner) {
            let s = &self.active_kernels[w].stats;
            let known_random = s.hits as u16 + s.misses as u16 >= min && unreliable(s, shift);
            if known_random && !target_in_frames(input, target, cfg.frame_words) {
                if let Some(st) = self.predictive.as_mut() {
                    st.gated += 1;
                }
                return;
            }
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
        let mut spawns: Vec<(Vec<u32>, Vec<u32>, Rate)> = Vec::new();
        self.generalize_in_place(input, target, cfg, &mut spawns);
        // the general copies: same output, the kept inputs, a threshold tolerating a frame's
        // worth of noise as at growth; skipped if an identical kernel exists
        let frame_bits = cfg.frame_words * 64;
        for (kept, output_set, source_rate) in spawns {
            if let Some(st) = self.predictive.as_ref() {
                if let Some(canon) = st.canon.as_ref() {
                    if let Some(&k) = canon.get(&connection_key(&kept, &output_set)) {
                        let kern = &self.active_kernels[k as usize];
                        if kern.input_set == kept && kern.output_set == output_set {
                            continue;
                        }
                    }
                }
            }
            // subsumed: a kernel with the same output already reads a subset of the kept
            // inputs (it matches whenever the copy would) and is at least as reliable as the
            // kernel the copy comes from (as sleep's merge requires); found through its
            // lowest input bit, which must be one of the kept bits
            let subsumed = self.predictive.as_ref().map_or(false, |st| {
                let free: crate::det::HashSet<usize> = st.free.iter().copied().collect();
                kept.iter().any(|&b| {
                    st.index.get(b as usize).map_or(false, |ks| {
                        ks.iter().any(|&k| {
                            let kern = &self.active_kernels[k as usize];
                            !free.contains(&(k as usize))
                                && kern.input_set.first() == Some(&b)
                                && kern.output_set == output_set
                                && kern.input_set.len() <= kept.len()
                                && Rate::of(&kern.stats) >= source_rate
                                && kern.input_set.iter().all(|x| kept.binary_search(x).is_ok())
                        })
                    })
                })
            });
            if subsumed {
                if let Some(st) = self.predictive.as_mut() {
                    st.spawn_subsumed += 1;
                }
                continue;
            }
            let mut per_frame: crate::det::HashMap<usize, usize> = crate::det::HashMap::default();
            for &b in &kept {
                *per_frame.entry(b as usize / frame_bits).or_default() += 1;
            }
            let smallest = per_frame.values().copied().min().unwrap_or(cfg.sample_bits).min(cfg.sample_bits);
            let tolerance = (smallest as f32 * (1.0 - cfg.match_fraction)).floor() as usize;
            let reach = kept.iter().map(|&b| b as usize / frame_bits + 1).max().unwrap_or(1);
            let threshold = kept.len().saturating_sub(tolerance).max(1);
            let mut k = SimpleKernel::sparse(kept, output_set, target.bit_len(), threshold, KernelOp::Or);
            k.context_frames = reach;
            k.stats.record_hit();
            k.stats.last_useful = self.tick;
            self.install(k, input, target);
            if let Some(st) = self.predictive.as_mut() {
                st.spawned += 1;
            }
        }
    }

    fn generalize_in_place(&mut self, input: &BitVector, target: &BitVector, cfg: &GrowthConfig, spawns: &mut Vec<(Vec<u32>, Vec<u32>, Rate)>) {
        let Some(st) = self.predictive.as_mut() else { return };
        let near = std::mem::take(&mut st.last_near);
        let min_bits = cfg.sample_bits.max(2);
        let need = if st.generalize_spawn { cfg.generalize_after.max(st.spawn_after).max(1) } else { cfg.generalize_after.max(1) };
        let frame_bits = cfg.frame_words * 64;
        // Bad credit: a tagged bit that was active when its kernel fired and mispredicted,
        // and whose copied bit is not in the target, loses its tag.
        if st.sticky_blame {
            for &k in &st.last_misses {
                if let Some(tags) = st.sticky_tags.get_mut(&k) {
                    tags.retain(|&b| {
                        let t = b % frame_bits;
                        !(input.bit_get(b) && t < target.bit_len() && !target.bit_get(t))
                    });
                }
            }
        }
        // A connection that is active while its kernel is right has proven compatible.
        for &k in &st.last_hits {
            if let Some(counts) = st.silent_counts.get_mut(&k) {
                counts.retain(|&b, _| !input.bit_get(b));
            }
            // Copy credit: an active input bit that carries the same bit (frame-relative)
            // as the target the kernel just predicted correctly gets a sticky tag.
            if st.sticky_factor > 0 {
                for &b in &self.active_kernels[k].input_set {
                    let b = b as usize;
                    let t = b % frame_bits;
                    if input.bit_get(b) && t < target.bit_len() && target.bit_get(t) {
                        st.sticky_tags.entry(k).or_default().insert(b);
                    }
                }
            }
        }
        for k in near {
            if !predicts(&self.active_kernels[k], target) {
                continue;
            }
            let old: Vec<usize> = self.active_kernels[k].input_set.iter().map(|&b| b as usize).collect();
            let counts = st.silent_counts.entry(k).or_default();
            let tags = st.sticky_tags.get(&k);
            let mut drop = Vec::new();
            for &b in &old {
                if input.bit_get(b) {
                    counts.remove(&b);
                } else {
                    let c = counts.entry(b).or_insert(0);
                    *c = c.saturating_add(1);
                    let sticky = st.sticky_factor > 0
                        && (tags.map_or(false, |t| t.contains(&b))
                            || st.sticky_mask.as_ref().map_or(false, |m| b < m.bit_len() && m.bit_get(b)));
                    // factor u8::MAX: tagged bits are never pruned
                    let never = sticky && st.sticky_factor == u8::MAX;
                    let needed = if sticky { need.saturating_mul(st.sticky_factor) } else { need };
                    if !never && *c >= needed {
                        drop.push(b);
                    }
                }
            }
            if drop.is_empty() || old.len() - drop.len() < min_bits {
                continue;
            }
            if st.generalize_spawn {
                // keep the specific kernel; queue a general copy without the unused inputs
                for &b in &drop {
                    counts.remove(&b);
                }
                let mut kept: Vec<u32> = old.iter().filter(|b| !drop.contains(b)).map(|&b| b as u32).collect();
                kept.sort_unstable();
                spawns.push((kept, self.active_kernels[k].output_set.clone(), Rate::of(&self.active_kernels[k].stats)));
                continue;
            }
            for &b in &drop {
                st.index[b].retain(|&x| x as usize != k);
                counts.remove(&b);
            }
            let old_key = connection_key(&self.active_kernels[k].input_set, &self.active_kernels[k].output_set);
            self.active_kernels[k].remove_inputs(&drop);
            if let Some(f) = st.frame_memo.as_mut() {
                f.note(k);
            }
            if let Some(canon) = st.canon.as_mut() {
                if canon.get(&old_key) == Some(&(k as u32)) {
                    canon.remove(&old_key);
                }
                let kern = &self.active_kernels[k];
                canon.entry(connection_key(&kern.input_set, &kern.output_set)).or_insert(k as u32);
            }
            // The match tolerance scales with the kernel's smallest remaining frame, so a
            // frame pruned to a few bits must still be (almost) fully present: otherwise
            // the kernel could fire with that frame absent and turn into a guesser.
            let frame_bits = cfg.frame_words * 64;
            let mut per_frame: crate::det::HashMap<usize, usize> = crate::det::HashMap::default();
            for &b in &old {
                if !drop.contains(&b) {
                    *per_frame.entry(b / frame_bits).or_default() += 1;
                }
            }
            let smallest = per_frame.values().copied().min().unwrap_or(cfg.sample_bits).min(cfg.sample_bits);
            let tolerance = (smallest as f32 * (1.0 - cfg.match_fraction)).floor() as usize;
            let kern = &mut self.active_kernels[k];
            kern.threshold = (old.len() - drop.len()).saturating_sub(tolerance).max(1);
            kern.stats.record_hit();
            kern.stats.last_useful = self.tick;
        }
    }

    /// Grow (or, at the budget, recycle) a kernel that recognises a sample of the
    /// active bits in the `depth` most recent frames of `input` and outputs `target`.
    pub fn grow<R: rand::Rng + ?Sized>(&mut self, input: &BitVector, target: &BitVector, depth: usize, rng: &mut R) {
        let tick = self.tick;
        let Some(st) = self.predictive.as_mut() else { return };
        let cfg = st.cfg;
        st.version += 1;

        let mut input_set: Vec<u32> = Vec::new();
        let mut sampled = 0;
        let mut reach = 0; // frames back the kernel actually connects to
        for f in 0..depth {
            let first = f * cfg.frame_words * 64;
            let last = ((f + 1) * cfg.frame_words * 64).min(input.bit_len());
            let allowed = |b: usize| st.growth_mask.as_ref().map_or(true, |m| b < m.bit_len() && m.bit_get(b));
            // the frame's active bits, word by word (not bit by bit)
            let mut active: Vec<usize> = Vec::new();
            for wi in first / 64..last.div_ceil(64) {
                let mut w = input.as_words()[wi];
                while w != 0 {
                    let b = wi * 64 + w.trailing_zeros() as usize;
                    w &= w - 1;
                    if b >= first && b < last && allowed(b) {
                        active.push(b);
                    }
                }
            }
            // Credit-guided growth: in a frame that carries the target's bits (a copy of
            // the word to predict, e.g. a recalled place), sample only those bits, so the
            // kernel is born keyed on the whole copied word rather than on incidental words.
            if st.copy_growth {
                let fb = cfg.frame_words * 64;
                let copies: Vec<usize> = active.iter().copied().filter(|&b| (b % fb) < target.bit_len() && target.bit_get(b % fb)).collect();
                if !copies.is_empty() {
                    active = copies;
                }
            }
            // canonical: the active bits with the smallest fixed hash, so the same context
            // always yields the same sample (and the same kernel); else a random sample
            let picked: Vec<usize> = if st.canon.is_some() {
                let mut v = active.clone();
                v.sort_unstable_by_key(|&b| bit_rank(b));
                v.truncate(cfg.sample_bits);
                v
            } else {
                active.choose_multiple(rng, cfg.sample_bits).copied().collect()
            };
            for &b in &picked {
                input_set.push(b as u32);
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

        // stored sparse: the sampled input positions and the target's bits
        let mut output_set: Vec<u32> = Vec::new();
        for (wi, &w) in target.as_words().iter().enumerate() {
            let mut w = w;
            while w != 0 {
                output_set.push((wi * 64 + w.trailing_zeros() as usize) as u32);
                w &= w - 1;
            }
        }
        // hash-consing: an identical kernel (same connections, same output) already
        // exists, so growing another would only duplicate it
        if let Some(canon) = st.canon.as_ref() {
            let mut sorted = input_set.clone();
            sorted.sort_unstable();
            sorted.dedup();
            if let Some(&k) = canon.get(&connection_key(&sorted, &output_set)) {
                let kern = &mut self.active_kernels[k as usize];
                if kern.input_set == sorted && kern.output_set == output_set {
                    kern.stats.last_useful = tick;
                    st.canon_reused += 1;
                    return;
                }
            }
        }
        let mut k = SimpleKernel::sparse(input_set, output_set, target.bit_len(), threshold, KernelOp::Or);
        // Depth is how far back the connections really reach: a kernel grown over
        // empty frames is no more specific than a shallower one and must not
        // outrank (or block the growth of) kernels that use those frames later.
        k.context_frames = reach;
        k.stats.last_useful = tick;
        self.install(k, input, target);
    }

    /// Place a new kernel: in a slot freed by sleep, at the end, or (at the budget) over the
    /// least recently useful one; index its connections and register it for hash-consing.
    fn install(&mut self, k: SimpleKernel, input: &BitVector, target: &BitVector) -> usize {
        let Some(st) = self.predictive.as_mut() else { return 0 };
        let cfg = st.cfg;
        st.version += 1;
        let canon_old = |kern: &SimpleKernel| connection_key(&kern.input_set, &kern.output_set);
        let slot = if let Some(free) = st.free.pop() {
            // a slot emptied by sleep: nothing points to it any more
            st.silent_counts.remove(&free);
            st.sticky_tags.remove(&free);
            if let Some(f) = st.fast.as_mut() {
                if let Some(u) = f.until.get_mut(free) {
                    *u = 0;
                }
            }
            self.active_kernels[free] = k;
            self.grown += 1;
            free
        } else if self.active_kernels.len() < cfg.max_kernels {
            self.active_kernels.push(k);
            self.grown += 1;
            self.active_kernels.len() - 1
        } else {
            let victim = (0..self.active_kernels.len())
                .filter(|&i| Some(i) != st.last_winner)
                .min_by_key(|&i| self.active_kernels[i].stats.last_useful)
                .expect("budget must allow at least two kernels");
            if let Some(canon) = st.canon.as_mut() {
                let key = canon_old(&self.active_kernels[victim]);
                if canon.get(&key) == Some(&(victim as u32)) {
                    canon.remove(&key);
                }
            }
            for &b in &self.active_kernels[victim].input_set {
                let b = b as usize;
                st.index[b].retain(|&x| x as usize != victim);
            }
            st.last_matches.retain(|&x| x != victim);
            st.last_hits.retain(|&x| x != victim);
            st.silent_counts.remove(&victim);
            st.sticky_tags.remove(&victim);
            st.last_misses.retain(|&x| x != victim);
            if let Some(f) = st.fast.as_mut() {
                if let Some(u) = f.until.get_mut(victim) {
                    *u = 0;
                }
            }
            self.active_kernels[victim] = k;
            self.recycled += 1;
            victim
        };

        if st.index.len() < input.bit_len() {
            st.index.resize(input.bit_len(), Vec::new());
        }
        let frame_bits = cfg.frame_words * 64;
        for &b in &self.active_kernels[slot].input_set {
            let b = b as usize;
            st.index[b].push(slot as u32);
            // Copy credit at birth: the kernel is grown to predict `target` from this
            // input, so its bits that carry the target's bits are tagged at once (before
            // a near miss could prune them).
            let t = b % frame_bits;
            if st.sticky_factor > 0 && t < target.bit_len() && target.bit_get(t) {
                st.sticky_tags.entry(slot).or_default().insert(b);
            }
        }
        // the slot's connections changed (new kernel, or a recycled one replaced)
        if let Some(f) = st.frame_memo.as_mut() {
            f.note(slot);
        }
        if let Some(canon) = st.canon.as_mut() {
            let kern = &self.active_kernels[slot];
            canon.insert(connection_key(&kern.input_set, &kern.output_set), slot as u32);
        }
        slot
    }
}

/// Whether frame `f` (0 = most recent) of a concatenated input has any active bit.
fn frame_has_bits(input: &BitVector, f: usize, frame_words: usize) -> bool {
    input.as_words().iter().skip(f * frame_words).take(frame_words).any(|&w| w != 0)
}

/// Smoothed hit rate of a predictive kernel.
/// Integer form of `reliability(s) < 2^k / (2^k + 1)`, with no divide:
/// (h+1)/(h+m+2) < 2^k/(2^k+1)  ⇔  (h+1)(2^k+1) < 2^k (h+m+2)  ⇔  h+1 < 2^k (m+1).
fn unreliable(s: &KernelStats, k: u32) -> bool {
    ((s.misses as u32 + 1) << k) > s.hits as u32 + 1
}

/// True if `target`'s bits are (at least half) present in some frame of `input`: a new
/// kernel could key on or copy them.
fn target_in_frames(input: &BitVector, target: &BitVector, frame_words: usize) -> bool {
    let fb = frame_words * 64;
    let bits: Vec<usize> = (0..target.bit_len().min(fb)).filter(|&b| target.bit_get(b)).collect();
    if bits.is_empty() {
        return false;
    }
    (0..input.bit_len() / fb).any(|f| bits.iter().filter(|&&b| input.bit_get(f * fb + b)).count() * 2 >= bits.len())
}

/// A fixed pseudo-random rank for an input bit (splitmix64): canonical sampling takes the
/// lowest-ranked active bits.
fn bit_rank(b: usize) -> u64 {
    let mut z = (b as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Hash of a kernel's connections (sorted input and output positions).
fn connection_key(input: &[u32], output: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in input {
        h = (h ^ x as u64).wrapping_mul(0x0100_0000_01b3);
    }
    h = (h ^ 0xFFFF_FFFF).wrapping_mul(0x0100_0000_01b3);
    for &x in output {
        h = (h ^ x as u64).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// A 64-bit hash of an input's active bits (word index and word, mixed).
fn input_hash(input: &BitVector) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (wi, &w) in input.as_words().iter().enumerate() {
        if w != 0 {
            h = (h ^ (wi as u64)).wrapping_mul(0x0100_0000_01b3);
            h = (h ^ w).wrapping_mul(0x0100_0000_01b3).rotate_left(29);
        }
    }
    h
}

/// Sorted-list subset test.
fn is_subset(small: &[u32], big: &[u32]) -> bool {
    let mut j = 0;
    for &x in small {
        while j < big.len() && big[j] < x {
            j += 1;
        }
        if j == big.len() || big[j] != x {
            return false;
        }
        j += 1;
    }
    true
}

fn reliability(s: &KernelStats) -> f32 {
    Rate::of(s).value()
}

/// A hit rate kept as an exact fraction of integers, num / den. Rates are compared by
/// cross-multiplication (a/b < c/d ⇔ a·d < c·b, both denominators positive), so ranking
/// and thresholds need no divide and no floats. `value` converts to f32 for readouts only
/// (confidence, target probability).
///
/// Sizes: with 8-bit hit / miss counters, num = hits + 1 ≤ 256 and den = hits + misses + 2
/// ≤ 512 fit in u16, and a cross-product is at most 256 · 512 = 2^17, so u32 holds it.
/// Thresholds (`new`) must also stay below 2^15 / 2^15 so their products fit.
#[derive(Clone, Copy, Debug)]
pub struct Rate {
    num: u16,
    den: u16,
}

impl Rate {
    pub fn new(num: u16, den: u16) -> Self {
        Self { num, den: den.max(1) }
    }

    /// A kernel's smoothed hit rate, (hits + 1) / (hits + misses + 2).
    pub fn of(s: &KernelStats) -> Self {
        Self::new(s.hits as u16 + 1, s.hits as u16 + s.misses as u16 + 2)
    }

    /// The rate as a float, for readouts (never used in a comparison).
    pub fn value(self) -> f32 {
        self.num as f32 / self.den as f32
    }
}

impl PartialEq for Rate {
    fn eq(&self, other: &Self) -> bool {
        self.num as u32 * other.den as u32 == other.num as u32 * self.den as u32
    }
}

impl Eq for Rate {}

impl PartialOrd for Rate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.num as u32 * other.den as u32).cmp(&(other.num as u32 * self.den as u32))
    }
}

/// True if `target` confirms at least half of `k`'s output bits.
fn predicts(k: &SimpleKernel, target: &BitVector) -> bool {
    // only the output's own bits are checked (event-style: ~32 bit tests, not a full-width AND)
    let hit = k.output_set.iter().filter(|&&b| (b as usize) < target.bit_len() && target.bit_get(b as usize)).count();
    hit * 2 >= k.output_set.len()
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
    #[test]
    fn eight_bit_counters_halve_together_and_keep_the_ratio() {
        let mut s = KernelStats::default();
        for i in 0..3000 {
            if i % 4 == 3 {
                s.record_miss();
            } else {
                s.record_hit();
            }
        }
        // 3:1 hits to misses, held in 8 bits after many halvings
        assert!(s.hits > 100 && s.hits as u32 >= 2 * s.misses as u32 && (s.hits as u32) <= 4 * (s.misses as u32 + 1));
        let mut t = KernelStats { hits: 255, misses: 10, ..Default::default() };
        t.record_hit();
        assert_eq!((t.hits, t.misses), (128, 5)); // both halved, then the hit counted
    }

    #[test]
    fn rates_order_like_the_fractions_they_hold() {
        // cross-multiplication gives the exact order of (hits+1)/(hits+misses+2)
        let stats: Vec<KernelStats> = (0..=255u8)
            .step_by(5)
            .flat_map(|h| (0..=255u8).step_by(3).map(move |m| KernelStats { hits: h, misses: m, ..Default::default() }))
            .collect();
        for a in stats.iter().step_by(7) {
            for b in &stats {
                let rate = |s: &KernelStats| (s.hits as f64 + 1.0) / (s.hits as f64 + s.misses as f64 + 2.0);
                let (fa, fb) = (rate(a), rate(b));
                assert_eq!(Rate::of(a).cmp(&Rate::of(b)), fa.partial_cmp(&fb).unwrap());
            }
        }
        // a floor of 1/2: equal rates pass (≥), lower ones do not
        let half = Rate::new(1, 2);
        assert!(Rate::of(&KernelStats { hits: 3, misses: 3, ..Default::default() }) >= half);
        assert!(Rate::of(&KernelStats { hits: 2, misses: 3, ..Default::default() }) < half);
    }

    #[test]
    fn integer_reliability_gate_matches_the_ratio() {
        // (misses+1) << k > hits+1  ⇔  (hits+1)/(hits+misses+2) < 2^k/(2^k+1), exactly
        for k in 0..6u32 {
            let cut = (1u64 << k) as f64 / ((1u64 << k) + 1) as f64;
            for hits in 0..=255u8 {
                for misses in 0..=255u8 {
                    let s = KernelStats { hits, misses, ..Default::default() };
                    let rate = (hits as f64 + 1.0) / (hits as f64 + misses as f64 + 2.0);
                    assert_eq!(unreliable(&s, k), rate < cut, "k={k} hits={hits} misses={misses}");
                }
            }
        }
    }

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
    fn growth_gate_stops_growing_guesses_in_a_known_random_context() {
        // context A is followed by one of four words at random; no frame carries the
        // target, so once A's kernels are known to be unreliable the gate stops growth
        // (with a varying previous word, so deeper kernels keep being grown for every
        // (previous word, outcome) pair without the gate)
        let cfg = GrowthConfig { frame_words: 1, max_frames: 2, sample_bits: 8, ..GrowthConfig::default() };
        let targets: Vec<BitVector> = (1..5u64).map(|i| BitVector::from_words(vec![0xFF << (8 * i)])).collect();
        let mut run = |gate: Option<(u32, u16)>| {
            let mut kc = KernelClass::predictive(cfg);
            kc.set_growth_gate(gate);
            let mut x: u64 = 12345;
            for _ in 0..2000 {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let prev = 1u64 << (32 + (x >> 33) % 32); // one of 32 previous words
                let ctx = frames(&[0xFF, prev]);
                step(&mut kc, &ctx, &targets[((x >> 50) % 4) as usize]);
            }
            (kc.live(), kc.gated_growth())
        };
        let (open, _) = run(None);
        let (gated, suppressed) = run(Some((3, 16)));
        // the gate fires once the context is known to be random, and never adds kernels
        // (the pile-up it prevents needs real history; see experiment 23)
        assert!(suppressed > 0);
        assert!(gated <= open, "gated {gated} vs open {open}");
    }

    #[test]
    fn canonical_growth_never_duplicates_a_kernel() {
        use rand::SeedableRng;
        let cfg = GrowthConfig { frame_words: 1, max_frames: 1, sample_bits: 4, ..GrowthConfig::default() };
        let mut kc = KernelClass::predictive(cfg);
        kc.set_canonical(true);
        let (ctx, b, c) = (frames(&[0xFFFF]), BitVector::from_words(vec![0xFF << 16]), BitVector::from_words(vec![0xFF << 32]));
        let mut r1 = rand::rngs::StdRng::seed_from_u64(1);
        let mut r2 = rand::rngs::StdRng::seed_from_u64(2);
        kc.grow(&ctx, &b, 1, &mut r1);
        kc.grow(&ctx, &b, 1, &mut r2); // different randomness, same canonical sample
        assert_eq!((kc.live(), kc.canon_reused()), (1, 1));
        kc.grow(&ctx, &c, 1, &mut r1); // a different output is a different kernel
        assert_eq!(kc.live(), 2);
        assert_eq!(kc.kernels()[0].input_set, kc.kernels()[1].input_set); // same canonical sample
    }

    #[test]
    fn frame_memo_gives_exactly_the_same_counts_through_growth_pruning_and_sleep() {
        use rand::{Rng, SeedableRng};
        let cfg = GrowthConfig { frame_words: 1, max_frames: 3, sample_bits: 8, generalize: Some(0.5), ..GrowthConfig::default() };
        let words: Vec<u64> = (0..8).map(|i| 0xFFu64 << (8 * i)).collect();
        let mut src = rand::rngs::StdRng::seed_from_u64(5);
        let seq: Vec<usize> = (0..4000).map(|i| if src.gen_bool(0.8) { (i * 3) % 8 } else { src.gen_range(0..8) }).collect();
        let run = |frame_memo: bool| {
            let mut kc = KernelClass::predictive(cfg);
            kc.set_frame_memo(frame_memo);
            kc.set_replay(64);
            let mut rng = rand::rngs::StdRng::seed_from_u64(9);
            let mut outs = Vec::new();
            for t in 3..seq.len() {
                if t % 500 == 0 {
                    kc.sleep();
                }
                let ctx = frames(&[words[seq[t - 1]], words[seq[t - 2]], words[seq[t - 3]]]);
                let mut out = BitVector::new(64, Some(0));
                kc.process(&ctx, &mut out, 0, 0);
                kc.feedback(&ctx, &frames(&[words[seq[t]]]), &mut rng);
                outs.push(out.as_words()[0]);
            }
            let stats: Vec<(u8, u8, usize)> = kc.kernels().iter().map(|k| (k.stats.hits, k.stats.misses, k.input_bits)).collect();
            (outs, stats, kc.frame_memo_stats())
        };
        let (plain, plain_stats, _) = run(false);
        let (memo, memo_stats, (lookups, hits, patched)) = run(true);
        assert_eq!(plain, memo);
        assert_eq!(plain_stats, memo_stats);
        assert!(hits + patched > lookups / 2, "lookups {lookups} hits {hits} patched {patched}");
    }

    #[test]
    fn spawned_general_kernels_transfer_without_losing_specifics() {
        // frame 0: a name (4 names, the last never trained), frame 1: a cue (2 cues)
        let name = |i: usize| 0xFFu64 << (8 * i);
        let cue = |c: usize| 0xFFu64 << (8 * (4 + c));
        let cfg = GrowthConfig { frame_words: 1, max_frames: 2, sample_bits: 8, generalize: Some(0.5), ..GrowthConfig::default() };
        // role rule: the target depends on the cue only; name rule: on name and cue
        let role_t = |_n: usize, c: usize| BitVector::from_words(vec![0xFFu64 << (8 * c)]);
        let name_t = |n: usize, c: usize| BitVector::from_words(vec![0xFFu64 << (8 * ((n * 2 + c) % 8))]);
        let run = |spawn: bool, generalize: bool, rule: &dyn Fn(usize, usize) -> BitVector| -> (usize, usize) {
            let mut kc = KernelClass::predictive(if generalize { cfg } else { GrowthConfig { generalize: None, ..cfg } });
            kc.set_generalize_spawn(spawn);
            for i in 0..120 {
                let (n, c) = (i % 3, (i / 3) % 2);
                step(&mut kc, &frames(&[name(n), cue(c)]), &rule(n, c));
            }
            // (known names right of 6, unseen name right of 2), predicting without learning
            let probe = |kc: &mut KernelClass<SimpleKernel>, n: usize, c: usize| {
                let mut out = BitVector::new(64, Some(0));
                kc.process(&frames(&[name(n), cue(c)]), &mut out, 0, 0);
                (out.as_words()[0] == rule(n, c).as_words()[0]) as usize
            };
            let known = (0..3).flat_map(|n| (0..2).map(move |c| (n, c))).map(|(n, c)| probe(&mut kc, n, c)).sum();
            let unseen = (0..2).map(|c| probe(&mut kc, 3, c)).sum();
            (known, unseen)
        };
        // role rule: spawning transfers to the unseen name; without generalisation it cannot
        assert_eq!(run(true, true, &role_t), (6, 2));
        assert_eq!(run(false, false, &role_t).1, 0);
        // name rule: spawning keeps every known name right
        assert_eq!(run(true, true, &name_t).0, 6);
    }

    #[test]
    fn sleep_forms_general_rules_from_replay() {
        // as above: frame 0 a name (the fourth never trained), frame 1 a cue; no waking
        // generalisation, only sleep's
        let name = |i: usize| 0xFFu64 << (8 * i);
        let cue = |c: usize| 0xFFu64 << (8 * (4 + c));
        let cfg = GrowthConfig { frame_words: 1, max_frames: 2, sample_bits: 8, generalize: None, ..GrowthConfig::default() };
        let role_t = |_n: usize, c: usize| BitVector::from_words(vec![0xFFu64 << (8 * c)]);
        let name_t = |n: usize, c: usize| BitVector::from_words(vec![0xFFu64 << (8 * ((n * 2 + c) % 8))]);
        let run = |rule: &dyn Fn(usize, usize) -> BitVector, sleep_gen: bool| -> (usize, usize, (usize, usize)) {
            let mut kc = KernelClass::predictive(cfg);
            kc.set_replay(256);
            kc.set_sleep_generalize(sleep_gen.then_some(2));
            for i in 0..120 {
                let (n, c) = (i % 3, (i / 3) % 2);
                step(&mut kc, &frames(&[name(n), cue(c)]), &rule(n, c));
            }
            kc.sleep();
            let probe = |kc: &mut KernelClass<SimpleKernel>, n: usize, c: usize| {
                let mut out = BitVector::new(64, Some(0));
                kc.process(&frames(&[name(n), cue(c)]), &mut out, 0, 0);
                (out.as_words()[0] == rule(n, c).as_words()[0]) as usize
            };
            let known = (0..3).flat_map(|n| (0..2).map(move |c| (n, c))).map(|(n, c)| probe(&mut kc, n, c)).sum();
            let unseen = (0..2).map(|c| probe(&mut kc, 3, c)).sum();
            (known, unseen, kc.slept_general())
        };
        // role rule: sleep forms the cue-only rules, and the unseen name is answered
        let (known, unseen, (made, _)) = run(&role_t, true);
        assert_eq!((known, unseen), (6, 2));
        assert!(made > 0);
        assert_eq!(run(&role_t, false).1, 0);
        // name rule: candidates fail the replay test or lose to the specifics; known pairs stay
        assert_eq!(run(&name_t, true).0, 6);
    }

    #[test]
    fn memo_gives_exactly_the_same_predictions_and_learning() {
        use rand::{Rng, SeedableRng};
        let cfg = GrowthConfig { frame_words: 1, max_frames: 2, sample_bits: 8, generalize: Some(0.5), ..GrowthConfig::default() };
        let words: Vec<u64> = (0..6).map(|i| 0xFFu64 << (8 * i)).collect();
        // a sequence that is mostly predictable with some noise
        let mut src = rand::rngs::StdRng::seed_from_u64(11);
        let seq: Vec<usize> = (0..3000).map(|i| if src.gen_bool(0.85) { i % 6 } else { src.gen_range(0..6) }).collect();
        let run = |memo: bool| {
            let mut kc = KernelClass::predictive(cfg);
            kc.set_surprise_gate(true);
            kc.set_memo(memo);
            let mut rng = rand::rngs::StdRng::seed_from_u64(3);
            let mut outs = Vec::new();
            for t in 2..seq.len() {
                let ctx = frames(&[words[seq[t - 1]], words[seq[t - 2]]]);
                let mut out = BitVector::new(64, Some(0));
                kc.process(&ctx, &mut out, 0, 0);
                kc.feedback(&ctx, &frames(&[words[seq[t]]]), &mut rng);
                outs.push(out.as_words()[0]);
            }
            (outs, kc.len(), kc.memo_stats())
        };
        let (plain, n_plain, _) = run(false);
        let (memo, n_memo, (lookups, hits)) = run(true);
        assert_eq!(plain, memo);
        assert_eq!(n_plain, n_memo);
        assert!(hits > lookups / 4, "hits {hits} of {lookups}");
    }

    #[test]
    fn surprise_gate_confirms_expected_words_and_learns_from_surprises() {
        let cfg = GrowthConfig { frame_words: 1, max_frames: 1, sample_bits: 8, ..GrowthConfig::default() };
        let mut kc = KernelClass::predictive(cfg);
        kc.set_surprise_gate(true);
        let (a, b, c) = (frames(&[0xFF]), BitVector::from_words(vec![0xFF00]), BitVector::from_words(vec![0xFF_0000]));
        assert_eq!(step(&mut kc, &a, &b), 0); // surprise: grow A -> B
        for _ in 0..5 {
            assert_eq!(step(&mut kc, &a, &b), 0xFF00); // expected: confirm only
        }
        assert_eq!(kc.expected_steps(), 5);
        assert_eq!(kc.kernels()[0].stats.hits, 5);
        let before = kc.live();
        step(&mut kc, &a, &c); // surprise: the full path scores and grows
        assert!(kc.live() > before);
        assert_eq!(kc.kernels()[0].stats.misses, 1);
    }

    #[test]
    fn sleep_replay_merges_kernels_that_respond_to_the_same_inputs() {
        // two kernels read different halves of the same word (neither contains the other),
        // predict the same thing, and always fire together: replay finds them
        let cfg = GrowthConfig { frame_words: 1, max_frames: 1, sample_bits: 8, ..GrowthConfig::default() };
        let mut kc = KernelClass::predictive(cfg);
        kc.set_replay(64);
        let b = BitVector::from_words(vec![0xFF00]);
        let mut rng = rand::thread_rng();
        kc.grow(&frames(&[0x0F]), &b, 1, &mut rng);
        kc.grow(&frames(&[0xF0]), &b, 1, &mut rng);
        for i in 0..8 {
            step(&mut kc, &frames(&[0xFF]), &b); // the word: both fire
            step(&mut kc, &frames(&[0xFF << (16 + i % 4)]), &BitVector::from_words(vec![0])); // others: neither
        }
        let (_, merged) = kc.sleep();
        assert_eq!(merged, 1);
        assert_eq!(kc.live(), 1);
        assert_eq!(step(&mut kc, &frames(&[0xFF]), &b), 0xFF00);
    }

    #[test]
    fn sleep_merges_a_redundant_kernel_and_reuses_its_slot() {
        let cfg = GrowthConfig { frame_words: 1, max_frames: 1, sample_bits: 8, ..GrowthConfig::default() };
        let mut kc = KernelClass::predictive(cfg);
        let b = BitVector::from_words(vec![0xFF00]);
        let mut rng = rand::thread_rng();
        // a general kernel (bits 0-3) and a specific one (bits 0-7) for the same output
        kc.grow(&frames(&[0x0F]), &b, 1, &mut rng);
        kc.grow(&frames(&[0xFF]), &b, 1, &mut rng);
        for _ in 0..4 {
            step(&mut kc, &frames(&[0xFF]), &b); // both fire and are right
        }
        assert_eq!(kc.live(), 2);
        let (pruned, merged) = kc.sleep();
        assert_eq!((pruned, merged), (0, 1));
        assert_eq!(kc.live(), 1);
        assert_eq!(step(&mut kc, &frames(&[0xFF]), &b), 0xFF00); // the general one still predicts
        kc.grow(&frames(&[0xF0]), &b, 1, &mut rng);
        assert_eq!((kc.len(), kc.live()), (2, 2)); // the freed slot was reused
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
        let outs: Vec<u64> = kc.kernels().iter().map(|k| k.output_vector().as_words()[0]).collect();
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
        assert_eq!(kc2.kernels()[0].input_words(2).as_slice(), &[0xFF, 0]);
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
        assert_eq!(kc.kernels()[0].input_words(1)[0], 0x0F);
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
        assert_eq!(kc.kernels()[0].input_words(2).as_slice(), &[0xFF, 0x0F]);
        step(&mut kc, &a_f2, &b); // second: filler connections dropped
        assert_eq!(kc.kernels()[0].input_words(2).as_slice(), &[0xFF, 0]);
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