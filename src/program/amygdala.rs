//! A fast learner of what cues are worth, and of which cues deserve attention (after the
//! basolateral and central amygdala), with the striatum's cholinergic interneurons' learned pause.
//!
//! - **Cues are onsets:** a feature (a cell of some sensing area's layer 4 code, as `slot * bits
//!   + cell`) that was not active the step before. Each onset leaves a trace for a few steps,
//!   tagged by its age (as time-locked "microstimuli": the cue at 1, 2, 3 steps back are different
//!   inputs), so value can attach to a cue across a delay.
//! - **Value** (basolateral amygdala, which gets sensory input directly and learns in a few
//!   pairings): per feature and age, Rescorla–Wagner toward the innate valuations that follow:
//!   v += α · (r − V(trace)) for each traced feature, where V is the mean over the trace.
//! - **Associability** (Pearce–Hall, with the central amygdala and cholinergic basal forebrain): a
//!   feature's α moves toward the size of the surprise that followed it, and a feature whose value
//!   has grown stays attended (Mackintosh: good predictors are attended). Only attended features
//!   reach the striatum, so a cue that predicts something is not swamped by everything that
//!   happens to be on screen; a new feature starts attended (novelty).
//! - **The cholinergic pause** (TANs: Aosaki et al. 1994; Morris et al. 2004): a feature in the
//!   recent trace when a large dopamine error came grows a pause weight (and loses it slowly
//!   otherwise); when a feature with a strong pause weight is present, the interneurons pause and
//!   striatal plasticity opens for whatever dopamine comes.

use std::collections::HashMap;

use crate::fixed::ONE;

/// Steps an onset stays in the trace.
pub const TRACE: usize = 4;
/// Features attended at most (reach the striatum).
const ATTEND: usize = 32;

#[derive(Clone, Copy, Default)]
struct Feature {
    /// value per age in the trace (`Q16`)
    value: [i32; TRACE],
    /// associability (`Q16`)
    alpha: i32,
    /// cholinergic pause weight (`Q16`)
    pause: i32,
    seen: bool,
}

pub struct Amygdala {
    features: HashMap<u32, Feature>,
    /// onsets of the last steps, newest first
    trace: Vec<Vec<u32>>,
    /// (updates, sum of |error|)
    pub stats: (u64, u64),
}

impl Default for Amygdala {
    fn default() -> Self {
        Self::new()
    }
}

impl Amygdala {
    pub fn new() -> Self {
        Self { features: HashMap::new(), trace: Vec::new(), stats: (0, 0) }
    }

    fn entry(&mut self, f: u32) -> &mut Feature {
        self.features.entry(f).or_insert(Feature { alpha: ONE as i32 / 2, ..Default::default() })
    }

    /// This step's onsets (features newly active). Shifts the trace.
    pub fn onsets(&mut self, onsets: Vec<u32>) {
        for &f in &onsets {
            self.entry(f).seen = true;
        }
        self.trace.insert(0, onsets);
        self.trace.truncate(TRACE);
    }

    /// What the trace predicts for this step (`Q16`): the mean over the features traced from
    /// earlier steps of their value at their age.
    pub fn predicted(&self) -> i32 {
        let mut sum = 0i64;
        let mut n = 0i64;
        for (age, fs) in self.trace.iter().enumerate().skip(1) {
            for f in fs {
                if let Some(x) = self.features.get(f) {
                    sum += x.value[age - 1] as i64;
                    n += 1;
                }
            }
        }
        if n == 0 {
            0
        } else {
            (sum / n) as i32
        }
    }

    /// The innate valuations `r` (`Q16`) came this step: each traced feature's value at its age
    /// moves toward them (a quarter of the error, times its associability), and its associability
    /// toward the size of the error.
    pub fn learn(&mut self, r: i32) {
        // the features traced at ages 1.. predicted this step (age 0 is this step's own onsets)
        let mut groups: Vec<(usize, Vec<u32>)> = Vec::new();
        for (age, fs) in self.trace.iter().enumerate().skip(1) {
            groups.push((age, fs.clone()));
        }
        for (age, fs) in groups {
            if fs.is_empty() {
                continue;
            }
            let v: i64 = fs.iter().map(|f| self.features.get(f).map_or(0, |x| x.value[age - 1]) as i64).sum::<i64>() / fs.len() as i64;
            let err = r - v as i32;
            self.stats.0 += 1;
            self.stats.1 += err.unsigned_abs() as u64;
            for f in fs {
                let x = self.entry(f);
                let a = x.alpha.max(ONE as i32 / 16) as i64;
                x.value[age - 1] += ((err as i64 * a) >> 18) as i32;
                x.alpha += (err.abs().min(ONE as i32) - x.alpha) / 8;
            }
        }
    }

    /// How much a feature deserves attention (`Q16`): its associability, or its largest learned
    /// value (a good predictor stays attended).
    fn attention(x: &Feature) -> i32 {
        let v = x.value.iter().map(|v| v.abs()).max().unwrap_or(0);
        x.alpha.max(2 * v)
    }

    /// The attended features in the trace, tagged by age (`feature * TRACE + age`): what the
    /// amygdala and the time-locked trace send the striatum. At most `ATTEND`, most attended first.
    pub fn attended(&self) -> Vec<usize> {
        let mut v: Vec<(i32, usize)> = Vec::new();
        for (age, fs) in self.trace.iter().enumerate() {
            for f in fs {
                if let Some(x) = self.features.get(f) {
                    let a = Self::attention(x);
                    if a >= ONE as i32 / 4 {
                        v.push((a, *f as usize * TRACE + age));
                    }
                }
            }
        }
        v.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        v.into_iter().take(ATTEND).map(|x| x.1).collect()
    }

    /// The value of some features if they came now (an outcome imagined: its features at age 0),
    /// `Q16`: the mean of what they have been worth one step on.
    pub fn value_of(&self, features: &[u32]) -> i32 {
        let vals: Vec<i64> = features.iter().filter_map(|f| self.features.get(f)).map(|x| x.value[0] as i64).collect();
        if vals.is_empty() {
            0
        } else {
            (vals.iter().sum::<i64>() / vals.len() as i64) as i32
        }
    }

    /// The dopamine error `delta` came (`Q16`): features in the last steps' trace learn whether
    /// they come before large errors (their pause weight). Returns whether the interneurons pause
    /// now: some feature in the trace has a pause weight above half.
    pub fn learn_pause(&mut self, delta: i32, large: i32) -> bool {
        // the cues one or two steps back came before this error; those present now and before
        // that learn it was small
        let before: Vec<u32> = self.trace.iter().skip(1).take(2).flatten().copied().collect();
        for f in before {
            let x = self.entry(f);
            if delta.abs() >= large {
                x.pause += (ONE as i32 - x.pause) / 8;
            } else {
                x.pause -= x.pause / 8;
            }
        }
        self.pausing()
    }

    /// The interneurons pause: a feature in the trace has a strong pause weight.
    pub fn pausing(&self) -> bool {
        self.trace.iter().take(3).flatten().any(|f| self.features.get(f).map_or(false, |x| x.pause > ONE as i32 / 2))
    }
}
