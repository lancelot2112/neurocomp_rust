//! The orbitofrontal cortex as a cortical area (`Ofc`): one more column, made of what every
//! column is made of, differing in what it is connected to and what it predicts.
//!
//! - **Inputs** (layer 4, four sources): the association areas' code (the cues), the efference
//!   copy (the act just made), affective touch (the outcome as sensed), and the hippocampus's
//!   recall at the top of the association hierarchy (context).
//! - **Layer 2/3** predicts the next input from this one, its own previous activity and, on its
//!   tuft, the state it holds. Its active cells are the area's state **Z**: the same cue after a
//!   different act or outcome history is coded by different cells, as the basal context differs.
//! - **The outcome layer** predicts the next outcome (**Ô**) from the same row: the
//!   orbitofrontal cortex's outcome expectation, learned as any column learns. Its target is the
//!   outcome's level on the labelled line (none, weak, middling, strong: a code per level), not
//!   where on the skin a stroke lands, which no prediction could know; so "nothing comes" is
//!   predicted too.
//! - **Holding.** What the area holds (on its tuft, until replaced) is chosen by a loop through
//!   the basal ganglia and the mediodorsal thalamus, as prefrontal working memory is gated
//!   (O'Reilly & Frank's PBWM); the loop is the striatum's, learned from dopamine.
//! - **Splitting on outcome surprise** (`split`): when an input the area expected well is
//!   followed by an outcome it badly mispredicted, and the cues are associable (a neuromodulatory
//!   gate: noradrenaline's network reset, Bouret & Sara 2005), the area holds its state now, the
//!   one that carries the surprising outcome. From then on the same cues meet a different tuft:
//!   their codes and the outcomes expected of them are learned in a new context, and the cells
//!   of the old one are not retrained.

use crate::bitvec::BitVector;
use crate::program::area::Pool;
use crate::program::column::{Column, Map};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Layer 4's sources.
pub const CUES: usize = 0;
pub const ACT: usize = 1;
pub const OUTCOME: usize = 2;
pub const CONTEXT: usize = 3;

fn overlap(a: &BitVector, b: &BitVector) -> u32 {
    a.as_words().iter().zip(b.as_words()).map(|(x, y)| (x & y).count_ones()).sum()
}

fn or(a: &BitVector, b: &BitVector) -> BitVector {
    BitVector::from_words(a.as_words().iter().zip(b.as_words()).map(|(x, y)| x | y).collect())
}

pub struct Ofc {
    bits: usize,
    pub col: Column,
    /// the next outcome (Ô), from the column's input and context
    pub outcome: Map,
    /// what the area holds on its tuft (empty: nothing)
    pub held: BitVector,
    /// with persistence (the column's own trace, `set_persistence`), what is held is the trace
    /// and holding refreshes it; else the separate copy `held`
    persist: bool,
    /// recurrence (`set_recurrence`)
    recur: bool,
    /// the row the outcome layer last predicted from (input, context), and what it predicted
    last: Option<(BitVector, BitVector, Option<BitVector>)>,
    /// how surprising the latest outcome was (`Q16`): 1 for a touch not expected or an
    /// expected one that did not come, the share of the touch missed otherwise; 0 for
    /// nothing expected and nothing come
    pub surprise: u32,
    /// how well the latest input was expected by layer 2/3 (`Q16`, the share of its bits)
    pub familiar: u32,
    /// `familiar` at the step before (when the outcome now checked was predicted)
    prev_familiar: u32,
    /// a code per outcome level (0: nothing)
    levels: Vec<BitVector>,
    rng: StdRng,
    /// (outcomes checked, right; holds; splits)
    pub stats: (u64, u64, u64, u64),
}

impl Ofc {
    /// An area of `bits` cells, `k` active, its layers up to `cap` cells.
    pub fn new(bits: usize, k: usize, cap: usize, seed: u64) -> Self {
        let z = BitVector::new(bits, Some(0));
        Self {
            bits,
            col: Column::granular(bits, Pool::new(4, bits, bits, k, 1, seed ^ 0x0FC0), cap),
            outcome: Map::new(bits, cap),
            held: z,
            persist: false,
            recur: false,
            last: None,
            surprise: 0,
            familiar: 0,
            prev_familiar: 0,
            levels: {
                let mut r = StdRng::seed_from_u64(seed ^ 0x0FC2);
                (0..4)
                    .map(|_| {
                        let mut v = BitVector::new(bits, Some(0));
                        while v.count_ones() < k {
                            v.bit_set(rand::Rng::gen_range(&mut r, 0..bits));
                        }
                        v
                    })
                    .collect()
            },
            rng: StdRng::seed_from_u64(seed ^ 0x0FC1),
            stats: (0, 0, 0, 0),
        }
    }

    /// Persistence: the column keeps its own trace (decaying with probability 2^-`tau` a bit a
    /// step); holding refreshes it, and the separate copy is not used.
    pub fn set_persistence(&mut self, tau: u32) {
        self.col.set_persistence(tau);
        self.persist = tau > 0;
    }

    /// Recurrence: the column holds its own state by self-excitation (`Column::set_recurrence`);
    /// holding or splitting opens its gate (the input becomes the state).
    pub fn set_recurrence(&mut self, gain: u32) {
        self.col.set_recurrence(gain);
        self.recur = gain > 0;
    }

    /// What is held: with recurrence the column's own state, else its trace, or the separate copy.
    pub fn held_state(&self) -> BitVector {
        if self.recur {
            return self.col.code.clone();
        }
        if self.persist {
            self.col.trace().clone()
        } else {
            self.held.clone()
        }
    }

    /// The tuft's context: what is held (with persistence the column puts its trace there itself).
    fn tuft(&self) -> BitVector {
        if self.persist || self.recur {
            BitVector::new(self.bits, Some(0))
        } else {
            self.held.clone()
        }
    }

    /// One step: `inputs` by source (cues, act, outcome as sensed, context; an empty one is
    /// silent) and the outcome's level now (0–3, 0: nothing). The outcome layer first learns what
    /// level followed its last prediction (and how surprising it was); layer 4 codes the input,
    /// layer 2/3 learns and predicts; the outcome layer predicts the next level. Returns the
    /// state Z.
    pub fn step(&mut self, inputs: &[BitVector; 4], level: usize, learn: bool) -> BitVector {
        let got = self.levels[level.min(3)].clone();
        // what came against what was expected (no expectation: surprising if anything came)
        if let Some((inp, ctx, pred)) = self.last.take() {
            let n = got.count_ones() as u32;
            let one = crate::fixed::ONE;
            self.surprise = match &pred {
                None => if level > 0 { one } else { 0 },
                Some(p) => one - overlap(p, &got).min(n) * one / n,
            };
            if level > 0 || pred.is_some() {
                self.stats.0 += 1;
                self.stats.1 += (self.surprise < one / 2) as u64;
            }
            if learn {
                self.outcome.learn(&inp, Some(&ctx), None, &got, &mut self.rng);
            }
        }
        let srcs: Vec<(usize, &BitVector)> = inputs.iter().enumerate().filter(|(_, x)| x.count_ones() > 0).collect();
        self.prev_familiar = self.familiar;
        let expected = self.col.expected.clone();
        let tuft = self.tuft();
        let code = self.col.sense(&srcs, Some(&tuft), learn, &mut self.rng);
        let n = code.count_ones() as u32;
        self.familiar = match (expected, n) {
            (_, 0) => 0,
            (Some(e), n) => overlap(&e, &code).min(n) * crate::fixed::ONE / n,
            (None, _) => 0,
        };
        let ctx = or(&self.col.activity(), &tuft);
        // (the column's activity carries its trace when it persists)
        if n > 0 {
            let pred = self.outcome.predict(&code, Some(&ctx), None);
            self.last = Some((code, ctx, pred));
        }
        self.z()
    }

    /// The state: layer 2/3's activity (projected).
    pub fn z(&self) -> BitVector {
        self.col.activity()
    }

    /// The loop chose to hold the state now (or, `false`, to let it go).
    pub fn hold(&mut self, on: bool) {
        if on {
            let z = self.z();
            if z.count_ones() > 0 {
                if self.persist || self.recur {
                    self.col.refresh(false);
                } else {
                    self.held = z;
                }
                self.stats.2 += 1;
            }
        }
    }

    /// Split on outcome surprise: if the latest outcome was badly mispredicted (at least ¾) after
    /// an input expected well (at least half) and `gate` is open, hold the state now (a new
    /// context). Returns whether it split.
    pub fn split(&mut self, gate: bool) -> bool {
        let one = crate::fixed::ONE;
        if gate && self.surprise >= one * 3 / 4 && self.prev_familiar >= one / 2 {
            let z = self.z();
            if z.count_ones() > 0 {
                if self.persist || self.recur {
                    self.col.refresh(true);
                } else {
                    self.held = z;
                }
                self.stats.3 += 1;
                return true;
            }
        }
        false
    }

    /// The outcome expected next (Ô), if any.
    pub fn expected_outcome(&self) -> Option<&BitVector> {
        self.last.as_ref().and_then(|l| l.2.as_ref())
    }

    /// Committed cells (layer 2/3 and the outcome layer).
    pub fn cells(&self) -> (usize, usize) {
        (self.col.l23.committed(), self.outcome.committed())
    }

    pub fn bits(&self) -> usize {
        self.bits
    }
}
