//! The orbitofrontal cortex as a cortical area (`Ofc`): one more column, made of what every
//! column is made of, differing in what it is connected to and what it predicts.
//!
//! - **Inputs** (layer 4, four sources): the association areas' code (the cues), the efference
//!   copy (the act just made), affective touch (the outcome as sensed), and the hippocampus's
//!   recall at the top of the association hierarchy (context).
//! - **Layer 2/3** predicts the next input from this one, its own previous activity and, on its
//!   tuft, the state it holds. Its active cells are the area's state **Z**: the same cue after a
//!   different act or outcome history is coded by different cells, as the basal context differs.
//! - **The outcome layer** predicts the next outcome code (**Ô**, which touch will come) from
//!   the same row: the orbitofrontal cortex's outcome expectation, learned as any column learns.
//! - **Holding.** What the area holds (on its tuft, until replaced) is chosen by a loop through
//!   the basal ganglia and the mediodorsal thalamus, as prefrontal working memory is gated
//!   (O'Reilly & Frank's PBWM); the loop is the striatum's, learned from dopamine.

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
    /// the row the outcome layer last predicted from (input, context), and what it predicted
    last: Option<(BitVector, BitVector, Option<BitVector>)>,
    /// how surprising the latest outcome was (`Q16`): 1 for a touch not expected or an
    /// expected one that did not come, the share of the touch missed otherwise; 0 for
    /// nothing expected and nothing come
    pub surprise: u32,
    /// how well the latest input was expected by layer 2/3 (`Q16`, the share of its bits)
    pub familiar: u32,
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
            last: None,
            surprise: 0,
            familiar: 0,
            rng: StdRng::seed_from_u64(seed ^ 0x0FC1),
            stats: (0, 0, 0, 0),
        }
    }

    /// The tuft's context: what is held.
    fn tuft(&self) -> BitVector {
        self.held.clone()
    }

    /// One step: `inputs` by source (cues, act, outcome, context; an empty one is silent). The
    /// outcome layer first learns what outcome followed its last prediction (and how surprising
    /// it was); layer 4 codes the input, layer 2/3 learns and predicts; the outcome layer
    /// predicts the next outcome. Returns the state Z.
    pub fn step(&mut self, inputs: &[BitVector; 4], learn: bool) -> BitVector {
        let got = &inputs[OUTCOME];
        // what came against what was expected
        if let Some((inp, ctx, pred)) = self.last.take() {
            let n = got.count_ones() as u32;
            let one = crate::fixed::ONE;
            self.surprise = match (&pred, n) {
                (None, 0) => 0,
                (Some(_), 0) | (None, _) => one,
                (Some(p), n) => one - overlap(p, got).min(n) * one / n,
            };
            if n > 0 || pred.is_some() {
                self.stats.0 += 1;
                self.stats.1 += (self.surprise < one / 2) as u64;
            }
            if learn && n > 0 {
                self.outcome.learn(&inp, Some(&ctx), None, got, &mut self.rng);
            }
        }
        let srcs: Vec<(usize, &BitVector)> = inputs.iter().enumerate().filter(|(_, x)| x.count_ones() > 0).collect();
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
                self.held = z;
                self.stats.2 += 1;
            }
        }
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
