//! Pattern separation by competition: cells with counting synapses compete for each input, the
//! `k` most driven fire (feedback inhibition), and learning makes overlapping inputs come apart.
//!
//! - Hebbian, for the winners: synapses from the input's lines strengthen, and the cell's other
//!   synapses weaken a little (its total stays bounded), so a winner comes to stand for the
//!   inputs it wins.
//! - Anti-Hebbian, for the cells that matched and lost: a cell the input drove at least half as
//!   much as the weakest winner (it would have fired but was held down) weakens its synapses from
//!   the input's lines (long-term depression).
//!
//! Inputs that overlap drive the same cells at first; each time one wins and the others are held
//! down, the losers lose exactly the lines the inputs share with the winner, so cells come to
//! answer to what is distinctive in each, and two inputs that share half their lines end up with
//! codes that barely overlap (as the dentate gyrus and cortical layer 4 separate patterns).
//! The inhibition is local in the input's own terms: only cells that matched the input (its
//! neighbours in pattern space) are depressed. And a cell's threshold rises with its own firing
//! (homeostasis): one that fires for every input, as the cells for what overlapping inputs
//! share would, is held back until cells that fire for one input alone win.
//!
//! All integer: counts, sums, compares.

use crate::bitvec::BitVector;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct Separator {
    n: usize,
    k: usize,
    bits: usize,
    /// synapses, `w[i * bits + b]`
    w: Vec<u8>,
    /// each cell's recent firing (0..=256: its share of inputs lately, times 256)
    rate: Vec<u32>,
}

impl Separator {
    /// `n` cells over `bits` input lines, `k` firing; each cell starts with weak synapses from a
    /// random `fan` of the lines.
    pub fn new(n: usize, k: usize, bits: usize, fan: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut w = vec![0u8; n * bits];
        for i in 0..n {
            for _ in 0..fan {
                w[i * bits + rng.gen_range(0..bits)] = 4;
            }
        }
        Self { n, k, bits, w, rate: vec![0; n] }
    }

    /// Code `input` (its winners as bits of an `n`-bit vector), learning if `learn`.
    pub fn code(&mut self, input: &BitVector, learn: bool) -> BitVector {
        let on: Vec<usize> = (0..self.bits).filter(|&b| input.bit_get(b)).collect();
        // the drive, divided by how far the cell fires above its share (k of n inputs): a cell
        // that fires for everything is held back (its own threshold rises with its firing:
        // homeostasis), so cells that would code every overlapping input come to code one
        let share = (256 * self.k / self.n).max(1) as u64;
        let mut drive: Vec<(u32, usize)> = (0..self.n)
            .map(|i| {
                let d = on.iter().map(|&b| self.w[i * self.bits + b] as u32).sum::<u32>() as u64;
                ((d * share / (share + self.rate[i] as u64)) as u32, i)
            })
            .collect();
        // the most driven fire (ties to the lower index, a fixed order)
        drive.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let winners: Vec<usize> = drive.iter().take(self.k).filter(|d| d.0 > 0).map(|d| d.1).collect();
        let weakest = drive.get(self.k.saturating_sub(1)).map_or(0, |d| d.0);
        if learn && !on.is_empty() {
            for (i, r) in self.rate.iter_mut().enumerate() {
                let won = winners.contains(&i);
                *r = if won { *r + (256 - (*r).min(256)) / 16 } else { *r - *r / 16 };
            }
            let on_set: Vec<bool> = {
                let mut v = vec![false; self.bits];
                for &b in &on {
                    v[b] = true;
                }
                v
            };
            for &i in &winners {
                let row = &mut self.w[i * self.bits..(i + 1) * self.bits];
                for (b, w) in row.iter_mut().enumerate() {
                    if on_set[b] {
                        *w = w.saturating_add(2);
                    } else if *w > 0 {
                        *w -= 1;
                    }
                }
            }
            // the cells that matched and were held down
            for &(d, i) in drive.iter().skip(self.k) {
                if d * 2 < weakest || d == 0 {
                    break;
                }
                for &b in &on {
                    let w = &mut self.w[i * self.bits + b];
                    *w = w.saturating_sub(1);
                }
            }
        }
        let mut v = BitVector::new(self.n, Some(0));
        for &i in &winners {
            v.bit_set(i);
        }
        v
    }
}
