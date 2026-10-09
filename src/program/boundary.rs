//! Event boundaries learned from prediction error (event segmentation theory, Zacks et al.
//! 2007): an event ends where what comes next stops being predictable.
//!
//! - **One cell.** Its synapses come from the current input's code. Its activity is its
//!   expectation of how surprising the *next* input will be: the mean weight of the active
//!   synapses.
//! - **Learning.** When the next input arrives, the column's surprise at it is the target.
//!   Each synapse active on the previous step moves by a quarter of the error (delta rule,
//!   integer). Nothing names a token: inputs that precede unpredictable stretches come to
//!   drive the cell, whatever they are.
//! - **Firing.** The cell fires (a boundary) when its expectation is above its own running
//!   mean by more than its running mean deviation. A refractory span (`min_len` inputs) and
//!   a capacity (`max_len`: a segment longer than this is closed) bound the segments.

use crate::fixed::{Q16, ONE};

pub struct BoundaryCell {
    w: Vec<i32>,
    last: Vec<usize>,
    pred: i32,
    mean: i32,
    dev: i32,
    since: usize,
    min_len: usize,
    max_len: usize,
    /// the expectation of the latest input, in `Q16` (for reports)
    pub level: i32,
}

impl BoundaryCell {
    pub fn new(bits: usize, min_len: usize, max_len: usize) -> Self {
        let half = (ONE / 2) as i32;
        Self { w: vec![half; bits], last: Vec::new(), pred: half, mean: half, dev: 0, since: 0, min_len, max_len, level: half }
    }

    /// The surprise (`Q16`, 0..=ONE) of the input that followed the last one observed.
    pub fn learn(&mut self, surprise: Q16) {
        if self.last.is_empty() {
            return;
        }
        let err = surprise.min(ONE) as i32 - self.pred;
        for &b in &self.last {
            self.w[b] += err / 4;
        }
    }

    /// The current input's active bits; true if the event ends here.
    pub fn observe(&mut self, active: &[usize]) -> bool {
        let n = active.len().max(1) as i64;
        self.pred = (active.iter().map(|&b| self.w[b] as i64).sum::<i64>() / n) as i32;
        self.level = self.pred;
        self.last = active.to_vec();
        self.since += 1;
        let fire = self.since >= self.min_len && (self.pred > self.mean + self.dev || self.since >= self.max_len);
        // running mean and mean absolute deviation of the expectation (rate 1/64)
        self.mean += (self.pred - self.mean) / 64;
        self.dev += ((self.pred - self.mean).abs() - self.dev) / 64;
        if fire {
            self.since = 0;
        }
        fire
    }

    /// A segment closed from outside (the page ended).
    pub fn reset(&mut self) {
        self.since = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_to_fire_before_unpredictable_inputs() {
        // input 0 is always followed by a surprise, inputs 1..4 never
        let mut c = BoundaryCell::new(64, 2, 100);
        let code = |i: usize| (i * 8..i * 8 + 8).collect::<Vec<_>>();
        let mut fired = [0usize; 5];
        for step in 0..2000 {
            let i = step % 5;
            if step > 0 {
                c.learn(if i == 1 { ONE } else { 0 });
            }
            if c.observe(&code(i)) && step > 1000 {
                fired[i] += 1;
            }
        }
        assert!(fired[0] > 150, "{fired:?}");
        assert_eq!(fired[1..].iter().sum::<usize>(), 0, "{fired:?}");
    }
}
