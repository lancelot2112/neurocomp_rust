//! The latent-cause model as attractors, on its own: which context the outcomes say we are in.
//!
//! Context assemblies (layer 2/3 of the orbitofrontal cortex) compete by mutual inhibition. The
//! active one holds itself by recurrent excitation and does not leak. Nonreward (an outcome the
//! active assembly did not predict) drives the interneurons that silence it (Deco & Rolls
//! 2005). With the slate cleared, the stored assembly whose predictions the outcomes now fit
//! takes over (the silenced one held back while it is adapted), and when none fits for long
//! enough, new cells are recruited: a new latent cause (Gershman, Blei & Niv 2010).
//!
//! The assemblies' predictions are the worths each context has locked in per cue (learned
//! elsewhere and passed in, `Q16` per bit, -1 unknown, with the outcomes each learned from), so
//! the model can be driven by any outcome stream: the mind's, or a synthetic one to see how it
//! behaves alone (`examples/latent_alone.rs`).
//!
//! All integers: counts, shifts and compares.

use crate::fixed::ONE;

/// How long the integrators run: a fixed 16 acts (a reset at half the leaky count), or set by
/// volatility (the bit length of the expected run between changes, as an opponent tally).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scale {
    Fixed,
    Volatile,
}

/// The acts a worth must have learned from to be locked in, and the recent-payout window: about
/// 16 expected rewards (a halving of the rate then stands two standard deviations out of the
/// count's noise: N/2 >= 2 sqrt(N) at N = 16).
pub fn window(v: i32) -> i32 {
    if v > 0 {
        (16 * ONE as i64 / v as i64).clamp(16, i32::MAX as i64) as i32
    } else {
        16
    }
}

#[derive(Clone, Debug)]
pub struct LatentCause {
    pub scale: Scale,
    /// the active context
    pub now: usize,
    /// the contexts known (assemblies recruited)
    pub contexts: usize,
    /// changes of context so far, and silencings (a silencing may settle back on the same one)
    pub switches: u32,
    pub resets: u32,
    /// each assembly's input drive: a leaky count of the acts it predicted
    score: Vec<i32>,
    /// unexplained acts while unsettled
    new: i32,
    /// what each cue paid lately (shared: every assembly sees the same outcomes), and how often
    /// it was in view
    recent: Vec<i32>,
    seen: Vec<u32>,
    /// the active assembly's nonreward
    nonreward: i32,
    /// silenced and not yet replaced
    unsettled: bool,
    /// the assembly silenced last and the acts until its adaptation wears off
    adapted: (usize, u32),
    /// volatility: acts since the last reset, and the usual gap between resets (0: none yet)
    since: u32,
    gap: u32,
    pub debug: bool,
}

impl LatentCause {
    pub fn new(bits: usize, scale: Scale) -> Self {
        Self {
            scale,
            now: 0,
            contexts: 1,
            switches: 0,
            resets: 0,
            score: vec![0],
            new: 0,
            recent: vec![-1; bits],
            seen: vec![0; bits],
            nonreward: 0,
            unsettled: false,
            adapted: (0, 0),
            since: 0,
            gap: 0,
            debug: false,
        }
    }

    /// Whether the active assembly has been silenced and none has taken over yet.
    pub fn unsettled(&self) -> bool {
        self.unsettled
    }

    /// The time scale now: the integration length and the count that silences or recruits.
    pub fn time_scale(&self) -> (i32, i32) {
        match self.scale {
            Scale::Volatile => {
                let run = self.gap.max(self.since);
                let h = (32 - run.leading_zeros() as i32).max(2);
                (h, h * ONE as i32)
            }
            Scale::Fixed => (16, 8 * ONE as i32),
        }
    }

    /// How assembly `c` answers an act: over the cues in view whose worth it has locked in,
    /// `Some(false)` when any paid lately outside a factor of two of its worth, `Some(true)` when
    /// all kept to it, `None` when it has locked in none of what is in view.
    pub fn answer(&self, worth: &[Vec<i32>], learned: &[Vec<u16>], c: usize, bits: &[usize]) -> Option<bool> {
        let vs = worth.get(c)?;
        let mut answer = None;
        for &b in bits.iter().filter(|&&b| vs[b] >= 0) {
            let (v, n) = (vs[b], learned[c][b] as i32);
            if 2 + n < window(v) {
                continue;
            }
            let r = self.recent[b];
            if 2 * r >= v && r <= 2 * v {
                answer = Some(true);
            } else {
                return Some(false);
            }
        }
        answer
    }

    /// One act's outcome (`got`, `Q16`, 0..=ONE) at the cues in view; returns the context the
    /// outcomes now say (a change when it differs from before).
    pub fn step(&mut self, worth: &[Vec<i32>], learned: &[Vec<u16>], bits: &[usize], got: i32) -> usize {
        // what each cue in view paid lately: over about 16 expected rewards at it, at the best
        // worth any assembly gives it (shared by all, so not slowed by the active one's
        // indifference to the cue)
        for &b in bits {
            let v = worth.iter().map(|vs| vs[b]).max().unwrap_or(-1);
            let step = (2 + self.seen[b].min(i32::MAX as u32 / 2) as i32).min(window(v));
            self.seen[b] = self.seen[b].saturating_add(1);
            let r = &mut self.recent[b];
            *r = if *r < 0 { got } else { *r + (got - *r) / step };
        }
        self.since = self.since.saturating_add(1);
        let (tau, full) = self.time_scale();
        let n_ctx = self.contexts.max(worth.len()).max(self.now + 1);
        self.contexts = n_ctx;
        self.score.resize(n_ctx, 0);
        let answers: Vec<Option<bool>> = (0..n_ctx).map(|c| self.answer(worth, learned, c, bits)).collect();
        for c in 0..n_ctx {
            self.score[c] += (if answers[c] == Some(true) { ONE as i32 } else { 0 }) - self.score[c] / tau;
        }
        self.adapted.1 = self.adapted.1.saturating_sub(1);
        let cur = self.now;
        let mut to = cur;
        if !self.unsettled {
            // the active assembly holds; only nonreward it did not predict can silence it, and
            // only acts it made a prediction for count
            if let Some(fit) = answers[cur] {
                self.nonreward = match self.scale {
                    // reward inhibits the nonreward cells: a fitting act that paid takes one
                    // off; a fitting act that paid nothing carries no news
                    Scale::Volatile => (self.nonreward + if !fit { ONE as i32 } else if got > 0 { -(ONE as i32) } else { 0 }).max(0),
                    Scale::Fixed => self.nonreward + (if fit { 0 } else { ONE as i32 }) - self.nonreward / tau,
                };
            }
            if self.nonreward >= full {
                self.unsettled = true;
                self.resets += 1;
                self.nonreward = 0;
                self.new = 0;
                self.adapted = (cur, tau as u32);
                self.gap = if self.gap == 0 { self.since } else { (self.gap + self.since) / 2 };
                self.since = 0;
            }
        }
        if self.unsettled {
            // settling: a stored assembly the outcomes now fit completes (the adapted one only
            // once its adaptation has worn off), the one with the most input drive first
            let adapted = self.adapted;
            let free = |c: usize| c != adapted.0 || adapted.1 == 0;
            let fit = (0..n_ctx).filter(|&c| answers[c] == Some(true) && free(c)).max_by_key(|&c| self.score[c]);
            self.new += (if fit.is_none() { ONE as i32 } else { 0 }) - if self.scale == Scale::Volatile { 0 } else { self.new / tau };
            if let Some(c) = fit {
                to = c;
                self.unsettled = false;
            } else if self.new >= full {
                // nothing stored explains what happens now: new cells
                self.score.push(0);
                self.contexts += 1;
                to = n_ctx;
                self.unsettled = false;
            }
            if !self.unsettled {
                self.new = 0;
            }
        }
        if self.debug {
            eprintln!(
                "attractor: got {:.2} scale {tau} answers {:?} drive {:?} nonreward {} unsettled {} new {} -> {to}",
                got as f64 / ONE as f64,
                answers,
                self.score.iter().map(|x| x / ONE as i32).collect::<Vec<_>>(),
                self.nonreward / ONE as i32,
                self.unsettled,
                self.new / ONE as i32
            );
        }
        if to != self.now {
            self.switches += 1;
            self.now = to;
        }
        to
    }
}
