//! A claustrum-like selector (`Claustrum`): what the cortex holds is chosen across areas, one
//! item a step at most.
//!
//! The claustrum is reciprocally connected with nearly all of the cortex and linked to resisting
//! distraction (Atlan et al. 2018); acetylcholine's release, which switches an area from
//! completing what it holds to taking in what came, is gated by salience, the central amygdala
//! driving the basal forebrain. Here every area with a held population proposes what surprised
//! it (`Column::proposal`), each proposal has a salience (how much surprised it, weighted by how
//! associable the cues are), and the selector lets the most salient through, if it stands out
//! from what is usual (half again the running mean of the best salience over every step, a
//! step with no surprise counting 0) and clears a bar that rises toward each item let through and
//! relaxes over about 32 steps (refractory: soon after a load, only something as salient), the
//! others not. Called every step.

pub struct Claustrum {
    /// the usual best salience a step (running mean over every step, ×16)
    mean: u64,
    /// the bar an item must clear (×16): raised toward each item let through, relaxing
    bar: u64,
    /// (steps with a proposal, items let through)
    pub stats: (u64, u64),
}

impl Default for Claustrum {
    fn default() -> Self {
        Self::new()
    }
}

impl Claustrum {
    pub fn new() -> Self {
        Self { mean: 0, bar: 0, stats: (0, 0) }
    }

    /// The proposals' saliences this step (0: none; call every step): which one, if any, is let
    /// through.
    pub fn choose(&mut self, salience: &[u32]) -> Option<usize> {
        let (best, s) = salience.iter().enumerate().max_by_key(|x| *x.1).map(|(i, &s)| (i, s)).unwrap_or((0, 0));
        let s16 = s as u64 * 16;
        let pass = s > 0 && s16 * 2 >= self.mean * 3 && s16 >= self.bar;
        self.mean = self.mean - self.mean / 32 + s16 / 32;
        self.bar -= self.bar / 32;
        if pass {
            self.bar += (s16 - self.bar.min(s16)) / 2;
        }
        if s == 0 {
            return None;
        }
        self.stats.0 += 1;
        if pass {
            self.stats.1 += 1;
            Some(best)
        } else {
            None
        }
    }
}
