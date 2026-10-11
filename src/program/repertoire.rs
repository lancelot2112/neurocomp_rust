//! The motor repertoire a basal-ganglia channel chooses from, grown by babbling, and the
//! curiosity that makes babbling worth it.
//!
//! - **Candidates are proposed, not listed.** The motor cortex proposes a new motor command
//!   (a few random active bits). It joins the repertoire as a new action of the channel; the
//!   striatum makes go and no-go cells for it as it is used.
//! - **What a command does is learned by the motor area** (`MotorArea::observe`: its forward
//!   model, motor command → what the senses report, the same predictive kernels as the rest of
//!   the cortex). The repertoire only keeps, per command, how that prediction error has gone.
//! - **Curiosity is learning progress** (Oudeyer & Kaplan 2007; Schmidhuber 1991): a slow
//!   average of the command's prediction error minus a fast one, beyond how much the error
//!   usually varies. A command whose effect is
//!   being learned pays; once its effect is predicted it stops paying (boredom); one whose
//!   effect is noise never pays, because nothing is learned.
//! - **Cost:** each command costs a little effort (`EFFORT`).
//! - **Room is limited:** at most `cap` commands; a new proposal takes the place of the one
//!   used longest ago (the caller clears its striatal cells).
//!
//! Everything is integer: `Q16` averages and shifts.

use rand::Rng;

use crate::bitvec::BitVector;
use crate::fixed::{Q16, ONE};

/// Effort per command (`Q16`): a sixteenth of a reward.
pub const EFFORT: i32 = (ONE / 16) as i32;

struct Command {
    code: BitVector,
    bits: Vec<usize>,
    /// prediction error, fast and slow averages (`Q16`)
    fast: u32,
    slow: u32,
    /// how much the error usually varies (mean absolute deviation from the slow average)
    dev: u32,
    /// what it has been worth: a running average of the dopamine that followed it (`Q16`)
    worth: i32,
    last_used: u64,
}

pub struct Repertoire {
    commands: Vec<Command>,
    pub cap: usize,
    bits: usize,
    active: usize,
    clock: u64,
    /// (proposals, commands replaced)
    pub stats: (u64, u64),
}

impl Repertoire {
    /// Commands of `active` bits among `bits`, at most `cap` of them.
    pub fn new(bits: usize, active: usize, cap: usize) -> Self {
        Self { commands: Vec::new(), cap, bits, active, clock: 0, stats: (0, 0) }
    }

    /// The motor cortex proposes a new command: half the time (once something has paid) a
    /// variation of the command worth most so far (two of its bits moved: exploration around
    /// what works, as in song learning), else a random one. Returns its index and whether it
    /// replaced an older one (the one worth least, of those worth no more than nothing first,
    /// the oldest among equals).
    pub fn propose<R: Rng>(&mut self, rng: &mut R) -> (usize, bool) {
        let best = (0..self.commands.len()).filter(|&i| self.commands[i].worth > 0).max_by_key(|&i| self.commands[i].worth);
        let mut bits: Vec<usize> = match best {
            Some(b) if rng.gen_bool(0.5) => {
                let mut v = self.commands[b].bits.clone();
                for _ in 0..2 {
                    if !v.is_empty() {
                        let k = rng.gen_range(0..v.len());
                        v[k] = rng.gen_range(0..self.bits);
                    }
                }
                v
            }
            _ => (0..self.active).map(|_| rng.gen_range(0..self.bits)).collect(),
        };
        bits.sort_unstable();
        bits.dedup();
        let c = Command { code: BitVector::from_bits(&bits, self.bits), bits: bits.clone(), fast: ONE, slow: ONE, dev: 0, worth: 0, last_used: self.clock };
        self.stats.0 += 1;
        if self.commands.len() < self.cap {
            self.commands.push(c);
            (self.commands.len() - 1, false)
        } else {
            let i = (0..self.commands.len()).min_by_key(|&i| (self.commands[i].worth.max(0), self.commands[i].last_used)).unwrap();
            self.commands[i] = c;
            self.stats.1 += 1;
            (i, true)
        }
    }

    /// Command `i` was issued and the motor area's forward model mispredicted its result by
    /// `err` (`Q16`). Returns the curiosity reward (`Q16`): the learning progress.
    pub fn observe(&mut self, i: usize, err: Q16) -> i32 {
        self.clock += 1;
        let c = &mut self.commands[i];
        c.last_used = self.clock;
        let err = err.min(ONE);
        c.fast = c.fast - c.fast / 4 + err / 4;
        c.slow = c.slow - c.slow / 16 + err / 16;
        c.dev = c.dev - c.dev / 16 + err.abs_diff(c.slow) / 16;
        // progress beyond the error's usual variation (a noisy error's dips are not progress)
        (c.slow as i32 - c.fast as i32 - c.dev as i32).max(0)
    }

    /// The dopamine that followed command `i` (`Q16`): its worth moves an eighth of the way.
    pub fn credit(&mut self, i: usize, dopamine: i32) {
        let c = &mut self.commands[i];
        c.worth += (dopamine - c.worth) / 8;
    }

    /// What command `i` has been worth (`Q16`).
    pub fn worth(&self, i: usize) -> i32 {
        self.commands[i].worth
    }

    /// The motor code of command `i`.
    pub fn code(&self, i: usize) -> &BitVector {
        &self.commands[i].code
    }

    /// The prediction error of command `i` now (`Q16`, fast average).
    pub fn error(&self, i: usize) -> Q16 {
        self.commands[i].fast
    }

    pub fn len(&self) -> usize {
        self.commands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::MotorArea;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// With the motor area's forward model: a command with a fixed effect pays while its effect
    /// is learned, then stops (boredom); a command whose effect is random never pays much.
    #[test]
    fn curiosity_is_learning_progress() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut motor = MotorArea::new(256);
        let mut r = Repertoire::new(256, 8, 8);
        let (fixed, _) = r.propose(&mut rng);
        let (noisy, _) = r.propose(&mut rng);
        let effect = BitVector::from_bits(&(100..124).collect::<Vec<_>>(), 256);
        let (mut early, mut late, mut noise, mut noise_late) = (0i64, 0i64, 0i64, 0i64);
        for t in 0..200 {
            let code = r.code(fixed).clone();
            let err = motor.observe(&code, &effect, &mut rng);
            let c = r.observe(fixed, err) as i64;
            if t < 10 {
                early += c;
            }
            if t >= 150 {
                late += c;
            }
            let mut o: Vec<usize> = (0..24).map(|_| rng.gen_range(0..256)).collect();
            o.sort_unstable();
            o.dedup();
            let code = r.code(noisy).clone();
            let err = motor.observe(&code, &BitVector::from_bits(&o, 256), &mut rng);
            let c = r.observe(noisy, err) as i64;
            noise += c;
            if t >= 150 {
                noise_late += c;
            }
        }
        eprintln!("curiosity: learnable early {early}, late {late}; noise over all {noise}, late {noise_late}");
        assert!(early > 10 * late.max(1), "early {early} late {late}");
        assert!(early > 2 * noise, "fixed {early} noise {noise}");
        assert!(noise_late * 20 < early, "noise late {noise_late}");
        assert!(r.error(fixed) < ONE / 8, "{}", r.error(fixed));
        assert!(r.error(noisy) > ONE / 2, "{}", r.error(noisy));
    }

    /// A full repertoire replaces the command used longest ago.
    #[test]
    fn replaces_the_least_recently_used() {
        let mut rng = StdRng::seed_from_u64(4);
        let mut r = Repertoire::new(256, 8, 3);
        for _ in 0..3 {
            r.propose(&mut rng);
        }
        r.observe(0, ONE);
        r.observe(2, ONE);
        let (i, replaced) = r.propose(&mut rng);
        assert!(replaced);
        assert_eq!(i, 1);
    }
}
