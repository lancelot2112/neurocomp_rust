//! Running a genome: the initial configuration it builds, and its update loop.
//!
//! A `Genome` describes a network (its definitions; the last is the whole architecture)
//! and a `Schedule` (what the network does at each clock event). The `Reader` is the
//! environment's side: it feeds the network one input per step, reads its prediction,
//! and announces the clock events (a sentence ended, a story ended, learning stopped);
//! the genome's schedule decides what each event does. Nothing about the task is in the
//! reader: what is read and what is asked comes from outside.

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::bitvec::BitVector;
use crate::program::modules::{Ctx, Do, Genome, Module, Network, On, Schedule};

pub struct Reader {
    pub net: Network,
    schedule: Schedule,
    rng: StdRng,
    /// Whether slow learning is on (it is off at test).
    pub learning: bool,
    stories: usize,
}

impl Reader {
    /// Build the genome's architecture (its last definition) with its schedule.
    pub fn new(genome: &Genome, seed: u64) -> Self {
        Self { net: genome.build_top(), schedule: genome.schedule.clone(), rng: StdRng::seed_from_u64(seed), learning: true, stories: 0 }
    }

    /// One step: the network reads `x` on its input port 0 (other ports unconnected) and
    /// ticks once. Returns its output port 0 (its prediction of the next input).
    pub fn step(&mut self, x: &BitVector) -> &BitVector {
        let mut ins: Vec<&BitVector> = vec![x];
        let empty = BitVector::EMPTY;
        while ins.len() < self.net.n_inputs() {
            ins.push(&empty);
        }
        let mut ctx = Ctx { rng: &mut self.rng, learn: self.learning };
        self.net.tick(&ins, &mut ctx);
        self.net.output(0)
    }

    /// A clock event: apply the schedule's matching rules, in order.
    pub fn event(&mut self, ev: On) {
        if ev == On::TestStart {
            self.learning = false;
        }
        let mut fire = vec![ev];
        if ev == On::Story && self.learning {
            self.stories += 1;
            for &(on, _) in &self.schedule.rules {
                if let On::Stories(n) = on {
                    if n > 0 && self.stories % n == 0 && !fire.contains(&on) {
                        fire.push(on);
                    }
                }
            }
        }
        let actions: Vec<Do> = self.schedule.rules.iter().filter(|(on, _)| fire.contains(on)).map(|&(_, d)| d).collect();
        for d in actions {
            match d {
                Do::Reset => self.net.reset(),
                Do::Sleep => self.net.sleep(),
            }
        }
    }

    /// Kernels held by the whole network.
    pub fn kernels(&self) -> usize {
        self.net.kernels()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::modules::{column_code, NetOp};

    fn code(bits: usize, i: usize) -> BitVector {
        BitVector::from_bits(&(0..8).map(|j| (i * 8 + j) % bits).collect::<Vec<_>>(), bits)
    }

    #[test]
    fn a_column_genome_learns_a_sequence_and_its_schedule_resets_it() {
        let bits = 256;
        let mut g = Genome::default();
        g.define("column", 1, column_code(bits, 8));
        g.schedule.rules.push((On::Story, Do::Reset));
        let mut r = Reader::new(&g, 1);
        // the story 0 1 2 3, read five times
        for _ in 0..5 {
            for i in 0..4 {
                r.step(&code(bits, i));
            }
            r.event(On::Story);
        }
        r.event(On::TestStart);
        r.step(&code(bits, 0));
        let p = r.step(&code(bits, 1)).clone();
        let hit = p.as_words().iter().zip(code(bits, 2).as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>();
        assert!(hit >= 6, "after 1 comes 2");
        assert!(!r.learning);
        // an empty slot keeps a frame reserved
        let mut g2 = Genome::default();
        g2.define("row", 1, vec![NetOp::In(0), NetOp::Zero, NetOp::In(0), NetOp::Place(crate::program::modules::Prim::Concat(3)), NetOp::Out]);
        let mut r2 = Reader::new(&g2, 1);
        assert_eq!(r2.step(&code(bits, 0)).bit_len(), 3 * bits);
    }
}
