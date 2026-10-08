//! Running a genome.
//!
//! The genome builds the network (its last definition) and says what happens besides
//! ticking (its `Schedule`, set by genes). The reader is the world's side: it hands the
//! network one word at a time on input port 0, and the sentence clock on input port 1 (a
//! bit at the first word after a sentence ended, so modules that keep a sentence's
//! content know to move on), and says when a story ends and when the test begins.

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::bitvec::BitVector;
use crate::program::modules::{Ctx, Genome, Module, Network, Schedule};

pub struct Reader {
    pub net: Network,
    schedule: Schedule,
    rng: StdRng,
    /// Slow learning is on until the test.
    pub learning: bool,
    stories: usize,
    sentence_ended: bool,
}

impl Reader {
    pub fn new(genome: &Genome, seed: u64) -> Self {
        let (net, schedule) = genome.build_top_and_schedule();
        Self::from_network(net, schedule, seed)
    }

    /// Run a network built elsewhere (e.g. from a gene list) with its update loop.
    pub fn from_network(net: Network, schedule: Schedule, seed: u64) -> Self {
        Self { net, schedule, rng: StdRng::seed_from_u64(seed), learning: true, stories: 0, sentence_ended: false }
    }

    /// Read one word: the network ticks once. Returns its prediction of the next word
    /// (output port 0).
    pub fn read(&mut self, word: &BitVector) -> &BitVector {
        let empty = BitVector::EMPTY;
        let clock = if std::mem::take(&mut self.sentence_ended) { crate::program::modules::on() } else { BitVector::EMPTY };
        let mut ins = vec![word, &clock];
        ins.resize(self.net.n_inputs().max(1), &empty);
        self.net.tick(&ins, &mut Ctx { rng: &mut self.rng, learn: self.learning });
        self.net.output(0)
    }

    pub fn end_sentence(&mut self) {
        self.sentence_ended = true;
        if self.schedule.reset_at_sentence {
            self.net.reset();
        }
    }

    pub fn end_story(&mut self) {
        if self.schedule.reset_at_story {
            self.net.reset();
        }
        if self.learning {
            self.stories += 1;
            if self.schedule.sleep_every > 0 && self.stories % self.schedule.sleep_every == 0 {
                self.net.sleep();
            }
        }
    }

    /// Slow learning stops (only fast inhibition runs from here on).
    pub fn start_test(&mut self) {
        self.learning = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::modules::{column_code, NetOp, Prim};

    fn code(bits: usize, i: usize) -> BitVector {
        BitVector::from_bits(&(0..8).map(|j| (i * 8 + j) % bits).collect::<Vec<_>>(), bits)
    }

    #[test]
    fn a_column_genome_learns_a_sequence() {
        let bits = 256;
        let mut g = Genome::default();
        let mut ops = vec![NetOp::Num(1), NetOp::Set(crate::program::modules::Gene::ResetAtStory)];
        ops.extend(column_code(bits, 8));
        g.define("column", 1, ops);
        let mut r = Reader::new(&g, 1);
        for _ in 0..5 {
            for i in 0..4 {
                r.read(&code(bits, i));
            }
            r.end_story();
        }
        r.start_test();
        r.read(&code(bits, 0));
        let p = r.read(&code(bits, 1)).clone();
        let hit = p.as_words().iter().zip(code(bits, 2).as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>();
        assert!(hit >= 6, "after 1 comes 2");
    }

    #[test]
    fn an_empty_slot_reserves_a_frame() {
        let bits = 256;
        let mut g = Genome::default();
        g.define("row", 1, vec![NetOp::In(0), NetOp::Zero, NetOp::In(0), NetOp::Place(Prim::Concat(3)), NetOp::Out]);
        assert_eq!(Reader::new(&g, 1).read(&code(bits, 0)).bit_len(), 3 * bits);
    }

    #[test]
    fn the_genome_files_parse_and_build() {
        for (name, text, parts) in [
            ("column", include_str!("../../genomes/column.gen"), vec!["Delay", "Concat(3)", "Predictor"]),
            ("hierarchy", include_str!("../../genomes/hierarchy.gen"), vec!["Surprise", "Bag", "Window(4", "Concat(2)", "Predictor", "Delay", "Concat(3)", "Predictor"]),
        ] {
            let g = Genome::parse(text).unwrap_or_else(|e| panic!("{name}: {e}"));
            let (net, schedule) = g.build_top_and_schedule();
            let d = net.describe(0);
            let mut at = 0;
            for p in parts {
                at = d[at..].find(p).map(|i| at + i + p.len()).unwrap_or_else(|| panic!("{name}: {p} missing or out of order in\n{d}"));
            }
            assert!(schedule.reset_at_story && schedule.sleep_every == 500, "{name}: its update loop");
            assert_eq!(net.n_outputs(), 1);
        }
        assert!(Genome::parse("def x 1 in:0 make:nothing out end").is_err());
        assert!(Genome::parse("def x 1 in:0 out").is_err());
    }
}
