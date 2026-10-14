//! Evolving graph programs in the world.
//!
//! A genome is a `GraphProgram` (the stack language that carves bit-vector nodes and kernel
//! edges) plus the few numbers that make each edge's kernels. The graph's first node reads
//! the world's senses and its last node is the act vector. Each edge gets `kernels` kernels:
//! an input mask of `syn` random lines over its source, a threshold, and an output mask (`out`
//! random lines over its destination, written with the genome's op). A kernel into the act node
//! writes one of the body's motor primitives instead (the body's pattern generators, given at
//! birth; the program only learns when to trigger them).
//!
//! Kernels learn as the base kernel does: one that fires moves a synapse from a silent line to
//! an active one (Hebbian); one that matched but was held down by an overlapping kernel moves a
//! synapse off its active lines (anti-Hebbian).
//!
//! Fitness is the observer's score of the world's events, over several world seeds. Nothing of
//! it reaches the program.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::bitvec::bithistory::AdvanceMode;
use crate::bitvec::BitVector;
use crate::kernel::{KernelClass, KernelGroup, KernelOp, SimpleKernel};
use crate::program::graph::{GraphInstruction, GraphProgram, ProgramDefaults};
use crate::program::neurocomp::{EdgeInfo, RuntimeNetwork};
use crate::world::{self, observe, Conditioning, World};

/// Every graph instruction, for mutation.
pub const INSTRUCTIONS: [GraphInstruction; 13] = [
    GraphInstruction::Stay(),
    GraphInstruction::Create(),
    GraphInstruction::Pop(),
    GraphInstruction::ReadInput(),
    GraphInstruction::WriteOutput(),
    GraphInstruction::Double(),
    GraphInstruction::Half(),
    GraphInstruction::Push(),
    GraphInstruction::Swap(),
    GraphInstruction::Dup(),
    GraphInstruction::Nop(),
    GraphInstruction::Jump(),
    GraphInstruction::Mark(),
];

const OPS: [KernelOp; 4] = [KernelOp::Or, KernelOp::And, KernelOp::Xor, KernelOp::Clear];

/// Bits of a hidden node (the program can double or halve it).
pub const NODE_BITS: usize = 512;

#[derive(Clone, Debug, PartialEq)]
pub struct Genome {
    pub code: Vec<GraphInstruction>,
    /// kernels per edge
    pub kernels: u8,
    /// input synapses per kernel
    pub syn: u8,
    /// active synapses needed to fire
    pub threshold: u8,
    /// output lines a hidden kernel writes
    pub out: u8,
    /// how hidden kernels write
    pub op: KernelOp,
    /// the wiring's random seed
    pub seed: u64,
}

impl Genome {
    pub fn random(rng: &mut StdRng) -> Self {
        let n = rng.gen_range(0..6);
        let syn = rng.gen_range(4..=32u8);
        Self {
            code: (0..n).map(|_| *INSTRUCTIONS.choose(rng).unwrap()).collect(),
            kernels: rng.gen_range(1..=16),
            syn,
            threshold: rng.gen_range(1..=syn.min(8)),
            out: rng.gen_range(4..=32),
            op: *OPS.choose(rng).unwrap(),
            seed: rng.r#gen(),
        }
    }

    /// One local change.
    pub fn mutate(&self, rng: &mut StdRng) -> Self {
        let mut g = self.clone();
        match rng.gen_range(0..9) {
            0 if g.code.len() < 48 => {
                let at = rng.gen_range(0..=g.code.len());
                g.code.insert(at, *INSTRUCTIONS.choose(rng).unwrap());
            }
            1 if !g.code.is_empty() => {
                let at = rng.gen_range(0..g.code.len());
                g.code.remove(at);
            }
            2 if !g.code.is_empty() => {
                let at = rng.gen_range(0..g.code.len());
                g.code[at] = *INSTRUCTIONS.choose(rng).unwrap();
            }
            3 => g.kernels = step(g.kernels, rng, 1, 32),
            4 => g.syn = step(g.syn, rng, 2, 64),
            5 => g.threshold = step(g.threshold, rng, 1, 64),
            6 => g.out = step(g.out, rng, 1, 64),
            7 => g.op = *OPS.choose(rng).unwrap(),
            _ => g.seed = rng.r#gen(),
        }
        g.threshold = g.threshold.min(g.syn);
        g
    }

    /// The network this genome builds for a world with these senses and acts.
    pub fn build(&self, w: &World, seed: u64) -> RuntimeNetwork {
        let defaults = ProgramDefaults { default_bits: NODE_BITS, min_bits: 64, default_threshold: self.threshold as usize, default_history_depth: 1, default_op: self.op };
        let prog = GraphProgram { code: self.code.clone() };
        let graph = prog.build_graph(&defaults);
        let body = w.body();
        let primitives: Vec<BitVector> = (0..body.voice.len())
            .map(|i| w.act(Some(i), None))
            .chain((0..body.hand.len()).map(|i| w.act(None, Some(i))))
            .collect();
        RuntimeNetwork::from_graph_with(&graph, world::SENSE_BITS, world::ACT_BITS, seed, |e: &EdgeInfo| {
            let mut r = StdRng::seed_from_u64(self.seed ^ (e.index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let kernels = (0..self.kernels)
                .map(|_| {
                    let syn = sparse(e.src_bits, self.syn as usize, &mut r);
                    let (out, op) = if e.to_output {
                        (primitives.choose(&mut r).unwrap().clone(), KernelOp::Or)
                    } else {
                        (sparse(e.dst_bits, self.out as usize, &mut r), self.op)
                    };
                    SimpleKernel::new(syn, 0, out, 0, self.threshold as usize, op)
                })
                .collect();
            KernelGroup::from_classes(vec![KernelClass::with_kernels(kernels)])
        })
    }
}

fn step(v: u8, rng: &mut StdRng, lo: u8, hi: u8) -> u8 {
    let d: i16 = if rng.gen_bool(0.5) { 1 } else { -1 } * rng.gen_range(1..=2);
    (v as i16 + d).clamp(lo as i16, hi as i16) as u8
}

fn sparse(bits: usize, k: usize, rng: &mut StdRng) -> BitVector {
    let all: Vec<usize> = (0..bits).collect();
    BitVector::from_bits(&all.choose_multiple(rng, k.min(bits)).copied().collect::<Vec<_>>(), bits)
}

/// The task a genome is scored on.
#[derive(Clone, Copy, Debug)]
pub enum Task {
    Conditioning(Conditioning, usize),
}

/// Where a lifetime's world physics comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Physics {
    /// the same codes in every lifetime (evolution can wire them in)
    Fixed(u64),
    /// new codes each lifetime (only learning within a lifetime can score)
    Varied,
}

/// One lifetime: the genome's network lives in a world with seed `seed`; returns the world's
/// events.
pub fn live(g: &Genome, task: Task, physics: Physics, seed: u64) -> Vec<world::Event> {
    let p = match physics {
        Physics::Fixed(p) => p,
        Physics::Varied => seed.wrapping_mul(0xA24B_AED4_963E_E407),
    };
    let mut w = World::with_physics(p, seed, 8);
    match task {
        Task::Conditioning(kind, trials) => {
            w.reversal_period = (trials / 4).max(1);
            w.conditioning(kind, trials)
        }
    }
    let mut net = g.build(&w, seed ^ 0x5EED);
    let mut act = BitVector::new(world::ACT_BITS, Some(0));
    while !w.done() {
        let senses = w.step(&act);
        net.tick(AdvanceMode::Clear, 0);
        net.write_input(&senses, 0);
        act = net.read_output().clone();
    }
    w.events
}

/// Fitness: the observer's discrimination (parts per 1024: click rate on paying trials minus on
/// the others), the mean over `seeds` world seeds.
pub fn fitness(g: &Genome, task: Task, physics: Physics, seeds: &[u64]) -> (i32, observe::Responding) {
    let mut total = 0i32;
    let mut sum = observe::Responding::default();
    for &s in seeds {
        let r = observe::responding(&live(g, task, physics, s));
        total += r.discrimination();
        sum.paying_trials += r.paying_trials;
        sum.paying_clicked += r.paying_clicked;
        sum.other_trials += r.other_trials;
        sum.other_clicked += r.other_clicked;
        sum.strokes += r.strokes;
    }
    (total / seeds.len().max(1) as i32, sum)
}

/// A generation's report.
pub struct Generation {
    pub best: Genome,
    pub best_fitness: i32,
    pub mean_fitness: i32,
    pub responding: observe::Responding,
}

/// A plain (mu + lambda) evolution: `pop` genomes, the best `elite` kept each generation, the
/// rest children of tournament winners (one mutation each, sometimes more). `report` sees every
/// generation.
pub fn evolve(task: Task, physics: Physics, seeds: &[u64], pop: usize, elite: usize, generations: usize, seed: u64, mut report: impl FnMut(usize, &Generation)) -> Genome {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut population: Vec<Genome> = (0..pop).map(|_| Genome::random(&mut rng)).collect();
    let mut best = population[0].clone();
    for generation in 0..generations {
        let mut scored: Vec<(i32, observe::Responding, Genome)> = population.drain(..).map(|g| { let (f, r) = fitness(&g, task, physics, seeds); (f, r, g) }).collect();
        // ties go to the shorter program, then to the earlier one
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.2.code.len().cmp(&b.2.code.len())));
        let mean = scored.iter().map(|s| s.0).sum::<i32>() / scored.len() as i32;
        best = scored[0].2.clone();
        report(generation, &Generation { best: best.clone(), best_fitness: scored[0].0, mean_fitness: mean, responding: scored[0].1 });
        let parents: Vec<Genome> = scored.iter().map(|s| s.2.clone()).collect();
        population = parents[..elite.min(parents.len())].to_vec();
        while population.len() < pop {
            let a = rng.gen_range(0..parents.len());
            let b = rng.gen_range(0..parents.len());
            let mut child = parents[a.min(b)].clone();
            loop {
                child = child.mutate(&mut rng);
                if rng.gen_bool(0.5) {
                    break;
                }
            }
            population.push(child);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lifetime_is_deterministic() {
        let mut rng = StdRng::seed_from_u64(3);
        let g = Genome::random(&mut rng);
        let t = Task::Conditioning(Conditioning::Instrumental, 20);
        let a: Vec<_> = live(&g, t, Physics::Varied, 1).iter().map(|e| e.moved).collect();
        let b: Vec<_> = live(&g, t, Physics::Varied, 1).iter().map(|e| e.moved).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn mutation_keeps_the_threshold_within_the_synapses() {
        let mut rng = StdRng::seed_from_u64(4);
        let mut g = Genome::random(&mut rng);
        for _ in 0..500 {
            g = g.mutate(&mut rng);
            assert!(g.threshold <= g.syn && g.threshold >= 1);
        }
    }
}
