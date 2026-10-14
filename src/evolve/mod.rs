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
use crate::kernel::KernelOp;
use crate::program::graph::{GraphInstruction, GraphProgram, ProgramDefaults};
use crate::program::neurocomp::{EdgeInfo, RuntimeNetwork};
use crate::world::{self, observe, Conditioning, World};

/// Every graph instruction, for mutation.
pub const INSTRUCTIONS: [GraphInstruction; 40] = [
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
    GraphInstruction::EdgeOr(),
    GraphInstruction::EdgeAnd(),
    GraphInstruction::EdgeXor(),
    GraphInstruction::EdgeClear(),
    GraphInstruction::EdgeWire(),
    GraphInstruction::EdgeRotate(),
    GraphInstruction::EdgeLag(),
    GraphInstruction::EdgeGrow(),
    GraphInstruction::EdgeShare(),
    GraphInstruction::EdgeGate(),
    GraphInstruction::EdgeGateNot(),
    GraphInstruction::EdgeNew(),
    GraphInstruction::EdgeWide(),
    GraphInstruction::EdgeSpan(),
    GraphInstruction::Noise(),
    GraphInstruction::EdgeKernels(),
    GraphInstruction::EdgeSynapses(),
    GraphInstruction::EdgeQuarter(),
    GraphInstruction::EdgeNand(),
    GraphInstruction::EdgeNor(),
    GraphInstruction::EdgeAdd(),
    GraphInstruction::EdgeSub(),
    GraphInstruction::EdgeShl(),
    GraphInstruction::EdgeShr(),
    GraphInstruction::EdgeRotl(),
    GraphInstruction::EdgeRotr(),
    GraphInstruction::EdgeWider(),
];

pub mod genes;
pub mod tags;

/// What evolution works on: something that builds a network for a world and mutates locally.
pub trait Evolvable: Clone + Send + Sync {
    fn build(&self, w: &World, seed: u64) -> RuntimeNetwork;
    fn mutate(&self, rng: &mut StdRng) -> Self;
    /// A size (reported).
    fn size(&self) -> usize;
    /// A child of this genome and another: part from each (recombination). By default, this one.
    fn crossover(&self, _other: &Self, _rng: &mut StdRng) -> Self {
        self.clone()
    }
}

impl Evolvable for Genome {
    fn build(&self, w: &World, seed: u64) -> RuntimeNetwork {
        Genome::build(self, w, seed)
    }
    fn mutate(&self, rng: &mut StdRng) -> Self {
        Genome::mutate(self, rng)
    }
    fn size(&self) -> usize {
        self.code.len()
    }
    /// One point: this program's front, the other's back.
    fn crossover(&self, other: &Self, rng: &mut StdRng) -> Self {
        let a = rng.gen_range(0..=self.code.len());
        let b = rng.gen_range(0..=other.code.len());
        Genome { code: self.code[..a].iter().chain(other.code[b..].iter()).copied().collect() }
    }
}

const OPS: [KernelOp; 4] = [KernelOp::Or, KernelOp::And, KernelOp::Xor, KernelOp::Clear];

/// Bits of a hidden node (the program can double or halve it).
pub const NODE_BITS: usize = 512;

/// A stack-grammar genome: only its program. Everything an edge does (its kernels, their
/// synapses, threshold and op) is set by the program's own instructions on that edge; there are
/// no genome-wide numbers.
#[derive(Clone, Debug, PartialEq)]
pub struct Genome {
    pub code: Vec<GraphInstruction>,
}

impl Genome {
    pub fn random(rng: &mut StdRng) -> Self {
        let n = rng.gen_range(0..8);
        Self { code: (0..n).map(|_| *INSTRUCTIONS.choose(rng).unwrap()).collect() }
    }

    /// One local change: an instruction inserted, removed or replaced.
    pub fn mutate(&self, rng: &mut StdRng) -> Self {
        let mut g = self.clone();
        let n = g.code.len();
        match rng.gen_range(0..6) {
            0 if n < 96 => {
                let at = rng.gen_range(0..=n);
                g.code.insert(at, *INSTRUCTIONS.choose(rng).unwrap());
            }
            1 if n > 0 => {
                let at = rng.gen_range(0..n);
                g.code.remove(at);
            }
            // two instructions swapped
            2 if n > 1 => {
                let (a, b) = (rng.gen_range(0..n), rng.gen_range(0..n));
                g.code.swap(a, b);
            }
            // a segment copied elsewhere (duplication)
            3 if n > 0 && n < 96 => {
                let a = rng.gen_range(0..n);
                let len = rng.gen_range(1..=8.min(n - a));
                let seg: Vec<GraphInstruction> = g.code[a..a + len].to_vec();
                let at = rng.gen_range(0..=n);
                for (k, i) in seg.into_iter().enumerate() {
                    g.code.insert(at + k, i);
                }
            }
            // a segment deleted
            4 if n > 1 => {
                let a = rng.gen_range(0..n);
                let len = rng.gen_range(1..=8.min(n - a));
                g.code.drain(a..a + len);
            }
            _ if n > 0 => {
                let at = rng.gen_range(0..n);
                g.code[at] = *INSTRUCTIONS.choose(rng).unwrap();
            }
            _ => g.code.push(*INSTRUCTIONS.choose(rng).unwrap()),
        }
        g
    }

    /// The network this genome builds for a world with these senses and acts.
    pub fn build(&self, w: &World, seed: u64) -> RuntimeNetwork {
        // the grammar's starting values for a new node or edge (not evolved numbers: each node
        // and edge then carries its own copy, which the program changes)
        let defaults = ProgramDefaults { default_bits: NODE_BITS, min_bits: 64, default_threshold: 1, default_history_depth: 1, default_op: KernelOp::Or };
        let prog = GraphProgram { code: self.code.clone() };
        let graph = prog.build_graph(&defaults);
        let body = w.body();
        let primitives: Vec<BitVector> = (0..body.voice.len())
            .map(|i| w.act(Some(i), None))
            .chain((0..body.hand.len()).map(|i| w.act(None, Some(i))))
            .collect();
        RuntimeNetwork::from_graph_with(&graph, world::SENSE_BITS, world::ACT_BITS, seed, |e: &EdgeInfo| genes::start_kernels(e, &primitives))
    }
}

fn sparse(bits: usize, k: usize, rng: &mut StdRng) -> BitVector {
    let all: Vec<usize> = (0..bits).collect();
    BitVector::from_bits(&all.choose_multiple(rng, k.min(bits)).copied().collect::<Vec<_>>(), bits)
}

/// The task a genome is scored on.
#[derive(Clone, Copy, Debug)]
pub enum Task {
    /// a conditioning scene of this many trials (a reversal swaps every quarter)
    Conditioning(Conditioning, usize),
    /// serial reversal: this many trials, the paying light swapping every `period`
    Reversal { trials: usize, period: usize },
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
pub fn live<G: Evolvable>(g: &G, task: Task, physics: Physics, seed: u64) -> Vec<world::Event> {
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
        Task::Reversal { trials, period } => {
            w.reversal_period = period.max(1);
            w.conditioning(Conditioning::Reversal, trials)
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
pub fn fitness<G: Evolvable>(g: &G, task: Task, physics: Physics, seeds: &[u64]) -> (i32, observe::Responding) {
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

/// Every genome's fitness, the population spread over the machine's cores (each lifetime is
/// deterministic by its seed, so the order of evaluation does not matter).
pub fn score_all<G: Evolvable>(population: Vec<G>, task: Task, physics: Physics, seeds: &[u64]) -> Vec<(i32, observe::Responding, G)> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).max(1);
    let chunk = population.len().div_ceil(threads).max(1);
    let mut out: Vec<(i32, observe::Responding, G)> = Vec::with_capacity(population.len());
    std::thread::scope(|sc| {
        let handles: Vec<_> = population
            .chunks(chunk)
            .map(|part| sc.spawn(move || part.iter().map(|g| {
                // a genome that breaks (a panic in its network) scores worst and dies; the run
                // goes on
                let (f, r) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| fitness(g, task, physics, seeds))).unwrap_or((i32::MIN / 2, observe::Responding::default()));
                (f, r, g.clone())
            }).collect::<Vec<_>>()))
            .collect();
        for h in handles {
            out.extend(h.join().unwrap());
        }
    });
    out
}

/// A generation's report.
pub struct Generation<G> {
    pub best: G,
    pub best_fitness: i32,
    pub mean_fitness: i32,
    pub responding: observe::Responding,
}

/// A plain (mu + lambda) evolution from random stack genomes: `pop` genomes, the best `elite`
/// kept each generation, the rest children of tournament winners (one mutation each, sometimes
/// more). `report` sees every generation.
pub fn evolve(task: Task, physics: Physics, seeds: &[u64], pop: usize, elite: usize, generations: usize, seed: u64, report: impl FnMut(usize, &Generation<Genome>)) -> Genome {
    let mut rng = StdRng::seed_from_u64(seed);
    let start: Vec<Genome> = (0..pop).map(|_| Genome::random(&mut rng)).collect();
    evolve_from(start, task, physics, seeds, elite, generations, seed, report)
}

/// The same evolution from a given starting population (for example copies of a hand-written
/// seed genome).
pub fn evolve_from<G: Evolvable>(start: Vec<G>, task: Task, physics: Physics, seeds: &[u64], elite: usize, generations: usize, seed: u64, report: impl FnMut(usize, &Generation<G>)) -> G {
    evolve_population(start, task, physics, seeds, elite, generations, seed, report).remove(0)
}

/// The same evolution, returning the whole final population, best first (so a curriculum can
/// carry the population from one task to the next).
pub fn evolve_population<G: Evolvable>(start: Vec<G>, task: Task, physics: Physics, seeds: &[u64], elite: usize, generations: usize, seed: u64, mut report: impl FnMut(usize, &Generation<G>)) -> Vec<G> {
    let mut rng = StdRng::seed_from_u64(seed ^ 0xE0E0);
    let pop = start.len();
    let mut population = start;
    let mut last: Vec<G> = population.clone();
    for generation in 0..generations {
        let mut scored: Vec<(i32, observe::Responding, G)> = score_all(population.drain(..).collect(), task, physics, seeds);
        // ties go to the newer genome (the offspring come after the kept ones), so on a flat stretch
        // the population drifts instead of keeping the same genomes forever
        let mut order: Vec<usize> = (0..scored.len()).collect();
        order.sort_by(|&a, &b| scored[b].0.cmp(&scored[a].0).then(b.cmp(&a)));
        let mut sorted = Vec::with_capacity(scored.len());
        let mut slots: Vec<Option<(i32, observe::Responding, G)>> = scored.drain(..).map(Some).collect();
        for i in order {
            sorted.push(slots[i].take().unwrap());
        }
        let scored = sorted;
        let mean = scored.iter().map(|s| s.0).sum::<i32>() / scored.len() as i32;
        report(generation, &Generation { best: scored[0].2.clone(), best_fitness: scored[0].0, mean_fitness: mean, responding: scored[0].1 });
        let parents: Vec<G> = scored.iter().map(|s| s.2.clone()).collect();
        last = parents.clone();
        population = parents[..elite.min(parents.len())].to_vec();
        while population.len() < pop {
            let a = rng.gen_range(0..parents.len());
            let b = rng.gen_range(0..parents.len());
            let mut child = parents[a.min(b)].clone();
            // a quarter of the children recombine two parents before mutating
            if rng.gen_range(0..4) == 0 {
                let c = rng.gen_range(0..parents.len());
                let d = rng.gen_range(0..parents.len());
                child = child.crossover(&parents[c.min(d)], &mut rng);
            }
            loop {
                child = child.mutate(&mut rng);
                if rng.gen_bool(0.5) {
                    break;
                }
            }
            population.push(child);
        }
    }
    last
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
    fn starting_kernels_come_from_the_edge_alone() {
        use crate::program::graph::GraphInstruction as I;
        let w = World::new(1, 4);
        let plain = Genome { code: vec![I::Create()] }.build(&w, 1);
        assert!(plain.edges.iter().all(|e| e.group_len() == 0), "no kernels unless an edge asks for them");
        let asked = Genome { code: vec![I::Create(), I::EdgeKernels(), I::EdgeKernels()] }.build(&w, 1);
        assert_eq!(asked.edges.iter().map(|e| e.group_len()).sum::<usize>(), 2, "the last edge made starts with two");
    }
}
