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
    /// serial reversal: this many trials, the paying light swapping every `period`; with
    /// `worst_phase`, a lifetime scores its worst phase (discrimination within each phase between
    /// swaps), so a genome must do well before and after each swap, not on average
    Reversal { trials: usize, period: usize, worst_phase: bool },
    /// reading: this many random tokens shown on the eye, one a step
    Sequence(usize),
    /// reading with a teacher: as `Sequence`, about half the tokens also said aloud
    Reading(usize),
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
    live_work(g, task, physics, seed, 0).0
}

/// As `live`, also returning the network's work over the lifetime (kernels matched plus words
/// wires and noise wrote; see `RuntimeNetwork::work`); with `gap` > 0, conditioning trials are
/// separated by 0 to `gap` empty steps (see `World::gap_max`).
pub fn live_work<G: Evolvable>(g: &G, task: Task, physics: Physics, seed: u64, gap: usize) -> (Vec<world::Event>, u64) {
    let p = match physics {
        Physics::Fixed(p) => p,
        Physics::Varied => seed.wrapping_mul(0xA24B_AED4_963E_E407),
    };
    let mut w = World::with_physics(p, seed, 8);
    w.gap_max = gap;
    match task {
        Task::Conditioning(kind, trials) => {
            w.reversal_period = (trials / 4).max(1);
            w.conditioning(kind, trials)
        }
        Task::Reversal { trials, period, .. } => {
            w.reversal_period = period.max(1);
            w.conditioning(Conditioning::Reversal, trials)
        }
        Task::Sequence(n) | Task::Reading(n) => {
            let mut r = StdRng::seed_from_u64(seed ^ 0x5E9);
            let toks: Vec<usize> = (0..n).map(|_| r.gen_range(0..8)).collect();
            w.teacher = matches!(task, Task::Reading(_));
            w.sequence(&toks)
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
    (w.events, net.work)
}

/// Fitness: the observer's discrimination (parts per 1024: click rate on paying trials minus on
/// the others), the mean over `seeds` world seeds. A lifetime (or, scoring the worst phase, a
/// phase) with no click counts 0, like clicking without telling the lights apart; a genome that
/// never clicks in any of its lifetimes scores `observe::SILENT`, below any clicking.
pub fn fitness<G: Evolvable>(g: &G, task: Task, physics: Physics, seeds: &[u64]) -> (i32, observe::Responding) {
    let (f, r, _) = fitness_detail(g, task, physics, seeds);
    (f, r)
}

/// As `fitness`, also counting the lifetimes with no click.
pub fn fitness_detail<G: Evolvable>(g: &G, task: Task, physics: Physics, seeds: &[u64]) -> (i32, observe::Responding, usize) {
    let mut total = 0i32;
    let mut silent = 0usize;
    let mut sum = observe::Responding::default();
    for &s in seeds {
        let events = live(g, task, physics, s);
        let r = observe::responding(&events);
        if r.paying_clicked + r.other_clicked == 0 {
            silent += 1;
        }
        total += match task {
            Task::Reversal { period, worst_phase: true, .. } => observe::worst_phase(&events, period),
            _ => r.discrimination(),
        };
        sum.paying_trials += r.paying_trials;
        sum.paying_clicked += r.paying_clicked;
        sum.other_trials += r.other_trials;
        sum.other_clicked += r.other_clicked;
        sum.strokes += r.strokes;
    }
    if silent == seeds.len() {
        return (observe::SILENT, sum, silent);
    }
    (total / seeds.len().max(1) as i32, sum, silent)
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

/// The probes of one genome in one world seed (see `observe::probes`), and its work per step.
/// Only the lifetimes the `wanted` probes need are lived (a serial-reversal lifetime for respond,
/// tell, discriminate, reverse and win-stay; a delayed one for remember; reading with a teacher
/// for copy and echo),
/// `trials` trials or tokens each, with gaps of 0 to `gap` steps between trials; the other
/// probes read 0.
pub fn probe_scores<G: Evolvable>(g: &G, trials: usize, physics: Physics, seed: u64, gap: usize, wanted: &[usize]) -> (observe::ProbeRow, u64) {
    let period = (trials / 4).max(1);
    let want = |ps: &[usize]| ps.iter().any(|p| wanted.contains(p));
    let (rev, w1) = if want(&[0, 1, 2, 3, 5]) { live_work(g, Task::Reversal { trials, period, worst_phase: false }, physics, seed, gap) } else { (vec![], 0) };
    let (del, w2) = if want(&[4]) { live_work(g, Task::Conditioning(Conditioning::Delayed, trials), physics, seed ^ 0xDE1A, gap) } else { (vec![], 0) };
    let (seq, w3) = if want(&[6, 7]) { live_work(g, Task::Reading(trials), physics, seed ^ 0x5E0, 0) } else { (vec![], 0) };
    let steps = (rev.len() + del.len() + seq.len()).max(1) as u64;
    let mut row = observe::probes(&rev, period, &del, &seq);
    for (p, x) in row.iter_mut().enumerate() {
        if !wanted.contains(&p) {
            *x = 0;
        }
    }
    (row, (w1 + w2 + w3) / steps)
}

/// All the probes.
pub const ALL_PROBES: [usize; observe::NPROBES] = [0, 1, 2, 3, 4, 5, 6, 7];

/// What probing a genome found: its probes and work per step in each seed, and its mean work
/// per step.
pub struct Probed<G> {
    pub probes: Vec<observe::ProbeRow>,
    pub works: Vec<u64>,
    pub work: u64,
    pub genome: G,
}

/// Each genome's probes in each seed, spread over the machine's cores (a genome whose network
/// panics scores worst on every probe).
pub fn probe_all<G: Evolvable>(population: Vec<G>, trials: usize, physics: Physics, seeds: &[u64], gap: usize, wanted: &[usize]) -> Vec<Probed<G>> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).max(1);
    let chunk = population.len().div_ceil(threads).max(1);
    let mut out = Vec::with_capacity(population.len());
    std::thread::scope(|sc| {
        let handles: Vec<_> = population
            .chunks(chunk)
            .map(|part| sc.spawn(move || part.iter().map(|g| {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| seeds.iter().map(|&s| probe_scores(g, trials, physics, s, gap, wanted)).collect::<Vec<_>>()))
                    .unwrap_or_else(|_| vec![([-1024; observe::NPROBES], u64::MAX / 2); seeds.len()]);
                let work = r.iter().map(|x| x.1).sum::<u64>() / seeds.len().max(1) as u64;
                Probed { probes: r.iter().map(|x| x.0).collect(), works: r.iter().map(|x| x.1).collect(), work, genome: g.clone() }
            }).collect::<Vec<_>>()))
            .collect();
        for h in handles {
            out.extend(h.join().unwrap());
        }
    });
    out
}

/// Mean of each probe over a genome's seeds.
pub fn probe_means(s: &[observe::ProbeRow]) -> observe::ProbeRow {
    let mut m = [0i64; observe::NPROBES];
    for x in s {
        for p in 0..observe::NPROBES {
            m[p] += x[p] as i64;
        }
    }
    m.map(|v| (v / s.len().max(1) as i64) as i32)
}

/// A generation of probe evolution: the genome with the highest summed probe means, its means
/// and work per step, and the population's means and work per step.
pub struct ProbeGeneration<G> {
    /// the whole population, highest summed probe means (on the training worlds) first
    pub ranked: Vec<G>,
    pub best: G,
    pub best_means: observe::ProbeRow,
    pub best_work: u64,
    pub pop_means: observe::ProbeRow,
    pub pop_work: u64,
}

/// Evolution on the probes by epsilon-lexicase selection: every (seed, probe) pair is a case,
/// and, with `thrift`, one more case per seed prefers less work per step in that world (kernels
/// matched plus words wired): a genome is cheaper by pausing or shedding populations, so among
/// genomes that do equally well the cheaper is kept (a sixth of the cases: enough to stop
/// bloat, not enough to win over every capacity). Each parent is chosen by taking the cases in a random order
/// and keeping, case by case, the genomes within epsilon (the case's median absolute deviation
/// over the population) of the best, until one is left. A genome best at anything (one probe, a
/// few worlds) can be chosen, so evolution can climb one capacity at a time instead of needing
/// all of them at once. The `elite` survivors are lexicase choices too; the rest are their
/// mutated (a quarter recombined) children. Returns the final population, highest summed probe
/// means first.
pub fn evolve_lexicase<G: Evolvable>(start: Vec<G>, trials: usize, physics: Physics, seeds: &[u64], gap: usize, wanted: &[usize], elite: usize, generations: usize, seed: u64, thrift: bool, mut report: impl FnMut(usize, &ProbeGeneration<G>)) -> Vec<G> {
    let mut rng = StdRng::seed_from_u64(seed ^ 0x1E81);
    let pop = start.len();
    let mut population = start;
    let mut last: Vec<G> = population.clone();
    for generation in 0..generations {
        let scored = probe_all(population.drain(..).collect(), trials, physics, seeds, gap, wanted);
        let n = scored.len();
        let (ranked, next) = lexicase_breed(&scored, wanted, thrift, pop, elite, &mut rng);
        let means: Vec<observe::ProbeRow> = scored.iter().map(|s| probe_means(&s.probes)).collect();
        let pop_means = probe_means(&means);
        let pop_work = scored.iter().map(|s| s.work.min(u64::MAX / 4)).sum::<u64>() / n.max(1) as u64;
        let b = ranked[0];
        report(generation, &ProbeGeneration { ranked: ranked.iter().map(|&g| scored[g].genome.clone()).collect(), best: scored[b].genome.clone(), best_means: means[b], best_work: scored[b].work, pop_means, pop_work });
        last = ranked.iter().map(|&g| scored[g].genome.clone()).collect();
        population = next;
    }
    last
}

/// One generation of epsilon-lexicase breeding on the `wanted` probes (and, with `thrift`, one
/// work case per seed): the population ranked by its summed probe means (highest first, ties to
/// the cheaper), and the next population: `elite` lexicase choices kept, the rest their mutated
/// (a quarter recombined) children.
fn lexicase_breed<G: Evolvable>(scored: &[Probed<G>], wanted: &[usize], thrift: bool, pop: usize, elite: usize, rng: &mut StdRng) -> (Vec<usize>, Vec<G>) {
    let n = scored.len();
    let nseeds = scored.first().map_or(0, |s| s.probes.len());
    // the cases: one per (seed, wanted probe), then one thrift case per seed (less work is better)
    let np = wanted.len();
    let ncases = nseeds * np + if thrift { nseeds } else { 0 };
    let value = |g: usize, c: usize| -> i64 {
        if c < nseeds * np { scored[g].probes[c / np][wanted[c % np]] as i64 } else { -(scored[g].works[c - nseeds * np].min(i64::MAX as u64 / 2) as i64) }
    };
    // epsilon per case: the median absolute deviation over the population
    let eps: Vec<i64> = (0..ncases).map(|c| {
        let mut v: Vec<i64> = (0..n).map(|g| value(g, c)).collect();
        v.sort_unstable();
        let med = v[n / 2];
        let mut d: Vec<i64> = v.iter().map(|x| (x - med).abs()).collect();
        d.sort_unstable();
        d[n / 2]
    }).collect();
    let select = |rng: &mut StdRng| -> usize {
        let mut order: Vec<usize> = (0..ncases).collect();
        order.shuffle(rng);
        let mut cand: Vec<usize> = (0..n).collect();
        for c in order {
            if cand.len() == 1 {
                break;
            }
            let best = cand.iter().map(|&g| value(g, c)).max().unwrap();
            cand.retain(|&g| value(g, c) >= best - eps[c]);
        }
        cand[rng.gen_range(0..cand.len())]
    };
    let means: Vec<observe::ProbeRow> = scored.iter().map(|s| probe_means(&s.probes)).collect();
    let total = |m: &observe::ProbeRow| wanted.iter().map(|&p| m[p] as i64).sum::<i64>();
    let mut ranked: Vec<usize> = (0..n).collect();
    ranked.sort_by(|&a, &b| total(&means[b]).cmp(&total(&means[a])).then(scored[a].work.cmp(&scored[b].work)).then(b.cmp(&a)));
    let mut kept: Vec<usize> = Vec::new();
    for _ in 0..elite * 4 {
        if kept.len() >= elite.min(n) {
            break;
        }
        let g = select(rng);
        if !kept.contains(&g) {
            kept.push(g);
        }
    }
    let mut next: Vec<G> = kept.iter().map(|&g| scored[g].genome.clone()).collect();
    while next.len() < pop {
        let mut child = scored[select(rng)].genome.clone();
        if rng.gen_range(0..4) == 0 {
            let other = select(rng);
            child = child.crossover(&scored[other].genome, rng);
        }
        loop {
            child = child.mutate(rng);
            if rng.gen_bool(0.5) {
                break;
            }
        }
        next.push(child);
    }
    (ranked, next)
}

/// An island: a population evolving on its own probes.
pub struct Island {
    pub name: String,
    pub probes: Vec<usize>,
}

/// One generation of island evolution, per island: the population ranked (by its own probes,
/// training worlds), its probe means and its work per step; and the plasmids moved this
/// generation: (from island, to island, what moved).
pub struct IslandGeneration<G> {
    pub ranked: Vec<Vec<G>>,
    pub means: Vec<observe::ProbeRow>,
    pub work: Vec<u64>,
    pub moved: Vec<(usize, usize, String)>,
}

/// Islands: each population evolves by lexicase on its own probes (`evolve_lexicase`'s breeding),
/// and every `every` generations each island takes `plasmids` circuits from the best genomes of
/// other islands (`transfer(recipient, donor, rng)` makes the child and says what moved): a
/// circuit that does well under one island's pressure gets tried under another's. The donors
/// are the top `elite` of a random other island; the recipients are this island's new
/// children (not its kept genomes). `start` gives each island's starting population (resized to
/// `pop`), and generations run from `first` to `generations` (so a checkpoint can be resumed).
pub fn evolve_islands<G: Evolvable>(
    islands: &[Island],
    start: Vec<Vec<G>>,
    first: usize,
    pop: usize,
    trials: usize,
    physics: Physics,
    seeds: &[u64],
    gap: usize,
    elite: usize,
    generations: usize,
    seed: u64,
    thrift: bool,
    every: usize,
    plasmids: usize,
    transfer: impl Fn(&G, &G, &mut StdRng) -> (G, String),
    mut report: impl FnMut(usize, &IslandGeneration<G>),
) -> Vec<Vec<G>> {
    let k = islands.len();
    let mut rng = StdRng::seed_from_u64(seed ^ 0x1514);
    let mut pops: Vec<Vec<G>> = start
        .into_iter()
        .map(|mut p| {
            let n = p.len().max(1);
            while p.len() < pop {
                let g = p[p.len() % n].clone();
                p.push(g);
            }
            p.truncate(pop);
            p
        })
        .collect();
    assert_eq!(pops.len(), k, "one starting population an island");
    let mut last: Vec<Vec<G>> = pops.clone();
    for generation in first..generations {
        let mut ranked: Vec<Vec<G>> = Vec::with_capacity(k);
        let mut means = Vec::with_capacity(k);
        let mut work = Vec::with_capacity(k);
        let mut next: Vec<Vec<G>> = Vec::with_capacity(k);
        for (i, isl) in islands.iter().enumerate() {
            let scored = probe_all(pops[i].drain(..).collect(), trials, physics, seeds, gap, &isl.probes);
            let (order, children) = lexicase_breed(&scored, &isl.probes, thrift, pop, elite, &mut rng);
            let m: Vec<observe::ProbeRow> = scored.iter().map(|s| probe_means(&s.probes)).collect();
            means.push(probe_means(&m));
            work.push(scored.iter().map(|s| s.work.min(u64::MAX / 4)).sum::<u64>() / scored.len().max(1) as u64);
            ranked.push(order.iter().map(|&g| scored[g].genome.clone()).collect());
            next.push(children);
        }
        // the exchange: plasmids from the best of other islands into new children
        let mut moved = Vec::new();
        if k > 1 && generation > first && every > 0 && generation % every == 0 {
            for i in 0..k {
                for _ in 0..plasmids {
                    let j = (i + rng.gen_range(1..k)) % k;
                    let donor = &ranked[j][rng.gen_range(0..elite.min(ranked[j].len()).max(1))];
                    let slot = rng.gen_range(elite.min(pop - 1)..pop);
                    let (child, what) = transfer(&next[i][slot], donor, &mut rng);
                    next[i][slot] = child;
                    moved.push((j, i, what));
                }
            }
        }
        report(generation, &IslandGeneration { ranked: ranked.clone(), means, work, moved });
        last = ranked;
        pops = next;
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
