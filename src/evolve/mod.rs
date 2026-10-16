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
    /// What its lifetimes found (`Note`s, summed over worlds), written into the genome: by default
    /// nothing (the tag grammar's epigenetic marks and explored connections).
    fn inherit(&mut self, _notes: &[Note]) {}
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
    /// a T-maze of this many trials (`World::maze`)
    Maze(usize),
    /// a screen of this many trials (`World::screen`)
    Screen(usize),
    /// task selection: this many blocks, alternating a conditioning block (a light to click, no
    /// swaps) and a reading block (tokens shown, no teacher), in one lifetime with one network;
    /// each block `Lesson::mixed_block` trials (a reading block ten tokens a trial), with the
    /// block's task felt as a cue on the touch channel if `Lesson::mixed_cue`
    Mixed(usize),
    /// feeling: this many steps with strokes at random, nothing to see or earn
    Feel(usize),
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
    live_lesson(g, task, physics, seed, gap, Lesson::default())
}

/// What a stage of a curriculum sets in the world (the harness's side only; the network is
/// never told): how many distinct tokens a reading lifetime uses (2 to 8); how often the teacher
/// says a shown token aloud, in eighths; how many times the paying light swaps in a reversal
/// lifetime; how many steps before the response window the delayed light flashes (1 to 5); and
/// the pattern of a prediction lifetime's stream: a cycle of `period` tokens drawn each lifetime
/// (0: no pattern, each token drawn alone), each token replaced by a random one with chance
/// `noise` in 8; and how a serial-reversal light is distorted each trial (shapes): `blur` (each
/// line moved within its word with chance `blur` in 8) and `shift` (the whole light rotated by up
/// to that many lines); and the T-maze: the stem's length, the cue only at its foot, the cue a
/// light's code rather than a side of the eye; and the screen: its columns and rows, how many
/// glyphs it shows (the target and distractors), its cells' width in bits (64, 32 or 16: at most
/// 8, 16 or 32 cells), how many walls it has (a 2-D maze) and whether a glyph shows only under
/// the cursor (a torch), and whether the cursor's mark is hidden (dark). The default is the plain world: 8 tokens, the
/// teacher on half of them, 3 swaps, a delay of 3, no pattern, fixed lights, a stem of 2 with the
/// cue on a side of the eye all along it, a 2 x 2 screen with the target alone on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lesson {
    pub tokens: usize,
    pub teach: u32,
    pub swaps: usize,
    pub delay: usize,
    pub period: usize,
    pub noise: u32,
    /// delayed conditioning (recall's long rungs): a delay drawn per trial up to `vary_delay` (0:
    /// fixed), a distracting flash of the other light, the other light on no-cue trials
    pub vary_delay: usize,
    pub distract: bool,
    pub two_lights: bool,
    pub blur: u32,
    pub shift: usize,
    pub maze_len: usize,
    pub cue_gone: bool,
    pub symbol: bool,
    pub maze_fixed: bool,
    pub maze_nocue: bool,
    /// task selection (`Task::Mixed`): trials a block, and a cue for the block's task
    pub mixed_block: usize,
    pub mixed_cue: bool,
    pub screen_w: usize,
    pub screen_h: usize,
    pub screen_lit: usize,
    pub screen_cell: usize,
    pub screen_walls: usize,
    pub screen_torch: bool,
    pub screen_dark: bool,
    /// energy, 0: none. Each step the network gets `ration` and pays its work; and it eats what it
    /// scores: every `FOOD_BLOCK` steps the lifetime's last block is scored by its task's probes,
    /// and each 1024 points (positive scores only) are `food` more. The store starts at 64 rations
    /// and holds at most 256; while it is below 0 the network rests (it does not run or act) and
    /// the ration refills it. So compute beyond the ration must be earned by scoring.
    pub ration: u64,
    pub food: u64,
}

/// A mixed lifetime's block `b` (`Task::Mixed`): even, a conditioning block; odd, a reading one.
fn mixed_block(w: &mut World, b: usize, lesson: &Lesson, tokens: usize, seed: u64) {
    let n = lesson.mixed_block.max(1);
    w.task_cue = lesson.mixed_cue.then_some(b % 2);
    if b % 2 == 0 {
        w.reversal_period = n;
        w.conditioning(Conditioning::Reversal, n);
    } else {
        let mut r = StdRng::seed_from_u64(seed ^ 0x313C ^ b as u64);
        let toks: Vec<usize> = (0..n * 10).map(|_| r.gen_range(0..tokens)).collect();
        w.teacher = false;
        w.sequence_marked(&toks, &vec![false; toks.len()]);
    }
}

/// Steps a lifetime is scored over for its food (`Lesson::food`).
pub const FOOD_BLOCK: usize = 50;

/// The default energy for lessons made from now on (`Lesson::ration`, `Lesson::food`): an
/// experiment's binary sets it once, before any lesson is made. 0: none.
static RATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FOOD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn set_energy(ration: u64, food: u64) {
    RATION.store(ration, std::sync::atomic::Ordering::Relaxed);
    FOOD.store(food, std::sync::atomic::Ordering::Relaxed);
}

pub fn energy() -> (u64, u64) {
    (RATION.load(std::sync::atomic::Ordering::Relaxed), FOOD.load(std::sync::atomic::Ordering::Relaxed))
}

impl Default for Lesson {
    fn default() -> Self {
        Self { tokens: 8, teach: 4, swaps: 3, delay: 3, vary_delay: 0, distract: false, two_lights: false, period: 0, noise: 0, blur: 0, shift: 0, maze_len: 2, cue_gone: false, symbol: false, maze_fixed: false, maze_nocue: false, mixed_block: 40, mixed_cue: false, screen_w: 2, screen_h: 2, screen_lit: 1, screen_cell: 64, screen_walls: 0, screen_torch: false, screen_dark: false, ration: energy().0, food: energy().1 }
    }
}

/// What a lifetime found about one population, for its genome: (its id, a count, a kind). Kind
/// 0: its kernels' predictions that came true. Kind 1 or 2: a trial connection grown from it by
/// exploration (back into its source, or on its destination), and that trial's predictions that
/// came true.
pub type Note = (u32, u32, u8);

thread_local! {
    /// The notes of the lifetimes run on this thread, while a probing job collects them (each
    /// job runs on one thread, so a lifetime adds to its own job's).
    static NOTES: std::cell::RefCell<Option<Vec<Note>>> = const { std::cell::RefCell::new(None) };
}

/// Notes summed by (id, kind).
fn merge_notes(notes: impl IntoIterator<Item = Note>) -> Vec<Note> {
    let mut out: Vec<Note> = Vec::new();
    for (id, c, k) in notes {
        match out.iter_mut().find(|n| n.0 == id && n.2 == k) {
            Some(n) => n.1 = n.1.saturating_add(c),
            None => out.push((id, c, k)),
        }
    }
    out
}

/// The epigenetic marks and the keeping of explored connections (`Evolvable::inherit`) are
/// on for every genome from now on. Off by default.
static EPIGENETICS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_epigenetics(on: bool) {
    EPIGENETICS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn epigenetics() -> bool {
    EPIGENETICS.load(std::sync::atomic::Ordering::Relaxed)
}

/// `live_work` in a world set by a lesson.
pub fn live_lesson<G: Evolvable>(g: &G, task: Task, physics: Physics, seed: u64, gap: usize, lesson: Lesson) -> (Vec<world::Event>, u64) {
    let p = match physics {
        Physics::Fixed(p) => p,
        Physics::Varied => seed.wrapping_mul(0xA24B_AED4_963E_E407),
    };
    // (a lesson's token count changes only reading lifetimes; conditioning keeps its 8)
    let reading = matches!(task, Task::Sequence(_) | Task::Reading(_) | Task::Mixed(_));
    let tokens = if reading { lesson.tokens.clamp(2, 8) } else { 8 };
    let mut w = World::with_physics(p, seed, tokens);
    w.gap_max = gap;
    w.teacher_rate = lesson.teach;
    w.delay = lesson.delay;
    w.vary_delay = lesson.vary_delay;
    w.distract = lesson.distract;
    w.two_lights = lesson.two_lights;
    w.blur = lesson.blur;
    w.shift = lesson.shift;
    w.maze_len = lesson.maze_len;
    w.cue_gone = lesson.cue_gone;
    w.symbol = lesson.symbol;
    w.maze_fixed = lesson.maze_fixed;
    w.maze_nocue = lesson.maze_nocue;
    w.screen_w = lesson.screen_w;
    w.screen_h = lesson.screen_h;
    w.screen_lit = lesson.screen_lit;
    w.screen_cell = lesson.screen_cell;
    w.screen_walls = lesson.screen_walls;
    w.screen_torch = lesson.screen_torch;
    w.screen_dark = lesson.screen_dark;
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
            // (a prediction lifetime: a cycle drawn for this lifetime, some tokens replaced by a
            // random other one, which the observer marks as breaking it)
            let cycle: Vec<usize> = (0..lesson.period).map(|_| r.gen_range(0..tokens)).collect();
            let mut odd = Vec::with_capacity(n);
            let toks: Vec<usize> = (0..n)
                .map(|i| {
                    if cycle.is_empty() {
                        odd.push(false);
                        return r.gen_range(0..tokens);
                    }
                    let c = cycle[i % cycle.len()];
                    let t = if lesson.noise > 0 && r.gen_ratio(lesson.noise.min(8), 8) { (c + r.gen_range(1..tokens)) % tokens } else { c };
                    odd.push(t != c);
                    t
                })
                .collect();
            w.teacher = matches!(task, Task::Reading(_));
            w.sequence_marked(&toks, &odd)
        }
        Task::Maze(n) => w.maze(n),
        Task::Screen(n) => w.screen(n),
        Task::Mixed(_) => mixed_block(&mut w, 0, &lesson, tokens, seed),
        Task::Feel(n) => w.feel(n),
    }
    let mut blk = 0usize;
    let mut net = g.build(&w, seed ^ 0x5EED);
    let mut act = BitVector::new(world::ACT_BITS, Some(0));
    // energy (`Lesson::ration`): the store, below 0 the network rests
    let (ration, food) = (lesson.ration as i64, lesson.food as i64);
    let mut store = ration.saturating_mul(64);
    let mut block = 0usize;
    loop {
        // (a mixed lifetime: the next block once one ends)
        if w.done() {
            match task {
                Task::Mixed(blocks) if blk + 1 < blocks => {
                    blk += 1;
                    mixed_block(&mut w, blk, &lesson, tokens, seed);
                }
                _ => break,
            }
        }
        let senses = w.step(&act);
        if ration > 0 {
            // it eats what it scores: the last block of steps, by its task's probes
            let mut fed = 0i64;
            if w.events.len() - block >= FOOD_BLOCK {
                let (b, e): (&[world::Event], &[world::Event]) = (&w.events[block..], &[]);
                let row = match task {
                    Task::Reversal { period, .. } => observe::probes(b, period, e, e, e, e, e),
                    Task::Conditioning(..) => observe::probes(e, 1, b, e, e, e, e),
                    Task::Reading(_) => observe::probes(e, 1, e, b, e, e, e),
                    Task::Sequence(_) => observe::probes(e, 1, e, e, b, e, e),
                    Task::Maze(_) => observe::probes(e, 1, e, e, e, b, e),
                    Task::Screen(_) => observe::probes(e, 1, e, e, e, e, b),
                    Task::Mixed(_) => {
                        let mut r = [0; observe::NPROBES];
                        r[13] = observe::select(b);
                        r
                    }
                    Task::Feel(_) => {
                        let mut r = [0; observe::NPROBES];
                        r[14] = observe::feel(b);
                        r
                    }
                };
                fed = food.saturating_mul(row.iter().map(|&x| x.max(0) as i64).sum::<i64>()) / 1024;
                block = w.events.len();
            }
            store = (store + ration + fed).min(ration.saturating_mul(256));
            if store < 0 {
                act = BitVector::new(world::ACT_BITS, Some(0));
                continue;
            }
        }
        let before = net.work;
        net.tick(AdvanceMode::Clear, 0);
        net.write_input(&senses, 0);
        act = net.read_output().clone();
        store -= (net.work - before).min(i64::MAX as u64 / 4) as i64;
    }
    NOTES.with(|n| if let Some(v) = n.borrow_mut().as_mut() { v.extend(net.notes()) });
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
/// for copy, echo and babble; a patterned stream for surprise; a T-maze for maze; a screen for screen),
/// `trials` trials or tokens each, with gaps of 0 to `gap` steps between trials; the other
/// probes read 0.
pub fn probe_scores<G: Evolvable>(g: &G, trials: usize, physics: Physics, seed: u64, gap: usize, wanted: &[usize]) -> (observe::ProbeRow, u64) {
    probe_scores_in(g, trials, physics, seed, gap, wanted, Lesson::default())
}

/// `probe_scores` in a world set by a lesson.
pub fn probe_scores_in<G: Evolvable>(g: &G, trials: usize, physics: Physics, seed: u64, gap: usize, wanted: &[usize], lesson: Lesson) -> (observe::ProbeRow, u64) {
    let period = (trials / (lesson.swaps + 1)).max(1);
    let want = |ps: &[usize]| ps.iter().any(|p| wanted.contains(p));
    let (rev, w1) = if want(&[0, 1, 2, 3, 5]) { live_lesson(g, Task::Reversal { trials, period, worst_phase: false }, physics, seed, gap, lesson) } else { (vec![], 0) };
    let (del, w2) = if want(&[4]) { live_lesson(g, Task::Conditioning(Conditioning::Delayed, trials), physics, seed ^ 0xDE1A, gap, lesson) } else { (vec![], 0) };
    let (seq, w3) = if want(&[6, 7, 8]) { live_lesson(g, Task::Reading(trials), physics, seed ^ 0x5E0, 0, Lesson { period: 0, noise: 0, ..lesson }) } else { (vec![], 0) };
    let (pat, w4) = if want(&[9]) { live_lesson(g, Task::Sequence(trials), physics, seed ^ 0x9ED, 0, lesson) } else { (vec![], 0) };
    let (maz, w5) = if want(&[10, 12]) { live_lesson(g, Task::Maze(trials), physics, seed ^ 0x3A2E, gap, lesson) } else { (vec![], 0) };
    let (scr, w6) = if want(&[11]) { live_lesson(g, Task::Screen(trials), physics, seed ^ 0x5C2E, gap, lesson) } else { (vec![], 0) };
    let blocks = (trials / lesson.mixed_block.max(1)).max(2);
    let (mix, w7) = if want(&[13]) { live_lesson(g, Task::Mixed(blocks), physics, seed ^ 0x313C, gap, lesson) } else { (vec![], 0) };
    let (fel, w8) = if want(&[14]) { live_lesson(g, Task::Feel(trials * 5), physics, seed ^ 0xFEE1, gap, lesson) } else { (vec![], 0) };
    let steps = (rev.len() + del.len() + seq.len() + pat.len() + maz.len() + scr.len() + mix.len() + fel.len()).max(1) as u64;
    let mut row = observe::probes(&rev, period, &del, &seq, &pat, &maz, &scr);
    row[13] = observe::select(&mix);
    row[14] = observe::feel(&fel);
    for (p, x) in row.iter_mut().enumerate() {
        if !wanted.contains(&p) {
            *x = 0;
        }
    }
    (row, (w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8) / steps)
}

/// All the probes.
pub const ALL_PROBES: [usize; observe::NPROBES] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];

/// What probing a genome found: its probes and work per step in each seed, and its mean work
/// per step.
pub struct Probed<G> {
    pub probes: Vec<observe::ProbeRow>,
    pub works: Vec<u64>,
    pub work: u64,
    pub genome: G,
    /// what its lifetimes found (summed over worlds)
    pub notes: Vec<Note>,
}

/// Each genome's probes in each seed, spread over the machine's cores (a genome whose network
/// panics scores worst on every probe).
pub fn probe_all<G: Evolvable>(population: Vec<G>, trials: usize, physics: Physics, seeds: &[u64], gap: usize, wanted: &[usize]) -> Vec<Probed<G>> {
    probe_all_in(population, trials, physics, seeds, gap, wanted, Lesson::default())
}

/// The carrying cost: work per step charged for each token of a genome (`set_carry`), so that
/// carrying what does nothing costs something, as a plasmid costs the cell that copies it. 0: none.
static CARRY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Sets the carrying cost for every probing from now on.
pub fn set_carry(per_token: u64) {
    CARRY.store(per_token, std::sync::atomic::Ordering::Relaxed);
}

pub fn carry() -> u64 {
    CARRY.load(std::sync::atomic::Ordering::Relaxed)
}

/// `probe_all` in a world set by a lesson.
pub fn probe_all_in<G: Evolvable>(population: Vec<G>, trials: usize, physics: Physics, seeds: &[u64], gap: usize, wanted: &[usize], lesson: Lesson) -> Vec<Probed<G>> {
    // one job a (genome, world) lifetime set, taken from a shared counter by whichever thread is
    // free: genomes differ in cost by orders of magnitude, so splitting by genome left threads
    // idle, and more cores than genomes went unused. Each job is deterministic by its seed, and
    // results are put back in order, so the outcome does not depend on the number of threads.
    let jobs = population.len() * seeds.len();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).max(1).min(jobs.max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut results: Vec<(observe::ProbeRow, u64)> = vec![([0; observe::NPROBES], 0); jobs];
    let mut notes: Vec<Vec<Note>> = vec![Vec::new(); jobs];
    std::thread::scope(|sc| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let (next, population) = (&next, &population);
                sc.spawn(move || {
                    let mut done = Vec::new();
                    loop {
                        let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if j >= jobs {
                            break;
                        }
                        let (g, s) = (&population[j / seeds.len()], seeds[j % seeds.len()]);
                        NOTES.with(|n| *n.borrow_mut() = Some(Vec::new()));
                        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| probe_scores_in(g, trials, physics, s, gap, wanted, lesson)))
                            .unwrap_or(([-1024; observe::NPROBES], u64::MAX / 2));
                        let found = NOTES.with(|n| n.borrow_mut().take()).unwrap_or_default();
                        done.push((j, r, found));
                    }
                    done
                })
            })
            .collect();
        for h in handles {
            for (j, r, found) in h.join().unwrap() {
                results[j] = r;
                notes[j] = found;
            }
        }
    });
    let carry = carry();
    population
        .into_iter()
        .enumerate()
        .map(|(i, g)| {
            let r = &results[i * seeds.len()..(i + 1) * seeds.len()];
            let cost = carry * g.size() as u64;
            let works: Vec<u64> = r.iter().map(|x| x.1.saturating_add(cost)).collect();
            let work = works.iter().sum::<u64>() / seeds.len().max(1) as u64;
            let found = merge_notes(notes[i * seeds.len()..(i + 1) * seeds.len()].iter().flatten().copied());
            Probed { probes: r.iter().map(|x| x.0).collect(), works, work, genome: g, notes: found }
        })
        .collect()
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
    pub stages: Vec<Stage>,
    /// scored at the frontier: on each other island's current stage, its probes in its world
    /// (each probe from the first island that has it), not on its own stages
    pub frontier: bool,
}

impl Island {
    /// An island with one stage: these probes in the plain world, forever.
    pub fn plain(name: &str, probes: Vec<usize>) -> Self {
        Self { name: name.into(), stages: vec![Stage { name: "all".into(), probes, lesson: Lesson::default(), goal: vec![], hold: 0 }], frontier: false }
    }
}

/// One stage of an island's curriculum: the probes it is selected on (list the earlier stages'
/// too, to keep them), the world it lives in, and its goal: the island moves to the next stage
/// once its population's training mean on each goal probe reaches its score for `hold`
/// generations in a row. A stage with no goal is the last. The network never sees the stage.
#[derive(Clone, Debug)]
pub struct Stage {
    pub name: String,
    pub probes: Vec<usize>,
    pub lesson: Lesson,
    pub goal: Vec<(usize, i32)>,
    pub hold: usize,
}

/// One generation of island evolution, per island: the population ranked (by its own probes,
/// training worlds), its probe means and its work per step; and the plasmids moved this
/// generation: (from island, to island, what moved).
pub struct IslandGeneration<G> {
    /// each island's stage this generation (what it was scored on), generations on it so far,
    /// and how many generations in a row its goal has been met
    pub stage: Vec<usize>,
    pub held: Vec<usize>,
    pub met: Vec<usize>,
    /// the elite's means on the held-out worlds, for each island checked before moving on
    pub rechecked: Vec<Option<observe::ProbeRow>>,
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
/// `pop`) and `stages` its starting stage, and generations run from `first` to `generations` (so
/// a checkpoint can be resumed). An island whose population meets its stage's goal for the
/// stage's `hold` generations in a row moves to the next stage, if its elite also meets the goal
/// on the `held_out` worlds (none: no check), so a population fitted to its training worlds does
/// not climb (and with `fresh` the training worlds are new each generation, so there is no fixed
/// set to fit); until it does, it is checked again each generation the training goal holds.
pub fn evolve_islands<G: Evolvable>(
    islands: &[Island],
    start: Vec<Vec<G>>,
    stages: Vec<usize>,
    first: usize,
    pop: usize,
    trials: usize,
    physics: Physics,
    seeds: &[u64],
    fresh: bool,
    held_out: &[u64],
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
    let mut stage: Vec<usize> = (0..k).map(|i| stages.get(i).copied().unwrap_or(0).min(islands[i].stages.len().saturating_sub(1))).collect();
    let mut met = vec![0usize; k];
    let mut on = vec![0usize; k];
    for generation in first..generations {
        // (`fresh`: the training worlds are new every generation, the same for every genome in
        // it, so a population cannot fit a fixed set of worlds: their codes and sequences)
        let train: Vec<u64> = if fresh { seeds.iter().map(|s| s.wrapping_add((generation as u64 + 1).wrapping_mul(1_000_003))).collect() } else { seeds.to_vec() };
        let seeds = &train[..];
        let now = stage.clone();
        let mut held = Vec::with_capacity(k);
        let mut met_now = Vec::with_capacity(k);
        let mut rechecked = vec![None; k];
        let mut ranked: Vec<Vec<G>> = Vec::with_capacity(k);
        let mut means = Vec::with_capacity(k);
        let mut work = Vec::with_capacity(k);
        let mut next: Vec<Vec<G>> = Vec::with_capacity(k);
        for (i, isl) in islands.iter().enumerate() {
            let st = &isl.stages[stage[i]];
            let (scored, wanted) = if isl.frontier {
                // at the frontier: each other island's current stage, in its own world, for the
                // probes no earlier island has given
                let genomes: Vec<G> = pops[i].drain(..).collect();
                let mut wanted: Vec<usize> = Vec::new();
                let mut rows = vec![vec![[0i32; observe::NPROBES]; seeds.len()]; genomes.len()];
                let mut works = vec![vec![0u64; seeds.len()]; genomes.len()];
                let mut found: Vec<Vec<Note>> = vec![Vec::new(); genomes.len()];
                for (j, other) in islands.iter().enumerate() {
                    if j == i || other.frontier {
                        continue;
                    }
                    let os = &other.stages[stage[j]];
                    let new: Vec<usize> = os.probes.iter().copied().filter(|p| !wanted.contains(p)).collect();
                    if new.is_empty() {
                        continue;
                    }
                    wanted.extend(&new);
                    for (g, s) in probe_all_in(genomes.clone(), trials, physics, seeds, gap, &new, os.lesson).iter().enumerate() {
                        for (k, r) in s.probes.iter().enumerate() {
                            for &p in &new {
                                rows[g][k][p] = r[p];
                            }
                            works[g][k] = works[g][k].saturating_add(s.works[k]);
                        }
                        found[g].extend(&s.notes);
                    }
                }
                let scored: Vec<Probed<G>> = genomes.into_iter().enumerate().map(|(g, genome)| {
                    let work = works[g].iter().sum::<u64>() / seeds.len().max(1) as u64;
                    Probed { probes: rows[g].clone(), works: works[g].clone(), work, genome, notes: merge_notes(found[g].iter().copied()) }
                }).collect();
                (scored, wanted)
            } else {
                (probe_all_in(pops[i].drain(..).collect(), trials, physics, seeds, gap, &st.probes, st.lesson), st.probes.clone())
            };
            // what each genome's lifetimes found goes into it before it breeds (its marks, and any
            // explored connection it keeps), so the elite and the children carry it
            let mut scored = scored;
            for s in scored.iter_mut() {
                let notes = std::mem::take(&mut s.notes);
                s.genome.inherit(&notes);
                s.notes = notes;
            }
            let (order, children) = lexicase_breed(&scored, &wanted, thrift, pop, elite, &mut rng);
            let m: Vec<observe::ProbeRow> = scored.iter().map(|s| probe_means(&s.probes)).collect();
            let mean = probe_means(&m);
            on[i] += 1;
            held.push(on[i]);
            // the curriculum: the goal met for `hold` generations in a row moves the island on
            if !st.goal.is_empty() && stage[i] + 1 < isl.stages.len() {
                met[i] = if st.goal.iter().all(|&(p, x)| mean[p] >= x) { met[i] + 1 } else { 0 };
                met_now.push(met[i]);
                if met[i] >= st.hold.max(1) {
                    let pass = held_out.is_empty() || {
                        let top: Vec<G> = order.iter().take(elite.max(1)).map(|&g| scored[g].genome.clone()).collect();
                        let h = probe_all_in(top, trials, physics, held_out, gap, &st.probes, st.lesson);
                        let m = probe_means(&h.iter().map(|s| probe_means(&s.probes)).collect::<Vec<_>>());
                        rechecked[i] = Some(m);
                        st.goal.iter().all(|&(p, x)| m[p] >= x)
                    };
                    if pass {
                        stage[i] += 1;
                        met[i] = 0;
                        on[i] = 0;
                    }
                }
            } else {
                met_now.push(0);
            }
            means.push(mean);
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
        report(generation, &IslandGeneration { stage: now, held, met: met_now, rechecked, ranked: ranked.clone(), means, work, moved });
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
    fn a_lesson_sets_the_tokens_and_the_teacher() {
        // a genome that does nothing: only the world's side shows
        let g = Genome { code: vec![] };
        let (ev, _) = live_lesson(&g, Task::Reading(200), Physics::Varied, 3, 0, Lesson { tokens: 2, teach: 8, ..Lesson::default() });
        assert!(ev.iter().filter_map(|e| e.shown).all(|t| t < 2), "two tokens only");
        assert!(ev.iter().filter(|e| e.shown.is_some()).all(|e| e.taught), "the teacher says every one");
        let (ev, _) = live_lesson(&g, Task::Reading(200), Physics::Varied, 3, 0, Lesson { tokens: 8, teach: 1, ..Lesson::default() });
        let taught = ev.iter().filter(|e| e.taught).count();
        assert!(taught > 5 && taught < 60, "about one in eight: {taught}");
        // the plain lesson is the old world
        let (a, _) = live_lesson(&g, Task::Reading(200), Physics::Varied, 3, 0, Lesson::default());
        let (b, _) = live_work(&g, Task::Reading(200), Physics::Varied, 3, 0);
        assert_eq!(a.iter().map(|e| (e.shown, e.taught)).collect::<Vec<_>>(), b.iter().map(|e| (e.shown, e.taught)).collect::<Vec<_>>());
    }

    #[test]
    fn a_lesson_reaches_the_conditioning_lifetimes() {
        // the delayed light flashes `delay` steps before the window, in the lifetime the remember
        // probe lives (a genome that does nothing: only the world shows)
        let g = Genome { code: vec![] };
        for d in [1usize, 3] {
            let (ev, _) = live_lesson(&g, Task::Conditioning(Conditioning::Delayed, 30), Physics::Varied, 4, 0, Lesson { delay: d, ..Lesson::default() });
            assert!(!ev.is_empty());
        }
        let w = |l: Lesson| {
            let (ev, _) = live_lesson(&g, Task::Reversal { trials: 30, period: 10, worst_phase: false }, Physics::Varied, 4, 0, l);
            ev.len()
        };
        // a shape lifetime runs as long as a plain one (only what the eye shows changes)
        assert_eq!(w(Lesson::default()), w(Lesson { blur: 4, shift: 32, ..Lesson::default() }));
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
