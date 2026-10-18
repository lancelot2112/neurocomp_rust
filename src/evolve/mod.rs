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
    /// The length of its wiring, for the wiring cost (`set_wire_cost`): by default none.
    fn wire_length(&self) -> usize {
        0
    }
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
    /// (delayed conditioning) a tone as the window opens (`World::go_tone`)
    pub go_tone: bool,
    pub blur: u32,
    pub shift: usize,
    pub maze_len: usize,
    pub cue_gone: bool,
    pub symbol: bool,
    pub maze_fixed: bool,
    pub maze_nocue: bool,
    /// reading: each token's sound made like the voice's code for it, in eighths of its lines (0:
    /// unrelated, as always; 8: heard exactly as it is said, so copying the ear into the voice
    /// echoes), `World::mirror_sounds`
    pub mirror: usize,
    /// reading: one's own voice is not heard, only the teacher (`World::quiet`)
    pub quiet: bool,
    /// the body's moves and tokens as place codes, the same every lifetime (`World::fix_motor`)
    pub motor_place: bool,
    /// reading: the teacher silent for the first this many tokens (`World::teacher_after`)
    pub babble: usize,
    /// sentences (`World::sentence`): a token written in any word of the voice is said, in order
    pub sentence: bool,
    /// reading: every token's glyph, sound and voice code in the first word of its channel
    /// (`World::one_word`)
    pub one_word: bool,
    /// (with `one_word`) the lines of each code, 0 for the usual 16
    pub code_lines: usize,
    /// the lights and the stroke at fixed places (`World::place_senses`)
    pub sense_place: bool,
    /// task selection (`Task::Mixed`): trials a block, and a cue for the block's task
    pub mixed_block: usize,
    pub mixed_cue: bool,
    /// (a mixed lifetime) the cue felt only for each block's first this many steps (0: all the
    /// block), the blocks' tasks drawn at random (not alternating), and a third task, still
    /// (glyphs shown, neither click nor voice)
    pub mixed_cue_steps: usize,
    pub mixed_random: bool,
    pub mixed_three: bool,
    /// (a mixed lifetime) the same stimuli in every block, whatever its task: a conditioning
    /// block's lights (paying only in a respond block), so only the cue tells the task
    pub mixed_same: bool,
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

/// A mixed lifetime's block `b` (`Task::Mixed`): even, a conditioning block; odd, a reading one
/// (`Lesson::mixed_random`: drawn; `mixed_three`: or a still block, glyphs shown as in reading).
fn mixed_block(w: &mut World, b: usize, lesson: &Lesson, tokens: usize, seed: u64) {
    let n = lesson.mixed_block.max(1);
    let kinds = if lesson.mixed_three { 3 } else { 2 };
    let task = if lesson.mixed_random { StdRng::seed_from_u64(seed ^ 0x7A5C ^ (b as u64).wrapping_mul(0x9E37_79B9)).gen_range(0..kinds) } else { b % kinds };
    w.task = Some(task);
    w.task_cue = lesson.mixed_cue.then_some(task);
    w.task_cue_left = (lesson.mixed_cue_steps > 0).then_some(lesson.mixed_cue_steps);
    w.no_pay = lesson.mixed_same && task != 0;
    if task == 0 || lesson.mixed_same {
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
        Self { tokens: 8, teach: 4, swaps: 3, delay: 3, vary_delay: 0, distract: false, two_lights: false, go_tone: false, period: 0, noise: 0, blur: 0, shift: 0, maze_len: 2, cue_gone: false, symbol: false, maze_fixed: false, maze_nocue: false, mirror: 0, quiet: false, motor_place: false, babble: 0, sentence: false, one_word: false, code_lines: 0, sense_place: false, mixed_block: 40, mixed_cue: false, mixed_cue_steps: 0, mixed_random: false, mixed_three: false, mixed_same: false, screen_w: 2, screen_h: 2, screen_lit: 1, screen_cell: 64, screen_walls: 0, screen_torch: false, screen_dark: false, ration: energy().0, food: energy().1 }
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
    /// (`set_trace`) a harness's look at each step of a lifetime on this thread
    static TRACE: std::cell::RefCell<Option<Box<dyn FnMut(&World, &RuntimeNetwork)>>> = const { std::cell::RefCell::new(None) };
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

/// A harness's look at every step of the lifetimes lived on this thread (the world and the network
/// after the step), for recording one lifetime; None stops it. It reads only.
pub fn set_trace(f: Option<Box<dyn FnMut(&World, &RuntimeNetwork)>>) {
    TRACE.with(|t| *t.borrow_mut() = f);
}

/// `live_work` in a world set by a lesson.
pub fn live_lesson<G: Evolvable>(g: &G, task: Task, physics: Physics, seed: u64, gap: usize, lesson: Lesson) -> (Vec<world::Event>, u64) {
    // (`set_twins`) lifetimes in pairs: seeds 2k and 2k+1 live the same world (seed k), the second
    // with its two lights' roles swapped and the maze's cue mirrored
    let (seed, twin) = if twins() { (seed >> 1, seed & 1 == 1) } else { (seed, false) };
    let p = match physics {
        Physics::Fixed(p) => p,
        Physics::Varied => seed.wrapping_mul(0xA24B_AED4_963E_E407),
    };
    // (a lesson's token count changes only reading lifetimes; conditioning keeps its 8)
    let reading = matches!(task, Task::Sequence(_) | Task::Reading(_) | Task::Mixed(_));
    let tokens = if reading { lesson.tokens.clamp(2, 128) } else { 8 };
    let mut w = World::with_physics(p, seed, tokens);
    w.gap_max = gap;
    w.teacher_rate = lesson.teach;
    // (`set_vary_lag`) a delayed lifetime's delay drawn for the lifetime, 1 to the rung's
    // (`set_vary_up`: the rung's to twice it, so a shorter memory scores in no lifetime)
    w.delay = if vary_lag() && vary_up() && lesson.delay > 0 && lesson.vary_delay == 0 {
        StdRng::seed_from_u64(seed ^ 0xDE1A_7A61).gen_range(lesson.delay..=2 * lesson.delay)
    } else if vary_lag() && lesson.delay > 1 && lesson.vary_delay == 0 {
        StdRng::seed_from_u64(seed ^ 0xDE1A_7A61).gen_range(1..=lesson.delay)
    } else {
        lesson.delay
    };
    w.vary_delay = lesson.vary_delay;
    w.distract = lesson.distract;
    w.two_lights = lesson.two_lights;
    w.go_tone = lesson.go_tone;
    // (`set_maze_mirror`) which half of the eye means which turn, drawn per lifetime
    if maze_mirror() {
        w.maze_mirror = StdRng::seed_from_u64(seed ^ 0x3A2F_A1E5).gen_bool(0.5);
    }
    // (`set_same_word_lights`) the shapes' two lights in one word
    if same_word_lights() && (lesson.blur > 0 || lesson.shift > 0) {
        w.same_word_lights();
    }
    w.blur = lesson.blur;
    w.shift = lesson.shift;
    w.maze_len = lesson.maze_len;
    w.cue_gone = lesson.cue_gone;
    w.symbol = lesson.symbol;
    w.maze_fixed = lesson.maze_fixed;
    w.maze_nocue = lesson.maze_nocue;
    w.quiet = lesson.quiet;
    w.teacher_after = lesson.babble;
    w.sentence = lesson.sentence || sentences();
    // (`set_body`) the body fixed for the whole run
    if body() != 0 {
        w.fix_body(body());
    }
    // (`set_one_word`) every reading lifetime's codes sparse, in one word
    if lesson.one_word || (reading && one_word() > 0) {
        w.one_word(p, if lesson.code_lines > 0 { lesson.code_lines } else { one_word() });
    } else if one_word() > 0 && one_word_voice() {
        // (`set_one_word_voice`) the voice's codes and their sounds one-word in every lifetime
        w.one_word_voice(p, one_word());
    }
    if lesson.motor_place {
        w.fix_motor();
    }
    if lesson.sense_place {
        w.place_senses();
    }
    if lesson.mirror > 0 {
        w.mirror_sounds(lesson.mirror, if body() != 0 { body() } else { p });
    }
    w.screen_w = lesson.screen_w;
    w.screen_h = lesson.screen_h;
    w.screen_lit = lesson.screen_lit;
    w.screen_cell = lesson.screen_cell;
    w.screen_walls = lesson.screen_walls;
    w.screen_torch = lesson.screen_torch;
    w.screen_dark = lesson.screen_dark;
    // (`set_cursor_away`) on every screen rung but the dark ones (glow), the cursor starts off the
    // target
    w.screen_away = cursor_away() && !lesson.screen_dark;
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
            // (`set_vary_period`) a cycle rung's period drawn for the lifetime, 1 to the rung's
            // (`set_vary_up`: the rung's to twice it, at most the tokens, so a detector of a shorter
            // cycle scores in no lifetime)
            let period = if vary_period() && vary_up() && lesson.period > 0 && lesson.period < tokens {
                r.gen_range(lesson.period..=(2 * lesson.period).min(tokens))
            } else if vary_period() && lesson.period > 1 && lesson.period <= tokens {
                r.gen_range(1..=lesson.period)
            } else {
                lesson.period
            };
            // (a prediction lifetime: a cycle drawn for this lifetime, some tokens replaced by a
            // random other one, which the observer marks as breaking it)
            let cycle: Vec<usize> = if fair_sequences() && period > 0 { fair_cycle(period, tokens, &mut r) } else { (0..period).map(|_| r.gen_range(0..tokens)).collect() };
            let (toks, odd) = prediction_stream(&cycle, n, lesson.noise, 0, tokens, &mut r);
            w.teacher = matches!(task, Task::Reading(_));
            w.sequence_marked(&toks, &odd)
        }
        Task::Maze(n) => w.maze(n),
        Task::Screen(n) => w.screen(n),
        Task::Mixed(_) => mixed_block(&mut w, 0, &lesson, tokens, seed),
        Task::Feel(n) => w.feel(n),
    }
    let mut blk = 0usize;
    // (`set_mixed_codes`) a mixed lifetime's lights the same kind of code as its glyphs
    if mixed_codes() && one_word() > 0 && matches!(task, Task::Mixed(_)) {
        w.one_word_lights(p, one_word());
    }
    // (`set_twins`) the second of a pair: the lights' roles swapped and the maze's cue mirrored,
    // so a policy fixed to one light or one side scores the opposite on its twin
    if twin {
        w.swap_lights();
        w.maze_mirror = !w.maze_mirror;
    }
    let mut net = g.build(&w, seed ^ 0x5EED);
    let mut act = BitVector::new(world::ACT_BITS, Some(0));
    // energy (`Lesson::ration`): the store, below 0 the network rests
    let (ration, food) = (lesson.ration as i64, lesson.food as i64);
    let mut store = ration.saturating_mul(64);
    // (`set_carry_energy`) what carrying the genome costs each step, from the same store
    let upkeep = if carry_energy() { (carry() as i64).saturating_mul(g.size() as i64).saturating_add((wire_cost() as i64).saturating_mul(g.wire_length() as i64)) } else { 0 };
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
        // (the muscle's response) the act shows what the body did
        if muscle() {
            let (said, moved) = w.events.last().map_or((None, None), |e| (e.said, e.moved));
            net.set_output(&w.act(said, moved));
        }
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
                fed = if food_from_world() {
                    // (`set_food_from_world`) what the world gave: an eighth of `food` for each
                    // stroke in the block (a block of 50 steps scoring about 300 fed about as much)
                    (food / 8).saturating_mul(b.iter().filter(|e| e.stroked).count() as i64)
                } else {
                    food.saturating_mul(row.iter().map(|&x| x.max(0) as i64).sum::<i64>()) / 1024
                };
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
        TRACE.with(|t| if let Some(f) = t.borrow_mut().as_mut() { f(&w, &net) });
        store -= (net.work - before).min(i64::MAX as u64 / 4) as i64 + upkeep;
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
    // (the novelty island) its behaviour reads every kind of lifetime
    let beh = wanted.iter().any(|&p| (21..=28).contains(&p));
    let want = |ps: &[usize]| beh || ps.iter().any(|p| wanted.contains(p));
    let (rev, w1) = if want(&[0, 1, 2, 3, 5]) { live_lesson(g, Task::Reversal { trials, period, worst_phase: false }, physics, seed, gap, lesson) } else { (vec![], 0) };
    let (del, w2) = if want(&[4]) { live_lesson(g, Task::Conditioning(Conditioning::Delayed, trials), physics, seed ^ 0xDE1A, gap, lesson) } else { (vec![], 0) };
    let (seq, w3) = if want(&[6, 7, 8, 15]) { live_lesson(g, Task::Reading(trials), physics, seed ^ 0x5E0, 0, Lesson { period: 0, noise: 0, ..lesson }) } else { (vec![], 0) };
    let (pat, w4) = if want(&[9, 16]) { live_lesson(g, Task::Sequence(trials), physics, seed ^ 0x9ED, 0, lesson) } else { (vec![], 0) };
    let (maz, w5) = if want(&[10, 12]) { live_lesson(g, Task::Maze(trials), physics, seed ^ 0x3A2E, gap, lesson) } else { (vec![], 0) };
    let (scr, w6) = if want(&[11]) { live_lesson(g, Task::Screen(trials), physics, seed ^ 0x5C2E, gap, lesson) } else { (vec![], 0) };
    let blocks = (trials / lesson.mixed_block.max(1)).max(2);
    let (mix, w7) = if want(&[13, 20]) { live_lesson(g, Task::Mixed(blocks), physics, seed ^ 0x313C, gap, lesson) } else { (vec![], 0) };
    let (fel, w8) = if want(&[14, 17]) { live_lesson(g, Task::Feel(trials * 5), physics, seed ^ 0xFEE1, gap, lesson) } else { (vec![], 0) };
    let steps = (rev.len() + del.len() + seq.len() + pat.len() + maz.len() + scr.len() + mix.len() + fel.len()).max(1) as u64;
    let mut row = observe::probes(&rev, period, &del, &seq, &pat, &maz, &scr);
    row[13] = observe::select(&mix);
    row[19] = observe::select_graded(&mix);
    // (selection's step rung) responding in a mixed lifetime's conditioning blocks, alone
    row[20] = observe::select_parts(&mix).0;
    if beh {
        let b = observe::behaviour(&[&rev, &del, &seq, &pat, &maz, &scr, &mix, &fel]);
        row[21..29].copy_from_slice(&b);
    }
    row[14] = observe::feel(&fel);
    row[17] = observe::feel_graded(&fel);
    for (p, x) in row.iter_mut().enumerate() {
        if !wanted.contains(&p) && !wanted.iter().any(|&w| graded(w) == p && w != p) {
            *x = 0;
        }
    }
    (row, (w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8) / steps)
}

/// All the probes.
/// All the probes (the graded ones, 15 to 17, stand in for echo, surprise and feel only in
/// selection, with `set_partial`).
pub const ALL_PROBES: [usize; 15] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];

/// Partial credit (`set_partial`): in selection, echo, surprise and feel are replaced by graded
/// versions (`observe::echo_graded`, `surprise_graded`, `feel_graded`) that give a near miss part
/// of the score; a rung's goal is still checked on the probe itself. Off by default.
static PARTIAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_partial(on: bool) {
    PARTIAL.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn partial() -> bool {
    PARTIAL.load(std::sync::atomic::Ordering::Relaxed)
}

/// The probe selection uses for probe `p`: its graded version with `set_partial`, else itself.
pub fn graded(p: usize) -> usize {
    if partial() { match p { 7 => 15, 9 if partial_surprise() => 31, 9 => 16, 14 => 17, 4 if partial_remember() => 18, 13 if partial_select() => 19, 11 if partial_screen() => 30, _ => p } } else { p }
}

/// Surprise shaped from clicking at all (`set_partial_surprise`, with `set_partial`): selection uses
/// `observe::surprise_shaped` in place of `surprise_graded`, clicking after breaks at all paying a
/// quarter before clicking only after them pays the rest. Off by default.
static PARTIAL_SURPRISE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_partial_surprise(on: bool) {
    PARTIAL_SURPRISE.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn partial_surprise() -> bool {
    PARTIAL_SURPRISE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Partial credit on the screen too (`set_partial_screen`, with `set_partial`): selection uses
/// `observe::screen_graded`, reaching the target without a hit counting half a hit. Off by default.
static PARTIAL_SCREEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_partial_screen(on: bool) {
    PARTIAL_SCREEN.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn partial_screen() -> bool {
    PARTIAL_SCREEN.load(std::sync::atomic::Ordering::Relaxed)
}

/// Partial credit on select too (`set_partial_select`, with `set_partial`): selection uses
/// `observe::select_graded`, the costs at a quarter. Off by default.
static PARTIAL_SELECT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_partial_select(on: bool) {
    PARTIAL_SELECT.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn partial_select() -> bool {
    PARTIAL_SELECT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Partial credit on remember too (`set_partial_remember`, with `set_partial`): selection uses
/// `observe::remember_graded`, a click in the two steps after the window counting half. Off by
/// default.
static PARTIAL_REMEMBER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_partial_remember(on: bool) {
    PARTIAL_REMEMBER.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn partial_remember() -> bool {
    PARTIAL_REMEMBER.load(std::sync::atomic::Ordering::Relaxed)
}

/// The elite's pass (`set_elite_pass`): a rung's goal is checked against the mean of the island's
/// elite (its best `elite` genomes) on the training worlds, not of its whole population (whose
/// fresh mutants pull the mean down), before the elite's held-out check. Off by default.
static ELITE_PASS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_elite_pass(on: bool) {
    ELITE_PASS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn elite_pass() -> bool {
    ELITE_PASS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Passed rungs kept (`set_keep_rungs`): each generation an island above its first rung is also
/// scored on one of the rungs it passed, drawn at random, in that rung's own world (its delay, its
/// blur, its lesson), and those rows join its lexicase cases, so a network that loses an earlier
/// ability is selected against and each rung builds on the ones below. A rung's goal is still
/// judged on the current rung's world alone. One more probing an island a generation. Off by
/// default.
static KEEP_RUNGS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_keep_rungs(on: bool) {
    KEEP_RUNGS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn keep_rungs() -> bool {
    KEEP_RUNGS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Size classes (`set_size_classes`): a population's genomes are sorted by size into classes a
/// factor of four apart (1-3 tokens, 4-15, 16-63, 64-255, 256 and more), and lexicase keeps a
/// mixture of them: each occupied class first keeps its own lexicase choice among the elite (as
/// far as the elite goes), and half of all children have a parent chosen by lexicase within a
/// class drawn at random from the occupied ones (the other parent of a recombined child is
/// chosen from the whole population, so circuits still move between sizes). A small network that
/// does a simple thing well is then not crowded out by bigger ones that score more, nor the other
/// way round. Off by default.
static SIZE_CLASSES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_size_classes(on: bool) {
    SIZE_CLASSES.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn size_classes() -> bool {
    SIZE_CLASSES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Thrift relaxed while stuck (`set_thrift_relax`, with thrift): a selection takes the thrift
/// cases with chance `scale / (scale + g)`, g the generations the island has been on its current
/// rung, else leaves them out: full thrift on a fresh rung, so the easy rungs are passed by small
/// networks; less and less the longer a rung goes unpassed, so networks can grow when a rung needs
/// more than they have; full again after each pass, so the growth is trimmed. 0: off (thrift
/// always taken, as before).
static THRIFT_RELAX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn set_thrift_relax(scale: usize) {
    THRIFT_RELAX.store(scale, std::sync::atomic::Ordering::Relaxed);
}

pub fn thrift_relax() -> usize {
    THRIFT_RELAX.load(std::sync::atomic::Ordering::Relaxed)
}

/// A genome's size class: floor(log4(size)), 4 and more together.
pub fn size_class(size: usize) -> usize {
    ((usize::BITS - size.max(1).leading_zeros() - 1) / 2).min(4) as usize
}

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

/// The carrying cost paid as energy (`set_carry_energy`, with energy on): each step a network
/// pays `carry` a token of its genome from the same store its work comes out of, so carrying and
/// computing compete for what it earns; it is then no longer added to the work that thrift and
/// the ranking compare. Off by default (the cost is added to the work per step, and spends no
/// energy).
static CARRY_ENERGY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_carry_energy(on: bool) {
    CARRY_ENERGY.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// A fixed body (`set_body`): the body's codes (`World::fix_body`) drawn from this seed in every
/// lifetime of a run, the world's still drawn with each world; 0 off (the default).
static BODY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn set_body(seed: u64) {
    BODY.store(seed, std::sync::atomic::Ordering::Relaxed);
}

pub fn body() -> u64 {
    BODY.load(std::sync::atomic::Ordering::Relaxed)
}

/// The generalist scored on passed rungs too (`set_frontier_passed`): at the frontier, besides
/// each other island's current rung, its last passed rung, in its own world, as more selection
/// cases. Off by default.
static FRONTIER_PASSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_frontier_passed(on: bool) {
    FRONTIER_PASSED.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn frontier_passed() -> bool {
    FRONTIER_PASSED.load(std::sync::atomic::Ordering::Relaxed)
}

/// The generalist on ladders of its own (`set_frontier_ladders`, the rung it starts on in each
/// island's ladder; None off, the default): at the frontier it holds its own rung on each other
/// island's ladder, scored on it in its world, and on one rung below it drawn at random, so it is
/// still shown what it passed; it climbs each ladder as that island does (the goal on the elite's
/// mean held for the rung's `hold` generations, then the held-out check). It replaces scoring on
/// the other islands' current rungs (and `set_frontier_passed`).
static FRONTIER_LADDERS: std::sync::Mutex<Option<Vec<usize>>> = std::sync::Mutex::new(None);

pub fn set_frontier_ladders(start: Option<Vec<usize>>) {
    *FRONTIER_LADDERS.lock().unwrap() = start;
}

pub fn frontier_ladders() -> Option<Vec<usize>> {
    FRONTIER_LADDERS.lock().unwrap().clone()
}

/// Fair prediction streams (`set_fair_sequences`): a cycle uses each of its tokens equally often
/// (`fair_cycle`), and a token that breaks it is one of the cycle's own tokens, one that may follow
/// the token before it where the cycle has such (`fair_odd`). So a rare token never marks a break
/// and, on a cycle where each token has two successors, the token before does not either: only the
/// two before do. Off by default.
static FAIR_SEQUENCES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_fair_sequences(on: bool) {
    FAIR_SEQUENCES.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn fair_sequences() -> bool {
    FAIR_SEQUENCES.load(std::sync::atomic::Ordering::Relaxed)
}

/// The generalist's passes reach the islands (`set_migrate`, with `set_frontier_ladders`): when it
/// passes a rung on an island's ladder that the island has not passed, its best genomes are copied
/// into that island's next population, to win or lose there. Off by default.
static MIGRATE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_migrate(on: bool) {
    MIGRATE.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn migrate() -> bool {
    MIGRATE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Novelty's species reach the islands (`set_novel_migrate`, with `set_novelty`): when the novelty
/// island founds a species useful on a probe, its founder is copied into the next population of
/// every island whose current rung is scored on that probe, to win or lose there. Off by default.
static NOVEL_MIGRATE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_novel_migrate(on: bool) {
    NOVEL_MIGRATE.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn novel_migrate() -> bool {
    NOVEL_MIGRATE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Step 2 of the learning plan (experiment 17), each off by default:
/// - `set_twins`: lifetimes in pairs (seeds 2k, 2k+1) in one world, the second with its lights'
///   roles swapped and the maze's cue mirrored, so a fixed policy nets 0 over a pair;
/// - `set_vary_lag`: a delayed lifetime's delay drawn from 1 to the rung's;
/// - `set_vary_period`: a cycle rung's period drawn from 1 to the rung's;
/// - `set_mixed_codes`: a mixed lifetime's lights drawn as one-word codes, as its glyphs are.
static TWINS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static VARY_LAG: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static VARY_PERIOD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static MIXED_CODES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_twins(on: bool) {
    TWINS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn twins() -> bool {
    TWINS.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_vary_lag(on: bool) {
    VARY_LAG.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn vary_lag() -> bool {
    VARY_LAG.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_vary_period(on: bool) {
    VARY_PERIOD.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn vary_period() -> bool {
    VARY_PERIOD.load(std::sync::atomic::Ordering::Relaxed)
}

/// The drawn delay and period from the rung's up (`set_vary_up`, with `set_vary_lag` or
/// `set_vary_period`): the rung's to twice it, not 1 to the rung's. Drawn down, a memory or cycle
/// detector for a shorter one scored in the shorter lifetimes, enough to pass: run 34's lag8 passer
/// scored 260 drawn and 0 at 8, its cycle3 passer (a cycle1 detector) 293 drawn and 5 at 3. Drawn
/// up, it scores in none, and one wired to the rung's value alone in about half. Off by default.
static VARY_UP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_vary_up(on: bool) {
    VARY_UP.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn vary_up() -> bool {
    VARY_UP.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_mixed_codes(on: bool) {
    MIXED_CODES.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn mixed_codes() -> bool {
    MIXED_CODES.load(std::sync::atomic::Ordering::Relaxed)
}

/// The islands' best migrate into the generalist (`set_migrate_up`, with `set_frontier_ladders`):
/// every this many generations each island's best (`set_migrate_up_n` of them), and an island's
/// best two whenever it passes a rung the generalist has not, beside the generalist's own children,
/// up to `set_generalist_room` genomes; 0 off (the default).
static MIGRATE_UP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn set_migrate_up(every: usize) {
    MIGRATE_UP.store(every, std::sync::atomic::Ordering::Relaxed);
}

pub fn migrate_up() -> usize {
    MIGRATE_UP.load(std::sync::atomic::Ordering::Relaxed)
}

/// (`set_migrate_up`) how many of each island's best go up at each periodic move (1 by default),
/// and the most genomes the generalist holds with its migrants (its own size by default: then no
/// room, so `set_generalist_room` gives it some).
static MIGRATE_UP_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);
static GENERALIST_ROOM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn set_migrate_up_n(n: usize) {
    MIGRATE_UP_N.store(n, std::sync::atomic::Ordering::Relaxed);
}

pub fn migrate_up_n() -> usize {
    MIGRATE_UP_N.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_generalist_room(n: usize) {
    GENERALIST_ROOM.store(n, std::sync::atomic::Ordering::Relaxed);
}

pub fn generalist_room() -> usize {
    GENERALIST_ROOM.load(std::sync::atomic::Ordering::Relaxed)
}

/// The novelty island (`ISL_NOVELTY` in examples/islands.rs): an island named "novelty" is selected
/// on the novelty of its behaviour (`observe::behaviour`: what it does in each kind of lifetime),
/// the mean distance to its 8 nearest among its population and an archive of past behaviours. A
/// novel genome whose mean on some probe (babble aside) reaches `NOVELTY_USEFUL` and whose behaviour
/// is at least `NOVELTY_APART` from every species so far founds a species: its founder is kept every
/// generation and the island grows by four genomes, up to `set_novelty_max`.
pub const NOVELTY_USEFUL: i32 = 200;
pub const NOVELTY_APART: i64 = 512;
static NOVELTY_MAX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(128);

pub fn set_novelty_max(n: usize) {
    NOVELTY_MAX.store(n, std::sync::atomic::Ordering::Relaxed);
}

pub fn novelty_max() -> usize {
    NOVELTY_MAX.load(std::sync::atomic::Ordering::Relaxed)
}

/// The cursor off the target at a screen trial's start (`set_cursor_away`, `World::screen_away`), on
/// the rungs that are not dark. Off by default.
static CURSOR_AWAY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_cursor_away(on: bool) {
    CURSOR_AWAY.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn cursor_away() -> bool {
    CURSOR_AWAY.load(std::sync::atomic::Ordering::Relaxed)
}

/// The voice one-word everywhere (`set_one_word_voice`, with `set_one_word`): its codes and the
/// sounds they make drawn as one-word codes in every lifetime, not only those with tokens to read,
/// so the voice is the same body in a conditioning or maze lifetime as in reading. Off by default.
static ONE_WORD_VOICE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_one_word_voice(on: bool) {
    ONE_WORD_VOICE.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn one_word_voice() -> bool {
    ONE_WORD_VOICE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Held-out worlds drawn fresh at every check (`set_fresh_held`): each check of a rung's goal on
/// held-out worlds uses as many worlds as it was given, new ones each time (seeds from 2^40 on, by
/// generation), so repeated checks on one set cannot let a lucky genome through. Off by default.
static FRESH_HELD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_fresh_held(on: bool) {
    FRESH_HELD.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn fresh_held() -> bool {
    FRESH_HELD.load(std::sync::atomic::Ordering::Relaxed)
}

/// The maze's spatial cue mirrored per lifetime (`set_maze_mirror`, `World::maze_mirror`). Off by
/// default.
static MAZE_MIRROR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_maze_mirror(on: bool) {
    MAZE_MIRROR.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn maze_mirror() -> bool {
    MAZE_MIRROR.load(std::sync::atomic::Ordering::Relaxed)
}

/// The shapes' two lights in one word (`set_same_word_lights`, `World::same_word_lights`), in
/// lifetimes with blur or shift. Off by default.
static SAME_WORD_LIGHTS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_same_word_lights(on: bool) {
    SAME_WORD_LIGHTS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn same_word_lights() -> bool {
    SAME_WORD_LIGHTS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Food from the world (`set_food_from_world`): with energy on, each block's food is what the
/// world gave in it (an eighth of `food` for each stroke), not the observer's probe scores on it, so no
/// score reaches the program. Off by default.
static FOOD_FROM_WORLD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_food_from_world(on: bool) {
    FOOD_FROM_WORLD.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn food_from_world() -> bool {
    FOOD_FROM_WORLD.load(std::sync::atomic::Ordering::Relaxed)
}

/// Order cycles of varied shape (`set_varied_orders`, with `set_fair_sequences`): see `fair_cycle`.
/// Off by default.
static VARIED_ORDERS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_varied_orders(on: bool) {
    VARIED_ORDERS.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn varied_orders() -> bool {
    VARIED_ORDERS.load(std::sync::atomic::Ordering::Relaxed)
}

/// A prediction lifetime's stream: `n` tokens of `cycle` (empty: random tokens), some replaced by
/// a break (`noise` in 8, fair with `set_fair_sequences`, each held `hold` steps), and which are
/// breaks. (Holding breaks was tried as an easier cycle3 and did not help a partly right
/// detector: `held_frequent_breaks_do_not_help_a_partly_right_detector`.)
pub fn prediction_stream(cycle: &[usize], n: usize, noise: u32, hold: usize, tokens: usize, r: &mut StdRng) -> (Vec<usize>, Vec<bool>) {
    let mut odd = Vec::with_capacity(n);
    let mut last: Option<usize> = None;
    // (`Lesson::odd_hold`) steps of a held break still to come
    let mut held = 0usize;
    let toks: Vec<usize> = (0..n)
        .map(|i| {
            if cycle.is_empty() {
                odd.push(false);
                return r.gen_range(0..tokens);
            }
            let c = cycle[i % cycle.len()];
            // (`set_fair_sequences`: never two breaks in a row, which would leave a pair the
            // cycle never has, telling the break by the token before alone)
            let go_on = held > 0;
            held = held.saturating_sub(1);
            let start = !go_on && noise > 0 && r.gen_ratio(noise.min(8), 8) && !(fair_sequences() && odd.last() == Some(&true));
            if start && hold > 1 {
                held = hold - 1;
            }
            let t = if go_on || start {
                if fair_sequences() { fair_odd(cycle, i, last, tokens, r) } else { (c + r.gen_range(1..tokens)) % tokens }
            } else {
                c
            };
            odd.push(t != c);
            last = Some(t);
            t
        })
        .collect();
    (toks, odd)
}

/// A cycle of `period` tokens out of `tokens`, each used equally often: up to `tokens` long, that
/// many distinct tokens in a random order; longer (a multiple), each token said `period / tokens`
/// times in a row, in a random order of tokens (period 6 of 3: A A B B C C, each token followed by
/// itself once and by the next once, so which comes is set by the two tokens before, not one).
pub fn fair_cycle(period: usize, tokens: usize, r: &mut StdRng) -> Vec<usize> {
    let tokens = tokens.max(1);
    let mut order: Vec<usize> = (0..tokens).collect();
    for i in (1..order.len()).rev() {
        order.swap(i, r.gen_range(0..=i));
    }
    if period <= tokens {
        order.truncate(period);
        return order;
    }
    // (`set_varied_orders`) each token said period / tokens times, in a random order whose pairs
    // (each token and the next, around the cycle) are all different and never a token twice: the
    // token before never says what comes, the two before always do, and the cycle's shape changes
    // from lifetime to lifetime (A B A C B C, A B C A C B ...), so no fixed rule on which tokens
    // repeat tells a break
    if varied_orders() && period % tokens == 0 {
        let reps = period / tokens;
        let mut c: Vec<usize> = (0..period).map(|i| i % tokens).collect();
        for _ in 0..10_000 {
            for i in (1..c.len()).rev() {
                c.swap(i, r.gen_range(0..=i));
            }
            let pairs: Vec<(usize, usize)> = (0..period).map(|i| (c[i], c[(i + 1) % period])).collect();
            let mut distinct = pairs.clone();
            distinct.sort_unstable();
            distinct.dedup();
            if distinct.len() == period && reps > 1 && pairs.iter().all(|p| p.0 != p.1) {
                return c;
            }
        }
    }
    let reps = period.div_ceil(tokens);
    order.iter().flat_map(|&t| std::iter::repeat(t).take(reps)).take(period).collect()
}

/// The token that breaks a fair cycle at position `i`: one the cycle has right after the token
/// shown before it (`before`; None: the cycle's own token before position `i`), other than the one
/// due; if there is none, any other of the cycle's tokens; if the cycle has one token only, any
/// other token. Breaks never come two in a row, so the token after a break makes a pair the cycle
/// has too.
pub fn fair_odd(cycle: &[usize], i: usize, before: Option<usize>, tokens: usize, r: &mut StdRng) -> usize {
    let p = cycle.len();
    let due = cycle[i % p];
    let before = before.unwrap_or(cycle[(i + p - 1) % p]);
    let mut next: Vec<usize> = (0..p).filter(|&j| cycle[j] == before).map(|j| cycle[(j + 1) % p]).filter(|&t| t != due).collect();
    next.sort_unstable();
    next.dedup();
    // (`set_varied_orders`, a cycle longer than its tokens) the break keeps the pair with the token
    // after it too, so the token before or after never tells it; where no token does, no break
    if varied_orders() && p > tokens {
        let after = cycle[(i + 1) % p];
        next.retain(|&x| (0..p).any(|j| cycle[j] == x && cycle[(j + 1) % p] == after));
        return if next.is_empty() { due } else { next[r.gen_range(0..next.len())] };
    }
    if next.is_empty() {
        next = cycle.iter().copied().filter(|&t| t != due).collect();
        next.sort_unstable();
        next.dedup();
    }
    if next.is_empty() {
        return (due + r.gen_range(1..tokens.max(2))) % tokens.max(2);
    }
    next[r.gen_range(0..next.len())]
}

/// Sparse one-word codes in every reading lifetime (`set_one_word`; a lesson's `one_word` sets
/// it for one): `World::one_word` with this many lines a code, 0 off (the default).
static ONE_WORD: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn set_one_word(lines: usize) {
    ONE_WORD.store(lines, std::sync::atomic::Ordering::Relaxed);
}

pub fn one_word() -> usize {
    ONE_WORD.load(std::sync::atomic::Ordering::Relaxed)
}

/// Sentences for every lifetime (`set_sentences`; a lesson's `sentence` sets it for one): see
/// `World::sentence`. Off by default.
static SENTENCES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_sentences(on: bool) {
    SENTENCES.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn sentences() -> bool {
    SENTENCES.load(std::sync::atomic::Ordering::Relaxed)
}

/// The muscle's response (`set_muscle`): after each step of the world, the act vector shows what
/// the body did: the clean code of the move made and the token said (the body's own commands for
/// them), or nothing where no move or token came of what was written. So every population that
/// writes the act sees, on its own lines, whether they drove the body (and, with copy learning,
/// learns toward what was done, not toward what it wrote), and what reads the act port reads what
/// was done. Off by default (the act keeps what the populations wrote).
static MUSCLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_muscle(on: bool) {
    MUSCLE.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn muscle() -> bool {
    MUSCLE.load(std::sync::atomic::Ordering::Relaxed)
}

/// The wiring cost (`set_wire_cost`): work per step charged for each unit of the genome's wiring
/// length (`Evolvable::wire_length`: for the tag grammar, how far apart in tag space the two ends
/// of each population are, ports included), charged as the carrying cost is (added to the work, or paid from
/// the energy store with `set_carry_energy`). A link between channels with near tags is cheap and
/// one between distant tags dear, so circuits that work together are pulled close in tag space
/// and joined to the rest by few long links: parts. 0: none.
static WIRE_COST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn set_wire_cost(per_bit: u64) {
    WIRE_COST.store(per_bit, std::sync::atomic::Ordering::Relaxed);
}

pub fn wire_cost() -> u64 {
    WIRE_COST.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn carry_energy() -> bool {
    CARRY_ENERGY.load(std::sync::atomic::Ordering::Relaxed)
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
            let cost = if carry_energy() { 0 } else { carry * g.size() as u64 + wire_cost() * g.wire_length() as u64 };
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
        let (ranked, next) = lexicase_breed(&scored, wanted, thrift, 0, pop, elite, &mut rng);
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
/// (a quarter recombined) children. `stuck`: the generations on the current rung, which relax the
/// thrift cases with `set_thrift_relax`.
fn lexicase_breed<G: Evolvable>(scored: &[Probed<G>], wanted: &[usize], thrift: bool, stuck: usize, pop: usize, elite: usize, rng: &mut StdRng) -> (Vec<usize>, Vec<G>) {
    let n = scored.len();
    let nseeds = scored.first().map_or(0, |s| s.probes.len());
    // the cases: one per (seed, wanted probe), then one thrift case per seed (less work is better)
    let np = wanted.len();
    let ncases = nseeds * np + if thrift { nseeds } else { 0 };
    let value = |g: usize, c: usize| -> i64 {
        if c < nseeds * np { scored[g].probes[c / np][graded(wanted[c % np])] as i64 } else { -(scored[g].works[c - nseeds * np].min(i64::MAX as u64 / 2) as i64) }
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
    // (`set_size_classes`) the genomes of each occupied size class
    let classes: Vec<Vec<usize>> = if size_classes() {
        let mut by = vec![Vec::new(); 5];
        for g in 0..n {
            by[size_class(scored[g].genome.size())].push(g);
        }
        by.into_iter().filter(|c| !c.is_empty()).collect()
    } else {
        Vec::new()
    };
    let select_in = |rng: &mut StdRng, mut cand: Vec<usize>| -> usize {
        let mut order: Vec<usize> = (0..ncases).collect();
        // (`set_thrift_relax`) the thrift cases taken with chance scale / (scale + stuck)
        let relax = thrift_relax();
        if thrift && relax > 0 && rng.gen_range(0..relax + stuck) >= relax {
            order.truncate(nseeds * np);
        }
        order.shuffle(rng);
        for c in order {
            if cand.len() == 1 {
                break;
            }
            let best = cand.iter().map(|&g| value(g, c)).max().unwrap();
            cand.retain(|&g| value(g, c) >= best - eps[c]);
        }
        cand[rng.gen_range(0..cand.len())]
    };
    let select = |rng: &mut StdRng| -> usize { select_in(rng, (0..n).collect()) };
    // a parent: half the time from a random occupied size class, with size classes on
    let parent = |rng: &mut StdRng| -> usize {
        if classes.len() > 1 && rng.gen_bool(0.5) {
            let c = rng.gen_range(0..classes.len());
            select_in(rng, classes[c].clone())
        } else {
            select(rng)
        }
    };
    let means: Vec<observe::ProbeRow> = scored.iter().map(|s| probe_means(&s.probes)).collect();
    let total = |m: &observe::ProbeRow| wanted.iter().map(|&p| m[p] as i64).sum::<i64>();
    let mut ranked: Vec<usize> = (0..n).collect();
    ranked.sort_by(|&a, &b| total(&means[b]).cmp(&total(&means[a])).then(scored[a].work.cmp(&scored[b].work)).then(b.cmp(&a)));
    let mut kept: Vec<usize> = Vec::new();
    // each occupied size class's own choice first, as far as the elite goes
    if classes.len() > 1 {
        for c in &classes {
            if kept.len() >= elite.min(n) {
                break;
            }
            kept.push(select_in(rng, c.clone()));
        }
    }
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
        let mut child = scored[parent(rng)].genome.clone();
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
    /// The novelty island (`set_novelty`): selected on how new its behaviour is, not on a goal.
    pub fn is_novelty(&self) -> bool {
        self.name == "novelty"
    }

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
    /// (populations competing, `program::neurocomp::set_write_compete`) each island's contest this
    /// generation, over all its probing: steps run, kernels held down by another population's
    /// claim, populations that let go of their destination
    pub contest: Vec<(u64, u64, u64)>,
    /// (`set_frontier_ladders`) each island's own rung on each island's ladder this generation
    /// (empty but at the frontier), the elite's training mean on it, how many generations in a
    /// row its goal has held, and the held-out checks made: (island, ladder, elite's held-out
    /// means, passed)
    pub ladders: Vec<Vec<usize>>,
    pub ladder_means: Vec<Vec<observe::ProbeRow>>,
    pub ladder_met: Vec<Vec<usize>>,
    pub ladder_checks: Vec<(usize, usize, observe::ProbeRow, bool)>,
    /// (`set_migrate`, `set_migrate_up`) genomes copied into another island's next population:
    /// (from island, to island, the rung passed on the ladder it concerns, or `usize::MAX` for the
    /// periodic move into the generalist, how many)
    pub migrated: Vec<(usize, usize, usize, usize)>,
    /// (`set_novelty`) species founded this generation (island, probe, score) and how many there are
    pub new_species: Vec<(usize, usize, i32)>,
    pub species: usize,
    /// (`set_novel_migrate`) species founders sent this generation: (novelty island, to island, the
    /// probe it was useful on)
    pub novel_sent: Vec<(usize, usize, usize)>,
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
    // (`set_frontier_ladders`) the generalist's own rung on each ladder, and its goal's run
    let mut lad: Vec<Vec<usize>> = (0..k).map(|i| match frontier_ladders() {
        Some(s) if islands[i].frontier => (0..k).map(|j| s.get(j).copied().unwrap_or(0).min(islands[j].stages.len().saturating_sub(1))).collect(),
        _ => Vec::new(),
    }).collect();
    let mut lmet: Vec<Vec<usize>> = lad.iter().map(|l| vec![0; l.len()]).collect();
    // (`set_novelty`) the novelty island's archive of behaviours and its species (a founder genome,
    // its behaviour, the probe it was useful on)
    let mut archive: Vec<[i64; 8]> = Vec::new();
    let mut species: Vec<(G, [i64; 8], usize)> = Vec::new();
    for generation in first..generations {
        let lad_now = lad.clone();
        let mut ladder_means: Vec<Vec<observe::ProbeRow>> = lad.iter().map(|l| vec![[0i32; observe::NPROBES]; l.len()]).collect();
        let mut ladder_checks = Vec::new();
        let mut ladder_met = lmet.clone();
        // (`set_novelty`) species founded this generation: (island, the probe, its score)
        let mut new_species: Vec<(usize, usize, i32)> = Vec::new();
        // (`set_novel_migrate`) their founders: (island, the probe, the genome)
        let mut founders: Vec<(usize, usize, G)> = Vec::new();
        // (`set_migrate`) the generalist's passes on its own ladders this generation
        let mut passes: Vec<(usize, usize, usize)> = Vec::new();
        // (`fresh`: the training worlds are new every generation, the same for every genome in
        // it, so a population cannot fit a fixed set of worlds: their codes and sequences)
        let train: Vec<u64> = if fresh { seeds.iter().map(|s| s.wrapping_add((generation as u64 + 1).wrapping_mul(1_000_003))).collect() } else { seeds.to_vec() };
        let seeds = &train[..];
        // (`set_fresh_held`) this generation's held-out worlds: new ones at every check
        let held_now: Vec<u64> = if fresh_held() { (0..held_out.len() as u64).map(|i| (1u64 << 40) + generation as u64 * 4096 + i).collect() } else { held_out.to_vec() };
        let held_out = &held_now[..];
        let now = stage.clone();
        let mut held = Vec::with_capacity(k);
        let mut met_now = Vec::with_capacity(k);
        let mut rechecked = vec![None; k];
        let mut ranked: Vec<Vec<G>> = Vec::with_capacity(k);
        let mut means = Vec::with_capacity(k);
        let mut work = Vec::with_capacity(k);
        let mut next: Vec<Vec<G>> = Vec::with_capacity(k);
        let mut contest = Vec::with_capacity(k);
        for (i, isl) in islands.iter().enumerate() {
            let _ = crate::program::neurocomp::take_contest();
            let st = &isl.stages[stage[i]];
            // (`set_frontier_ladders`) each ladder's own rung: (ladder, its rows' range)
            let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
            let (scored, wanted) = if isl.frontier && !lad[i].is_empty() {
                // on ladders of its own: each other island's ladder at its own rung, and one rung
                // below it drawn at random, each in its own world, rows of their own
                let genomes: Vec<G> = pops[i].drain(..).collect();
                let mut wanted: Vec<usize> = Vec::new();
                let mut rows = vec![Vec::new(); genomes.len()];
                let mut works = vec![Vec::new(); genomes.len()];
                let mut found: Vec<Vec<Note>> = vec![Vec::new(); genomes.len()];
                for (j, other) in islands.iter().enumerate() {
                    if j == i || other.frontier || other.is_novelty() {
                        continue;
                    }
                    let c = lad[i][j];
                    let mut rungs = vec![c];
                    if c > 0 {
                        rungs.push(rng.gen_range(0..c));
                    }
                    for (n, &r) in rungs.iter().enumerate() {
                        let os = &other.stages[r];
                        for &p in &os.probes {
                            if !wanted.contains(&p) {
                                wanted.push(p);
                            }
                        }
                        let a = rows.first().map_or(0, |x: &Vec<observe::ProbeRow>| x.len());
                        for (g, s) in probe_all_in(genomes.clone(), trials, physics, seeds, gap, &os.probes, os.lesson).iter().enumerate() {
                            for (kk, r) in s.probes.iter().enumerate() {
                                let mut row = [0i32; observe::NPROBES];
                                for &p in &os.probes {
                                    row[p] = r[p];
                                }
                                rows[g].push(row);
                                works[g].push(s.works[kk]);
                            }
                            found[g].extend(&s.notes);
                        }
                        if n == 0 {
                            blocks.push((j, a, rows.first().map_or(0, |x| x.len())));
                        }
                    }
                }
                let scored: Vec<Probed<G>> = genomes.into_iter().enumerate().map(|(g, genome)| {
                    let work = works[g].iter().sum::<u64>() / works[g].len().max(1) as u64;
                    Probed { probes: rows[g].clone(), works: works[g].clone(), work, genome, notes: merge_notes(found[g].iter().copied()) }
                }).collect();
                (scored, wanted)
            } else if isl.frontier {
                // at the frontier: each other island's current stage, in its own world, for the
                // probes no earlier island has given
                let genomes: Vec<G> = pops[i].drain(..).collect();
                let mut wanted: Vec<usize> = Vec::new();
                let mut rows = vec![vec![[0i32; observe::NPROBES]; seeds.len()]; genomes.len()];
                let mut works = vec![vec![0u64; seeds.len()]; genomes.len()];
                let mut found: Vec<Vec<Note>> = vec![Vec::new(); genomes.len()];
                for (j, other) in islands.iter().enumerate() {
                    if j == i || other.frontier || other.is_novelty() {
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
                // (`set_frontier_passed`) and each other island's last passed rung, in its own
                // world: rows of their own, joining the cases, so the generalist is selected on
                // what the specialists already do as well as on what none does yet
                if frontier_passed() {
                    for (j, other) in islands.iter().enumerate() {
                        if j == i || other.frontier || other.is_novelty() || stage[j] == 0 {
                            continue;
                        }
                        let ps = &other.stages[stage[j] - 1];
                        for &p in &ps.probes {
                            if !wanted.contains(&p) {
                                wanted.push(p);
                            }
                        }
                        for (g, s) in probe_all_in(genomes.clone(), trials, physics, seeds, gap, &ps.probes, ps.lesson).iter().enumerate() {
                            for (k, r) in s.probes.iter().enumerate() {
                                let mut row = [0i32; observe::NPROBES];
                                for &p in &ps.probes {
                                    row[p] = r[p];
                                }
                                rows[g].push(row);
                                works[g].push(s.works[k]);
                            }
                        }
                    }
                }
                let scored: Vec<Probed<G>> = genomes.into_iter().enumerate().map(|(g, genome)| {
                    let work = works[g].iter().sum::<u64>() / works[g].len().max(1) as u64;
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
            // (`set_keep_rungs`) one passed rung, drawn at random, scored again in its own world:
            // its rows join the lexicase cases, so a climb keeps what the rungs below it asked;
            // the goal below is still judged on the current rung's rows alone
            let (order, children) = if keep_rungs() && !isl.frontier && stage[i] > 0 {
                let j = rng.gen_range(0..stage[i]);
                let earlier = &isl.stages[j];
                let again = probe_all_in(scored.iter().map(|s| s.genome.clone()).collect(), trials, physics, seeds, gap, &earlier.probes, earlier.lesson);
                let both: Vec<Probed<G>> = scored.iter().zip(again).map(|(s, a)| {
                    let mut probes = s.probes.clone();
                    probes.extend(a.probes);
                    let mut works = s.works.clone();
                    works.extend(a.works);
                    Probed { probes, works, work: s.work, genome: s.genome.clone(), notes: Vec::new() }
                }).collect();
                let mut cases = wanted.clone();
                for &p in &earlier.probes {
                    if !cases.contains(&p) {
                        cases.push(p);
                    }
                }
                lexicase_breed(&both, &cases, thrift, on[i], pop, elite, &mut rng)
            } else if isl.is_novelty() {
                // (`set_novelty`) each genome's behaviour, its mean over its lifetimes; its novelty,
                // the mean distance (L1) to its 8 nearest among the others and the archive, written
                // as probe 29 on its rows, the one case it is selected on (with thrift)
                let behs: Vec<[i64; 8]> = scored.iter().map(|s| {
                    let mut b = [0i64; 8];
                    for r in &s.probes {
                        for d in 0..8 {
                            b[d] += r[21 + d] as i64;
                        }
                    }
                    let n = s.probes.len().max(1) as i64;
                    b.map(|x| x / n)
                }).collect();
                let dist = |a: &[i64; 8], b: &[i64; 8]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<i64>();
                let nov: Vec<i64> = (0..behs.len()).map(|g| {
                    let mut d: Vec<i64> = behs.iter().enumerate().filter(|(h, _)| *h != g).map(|(_, b)| dist(&behs[g], b)).chain(archive.iter().map(|b| dist(&behs[g], b))).collect();
                    d.sort_unstable();
                    let k = d.len().min(8).max(1);
                    d.iter().take(k).sum::<i64>() / k as i64
                }).collect();
                for (s, &v) in scored.iter_mut().zip(&nov) {
                    for r in s.probes.iter_mut() {
                        r[29] = v.min(i32::MAX as i64) as i32;
                    }
                }
                // the archive: the two most novel behaviours each generation
                let mut by: Vec<usize> = (0..nov.len()).collect();
                by.sort_by_key(|&g| std::cmp::Reverse(nov[g]));
                for &g in by.iter().take(2) {
                    if archive.len() < 4096 {
                        archive.push(behs[g]);
                    }
                }
                // species: a novel genome useful on some probe (its mean at least
                // `NOVELTY_USEFUL`, babble aside) and unlike every species so far founds one
                for &g in by.iter().take(4) {
                    let m = probe_means(&scored[g].probes);
                    let useful = ALL_PROBES.iter().copied().filter(|&p| p != 8).max_by_key(|&p| m[p]);
                    if let Some(p) = useful.filter(|&p| m[p] >= NOVELTY_USEFUL) {
                        if species.iter().all(|s| dist(&s.1, &behs[g]) >= NOVELTY_APART) && species.len() < 64 {
                            species.push((scored[g].genome.clone(), behs[g], p));
                            new_species.push((i, p, m[p]));
                            founders.push((i, p, scored[g].genome.clone()));
                        }
                    }
                }
                let size = (pop + 4 * species.len()).min(novelty_max().max(pop));
                let (order, mut children) = lexicase_breed(&scored, &[29], thrift, on[i], size, elite, &mut rng);
                // each species' founder lives on beside the children
                for s in &species {
                    children.push(s.0.clone());
                }
                (order, children)
            } else {
                lexicase_breed(&scored, &wanted, thrift, on[i], pop, elite, &mut rng)
            };
            let m: Vec<observe::ProbeRow> = scored.iter().map(|s| probe_means(&s.probes)).collect();
            let mean = probe_means(&m);
            // (`set_elite_pass`) the goal on the elite's mean, else the whole population's
            let judged = if elite_pass() { probe_means(&order.iter().take(elite.max(1)).map(|&g| m[g]).collect::<Vec<_>>()) } else { mean };
            on[i] += 1;
            held.push(on[i]);
            // (`set_frontier_ladders`) each ladder climbed as its island climbs it: the goal on the
            // elite's mean over that rung's rows, held, then checked on the held-out worlds
            for &(j, a, b) in &blocks {
                let ost = &islands[j].stages[lad[i][j]];
                let top: Vec<observe::ProbeRow> = order.iter().take(elite.max(1)).map(|&g| probe_means(&scored[g].probes[a..b])).collect();
                let jm = probe_means(&top);
                ladder_means[i][j] = jm;
                if ost.goal.is_empty() || lad[i][j] + 1 >= islands[j].stages.len() {
                    continue;
                }
                lmet[i][j] = if ost.goal.iter().all(|&(p, x)| jm[p] >= x) { lmet[i][j] + 1 } else { 0 };
                ladder_met[i][j] = lmet[i][j];
                if lmet[i][j] >= ost.hold.max(1) {
                    let pass = held_out.is_empty() || {
                        let tg: Vec<G> = order.iter().take(elite.max(1)).map(|&g| scored[g].genome.clone()).collect();
                        let h = probe_all_in(tg, trials, physics, held_out, gap, &ost.probes, ost.lesson);
                        let m = probe_means(&h.iter().map(|s| probe_means(&s.probes)).collect::<Vec<_>>());
                        let pass = ost.goal.iter().all(|&(p, x)| m[p] >= x);
                        ladder_checks.push((i, j, m, pass));
                        pass
                    };
                    if pass {
                        passes.push((i, j, lad[i][j]));
                        lad[i][j] += 1;
                        lmet[i][j] = 0;
                    }
                }
            }
            // the curriculum: the goal met for `hold` generations in a row moves the island on
            if !st.goal.is_empty() && stage[i] + 1 < isl.stages.len() {
                met[i] = if st.goal.iter().all(|&(p, x)| judged[p] >= x) { met[i] + 1 } else { 0 };
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
            contest.push(crate::program::neurocomp::take_contest());
        }
        // (`set_migrate`) a generalist that passed a rung on an island's ladder the island has not
        // passed: its best (up to two of its elite) join that island's next population, in place
        // of its last children, and are judged there like any other genome
        let mut migrated = Vec::new();
        if migrate() {
            for &(i, j, r) in &passes {
                if stage[j] > r || next[j].is_empty() {
                    continue;
                }
                let m = elite.clamp(1, 2).min(ranked[i].len()).min(next[j].len());
                let n = next[j].len();
                for k in 0..m {
                    next[j][n - 1 - k] = ranked[i][k].clone();
                }
                migrated.push((i, j, r, m));
            }
        }
        // (`set_novel_migrate`) each new species' founder into every island whose current rung is
        // scored on the probe it was useful on, in place of one of its last children (behind any
        // genomes migrated there already)
        let mut novel_sent = Vec::new();
        if novel_migrate() {
            let mut placed = vec![0usize; k];
            for (i, p, g) in &founders {
                for j in 0..k {
                    let isl = &islands[j];
                    if j == *i || isl.frontier || isl.is_novelty() || next[j].is_empty() || !isl.stages[stage[j]].probes.contains(p) {
                        continue;
                    }
                    let taken = migrated.iter().filter(|m: &&(usize, usize, usize, usize)| m.1 == j).map(|m| m.3).sum::<usize>() + placed[j];
                    let n = next[j].len();
                    if taken < n {
                        next[j][n - 1 - taken] = g.clone();
                        placed[j] += 1;
                        novel_sent.push((*i, j, *p));
                    }
                }
            }
        }
        // (`set_migrate_up`) the islands' best into the generalist, beside its own children (not in
        // their place): every `migrate_up()` generations each island's best `migrate_up_n()`, and
        // an island's best two at least whenever it passed a rung the generalist has not passed on
        // its ladder, up to `generalist_room()` genomes in all; they are scored with the rest, and
        // breeding brings the generalist back to its own size, so a migrant that wins a place among
        // its parents or its elite carries its circuits on
        let up = migrate_up();
        if up > 0 {
            for gi in (0..k).filter(|&i| !lad[i].is_empty()) {
                let room = generalist_room().max(next[gi].len());
                for j in 0..k {
                    if j == gi || islands[j].frontier || ranked[j].is_empty() {
                        continue;
                    }
                    let passed = stage[j] != now[j] && lad[gi][j] <= now[j];
                    let m = if generation % up == 0 { migrate_up_n().max(1) } else { 0 }.max(if passed { 2 } else { 0 });
                    let mut sent = 0;
                    for q in 0..m.min(ranked[j].len()) {
                        if next[gi].len() >= room {
                            break;
                        }
                        next[gi].push(ranked[j][q].clone());
                        sent += 1;
                    }
                    if sent > 0 {
                        migrated.push((j, gi, if passed { now[j] } else { usize::MAX }, sent));
                    }
                }
            }
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
        report(generation, &IslandGeneration { stage: now, held, met: met_now, rechecked, ranked: ranked.clone(), means, work, moved, contest, ladders: lad_now, ladder_means, ladder_met, ladder_checks, migrated, new_species, species: species.len(), novel_sent });
        last = ranked;
        pops = next;
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn held_frequent_breaks_do_not_help_a_partly_right_detector() {
        // surprise as the observer scores it: clicks in the 3 steps after a break, less after
        // steps with no break in the 3 before; strategies on the token stream itself
        set_fair_sequences(true);
        let score = |noise: u32, hold: usize, policy: u8| -> i64 {
            let mut total = 0;
            for seed in 0..24u64 {
                let mut r = StdRng::seed_from_u64(seed);
                let cycle = fair_cycle(3, 8, &mut r);
                let (tok, odd) = prediction_stream(&cycle, 400, noise, hold, 8, &mut r);
                let mut click = vec![false; tok.len() + 1];
                for t in 3..tok.len() {
                    click[t + 1] = match policy {
                        0 => tok[t] != tok[t - 3],                                // lag-3 comparator
                        1 => tok[t] != tok[t - 3] && r.gen_ratio(1, 3),           // right a third of the time
                        2 => tok[t] == tok[t - 1],                                // the cycle1 repeat detector
                        _ => r.gen_ratio(1, 4),                                   // clicking at random
                    };
                }
                let after = |t: usize| (t + 1..=(t + 3).min(tok.len())).any(|u| click[u]);
                let (mut o, mut oc, mut k, mut kc) = (0i64, 0i64, 0i64, 0i64);
                for t in 0..tok.len() - 3 {
                    if odd[t] {
                        o += 1;
                        oc += after(t) as i64;
                    } else if !(t.saturating_sub(3)..t).any(|u| odd[u]) {
                        k += 1;
                        kc += after(t) as i64;
                    }
                }
                total += oc * 1024 / o.max(1) - kc * 1024 / k.max(1);
            }
            total / 24
        };
        let (full3, part3, rep3, rand3) = (score(1, 0, 0), score(1, 0, 1), score(1, 0, 2), score(1, 0, 3));
        let (full_e, part_e, rep_e, rand_e) = (score(2, 2, 0), score(2, 2, 1), score(2, 2, 2), score(2, 2, 3));
        eprintln!("cycle3: lag-3 {full3}, a third right {part3}, repeats {rep3}, random {rand3}; cycle3e: lag-3 {full_e}, a third right {part_e}, repeats {rep_e}, random {rand_e}");
        set_fair_sequences(false);
        // cycle3 already pays a partly right detector in proportion (a third right passes its goal
        // of 200), holding breaks adds nothing to it and costs the full detector its false alarms
        // after each break; and a break may repeat the token before, which a cycle of distinct
        // tokens never does, so sameness tells it (a leak, unused by run 33's population)
        assert!(full3 > 600 && part3 > 200 && (part_e - part3).abs() < 60 && full_e < full3);
        assert!(rep3 > 600 && rep_e > 300);
        assert!(rand3.abs() < 60 && rand_e.abs() < 60);
    }

    #[test]
    fn fair_cycles_use_each_token_equally_and_breaks_keep_to_its_tokens() {
        let mut r = StdRng::seed_from_u64(5);
        for (period, tokens) in [(1usize, 8usize), (2, 8), (4, 8), (8, 8), (6, 3)] {
            let c = fair_cycle(period, tokens, &mut r);
            assert_eq!(c.len(), period);
            let mut count = vec![0; tokens];
            for &t in &c {
                count[t] += 1;
            }
            let used: Vec<usize> = count.iter().copied().filter(|&x| x > 0).collect();
            assert!(used.iter().all(|&x| x == used[0]), "{c:?}");
            for i in 0..50 {
                let o = fair_odd(&c, i, None, tokens, &mut r);
                assert_ne!(o, c[i % period]);
                if period > 1 {
                    assert!(c.contains(&o), "{c:?} {o}");
                }
            }
        }
        // period 6 over 3: A A B B C C; a break keeps the pair legal (the token before can be
        // followed by it somewhere in the cycle), so only the two before tell it
        let c = fair_cycle(6, 3, &mut r);
        assert!(c[0] == c[1] && c[2] == c[3] && c[4] == c[5] && c[0] != c[2] && c[2] != c[4] && c[0] != c[4], "{c:?}");
        for i in 0..60 {
            let o = fair_odd(&c, i, None, 3, &mut r);
            let before = c[(i + 5) % 6];
            assert!((0..6).any(|j| c[j] == before && c[(j + 1) % 6] == o), "{c:?} at {i}: {o}");
        }
    }

    #[test]
    fn size_classes_are_a_factor_of_four_apart() {
        let c: Vec<usize> = [0, 1, 3, 4, 15, 16, 63, 64, 255, 256, 5000].iter().map(|&s| size_class(s)).collect();
        assert_eq!(c, vec![0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4]);
    }

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
