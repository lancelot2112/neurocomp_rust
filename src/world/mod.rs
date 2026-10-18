//! The world a bit-vector program lives in, kept small for evolving programs.
//!
//! - **The world and the program meet only through bit vectors.** Each step the program
//!   writes one act vector and the world answers with one senses vector. Both are split
//!   into fixed, word-aligned channels, so a kernel can read or write one channel by its
//!   offset.
//! - **Acts go through a body.** Each effector (voice, hand) has a set of motor primitives,
//!   random sparse codes given at birth. An act channel drives the primitive it holds at
//!   least half of (best first); a command that matches none does nothing.
//! - **The world gives no rewards.** Nothing in the senses says good or bad. Affective touch
//!   (a slow stroke on the skin, as a run of neighbouring receptors) is a sense like the
//!   others; whether it is valued is up to the program (and to evolution).
//! - **Scoring is not the world's job.** Each step leaves an `Event` for an observer outside
//!   both (`observe`), which is what evolution's fitness reads.
//! - **Deterministic by seed.** Every code and every scene draw comes from the world's seed.
//!
//! Scenes:
//! - `Conditioning`: trials of 10 steps with a caregiver there; Pavlovian, instrumental,
//!   delayed and serial reversal (the paying light swaps every `reversal_period` trials).
//! - `Sequence`: a stream of tokens shown one per step; saying the next token before it is
//!   shown is what the observer counts.
//! - `Echo`: a caregiver comes and goes and mostly answers a sound made aloud with the same
//!   sound.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::bitvec::BitVector;

pub mod observe;

/// Bits per channel, and active bits per code.
pub const CH_BITS: usize = 512;
pub const ACTIVE: usize = 16;

/// Sense channels, in order in the senses vector.
pub const EYE: usize = 0;
pub const EAR: usize = 1;
/// what the hand feels (its own move)
pub const TOUCH: usize = 2;
/// affective touch
pub const AFFECT: usize = 3;
pub const SENSE_CHANNELS: usize = 4;

/// Act channels, in order in the act vector.
pub const VOICE: usize = 0;
pub const HAND: usize = 1;
pub const ACT_CHANNELS: usize = 2;

/// The hand's moves.
pub const LEFT: usize = 0;
pub const RIGHT: usize = 1;
pub const CLICK: usize = 2;
/// up and down: for a screen (two dimensions). No scene uses them yet, so they do nothing but be
/// felt, like any move that does nothing.
pub const UP: usize = 3;
pub const DOWN: usize = 4;
const MOVES: usize = 5;

/// Width of the senses vector.
pub const SENSE_BITS: usize = SENSE_CHANNELS * CH_BITS;
/// Width of the act vector.
pub const ACT_BITS: usize = ACT_CHANNELS * CH_BITS;

/// Steps in a conditioning trial.
pub const TRIAL: usize = 10;

/// A conditioning scene's kind.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Conditioning {
    /// a tone (CS+, half the trials) is followed two steps later by a stroke; another tone
    /// (CS−) by nothing; nothing to do
    Pavlov,
    /// a light is on for three steps (half the trials); a click while it is on, or the step
    /// after, gets a stroke the next step
    Instrumental,
    /// a light is on for one step; a click three to five steps later gets a stroke
    Delayed,
    /// one of two lights (A on cue trials, B on the others) is on for three steps every trial;
    /// a click while it is on, or the step after, gets a stroke if that light is the one paying
    /// now: A for the first `reversal_period` trials, then B, then A again...
    Reversal,
}

/// Which scene a step belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SceneKind {
    Idle,
    Conditioning(Conditioning),
    Sequence,
    Echo,
    Maze,
    Screen,
    Feel,
}

/// A record of one step, for the observer (never sent to the program).
#[derive(Clone, Debug)]
pub struct Event {
    pub scene: SceneKind,
    /// conditioning: the trial's index, the step in the trial, a cue trial (CS+ or light A), and
    /// whether a click in the window pays on this trial
    pub trial: Option<(usize, usize, bool, bool)>,
    /// the hand's move this step, if any
    pub moved: Option<usize>,
    /// the voice's token this step, if any (with sentences: the first)
    pub said: Option<usize>,
    /// (`World::sentence`) the tokens said this step, in the order of the voice's words; empty
    /// without sentences
    pub sentence: Vec<usize>,
    /// sequence: the token shown this step
    pub shown: Option<usize>,
    /// echo: the caregiver answered with the program's last sound
    pub echoed: bool,
    /// a stroke was felt this step
    pub stroked: bool,
    /// someone is there
    pub present: bool,
    /// reading: a teacher said the shown token aloud this step
    pub taught: bool,
    /// sequence: the token shown this step broke the stream's pattern (the observer's mark; the
    /// program only sees the token)
    pub odd: bool,
    /// maze: the trial's index, the place on the stem (the stem's length: at the junction), and
    /// the side the food is on (true: right)
    pub maze: Option<(usize, usize, bool)>,
    /// maze: a choice at the junction this step, and whether it was the food's side; screen: a
    /// click this step, and whether it was on the target
    pub chose: Option<bool>,
    /// screen: the trial's index and how many cells the screen has
    pub screen: Option<(usize, usize)>,
    /// conditioning: this step is in the trial's response window (the observer's mark)
    pub window: bool,
}

/// An effector's motor primitives: random sparse codes, one per move or token.
pub struct Effector {
    codes: Vec<BitVector>,
    /// a place code (`fix_place`): primitive i owns word i of the channel, and a command is the
    /// primitive whose word has the most lines on
    place: bool,
}

/// The lines of a place code inside its word (16, one in four).
pub const PLACE_LINES: u64 = 0x1111_1111_1111_1111u64;

impl Effector {
    fn new(n: usize, rng: &mut StdRng) -> Self {
        Self { codes: codes(n, rng), place: false }
    }

    /// Redraw its codes in the first word of the channel, `k` lines each (`World::one_word`).
    fn one_word(&mut self, k: usize, rng: &mut StdRng) {
        let n = self.codes.len();
        self.codes = one_word_codes(n, k, rng);
    }

    /// Make it a place code, the same in every lifetime, as a muscle is a labelled line:
    /// primitive i is `PLACE_LINES` in word i of the channel, and any command is read as the
    /// primitive whose word has the most lines on (at least half a code's), so whatever lands in
    /// a word drives that primitive.
    pub fn fix_place(&mut self) {
        let words = CH_BITS / 64;
        for (i, c) in self.codes.iter_mut().enumerate() {
            let mut v = vec![0u64; words];
            v[i % words] = PLACE_LINES;
            *c = BitVector::from_words(v);
        }
        self.place = true;
    }

    /// The command for primitive `i` (what a body's pattern generators can produce).
    pub fn command(&self, i: usize) -> &BitVector {
        &self.codes[i]
    }

    pub fn len(&self) -> usize {
        self.codes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// The primitive the command holds at least half of, best first; None if none.
    pub fn articulate(&self, m: &BitVector) -> Option<usize> {
        if self.place {
            let w = m.as_words();
            let half = PLACE_LINES.count_ones() / 2;
            return (0..self.codes.len())
                .map(|i| (w.get(i).map_or(0, |x| x.count_ones()), i))
                .filter(|&(c, _)| c >= half)
                .max_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)))
                .map(|x| x.1);
        }
        (0..self.codes.len())
            .map(|i| (overlap(&self.codes[i], m), i))
            .filter(|&(o, i)| 2 * o >= self.codes[i].count_ones())
            .max_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)))
            .map(|x| x.1)
    }
}

/// What a program is born with: its effectors' primitives.
pub struct Body<'a> {
    pub voice: &'a Effector,
    pub hand: &'a Effector,
}

/// A screen trial's layout: the cursor's cell, the target's cell, the glyph in each cell, and the
/// walls (one bit a cell).
#[derive(Clone)]
struct ScreenTrial {
    cursor: usize,
    target: usize,
    glyph: [Option<usize>; 32],
    walls: u32,
}

enum Scene {
    Idle,
    /// (kind, trials left, step in the trial, a cue trial, stroked this trial)
    Conditioning(Conditioning, usize, usize, bool, bool),
    /// the tokens still to be shown
    Sequence(std::collections::VecDeque<(usize, bool)>),
    /// steps left, a caregiver there
    Echo(usize, bool),
    /// a gap between conditioning trials: (kind, trials left after it, steps left in the gap)
    Gap(Conditioning, usize, usize),
    /// a T-maze trial: (trials left, place on the stem, steps in the trial, food on the right)
    Maze(usize, usize, usize, bool),
    /// between maze trials: (trials left after it, steps left)
    MazeGap(usize, usize),
    /// a screen trial: (trials left, steps in the trial, its layout)
    Screen(usize, usize, Box<ScreenTrial>),
    /// between screen trials: (trials left after it, steps left)
    ScreenGap(usize, usize),
    /// feeling: steps left; a stroke comes now and then, whatever the program does
    Feel(usize),
}

/// The world.
pub struct World {
    rng: StdRng,
    /// tokens: how each is seen and heard
    glyphs: Vec<BitVector>,
    sounds: Vec<BitVector>,
    /// tones (heard) and two lights (seen) for conditioning
    tones: Vec<BitVector>,
    lights: Vec<BitVector>,
    /// how each hand move feels
    felt: Vec<BitVector>,
    /// task cues: a code per task, felt on the touch channel while `task_cue` names it
    cues: Vec<BitVector>,
    /// the task on now, felt as its cue every step (none: no cue)
    pub task_cue: Option<usize>,
    voice: Effector,
    hand: Effector,
    cond_total: usize,
    pub reversal_period: usize,
    /// conditioning: after each trial, a gap of 0 to `gap_max` steps (drawn each time) with
    /// nothing shown, so when the next light comes cannot be counted from the last; 0 (the
    /// default): trials follow each other directly
    pub gap_max: usize,
    /// reading: with a teacher, about half the tokens shown are also said aloud (heard as the
    /// network hears its own voice saying them), so a glyph's sound can be learned
    pub teacher: bool,
    /// how often the teacher says a shown token, in eighths (4: half of them)
    pub teacher_rate: u32,
    /// (a reading lesson's `babble`) the teacher is silent for the first this many tokens shown,
    /// so one hears only one's own voice: babbling before imitation; and the tokens shown so far
    pub teacher_after: usize,
    shown_so_far: usize,
    /// delayed conditioning: how many steps before the response window (steps 5 to 7) the light
    /// flashes, 1 to 5 (3: at step 2, three to five steps before a click pays)
    pub delay: usize,
    /// delayed conditioning: each trial's delay drawn from 1 to `vary_delay` (0: always `delay`);
    /// with `distract`, on half the trials the other light flashes once during the delay (it does
    /// not pay, and on a trial with no cue it is all that shows); with `two_lights`, a trial with no
    /// cue shows the other light where a cue trial shows the first, and a distractor is the light
    /// that trial did not show first (so remembering the latest light is wrong on every trial
    /// with one). A delay over 5 starts the window at the delay's step and lengthens the trial to
    /// fit (up to 5: the window at 5-7, as always). With `go_tone`, a tone sounds as the window
    /// opens on every trial: a delay drawn each trial cannot be timed from the light, so the
    /// network holds which light came until the tone asks for it; a click before the window spends
    /// the trial (it cannot pay, and its window is not marked), so clicking at fixed times after
    /// the light is right only on the trials whose delay it matches (one delay in about three).
    pub vary_delay: usize,
    pub distract: bool,
    pub two_lights: bool,
    pub go_tone: bool,
    /// (the maze, a spatial cue) the cue's half of the eye is the side that pays, or with
    /// `maze_mirror` the other side, drawn per lifetime (`evolve::set_maze_mirror`): which half
    /// means which turn is the world's, learned in the lifetime from the strokes, not wired
    pub maze_mirror: bool,
    trial_delay: usize,
    distract_at: Option<usize>,
    scene: Scene,
    stroke_next: bool,
    /// serial reversal: shapes, not fixed lights. Each trial shows a fresh exemplar of its light:
    /// each line moved, with chance `blur` in 8, to another place in its word, and the whole
    /// pattern rotated by 0 to `shift` lines across the eye (0 and 0, the default: the light as is)
    pub blur: u32,
    pub shift: usize,
    exemplar: Option<BitVector>,
    /// maze: steps on the stem before the junction, the cue shown only at the start of the stem
    /// (not all along it), and the cue a light's code (which side it means must be learned) rather
    /// than a mark in the eye's left or right half (a side to turn to)
    pub maze_len: usize,
    pub cue_gone: bool,
    pub symbol: bool,
    /// the maze's food stays on one side for the whole lifetime (drawn by its seed), and with
    /// `maze_nocue` no cue shows where: which turn pays is learned from the stroke alone
    pub maze_fixed: bool,
    pub maze_nocue: bool,
    /// one's own voice not heard (a reading lesson's `quiet`): the ear has only the teacher
    pub quiet: bool,
    /// sentences (a lesson's `sentence`): each word of the voice says the token whose pattern (its
    /// code's lines inside a word, wherever the code was drawn) it holds at least half of, so a
    /// token written in any word is said, and the voice says its words in order, a sentence a
    /// step; one's own words are heard at the same places in the ear, and the teacher speaks at
    /// the first
    pub sentence: bool,
    /// (`fix_body`) the body's own seed: its voice and hand codes, the sounds its voice makes and
    /// the feel of its moves, the same in every lifetime of a run; None, drawn with the world
    pub body: Option<u64>,
    /// (`place_senses`) the stroke always on the first word of the skin's map
    pub stroke_place: bool,
    maze_first: Option<bool>,
    junction: BitVector,
    maze_total: usize,
    /// screen: a grid of `screen_w` x `screen_h` cells of `screen_cell` bits each (64, 32 or 16:
    /// at most 8, 16 or 32 cells on the eye), with `screen_lit` glyphs on it (the target, the
    /// first glyph, and different distractors) and `screen_walls` walls (a 2-D maze: a move into
    /// a wall does nothing; every free cell is reachable); with `screen_torch`, a glyph shows only
    /// while the cursor is on its cell (where the cursor is matters, not how it relates to the
    /// glyph: clicking at a glyph is enough to score)
    pub screen_w: usize,
    pub screen_h: usize,
    pub screen_lit: usize,
    pub screen_cell: usize,
    pub screen_walls: usize,
    pub screen_torch: bool,
    /// the cursor's mark is not shown (with the torch, the eye is then empty unless the cursor is
    /// on the target: clicking at whatever is seen is a respond, not telling glyph-and-cursor from
    /// cursor alone)
    pub screen_dark: bool,
    /// programs built for this world keep each declared channel in one frame, held in place
    /// from step to step (only clearing writes take lines away, and a write is seen by the edges
    /// after it in the same step); default: the process's (`neurocomp::set_in_place`)
    pub in_place: bool,
    /// programs built for this world count the words their kernel populations read as work;
    /// default: the process's (`neurocomp::set_count_reads`)
    pub count_reads: bool,
    /// programs built for this world explore: a stuck population grows a trial connection;
    /// default: the process's (`neurocomp::set_explore`)
    pub explore: bool,
    /// (with `explore`) explorers are shadow predictors; default: the process's
    /// (`neurocomp::set_shadow`)
    pub shadow: bool,
    cursor_mark: u64,
    wall_mark: u64,
    screen_total: usize,
    present: bool,
    pub events: Vec<Event>,
}

fn overlap(a: &BitVector, b: &BitVector) -> usize {
    a.as_words().iter().zip(b.as_words()).map(|(x, y)| (x & y).count_ones() as usize).sum()
}

fn codes(n: usize, rng: &mut StdRng) -> Vec<BitVector> {
    (0..n).map(|_| clustered_code(CH_BITS, ACTIVE, rng)).collect()
}

/// `n` codes of `k` random lines, all in the first word of a channel.
fn one_word_codes(n: usize, k: usize, rng: &mut StdRng) -> Vec<BitVector> {
    (0..n).map(|_| clustered_code(64, k, rng)).map(|c| { let mut v = vec![0u64; CH_BITS / 64]; v[0] = c.as_words()[0]; BitVector::from_words(v) }).collect()
}

/// A topographic code: `k` random lines inside one random 64-bit word of a `bits`-wide vector.
/// A kernel reading that word sees the code at density k/64 (a quarter, for 16), so one-word
/// kernels can tell codes apart; two random codes in the same word share about k²/64 lines.
pub fn clustered_code<R: Rng + ?Sized>(bits: usize, k: usize, rng: &mut R) -> BitVector {
    let w = rng.gen_range(0..bits / 64);
    let lines: Vec<usize> = (0..64).collect();
    let picked: Vec<usize> = lines.choose_multiple(rng, k.min(64)).map(|b| w * 64 + b).collect();
    BitVector::from_bits(&picked, bits)
}

/// Channel `ch` of a vector (word-aligned).
pub fn channel(v: &BitVector, ch: usize) -> BitVector {
    let w = CH_BITS / 64;
    BitVector::from_words(v.as_words()[ch * w..(ch + 1) * w].to_vec())
}

fn put(v: &mut BitVector, ch: usize, x: &BitVector) {
    let w = CH_BITS / 64;
    for (d, s) in v.as_words_mut()[ch * w..(ch + 1) * w].iter_mut().zip(x.as_words()) {
        *d |= s;
    }
}

impl World {
    /// An exemplar of a code: each line moved, with chance `blur` in 8, to a line off in its word,
    /// then the whole code rotated by 0 to `shift` lines (draws from the world's own stream).
    fn distort(&mut self, code: &BitVector) -> BitVector {
        let mut w: Vec<u64> = code.as_words().to_vec();
        if self.blur > 0 {
            for word in w.iter_mut() {
                let mut x = *word;
                while x != 0 {
                    let b = x.trailing_zeros();
                    x &= x - 1;
                    if self.rng.gen_ratio(self.blur.min(8), 8) && word.count_ones() < 64 {
                        // to a random line of the word that is off
                        let free = !*word;
                        let k = self.rng.gen_range(0..free.count_ones());
                        let mut f = free;
                        for _ in 0..k {
                            f &= f - 1;
                        }
                        *word &= !(1u64 << b);
                        *word |= 1u64 << f.trailing_zeros();
                    }
                }
            }
        }
        let bits = w.len() * 64;
        let r = if self.shift > 0 { self.rng.gen_range(0..=self.shift) % bits } else { 0 };
        if r == 0 {
            return BitVector::from_words(w);
        }
        let mut out = vec![0u64; w.len()];
        for i in 0..bits {
            if w[i / 64] >> (i % 64) & 1 == 1 {
                let j = (i + r) % bits;
                out[j / 64] |= 1 << (j % 64);
            }
        }
        BitVector::from_words(out)
    }

    /// A world with `tokens` distinct tokens (seen, heard and sayable); its physics (every code)
    /// and its draws both come from `seed`.
    pub fn new(seed: u64, tokens: usize) -> Self {
        Self::with_physics(seed, seed, tokens)
    }

    /// A world whose physics (how each token, tone, light and move looks, and the body's motor
    /// primitives) comes from `physics` and whose draws (cues, trial order, the caregiver) come
    /// from `seed`. Fixed physics across lifetimes lets evolution wire in what the codes are;
    /// varied physics leaves only what a program learns within a lifetime.
    /// Each token's sound made like the voice's code for the same token (a reading lesson's
    /// `mirror`): it keeps `eighths` of the voice code's lines and draws the rest afresh in the
    /// same word (from the physics, so held-out worlds hear the same). 8: a token is heard
    /// exactly as it is said, so a wire from the ear to the voice echoes it; 4: half the lines,
    /// the least a said token needs (`Effector::articulate`); fewer, a copy says nothing.
    /// A code's pattern: its lines inside its word, wherever that word is.
    fn pattern(code: &BitVector) -> u64 {
        code.as_words().iter().copied().find(|&w| w != 0).unwrap_or(0)
    }

    /// (sentences) The tokens a voice vector says, word by word: (word, token) for each word that
    /// holds at least half of some token's pattern (the most-held; ties, the lower token).
    pub fn sentence_of(&self, voice: &BitVector) -> Vec<(usize, usize)> {
        let pats: Vec<u64> = (0..self.voice.len()).map(|t| Self::pattern(self.voice.command(t))).collect();
        voice
            .as_words()
            .iter()
            .enumerate()
            .filter_map(|(w, &x)| {
                pats.iter()
                    .enumerate()
                    .map(|(t, &p)| ((x & p).count_ones(), t, p.count_ones()))
                    .filter(|&(o, _, n)| n > 0 && 2 * o >= n)
                    .max_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)))
                    .map(|(_, t, _)| (w, t))
            })
            .collect()
    }

    /// The body's moves and tokens as place codes (`Effector::fix_place`), the same every lifetime:
    /// what is learned in a lifetime is what the senses mean, not which line moves which muscle.
    pub fn fix_motor(&mut self) {
        self.hand.fix_place();
        self.voice.fix_place();
    }

    /// The simple stimuli at fixed places, as touch and sight are mapped: light k always in word k
    /// of the eye (`PLACE_LINES`), and a stroke always in the first word of the skin's map.
    /// Tokens' glyphs and sounds stay drawn afresh (what a word means is learned).
    pub fn place_senses(&mut self) {
        let words = CH_BITS / 64;
        for (k, l) in self.lights.iter_mut().enumerate() {
            let mut v = vec![0u64; words];
            v[k % words] = PLACE_LINES;
            *l = BitVector::from_words(v);
        }
        self.stroke_place = true;
    }

    /// Tokens' glyphs, sounds and voice codes all in the first word of their channels (still drawn
    /// afresh each lifetime), `k` random lines of 64 each (0: `ACTIVE`, 16, so two share about 4):
    /// a glyph, its sound and its voice code then sit at the same place, and a topographic kernel
    /// can write one from the other. Sparser codes crowd a word less: with 16 lines, a word holding
    /// three codes shows half the lines of 84% of other random codes; with 4 to 6, 14 to 17%.
    pub fn one_word(&mut self, physics: u64, k: usize) {
        let k = if k == 0 { ACTIVE } else { k.min(64) };
        let mut rng = StdRng::seed_from_u64(physics ^ 0x0E0D);
        let n = self.glyphs.len();
        self.glyphs = one_word_codes(n, k, &mut rng);
        if let Some(b) = self.body {
            // (a fixed body) its sounds and voice codes from its own seed, each from its own
            // stream, so a lifetime with fewer tokens has the first of the same codes
            self.sounds = one_word_codes(n, k, &mut StdRng::seed_from_u64(b ^ 0x0E0E));
            self.voice.one_word(k, &mut StdRng::seed_from_u64(b ^ 0x0E0F));
            return;
        }
        self.sounds = one_word_codes(n, k, &mut rng);
        self.voice.one_word(k, &mut rng);
    }

    /// The voice's codes and their sounds as one-word codes of `k` lines (as `one_word` makes them,
    /// from the fixed body's seed when there is one), the glyphs left as they are: for lifetimes
    /// with nothing to read (`evolve::set_one_word_voice`).
    pub fn one_word_voice(&mut self, physics: u64, k: usize) {
        let k = if k == 0 { ACTIVE } else { k.min(64) };
        let n = self.sounds.len();
        if let Some(b) = self.body {
            self.sounds = one_word_codes(n, k, &mut StdRng::seed_from_u64(b ^ 0x0E0E));
            self.voice.one_word(k, &mut StdRng::seed_from_u64(b ^ 0x0E0F));
            return;
        }
        let mut rng = StdRng::seed_from_u64(physics ^ 0x0E1D);
        self.sounds = one_word_codes(n, k, &mut rng);
        self.voice.one_word(k, &mut rng);
    }

    /// The body fixed by its own seed, the same in every lifetime of a run (anatomy, not
    /// learned): the voice's and the hand's codes, the sound the voice makes for each token (the
    /// teacher speaks the same language) and the feel of each move. What the world shows (glyphs,
    /// lights, tones, cues) and what pays stay drawn with the world each lifetime, so learning is
    /// forced by the world, not by a new body. A lifetime with fewer tokens has the first of the
    /// same codes.
    pub fn fix_body(&mut self, body: u64) {
        let r = |salt: u64| StdRng::seed_from_u64(body ^ salt);
        let n = self.sounds.len();
        self.sounds = codes(n, &mut r(0x50D5));
        self.felt = codes(MOVES, &mut r(0xFE17));
        self.voice = Effector::new(n, &mut r(0x7AC7));
        self.hand = Effector::new(MOVES, &mut r(0x40B5));
        self.body = Some(body);
    }

    pub fn mirror_sounds(&mut self, eighths: usize, physics: u64) {
        let mut rng = StdRng::seed_from_u64(physics ^ 0x3122);
        for k in 0..self.sounds.len() {
            let v = self.voice.command(k);
            let w = (0..v.word_len()).find(|&i| v.word(i) != 0).unwrap_or(0);
            let on: Vec<usize> = (0..64).filter(|b| v.word(w) >> b & 1 == 1).collect();
            let off: Vec<usize> = (0..64).filter(|b| v.word(w) >> b & 1 == 0).collect();
            let keep = (on.len() * eighths.min(8) / 8).min(on.len());
            let mut lines: Vec<usize> = on.choose_multiple(&mut rng, keep).map(|b| w * 64 + b).collect();
            lines.extend(off.choose_multiple(&mut rng, on.len() - keep).map(|b| w * 64 + b));
            self.sounds[k] = BitVector::from_bits(&lines, CH_BITS);
        }
    }

    pub fn with_physics(physics: u64, seed: u64, tokens: usize) -> Self {
        let r = |salt: u64| StdRng::seed_from_u64(physics ^ salt);
        Self {
            rng: StdRng::seed_from_u64(seed ^ 0xD1CE),
            glyphs: codes(tokens, &mut r(0x6171)),
            sounds: codes(tokens, &mut r(0x50D5)),
            tones: codes(2, &mut r(0x7070)),
            lights: codes(2, &mut r(0x1167)),
            felt: codes(MOVES, &mut r(0xFE17)),
            cues: codes(2, &mut r(0xC0E5)),
            task_cue: None,
            voice: Effector::new(tokens, &mut r(0x7AC7)),
            hand: Effector::new(MOVES, &mut r(0x40B5)),
            cond_total: 0,
            reversal_period: 500,
            gap_max: 0,
            teacher: false,
            teacher_rate: 4,
            teacher_after: 0,
            shown_so_far: 0,
            delay: 3,
            vary_delay: 0,
            distract: false,
            two_lights: false,
            go_tone: false,
            maze_mirror: false,
            trial_delay: 3,
            distract_at: None,
            scene: Scene::Idle,
            stroke_next: false,
            blur: 0,
            shift: 0,
            exemplar: None,
            maze_len: 2,
            cue_gone: false,
            symbol: false,
            maze_fixed: false,
            maze_nocue: false,
            quiet: false,
            sentence: false,
            body: None,
            stroke_place: false,
            maze_first: None,
            junction: codes(1, &mut r(0x7A2E)).remove(0),
            maze_total: 0,
            screen_w: 2,
            screen_h: 2,
            screen_lit: 1,
            screen_cell: 64,
            screen_walls: 0,
            screen_torch: false,
            screen_dark: false,
            in_place: crate::program::neurocomp::in_place(),
            count_reads: crate::program::neurocomp::count_reads(),
            explore: crate::program::neurocomp::explore(),
            shadow: crate::program::neurocomp::shadow(),
            cursor_mark: {
                let mut c = StdRng::seed_from_u64(physics ^ 0xC025);
                let lines: Vec<u64> = (0..64).collect();
                lines.choose_multiple(&mut c, 8).fold(0u64, |m, &b| m | 1 << b)
            },
            wall_mark: {
                let mut c = StdRng::seed_from_u64(physics ^ 0x3A11);
                let lines: Vec<u64> = (0..64).collect();
                lines.choose_multiple(&mut c, 24).fold(0u64, |m, &b| m | 1 << b)
            },
            screen_total: 0,
            present: false,
            events: Vec::new(),
        }
    }

    /// The body, for the program at birth.
    pub fn body(&self) -> Body<'_> {
        Body { voice: &self.voice, hand: &self.hand }
    }

    /// The tokens' glyphs and sounds this lifetime, for a harness that inspects a lifetime (never
    /// given to the network).
    pub fn codes(&self) -> (&[BitVector], &[BitVector]) {
        (&self.glyphs, &self.sounds)
    }

    /// The scene is over (a new one may begin).
    pub fn done(&self) -> bool {
        matches!(self.scene, Scene::Idle)
    }

    /// Where a delayed trial's response window starts for a delay of `d`: step 5 for a delay up to
    /// 5 (the light at 5 - d), else step d (the light at 0).
    fn window_start(d: usize) -> usize {
        d.max(5)
    }

    /// A trial's length: `TRIAL` steps, or for a delayed trial with a long delay, long enough for
    /// its window and the stretch after it.
    fn trial_len(&self, kind: Conditioning) -> usize {
        match kind {
            Conditioning::Delayed => TRIAL.max(Self::window_start(self.trial_delay) + 5),
            _ => TRIAL,
        }
    }

    /// Conditioning: `trials` trials of `TRIAL` steps, a caregiver there throughout.
    pub fn conditioning(&mut self, kind: Conditioning, trials: usize) {
        self.present = true;
        self.cond_total = trials;
        let cue = self.rng.gen_bool(0.5);
        self.scene = Scene::Conditioning(kind, trials, 0, cue, false);
    }

    /// A sequence: the tokens shown one per step.
    pub fn sequence(&mut self, tokens: &[usize]) {
        self.sequence_marked(tokens, &[]);
    }

    /// A sequence with some tokens marked as breaking its pattern (for the observer only).
    pub fn sequence_marked(&mut self, tokens: &[usize], odd: &[bool]) {
        self.present = false;
        self.scene = Scene::Sequence(tokens.iter().enumerate().map(|(i, &t)| (t, odd.get(i).copied().unwrap_or(false))).collect());
    }

    /// A T-maze: `trials` trials. Each starts at the foot of a stem `maze_len` steps long; a click
    /// steps forward, and at the junction (the eye shows its code) a move left or right enters an
    /// arm: the food's side is stroked (felt the step after), the other gives nothing, and either
    /// ends the trial, as does taking `maze_len + 6` steps. The cue (where the food is) is on the
    /// eye all along the stem, or only on the trial's first step, at its foot (`cue_gone`).
    pub fn maze(&mut self, trials: usize) {
        self.present = false;
        self.maze_total = trials;
        let side = self.maze_side();
        self.scene = Scene::Maze(trials, 0, 0, side);
    }

    /// The cue for a maze trial: a light's code (symbolic), or the first light's lines moved into
    /// the eye's left or right half (spatial).
    fn maze_cue(&self, side: bool) -> BitVector {
        if self.symbol {
            return self.lights[side as usize].clone();
        }
        let w = self.lights[0].as_words();
        let half = w.len() / 2;
        let mut out = vec![0u64; w.len()];
        let at = side != self.maze_mirror;
        for (i, &x) in w.iter().enumerate() {
            out[i % half + at as usize * half] |= x;
        }
        BitVector::from_words(out)
    }

    /// A maze trial's food side: drawn each trial, or (`maze_fixed`) drawn once for the lifetime.
    fn maze_side(&mut self) -> bool {
        let s = self.rng.gen_bool(0.5);
        if self.maze_fixed { *self.maze_first.get_or_insert(s) } else { s }
    }

    /// The two lights' roles swapped (`evolve::set_twins`): the first light (the one that pays first,
    /// the delayed cue) becomes the second, and the second the first.
    pub fn swap_lights(&mut self) {
        self.lights.swap(0, 1);
    }

    /// The two lights as one-word codes of `k` lines (`evolve::set_mixed_codes`), as a reading
    /// lifetime's glyphs are, drawn from the world's own stream.
    pub fn one_word_lights(&mut self, physics: u64, k: usize) {
        let k = if k == 0 { ACTIVE } else { k.min(64) };
        self.lights = one_word_codes(2, k, &mut StdRng::seed_from_u64(physics ^ 0x0E11));
    }

    /// The two lights in one word (`evolve::set_same_word_lights`): the second light redrawn with
    /// the first's number of lines inside the first's word, so the two differ by their pattern,
    /// not by where they are.
    pub fn same_word_lights(&mut self) {
        let a = self.lights[0].as_words().to_vec();
        let Some(w) = a.iter().position(|&x| x != 0) else { return };
        let k = a[w].count_ones() as usize;
        let lines: Vec<usize> = (0..64).collect();
        let picked: Vec<usize> = lines.choose_multiple(&mut self.rng, k).map(|b| w * 64 + b).collect();
        self.lights[1] = BitVector::from_bits(&picked, a.len() * 64);
    }

    /// The scene after a maze trial: a gap of 0 to `gap_max` steps, then the next trial.
    fn after_maze(&mut self, left: usize) -> Scene {
        if left <= 1 {
            return Scene::Idle;
        }
        let g = if self.gap_max > 0 { self.rng.gen_range(0..=self.gap_max) } else { 0 };
        if g > 0 {
            Scene::MazeGap(left - 1, g)
        } else {
            let side = self.maze_side();
            Scene::Maze(left - 1, 0, 0, side)
        }
    }

    /// A screen: `trials` trials. The eye shows a grid of `screen_w` x `screen_h` cells of
    /// `screen_cell` bits: `screen_lit` cells hold a glyph (the target, the first glyph, and
    /// different distractors; in a cell under 64 bits, a glyph's lines folded to its width),
    /// `screen_walls` hold the wall's mark, and the cell under the cursor also holds the cursor's
    /// mark. Left, right, up and down move the cursor one cell (off the edge or into a wall:
    /// nothing); a click selects the cell under it, stroked (the step after) if it is the target,
    /// and ends the trial, as do 4 x (columns + rows) + 4 steps. The cursor, glyphs and walls are
    /// placed at random, every free cell reachable from every other.
    /// Feeling: `steps` steps with nothing to see or hear, a stroke on one step in eight at random,
    /// whatever the program does: noticing reward with nothing to earn it (the `feel` probe).
    pub fn feel(&mut self, steps: usize) {
        self.present = false;
        self.scene = Scene::Feel(steps);
    }

    pub fn screen(&mut self, trials: usize) {
        self.present = false;
        self.screen_total = trials;
        self.scene = self.screen_trial(trials);
    }

    /// The screen's columns and cells, within what the eye holds at this cell size.
    fn screen_shape(&self) -> (usize, usize) {
        let most = CH_BITS / self.screen_cell.clamp(16, 64).next_power_of_two();
        let w = self.screen_w.clamp(1, most);
        (w, (w * self.screen_h.max(1)).clamp(2, most.min(32)))
    }

    /// A new screen trial's layout: walls first, then the cursor and the glyphs on free cells,
    /// drawn again until every free cell can be reached from every other.
    fn screen_trial(&mut self, left: usize) -> Scene {
        let (w, cells) = self.screen_shape();
        loop {
            let mut order: Vec<usize> = (0..cells).collect();
            order.shuffle(&mut self.rng);
            let walls_n = self.screen_walls.min(cells.saturating_sub(2));
            let walls = order[..walls_n].iter().fold(0u32, |m, &c| m | 1 << c);
            let free = &order[walls_n..];
            let lit = self.screen_lit.clamp(1, (free.len() - 1).min(self.glyphs.len()).max(1));
            let mut glyph = [None; 32];
            for (k, &c) in free.iter().take(lit).enumerate() {
                glyph[c] = Some(k);
            }
            let (target, cursor) = (free[0], free[self.rng.gen_range(0..free.len())]);
            // reachable: a flood from the cursor through free cells
            let mut seen = 1u32 << cursor;
            let mut stack = vec![cursor];
            while let Some(c) = stack.pop() {
                let next = [(c % w > 0).then(|| c - 1), (c % w + 1 < w && c + 1 < cells).then(|| c + 1), (c >= w).then(|| c - w), (c + w < cells).then(|| c + w)];
                for n in next.into_iter().flatten() {
                    if walls >> n & 1 == 0 && seen >> n & 1 == 0 {
                        seen |= 1 << n;
                        stack.push(n);
                    }
                }
            }
            // every free cell reachable (so the walls cut no pocket off: a blind click then hits
            // the target 1 time in the free cells, not 1 in a smaller pocket shared with it)
            let all_free = free.iter().fold(0u32, |m, &c| m | 1 << c);
            if seen == all_free {
                let _ = target;
                return Scene::Screen(left, 0, Box::new(ScreenTrial { cursor, target, glyph, walls }));
            }
        }
    }

    /// The scene after a screen trial: a gap of 0 to `gap_max` steps, then the next trial.
    fn after_screen(&mut self, left: usize) -> Scene {
        if left <= 1 {
            return Scene::Idle;
        }
        let g = if self.gap_max > 0 { self.rng.gen_range(0..=self.gap_max) } else { 0 };
        if g > 0 {
            Scene::ScreenGap(left - 1, g)
        } else {
            self.screen_trial(left - 1)
        }
    }

    /// Echo play: `steps` steps; a caregiver comes or goes every 8, there half the time.
    pub fn echo(&mut self, steps: usize) {
        self.scene = Scene::Echo(steps, false);
    }

    /// Whether light A pays on the trial with `left` trials to go (counting this one).
    pub fn a_pays(&self, left: usize) -> bool {
        (self.cond_total - left) / self.reversal_period.max(1) % 2 == 0
    }

    /// One step: the program's act in, its senses out.
    pub fn step(&mut self, act: &BitVector) -> BitVector {
        // (sentences) each word of the voice says the token it holds, in order
        let words: Vec<(usize, usize)> = if self.sentence { self.sentence_of(&channel(act, VOICE)) } else { Vec::new() };
        let said = if self.sentence { words.first().map(|x| x.1) } else { self.voice.articulate(&channel(act, VOICE)) };
        let moved = self.hand.articulate(&channel(act, HAND));
        let mut ev = Event { scene: SceneKind::Idle, trial: None, moved, said, sentence: words.iter().map(|x| x.1).collect(), shown: None, echoed: false, stroked: false, present: self.present, taught: false, odd: false, maze: None, chose: None, screen: None, window: false };
        let mut senses = BitVector::new(SENSE_BITS, Some(0));
        // the body: one's own voice is heard, one's own move felt
        if self.sentence {
            if !self.quiet {
                for &(w, t) in &words {
                    let mut v = vec![0u64; CH_BITS / 64];
                    v[w] = Self::pattern(&self.sounds[t]);
                    put(&mut senses, EAR, &BitVector::from_words(v));
                }
            }
        } else if let Some(w) = said.filter(|_| !self.quiet) {
            put(&mut senses, EAR, &self.sounds[w].clone());
        }
        if let Some(m) = moved {
            put(&mut senses, TOUCH, &self.felt[m].clone());
        }
        // a stroke owed from the step before
        let mut stroke = std::mem::take(&mut self.stroke_next);
        let scene = std::mem::replace(&mut self.scene, Scene::Idle);
        self.scene = match scene {
            Scene::Idle => Scene::Idle,
            Scene::Conditioning(kind, left, t, cue, mut stroked) => {
                ev.scene = SceneKind::Conditioning(kind);
                let a_pays = self.a_pays(left);
                let pays = match kind {
                    Conditioning::Reversal => cue == a_pays,
                    _ => cue,
                };
                ev.trial = Some((self.cond_total - left, t, cue, pays));
                let click = moved == Some(CLICK);
                match kind {
                    Conditioning::Pavlov => {
                        if t == 2 {
                            put(&mut senses, EAR, &self.tones[!cue as usize].clone());
                        }
                        if t == 4 && cue {
                            stroke = true;
                        }
                    }
                    Conditioning::Instrumental => {
                        if cue && (2..=4).contains(&t) {
                            put(&mut senses, EYE, &self.lights[0].clone());
                        }
                        if cue && click && (3..=5).contains(&t) && !stroked {
                            self.stroke_next = true;
                            stroked = true;
                        }
                    }
                    Conditioning::Delayed => {
                        // a trial's delay and distractor, drawn as it starts (draws only when
                        // asked for, so a plain lifetime runs as it always did)
                        if t == 0 {
                            self.trial_delay = if self.vary_delay > 0 { self.rng.gen_range(1..=self.vary_delay.min(32)) } else { self.delay.clamp(1, 32) };
                            let d = self.trial_delay;
                            self.distract_at = (self.distract && d >= 2 && self.rng.gen_bool(0.5)).then(|| {
                                let from = Self::window_start(d) - d + 1;
                                self.rng.gen_range(from..Self::window_start(d))
                            });
                        }
                        let d = self.trial_delay;
                        let ws = Self::window_start(d);
                        if t + d == ws {
                            if cue {
                                put(&mut senses, EYE, &self.lights[0].clone());
                            } else if self.two_lights {
                                put(&mut senses, EYE, &self.lights[1].clone());
                            }
                        }
                        if self.distract_at == Some(t) {
                            let other = if self.two_lights && !cue { 0 } else { 1 };
                            put(&mut senses, EYE, &self.lights[other].clone());
                        }
                        // (`go_tone`) a click before the window spends the trial
                        if self.go_tone && click && t < ws {
                            stroked = true;
                        }
                        if self.go_tone && t == ws {
                            put(&mut senses, EAR, &self.tones[0].clone());
                        }
                        ev.window = (ws..=ws + 2).contains(&t) && !(self.go_tone && stroked);
                        if cue && click && ev.window && !stroked {
                            self.stroke_next = true;
                            stroked = true;
                        }
                    }
                    Conditioning::Reversal => {
                        if (2..=4).contains(&t) {
                            if t == 2 || self.exemplar.is_none() {
                                let light = self.lights[!cue as usize].clone();
                                self.exemplar = Some(if self.blur > 0 || self.shift > 0 { self.distort(&light) } else { light });
                            }
                            put(&mut senses, EYE, &self.exemplar.clone().unwrap());
                        }
                        if click && (3..=5).contains(&t) && !stroked && pays {
                            self.stroke_next = true;
                            stroked = true;
                        }
                    }
                }
                if t + 1 < self.trial_len(kind) {
                    Scene::Conditioning(kind, left, t + 1, cue, stroked)
                } else if left > 1 && self.gap_max > 0 {
                    let g = self.rng.gen_range(0..=self.gap_max);
                    if g == 0 {
                        let cue = self.rng.gen_bool(0.5);
                        Scene::Conditioning(kind, left - 1, 0, cue, false)
                    } else {
                        Scene::Gap(kind, left - 1, g)
                    }
                } else if left > 1 {
                    let cue = self.rng.gen_bool(0.5);
                    Scene::Conditioning(kind, left - 1, 0, cue, false)
                } else {
                    self.present = false;
                    Scene::Idle
                }
            }
            Scene::Gap(kind, left, g) => {
                ev.scene = SceneKind::Conditioning(kind);
                if g > 1 {
                    Scene::Gap(kind, left, g - 1)
                } else {
                    let cue = self.rng.gen_bool(0.5);
                    Scene::Conditioning(kind, left, 0, cue, false)
                }
            }
            Scene::Sequence(mut q) => {
                ev.scene = SceneKind::Sequence;
                if let Some((tok, odd)) = q.pop_front() {
                    put(&mut senses, EYE, &self.glyphs[tok].clone());
                    ev.shown = Some(tok);
                    ev.odd = odd;
                    self.shown_so_far += 1;
                    if self.teacher && self.shown_so_far > self.teacher_after && self.rng.gen_ratio(self.teacher_rate.min(8), 8) {
                        if self.sentence {
                            // (sentences) the teacher's word at the ear's first place
                            let mut v = vec![0u64; CH_BITS / 64];
                            v[0] = Self::pattern(&self.sounds[tok]);
                            put(&mut senses, EAR, &BitVector::from_words(v));
                        } else {
                            put(&mut senses, EAR, &self.sounds[tok].clone());
                        }
                        ev.taught = true;
                    }
                }
                if q.is_empty() { Scene::Idle } else { Scene::Sequence(q) }
            }
            Scene::Maze(left, mut pos, t, side) => {
                ev.scene = SceneKind::Maze;
                // the move made acts on where the program is (a move that does nothing does
                // nothing: a click at the junction, a turn on the stem); a trial's first step only
                // shows where it starts (the move then was made before it began)
                let mut done = false;
                if t == 0 {
                } else if pos < self.maze_len && moved == Some(CLICK) {
                    pos += 1;
                } else if pos >= self.maze_len && (moved == Some(LEFT) || moved == Some(RIGHT)) {
                    let right = moved == Some(RIGHT);
                    ev.chose = Some(right == side);
                    if right == side {
                        self.stroke_next = true;
                    }
                    done = true;
                }
                ev.maze = Some((self.maze_total - left, pos, side));
                if done || t + 1 >= self.maze_len + 6 {
                    self.after_maze(left)
                } else {
                    // what the program sees where it now is
                    if !self.maze_nocue && (!self.cue_gone || t == 0) {
                        put(&mut senses, EYE, &self.maze_cue(side));
                    }
                    if pos >= self.maze_len {
                        put(&mut senses, EYE, &self.junction.clone());
                    }
                    Scene::Maze(left, pos, t + 1, side)
                }
            }
            Scene::Screen(left, t, mut tr) => {
                ev.scene = SceneKind::Screen;
                let (w, cells) = self.screen_shape();
                let bits = self.screen_cell.clamp(16, 64).next_power_of_two();
                ev.screen = Some((self.screen_total - left, cells - (tr.walls.count_ones() as usize)));
                // the move made acts on the cursor (a trial's first step only shows the screen)
                let mut done = false;
                if t > 0 {
                    let c = tr.cursor;
                    let to = match moved {
                        Some(LEFT) if c % w > 0 => Some(c - 1),
                        Some(RIGHT) if c % w + 1 < w && c + 1 < cells => Some(c + 1),
                        Some(UP) if c >= w => Some(c - w),
                        Some(DOWN) if c + w < cells => Some(c + w),
                        _ => None,
                    };
                    if let Some(n) = to.filter(|&n| tr.walls >> n & 1 == 0) {
                        tr.cursor = n;
                    }
                    if moved == Some(CLICK) {
                        ev.chose = Some(tr.cursor == tr.target);
                        if tr.cursor == tr.target {
                            self.stroke_next = true;
                        }
                        done = true;
                    }
                }
                if done || t + 1 >= 4 * (w + cells / w) + 4 {
                    self.after_screen(left)
                } else {
                    // each cell's lines (a pattern folded to the cell's width), at its place
                    let fold = |x: u64| -> u64 {
                        let mut f = 0u64;
                        let mut y = x;
                        while y != 0 {
                            f |= y & (u64::MAX >> (64 - bits));
                            y = if bits == 64 { 0 } else { y >> bits };
                        }
                        f
                    };
                    let mut eye = vec![0u64; CH_BITS / 64];
                    let mut place = |c: usize, x: u64| {
                        let at = c * bits;
                        eye[at / 64] |= fold(x) << (at % 64);
                    };
                    for c in 0..cells {
                        if let Some(g) = tr.glyph[c].filter(|_| !self.screen_torch || c == tr.cursor) {
                            place(c, self.glyphs[g].as_words().iter().fold(0, |m, &x| m | x));
                        }
                        if tr.walls >> c & 1 == 1 {
                            place(c, self.wall_mark);
                        }
                    }
                    if !self.screen_dark {
                        place(tr.cursor, self.cursor_mark);
                    }
                    put(&mut senses, EYE, &BitVector::from_words(eye));
                    Scene::Screen(left, t + 1, tr)
                }
            }
            Scene::Feel(left) => {
                ev.scene = SceneKind::Feel;
                if self.rng.gen_ratio(1, 8) {
                    stroke = true;
                }
                if left > 1 { Scene::Feel(left - 1) } else { Scene::Idle }
            }
            Scene::ScreenGap(left, g) => {
                ev.scene = SceneKind::Screen;
                if g > 1 {
                    Scene::ScreenGap(left, g - 1)
                } else {
                    self.screen_trial(left)
                }
            }
            Scene::MazeGap(left, g) => {
                ev.scene = SceneKind::Maze;
                if g > 1 {
                    Scene::MazeGap(left, g - 1)
                } else {
                    let side = self.maze_side();
                    Scene::Maze(left, 0, 0, side)
                }
            }
            Scene::Echo(left, mut caregiver) => {
                ev.scene = SceneKind::Echo;
                if left % 8 == 0 {
                    caregiver = self.rng.gen_bool(0.5);
                    self.present = caregiver;
                }
                // the caregiver answers the last sound made aloud, most of the time (heard the
                // step after; one's own voice is heard now)
                if said.is_none() && caregiver {
                    if let Some(w) = self.events.last().and_then(|e| e.said) {
                        if self.rng.gen_bool(0.7) {
                            put(&mut senses, EAR, &self.sounds[w].clone());
                            ev.echoed = true;
                        }
                    }
                }
                if left > 1 {
                    Scene::Echo(left - 1, caregiver)
                } else {
                    self.present = false;
                    Scene::Idle
                }
            }
        };
        if stroke {
            // a slow stroke: a run of neighbouring receptors on the skin's map
            // (inside one word of the skin's map)
            let start = if self.stroke_place { 0 } else { self.rng.gen_range(0..CH_BITS / 64) * 64 } + self.rng.gen_range(0..=64 - ACTIVE);
            let run: Vec<usize> = (start..start + ACTIVE).collect();
            put(&mut senses, AFFECT, &BitVector::from_bits(&run, CH_BITS));
            ev.stroked = true;
        }
        if let Some(k) = self.task_cue {
            put(&mut senses, TOUCH, &self.cues[k % self.cues.len()].clone());
        }
        ev.present = self.present;
        self.events.push(ev);
        senses
    }

    /// The screen's cursor and target cells, its walls, its columns and cells, if a screen trial
    /// is on (for tests' policies only).
    pub fn screen_state_for_test(&self) -> Option<(usize, usize, u32, usize, usize)> {
        let (w, cells) = self.screen_shape();
        match self.scene {
            Scene::Screen(_, _, ref tr) => Some((tr.cursor, tr.target, tr.walls, w, cells)),
            _ => None,
        }
    }

    /// The two lights' codes (for tests outside the program; a program only sees them).
    pub fn lights_for_test(&self) -> &[BitVector] {
        &self.lights
    }

    /// The act vector that drives voice token `say` and hand move `hand` (either may be None).
    pub fn act(&self, say: Option<usize>, hand: Option<usize>) -> BitVector {
        let mut a = BitVector::new(ACT_BITS, Some(0));
        if let Some(w) = say {
            put(&mut a, VOICE, &self.voice.command(w).clone());
        }
        if let Some(m) = hand {
            put(&mut a, HAND, &self.hand.command(m).clone());
        }
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fixed_body_is_the_same_in_every_world_and_the_world_is_not() {
        let mut a = World::with_physics(11, 1, 8);
        let mut b = World::with_physics(22, 2, 4);
        a.fix_body(7);
        b.fix_body(7);
        for t in 0..4 {
            assert_eq!(a.sounds[t].as_words(), b.sounds[t].as_words());
            assert_eq!(a.voice.command(t).as_words(), b.voice.command(t).as_words());
        }
        assert_eq!(a.hand.command(0).as_words(), b.hand.command(0).as_words());
        assert_ne!(a.glyphs[0].as_words(), b.glyphs[0].as_words());
        a.one_word(11, 8);
        b.one_word(22, 8);
        for t in 0..4 {
            assert_eq!(a.sounds[t].as_words(), b.sounds[t].as_words());
            assert_eq!(a.voice.command(t).as_words(), b.voice.command(t).as_words());
        }
        assert_ne!(a.glyphs[0].as_words(), b.glyphs[0].as_words());
    }

    #[test]
    fn a_sentence_says_a_token_in_any_word_in_order() {
        let mut w = World::with_physics(9, 1, 2);
        w.sentence = true;
        let p = |t: usize| World::pattern(w.voice.command(t));
        // token 1 in word 0, token 0 in word 5: said in that order, wherever the codes were drawn
        let mut v = vec![0u64; CH_BITS / 64];
        v[0] = p(1);
        v[5] = p(0);
        assert_eq!(w.sentence_of(&BitVector::from_words(v)), vec![(0, 1), (5, 0)]);
    }

    #[test]
    fn a_mirrored_sound_is_said_as_its_token_when_copied() {
        for m in [8usize, 6, 4] {
            let mut w = World::with_physics(5, 1, 2);
            w.mirror_sounds(m, 5);
            for k in 0..2 {
                assert_eq!(w.body().voice.articulate(&w.sounds[k]), Some(k), "mirror {m}, token {k}");
            }
        }
        // below half the lines a copied sound says nothing of its own
        let mut w = World::with_physics(5, 1, 2);
        w.mirror_sounds(2, 5);
        assert!((0..2).all(|k| w.body().voice.articulate(&w.sounds[k]) != Some(k)));
    }

    #[test]
    fn a_command_drives_its_primitive_and_noise_does_nothing() {
        let w = World::new(1, 8);
        for m in [LEFT, RIGHT, CLICK, UP, DOWN] {
            let a = w.act(None, Some(m));
            assert_eq!(w.hand.articulate(&channel(&a, HAND)), Some(m), "move {m}");
        }
        let a = w.act(Some(3), Some(CLICK));
        assert_eq!(w.voice.articulate(&channel(&a, VOICE)), Some(3));
        assert_eq!(w.hand.articulate(&channel(&a, HAND)), Some(CLICK));
        assert_eq!(w.voice.articulate(&BitVector::new(CH_BITS, Some(0))), None);
    }

    #[test]
    fn instrumental_click_in_the_window_is_stroked_next_step() {
        let mut w = World::new(2, 4);
        w.conditioning(Conditioning::Instrumental, 40);
        let mut strokes = 0;
        while !w.done() {
            let (cue, t) = match w.scene {
                Scene::Conditioning(_, _, t, cue, _) => (cue, t),
                _ => unreachable!(),
            };
            let act = if t == 3 { w.act(None, Some(CLICK)) } else { w.act(None, None) };
            let s = w.step(&act);
            if channel(&s, AFFECT).count_ones() > 0 {
                strokes += 1;
                assert!(w.events.last().unwrap().trial.unwrap().1 == 4);
            }
            let _ = cue;
        }
        let cue_trials = w.events.iter().filter(|e| matches!(e.trial, Some((_, 0, true, _)))).count();
        assert_eq!(strokes, cue_trials);
    }

    #[test]
    fn reversal_swaps_the_paying_light() {
        let mut w = World::new(3, 4);
        w.reversal_period = 5;
        w.conditioning(Conditioning::Reversal, 20);
        while !w.done() {
            let a = w.act(None, None);
            w.step(&a);
        }
        for e in &w.events {
            let (trial, _, cue, pays) = e.trial.unwrap();
            assert_eq!(pays, cue == ((trial / 5) % 2 == 0));
        }
    }

    #[test]
    fn a_sequence_shows_each_token_once() {
        let mut w = World::new(4, 5);
        w.sequence(&[0, 1, 2, 1]);
        let mut shown = Vec::new();
        while !w.done() {
            let a = w.act(None, None);
            let s = w.step(&a);
            assert!(channel(&s, EYE).count_ones() == ACTIVE);
            shown.push(w.events.last().unwrap().shown.unwrap());
        }
        assert_eq!(shown, vec![0, 1, 2, 1]);
    }
}
