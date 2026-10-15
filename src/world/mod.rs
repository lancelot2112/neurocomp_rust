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
    /// the voice's token this step, if any
    pub said: Option<usize>,
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
}

/// An effector's motor primitives: random sparse codes, one per move or token.
pub struct Effector {
    codes: Vec<BitVector>,
}

impl Effector {
    fn new(n: usize, rng: &mut StdRng) -> Self {
        Self { codes: codes(n, rng) }
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
    /// delayed conditioning: how many steps before the response window (steps 5 to 7) the light
    /// flashes, 1 to 5 (3: at step 2, three to five steps before a click pays)
    pub delay: usize,
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
    /// programs built for this world keep each declared channel in one frame, held in place
    /// from step to step (only clearing writes take lines away, and a write is seen by the edges
    /// after it in the same step); default: the process's (`neurocomp::set_in_place`)
    pub in_place: bool,
    /// programs built for this world count the words their kernel populations read as work;
    /// default: the process's (`neurocomp::set_count_reads`)
    pub count_reads: bool,
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
    pub fn with_physics(physics: u64, seed: u64, tokens: usize) -> Self {
        let r = |salt: u64| StdRng::seed_from_u64(physics ^ salt);
        Self {
            rng: StdRng::seed_from_u64(seed ^ 0xD1CE),
            glyphs: codes(tokens, &mut r(0x6171)),
            sounds: codes(tokens, &mut r(0x50D5)),
            tones: codes(2, &mut r(0x7070)),
            lights: codes(2, &mut r(0x1167)),
            felt: codes(MOVES, &mut r(0xFE17)),
            voice: Effector::new(tokens, &mut r(0x7AC7)),
            hand: Effector::new(MOVES, &mut r(0x40B5)),
            cond_total: 0,
            reversal_period: 500,
            gap_max: 0,
            teacher: false,
            teacher_rate: 4,
            delay: 3,
            scene: Scene::Idle,
            stroke_next: false,
            blur: 0,
            shift: 0,
            exemplar: None,
            maze_len: 2,
            cue_gone: false,
            symbol: false,
            junction: codes(1, &mut r(0x7A2E)).remove(0),
            maze_total: 0,
            screen_w: 2,
            screen_h: 2,
            screen_lit: 1,
            screen_cell: 64,
            screen_walls: 0,
            screen_torch: false,
            in_place: crate::program::neurocomp::in_place(),
            count_reads: crate::program::neurocomp::count_reads(),
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

    /// The scene is over (a new one may begin).
    pub fn done(&self) -> bool {
        matches!(self.scene, Scene::Idle)
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
        let side = self.rng.gen_bool(0.5);
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
        for (i, &x) in w.iter().enumerate() {
            out[i % half + side as usize * half] |= x;
        }
        BitVector::from_words(out)
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
            let side = self.rng.gen_bool(0.5);
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
        let said = self.voice.articulate(&channel(act, VOICE));
        let moved = self.hand.articulate(&channel(act, HAND));
        let mut ev = Event { scene: SceneKind::Idle, trial: None, moved, said, shown: None, echoed: false, stroked: false, present: self.present, taught: false, odd: false, maze: None, chose: None, screen: None };
        let mut senses = BitVector::new(SENSE_BITS, Some(0));
        // the body: one's own voice is heard, one's own move felt
        if let Some(w) = said {
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
                        if cue && t + self.delay.clamp(1, 5) == 5 {
                            put(&mut senses, EYE, &self.lights[0].clone());
                        }
                        if cue && click && (5..=7).contains(&t) && !stroked {
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
                if t + 1 < TRIAL {
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
                    if self.teacher && self.rng.gen_ratio(self.teacher_rate.min(8), 8) {
                        put(&mut senses, EAR, &self.sounds[tok].clone());
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
                    if !self.cue_gone || t == 0 {
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
                    place(tr.cursor, self.cursor_mark);
                    put(&mut senses, EYE, &BitVector::from_words(eye));
                    Scene::Screen(left, t + 1, tr)
                }
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
                    let side = self.rng.gen_bool(0.5);
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
            let start = self.rng.gen_range(0..CH_BITS / 64) * 64 + self.rng.gen_range(0..=64 - ACTIVE);
            let run: Vec<usize> = (start..start + ACTIVE).collect();
            put(&mut senses, AFFECT, &BitVector::from_bits(&run, CH_BITS));
            ev.stroked = true;
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
