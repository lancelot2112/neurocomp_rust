//! The world a mind lives in: a body, a screen, other people. The world and the mind meet only
//! through the senses and the acts.
//!
//! - **The mind acts** (`Acts`): a command to the vocal tract, a command to the hand (the mouse).
//!   The body carries them out: a command that matches no motor pattern does nothing.
//! - **The world answers with senses** (`Senses`): what the eye sees (the screen: the cursors,
//!   an open book's words, what moved), what the ear hears (one's own voice, someone else's,
//!   silence), what the hand feels (its motion, the texture and edges under the cursor), and
//!   whether someone is there and attending, and smiling.
//! - **The world gives no rewards.** Nothing in `Senses` says good or bad. A smile is a sight
//!   like any other; that it is valued is the mind's (innate) business, as is valuing its own
//!   effect on what it senses.
//! - **Other people respond only to what they can observe:** a word said aloud (never a word
//!   meant or said inwardly), an act on the screen.
//!   - *The caregiver* (speech play) is there now and then; when there, they mostly answer a
//!     sound the child makes with the same sound.
//!   - *The teacher* (lessons) reads a story from a book, opening it on screen with their own
//!     cursor; pauses; asks "now read it" or "now tell it"; waits (showing how to open the book,
//!     on their own copy, if asked to); then either does the task or listens while the child does.
//!     Listening, they smile at a right word said aloud, and (if correcting) say the right word
//!     after a wrong one. If the child chimes in while the teacher is talking, a right next word
//!     gets a smile and the teacher goes on from there; a wrong one, the teacher says the word.
//! - **Scoring is not the world's job either:** each step leaves an `Event` record (what was said
//!   and by whom, what was wanted) for an observer outside both.
//!
//! The body's motor primitives (the syllables its brainstem can produce, the hand's basic
//! moves) and the receptors' innate tuning (a smile) are given to the mind at birth (`Body`):
//! that is the genome's part, not the world's.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::bitvec::BitVector;
use crate::program::motor::VocalTract;

/// Bits per sensory frame.
pub const BITS: usize = 2048;
/// Active bits per code.
pub const ACTIVE: usize = 32;

// ---- the stories ----

pub const NAMES: [(&str, &str); 6] = [("mary", "she"), ("sue", "she"), ("anna", "she"), ("john", "he"), ("tom", "he"), ("bob", "he")];
pub const PLACES: [&str; 8] = ["park", "farm", "river", "shop", "school", "garden", "forest", "beach"];
pub const ANIMALS: [&str; 8] = ["cat", "dog", "bird", "fox", "horse", "duck", "frog", "owl"];
pub const ADJS: [&str; 6] = ["happy", "hungry", "small", "big", "sleepy", "wet"];
pub const FOODS: [&str; 6] = ["apple", "bread", "cake", "fish", "seed", "carrot"];
pub const THINGS: [&str; 6] = ["ball", "hat", "box", "stick", "key", "shell"];
pub const OTHER: [&str; 16] = ["went", "to", "the", "a", ".", "saw", "was", "gave", "found", "played", "with", "then", "home", "now", "read", "tell"];
pub const IT: &str = "it";

/// The words the stories and the teacher use.
pub fn vocabulary() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = Vec::new();
    for (n, p) in NAMES {
        v.push(n);
        if !v.contains(&p) {
            v.push(p);
        }
    }
    for list in [&PLACES[..], &ANIMALS, &ADJS, &FOODS, &THINGS, &OTHER, &[IT]] {
        v.extend_from_slice(list);
    }
    v
}

/// A story: a name goes somewhere, sees an animal, one to three things happen, they go home.
pub fn story(rng: &mut StdRng) -> Vec<&'static str> {
    let (n, p) = NAMES[rng.gen_range(0..NAMES.len())];
    let place = PLACES[rng.gen_range(0..PLACES.len())];
    let animal = ANIMALS[rng.gen_range(0..ANIMALS.len())];
    let adj = ADJS[rng.gen_range(0..ADJS.len())];
    let food = FOODS[rng.gen_range(0..FOODS.len())];
    let thing = THINGS[rng.gen_range(0..THINGS.len())];
    let mut s = vec![n, "went", "to", "the", place, ".", p, "saw", "a", animal, "."];
    let mut middle: Vec<Vec<&str>> = vec![
        vec!["the", animal, "was", adj, "."],
        vec![n, "gave", "the", animal, "a", food, "."],
        vec![p, "found", "a", thing, "."],
        vec![p, "played", "with", "the", animal, "."],
    ];
    middle.shuffle(rng);
    for m in middle.into_iter().take(rng.gen_range(1..=3)) {
        s.extend(m);
    }
    s.extend(["then", n, "went", "home", "."]);
    s
}

fn codes(n: usize, active: usize, rng: &mut StdRng) -> Vec<BitVector> {
    let all: Vec<usize> = (0..BITS).collect();
    (0..n).map(|_| BitVector::from_bits(&all.choose_multiple(rng, active).copied().collect::<Vec<_>>(), BITS)).collect()
}

fn or_into(v: &mut BitVector, x: &BitVector) {
    for (w, y) in v.as_words_mut().iter_mut().zip(x.as_words()) {
        *w |= y;
    }
}

fn zero() -> BitVector {
    BitVector::new(BITS, Some(0))
}

/// What the senses report this step.
#[derive(Clone)]
pub struct Senses {
    pub eye: BitVector,
    pub ear: BitVector,
    pub touch: BitVector,
    pub presence: BitVector,
}

/// What the mind does this step: a command to each effector, or none.
#[derive(Clone, Default)]
pub struct Acts {
    pub voice: Option<BitVector>,
    pub hand: Option<BitVector>,
}

/// What the genome gives the mind: the body's motor primitives (commands its pattern
/// generators can produce spontaneously) and innately tuned receptors.
pub struct Body {
    pub voice: Vec<BitVector>,
    pub hand: Vec<BitVector>,
    /// a smile, as the presence sense reports it (an innately recognised face pattern)
    pub smile: BitVector,
}

/// The hand's moves.
pub const LEFT: usize = 0;
pub const RIGHT: usize = 1;
pub const CLICK: usize = 2;
const MOVES: usize = 3;

/// Who is speaking or acting, for the observer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Who {
    Child,
    Teacher,
    Caregiver,
}

/// The part of a lesson the step belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    SpeechPlay,
    MousePlay,
    Story,
    Instruction,
    Wait,
    /// the task, with the child doing it (`true`) or the teacher
    Task(bool),
    Pause,
    Conditioning,
}

/// A conditioning scene's kind (a caregiver is there throughout; a smile is the outcome).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Conditioning {
    /// a tone (CS+, half the trials) is followed two steps later by a smile; another tone (CS−)
    /// by nothing; nothing to do
    Pavlov,
    /// a light is on for three steps (half the trials); a click while it is on, or the step
    /// after, gets a smile; clicks at other times get nothing
    Instrumental,
    /// a light is on for one step; a click three to five steps later gets a smile
    Delayed,
}

/// A record of one step, for an observer (never sent to the mind).
#[derive(Clone, Debug)]
pub struct Event {
    pub phase: Phase,
    /// the word wanted now (in a task done by the child)
    pub want: Option<usize>,
    /// words said aloud this step, and by whom
    pub said: Vec<(Who, usize)>,
    /// the child's hand move, if any happened
    pub moved: Option<usize>,
    /// a book opened by the child
    pub opened: bool,
    /// the teacher smiled
    pub smiled: bool,
    /// in conditioning: (the trial's step, a cue trial: the CS+ or the light)
    pub cond: Option<(usize, bool)>,
}

/// What the book on the desk is in a lesson.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Book {
    /// the story just heard
    Same,
    /// another story
    Other,
    /// no book
    Closed,
}

/// The kind of task a lesson asks for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Task {
    Read,
    Tell,
}

/// A lesson's settings.
#[derive(Clone, Copy, Debug)]
pub struct Lesson {
    pub task: Task,
    pub book: Book,
    /// the child does the task (else the teacher does)
    pub child_does: bool,
    /// the teacher shows how to open the book while waiting
    pub demonstrate: bool,
    /// the teacher says the right word after a wrong one
    pub correct: bool,
}

enum Scene {
    Idle,
    /// a caregiver comes and goes; steps left
    SpeechPlay { left: usize, caregiver: bool },
    /// books on some places; steps left
    MousePlay { left: usize },
    Lesson(LessonState),
    /// (kind, trials left, step in the trial, a cue trial, smiled this trial)
    Conditioning(Conditioning, usize, usize, bool, bool),
}

struct LessonState {
    lesson: Lesson,
    /// the teacher's script: (phase, word or silence) still to come, front first
    queue: std::collections::VecDeque<(Phase, Option<usize>)>,
    /// the words of the task (what is read or told)
    target: Vec<usize>,
    /// the wait's steps left, and where the teacher's cursor is (once shown)
    wait_left: usize,
    tcur: Option<usize>,
    tdone: bool,
    /// the task's index
    t: usize,
    /// the teacher owes a word next step (a correction or a word the child talked over)
    owed: Option<usize>,
    smile_next: bool,
    started_task: bool,
    steps: usize,
}

/// The world.
pub struct World {
    rng: StdRng,
    pub vocab: Vec<&'static str>,
    /// how each word sounds (and silence, after the words)
    sounds: Vec<BitVector>,
    glyphs: Vec<BitVector>,
    cursor_seen: Vec<BitVector>,
    teacher_seen: Vec<BitVector>,
    motion_seen: Vec<BitVector>,
    motion_felt: Vec<BitVector>,
    modes: Vec<BitVector>,
    edges: Vec<BitVector>,
    attending: BitVector,
    smile: BitVector,
    /// conditioning cues: two tones (heard) and a light (seen)
    tones: Vec<BitVector>,
    light: BitVector,
    tract: VocalTract,
    mouse: VocalTract,
    /// the screen: what lies at each place (none, or a book's words), and the child's cursor
    places: Vec<Option<Vec<usize>>>,
    cursor: usize,
    /// a book open in view: its words and the next one to be seen
    page: Option<(Vec<usize>, usize)>,
    /// what moved on screen at the last step, felt at the last step (the senses lag the act)
    seen_motion: Option<usize>,
    felt: Option<usize>,
    present: bool,
    /// a word of the teacher's book in view this step
    glyph_now: Option<usize>,
    /// a conditioning cue sounding or in view this step, and a smile owed next step
    extra_sound: Option<BitVector>,
    extra_sight: Option<BitVector>,
    smile_next: bool,
    smile_pending: bool,
    scene: Scene,
    pub events: Vec<Event>,
}

impl World {
    pub fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let vocab = vocabulary();
        let n = vocab.len();
        let mut r = |salt: u64| StdRng::seed_from_u64(seed ^ salt);
        let sounds = codes(n + 1, ACTIVE, &mut r(0x50D5));
        let glyphs = codes(n, ACTIVE, &mut r(0x6171));
        let npos = 5;
        let cursor_seen = codes(npos, ACTIVE, &mut r(0xC0C5));
        let teacher_seen = codes(npos, ACTIVE, &mut r(0x7EAC));
        let motion_seen = codes(MOVES, ACTIVE, &mut r(0x5EE5));
        let motion_felt = codes(MOVES, ACTIVE, &mut r(0xFE17));
        let modes = codes(2, ACTIVE / 2, &mut r(0x7E47));
        let edges = codes(2, ACTIVE / 4, &mut r(0xED6E));
        let face = codes(2, ACTIVE, &mut r(0xFACE));
        let tract = VocalTract::new(n + 1, BITS, ACTIVE, &mut r(0x7AC7));
        let mouse = VocalTract::new(MOVES, BITS, ACTIVE, &mut r(0x40B5));
        let cursor = rng.gen_range(0..npos);
        Self {
            rng,
            vocab,
            sounds,
            glyphs,
            cursor_seen,
            teacher_seen,
            motion_seen,
            motion_felt,
            modes,
            edges,
            attending: face[0].clone(),
            smile: face[1].clone(),
            tones: codes(2, ACTIVE, &mut r(0x7070)),
            light: codes(1, ACTIVE, &mut r(0x1167)).remove(0),
            tract,
            mouse,
            places: vec![None; npos],
            cursor,
            page: None,
            seen_motion: None,
            felt: None,
            present: false,
            glyph_now: None,
            extra_sound: None,
            extra_sight: None,
            smile_next: false,
            smile_pending: false,
            scene: Scene::Idle,
            events: Vec::new(),
        }
    }

    /// The body and the receptors' innate tuning, for the mind at birth.
    pub fn body(&self) -> Body {
        Body {
            voice: (0..self.tract.len()).map(|w| self.tract.command(w).clone()).collect(),
            hand: (0..MOVES).map(|m| self.mouse.command(m).clone()).collect(),
            smile: self.smile.clone(),
        }
    }

    pub fn id(&self, w: &str) -> usize {
        self.vocab.iter().position(|v| *v == w).unwrap()
    }

    /// The index of silence among the sounds (the vocal tract can make it: closing the mouth).
    pub fn quiet(&self) -> usize {
        self.vocab.len()
    }

    /// The scene is over (a new one may begin).
    pub fn done(&self) -> bool {
        matches!(self.scene, Scene::Idle)
    }

    /// Speech play: `steps` steps; a caregiver comes or goes every 8, there half the time.
    pub fn speech_play(&mut self, steps: usize) {
        self.reset_screen(false);
        self.scene = Scene::SpeechPlay { left: steps, caregiver: false };
    }

    /// Mouse play: books lie on some places (at least one); 12 steps, or until one is opened and
    /// its first words seen.
    pub fn mouse_play(&mut self) {
        self.reset_screen(false);
        let n = self.places.len();
        for p in 0..n {
            if self.rng.gen_bool(0.5) {
                self.places[p] = Some(self.text());
            }
        }
        if self.places.iter().all(|p| p.is_none()) {
            let k = self.rng.gen_range(0..n);
            self.places[k] = Some(self.text());
        }
        self.cursor = self.rng.gen_range(0..n);
        self.present = false;
        self.scene = Scene::MousePlay { left: 12 };
    }

    /// Conditioning: `trials` trials of 10 steps, a caregiver there throughout.
    pub fn conditioning(&mut self, kind: Conditioning, trials: usize) {
        self.reset_screen(false);
        self.present = true;
        let cue = self.rng.gen_bool(0.5);
        self.scene = Scene::Conditioning(kind, trials, 0, cue, false);
    }

    /// A lesson: the teacher reads a story, asks for it to be read or told, waits, and the task
    /// is done (by the teacher or the child).
    pub fn lesson(&mut self, lesson: Lesson) {
        self.reset_screen(true);
        let s = self.text();
        // the desk (place 0) holds the book asked about, the other places other books
        let p = match lesson.book {
            Book::Same => Some(s.clone()),
            Book::Other => Some(self.text()),
            Book::Closed => None,
        };
        self.places[0] = p.clone();
        for k in 1..self.places.len() {
            self.places[k] = Some(self.text());
        }
        let mut queue = std::collections::VecDeque::new();
        // the teacher opens their book (seen: their cursor's click), then reads it aloud
        queue.push_back((Phase::Story, None));
        for &w in &s {
            queue.push_back((Phase::Story, Some(w)));
        }
        queue.push_back((Phase::Pause, None));
        let verb = match lesson.task {
            Task::Read => "read",
            Task::Tell => "tell",
        };
        for w in ["now", verb, IT] {
            queue.push_back((Phase::Instruction, Some(self.id(w))));
        }
        let target = match lesson.task {
            Task::Read => p.clone().unwrap_or_default(),
            Task::Tell => s.clone(),
        };
        // the wait: long enough, with a book to open, for the teacher to show how from anywhere
        let wait_left = if lesson.task == Task::Read { 3 + self.places.len() - 1 } else { 3 };
        self.present = true;
        self.scene = Scene::Lesson(LessonState { lesson, queue, target, wait_left, tcur: None, tdone: false, t: 0, owed: None, smile_next: false, started_task: false, steps: 0 });
    }

    /// A story's words, as indices.
    fn text(&mut self) -> Vec<usize> {
        let s = story(&mut self.rng);
        s.iter().map(|w| self.id(w)).collect()
    }

    fn reset_screen(&mut self, keep_cursor: bool) {
        for p in self.places.iter_mut() {
            *p = None;
        }
        if !keep_cursor {
            self.cursor = 0;
        }
        self.page = None;
        self.seen_motion = None;
        self.felt = None;
    }

    /// The word the lesson's target wants now.
    fn sound(&self, w: Option<usize>) -> BitVector {
        w.map_or_else(|| self.sounds[self.vocab.len()].clone(), |w| self.sounds[w].clone())
    }

    /// One step: the mind's acts are carried out, the people respond, and the senses report.
    pub fn step(&mut self, acts: &Acts) -> Senses {
        let mut ev = Event { phase: Phase::Pause, want: None, said: Vec::new(), moved: None, opened: false, smiled: false, cond: None };
        // the body: the voice says a word (or silence, or nothing for a garbled command)
        let said = acts.voice.as_ref().and_then(|c| self.tract.articulate(c));
        let child_word = said.filter(|&w| w < self.vocab.len());
        if let Some(w) = child_word {
            ev.said.push((Who::Child, w));
        }
        // the hand
        let moved = acts.hand.as_ref().and_then(|c| self.mouse.articulate(c));
        let mut opened = false;
        let mut blocked = false;
        let npos = self.places.len();
        match moved {
            Some(LEFT) if self.cursor > 0 => self.cursor -= 1,
            Some(RIGHT) if self.cursor + 1 < npos => self.cursor += 1,
            Some(CLICK) if self.places[self.cursor].is_some() => {
                let words = self.places[self.cursor].clone().unwrap();
                self.page = Some((words, 0));
                opened = true;
            }
            Some(_) => blocked = true,
            None => {}
        }
        ev.moved = moved;
        ev.opened = opened;
        // what is heard this step: one's own voice, else what someone says
        let mut heard: Option<usize> = said;
        let mut eye = zero();
        let mut teacher_motion: Option<usize> = None;
        let mut smile = false;
        // the people
        let scene = std::mem::replace(&mut self.scene, Scene::Idle);
        self.scene = match scene {
            Scene::Idle => Scene::Idle,
            Scene::SpeechPlay { left, mut caregiver } => {
                ev.phase = Phase::SpeechPlay;
                if left % 8 == 0 {
                    caregiver = self.rng.gen_bool(0.5);
                    self.present = caregiver;
                }
                // the caregiver answers a sound made aloud, with the same sound, most of the time
                // (heard the step after: the child's own voice is heard now)
                if heard.is_none() && caregiver {
                    if let Some(w) = self.events.last().and_then(|e| e.said.iter().find(|s| s.0 == Who::Child).map(|s| s.1)) {
                        if self.rng.gen_bool(0.7) {
                            heard = Some(w);
                            ev.said.push((Who::Caregiver, w));
                        }
                    }
                }
                if left > 1 {
                    Scene::SpeechPlay { left: left - 1, caregiver }
                } else {
                    self.present = false;
                    Scene::Idle
                }
            }
            Scene::MousePlay { left } => {
                ev.phase = Phase::MousePlay;
                if left > 1 && !(self.page.as_ref().map_or(false, |p| p.1 >= 3)) {
                    Scene::MousePlay { left: left - 1 }
                } else {
                    self.page = None;
                    Scene::Idle
                }
            }
            Scene::Conditioning(kind, left, t, cue, mut smiled) => {
                ev.phase = Phase::Conditioning;
                ev.cond = Some((t, cue));
                let click = moved == Some(CLICK);
                match kind {
                    Conditioning::Pavlov => {
                        if t == 2 {
                            self.extra_sound = Some(self.tones[if cue { 0 } else { 1 }].clone());
                        }
                        if t == 4 && cue {
                            smile = true;
                        }
                    }
                    Conditioning::Instrumental => {
                        if cue && (2..=4).contains(&t) {
                            self.extra_sight = Some(self.light.clone());
                        }
                        if cue && click && (3..=5).contains(&t) && !smiled {
                            self.smile_next = true;
                            smiled = true;
                        }
                    }
                    Conditioning::Delayed => {
                        if cue && t == 2 {
                            self.extra_sight = Some(self.light.clone());
                        }
                        if cue && click && (5..=7).contains(&t) && !smiled {
                            self.smile_next = true;
                            smiled = true;
                        }
                    }
                }
                if t + 1 < 10 {
                    Scene::Conditioning(kind, left, t + 1, cue, smiled)
                } else if left > 1 {
                    let cue = self.rng.gen_bool(0.5);
                    Scene::Conditioning(kind, left - 1, 0, cue, false)
                } else {
                    self.present = false;
                    Scene::Idle
                }
            }
            Scene::Lesson(mut l) => {
                let done = self.lesson_step(&mut l, child_word, &mut heard, &mut ev, &mut teacher_motion, &mut smile);
                if done {
                    self.present = false;
                    self.page = None;
                    Scene::Idle
                } else {
                    Scene::Lesson(l)
                }
            }
        };
        if std::mem::take(&mut self.smile_pending) {
            smile = true;
        }
        if std::mem::take(&mut self.smile_next) {
            self.smile_pending = true;
        }
        ev.smiled = smile;
        // the eye: the screen (the cursors, what moved, an open page's next word)
        or_into(&mut eye, &self.cursor_seen[self.cursor]);
        if let Scene::Lesson(l) = &self.scene {
            if let Some(t) = l.tcur {
                or_into(&mut eye, &self.teacher_seen[t]);
            }
        }
        if let Some(m) = self.seen_motion.take() {
            or_into(&mut eye, &self.motion_seen[m]);
        }
        if let Some(w) = self.glyph_now.take() {
            or_into(&mut eye, &self.glyphs[w]);
        }
        if let Some(l) = self.extra_sight.take() {
            or_into(&mut eye, &l);
        }
        if let Some((words, k)) = self.page.as_mut() {
            if let Some(&w) = words.get(*k) {
                or_into(&mut eye, &self.glyphs[w]);
            }
            *k += 1;
        }
        // what moved this step is seen at the next (an act that did nothing moved nothing)
        self.seen_motion = if blocked { None } else { moved.or(teacher_motion) };
        // touch: the hand's own motion (if it moved anything), the texture under the cursor (a
        // book vibrates), and an edge where a neighbour differs
        let mut touch = zero();
        if let (Some(m), false) = (moved, blocked) {
            or_into(&mut touch, &self.motion_felt[m]);
        }
        let kind = |p: usize| self.places[p].is_some() as usize;
        if kind(self.cursor) > 0 {
            or_into(&mut touch, &self.modes[1]);
        }
        if self.cursor > 0 && kind(self.cursor - 1) != kind(self.cursor) {
            or_into(&mut touch, &self.edges[0]);
        }
        if self.cursor + 1 < npos && kind(self.cursor + 1) != kind(self.cursor) {
            or_into(&mut touch, &self.edges[1]);
        }
        let mut presence = zero();
        if self.present {
            or_into(&mut presence, &self.attending);
        }
        if smile {
            or_into(&mut presence, &self.smile);
        }
        self.felt = moved;
        self.events.push(ev);
        let mut ear = self.sound(heard);
        if let (Some(t), None) = (self.extra_sound.take(), heard) {
            ear = t;
        }
        Senses { eye, ear, touch, presence }
    }

    /// The teacher's part of a lesson step. Returns whether the lesson is over.
    fn lesson_step(&mut self, l: &mut LessonState, child_word: Option<usize>, heard: &mut Option<usize>, ev: &mut Event, teacher_motion: &mut Option<usize>, smile: &mut bool) -> bool {
        if std::mem::take(&mut l.smile_next) {
            *smile = true;
        }
        // a word the teacher owes (said after the child's turn), if the child is quiet now
        // (if the child talks over it, the teacher lets it go)
        if let Some(w) = l.owed.take() {
            if heard.is_none() {
                *heard = Some(w);
                ev.said.push((Who::Teacher, w));
            }
            ev.phase = l.queue.front().map_or(Phase::Task(l.lesson.child_does), |x| x.0);
            return false;
        }
        // a lesson has an end, however it goes
        l.steps += 1;
        if l.steps > 200 {
            return true;
        }
        // the script: the story, the pause, the instruction
        if let Some((phase, w)) = l.queue.front().copied() {
            ev.phase = phase;
            match (child_word, w) {
                // the child chimes in: a right next word is smiled at and the teacher goes on; a
                // wrong one, the teacher says the word after it
                (Some(c), Some(w)) => {
                    l.queue.pop_front();
                    if c == w {
                        l.smile_next = true;
                    } else {
                        l.owed = Some(w);
                    }
                }
                (Some(_), None) => {
                    l.queue.pop_front();
                }
                (None, w) => {
                    l.queue.pop_front();
                    match w {
                        Some(w) => {
                            *heard = Some(w);
                            ev.said.push((Who::Teacher, w));
                        }
                        None if phase == Phase::Story => {
                            // the teacher opens their book: their cursor clicks it (seen)
                            *teacher_motion = Some(CLICK);
                            l.tcur = Some(0);
                        }
                        None => {}
                    }
                    if phase == Phase::Story && w.is_some() {
                        // the teacher's book is on the screen: the word read is seen
                        self.glyph_now = w;
                    }
                }
            }
            if phase == Phase::Instruction && l.queue.front().map_or(true, |x| x.0 != Phase::Instruction) {
                // the teacher's cursor is put away; the wait begins
                l.tcur = None;
            }
            return false;
        }
        // the wait: the child may use the mouse; the teacher shows how if asked
        if l.wait_left > 0 && self.page.is_none() {
            ev.phase = Phase::Wait;
            l.wait_left -= 1;
            if l.lesson.demonstrate && !l.tdone && l.lesson.task == Task::Read && l.lesson.book != Book::Closed {
                let from = l.tcur.unwrap_or(self.cursor);
                if l.tcur.is_some() || l.wait_left < from + 1 {
                    let m = if from > 0 { LEFT } else { CLICK };
                    let t = l.tcur.get_or_insert(from);
                    match m {
                        LEFT => *t -= 1,
                        _ => l.tdone = true,
                    }
                    *teacher_motion = Some(m);
                }
            }
            return false;
        }
        if !l.started_task {
            l.started_task = true;
            l.tcur = None;
        }
        // the task
        ev.phase = Phase::Task(l.lesson.child_does);
        let Some(&want) = l.target.get(l.t) else {
            // done: the teacher falls silent, and the lesson ends
            ev.phase = Phase::Pause;
            return true;
        };
        l.t += 1;
        ev.want = Some(want);
        if l.lesson.child_does {
            match child_word {
                Some(c) if c == want => l.smile_next = true,
                Some(_) if l.lesson.correct => l.owed = Some(want),
                _ => {}
            }
        } else {
            match child_word {
                Some(c) if c == want => l.smile_next = true,
                Some(_) => l.owed = Some(want),
                None => {
                    *heard = Some(want);
                    ev.said.push((Who::Teacher, want));
                }
            }
        }
        false
    }
}
