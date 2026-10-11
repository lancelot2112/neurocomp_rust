//! Reading aloud and retelling, learned from a teacher's examples, chosen by a spoken
//! instruction.
//!
//! Each trial: the teacher tells a short story (S) aloud. Then a book is opened, at the same
//! story or at another one (P), or stays closed, and the teacher says "now read it" or "now
//! tell it". The teacher then does it, word by word: reading says the book's words (P),
//! telling says the heard story (S) whatever the book shows. The network hears and sees
//! everything the child would and learns, at each word, to plan the word the teacher says.
//! Nothing says which source to use: when the book shows another story, the eye and the
//! memory propose different words, and only the instruction tells which one is wanted.
//!
//! At test (new stories, cortex learning off), the network does it itself: it plans, says the
//! word through the learned motor path, and hears itself.
//!
//! Everything goes through the network; no word reaches speech except through the cortex:
//! - **Eye and ear** are separate senses with their own codes: the visual word at the fixation
//!   (the book's word t, or nothing when it is closed) and the heard word (the last word said).
//!   Each has its own layer 4 (learned k-winners-take-all).
//! - **The hippocampus** stores the story as it is heard: each word (content) with the word
//!   before it and the story's context (context). Cued by the last word said and the context,
//!   it recalls the next word, and the recall returns to the cortex as an input (EC → cortex).
//! - **The prefrontal cortex** holds what a basal-ganglia gate loads from what is heard
//!   (`PfcGate`, after PBWM). Its credit is whether the words spoken while it held that item
//!   were right, so it learns to hold the instruction word. What it holds reaches the cells'
//!   tuft (layer 1) as context and the thalamic relay's context.
//! - **Recurrent layer 2/3** (bitwise primed cells): basal input is the eye's layer 4, the
//!   ear's layer 4, the hippocampal recall and its own previous activity; its tuft gets the
//!   prefrontal content and the cerebellum's output. A cell whose tuft matches bursts and
//!   wins over one that only spikes.
//! - **Layer 5**, **the cerebellar circuit** (mossy fibres: the ear and the pons) and **the
//!   thalamic relay** as in the prose driver; the plan is the relay's strongest 32 bits.
//! - **Speaking:** the motor area's inverse model (learned by babbling) turns the plan into a
//!   command; the vocal tract says a word; the network hears it.
//!
//! Options: RT_TRAIN (trials, default 3000), RT_TEST (default 300), RT_GATE=fixed (diagnosis:
//! the instruction word is always loaded, no gate), RT_TELL=apart (the teacher never retells
//! with the book open at the same story), RT_PRACTICE (share of training tasks done by the
//! network in its own voice, the teacher's word the target; default 0), RT_HC_TIME=1 (a time
//! code in the hippocampal cue), RT_MAKE=1 (a third instruction, "now make one": the teacher
//! makes up a new story; at test the network speaks freely, with RT_TEMP noise on the relay,
//! default 30, and each invented story is judged for grammar, coherence and novelty),
//! RT_TRACE, SEED.

mod common;

use common::Encoder;
use neurocomp::bitvec::BitVector;
use neurocomp::fixed::ONE;
use neurocomp::program::area::{Brain, Kind, Motor};
use neurocomp::program::column::Column;
use neurocomp::program::{
    BitCells, BoundaryCell, CerebellarCircuit, Driver, EpisodicCircuit, Hippocampus, HippocampusConfig, IndexConfig, IndexMemory, Layer4, MotorArea,
    MaskStriatum, Striatum, ThalamicRelay, VocalTract, WorkingMemory,
};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};

const BITS: usize = 2048;
const ACTIVE: usize = 32;
/// The hippocampus's context space (after the BITS content bits).
const CTX_SPACE: usize = 65_536;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn frames(fs: &[&BitVector]) -> BitVector {
    let mut w = Vec::new();
    for f in fs {
        w.extend_from_slice(f.as_words());
    }
    BitVector::from_words(w)
}

fn overlap(a: &BitVector, b: &BitVector) -> u32 {
    a.as_words().iter().zip(b.as_words()).map(|(x, y)| (x & y).count_ones()).sum()
}

fn ones(v: &BitVector) -> Vec<usize> {
    let mut out = Vec::new();
    for (wi, &x) in v.as_words().iter().enumerate() {
        let mut x = x;
        while x != 0 {
            out.push(wi * 64 + x.trailing_zeros() as usize);
            x &= x - 1;
        }
    }
    out
}

fn zero() -> BitVector {
    BitVector::new(BITS, Some(0))
}

/// A fixed projection of cells into one frame (8 bits per cell): the recurrent frame and the pons.
fn project(cells: &[usize]) -> BitVector {
    let mut v = zero();
    for &c in cells {
        for i in 0..8u64 {
            let mut h = (c as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            h ^= h >> 31;
            v.bit_set((h % BITS as u64) as usize);
        }
    }
    v
}

// ---- the stories ----

const NAMES: [(&str, &str); 6] = [("mary", "she"), ("sue", "she"), ("anna", "she"), ("john", "he"), ("tom", "he"), ("bob", "he")];
const PLACES: [&str; 8] = ["park", "farm", "river", "shop", "school", "garden", "forest", "beach"];
const ANIMALS: [&str; 8] = ["cat", "dog", "bird", "fox", "horse", "duck", "frog", "owl"];
const ADJS: [&str; 6] = ["happy", "hungry", "small", "big", "sleepy", "wet"];
const FOODS: [&str; 6] = ["apple", "bread", "cake", "fish", "seed", "carrot"];
const THINGS: [&str; 6] = ["ball", "hat", "box", "stick", "key", "shell"];
const OTHER: [&str; 16] =
    ["went", "to", "the", "a", ".", "saw", "was", "gave", "found", "played", "with", "then", "home", "now", "read", "tell"];
const IT: &str = "it";

/// The words; with `make`, also "make" and "one" (the instruction "now make one").
fn vocabulary(make: bool, find: bool) -> Vec<&'static str> {
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
    if make {
        v.extend_from_slice(&["make", "one"]);
    }
    if find {
        v.push("find");
    }
    v
}

/// A story's animal (the one it is about: "... saw a <animal> .").
fn animal_of(s: &[&'static str]) -> &'static str {
    s[9]
}

/// A new book for the shelf, about an animal no other book there is about.
fn shelf_book(rng: &mut StdRng, shelf: &[Vec<&'static str>], skip: usize) -> Vec<&'static str> {
    loop {
        let b = story(rng);
        if !shelf.iter().enumerate().any(|(i, o)| i != skip && animal_of(o) == animal_of(&b)) {
            return b;
        }
    }
}

/// A sentence's form, if it is one the stories use: (form, name, pronoun, animal).
fn sentence_form<'a>(w: &[&'a str]) -> Option<(usize, Option<&'a str>, Option<&'a str>, Option<&'a str>)> {
    let is = |x: &str, list: &[&str]| list.contains(&x);
    let name = |x: &str| NAMES.iter().any(|n| n.0 == x);
    let pron = |x: &str| x == "he" || x == "she";
    match w {
        [n, "went", "to", "the", pl] if name(n) && is(pl, &PLACES) => Some((0, Some(*n), None, None)),
        [p, "saw", "a", a] if pron(p) && is(a, &ANIMALS) => Some((1, None, Some(*p), Some(*a))),
        ["the", a, "was", j] if is(a, &ANIMALS) && is(j, &ADJS) => Some((2, None, None, Some(*a))),
        [n, "gave", "the", a, "a", f] if name(n) && is(a, &ANIMALS) && is(f, &FOODS) => Some((3, Some(*n), None, Some(*a))),
        [p, "found", "a", t] if pron(p) && is(t, &THINGS) => Some((4, None, Some(*p), None)),
        [p, "played", "with", "the", a] if pron(p) && is(a, &ANIMALS) => Some((5, None, Some(*p), Some(*a))),
        ["then", n, "went", "home"] if name(n) => Some((6, Some(*n), None, None)),
        _ => None,
    }
}

/// Judge an invented story: (sentences, grammatical sentences, starts and ends as a story,
/// coherent: one name, its pronoun, one animal).
fn judge(words: &[&str]) -> (usize, usize, bool, bool) {
    let sentences: Vec<&[&str]> = words.split(|w| *w == ".").filter(|x| !x.is_empty()).collect();
    let forms: Vec<_> = sentences.iter().map(|x| sentence_form(x)).collect();
    let ok = forms.iter().filter(|f| f.is_some()).count();
    let framed = forms.first().map_or(false, |f| matches!(f, Some((0, ..)))) && forms.last().map_or(false, |f| matches!(f, Some((6, ..)))) && words.last() == Some(&".");
    let names: std::collections::BTreeSet<&str> = forms.iter().flatten().filter_map(|f| f.1).collect();
    let prons: std::collections::BTreeSet<&str> = forms.iter().flatten().filter_map(|f| f.2).collect();
    let animals: std::collections::BTreeSet<&str> = forms.iter().flatten().filter_map(|f| f.3).collect();
    let coherent = names.len() == 1
        && prons.iter().all(|p| NAMES.iter().any(|n| names.contains(n.0) && n.1 == *p))
        && animals.len() <= 1;
    (sentences.len(), ok, framed, coherent)
}

fn story(rng: &mut StdRng) -> Vec<&'static str> {
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

// ---- the network ----

struct Net {
    l4e: Layer4,
    l4a: Layer4,
    /// the association area: eye and ear converge (`assoc_of`); what the index points to
    assoc: Layer4,
    l23: BitCells,
    l5: BitCells,
    cb: CerebellarCircuit,
    relay: ThalamicRelay,
    rec23: BitVector,
    pons: BitVector,
    wm: WorkingMemory,
    hc: Hippocampus,
    /// RT_AREAS=1: each device has its own area (eye, ear, hand), and association areas grow
    /// by conjunction; the hand's felt motion is its own input frame, not the ear's
    brain: Option<Brain>,
    /// what the hand feels now (its own motion), for its area
    felt: BitVector,
    /// RT_TOP=1: the top association column replaces the shared layers 2/3 and 5. Its input is
    /// the scene (the association of every area) and the hippocampus's return; its tuft the
    /// prefrontal content and the cerebellum's output; its layer 2/3 predicts the next scene,
    /// its layer 5 sends the goal (the meant sound or act) to the motor columns.
    top: Option<Column>,
    /// RT_STREAMS=1: the streams replace the one top column (see `Streams`)
    streams: Option<Streams>,
    /// what the senses send up: their layer 4 code and layer 2/3 activity (RT_FF=l4: the code)
    ff_l4: bool,
    /// self-organizing (the default; RT_SELFORG=0 off): no streams wired: every association column grown by co-activity has
    /// layers 2/3 and 5, and every column's layer 5 offers goals to every motor area; which
    /// column drives which is learned (the relay's reliability per source and context), and a
    /// column learns only where its input is active
    selforg: bool,
    /// self-organizing, diagnosis: per association column, (goals offered, right) for sounds
    /// and for mouse acts
    goal_hits: Vec<[(u64, u64); 2]>,
    /// the target being learned is a mouse act's result (set by `learn_motor`)
    learning_motion: bool,
}

/// The hierarchy as streams (RT_STREAMS=1), after the cortex's:
/// - **ventral** (word forms, the story): the association of eye and ear (a word seen and
///   heard) and the hippocampus's return; its layer 2/3 predicts the next word assembly, which
///   is sent down to the ear as the sound expected (a source of the plan), and its activity is
///   feedback on the dorsal columns' tufts;
/// - **dorsal auditory–motor** (as the area at the parietal–temporal junction and Broca's): what
///   is heard and the word scene; its layer 5 sends the meant sound to the voice;
/// - **dorsal visual–motor** (parietal–premotor): what is seen and felt; its layer 5 proposes the
///   mouse's acts.
/// Both dorsal layer 5s reach the basal ganglia and the pons.
struct Streams {
    ventral: Column,
    audio_motor: Column,
    visuo_motor: Column,
}

const EYE: usize = 0;
const EAR: usize = 1;
const HAND: usize = 2;
const VOICE: usize = 3;
/// seen actions (the superior temporal sulcus's biological-motion stream): what an act looks
/// like, wherever it happens
const ACTS: usize = 4;

struct Step {
    /// with the top column: (scene, tuft, layer 2/3's expected scene), for layer 5 to learn from
    top_in: Option<(BitVector, BitVector, BitVector)>,
    /// with streams: (auditory–motor input, visual–motor input, tuft, their layer 2/3s' expectations)
    streams_in: Option<(BitVector, BitVector, BitVector, BitVector, BitVector)>,
    /// self-organizing: (per association column: its input and expectation; the tuft; per
    /// sensory column: (area, its code, its expectation))
    selforg_in: Option<(Vec<(BitVector, BitVector)>, BitVector, Vec<(usize, BitVector, BitVector)>)>,
    l23_row: BitVector,
    l5_row: BitVector,
    mossy: BitVector,
    ctx: BitVector,
    props: Vec<(u8, Driver, BitVector)>,
    plan: BitVector,
}

impl Net {
    fn new(seed: u64) -> Self {
        let mk = |cells: usize, tail: usize| {
            let mut l = BitCells::new(BITS / 64, 16, cells);
            l.set_meta(true);
            l.set_tag_capture(true);
            l.set_interneurons(true);
            l.set_basal_tail(tail);
            l.set_clustered(std::env::var("RT_CLUSTER").map_or(true, |v| v != "0"));
            l
        };
        // RT_L4_SETTLE=0: layer 4 keeps learning at full rate (default: it settles with experience)
        let l4 = |sd: u64| {
            let mut l = Layer4::new(BITS, BITS, 16, ACTIVE, sd);
            l.set_settling(std::env::var("RT_L4_SETTLE").map_or(true, |v| v != "0"));
            l
        };
        // RT_COLUMNS=1: areas (implied) whose motor sides and association areas are columns;
        // RT_TOP=1 (implies columns): the senses are columns too, and the top association column
        // replaces the shared layers
        let top = std::env::var("RT_TOP").map_or(false, |v| v == "1");
        let streams = std::env::var("RT_STREAMS").map_or(false, |v| v == "1");
        // (the default; RT_SELFORG=0: the shared layers, or what RT_TOP / RT_STREAMS / RT_AREAS ask)
        let selforg = !top && !streams && std::env::var("RT_SELFORG").map_or(true, |v| v != "0") && std::env::var("RT_AREAS").map_or(true, |v| v != "0");
        let columns = top || streams || selforg || std::env::var("RT_COLUMNS").map_or(false, |v| v == "1");
        let areas = columns || std::env::var("RT_AREAS").map_or(false, |v| v == "1");
        let mut cb = CerebellarCircuit::new(131_072, BITS, ONE / 256, seed.wrapping_add(77));
        cb.set_rates(2, 0);
        cb.set_cross_frames(true);
        Self {
            l4e: l4(seed.wrapping_add(98)),
            l4a: l4(seed.wrapping_add(99)),
            assoc: {
                let mut l = Layer4::new(2 * BITS, BITS, 16, ACTIVE, seed.wrapping_add(97));
                // eye and ear are two pathways: an absent sense does not unlearn its synapses
                l.set_pathways(BITS);
                l.set_settling(std::env::var("RT_L4_SETTLE").map_or(true, |v| v != "0"));
                l
            },
            // [eye L4 | PFC (tuft) | cerebellum (tuft) | ear L4 | recall | own previous activity]
            // (with areas, a last basal frame: the hand's area)
            l23: mk(16384, if areas { 4 } else { 3 }),
            // [eye L4 | PFC (tuft) | cerebellum (tuft) | ear L4 | layer 2/3's prediction (| hand)]
            l5: mk(8192, if areas { 3 } else { 2 }),
            brain: areas.then(|| {
                let mut b = Brain::new(BITS, ACTIVE, seed.wrapping_add(500));
                b.columns = columns;
                b.selforg = selforg;
                b.attach("eye", Kind::Sensor);
                b.attach("ear", Kind::Sensor);
                b.attach("hand", Kind::SensorMotor);
                // the voice: a motor area (it is heard through the ear)
                b.attach("voice", Kind::Motor);
                b.attach("seen acts", Kind::Sensor);
                b
            }),
            felt: zero(),
            top: (top && !streams).then(|| Column::agranular(BITS, 16384)),
            streams: streams.then(|| Streams { ventral: Column::agranular(BITS, 16384), audio_motor: Column::agranular(BITS, 16384), visuo_motor: Column::agranular(BITS, 8192) }),
            ff_l4: std::env::var("RT_FF").map_or(false, |v| v == "l4"),
            selforg,
            goal_hits: Vec::new(),
            learning_motion: false,
            cb,
            relay: ThalamicRelay::new(),
            rec23: zero(),
            pons: zero(),
            wm: WorkingMemory::new(BITS, 1),
            hc: {
                // RT_HC_DECAY (per-store decay of every weight, default 0.999), RT_HC_CENTER=1
                // (homeostatic centering of CA3's drive), RT_HC_SCALE=1 (presynaptic scaling
                // on every pathway)
                // input: the content (BITS) and the sparse context space; output: the content
                let mut cfg = HippocampusConfig::new(BITS + CTX_SPACE, seed.wrapping_add(300));
                cfg.out_bits = BITS;
                cfg.hashed_fan_out = Some(env("RT_HC_FANOUT", 300usize));
                cfg.decay = env("RT_HC_DECAY", 0.999f32);
                cfg.center = std::env::var("RT_HC_CENTER").is_ok();
                cfg.scale_all = std::env::var("RT_HC_SCALE").is_ok();
                // RT_HC_WRITES=lifetime: familiarity counts that never decay (the first version)
                cfg.decay_writes = std::env::var("RT_HC_WRITES").map_or(true, |v| v != "lifetime");
                Hippocampus::new(cfg)
            },
        }
    }

    /// The association area's assembly for what is seen (`eye`, a visual word code, or nothing)
    /// and heard (`ear`, a sound code, or nothing): a competitive layer (k-winners-take-all,
    /// learned when `learn`) over the two layer 4s. Seen and heard together when reading aloud,
    /// a word's cells grow synapses on both, so either sense alone comes to evoke the same
    /// assembly.
    fn assoc_of(&mut self, eye: &BitVector, ear: &BitVector, learn: bool, rng: &mut StdRng) -> BitVector {
        if let Some(b) = self.brain.as_mut() {
            // the association areas (grown by conjunction); a sense with no partner yet is
            // represented by its own area's code
            let e4 = b.encode(EYE, eye, false);
            let a4 = b.encode(EAR, ear, false);
            let c = b.associate_codes(&[e4.clone(), a4.clone(), zero()], learn);
            if c.count_ones() > 0 {
                return c;
            }
            return if a4.count_ones() > 0 { a4 } else { e4 };
        }
        let e4 = if eye.count_ones() == 0 { zero() } else { self.l4e.encode(eye) };
        let a4 = if ear.count_ones() == 0 { zero() } else { self.l4a.encode(ear) };
        let row = frames(&[&e4, &a4]);
        if learn {
            self.assoc.encode_learn(&row, rng)
        } else {
            self.assoc.encode(&row)
        }
    }

    /// The hippocampus's cue and stored context: a sparse conjunctive code (64 of CTX_SPACE
    /// bits, after the content half) set by the story's context and the last two sounds heard
    /// (silence before anyone has spoken), through a fixed random projection, as entorhinal
    /// conjunctive cells would give. A conjunction is rarely repeated, so its bits keep their
    /// drive under the hippocampus's presynaptic scaling (single words and a reused context,
    /// written thousands of times, are scaled to nothing).
    ///
    /// With `pos` (RT_HC_TIME=1), 32 more bits from the story and the position in the telling,
    /// as time cells give: a cue whose last words were wrong still matches in time.
    fn hc_context(prev: Option<&BitVector>, prev2: Option<&BitVector>, silence: &BitVector, story: u64, pos: Option<usize>) -> Vec<usize> {
        let mut h: u64 = story.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for (k, v) in [prev.unwrap_or(silence), prev2.unwrap_or(silence)].into_iter().enumerate() {
            for w in v.as_words() {
                h = (h ^ w.rotate_left(17 * k as u32 + 5)).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
                h ^= h >> 29;
            }
        }
        let mut c: Vec<usize> = (0..2 * ACTIVE as u64)
            .map(|i| {
                let mut x = h ^ i.wrapping_mul(0xD6E8_FEB8_6659_FD93);
                x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                x ^= x >> 31;
                BITS + (x % CTX_SPACE as u64) as usize
            })
            .collect();
        if let Some(t) = pos {
            let h = story.wrapping_mul(0xA24B_AED4_963E_E407) ^ (t as u64 + 1).wrapping_mul(0x9FB2_1C65_1E98_DF25);
            c.extend((0..ACTIVE as u64).map(|i| {
                let mut x = (h ^ i.wrapping_mul(0xD6E8_FEB8_6659_FD93)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                x ^= x >> 31;
                BITS + (x % CTX_SPACE as u64) as usize
            }));
        }
        c.sort_unstable();
        c.dedup();
        c
    }

    /// What the hippocampus recalls for this cue: the content half, as a frame.
    fn recall(&self, cue: &[usize]) -> BitVector {
        let r = self.hc.recall(cue);
        let mut v = zero();
        if r.strength > 0 {
            for &b in r.ec.iter().filter(|&&b| b < BITS) {
                v.bit_set(b);
            }
        }
        v
    }

    /// See `eye` (or nothing), hear `ear` (or nothing), with `recall` from the hippocampus;
    /// form the plan for the word to say.
    /// `temp`: noise on the relay's weights (0–100), for free speech.
    fn step(&mut self, eye: &BitVector, ear: &BitVector, recall: &BitVector, learn: bool, temp: u32, rng: &mut StdRng) -> Step {
        let (e4, a4, h4) = match self.brain.as_mut() {
            Some(b) => (b.encode(EYE, eye, learn), b.encode(EAR, ear, learn), Some(b.encode(HAND, &self.felt, learn))),
            None => (
                if eye.count_ones() == 0 { zero() } else if learn { self.l4e.encode_learn(eye, rng) } else { self.l4e.encode(eye) },
                if ear.count_ones() == 0 { zero() } else if learn { self.l4a.encode_learn(ear, rng) } else { self.l4a.encode(ear) },
                None,
            ),
        };
        let pfc = self.wm.content();
        let mossy = frames(&[ear, &self.pons]);
        let cb_out = self.cb.predict(&mossy).clone();
        let mut props: Vec<(u8, Driver, BitVector)> = Vec::new();
        if cb_out.count_ones() > 0 {
            props.push((2, Driver::Tonic(0), cb_out.clone()));
        }
        if let (true, Some(b)) = (self.selforg, self.brain.as_mut()) {
            let h4 = h4.unwrap_or_else(zero);
            // the senses' columns, with the association columns' activity as feedback
            let (fb, _) = b.columns_activity();
            b.sense_step(EYE, &e4, Some(&fb), learn);
            b.sense_step(EAR, &a4, Some(&fb), learn);
            b.sense_step(HAND, &h4, Some(&fb), learn);
            // association areas grow from this step's co-activity
            b.associate_codes(&[e4.clone(), a4.clone(), h4.clone()], learn);
            let mut tuft = pfc.clone();
            for (w, x) in tuft.as_words_mut().iter_mut().zip(cb_out.as_words()) {
                *w |= x;
            }
            // recall: the index's pointers name cells of the eye–ear association (a word seen or
            // heard); the hippocampus reinstates those cells there, and nowhere else
            let extra: Vec<(usize, BitVector)> = b.pair_index(EYE, EAR).filter(|_| recall.count_ones() > 0).map(|i| (i, recall.clone())).into_iter().collect();
            let outs = b.step_columns(&[e4.clone(), a4.clone(), h4.clone()], &extra, &tuft, learn);
            // every area's layer 5 offers a goal, the senses' too (sources 30 + the area)
            let mut sense_ins = Vec::new();
            for a in [EYE, EAR, HAND] {
                if let Some((code, exp)) = b.sense_state(a) {
                    if code.count_ones() > 0 {
                        if let Some((o, burst)) = b.sense_goal(a, &tuft) {
                            props.push((30 + a as u8, if burst { Driver::Burst } else { Driver::Spike }, o));
                        }
                        sense_ins.push((a, code, exp));
                    }
                }
            }
            let (act, fired) = b.columns_activity();
            self.rec23 = act;
            self.pons = project(&fired);
            // every column's goal is a source of the plan (source 10 + its index)
            let mut spiked = false;
            for (i, (goal, _, _)) in outs.iter().enumerate() {
                if let Some((o, burst)) = goal {
                    props.push((10 + i.min(200) as u8, if *burst { Driver::Burst } else { Driver::Spike }, o.clone()));
                    spiked |= !burst;
                }
            }
            self.relay.step(spiked);
            let ctx = frames(&[&pfc, ear]);
            let plan = self.mix(&props, &ctx, &cb_out, temp, rng);
            let ins = outs.into_iter().map(|(_, x, e)| (x, e)).collect();
            return Step { top_in: None, streams_in: None, selforg_in: Some((ins, tuft, sense_ins)), l23_row: zero(), l5_row: zero(), mossy, ctx, props, plan };
        }
        if let (Some(sx), Some(b)) = (self.streams.as_mut(), self.brain.as_mut()) {
            let or = |a: &BitVector, b: &BitVector| {
                let mut v = a.clone();
                for (w, x) in v.as_words_mut().iter_mut().zip(b.as_words()) {
                    *w |= x;
                }
                v
            };
            let h4 = h4.unwrap_or_else(zero);
            // the senses' columns, with the ventral stream's activity as feedback
            let fb = sx.ventral.activity();
            b.sense_step(EYE, &e4, Some(&fb), learn);
            b.sense_step(EAR, &a4, Some(&fb), learn);
            b.sense_step(HAND, &h4, Some(&fb), learn);
            let (fe, fa, fh) = if self.ff_l4 { (e4.clone(), a4.clone(), h4.clone()) } else { (b.feedforward(EYE, &e4), b.feedforward(EAR, &a4), b.feedforward(HAND, &h4)) };
            // ventral: the word (seen and heard; each alone where it has no partner yet) with the
            // hippocampus's return
            let mut word = b.associate_codes(&[e4.clone(), a4.clone()], false);
            if word.count_ones() == 0 {
                word = or(&e4, &a4);
            }
            let word = or(&word, recall);
            sx.ventral.sense(&[(0, &word)], Some(&pfc), learn, rng);
            let vexp = sx.ventral.expected.clone().unwrap_or_else(zero);
            let tuft = or(&or(&pfc, &cb_out), &sx.ventral.activity());
            // dorsal auditory–motor: what is heard and the word scene; layer 5: the meant sound
            let am_in = or(&fa, &word);
            sx.audio_motor.sense(&[(0, &am_in)], Some(&tuft), learn, rng);
            let am_exp = sx.audio_motor.expected.clone().unwrap_or_else(zero);
            let p5 = sx.audio_motor.l5.predict_full(&am_in, Some(&tuft), Some(&am_exp));
            // dorsal visual–motor: what is seen and felt; layer 5: the mouse's acts
            let vm_in = or(&fe, &fh);
            sx.visuo_motor.sense(&[(0, &vm_in)], Some(&tuft), learn, rng);
            let vm_exp = sx.visuo_motor.expected.clone().unwrap_or_else(zero);
            sx.visuo_motor.l5.predict_full(&vm_in, Some(&tuft), Some(&vm_exp));
            self.rec23 = or(&or(&sx.ventral.activity(), &sx.audio_motor.activity()), &sx.visuo_motor.activity());
            self.pons = or(&project(&sx.audio_motor.l5.last_fired()), &project(&sx.visuo_motor.l5.last_fired()));
            self.relay.step(matches!(p5, Some((_, false))));
            if let Some((o, burst)) = p5 {
                props.push((1, if burst { Driver::Burst } else { Driver::Spike }, o));
            }
            // the ventral stream's expected word, sent down to the ear: the sound expected
            let down = b.down(EYE, EAR, &vexp, EAR);
            if down.count_ones() > 0 {
                props.push((0, Driver::Tonic(1), down));
            }
            let ctx = frames(&[&pfc, ear]);
            let plan = self.mix(&props, &ctx, &cb_out, temp, rng);
            return Step { top_in: None, streams_in: Some((am_in, vm_in, tuft, am_exp, vm_exp)), selforg_in: None, l23_row: zero(), l5_row: zero(), mossy, ctx, props, plan };
        }
        if let (Some(top), Some(b)) = (self.top.as_mut(), self.brain.as_mut()) {
            let h4 = h4.unwrap_or_else(zero);
            // the senses' columns: each predicts its next code, with the top's activity as feedback
            let fb = top.activity();
            b.sense_step(EYE, &e4, Some(&fb), learn);
            b.sense_step(EAR, &a4, Some(&fb), learn);
            b.sense_step(HAND, &h4, Some(&fb), learn);
            // the scene: the association of the areas (each sense alone where it has no partner
            // yet), with the hippocampus's return
            let mut scene = b.associate_codes(&[e4.clone(), a4.clone(), h4.clone()], false);
            if scene.count_ones() == 0 {
                for c in [&e4, &a4, &h4] {
                    for (w, x) in scene.as_words_mut().iter_mut().zip(c.as_words()) {
                        *w |= x;
                    }
                }
            }
            for (w, x) in scene.as_words_mut().iter_mut().zip(recall.as_words()) {
                *w |= x;
            }
            let mut tuft = pfc.clone();
            for (w, x) in tuft.as_words_mut().iter_mut().zip(cb_out.as_words()) {
                *w |= x;
            }
            top.sense(&[(0, &scene)], Some(&tuft), learn, rng);
            let expected = top.expected.clone().unwrap_or_else(zero);
            self.rec23 = top.activity();
            let p5 = top.l5.predict_full(&scene, Some(&tuft), Some(&expected));
            self.pons = project(&top.l5.last_fired());
            self.relay.step(matches!(p5, Some((_, false))));
            if let Some((o, burst)) = p5 {
                props.push((1, if burst { Driver::Burst } else { Driver::Spike }, o));
            }
            let ctx = frames(&[&pfc, ear]);
            let plan = self.mix(&props, &ctx, &cb_out, temp, rng);
            return Step { top_in: Some((scene, tuft, expected)), streams_in: None, selforg_in: None, l23_row: zero(), l5_row: zero(), mossy, ctx, props, plan };
        }
        let l23_row = match h4.as_ref() {
            Some(h) => frames(&[&e4, &pfc, &cb_out, &a4, recall, &self.rec23, h]),
            None => frames(&[&e4, &pfc, &cb_out, &a4, recall, &self.rec23]),
        };
        let p23 = self.l23.predict(&l23_row, BITS);
        self.rec23 = project(&self.l23.last_fired());
        let o23 = p23.as_ref().map(|x| x.0.clone()).unwrap_or_else(zero);
        if let Some((o, _, burst)) = p23 {
            props.push((0, if burst { Driver::Burst } else { Driver::Tonic(1) }, o));
        }
        let l5_row = match h4.as_ref() {
            Some(h) => frames(&[&e4, &pfc, &cb_out, &a4, &o23, h]),
            None => frames(&[&e4, &pfc, &cb_out, &a4, &o23]),
        };
        let p5 = self.l5.predict(&l5_row, BITS);
        self.pons = project(&self.l5.last_fired());
        self.relay.step(matches!(p5, Some((_, _, false))));
        if let Some((o, _, burst)) = p5 {
            props.push((1, if burst { Driver::Burst } else { Driver::Spike }, o));
        }
        // the relay's context: what the prefrontal cortex holds and what was heard
        let ctx = frames(&[&pfc, ear]);
        let plan = self.mix(&props, &ctx, &cb_out, temp, rng);
        Step { top_in: None, streams_in: None, selforg_in: None, l23_row, l5_row, mossy, ctx, props, plan }
    }

    /// The plan: the relay mixes the sources' proposals, each weighted by its learned
    /// reliability in this context; the strongest 32 bits.
    fn mix(&mut self, props: &[(u8, Driver, BitVector)], ctx: &BitVector, cb_out: &BitVector, temp: u32, rng: &mut StdRng) -> BitVector {
        let mut score = vec![0u32; BITS];
        for (src, drv, pat) in props {
            let agrees = *src != 2 && overlap(pat, cb_out) >= 16;
            let mut w = self.relay.weight(*src, *drv, ctx, agrees) as u64;
            if temp > 0 {
                w = w * (100 + rng.gen_range(0..=temp) as u64) / 100;
            }
            for b in ones(pat) {
                score[b] += (w >> 4) as u32 + 1;
            }
        }
        let mut order: Vec<usize> = (0..BITS).filter(|&b| score[b] > 0).collect();
        order.sort_unstable_by(|a, b| score[*b].cmp(&score[*a]).then(a.cmp(b)));
        let mut plan = zero();
        for &b in order.iter().take(ACTIVE) {
            plan.bit_set(b);
        }
        plan
    }

    /// Learn a mouse act's result (in the wait): with streams, the visual–motor column's layer 5
    /// learns it (the auditory–motor one learns sounds); otherwise as any target.
    fn learn_motor(&mut self, st: &Step, target: &BitVector, rng: &mut StdRng) {
        if st.selforg_in.is_some() {
            self.learning_motion = true;
            self.learn(st, target, rng);
            self.learning_motion = false;
            return;
        }
        if let (Some(sx), Some((_, vm_in, tuft, _, vm_exp))) = (self.streams.as_mut(), st.streams_in.as_ref()) {
            sx.visuo_motor.l5.learn(vm_in, Some(tuft), Some(vm_exp), target, rng);
            self.cb.learn(&st.mossy, target, rng);
            for (src, drv, pat) in &st.props {
                self.relay.record(*src, *drv, &st.ctx, overlap(pat, target) >= 16, rng);
            }
            return;
        }
        self.learn(st, target, rng);
    }

    fn learn(&mut self, st: &Step, target: &BitVector, rng: &mut StdRng) {
        if let (Some(b), Some((ins, tuft, sense_ins))) = (self.brain.as_mut(), st.selforg_in.as_ref()) {
            for (i, (x, e)) in ins.iter().enumerate() {
                b.learn_goal(i, x, tuft, e, target);
            }
            for (a, code, exp) in sense_ins {
                b.learn_sense_goal(*a, code, tuft, exp, target);
            }
            // which column's goal was right, for which kind of target
            let kind = self.learning_motion as usize;
            for (src, _, pat) in &st.props {
                if *src >= 10 {
                    // association columns at 0.., sensory columns at 20 + the area
                    let i = if *src >= 30 { 20 + (*src - 30) as usize } else { (*src - 10) as usize };
                    if self.goal_hits.len() <= i {
                        self.goal_hits.resize(i + 1, [(0, 0); 2]);
                    }
                    self.goal_hits[i][kind].0 += 1;
                    self.goal_hits[i][kind].1 += (overlap(pat, target) >= 16) as u64;
                }
            }
        } else if let (Some(sx), Some((am_in, _, tuft, am_exp, _))) = (self.streams.as_mut(), st.streams_in.as_ref()) {
            sx.audio_motor.l5.learn(am_in, Some(tuft), Some(am_exp), target, rng);
        } else if let (Some(top), Some((scene, tuft, expected))) = (self.top.as_mut(), st.top_in.as_ref()) {
            top.l5.learn(scene, Some(tuft), Some(expected), target, rng);
        } else {
            self.l23.learn(&st.l23_row, target, rng);
            self.l5.learn(&st.l5_row, target, rng);
        }
        self.cb.learn(&st.mossy, target, rng);
        for (src, drv, pat) in &st.props {
            self.relay.record(*src, *drv, &st.ctx, overlap(pat, target) >= 16, rng);
        }
    }
}

/// What the striatum sees, as groups of active bits, one per input pathway (no hand-made
/// features): working memory; the association area's assemblies for the last sound and the one
/// before (an auditory trace); layer 2/3's and layer 5's fired cells (the corticostriatal
/// inputs); the eye's layer 4 (a book in view is visual input); the hippocampus's output (recall
/// reaching the cortex); the context held for the hippocampus; and the place cells it recalled.
#[allow(clippy::too_many_arguments)]
fn state_groups(
    pfc: &BitVector,
    heard: &BitVector,
    before: &BitVector,
    l23: &BitVector,
    l5: &BitVector,
    eye4: &BitVector,
    recall: &BitVector,
    held: &[usize],
    place: Option<usize>,
) -> Vec<Vec<usize>> {
    let off = |v: &BitVector, k: usize| ones(v).into_iter().map(|b| b + k * BITS).collect::<Vec<usize>>();
    let held_ids = held.iter().map(|&c| 8 * BITS + c).collect();
    let where_ = place.map_or(Vec::new(), |k| (0..16).map(|j| 8 * BITS + CTX_POOL + k * 16 + j).collect());
    vec![off(pfc, 0), off(heard, 1), off(before, 2), off(l23, 3), off(l5, 4), off(eye4, 5), off(recall, 6), held_ids, where_]
}

/// The striatum's channels (one per gate).
const CH_HOLD: usize = 1;
const CH_REINSTATE: usize = 2;
const CH_LOAD: usize = 3;
/// what the mouse does (candidates: the cortex's layer 5 predictions of a mouse motion)
const CH_ACT: usize = 5;
/// what the cortex imagines instead of doing (the same candidates: a motion imagined cues the
/// hippocampus with it)
const CH_IMAGINE: usize = 7;
/// steps a request to retrieve lasts unless renewed
const REQUEST_STEPS: usize = 32;
/// where the cortex's proposals sit in the striatum's input ids
const ACT_BASE: usize = 1 << 25;
/// the mouse's motor primitives: move the cursor left, right, and click
const MOUSE: usize = 3;
const LEFT: usize = 0;
const RIGHT: usize = 1;
const CLICK: usize = 2;
/// spontaneous mouse movements while learning: 1 in this many waiting steps (motor babbling)
const BABBLE: u32 = 6;
/// effort per mouse command (`Q16`)
const EFFORT: i32 = (ONE / 16) as i32;

/// The hippocampal context's cells are drawn from this many.
const CTX_POOL: usize = 1 << 16;
/// Place cells (16 per place on the shelf) and the content keys (association-area cells) come
/// after the context's ids in the index's key space.
const PLACE_BASE: usize = CTX_POOL;
const CONTENT_BASE: usize = CTX_POOL + 4096;

/// The hippocampus as an index (`IndexMemory`, after Teyler & DiScenna): one row per event,
/// grown as needed, forgotten when unused.
/// - **The context is driven by what is heard** (the temporal context model, Howard & Kahana
///   2002): each sound replaces a few of the context's 32 cells with cells set by layer 4's
///   response to it and the cell it replaces; a learned event boundary (`BoundaryCell`, from the cortex's
///   surprise) replaces half. The same words heard again from the same start give the same
///   contexts; a wrong word changes only a few cells.
/// - **A row's keys are the context in force; it points to a cortical assembly,** never to the
///   content: the ear's layer 4 cells active when the next sound was heard. Recall is "which
///   row had this context"; it reinstates those cells, layer 2/3 completes the assembly, and the
///   cortex says the word. Content comes back through the cortex (or by rereading).
/// - **Retrieval mode:** the prefrontal cortex requests retrieval (reinstating; the nucleus
///   reuniens route), and the hippocampus retrieves while the request stands and acetylcholine
///   is low. Then nothing new is stored, so its own retelling does not overwrite the memory it
///   reads, and only then does recall reach the cortex (Hasselmo's encoding and retrieval
///   modes). Novelty, a mismatch between recall and what others say (its own speech does not
///   count), raises acetylcholine, which holds it in encoding until it decays; holding a new
///   episode's start withdraws the request.
/// - **Reconsolidation:** when the cortex completes a recalled pointer to an assembly that
///   still holds at least half the pointed cells, the row is relearned to the assembly as it is
///   now, so the index follows layer 4's drift.
/// - **Order:** each row also links to the next. Recall expects the successor of the row just
///   recalled (CA3's sequence bias) and gives it unless another row matches the cue better by
///   12 of 32 cells; when nothing matches, the successor anyway.
/// - **The prefrontal cortex holds and reinstates context:** at a boundary one learned gate may
///   hold the context where the new episode begins; on any sound another may reinstate the
///   held context as the current one (both keyed by the last two sounds). Their dopamine is the
///   hippocampus's own comparator: whether the row it recalled predicted the sound heard next.
///   Pauses are heard as silence, a natural boundary. "Tell
///   it" can then return to the start of the story and retell it, the context evolving as it
///   did the first time.
struct IndexHc {
    mem: IndexMemory,
    ctx: Vec<usize>,
    boundary: BoundaryCell,
    held: Option<Vec<usize>>,
    /// the place on the shelf whose book is in view (its place cells join every stored row)
    place: Option<usize>,
    last_word: usize,
    last_recalled: Option<u32>,
    /// the rows recalled in the last few steps (refractory)
    refractory: Vec<u32>,
    /// each row's context slot by slot, as it was stored (the keys keep it only as a set)
    /// (and whether an event boundary followed it, moving the context by half)
    ordered: std::collections::HashMap<u32, (Vec<usize>, bool)>,
    /// the row stored at this step, if any
    stored_now: Option<u32>,
    /// a xorshift state for the context blend
    blend_rng: u64,
    /// how many best matches vote on the output when recall does not follow a link
    top_k: usize,
    /// the row recalled since the last sound heard, to relearn if the cortex completes it
    to_reconsolidate: Option<u32>,
    /// the prefrontal cortex's standing request to retrieve (through the nucleus reuniens):
    /// set by reinstating, cleared by holding a new episode's start
    request: bool,
    /// acetylcholine (`Q16`): raised by novelty (the CA1 comparator's mismatch with what others
    /// say, through the septum), decaying each step; high, it holds the hippocampus in encoding
    ach: u32,
    /// the output reaches the cortex only in retrieval mode
    gate_output: bool,
    plan: BitVector,
    boundaries: usize,
    holds: usize,
    reinstated: usize,
    relearned: usize,
    replayed: usize,
    relearned_asleep: usize,
    recalls: usize,
    debug: bool,
    had_prev: usize,
    followed: usize,
    /// diagnosis: the hold gate does nothing (the driver holds at the story's start)
    hold_off: bool,
    /// going back by recall (RT_RETRIEVE=pause, default): pauses are stored like any sound; when
    /// the striatum's retrieve gate fires, the hippocampus, cued with a pause in the current
    /// context, goes back to the row it recalls best (the pause before this story) and recall
    /// carries on along the links. RT_RETRIEVE=hold: the striatum's hold and reinstate gates.
    bookmarks: bool,
    /// the striatum's retrieve gate goes back to the recalled pause (RT_RETRIEVE=pause)
    pause_gate: bool,
    /// steps the prefrontal request to retrieve lasts unless renewed (with bookmarks, where no
    /// hold ends it)
    request_left: usize,
    /// storing goes on while retrieving (encoding and retrieval alternate within each theta
    /// cycle), except for its own speech (corollary discharge)
    theta: bool,
    /// diagnosis: the gates do nothing (the driver holds and reinstates)
    gates_off: bool,
    /// diagnosis: (boundary fired, recalled row) per heard sound, while tracing
    pub log: Vec<(bool, Option<u32>)>,
    /// always completing (RT_HC_AUTO=1): no cue and no request. Every step the entorhinal
    /// input (the context and the association area's assembly for what the cortex represents:
    /// heard, seen, said, meant or imagined) is completed to the stored row it matches best;
    /// what followed that row is the output, and the context is pulled half way toward the
    /// row's own. Acetylcholine alone sets the mode: high, the output is suppressed (encoding);
    /// low, it reaches the cortex (retrieval). Storing goes on in both (the two halves of a
    /// theta cycle), except for its own speech.
    auto: bool,
    /// what the cortex represents now, as association cells (the content half of the input)
    content: Vec<usize>,
    /// the row stored at the latest sound (not yet available to recall: it was written in the
    /// other half of the cycle)
    just_stored: Option<u32>,
    /// diagnosis: completions, and of those that reached the cortex
    completions: (usize, usize),
}

impl IndexHc {
    fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(4242));
        // RT_CTX_BANKS=1: two slower banks follow the fast one (16 and 32 slots)
        let slots = if std::env::var("RT_CTX_BANKS").map_or(false, |v| v == "1") { ACTIVE + 48 } else { ACTIVE };
        let ctx = (0..slots).map(|_| rng.gen_range(0..CTX_POOL)).collect();
        // RT_HC_TIERS=0: rarity weights 1/n (default: powers of two, 2^−⌊log₂ n⌋)
        let cfg = IndexConfig { min_overlap: 12, tiers: std::env::var("RT_HC_TIERS").map_or(true, |v| v != "0"), ..Default::default() };
        Self {
            mem: IndexMemory::new(cfg),
            ctx,
            boundary: BoundaryCell::new(BITS, 3, 80),
            held: None,
            place: None,
            last_word: 0,
            last_recalled: None,
            refractory: Vec::new(),
            ordered: std::collections::HashMap::new(),
            stored_now: None,
            blend_rng: 0x9E37_79B9_7F4A_7C15 ^ seed,
            top_k: env("RT_HC_K", 1usize),
            to_reconsolidate: None,
            request: false,
            ach: 0,
            gate_output: std::env::var("RT_HC_OUT").map_or(true, |v| v != "always"),
            plan: zero(),
            boundaries: 0,
            holds: 0,
            reinstated: 0,
            relearned: 0,
            replayed: 0,
            relearned_asleep: 0,
            recalls: 0,
            debug: false,
            had_prev: 0,
            followed: 0,
            gates_off: false,
            hold_off: false,
            bookmarks: false,
            pause_gate: false,
            request_left: 0,
            theta: false,
            log: Vec::new(),
            auto: std::env::var("RT_HC_AUTO").map_or(false, |v| v == "1"),
            content: Vec::new(),
            just_stored: None,
            completions: (0, 0),
        }
    }

    /// The context moves on with layer 4's response: `n` of its 32 cells are replaced, each by
    /// one layer 4 cell. The cells that act are the `n` with the smallest hash (min-hash), so a
    /// layer 4 response that drifted by a cell or two moves the context almost the same way:
    /// each acting cell picks a slot and sets it from itself and the cell it replaces (a fixed
    /// random projection).
    fn drift(&mut self, cells: &[usize], n: usize, salt: u64) {
        let mix = |x: u64| {
            let mut y = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            y ^= y >> 29;
            y = y.wrapping_mul(0xD6E8_FEB8_6659_FD93);
            y ^ (y >> 31)
        };
        let mut acting: Vec<(u64, usize)> = cells.iter().map(|&c| (mix(c as u64 ^ salt.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)), c)).collect();
        acting.sort_unstable();
        // the fast bank: the first ACTIVE slots (the slower banks, if any, follow it)
        let len = ACTIVE.min(self.ctx.len()) as u64;
        for &(h, c) in acting.iter().take(n) {
            let slot = (h % len) as usize;
            self.ctx[slot] = (mix(c as u64 ^ (self.ctx[slot] as u64).rotate_left(21) ^ salt) % CTX_POOL as u64) as usize;
        }
    }

    /// One step's drift: 4 of the fast bank's slots, and with banks (RT_CTX_BANKS=1) one slot of
    /// each slower bank (16 slots: a half-life of about 11 steps; 32 slots: about 22).
    fn step_drift(&mut self, cells: &[usize]) {
        self.drift(cells, 4, 1);
        let mut start = ACTIVE;
        for (bank, size) in [(1u64, 16usize), (2, 32)] {
            if self.ctx.len() < start + size {
                break;
            }
            let mix = |x: u64| {
                let mut y = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                y ^= y >> 29;
                y = y.wrapping_mul(0xD6E8_FEB8_6659_FD93);
                y ^ (y >> 31)
            };
            if let Some(&c) = cells.iter().min_by_key(|&&c| mix(c as u64 ^ bank.wrapping_mul(0xA24B_AED4_963E_E407))) {
                let h = mix(c as u64 ^ bank.wrapping_mul(0xA24B_AED4_963E_E407));
                let slot = start + (h % size as u64) as usize;
                self.ctx[slot] = (mix(c as u64 ^ (self.ctx[slot] as u64).rotate_left(13) ^ (bank << 40)) % CTX_POOL as u64) as usize;
            }
            start += size;
        }
    }

    /// What the hippocampus reinstates now: the layer 4 cells stored under the context most
    /// like the current one; else the successor of the last row recalled.
    fn recall(&mut self) -> BitVector {
        if self.auto {
            return self.complete();
        }
        // the sequence bias (CA3's recurrent links): the successor of the row just recalled is
        // expected, and is recalled whenever the cue still matches it well enough
        // expected, and is recalled unless another row matches the cue clearly better (by 12 of
        // 32 cells): the context finds where a sequence starts and corrects large jumps; the
        // links carry it on through boundaries that fall differently the second time
        // while retrieving, what the cortex represents is the recalled assembly (the cells that
        // were active when the last row was stored), and it moves the context as they did then:
        // the context replays its trajectory along the episode
        if let (Some(l), true) = (self.last_recalled, self.retrieving()) {
            let cells: Vec<usize> = self.mem.out_of(l).to_vec();
            if !cells.is_empty() {
                self.step_drift(&cells);
                // the event boundary that followed it then (the context jumped by half)
                if self.ordered.get(&l).map_or(false, |x| x.1) {
                    self.drift(&cells, ACTIVE / 2, 2);
                }
            }
        }
        let best = self.mem.recall(&self.ctx);
        let best_overlap = best.strength / 16;
        // the rows recalled in the last few steps are refractory: if one is still the best match,
        // recall moves on along the link instead of going back
        let refractory = best.ca3.first().map_or(false, |b| self.refractory.contains(b));
        let follow = self.last_recalled.and_then(|l| self.mem.successor_match(l, &self.ctx)).filter(|&(_, o)| refractory || best.ec.is_empty() || o + 12 > best_overlap);
        self.recalls += 1;
        if self.debug {
            let sm = self.last_recalled.and_then(|l| self.mem.successor_match(l, &self.ctx));
            println!("    recall: last {:?} successor {:?} best row {:?} overlap {} -> follow {}", self.last_recalled, sm, best.ca3.first(), best_overlap, follow.is_some());
        }
        self.had_prev += self.last_recalled.is_some() as usize;
        let followed = follow.is_some();
        let mut r = match follow {
            Some(_) => {
                self.followed += 1;
                self.mem.successor(self.last_recalled.unwrap())
            }
            None => best,
        };
        // top-k (RT_HC_K): when recall does not follow a link, the k best matches vote, and the
        // output keeps the bits most of them share (a bit-sliced majority)
        if !followed && self.top_k > 1 && !r.ec.is_empty() {
            let rows = self.mem.matches(&self.ctx, self.top_k);
            if rows.len() > 1 {
                let mut votes: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
                for &(row, _) in &rows {
                    for &b in self.mem.out_of(row) {
                        *votes.entry(b).or_default() += 1;
                    }
                }
                let need = rows.len() / 2 + 1;
                let mut ec: Vec<usize> = votes.into_iter().filter(|&(_, v)| v >= need).map(|(b, _)| b).collect();
                ec.sort_unstable();
                if !ec.is_empty() {
                    r.ec = ec;
                }
            }
        }
        if r.ec.is_empty() {
            if let Some(l) = self.last_recalled {
                r = self.mem.successor(l);
            }
        }
        self.last_recalled = r.ca3.first().copied();
        // context reinstatement, partial: while retrieving, each slot takes the recalled row's
        // stored value with probability 1/4, so the context stays in this episode even as the
        // replay slips (a word's assembly is the same in every episode; its stored context is not)
        if let (Some(l), true) = (self.last_recalled, self.retrieving()) {
            if let Some((stored, _)) = self.ordered.get(&l) {
                if stored.len() == self.ctx.len() {
                    for i in 0..self.ctx.len() {
                        self.blend_rng ^= self.blend_rng << 13;
                        self.blend_rng ^= self.blend_rng >> 7;
                        self.blend_rng ^= self.blend_rng << 17;
                        if self.blend_rng & 3 == 0 {
                            self.ctx[i] = stored[i];
                        }
                    }
                }
            }
        }
        if let Some(l) = self.last_recalled {
            self.refractory.push(l);
            if self.refractory.len() > 4 {
                self.refractory.remove(0);
            }
        }
        self.to_reconsolidate = self.last_recalled;
        let mut v = zero();
        // in encoding mode the hippocampus's output to the cortex is suppressed (Hasselmo): what
        // reaches the cortex is a memory being retrieved, never a guess during new input
        if self.gate_output && !self.retrieving() {
            return v;
        }
        for &b in r.ec.iter().filter(|&&b| b < BITS) {
            v.bit_set(b);
        }
        v
    }

    /// Always completing (`auto`): the entorhinal input now (context and content) is completed
    /// to the best matching row other than the one just written; the row that followed it is
    /// the output (CA3's sequence step), and in retrieval mode the context is pulled half way
    /// toward the completed row's own (each slot with probability 1/2).
    fn complete(&mut self) -> BitVector {
        self.recalls += 1;
        let cue = self.ec(&self.content);
        let c = self.mem.matches(&cue, 2).into_iter().map(|x| x.0).find(|&r| Some(r) != self.just_stored);
        let r = c.map_or_else(Default::default, |c| self.mem.successor(c));
        self.last_recalled = r.ca3.first().copied();
        self.to_reconsolidate = self.last_recalled;
        self.completions.0 += 1;
        let mut v = zero();
        if !self.retrieving() {
            return v;
        }
        if let Some((stored, _)) = c.and_then(|c| self.ordered.get(&c)) {
            if stored.len() == self.ctx.len() {
                for i in 0..self.ctx.len() {
                    self.blend_rng ^= self.blend_rng << 13;
                    self.blend_rng ^= self.blend_rng >> 7;
                    self.blend_rng ^= self.blend_rng << 17;
                    if self.blend_rng & 1 == 0 {
                        self.ctx[i] = stored[i];
                    }
                }
            }
        }
        self.completions.1 += 1;
        for &b in r.ec.iter().filter(|&&b| b < BITS) {
            v.bit_set(b);
        }
        v
    }

    /// The cortex represents something not heard (meant, said silently, imagined): it becomes
    /// the content half of the input, and the context moves with it.
    fn represent(&mut self, cells: &[usize]) {
        self.content = cells.to_vec();
        self.step_drift(cells);
    }

    /// A sound is heard (`key` its word, `code` its sensory code, `cells` its layer 4 cells):
    /// store a pointer to the layer 4 cells under the context in force (if `store`), move the
    /// context on with those cells, learn the boundary cell (its input: layer 4) from the
    /// cortex's surprise at the sound, and at a boundary let the gates hold and reinstate.
    /// Returns whether an event boundary fired.
    /// `heard`: the content keys: the association area's assembly for what is seen while a
    /// book is in view, else for the sound alone.
    fn hear(&mut self, key: usize, code: &BitVector, cells: &BitVector, heard: &BitVector, store: bool, own: bool) -> bool {
        // acetylcholine decays each step (a quarter)
        self.ach -= self.ach / 4;
        let surprise = ONE - (overlap(&self.plan, code).min(ACTIVE as u32) * ONE / ACTIVE as u32);
        self.boundary.learn(surprise);
        let cells = ones(cells);
        // reconsolidation: the row just recalled pointed to an assembly; if what the cortex
        // completed and what was then heard is that assembly (at least half its pointed cells
        // still in it), the row is relearned to the assembly as it is now
        if let Some(r) = self.to_reconsolidate.take() {
            let old = self.mem.out_of(r).to_vec();
            let kept = old.iter().filter(|c| cells.binary_search(c).is_ok()).count();
            let matched = !old.is_empty() && kept * 2 >= old.len();
            if matched && old != cells {
                self.mem.reconsolidate(r, &cells);
                self.relearned += 1;
            }
            // a mismatch is novelty: it raises acetylcholine, which holds the hippocampus in
            // encoding until it decays; its own speech is not novel (corollary discharge damps
            // the response to self-made sounds)
            if !matched && !own {
                self.ach += (ONE - self.ach.min(ONE)) / 2;
            }
        }
        // retrieval mode (requested and acetylcholine low): nothing new is stored (encoding and
        // retrieval are separate modes, Hasselmo)
        if self.auto {
            self.content = ones(heard);
            self.just_stored = None;
        }
        let may_store = if self.auto { !own } else { !self.retrieving() || (self.theta && !own) };
        if store && may_store && !cells.is_empty() {
            // the keys: the context, the place (if a book from the shelf is in view), and the
            // content (the assembly), so the episode can be completed from any of them
            let mut keys = self.ctx.clone();
            if let Some(k) = self.place {
                keys.extend((0..16).map(|j| PLACE_BASE + k * 16 + j));
            }
            keys.extend(ones(heard).into_iter().map(|c| CONTENT_BASE + c));
            let id = self.mem.rows() as u32;
            self.mem.store_split(&keys, &[], &cells);
            if self.mem.rows() as u32 > id {
                self.ordered.insert(id, (self.ctx.clone(), false));
                self.stored_now = Some(id);
                self.just_stored = Some(id);
            }
        }
        // the context moves on with layer 4's cells, not the raw sound; while retrieving, the
        // network's own speech does not move it (it is predicted: corollary discharge), the
        // recall does (see `recall`)
        if self.auto || !(own && self.retrieving()) {
            self.step_drift(&cells);
        }
        self.last_word = key;
        let fired = self.boundary.observe(&cells);
        self.log.push((fired, self.last_recalled));
        // the request lapses unless renewed (prefrontal activity that is not maintained fades)
        if self.request && self.bookmarks && self.request_left > 0 {
            self.request_left -= 1;
            if self.request_left == 0 {
                self.request = false;
            }
        }
        if fired {
            self.boundaries += 1;
            self.drift(&cells, ACTIVE / 2, 2);
            if let Some(e) = self.stored_now.and_then(|id| self.ordered.get_mut(&id)) {
                e.1 = true;
            }
        }
        self.stored_now = None;
        fired
    }

    /// Retrieval mode: the prefrontal request stands and acetylcholine is low (Hasselmo: high
    /// acetylcholine favours encoding, low favours retrieval).
    fn retrieving(&self) -> bool {
        (self.request || self.auto) && self.ach < ONE / 2
    }

    /// A reach (a motor act that brings a new scene, a book) is an event boundary: the context
    /// moves on by half.
    fn reach_event(&mut self) {
        self.boundaries += 1;
        let c = self.ctx.clone();
        self.drift(&c, ACTIVE / 2, 3);
    }

    /// The entorhinal input now, the only way anything cues the hippocampus: the context cells,
    /// the place cells (when a scene from the shelf is in view) and the content (the association
    /// area's assembly for what is being heard).
    fn ec(&self, content: &[usize]) -> Vec<usize> {
        let mut v = self.ctx.clone();
        if let Some(k) = self.place {
            v.extend((0..16).map(|j| PLACE_BASE + k * 16 + j));
        }
        v.extend(content.iter().map(|&c| CONTENT_BASE + c));
        v
    }

    /// Where was this seen? The hippocampus completes the entorhinal input (`content`: what is
    /// active in the cortex now) and reads out the place component: each matching episode (up
    /// to 256) votes for its place with its match score; episodes with no place vote nothing.
    fn where_of(&self, content: &[usize]) -> Option<usize> {
        let mut count = [0u64; 16];
        for (row, score) in self.mem.matches(&self.ec(content), 256) {
            for &k in self.mem.keys_of(row) {
                let k = k as usize;
                if k >= PLACE_BASE && k < PLACE_BASE + 16 * 16 && (k - PLACE_BASE) % 16 == 0 {
                    count[(k - PLACE_BASE) / 16] += score;
                }
            }
        }
        let (best, n) = count.iter().enumerate().max_by_key(|x| (*x.1, std::cmp::Reverse(x.0)))?;
        (*n > 0).then_some(best)
    }

    /// The cortex's own state moves the context when nothing is heard: the association area's
    /// assembly of what it represents (its plan: a word recalled or meant, said or not).
    fn drift_with(&mut self, cells: &[usize]) {
        if self.auto {
            self.represent(cells);
            return;
        }
        // while retrieving, the cortex represents what is recalled, and that moved it already
        if !self.retrieving() {
            self.step_drift(cells);
        }
    }

    /// The prefrontal cortex holds the context in force: where this episode begins.
    fn hold(&mut self) {
        self.held = Some(self.ctx.clone());
        self.request = false;
        self.holds += 1;
    }

    /// Where to go back to: cued with a pause (`pause`: the association cells of silence) in
    /// the current context, the row the hippocampus recalls best. The pause before this story
    /// matches best, since the drifting context still overlaps it most; its link leads to the
    /// story's first word.
    fn pause_row(&self, pause: &[usize]) -> Option<u32> {
        self.best_row(pause)
    }

    /// The row the hippocampus recalls best for some content (the association cells of what
    /// is imagined, heard or seen), cued by the content alone: every row with that content
    /// matches as well, and the most recent wins. (With the current context in the cue as
    /// well, the context's rare bits outweigh a content seen in every episode, and recall goes
    /// to whatever was stored last.)
    fn best_row(&self, content: &[usize]) -> Option<u32> {
        let cue: Vec<usize> = content.iter().map(|&c| CONTENT_BASE + c).collect();
        self.mem.matches(&cue, 1).first().map(|x| x.0)
    }

    /// Row `r` is gone back to: its context becomes the current one, recall carries on from it
    /// along the links, and retrieval begins.
    fn reinstate_row(&mut self, r: u32) {
        self.ctx = self.ordered.get(&r).map(|x| x.0.clone()).unwrap_or_else(|| self.mem.keys_of(r).iter().map(|&k| k as usize).filter(|&k| k < CTX_POOL).collect());
        self.last_recalled = Some(r);
        self.refractory = vec![r];
        self.request = true;
        self.request_left = REQUEST_STEPS;
        self.reinstated += 1;
    }

    /// The held context is reinstated as the current one, and retrieval begins.
    fn reinstate(&mut self) {
        if let Some(h) = self.held.clone() {
            self.ctx = h;
            self.last_recalled = None;
            self.request = true;
            self.request_left = REQUEST_STEPS;
            self.reinstated += 1;
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Book {
    Same,
    Other,
    Closed,
}

#[derive(Default, Clone, Copy)]
struct Tally {
    words: usize,
    right: usize,
    said_story: usize,
    said_book: usize,
    silent: usize,
}

/// The basal ganglia in use: spiny popcount cells (`Striatum`) or masks (`MaskStriatum`).
enum Bg {
    Cells(Striatum),
    Masks(MaskStriatum),
    /// diagnosis: masks, with the channels listed handled by cells (RT_CELLS_FOR)
    Mixed(Box<Striatum>, Box<MaskStriatum>, Vec<usize>),
}

impl Bg {
    fn begin_groups(&mut self, s: &[Vec<usize>], learn: bool, rng: &mut StdRng) {
        match self {
            Bg::Cells(x) => x.begin_groups(s, learn, rng),
            Bg::Masks(x) => x.begin_groups(s, learn),
            Bg::Mixed(c, m, _) => {
                c.begin_groups(s, learn, rng);
                m.begin_groups(s, learn);
            }
        }
    }
    fn value(&self) -> i32 {
        match self {
            Bg::Cells(_) => 0,
            Bg::Masks(x) => x.value(),
            Bg::Mixed(_, m, _) => m.value(),
        }
    }
    /// curiosity as noise: extra GPi noise for the next choices (masks in grow mode only)
    fn set_noise(&mut self, n: i32) {
        match self {
            Bg::Cells(_) => {}
            Bg::Masks(x) => x.extra_noise = n,
            Bg::Mixed(_, m, _) => m.extra_noise = n,
        }
    }
    fn choose(&mut self, ch: usize, n: usize, rng: Option<&mut StdRng>) -> usize {
        match self {
            Bg::Cells(x) => x.choose(ch, n, rng),
            Bg::Masks(x) => x.choose(ch, n, rng.is_some()),
            Bg::Mixed(c, m, on) => {
                if on.contains(&ch) {
                    c.choose(ch, n, rng)
                } else {
                    m.choose(ch, n, rng.is_some())
                }
            }
        }
    }
    fn choose_among(&mut self, ch: usize, cands: &[Vec<Vec<usize>>], rng: Option<&mut StdRng>) -> Option<usize> {
        match self {
            Bg::Cells(x) => x.choose_among(ch, cands, rng),
            Bg::Masks(x) => x.choose_among(ch, cands, rng.is_some()),
            Bg::Mixed(c, m, on) => {
                if on.contains(&ch) {
                    c.choose_among(ch, cands, rng)
                } else {
                    m.choose_among(ch, cands, rng.is_some())
                }
            }
        }
    }
    fn did_among(&mut self, ch: usize, cand: &[Vec<usize>]) {
        match self {
            Bg::Cells(x) => x.did_among(ch, cand),
            Bg::Masks(x) => x.did_among(ch, cand),
            Bg::Mixed(c, m, on) => {
                if on.contains(&ch) {
                    c.did_among(ch, cand)
                } else {
                    m.did_among(ch, cand)
                }
            }
        }
    }
    fn maintain(&mut self, ch: usize) {
        match self {
            Bg::Cells(x) => x.maintain(ch),
            Bg::Masks(x) => x.maintain(ch),
            Bg::Mixed(c, m, _) => {
                c.maintain(ch);
                m.maintain(ch);
            }
        }
    }
    fn retag(&mut self, ch: usize) {
        match self {
            Bg::Cells(x) => x.retag(ch),
            Bg::Masks(x) => x.retag(ch),
            Bg::Mixed(c, m, _) => {
                c.retag(ch);
                m.retag(ch);
            }
        }
    }
    fn reward(&mut self, r: i32) {
        match self {
            Bg::Cells(x) => x.reward(r),
            Bg::Masks(x) => x.reward(r),
            Bg::Mixed(c, m, _) => {
                c.reward(r);
                m.reward(r);
            }
        }
    }
    fn end(&mut self, learn: bool, rng: &mut StdRng) {
        match self {
            Bg::Cells(x) => x.end(learn, rng),
            Bg::Masks(x) => x.end(learn),
            Bg::Mixed(c, m, _) => {
                c.end(learn, rng);
                m.end(learn);
            }
        }
    }
    /// (the current state's value, 0), for reports
    fn debug_now(&mut self, ch: usize) -> (i32, i32) {
        match self {
            Bg::Cells(x) => x.debug_now(ch),
            Bg::Masks(x) => (x.value(), 0),
            Bg::Mixed(_, m, _) => (m.value(), 0),
        }
    }
    fn last_delta(&self) -> i32 {
        match self {
            Bg::Cells(x) => x.last_delta,
            Bg::Masks(x) => x.last_delta,
            Bg::Mixed(_, m, _) => m.last_delta,
        }
    }
    fn report(&self) -> String {
        let cells = |x: &Striatum| {
            format!(
                "cells: {} dopamine events, mean |dopamine| {:.3}, {} cholinergic pauses; striatal cells: {} striosome, {} go, {} no-go ({} recruited, {} removed)",
                x.stats.0,
                x.stats.1 as f64 / x.stats.0.max(1) as f64 / ONE as f64,
                x.pauses,
                x.sizes().0,
                x.sizes().1,
                x.sizes().2,
                x.changes.0,
                x.changes.1
            )
        };
        let masks = |x: &MaskStriatum| {
            format!(
                "masks: {} dopamine events, mean |dopamine| {:.3}, {} cholinergic pauses, {} bits changed",
                x.stats.0,
                x.stats.1 as f64 / x.stats.0.max(1) as f64 / ONE as f64,
                x.pauses,
                x.swaps
            )
        };
        match self {
            Bg::Cells(x) => cells(x),
            Bg::Masks(x) => masks(x),
            Bg::Mixed(c, m, on) => format!("{}; channels {:?} on {}", masks(m), on, cells(c)),
        }
    }
}

fn main() {
    let seed: u64 = env("SEED", 0);
    let mut rng = StdRng::seed_from_u64(seed);
    // RT_MAKE=1: a third instruction, "now make one": the teacher makes up a new story
    let make_on = std::env::var("RT_MAKE").map_or(false, |v| v == "1");
    let make_temp: u32 = env("RT_TEMP", 30);
    // RT_SHELF (default 4): books on a shelf at that many places; "now find the <animal>" asks
    // for one (a quarter of the trials). 0: no shelf
    let shelf_k: usize = env("RT_SHELF", 4);
    let vocab = vocabulary(make_on, shelf_k > 0);
    let id = |w: &str| vocab.iter().position(|v| *v == w).unwrap();
    let ear = Encoder::new(vocab.len(), BITS, ACTIVE, &mut rng);
    let eye = Encoder::new(vocab.len(), BITS, ACTIVE, &mut rng);
    // the sound of silence: what the ear hears before anyone speaks
    let silence = Encoder::new(1, BITS, ACTIVE, &mut rng).codes[0].clone();
    let tract = VocalTract::new(vocab.len() + 1, BITS, ACTIVE, &mut rng);
    let mut motor = MotorArea::new(BITS);
    // RT_STOP=learned (default): silence is a sound the tract can make (closing the mouth),
    // learned by babbling like the words; the teacher's silence after the last word is heard,
    // so the network learns to predict the end, and stops by saying silence. RT_STOP=hand:
    // the driver ends an invented story at "home ." (and silence is not a sound)
    let stop_learned = std::env::var("RT_STOP").map_or(true, |v| v != "hand");
    let quiet = vocab.len();
    let sounds: Vec<BitVector> = ear.codes.iter().cloned().chain(stop_learned.then(|| silence.clone())).collect();
    motor.babble(&tract, &sounds, 3, &mut rng);
    // RT_CORRECT=heard (default): in practice, a wrong word is followed by the teacher saying
    // the right one aloud, which the network hears; RT_CORRECT=oracle: the teacher's word is
    // the target without being heard (the first version)
    let heard_correction = std::env::var("RT_CORRECT").map_or(true, |v| v != "oracle");
    // after a step where nothing came out, the ear hears nothing (default): a plan that could
    // not be said is not a sound, and the hippocampus, moving on along its links, carries the
    // next word. RT_SILENT=efference: the unsaid plan is heard as if said (the first version)
    let heard_plan = std::env::var("RT_SILENT").map_or(false, |v| v == "efference");
    // RT_REACH=learned (default): a book stays shut until the network reaches for it, a motor
    // act chosen by a basal-ganglia gate (keyed by the last two sounds heard, credited by the
    // words said right); the world then hands it the book from its first word (a file retrieved),
    // and the reach is an event boundary. RT_REACH=given: the book is open from the start.
    let reach_learned = std::env::var("RT_REACH").map_or(true, |v| v != "given");
    // the striatum: one critic, and an actor channel per gate (reach, hold, reinstate, load into
    // working memory); dopamine is the TD error, the reward the words that come out right
    // RT_STRIATUM=masks (default: go/no-go masks, no STN hold, sticky habit bits, pair features,
    // the swap rule) or cells (spiny popcount cells)
    let mut striatum = match std::env::var("RT_STRIATUM").unwrap_or_else(|_| "masks".into()).as_str() {
        "cells" => Bg::Cells(Striatum::new(1 << 18)),
        _ => {
            let mut m = MaskStriatum::new(2048, 64, seed.wrapping_add(31));
            m.hold = false;
            m.sticky = true;
            m.conj = true;
            m.push_pull = false;
            // RT_MASK_RELEASE: the masks' tonic brake (default 2)
            m.release = env("RT_MASK_RELEASE", 2);
            // pools of mask-cells recruited as needed (RT_MASK_GROW=0: one mask per pathway); RT_MASK_STICKY=0: no
            // habit bits
            m.grow = std::env::var("RT_MASK_GROW").map_or(true, |v| v != "0");
            m.sticky = std::env::var("RT_MASK_STICKY").map_or(true, |v| v != "0");
            // RT_CELLS_FOR (diagnosis): channels handled by cells, the rest by masks
            match std::env::var("RT_CELLS_FOR") {
                Ok(v) => Bg::Mixed(Box::new(Striatum::new(1 << 18)), Box::new(m), v.split(',').filter_map(|x| x.parse().ok()).collect()),
                Err(_) => Bg::Masks(m),
            }
        }
    };
    // the screen and the mouse (the network's body): a row of places, the desk (where the
    // story's book lies, unless it is closed) and then the shelf's places. The eye sees where the
    // cursor is (a visual code per place); a mouse command is felt as its motion (a code per
    // primitive: left, right, click), sensed like a sound. The mouse's motor area is learned by
    // babbling before the trials, as speech is; a command that matches no primitive does nothing.
    let npos = 1 + shelf_k;
    let cursor_seen = Encoder::new(npos, BITS, ACTIVE, &mut rng).codes;
    let motion = Encoder::new(MOUSE, BITS, ACTIVE, &mut rng).codes;
    // what a mouse act looks like on the screen (the cursor moving, a click's flash), whoever
    // does it; drawn from its own generator so the other codes stay as they were
    let motion_seen = Encoder::new(MOUSE, BITS, ACTIVE, &mut StdRng::seed_from_u64(seed ^ 0x5EE5)).codes;
    // RT_TEACH_SEEN=1: the teacher's act is seen (on the screen), not felt as one's own motion
    // the teacher shows how with a cursor of their own, on their own copy
    // of the book: the network's cursor stays where it is, its book opens only when it clicks it
    // itself, and the teacher's act is seen (implies RT_TEACH_SEEN=1)
    // (the default; RT_TEACHER_CURSOR=0: the teacher moves the network's own cursor)
    let teacher_cursor = std::env::var("RT_TEACHER_CURSOR").map_or(true, |v| v != "0");
    let teach_seen = teacher_cursor || std::env::var("RT_TEACH_SEEN").map_or(false, |v| v == "1");
    // (the teacher's demonstrations that ended with their book open)
    let mut teacher_opened = 0usize;
    let wait_end = std::env::var("RT_WAIT_END").map_or(false, |v| v == "1");
    // the row holding this trial's story's opening, and (imagined clicks while telling, landing
    // on it)
    let mut story_open_row: Option<u32> = None;
    let mut open_recall = (0usize, 0usize);
    // RT_CHECK=1: an act just credited by a reward is checked the next time it is done: if it
    // has none of the effect the forward model expects, that is a dip; =always: every act
    let check_mode = std::env::var("RT_CHECK").unwrap_or_default();
    let mut check_due = [false; 3];
    // (checks made, of which failed)
    let mut checks = (0usize, 0usize);
    // the mirror (with areas and RT_TEACH_SEEN=1): (teacher's acts seen, mirrored to the right felt motion)
    let mut mirror_stats = (0usize, 0usize);
    let mouse = VocalTract::new(MOUSE, BITS, ACTIVE, &mut rng);
    let mut hand = Motor::Kernels(MotorArea::new(BITS));
    hand.babble(&mouse, &motion, 3, &mut rng);
    let mut cursor = 0usize;
    // curiosity: the cortex's misses at predicting what follows, one bit per waiting step (a
    // shift register); its learning progress is the long-run miss rate minus the recent one
    // one register per context (the working memory's content and the cursor's place, hashed
    // into 64): progress is learning in this situation, not on average
    // A register starts empty: a situation with no history shows no progress (starting it full of
    // misses made every newly visited situation look like learning, and paid the mouse to wander)
    let mut misses: Vec<u64> = vec![0; 64];
    // cortical curiosity: (sum of the extra noise, choices)
    let mut curiosity_stats = (0usize, 0usize);
    // (babbled, released, left, right, clicks, books opened, steps shown by the teacher)
    let mut mouse_stats = [0usize; 7];
    let observe = std::env::var("RT_OBSERVE").map_or(false, |v| v == "1");
    let blocked_senses = std::env::var("RT_BLOCKED").map_or(true, |v| v != "0");
    // diagnosis, read trials with the cursor on the book (training, test): steps, CLICK
    // proposed by the cortex, LEFT proposed, CLICK released, LEFT released
    let mut on_book = [[0usize; 5]; 2];
    // diagnosis, read trials at test (instruction held), cursor elsewhere: (place, what the
    // cortex proposed, what was released) counts, as "place:proposed>released"
    let mut off_book: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    // RT_TEACH=none: the teacher never shows how to use the mouse
    let teach_mouse = std::env::var("RT_TEACH").map_or(true, |v| v != "none");
    // RT_STORY=book (default): the teacher reads the story from a book, opened with a click the
    // network sees, and the network sees each word as it is read; RT_STORY=heard: told, no book
    let story_book = std::env::var("RT_STORY").map_or(true, |v| v != "heard");
    let imagine_on = std::env::var("RT_RETRIEVE").map_or(true, |v| v == "imagine");
    // (imagined, of which while retelling)
    let mut imagined = (0usize, 0usize);
    let mut curiosity_sum = 0i64;
    // RT_CURIOSITY=0: learning progress is not rewarded
    // curiosity: RT_CURIOSITY=noise adds a little exploration noise where the cortex's
    // predictions are improving; =reward pays it as reward; =0 off
    // =cortex (default): the noise is the learning progress of the layer 5 cell predicting now
    // (its own record of hits and misses), damped while reward is expected (RT_LC_DAMP=1)
    let curiosity_mode = std::env::var("RT_CURIOSITY").unwrap_or_else(|_| "cortex".into());
    let curiosity_cortex = curiosity_mode == "cortex";
    let lc_damp = std::env::var("RT_LC_DAMP").map_or(false, |v| v == "1");
    let curiosity_on = curiosity_mode == "reward" || curiosity_mode == "1";
    let curiosity_noise = curiosity_mode == "noise";
    // a trial is a finite episode: no discounting within it (an extra step, such as the event a
    // reach makes, costs nothing); the eligibility traces still decay by lambda
    let mut shelf: Vec<Vec<&str>> = Vec::new();
    for k in 0..shelf_k {
        let b = shelf_book(&mut rng, &shelf, k);
        shelf.push(b);
    }
    // "find" trials at test: (trials, reached the right place, reached a place, words, right)
    let mut find_stats = (0usize, 0usize, 0usize, 0usize, 0usize);
    // all "find" trials: (trials, working memory held the animal, recalled the right place, recalled a place)
    let mut find_diag = (0usize, 0usize, 0usize, 0usize);
    // training read trials: words planned right without and with a reach
    let mut read_by_reach = [(0usize, 0usize); 2];
    // training: the dopamine signal right after a reach, per instruction (read, tell, make)
    let mut reach_delta = [(0i64, 0usize); 3];
    let mut wait_states: [Vec<Vec<usize>>; 2] = [Vec::new(), Vec::new()];
    // per (instruction read/tell, book): (trials, reached, words before the reach), at test
    let mut reach_stats = [[(0usize, 0usize, 0usize); 3]; 2];
    // RT_HC_STORE=all (default): the hippocampus stores every word heard (the instruction,
    // the task, its own speech), its novelty setting the strength; RT_HC_STORE=listen: only
    // the story as first heard
    let store_all = std::env::var("RT_HC_STORE").map_or(true, |v| v != "listen");
    let mut net = Net::new(seed);
    // with areas (the default), proposals are offered undecoded (RT_DECODE=1: only those near a
    // known motion, labelled)
    let undecoded = net.brain.is_some() && std::env::var("RT_DECODE").map_or(true, |v| v != "1");
    // with areas, the voice's own motor area learns by babbling (the shared one is unused)
    if let Some(b) = net.brain.as_mut() {
        b.areas[VOICE].motor.as_mut().unwrap().babble(&tract, &sounds, 3, &mut rng);
    }
    // with areas, the hand's own area's motor side learns by babbling too, and is the one used
    if let Some(b) = net.brain.as_mut() {
        b.areas[HAND].motor.as_mut().unwrap().babble(&mouse, &motion, 3, &mut rng);
    }
    macro_rules! hand {
        () => {
            match net.brain.as_mut().and_then(|b| b.areas[HAND].motor.as_mut()) {
                Some(m) => m,
                None => &mut hand,
            }
        };
    }
    // the cortex's assembly for a mouse act (done, seen or imagined): with areas, the association
    // of the hand (how it feels) and seen acts (how it looks); else the old shortcut, the motion
    // as an ear input
    macro_rules! act_assembly {
        ($m:expr) => {{
            let m: usize = $m;
            match net.brain.as_mut() {
                Some(b) => {
                    let h = b.encode(HAND, &motion[m], false);
                    let e = b.encode(ACTS, &motion_seen[m], false);
                    let c = b.associate_codes(&[zero(), zero(), h.clone(), zero(), e], false);
                    if c.count_ones() > 0 {
                        c
                    } else {
                        h
                    }
                }
                None => net.assoc_of(&zero(), &motion[m], false, &mut rng),
            }
        }};
    }
    // an act imagined (with areas): the cortex's proposal, as the hand's motor area would carry
    // it out, run through its forward model without acting (motor imagery): how it would feel;
    // the mirror adds how it would look; their association is the act as the cortex represents
    // it. No label: whatever the proposal is.
    macro_rules! imagined_assembly {
        ($pat:expr) => {{
            let pat: &BitVector = $pat;
            let cmd = hand!().plan(pat);
            let felt = cmd.and_then(|c| hand!().predict(&c)).unwrap_or_else(zero);
            let b = net.brain.as_mut().unwrap();
            let h = b.encode(HAND, &felt, false);
            let seen = b.mirror(HAND, &h, ACTS);
            let e = b.encode_only(ACTS, &seen);
            let c = b.associate_codes(&[zero(), zero(), h.clone(), zero(), e], false);
            if c.count_ones() > 0 {
                c
            } else {
                h
            }
        }};
    }
    // an act seen (with areas): how it looks, and through the mirror how it would feel
    macro_rules! seen_assembly {
        ($seen:expr) => {{
            let seen: &BitVector = $seen;
            let b = net.brain.as_mut().unwrap();
            let e = b.encode_only(ACTS, seen);
            let felt = b.mirror(ACTS, &e, HAND);
            let h = b.encode_only(HAND, &felt);
            let c = b.associate_codes(&[zero(), zero(), h, zero(), e.clone()], false);
            if c.count_ones() > 0 {
                c
            } else {
                e
            }
        }};
    }
    // the voice's corollary discharge (with areas): (words said, of which not heard as expected)
    let mut voice_check = (0usize, 0usize);
    let imitate = std::env::var("RT_IMITATE").map_or(true, |v| v != "0");
    // (words said through the inverse model, through the imitation route)
    let mut voice_route = [0usize; 2];
    // a word said: None for nothing said or (with the learned stop) silence. With areas, the
    // voice's area turns the plan into a command and keeps its efference copy, which is compared
    // with what the ear then hears
    macro_rules! say {
        ($plan:expr) => {{
            let plan: &BitVector = $plan;
            match net.brain.as_mut() {
                Some(b) => {
                    // the imitation route first (the meant sound, as the ear would hear it, evokes
                    // through the ear–voice association the command that says it), else the
                    // voice's inverse model (RT_IMITATE=0: the inverse model only)
                    let mirrored = if imitate {
                        let e = b.encode_only(EAR, plan);
                        Some(b.mirror(EAR, &e, VOICE)).filter(|m| tract.articulate(m).is_some())
                    } else {
                        None
                    };
                    voice_route[mirrored.is_some() as usize] += 1;
                    let v = &mut b.areas[VOICE];
                    let cmd = mirrored.or_else(|| v.command_for(plan));
                    cmd.and_then(|c| {
                        v.issue(&c);
                        let w = tract.articulate(&c);
                        let heard = w.map_or_else(zero, |w| sounds[w].clone());
                        v.sense(&heard, false);
                        if w.is_some() {
                            voice_check.0 += 1;
                            voice_check.1 += (v.effect_error.unwrap_or(0) >= ONE / 2) as usize;
                        }
                        w
                    })
                }
                None => motor.plan(plan).and_then(|m| tract.articulate(&m)),
            }
        }};
    }
    let fixed_gate = std::env::var("RT_GATE").map_or(false, |v| v == "fixed");
    let td_debug = std::env::var("RT_TD_DEBUG").is_ok();
    // the striatum's reward is the outcome: words said right in the task (the teacher's
    // words); RT_REWARD=all also rewards every heard word the cortex predicted
    let reward_all = std::env::var("RT_REWARD").map_or(false, |v| v == "all");
    // RT_HC=index (default): the hippocampus as a growing index with learned event boundaries
    // (`IndexHc`); RT_HC=circuit: the fixed circuit cued by the driver's story context
    // RT_IX_HOLD=oracle (diagnosis only): the context is held at the story's first word and
    // reinstated when the task begins, instead of by the learned gates; RT_IX_HOLD=hold: only
    // the hold is given, reinstating is learned
    let ix_oracle = std::env::var("RT_IX_HOLD").map_or(false, |v| v == "oracle");
    let hold_oracle = std::env::var("RT_IX_HOLD").map_or(false, |v| v == "oracle" || v == "hold");
    // RT_RESTART=on: reinstating while retrieving, which starts the sequence over (default off:
    // the gate then fires on almost every step)
    let restart = std::env::var("RT_RESTART").map_or(false, |v| v == "on");
    // RT_IX_HOLD=pause (diagnosis only): the driver triggers retrieval as the task begins, the
    // hippocampus going back to the pause it recalls best (the learned gate is off)
    let pause_oracle = std::env::var("RT_IX_HOLD").map_or(false, |v| v == "pause");
    // RT_IX_HOLD=imagine (diagnosis only): the driver imagines the click that opened the book as
    // the task begins (the learned gates are off)
    let imagine_oracle = std::env::var("RT_IX_HOLD").map_or(false, |v| v == "imagine");
    let sleep_every: usize = env("RT_SLEEP_EVERY", 10);
    let replays: usize = env("RT_REPLAY", 3);
    let mut ix: Option<IndexHc> = std::env::var("RT_HC").map_or(true, |v| v != "circuit").then(|| IndexHc::new(seed));
    // with areas (the default), the hippocampus always completes (no cue, no request; acetylcholine
    // sets the mode); RT_HC_AUTO=0: the imagine-to-reinstate route
    if let Some(x) = ix.as_mut() {
        x.auto = std::env::var("RT_HC_AUTO").map_or(net.brain.is_some(), |v| v == "1");
    }
    if let Some(x) = ix.as_mut() {
        x.gates_off = ix_oracle || std::env::var("RT_IX_HOLD").map_or(false, |v| v == "pause" || v == "imagine");
        x.hold_off = hold_oracle;
        // RT_RETRIEVE=imagine (default): the cortex's predictions of a mouse motion can be
        // imagined instead of done, and what is imagined cues the hippocampus; =pause: one
        // retrieve gate, the hippocampus going back to the recent pause it recalls best; =hold:
        // the striatum's hold and reinstate gates
        let mode = std::env::var("RT_RETRIEVE").unwrap_or_else(|_| "imagine".into());
        if mode != "hold" && !hold_oracle {
            x.bookmarks = true;
            x.hold_off = true;
            x.pause_gate = mode == "pause";
        }
        // RT_THETA=on: storing goes on while retrieving (default off: it files the retelling
        // under the reinstated context, and those rows then win recall)
        x.theta = std::env::var("RT_THETA").map_or(false, |v| v == "on");
    }
    // the hippocampus hears word `w`: the index learns its boundary and (with `store`) stores
    // the event; the circuit stores it with `cue` when `store`
    // what the eye fixates now (the book's word in the task, else nothing): with the heard
    // word, the association area's input
    let mut eye_now = zero();
    // the association area's assembly for the last sound heard, and whether the book is open:
    // part of what the striatum sees
    let mut last_assoc: BitVector = zero();
    let mut prev_assoc: BitVector = zero();
    // what the hippocampus last gave the cortex (its output, as the striatum sees it)
    let mut last_recall = zero();
    // the place the hippocampus recalled for what is held (in "find" trials)
    let mut recalled_place: Option<usize> = None;
    macro_rules! hc_hear {
        ($w:expr, $cue:expr, $store:expr, $explore:expr, $own:expr) => {{
            let w: usize = $w;
            let cells = net.assoc_of(&eye_now, &ear.codes[w], $explore, &mut rng);
            // the content keys: what is seen while a book is in view (the words at that place),
            // else what is heard
            let heard_only = if eye_now.count_ones() == 0 { cells.clone() } else { net.assoc_of(&eye_now, &zero(), false, &mut rng) };
            let fired = match ix.as_mut() {
                Some(x) => x.hear(w, &ear.codes[w], &cells, &heard_only, $store, $own),
                None => {
                    if $store {
                        net.hc.store_split(&ones(&ear.codes[w]), $cue, &ones(&ear.codes[w]));
                    }
                    false
                }
            };
            prev_assoc = std::mem::replace(&mut last_assoc, cells);
            td_step!(fired, $explore);
            // the prefrontal gate: load what was just heard into working memory, or keep
            if !fixed_gate && striatum.choose(CH_LOAD, 2, if $explore { Some(&mut rng) } else { None }) == 1 {
                net.wm.load(0, &ear.codes[w]);
            }
        }};
    }
    // a new step for the striatum (after a sound): the TD update, then the hippocampal gates:
    // reinstate what is held, or (at a boundary) hold where this episode begins
    // the context moves every step with what the cortex represents: what is heard (in `hear`),
    // or when nothing is, its plan (a word recalled or meant, said or not)
    macro_rules! cortex_drift {
        ($plan:expr) => {{
            if ix.is_some() && ($plan).count_ones() > 0 {
                let cells = ones(&net.assoc_of(&eye_now, $plan, false, &mut rng));
                ix.as_mut().unwrap().drift_with(&cells);
            }
        }};
    }
    macro_rules! td_step {
        ($fired:expr, $explore:expr) => {{
            let (retr, held) = ix.as_ref().map_or((false, false), |x| (x.request, x.held.is_some()));
            let held_ctx = ix.as_ref().and_then(|x| x.held.clone()).unwrap_or_default();
            let eye4 = match net.brain.as_mut() {
                Some(b) => b.encode(EYE, &eye_now, false),
                None => if eye_now.count_ones() == 0 { zero() } else { net.l4e.encode(&eye_now) },
            };
            let state = state_groups(&net.wm.content(), &last_assoc, &prev_assoc, &net.rec23, &net.pons, &eye4, &last_recall, &held_ctx, recalled_place);
            striatum.begin_groups(&state, $explore, &mut rng);
            if let Some(x) = ix.as_mut() {
                if !x.gates_off {
                    // reinstating is possible only when not already retrieving (it would only
                    // start the sequence over)
                    // the held content's tag is kept with it, and re-armed when it is used, so the
                    // dopamine that follows using it reaches the decision to hold it
                    if x.pause_gate {
                        // whether to go back now (the striatum); where to (the hippocampus: the
                        // recent pause it recalls best)
                        if (!retr || restart) && striatum.choose(CH_REINSTATE, 2, if $explore { Some(&mut rng) } else { None }) == 1 {
                            let pause = ones(&net.assoc_of(&zero(), &silence, false, &mut rng));
                            if let Some(r) = x.pause_row(&pause) {
                                x.reinstate_row(r);
                            }
                        }
                    } else if held && (!retr || restart) && striatum.choose(CH_REINSTATE, 2, if $explore { Some(&mut rng) } else { None }) == 1 {
                        x.reinstate();
                        striatum.retag(CH_HOLD);
                    } else if $fired && !x.hold_off && striatum.choose(CH_HOLD, 2, if $explore { Some(&mut rng) } else { None }) == 1 {
                        x.hold();
                        striatum.maintain(CH_HOLD);
                    }
                }
            }
        }};
    }
    macro_rules! hc_recall {
        ($cue:expr) => {{
            let r = match ix.as_mut() {
                Some(x) => x.recall(),
                None => net.recall($cue),
            };
            last_recall = r.clone();
            r
        }};
    }
    // the index's boundary cell judges surprise against the plan
    macro_rules! hc_plan {
        ($st:expr) => {
            if let Some(x) = ix.as_mut() {
                x.plan = $st.plan.clone();
            }
        };
    }
    let tell_apart = std::env::var("RT_TELL").map_or(false, |v| v == "apart");
    let time = std::env::var("RT_HC_TIME").map_or(false, |v| v == "1");
    // RT_PRACTICE: the share of training tasks the network does itself, hearing its own words,
    // with the teacher's word as the target (practice with correction)
    let practice_p: f64 = env("RT_PRACTICE", 0.0);
    let (n_train, n_test): (usize, usize) = (env("RT_TRAIN", 3000), env("RT_TEST", 300));
    println!("vocabulary {}; {} training trials, {} test trials; gate {}", vocab.len(), n_train, n_test, if fixed_gate { "fixed (diagnosis)" } else { "learned" });
    let t0 = std::time::Instant::now();
    // results per (instruction read/tell, book), at test; and with the prefrontal content removed
    let mut res = [[Tally::default(); 3]; 2];
    let mut res_noinstr = [[Tally::default(); 3]; 2];
    let mut train_right = (0usize, 0usize);
    let mut practice_right = (0usize, 0usize);
    let mut corrections = 0usize;
    let mut stopped_itself = 0usize;
    // invented stories at test: (the story heard just before, what it said); every story the
    // network heard or saw in training
    let mut invented: Vec<(Vec<&str>, Vec<&str>)> = Vec::new();
    let mut heard_stories: std::collections::HashSet<String> = std::collections::HashSet::new();
    // the hippocampal recall's own accuracy while telling: (recalled the wanted word, steps)
    let mut hc_right = (0usize, 0usize);
    let mut hc_bits = 0usize;
    // the last 500 training trials (the teacher's words heard): recall right, and the plan
    // right per (instruction, book)
    let mut hc_train = (0usize, 0usize);
    let mut trace: usize = env("RT_TRACE", 0);
    // RT_TRACE_NOINSTR=1: trace the trials with the prefrontal content removed instead
    let trace_noinstr = std::env::var("RT_TRACE_NOINSTR").map_or(false, |v| v == "1");
    // RT_TRACE_ALL=1: trace every kind of trial, not only telling with the book closed
    let trace_all = std::env::var("RT_TRACE_ALL").map_or(false, |v| v == "1");
    let mut by_cond = [[(0usize, 0usize); 3]; 2];
    let books = [Book::Same, Book::Other, Book::Closed];
    // touch (the default; RT_TOUCH=0 off): a fingertip's window over the place under the cursor
    // and its two neighbours, as the skin's receptors report it:
    // - a vibration mode for what is under the cursor (its kind: the desk's book, a shelf's
    //   book, or nothing), as fast receptors report texture;
    // - an edge on the side where a neighbour differs, as slow receptors report edges: so where
    //   a thing begins and ends, and whether the cursor is inside it or at its border;
    // - while moving, the motion itself.
    // Each part is sparse (the mode 16 bits, an edge 8), so they add up.
    let touch_on = std::env::var("RT_TOUCH").map_or(true, |v| v != "0");
    let modes = Encoder::new(3, BITS, ACTIVE / 2, &mut StdRng::seed_from_u64(seed ^ 0x7E47)).codes;
    let edges = Encoder::new(2, BITS, ACTIVE / 4, &mut StdRng::seed_from_u64(seed ^ 0xED6E)).codes;
    // what lies at each place: 1 the desk's book, 2 a shelf's book, 0 nothing
    let kinds: Vec<usize> = (0..npos).map(|p| if p == 0 { 1 } else { 2 }).collect();
    macro_rules! touch {
        ($sensed:expr, $pos:expr, $kinds:expr) => {{
            let sensed: &BitVector = $sensed;
            let kinds: &[usize] = $kinds;
            let pos: usize = $pos;
            if !touch_on {
                sensed.clone()
            } else {
                // what moved (not the silence of nothing sensed)
                let mut v = if sensed.as_words() == silence.as_words() { zero() } else { sensed.clone() };
                let mut add = |c: &BitVector| {
                    for (w, x) in v.as_words_mut().iter_mut().zip(c.as_words()) {
                        *w |= x;
                    }
                };
                if kinds[pos] > 0 {
                    add(&modes[kinds[pos]]);
                }
                if pos > 0 && kinds[pos - 1] != kinds[pos] {
                    add(&edges[0]);
                }
                if pos + 1 < kinds.len() && kinds[pos + 1] != kinds[pos] {
                    add(&edges[1]);
                }
                if v.count_ones() == 0 {
                    silence.clone()
                } else {
                    v
                }
            }
        }};
    }
    // speech play (with areas; RT_SPEECH_PLAY sounds, default 600, 0 none), before any teaching,
    // as an infant's canonical babbling: the network makes a sound (a random command to the vocal
    // tract) and hears it. The ear's area and column hear it, and the voice's area holds the
    // command issued (its efference copy), so the two are active together and an auditory–motor
    // association area grows between them (the dorsal speech stream). Half the time a caregiver
    // answers with the same sound, heard in their voice (contingent replies shape babbling).
    // Then a word heard alone evokes, through that association, the command that says it: the
    // mirror for speech, a learned route from hearing to saying (imitation).
    let n_speech: usize = env("RT_SPEECH_PLAY", 600);
    if let (true, Some(_)) = (n_speech > 0, net.brain.as_ref()) {
        for _ in 0..n_speech {
            let w = rng.gen_range(0..tract.len());
            let cmd = tract.command(w).clone();
            let heard = tract.articulate(&cmd).map_or_else(|| silence.clone(), |x| sounds[x].clone());
            let answer = rng.gen_bool(0.5);
            let b = net.brain.as_mut().unwrap();
            b.areas[VOICE].motor.as_mut().unwrap().observe(&cmd, &heard, &mut rng);
            let a4 = b.encode(EAR, &heard, true);
            b.sense_step(EAR, &a4, None, true);
            b.associate_codes(&[zero(), a4, zero(), cmd], true);
            if answer {
                let a4 = b.encode(EAR, &heard, true);
                b.sense_step(EAR, &a4, None, true);
            }
        }
        // imitation: each word heard alone, through the ear–voice association, to a command
        let b = net.brain.as_ref().unwrap();
        let mut right = 0;
        for w in 0..sounds.len().min(tract.len()) {
            let e = b.encode_only(EAR, &sounds[w]);
            right += (tract.articulate(&b.mirror(EAR, &e, VOICE)) == Some(w)) as usize;
        }
        println!("speech play: {n_speech} sounds babbled; a word heard evokes the command that says it for {right} of {} words (the ear–voice association)", sounds.len().min(tract.len()));
    }
    // RT_PLAY: trials of play before any teaching (default 300; 0 none). No teacher, no story: the cursor
    // starts somewhere, and the network moves (often babbling) and clicks. A click on a book opens
    // it: the eye sees its page. That change, made by its own act, is rewarding in itself (as
    // contingent sensory change is for an infant), so the basal ganglia learn when a click pays,
    // and the cortex what each act does to what is seen and felt.
    let n_play: usize = env("RT_PLAY", 300);
    let mut play_opened = 0usize;
    // (clicks on a book, clicks in all) in the first and the last half of play, chosen (not babbled)
    let mut play_clicks = [(0usize, 0usize); 2];
    for play_trial in 0..n_play {
        cursor = rng.gen_range(0..npos);
        // which places hold a book this time (each half the time, at least one)
        let mut present: Vec<bool> = (0..npos).map(|_| rng.gen_bool(0.5)).collect();
        if !present.iter().any(|&x| x) {
            let k = rng.gen_range(0..npos);
            present[k] = true;
        }
        let play_kinds: Vec<usize> = (0..npos).map(|p| if !present[p] { 0 } else if p == 0 { 1 } else { 2 }).collect();
        let mut sensed = silence.clone();
        let mut seen_motion: Option<usize> = None;
        eye_now = zero();
        for _ in 0..12 {
            let mut eye_in = cursor_seen[cursor].clone();
            if let Some(m) = seen_motion {
                for (w, x) in eye_in.as_words_mut().iter_mut().zip(motion_seen[m].as_words()) {
                    *w |= x;
                }
            }
            eye_now = eye_in.clone();
            td_step!(false, true);
            // (an empty place feels like nothing)
            let felt_now = touch!(&sensed, cursor, &play_kinds);
            let st = if net.brain.is_some() {
                net.felt = felt_now;
                net.step(&eye_in, &silence, &zero(), true, 0, &mut rng)
            } else {
                net.step(&eye_in, &felt_now, &zero(), true, 0, &mut rng)
            };
            let mut cands: Vec<(usize, BitVector)> = Vec::new();
            let proposals = match (net.streams.as_ref(), net.top.as_ref()) {
                _ if net.selforg => net.brain.as_mut().unwrap().goal_proposals().into_iter().map(|(_, p, b)| (p, b)).collect(),
                (Some(sx), _) => sx.visuo_motor.l5.proposals(),
                (None, Some(t)) => t.l5.proposals(),
                _ => net.l5.proposals(BITS),
            };
            for (pat, _) in proposals {
                if !undecoded {
                    let best = (0..MOUSE).max_by_key(|&m| overlap(&pat, &motion[m])).unwrap();
                    if 2 * overlap(&pat, &motion[best]) >= ACTIVE as u32 && !cands.iter().any(|c| c.0 == best) {
                        cands.push((best, pat));
                    }
                } else if cands.len() < 8 && !cands.iter().any(|c| overlap(&c.1, &pat) * 2 >= ACTIVE as u32) {
                    // every distinct proposal is offered as it is (the motor area makes of it what
                    // it can); the motion it is nearest is kept for the readouts only
                    let label = (0..MOUSE).max_by_key(|&m| overlap(&pat, &motion[m])).unwrap();
                    cands.push((label, pat));
                }
            }
            // babbling, often: half the steps
            let babbled = cands.is_empty() || rng.gen_bool(0.5);
            let command: Option<BitVector> = if babbled {
                let m = rng.gen_range(0..MOUSE);
                striatum.did_among(CH_ACT, &[ones(&motion[m]).into_iter().map(|b| ACT_BASE + b).collect()]);
                Some(mouse.command(m).clone())
            } else {
                let offer: Vec<Vec<Vec<usize>>> = cands.iter().map(|c| vec![ones(&c.1).into_iter().map(|b| ACT_BASE + b).collect()]).collect();
                striatum.choose_among(CH_ACT, &offer, Some(&mut rng)).and_then(|i| hand!().plan(&cands[i].1))
            };
            let done = command.as_ref().and_then(|c| mouse.articulate(c));
            if let (Some(c), Some(m)) = (command.as_ref(), done) {
                hand!().observe(c, &motion[m], &mut rng);
            }
            if done == Some(CLICK) && !babbled {
                let half = (play_trial * 2 >= n_play) as usize;
                play_clicks[half].0 += present[cursor] as usize;
                play_clicks[half].1 += 1;
            }
            // a click opens the book under the cursor; on an empty place, or a move against the
            // edge, nothing happens and nothing is sensed
            let blocked = matches!((done, cursor), (Some(LEFT), 0)) || done == Some(RIGHT) && cursor + 1 == npos || done == Some(CLICK) && !present[cursor];
            let next = if blocked { silence.clone() } else { done.map_or(silence.clone(), |m| motion[m].clone()) };
            if let (Some(b), Some(m), false) = (net.brain.as_mut(), done, blocked) {
                let e = b.encode(ACTS, &motion_seen[m], true);
                let h = b.encode(HAND, &motion[m], true);
                b.associate_codes(&[zero(), zero(), h, zero(), e], true);
            }
            net.learn_motor(&st, &next, &mut rng);
            if done.is_some() {
                striatum.reward(-EFFORT);
            }
            seen_motion = if blocked { None } else { done };
            sensed = next;
            match done {
                Some(LEFT) => cursor = cursor.saturating_sub(1),
                Some(RIGHT) => cursor = (cursor + 1).min(npos - 1),
                Some(CLICK) => {
                    // the page appears: its own act changed what it sees
                    play_opened += 1;
                    striatum.reward(ONE as i32 / 2);
                    td_step!(true, true);
                    // the page: a text's first words come into view, one after another (the
                    // cortex's columns learn that after a click on a book, words begin)
                    let text = story(&mut rng);
                    net.felt = zero();
                    for w in text.iter().take(3) {
                        eye_now = eye.codes[id(w)].clone();
                        net.step(&eye_now.clone(), &silence, &zero(), true, 0, &mut rng);
                    }
                    break;
                }
                _ => {}
            }
        }
        striatum.end(true, &mut rng);
    }
    net.felt = zero();
    eye_now = zero();
    if n_play > 0 {
        println!(
            "play: {n_play} trials, a book opened in {play_opened}; chosen clicks on a book: first half {} of {}, second half {} of {} (books on half the places)",
            play_clicks[0].0, play_clicks[0].1, play_clicks[1].0, play_clicks[1].1
        );
    }
    for trial in 0..n_train + 2 * n_test {
        let testing = trial >= n_train;
        let no_instr = trial >= n_train + n_test;
        let learn = !testing;
        let s = story(&mut rng);
        // the story's context (a lateral-EC-like pattern as large as the two-word cue), new for
        // each story told
        let story_ctx: u64 = rng.next_u64();
        // 1. the teacher tells the story (with RT_STORY=book, reads it from a book opened with a
        // click the network sees, an event the hippocampus stores); the network predicts each
        // next word and stores it
        if story_book {
            // (with areas, the click is seen, not heard: an act, as the cortex represents acts)
            let cells = if net.brain.is_some() { seen_assembly!(&motion_seen[CLICK]) } else { act_assembly!(CLICK) };
            if ix.is_some() {
                let before = ix.as_ref().unwrap().mem.rows();
                let fired = ix.as_mut().unwrap().hear(quiet, &motion[CLICK], &cells, &cells, true, false);
                // the row that holds this story's opening (diagnosis)
                story_open_row = (ix.as_ref().unwrap().mem.rows() > before).then_some(before as u32);
                prev_assoc = std::mem::replace(&mut last_assoc, cells);
                td_step!(fired, learn);
            }
            // the cortex: after the book opens, the first word
            let st = if net.brain.is_some() {
                net.step(&motion_seen[CLICK], &zero(), &zero(), learn, 0, &mut rng)
            } else {
                net.step(&zero(), &motion[CLICK], &zero(), learn, 0, &mut rng)
            };
            if learn {
                net.learn(&st, &ear.codes[id(s[0])], &mut rng);
            }
        }
        let mut prev: Option<BitVector> = None;
        let mut prev2: Option<BitVector> = None;
        for t in 0..s.len() {
            let w = id(s[t]);
            // the network sees the word the teacher reads (a book open), or nothing (told)
            eye_now = if story_book { eye.codes[w].clone() } else { zero() };
            let heard = &ear.codes[w];
            let cue = Net::hc_context(prev.as_ref(), prev2.as_ref(), &silence, story_ctx, time.then_some(t));
            if ix.is_some() {
                // RT_IX_HOLD=oracle (diagnosis only): hold the context in force at the story's
                // first word, instead of the learned gate
                if t == 0 && hold_oracle {
                    let x = ix.as_mut().unwrap();
                    x.held = Some(x.ctx.clone());
                    x.request = false;
                }
                hc_hear!(w, &cue, true, learn, false);
            }
            let rec = hc_recall!(&cue);
            let st = net.step(&eye_now.clone(), heard, &rec, learn, 0, &mut rng);
            hc_plan!(st);
            // with RT_REWARD=all, a heard word the cortex predicted is a reward too (the default
            // rewards only outcomes; the cortex learns its predictions locally)
            if learn && reward_all {
                let next = if t + 1 < s.len() { &ear.codes[id(s[t + 1])] } else { &silence };
                if overlap(&st.plan, next) >= 24 {
                    striatum.reward(ONE as i32);
                }
            }
            if learn && t + 1 < s.len() {
                net.learn(&st, &ear.codes[id(s[t + 1])], &mut rng);
            } else if learn && stop_learned {
                // the teacher stops: silence is heard next
                net.learn(&st, &silence, &mut rng);
            }
            if ix.is_none() {
                hc_hear!(w, &cue, true, learn, false);
            }
            prev2 = prev.replace(heard.clone());
        }
        if trace > 0 && testing && ix.is_none() {
            // probe: cue the hippocampus with the true preceding words, right after hearing
            let (mut p1, mut p2): (Option<BitVector>, Option<BitVector>) = (None, None);
            let mut line = String::new();
            for t in 0..s.len().min(8) {
                let r = net.hc.recall(&Net::hc_context(p1.as_ref(), p2.as_ref(), &silence, story_ctx, time.then_some(t)));
                let mut v = zero();
                for &b in r.ec.iter().filter(|&&b| b < BITS) {
                    v.bit_set(b);
                }
                line += &format!(" {}:{}/{}(s{})", s[t], overlap(&v, &ear.codes[id(s[t])]), v.count_ones(), r.strength);
                p2 = p1.replace(ear.codes[id(s[t])].clone());
            }
            println!("  probe right after hearing:{line}");
        }
        // the book is closed again
        eye_now = zero();
        // the teacher pauses: silence is heard (a sound, and a natural event boundary), and
        // stored like any sound (with bookmarks, a pause row is where an episode can be gone
        // back to)
        if ix.is_some() {
            let cells = net.assoc_of(&zero(), &silence, false, &mut rng);
            let keep = ix.as_ref().unwrap().pause_gate;
            let fired = ix.as_mut().unwrap().hear(quiet, &silence, &cells, &cells, keep, false);
            prev_assoc = std::mem::replace(&mut last_assoc, cells);
            td_step!(fired, learn);
        }
        let listen_log: Vec<bool> = ix.as_mut().map_or(Vec::new(), |x| x.log.drain(..).map(|l| l.0).collect());
        // 2. the book and the instruction
        // now and then a book on the shelf is replaced by a new one
        if shelf_k > 0 && rng.gen_bool(0.1) {
            let k = rng.gen_range(0..shelf_k);
            shelf[k] = shelf_book(&mut rng, &shelf, k);
        }
        let find = shelf_k > 0 && rng.gen_bool(0.25);
        let find_at = if find { rng.gen_range(0..shelf_k) } else { 0 };
        let make = !find && make_on && rng.gen_bool(0.25);
        let read = !find && !make && rng.gen_bool(0.5);
        // RT_TELL=apart: in training the teacher retells with the book closed or open at
        // another story, never at the same one (the test keeps all three)
        let book = if make || find {
            Book::Closed
        } else if read {
            [Book::Same, Book::Other][rng.gen_range(0..2)]
        } else if tell_apart && !testing {
            [Book::Other, Book::Closed][rng.gen_range(0..2)]
        } else {
            books[rng.gen_range(0..3)]
        };
        let mut p = match book {
            Book::Same => s.clone(),
            Book::Other => story(&mut rng),
            Book::Closed => Vec::new(),
        };
        let verb = if find { "find" } else if make { "make" } else if read { "read" } else { "tell" };
        let (mut ip, mut ip2): (Option<BitVector>, Option<BitVector>) = (None, None);
        let instr = ["now", verb, if find { animal_of(&shelf[find_at]) } else if make { "one" } else { IT }];
        for (it, w) in instr.into_iter().enumerate() {
            let wi = id(w);
            let cue = Net::hc_context(ip.as_ref(), ip2.as_ref(), &silence, story_ctx, time.then_some(it));
            hc_hear!(wi, &cue, store_all, learn, false);
            ip2 = ip.replace(ear.codes[wi].clone());
            if fixed_gate && w == verb {
                net.wm.load(0, &ear.codes[wi]);
            }
            // the cortex learns the instruction like any speech: the next sound, and after its last
            // word the teacher's pause
            let st = net.step(&zero(), &ear.codes[wi], &zero(), learn, 0, &mut rng);
            hc_plan!(st);
            let next = if it + 1 < instr.len() { ear.codes[id(instr[it + 1])].clone() } else { silence.clone() };
            if learn {
                net.learn(&st, &next, &mut rng);
            }
            if learn && reward_all && overlap(&st.plan, &next) >= 24 {
                striatum.reward(ONE as i32);
            }
        }
        if no_instr {
            net.wm.clear();
        }
        // 3. the task: the teacher does it (training), or the network does (test)
        if make && testing {
            // free speech: the network makes one up, hearing itself, until it ends a story
            if !no_instr {
                let mut said_words: Vec<&str> = Vec::new();
                let (mut prev, mut prev2): (Option<BitVector>, Option<BitVector>) = (None, None);
                let mut stopped = false;
                for t in 0..40 {
                    let cue = Net::hc_context(prev.as_ref(), prev2.as_ref(), &silence, story_ctx, time.then_some(t));
                    let rec = hc_recall!(&cue);
                    let heard = prev.clone().unwrap_or_else(zero);
                    let st = net.step(&zero(), &heard, &rec, false, make_temp, &mut rng);
                    hc_plan!(st);
                    match say!(&st.plan) {
                        Some(w) if w == quiet => {
                            // it says nothing more: the story is over
                            stopped = true;
                            break;
                        }
                        Some(w) => {
                            said_words.push(vocab[w]);
                            hc_hear!(w, &cue, store_all, false, true);
                            prev2 = prev.replace(ear.codes[w].clone());
                        }
                        None => {
                            said_words.push("…");
                            prev2 = if heard_plan { prev.replace(st.plan.clone()) } else { prev.take() };
                        }
                    }
                    if !stop_learned && said_words.ends_with(&["home", "."]) {
                        stopped = true;
                        break;
                    }
                }
                stopped_itself += stopped as usize;
                invented.push((s.clone(), said_words));
            }
            continue;
        }
        // the teacher's new story, when asked to make one
        let q = if make { story(&mut rng) } else { Vec::new() };
        if learn {
            heard_stories.insert(s.join(" "));
            heard_stories.insert(p.join(" "));
            heard_stories.insert(q.join(" "));
        }
        let practice = learn && rng.gen_bool(practice_p);
        let mut prev: Option<BitVector> = None;
        let mut prev2: Option<BitVector> = None;
        let mut tally = Tally::default();
        let mut trace_recall: Vec<bool> = Vec::new();
        let mut said_seq: Vec<&str> = Vec::new();
        if let Some(x) = ix.as_mut() {
            x.debug = trace > 0 && testing && !read && !make && book == Book::Closed && no_instr == trace_noinstr;
            if x.debug {
                println!("  story rows end at {} (stored so far)", x.mem.rows());
            }
        }
        let mut book_open = !reach_learned;

        let mut eye_pos = 0usize;
        let mut debug_right = 0usize;
        let mut reached_at: Option<usize> = None;
        // in "find" trials, where was the book about what is held? The hippocampus completes the
        // episode from the content and gives the place, which joins what the striatum sees
        recalled_place = None;
        if find {
            let held = net.wm.content();
            // diagnosis: does working memory hold the animal asked for?
            find_diag.0 += 1;
            find_diag.1 += (overlap(&held, &ear.codes[id(animal_of(&shelf[find_at]))]) >= 24) as usize;
            // the cue is the entorhinal input: what is active in the cortex now (the last word
            // heard, the animal) with the context; working memory never cues it directly
            if let Some(x) = ix.as_ref() {
                recalled_place = x.where_of(&ones(&last_assoc));
            }
            find_diag.2 += (recalled_place == Some(find_at)) as usize;
            find_diag.3 += recalled_place.is_some() as usize;
        }
        // diagnosis: layer 2/3's activity at the wait, by instruction
        if learn && trial + 300 >= n_train && (read || (!make && !find)) {
            let k = if read { 0 } else { 1 };
            wait_states[k].push(ones(&net.rec23));
        }
        // the teacher waits before starting (three pauses; in "find" trials long enough to reach
        // any place on the shelf); the network may use the mouse meanwhile. Once the teacher
        // starts, it goes on whether a book is open or not.
        // (with a book to open, long enough for the teacher to show how from anywhere)
        let waits = if find || read { 3 + shelf_k } else { 3 };
        // where the book to open lies, if there is one to open (the teacher shows it)
        let target_pos: Option<usize> = if read { Some(0) } else if find { Some(1 + find_at) } else { None };
        let mut sensed = silence.clone();
        // what the eye saw happen at the last step (an act on the screen), and the network's own
        // acts in this wait
        let mut seen_motion: Option<usize> = None;
        let mut own_acts: Vec<usize> = Vec::new();
        // the teacher's cursor (RT_TEACHER_CURSOR=1): where it is once the teacher starts showing,
        // and whether the teacher has opened their book
        let mut tcur: Option<usize> = None;
        let mut tdone = false;
        if !book_open {
            for k in 0..waits {
                // the eye sees the screen: where the cursor is (the striatum's visual input)
                eye_now = cursor_seen[cursor].clone();
                if ix.is_some() {
                    let cells = net.assoc_of(&zero(), &silence, false, &mut rng);
                    let fired = ix.as_mut().unwrap().hear(quiet, &silence, &cells, &cells, false, false);
                    prev_assoc = std::mem::replace(&mut last_assoc, cells);
                    td_step!(fired, learn);
                } else {
                    prev_assoc = std::mem::replace(&mut last_assoc, net.assoc_of(&zero(), &silence, false, &mut rng));
                    td_step!(false, learn);
                }
                // the cortex: what it expects to sense next, seeing the cursor and sensing the
                // last motion (or silence). Its proposals are its layer 5 cells' predictions that
                // are a mouse motion; the basal ganglia release one or none, and the motor area
                // turns it into a command. While learning, the mouse also moves on its own now
                // and then (motor babbling), so the cortex learns what follows what.
                // (and the teacher's cursor, if it is on the screen: a cursor looks the same
                // whoever's it is)
                let mut scene = cursor_seen[cursor].clone();
                if let Some(t) = tcur {
                    for (w, x) in scene.as_words_mut().iter_mut().zip(cursor_seen[t].as_words()) {
                        *w |= x;
                    }
                }
                let eye_in = match seen_motion.filter(|_| teach_seen) {
                    Some(m) => {
                        let mut v = scene.clone();
                        for (w, x) in v.as_words_mut().iter_mut().zip(motion_seen[m].as_words()) {
                            *w |= x;
                        }
                        v
                    }
                    None => scene,
                };
                // with areas, the hand feels its motion in its own area; the ear hears nothing
                let felt_now = touch!(&sensed, cursor, &kinds);
                let st = if net.brain.is_some() {
                    net.felt = felt_now;
                    net.step(&eye_in, &silence, &zero(), learn, 0, &mut rng)
                } else {
                    net.step(&eye_in, &felt_now, &zero(), learn, 0, &mut rng)
                };
                // curiosity as noise: the learning progress where the network is now (from the
                // steps before) loosens the choice a little, up to ±2 more; no reward
                let here = {
                    let wm: u64 = ones(&net.wm.content()).iter().take(4).fold(0u64, |h, &b| (h ^ b as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    ((wm ^ (cursor as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)) >> 58) as usize
                };
                if curiosity_noise {
                    let p = (misses[here] & 0xFF00).count_ones() as i32 - (misses[here] & 0xFF).count_ones() as i32 - 1;
                    striatum.set_noise(p.clamp(0, 2));
                }
                // cortical curiosity, as the locus coeruleus's tonic gain: where the cortex is
                // getting better at predicting what comes, choices loosen a little; while reward is
                // expected (the critic's value above zero), they do not (phasic mode, exploit)
                if curiosity_cortex {
                    let p = match (net.streams.as_ref(), net.top.as_ref()) {
                        _ if net.selforg => net.brain.as_mut().unwrap().goal_progress(),
                        (Some(sx), _) => sx.visuo_motor.l5.progress(),
                        (None, Some(t)) => t.l5.progress(),
                        _ => net.l5.progress(),
                    }
                    .clamp(0, 2);
                    let n = if lc_damp && striatum.value() > 0 { 0 } else { p };
                    striatum.set_noise(n);
                    curiosity_stats.0 += n as usize;
                    curiosity_stats.1 += 1;
                }
                let mut cands: Vec<(usize, BitVector)> = Vec::new();
                let proposals = match (net.streams.as_ref(), net.top.as_ref()) {
                    _ if net.selforg => net.brain.as_mut().unwrap().goal_proposals().into_iter().map(|(_, p, b)| (p, b)).collect(),
                    (Some(sx), _) => sx.visuo_motor.l5.proposals(),
                    (None, Some(t)) => t.l5.proposals(),
                    _ => net.l5.proposals(BITS),
                };
                for (pat, _) in proposals {
                    if !undecoded {
                        let best = (0..MOUSE).max_by_key(|&m| overlap(&pat, &motion[m])).unwrap();
                        if 2 * overlap(&pat, &motion[best]) >= ACTIVE as u32 && !cands.iter().any(|c| c.0 == best) {
                            cands.push((best, pat));
                        }
                    } else if cands.len() < 8 && !cands.iter().any(|c| overlap(&c.1, &pat) * 2 >= ACTIVE as u32) {
                        // every distinct proposal is offered as it is (the motor area makes of it what
                        // it can); the motion it is nearest is kept for the readouts only
                        let label = (0..MOUSE).max_by_key(|&m| overlap(&pat, &motion[m])).unwrap();
                        cands.push((label, pat));
                    }
                }
                // the teacher shows how (training, RT_TEACH=mouse, default): if the network has not
                // opened the book by the time the teacher needs the remaining steps, the teacher
                // moves the cursor to it and clicks, a step at a time. The network sees the cursor
                // move and senses each motion as it would its own; the cortex learns the sequence
                // as it learns the teacher's words. It is the teacher's act, not the network's:
                // the basal ganglia are not tagged.
                let from = if teacher_cursor { tcur.unwrap_or(cursor) } else { cursor };
                let shown: Option<usize> = target_pos.filter(|&tp| learn && teach_mouse && !tdone && (tcur.is_some() || waits - k <= from.abs_diff(tp) + 1)).map(|tp| {
                    if from > tp {
                        LEFT
                    } else if from < tp {
                        RIGHT
                    } else {
                        CLICK
                    }
                });
                mouse_stats[6] += shown.is_some() as usize;
                // imagine instead of do: one of the same proposals, not sent to the mouse but to
                // the hippocampus as a cue (what is imagined, as if sensed); recall goes back to
                // the row it matches best and carries on along the links
                if imagine_on && !cands.is_empty() && !ix.as_ref().map_or(true, |x| x.request) {
                    let offer: Vec<Vec<Vec<usize>>> = cands.iter().map(|c| vec![ones(&c.1).into_iter().map(|b| ACT_BASE + b).collect()]).collect();
                    if let Some(i) = striatum.choose_among(CH_IMAGINE, &offer, if learn { Some(&mut rng) } else { None }) {
                        // the act the proposal is (the motion it matched), as the cortex represents it
                        let cue = if net.brain.is_some() { ones(&imagined_assembly!(&cands[i].1)) } else { ones(&net.assoc_of(&zero(), &cands[i].1, false, &mut rng)) };
                        if let Some(x) = ix.as_mut() {
                            if x.auto {
                                // the cortex represents the imagined act; the hippocampus completes it
                                x.represent(&cue);
                                last_recall = x.complete();
                            } else if let Some(r) = x.best_row(&cue) {
                                x.reinstate_row(r);
                                // diagnosis: imagining the click while telling, did recall land on
                                // this story's opening?
                                if cands[i].0 == CLICK && !read && !make && !find {
                                    open_recall.0 += 1;
                                    open_recall.1 += (Some(r) == story_open_row) as usize;
                                }
                            }
                        }
                        imagined.0 += 1;
                        imagined.1 += (!read && !make && !find) as usize;
                    }
                }
                let command: Option<BitVector> = if let Some(m) = shown {
                    // RT_OBSERVE=1: the teacher's act, seen, is tagged as if done (the
                    // mirror system's view of it), so the dopamine that follows credits it and not
                    // whatever the network did before
                    if observe {
                        striatum.did_among(CH_ACT, &[ones(&motion[m]).into_iter().map(|b| ACT_BASE + b).collect()]);
                    }
                    None
                } else if learn && rng.gen_range(0..BABBLE) == 0 {
                    let m = rng.gen_range(0..MOUSE);
                    striatum.did_among(CH_ACT, &[ones(&motion[m]).into_iter().map(|b| ACT_BASE + b).collect()]);
                    mouse_stats[0] += 1;
                    Some(mouse.command(m).clone())
                } else {
                    let offer: Vec<Vec<Vec<usize>>> = cands.iter().map(|c| vec![ones(&c.1).into_iter().map(|b| ACT_BASE + b).collect()]).collect();
                    let ob = if read && cursor == 0 && !no_instr { Some(testing as usize) } else { None };
                    if let Some(k) = ob {
                        on_book[k][0] += 1;
                        on_book[k][1] += cands.iter().any(|c| c.0 == CLICK) as usize;
                        on_book[k][2] += cands.iter().any(|c| c.0 == LEFT) as usize;
                    }
                    let off = read && cursor != 0 && testing && !no_instr;
                    let names = ["L", "R", "C"];
                    let proposed: String = cands.iter().map(|c| names.get(c.0).copied().unwrap_or("?")).collect();
                    match striatum.choose_among(CH_ACT, &offer, if learn { Some(&mut rng) } else { None }) {
                        Some(i) => {
                            if off {
                                *off_book.entry(format!("{cursor}:{proposed}>{}", names.get(cands[i].0).copied().unwrap_or("?"))).or_default() += 1;
                            }
                            if let Some(k) = ob {
                                on_book[k][3] += (cands[i].0 == CLICK) as usize;
                                on_book[k][4] += (cands[i].0 == LEFT) as usize;
                            }
                            mouse_stats[1] += 1;
                            hand!().plan(&cands[i].1)
                        }
                        None => {
                            if off {
                                *off_book.entry(format!("{cursor}:{proposed}>-")).or_default() += 1;
                            }
                            None
                        }
                    }
                };
                striatum.set_noise(0);
                // the world: the mouse does what the command articulates
                // with the teacher's own cursor, the teacher's act moves that cursor, not the network's
                let done = if teacher_cursor && shown.is_some() { None } else { shown.or_else(|| command.as_ref().and_then(|c| mouse.articulate(c))) };
                if let (true, Some(m), Some(tp)) = (teacher_cursor, shown, target_pos) {
                    let t = tcur.get_or_insert(cursor);
                    match m {
                        LEFT => *t = t.saturating_sub(1),
                        RIGHT => *t = (*t + 1).min(npos - 1),
                        _ => {
                            if *t == tp {
                                tdone = true;
                                teacher_opened += 1;
                            }
                        }
                    }
                }
                if let (Some(c), Some(m)) = (command.as_ref(), done) {
                    hand!().observe(c, &motion[m], &mut rng);
                }
                // what is sensed is the act's effect on the screen: a move against the screen's
                // edge moves nothing, and nothing is sensed (RT_BLOCKED=0: the
                // motion is sensed anyway)
                // (likewise a click on nothing: nothing opens)
                let opens = if find { cursor >= 1 && cursor <= shelf_k } else { cursor == 0 && !p.is_empty() };
                let blocked = blocked_senses && (matches!((done, cursor), (Some(LEFT), 0)) || done == Some(RIGHT) && cursor + 1 == npos || done == Some(CLICK) && !opens);
                // (with RT_TEACH_SEEN=1 the teacher's act is not felt: it is seen, at the next step)
                let mut next = if blocked || (teach_seen && shown.is_some()) { silence.clone() } else { done.map_or(silence.clone(), |m| motion[m].clone()) };
                if let (Some(b), Some(m), false) = (net.brain.as_mut(), done.or(shown), blocked) {
                    if shown.is_none() {
                        // an own act, seen and felt: the association of seen acts and the hand
                        // learns the pair
                        if learn {
                            let e = b.encode(ACTS, &motion_seen[m], true);
                            let h = b.encode(HAND, &motion[m], true);
                            b.associate_codes(&[zero(), zero(), h, zero(), e], true);
                        }
                    } else if teach_seen {
                        // the teacher's act, seen only: the mirror (that association)
                        // evokes how it would feel, and that is what the cortex takes as come
                        let e = b.encode(ACTS, &motion_seen[m], false);
                        let felt = b.mirror(ACTS, &e, HAND);
                        mirror_stats.0 += 1;
                        mirror_stats.1 += (overlap(&felt, &motion[m]) * 2 >= ACTIVE as u32) as usize;
                        if felt.count_ones() > 0 {
                            next = felt;
                        }
                    }
                }
                seen_motion = if blocked { None } else { done.or(shown) };
                // the check: an own act, due for one (or every act, RT_CHECK=always), compared
                // with what the forward model expected it to do; no effect at all is a dip
                if let (Some(c), Some(m), true) = (command.as_ref(), done, learn && shown.is_none()) {
                    own_acts.push(m);
                    if check_mode == "always" || (check_mode == "1" && check_due[m]) {
                        check_due[m] = false;
                        let expected = hand!().predict(c).unwrap_or_else(zero);
                        checks.0 += 1;
                        if expected.count_ones() > 0 && overlap(&expected, &next) * 2 < ACTIVE as u32 {
                            checks.1 += 1;
                            striatum.reward(-(ONE as i32) / 2);
                        }
                    }
                }
                // the cortex learns what came; curiosity is the progress of its predictions here
                misses[here] = misses[here] << 1 | (overlap(&st.plan, &next) < 24) as u64;
                // misses 9..16 steps back minus the last 8 (no divide), beyond one of noise
                let progress = (misses[here] & 0xFF00).count_ones() as i32 - (misses[here] & 0xFF).count_ones() as i32 - 1;
                if learn {
                    net.learn_motor(&st, &next, &mut rng);
                    if progress > 0 && curiosity_on {
                        striatum.reward(progress * ONE as i32 / 8);
                        curiosity_sum += (progress * ONE as i32 / 8) as i64;
                    }
                    if done.is_some() && shown.is_none() {
                        striatum.reward(-EFFORT);
                    }
                }
                sensed = next;
                let mut opened: Option<usize> = None;
                match done {
                    Some(LEFT) => {
                        cursor = cursor.saturating_sub(1);
                        mouse_stats[2] += 1;
                    }
                    Some(RIGHT) => {
                        cursor = (cursor + 1).min(npos - 1);
                        mouse_stats[3] += 1;
                    }
                    Some(CLICK) => {
                        mouse_stats[4] += 1;
                        // a click on a book opens it: the story's book on the desk, a shelf's
                        // book at its place (in "find" trials)
                        opened = if find { (cursor >= 1 && cursor <= shelf_k).then(|| cursor - 1) } else { (cursor == 0 && !p.is_empty()).then_some(0) };
                        mouse_stats[5] += opened.is_some() as usize;
                    }
                    _ => {}
                }
                if let Some(d) = opened {
                    if find {
                        // the book at that place, from its first word; its place joins the
                        // hippocampus's context while it is in view
                        p = shelf[d].clone();
                    }
                    // a reward came: the network's own acts in this wait are due a check
                    for &a in &own_acts {
                        check_due[a] = true;
                    }
                    // the book, from its first word; the reach is an event boundary
                    book_open = true;
                    eye_now = p.first().map_or(zero(), |w| eye.codes[id(w)].clone());
                    reached_at = Some(d);
                    if let Some(x) = ix.as_mut() {
                        x.reach_event();
                        if find {
                            x.place = Some(d);
                        }
                    }
                    td_step!(true, learn);
                    if learn && !find {
                        let k = if read { 0 } else if make { 2 } else { 1 };
                        reach_delta[k].0 += striatum.last_delta() as i64;
                        reach_delta[k].1 += 1;
                    }
                    break;
                }
            }
        }
        // the hand is still once the wait is over
        net.felt = zero();
        // RT_WAIT_END=1: the wait is an episode of its own for the basal ganglia: its credit
        // ends with it, so the story's rewards do not reach the acts made while waiting
        if wait_end {
            striatum.end(learn, &mut rng);
        }
        if imagine_oracle && !read && !make {
            let cue = if net.brain.is_some() { ones(&seen_assembly!(&motion_seen[CLICK])) } else { ones(&act_assembly!(CLICK)) };
            if let Some(x) = ix.as_mut() {
                if let Some(r) = x.best_row(&cue) {
                    x.reinstate_row(r);
                }
            }
        }
        if pause_oracle && !read && !make {
            let pause = ones(&net.assoc_of(&zero(), &silence, false, &mut rng));
            if let Some(x) = ix.as_mut() {
                if let Some(r) = x.pause_row(&pause) {
                    x.reinstate_row(r);
                }
            }
        }
        // RT_IX_HOLD=oracle: the held start is reinstated as the task begins (after the wait)
        if ix_oracle && !read && !make {
            if let Some(x) = ix.as_mut() {
                if let Some(h) = x.held.clone() {
                    x.ctx = h;
                    x.last_recalled = None;
                    x.request = true;
                }
            }
        }
        let (debug_v0, debug_p0) = striatum.debug_now(CH_LOAD);
        // the teacher reads the book asked for in "find" trials
        let find_target: Vec<&str> = if find { shelf[find_at].clone() } else { Vec::new() };
        let target: &Vec<&str> = if find { &find_target } else if make { &q } else if read { &p } else { &s };
        for t in 0..target.len() {
            let seen = if book_open { p.get(eye_pos).map(|w| eye.codes[id(w)].clone()).unwrap_or_else(zero) } else { zero() };
            if book_open {
                eye_pos += 1;
            }
            eye_now = seen.clone();
            let cue = Net::hc_context(prev.as_ref(), prev2.as_ref(), &silence, story_ctx, time.then_some(t));
            let rec = hc_recall!(&cue);
            let heard = prev.clone().unwrap_or_else(zero);
            let st = net.step(&seen, &heard, &rec, learn, 0, &mut rng);
            hc_plan!(st);
            let want = id(target[t]);
            // what recall should give: the wanted word's sound (circuit) or its layer 4 cells (index)
            let want_rec = if ix.is_some() { net.assoc_of(&zero(), &ear.codes[want], false, &mut rng) } else { ear.codes[want].clone() };
            if !read && !make {
                trace_recall.push(overlap(&rec, &want_rec) >= 16);
                hc_right.0 += (overlap(&rec, &want_rec) >= 16) as usize;
                hc_right.1 += 1;
                hc_bits += rec.count_ones() as usize;
                if learn && trial + 500 >= n_train {
                    hc_train.0 += (overlap(&rec, &want_rec) >= 16) as usize;
                    hc_train.1 += 1;
                }
            }
            if learn && !make && trial + 500 >= n_train {
                let bi = books.iter().position(|b| *b == book).unwrap();
                let c = &mut by_cond[!read as usize][bi];
                c.0 += (overlap(&st.plan, &ear.codes[want]) >= 24) as usize;
                c.1 += 1;
            }
            if learn {
                let right = overlap(&st.plan, &ear.codes[want]) >= 24;
                train_right.0 += right as usize;
                train_right.1 += 1;
                net.learn(&st, &ear.codes[want], &mut rng);
                if practice {
                    // the network says it and hears itself
                    let said = say!(&st.plan).filter(|&w| w != quiet);
                    let ok = said == Some(want);
                    practice_right.0 += ok as usize;
                    practice_right.1 += 1;
                    if let Some(w) = said {
                        prev2 = prev.replace(ear.codes[w].clone());
                    } else {
                        prev2 = if heard_plan { prev.replace(st.plan.clone()) } else { prev.take() };
                    }
                    if heard_correction && !ok {
                        // the teacher says the right word aloud, and it is heard
                        prev2 = prev.replace(ear.codes[want].clone());
                        corrections += 1;
                    }
                    // the reward: no correction came (with the oracle: the unheard comparison)
                    if ok {
                        striatum.reward(ONE as i32);
                    }

                    if ix.is_some() {
                        // its own word was heard, then (if wrong) the teacher's
                        if let Some(w) = said {
                            hc_hear!(w, &cue, store_all, true, true);
                        }
                        if heard_correction && !ok {
                            hc_hear!(want, &cue, store_all, true, false);
                        }
                        if said.is_none() && !(heard_correction && !ok) {
                            cortex_drift!(&st.plan);
                        }
                    } else if store_all {
                        let w = if ok || heard_correction { Some(want) } else { said };
                        if let Some(w) = w {
                            hc_hear!(w, &cue, true, true, false);
                        }
                    }
                } else {
                    // the reward: the planned word was the one the teacher then said
                    if right {
                        striatum.reward(ONE as i32);
                        debug_right += 1;
                    }
                    // the teacher's word is heard
                    prev2 = prev.replace(ear.codes[want].clone());
                    hc_hear!(want, &cue, store_all, true, false);
                }
            } else {
                let said = say!(&st.plan).filter(|&w| w != quiet);
                if trace > 0 && (trace_all || (book == Book::Closed && !read)) && t < 6 && no_instr == trace_noinstr {
                    trace -= (t == 5) as usize;
                    let near = ear.decode(&st.plan).map_or("-", |w| vocab[w]);
                    println!(
                        "  trace {} book {} t={t} want {:<7} seen {:>2} heard {:>2} pfc {:>2} recall {:>2} plan bits {:>2} (nearest {:<7} overlap {:>2}) sources {:?} recall overlap {:>2} said {:?}",
                        if read { "read" } else { "tell" },
                        if book_open { "open" } else { "shut" },
                        target[t],
                        seen.count_ones(),
                        heard.count_ones(),
                        net.wm.content().count_ones(),
                        rec.count_ones(),
                        st.plan.count_ones(),
                        near,
                        overlap(&st.plan, &ear.codes[want]),
                        st.props.iter().map(|x| x.0).collect::<Vec<_>>(),
                        overlap(&rec, &want_rec),
                        said.map(|w| vocab[w])
                    );
                }
                tally.words += 1;
                said_seq.push(match say!(&st.plan) {
                    Some(w) if w == quiet => "<silence>",
                    Some(w) => vocab[w],
                    None => "…",
                });
                match said {
                    Some(w) => {
                        tally.right += (w == want) as usize;
                        tally.said_story += (w == id(s[t.min(s.len() - 1)]) && t < s.len()) as usize;
                        tally.said_book += p.get(t).map_or(false, |b| w == id(b)) as usize;
                        prev2 = prev.replace(ear.codes[w].clone());
                        hc_hear!(w, &cue, store_all, false, true);
                    }
                    None => {
                        tally.silent += 1;
                        prev2 = if heard_plan { prev.replace(st.plan.clone()) } else { prev.take() };
                        cortex_drift!(&st.plan);
                    }
                }
            }
        }
        eye_now = zero();
        if learn && stop_learned && !target.is_empty() {
            // the task is done and the teacher falls silent: silence is the next sound
            let cue = Net::hc_context(prev.as_ref(), prev2.as_ref(), &silence, story_ctx, time.then_some(target.len()));
            let rec = hc_recall!(&cue);
            let heard = prev.clone().unwrap_or_else(zero);
            let st = net.step(&zero(), &heard, &rec, true, 0, &mut rng);
            net.learn(&st, &silence, &mut rng);
        }
        if ix.is_some() {
            // the pause after the task (stored with bookmarks: the next story follows it)
            let cells = net.assoc_of(&zero(), &silence, false, &mut rng);
            let keep = ix.as_ref().unwrap().pause_gate;
            let fired = ix.as_mut().unwrap().hear(quiet, &silence, &cells, &cells, keep, false);
            prev_assoc = std::mem::replace(&mut last_assoc, cells);
            td_step!(fired, learn);
            let task_log: Vec<bool> = ix.as_mut().unwrap().log.drain(..).map(|l| l.0).collect();
            if trace > 0 && testing && !read && !make && book == Book::Closed && no_instr == trace_noinstr {
                trace -= 1;
                let marks = |l: &[bool], w: &[&str]| w.iter().zip(l.iter().chain(std::iter::repeat(&false))).map(|(w, &b)| if b { format!("{w}|") } else { w.to_string() }).collect::<Vec<_>>().join(" ");
                println!("  boundaries while listening: {}", marks(&listen_log, &s));
                println!("  boundaries in the task (instruction first): {}", task_log.iter().map(|&b| if b { '|' } else { '.' }).collect::<String>());
                println!("  story: {}\n  said:  {}", s.join(" "), said_seq.join(" "));
                println!("  retold: {:.0}% right; recall right on {}", 100.0 * tally.right as f64 / tally.words.max(1) as f64, trace_recall.iter().map(|&r| if r { '+' } else { '-' }).collect::<String>());
            }
        }
        // the trial ends: the last dopamine, and the traces clear
        if td_debug && learn && trial % 50 == 0 {
            println!(
                "  td trial {trial}: {} book {:?}, reached at {:?}, words right {} of {}, value at task start {:.2}, reach pref there {:+.2}",
                verb,
                books.iter().position(|b| *b == book),
                reached_at,
                debug_right,
                target.len(),
                debug_v0 as f64 / ONE as f64,
                debug_p0 as f64 / ONE as f64
            );
        }
        if learn && read {
            let k = reached_at.is_some() as usize;
            read_by_reach[k].0 += debug_right;
            read_by_reach[k].1 += target.len();
        }
        striatum.end(learn, &mut rng);
        recalled_place = None;
        if let Some(x) = ix.as_mut() {
            x.place = None;
        }
        if find {
            if testing && !no_instr {
                find_stats.0 += 1;
                find_stats.1 += (reached_at == Some(find_at)) as usize;
                find_stats.2 += reached_at.is_some() as usize;
                find_stats.3 += tally.words;
                find_stats.4 += tally.right;
            }
            continue;
        }
        if testing && !no_instr && !make && reach_learned {
            let bi = books.iter().position(|b| *b == book).unwrap();
            let r = &mut reach_stats[!read as usize][bi];
            r.0 += 1;
            if let Some(at) = reached_at {
                r.1 += 1;
                r.2 += at;
            }
        }
        if testing {
            let bi = books.iter().position(|b| *b == book).unwrap();
            let slot = if no_instr { &mut res_noinstr[!read as usize][bi] } else { &mut res[!read as usize][bi] };
            slot.words += tally.words;
            slot.right += tally.right;
            slot.said_story += tally.said_story;
            slot.said_book += tally.said_book;
            slot.silent += tally.silent;
        }
        // sleep: every RT_SLEEP_EVERY training trials (default 10), the index replays RT_REPLAY
        // chains (default 3, up to 30 rows each, by strength). Each pointer is reinstated in the
        // cortex, which completes it; the row is relearned to the layer 4 assembly that the
        // completion evokes now, if it keeps at least half the pointed cells (so the index
        // follows layer 4's drift between uses)
        if learn && sleep_every > 0 && (trial + 1) % sleep_every == 0 {
            if let Some(x) = ix.as_mut() {
                for _ in 0..replays {
                    let chain = x.mem.replay_prioritized(&mut rng, |_| 1, 30);
                    for r in chain {
                        let (Some(&row), false) = (r.ca3.first(), r.ec.is_empty()) else { continue };
                        let mut frame = zero();
                        for &b in r.ec.iter().filter(|&&b| b < BITS) {
                            frame.bit_set(b);
                        }
                        let st = net.step(&zero(), &zero(), &frame, false, 0, &mut rng);
                        let now = ones(&net.assoc_of(&zero(), &st.plan, false, &mut rng));
                        let kept = r.ec.iter().filter(|c| now.binary_search(c).is_ok()).count();
                        x.replayed += 1;
                        if kept * 2 >= r.ec.len() && now != r.ec {
                            x.mem.reconsolidate(row, &now);
                            x.relearned_asleep += 1;
                        }
                    }
                }
            }
        }
        if learn && (trial + 1) % 500 == 0 {
            println!(
                "  trial {:>5}: words planned right {:.1}% over the last 500 trials, said right in practice {:.1}% ({:.0} s)",
                trial + 1,
                100.0 * train_right.0 as f64 / train_right.1.max(1) as f64,
                100.0 * practice_right.0 as f64 / practice_right.1.max(1) as f64,
                t0.elapsed().as_secs_f64()
            );
            train_right = (0, 0);
            practice_right = (0, 0);
        }
    }
    let pct = |a: usize, b: usize| 100.0 * a as f64 / b.max(1) as f64;
    for (title, r) in [("with the instruction held", &res), ("prefrontal content removed before the task", &res_noinstr)] {
        println!("\ntest, {title} (new stories, words said aloud):");
        println!("  {:<6} {:<12} {:>6} {:>8} {:>13} {:>12} {:>7}", "asked", "book", "words", "right", "said story's", "said book's", "silent");
        for (ii, instr) in ["read", "tell"].iter().enumerate() {
            for (bi, b) in ["same story", "other story", "closed"].iter().enumerate() {
                let x = r[ii][bi];
                if x.words == 0 {
                    continue;
                }
                println!(
                    "  {:<6} {:<12} {:>6} {:>7.1}% {:>12.1}% {:>11.1}% {:>6.1}%",
                    instr,
                    b,
                    x.words,
                    pct(x.right, x.words),
                    pct(x.said_story, x.words),
                    pct(x.said_book, x.words),
                    pct(x.silent, x.words)
                );
            }
        }
    }
    println!("\nhippocampal recall held the wanted word on {:.1}% of telling steps (overlap at least 16 of 32 bits); {:.0} bits recalled on average", pct(hc_right.0, hc_right.1), hc_bits as f64 / hc_right.1.max(1) as f64);
    println!("last 500 training trials, the teacher's words heard: recall right {:.1}% of telling steps; plan right: read same {:.1}%, read other {:.1}%, tell same {:.1}%, tell other {:.1}%, tell closed {:.1}%",
        pct(hc_train.0, hc_train.1),
        pct(by_cond[0][0].0, by_cond[0][0].1), pct(by_cond[0][1].0, by_cond[0][1].1),
        pct(by_cond[1][0].0, by_cond[1][0].1), pct(by_cond[1][1].0, by_cond[1][1].1), pct(by_cond[1][2].0, by_cond[1][2].1));
    if make_on {
        let n = invented.len();
        let (mut sent, mut ok, mut framed, mut well, mut coherent, mut novel, mut copied, mut len) = (0, 0, 0, 0, 0, 0, 0, 0);
        for (heard, w) in &invented {
            let (a, b, f, c) = judge(w);
            sent += a;
            ok += b;
            framed += f as usize;
            well += (f && a == b) as usize;
            coherent += (f && c && a == b) as usize;
            novel += (f && a == b && !heard_stories.contains(&w.join(" "))) as usize;
            copied += (w == heard) as usize;
            len += w.len();
        }
        println!(
            "\ninvented stories (\"now make one\", relay noise {make_temp}): {n}; {:.1} words long on average; sentences grammatical {:.1}%; begin and end as a story {:.1}%; and every sentence grammatical {:.1}%; well formed and coherent (one name, its pronoun, one animal) {:.1}%; well formed and never heard in training {:.1}%; a copy of the story just heard {:.1}%; ended by itself {:.1}%",
            len as f64 / n.max(1) as f64,
            pct(ok, sent),
            pct(framed, n),
            pct(well, n),
            pct(coherent, n),
            pct(novel, n),
            pct(copied, n),
            pct(stopped_itself, n)
        );
        for (_, w) in invented.iter().take(8) {
            println!("  {}", w.join(" "));
        }
    }
    println!(
        "\ntraining read trials, words planned right: without reaching {:.1}% of {}, after reaching {:.1}% of {}",
        pct(read_by_reach[0].0, read_by_reach[0].1),
        read_by_reach[0].1,
        pct(read_by_reach[1].0, read_by_reach[1].1),
        read_by_reach[1].1
    );
    {
        let sim = |a: &Vec<usize>, b: &Vec<usize>| a.iter().filter(|x| b.binary_search(x).is_ok()).count() as f64 / a.len().max(b.len()).max(1) as f64;
        let mut within = (0.0, 0usize);
        let mut across = (0.0, 0usize);
        for i in 0..wait_states[0].len().min(40) {
            for j in 0..wait_states[0].len().min(40) {
                if i != j {
                    within.0 += sim(&wait_states[0][i], &wait_states[0][j]);
                    within.1 += 1;
                }
            }
            for j in 0..wait_states[1].len().min(40) {
                across.0 += sim(&wait_states[0][i], &wait_states[1][j]);
                across.1 += 1;
            }
        }
        println!(
            "layer 2/3 at the wait: {} bits on average; overlap read-read {:.2}, read-tell {:.2}",
            wait_states[0].iter().map(|v| v.len()).sum::<usize>() / wait_states[0].len().max(1),
            within.0 / within.1.max(1) as f64,
            across.0 / across.1.max(1) as f64
        );
    }
    println!(
        "dopamine right after a reach (training): read {:+.2} ({}), tell {:+.2} ({}), make {:+.2} ({})",
        reach_delta[0].0 as f64 / reach_delta[0].1.max(1) as f64 / ONE as f64,
        reach_delta[0].1,
        reach_delta[1].0 as f64 / reach_delta[1].1.max(1) as f64 / ONE as f64,
        reach_delta[1].1,
        reach_delta[2].0 as f64 / reach_delta[2].1.max(1) as f64 / ONE as f64,
        reach_delta[2].1
    );
    if shelf_k > 0 {
        println!(
            "\nfinding a book on the shelf ({} places; test, instruction held): {} trials; reached the right place {:.1}% (chance {:.1}%), reached a place {:.1}%; words read right {:.1}%",
            shelf_k,
            find_stats.0,
            pct(find_stats.1, find_stats.0),
            100.0 / shelf_k as f64,
            pct(find_stats.2, find_stats.0),
            pct(find_stats.4, find_stats.3)
        );
        println!(
            "  over all {} find trials: working memory held the animal {:.1}%; the hippocampus recalled a place {:.1}%, the right one {:.1}%",
            find_diag.0,
            pct(find_diag.1, find_diag.0),
            pct(find_diag.3, find_diag.0),
            pct(find_diag.2, find_diag.0)
        );
    }
    if let Some(x) = &ix {
        let (live, evicted, work) = x.mem.report();
        println!(
            "\nhippocampal index: {} rows stored ({} live, {} forgotten), {} postings visited per recall; {} learned event boundaries ({:.1} per trial); an episode's start held {} times, reinstated {} times; rows relearned on recall {}; asleep: {} replayed, {} relearned; recalls {} (after a recall {}, followed the link {})",
            x.mem.rows(),
            live,
            evicted,
            work,
            x.boundaries,
            x.boundaries as f64 / (n_train + 2 * n_test) as f64,
            x.holds,
            x.reinstated,
            x.relearned,
            x.replayed,
            x.relearned_asleep,
            x.recalls,
            x.had_prev,
            x.followed
        );
    }
    {
        // is the association area cross-modal? the assembly a word evokes seen and heard
        let (mut same, mut n, mut ov) = (0usize, 0usize, 0u32);
        for w in 0..vocab.len() {
            let seen = net.assoc_of(&eye.codes[w], &zero(), false, &mut rng);
            let heard = net.assoc_of(&zero(), &ear.codes[w], false, &mut rng);
            let o = overlap(&seen, &heard);
            same += (o >= 16) as usize;
            ov += o;
            n += 1;
        }
        println!("association area: a word seen and the same word heard share {:.1} of 32 cells on average; at least half for {:.1}% of words", ov as f64 / n as f64, pct(same, n));
    }
    if reach_learned {
        let mut line = String::new();
        for (ii, instr) in ["read", "tell"].iter().enumerate() {
            for (bi, b) in ["same story", "other story", "closed"].iter().enumerate() {
                let (n, r, d) = reach_stats[ii][bi];
                if n > 0 {
                    line += &format!(" {instr} ({b}): {:.0}% after {:.1} words;", pct(r, n), d as f64 / r.max(1) as f64);
                }
            }
        }
        println!("\nreaching for the book (test, instruction held):{line}");
    }
    if practice_p > 0.0 {
        println!("practice: {corrections} corrections heard ({})", if heard_correction { "heard" } else { "oracle: counted, not heard" });
    }
    println!("\nbasal ganglia: {}", striatum.report());
    println!(
        "imagined (cues to the hippocampus): {} ({} in telling trials)\nmouse: {} babbled, {} released by the basal ganglia, {} shown by the teacher ({} left, {} right, {} clicks in all), {} books opened; curiosity per training trial {:+.3}; motor area {:?}",
        imagined.0,
        imagined.1,
        mouse_stats[0],
        mouse_stats[1],
        mouse_stats[6],
        mouse_stats[2],
        mouse_stats[3],
        mouse_stats[4],
        mouse_stats[5],
        curiosity_sum as f64 / n_train.max(1) as f64 / ONE as f64,
        hand!().stats()
    );
    println!("read trials, cursor on the book (steps, CLICK proposed, LEFT proposed, CLICK released, LEFT released): training {:?}, test {:?}", on_book[0], on_book[1]);
    if let Some(x) = ix.as_ref().filter(|x| x.auto) {
        println!("always completing: {} completions, {} reached the cortex (retrieval mode)", x.completions.0, x.completions.1);
    }
    if let Some(b) = net.brain.as_ref() {
        println!("areas: {:?}; association areas (pair, cells): {:?}; full area: {:?} cells", b.areas.iter().map(|a| (a.name.as_str(), a.l4.as_ref().map_or(0, |p| p.cells()))).collect::<Vec<_>>(), b.pairs.iter().map(|p| (p.0, p.1.l4.as_ref().map_or(0, |x| x.cells()))).collect::<Vec<_>>(), b.full.as_ref().map(|f| f.l4.as_ref().map_or(0, |x| x.cells())));
        if b.columns {
            let (n, right) = b.assoc_hits();
            println!("association columns: predicted the next scene {right} of {n} times");
            let (n, right) = b.sense_hits();
            println!("sensory columns: predicted the next input {right} of {n} times");
        }
        if net.selforg {
            let names: Vec<String> = b.pairs.iter().map(|p| format!("{}+{}", b.areas[p.0 .0].name, b.areas[p.0 .1].name)).chain(b.full.iter().map(|_| "full".to_string())).collect();
            for (i, h) in net.goal_hits.iter().enumerate().filter(|(_, h)| h[0].0 + h[1].0 > 0) {
                let name = if i >= 20 { format!("sense {}", b.areas[i - 20].name) } else { names.get(i).cloned().unwrap_or_else(|| "?".into()) };
                println!(
                    "  column {}: sounds right {} of {} offered, mouse acts right {} of {}",
                    name,
                    h[0].1,
                    h[0].0,
                    h[1].1,
                    h[1].0
                );
            }
        }
        if let Some(sx) = net.streams.as_ref() {
            println!(
                "streams: ventral predicted the next word {} of {}; auditory–motor the next input {} of {}; visual–motor {} of {}; layer 5 cells {} / {}",
                sx.ventral.hits.1, sx.ventral.hits.0, sx.audio_motor.hits.1, sx.audio_motor.hits.0, sx.visuo_motor.hits.1, sx.visuo_motor.hits.0, sx.audio_motor.l5.committed(), sx.visuo_motor.l5.committed()
            );
        }
        if let Some(t) = net.top.as_ref() {
            println!("top association column: predicted the next scene {} of {} times; layer 5 {} cells", t.hits.1, t.hits.0, t.l5.committed());
        }
    }
    if net.brain.is_some() {
        println!("voice area: {} words said, {} not heard as its efference copy expected; plans said through the imitation route {}, the inverse model {}", voice_check.0, voice_check.1, voice_route[1], voice_route[0]);
    }
    if teacher_cursor {
        println!("teacher's cursor: {teacher_opened} demonstrations ended with the teacher's book open");
    }
    println!("mirror: {} of the teacher's acts seen, {} evoked the right felt motion", mirror_stats.0, mirror_stats.1);
    println!("imagining the click while telling: {} times, recall went to this story's opening {} times", open_recall.0, open_recall.1);
    println!("checks of newly credited acts: {} made, {} found no effect", checks.0, checks.1);
    println!("cortical curiosity: extra noise {:.3} on average over {} choices", curiosity_stats.0 as f64 / curiosity_stats.1.max(1) as f64, curiosity_stats.1);
    println!("read trials at test, cursor off the book (place:proposed>released): {off_book:?}");
}
