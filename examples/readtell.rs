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
use neurocomp::program::{
    BitCells, BoundaryCell, CerebellarCircuit, Driver, EpisodicCircuit, Hippocampus, HippocampusConfig, IndexConfig, IndexMemory, Layer4, MotorArea,
    Striatum, ThalamicRelay, VocalTract, WorkingMemory,
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
}

struct Step {
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
            l23: mk(16384, 3),
            // [eye L4 | PFC (tuft) | cerebellum (tuft) | ear L4 | layer 2/3's prediction]
            l5: mk(8192, 2),
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
        let e4 = if eye.count_ones() == 0 { zero() } else if learn { self.l4e.encode_learn(eye, rng) } else { self.l4e.encode(eye) };
        let a4 = if ear.count_ones() == 0 { zero() } else if learn { self.l4a.encode_learn(ear, rng) } else { self.l4a.encode(ear) };
        let pfc = self.wm.content();
        let mossy = frames(&[ear, &self.pons]);
        let cb_out = self.cb.predict(&mossy).clone();
        let mut props: Vec<(u8, Driver, BitVector)> = Vec::new();
        if cb_out.count_ones() > 0 {
            props.push((2, Driver::Tonic(0), cb_out.clone()));
        }
        let l23_row = frames(&[&e4, &pfc, &cb_out, &a4, recall, &self.rec23]);
        let p23 = self.l23.predict(&l23_row, BITS);
        self.rec23 = project(&self.l23.last_fired());
        let o23 = p23.as_ref().map(|x| x.0.clone()).unwrap_or_else(zero);
        if let Some((o, _, burst)) = p23 {
            props.push((0, if burst { Driver::Burst } else { Driver::Tonic(1) }, o));
        }
        let l5_row = frames(&[&e4, &pfc, &cb_out, &a4, &o23]);
        let p5 = self.l5.predict(&l5_row, BITS);
        self.pons = project(&self.l5.last_fired());
        self.relay.step(matches!(p5, Some((_, _, false))));
        if let Some((o, _, burst)) = p5 {
            props.push((1, if burst { Driver::Burst } else { Driver::Spike }, o));
        }
        // the relay's context: what the prefrontal cortex holds and what was heard
        let ctx = frames(&[&pfc, ear]);
        let mut score = vec![0u32; BITS];
        for (src, drv, pat) in &props {
            let agrees = *src != 2 && overlap(pat, &cb_out) >= 16;
            let mut w = self.relay.weight(*src, *drv, &ctx, agrees) as u64;
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
        Step { l23_row, l5_row, mossy, ctx, props, plan }
    }

    fn learn(&mut self, st: &Step, target: &BitVector, rng: &mut StdRng) {
        self.l23.learn(&st.l23_row, target, rng);
        self.l5.learn(&st.l5_row, target, rng);
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
const CH_REACH: usize = 0;
const CH_HOLD: usize = 1;
const CH_REINSTATE: usize = 2;
const CH_LOAD: usize = 3;
const CH_PLACE: usize = 4;

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
    /// diagnosis: the gates do nothing (the driver holds and reinstates)
    gates_off: bool,
    /// diagnosis: (boundary fired, recalled row) per heard sound, while tracing
    pub log: Vec<(bool, Option<u32>)>,
}

impl IndexHc {
    fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(4242));
        let ctx = (0..ACTIVE).map(|_| rng.gen_range(0..CTX_POOL)).collect();
        let cfg = IndexConfig { min_overlap: 12, ..Default::default() };
        Self {
            mem: IndexMemory::new(cfg),
            ctx,
            boundary: BoundaryCell::new(BITS, 3, 80),
            held: None,
            place: None,
            last_word: 0,
            last_recalled: None,
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
            log: Vec::new(),
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
        let len = self.ctx.len() as u64;
        for &(h, c) in acting.iter().take(n) {
            let slot = (h % len) as usize;
            self.ctx[slot] = (mix(c as u64 ^ (self.ctx[slot] as u64).rotate_left(21) ^ salt) % CTX_POOL as u64) as usize;
        }
    }

    /// What the hippocampus reinstates now: the layer 4 cells stored under the context most
    /// like the current one; else the successor of the last row recalled.
    fn recall(&mut self) -> BitVector {
        // the sequence bias (CA3's recurrent links): the successor of the row just recalled is
        // expected, and is recalled whenever the cue still matches it well enough
        // expected, and is recalled unless another row matches the cue clearly better (by 12 of
        // 32 cells): the context finds where a sequence starts and corrects large jumps; the
        // links carry it on through boundaries that fall differently the second time
        let best = self.mem.recall(&self.ctx);
        let best_overlap = best.strength / 16;
        let follow = self.last_recalled.and_then(|l| self.mem.successor_match(l, &self.ctx)).filter(|&(_, o)| best.ec.is_empty() || o + 12 > best_overlap);
        self.recalls += 1;
        if self.debug {
            let sm = self.last_recalled.and_then(|l| self.mem.successor_match(l, &self.ctx));
            println!("    recall: last {:?} successor {:?} best row {:?} overlap {} -> follow {}", self.last_recalled, sm, best.ca3.first(), best_overlap, follow.is_some());
        }
        self.had_prev += self.last_recalled.is_some() as usize;
        let mut r = match follow {
            Some(_) => {
                self.followed += 1;
                self.mem.successor(self.last_recalled.unwrap())
            }
            None => best,
        };
        if r.ec.is_empty() {
            if let Some(l) = self.last_recalled {
                r = self.mem.successor(l);
            }
        }
        self.last_recalled = r.ca3.first().copied();
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
        if store && !self.retrieving() && !cells.is_empty() {
            // the keys: the context, the place (if a book from the shelf is in view), and the
            // content (the assembly), so the episode can be completed from any of them
            let mut keys = self.ctx.clone();
            if let Some(k) = self.place {
                keys.extend((0..16).map(|j| PLACE_BASE + k * 16 + j));
            }
            keys.extend(ones(heard).into_iter().map(|c| CONTENT_BASE + c));
            self.mem.store_split(&keys, &[], &cells);
        }
        // the context moves on with layer 4's cells, not the raw sound
        self.drift(&cells, 4, 1);
        self.last_word = key;
        let fired = self.boundary.observe(&cells);
        self.log.push((fired, self.last_recalled));
        if fired {
            self.boundaries += 1;
            self.drift(&cells, self.ctx.len() / 2, 2);
        }
        fired
    }

    /// Retrieval mode: the prefrontal request stands and acetylcholine is low (Hasselmo: high
    /// acetylcholine favours encoding, low favours retrieval).
    fn retrieving(&self) -> bool {
        self.request && self.ach < ONE / 2
    }

    /// A reach (a motor act that brings a new scene, a book) is an event boundary: the context
    /// moves on by half.
    fn reach_event(&mut self) {
        self.boundaries += 1;
        let c = self.ctx.clone();
        self.drift(&c, self.ctx.len() / 2, 3);
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

    /// The prefrontal cortex holds the context in force: where this episode begins.
    fn hold(&mut self) {
        self.held = Some(self.ctx.clone());
        self.request = false;
        self.holds += 1;
    }

    /// The held context is reinstated as the current one, and retrieval begins.
    fn reinstate(&mut self) {
        if let Some(h) = self.held.clone() {
            self.ctx = h;
            self.last_recalled = None;
            self.request = true;
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
    // a word said: None for nothing said or (with the learned stop) silence
    let say = |plan: &BitVector| motor.plan(plan).and_then(|m| tract.articulate(&m));
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
    let mut striatum = Striatum::new(1 << 18);
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
    let fixed_gate = std::env::var("RT_GATE").map_or(false, |v| v == "fixed");
    let td_debug = std::env::var("RT_TD_DEBUG").is_ok();
    // the striatum's reward is the outcome: words said right in the task (the teacher's
    // words); RT_REWARD=all also rewards every heard word the cortex predicted
    let reward_all = std::env::var("RT_REWARD").map_or(false, |v| v == "all");
    // RT_HC=index (default): the hippocampus as a growing index with learned event boundaries
    // (`IndexHc`); RT_HC=circuit: the fixed circuit cued by the driver's story context
    // RT_IX_HOLD=oracle (diagnosis only): the context is held at the story's first word and
    // reinstated when the task begins, instead of by the learned gates
    let ix_oracle = std::env::var("RT_IX_HOLD").map_or(false, |v| v == "oracle");
    let sleep_every: usize = env("RT_SLEEP_EVERY", 10);
    let replays: usize = env("RT_REPLAY", 3);
    let mut ix: Option<IndexHc> = std::env::var("RT_HC").map_or(true, |v| v != "circuit").then(|| IndexHc::new(seed));
    if let Some(x) = ix.as_mut() {
        x.gates_off = ix_oracle;
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
    macro_rules! td_step {
        ($fired:expr, $explore:expr) => {{
            let (retr, held) = ix.as_ref().map_or((false, false), |x| (x.request, x.held.is_some()));
            let held_ctx = ix.as_ref().and_then(|x| x.held.clone()).unwrap_or_default();
            let eye4 = if eye_now.count_ones() == 0 { zero() } else { net.l4e.encode(&eye_now) };
            let state = state_groups(&net.wm.content(), &last_assoc, &prev_assoc, &net.rec23, &net.pons, &eye4, &last_recall, &held_ctx, recalled_place);
            striatum.begin_groups(&state, $explore, &mut rng);
            if let Some(x) = ix.as_mut() {
                if !x.gates_off {
                    // reinstating is possible only when not already retrieving (it would only
                    // start the sequence over)
                    if held && !retr && striatum.choose(CH_REINSTATE, 2, if $explore { Some(&mut rng) } else { None }) == 1 {
                        x.reinstate();
                    } else if $fired && striatum.choose(CH_HOLD, 2, if $explore { Some(&mut rng) } else { None }) == 1 {
                        x.hold();
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
    let mut by_cond = [[(0usize, 0usize); 3]; 2];
    let books = [Book::Same, Book::Other, Book::Closed];
    for trial in 0..n_train + 2 * n_test {
        let testing = trial >= n_train;
        let no_instr = trial >= n_train + n_test;
        let learn = !testing;
        let s = story(&mut rng);
        // the story's context (a lateral-EC-like pattern as large as the two-word cue), new for
        // each story told
        let story_ctx: u64 = rng.next_u64();
        // 1. the teacher tells the story; the network predicts each next word and stores it
        let mut prev: Option<BitVector> = None;
        let mut prev2: Option<BitVector> = None;
        for t in 0..s.len() {
            let w = id(s[t]);
            let heard = &ear.codes[w];
            let cue = Net::hc_context(prev.as_ref(), prev2.as_ref(), &silence, story_ctx, time.then_some(t));
            if ix.is_some() {
                // RT_IX_HOLD=oracle (diagnosis only): hold the context in force at the story's
                // first word, instead of the learned gate
                if t == 0 && ix_oracle {
                    let x = ix.as_mut().unwrap();
                    x.held = Some(x.ctx.clone());
                    x.request = false;
                }
                hc_hear!(w, &cue, true, learn, false);
            }
            let rec = hc_recall!(&cue);
            let st = net.step(&zero(), heard, &rec, learn, 0, &mut rng);
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
        // the teacher pauses: silence is heard (a sound, and a natural event boundary)
        if ix.is_some() {
            let cells = net.assoc_of(&zero(), &silence, false, &mut rng);
            let fired = ix.as_mut().unwrap().hear(quiet, &silence, &cells, &cells, false, false);
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
        if ix_oracle && !read && !make {
            if let Some(x) = ix.as_mut() {
                if let Some(h) = x.held.clone() {
                    x.ctx = h;
                    x.last_recalled = None;
                    x.request = true;
                }
            }
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
                    match say(&st.plan) {
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
            x.debug = trace > 0 && testing && !read && !make && book == Book::Closed && !no_instr;
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
        // the teacher waits (up to three pauses) before starting; the network may reach for the
        // book meanwhile (in "find" trials: to one of the shelf's places). Once the teacher
        // starts, it goes on whether a book is open or not.
        if !book_open {
            for _ in 0..3 {
                if ix.is_some() {
                    let cells = net.assoc_of(&zero(), &silence, false, &mut rng);
                    let fired = ix.as_mut().unwrap().hear(quiet, &silence, &cells, &cells, false, false);
                    prev_assoc = std::mem::replace(&mut last_assoc, cells);
                    td_step!(fired, learn);
                } else {
                    prev_assoc = std::mem::replace(&mut last_assoc, net.assoc_of(&zero(), &silence, false, &mut rng));
                    td_step!(false, learn);
                }
                if find {
                    let a = striatum.choose(CH_PLACE, shelf_k + 1, if learn { Some(&mut rng) } else { None });
                    if a > 0 {
                        // the book at that place, from its first word; its place joins the
                        // hippocampus's context while it is in view
                        p = shelf[a - 1].clone();
                        book_open = true;
                        eye_now = p.first().map_or(zero(), |w| eye.codes[id(w)].clone());
                        reached_at = Some(a - 1);
                        if let Some(x) = ix.as_mut() {
                            x.reach_event();
                            x.place = Some(a - 1);
                        }
                        td_step!(true, learn);
                        break;
                    }
                } else if striatum.choose(CH_REACH, 2, if learn { Some(&mut rng) } else { None }) == 1 {
                    // the world hands over the book, from its first word; the reach is an
                    // event boundary, and the striatum may hold where this new episode begins
                    book_open = true;
                    eye_now = p.first().map_or(zero(), |w| eye.codes[id(w)].clone());
                    reached_at = Some(0);
                    if let Some(x) = ix.as_mut() {
                        x.reach_event();
                    }
                    td_step!(true, learn);
                    if learn {
                        let k = if read { 0 } else if make { 2 } else { 1 };
                        reach_delta[k].0 += striatum.last_delta as i64;
                        reach_delta[k].1 += 1;
                    }
                    break;
                }
            }
        }
        let (debug_v0, debug_p0) = striatum.debug_now(CH_REACH);
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
                    let said = say(&st.plan).filter(|&w| w != quiet);
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
                let said = say(&st.plan).filter(|&w| w != quiet);
                if trace > 0 && book == Book::Closed && !read && t < 6 {
                    trace -= (t == 5) as usize;
                    let near = ear.decode(&st.plan).map_or("-", |w| vocab[w]);
                    println!(
                        "  trace t={t} want {:<7} plan bits {:>2} (nearest {:<7} overlap {:>2}) sources {:?} recall overlap {:>2} said {:?}",
                        target[t],
                        st.plan.count_ones(),
                        near,
                        overlap(&st.plan, &ear.codes[want]),
                        st.props.iter().map(|x| x.0).collect::<Vec<_>>(),
                        overlap(&rec, &ear.codes[want]),
                        said.map(|w| vocab[w])
                    );
                }
                tally.words += 1;
                said_seq.push(match say(&st.plan) {
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
            // the pause after the task
            let cells = net.assoc_of(&zero(), &silence, false, &mut rng);
            let fired = ix.as_mut().unwrap().hear(quiet, &silence, &cells, &cells, false, false);
            prev_assoc = std::mem::replace(&mut last_assoc, cells);
            td_step!(fired, learn);
            let task_log: Vec<bool> = ix.as_mut().unwrap().log.drain(..).map(|l| l.0).collect();
            if trace > 0 && testing && !read && !make && book == Book::Closed && !no_instr {
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
        // the striatum's preference for reaching over waiting, holding "read" or "tell" and
        // having just heard "it"
        let heard_it = net.assoc_of(&zero(), &ear.codes[id(IT)], false, &mut rng);
        let mut pref = |v: &str| {
            let before = net.assoc_of(&zero(), &ear.codes[id(v)], false, &mut rng);
            let st = state_groups(&ear.codes[id(v)], &heard_it, &before, &zero(), &zero(), &zero(), &zero(), &[], None);
            striatum.begin_groups(&st, false, &mut rng);
            let d = striatum.drives(CH_REACH, 2);
            striatum.end(false, &mut rng);
            (d[1] - d[0]) as f64
        };
        println!("\nreaching for the book (test, instruction held):{line} preference for reaching over waiting after \"read it\" {:+.2}, \"tell it\" {:+.2}", pref("read"), pref("tell"));
    }
    if practice_p > 0.0 {
        println!("practice: {corrections} corrections heard ({})", if heard_correction { "heard" } else { "oracle: counted, not heard" });
    }
    println!(
        "\nstriatum: {} TD updates, mean |dopamine| {:.3}; cells: {} striosome, {} go, {} no-go ({} recruited, {} removed)",
        striatum.stats.0,
        striatum.stats.1 as f64 / striatum.stats.0.max(1) as f64 / ONE as f64,
        striatum.sizes().0,
        striatum.sizes().1,
        striatum.sizes().2,
        striatum.changes.0,
        striatum.changes.1
    );
}
