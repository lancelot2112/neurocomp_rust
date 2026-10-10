//! Experiment 11: episodic autoassociative memory for fact binding.
//!
//! Run:  cargo run --release --example episodic
//!
//! Stories as in experiments 09-10 (random filler words between stories, 1-N
//! facts about distinct people, a question about a random one, one place per
//! name held out of training), in three forms:
//!   short:  "X went to the P ."                          "where is X ?"
//!   long:   "X went [all the way over] to the P ."        "where is X right now ?"
//!   varied: "X [quickly|slowly] went [all the way over] to the [big|old] P ."
//!           with every bracket optional, so the place sits at a variable
//!           distance from the name and no fixed offset finds it.
//!
//! Episodic memory (`EpisodicMemory`): each sentence (up to ".") is stored once as
//! the union of its word codes. While reading, the current sentence so far,
//! minus habituated (frequent) bits, is the cue; the most recent episode with the
//! largest overlap is recalled, and its novel bits, minus the cue, are given to
//! the predictor. No offsets, no routes, no gradual learning in the memory.
//! The memory keeps storing at test time (storing is reading); the predictor's
//! learning is off at test.
//!
//! Predictor input: [current word | memory (or relay) frames | previous word].
//!
//! Experiment 12 adds `Ca3` policies: episodes are stored in Hebbian weights
//! (EC->CA3, CA3 recurrent, CA3->EC; decaying palimpsest) under a dentate-gyrus
//! code, and recalled by driving CA3 from the cue and letting it settle. Low vs high
//! pattern separation = few dense vs many sparse granule/CA3 cells.

mod common;

use neurocomp::det::HashMap;

use common::Encoder;
use neurocomp::bitvec::BitVector;
use neurocomp::fixed::{chance, q16, q16x, ratio as ratio_q, to_f32, Q16, ONE};
use neurocomp::kernel::{GrowthConfig, KernelClass, SimpleKernel};
use neurocomp::program::{BitCells, BoundaryCell, Layer5, PrimedLayer5, Curiosity, Dedup, EngramConfig, BeliefRule, EngramStore, EpisodicCircuit, MotorArea, OutputBuffer, PhonologicalLoop, RelationStore, VocalTract, Hippocampus, HippocampusConfig, IndexConfig, IndexMemory, Autoassociative, BasalGanglia, Ca3FloatMemory, Ca3Memory, CorticalColumn, AreaContext, CorticothalamicGate, DentateGyrus, RoleArea, SourceMix, HigherArea, EpisodicMemory, Gate, PfcGate, RelayChannel, RouteScores, Thalamus, WorkingMemory};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

// Runs (one per policy and seed) go in parallel threads; each thread's printing is kept in
// a buffer of its own and printed in run order when the run ends, so the output reads as
// if the runs went one after another.
thread_local! {
    static OUT: std::cell::RefCell<Option<(String, String)>> = const { std::cell::RefCell::new(None) };
}

fn emit(err: bool, newline: bool, text: String) {
    let text = if newline { text + "\n" } else { text };
    let rest = OUT.with(|o| match o.borrow_mut().as_mut() {
        Some((out, er)) => {
            if err { er } else { out }.push_str(&text);
            None
        }
        None => Some(text),
    });
    if let Some(text) = rest {
        if err {
            std::eprint!("{text}");
        } else {
            std::print!("{text}");
        }
    }
}

macro_rules! println {
    () => { emit(false, true, String::new()) };
    ($($t:tt)*) => { emit(false, true, format!($($t)*)) };
}
macro_rules! eprintln {
    () => { emit(true, true, String::new()) };
    ($($t:tt)*) => { emit(true, true, format!($($t)*)) };
}
macro_rules! eprint {
    ($($t:tt)*) => { emit(true, false, format!($($t)*)) };
}

/// Run every (policy, seed) job on up to THREADS threads (default: the machine's cores),
/// each with its output buffered: (outcome, printed, printed to stderr) in job order. Each
/// run depends only on its seed, so the results do not depend on the thread count.
fn run_all(jobs: &[(Policy, u64)], task: Task, max_facts: usize) -> Vec<(Outcome, String, String)> {
    let threads = std::env::var("THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get())).clamp(1, jobs.len().max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done: Vec<std::sync::Mutex<Option<(Outcome, String, String)>>> = jobs.iter().map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|sc| {
        for _ in 0..threads {
            std::thread::Builder::new()
                .stack_size(256 << 20)
                .spawn_scoped(sc, || loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&(policy, seed)) = jobs.get(i) else { break };
                    OUT.with(|o| *o.borrow_mut() = Some((String::new(), String::new())));
                    let outcome = run(policy, task, max_facts, seed);
                    let (out, err) = OUT.with(|o| o.borrow_mut().take()).unwrap_or_default();
                    *done[i].lock().unwrap() = Some((outcome, out, err));
                })
                .expect("spawn a run thread");
        }
    });
    done.into_iter().map(|m| m.into_inner().unwrap().expect("every job ran")).collect()
}

/// Fixed-point constants (`Q16`, `ONE` = 1): the model's per-step arithmetic is integer.
const Q_TENTH: Q16 = 6554;
const Q_03: Q16 = 19661;
const Q_04: Q16 = 26214;
const Q_HALF: Q16 = 32768;
const Q_08: Q16 = 52429;
const SEVEN_TENTHS: Q16 = 45875;
/// Confidence bands 0.5 / 0.7 / 0.8 / 0.9 (mix keys and calibration).
const CONF_BANDS: [Q16; 4] = [32768, 45875, 52429, 58982];
const BITS: usize = 8192; // sparse enough that words rarely share bits (habituation is per bit)
const NAMES: &[&str] = &["mary", "john", "sandra", "daniel", "anna", "peter"];
const PLACES: &[&str] = &["kitchen", "garden", "office", "hallway", "bathroom", "bedroom"];
const FILLERS: &[&str] = &["then", "later", "so", "next", "after"];
const OBJECTS: &[&str] = &["ball", "apple", "book", "key", "cup", "box"];
/// Topic task: distractor sentences with no names or places.
const DISTRACTORS: &[&[&str]] = &[&["the", "cat", "slept", "."], &["the", "dog", "ran", "away", "."], &["it", "rained", "."]];
/// Persist task: names whose places are fixed and stated only early in training.
const ANCHOR_NAMES: &[&str] = &["bill", "fred", "julie"];
const TRAIN: usize = 3000;
/// The frame gate's own records in the mix (HIER_TRUST_GATE): the higher area, the column.
const TRUST_AREA: u8 = 40;
const TRUST_COLUMN: u8 = 41;
/// The frame's counterfactual record (HIER_TRUST_GATE=cf).
const TRUST_CF: u8 = 42;
/// The routing records (ROUTE): channel c's counterfactual record is source ROUTE_SRC + c.
const ROUTE_SRC: u8 = 100;
/// The burst code's rotation (HIER_UP=both) and the per-source binding step (ROUTE).
const BURST_ROT: usize = 4099;
const ROUTE_ROT: usize = 997;
/// The entorhinal feedback channel's source id (HC_ROUTE).
const HC_CHANNEL: usize = 48;
/// The slow cortex's source id in the mix (SLOW_CORTEX).
const SLOW_SRC: u8 = 12;
/// ASSOC: the association area's vote in the mix.
const ASSOC_SRC: u8 = 15;
/// The cerebellum's source id in the mix and its thalamic channel (LEARNING=three).
const CB_SRC: u8 = 13;
const CB_CHANNEL: usize = 49;
/// QQUERY with ROUTE: the held item's query answer as a routed channel.
const Q_CHANNEL: usize = 50;
/// Persist: anchor facts only appear in the first anchor_stories() training stories (env).
fn anchor_stories() -> usize {
    std::env::var("ANCHOR_STORIES").ok().and_then(|v| v.parse().ok()).unwrap_or(300)
}
const ANCHORS: usize = 3; // Persist: ANCHOR_NAMES; all of NAMES stay active
const TEST: usize = 1000;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Task {
    Short,
    Long,
    Varied,
    /// Two-hop questions: "X picked up the O" + "X went to the P" ... "where is the O ?"
    TwoHop,
    /// Consolidation: anchor names (bill, fred, julie) each have one fixed place, stated
    /// only in the first anchor_stories() training stories; the other names live in varied
    /// stories. Test asks about an anchor half of the time (reported as "held-out").
    Persist,
    /// Working memory: "X went to the P . <distractors> where is the person ?" -> P. The
    /// question names nobody; the subject must be held across the distractor sentences.
    Topic,
    /// Several routes needed: 2-3 facts "X gave the O to Y ." (all names distinct), then
    /// "what did X give ?" -> O (route (2,3): the word 3 after the earlier X) or
    /// "who got the O ?" -> Y (route (1,2): the word 2 after the earlier O). Recall of the
    /// fact sentence returns both O and Y, so memory alone is ambiguous.
    Give,
    /// Elimination: "is it the P ? no ." for 5 of the 6 places in random order, then
    /// "is it the" -> the one place not yet named. Nothing in the story states the answer;
    /// the column has to avoid what it has just seen (fast inhibition).
    Elim,
    /// Hierarchy: "in the morning ." or "at night .", 1-3 filler sentences, then
    /// "X went to the" -> X's habitual place for that time of day (a fixed mapping learned
    /// over training). The lower column cannot see the name and the time at the answer.
    Habit,
    /// Long reach (experiment 25): one episode per story. A season is announced ("winter
    /// came ."), then D season-free filler stories (1-2 distractor sentences each; D drawn
    /// from 0 to SEASON_LEN − 1, default 32), then "X went to the" -> X's place in that
    /// season. Nothing between the announcement and the question reveals the season, so
    /// accuracy by D shows how far back the areas reach.
    Season,
    /// Reading with actions (experiment 28): three books are read in interleaved sessions.
    /// A session is "@open_x" (an action: book x is opened), the book's season announcement
    /// if this is its first session, 0-3 filler stories, "X went to the" -> X's place in
    /// that book's season, then "@close". A book lasts 3-8 sessions, then a new one with a
    /// new season replaces it. Returning to a book, its season was announced sessions ago,
    /// with other books' seasons read in between.
    Books,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Policy {
    NoMemory,
    FixedRelay, // hand-set thalamic routes from experiments 09-10: (1,4), (3,4), (3,8)
    Episodic,
    /// Experiment 12: dentate-gyrus code + Hebbian CA3 store instead of a list.
    /// (granule cells, active cells, recurrent settle steps)
    Ca3 { cells: usize, k: usize, settle: usize },
    /// Big-loop recall (`EpisodicMemory::recall_chain`) with this many hops; one
    /// predictor frame per hop.
    Loop(usize),
    /// Branching big loop (`EpisodicMemory::recall_branches`): hop 1 plus this many
    /// hop-2 branches, one per rare item of hop 1; one frame each.
    Branch(usize),
    /// Big loop where a basal-ganglia selector (`BasalGanglia`) picks which item of
    /// hop 1 to follow, learned from reward (did hop 2 recall the next word?).
    /// Frames: hop 1, hop 2.
    Select,
    /// Thalamic gate driven by the basal ganglia: the relay routes and memory recall are
    /// channels; `BasalGanglia` releases one per step, valued per (channel, current word);
    /// reward = the released content contained the next word. One frame.
    ThalamicGate,
    /// Episodic memory plus consolidation: after each training story the hippocampus
    /// replays REPLAYS stored episodes into a cortical semantic store (a predictive
    /// KernelClass learning cue -> content); when hippocampal recall fails, the cortex
    /// supplies the memory frame.
    Consolidate,
    /// Episodic recall cued by prefrontal working memory (`WorkingMemory`), whose content
    /// is loaded by a gate: `learned` = `PfcGate` (basal ganglia, reward at the question
    /// if the recall contained the answer, delayed through the eligibility trace);
    /// otherwise a hand-set rule (load names) as the upper bound.
    Pfc { learned: bool },
    /// Layer-6 corticothalamic gating: each channel (the fixed relay routes, then memory
    /// recall) has its own frame, and `CorticothalamicGate` opens any number of them per
    /// context, learned Hebbian-style from L5 attribution (no reward). `gated: false`
    /// keeps every channel open (the baseline).
    L6Gate { gated: bool },
    /// Same gate, but the relay routes are learned: discovered from surprises and ranked
    /// by consistency (`RouteScores`, as in experiment 10), top 8 refreshed every 50
    /// stories; memory recall is one more channel.
    LearnedGate,
}

#[derive(Clone)]
struct Story {
    words: Vec<&'static str>,
    answer_at: usize,
    held_out: bool,
}

fn allowed(n: usize, p: usize) -> bool {
    p != n % PLACES.len()
}

/// Two-hop story: 2-3 people move around (1-2 moves each) and 1-2 of them pick up an
/// object, in random order; the question asks where an object is, i.e. the last place
/// its holder went. Held-out: (object, place) answers never seen in training.
fn two_hop_story(rng: &mut StdRng, want_held_out: bool) -> Story {
    loop {
        let mut words: Vec<&'static str> = Vec::new();
        for _ in 0..rng.gen_range(0..=4) {
            words.push(FILLERS.choose(rng).unwrap());
        }
        let mut names: Vec<usize> = (0..NAMES.len()).collect();
        names.shuffle(rng);
        let people = &names[..rng.gen_range(2..=3)];
        let mut objects: Vec<usize> = (0..OBJECTS.len()).collect();
        objects.shuffle(rng);
        let held: Vec<(usize, usize)> = objects[..rng.gen_range(1..=2)].iter().map(|&o| (o, *people.choose(rng).unwrap())).collect();
        let mut sentences: Vec<(Vec<&'static str>, Option<(usize, usize)>)> = Vec::new(); // (words, (person, place) if a move)
        for &p in people {
            for _ in 0..rng.gen_range(1..=2) {
                let place = rng.gen_range(0..PLACES.len());
                sentences.push((vec![NAMES[p], "went", "to", "the", PLACES[place], "."], Some((p, place))));
            }
        }
        for &(o, p) in &held {
            sentences.push((vec![NAMES[p], "picked", "up", "the", OBJECTS[o], "."], None));
        }
        sentences.shuffle(rng);
        let (o, holder) = held[rng.gen_range(0..held.len())];
        let answer = sentences.iter().filter_map(|(_, m)| *m).filter(|&(p, _)| p == holder).last().unwrap().1;
        if (answer != o % PLACES.len()) == want_held_out {
            continue; // held-out answers: place == object index (a diagonal)
        }
        for (w, _) in sentences {
            words.extend(w);
        }
        words.extend(["where", "is", "the", OBJECTS[o], "?"]);
        let answer_at = words.len();
        words.extend([PLACES[answer], "."]);
        return Story { words, answer_at, held_out: want_held_out };
    }
}

/// Give task: held-out = the question's (name, object) pair is never asked in training.
fn give_story(rng: &mut StdRng, want_held_out: bool) -> Story {
    loop {
        let facts = rng.gen_range(2..=3);
        let mut names: Vec<usize> = (0..NAMES.len()).collect();
        names.shuffle(rng);
        let mut objects: Vec<usize> = (0..OBJECTS.len()).collect();
        objects.shuffle(rng);
        let mut words: Vec<&'static str> = Vec::new();
        for f in 0..facts {
            if rng.gen_bool(0.5) {
                words.push(FILLERS.choose(rng).unwrap());
            }
            words.extend([NAMES[names[2 * f]], "gave", "the", OBJECTS[objects[f]], "to", NAMES[names[2 * f + 1]], "."]);
        }
        let f = rng.gen_range(0..facts);
        let (giver, receiver, object) = (names[2 * f], names[2 * f + 1], objects[f]);
        let ask_object = rng.gen_bool(0.5);
        let name = if ask_object { giver } else { receiver };
        if allowed(name, object) == want_held_out {
            continue;
        }
        let answer = if ask_object {
            words.extend(["what", "did", NAMES[giver], "give", "?"]);
            OBJECTS[object]
        } else {
            words.extend(["who", "got", "the", OBJECTS[object], "?"]);
            NAMES[receiver]
        };
        let answer_at = words.len();
        words.extend([answer, "."]);
        return Story { words, answer_at, held_out: want_held_out };
    }
}

/// Habit task: the place name `n` habitually goes to in the morning (t = 0) or at night (1).
fn habit_place(n: usize, t: usize) -> usize {
    (2 * n + 3 * t + 1) % PLACES.len()
}

fn habit_story(rng: &mut StdRng, held_out: bool) -> Story {
    let mut words: Vec<&'static str> = Vec::new();
    if rng.gen_bool(0.5) {
        words.push(FILLERS.choose(rng).unwrap());
    }
    let t = rng.gen_range(0..2);
    words.extend(if t == 0 { ["in", "the", "morning", "."] } else { ["at", "the", "night", "."] });
    for _ in 0..rng.gen_range(1..=3) {
        words.extend(DISTRACTORS.choose(rng).unwrap().iter().copied());
    }
    let n = rng.gen_range(0..NAMES.len());
    words.extend([NAMES[n], "went", "to", "the"]);
    let answer_at = words.len();
    words.extend([PLACES[habit_place(n, t)], "."]);
    Story { words, answer_at, held_out }
}

const SEASONS: &[&str] = &["spring", "summer", "autumn", "winter"];

/// Season task: `n`'s place in season `s` (distinct per season for every name). With
/// SEASON_RULE=season the place depends on the season only (everyone goes to the same
/// place), a rule over the person's role rather than their identity.
fn season_place(n: usize, s: usize) -> usize {
    if std::env::var("SEASON_RULE").map_or(false, |v| v == "season") {
        return [0, 2, 3, 5][s];
    }
    (2 * n + [0, 1, 3, 4][s] + 1) % PLACES.len()
}

/// FAMILY=1 (schema test with structure): names belong to families, and the family decides
/// the place. Questions read "X <surname> went to the"; trained names: mary, john, sandra
/// smith and daniel, anna, peter jones; new names (held-out): tom smith, lucy jones, sam
/// smith, whose places follow their family's rule.
const SURNAMES: &[&str] = &["smith", "jones"];
fn family() -> bool {
    std::env::var("FAMILY").is_ok()
}
fn family_place(f: usize, s: usize) -> usize {
    [[0, 2, 3, 5], [1, 4, 5, 3]][f][s]
}
fn family_of(name_i: usize, new: bool) -> usize {
    if new { [0, 1, 0][name_i % 3] } else { name_i / 3 }
}
/// FAMILY_STATED=1 (with FAMILY): a new member's family is stated, not named in the
/// question. Some training stories carry a statement "X is a <surname> ." (a trained name,
/// with probability 1/4); with SCHEMA_K=k, each new name is stated in k training stories
/// ("tom is a smith ."), and test questions about new names omit the surname ("tom went
/// to the"). Answering needs the family completed from memory, then the family rule.
fn family_stated() -> bool {
    std::env::var("FAMILY_STATED").is_ok()
}
/// A belief's band, in eighths (0..=7).
fn belief_band(b: Q16) -> usize {
    ((b as u64 * 8) >> 16).min(7) as usize
}

/// What the answer-or-unknown go/no-go sees about a key: the believed value, and the
/// context (the band of its belief, in eighths, × the band of its lead over the runner-up,
/// in sixteenths up to 7/16): how strongly it is believed, and how decisively.
fn decisiveness(rel: &RelationStore, w: usize, r: usize) -> Option<(usize, usize)> {
    let (v, top, lead) = lead_of(rel, w, r)?;
    Some((v, belief_band(top) * 8 + ((lead as u64 * 16) >> 16).min(7) as usize))
}

/// The believed value of (w, relation r), its belief, and its lead over the runner-up
/// (`Q16`).
fn lead_of(rel: &RelationStore, w: usize, r: usize) -> Option<(usize, u32, u32)> {
    let v = rel.bayes.believed(&(r, w, 0, 1))?;
    let top = rel.belief(w, r, 0, 1, v);
    let second = rel.claims(w, r, 0, 1).iter().filter(|c| c.0 != v).map(|c| c.2).max().unwrap_or(0);
    Some((v, top, top.saturating_sub(second)))
}

/// A person's true family (for the teacher and the practice quiz): trained names, new
/// names, practice names.
fn true_family(name: &str, practice_truth: &[usize]) -> Option<usize> {
    if let Some(m) = NAMES.iter().position(|n| *n == name) {
        return Some(family_of(m, false));
    }
    if let Some(i) = NEW_NAMES.iter().position(|n| *n == name) {
        return Some(family_of(i, true));
    }
    PRACTICE_NAMES.iter().position(|n| *n == name).and_then(|j| practice_truth.get(j).copied())
}

/// BELIEF_Q=1: test questions that ask a new name's family (see `season_story_with`).
fn belief_q() -> bool {
    std::env::var("BELIEF_Q").is_ok()
}
/// FAMILY_SHORT=p (with FAMILY): a trained name's question omits the surname with
/// probability p ("mary went to the"), as people are often named by first name only.
fn family_short() -> Option<f64> {
    std::env::var("FAMILY_SHORT").ok().and_then(|v| v.parse().ok())
}

/// Schema test (SCHEMA_K): the new names' own places, one per season. The mapping has the
/// opposite parity to every trained name's, so it cannot be copied from any of them.
fn new_place(i: usize, s: usize) -> usize {
    (2 * i + [0, 1, 3, 4][s] + 4) % PLACES.len()
}

/// SEASON_RULE=random: places are random in every story (no structure to learn; the
/// no-schema control of the schema test).
fn random_places() -> bool {
    std::env::var("SEASON_RULE").map_or(false, |v| v == "random")
}

/// NEW_NAMES=1: held-out test stories use names never seen in training.
const NEW_NAMES: &[&str] = &["tom", "lucy", "sam"];
/// PRACTICE=1 (with BELIEF_UNKNOWN=learned): people met only in training, whose family is
/// stated with a conflict of a known kind and then revealed in a quiz (see `practice_at`).
const PRACTICE_NAMES: &[&str] = &["kim", "joe", "eve", "ian", "amy", "bob", "ann", "dan", "liz", "max", "pam", "ray", "sue", "ted", "una", "vic", "wes", "zoe"];
fn practice() -> bool {
    std::env::var("PRACTICE").is_ok()
}
fn new_names() -> bool {
    std::env::var("NEW_NAMES").is_ok()
}

/// SEASON_NEW_WORDING=1: held-out test questions read "X walked into the" (never seen in
/// training) instead of "X went to the".
fn new_wording() -> bool {
    std::env::var("SEASON_NEW_WORDING").is_ok()
}

/// QUESTION=1 (with FAMILY, FAMILY_STATED): stories where a fact about a stranger arrives
/// without the stranger's name, so answering needs it bound to them within the story:
/// "winter came . kim came . the dog ran away . the person is a smith . mary is a jones . it
/// rained . kim went to the <place by the smith rule> ." The stranger's family is random
/// per story (only this story can tell it), and a known person of the other family is
/// stated too (the story holds both surnames). Strangers: practice names in training (a
/// share QUESTION_P of stories, default 0.5) and at unheld test stories, new names in
/// held-out ones.
fn question() -> bool {
    std::env::var("QUESTION").is_ok()
}

/// QUESTION_POOL=n: training strangers are drawn from n made-up names instead of the 18
/// practice names, so each is met only a few times (a stranger is someone new, as the test's
/// new names are; a practice name met in ~80 stories with random families has a crowded
/// memory that a restated fact cannot stand out in).
fn question_pool() -> &'static [&'static str] {
    static POOL: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let n: usize = std::env::var("QUESTION_POOL").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        (0..n).map(|i| &*Box::leak(format!("stranger{i}").into_boxed_str())).collect()
    })
}

fn question_story(rng: &mut StdRng, held_out: bool) -> Story {
    let season = rng.gen_range(0..SEASONS.len());
    let pool = question_pool();
    let stranger = if held_out {
        NEW_NAMES[rng.gen_range(0..NEW_NAMES.len())]
    } else if !pool.is_empty() {
        pool[rng.gen_range(0..pool.len())]
    } else {
        PRACTICE_NAMES[rng.gen_range(0..PRACTICE_NAMES.len())]
    };
    let f = rng.gen_range(0..SURNAMES.len());
    let known = loop {
        let m = rng.gen_range(0..NAMES.len());
        if family_of(m, false) != f {
            break m;
        }
    };
    let mut words: Vec<&'static str> = vec![SEASONS[season], "came", "."];
    words.extend([stranger, "came", "."]);
    let mut facts: Vec<Vec<&'static str>> = vec![vec!["the", "person", "is", "a", SURNAMES[f], "."], vec![NAMES[known], "is", "a", SURNAMES[1 - f], "."]];
    if rng.gen_bool(0.5) {
        facts.swap(0, 1);
    }
    for fact in facts {
        words.extend(DISTRACTORS.choose(rng).unwrap().iter().copied());
        words.extend(fact);
    }
    words.extend(DISTRACTORS.choose(rng).unwrap().iter().copied());
    words.extend([stranger, "went", "to", "the"]);
    let answer_at = words.len();
    words.extend([PLACES[family_place(f, season)], "."]);
    Story { words, answer_at, held_out }
}

fn season_story(rng: &mut StdRng, distance: usize, held_out: bool) -> Story {
    season_story_with(rng, distance, held_out, None, None)
}

/// `forced`: (new-name index, season) for a schema-test story about a new name.
/// `stated`: a new name whose family this story states (FAMILY_STATED).
fn season_story_with(rng: &mut StdRng, distance: usize, held_out: bool, forced: Option<(usize, usize)>, stated: Option<usize>) -> Story {
    let season = forced.map_or_else(|| rng.gen_range(0..SEASONS.len()), |f| f.1);
    let mut words: Vec<&'static str> = vec![SEASONS[season], "came", "."];
    for _ in 0..distance {
        for _ in 0..rng.gen_range(1..=2) {
            words.extend(DISTRACTORS.choose(rng).unwrap().iter().copied());
        }
    }
    if rng.gen_bool(0.5) {
        words.push(FILLERS.choose(rng).unwrap());
    }
    words.extend(DISTRACTORS.choose(rng).unwrap().iter().copied());
    if family() && family_stated() {
        let stmt = match stated {
            Some(i) => Some((NEW_NAMES[i], family_of(i, true))),
            None if rng.gen_bool(0.25) => {
                let m = rng.gen_range(0..NAMES.len());
                Some((NAMES[m], family_of(m, false)))
            }
            None => None,
        };
        if let Some((who, f)) = stmt {
            words.extend([who, "is", "a", SURNAMES[f], "."]);
        }
    }
    let n = rng.gen_range(0..NAMES.len());
    // schema test: a forced new-name story, or a held-out test question about a new name
    let schema = std::env::var("SCHEMA_K").is_ok();
    let new_i = forced.map(|f| f.0).or_else(|| (schema && held_out).then(|| rng.gen_range(0..NEW_NAMES.len())));
    let name = match new_i {
        Some(i) => NEW_NAMES[i],
        None if held_out && new_names() => NEW_NAMES[n % NEW_NAMES.len()],
        None => NAMES[n],
    };
    let fam = family().then(|| match new_i {
        Some(i) => family_of(i, true),
        None => family_of(n, false),
    });
    // BELIEF_Q=1 (with FAMILY_STATED): half the held-out questions about a new name ask its
    // family itself ("tom is a"), the fact the narrators gave; the other half its place, which
    // the family decides
    if let (Some(_), true, true, Some(f)) = (new_i, family_stated(), forced.is_none() && held_out && belief_q(), fam) {
        if rng.gen_bool(0.5) {
            words.extend([name, "is", "a"]);
            let answer_at = words.len();
            words.extend([SURNAMES[f], "."]);
            return Story { words, answer_at, held_out };
        }
    }
    if held_out && new_wording() {
        words.extend([name, "walked", "into", "the"]);
    } else if let (Some(_), true, true) = (new_i, family_stated(), forced.is_none()) {
        words.extend([name, "went", "to", "the"]);
    } else if let (Some(_), Some(p)) = (fam.filter(|_| new_i.is_none()), family_short().filter(|&p| rng.gen_bool(p))) {
        // FAMILY_SHORT: a trained name's question without the surname
        let _ = p;
        words.extend([name, "went", "to", "the"]);
    } else if let Some(f) = fam {
        words.extend([name, SURNAMES[f], "went", "to", "the"]);
    } else {
        words.extend([name, "went", "to", "the"]);
    }
    let answer_at = words.len();
    let place = match (new_i, fam) {
        (_, Some(_)) if random_places() && new_i.is_none() => rng.gen_range(0..PLACES.len()),
        (_, Some(f)) => family_place(f, season),
        (Some(i), None) => new_place(i, season),
        (None, None) if random_places() => rng.gen_range(0..PLACES.len()),
        (None, None) => season_place(n, season),
    };
    words.extend([PLACES[place], "."]);
    Story { words, answer_at, held_out }
}

/// Book actions: "@open_x" for each physical book. Three books are read at a time (slots);
/// with BOOK_IDS=n a new book gets the next of n actions in turn (default 3: a new book
/// reuses its slot's action).
const BOOKS: &[&str] = &["@open_a", "@open_b", "@open_c", "@open_d", "@open_e", "@open_f", "@open_g", "@open_h", "@open_i", "@open_j", "@open_k", "@open_l"];

/// One reading session of book `b` in `season`; `first` = the book's first session.
fn book_ids() -> usize {
    std::env::var("BOOK_IDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3).clamp(3, BOOKS.len())
}

fn book_session(rng: &mut StdRng, id: usize, season: usize, first: bool, held_out: bool) -> Story {
    let mut words: Vec<&'static str> = vec![BOOKS[id]];
    if first {
        words.extend([SEASONS[season], "came", "."]);
    }
    for _ in 0..rng.gen_range(0..=3) {
        for _ in 0..rng.gen_range(1..=2) {
            words.extend(DISTRACTORS.choose(rng).unwrap().iter().copied());
        }
    }
    if rng.gen_bool(0.5) {
        words.push(FILLERS.choose(rng).unwrap());
    }
    words.extend(DISTRACTORS.choose(rng).unwrap().iter().copied());
    let n = rng.gen_range(0..NAMES.len());
    words.extend([NAMES[n], "went", "to", "the"]);
    let answer_at = words.len();
    words.extend([PLACES[season_place(n, season)], ".", "@close"]);
    Story { words, answer_at, held_out }
}

fn elim_story(rng: &mut StdRng, held_out: bool) -> Story {
    let mut places: Vec<usize> = (0..PLACES.len()).collect();
    places.shuffle(rng);
    let mut words: Vec<&'static str> = Vec::new();
    if rng.gen_bool(0.5) {
        words.push(FILLERS.choose(rng).unwrap());
    }
    for &p in &places[..PLACES.len() - 1] {
        words.extend(["is", "it", "the", PLACES[p], "?", "no", "."]);
    }
    words.extend(["is", "it", "the"]);
    let answer_at = words.len();
    words.extend([PLACES[places[PLACES.len() - 1]], "?", "yes", "."]);
    Story { words, answer_at, held_out }
}

fn topic_story(rng: &mut StdRng, want_held_out: bool) -> Story {
    loop {
        let mut words: Vec<&'static str> = Vec::new();
        for _ in 0..rng.gen_range(0..=4) {
            words.push(FILLERS.choose(rng).unwrap());
        }
        let n = rng.gen_range(0..NAMES.len());
        let p = rng.gen_range(0..PLACES.len());
        if allowed(n, p) == want_held_out {
            continue; // held-out: (name, place) pairs never answered in training
        }
        words.push(NAMES[n]);
        if rng.gen_bool(0.5) {
            words.push(if rng.gen_bool(0.5) { "quickly" } else { "slowly" });
        }
        words.push("went");
        if rng.gen_bool(0.5) {
            words.extend(["all", "the", "way", "over"]);
        }
        words.extend(["to", "the"]);
        if rng.gen_bool(0.5) {
            words.push(if rng.gen_bool(0.5) { "big" } else { "old" });
        }
        words.extend([PLACES[p], "."]);
        for _ in 0..rng.gen_range(1..=3) {
            words.extend(DISTRACTORS.choose(rng).unwrap().iter().copied());
        }
        words.extend(["where", "is", "the", "person", "?"]);
        let answer_at = words.len();
        words.extend([PLACES[p], "."]);
        return Story { words, answer_at, held_out: want_held_out };
    }
}

/// Persist task. Anchors: ANCHOR_NAMES[a], each always at PLACES[(2a + 1) % 6]. All six
/// NAMES stay active, so a name is rarer than the question words (with only three active
/// names they were not, and the recall cue picked up "where is ... right now").
fn anchor_place(a: usize) -> usize {
    (2 * a + 1) % PLACES.len()
}

fn persist_story(rng: &mut StdRng, s_i: usize, ask_anchor: bool, testing: bool) -> Story {
    let mut words: Vec<&'static str> = Vec::new();
    for _ in 0..rng.gen_range(0..=4) {
        words.push(FILLERS.choose(rng).unwrap());
    }
    // (name, place, is_anchor)
    let mut facts: Vec<(&'static str, usize, bool)> = Vec::new();
    if s_i < anchor_stories() {
        // one anchor statement per early story, cycling through the anchors
        let a = s_i % ANCHORS;
        facts.push((ANCHOR_NAMES[a], anchor_place(a), true));
    }
    let mut active: Vec<usize> = (0..NAMES.len()).collect();
    active.shuffle(rng);
    for &n in &active[..rng.gen_range(1..=2)] {
        facts.push((NAMES[n], rng.gen_range(0..PLACES.len()), false));
    }
    facts.shuffle(rng);
    let mut asked = None;
    for &(name, p, anchor) in &facts {
        words.push(name);
        if rng.gen_bool(0.5) {
            words.push(if rng.gen_bool(0.5) { "quickly" } else { "slowly" });
        }
        words.push("went");
        if rng.gen_bool(0.5) {
            words.extend(["all", "the", "way", "over"]);
        }
        words.extend(["to", "the"]);
        if rng.gen_bool(0.5) {
            words.push(if rng.gen_bool(0.5) { "big" } else { "old" });
        }
        words.extend([PLACES[p], "."]);
        if !anchor {
            asked = Some((name, p));
        }
    }
    // during the anchor phase, half the training questions ask about the anchor just
    // stated (retrieval practice: these are what question-tagged replay can tag)
    let practice = !testing && s_i < anchor_stories() && rng.gen_bool(0.5);
    let (q, a) = if practice {
        let n = s_i % ANCHORS;
        (ANCHOR_NAMES[n], anchor_place(n))
    } else if ask_anchor {
        let n = rng.gen_range(0..ANCHORS);
        (ANCHOR_NAMES[n], anchor_place(n))
    } else {
        asked.unwrap()
    };
    words.extend(["where", "is", q, "right", "now", "?"]);
    let answer_at = words.len();
    words.extend([PLACES[a], "."]);
    Story { words, answer_at, held_out: ask_anchor }
}

fn story(rng: &mut StdRng, task: Task, max_facts: usize, want_held_out: bool) -> Story {
    if task == Task::TwoHop {
        return two_hop_story(rng, want_held_out);
    }
    loop {
        let mut words: Vec<&'static str> = Vec::new();
        for _ in 0..rng.gen_range(0..=4) {
            words.push(FILLERS.choose(rng).unwrap());
        }
        let facts = rng.gen_range(1..=max_facts);
        let mut names: Vec<usize> = (0..NAMES.len()).collect();
        names.shuffle(rng);
        let mut loc = Vec::new();
        for &n in &names[..facts] {
            let p = rng.gen_range(0..PLACES.len());
            loc.push((n, p));
            words.push(NAMES[n]);
            if task == Task::Varied && rng.gen_bool(0.5) {
                words.push(if rng.gen_bool(0.5) { "quickly" } else { "slowly" });
            }
            words.push("went");
            if task != Task::Short && rng.gen_bool(0.5) {
                words.extend(["all", "the", "way", "over"]);
            }
            words.extend(["to", "the"]);
            if task == Task::Varied && rng.gen_bool(0.5) {
                words.push(if rng.gen_bool(0.5) { "big" } else { "old" });
            }
            words.extend([PLACES[p], "."]);
        }
        let (q, a) = loc[rng.gen_range(0..facts)];
        let ok = want_held_out || loc.iter().all(|&(n, p)| allowed(n, p));
        if !ok || allowed(q, a) == want_held_out {
            continue;
        }
        words.extend(["where", "is", NAMES[q]]);
        if task != Task::Short {
            words.extend(["right", "now"]);
        }
        words.push("?");
        let answer_at = words.len();
        words.extend([PLACES[a], "."]);
        return Story { words, answer_at, held_out: want_held_out };
    }
}

struct Outcome {
    seen: f64,
    held_out: f64,
    /// % of test questions where the memory (or relay) frames contained the answer word
    recall: f64,
    /// mean number of distinct place words in the memory frames at the answer
    places: f64,
}

/// The words of a task's stories, in a fixed order (their codes follow it).
fn task_vocab(task: Task) -> Vec<&'static str> {
    let mut vocab: Vec<&'static str> = vec![
        "went", "to", "the", ".", "where", "is", "?", "all", "way", "over", "right", "now", "quickly", "slowly", "big", "old", "picked", "up",
    ];
    vocab.extend(OBJECTS);
    vocab.extend(ANCHOR_NAMES);
    vocab.extend(["person", "cat", "slept", "dog", "ran", "away", "it", "rained"]);
    vocab.extend(NAMES);
    vocab.extend(PLACES);
    vocab.extend(FILLERS);
    if task == Task::Give {
        // appended only for this task, so the other tasks' codes and rng are unchanged
        vocab.extend(["gave", "what", "did", "give", "who", "got"]);
    }
    if task == Task::Elim {
        vocab.extend(["no", "yes"]);
    }
    if task == Task::Habit {
        vocab.extend(["in", "morning", "at", "night"]);
    }
    if task == Task::Season || task == Task::Books {
        vocab.extend(SEASONS);
        vocab.push("came");
    }
    if task == Task::Season && new_wording() {
        vocab.extend(["walked", "into"]);
    }
    if task == Task::Season && (new_names() || std::env::var("SCHEMA_K").is_ok() || question()) {
        vocab.extend(NEW_NAMES);
    }
    if task == Task::Season && (practice() || question()) {
        vocab.extend(PRACTICE_NAMES);
        if question() {
            vocab.extend(question_pool());
        }
    }
    if task == Task::Season && family() {
        vocab.extend(SURNAMES);
        if family_stated() {
            vocab.push("a");
        }
    }
    if task == Task::Books {
        vocab.extend(&BOOKS[..book_ids()]);
        vocab.push("@close");
    }
    vocab
}

fn run(policy: Policy, task: Task, max_facts: usize, seed: u64) -> Outcome {
    if std::env::var("GENOME").is_ok() {
        return run_genome(task, max_facts, seed);
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let vocab = task_vocab(task);
    let index: HashMap<&str, usize> = vocab.iter().enumerate().map(|(i, w)| (*w, i)).collect();
    let enc = Encoder::new(vocab.len(), BITS, 32, &mut rng);
    // Independent random streams, one per subsystem, split off the seed: switching a
    // subsystem on (or changing what it does) draws only from its own stream, so the rest
    // of the run is unchanged and differences between configurations can be attributed.
    // `rng` stays for setup and the cortex's waking learning.
    let stream = |k: u64| StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    let mut story_rng = stream(1); // which story is read, and its words
    let mut bg_rng = stream(2); // the basal ganglia's exploration and learning
    let mut sleep_rng = stream(3); // offline: replay order, consolidation, sleep learning
    let mut rel_rng = stream(4); // the relation store
    let mut noise_rng = stream(5); // altered feedback

    let mut routes = vec![
        RelayChannel { query_lag: 1, value_offset: 4 },
        RelayChannel { query_lag: 3, value_offset: 4 },
        RelayChannel { query_lag: 3, value_offset: 8 },
    ];
    if task == Task::Give {
        // the two routes this task needs, alongside the three of 09-10
        routes.extend([RelayChannel { query_lag: 2, value_offset: 3 }, RelayChannel { query_lag: 1, value_offset: 2 }]);
    }
    // HIER=1: a higher cortical area above the column; its prediction is one more frame
    let hier = std::env::var("HIER").is_ok();
    let base_frames = match policy {
        Policy::NoMemory => !hier as usize,
        Policy::Episodic | Policy::Consolidate | Policy::Pfc { .. } | Policy::Ca3 { .. } => 1,
        Policy::Loop(hops) => hops,
        Policy::Branch(b) => b + 1,
        Policy::Select => 2,
        Policy::ThalamicGate | Policy::LearnedGate => 1,
        Policy::L6Gate { .. } => routes.len() + 1,
        Policy::FixedRelay => routes.len(),
    };
    let mid_frames = base_frames + hier as usize;
    let mut th = Thalamus::new(BITS, 60, routes.clone());
    let capacity = std::env::var("CAPACITY").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let mut memory = EpisodicMemory::new(BITS, capacity); // also keeps the habituation statistics
    // Consolidate: the cortical semantic store (cue -> content), trained only by replay
    let mut semantic: KernelClass<SimpleKernel> = KernelClass::predictive(GrowthConfig {
        max_kernels: 20_000,
        frame_words: BITS / 64,
        max_frames: 1,
        sample_bits: 16,
        match_fraction: 0.8,
        surprise_fraction: 0.5,
        generalize: None,
        generalize_after: 1,
    });
    let replays: usize = std::env::var("REPLAYS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    // sleep happens after a training story with this probability (small budgets)
    let replay_prob: Q16 = q16(std::env::var("REPLAY_PROB").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.0));
    // REPLAY_MODE: random (default) | tagged (a question whose recall predicted the answer
    // tags that episode; sleep replay favours tags) | awake (that episode is replayed into
    // cortex at once, at the question: prefrontal-driven retrieval)
    let replay_mode = std::env::var("REPLAY_MODE").unwrap_or_else(|_| "random".into());
    let tag_boost: u32 = std::env::var("TAG_BOOST").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let mut last_recall_id: Option<usize> = None; // hippocampal episode recalled this step
    let mut last_recall_cue: Option<BitVector> = None; // the cue that recalled it
    // tagged replay: the question's cue, stored with the tag, keys the sleep replay
    let mut tag_cues: HashMap<usize, BitVector> = HashMap::default();
    // Pfc: one-slot working memory and its basal-ganglia gate
    let mut wm = WorkingMemory::new(BITS, 1);
    // REWARD=l5: every basal-ganglia selector learns from the column's L5 outcome (did the
    // whole prediction come true?) instead of checking its own content against the target
    let l5_reward = std::env::var("REWARD").map_or(false, |v| v.starts_with("l5"));
    // REWARD=l5_used: only if the prediction read the chosen content's L4 frame
    let l5_used = std::env::var("REWARD").map_or(false, |v| v == "l5_used");
    // L4 frame that the selector's choice fills: hop 2 for Select, else the memory frame
    // (with HIER_EARLY the top-down frame sits at 1 and the policy's frames move up by one)
    let early_shift = (std::env::var("HIER").is_ok() && std::env::var("HIER_EARLY").is_ok()) as usize;
    let choice_frame = if policy == Policy::Select { 2 } else { 1 } + early_shift;
    let mut l5_sum = [0f64; 2]; // training: summed L5 reward, count
    let pfc_trace: usize = std::env::var("PFC_TRACE").ok().and_then(|v| v.parse().ok()).unwrap_or(24);
    let mut pfc_gate = PfcGate::new(BITS, pfc_trace, q16(0.9), seed + 11);
    let mut pfc_loads = HashMap::<&str, (usize, usize)>::default(); // word -> (loads, decisions) at test
    let (mut pfc_rewards, mut pfc_questions) = (0usize, 0usize); // training
    let names_set: neurocomp::det::HashSet<usize> = NAMES.iter().map(|n| index[n]).collect();
    let mut tagged_or_replayed = 0usize;
    let mut from_cortex = 0usize; // test answers where the memory frame came from the cortex
    let (dg, mut ca3) = match policy {
        Policy::Ca3 { cells, k, settle } => {
            let env = |name: &str, default: f32| std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
            let fan_in = env("DG_FAN_IN", 300.0) as usize;
            let decay = env("CA3_DECAY", 0.7);
            let mut ca3: Box<dyn Autoassociative> = match std::env::var("CA3_STORE").as_deref() {
                Ok("ring") => Box::new(Ca3Memory::new_delay_line(BITS, cells, k, decay, settle)),
                Ok("shift") => Box::new(Ca3Memory::new_shift_register(BITS, cells, k, 11, settle)),
                Ok("float") => Box::new(Ca3FloatMemory::new(BITS, cells, k, decay, settle)),
                _ => Box::new(Ca3Memory::new(BITS, cells, k, decay, settle)),
            };
            ca3.set_readout_fraction(q16(env("CA3_READOUT", 0.5) as f64));
            (Some(DentateGyrus::new(BITS, cells, fan_in, k, seed + 100)), Some(ca3))
        }
        _ => (None, None),
    };
    let mut sentence = BitVector::new(BITS, Some(0)); // bag of the current sentence so far
    // COOPERATE=1: what memory says about the sentence's uncertain words, added to the higher
    // area's sentence context (experiment 63)
    let cooperate = std::env::var("COOPERATE").is_ok();
    // GRADED=1 (with MIX): both stores always answer; the mix weighs each by the column's
    // confidence band (experiment 64)
    let graded = std::env::var("GRADED").map_or(false, |v| v != "enrich");
    // GRADED=enrich: the semantic store always answers into the higher areas' context, each of
    // its bits getting in with probability 1 − the column's confidence
    let graded_enrich = std::env::var("GRADED").map_or(false, |v| v == "enrich");
    let mut graded_stats = [0usize; 1];
    let mut coop_stats = [0usize; 2]; // enrichments, of them in held-out test stories
    let mut bg = BasalGanglia::new(BITS);
    bg.trace_len = std::env::var("BG_TRACE").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    // BG_BASELINE=rate: the selector's error is reward − a running-average reward
    bg.baseline_rate = std::env::var("BG_BASELINE").ok().and_then(|v| v.parse().ok());
    let mut bg_pending: Option<BitVector> = None; // hop-2 content of the latest choice, awaiting reward
    // ThalamicGate: channel identity codes (routes, then memory), bound to the current word
    // by rotation, so the striatum holds a value per (channel, context)
    let mut gate_bg = BasalGanglia::new(BITS);
    // channel identity code: a fixed random sparse code per route (q, v), and one for memory
    let channel_code = |r: Option<RelayChannel>| -> BitVector {
        let key = r.map_or(0, |r| 1 + r.query_lag as u64 * 64 + r.value_offset as u64);
        let mut crng = StdRng::seed_from_u64(seed.wrapping_mul(1_000_003) ^ (key + 7));
        let all: Vec<usize> = (0..BITS).collect();
        BitVector::from_bits(&all.choose_multiple(&mut crng, 32).copied().collect::<Vec<_>>(), BITS)
    };
    let mut gate_pending: Option<BitVector> = None; // released content, awaiting reward
    // MIX=1: precision-weighted mixing of sources (thalamic gain, not a switch). The
    // column's own prediction, memory (recall / relay frames) and the higher area's
    // top-down prediction each vote for their words, weighted by their learned reliability
    // per (previous word, current word, own-confidence bucket); the word with the most
    // evidence is the prediction (see `SourceMix`)
    let mixing = std::env::var("MIX").is_ok();
    // MIX_TEST_LEARN=1: the mix's reliability counters keep learning at test (online
    // arbitration; nothing else learns, and test stories are not stored in memory)
    let mix_test_learn = std::env::var("MIX_TEST_LEARN").is_ok();
    // held-out (new-name) answers right, in the first and second half of the test
    let mut held_halves = [(0usize, 0usize); 2];
    let mut mix = SourceMix::new();
    // test answers: [answers, column right, mix right, changed, changed and right,
    // sources agreed, agreed and right]
    let mut mix_stats = [0usize; 7];
    // L6Gate: the corticothalamic gate, this step's context and which channels were open
    // with content; test counts: open channels per word, and per channel at answers
    let mut l6_gate = CorticothalamicGate::new(BITS, routes.len() + 1, seed + 21);
    let mut l6_step: Option<(BitVector, Vec<bool>)> = None;
    let l6_pair = std::env::var("L6_CONTEXT").map_or(false, |v| v == "pair");
    // L6_WARMUP=n: all channels stay open for the first n training stories while the gate
    // learns (cortex first learns to use the relays); L6_WEAKEN=w: weakening rate factor
    let l6_warmup: usize = std::env::var("L6_WARMUP").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    l6_gate.weaken = q16x(std::env::var("L6_WEAKEN").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.0));
    let (mut l6_open_sum, mut l6_words) = (0usize, 0usize);
    // [question kind][channel]: kind 1 = "what did X give ?", else 0
    let mut l6_open_at_answer = vec![vec![0usize; routes.len() + 1]; 2];
    let mut l6_answers = [0usize; 2];
    let mut gate_chosen_at_answer: HashMap<String, usize> = HashMap::default();
    // LearnedGate: route discovery and the current pool
    let mut route_scores = RouteScores::default();
    let mut gate_routes: Vec<RelayChannel> = if policy == Policy::ThalamicGate { routes.clone() } else { Vec::new() };
    // CA1-style comparator (NOVELTY=prediction): store, cue and read out only what the
    // predictor failed to predict, instead of frequency habituation.
    let predictive_novelty = std::env::var("NOVELTY").map_or(false, |v| v == "prediction");
    let mut surprising = BitVector::new(BITS, Some(0)); // unpredicted bits of the sentence so far
    let predicted_share: Q16 = q16(std::env::var("PREDICTED_SHARE").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.5));
    // recalled content: drop bits in more than 40% of episodes (all kept with prediction novelty)
    let habituation: Q16 = if predictive_novelty { ONE } else { Q_04 };
    // which recalled item to cue with: the rarest among stored items (an IDF-like
    // specificity); RARITY=all cues with everything unpredicted
    let rarity_ratio: Q16 = if std::env::var("RARITY").map_or(false, |v| v == "all") { u32::MAX } else { ONE * 3 / 2 };
    let min_overlap = 8; // floor for the recall threshold

    // LEARNING=three: three learning systems. The column's L2/3 becomes the slow cortex (a
    // miss grows a kernel only with probability SLOW_P, 1/16, with near-miss generalisation
    // SLOW_GEN, 0.5); the fast one-shot rule it used moves to a `Cerebellum`, which reads a
    // copy of the column's input (cortex → pons → mossy fibres) and whose prediction returns
    // through the thalamus (deep nuclei → thalamus → cortex) as a routed channel before the
    // cortex predicts (with ROUTE), and votes in the mix. The hippocampus is the third.
    let three = std::env::var("LEARNING").map_or(false, |v| v == "three");
    let slow_p: Q16 = q16(std::env::var("SLOW_P").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.0 / 16.0));
    let slow_gen: f32 = std::env::var("SLOW_GEN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
    let make_l23 = |generalize: Option<f32>| -> KernelClass<SimpleKernel> {
    let mut class: KernelClass<SimpleKernel> = KernelClass::predictive(GrowthConfig {
        max_kernels: 100_000,
        frame_words: BITS / 64,
        // ROUTE_EXTRA: room for routed channels beyond the hand row (HC_ROUTE: one, for the
        // entorhinal feedback channel)
        max_frames: mid_frames + 2 + route_extra_slots(),
        sample_bits: 16,
        match_fraction: 0.8,
        surprise_fraction: 0.5,
        // GENERALIZE=f: drop silent inputs of near-matching kernels that would have been
        // right (synapse-level credit), after GENERALIZE_AFTER misses; off by default
        generalize: generalize.or_else(|| std::env::var("GENERALIZE").ok().and_then(|v| v.parse().ok())),
        generalize_after: std::env::var("GENERALIZE_AFTER").ok().and_then(|v| v.parse().ok()).unwrap_or(1),
    });

    // TRUST=f: depth only outranks reliability among kernels at least f reliable
    class.set_trust_floor(ratio_env("TRUST"));
    // SPECIFIC=1: within a depth, the kernel that matched more of the input wins before
    // reliability is compared (consolidated, context-specific kernels over general ones)
    class.set_specificity(std::env::var("SPECIFIC").is_ok());
    // COMPETE=evidence: the column's winner by evidence (each output's matched kernels'
    // reliability × their depth's learned gain), not by the ranking depth-then-reliability
    class.set_evidence_competition(std::env::var("COMPETE").map_or(false, |v| v == "evidence"));
    // COMPETE=burst: the winner is the matched kernel reading input and context together
    // whose recent coincidences were confirmed (KernelClass::set_burst_competition)
    class.set_burst_competition(std::env::var("COMPETE").map_or(false, |v| v == "burst"));
    // STICKY=f: credit-tagged synapses (input bits that carried the correctly predicted
    // word) need f times as many silent confirmations before pruning
    if let Some(f) = std::env::var("STICKY").ok().and_then(|v| v.parse().ok()) {
        class.set_sticky(f, None);
    }
    // STICKY_BLAME=1: a tag is released when its bit was active on a misprediction
    class.set_sticky_blame(std::env::var("STICKY_BLAME").is_ok());
    // COPY_GROW=1: new kernels sample only the target's bits in a frame that contains them
    class.set_copy_growth(std::env::var("COPY_GROW").is_ok());
    // GROW_TRUST=f: a lucky unreliable kernel (< f) does not block growth of a better one
    // FAST_INHIBIT=hits|misses|both: L2/3 fast inhibitory loop, tags lasting FAST_TTL steps
    if let Ok(mode) = std::env::var("FAST_INHIBIT") {
        let ttl: u8 = std::env::var("FAST_TTL").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
        let (h, m) = (mode == "hits" || mode == "both", mode == "misses" || mode == "both");
        let mut fast = neurocomp::kernel::FastInhibition::new(ttl, h, m);
        // FAST_RELSHIFT=k: only kernels with hit rate below 2^k/(2^k+1) are tagged
        // (integer test (misses+1) << k > hits+1; k = 3 ≈ 0.89)
        fast.reliable_shift = std::env::var("FAST_RELSHIFT").ok().and_then(|v| v.parse().ok());
        class.set_fast_inhibition(Some(fast));
    }
    class.set_growth_trust(ratio_env("GROW_TRUST"));
    // SURPRISE_GATE=1: learn fully only on surprise; an expected word confirms the winner
    class.set_surprise_gate(std::env::var("SURPRISE_GATE").is_ok());
    // MEMO=1: memoised interpretation (input hash -> winner while the prior is unchanged)
    class.set_memo(std::env::var("MEMO").is_ok());
    // FRAME_MEMO=1: per-frame memo of match counts, invalidated per kernel
    class.set_frame_memo(std::env::var("FRAME_MEMO").is_ok());
    // CANON=1: canonical kernels (deterministic sampling + hash-consing at growth)
    class.set_canonical(std::env::var("CANON").is_ok());
    // REPLAY_LEN=n: recent inputs kept for sleep replay (default 512 when SLEEP_EVERY is set)
    if std::env::var("SLEEP_EVERY").is_ok() {
        class.set_replay(std::env::var("REPLAY_LEN").ok().and_then(|v| v.parse().ok()).unwrap_or(512));
    }
    // GROW_GATE=k: uncertainty-gated growth (no growth on misses in contexts known to be
    // random, hit rate < 2^k/(2^k+1) over at least UNC_MIN observations, when no input
    // frame carries the target)
    if let Some(k) = std::env::var("GROW_GATE").ok().and_then(|v| v.parse().ok()) {
        let min: u16 = std::env::var("UNC_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
        class.set_growth_gate(Some((k, min)));
    }
    class
    };
    let mut class = make_l23(three.then_some(slow_gen).filter(|&f| f > 0.0));
    if three {
        class.set_growth_probability(Some(slow_p));
        // SLOW_GATE=0: no uncertainty growth gate on the slow cortex (a slow learner keeps
        // missing while it has not grown yet, which the gate reads as a noisy context)
        if std::env::var("SLOW_GATE").map_or(false, |v| v == "0") {
            class.set_growth_gate(None);
        }
    }
    let mut cerebellum: Option<neurocomp::program::Cerebellum> = three.then(|| neurocomp::program::Cerebellum::new(BITS, make_l23(None)));
    let mut cb_rng = StdRng::seed_from_u64(seed.wrapping_add(5151));
    let mut cb_input: Option<BitVector> = None; // the mossy-fibre input of this step, for learning
    let mut cb_word: Option<usize> = None;
    let mut cb_conf: Q16 = 0;
    let mut cb_stats = [0usize; 2]; // test answers: proposed, right
    // the cortical column: L4 input assembly, L2/3 predictor (`class`), L5 prediction /
    // confidence / surprise, L6 context (`th`, whose match rules the thalamus gates)
    let mut column = CorticalColumn::new(BITS, class, th);
    // SLOW_CORTEX=1: three learning systems. The column's kernels grow in one shot on every
    // miss, each output corrected by the next word: the cerebellum's rule, kept as the fast,
    // precise learner. Beside it, a slow cortex reads the same input and learns slowly and
    // generally: a miss grows a kernel only with probability SLOW_P (1/16), so a context is
    // learned once it recurs, with near-miss generalisation (SLOW_GEN, 0.5), from waking and
    // replay alike (complementary learning systems: the hippocampus fast and detailed, the
    // cortex slow and statistical). It votes in the thalamic mix with its own record.
    let mut slow: Option<KernelClass<SimpleKernel>> = std::env::var("SLOW_CORTEX").is_ok().then(|| {
        let mut c = KernelClass::predictive(GrowthConfig {
            max_kernels: 100_000,
            frame_words: BITS / 64,
            max_frames: mid_frames + 2 + route_extra_slots(),
            sample_bits: 16,
            match_fraction: 0.8,
            surprise_fraction: 0.5,
            generalize: Some(std::env::var("SLOW_GEN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5)).filter(|&f: &f32| f > 0.0),
            generalize_after: 1,
        });
        c.set_surprise_gate(true);
        c.set_canonical(true);
        c.set_growth_probability(Some(slow_p));
        c
    });
    let mut slow_rng = StdRng::seed_from_u64(seed.wrapping_add(4242));
    let mut slow_word: Option<usize> = None;
    let mut slow_conf: Q16 = 0;
    let mut slow_stats = [0usize; 3]; // answers: proposed, right; the mix chose its word
    // the higher area (HIER=1): its own predictive L2/3 over [sentence bag | slow state],
    // the slow state spanning the last HIER_SPAN sentences' surprises
    let hier_span: usize = std::env::var("HIER_SPAN").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    // HIER_LEVELS=n: n higher areas in a chain (default 1). Area k+1's window is HIER_SPAN
    // times area k's, and each area (but the top) also reads the prediction of the area
    // above it as one more frame, so top-down flows down the chain to the column
    let hier_levels: usize = std::env::var("HIER_LEVELS").ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
    // HIER_CHAIN=mix: the upper areas do not feed the area below; each votes in the
    // precision-weighted mix (MIX) as one more source, weighted by its own reliability
    let chain_mix = std::env::var("HIER_CHAIN").map_or(false, |v| v == "mix");
    // ROLE=cells|raw: a leading frame for area 2, [role | slow state | sentence]: the role
    // cells' code for the column's expectation of the next slot (cells), or that expectation
    // itself (raw), both learned
    let role_mode = std::env::var("ROLE").ok();
    let make_area = |span: usize, with_above: bool, lead: bool| {
        let mut c: KernelClass<SimpleKernel> = KernelClass::predictive(GrowthConfig {
            max_kernels: 100_000,
            frame_words: BITS / 64,
            max_frames: if std::env::var("HIER_SEPARATE").is_ok() { 1 + span } else { 2 } + with_above as usize + lead as usize,
            sample_bits: 16,
            match_fraction: 0.8,
            surprise_fraction: 0.5,
            // HIER_GENERALIZE=f: near-miss generalization in the higher area (off: it pruned
            // the time-of-day bits along with the noise)
            generalize: std::env::var("HIER_GENERALIZE").ok().and_then(|v| v.parse().ok()).filter(|&f: &f32| f > 0.0),
            generalize_after: std::env::var("GENERALIZE_AFTER").ok().and_then(|v| v.parse().ok()).unwrap_or(1),
        });
        c.set_surprise_gate(true);
        c.set_canonical(true);
        // HIER_GEN_SPAWN=1: near-miss generalisation in the higher area spawns a general
        // copy and keeps the specific kernel (with HIER_GENERALIZE)
        c.set_generalize_spawn(std::env::var("HIER_GEN_SPAWN").is_ok());
        // HIER_SPAWN_AFTER=n: spawn a copy only after n confirmed near misses without the
        // dropped inputs (default 1)
        c.set_spawn_after(std::env::var("HIER_SPAWN_AFTER").ok().and_then(|v| v.parse().ok()).unwrap_or(1));
        // HIER_GROW_TRUST=p/q: growth trust floor in the higher area (did not help)
        c.set_growth_trust(ratio_env("HIER_GROW_TRUST"));
        // HIER_SLEEP=1: sleep compacts the higher area too (half the cost, a few points less
        // on habit). No uncertainty gate on its growth: the gate stops growth when no input
        // frame carries the target, but the higher area's target (the column's residual) is
        // never in its input, by design
        // HIER_SLEEP_GEN=n: generalisation during the higher area's sleep (inputs absent in n
        // confirmed near misses over the replay; candidates tested on the replay)
        c.set_sleep_generalize(std::env::var("HIER_SLEEP_GEN").ok().and_then(|v| v.parse().ok()));
        // HIER_DREAM=1: the higher area only generalises from its replay at each sleep time
        // (no downscaling, pruning or merging)
        if std::env::var("HIER_DREAM").is_ok() {
            c.set_replay(std::env::var("HIER_REPLAY_LEN").ok().and_then(|v| v.parse().ok()).unwrap_or(512));
        }
        if std::env::var("HIER_SLEEP").is_ok() {
            c.set_replay(std::env::var("HIER_REPLAY_LEN").or_else(|_| std::env::var("REPLAY_LEN")).ok().and_then(|v| v.parse().ok()).unwrap_or(512));
        }
        // REPLAY_GEN=1: sleep generalisation reads only the hippocampus's free-settling
        // replay (see below), not a buffer of the area's own waking experience
        if std::env::var("REPLAY_GEN").is_ok() {
            c.set_record_waking(false);
        }
        let mut a = HigherArea::new(BITS, c, span);
        // HIER_FADE=f: a fading state instead of the window, half-life f × the area's span
        // (in sentences)
        a.set_fade(std::env::var("HIER_FADE").ok().and_then(|v| v.parse::<f64>().ok()).map(|f| f * span as f64));
        // HIER_SEPARATE=1: one slow-state frame per recent sentence (worse: deep kernels at
        // varying lags); default: one frame with all recent surprises
        a.separate = std::env::var("HIER_SEPARATE").is_ok();
        // HIER_FOCUS=1: attention to one remembered word at growth (did not help)
        a.focus = std::env::var("HIER_FOCUS").is_ok();
        a
    };
    let mut area = make_area(hier_span, hier_levels > 1 && !chain_mix && role_mode.is_none(), role_mode.is_some());
    // ASSOC=1 (with HIER): an association area, the hippocampus's cortical partner
    // (perirhinal / parahippocampal cortex behind the entorhinal gate). An area like the
    // higher area ([sentence | slow context] → next word), with its own role:
    // - it learns slowly while awake (growth probability ASSOC_P, default 1/16) and readily
    //   from consolidation replay while asleep (growth probability 1, gate open): replay goes
    //   to it instead of the higher area;
    // - the hippocampus's return goes to it: reinstated states (REINSTATE) join its context,
    //   not the higher area's;
    // - it reaches reading through the thalamic mix, as a source of its own (ASSOC_SRC) whose
    //   reliability per context is learned like any other's.
    let assoc_on = hier && std::env::var("ASSOC").is_ok();
    let assoc_p: Q16 = q16(std::env::var("ASSOC_P").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.0 / 16.0));
    let mut assoc: Option<HigherArea> = assoc_on.then(|| {
        let mut a = make_area(hier_span, false, false);
        a.column.l23.set_growth_probability(Some(assoc_p));
        a
    });
    let mut own_diag = [0usize; 3]; // OWNDIAG, test answers: all, the column decodes a word, any bits
    let mut own_diag_k = [0usize; 4]; // OWNDIAG: Σ matched kernels, Σ proposing a place, Σ proposing the answer, answers where any matched kernel proposes it
    let mut assoc_in: Option<BitVector> = None; // this step's input, for learning
    let mut assoc_word: Option<usize> = None; // this step's proposal
    let mut assoc_conf: Q16 = 0;
    let mut assoc_stats = [0usize; 3]; // test answers: proposed, right; replays taught
    let mut upper: Vec<HigherArea> = (1..hier_levels).map(|j| make_area(hier_span.pow(j as u32 + 1), j + 1 < hier_levels && !chain_mix, false)).collect();
    // HIER_GROW=1 (with HIER_CHAIN=mix): areas grow by need. The top area keeps a bud above
    // it, the last of `upper`, with a window HIER_SPAN times longer. The bud reads and learns
    // like any area (the column's residual) and its vote is weighed by the mix (its
    // reliability is learned) but not counted: it runs in shadow. Each training word, the
    // mix's choice with and without the bud is compared: a fix (wrong without it, right with
    // it) or a break (the reverse); a story's answer counts HIER_GROW_ANSWER times (16).
    // Every HIER_GROW_EVERY training stories (250), a bud whose fixes beat its breaks by more
    // than HIER_GROW_Z (2) standard deviations (a sign test), with at least HIER_GROW_MIN (32)
    // fixes, is promoted to a full area and a new bud starts above it (up to
    // HIER_GROW_MAX areas above the column, 4); a bud not promoted in HIER_GROW_PATIENCE
    // checks (4) is pruned and a fresh one starts. No bud at test.
    let hier_grow = std::env::var("HIER_GROW").is_ok() && chain_mix;
    let grow_every: usize = std::env::var("HIER_GROW_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(250);
    let grow_answer: u64 = std::env::var("HIER_GROW_ANSWER").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let grow_z: u64 = std::env::var("HIER_GROW_Z").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let grow_min: u64 = std::env::var("HIER_GROW_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
    let grow_max: usize = std::env::var("HIER_GROW_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    let grow_patience: usize = std::env::var("HIER_GROW_PATIENCE").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    if hier_grow {
        upper.push(make_area(hier_span.pow(upper.len() as u32 + 2), false, false));
    }
    let mut bud_live = hier_grow;
    let mut bud_tally = (0u64, 0u64); // (fixes, breaks) since the last check
    let mut bud_checks = 0usize; // checks the current bud has failed
    let mut grow_log: Vec<String> = Vec::new();
    // the mix's source id of upper area i (3, 4, 5, then past the other sources)
    let upper_src = |i: usize| if i < 3 { 3 + i as u8 } else { 10 + i as u8 };
    let mut role_rng = StdRng::seed_from_u64(seed.wrapping_add(77));
    let mut roles = RoleArea::new(BITS, 64, &mut role_rng);
    // test: per role cell, how often each word filled the slot it fired for (description only)
    let mut role_words: HashMap<usize, HashMap<usize, u32>> = HashMap::default();
    let mut role_now: Option<usize> = None;
    // BIND=1: slot ⊗ content episodes (after the Tolman-Eichenbaum Machine).
    // - Slots: role cells fed with the column's expectation for the next slot and the
    //   previous slot's code (so slots follow the sequence of a story's structure).
    // - Binding: each surprising word is stored rotated by its slot's offset; a story's
    //   episode is the union of its bindings (training stories only).
    // - Recall: the story's bindings so far cue the store; the recalled episode, unbound
    //   with the slot the column expects next, answers "what filled this slot in the
    //   matching episode?". The answer is a source in the mix (MIX), source 6.
    let bind = std::env::var("BIND").is_ok();
    // BIND_HAB=f: habituation of the cue, leaving out bindings present in more than a share f
    // of stored episodes (the store's own frequency statistics)
    let bind_hab: Option<Q16> = std::env::var("BIND_HAB").ok().and_then(|v| v.parse::<f64>().ok()).map(q16);
    let mut bind_mem = EpisodicMemory::new(BITS, 5000);
    // HIPPO=ca3: the slot ⊗ content episodes are stored in and recalled from the learned
    // hippocampus of experiment 12: a dentate gyrus (random expansion + k-WTA) gives each
    // episode a sparse CA3 code; CA3 stores EC→CA3, CA3↔CA3 and CA3→EC with Hebbian,
    // bit-sliced, decaying weights; recall drives CA3 from the cue, settles through the
    // recurrent weights (pattern completion) and reads the EC pattern back out. The list
    // store `bind_mem` then only supplies the familiarity statistics (perirhinal-like) and
    // the habituated cue. HIPPO_CELLS (8192), HIPPO_K (32), HIPPO_SETTLE (2),
    // HIPPO_DECAY (0.999 per store), HIPPO_HAB (cue habituation, default 0.3: bindings in
    // more than that fraction of episodes are left out of the cue, as entorhinal
    // adaptation would; 0 = the whole story's bindings).
    let hippo_ca3 = std::env::var("HIPPO").map_or(false, |v| v == "ca3");
    let henv = |name: &str, default: f32| std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
    let (hcells, hk) = (henv("HIPPO_CELLS", 8192.0) as usize, henv("HIPPO_K", 32.0) as usize);
    let hippo_hab: Q16 = q16(henv("HIPPO_HAB", 0.3) as f64);
    let bind_dg = hippo_ca3.then(|| DentateGyrus::new(BITS, hcells, 300, hk, seed + 200));
    // HIPPO=full: the full circuit (`program::Hippocampus`): EC II → DG → CA3 by mossy
    // fibres at storage, EC II → CA3 (presynaptically scaled) at recall, CA3 recurrent
    // settling, CA3 → CA1 (Schaffer) and EC III → CA1 (comparator), CA2 temporal context,
    // CA1 → subiculum → EC V readout, novelty-gated encoding, cached (event-based) recall.
    // HIPPO_GAIN (novelty gain, 3), HIPPO_CA2 (CA2 → CA1 weight in quarters, 0 = off),
    // HIPPO_DECAY (per-store decay, 0.999), HIPPO_CELLS (CA3 and CA1 cells; DG twice that).
    let mut index_hc: Option<Box<dyn EpisodicCircuit>> = None;
    // HIPPO=index: the index memory (`program::IndexMemory`): one row per event, recall by
    // winner-take-all over an inverted index with 1/n cue weights, successor pointers,
    // strength-based forgetting (INDEX_PERIOD stores per unit, 64; INDEX_PERIOD=0: none),
    // INDEX_PLAIN=1: plain overlap. Use with HIPPO_SELF=1 SPARSE_BIND=1.
    // HIPPO=engram: the engram store (`program::EngramStore`): rows of binding ids and grid
    // phases (the story is the place), theta-cycle deduplication, a ring buffer whose write
    // head evicts weak rows, replay by strength × the cortex's error. ENGRAM_TAU (64),
    // ENGRAM_CAPACITY (65536), ENGRAM_BONUS (where bonus, in ids; 0: an additive bonus
    // even of 1/16 id outweighs content matches of common bindings, experiment 56).
    if engram_mode() {
        let mut cfg = EngramConfig::default();
        cfg.tau = henv("ENGRAM_TAU", 64.0) as u32;
        if cfg.tau == 0 {
            cfg.tau = u32::MAX;
        }
        cfg.capacity = henv("ENGRAM_CAPACITY", 65536.0) as usize;
        cfg.where_bonus = q16x(henv("ENGRAM_BONUS", 0.0) as f64);
        cfg.seed_place = std::env::var("ENGRAM_SEED_PLACE").is_ok();
        // ENGRAM_WALK=1: recall walks one step through a rare cue binding (experiment 59)
        cfg.walk = std::env::var("ENGRAM_WALK").is_ok();
        cfg.walk_rare = henv("ENGRAM_WALK_RARE", 2.0) as usize;
        // INFER_KEEP_ONLY=1: inferred events keep the partner word ("lucy jones went to …")
        cfg.infer_drop = std::env::var("INFER_KEEP_ONLY").is_err();
        // ENGRAM_DEDUP=any | move (default) | place
        // INFER_EVERY=k: infer from every statement of a fact about a rare word (held by at
        // most k rows), not only the first statement of a word never seen before
        if let Some(k) = std::env::var("INFER_EVERY").ok().and_then(|v| v.parse().ok()) {
            cfg.infer_rare = k;
        }
        cfg.dedup = match std::env::var("ENGRAM_DEDUP").as_deref() {
            Ok("any") => Dedup::Any,
            Ok("place") => Dedup::Place,
            _ => Dedup::Move,
        };
        index_hc = Some(Box::new(EngramStore::new(cfg)) as Box<dyn EpisodicCircuit>);
    }
    if std::env::var("HIPPO").map_or(false, |v| v == "index") {
        let mut cfg = IndexConfig::default();
        let period = henv("INDEX_PERIOD", 64.0) as u32;
        cfg.period = if period == 0 { u32::MAX } else { period };
        cfg.inverse = std::env::var("INDEX_PLAIN").is_err();
        cfg.cap = henv("INDEX_CAP", 4096.0) as usize;
        cfg.min_overlap = henv("INDEX_MIN", 16.0) as u32;
        // INDEX_PLACE=k: the place code as k input indices per story (place cells as more
        // input), in a range above the binding fields
        cfg.place_bits = henv("INDEX_PLACE", 0.0) as usize;
        cfg.place_base = (2 * SPARSE_FIELDS + 1) * BITS;
        index_hc = Some(Box::new(IndexMemory::new(cfg)) as Box<dyn EpisodicCircuit>);
    }
    let mut bind_hc = (std::env::var("HIPPO").map_or(false, |v| v == "full")).then(|| {
        let mut cfg = HippocampusConfig::new(BITS, seed + 300);
        cfg.novelty_gain = q16x(henv("HIPPO_GAIN", 3.0) as f64);
        cfg.ca2_weight = henv("HIPPO_CA2", 0.0) as u32;
        cfg.decay = henv("HIPPO_DECAY", 0.999);
        cfg.scale_all = std::env::var("HIPPO_SCALE_ALL").is_ok();
        // HIPPO_CENTER=1: homeostatic centering of CA3's cue drive (experiment 54)
        cfg.center = std::env::var("HIPPO_CENTER").is_ok();
        // SPARSE_BIND=1 (with HIPPO_SELF): the input is the sparse binding space (64 content
        // and 64 context fields of BITS bits), the output stays the compact rotated code;
        // DG and EC III → CA1 project by hash (HIPPO_FANOUT, 300, cells per active input bit)
        if std::env::var("SPARSE_BIND").is_ok() {
            cfg.ec_bits = 2 * SPARSE_FIELDS * BITS;
            cfg.out_bits = BITS;
            cfg.hashed_fan_out = Some(henv("HIPPO_FANOUT", 300.0) as usize);
        }
        if let Ok(v) = std::env::var("HIPPO_CELLS").map(|v| v.parse::<usize>().unwrap_or(4096)) {
            (cfg.ca3_cells, cfg.ca1_cells, cfg.dg_cells) = (v, v, 2 * v);
        }
        Box::new(Hippocampus::new(cfg)) as Box<dyn EpisodicCircuit>
    });
    if index_hc.is_some() {
        bind_hc = index_hc.take();
    }
    // HIPPO_SELF=1 (with HIPPO=full): the hippocampus on its own. Its own familiarity counts
    // replace the list store's statistics; it stores events (one per sentence, with the
    // story's earlier bindings as a context code) instead of whole stories; and with
    // SEMANTIC, its own cue-free replay, decoded through the slot cells, teaches the
    // semantic store (no sentence buffer, no word counts). SEMANTIC_HREPLAYS: replays
    // per sleep (default 400).
    let hippo_self = std::env::var("HIPPO_SELF").is_ok() && bind_hc.is_some();
    let bind_all = std::env::var("HIPPO_BIND_ALL").is_ok();
    let sparse_bind = std::env::var("SPARSE_BIND").is_ok() && hippo_self;
    let mut bind_sentence_pairs: Vec<(usize, usize)> = Vec::new(); // (word, slot) of bind_sentence
    let ctx_offset = BITS / 2 + 12_345 % (BITS / 2);
    let mut bind_prev = BitVector::new(BITS, Some(0)); // the story's bindings before this sentence
    let sem_hreplays: usize = std::env::var("SEMANTIC_HREPLAYS").ok().and_then(|v| v.parse().ok()).unwrap_or(400);
    // INFER_REPLAY=reps (generative replay into the higher area), INFER_ROWS (source rows per
    // new fact, 32)
    let infer_reps: Option<usize> = std::env::var("INFER_REPLAY").ok().and_then(|v| v.parse().ok());
    let infer_rows = henv("INFER_ROWS", 32.0) as usize;
    // inferred pairs only grow kernels, masked to the new word, the word before and the
    // shared state; INFER_LEARN=1 uses area.learn instead (it blames the kernels that fire
    // and masks growth by the live window: it lowered accuracy in every test, experiment 61)
    let infer_grow = std::env::var("INFER_LEARN").is_err();
    let mut infer_stats = [0usize; 2]; // inferred events, (prefix → word) pairs taught
    // PROPOSALS=1: the network's own inferences are proposals, not facts. Each inferred event
    // ("lucy went to the hallway", in its source story's season) is a proposal, stored in
    // the hippocampus tagged as one (source 2), with its evidence:
    // - support: the distinct source episodes it was derived from (convergence);
    // - confirmed / contradicted: a sentence later read in a story of the same season that
    //   is the proposal, or differs from it in one word (the same name, another place).
    // A proposal is validated when confirmed, or supported by PROPOSAL_SUPPORT (default 2)
    // independent premises, and never contradicted. Independent premises: the weaker of
    // the fact's testimonies (in how many episodes it was stated) and the distinct source
    // events; derivations that share one fact stated once are not independent. Only
    // validated proposals are replayed to the cortex (once, `reps` times, when validated). The open ones (neither validated nor
    // contradicted) are what to investigate next.
    let proposals_on = std::env::var("PROPOSALS").is_ok();
    let proposal_support: usize = std::env::var("PROPOSAL_SUPPORT").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    // (season, sentence words) → (source rows, confirmed, contradicted, replayed, each source's
    // prefix, fact rows, partner words)
    let mut proposals: HashMap<(usize, Vec<usize>), (Vec<u32>, u32, u32, bool, Vec<Vec<usize>>, Vec<u32>, Vec<usize>)> = HashMap::default();
    // with the relation store, a proposal's credibility comes from the Bayes module: the
    // belief in its fact ("lucy is a jones") times the credibility of the narrator who told
    // its source event; validated at PROPOSAL_MIN (default 0.5)
    let proposal_min: Q16 = q16(std::env::var("PROPOSAL_MIN").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.5));
    // the narrator of each stored hippocampal event (who told it)
    let mut row_narrator: HashMap<u32, u16> = HashMap::default();
    let mut row_state: HashMap<u32, Vec<u32>> = HashMap::default(); // engram row → higher-area state
    // REINSTATE=1 (with HIER, HIPPO_SELF, SPARSE_BIND, engram): the hippocampus as an index
    // to the cortex (Teyler & Rudy). Each event is stored with the cortical state it was read
    // in (the higher area's slow context: the entorhinal summary), and at every step the
    // current sentence's content cues the hippocampus, with a bonus of REINSTATE_BONUS
    // (default 8) for this story's events (context-dependent retrieval, `recall_soft`). The
    // cortical states stored with the best events (up to 4) are reinstated in the higher
    // areas' context, beside their own: recall brings back the state, not words.
    let reinstate = std::env::var("REINSTATE").is_ok();
    let reinstate_here = std::env::var("REINSTATE").map_or(false, |v| v == "here");
    // ACH=1: an acetylcholine-like mode set by the hippocampus's own novelty (Hasselmo). Each
    // step's recall gives the episode's novelty (1 when nothing is recalled or the best event
    // is from another story: seen, but not here; else CA1's mismatch, 1 − its match), and a
    // tonic level follows it (a quarter of the way each step). High (novel input): encoding
    // mode, recall's pull on the cortex is weakened, so only a share 1 − ACh of the
    // reinstated state and of the entorhinal feedback passes. Low (familiar): retrieval mode,
    // both pass. (Storage already follows novelty: a familiar event strengthens its row, a
    // new one is appended.)
    let ach_on = std::env::var("ACH").is_ok();
    // NE=1: a norepinephrine-like gain (locus coeruleus).
    // - Phasic, a salience tag: a sentence that surprised the column at two or more words is
    //   stored twice, so its event is strengthened as a repeat would strengthen it.
    // - Tonic, adaptive exploration (Aston-Jones & Cohen): a running error rate of training
    //   answers (1/64 per answer) raises the basal ganglia's exploration while outcomes are
    //   poor, from 0.1 up to 0.5, and lowers it as they improve.
    let ne_on = std::env::var("NE").is_ok();
    let mut ne_tonic: Q16 = ONE / 2;
    let mut sent_surprises = 0usize; // this sentence's surprising words
    let mut ne_stats = [0u64; 3]; // salient sentences stored twice, answers seen, Σ exploration (Q16)
    let mut ach: Q16 = ONE;
    let mut ach_stats = [0u64; 2]; // test steps, Σ ACh
    // STORE_TEST=1: the hippocampus encodes test stories too (it is never off); without it,
    // only training stories are stored (QUESTION stores test events by itself)
    let store_test = std::env::var("STORE_TEST").is_ok();
    // REINSTATE_TOP=k: the states of the best k events are reinstated (default 4); 1 is the
    // best event's own state only
    let reinstate_top: usize = std::env::var("REINSTATE_TOP").ok().and_then(|v| v.parse().ok()).unwrap_or(4).max(1);
    let reinstate_bonus: usize = std::env::var("REINSTATE_BONUS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let mut row_cortex: HashMap<u32, BitVector> = HashMap::default(); // engram row → the cortical state it was stored in
    let mut reinstate_stats = [0u64; 3]; // test steps: reinstated, from this story's events only, bits reinstated
    // engram row → the words of its story before its sentence (what was read up to it)
    let mut row_prefix: HashMap<u32, Vec<usize>> = HashMap::default();
    // stories queued for replay through the reading steps (INFER_REPLAY, read mode)
    let mut replay_queue: Vec<Story> = Vec::new();
    let mut replay_words = 0usize;
    let mut prev_replaying = false;
    // INFER_PAIRS=1: teach inferred events as (input, target) pairs to the higher area
    // (experiment 61) instead of reading them as stories
    let infer_pairs = std::env::var("INFER_PAIRS").is_ok();
    let mut sem_hstats = [0usize; 4]; // replays, decoded with content, cued by a new name, marked consolidated
    // REPLAY_GEN=1 (with HIPPO_SELF, SPARSE_BIND, a higher area with HIER_SLEEP_GEN): at each
    // sleep the hippocampus replays freely (from random CA3 cells, settling into attractors:
    // prototypes of overlapping events). Each replay is decoded into its content words (by
    // slot) and context words; every content word becomes a target predicted from
    // [the event's other words | its context], and these pairs are what the higher area's
    // sleep generalisation reads. GEN_REPLAYS (512) per sleep. Events then keep their
    // context in the output (EC V) so a replay brings it back.
    let replay_gen = std::env::var("REPLAY_GEN").is_ok();
    let gen_replays: usize = std::env::var("GEN_REPLAYS").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    let mut gen_stats = [0usize; 3]; // replays, decoded, (input, target) pairs given
    let mut bind_ca3 = hippo_ca3.then(|| Ca3Memory::new(BITS, hcells, hk, henv("HIPPO_DECAY", 0.999), henv("HIPPO_SETTLE", 2.0) as usize));
    let mut bind_story = BitVector::new(BITS, Some(0));
    let mut expect_prev = BitVector::new(BITS, Some(0));
    let mut slot_prev: Option<usize> = None;
    let mut bind_answer: Option<usize> = None;
    // the recall's strength in bands of 64 (about two rare-weighted bindings), 0..7, as the
    // memory source's own confidence in the mix
    let mut bind_strength = 0u64;
    // BIND_FAM=1: familiarity-gated arbitration. The current sentence's bindings (word in
    // slot) are looked up in the store's own statistics; the rarest one's episode count, in
    // log2 bands 0..7, is the sentence's familiarity, and every source's reliability in the
    // mix is kept per band (a name seen once is a low band, a trained name a high one)
    let bind_fam = std::env::var("BIND_FAM").is_ok();
    let mut bind_sentence: Vec<BitVector> = Vec::new();
    // CLASS_READ=1: the slot memory's readout is filtered by the column's expectation (its
    // possible continuations): among the words the unbound episode holds, the one the
    // cortex expects as a kind wins. CLASS_VOTE=1: for novel items (familiarity band < 4,
    // i.e. the rarest binding in fewer than 8 episodes; hand-set) the column and the
    // higher area vote their whole class of continuations instead of one guessed word
    let class_read = std::env::var("CLASS_READ").is_ok();
    let class_vote = std::env::var("CLASS_VOTE").is_ok();
    // CONSOLIDATE=r: systems consolidation. Each training story leaves a trace at its answer
    // (the question sentence, the story's slot bindings, the answer, its familiarity band).
    // At every sleep, and once more before the test, the novel traces (band < 4) are
    // replayed r times to the higher area, which learns from them with its normal rule. The
    // replayed slow state is the episode's gist: only its uncommon bindings (in at most 30%
    // of stored episodes) are reinstated, not the filler of that moment.
    // BIND_LESION=1: the slot memory is switched off at test (only the cortex can answer)
    let consolidate: Option<usize> = std::env::var("CONSOLIDATE").ok().and_then(|v| v.parse().ok());
    // CONSOLIDATE_INTERLEAVE=1: interleaved replay (novel and familiar traces mixed) that
    // also feeds sleep generalisation (needs HIER_DREAM / HIER_SLEEP_GEN)
    let consolidate_interleave = std::env::var("CONSOLIDATE_INTERLEAVE").is_ok();
    // SLEEP_P=p (with CONSOLIDATE): sleep-gated plasticity of the cortex. Each trace also keeps
    // the column's own input row at the answer (the cortical state of that moment, as the
    // hippocampus indexes it), and at every sleep the replayed trace teaches the column too,
    // with its growth probability raised to p while it sleeps (low acetylcholine: the slow
    // cortex learns readily from replay, slowly while awake: SLOW_P under LEARNING=three).
    // REPLAY_PREDICT=1: a replayed trace is predicted before it is learned. Learning credits
    // hits and misses to the kernels that matched at the last prediction, and grows from the
    // last winner; without it, replay was credited against the last awake step's matches.
    let replay_predict = std::env::var("REPLAY_PREDICT").is_ok();
    let sleep_p: Option<Q16> = std::env::var("SLEEP_P").ok().and_then(|v| v.parse::<f64>().ok()).map(q16);
    let mut trace_rows: Vec<BitVector> = Vec::new(); // the column's input at each trace's answer
    // DA=1: the dopamine–novelty loop (Lisman & Grace 2005). Each consolidation trace carries
    // a dopamine level: the hippocampus's novelty for the event when it is laid down (CA1's
    // mismatch, through the subiculum–accumbens–pallidum–VTA loop back to the hippocampus),
    // plus 1/4 if the answer was right (reward). Replay samples traces in proportion to it,
    // instead of the familiarity-band rule (band < 4).
    let da_on = std::env::var("DA").is_ok();
    let mut trace_da: Vec<Q16> = Vec::new();
    let mut da_stats = [0u64; 3]; // traces, Σ dopamine (Q16), traces replayed
    let mut sleep_column = 0usize; // replays taught to the column
    // CONSOLIDATE_STEPS=1: besides each training story's answer, every word the network
    // failed to predict in a novel sentence (familiarity band < 4) leaves a trace, so
    // replay covers what surprised it, such as "smith" after "tom is a"
    // CONSOLIDATE_STEPS=assoc: the step trace's input is only the sentence's rare words
    // (under 1% of sentences), an association ("tom" goes with "smith") rather than a
    // sequence ("tom is a" → "smith")
    let consolidate_steps = std::env::var("CONSOLIDATE_STEPS").is_ok();
    let consolidate_assoc = std::env::var("CONSOLIDATE_STEPS").map_or(false, |v| v == "assoc");
    // ROLLOUT_AREA=1: a rollout step's word may also come from the higher area's
    // prediction (slot memory first, then the higher area, then the column), each only if
    // it is of the kind the column expects
    let rollout_area = std::env::var("ROLLOUT_AREA").is_ok();
    // SEMANTIC=r: a cortical semantic store (the cue -> content store of experiment 17),
    // trained only by sleep replay. Each training sentence is kept as an episode until the
    // next sleep; at sleep the novel ones (with a word in under 1% of sentences so far) are
    // replayed, interleaved with as many others, r rounds: the sentence's rarest word cues
    // the codes of its other words ("tom" -> {is, a, smith}). At a rollout step it is
    // queried with the current sentence's rarest word, after the slot memory and before
    // the higher area and the column, and only a word of the expected kind is taken.
    // SEMANTIC_RANDOM=n (a control): replay n random sentences per sleep instead.
    let semantic_reps: Option<usize> = std::env::var("SEMANTIC").ok().and_then(|v| v.parse().ok());
    let mut sem_store: KernelClass<SimpleKernel> = KernelClass::predictive(GrowthConfig {
        max_kernels: 20_000,
        frame_words: BITS / 64,
        max_frames: 1,
        sample_bits: 16,
        match_fraction: 0.8,
        surprise_fraction: 0.5,
        generalize: None,
        generalize_after: 1,
    });
    let mut sem_buf: Vec<Vec<usize>> = Vec::new(); // training sentences since the last sleep
    // SEM_TYPED=roles|dir (with HIPPO_SELF): typed relations in the semantic store. A fact's
    // content is each word bound to its learned role (its code rotated by the role's offset,
    // as in the hippocampus's slot bindings): "lucy" → {is@1, a@2, jones@3}, not a bag.
    // dir: the cue is bound to its role too ("lucy" as the subject ≠ "lucy" as the
    // object), so a relation has a direction. Readers that want words unbind every role
    // and keep the words found; readers that take context (the higher area) get the
    // role-bound content as is.
    let sem_typed: u8 = match std::env::var("SEM_TYPED").as_deref() {
        Ok("dir") => 2,
        Ok(_) => 1,
        Err(_) => 0,
    };
    // SEM_FRAME=1 (with SEM_TYPED): the role binding is keyed on the fact's frame too, so
    // two relations of one word that fill the same role ("lucy is a jones", "lucy likes
    // tea") do not collide. The frame is learned from familiarity: a fact's bindings at
    // least half as familiar as its most familiar one ("is@1", "a@2": what every fact of
    // this kind shares), the rest are its fillers ("jones@3"). A frame's id is a hash of
    // its bindings; each binding is rotated by its role's offset plus its frame's. The
    // store remembers which frames each cue was stored with, so a word reader unbinds
    // those (frame, role) pairs only.
    // REL=reps: the relation store (src/program/relations.rs) stands in for the semantic
    // store's answers. Training sentences are read into it (word counts, buffered facts);
    // at each sleep the facts are parsed (frame = the positions where most facts of the
    // same shape agree, fillers the rest) and replayed `reps` times. Asked about a word, it
    // answers the relation named by the current sentence's words if the word has one,
    // else everything it knows about the word (`about`): the fillers' codes, plain.
    let rel_reps: Option<usize> = std::env::var("REL").ok().and_then(|v| v.parse().ok());
    let narrators: Option<usize> = std::env::var("NARRATORS").ok().and_then(|v| v.parse().ok()).filter(|&k: &usize| k > 0);
    let liar: Q16 = q16(std::env::var("LIAR").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0));
    let mut lies_told = 0usize;
    let mut rel = RelationStore::new(BITS);
    // REL_LIFT=1: facts are read twice, the second time with weak frame positions and
    // frame words that are entities elsewhere lifted into the fillers ("X _ went to the
    // Y": john, jones, hallway). REL_HOPS=2: the answer adds a second hop, forward steps
    // only (from an earlier filler to a later one) from each first answer: lucy → jones →
    // the places joneses went.
    rel.lift = std::env::var("REL_LIFT").is_ok();
    // REL_COMPLETE=1: the relation store answers by fact completion (a kernel class that
    // completes "tom is a _" from the rest of the fact, one frame per word) instead of a
    // store keyed by the entity's code rotated by a hashed relation offset
    rel.completion = std::env::var("REL_COMPLETE").is_ok();
    // trust per source per relation (default; a narrator honest about places can still lie
    // about families). TRUST_TOPIC=0: one trust per source
    rel.set_trust_by_relation(std::env::var("TRUST_TOPIC").map_or(true, |v| v != "0"));
    // BELIEF=full|vote|graded|posterior: the Bayes module's belief rule (default graded);
    // TRUST=vote is the same as BELIEF=vote
    rel.bayes.rule = BeliefRule::named(&std::env::var("BELIEF").unwrap_or_else(|_| if std::env::var("TRUST").map_or(false, |v| v == "vote") { "vote".into() } else { "graded".into() }));
    let rel_hops: usize = std::env::var("REL_HOPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut rel_stats = [0usize; 3]; // facts parsed at sleep, answers given at test, of those for held-out stories
    let sem_frame = sem_typed > 0 && std::env::var("SEM_FRAME").is_ok();
    let mut sem_frames: HashMap<usize, Vec<u64>> = HashMap::default(); // cue word → its frames
    let mut frame_names: HashMap<u64, Vec<(usize, usize)>> = HashMap::default(); // frame → its bindings (report)
    let mut sem_replays = 0usize;
    let mut sem_used = [0usize; 2]; // rollout steps whose word came from the store: (all, held-out stories)
    let bind_lesion = std::env::var("BIND_LESION").is_ok();
    // COMPLETE=1: pattern completion of a skipped slot. When the next word is not of the kind
    // the column expects here, the sentence is novel (familiarity band < 4) and the slot
    // memory has a filler for the expected slot, that filler is inserted as an internal step
    // (heard from memory, not read from the page), at most once per sentence. With
    // COMPLETE=test only at test.
    // COMPLETE=rollout (rollout-test: only at test): the cortex follows its own expectation
    // instead. When the next word contradicts a definite expectation (the column expects one
    // word only), and then while it is not one the column expects, the expected word is
    // inserted as an internal step, up to 4 per sentence, until the page matches the
    // expectation again: "tom [is a smith] went". The step's word is the slot memory's if it
    // is of the kind the column expects (the schema gives the kind, memory which one), else
    // the column's own.
    let complete = std::env::var("COMPLETE").ok();
    let rollout = complete.as_deref().map_or(false, |m| m.starts_with("rollout"));
    // COMPLETE=speech (speech-test: only at test): inner speech in place of the rollout.
    // The network says its integrated prediction (the mix's word, else the column's
    // output: the vector it would speak) into a phonological loop, and hears it as its next
    // input, marked as its own (the efference copy; nothing learns from it as the world's).
    // When: where the page contradicts what it predicted (a surprise), and then while it
    // still does, up to 4 times a sentence, never at the answer. INNER_GATE=learned: the
    // basal ganglia decide speak or read on at each such surprise, per (confidence band,
    // already speaking, novel sentence), rewarded at the answer less STEP_COST a step.
    // What is said is not chosen among sources: memory, the semantic store and the higher
    // area act only through the prediction they shaped.
    let inner_speech = complete.as_deref().map_or(false, |m| m.starts_with("speech"));
    let inner_learned = std::env::var("INNER_GATE").map_or(false, |v| v == "learned");
    let inner_when_definite = std::env::var("INNER_WHEN").map_or(false, |v| v == "definite");
    let inner_say_recall = std::env::var("INNER_SAY").map_or(false, |v| v == "recall");
    let mut phono = PhonologicalLoop::new();
    // QHOLD=learned (with HIPPO_SELF, SPARSE_BIND, engram): holding an item in working memory,
    // learned. No sentence is rewritten, nothing names the item from outside, no hand rule
    // decides what is held or where it goes (the oracles and operators of 89–90 are removed).
    // - Holding: as each word is bound, the basal ganglia choose to take it into working
    //   memory or not, seeing the hippocampus's novelty for its binding (1 − its recall match)
    //   and that of what is held now (bands), rewarded at the story's answer less STEP_COST.
    // - Binding: the held item is active, so it is part of every recall cue, and at a
    //   sentence's end it may be stored with that sentence's event in a field of its own
    //   (QATTACH=learned): the basal ganglia choose, seeing the sentence itself (its words'
    //   codes as the choice's code, so what is learned for one sentence carries over to
    //   sentences that share words). Credit is tagged by recall: an attached event recalled
    //   for the answer gets the outcome (+1 right, −1 wrong); otherwise only STEP_COST.
    // - Use: recall reaches the cortex as usual; QQUERY (below) adds the held item's own
    //   query.
    let qhold_learned = std::env::var("QHOLD").map_or(false, |v| v == "learned");
    let qhold = qhold_learned.then_some(());
    let qattach_learned = std::env::var("QATTACH").map_or(false, |v| v == "learned");
    let mut attach_pending: Vec<(BitVector, bool, Option<u32>, usize)> = Vec::new(); // (choice, attached, its event's row, the held word)
    // Recall-tagged credit for the hold (default; QHOLD_CREDIT=story: the story's reward for
    // every hold choice, as in 91): taking a word into working memory gets the answer's outcome
    // (+1 right, −1 wrong) only if an event it was stored with was recalled for that answer;
    // otherwise only STEP_COST, and leaving a word 0.
    let hold_credit_story = std::env::var("QHOLD_CREDIT").map_or(false, |v| v == "story");
    let mut hold_pending: Vec<(BitVector, Option<usize>)> = Vec::new(); // (choice, the word taken)
    let mut answer_rows: Vec<u32> = Vec::new(); // the events recalled for the answer
    let mut held_item: Option<(usize, Q16)> = None; // the held word and its novelty when taken
    let mut q_err = [0usize; 5]; // QUESTION, held-out answers: right, right family wrong season, other family right season, other, no place
    let mut held_words: HashMap<&str, usize> = HashMap::default(); // test, held out: what was held at the answer
    // QQUERY=1 (with QHOLD): the held item as a recall query of its own. When the held item is
    // read again, it cues the hippocampus alone, among the current story's events only
    // (context-dependent recall, `recall_here`): pattern completion from the item to the
    // event it was stored with. The event's words join the entorhinal feedback (HC_EC) for the
    // rest of the sentence, and the event counts as used for the answer's credit.
    // QQUERY=all: the blend of every event bound with the item here. QQUERY=soft: the item and
    // the context as a weighted match (overlap plus a bonus for this story's events), not a
    // gate. QQUERY_FULL: the answer passes whole, not scaled by the cortex's uncertainty.
    // With ROUTE: the answer is a routed channel of its own (Q_CHANNEL). QAREA=1: it goes to
    // the higher areas' sentence context only.
    let qquery = std::env::var("QQUERY").is_ok();
    let qquery_all = std::env::var("QQUERY").map_or(false, |v| v == "all");
    let qquery_full = std::env::var("QQUERY_FULL").is_ok();
    let qquery_soft = std::env::var("QQUERY").map_or(false, |v| v == "soft");
    let qarea = std::env::var("QAREA").is_ok();
    let mut query_rows: Vec<u32> = Vec::new(); // every event the query blended
    let mut query_ec: Option<(u32, BitVector)> = None; // the query's event and its words' codes
    let mut query_stats = [0usize; 3]; // test, held-out: queries, an event found, the event holds a surname
    let mut hold_stats = [0usize; 4]; // test, held-out: answers with an item held, a new name held, events attached, attached events recalled for the answer
    let mut inner_diag = 0usize;
    let mut inner_stats = [0usize; 3]; // test: surprises where it could speak, spoken, spoken in held-out stories
    let mut completed_sentence = false;
    let mut rolled = 0usize; // internal steps inserted in this sentence
    // held-out answers by the surname the rollout supplied: (right, total) for none, the
    // right family, the wrong family
    let mut rolled_surname: Option<bool> = None;
    let mut by_surname = [(0usize, 0usize); 3];
    let mut complete_stats = [0usize; 3]; // completions: training, test (trained names), test (held out)
    let mut complete_words: HashMap<String, usize> = HashMap::default(); // test, held out: "name -> word"
    // (question sentence bag, the story's (word, slot) bindings, answer word, familiarity band)
    let mut traces: Vec<(BitVector, Vec<(usize, usize)>, usize, u64)> = Vec::new();
    let mut replayed = 0usize;
    let mut bind_list: Vec<(usize, usize)> = Vec::new(); // this story's (word, slot) bindings
    let mut bind_diag = 0usize;
    let slot_offset = |c: usize| ((c + 1) * 2_654_435_761usize) % BITS;
    let slot_input = |expect: &BitVector, prev: Option<usize>, roles: &RoleArea| {
        let mut x = expect.clone();
        if let Some(p) = prev {
            let mut c = roles.code(p).clone();
            c.rotl_mut(BITS / 2);
            x.or_mut(&c);
        }
        x
    };
    // test answers with a binding answer: (answers, binding answer right)
    let mut bind_stats = (0usize, 0usize);
    let mut bind_new = (0usize, 0usize); // the same, for held-out (new-name) answers
    // this step's prediction of each upper area (for the mix)
    let mut upper_pred: Vec<Option<BitVector>> = vec![None; upper.len()];
    let mut upper_in: Vec<Option<BitVector>> = vec![None; upper.len()];
    // test answers where each upper area's prediction held the answer
    let mut upper_has_answer = vec![0usize; upper.len()];
    // HIER_RESET=1: a story boundary is a context boundary: every area forgets its window
    let hier_reset = std::env::var("HIER_RESET").is_ok();
    // BOUNDARY=1: detect context boundaries instead of being told. A boundary is a conflict
    // of facts: a rare surprising word arrives while an area's window holds a different rare
    // word of the same kind. Kind is learned from the input: two words are of a kind when
    // the words seen before and after them mostly coincide (Jaccard >= BOUNDARY_KIND, default
    // 0.5). The boundary lies just after the older fact: each area forgets it and everything
    // older (`HigherArea::forget_through`)
    let bound_detect = std::env::var("BOUNDARY").is_ok();
    // EVENT_BOUNDARY=learned: no "." trigger. An event-boundary cell (BoundaryCell) learns from
    // the column's surprise which inputs precede unpredictable stretches, and its firing ends
    // the event: storage, resets and everything else the "." used to trigger. BOUND_MIN is
    // its refractory span, BOUND_MAX the longest event it holds. The page's end still closes
    // the last event.
    let learned_bound = std::env::var("EVENT_BOUNDARY").map_or(false, |v| v == "learned");
    let bound_min: usize = std::env::var("BOUND_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let bound_max: usize = std::env::var("BOUND_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let mut bcell = BoundaryCell::new(BITS, bound_min, bound_max);
    // GATE=burst: the thalamic mix weighs each source by its burst rate (how often its recent
    // predictions were confirmed where the sources disagreed), not by word-keyed tables. Only
    // sources at or above a threshold pass; the threshold falls while none passes and rises
    // slowly while some do. When none passes, the cerebellum's prediction goes through.
    // L5=two: layer 5 two-compartment cells (basal: the input and previous input; apical:
    // the context frames) whose bursts override L2/3 (Layer5); L5_MAX cells at most
    let mut layer5: Option<Layer5> = std::env::var("L5").map_or(false, |v| v == "two").then(|| Layer5::new(BITS / 64, 16, 0.8, std::env::var("L5_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(50_000)));
    // L5=primed: two popcounts, the tuft primes and the input triggers (PrimedLayer5); a pool
    // of L5_CELLS cells wired to random contexts
    // Default (experiment 103): primed layer 5 with SST/VIP interneurons, vectorized; L5=off
    // (or another L5 value) turns it off
    let mut primed5: Option<PrimedLayer5> = std::env::var("L5").map_or(true, |v| v == "primed").then(|| {
        let n = std::env::var("L5_CELLS").ok().and_then(|v| v.parse().ok()).unwrap_or(8192);
        // L5_GROW=1: cells grown as needed (up to L5_CELLS) instead of a random-context pool
        if std::env::var("L5_GROW").is_ok() { PrimedLayer5::grown(BITS / 64, 16, n) } else { PrimedLayer5::new(BITS / 64, 16, n) }
    });
    // L23=primed: the column's L2/3 prediction from two-compartment primed cells (input on the
    // basal dendrites, context on the tuft; spikes and bursts both predict); L23_GROW=1: grown
    // as needed; L23_CELLS cells (default 16384)
    let mut primed23: Option<PrimedLayer5> = std::env::var("L23").map_or(false, |v| v == "primed").then(|| {
        let n = std::env::var("L23_CELLS").ok().and_then(|v| v.parse().ok()).unwrap_or(16384);
        if std::env::var("L23_GROW").is_ok() { PrimedLayer5::grown(BITS / 64, 16, n) } else { PrimedLayer5::new(BITS / 64, 16, n) }
    });
    // INTERNEURONS=1: SST and VIP cells set the primed layers' context threshold (instead of
    // the gain rule)
    // L5_VEC=1: the primed layers count with bitsets and bit-sliced counters (same counts)
    // L5_SYN=bits: the primed layers' synapses as bits (absent, silent, active, sticky), no
    // strengths (PrimedLayer5::set_bit_synapses); set before the backend is built
    if std::env::var("L5_SYN").map_or(false, |v| v == "bits") {
        for l in primed5.iter_mut().chain(primed23.iter_mut()) {
            l.set_bit_synapses(true);
        }
    }
    if std::env::var("L5_VEC").map_or(true, |v| v != "0") {
        for l in primed5.iter_mut().chain(primed23.iter_mut()) {
            l.set_vectorized(true);
        }
    }
    if std::env::var("INTERNEURONS").map_or(true, |v| v != "0") {
        for l in primed5.iter_mut().chain(primed23.iter_mut()) {
            l.set_interneurons(true);
        }
    }
    let mut l23_stats = [0usize; 2]; // predictions by primed cells, by the old kernels
    // L5_SYN=bitwise: the primed layers as BitCells (cells are masks; learning is mask
    // arithmetic), in place of PrimedLayer5
    // Default (experiment 105): the primed layers are BitCells (fully bitwise, self-calibrating);
    // L5_SYN=strength (or bits) keeps PrimedLayer5
    let bitwise = std::env::var("L5_SYN").map_or(true, |v| v == "bitwise");
    let l5_out = bitwise && std::env::var("L5_OUT").is_ok();
    // L5_OUT=relay: as L5_OUT, but a silent layer 5 relays L2/3's prediction (the thick-tufted
    // cells' regular firing on L2/3 drive) instead of sending nothing
    let l5_relay = l5_out && std::env::var("L5_OUT").map_or(false, |v| v == "relay");
    let mut bit23: Option<BitCells> = (bitwise && primed23.is_some()).then(|| BitCells::new(BITS / 64, 16, std::env::var("L23_CELLS").ok().and_then(|v| v.parse().ok()).unwrap_or(16384)));
    let mut bit5: Option<BitCells> = (bitwise && primed5.is_some()).then(|| BitCells::new(BITS / 64, 16, std::env::var("L5_CELLS").ok().and_then(|v| v.parse().ok()).unwrap_or(8192)));
    if bitwise {
        primed23 = None;
        primed5 = None;
        if std::env::var("INTERNEURONS").map_or(true, |v| v != "0") {
            for l in bit23.iter_mut().chain(bit5.iter_mut()) {
                l.set_interneurons(true);
            }
        }
        // L5_META=1: each cell's learning rate and threshold calibrate from its own outcomes
        if std::env::var("L5_META").map_or(true, |v| v != "0") {
            for l in bit23.iter_mut().chain(bit5.iter_mut()) {
                l.set_meta(true);
            }
        }
        // L5_STICK=tag: stickiness by synaptic tagging and capture
        if std::env::var("L5_STICK").map_or(false, |v| v == "tag") {
            for l in bit23.iter_mut().chain(bit5.iter_mut()) {
                l.set_tag_capture(true);
            }
        }
        // L5_STICKY=0: no consolidation into sticky synapses
        if std::env::var("L5_STICKY").map_or(false, |v| v == "0") {
            for l in bit23.iter_mut().chain(bit5.iter_mut()) {
                l.set_sticky(false);
            }
        }
        // L5_OUT=1: L2/3's prediction is part of layer 5's input side, and only layer 5 reaches
        // the thalamus
        if l5_out {
            for l in bit5.iter_mut() {
                l.set_basal_tail(2);
            }
        }
        // L5_IDX=id: plasticity indexed by the cell's id and the step, no random draw
        if std::env::var("L5_IDX").map_or(false, |v| v == "id") {
            for l in bit23.iter_mut().chain(bit5.iter_mut()) {
                l.set_id_index(true);
            }
        }
    }
    let burst_vote = std::env::var("BURST_VOTE").is_ok();
    let burst_key = std::env::var("BURST_KEY").is_ok();
    let burst_gate = std::env::var("GATE").map_or(false, |v| v == "burst");
    let mut src_burst: HashMap<u8, Q16> = HashMap::default();
    let mut gate_theta: Q16 = ONE / 2;
    let gate_on_surprise = std::env::var("GATE_ON").map_or(false, |v| v == "surprise");
    let mut gate_stats = [0usize; 4]; // at test answers: gated, passed, cerebellum fallback, ungated
    let mut bound_stats = [0usize; 3]; // at test: boundaries, of them at ".", periods
    let bound_kind: Q16 = q16(std::env::var("BOUNDARY_KIND").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.5));
    // context signature per word: the words seen just before (< V) and just after (V + w)
    let mut word_ctx: Vec<neurocomp::det::HashSet<usize>> = vec![neurocomp::det::HashSet::default(); vocab.len()];
    // test sentences with a detected boundary: (story-opening, other); story-opening sentences
    let mut bound_hits = (0usize, 0usize);
    let mut story_openings = 0usize;
    let mut bound_fired = false; // a boundary was detected in the current sentence
    // READBACK=top|all: self-supervised read-back. At each sentence end, an area says back
    // the most recent rare word its window holds (rare: seen in fewer than READBACK_RARE of
    // the sentences so far, default 0.02). Its target is that word, from its own input; it
    // learns from the mismatch (training only), and what it says is heard: it joins the
    // next sentence's surprising words, so it re-enters every area's window (rehearsal)
    let readback = std::env::var("READBACK").ok();
    let readback_rare: Q16 = q16(std::env::var("READBACK_RARE").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.02));
    let mut word_count = vec![0u32; vocab.len()];
    // BOUNDDIAG: the column's surprise per sentence (sum, words, first word's), and per
    // test sentence (is it a story's first sentence?, mean surprise, first-word surprise)
    let (mut sent_surprise, mut sent_words, mut first_surprise) = (0f32, 0usize, 0f32);
    let mut bound_log: Vec<(bool, f32, f32)> = Vec::new();
    let mut sentence_count = 0u32;
    // read-back at test: (attempts, spoke the target)
    let mut readback_stats = (0usize, 0usize);
    let mut hier_in: Option<BitVector> = None;
    let hier_residual = std::env::var("HIER_RESIDUAL").map_or(true, |v| v != "0");
    // HIER_GATE=1: a corticothalamic gate on the top-down channel, learned from use (L5
    // attribution); HIER_GATE_WARMUP stories all open, weakening × HIER_GATE_WEAKEN
    let hier_gate_on = std::env::var("HIER_GATE").is_ok();
    let hier_gate_area = std::env::var("HIER_GATE_SIGNAL").map_or(false, |v| v == "area");
    let hier_gate_conf: Option<Q16> = std::env::var("HIER_GATE_CONF").ok().and_then(|v| v.parse::<f64>().ok()).map(q16);
    let hier_gate_warmup: usize = std::env::var("HIER_GATE_WARMUP").ok().and_then(|v| v.parse().ok()).unwrap_or(1000);
    let mut hier_gate = CorticothalamicGate::new(BITS, 1, seed + 31);
    hier_gate.weaken = q16x(std::env::var("HIER_GATE_WEAKEN").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.25));
    let mut hier_gate_step: Option<BitVector> = None;
    let mut topdown_passed_at_answer = 0usize;
    // HIER_EARLY=1: the top-down frame right after the current word
    let hier_early = std::env::var("HIER_EARLY").is_ok();
    let mut topdown_has_answer = 0usize;
    let mut prev: Option<usize> = None;
    let (mut seen, mut held) = ((0usize, 0usize), (0usize, 0usize));
    // BELIEF_Q: held-out answers (right, asked) for place questions and family questions
    let mut belief_tally = [(0usize, 0usize); 2];
    let (mut mix_dbg, mut bq_diag) = (String::new(), 0usize);
    let mut belief_names = [(0usize, 0usize); 3]; // place questions per new name: (right, asked)
    // calibration of the column's L5 confidence at test answers: (answers, right) per
    // confidence bucket [0, .5), [.5, .7), [.7, .8), [.8, .9), [.9, 1], plus the summed
    // confidence per bucket (for the expected calibration error)
    let mut calib = [(0usize, 0usize, 0f64); 5];
    let mut recall_has_answer = 0usize;
    let mut recall_places = 0usize;
    // DIAG: what the predictor got at the answer, split by right / wrong
    // [answer bits recalled, recall size in bits, whole words recalled, count]
    let mut diag = [[0usize; 4]; 2];
    // DIAG: the winning kernel at held-out answers, split by right / wrong:
    // [no winner, reads the memory frame, mask bits in current/memory/previous frames
    //  summed (3 entries), reliability ×1000 summed, count]
    let mut kdiag = [[0usize; 7]; 2];
    // DIAG coverage at wrong held-out answers: [cases, cases with any memory-reading
    // kernel for the answer, best match ratio ×100 summed, missing bits per frame
    // (current, memory, previous) of that best kernel summed]
    let mut cover = [0usize; 8]; // + [6] best kernel's reliability ×100 summed, [7] its hits+misses summed
    let mut pending_diag = [0usize; 3];
    let full_stop = index["."];
    // DUMP_STORIES=dir: write this task's stories (train, then test) for outside baselines
    // and skip the network entirely
    let mut dump = std::env::var("DUMP_STORIES").ok().map(|dir| {
        let path = format!("{dir}/{task:?}_{max_facts}_{seed}.txt").to_lowercase();
        std::io::BufWriter::new(std::fs::File::create(path).expect("dump file"))
    });
    // Season task: filler stories between the announcement and the question, and test
    // accuracy by that distance in bins [0, 1, 2-3, 4-7, 8-15, 16+]: (answers, right)
    let season_len: usize = std::env::var("SEASON_LEN").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
    let mut season_distance = 0usize;
    // SCHEMA_K=k: each (new name, season) pair appears in exactly k training stories, at
    // random positions among the last SCHEMA_PHASE (default 600) training stories
    let schema_k: Option<usize> = std::env::var("SCHEMA_K").ok().and_then(|v| v.parse().ok());
    let schema_at: HashMap<usize, (usize, usize)> = {
        let mut m = HashMap::default();
        if let Some(k) = schema_k {
            let phase: usize = std::env::var("SCHEMA_PHASE").ok().and_then(|v| v.parse().ok()).unwrap_or(600);
            let mut srng = StdRng::seed_from_u64(seed.wrapping_add(991));
            let mut slots: Vec<usize> = (TRAIN - phase..TRAIN).collect();
            slots.shuffle(&mut srng);
            let mut it = slots.into_iter();
            for i in 0..NEW_NAMES.len() {
                // FAMILY_STATED: k statements per new name (the season is unused)
                let seasons = if family_stated() { 1 } else { SEASONS.len() };
                for sn in 0..seasons {
                    for _ in 0..k {
                        if let Some(p) = it.next() {
                            m.insert(p, (i, sn));
                        }
                    }
                }
            }
        }
        m
    };
    // each statement about a new name: (name, its place among that name's statements in time)
    let schema_ord: HashMap<usize, (usize, usize)> = {
        let mut slots: Vec<(usize, usize)> = schema_at.iter().map(|(&p, &(i, _))| (p, i)).collect();
        slots.sort_unstable();
        let mut count = [0usize; 8];
        slots.into_iter().map(|(p, i)| { let o = count[i % 8]; count[i % 8] += 1; (p, (i, o)) }).collect()
    };
    // NARRATOR_SPLIT=1 (with NARRATORS, LIAR): the statements about each new name alternate
    // between the liar, who always lies in them, and the first (honest) narrator, so each new
    // name's family is a 1:1 conflict a vote cannot settle; which comes first alternates by
    // name. Trust must come from the other facts (the trained names'), where the liar lies
    // with probability LIAR among honest narrators.
    let narrator_split = std::env::var("NARRATOR_SPLIT").is_ok();
    // UNDECIDED=1 (with NARRATOR_SPLIT): the last new name's statements come from two honest
    // narrators who disagree (the second states the wrong family, once): a conflict that
    // trust cannot settle, where "unknown" is the right answer
    let undecided = std::env::var("UNDECIDED").is_ok();
    // BELIEF_UNKNOWN=1: at a question that names a relation the relation store holds for
    // the word ("sam is a"), the answer is "unknown" unless the believed value is believed
    // more than half (more likely than every alternative together)
    let belief_unknown = std::env::var("BELIEF_UNKNOWN").is_ok();
    // BELIEF_UNKNOWN=learned: answer or "unknown" is a basal-ganglia go/no-go per band of
    // the rule's own belief in its best value (eighths), learned from practice quizzes:
    // answering right is worth 1, answering wrong 0, "unknown" one half, so answering wins
    // a band once its answers there are right more often than not
    let unknown_learned = std::env::var("BELIEF_UNKNOWN").map_or(false, |v| v == "learned");
    let mut unknown_bg = BasalGanglia::new(256);
    // one block per (state, choice) (default). GONOGO_CODES=shared: belief band's bits joined
    // with the lead band's, so unpractised states borrow from their neighbours; tried and
    // backed out as default: the belief band shared with a settleable state leaked "answer"
    // into the undecidable one (sam answered wrong on two seeds of three)
    let gonogo_separate = std::env::var("GONOGO_CODES").map_or(true, |v| v != "shared");
    let unknown_code = |ctx: usize, answer: bool| {
        if gonogo_separate {
            let at = (ctx * 2 + answer as usize) * 2;
            return BitVector::from_bits(&(at..at + 2).collect::<Vec<_>>(), 256);
        }
        let base = answer as usize * 128;
        let (b, l) = (ctx / 8, ctx % 8);
        let mut bits: Vec<usize> = (base + b * 8..base + b * 8 + 8).collect();
        bits.extend(base + 64 + l * 8..base + 64 + l * 8 + 8);
        BitVector::from_bits(&bits, 256)
    };
    let mut quiz_rng = StdRng::seed_from_u64(seed ^ 0x5157_4954);
    let mut quiz_stats = [[0usize; 3]; 64]; // per context: (quizzes, answered, answered right)
    let quiz_reps: usize = std::env::var("QUIZ_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
    // ASK=n (with NARRATORS): at each sleep the network may ask a teacher n questions ("who
    // is sam?"); the teacher answers with the true family, as a source of its own whose
    // trust is earned like any narrator's. ASK_POLICY=curious (default): the curiosity
    // module picks the questions by their learned value of information; =random: any n
    // people the module has claims about.
    let ask: usize = std::env::var("ASK").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    // =learned: the network's own policy, a basal-ganglia selector over the questions'
    // states, rewarded by the information each answer brought
    let ask_random = std::env::var("ASK_POLICY").map_or(false, |v| v == "random");
    let ask_learned = std::env::var("ASK_POLICY").map_or(false, |v| v == "learned");
    // HC_SURPRISE=1: within a sentence the hippocampus's recall (CA3's settled state) is
    // reused while the column needs no help: a surprising word (or a sentence's first), or a
    // column unsure of the next word (its own prediction, without memory or top-down, under
    // half reliable), recalls again. HIER_SURPRISE=1: the higher areas predict and learn only
    // then too; otherwise they send an empty top-down frame (a stale one, computed for another
    // next word, misleads: gating on the last word's surprise alone lost the answers, which
    // follow predictable words: "X went to the _").
    let hc_surprise = std::env::var("HC_SURPRISE").is_ok();
    // HIER_SURPRISE=learned: whether to consult the higher areas is the network's own choice: a basal-ganglia go/no-go per context (the column's own confidence
    // band, whether the last word surprised it, the current word; the code shares bits across
    // words and across bands). Consulting is rewarded by what the frame changed against the column's own prediction without it (one
    // look-up): right where it would be wrong 1, wrong where it would be right 0, no
    // difference one half less HIER_COST. Skipping is worth one half. Skipping sends an empty
    // frame. The gate also decides while learning (with exploration); HIER_LEARN_ALWAYS=1
    // consults on every learning step instead, which leaves the column untrained on empty
    // frames (story boundary 0% at test). At HIER_COST 0.05 the gate rarely consults; 0 is
    // the setting that keeps the tasks.
    let hier_learned = std::env::var("HIER_SURPRISE").map_or(false, |v| v == "learned");
    let hier_surprise = std::env::var("HIER_SURPRISE").is_ok() && !hier_learned;
    // HIER_TRUST_GATE=1: the top-down frame reaches the column's L4 only as a trusted
    // witness: where the thalamus's record of the higher area (how often its proposed word
    // was right, kept per previous word, current word and the area's confidence band, from
    // every training word) is at least one half. An area never heard in a context passes, so
    // the record can form; the gate opens and closes as the area grows more or less
    // reliable. The area itself still predicts, learns and votes in the mix every step.
    // HIER_TRUST_GATE=column: pass where the area's record is at least the column's own.
    // HIER_TRUST_GATE=cf: the frame's counterfactual record instead. On every training step
    // the column's prediction with the full frame and with none are compared (one extra
    // look-up); where they differ, a fix (right only with it) or a break (right only
    // without), kept per the same context. The frame enters L4 scaled: a fixed subset of its
    // bits, the share 2 × (fixes + 1) / (fixes + breaks + 2) up to all of it, so a frame with no
    // record passes whole and weakens as its breaks outnumber its fixes. The record is always
    // taken on the full frame, so a weakened frame can earn its way back.
    let hier_trust_gate = std::env::var("HIER_TRUST_GATE").ok();
    let hier_up_surprise = std::env::var("HIER_UP").map_or(false, |v| v == "surprise");
    // HIER_UP=both: predicted and unpredicted words both go up, told apart. The sentence
    // frame holds the predicted words as they are and the surprising ones as a burst code
    // (the same word, rotated), as a burst on a pyramidal cell carries more than a single
    // spike does
    let hier_up_both = std::env::var("HIER_UP").map_or(false, |v| v == "both");
    // ROUTE=1: learned input routing. The column's L4 row is no longer laid out by hand.
    // The current word drives slot 0; every other frame of the row (memory, relays, the
    // top-down frame, the previous input) and each promoted higher area's prediction is a
    // channel. Each channel keeps its own code (topographic: the column can copy a word
    // from it; ROUTE_BIND=1 rotates it by a per-source offset instead, which stops copying),
    // passes a share of its bits set by its counterfactual record (as HIER_TRUST_GATE=cf),
    // and is placed in a slot by the thalamus: every ROUTE_EVERY (250) training stories the
    // channels are ranked by fixes − breaks, the most useful first (kernels read slots in
    // order). The row keeps its width: channels beyond its slots are ORed into the last one.
    // The record: each training step one channel in turn is removed (one extra look-up).
    let route_on = std::env::var("ROUTE").is_ok();
    // HC_ROUTE=1 (with ROUTE): the hippocampus reaches the column as entorhinal feedback, not
    // as a vote. Within a word's step, in theta order: the column's feedforward sweep gives
    // its expectation, the hippocampus recalls (cued as before), CA1 decodes the recall into
    // word codes (the words the unbound episode holds, best overlap first, up to three), and
    // they arrive as one more channel before the column predicts: context, never the driver,
    // its gain learned like any channel's. The slot memory's word no longer votes in the mix.
    let hc_route = route_on && std::env::var("HC_ROUTE").is_ok();
    // HC_EC=1 (without ROUTE): the plausible hippocampal path. Its only output is entorhinal
    // feedback (CA1 → subiculum → deep EC → association cortex): within a word's step, the
    // column's feedforward sweep gives its expectation and its confidence; the hippocampus
    // recalls (cued as before); CA1 decodes the recall into word codes (up to three, best
    // overlap first); they reach the column as context in their own slot, never as the
    // driver. Their gain is the cortex's uncertainty: the share of the feedback's bits that
    // pass is 1 − the confidence of the feedforward sweep's prediction (memory counts where
    // the cortex is unsure, as prefrontal control retrieves when monitoring finds
    // uncertainty). Memory's word no longer votes in the thalamic mix, and the rollout no
    // longer inserts it as the next input or feeds its readout back.
    let hc_ec = !route_on && std::env::var("HC_EC").is_ok();
    // INNER_SLOT=1 (with COMPLETE=speech*, without ROUTE): inner speech is heard through its
    // own input, a slot of the column's row (as auditory cortex beside the visual word), not
    // in the page word's slot: at an inner step the word slot is empty and the said vector
    // is in the heard slot, so the column learns separately how far to go by its own speech.
    let inner_slot = !route_on && std::env::var("INNER_SLOT").is_ok();
    let inner_echo = std::env::var("INNER_SLOT").map_or(false, |v| v == "echo");
    let mut hc_ec_stats = [0u64; 3]; // steps with feedback, Σ share passed (Q16), test steps with feedback
    let mut route_cur: Option<(BitVector, Vec<(usize, BitVector)>, usize)> = None; // this step's row parts
    // The record (default): each training step, for every slot, the column's prediction
    // against its prediction from the kernels that do not read that slot or any later one
    // (`peek_shallow`, the same match: no second run); a slot's fix or break is credited to
    // the channels in it. ROUTE_RECORD=rerun: one channel in turn removed and the column
    // asked again (one extra look-up; not a mechanism a brain has).
    // The slot order changes only in a critical period, the first ROUTE_CRITICAL (1000)
    // training stories; after it the layout is fixed and only the gains (shares) adapt, as
    // laminar targets are fixed after development while thalamic gain stays plastic.
    // ROUTE_NOSCALE=1: every channel passes whole (binding and slot order only; a control)
    let route_noscale = std::env::var("ROUTE_NOSCALE").is_ok();
    let route_rerun = std::env::var("ROUTE_RECORD").map_or(false, |v| v == "rerun");
    let route_critical: usize = std::env::var("ROUTE_CRITICAL").ok().and_then(|v| v.parse().ok()).unwrap_or(1000);
    let route_every: usize = std::env::var("ROUTE_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(250);
    let mut route_order: Vec<usize> = Vec::new(); // channel ids, best slot first
    let mut route_score: HashMap<usize, (u64, u64)> = HashMap::default(); // (fixes, breaks) per channel
    let mut route_step: Option<(BitVector, Vec<(usize, BitVector)>, usize)> = None; // (word, channels, slots)
    let mut route_turn = 0usize;
    let mut hc_routed = 0usize; // steps the hippocampus reached the column (HC_ROUTE)
    let mut route_log: Vec<String> = Vec::new();
    let mut cf_scaled = [0usize; 2]; // steps whose frame was scaled down
    let mut cf_share_sum = [0u64; 2]; // Σ share passed (Q16), training and test
    let mut td_full: Option<BitVector> = None; // this step's unscaled top-down frame (cf)
    let mut trust_passed = [[0usize; 2]; 2]; // (training, test) × (withheld, passed)
    let hier_learn_always = std::env::var("HIER_LEARN_ALWAYS").map(|v| v == "1").unwrap_or(false);
    let hier_cost: Q16 = q16(std::env::var("HIER_COST").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.05));
    let mut hier_bg = BasalGanglia::new(512);
    let hier_code = |word: usize, band: usize, surprised: bool, consult: bool| {
        let base = consult as usize * 256;
        let ctx = (band.min(4) * 2 + surprised as usize) * 8;
        let mut h = (word as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 31;
        let wb = 96 + (h % 20) as usize * 8;
        let mut bits: Vec<usize> = (base + ctx..base + ctx + 8).collect();
        bits.extend(base + wb..base + wb + 8);
        BitVector::from_bits(&bits, 512)
    };
    let mut hier_pending: Option<BitVector> = None; // the chosen action's code, until its reward
    // HIER_TRACE=n: credit over time. Each choice stays eligible for n page steps and is
    // rewarded by the column's accuracy over them (its own step's prediction and the next
    // n − 1), each step weighted HIER_DECAY (default one half) per step of distance, instead
    // of the one-step counterfactual. Consulting and skipping are valued alike, as the
    // accuracy that followed them in that context; nothing is charged unless HIER_COST is set
    // (taken off a consultation's credit).
    let hier_trace_len: usize = std::env::var("HIER_TRACE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    // HIER_ANSWER_WEIGHT=w: a story's answer counts w times an ordinary word (the world's
    // reward is the answer; the other words are the column's own check on itself)
    let hier_answer_weight: u64 = std::env::var("HIER_ANSWER_WEIGHT").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    // HIER_MARGIN=m: skip only where skipping's value exceeds consulting's by more than m
    let hier_margin: Q16 = q16(std::env::var("HIER_MARGIN").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0));
    let hier_decay: Q16 = q16(std::env::var("HIER_DECAY").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.5));
    // (code, consulted, Σ weight·outcome, Σ weight, current weight, steps seen)
    let mut hier_trace: std::collections::VecDeque<(BitVector, bool, u64, u64, Q16, usize)> = std::collections::VecDeque::new();
    let mut hier_choices = [[0usize; 2]; 2]; // (training, test) × (skipped, consulted)
    let mut hier_consult = true;
    let mut td_span: Option<(usize, usize)> = None; // where this step's top-down frame sits in the input
    let mut step_surprised = true;
    let mut hc_cached: Option<neurocomp::program::Recall> = None;
    let mut step_needs_help = true;
    let mut gated_steps = [0usize; 4]; // (recalls reused, recalls made, area steps reused, area steps run)
    let ask_rounds: usize = std::env::var("ASK_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
    // =cost: the network's own policy with a cost. Each question uses energy, the compute its
    // answer took to weigh (relation-store replays, per COST_UNIT replays a full reserve),
    // from a reserve refilled by POWER (a share of a full reserve) at each sleep; at each
    // step the policy asks or stops, knowing the reserve and how far into the sleep it is
    // (see `Curiosity::decide`). ASK is then only the most it may ask in a round.
    let ask_cost = std::env::var("ASK_POLICY").map_or(false, |v| v == "cost");
    let power: u32 = q16(std::env::var("POWER").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.5));
    let cost_unit: u64 = std::env::var("COST_UNIT").ok().and_then(|v| v.parse().ok()).unwrap_or(400);
    let mut energy: u32 = ONE;
    let mut work_total = 0u64;
    let mut curiosity: Curiosity<usize> = Curiosity::new(8, 8);
    let mut ask_rng = StdRng::seed_from_u64(seed ^ 0xA5C0_0001);
    let mut asked_log: Vec<(usize, usize)> = Vec::new(); // (person asked about, at which sleep)
    let mut sleeps_seen = 0usize;
    // PRACTICE: each practice name has a true family and one of three kinds of statement:
    // 0, the first honest narrator (truth) against the liar (a lie); 1, the first honest
    // narrator (truth) against the second (wrong, once: undecidable); 2, one honest
    // statement. Statements fall in stories told by the right narrator, before the new
    // names' phase. (slot -> (practice name, family stated))
    let (practice_at, practice_truth): (HashMap<usize, (usize, usize)>, Vec<usize>) = {
        let mut m = HashMap::default();
        let mut truth = Vec::new();
        if let (true, Some(k)) = (practice(), narrators) {
            let mut prng = StdRng::seed_from_u64(seed.wrapping_add(4242));
            let phase: usize = std::env::var("SCHEMA_PHASE").ok().and_then(|v| v.parse().ok()).unwrap_or(600);
            for j in 0..PRACTICE_NAMES.len() {
                // families balanced within each kind: a lopsided share would make the
                // majority family part of the learned frame ("X is a smith"), and those
                // facts would no longer parse
                let f = (j / 3) % SURNAMES.len();
                truth.push(f);
                let tellers: Vec<(usize, usize)> = match j % 3 {
                    0 => vec![(0, f), (k - 1, 1 - f)],
                    // which honest narrator errs alternates, so neither earns less trust
                    1 => if (j / 3) % 2 == 0 { vec![(0, f), (1, 1 - f)] } else { vec![(1, f), (0, 1 - f)] },
                    _ => vec![(prng.gen_range(0..k - 1), f)],
                };
                let mut at = prng.gen_range(100..TRAIN - phase - 200);
                for (n, fam) in tellers {
                    while at % k != n || m.contains_key(&at) {
                        at += 1;
                    }
                    m.insert(at, (j, fam));
                    at += 1 + prng.gen_range(0..60);
                }
            }
        }
        (m, truth)
    };
    let mut unknown_tally = [[0usize; 4]; 3]; // per new name: (asked, right, wrong, unknown)
    // Books task: each book's season, sessions left, last session read; this session's
    // book and its bin [first session, back to back, after 1-2 other sessions, after 3+]
    let (mut book_season, mut book_left, mut book_last) = ([0usize; 3], [0usize; 3], [0usize; 3]);
    let (mut book_bin, mut open_book_next) = (0usize, 0usize);
    let (mut book_id, mut next_book_id) = ([0usize, 1, 2], 3usize);
    let mut book_bins = [(0usize, 0usize); 4];
    // BOOK_CTX: what an "@open" action does to the areas' context: none (default), reset,
    // reinstate (restore the context saved at that book's last "@close", else reset), or
    // learned (a basal-ganglia choice of keep / reset / reinstate per book action,
    // rewarded by whether the session's answer comes out right)
    let book_ctx = std::env::var("BOOK_CTX").unwrap_or_else(|_| "none".into());
    let mut saved_ctx: HashMap<usize, Vec<AreaContext>> = HashMap::default();
    let mut ctx_bg = BasalGanglia::new(BITS);
    let ctx_code = |a: usize| -> BitVector {
        let mut crng = StdRng::seed_from_u64(seed.wrapping_mul(3_000_017) ^ (a as u64 + 2000));
        let all: Vec<usize> = (0..BITS).collect();
        BitVector::from_bits(&all.choose_multiple(&mut crng, 32).copied().collect::<Vec<_>>(), BITS)
    };
    let mut ctx_pending = false; // a learned choice awaits this session's answer
    let mut ctx_chosen = [0usize; 3]; // test: keep, reset, reinstate
    let mut open_book: Option<usize> = None;
    // SACCADE=learned|oracle: active reading. The page stays available; at each word the
    // basal ganglia choose a saccade: read on (0), look back to the previous sentence (1),
    // or to the top of the page (2), per context (previous word, current word). A
    // regression re-reads that sentence (perception only: surprise as usual, no learning),
    // its surprising words entering the areas' windows, then the eye returns to the current
    // word. Reward: the next prediction right, minus SACCADE_COST (default 0.1) for a
    // regression. oracle: look back to the page top exactly at the answer, never elsewhere
    let saccade = std::env::var("SACCADE").ok();
    let saccade_cost: i32 = q16(std::env::var("SACCADE_COST").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.1)) as i32;
    let mut sacc_bg = BasalGanglia::new(BITS);
    // STEP=learned (with COMPLETE=rollout*): the basal ganglia decide each internal step.
    // Wherever the page contradicts the column's expectation and some source offers a word
    // of the expected kind, they choose whether to start a rollout ("look again") or read
    // on, per context: (definite expectation, column confidence in 3 bands, novel sentence,
    // the offering source: slot memory / semantic store / higher area / column). A started
    // rollout runs on until the page fits again, as with the hand-set trigger.
    // Every choice in a story is rewarded at its answer: right (0 or 1), minus STEP_COST
    // (default 0.05) for a step. They learn in training, and at test with STEP_TEST_LEARN.
    let step_learned = std::env::var("STEP").map_or(false, |v| v == "learned");
    let step_test_learn = std::env::var("STEP_TEST_LEARN").is_ok();
    let step_cost: i32 = q16(std::env::var("STEP_COST").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.05)) as i32;
    let mut step_bg = BasalGanglia::new(BITS);
    let step_code = |ctx: usize, act: usize| {
        let mut crng = StdRng::seed_from_u64(seed.wrapping_mul(7_000_003) ^ ((ctx * 2 + act) as u64 + 5000));
        let all: Vec<usize> = (0..BITS).collect();
        BitVector::from_bits(&all.choose_multiple(&mut crng, 32).copied().collect::<Vec<_>>(), BITS)
    };
    let mut step_pending: Vec<(BitVector, bool)> = Vec::new(); // this story's choices
    // CUE_CTL=1 (with HIPPO_SELF + SPARSE_BIND): a cue controller. The hippocampus is still
    // cued at every word (automatic recall), but the basal ganglia choose how the cue is
    // edited first, per context (the column's confidence band × the sentence's familiarity
    // band): 0 the cue as is, 1 the cue as is with the walk's bridge allowed, 2 the
    // sentence's content only (the story's context dropped), 3 the sentence's least
    // familiar binding + the story's context (focus). Reward: the recalled word is the
    // next word read (1, else 0), minus CUE_COST (default 0.02) for the walk's second
    // recall. Learns in training (not in replay); CUE_TEST_LEARN=1 at test too.
    // Sparse gating of memory (after 63, 64, 67).
    // SPARSE_HC=1: the hippocampus's answers (its memory frames and its slot readout) enter
    // the source mix only where the column is unsure (its own prediction under half
    // reliable), as the semantic/relation store's does under COOPERATE: sparse in time for
    // both stores.
    // GATE=learned (with COOPERATE): the basal ganglia decide whether the cortex asks the
    // store at this step, instead of the fixed "under half reliable" threshold, per
    // (the column's own confidence band × the cue word's familiarity band). Reward: how
    // well the higher area then predicts the next word (its L5 outcome), minus GATE_COST
    // (default 0.05) for asking. Learns in training (not in replay).
    // SPEAK=1: answering by speaking. At a test question the network writes its answer to
    // an output buffer: the source mix's word, with the mix's confidence (else the
    // column's), or "unknown" under SPEAK_MIN (default 0: always speak). The page's answer
    // is not read: the spoken word comes back as the next input in its place (an internal
    // step: nothing learns from it as the world's word; "unknown" is heard as "."). The
    // report scores the buffer against the page: accuracy, coverage, and the curve at
    // other thresholds.
    let speak = std::env::var("SPEAK").is_ok();
    // SPEECH=motor (with SPEAK and/or RECITE): speech routed as in the brain.
    // - The vocal tract (the world) says a word from its own motor code, unrelated to how
    //   the word sounds. A motor area learned by babbling (before reading; SPEECH_BABBLE
    //   rounds, default 10) holds an inverse model (sound → command) and a forward model
    //   (command → expected sound).
    // - Plan: the cortex's evidence for the word (the source mix's choice); the motor area
    //   turns it into a command, the tract says it (or nothing, for a garbled command).
    // - Select: at a question, the basal ganglia decide speak or stay silent, per the mix's
    //   confidence band. They learn in training, where the page's answer follows: speaking
    //   right +1, speaking wrong −SPEAK_PENALTY (default 1), silence 0.
    // - The efference copy is the forward model's prediction, compared with what is heard
    //   (not a flag): a word heard as predicted is not surprising.
    let motor_speech = std::env::var("SPEECH").map_or(false, |v| v == "motor");
    // SPEAK_CTX: what the go/no-go sees besides the mix's confidence band: "novelty" (the
    // sentence's least familiar binding, in four bands: a new name is new), "agree" (every
    // source in the mix proposed the same word), or both ("novelty,agree"). Default: the
    // confidence band only.
    let speak_ctx_cfg = std::env::var("SPEAK_CTX").unwrap_or_default();
    // FAMILIARITY=cortex: novelty from a cortical familiarity signal (as perirhinal
    // cortex's, which survives a hippocampal lesion), not the hippocampus's counts: the
    // word's exposure, how many sentences holding it were read (replays not counted). Bands
    // of the sentence's least exposed word: under 16, under 64, under 256, more.
    // FAMILIARITY=kernels: the number of the column's kernels keyed on the word (they sample
    // its current-word bits), recomputed every 100 stories (a measure that failed: growth
    // is driven by surprise and replay, so new names have more kernels than trained ones).
    let cortex_fam_on = std::env::var("FAMILIARITY").map_or(false, |v| v == "cortex");
    let kernel_fam_on = std::env::var("FAMILIARITY").map_or(false, |v| v == "kernels");
    let mut cortex_fam: Vec<u32> = Vec::new();
    let (ctx_novelty, ctx_agree) = (speak_ctx_cfg.contains("novelty"), speak_ctx_cfg.contains("agree"));
    let speak_ctx = |cb: usize, fam_band: u64, agreed: bool| -> usize {
        if !ctx_novelty && !ctx_agree {
            return cb;
        }
        cb * 8 + if ctx_novelty { (fam_band.min(7) / 2) as usize * 2 } else { 0 } + if ctx_agree { agreed as usize } else { 0 }
    };
    // at test: (questions, spoken, right) when the sources agreed, and when they did not
    let mut sp_agree = [[0usize; 3]; 2];
    // at test: (questions, spoken, right) per novelty band
    let mut sp_novel = [[0usize; 3]; 4];
    let speak_penalty: i32 = q16(std::env::var("SPEAK_PENALTY").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(1.0)) as i32;
    let tract = VocalTract::new(vocab.len(), BITS, 32, &mut StdRng::seed_from_u64(seed ^ 0x5eec_0001));
    let mut motor = MotorArea::new(BITS);
    if motor_speech {
        let rounds = std::env::var("SPEECH_BABBLE").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
        motor.babble(&tract, &enc.codes, rounds, &mut StdRng::seed_from_u64(seed ^ 0xbab1_e000));
    }
    let mut speak_bg = BasalGanglia::new(BITS);
    // its own random draws, so that what the go/no-go sees does not change the rest of the
    // run's training (a shared generator made every variant a different training run)
    let mut speak_rng = StdRng::seed_from_u64(seed ^ 0x5bea_7000);
    let speak_code = |ctx: usize, act: usize| {
        let mut crng = StdRng::seed_from_u64(seed.wrapping_mul(13_000_027) ^ ((ctx * 2 + act) as u64 + 11_000));
        let all: Vec<usize> = (0..BITS).collect();
        BitVector::from_bits(&all.choose_multiple(&mut crng, 32).copied().collect::<Vec<_>>(), BITS)
    };
    // at test, per mix-confidence band: (questions, spoken, spoken right); in training: (questions, spoken)
    let mut sp_stats = [[0usize; 3]; 5];
    let mut sp_train = [0usize; 2];
    // the forward model's prediction for the word heard at each position (motor speech)
    let mut eff_pred: Vec<Option<BitVector>> = Vec::new();
    // story boundaries after an altered ending (an answer spoken or withheld): the step
    // carries of the last story that ended as read (previous word, its slot, the column's
    // expectation), restored at the next story's start, as a boundary resets them. The
    // role cells chain each word's slot on the previous one, so an unusual ending ("went to
    // the .") would otherwise shift every slot of the next story.
    let mut boundary_carry: Option<(Option<usize>, Option<usize>, Option<usize>, BitVector)> = None;
    let mut story_altered = false;
    let mut prev_altered = false;
    let mut speech = OutputBuffer::new(q16(std::env::var("SPEAK_MIN").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0)));
    let mut transcripts: Vec<String> = Vec::new();
    // RECITE=k: recitation. After each test story, the network retells it: a copy is read
    // with the first k words given (the cue), and from there each word the network speaks
    // (the mix's choice, else the column's) is its next input. Nothing learns or is stored
    // (as a test), nothing is counted in the answer statistics. Scored against the story:
    // words right in their position, and the story's content words (in under 5% of
    // sentences: names, places, seasons) said anywhere.
    // EFFERENCE=1: the efference copy. A word the network spoke is marked as its own when
    // it comes back: predicted by the copy (no surprise; it does not enter the areas'
    // windows of surprising words). A mismatch between the copy and what is heard is a full
    // surprise. SELF_NOISE=p: what is heard is a random other word with probability p
    // (altered feedback), to test that the mismatch is caught.
    let recite: Option<usize> = std::env::var("RECITE").ok().and_then(|v| v.parse().ok());
    // RECITE_PLAN=1: the hippocampus plans the retelling, the cortex speaks it. At the
    // first spoken step the hippocampus recalls the most recent episode (the story just
    // read, by recency: `recent_episode`) and plays it forward from the event after the
    // one the cue matches (else the episode the cue's bindings recall: `sequence_from`):
    // the events' words in reading order, each ended by ".". With RECITE_PLAN the test
    // stories are stored in the hippocampus too (experience to retell), as training ones. At each step the planned word is spoken
    // if the column's expectation admits it (its kind fits here), else the cortex's own.
    let recite_plan = std::env::var("RECITE_PLAN").is_ok();
    // Source memory. SELF_STORE=1: the network's retellings are stored in the hippocampus
    // too, as events of its own (tagged "self"; what is read is tagged "world").
    // SOURCE_TAG=1: reality monitoring, recall for reading and answering returns world
    // events only. Without it, recall may return what the network itself said.
    let self_store = std::env::var("SELF_STORE").is_ok();
    let source_tag = std::env::var("SOURCE_TAG").is_ok();
    if source_tag {
        if let Some(hc) = &bind_hc {
            hc.set_recall_sources(1);
        }
    }
    // at test questions: recalls that returned a world event, a self event
    let mut source_stats = [0usize; 2];
    let mut plan: Vec<usize> = Vec::new();
    let mut plan_stats = [0usize; 2]; // steps with a planned word, planned word spoken
    let efference_on = std::env::var("EFFERENCE").is_ok();
    let self_noise: Q16 = q16(std::env::var("SELF_NOISE").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0));
    // recitation: positions scored, right; content words, said
    let mut recite_stats = [0usize; 4];
    let mut recite_sample: Vec<String> = Vec::new();
    // efference: own words heard, of those altered, altered caught; own words that entered a surprise window
    let mut eff_stats = [0usize; 4];
    let mut efference: Vec<Option<usize>> = Vec::new();
    let sparse_hc = std::env::var("SPARSE_HC").is_ok();
    let gate_learned = std::env::var("GATE").map_or(false, |v| v == "learned");
    let gate_cost: i32 = q16(std::env::var("GATE_COST").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.05)) as i32;
    let mut mg_bg = BasalGanglia::new(BITS);
    let mg_code = |ctx: usize, act: usize| {
        let mut crng = StdRng::seed_from_u64(seed.wrapping_mul(11_000_027) ^ ((ctx * 2 + act) as u64 + 9000));
        let all: Vec<usize> = (0..BITS).collect();
        BitVector::from_bits(&all.choose_multiple(&mut crng, 32).copied().collect::<Vec<_>>(), BITS)
    };
    let mut mg_pending: Option<(BitVector, bool)> = None;
    // at test, per own-confidence band: (steps, asked); hippocampal answers withheld
    let mut mg_stats = [[0usize; 2]; 5];
    let mut hc_withheld = 0usize;
    let cue_ctl = std::env::var("CUE_CTL").is_ok();
    let cue_test_learn = std::env::var("CUE_TEST_LEARN").is_ok();
    let cue_cost: i32 = q16(std::env::var("CUE_COST").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.02)) as i32;
    let mut cue_bg = BasalGanglia::new(BITS);
    let cue_code = |ctx: usize, act: usize| {
        let mut crng = StdRng::seed_from_u64(seed.wrapping_mul(9_000_011) ^ ((ctx * 4 + act) as u64 + 7000));
        let all: Vec<usize> = (0..BITS).collect();
        BitVector::from_bits(&all.choose_multiple(&mut crng, 32).copied().collect::<Vec<_>>(), BITS)
    };
    let mut cue_pending: Option<(BitVector, usize)> = None;
    // at test: [action] → (chosen, recalled the next word); in training: chosen
    let mut cue_stats = [[0usize; 2]; 4];
    let mut cue_train = [0usize; 4];
    let mut cue_last_test: Option<usize> = None;
    let mut step_stats = [[0usize; 2]; 4]; // at test, per offering source: (offered, stepped)
    // SEMANTIC_MIX=1: the semantic store votes in the mix as source 8 (its word of the kind
    // the column expects, for the sentence's rarest word), with its own learned reliability.
    // ROLLOUT_MIX=1: a rollout step's word comes from the mix of the offering sources (each
    // with its learned weight here) instead of a fixed order.
    let semantic_mix = std::env::var("SEMANTIC_MIX").is_ok();
    let rollout_mix = std::env::var("ROLLOUT_MIX").is_ok();
    let mut inner: Vec<bool> = Vec::new(); // per position of the current story: an internal step
    // ROLLOUT_LOOP=1: the closed loop. A rollout step feeds the network's own output back as
    // its next input, with no word decoded and re-encoded: the offering source's output
    // vector gated by the column's expectation (bitwise AND, a thalamic gate), taken if it
    // keeps at least one word's worth of bits (24). "Definite" is likewise a bit count: the
    // expectation holds at most 1.5 words' worth of bits. A word is still decoded from the
    // fed-back vector, but only for the harness's bookkeeping (reports, context keys).
    let rollout_loop = std::env::var("ROLLOUT_LOOP").is_ok();
    // ROLLOUT_SUPER=1 (with ROLLOUT_LOOP): all sources at once. Every source's output,
    // gated by the expectation, is OR-ed into one fed-back vector, and the column resolves
    // the superposition; no order and no per-source reliabilities. (The basal ganglia's
    // context still names the first source present.)
    // ROLLOUT_SUPER=evidence: only the evidence sources (slot memory, semantic store, higher
    // area) are superposed; the column's own prediction, which is the expectation (the
    // prior) the gate already applies, is fed back only when none of them offers anything.
    let rollout_super = std::env::var("ROLLOUT_SUPER").is_ok();
    let super_evidence = std::env::var("ROLLOUT_SUPER").map_or(false, |v| v == "evidence");
    let mut inner_code: Vec<Option<BitVector>> = Vec::new(); // the fed-back vector of an internal step
    let mut bind_raw: Option<BitVector> = None; // the slot memory's unbound readout (before decoding)
    let mut loop_stats = [0usize; 3]; // test: fed-back vectors, exactly one word's code, a blend (2+ words' worth)
    let sacc_code = |a: usize| -> BitVector {
        let mut crng = StdRng::seed_from_u64(seed.wrapping_mul(4_000_037) ^ (a as u64 + 3000));
        let all: Vec<usize> = (0..BITS).collect();
        BitVector::from_bits(&all.choose_multiple(&mut crng, 32).copied().collect::<Vec<_>>(), BITS)
    };
    let mut sacc_pending: Option<usize> = None;
    // test: regressions to the previous sentence / page top, words re-read, words read,
    // answers preceded by a regression at the step before
    let mut sacc_stats = [0usize; 5];
    let mut l4_mid = 0usize; // frames between current and previous in the column's L4
    // SACCADE_CONF=1: the selector's context also holds the column's own confidence about
    // the next word (a peek without top-down, no state change), in 4 buckets: no prediction,
    // < 0.5, < 0.8, >= 0.8
    let sacc_conf = std::env::var("SACCADE_CONF").is_ok();
    // SACCADE_CTX=cortex: the selector's context is the column's own state, not the word
    // identities: the union of what its matching kernels predict next (one word when
    // confident, a superposed class when not) plus a code for its confidence bucket. Each
    // candidate is bound to that state by its own rotation, so similar states share values
    let sacc_cortex = std::env::var("SACCADE_CTX").map_or(false, |v| v == "cortex");
    // test answers (trained wording, new wording): (answers, re-read the first sentence
    // just before, right)
    let mut sacc_wording = [(0usize, 0usize, 0usize); 2];
    // SACCADE=index: no fixed targets. The page keeps an index of where each surprising
    // word was read (landmarks); a regression's candidates are "read on" plus each landmark
    // word (its latest position before the current sentence), valued per context
    let mut page_marks: Vec<bool> = Vec::new(); // per position of the current story
    // test: answers whose regression re-read the story's first sentence (the announcement)
    let mut sacc_hit_first = 0usize;
    let mut season_bins = [(0usize, 0usize); 6];
    // COST: wall time and words for training and test
    let (mut train_secs, mut test_secs, mut train_words, mut test_words) = (0f64, 0f64, 0usize, 0usize);
    let mut train_text_bytes = 0usize;
    let mut phase_start = std::time::Instant::now();
    let mut prof = [0f64; 6];
    let sleep_every: Option<usize> = std::env::var("SLEEP_EVERY").ok().and_then(|v| v.parse().ok());
    let (mut sleeps, mut slept_pruned, mut slept_merged) = (0usize, 0usize, 0usize);
    let mut prof_t = std::time::Instant::now();
    for s_i in 0..TRAIN + TEST {
        let testing = s_i >= TRAIN;
        if kernel_fam_on && (s_i % 100 == 0 || s_i == TRAIN) {
            cortex_fam = cortical_familiarity(column.l23.kernels(), &enc.codes, BITS);
        }
        // SLEEP_EVERY=n: an offline sleep pass for the column every n training stories
        // consolidation replay: at every sleep, and the night before the test
        if let (Some(reps), true) = (consolidate, hier && bind && s_i > 0 && (s_i == TRAIN || (!testing && sleep_every.map_or(false, |n| s_i % n == 0)))) {
            // the replayed input of a trace: [question sentence | gist], the gist being the
            // words of the story's uncommon bindings
            let replay_input = |sent: &BitVector, bl: &[(usize, usize)]| -> BitVector {
                let mut state = BitVector::new(BITS, Some(0));
                for &(w, sl) in bl {
                    let mut b = enc.codes[w].clone();
                    b.rotl_mut(slot_offset(sl));
                    if (fam_binding(&bind_hc, &bind_mem, hippo_self, sparse_bind, &b, &enc.codes[w], w, sl) << 16) <= Q_03 as u64 * mem_size(&bind_hc, &bind_mem, hippo_self) as u64 {
                        state.or_mut(&enc.codes[w]);
                    }
                }
                let mut words = sent.as_words().to_vec();
                words.extend_from_slice(state.as_words());
                BitVector::from_words(words)
            };
            // which traces replay: the novel ones (band < 4), or with DA those the dopamine
            // tag selects (each with probability its level)
            let novel: Vec<usize> = if da_on {
                let v: Vec<usize> = (0..traces.len()).filter(|&i| chance(&mut sleep_rng, trace_da.get(i).copied().unwrap_or(0))).collect();
                da_stats[2] += v.len() as u64;
                v
            } else {
                (0..traces.len()).filter(|&i| traces[i].3 < 4).collect()
            };
            if consolidate_interleave {
                // interleaved: the novel traces mixed with as many familiar ones, shuffled each
                // round; every replay also feeds sleep generalisation, which then runs
                let familiar: Vec<usize> = (0..traces.len()).filter(|&i| !novel.contains(&i)).collect();
                let mut order: Vec<usize> = novel.clone();
                order.extend(familiar.choose_multiple(&mut sleep_rng, novel.len()).copied());
                for _ in 0..reps {
                    order.shuffle(&mut sleep_rng);
                    for &i in &order {
                        let (sent, bl, ans, _) = &traces[i];
                        let x = replay_input(sent, bl);
                        if let Some(a) = assoc.as_mut() {
                            // the association area learns from replay, its plasticity open
                            let gate = a.column.l23.growth_gate();
                            a.column.l23.set_growth_gate(None);
                            a.column.l23.set_growth_probability(Some(ONE));
                            if replay_predict {
                                a.predict(&x);
                            }
                            a.learn(&x, &enc.codes[*ans], &mut sleep_rng);
                            a.column.l23.add_replay(&x, &enc.codes[*ans]);
                            a.column.l23.set_growth_probability(Some(assoc_p));
                            a.column.l23.set_growth_gate(gate);
                            assoc_stats[2] += 1;
                        } else {
                            if replay_predict {
                                area.predict(&x);
                            }
                            area.learn(&x, &enc.codes[*ans], &mut sleep_rng);
                            area.column.l23.add_replay(&x, &enc.codes[*ans]);
                        }
                        replayed += 1;
                        if let (Some(p), Some(row)) = (sleep_p, trace_rows.get(i).filter(|r| r.count_ones() > 0)) {
                            // asleep, the column's plasticity is open: growth probability p,
                            // no uncertainty gate
                            let gate = column.l23.growth_gate();
                            column.l23.set_growth_gate(None);
                            column.l23.set_growth_probability(Some(p));
                            if replay_predict {
                                column.predict(row);
                            }
                            column.learn(row, &enc.codes[*ans], &mut sleep_rng);
                            if let Some(l) = layer5.as_mut() {
                                l.learn(row, &enc.codes[*ans], &mut sleep_rng);
                            }
                            if let Some(l) = primed5.as_mut() {
                                l.learn(row, &enc.codes[*ans], &mut sleep_rng);
                            }
                            if let Some(l) = primed23.as_mut() {
                                l.learn(row, &enc.codes[*ans], &mut sleep_rng);
                            }
                            for l in bit23.iter_mut() {
                                l.learn(row, &enc.codes[*ans], &mut sleep_rng);
                            }
                            if let Some(l) = bit5.as_mut() {
                                // replay: L2/3's prediction for the replayed row, on layer 5's input side
                                let r5 = l5_out.then(|| with_frame(row, &column.l23.peek(row).unwrap_or_else(|| BitVector::new(BITS, Some(0)))));
                                l.learn(r5.as_ref().unwrap_or(row), &enc.codes[*ans], &mut sleep_rng);
                            }
                            column.l23.set_growth_probability(three.then_some(slow_p));
                            column.l23.set_growth_gate(gate);
                            sleep_column += 1;
                        }
                    }
                }
                match assoc.as_mut() {
                    Some(a) => a.column.l23.generalize_from_replay(),
                    None => area.column.l23.generalize_from_replay(),
                }
            } else {
                for &i in &novel {
                    let (sent, bl, ans, _) = &traces[i];
                    let x = replay_input(sent, bl);
                    for _ in 0..reps {
                        if let Some(a) = assoc.as_mut() {
                            let gate = a.column.l23.growth_gate();
                            a.column.l23.set_growth_gate(None);
                            a.column.l23.set_growth_probability(Some(ONE));
                            if replay_predict {
                                a.predict(&x);
                            }
                            a.learn(&x, &enc.codes[*ans], &mut sleep_rng);
                            a.column.l23.set_growth_probability(Some(assoc_p));
                            a.column.l23.set_growth_gate(gate);
                            assoc_stats[2] += 1;
                        } else {
                            if replay_predict {
                                area.predict(&x);
                            }
                            area.learn(&x, &enc.codes[*ans], &mut sleep_rng);
                        }
                        replayed += 1;
                        if let (Some(p), Some(row)) = (sleep_p, trace_rows.get(i).filter(|r| r.count_ones() > 0)) {
                            // asleep, the column's plasticity is open: growth probability p,
                            // no uncertainty gate
                            let gate = column.l23.growth_gate();
                            column.l23.set_growth_gate(None);
                            column.l23.set_growth_probability(Some(p));
                            if replay_predict {
                                column.predict(row);
                            }
                            column.learn(row, &enc.codes[*ans], &mut sleep_rng);
                            if let Some(l) = layer5.as_mut() {
                                l.learn(row, &enc.codes[*ans], &mut sleep_rng);
                            }
                            if let Some(l) = primed5.as_mut() {
                                l.learn(row, &enc.codes[*ans], &mut sleep_rng);
                            }
                            if let Some(l) = primed23.as_mut() {
                                l.learn(row, &enc.codes[*ans], &mut sleep_rng);
                            }
                            for l in bit23.iter_mut() {
                                l.learn(row, &enc.codes[*ans], &mut sleep_rng);
                            }
                            if let Some(l) = bit5.as_mut() {
                                // replay: L2/3's prediction for the replayed row, on layer 5's input side
                                let r5 = l5_out.then(|| with_frame(row, &column.l23.peek(row).unwrap_or_else(|| BitVector::new(BITS, Some(0)))));
                                l.learn(r5.as_ref().unwrap_or(row), &enc.codes[*ans], &mut sleep_rng);
                            }
                            column.l23.set_growth_probability(three.then_some(slow_p));
                            column.l23.set_growth_gate(gate);
                            sleep_column += 1;
                        }
                    }
                }
            }
        }
        // INFER_REPLAY=reps: generative replay. At each sleep the hippocampus composes events
        // from the facts stored since the last one (`infer`: "lucy is a jones" + a jones event
        // of another story → "lucy went to the hallway", in that story's context), and the
        // higher area learns each word of them from the words before it and the story's
        // unfamiliar context words, `reps` times: the walk's results taught to the cortex.
        if let (Some(reps), true, Some(hc)) = (infer_reps, hier && s_i > 0 && (s_i == TRAIN || (!testing && sleep_every.map_or(false, |n| s_i % n == 0))), bind_hc.as_mut()) {
            let events = hc.infer(infer_rows);
            if !infer_pairs {
                // read mode: each inferred event becomes a story, its source story's opening
                // (what was read before the source sentence) then the inferred sentence, and
                // is read like any story before the next one (`reps` times, in a shuffled order)
                let mut stories: Vec<Story> = Vec::new();
                if proposals_on {
                    // the inferences become proposals (and hippocampal events tagged as such)
                    for (seq, ctx, src, fact, partner) in &events {
                        let Some(prefix) = row_prefix.get(src) else { continue };
                        let Some(season) = prefix.iter().find_map(|&w| SEASONS.iter().position(|x| *x == vocab[w])) else { continue };
                        let words: Vec<usize> = seq.iter().map(|&i| i % 4096).filter(|&w| w < vocab.len() && vocab[w] != ".").collect();
                        // the event's pattern, as a read sentence's: each word's code bound to its slot
                        let mut ev = BitVector::new(BITS, Some(0));
                        for &i in seq.iter() {
                            if i % 4096 < vocab.len() {
                                let mut b = enc.codes[i % 4096].clone();
                                b.rotl_mut(slot_offset((i / 4096) % SPARSE_FIELDS));
                                ev.or_mut(&b);
                            }
                        }
                        let e = proposals.entry((season, words)).or_insert_with(|| {
                            hc.set_source(2);
                            hc.store_split(seq, ctx, &set_bits(&ev));
                            hc.set_source(0);
                            (Vec::new(), 0, 0, false, Vec::new(), Vec::new(), Vec::new())
                        });
                        if !e.6.contains(partner) {
                            e.6.push(*partner);
                        }
                        if !e.0.contains(src) {
                            e.0.push(*src);
                            e.4.push(prefix.clone());
                        }
                        if !e.5.contains(fact) {
                            e.5.push(*fact);
                        }
                    }
                    // the validated ones, not yet replayed, are replayed now
                    let isa = rel_reps.and(rel.relation_for(&[index["is"], index["a"]]));
                    let st = validate_proposals(&mut proposals, isa.map(|i| (&rel, i)), &**hc, &row_narrator, &vocab, proposal_min, proposal_support);
                    infer_stats[0] += st.len();
                    stories.extend(st);
                }
                for (seq, _ctx, src, _, _) in events.iter().filter(|_| !proposals_on) {
                    let Some(prefix) = row_prefix.get(src) else { continue };
                    let mut words: Vec<&'static str> = prefix.iter().map(|&w| vocab[w]).collect();
                    words.extend(seq.iter().map(|&i| i % 4096).filter(|&w| w < vocab.len()).map(|w| vocab[w]));
                    if words.last() != Some(&".") {
                        words.push(".");
                    }
                    let answer_at = words.len().saturating_sub(2);
                    if std::env::var("INFERDIAG").is_ok() && stories.len() < 12 {
                        eprintln!("  INFERDIAG s_i {s_i} replay story: {}", words.join(" "));
                    }
                    stories.push(Story { words, answer_at, held_out: false });
                    infer_stats[0] += 1;
                }
                for _ in 0..reps {
                    let mut order: Vec<usize> = (0..stories.len()).collect();
                    order.shuffle(&mut sleep_rng);
                    for i in order {
                        let st = &stories[i];
                        replay_queue.push(Story { words: st.words.clone(), answer_at: st.answer_at, held_out: false });
                    }
                }
            }
            let events = if infer_pairs { events } else { Vec::new() };
            let size = hc.len().max(1) as u64;
            // per inferred event: its words and the higher area's state when its source was read
            let mut items: Vec<(Vec<usize>, BitVector)> = Vec::new();
            for (seq, ctx, src, _fact, _) in &events {
                let mut state = BitVector::new(BITS, Some(0));
                match row_state.get(src).filter(|_| std::env::var("INFER_CTX_STATE").is_err()) {
                    // the higher area's own state when the source event was read
                    Some(bits) => {
                        for &b in bits {
                            state.bit_set(b as usize);
                        }
                    }
                    None => {
                        for &i in ctx {
                            // unfamiliar context only, as in answer-trace consolidation
                            if (hc.familiarity(&[i]) << 16) <= Q_03 as u64 * size {
                                state.or_mut(&enc.codes[i % 4096]);
                            }
                        }
                    }
                }
                let words: Vec<usize> = seq.iter().map(|&i| i % 4096).filter(|&w| w < vocab.len()).collect();
                if words.len() >= 2 {
                    items.push((words, state));
                }
            }
            let has = |st: &BitVector, w: usize| st.as_words().iter().zip(enc.codes[w].as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 28;
            // the schema across instances: for each new word, the targets every one of its
            // events has are frame (went, to, the); the varying ones are taught. For each
            // distinct event, the state is what its sources share (the season), not each
            // source story's own names and filler.
            let mut by_new: HashMap<usize, Vec<usize>> = HashMap::default();
            for (i, (w, _)) in items.iter().enumerate() {
                by_new.entry(w[0]).or_default().push(i);
            }
            for (_, idx) in by_new.iter() {
                let mut freq: HashMap<usize, usize> = HashMap::default();
                for &i in idx {
                    let mut seen: Vec<usize> = items[i].0[1..].to_vec();
                    seen.sort_unstable();
                    seen.dedup();
                    for w in seen {
                        *freq.entry(w).or_default() += 1;
                    }
                }
                let frame = |w: usize| freq.get(&w).copied().unwrap_or(0) * 10 >= idx.len() * 9;
                let mut done: Vec<Vec<usize>> = Vec::new();
                for &i in idx {
                    let words = &items[i].0;
                    if done.contains(words) {
                        continue;
                    }
                    done.push(words.clone());
                    let same: Vec<&BitVector> = idx.iter().filter(|&&j| &items[j].0 == words).map(|&j| &items[j].1).collect();
                    let mut state = BitVector::new(BITS, Some(0));
                    for w in 0..vocab.len() {
                        if same.iter().filter(|st| has(st, w)).count() * 2 > same.len() {
                            state.or_mut(&enc.codes[w]);
                        }
                    }
                    if std::env::var("INFERDIAG").is_ok() {
                        let st: Vec<&str> = (0..vocab.len()).filter(|&w| has(&state, w)).map(|w| vocab[w]).collect();
                        eprintln!("  INFERDIAG s_i {s_i}: {:?} from {} sources | shared state {:?}", words.iter().map(|&w| vocab[w]).collect::<Vec<_>>(), same.len(), st);
                    }
                    for k in 1..words.len() {
                        if infer_grow && frame(words[k]) {
                            continue;
                        }
                        let mut bag = BitVector::new(BITS, Some(0));
                        for &w in &words[..k] {
                            bag.or_mut(&enc.codes[w]);
                        }
                        let mut x = bag.as_words().to_vec();
                        x.extend_from_slice(state.as_words());
                        let x = BitVector::from_words(x);
                        if infer_grow {
                            // grow only: a kernel keyed on the new word, the word before the
                            // target and the shared state (no blame on the kernels that fire)
                            let mut mask = BitVector::new(2 * BITS, Some(0));
                            let (n, p) = (enc.codes[words[0]].as_words(), enc.codes[words[k - 1]].as_words());
                            for (i, m) in mask.as_words_mut().iter_mut().enumerate() {
                                *m = if i < BITS / 64 { n[i] | p[i] } else { state.as_words()[i - BITS / 64] };
                            }
                            area.column.l23.set_growth_mask(Some(mask));
                            for _ in 0..reps {
                                area.column.l23.grow(&x, &enc.codes[words[k]], 2, &mut sleep_rng);
                            }
                            area.column.l23.set_growth_mask(None);
                        } else {
                            for _ in 0..reps {
                                area.learn(&x, &enc.codes[words[k]], &mut sleep_rng);
                            }
                        }
                        infer_stats[1] += 1;
                    }
                    infer_stats[0] += 1;
                }
            }
        }
        // SEMANTIC: sleep replay of the sentences since the last sleep into the semantic store
        // REPLAY_GEN: the hippocampus's free-settling replay feeds sleep generalisation
        if let (true, Some(hc)) = (replay_gen && hippo_self && hier && !testing && s_i > 0 && sleep_every.map_or(false, |n| s_i % n == 0), bind_hc.as_ref()) {
            let ov = |i: usize, v: &BitVector| enc.codes[i].as_words().iter().zip(v.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>();
            let cued = std::env::var("REPLAY_GEN").map_or(false, |v| v == "cued");
            for _ in 0..gen_replays {
                // REPLAY_GEN=cued: a random familiar binding (a word in a slot that some event
                // stored) cues recall, which anchors the settling: a specific event, not a
                // prototype (as cortical slow oscillations cue hippocampal replay)
                let r = if cued {
                    let mut cue = None;
                    for _ in 0..64 {
                        let (w, c) = (sleep_rng.gen_range(0..vocab.len()), sleep_rng.gen_range(0..roles.used().max(1)));
                        let b = sparse_binding(&enc.codes[w], w, c, false);
                        if hc.familiarity(&b) > 0 {
                            cue = Some(b);
                            break;
                        }
                    }
                    match cue {
                        Some(b) => hc.recall(&b),
                        None => continue,
                    }
                } else {
                    hc.replay(&mut sleep_rng)
                };
                gen_stats[0] += 1;
                if r.ec.is_empty() {
                    continue;
                }
                let ep = BitVector::from_bits(&r.ec, BITS);
                let (mut content, mut context): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
                for c in 0..roles.used() {
                    let mut u = ep.clone();
                    u.rotr_mut(slot_offset(c));
                    let mut v = ep.clone();
                    v.rotr_mut(ctx_offset);
                    v.rotr_mut(slot_offset(c));
                    for w in 0..vocab.len() {
                        if ov(w, &u) >= 28 && !content.contains(&w) {
                            content.push(w);
                        }
                        if ov(w, &v) >= 28 && !context.contains(&w) {
                            context.push(w);
                        }
                    }
                }
                if std::env::var("GENDIAG").is_ok() && gen_stats[0] % 200 == 1 && s_i == TRAIN - sleep_every.unwrap_or(500) {
                    let names = |v: &[usize]| v.iter().map(|&w| vocab[w]).collect::<Vec<_>>();
                    eprintln!("  GENDIAG replay {}: content {:?} | context {:?}", gen_stats[0], names(&content), names(&context));
                }
                if content.len() < 2 {
                    continue;
                }
                gen_stats[1] += 1;
                let mut state = BitVector::new(BITS, Some(0));
                for &w in &context {
                    state.or_mut(&enc.codes[w]);
                }
                for &target in &content {
                    let mut bag = BitVector::new(BITS, Some(0));
                    for &w in content.iter().filter(|&&w| w != target) {
                        bag.or_mut(&enc.codes[w]);
                    }
                    let mut words = bag.as_words().to_vec();
                    words.extend_from_slice(state.as_words());
                    area.column.l23.add_replay(&BitVector::from_words(words), &enc.codes[target]);
                    gen_stats[2] += 1;
                }
            }
        }
        if let (Some(reps), true, Some(hc)) = (semantic_reps, hippo_self && s_i > 0 && (s_i == TRAIN || (!testing && sleep_every.map_or(false, |n| s_i % n == 0))), bind_hc.as_mut()) {
            // novelty-tagged events are replayed first (REPLAY_TAGGED=1), each `reps` times,
            // then the cue-free random replays
            let tags = if std::env::var("REPLAY_TAGGED").is_ok() { hc.take_tags() } else { Vec::new() };
            hc.begin_sleep();
            let hc = &*hc;
            // the hippocampus replays on its own (cue-free, from random CA3 starts); each
            // replay is read through the slot cells: in every slot, the words bound there
            let ov = |i: usize, v: &BitVector| enc.codes[i].as_words().iter().zip(v.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>();
            let tagged: Vec<&(Vec<u32>, Vec<usize>)> = tags.iter().flat_map(|t| std::iter::repeat(t).take(reps)).collect();
            // a tag's new inputs, read as a word: field → slot, bits → the word code they match
            let tag_word = |novel: &[usize]| -> Option<usize> {
                let mut code = BitVector::new(BITS, Some(0));
                if engram_mode() {
                    // binding ids: the word is the id's low part
                    return novel.first().map(|&i| i % 4096).filter(|&w| w < vocab.len());
                }
                for &b in novel {
                    code.bit_set(b % BITS);
                }
                (0..vocab.len()).max_by_key(|&w| enc.codes[w].as_words().iter().zip(code.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>())
            };
            for i in 0..tagged.len() + sem_hreplays * reps {
                let r = if i < tagged.len() { hc.replay_from(&tagged[i].0) } else { hc.replay(&mut sleep_rng) };
                sem_hstats[0] += 1;
                if r.ec.is_empty() {
                    continue;
                }
                let ep = BitVector::from_bits(&r.ec, BITS);
                let mut items: Vec<(usize, u64, usize)> = Vec::new(); // (word, familiarity of its binding, slot)
                for c in 0..roles.used() {
                    let mut u = ep.clone();
                    u.rotr_mut(slot_offset(c));
                    for w in 0..vocab.len() {
                        if ov(w, &u) >= 28 && !items.iter().any(|x| x.0 == w) {
                            let mut b = enc.codes[w].clone();
                            b.rotl_mut(slot_offset(c));
                            items.push((w, if sparse_bind { hc.familiarity(&sparse_binding(&enc.codes[w], w, c, false)) } else { hc.familiarity(&set_bits(&b)) }, c));
                        }
                    }
                }
                if std::env::var("TAGDIAG").is_ok() && i < tagged.len() && s_i >= TRAIN - 500 {
                    let d: Vec<String> = items.iter().map(|(w, f, _)| format!("{}:{}", vocab[*w], f)).collect();
                    eprintln!("  TAGDIAG s_i {s_i} tagged replay {i}/{}: {} bits -> {:?}", tagged.len(), r.ec.len(), d);
                }
                // the cue: for a tagged replay, what was new in the event (its tag); else the
                // least familiar binding read out
                let Some(cue_w) = (if i < tagged.len() { tag_word(&tagged[i].1) } else { None }).or_else(|| items.iter().min_by_key(|x| x.1).map(|x| x.0)) else { continue };
                // the fact's frame: its bindings at least half as familiar as the most familiar
                let frame: Option<u64> = sem_frame.then(|| {
                    let top = items.iter().filter(|x| x.0 != cue_w).map(|x| x.1).max().unwrap_or(0);
                    let mut fb: Vec<(usize, usize)> = items.iter().filter(|x| x.0 != cue_w && x.1 * 2 >= top && top > 0).map(|x| (x.0, x.2)).collect();
                    fb.sort_unstable();
                    let id = fb.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &(w, c)| (h ^ (w as u64 * 131 + c as u64 + 1)).wrapping_mul(0x100_0000_01b3));
                    frame_names.entry(id).or_insert(fb);
                    let fs = sem_frames.entry(cue_w).or_default();
                    if !fs.contains(&id) {
                        fs.push(id);
                    }
                    id
                });
                let mut content = BitVector::new(BITS, Some(0));
                for &(w, _, c) in items.iter().filter(|x| x.0 != cue_w) {
                    content.or_mut(&sem_bind_in(&enc.codes[w], (sem_typed > 0).then_some(c), frame));
                }
                if content.count_ones() == 0 {
                    continue;
                }
                sem_hstats[1] += 1;
                sem_hstats[2] += NEW_NAMES.contains(&vocab[cue_w]) as usize;
                let cue_slot = items.iter().find(|x| x.0 == cue_w).map(|x| x.2);
                let cue = &sem_bind(&enc.codes[cue_w], if sem_typed > 1 { cue_slot } else { None });
                let mut out = BitVector::new(BITS, Some(0));
                sem_store.process_predictive(cue, &mut out);
                // consolidation: the cortex already gives this tagged event's content back
                // from its cue, so the hippocampus may let it fade faster
                let got: u32 = out.as_words().iter().zip(content.as_words()).map(|(a, b)| (a & b).count_ones()).sum();
                // the cortex's error on this event sets its replay priority (engram store)
                hc.report_error(&r.ca3, ONE - ratio_q(got as u64, content.count_ones() as u64));
                if i < tagged.len() && std::env::var("NO_CONSOLIDATE_MARK").is_err() && got as usize * 10 >= content.count_ones() * 8 {
                    hc.mark_consolidated(&tagged[i].0);
                    sem_hstats[3] += 1;
                }
                sem_store.feedback(cue, &content, &mut sleep_rng);
                sem_replays += 1;
            }
        }
        if let (Some(reps), true) = (rel_reps, s_i > 0 && (s_i == TRAIN || (!testing && sleep_every.map_or(false, |n| s_i % n == 0)))) {
            rel_stats[0] += rel.consolidate(&enc.codes, reps, &mut rel_rng);
            // curiosity: once this sleep's facts are weighed, the open questions (people whose
            // family the module has claims about) are ranked by the value of information; the
            // chosen ones are asked of the teacher, and its answers weighed in a second pass
            // ASK_ROUNDS=r: asking in r rounds, each followed by a share of the practice
            // quiz, so the asking policy learns from each round's gains before the next
            energy = (energy + power).min(ONE);
            for round in 0..ask_rounds {
                if let (true, Some(k), Some(isa)) = (ask > 0, narrators, rel.relation_for(&[index["is"], index["a"]])) {
                    let mut asked: Vec<(usize, usize, u32)> = Vec::new(); // (person, context, lead before)
                    let people: Vec<usize> = NAMES.iter().chain(NEW_NAMES).chain(PRACTICE_NAMES).filter_map(|n| index.get(n).copied()).collect();
                    let mut open: Vec<usize> = Vec::new();
                    for &w in &people {
                        if let (Some((_, ctx)), Some((_, _, lead))) = (decisiveness(&rel, w, isa), lead_of(&rel, w, isa)) {
                            curiosity.note(w, ctx, ONE - lead.min(ONE));
                            open.push(w);
                        }
                    }
                    // the cost-aware policy's steps this round: (person, energy when chosen)
                    let time_band = round * neurocomp::program::curiosity::TIME_BANDS / ask_rounds;
                    let mut costed: Vec<(usize, u32)> = Vec::new();
                    if ask_cost {
                        for _ in 0..ask {
                            if energy == 0 {
                                break;
                            }
                            match curiosity.decide(energy, time_band, &mut ask_rng) {
                                Some((w, _)) => costed.push((w, energy)),
                                None => {
                                    curiosity.reward_stop(energy, time_band, &mut ask_rng);
                                    break;
                                }
                            }
                        }
                    }
                    let chosen: Vec<usize> = if ask_cost {
                        costed.iter().map(|x| x.0).collect()
                    } else if ask_random {
                        open.choose_multiple(&mut ask_rng, ask).copied().collect()
                    } else if ask_learned {
                        curiosity.pick_learned(ask, &mut ask_rng)
                    } else {
                        curiosity.pick(ask)
                    };
                    for w in chosen {
                        let (Some(f), Some((_, ctx))) = (true_family(vocab[w], &practice_truth), decisiveness(&rel, w, isa)) else { continue };
                        let before = lead_of(&rel, w, isa).map_or(0, |x| x.2);
                        rel.observe_from(&[w, index["is"], index["a"], index[SURNAMES[f]]], k as u16 + 1);
                        asked.push((w, ctx, before));
                        asked_log.push((w, sleeps_seen));
                    }
                    let work_before = rel.stats().1;
                    if !asked.is_empty() {
                        rel.consolidate(&enc.codes, reps, &mut rel_rng);
                    }
                    // the compute the answers took, shared among the questions
                    let work = (rel.stats().1 - work_before) as u64;
                    work_total += work;
                    let cost = ((work << 16) / (cost_unit * asked.len().max(1) as u64)).min(ONE as u64) as u32;
                    // the information each question gained: the rise in its answer's lead
                    for &(w, ctx, before) in &asked {
                        let after = lead_of(&rel, w, isa).map_or(0, |x| x.2);
                        let gain = after.saturating_sub(before);
                        match costed.iter().find(|x| x.0 == w) {
                            Some(&(_, e)) => {
                                curiosity.reward_ask(ctx, e, time_band, gain, cost, &mut ask_rng);
                                energy = energy.saturating_sub(cost);
                            }
                            None => curiosity.learn(ctx, gain, &mut ask_rng),
                        }
                    }
                }
                // the practice quiz: each practice name the module has claims about is asked its
                // family; the go/no-go answers (the believed family) or says "unknown", and the
                // world then reveals the truth
                if let (true, Some(isa), false) = (unknown_learned, rel.relation_for(&[index["is"], index["a"]]), testing) {
                    for _ in 0..quiz_reps.div_ceil(ask_rounds) {
                        for (j, n) in PRACTICE_NAMES.iter().enumerate() {
                            let Some(&w) = index.get(n) else { continue };
                            let Some((v, band)) = decisiveness(&rel, w, isa) else { continue };
                            let cands = [unknown_code(band, false), unknown_code(band, true)];
                            let answer = unknown_bg.select(&cands, Some(&mut quiz_rng)) == Some(1);
                            let right = vocab[v] == SURNAMES[practice_truth[j]];
                            let r = if !answer { ONE / 2 } else if right { ONE } else { 0 };
                            unknown_bg.reward_candidate(&cands[answer as usize], r as i32, &mut quiz_rng);
                            let q = &mut quiz_stats[band];
                            q[0] += 1;
                            q[1] += answer as usize;
                            q[2] += (answer && right) as usize;
                        }
                    }
                }
            }
            sleeps_seen += 1;
            // proposals judged again now that the facts of this stretch are in the module
            if let (true, Some(ireps), Some(hc), Some(isa)) = (proposals_on, infer_reps, bind_hc.as_ref(), rel.relation_for(&[index["is"], index["a"]])) {
                let stories = validate_proposals(&mut proposals, Some((&rel, isa)), &**hc, &row_narrator, &vocab, proposal_min, proposal_support);
                infer_stats[0] += stories.len();
                for _ in 0..ireps {
                    let mut order: Vec<usize> = (0..stories.len()).collect();
                    order.shuffle(&mut sleep_rng);
                    for i in order {
                        let st = &stories[i];
                        replay_queue.push(Story { words: st.words.clone(), answer_at: st.answer_at, held_out: false });
                    }
                }
            }
        }
        if let (Some(reps), true) = (semantic_reps, !hippo_self && s_i > 0 && (s_i == TRAIN || (!testing && sleep_every.map_or(false, |n| s_i % n == 0)))) {
            let rare = |w: usize| (word_count[w] as u64) * 100 < sentence_count as u64;
            let novel: Vec<usize> = (0..sem_buf.len()).filter(|&i| sem_buf[i].iter().any(|&w| rare(w))).collect();
            let others: Vec<usize> = (0..sem_buf.len()).filter(|&i| !novel.contains(&i)).collect();
            let mut order = novel.clone();
            order.extend(others.choose_multiple(&mut sleep_rng, novel.len()).copied());
            // SEMANTIC_RANDOM=n (control): n sentences drawn at random, no novelty priority
            if let Some(n) = std::env::var("SEMANTIC_RANDOM").ok().and_then(|v| v.parse::<usize>().ok()) {
                let all: Vec<usize> = (0..sem_buf.len()).collect();
                order = all.choose_multiple(&mut sleep_rng, n).copied().collect();
            }
            for _ in 0..reps {
                order.shuffle(&mut sleep_rng);
                for &i in &order {
                    let sent = &sem_buf[i];
                    let Some(&cue_w) = sent.iter().min_by_key(|&&w| word_count[w]) else { continue };
                    let mut content = BitVector::new(BITS, Some(0));
                    for &w in sent.iter().filter(|&&w| w != cue_w) {
                        content.or_mut(&enc.codes[w]);
                    }
                    if content.count_ones() == 0 {
                        continue;
                    }
                    let cue = &enc.codes[cue_w];
                    let mut out = BitVector::new(BITS, Some(0));
                    sem_store.process_predictive(cue, &mut out);
                    sem_store.feedback(cue, &content, &mut sleep_rng);
                    sem_replays += 1;
                }
            }
            sem_buf.clear();
        }
        if !testing && s_i > 0 && sleep_every.map_or(false, |n| s_i % n == 0) {
            let (p, m) = column.l23.sleep();
            if hier && std::env::var("HIER_DREAM").is_ok() {
                area.column.l23.generalize_from_replay();
            }
            if hier && std::env::var("HIER_SLEEP").is_ok() {
                area.column.l23.sleep();
                for u in upper.iter_mut() {
                    u.column.l23.sleep();
                }
            }
            sleeps += 1;
            slept_pruned += p;
            slept_merged += m;
        }
        if s_i == TRAIN {
            train_secs = phase_start.elapsed().as_secs_f64();
            phase_start = std::time::Instant::now();
        }
        if policy == Policy::LearnedGate && !testing && s_i % 50 == 0 && s_i > 0 {
            gate_routes = route_scores.top(8, 2 * ONE as u64);
        }
        if s_i == TRAIN && std::env::var("DIAG").is_ok() {
            // Coverage after training: per place, kernels that predict it from the
            // question context ("?" as current word) and read the memory frame.
            let frame = BITS / 64;
            let q = enc.codes[index["?"]].as_words();
            let mut line = String::new();
            for p in PLACES {
                let code = &enc.codes[index[p]];
                let (mut copy_q, mut blind_q, mut best_rel) = (0usize, 0usize, 0f32);
                for k in column.l23.kernels() {
                    if k.output_set.iter().filter(|&&b| code.bit_get(b as usize)).count() * 2 < k.output_set.len() {
                        continue;
                    }
                    let mut cur_ok = 0u32;
                    let mut cur_n = 0u32;
                    let mut mem = false;
                    for (wi, m) in k.input_words(k.input_set.last().map_or(0, |&b| b as usize / 64 + 1)).into_iter().enumerate() {
                        let w = k.input_idx + wi;
                        if w < frame {
                            cur_ok += (m & q[w]).count_ones();
                            cur_n += m.count_ones();
                        } else if w < frame * (1 + mid_frames) && m != 0 {
                            mem = true;
                        }
                    }
                    if cur_n == 0 || cur_ok * 10 < cur_n * 8 {
                        continue; // not a "?"-context kernel
                    }
                    let rel = (k.stats.hits as f32 + 1.0) / (k.stats.hits as f32 + k.stats.misses as f32 + 2.0);
                    if mem {
                        copy_q += 1;
                        best_rel = best_rel.max(rel);
                    } else {
                        blind_q += 1;
                    }
                }
                line += &format!(" {p}: copy {copy_q} (best {best_rel:.2}), blind {blind_q};");
            }
            eprintln!("  COVER seed {seed} after training:{line}");
        }
        if s_i == TRAIN {
            // TRUST_AT_TEST=f: reliability-aware ranking only when answering, so training
            // keeps the depth-first ranking that drives growth
            if let Some(f) = ratio_env("TRUST_AT_TEST") {
                column.l23.set_trust_floor(Some(f));
            }
            // HIER_TRUST_AT_TEST=f: the same reliability floor in the higher areas
            if let Some(f) = ratio_env("HIER_TRUST_AT_TEST") {
                area.column.l23.set_trust_floor(Some(f));
                for u in upper.iter_mut() {
                    u.column.l23.set_trust_floor(Some(f));
                }
            }
        }
        let mut s = if task == Task::Books {
            // pick a book; a finished one is replaced by a new book with a new season
            let b = story_rng.gen_range(0..3);
            let first = book_left[b] == 0;
            if first {
                book_season[b] = story_rng.gen_range(0..SEASONS.len());
                book_left[b] = story_rng.gen_range(3..=8);
                // the new book's action: its slot's (3 ids), or the next of BOOK_IDS in turn
                book_id[b] = if book_ids() == 3 { b } else { next_book_id };
                next_book_id = (next_book_id + 1) % book_ids();
            }
            book_left[b] -= 1;
            book_bin = if first {
                0
            } else {
                match s_i - book_last[b] - 1 {
                    0 => 1,
                    1..=2 => 2,
                    _ => 3,
                }
            };
            book_last[b] = s_i;
            open_book_next = book_id[b];
            book_session(&mut story_rng, book_id[b], book_season[b], first, testing && s_i % 2 == 1)
        } else if task == Task::Season {
            season_distance = story_rng.gen_range(0..season_len);
            let at = schema_at.get(&s_i).copied();
            let q_share: f64 = std::env::var("QUESTION_P").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
            if question() && (testing || story_rng.gen_bool(q_share)) {
                question_story(&mut story_rng, testing && s_i % 2 == 1)
            } else if family_stated() {
                season_story_with(&mut story_rng, season_distance, testing && s_i % 2 == 1, None, at.map(|a| a.0))
            } else {
                season_story_with(&mut story_rng, season_distance, testing && s_i % 2 == 1, at, None)
            }
        } else if task == Task::Habit {
            habit_story(&mut story_rng, testing && s_i % 2 == 1)
        } else if task == Task::Elim {
            elim_story(&mut story_rng, testing && s_i % 2 == 1)
        } else if task == Task::Give {
            give_story(&mut story_rng, testing && s_i % 2 == 1)
        } else if task == Task::Topic {
            topic_story(&mut story_rng, testing && s_i % 2 == 1)
        } else if task == Task::Persist {
            persist_story(&mut story_rng, s_i, testing && s_i % 2 == 1, testing)
        } else {
            story(&mut story_rng, task, max_facts, testing && s_i % 2 == 1)
        };
        // NARRATORS=k: each training story is told by narrator s_i % k (the reader knows who
        // tells it, as one knows a book's author). LIAR=p: the last narrator swaps the family
        // in a fact sentence ("X is a jones" → "smith") with probability p.
        // (narrator, whether this statement is a lie) for a statement about a new name
        let split = narrators.zip(schema_ord.get(&s_i)).filter(|_| narrator_split && !testing).map(|(k, &(i, o))| {
            if undecided && i == NEW_NAMES.len() - 1 {
                // UNDECIDED: two honest narrators disagree; the second is wrong this once
                (o % 2, o % 2 == 1)
            } else if (i + o) % 2 == 0 {
                (k - 1, true)
            } else {
                (0, false)
            }
        });
        let narrator: u16 = split.map_or_else(|| narrators.map_or(0, |k| (s_i % k) as u16), |n| n.0 as u16);
        // a practice statement, after the story's first sentence
        if let (Some(&(j, f)), false) = (practice_at.get(&s_i), testing) {
            let at = s.words.iter().position(|w| *w == ".").map_or(0, |i| i + 1);
            s.words.splice(at..at, [PRACTICE_NAMES[j], "is", "a", SURNAMES[f], "."]);
            if s.answer_at >= at {
                s.answer_at += 5;
            }
        }
        if let (Some(k), false) = (narrators, testing) {
            if (narrator as usize == k - 1 && liar > 0) || split.is_some_and(|x| x.1) {
                let mut lrng = StdRng::seed_from_u64(seed ^ (s_i as u64).wrapping_mul(0x9E37_79B9));
                let p = match split {
                    Some((_, lie)) => if lie { ONE } else { 0 },
                    None => liar,
                };
                for i in 3..s.words.len() {
                    // (a practice statement is stated as designed)
                    if s.words[i - 2] == "is" && s.words[i - 1] == "a" && !PRACTICE_NAMES.contains(&s.words[i - 3]) && chance(&mut lrng, p) {
                        if let Some(f) = SURNAMES.iter().position(|x| *x == s.words[i]) {
                            s.words[i] = SURNAMES[1 - f];
                            lies_told += 1;
                        }
                    }
                }
            }
        }
        if let Some(out) = dump.as_mut() {
            use std::io::Write;
            writeln!(out, "{}\t{}\t{}\t{}", if testing { "test" } else { "train" }, s.held_out as u8, s.answer_at, s.words.join(" ")).unwrap();
            continue;
        }
        // replay stories queued by this sleep (generative replay) are read first, through the
        // same steps as any story, as training: the cortex learns from them; nothing is
        // stored in memory and nothing is counted
        let mut stories_now: Vec<(Story, bool, bool)> = replay_queue.drain(..).map(|r| (r, true, false)).collect();
        // a retelling of the test story, read after it (RECITE)
        let retell = (recite.is_some() && testing).then(|| {
            let mut r = s.clone();
            r.answer_at = usize::MAX;
            r
        });
        stories_now.push((s, false, false));
        if let Some(r) = retell {
            stories_now.push((r, true, true));
        }
        for (s, replaying, reciting) in stories_now {
        #[allow(unused_mut)]
        let mut s = s;
        // a retelling: nothing stored or counted (as a replay), nothing learned (as a test)
        let testing = (testing && !replaying) || reciting;
        // the reading context is saved before a retelling and put back after it: what the
        // network said to itself does not become the context of the next story it reads
        let saved_reading = (reciting && hier && std::env::var("RECITE_KEEP_CONTEXT").is_err()).then(|| {
            let mut c = vec![area.save_context()];
            c.extend(upper.iter().map(|u| u.save_context()));
            c
        });
        // and the step-to-step carries: the previous word, its slot (the role cells chain
        // each word's slot on the previous one), the column's expectation
        let saved_carry = reciting.then(|| (prev, slot_prev, role_now, expect_prev.clone()));
        if prev_altered && !reciting {
            if let Some((p, sp, rn, ep)) = boundary_carry.clone() {
                prev = p;
                slot_prev = sp;
                role_now = rn;
                expect_prev = ep;
            }
        }
        story_altered = false;
        if reciting {
        } else if replaying {
            replay_words += s.words.len();
        } else if testing {
            test_words += s.words.len();
        } else {
            train_words += s.words.len();
            train_text_bytes += s.words.iter().map(|w| w.len() + 1).sum::<usize>();
        }
        let mut ids: Vec<usize> = s.words.iter().map(|w| index[w]).collect();
        completed_sentence = false;
        rolled = 0;
        rolled_surname = None;
        held_item = None;
        query_ec = None;
        query_rows.clear();
        attach_pending.clear();
        hold_pending.clear();
        answer_rows.clear();
        page_marks = vec![false; ids.len()];
        inner = vec![false; ids.len()];
        inner_code = vec![None; ids.len()];
        efference = vec![None; ids.len()];
        eff_pred = vec![None; ids.len()];
        let truth_ids = ids.clone();
        let mut recited: Vec<usize> = Vec::new();
        plan.clear();
        step_pending.clear();
        // the previous story's bindings become one episode (training stories only)
        if bind {
            if bind_story.count_ones() > 0 && s_i > 0 && s_i - 1 < TRAIN && !hippo_self && !prev_replaying {
                bind_mem.store(&bind_story);
                if let (Some(dg), Some(ca3)) = (&bind_dg, &mut bind_ca3) {
                    let x = set_bits(&bind_story);
                    ca3.store(&x, &dg.separate(&x));
                }
                if let Some(hc) = &mut bind_hc {
                    hc.store(&set_bits(&bind_story));
                    hc.advance_time();
                }
            }
            bind_story = BitVector::new(BITS, Some(0));
            bind_prev = BitVector::new(BITS, Some(0));
            bind_list.clear();
            if let Some(hc) = &mut bind_hc {
                hc.end_sequence();
            }
        }
        // ROUTE: the thalamus re-ranks the channels into slots, most useful first
        if route_on && !testing && !replaying && s_i > 0 && s_i <= route_critical && s_i % route_every == 0 && !route_order.is_empty() {
            let score = |c: &usize| route_score.get(c).map_or(0i64, |&(f, b)| f as i64 - b as i64);
            let before = route_order.clone();
            // with hysteresis: a channel moves up past its neighbour only when its net help
            // is more than twice the neighbour's (a swap moves every kernel's reading)
            for _ in 0..route_order.len() {
                for i in 0..route_order.len().saturating_sub(1) {
                    let (a, b) = (score(&route_order[i]), score(&route_order[i + 1]));
                    if b > 0 && b > 2 * a.max(0) {
                        route_order.swap(i, i + 1);
                    }
                }
            }
            if route_order != before {
                route_log.push(format!("story {s_i}: {:?}", route_order.iter().map(|c| (*c, score(c))).collect::<Vec<_>>()));
            }
        }
        // HIER_GROW: promote, prune or keep the bud
        if hier && bud_live && !testing && !replaying && s_i > 0 && s_i % grow_every == 0 {
            let (fixes, breaks) = bud_tally;
            let k = upper.len() - 1; // the bud's index
            // a sign test: fixes beat breaks by more than HIER_GROW_Z (2) standard deviations
            // (in integers: (fixes − breaks)² > z² · (fixes + breaks))
            let lead = fixes.saturating_sub(breaks);
            if fixes >= grow_min && lead * lead > grow_z * grow_z * (fixes + breaks) {
                grow_log.push(format!("story {s_i}: area {} (window {}) promoted, fixes {fixes} breaks {breaks}", k + 3, upper[k].span()));
                if upper.len() + 1 < grow_max {
                    upper.push(make_area(hier_span.pow(upper.len() as u32 + 2), false, false));
                } else {
                    bud_live = false;
                }
                bud_checks = 0;
            } else {
                bud_checks += 1;
                if bud_checks >= grow_patience {
                    grow_log.push(format!("story {s_i}: bud (window {}) pruned, fixes {fixes} breaks {breaks}", upper[k].span()));
                    upper[k] = make_area(upper[k].span(), false, false);
                    bud_checks = 0;
                }
            }
            bud_tally = (0, 0);
            upper_pred.resize(upper.len(), None);
            upper_in.resize(upper.len(), None);
            upper_has_answer.resize(upper.len(), 0);
        }
        if hier && hier_reset {
            area.clear();
            if let Some(a) = assoc.as_mut() {
                a.clear();
            }
            for u in upper.iter_mut() {
                u.clear();
            }
        }
        let mut t_next = 0;
        let mut seg_start = 0usize; // where the current event began (EVENT_BOUNDARY=learned)
        bcell.reset();
        while t_next < ids.len() {
            // (a while loop: completion can insert an internal word into the stream)
            let t = t_next;
            t_next += 1;
            // PROF: [4] storage and everything after learning (from the previous word)
            prof[4] += prof_t.elapsed().as_secs_f64();
            prof_t = std::time::Instant::now();
            // actions (efference copies of the reader's own actions)
            if task == Task::Books && hier {
                let w = s.words[t];
                if w == "@close" {
                    if let Some(b) = open_book.take() {
                        let mut c = vec![area.save_context()];
                        c.extend(upper.iter().map(|u| u.save_context()));
                        saved_ctx.insert(b, c);
                    }
                } else if w.starts_with("@open") {
                    let b = open_book_next;
                    open_book = Some(b);
                    let choice = match book_ctx.as_str() {
                        "reset" => 1,
                        "reinstate" => 2,
                        "learned" => {
                            let cands: Vec<BitVector> = (0..3)
                                .map(|a| {
                                    let mut c = ctx_code(a);
                                    c.rotl_mut((ids[t] * 131) % BITS);
                                    c
                                })
                                .collect();
                            let explore = if testing { None } else { Some(&mut bg_rng) };
                            ctx_pending = true;
                            ctx_bg.select(&cands, explore).unwrap_or(0)
                        }
                        _ => 0,
                    };
                    if testing {
                        ctx_chosen[choice] += 1;
                    }
                    match (choice, saved_ctx.get(&b)) {
                        (1, _) | (2, None) => {
                            area.clear();
                            for u in upper.iter_mut() {
                                u.clear();
                            }
                        }
                        (2, Some(c)) => {
                            area.restore_context(&c[0]);
                            for (u, c) in upper.iter_mut().zip(&c[1..]) {
                                u.restore_context(c);
                            }
                        }
                        _ => {}
                    }
                }
            }
            // an internal step of the closed loop is heard as the vector fed back
            let fed = inner_code[t].clone();
            let code = fed.as_ref().unwrap_or(&enc.codes[ids[t]]);
            if let Some(p) = prev {
                word_ctx[ids[t]].insert(p);
                word_ctx[p].insert(vocab.len() + ids[t]);
            }
            column.observe(code);
            sentence.or_mut(code);
            // Comparator: was this word predicted? Graded, at the word level: the share of
            // the prediction that this word accounts for. A prediction that superimposes a
            // whole class ("some name", "some place") gives each member a small share, so
            // the actual member still counts as unpredicted. Bitwise mismatch would not.
            // L5: the probability the column gave this word (its share of the prediction
            // times the predicting kernel's reliability); a lucky guess is still a surprise
            let mut share = ONE - column.surprise(code);
            if let Some(p) = eff_pred[t].as_ref() {
                // motor speech: the forward model's prediction of the sound is the copy
                share = if code.as_words().iter().zip(p.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24 { ONE } else { 0 };
            } else if let (true, Some(meant)) = (efference_on, efference[t]) {
                // the efference copy predicted this input: no surprise if it is what was
                // said, a full one if what was heard differs
                share = if ids[t] == meant { ONE } else { 0 };
            }
            if efference[t].is_some() {
                eff_stats[0] += 1;
                let altered = efference[t] != Some(ids[t]);
                eff_stats[1] += altered as usize;
                eff_stats[2] += (altered && share < predicted_share) as usize;
                eff_stats[3] += (!altered && share < predicted_share) as usize;
            }
            if learned_bound && !replaying {
                bcell.learn(ONE - share);
            }
            if sent_words == 0 {
                first_surprise = to_f32(ONE - share); // report
            }
            sent_surprise += to_f32(ONE - share); // report
            step_surprised = share < predicted_share || sent_words == 0;
            sent_surprises += (share < predicted_share) as usize;
            sent_words += 1;
            if share < predicted_share {
                surprising.or_mut(code);
                page_marks[t] = true;

                // a fact conflict (BOUNDARY): this rare surprising word against a different
                // rare word of the same kind still held; checked as the word arrives, so the
                // old context is gone before the next prediction
                if hier && bound_detect {
                    let w = ids[t];
                    let rare = |w: usize| sentence_count > 50 && ((word_count[w] as u64) << 16) < readback_rare as u64 * sentence_count as u64;
                    if rare(w) {
                        let same_kind = |a: usize, b: usize| {
                            let (x, y) = (&word_ctx[a], &word_ctx[b]);
                            let union = x.union(y).count();
                            union > 0 && ((x.intersection(y).count() as u64) << 16) >= bound_kind as u64 * union as u64
                        };
                        let conflicts = |c: &BitVector| enc.decode(c).map_or(false, |h| h != w && rare(h) && same_kind(h, w));
                        bound_fired |= area.forget_through(conflicts);
                        for u in upper.iter_mut() {
                            bound_fired |= u.forget_through(conflicts);
                        }
                    }
                }
                if hier {
                    area.note_word(code);
                    if let Some(a) = assoc.as_mut() {
                        a.note_word(code);
                    }
                    for u in upper.iter_mut() {
                        u.note_word(code);
                    }
                }
            }
            if std::env::var("TRACE_SHARE").is_ok() && testing && s_i < TRAIN + 3 {
                eprint!("{}:{:.2} ", s.words[t], to_f32(share));
                if t + 1 == ids.len() {
                    eprintln!();
                }
            }
            // slot of this word, and its binding if it was surprising
            if bind {
                let x = slot_input(&expect_prev, slot_prev, &roles);
                let slot = roles.observe(&x, !testing, &mut role_rng);
                // (HIPPO_BIND_ALL: the hippocampus binds every word of the event, not only the
                // words the column found surprising)
                if let (Some(c), true) = (slot, share < predicted_share || bind_all) {
                    let mut b = code.clone();
                    b.rotl_mut(slot_offset(c));
                    bind_story.or_mut(&b);
                    bind_list.push((ids[t], c));
                    bind_sentence.push(b);
                    bind_sentence_pairs.push((ids[t], c));
                    if let (Some(_), Some(hc), false) = (qhold, bind_hc.as_ref(), replaying) {
                        // take this word into working memory? The basal ganglia decide, seeing
                        // its novelty and that of what is held (none: band 4)
                        let nov = hc.novelty(&sparse_binding(code, ids[t], c, false));
                        let band = |n: Q16| (n as u64 * 4 >> 16).min(3) as usize;
                        if held_item.map_or(true, |(hw, _)| hw != ids[t]) {
                            let ctx = 6000 + band(nov) * 5 + held_item.map_or(4, |(_, hn)| band(hn));
                            let cands = [step_code(ctx, 0), step_code(ctx, 1)];
                            let a = step_bg.select(&cands, if !testing { Some(&mut bg_rng) } else { None }).unwrap_or(0);
                            if !testing {
                                if hold_credit_story {
                                    step_pending.push((cands[a].clone(), a == 1));
                                } else {
                                    hold_pending.push((cands[a].clone(), (a == 1).then_some(ids[t])));
                                }
                            }
                            if a == 1 {
                                held_item = Some((ids[t], nov));
                            }
                        }
                    }
                    // the held item read again: it queries what was stored with it here
                    if let (true, Some((hw, _)), Some(hc), false) = (qquery, held_item, bind_hc.as_ref(), replaying) {
                        let start = if learned_bound { seg_start } else { s.words[..t].iter().rposition(|w| *w == ".").map_or(0, |p| p + 1) };
                        if ids[t] == hw && !ids[..start].is_empty() && ids[..start].contains(&hw) {
                            let cue = sparse_binding(&enc.codes[hw], hw, HELD_FIELD, false);
                            // QQUERY=all: every event bound with the item here, blended
                            let found = if qquery_soft {
                                // soft: item × context, a bonus for this story's events
                                let (rows, words) = hc.recall_soft(&cue, 2, 8);
                                query_rows = rows.clone();
                                (!rows.is_empty()).then(|| (rows[0], words))
                            } else if qquery_all {
                                let (rows, words) = hc.recall_here_all(&cue);
                                query_rows = rows.clone();
                                (!rows.is_empty()).then(|| (rows[rows.len() - 1], words))
                            } else {
                                hc.recall_here(&cue)
                            };
                            if testing && s.held_out {
                                query_stats[0] += 1;
                                query_stats[1] += found.is_some() as usize;
                            }
                            query_ec = found.map(|(row, words)| {
                                let mut v = BitVector::new(BITS, Some(0));
                                for &w in words.iter().filter(|&&w| w != hw && w < vocab.len()) {
                                    v.or_mut(&enc.codes[w]);
                                }
                                if testing && s.held_out {
                                    query_stats[2] += words.iter().any(|&w| SURNAMES.contains(&vocab.get(w).copied().unwrap_or(""))) as usize;
                                }
                                (row, v)
                            });
                        }
                    }
                }
                slot_prev = slot;
            }
            // active reading: choose a saccade before predicting the next word
            if let (true, Some(mode)) = (hier && t + 1 < ids.len(), saccade.as_deref()) {
                let cur_start = s.words[..=t].iter().rposition(|w| *w == ".").map_or(0, |i| i + 1);
                // the column's own confidence about the next word (peek, no top-down)
                let peek_l4 = (sacc_conf || sacc_cortex).then(|| {
                    let empty = BitVector::new(BITS, Some(0));
                    l4_row(&column, code, &vec![empty; l4_mid], route_on.then_some(&route_order[..]))
                });
                let bucket = match (sacc_conf || sacc_cortex, peek_l4.as_ref().and_then(|x| column.l23.peek_scored(x))) {
                    (false, _) => 0,
                    (true, None) => 0,
                    (true, Some((_, c))) if c < Q_HALF => 1,
                    (true, Some((_, c))) if c < Q_08 => 2,
                    _ => 3,
                };
                // the cortical state: possible continuations + confidence code
                let state = peek_l4.as_ref().filter(|_| sacc_cortex).map(|x| {
                    let mut st = column.l23.peek_union(x, BITS);
                    st.or_mut(&sacc_code(10 + bucket));
                    st
                });
                let prev_w = if t > 0 { ids[t - 1] } else { vocab.len() };
                let ctx = (prev_w * 131 + ids[t] * 7919 + bucket * 104_729) % BITS;
                // landmarks (index mode): distinct surprising words before the current
                // sentence, each at its latest position
                let mut marks: Vec<(usize, usize)> = Vec::new(); // (word, position)
                if mode == "index" {
                    for j in (0..cur_start).rev() {
                        if page_marks[j] && s.words[j] != "." && !marks.iter().any(|m| m.0 == ids[j]) {
                            marks.push((ids[j], j));
                        }
                    }
                }
                let a = if mode == "oracle" {
                    if t + 1 == s.answer_at { 2 } else { 0 }
                } else {
                    let cands: Vec<BitVector> = if let Some(state) = state.as_ref() {
                        // candidate = the cortical state rotated by the candidate's own amount
                        let ids_c: Vec<usize> = if mode == "index" { std::iter::once(0).chain(marks.iter().map(|m| m.0 + 1)).collect() } else { (0..3).collect() };
                        ids_c
                            .iter()
                            .map(|&c| {
                                let mut x = state.clone();
                                x.rotl_mut((c * 2_654_435_761 + 97) % BITS);
                                x
                            })
                            .collect()
                    } else if mode == "index" {
                        std::iter::once(sacc_code(0))
                            .chain(marks.iter().map(|m| enc.codes[m.0].clone()))
                            .map(|mut c| {
                                c.rotl_mut(ctx);
                                c
                            })
                            .collect()
                    } else {
                        (0..3)
                            .map(|a| {
                                let mut c = sacc_code(a);
                                c.rotl_mut(ctx);
                                c
                            })
                            .collect()
                    };
                    let explore = if testing { None } else { Some(&mut bg_rng) };
                    let a = sacc_bg.select(&cands, explore).unwrap_or(0);
                    sacc_pending = Some(a);
                    a
                };
                // the target sentence: [start, end] word indices, end at its "."
                let sentence_of = |j: usize| {
                    let start = s.words[..j].iter().rposition(|w| *w == ".").map_or(0, |i| i + 1);
                    let end = s.words[j..].iter().position(|w| *w == ".").map_or(s.words.len() - 1, |e| j + e);
                    (start, end)
                };
                let target = match a {
                    a if mode == "index" && a > 0 => Some(sentence_of(marks[a - 1].1)),
                    1 if cur_start > 0 => {
                        let prev_start = s.words[..cur_start - 1].iter().rposition(|w| *w == ".").map_or(0, |i| i + 1);
                        Some((prev_start, cur_start - 1))
                    }
                    2 => s.words.iter().position(|w| *w == ".").filter(|&e| e < cur_start).map(|e| (0, e)),
                    _ => None,
                };
                if testing {
                    sacc_stats[3] += 1;
                }
                if testing && t + 1 == s.answer_at && task == Task::Season {
                    let k = (s.held_out && new_wording()) as usize;
                    sacc_wording[k].0 += 1;
                    sacc_wording[k].1 += target.map_or(false, |(st, _)| st == 0) as usize;
                }
                if let Some((start, end)) = target {
                    if testing {
                        sacc_stats[if mode == "index" { 1 } else { a - 1 }] += 1;
                        sacc_hit_first += (t + 1 == s.answer_at && start == 0) as usize;
                        sacc_stats[2] += end + 1 - start;
                        sacc_stats[4] += (t + 1 == s.answer_at) as usize;
                    }
                    let empty = BitVector::new(BITS, Some(0));
                    let mids = vec![empty.clone(); l4_mid];
                    let mut re_surprising = BitVector::new(BITS, Some(0));
                    for j in start..=end {
                        let c = &enc.codes[ids[j]];
                        column.observe(c);
                        if ONE - column.surprise(c) < predicted_share {
                            re_surprising.or_mut(c);
                            area.note_word(c);
                            for u in upper.iter_mut() {
                                u.note_word(c);
                            }
                        }
                        let l4 = l4_row(&column, c, &mids, route_on.then_some(&route_order[..]));
                        column.predict(&l4);
                    }
                    area.end_sentence(&re_surprising);
                    for u in upper.iter_mut() {
                        u.end_sentence(&re_surprising);
                    }
                    // the return saccade: fixate the previous and the current word again
                    if t > 0 {
                        column.observe(&enc.codes[ids[t - 1]]);
                    }
                    column.observe(code);
                }
            }
            // MEM_CONTEXT=1: episodes are bound to their context. The higher area's slow state
            // (what the story has established, e.g. the season) joins every stored episode and
            // every recall cue
            let mem_context = hier && std::env::var("MEM_CONTEXT").is_ok();
            let cue_with_context = mem_context.then(|| {
                let mut c = if predictive_novelty { surprising.clone() } else { sentence.clone() };
                c.or_mut(&area.state(&surprising));
                c
            });
            let cue_source = match cue_with_context.as_ref() {
                Some(c) => c,
                None if predictive_novelty => &surprising,
                None => &sentence,
            };
            if let Policy::Pfc { learned } = policy {
                let load = if learned {
                    let explore = if testing { None } else { Some(&mut bg_rng) };
                    pfc_gate.decide(ids[t], explore) == Gate::Load
                } else {
                    names_set.contains(&ids[t]) // hand-set rule: hold the last name
                };
                if load {
                    wm.load(0, code);
                }
                if testing {
                    let e = pfc_loads.entry(vocab[ids[t]]).or_default();
                    e.0 += load as usize;
                    e.1 += 1;
                }
            }

            // PROF: [5] observe, surprise, sentence bag, working-memory gate
            prof[5] += prof_t.elapsed().as_secs_f64();
            prof_t = std::time::Instant::now();
            if t + 1 < ids.len() {
                // INNER_SLOT: an inner step has no page word; what was said arrives in the
                // heard slot instead
                let mut words = if inner_slot && inner[t] { vec![0u64; BITS / 64] } else { code.as_words().to_vec() };
                match policy {
                    // (with HIER the top-down frame takes the empty frame's place: growth
                    // deepens one frame at a time and would stop at an always-empty frame)
                    Policy::NoMemory => words.extend(std::iter::repeat(0).take(base_frames * BITS / 64)),
                    Policy::FixedRelay => words.extend_from_slice(column.l6.relay().as_words()),
                    Policy::ThalamicGate | Policy::LearnedGate => {
                        // channel contents: each relay route, then memory recall (None)
                        let mut contents: Vec<(Option<RelayChannel>, BitVector)> = Vec::new();
                        for r in &gate_routes {
                            if let Some(v) = column.l6.relay_channel(*r) {
                                contents.push((Some(*r), v.clone()));
                            }
                        }
                        let cue = memory.rarest(cue_source, Q_TENTH, rarity_ratio);
                        if cue.count_ones() > 0 {
                            let need = (neurocomp::fixed::mul_floor(cue.count_ones() as u64, SEVEN_TENTHS) as u32).max(min_overlap);
                            if let Some(ep) = memory.recall(&cue, need) {
                                let mut recalled = memory.novel(ep, habituation);
                                for (r, &c) in recalled.as_words_mut().iter_mut().zip(cue.as_words()) {
                                    *r &= !c;
                                }
                                if recalled.count_ones() > 0 {
                                    contents.push((None, recalled));
                                }
                            }
                        }
                        let mut released = BitVector::new(BITS, Some(0));
                        if !contents.is_empty() {
                            // candidate = channel code bound to the current word (rotation)
                            let cands: Vec<BitVector> = contents
                                .iter()
                                .map(|(c, _)| {
                                    let mut code = channel_code(*c);
                                    code.rotl_mut((ids[t] * 131) % BITS);
                                    code
                                })
                                .collect();
                            let explore = if testing { None } else { Some(&mut bg_rng) };
                            if let Some(i) = gate_bg.select(&cands, explore) {
                                released = contents[i].1.clone();
                                gate_pending = Some(released.clone());
                                if testing && t + 1 == s.answer_at {
                                    let name = contents[i].0.map_or("memory".to_string(), |r| format!("({},{})", r.query_lag, r.value_offset));
                                    *gate_chosen_at_answer.entry(name).or_default() += 1;
                                }
                            }
                        }
                        words.extend_from_slice(released.as_words());
                    }
                    Policy::L6Gate { gated } => {
                        // channel contents: each route's relay, then memory recall
                        let mut contents: Vec<Option<BitVector>> = routes.iter().map(|r| column.l6.relay_channel(*r).cloned()).collect();
                        let cue = memory.rarest(cue_source, Q_TENTH, rarity_ratio);
                        let mut recalled = None;
                        if cue.count_ones() > 0 {
                            let need = (neurocomp::fixed::mul_floor(cue.count_ones() as u64, SEVEN_TENTHS) as u32).max(min_overlap);
                            if let Some(ep) = memory.recall(&cue, need) {
                                let mut r = memory.novel(ep, habituation);
                                for (x, &c) in r.as_words_mut().iter_mut().zip(cue.as_words()) {
                                    *x &= !c;
                                }
                                if r.count_ones() > 0 {
                                    recalled = Some(r);
                                }
                            }
                        }
                        contents.push(recalled);
                        // L6: the cortical context is the current input, or with L6_CONTEXT=pair
                        // the current and previous inputs (previous bound by a 1-bit rotation)
                        let ctx = if l6_pair {
                            let mut c = column.previous();
                            c.rotl_mut(1);
                            c.or_mut(code);
                            c
                        } else {
                            code.clone()
                        };
                        let open = if gated && s_i >= l6_warmup {
                            let explore = if testing { None } else { Some(&mut bg_rng) };
                            l6_gate.open(&ctx, explore)
                        } else {
                            vec![true; contents.len()]
                        };
                        let mut passed = vec![false; contents.len()];
                        for (c, content) in contents.iter().enumerate() {
                            match content {
                                Some(v) if open[c] => {
                                    passed[c] = true;
                                    words.extend_from_slice(v.as_words());
                                }
                                _ => words.extend(std::iter::repeat(0).take(BITS / 64)),
                            }
                        }
                        if testing {
                            l6_open_sum += passed.iter().filter(|&&p| p).count();
                            l6_words += 1;
                            if t + 1 == s.answer_at {
                                let kind = (t >= 1 && s.words[t - 1] == "give") as usize;
                                l6_answers[kind] += 1;
                                for (c, &p) in passed.iter().enumerate() {
                                    l6_open_at_answer[kind][c] += p as usize;
                                }
                            }
                        }
                        l6_step = Some((ctx, passed));
                    }
                    Policy::Loop(hops) => {
                        let cue = memory.rarest(cue_source, Q_TENTH, rarity_ratio);
                        let chain = memory.recall_chain(&cue, hops, habituation, Q_TENTH, rarity_ratio);
                        if std::env::var("TRACE").is_ok() && testing && t + 1 == s.answer_at && s_i < TRAIN + 3 {
                            let names = |bv: &BitVector| -> Vec<&str> {
                                (0..vocab.len())
                                    .filter(|&i| enc.codes[i].as_words().iter().zip(bv.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24)
                                    .map(|i| vocab[i])
                                    .collect()
                            };
                            let hops_named: Vec<Vec<&str>> = chain.iter().map(|h| names(h)).collect();
                            eprintln!("{:?}\n  cue {:?} -> hops {:?}", s.words, names(&cue), hops_named);
                            let freqs: Vec<String> = ["picked", "up", "the", "went", "where", "mary", "john", "kitchen", "ball"]
                                .iter()
                                .map(|w| format!("{w}:{:.2}", memory.frequency(&enc.codes[index[w]])))
                                .collect();
                            eprintln!("  frequencies: {}", freqs.join(" "));
                        }
                        for h in 0..hops {
                            match chain.get(h) {
                                Some(frame) => words.extend_from_slice(frame.as_words()),
                                None => words.extend(std::iter::repeat(0).take(BITS / 64)),
                            }
                        }
                    }
                    Policy::Select => {
                        let cue = memory.rarest(cue_source, Q_TENTH, rarity_ratio);
                        let (mut hop1, mut hop2) = (BitVector::new(BITS, Some(0)), BitVector::new(BITS, Some(0)));
                        let need = (neurocomp::fixed::mul_ceil(cue.count_ones() as u64, SEVEN_TENTHS) as u32).max(1);
                        if cue.count_ones() > 0 {
                            if let Some((id, ep)) = memory.recall_excluding(&cue, need, &[]) {
                                hop1 = memory.novel(ep, habituation);
                                for (c, &u) in hop1.as_words_mut().iter_mut().zip(cue.as_words()) {
                                    *c &= !u;
                                }
                                let items = memory.items(&hop1, 16);
                                let explore = if testing { None } else { Some(&mut bg_rng) };
                                if let Some(i) = bg.select(&items, explore) {
                                    let item = &items[i];
                                    let need = (neurocomp::fixed::mul_ceil(item.count_ones() as u64, SEVEN_TENTHS) as u32).max(1);
                                    if let Some((_, ep2)) = memory.recall_excluding(item, need, &[id]) {
                                        hop2 = memory.novel(ep2, habituation);
                                        for (c, (&u, &it)) in hop2.as_words_mut().iter_mut().zip(cue.as_words().iter().zip(item.as_words())) {
                                            *c &= !(u | it);
                                        }
                                    }
                                    bg_pending = Some(hop2.clone());
                                }
                                if std::env::var("TRACE").is_ok() && testing && t + 1 == s.answer_at && s_i < TRAIN + 5 {
                                    let names = |bv: &BitVector| -> Vec<&str> {
                                        (0..vocab.len())
                                            .filter(|&i| enc.codes[i].as_words().iter().zip(bv.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24)
                                            .map(|i| vocab[i])
                                            .collect()
                                    };
                                    let vals: Vec<String> = items.iter().map(|it| format!("{:?}={:.2}", names(it), bg.value(it))).collect();
                                    eprintln!("{:?}\n  cue {:?} -> hop1 {:?}; items {}; hop2 {:?}", s.words, names(&cue), names(&hop1), vals.join(" "), names(&hop2));
                                }
                            }
                        }
                        words.extend_from_slice(hop1.as_words());
                        words.extend_from_slice(hop2.as_words());
                    }
                    Policy::Branch(b) => {
                        let cue = memory.rarest(cue_source, Q_TENTH, rarity_ratio);
                        for frame in memory.recall_branches(&cue, b, habituation, 16) {
                            words.extend_from_slice(frame.as_words());
                        }
                    }
                    Policy::Ca3 { .. } => {
                        let cue = memory.rarest(cue_source, Q_TENTH, rarity_ratio);
                        let mut recalled = BitVector::new(BITS, Some(0));
                        let cue_bits = set_bits(&cue);
                        if !cue_bits.is_empty() {
                            let (bits, strength) = ca3.as_ref().unwrap().recall(&cue_bits, BITS);
                            if std::env::var("TRACE").is_ok() && testing && t + 1 == s.answer_at && s_i < TRAIN + 3 {
                                let raw = BitVector::from_bits(&bits, BITS);
                                let names = |bv: &BitVector| -> Vec<&str> {
                                    (0..vocab.len())
                                        .filter(|&i| enc.codes[i].as_words().iter().zip(bv.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24)
                                        .map(|i| vocab[i])
                                        .collect()
                                };
                                eprintln!("{:?}\n  CA3 cue {:?} -> raw recall {:?} ({} bits, strength {strength:.2})", s.words, names(&cue), names(&raw), bits.len());
                            }
                            if strength > 0 {
                                recalled = memory.novel(&BitVector::from_bits(&bits, BITS), habituation);
                                for (r, &c) in recalled.as_words_mut().iter_mut().zip(cue.as_words()) {
                                    *r &= !c;
                                }
                            }
                        }
                        words.extend_from_slice(recalled.as_words());
                    }
                    Policy::Episodic | Policy::Consolidate | Policy::Pfc { .. } => {
                        // Pfc: the cue is what working memory holds (prefrontal-directed retrieval)
                        let cue = if matches!(policy, Policy::Pfc { .. }) { wm.content() } else { memory.rarest(cue_source, Q_TENTH, rarity_ratio) };
                        let mut recalled = BitVector::new(BITS, Some(0));
                        if cue.count_ones() > 0 {
                            // recall needs most of the cue to be present in the episode
                            let need = (neurocomp::fixed::mul_floor(cue.count_ones() as u64, SEVEN_TENTHS) as u32).max(min_overlap);
                            if let Some((id, ep)) = memory.recall_excluding(&cue, need, &[]) {
                                last_recall_id = Some(id);
                                last_recall_cue = Some(cue.clone());
                                recalled = memory.novel(ep, habituation);
                                // what the memory adds beyond the cue
                                for (r, &c) in recalled.as_words_mut().iter_mut().zip(cue.as_words()) {
                                    *r &= !c;
                                }
                            } else if policy == Policy::Consolidate {
                                // hippocampus has nothing: the cortex's consolidated knowledge
                                if let Some(out) = semantic.peek(&cue) {
                                    recalled = out.clone();
                                    for (r, &c) in recalled.as_words_mut().iter_mut().zip(cue.as_words()) {
                                        *r &= !c;
                                    }
                                    if testing && t + 1 == s.answer_at {
                                        from_cortex += 1;
                                    }
                                }
                            }
                        }
                        if std::env::var("TRACE").is_ok() && testing && t + 1 == s.answer_at && s_i < TRAIN + 3 {
                            let names = |bv: &BitVector| -> Vec<&str> {
                                (0..vocab.len())
                                    .filter(|&i| enc.codes[i].as_words().iter().zip(bv.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24)
                                    .map(|i| vocab[i])
                                    .collect()
                            };
                            eprintln!(
                                "{:?}\n  cue {:?} ({} bits; sentence {} bits) -> recalled novel {:?}",
                                s.words,
                                names(&cue),
                                cue.count_ones(),
                                sentence.count_ones(),
                                names(&recalled)
                            );
                            let freqs: Vec<String> = s.words[..=t].iter().map(|w| format!("{w}:{:.2}", memory.frequency(&enc.codes[index[w]]))).collect();
                            eprintln!("  word frequency over {} stored episodes: {}", memory.len(), freqs.join(" "));
                        }
                        words.extend_from_slice(recalled.as_words());
                    }
                }
                // the higher area: [sentence bag | slow state] -> top-down frame, through a
                // thalamic gate (HIER_GATE) and placed after the memory / relay frames, or right
                // after the current word with HIER_EARLY
                let mut td_src: Option<BitVector> = None;
                // the column's own memory / relay frames, for arbitration
                let mem_src = {
                    let frame = BITS / 64;
                    let mut m = vec![0u64; frame];
                    for f in 1..words.len() / frame {
                        for (a, b) in m.iter_mut().zip(&words[frame * f..frame * (f + 1)]) {
                            *a |= b;
                        }
                    }
                    BitVector::from_words(m)
                };
                // COOPERATE: where the column is uncertain about the next word (its own prediction,
                // without top-down or memory frames, is missing or under half reliable), the
                // cortex asks memory about the sentence's least familiar word: the semantic
                // store's content for it ("lucy" → "is a jones") joins the higher areas'
                // sentence context for this step. Where the column is sure, nothing is asked.
                let mut own_conf_step: Q16 = ONE;
                let sentence_plus = {
                    // HIER_UP=surprise: only surprisal goes up; the sentence frame holds the
                    // sentence's surprising words and the current word (where the area is),
                    // not the words the column predicted
                    let mut x = if hier_up_surprise {
                        let mut x = surprising.clone();
                        x.or_mut(code);
                        x
                    } else if hier_up_both {
                        let mut quiet = surprising.clone();
                        quiet.not_mut();
                        quiet.and_mut(&sentence);
                        let mut burst = surprising.clone();
                        burst.rotl_mut(BURST_ROT);
                        quiet.or_mut(&burst);
                        quiet
                    } else {
                        sentence.clone()
                    };
                    // QAREA: the held item's queried event is reinstated in the higher areas'
                    // working context (hippocampus → entorhinal → association cortex), as
                    // the semantic store's content is under COOPERATE
                    if let (true, Some((_, q))) = (qarea, query_ec.as_ref().filter(|_| !(testing && bind_lesion))) {
                        x.or_mut(q);
                    }
                    if cooperate || graded_enrich || sparse_hc {
                        let empty = BitVector::new(BITS, Some(0));
                        let own = column.l23.peek_scored(&l4_row(&column, code, &vec![empty; l4_mid], route_on.then_some(&route_order[..])));
                        let conf = own.map_or(0, |(_, c)| c);
                        own_conf_step = conf;
                        let start = if learned_bound { seg_start } else { s.words[..=t].iter().rposition(|x| *x == ".").map_or(0, |i| i + 1) };
                        let (cue, cue_fam) = if hippo_self {
                            bind_sentence_pairs
                                .iter()
                                .zip(&bind_sentence)
                                .map(|((w, c), b)| (*w, fam_binding(&bind_hc, &bind_mem, true, sparse_bind, b, &enc.codes[*w], *w, *c)))
                                .min_by_key(|x| x.1)
                                .map_or((None, 0), |(w, f)| (Some(w), f))
                        } else {
                            ids[start..=t].iter().map(|&w| (w, word_count[w] as u64)).min_by_key(|x| x.1).map_or((None, 0), |(w, f)| (Some(w), f))
                        };
                        let cb = CONF_BANDS.iter().filter(|&&e| conf >= e).count();
                        let unsure = if graded_enrich {
                            true
                        } else if gate_learned && cue.is_some() {
                            let fb = (64 - cue_fam.leading_zeros() as usize).min(7);
                            let cands = [mg_code(cb * 8 + fb, 0), mg_code(cb * 8 + fb, 1)];
                            let learn = !testing && !replaying;
                            let a = mg_bg.select(&cands, if learn { Some(&mut bg_rng) } else { None }).unwrap_or(0);
                            if learn {
                                mg_pending = Some((cands[a].clone(), a == 1));
                            }
                            a == 1
                        } else {
                            conf < Q_HALF
                        };
                        if testing && !replaying {
                            mg_stats[cb][0] += 1;
                            mg_stats[cb][1] += (unsure && (cooperate || graded_enrich)) as usize;
                        }
                        if unsure && (cooperate || graded_enrich) {
                            if let Some(mut out) = cue.and_then(|w| sem_read(&sem_store, &enc.codes, w, slot_in(&bind_sentence_pairs, w), sem_typed, roles.used(), false, &sem_frames, rel_reps.map(|_| (&rel, rel_hops)), &bind_sentence_pairs)) {
                                if graded_enrich {
                                    // graded: each bit of the store's answer gets in with
                                    // probability 1 − the column's confidence (a fixed hash per
                                    // bit and step: a sure column lets almost nothing in)
                                    let doubt = (ONE - conf.min(ONE)) as u64;
                                    let words: Vec<u64> = out
                                        .as_words()
                                        .iter()
                                        .enumerate()
                                        .map(|(i, &w)| {
                                            let mut m = 0u64;
                                            for b in 0..64 {
                                                if w >> b & 1 == 1 {
                                                    let mut h = ((i * 64 + b) as u64) ^ ((t as u64) << 32) ^ 0x9E37_79B9_7F4A_7C15;
                                                    h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                                                    h ^= h >> 31;
                                                    if (h & 0xFFFF) < doubt {
                                                        m |= 1 << b;
                                                    }
                                                }
                                            }
                                            m
                                        })
                                        .collect();
                                    out = BitVector::from_words(words);
                                }
                                x.or_mut(&out);
                                coop_stats[0] += 1;
                                coop_stats[1] += (testing && s.held_out) as usize;
                            }
                        }
                    }
                    x
                };
                // does the column need help with the next word? (HC_SURPRISE, HIER_SURPRISE)
                if hc_surprise || hier_surprise || hier_learned {
                    let empty = BitVector::new(BITS, Some(0));
                    let own = column.l23.peek_scored(&l4_row(&column, code, &vec![empty; l4_mid], route_on.then_some(&route_order[..]))).map_or(0, |(_, c)| c);
                    step_needs_help = step_surprised || own < Q_HALF;
                    if hier_learned && hier {
                        let band = CONF_BANDS.iter().filter(|&&e| own >= e).count();
                        // consult first: a tie (nothing learned yet) consults, as before the gate
                        let cands = [hier_code(ids[t], band, step_surprised, true), hier_code(ids[t], band, step_surprised, false)];
                        let learn = !testing && !replaying && !inner[t + 1];
                        // while learning the area is always consulted (it and the column keep
                        // learning as before) and the gate learns, from the counterfactual,
                        // where the frame helps; answering, the gate decides
                        let i = if learn && hier_learn_always {
                            0
                        } else if hier_margin > 0 {
                            // skip only where skipping is clearly worth more (HIER_MARGIN)
                            let i = (hier_bg.value(&cands[1]) > hier_bg.value(&cands[0]) + hier_margin) as usize;
                            let i = if learn && chance(&mut bg_rng, hier_bg.explore) { bg_rng.gen_range(0..2) } else { i };
                            hier_bg.select(&cands[i..=i], None::<&mut StdRng>);
                            i
                        } else {
                            hier_bg.select(&cands, if learn { Some(&mut bg_rng) } else { None }).unwrap_or(0)
                        };
                        hier_consult = i == 0;
                        hier_choices[testing as usize][hier_consult as usize] += 1;
                        hier_pending = learn.then(|| cands[i].clone());
                    }
                }
                // REINSTATE: what the hippocampus brings back is the cortical state it indexed
                let surprising_r = match (reinstate && hier && !bind_sentence_pairs.is_empty() && !(testing && bind_lesion), bind_hc.as_ref()) {
                    (true, Some(hc)) => {
                        let cue: Vec<usize> = bind_sentence_pairs.iter().flat_map(|&(w, c)| sparse_binding(&enc.codes[w], w, c, false)).collect();
                        // REINSTATE=here: only this story's events (no fallback to other stories)
                        let (rows, _) = if reinstate_here {
                            let (mut r, w) = hc.recall_here_all(&cue);
                            r.sort_unstable_by(|a, b| b.cmp(a));
                            r.truncate(reinstate_top);
                            (r, w)
                        } else {
                            hc.recall_soft(&cue, reinstate_bonus, reinstate_top)
                        };
                        // ACH with REINSTATE: the mode's signal comes from this same retrieval.
                        // Its best event from this story: familiar here (CA1's mismatch on
                        // it, approximated as 0); none, or only other stories': novel
                        if ach_on && !reciting {
                            let nov = if rows.iter().any(|&r| hc.row_here(r)) { 0 } else { ONE };
                            ach = ((3 * ach as u64 + nov as u64) / 4) as Q16;
                            if testing {
                                ach_stats[0] += 1;
                                ach_stats[1] += ach as u64;
                            }
                        }
                        let mut x = surprising.clone();
                        let mut n = 0u64;
                        // ACH: only a share 1 − ACh of the reinstated bits passes
                        let pass = if ach_on { (ONE - ach.min(ONE)) as u64 } else { ONE as u64 };
                        for r in &rows {
                            if let Some(st) = row_cortex.get(r) {
                                let mut v = st.clone();
                                if pass < ONE as u64 {
                                    for (wi, w) in v.as_words_mut().iter_mut().enumerate() {
                                        let mut keep = 0u64;
                                        for b in 0..64 {
                                            let h = ((wi * 64 + b) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 48;
                                            if h < pass {
                                                keep |= 1 << b;
                                            }
                                        }
                                        *w &= keep;
                                    }
                                }
                                x.or_mut(&v);
                                n += 1;
                            }
                        }
                        if testing && n > 0 {
                            reinstate_stats[0] += 1;
                            reinstate_stats[2] += x.count_ones() as u64;
                        }
                        x
                    }
                    _ => surprising.clone(),
                };
                if hier {
                    // the chain, top down: each upper area predicts from its window and the
                    // prediction of the area above it
                    let hier_skip = (hier_surprise && !step_needs_help) || (hier_learned && !hier_consult);
                    gated_steps[2 + !hier_skip as usize] += hier_surprise as usize;
                    let mut above: Option<BitVector> = None;
                    for i in (0..upper.len()).rev() {
                        let hin_u = upper[i].input_with(&sentence_plus, if assoc_on { &surprising } else { &surprising_r }, if chain_mix { None } else { above.as_ref() });
                        let p = if hier_skip { BitVector::new(BITS, Some(0)) } else { upper[i].predict(&hin_u) };
                        upper_pred[i] = (p.count_ones() > 0).then(|| p.clone());
                        if testing && t + 1 == s.answer_at {
                            upper_has_answer[i] += (p.as_words().iter().zip(enc.codes[ids[t + 1]].as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24) as usize;
                        }
                        above = Some(p);
                        upper_in[i] = (!hier_skip).then_some(hin_u);
                    }
                    let hin = if let Some(mode) = role_mode.as_deref() {
                        // the column's expectation for the next slot (peek: no top-down, no state)
                        let empty = BitVector::new(BITS, Some(0));
                        let expect = column.l23.peek_union(&l4_row(&column, code, &vec![empty.clone(); l4_mid], route_on.then_some(&route_order[..])), BITS);
                        let lead = if mode == "raw" {
                            expect
                        } else {
                            role_now = roles.observe(&expect, !testing, &mut role_rng);
                            role_now.map_or(empty, |c| roles.code(c).clone())
                        };
                        area.input_lead(&lead, &sentence_plus, if assoc_on { &surprising } else { &surprising_r })
                    } else {
                        area.input_with(&sentence_plus, if assoc_on { &surprising } else { &surprising_r }, if chain_mix { None } else { above.as_ref() })
                    };
                    let td = if hier_skip { BitVector::new(BITS, Some(0)) } else { area.predict(&hin) };
                    if std::env::var("HIERDIAG").is_ok() && testing && t + 1 == s.answer_at && s_i < TRAIN + 8 {
                        let names = |bv: &BitVector| -> Vec<&str> {
                            (0..vocab.len()).filter(|&i| enc.codes[i].as_words().iter().zip(bv.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24).map(|i| vocab[i]).collect()
                        };
                        eprintln!(
                            "  HIERDIAG {:?}\n    state {:?} -> top-down {:?} (confidence {:.2}), answer {}",
                            s.words,
                            names(&area.state(&surprising)),
                            names(&td),
                            area.column.confidence(),
                            s.words[t + 1]
                        );
                    }
                    if testing && t + 1 == s.answer_at {
                        topdown_has_answer += (td.as_words().iter().zip(enc.codes[ids[t + 1]].as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24) as usize;
                    }
                    // HIER_GATE: the corticothalamic gain on the top-down channel, per context
                    // (the current word); all open during the warm-up
                    // HIER_GATE_CONF=c: pass only predictions the higher area is confident in
                    // (its winning kernel's reliability, the L5 confidence it sends up)
                    let confident = s_i < hier_gate_warmup || hier_gate_conf.map_or(true, |c| area.column.confidence() >= c);
                    let passed = confident && if hier_gate_on && s_i >= hier_gate_warmup {
                        let explore = if testing { None } else { Some(&mut bg_rng) };
                        hier_gate.open(code, explore)[0]
                    } else {
                        true
                    };
                    let has = td.count_ones() > 0;
                    td_src = has.then(|| td.clone());
                    if testing && t + 1 == s.answer_at {
                        topdown_passed_at_answer += (passed && has) as usize;
                    }
                    hier_gate_step = (hier_gate_on && has && passed).then(|| code.clone());
                    let mut scaled: Option<BitVector> = None;
                    let trusted = match (hier_trust_gate.as_deref(), has) {
                        (Some("cf"), true) => {
                            let p = if t > 0 { ids[t - 1] } else { vocab.len() };
                            let key = (p * (vocab.len() + 1) + ids[t]) as u64 * 64;
                            let bucket = |c: Q16| CONF_BANDS.iter().filter(|&&e| c >= e).count() as u64;
                            let share = (2 * mix.rate(TRUST_CF, key + bucket(area.column.confidence())) as u64).min(ONE as u64);
                            let share = if std::env::var("HIER_CF_NOSCALE").is_ok() { ONE as u64 } else { share };
                            cf_scaled[testing as usize] += (share < ONE as u64) as usize;
                            cf_share_sum[testing as usize] += share;
                            trust_passed[testing as usize][1] += 1;
                            td_full = (!testing).then(|| td.clone());
                            if share < ONE as u64 {
                                // a fixed subset of bit positions (by a hash of the position)
                                let mut words = td.as_words().to_vec();
                                for (wi, w) in words.iter_mut().enumerate() {
                                    let mut keep = 0u64;
                                    for b in 0..64 {
                                        let h = ((wi * 64 + b) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 48;
                                        if h < share {
                                            keep |= 1 << b;
                                        }
                                    }
                                    *w &= keep;
                                }
                                scaled = Some(BitVector::from_words(words));
                            }
                            true
                        }
                        (Some(mode), true) => {
                            let p = if t > 0 { ids[t - 1] } else { vocab.len() };
                            let key = (p * (vocab.len() + 1) + ids[t]) as u64 * 64;
                            let bucket = |c: Q16| CONF_BANDS.iter().filter(|&&e| c >= e).count() as u64;
                            let area_rate = mix.rate(TRUST_AREA, key + bucket(area.column.confidence()));
                            let ok = if mode == "column" { area_rate >= mix.rate(TRUST_COLUMN, key + bucket(column.confidence())) } else { area_rate >= ONE / 2 };
                            trust_passed[testing as usize][ok as usize] += 1;
                            ok
                        }
                        _ => true,
                    };
                    let frame = if passed && trusted { scaled.unwrap_or(td) } else { BitVector::new(BITS, Some(0)) };
                    let fw = frame.as_words().len();
                    if hier_early {
                        let at = BITS / 64; // right after the current word
                        td_span = Some((at, fw));
                        words.splice(at..at, frame.as_words().iter().copied());
                    } else {
                        td_span = Some((words.len(), fw));
                        words.extend_from_slice(frame.as_words());
                    }
                    hier_in = (!hier_skip).then_some(hin);
                    // the association area: [sentence | its slow context + the hippocampus's return]
                    if let Some(a) = assoc.as_mut() {
                        let ain = a.input(&sentence_plus, &surprising_r);
                        let p = a.predict(&ain);
                        assoc_word = enc.decode(&p);
                        assoc_conf = a.column.confidence();
                        assoc_in = Some(ain);
                    }
                }
                if inner_slot {
                    // the heard slot (auditory input, the phonological store): one's own speech
                    // at an inner step, else silence; INNER_SLOT=echo: subvocalisation, the
                    // page's word is heard there too as it is read
                    if inner[t] || inner_echo {
                        words.extend_from_slice(code.as_words());
                    } else {
                        words.extend(std::iter::repeat(0).take(BITS / 64));
                    }
                }
                if hc_ec {
                    // the entorhinal feedback slot, filled once the hippocampus has recalled
                    words.extend(std::iter::repeat(0).take(BITS / 64));
                }
                words.extend_from_slice(column.previous().as_words()); // L6: the previous input
                if testing && t + 1 == s.answer_at {
                    // memory diagnostic: does any recalled frame contain the answer word?
                    let frame = BITS / 64;
                    let answer = enc.codes[ids[t + 1]].as_words();
                    let found = (0..mid_frames).any(|f| {
                        words[frame * (1 + f)..frame * (2 + f)].iter().zip(answer).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24
                    });
                    recall_has_answer += found as usize;
                    // how many places the recall offers (1 = clean, more = blended)
                    let mem = &words[frame..frame * (1 + mid_frames)];
                    {
                        let mem = &words[frame..frame * (1 + mid_frames)];
                        let ans_bits: usize = (0..mid_frames)
                            .map(|f| mem[frame * f..frame * (f + 1)].iter().zip(answer).map(|(a, b)| (a & b).count_ones() as usize).sum::<usize>())
                            .max()
                            .unwrap_or(0);
                        let size: usize = mem.iter().map(|w| w.count_ones() as usize).sum();
                        let whole = (0..vocab.len())
                            .filter(|&i| {
                                let code = enc.codes[i].as_words();
                                (0..mid_frames).any(|f| mem[frame * f..frame * (f + 1)].iter().zip(code).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24)
                            })
                            .count();
                        pending_diag = [ans_bits, size, whole];
                    }
                    recall_places += PLACES
                        .iter()
                        .filter(|p| {
                            let code = enc.codes[index[*p]].as_words();
                            (0..mid_frames).any(|f| mem[frame * f..frame * (f + 1)].iter().zip(code).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24)
                        })
                        .count();
                }
                // PROF: [0] L4 assembly: recall, relays, gates
                prof[0] += prof_t.elapsed().as_secs_f64();
                prof_t = std::time::Instant::now();
                let words = if route_on && t + 1 < ids.len() {
                    let fw = BITS / 64;
                    let nf = words.len() / fw;
                    let word = BitVector::from_words(words[..fw].to_vec());
                    // channels: the hand row's frames 1.. (their index is their source), then
                    // each promoted higher area (not the bud)
                    let mut ch: Vec<(usize, BitVector)> = (1..nf).map(|f| (f, BitVector::from_words(words[f * fw..(f + 1) * fw].to_vec()))).collect();
                    for (i, p) in upper_pred.iter().enumerate() {
                        if bud_live && i + 1 == upper.len() {
                            continue;
                        }
                        if let Some(p) = p {
                            ch.push((32 + i, p.clone()));
                        }
                    }
                    for (c, _) in &ch {
                        if !route_order.contains(c) {
                            route_order.push(*c);
                        }
                    }
                    let slots = nf - 1 + route_extra_slots();
                    let key = ((if t > 0 { ids[t - 1] } else { vocab.len() }) * (vocab.len() + 1) + ids[t]) as u64 * 64;
                    let shares: Vec<u64> = ch.iter().map(|(c, _)| if route_noscale { ONE as u64 } else { (2 * mix.rate(ROUTE_SRC + *c as u8, key) as u64).min(ONE as u64) }).collect();
                    let row = route_row(&word, &ch, &shares, &route_order, slots, None);
                    route_cur = Some((word.clone(), ch.clone(), slots));
                    route_step = (!testing && !inner[t + 1]).then(|| (word, ch, slots));
                    row
                } else {
                    words
                };
                let input = BitVector::from_words(words);
                l4_mid = (input.as_words().len() / (BITS / 64)).saturating_sub(2);
                // LEARNING=three: the cerebellum works in parallel with the cortex's
                // feedforward sweep, from a copy of its input (cortex → pons → mossy fibres)
                // its candidates (every matching kernel's prediction), as the column's
                // expectation is the union of its own
                let cb_union: Option<BitVector> = cerebellum.as_ref().map(|cb| cb.kernels.peek_union(&input, BITS));
                let cb_early: Option<BitVector> = cerebellum.as_mut().map(|cb| {
                    let p = cb.predict(&input).clone();
                    cb_word = enc.decode(&p);
                    cb_conf = cb.confidence();
                    cb_input = Some(input.clone());
                    p
                });
                if bind && !(testing && bind_lesion) {
                    // the column's expectation for the next slot, and that slot's cell
                    expect_prev = column.l23.peek_union(&input, BITS);
                    // LEARNING=three: the cerebellum's candidates join the expectation
                    if let Some(u) = cb_union.as_ref() {
                        expect_prev.or_mut(u);
                    }
                    let next_slot = roles.winner(&slot_input(&expect_prev, slot_prev, &roles)).map(|w| w.0);
                    bind_answer = None;
                    bind_raw = None;
                    if let (Some(c), true) = (next_slot, bind_story.count_ones() > 0) {
                        let cue = match bind_hab {
                            Some(f) if bind_mem.len() > 50 => bind_mem.novel(&bind_story, f),
                            _ => bind_story.clone(),
                        };
                        // BIND_RARE=1: rarity-weighted recall
                        let found = if let Some(hc) = &bind_hc {
                            // the full circuit: the story's bindings so far are the cue (on its
                            // own: this sentence's bindings with the story so far as context)
                            let cue = if hippo_self { event_vec(&bind_sentence, &bind_prev, ctx_offset) } else { cue };
                            let reuse = hc_surprise && !step_needs_help && hc_cached.is_some();
                            let r = if reuse {
                                gated_steps[0] += 1;
                                hc_cached.clone().unwrap_or_default()
                            } else if sparse_bind {
                                // the sparse cue: this sentence's bindings, and the story's earlier ones as context
                                let n_prev = bind_list.len().saturating_sub(bind_sentence_pairs.len());
                                let mut idx: Vec<usize> = bind_sentence_pairs.iter().flat_map(|&(w, c)| sparse_binding(&enc.codes[w], w, c, false)).collect();
                                idx.extend(bind_list[..n_prev].iter().flat_map(|&(w, c)| sparse_binding(&enc.codes[w], w, c, true)));
                                // what is held in working memory is active, so it cues too
                                if let Some((hw, _)) = held_item {
                                    idx.extend(sparse_binding(&enc.codes[hw], hw, HELD_FIELD, false));
                                }
                                idx.sort_unstable();
                                idx.dedup();
                                if cue_ctl && !bind_sentence_pairs.is_empty() {
                                    let fams: Vec<u64> = bind_sentence_pairs.iter().zip(&bind_sentence).map(|(&(w, c), b)| fam_binding(&bind_hc, &bind_mem, true, true, b, &enc.codes[w], w, c)).collect();
                                    let band = fams.iter().min().map_or(7, |&c| (64 - c.leading_zeros() as usize).min(7));
                                    let conf = CONF_BANDS.iter().filter(|&&e| column.confidence() >= e).count();
                                    let cctx = conf * 8 + band;
                                    let cands: Vec<BitVector> = (0..4).map(|a| cue_code(cctx, a)).collect();
                                    let learn = !replaying && (!testing || cue_test_learn);
                                    let a = cue_bg.select(&cands, if learn { Some(&mut bg_rng) } else { None }).unwrap_or(0);
                                    if learn {
                                        cue_pending = Some((cands[a].clone(), a));
                                    }
                                    if testing && !replaying {
                                        cue_stats[a][0] += 1;
                                    } else if !replaying {
                                        cue_train[a] += 1;
                                    }
                                    if testing && !replaying {
                                        cue_last_test = Some(a);
                                    }
                                    match a {
                                        0 => hc.recall_as(&idx, false),
                                        1 => hc.recall_as(&idx, true),
                                        2 => {
                                            let mut c: Vec<usize> = bind_sentence_pairs.iter().flat_map(|&(w, c)| sparse_binding(&enc.codes[w], w, c, false)).collect();
                                            c.sort_unstable();
                                            c.dedup();
                                            hc.recall_as(&c, false)
                                        }
                                        _ => {
                                            let i = (0..fams.len()).min_by_key(|&i| fams[i]).unwrap();
                                            let (w, c) = bind_sentence_pairs[i];
                                            let mut c: Vec<usize> = sparse_binding(&enc.codes[w], w, c, false);
                                            c.extend(bind_list[..n_prev].iter().flat_map(|&(w, c)| sparse_binding(&enc.codes[w], w, c, true)));
                                            c.sort_unstable();
                                            c.dedup();
                                            hc.recall_as(&c, false)
                                        }
                                    }
                                } else if reciting {
                                    // cued by the network's own words: recall without
                                    // strengthening what is recalled
                                    hc.recall_peek(&idx)
                                } else {
                                    hc.recall(&idx)
                                }
                            } else {
                                hc.recall(&set_bits(&cue))
                            };
                            if ach_on && !reinstate && !reciting {
                                // novelty is the episode's, not the words': content recalled
                                // from another story is "seen, but not here" (full novelty);
                                // from this story, CA1's mismatch
                                let nov = match r.ca3.first() {
                                    None => ONE,
                                    Some(&row) if !hc.row_here(row) => ONE,
                                    Some(_) => ONE - r.ca1_match.min(ONE),
                                };
                                ach = ((3 * ach as u64 + nov as u64) / 4) as Q16;
                                if testing {
                                    ach_stats[0] += 1;
                                    ach_stats[1] += ach as u64;
                                }
                            }
                            if t + 1 == s.answer_at {
                                answer_rows = r.ca1.clone();
                                if let Some((qr, _)) = query_ec.as_ref() {
                                    answer_rows.push(*qr);
                                    answer_rows.extend(query_rows.iter().copied());
                                }
                            }
                            if hc_surprise && !reuse {
                                gated_steps[1] += 1;
                                hc_cached = Some(r.clone());
                            }
                            if testing && !reciting && t + 1 == s.answer_at {
                                if let Some(src) = r.ca3.last().and_then(|&row| hc.row_source(row as u32)) {
                                    source_stats[(src > 0) as usize] += 1;
                                }
                            }
                            if std::env::var("SELFDIAG").is_ok() && testing && s.held_out && bind_diag < 10 && (qhold.is_none() || t + 1 == s.answer_at) && s.words[..=t].iter().any(|w| NEW_NAMES.contains(w)) {
                                bind_diag += 1;
                                let decode = |v: &BitVector| -> Vec<String> {
                                    let mut out = Vec::new();
                                    for sl in 0..roles.used() {
                                        let mut u = v.clone();
                                        u.rotr_mut(slot_offset(sl));
                                        for w in 0..vocab.len() {
                                            if enc.codes[w].as_words().iter().zip(u.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 28 {
                                                out.push(format!("{}@{}", vocab[w], sl));
                                            }
                                        }
                                    }
                                    out
                                };
                                let ep = BitVector::from_bits(&r.ec, BITS);
                                eprintln!("  SELFDIAG at {:?} (next slot {c}): cue {:?}\n    recalled ({} bits, strength {}, CA1 match {}): {:?}", &s.words[..=t], decode(&cue), r.ec.len(), r.strength, r.ca1_match, decode(&ep));
                            }
                            (!r.ec.is_empty()).then(|| (64 * (32 - r.strength.leading_zeros() as u64).saturating_sub(5), BitVector::from_bits(&r.ec, BITS)))
                        } else if let Some(ca3) = &bind_ca3 {
                            // the learned hippocampus: pattern completion from the (habituated) cue
                            let cue = if hippo_hab > 0 && bind_mem.len() > 50 { bind_mem.novel(&bind_story, hippo_hab) } else { bind_story.clone() };
                            let (bits, strength) = ca3.recall(&set_bits(&cue), BITS);
                            (!bits.is_empty()).then(|| (64 * (64 - (strength as u64).leading_zeros() as u64).saturating_sub(5), BitVector::from_bits(&bits, BITS)))
                        } else if std::env::var("BIND_RARE").is_ok() {
                            bind_mem.recall_rare_scored(&cue, 64).map(|(sc, e)| (sc, e.clone()))
                        } else {
                            bind_mem.recall(&cue, 64).map(|e| (64, e.clone()))
                        };
                        if let Some((score, ep)) = found {
                            bind_strength = (score / 64).min(7);
                            let mut u = ep.clone();
                            u.rotr_mut(slot_offset(c));
                            bind_raw = Some(u.clone());
                            bind_answer = if class_read {
                                // candidates in the unbound episode, best overlap first, kept
                                // if the cortex expects them here
                                let ov = |i: usize, v: &BitVector| enc.codes[i].as_words().iter().zip(v.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>();
                                let mut cands: Vec<(u32, usize)> = (0..vocab.len()).map(|i| (ov(i, &u), i)).filter(|x| x.0 >= 24).collect();
                                cands.sort_by(|a, b| b.cmp(a));
                                cands.iter().find(|&&(_, i)| ov(i, &expect_prev) >= 24).or(cands.first()).map(|x| x.1)
                            } else {
                                enc.decode(&u)
                            };
                            if std::env::var("BINDDIAG").is_ok() && testing && s.held_out && t + 1 == s.answer_at && bind_diag < 6 {
                                bind_diag += 1;
                                let unbind = |slot: usize| -> Vec<&str> {
                                    let mut v = ep.clone();
                                    v.rotr_mut(slot_offset(slot));
                                    (0..vocab.len()).filter(|&i| enc.codes[i].as_words().iter().zip(v.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24).map(|i| vocab[i]).collect()
                                };
                                let story: Vec<String> = bind_list.iter().map(|(w, sl)| format!("{}@{}", vocab[*w], sl)).collect();
                                let mut slots: Vec<usize> = bind_list.iter().map(|x| x.1).chain(std::iter::once(c)).collect();
                                slots.sort_unstable();
                                slots.dedup();
                                let rec: Vec<String> = slots.iter().map(|&sl| format!("{}:{:?}", sl, unbind(sl))).collect();
                                eprintln!("  BINDDIAG {:?}\n    story bindings {:?}; next slot {c}; answer {} -> read {:?}\n    recalled episode by slot: {}", s.words, story, s.words[t + 1], bind_answer.map(|w| vocab[w]), rec.join(" "));
                            }
                        }
                    }
                }

                // the cue controller's outcome: did the recall it shaped give the next word?
                let cue_right = bind_answer == Some(ids[t + 1]);
                if let Some((code, a)) = cue_pending.take() {
                    cue_bg.reward_candidate(&code, if cue_right { ONE as i32 } else { 0 } - if a == 1 { cue_cost } else { 0 }, &mut bg_rng);
                }
                if let Some(a) = cue_last_test.take() {
                    cue_stats[a][1] += cue_right as usize;
                }
                if std::env::var("COMPLETEDIAG").is_ok() && bind && (s.held_out || !testing) && NEW_NAMES.contains(&s.words[t]) && bind_diag < 12 {
                    bind_diag += 1;
                    let wl = |v: &BitVector| -> Vec<&str> { (0..vocab.len()).filter(|&i| enc.codes[i].as_words().iter().zip(v.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24).map(|i| vocab[i]).collect() };
                    let next_slot = roles.winner(&slot_input(&expect_prev, slot_prev, &roles)).map(|w| w.0);
                    let story: Vec<String> = bind_list.iter().map(|(w, sl)| format!("{}@{}", vocab[*w], sl)).collect();
                    eprintln!("  COMPLETEDIAG {} {:?}\n    at {}: expects {:?}; next slot {:?}; memory reads {:?}; bindings {:?}", if testing { "test" } else { "train" }, s.words, s.words[t], wl(&expect_prev), next_slot, bind_answer.map(|w| vocab[w]), story);
                }
                // the semantic store's cue: the sentence's least familiar word (on its own: by
                // the hippocampus's counts of its bindings; else by word counts)
                let sem_cue_w: Option<usize> = if hippo_self {
                    bind_sentence_pairs.iter().zip(&bind_sentence).min_by_key(|((w, c), b)| fam_binding(&bind_hc, &bind_mem, true, sparse_bind, b, &enc.codes[*w], *w, *c)).map(|((w, _), _)| *w)
                } else {
                    let start = if learned_bound { seg_start } else { s.words[..=t].iter().rposition(|x| *x == ".").map_or(0, |i| i + 1) };
                    ids[start..=t].iter().min_by_key(|&&w| word_count[w]).copied()
                };
                let mut fed_vec: Option<BitVector> = None;
                let rollout_w = if rollout && rolled < 4 && (testing || complete.as_deref() == Some("rollout")) && bind && mem_size(&bind_hc, &bind_mem, hippo_self) > 0 {
                    let ov = |i: usize, v: &BitVector| enc.codes[i].as_words().iter().zip(v.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>();
                    let band = bind_sentence.iter().zip(&bind_sentence_pairs).map(|(b, &(w, c))| fam_binding(&bind_hc, &bind_mem, hippo_self, sparse_bind, b, &enc.codes[w], w, c)).min().map_or(7, |c| (64 - c.leading_zeros() as u64).min(7));
                    // the column's expectation: with LEARNING=three, the integrated one (the slow
                    // cortex's and the cerebellum's), as the column's output is
                    let mut expect = column.l23.peek_union(&input, BITS);
                    if let Some(u) = cb_union.as_ref() {
                        expect.or_mut(u);
                    }
                    // a definite expectation (one word) violated starts a rollout; a started
                    // rollout continues while the page does not match the expectation
                    let col_w = match (column.l23.peek_scored(&input), cb_early.as_ref()) {
                        // LEARNING=three: the more confident of the slow cortex and the cerebellum
                        (Some((o, c)), Some(p)) if c < cb_conf => enc.decode(p).or_else(|| enc.decode(&o)),
                        (Some((o, _)), _) => enc.decode(&o),
                        (None, Some(p)) => enc.decode(p),
                        (None, None) => None,
                    };
                    // what starts a recall is a definite prediction the page contradicts. Under
                    // LEARNING=three that prediction is the integrated winner (`col_w`); the
                    // union of candidates (`expect`) only filters what recall may offer
                    let (definite, contradicted) = if three {
                        (col_w.is_some(), col_w.map_or(false, |w| ov(ids[t + 1], &enc.codes[w]) < 24))
                    } else {
                        (
                            if rollout_loop { expect.count_ones() <= 48 } else { (0..vocab.len()).filter(|&i| ov(i, &expect) >= 24).count() == 1 },
                            expect.count_ones() > 0 && ov(ids[t + 1], &expect) < 24,
                        )
                    };
                    let skipped = if step_learned { contradicted } else { (definite || rolled > 0) && contradicted };
                    let from_area = || -> Option<usize> {
                        let td = td_src.as_ref().filter(|_| rollout_area)?;
                        (0..vocab.len()).filter(|&i| ov(i, td) >= 24 && ov(i, &expect) >= 24).max_by_key(|&i| ov(i, td))
                    };
                    let from_sem = || -> Option<usize> {
                        semantic_reps?;
                        let out = sem_read(&sem_store, &enc.codes, sem_cue_w?, slot_in(&bind_sentence_pairs, sem_cue_w?), sem_typed, roles.used(), true, &sem_frames, rel_reps.map(|_| (&rel, rel_hops)), &bind_sentence_pairs)?;
                        (0..vocab.len()).filter(|&i| ov(i, &out) >= 24 && ov(i, &expect) >= 24).max_by_key(|&i| ov(i, &out))
                    };
                    let mem_w = bind_answer.filter(|&w| !(testing && bind_lesion) && ov(w, &expect) >= 24);
                    let sem_w = if mem_w.is_none() { from_sem() } else { None };
                    if sem_w.is_some() && skipped {
                        sem_used[0] += 1;
                        sem_used[1] += (testing && s.held_out) as usize;
                    }
                    // (word, offering source: 0 slot memory, 1 semantic store, 2 higher area, 3 column)
                    let offer: Option<(usize, usize)> = if rollout_mix && mixing {
                        let prev = if t > 0 { ids[t - 1] } else { vocab.len() };
                        let fb = if bind_fam { band } else { 0 };
                        let ctx = ((prev * (vocab.len() + 1) + ids[t]) as u64 * 8 + fb) * 8;
                        let bucket = |c: Q16| CONF_BANDS.iter().filter(|&&e| c >= e).count() as u64;
                        let sem_any = from_sem();
                        let cands: Vec<(usize, u8, u64, usize)> = [
                            mem_w.map(|w| (w, 6u8, ctx + bind_strength, 0usize)),
                            sem_any.map(|w| (w, 8, ctx, 1)),
                            from_area().map(|w| (w, 2, ctx + bucket(area.column.confidence()), 2)),
                            col_w.filter(|&w| ov(w, &expect) >= 24).map(|w| (w, 0, ctx + bucket(column.confidence()), 3)),
                        ]
                        .into_iter()
                        .flatten()
                        .collect();
                        let votes: Vec<(usize, u32)> = cands.iter().map(|&(w, src, key, _)| (w, mix.weight(src, key))).collect();
                        SourceMix::combine(&votes).map(|(w, _)| {
                            let src = cands.iter().filter(|c| c.0 == w).max_by_key(|c| mix.weight(c.1, c.2)).map_or(3, |c| c.3);
                            (w, src)
                        })
                    } else {
                        mem_w.map(|w| (w, 0)).or(sem_w.map(|w| (w, 1))).or_else(|| from_area().map(|w| (w, 2))).or(col_w.map(|w| (w, 3)))
                    };
                    // the closed loop: the first source whose output, gated by the expectation,
                    // keeps a word's worth of bits; that vector is what is fed back
                    let offer = if rollout_loop {
                        let gate = |v: &BitVector| -> Option<BitVector> {
                            let mut g = v.clone();
                            for (a, b) in g.as_words_mut().iter_mut().zip(expect.as_words()) {
                                *a &= *b;
                            }
                            (g.count_ones() >= 24).then_some(g)
                        };
                        let start = if learned_bound { seg_start } else { s.words[..=t].iter().rposition(|x| *x == ".").map_or(0, |i| i + 1) };
                        let sem_out = semantic_reps.and(sem_cue_w).and_then(|cw| sem_read(&sem_store, &enc.codes, cw, slot_in(&bind_sentence_pairs, cw), sem_typed, roles.used(), true, &sem_frames, rel_reps.map(|_| (&rel, rel_hops)), &bind_sentence_pairs));
                        let fed = [
                            bind_raw.as_ref().filter(|_| !(testing && bind_lesion) && !hc_ec).and_then(|v| gate(v)).map(|v| (v, 0usize)),
                            sem_out.as_ref().and_then(|v| gate(v)).map(|v| (v, 1)),
                            td_src.as_ref().filter(|_| rollout_area).and_then(|v| gate(v)).map(|v| (v, 2)),
                            column.l23.peek(&input).as_ref().and_then(|v| gate(v)).map(|v| (v, 3)),
                        ]
                        .into_iter()
                        .flatten()
                        .fold(None, |acc: Option<(BitVector, usize)>, (v, src)| match acc {
                            None => Some((v, src)),
                            Some((mut a, s0)) if rollout_super && !(super_evidence && src == 3) => {
                                a.or_mut(&v);
                                Some((a, s0))
                            }
                            keep => keep,
                        });
                        let o = fed.as_ref().and_then(|(v, src)| enc.decode(v).map(|w| (w, *src)));
                        fed_vec = fed.map(|f| f.0);
                        o
                    } else {
                        offer
                    };
                    let own = offer.map(|o| o.0);
                    if std::env::var("COMPLETEDIAG").is_ok() && testing && s.words[t] == "a" && t >= 2 && NEW_NAMES.contains(&s.words[t - 2]) {
                        let wl = |v: &BitVector| -> Vec<&str> { (0..vocab.len()).filter(|&i| ov(i, v) >= 24).map(|i| vocab[i]).collect() };
                        eprintln!("  AREADIAG {} is a: area {:?} (conf {:.2}), column expects {:?}, column own {:?}", s.words[t - 2], td_src.as_ref().map(|v| wl(v)), area.column.confidence(), wl(&expect), column.l23.peek(&input).and_then(|o| enc.decode(&o)).map(|w| vocab[w]));
                    }
                    if std::env::var("COMPLETEDIAG").is_ok() && testing && NEW_NAMES.contains(&s.words[t]) {
                        eprintln!("  ROLLDIAG at {}: band {band}, skipped {skipped}, own {:?}, next {}", s.words[t], own.map(|w| vocab[w]), s.words[t + 1]);
                    }
                    let _ = band;
                    let offered = own.filter(|&w| skipped && w != ids[t + 1] && vocab[w] != "." && s.words[t + 1] != ".");
                    match (step_learned, offered, offer) {
                        // a started rollout runs on until the page fits again: the choice is
                        // whether to start one ("look again" or "read on")
                        (true, Some(w), Some(_)) if rolled > 0 => Some(w),
                        (true, Some(w), Some((_, src))) => {
                            let conf = column.confidence();
                            let cb = if conf < Q_HALF { 0 } else if conf < Q_08 { 1 } else { 2 };
                            let ctx = definite as usize + 2 * cb + 12 * (band < 4) as usize + 24 * src;
                            let cands = [step_code(ctx, 0), step_code(ctx, 1)];
                            let explore = if !testing || step_test_learn { Some(&mut bg_rng) } else { None };
                            let choice = step_bg.select(&cands, explore).unwrap_or(1);
                            if !testing || step_test_learn {
                                step_pending.push((cands[choice].clone(), choice == 0));
                            }
                            if testing {
                                step_stats[src][0] += 1;
                                step_stats[src][1] += (choice == 0) as usize;
                            }
                            (choice == 0).then_some(w)
                        }
                        _ => offered,
                    }
                } else {
                    None
                };
                if let Some(w) = rollout_w {
                    let k = if !testing { 0 } else if s.held_out { 2 } else { 1 };
                    complete_stats[k] += 1;
                    if k == 2 {
                        let start = if learned_bound { seg_start } else { s.words[..=t].iter().rposition(|x| *x == ".").map_or(0, |i| i + 1) };
                        *complete_words.entry(format!("{} -> {}", s.words[start..=t].join(" "), vocab[w])).or_default() += 1;
                    }
                    ids.insert(t + 1, w);
                    s.words.insert(t + 1, vocab[w]);
                    page_marks.insert(t + 1, false);
                    inner.insert(t + 1, true);
                    efference.insert(t + 1, None);
                    eff_pred.insert(t + 1, None);
                    let fv = fed_vec.take().filter(|_| rollout_loop);
                    if let (Some(v), true) = (fv.as_ref(), testing) {
                        loop_stats[0] += 1;
                        loop_stats[1] += (v.as_words() == enc.codes[w].as_words()) as usize;
                        loop_stats[2] += ((0..vocab.len()).filter(|&i| enc.codes[i].as_words().iter().zip(v.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24).count() > 1) as usize;
                    }
                    inner_code.insert(t + 1, fv);
                    if s.answer_at > t {
                        s.answer_at += 1;
                    }
                    rolled += 1;
                    if let (true, Some(i)) = (SURNAMES.contains(&vocab[w]), NEW_NAMES.iter().position(|n| s.words[..=t].contains(n))) {
                        rolled_surname = Some(vocab[w] == SURNAMES[family_of(i, true)]);
                    }
                }
                if let (false, Some(mode), Some(w), false, false) = (rollout || inner_speech, complete.as_deref(), bind_answer.filter(|_| !hc_ec), completed_sentence, testing && bind_lesion) {
                    let ov = |i: usize, v: &BitVector| enc.codes[i].as_words().iter().zip(v.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>();
                    let band = bind_sentence.iter().zip(&bind_sentence_pairs).map(|(b, &(w, c))| fam_binding(&bind_hc, &bind_mem, hippo_self, sparse_bind, b, &enc.codes[w], w, c)).min().map_or(7, |c| (64 - c.leading_zeros() as u64).min(7));
                    let start = if learned_bound { seg_start } else { s.words[..=t].iter().rposition(|x| *x == ".").map_or(0, |i| i + 1) };
                    let skipped = ov(ids[t + 1], &expect_prev) < 24;
                    let fresh = w != ids[t + 1] && !ids[start..=t].contains(&w) && s.words[t + 1] != ".";
                    if skipped && fresh && band < 4 && (testing || mode != "test") {
                        let k = if !testing { 0 } else if s.held_out { 2 } else { 1 };
                        complete_stats[k] += 1;
                        if k == 2 {
                            *complete_words.entry(format!("{} -> {}", s.words[t], vocab[w])).or_default() += 1;
                        }
                        ids.insert(t + 1, w);
                        s.words.insert(t + 1, vocab[w]);
                        page_marks.insert(t + 1, false);
                        inner.insert(t + 1, true);
                        efference.insert(t + 1, None);
                        eff_pred.insert(t + 1, None);
                        inner_code.insert(t + 1, None);
                        if s.answer_at > t {
                            s.answer_at += 1;
                        }
                        completed_sentence = true;
                    }
                }
                // HC_EC: the recall, decoded by CA1, returns as entorhinal feedback in its slot,
                // passed in proportion to the feedforward sweep's uncertainty
                let empty_raw = BitVector::new(BITS, Some(0));
                let raw_now = bind_raw.as_ref().filter(|_| !(testing && bind_lesion));
                let input = match (hc_ec, raw_now.or(query_ec.as_ref().map(|_| &empty_raw))) {
                    (true, Some(raw)) => {
                        let ov = |i: usize| enc.codes[i].as_words().iter().zip(raw.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>();
                        let mut cands: Vec<(u32, usize)> = (0..vocab.len()).map(|i| (ov(i), i)).filter(|x| x.0 >= 24).collect();
                        cands.sort_by(|a, b| b.cmp(a));
                        let mut ec = BitVector::new(BITS, Some(0));
                        for &(_, i) in cands.iter().take(3) {
                            ec.or_mut(&enc.codes[i]);
                        }
                        // QQUERY: what the held item's own query found, added to the feedback
                        if let Some((_, q)) = query_ec.as_ref().filter(|_| !(testing && bind_lesion) && !qarea) {
                            ec.or_mut(q);
                        }
                        if ec.count_ones() == 0 {
                            input
                        } else {
                            let conf = column.l23.peek_scored(&input).map_or(0, |(_, c)| c) as u64;
                            let mut share = (ONE as u64).saturating_sub(conf);
                            if ach_on {
                                share = (share * (ONE - ach.min(ONE)) as u64) >> 16;
                            }
                            let fw = BITS / 64;
                            let mut w = input.as_words().to_vec();
                            let at = w.len() - 2 * fw; // the slot before the previous input
                            for (wi, (d, x)) in w[at..at + fw].iter_mut().zip(ec.as_words()).enumerate() {
                                let mut keep = 0u64;
                                for b in 0..64 {
                                    let h = ((wi * 64 + b) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 48;
                                    if h < share {
                                        keep |= 1 << b;
                                    }
                                }
                                *d = x & keep;
                            }
                            // QQUERY_FULL: the held item's answer passes whole, not scaled by
                            // the sweep's uncertainty (a query asked on purpose, not a recall
                            // that competes with the cortex's own guess)
                            if let (true, Some((_, q))) = (qquery_full && !qarea, query_ec.as_ref().filter(|_| !(testing && bind_lesion))) {
                                for (d, x) in w[at..at + fw].iter_mut().zip(q.as_words()) {
                                    *d |= *x;
                                }
                            }
                            hc_ec_stats[0] += 1;
                            hc_ec_stats[1] += share;
                            hc_ec_stats[2] += testing as u64;
                            BitVector::from_words(w)
                        }
                    }
                    _ => input,
                };
                // HC_ROUTE: the recall, decoded by CA1, returns as entorhinal feedback
                let input = match (hc_route, bind_raw.as_ref().filter(|_| !(testing && bind_lesion)), route_cur.clone()) {
                    (true, Some(raw), Some((word, mut ch, slots))) => {
                        let ov = |i: usize| enc.codes[i].as_words().iter().zip(raw.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>();
                        let mut cands: Vec<(u32, usize)> = (0..vocab.len()).map(|i| (ov(i), i)).filter(|x| x.0 >= 24).collect();
                        cands.sort_by(|a, b| b.cmp(a));
                        let mut ec = BitVector::new(BITS, Some(0));
                        for &(_, i) in cands.iter().take(3) {
                            ec.or_mut(&enc.codes[i]);
                        }
                        if ec.count_ones() == 0 {
                            input
                        } else {
                            if !route_order.contains(&HC_CHANNEL) {
                                route_order.push(HC_CHANNEL);
                            }
                            ch.push((HC_CHANNEL, ec));
                            let key = ((if t > 0 { ids[t - 1] } else { vocab.len() }) * (vocab.len() + 1) + ids[t]) as u64 * 64;
                            let shares: Vec<u64> = ch.iter().map(|(c, _)| if route_noscale { ONE as u64 } else { (2 * mix.rate(ROUTE_SRC + *c as u8, key) as u64).min(ONE as u64) }).collect();
                            let row = route_row(&word, &ch, &shares, &route_order, slots, None);
                            if let Some(st) = route_step.as_mut() {
                                st.1 = ch.clone();
                            }
                            route_cur = Some((word, ch, slots));
                            hc_routed += 1;
                            BitVector::from_words(row)
                        }
                    }
                    _ => input,
                };
                // QQUERY with ROUTE: the held item's answer as a channel of its own
                let input = match (route_on && qquery && !qarea, query_ec.as_ref().filter(|_| !(testing && bind_lesion)), route_cur.clone()) {
                    (true, Some((_, q)), Some((word, mut ch, slots))) if q.count_ones() > 0 => {
                        if !route_order.contains(&Q_CHANNEL) {
                            route_order.push(Q_CHANNEL);
                        }
                        ch.push((Q_CHANNEL, q.clone()));
                        let key = ((if t > 0 { ids[t - 1] } else { vocab.len() }) * (vocab.len() + 1) + ids[t]) as u64 * 64;
                        let shares: Vec<u64> = ch.iter().map(|(c, _)| if route_noscale { ONE as u64 } else { (2 * mix.rate(ROUTE_SRC + *c as u8, key) as u64).min(ONE as u64) }).collect();
                        let row = route_row(&word, &ch, &shares, &route_order, slots, None);
                        if let Some(st) = route_step.as_mut() {
                            st.1 = ch.clone();
                        }
                        route_cur = Some((word, ch, slots));
                        BitVector::from_words(row)
                    }
                    _ => input,
                };
                // LEARNING=three: the cerebellum predicts from a copy of the cortex's input, and
                // its prediction returns through the thalamus as a channel
                let input = match cb_early.clone() {
                    Some(p) => {
                        match (route_on && p.count_ones() > 0, route_cur.take()) {
                            (true, Some((word, mut ch, slots))) => {
                                if !route_order.contains(&CB_CHANNEL) {
                                    route_order.push(CB_CHANNEL);
                                }
                                ch.push((CB_CHANNEL, p));
                                let key = ((if t > 0 { ids[t - 1] } else { vocab.len() }) * (vocab.len() + 1) + ids[t]) as u64 * 64;
                                let shares: Vec<u64> = ch.iter().map(|(c, _)| if route_noscale { ONE as u64 } else { (2 * mix.rate(ROUTE_SRC + *c as u8, key) as u64).min(ONE as u64) }).collect();
                                let row = route_row(&word, &ch, &shares, &route_order, slots, None);
                                if let Some(st) = route_step.as_mut() {
                                    st.1 = ch;
                                }
                                BitVector::from_words(row)
                            }
                            _ => input,
                        }
                    }
                    None => input,
                };
                route_cur = None;
                let mut out = BitVector::new(BITS, Some(0));
                out.or_mut(column.predict(&input));
                // the column's burst this step (its strength), for BURST_VOTE
                let mut col_burst: Option<Q16> = None;
                // the primed column's state this step: (burst?, priming), for BURST_KEY
                let mut col_state: Option<(bool, Q16)> = None;
                // L5=two: a burst of a two-compartment cell (input and context together)
                // overrides L2/3's habit; without a burst L2/3 speaks, and where it has
                // nothing to say the mix falls to the other sources (the cerebellum)
                if let Some(l) = layer5.as_mut() {
                    if let Some((o, c)) = l.predict(&input, BITS) {
                        out = o;
                        column.set_output(out.clone(), c);
                        col_burst = Some(c);
                    }
                }
                // L23=primed: L2/3 as two-compartment primed cells: whenever one fires (spike or
                // burst) it is the column's prediction; the old kernels speak only when none fires
                if let Some(l) = bit23.as_mut() {
                    match l.predict(&input, BITS) {
                        Some((o, c, burst)) => {
                            out = o;
                            column.set_output(out.clone(), c);
                            l23_stats[0] += 1;
                            col_state = Some((burst, c));
                            if burst {
                                col_burst = Some(c);
                            }
                        }
                        None => l23_stats[1] += 1,
                    }
                }
                if let Some(l) = primed23.as_mut() {
                    match l.predict(&input, BITS) {
                        Some((o, c, burst)) => {
                            out = o;
                            column.set_output(out.clone(), c);
                            l23_stats[0] += 1;
                            col_state = Some((burst, c));
                            if burst {
                                col_burst = Some(c);
                            }
                        }
                        None => l23_stats[1] += 1,
                    }
                }
                // L5=primed: the same, with priming (PrimedLayer5): a burst overrides L2/3
                // L5_OUT: layer 5 reads L2/3's prediction on its input side, and its output (spike or
                // burst) is all the thalamus receives from the column; silent layer 5, no output
                let row5: Option<BitVector> = l5_out.then(|| with_frame(&input, &out));
                if let Some(l) = bit5.as_mut() {
                    if let Some(r5) = row5.as_ref() {
                        match l.predict(r5, BITS) {
                            Some((o, c, burst)) => {
                                out = o;
                                column.set_output(out.clone(), c);
                                col_state = Some((burst, c));
                                if burst {
                                    col_burst = Some(c);
                                }
                            }
                            None if l5_relay => {} // L2/3's prediction passes through
                            None => {
                                out = BitVector::new(BITS, Some(0));
                                column.set_output(out.clone(), 0);
                            }
                        }
                    } else if let Some((o, c, true)) = l.predict(&input, BITS) {
                        out = o;
                        column.set_output(out.clone(), c);
                        col_burst = Some(c);
                        col_state = Some((true, c));
                    }
                }
                if let Some(l) = primed5.as_mut() {
                    if let Some((o, c, true)) = l.predict(&input, BITS) {
                        out = o;
                        column.set_output(out.clone(), c);
                        col_burst = Some(c);
                        col_state = Some((true, c));
                    }
                }
                // the slow cortex's prediction (SLOW_CORTEX)
                if let Some(sc) = slow.as_mut() {
                    let mut so = BitVector::new(BITS, Some(0));
                    sc.process_predictive(&input, &mut so);
                    slow_word = enc.decode(&so);
                    slow_conf = sc.confidence().unwrap_or(0);
                }
                // PROF: [1] L2/3 prediction
                prof[1] += prof_t.elapsed().as_secs_f64();
                prof_t = std::time::Instant::now();
                let next = ids[t + 1];
                if std::env::var("FASTDIAG").is_ok() && testing && t + 1 == s.answer_at && s_i < TRAIN + 8 {
                    eprintln!(
                        "  FASTDIAG answer step: {} kernels matched, {} distinct outputs, {} inhibited, winner reliability {:.2}, predicted {:?}, answer {}",
                        column.l23.matched(),
                        column.l23.matched_outputs(),
                        column.l23.inhibited(),
                        column.confidence(),
                        enc.decode(&out).map(|i| vocab[i]),
                        vocab[next]
                    );
                }
                // Books, learned context action: reward the choice made at "@open" by this
                // session's answer (training only)
                if task == Task::Books && ctx_pending && t + 1 == s.answer_at {
                    ctx_pending = false;
                    if !testing {
                        let right = enc.decode(&out) == Some(next);
                        ctx_bg.reward(if right { ONE as i32 } else { 0 }, &mut bg_rng);
                    }
                }
                // MIX: every source votes for its words with its reliability as the weight
                let mut mix_conf: Option<Q16> = None;
                let mut mix_agreed = false;
                // familiarity band of the current sentence (BIND_FAM), 7 = familiar / none
                let fam_band: u64 = if bind && bind_fam && mem_size(&bind_hc, &bind_mem, hippo_self) > 0 {
                    bind_sentence
                        .iter()
                        .zip(&bind_sentence_pairs)
                        .map(|(b, &(w, c))| fam_binding(&bind_hc, &bind_mem, hippo_self, sparse_bind, b, &enc.codes[w], w, c))
                        .min()
                        .map_or(7, |c| (64 - c.leading_zeros() as u64).min(7))
                } else {
                    0
                };
                if mixing {
                    let prev = if t > 0 { ids[t - 1] } else { vocab.len() };
                    let ctx = ((prev * (vocab.len() + 1) + ids[t]) as u64 * 8 + fam_band) * 8;
                    let bucket = |c: Q16| CONF_BANDS.iter().filter(|&&e| c >= e).count() as u64;
                    let words_of = |bv: &BitVector| -> Vec<usize> {
                        (0..vocab.len()).filter(|&i| enc.codes[i].as_words().iter().zip(bv.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24).collect()
                    };
                    // (source, key, proposed words): 0 column, 1 memory, 2 top-down
                    let mut proposals: Vec<(u8, u64, Vec<usize>)> = Vec::new();
                    let own = enc.decode(&out);
                    if std::env::var("OWNDIAG").is_ok() && testing && t + 1 == s.answer_at {
                        own_diag[0] += 1;
                        own_diag[1] += own.is_some() as usize;
                        own_diag[2] += (out.count_ones() > 0) as usize;
                        // which matched kernels propose the answer, how deep, how reliable
                        let mk = column.l23.matched_kernels(&input);
                        let place = |v: &BitVector| enc.decode(v).map_or(false, |w| PLACES.contains(&vocab[w]));
                        let n_place = mk.iter().filter(|k| place(&k.2)).count();
                        let n_right = mk.iter().filter(|k| enc.decode(&k.2) == Some(next)).count();
                        own_diag_k[0] += mk.len();
                        own_diag_k[1] += n_place;
                        own_diag_k[2] += n_right;
                        own_diag_k[3] += (n_right > 0) as usize;
                        if own_diag[0] <= 8 {
                            let best_right = mk.iter().filter(|k| enc.decode(&k.2) == Some(next)).map(|k| (k.0, k.1)).max();
                            let best_any = mk.iter().map(|k| (k.0, k.1, enc.decode(&k.2).map(|w| vocab[w]))).max();
                            eprintln!("  OWNDIAG matched {} kernels, {} propose a place, {} the answer; best for the answer (depth, rate) {:?}; best overall {:?}", mk.len(), n_place, n_right, best_right, best_any);
                            eprintln!("  OWNDIAG {:?}: column says {:?} ({} bits), answer {}", &s.words[if learned_bound { seg_start } else { s.words[..=t].iter().rposition(|x| *x == ".").map_or(0, |i| i + 1) }..=t], own.map(|w| vocab[w]), out.count_ones(), vocab[next]);
                        }
                    }
                    // the cortex's class: every word in its possible continuations
                    let class_words = |bv: &BitVector| -> Vec<usize> { (0..vocab.len()).filter(|&i| enc.codes[i].as_words().iter().zip(bv.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24).collect() };
                    let novel = class_vote && bind && fam_band < 4;
                    if novel {
                        let cw = class_words(&expect_prev);
                        if !cw.is_empty() {
                            proposals.push((7, ctx + bucket(column.confidence()), cw));
                        }
                    } else if let Some(w) = own {
                        // BURST_KEY: the column's reliability is learned per burst state (burst
                        // or spike × priming band), not per word pair
                        let key = match (burst_key, col_state) {
                            (true, Some((b, p))) => (1u64 << 61) | ((b as u64) << 4) | bucket(p),
                            (true, None) => 1u64 << 61 | 1 << 5,
                            _ => ctx + bucket(column.confidence()),
                        };
                        proposals.push((0, key, vec![w]));
                    }
                    if let (Some(w), true) = (cb_word, cerebellum.is_some()) {
                        if std::env::var("CBDIAG").is_ok() && testing && t + 1 == s.answer_at && cb_stats[0] < 8 {
                            eprintln!("  CBDIAG {:?}: cerebellum says {}, answer {}, own {:?}", &s.words[..=t], vocab[w], vocab[next], own.map(|o| vocab[o]));
                        }
                        proposals.push((CB_SRC, ctx + bucket(cb_conf), vec![w]));
                        if testing && t + 1 == s.answer_at {
                            cb_stats[0] += 1;
                            cb_stats[1] += (w == next) as usize;
                        }
                    }
                    if let Some(w) = assoc_word.take() {
                        proposals.push((ASSOC_SRC, ctx + bucket(assoc_conf), vec![w]));
                        if testing && t + 1 == s.answer_at {
                            assoc_stats[0] += 1;
                            assoc_stats[1] += (w == next) as usize;
                        }
                    }
                    if let (Some(w), true) = (slow_word, slow.is_some()) {
                        proposals.push((SLOW_SRC, ctx + bucket(slow_conf), vec![w]));
                        if testing && t + 1 == s.answer_at {
                            slow_stats[0] += 1;
                            slow_stats[1] += (w == next) as usize;
                        }
                    }
                    let mut mw = words_of(&mem_src);
                    if sparse_hc && own_conf_step >= Q_HALF && !mw.is_empty() {
                        // sparse in time: the column is sure here, the hippocampus's answer is not sent
                        mw.clear();
                        hc_withheld += (testing && !replaying) as usize;
                    }
                    if !mw.is_empty() {
                        // GRADED: the hippocampus's answer is weighed by the cortex's uncertainty
                        // (the column's confidence band), learned per context
                        // (keys of their own: a context has room for only 8 offsets)
                        let key = if graded { (1u64 << 62) | (ctx << 4) | (mw.len().min(3) as u64) << 2 | bucket(column.confidence()).min(3) } else { ctx + mw.len().min(3) as u64 };
                        proposals.push((1, key, mw));
                    }
                    // GRADED: the semantic store always answers too, through the cortex: the
                    // higher area's prediction with the store's content for the sentence's least
                    // familiar word added to its sentence context (peeked: no side effects),
                    // weighed by the column's confidence band
                    if let (true, Some(hin), Some(cue_w)) = (graded, hier_in.as_ref(), sem_cue_w) {
                        if let Some(o) = sem_read(&sem_store, &enc.codes, cue_w, slot_in(&bind_sentence_pairs, cue_w), sem_typed, roles.used(), false, &sem_frames, rel_reps.map(|_| (&rel, rel_hops)), &bind_sentence_pairs) {
                            let mut x = hin.as_words().to_vec();
                            for (w, &b) in x.iter_mut().zip(o.as_words()) {
                                *w |= b;
                            }
                            if let Some(p) = area.column.l23.peek(&BitVector::from_words(x)) {
                                let sw = words_of(&p);
                                if !sw.is_empty() {
                                    proposals.push((9, (ctx << 2) | bucket(column.confidence()).min(3), sw));
                                    graded_stats[0] += 1;
                                }
                            }
                        }
                    }
                    if let (Some(td), false) = (td_src.as_ref(), novel) {
                        let tw = words_of(td);
                        if !tw.is_empty() {
                            proposals.push((2, ctx + bucket(area.column.confidence()), tw));
                        }
                    }
                    // the semantic store (SEMANTIC_MIX)
                    if let (true, Some(_)) = (semantic_mix, semantic_reps) {
                        if let Some(cue_w) = sem_cue_w {
                            if let Some(o) = sem_read(&sem_store, &enc.codes, cue_w, slot_in(&bind_sentence_pairs, cue_w), sem_typed, roles.used(), true, &sem_frames, rel_reps.map(|_| (&rel, rel_hops)), &bind_sentence_pairs) {
                                let ov = |i: usize, v: &BitVector| enc.codes[i].as_words().iter().zip(v.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>();
                                // of the kind the column expects here (read now: `expect_prev` is
                                // only kept while the hippocampus is intact)
                                let expect = column.l23.peek_union(&input, BITS);
                                // a question that names a relation the store holds for the word
                                // ("tom is a": family) asks for that relation's answer, of its
                                // kind by construction, whatever the column expects
                                let named = rel_reps.is_some() && {
                                    let q: Vec<usize> = bind_sentence_pairs.iter().map(|x| x.0).filter(|&x| x != cue_w).collect();
                                    rel.relation_for(&q).is_some_and(|r| rel.about(&enc.codes, cue_w).iter().any(|a| a.0 == r))
                                };
                                if let Some(w) = (0..vocab.len()).filter(|&i| ov(i, &o) >= 24 && (named || ov(i, &expect) >= 24)).max_by_key(|&i| ov(i, &o)) {
                                    proposals.push((8, ctx, vec![w]));
                                }
                            }
                        }
                    }
                    // slot ⊗ content memory (BIND)
                    if let Some(w) = bind_answer {
                        if sparse_hc && own_conf_step >= Q_HALF {
                            // sparse in time: the column is sure here, the hippocampus's answer is not sent
                            hc_withheld += (testing && !replaying) as usize;
                        } else {
                            if !hc_route && !hc_ec {
                                proposals.push((6, ctx + bind_strength, vec![w]));
                            }
                        }
                    }
                    // the upper areas of the chain, one source each (the bud, HIER_GROW, apart)
                    let mut bud_prop: Option<(u8, u64, Vec<usize>)> = None;
                    for (i, p) in upper_pred.iter().enumerate() {
                        if let Some(p) = p {
                            let uw = words_of(p);
                            if !uw.is_empty() {
                                let prop = (upper_src(i), ctx + bucket(upper[i].column.confidence()), uw);
                                if bud_live && i + 1 == upper.len() {
                                    bud_prop = Some(prop);
                                } else {
                                    proposals.push(prop);
                                }
                            }
                        }
                    }
                    // BURST_VOTE: a bursting column votes with the burst's own evidence (its
                    // strength now), not the word-keyed reliability learned for the old column
                    let burst_weight = |src: u8, key: u64| -> u32 {
                        match (burst_vote, src, col_burst) {
                            (true, 0, Some(c)) => mix.weight_of_rate(c),
                            _ => mix.weight(src, key),
                        }
                    };
                    let votes: Vec<(usize, u32)> = if burst_gate {
                        let beta = |src: u8| src_burst.get(&src).copied().unwrap_or(ONE / 2);
                        let mut pass: Vec<&(u8, u64, Vec<usize>)> = proposals.iter().filter(|p| beta(p.0) >= gate_theta).collect();
                        let at_answer = testing && t + 1 == s.answer_at;
                        gate_stats[0] += at_answer as usize;
                        if pass.is_empty() {
                            gate_theta -= gate_theta / 16;
                            pass = proposals.iter().filter(|p| p.0 == CB_SRC).collect();
                            if pass.is_empty() {
                                pass = proposals.iter().collect();
                                gate_stats[3] += at_answer as usize;
                            } else {
                                gate_stats[2] += at_answer as usize;
                            }
                        } else {
                            gate_theta += (ONE - gate_theta.min(ONE)) / 64;
                            gate_stats[1] += at_answer as usize;
                        }
                        pass.iter().flat_map(|(src, _, ws)| ws.iter().map(|&w| (w, mix.weight_of_rate(beta(*src)))).collect::<Vec<_>>()).collect()
                    } else {
                        proposals.iter().flat_map(|(src, key, ws)| ws.iter().map(|&w| (w, burst_weight(*src, *key))).collect::<Vec<_>>()).collect()
                    };
                    // the bud in shadow: would its vote have fixed or broken the mix's choice?
                    if let Some((src, key, ws)) = bud_prop {
                        if !testing && !inner[t + 1] && !reciting {
                            let without = SourceMix::combine(&votes).map(|(w, _)| w).or(own);
                            let mut with_votes = votes.clone();
                            with_votes.extend(ws.iter().map(|&w| (w, mix.weight(src, key))));
                            let with = SourceMix::combine(&with_votes).map(|(w, _)| w);
                            let n = if t + 1 == s.answer_at { grow_answer } else { 1 };
                            match (without == Some(next), with == Some(next)) {
                                (false, true) => bud_tally.0 += n,
                                (true, false) => bud_tally.1 += n,
                                _ => {}
                            }
                            for &w in &ws {
                                mix.record(src, key, w == next);
                            }
                        }
                    }
                    if std::env::var("BQDIAG").is_ok() && testing && t + 1 == s.answer_at && s.held_out && s.words[t] == "a" {
                        mix_dbg = format!("cue {:?} ", sem_cue_w.map(|w| vocab[w])) + &proposals.iter().map(|(src, key, ws)| format!("{src}:{:?}@{:.2}", ws.iter().map(|&w| vocab[w]).collect::<Vec<_>>(), to_f32(mix.weight(*src, *key)))).collect::<Vec<_>>().join(" ");
                    }
                    if let Some((w, total)) = SourceMix::combine(&votes) {
                        if testing && t + 1 == s.answer_at {
                            let agreed = proposals.len() > 1 && proposals.iter().all(|p| p.2 == vec![w]);
                            mix_stats[0] += 1;
                            mix_stats[1] += (own == Some(next)) as usize;
                            mix_stats[2] += (w == next) as usize;
                            mix_stats[3] += (own != Some(w)) as usize;
                            mix_stats[4] += (own != Some(w) && w == next) as usize;
                            mix_stats[5] += agreed as usize;
                            mix_stats[6] += (agreed && w == next) as usize;
                        }
                        out = enc.codes[w].clone();
                        mix_conf = Some(SourceMix::confidence(total));
                        if three {
                            column.set_output(out.clone(), SourceMix::confidence(total));
                        }
                        // every source supports the chosen word (a source may offer several)
                        mix_agreed = proposals.len() > 1 && proposals.iter().all(|p| p.2.contains(&w));
                    }
                    // bursts: where the sources disagreed, each source's trace moves toward
                    // whether it was confirmed (activity, not learning: at test too)
                    if burst_gate && !inner[t + 1] && !reciting && !replaying {
                        let hits: Vec<bool> = proposals.iter().map(|p| p.2.contains(&next)).collect();
                        // GATE_ON=surprise: only where the column's own prediction missed, the
                        // moments the gate has to decide
                        let moment = !gate_on_surprise || own != Some(next);
                        if moment && hits.iter().any(|&h| h) && hits.iter().any(|&h| !h) {
                            for (p, &h) in proposals.iter().zip(&hits) {
                                let b = src_burst.entry(p.0).or_insert(ONE / 2);
                                *b = if h { *b + (ONE - (*b).min(ONE)) / 4 } else { *b - *b / 4 };
                            }
                        }
                    }
                    // (not on an internal step: its "next word" is the network's own)
                    if (!testing || mix_test_learn) && !inner[t + 1] && !reciting {
                        for (src, key, ws) in &proposals {
                            for &w in ws {
                                mix.record(*src, *key, w == next);
                            }
                        }
                        // the frame gate's record (HIER_TRUST_GATE): the same outcomes, under a
                        // context known before the frame is placed (no familiarity band)
                        if hier_trust_gate.is_some() {
                            let key = (prev * (vocab.len() + 1) + ids[t]) as u64 * 64;
                            if let Some(td) = td_src.as_ref() {
                                let tw = words_of(td);
                                if !tw.is_empty() {
                                    mix.record(TRUST_AREA, key + bucket(area.column.confidence()), tw.contains(&next));
                                }
                            }
                            if let Some(w) = own {
                                mix.record(TRUST_COLUMN, key + bucket(column.confidence()), w == next);
                            }
                        }
                    }
                }
                if testing && bind && t + 1 == s.answer_at {
                    if let Some(w) = bind_answer {
                        bind_stats.0 += 1;
                        bind_stats.1 += (w == next) as usize;
                        if s.held_out {
                            bind_new.0 += 1;
                            bind_new.1 += (w == next) as usize;
                        }
                    }
                }
                if testing {
                    if let Some(c) = role_now.take() {
                        *role_words.entry(c).or_default().entry(next).or_default() += 1;
                    }
                }
                // inner speech: surprised by the page, the network says its prediction to
                // itself and hears it before reading on
                if inner_speech && !replaying && !reciting && rolled < 4 && (testing || complete.as_deref() == Some("speech")) && t + 1 != s.answer_at && s.words[t + 1] != "." {
                    // INNER_WHEN=definite: only where the column's own expectation holds one
                    // word (as the rollout's trigger), or while already speaking
                    let ex = column.l23.peek_union(&input, BITS);
                    let fits = |w: usize| enc.codes[w].as_words().iter().zip(ex.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24;
                    let definite = !inner_when_definite || rolled > 0 || (0..vocab.len()).filter(|&i| fits(i)).count() == 1;
                    // INNER_SAY=recall: recall plans the utterance. The slot memory's word is
                    // said if it is of the kind the column expects here, else the prediction
                    // (the hippocampus's slot memory first, else the semantic store's word)
                    let recalled = bind_answer.filter(|&w| inner_say_recall && !hc_ec && !(testing && bind_lesion) && fits(w)).or_else(|| {
                        if !inner_say_recall {
                            return None;
                        }
                        semantic_reps?;
                        let cw = sem_cue_w?;
                        let o = sem_read(&sem_store, &enc.codes, cw, slot_in(&bind_sentence_pairs, cw), sem_typed, roles.used(), true, &sem_frames, rel_reps.map(|_| (&rel, rel_hops)), &bind_sentence_pairs)?;
                        let ovo = |i: usize| enc.codes[i].as_words().iter().zip(o.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>();
                        (0..vocab.len()).filter(|&i| ovo(i) >= 24 && fits(i)).max_by_key(|&i| ovo(i))
                    });
                    let said_vec = recalled.map(|w| enc.codes[w].clone()).unwrap_or_else(|| out.clone());
                    // what is said must be of the kind the column expects here (the expectation
                    // gates the loop, as the thalamic gate did the rollout's offers)
                    let said = enc.decode(&said_vec).filter(|&w| definite && fits(w) && w != next && vocab[w] != ".");
                    if std::env::var("INNERDIAG").is_ok() && testing && s.held_out && NEW_NAMES.iter().any(|n| s.words[..=t].contains(n)) && inner_diag < 40 {
                        inner_diag += 1;
                        let start = if learned_bound { seg_start } else { s.words[..=t].iter().rposition(|x| *x == ".").map_or(0, |i| i + 1) };
                        eprintln!("  INNERDIAG {:?} | page {} | out {:?} recalled {:?} definite {definite} rolled {rolled} -> said {:?}", &s.words[start..=t], vocab[next], enc.decode(&out).map(|w| vocab[w]), bind_answer.map(|w| vocab[w]), said.map(|w| vocab[w]));
                    }
                    if let Some(w) = said {
                        let go = if inner_learned {
                            let c = mix_conf.unwrap_or_else(|| column.confidence());
                            let cb = if c < Q_HALF { 0 } else if c < Q_08 { 1 } else { 2 };
                            let ctx = 1000 + cb + 3 * (rolled > 0) as usize + 6 * (fam_band < 4) as usize;
                            let cands = [step_code(ctx, 0), step_code(ctx, 1)];
                            let explore = if !testing || step_test_learn { Some(&mut bg_rng) } else { None };
                            let choice = step_bg.select(&cands, explore).unwrap_or(1);
                            if !testing || step_test_learn {
                                step_pending.push((cands[choice].clone(), choice == 0));
                            }
                            choice == 0
                        } else {
                            true
                        };
                        if testing {
                            inner_stats[0] += 1;
                        }
                        if go {
                            phono.say(&said_vec);
                            let heard = phono.hear();
                            if testing {
                                inner_stats[1] += 1;
                                inner_stats[2] += s.held_out as usize;
                                let k = if s.held_out { 2 } else { 1 };
                                complete_stats[k] += 1;
                                if k == 2 {
                                    let start = if learned_bound { seg_start } else { s.words[..=t].iter().rposition(|x| *x == ".").map_or(0, |i| i + 1) };
                                    *complete_words.entry(format!("{} -> {}", s.words[start..=t].join(" "), vocab[w])).or_default() += 1;
                                }
                            } else {
                                complete_stats[0] += 1;
                            }
                            ids.insert(t + 1, w);
                            s.words.insert(t + 1, vocab[w]);
                            page_marks.insert(t + 1, false);
                            inner.insert(t + 1, true);
                            efference.insert(t + 1, Some(w));
                            eff_pred.insert(t + 1, None);
                            inner_code.insert(t + 1, heard);
                            if s.answer_at > t {
                                s.answer_at += 1;
                            }
                            rolled += 1;
                            if let (true, Some(i)) = (SURNAMES.contains(&vocab[w]), NEW_NAMES.iter().position(|n| s.words[..=t].contains(n))) {
                                rolled_surname = Some(vocab[w] == SURNAMES[family_of(i, true)]);
                            }
                        }
                    }
                }
                if qhold.is_some() && testing && s.held_out && t + 1 == s.answer_at {
                    if let Some((hw, _)) = held_item {
                        *held_words.entry(vocab[hw]).or_default() += 1;
                        hold_stats[0] += 1;
                        hold_stats[1] += NEW_NAMES.contains(&vocab[hw]) as usize;
                    }
                }
                // QUESTION: what a held-out answer got wrong: the family, the season, or both
                if question() && testing && s.held_out && t + 1 == s.answer_at {
                    let season = SEASONS.iter().position(|x| *x == s.words[0]);
                    let fam = s.words.iter().position(|w| *w == "person").and_then(|p| s.words.get(p + 3)).and_then(|w| SURNAMES.iter().position(|x| x == w));
                    let pred = enc.decode(&out).and_then(|w| PLACES.iter().position(|p| *p == vocab[w]));
                    if let (Some(se), Some(f), Some(pl)) = (season, fam, pred) {
                        let k = if pl == family_place(f, se) {
                            0
                        } else if (0..4).any(|o| o != se && family_place(f, o) == pl) {
                            1 // a place of the right family, another season
                        } else if family_place(1 - f, se) == pl {
                            2 // the other family's place for this season
                        } else {
                            3
                        };
                        q_err[k] += 1;
                    } else {
                        q_err[4] += 1;
                    }
                }
                // NE, tonic: the running error rate of training answers sets exploration
                if ne_on && t + 1 == s.answer_at && !testing && !replaying {
                    let wrong = enc.decode(&out) != Some(next);
                    ne_tonic = ((63 * ne_tonic as u64 + if wrong { ONE as u64 } else { 0 }) / 64) as Q16;
                    let explore = ONE / 10 + ((ne_tonic as u64 * 4 / 10) as Q16).min(ONE * 4 / 10);
                    step_bg.explore = explore;
                    ne_stats[1] += 1;
                    ne_stats[2] += explore as u64;
                }
                if (qattach_learned || qhold_learned) && t + 1 == s.answer_at {
                    let right = enc.decode(&out) == Some(next);
                    let outcome = if right { ONE as i32 } else { -(ONE as i32) };
                    // the held words whose stored events served this answer
                    let used_items: Vec<usize> = attach_pending.iter().filter(|x| x.1 && x.2.map_or(false, |r| answer_rows.contains(&r))).map(|x| x.3).collect();
                    for (code, took) in hold_pending.drain(..) {
                        if !testing {
                            let r = match took {
                                Some(w) if used_items.contains(&w) => outcome - step_cost,
                                Some(_) => -step_cost,
                                None => 0,
                            };
                            step_bg.reward_candidate(&code, r, &mut bg_rng);
                        }
                    }
                    for (code, acted, row, _) in attach_pending.drain(..) {
                        let used = acted && row.map_or(false, |r| answer_rows.contains(&r));
                        if testing && s.held_out && used {
                            hold_stats[3] += 1;
                        }
                        if !testing {
                            let r = if used { if right { ONE as i32 } else { -(ONE as i32) } } else { 0 } - if acted { step_cost } else { 0 };
                            step_bg.reward_candidate(&code, r, &mut bg_rng);
                        }
                    }
                }
                if (step_learned || inner_learned || (qhold_learned && hold_credit_story)) && t + 1 == s.answer_at {
                    let right = enc.decode(&out) == Some(next);
                    for (code, stepped) in step_pending.drain(..) {
                        step_bg.reward_candidate(&code, if right { ONE as i32 } else { 0 } - if stepped { step_cost } else { 0 }, &mut bg_rng);
                    }
                }
                if bind && consolidate.is_some() && !testing && !replaying && t + 1 == s.answer_at {
                    traces.push((sentence.clone(), bind_list.clone(), next, fam_band));
                    if sleep_p.is_some() {
                        trace_rows.push(input.clone());
                    }
                    if da_on {
                        let nov = match bind_hc.as_ref() {
                            Some(hc) if !bind_sentence_pairs.is_empty() => {
                                let cue: Vec<usize> = bind_sentence_pairs.iter().flat_map(|&(w, c)| sparse_binding(&enc.codes[w], w, c, false)).collect();
                                hc.novelty(&cue)
                            }
                            _ => if fam_band < 4 { ONE } else { 0 },
                        };
                        let reward = if enc.decode(&out) == Some(next) { ONE / 4 } else { 0 };
                        let da = (nov / 2 + reward + if nov > 0 { ONE / 4 } else { 0 }).min(ONE);
                        trace_da.push(da);
                        da_stats[0] += 1;
                        da_stats[1] += da as u64;
                    }
                } else if bind && consolidate_steps && !testing && !replaying && fam_band < 4 && enc.decode(&out) != Some(next) && s.words[t + 1] != "." {
                    let input = if consolidate_assoc {
                        let start = if learned_bound { seg_start } else { s.words[..=t].iter().rposition(|x| *x == ".").map_or(0, |i| i + 1) };
                        let mut bag = BitVector::new(BITS, Some(0));
                        for &w in &ids[start..=t] {
                            if sentence_count > 50 && (word_count[w] as u64) * 100 < sentence_count as u64 {
                                bag.or_mut(&enc.codes[w]);
                            }
                        }
                        bag
                    } else {
                        sentence.clone()
                    };
                    if sleep_p.is_some() {
                        trace_rows.push(BitVector::new(BITS, Some(0))); // (no row: not replayed to the column)
                    }
                    if da_on {
                        trace_da.push(if fam_band < 4 { ONE } else { 0 });
                    }
                    traces.push((input, bind_list.clone(), next, fam_band));
                }
                // saccade reward: the prediction right, minus the cost of a regression
                if let Some(a) = sacc_pending.take() {
                    if !testing {
                        let right = enc.decode(&out) == Some(next);
                        let r = (if right { ONE as i32 } else { 0 } - if a > 0 { saccade_cost } else { 0 }).max(0);
                        sacc_bg.reward(r, &mut bg_rng);
                    }
                }
                if testing && t + 1 == s.answer_at {
                    let right = enc.decode(&out) == Some(next);
                    if std::env::var("BQDIAG").is_ok() && s.held_out && s.words[t] == "a" && bq_diag < 20 {
                        bq_diag += 1;
                        eprintln!("  BQDIAG {} is a {}: said {:?}; mix {}", s.words[t - 2], vocab[next], enc.decode(&out).map(|w| vocab[w]), std::mem::take(&mut mix_dbg));
                    }
                    if s.held_out && !right && std::env::var("DIAG").is_ok() {
                        // is there a copy kernel for the answer, and how close did it come?
                        let frame = BITS / 64;
                        let answer = &enc.codes[next];
                        let mut best: Option<(f32, [usize; 3], f32, u32)> = None;
                        for k in column.l23.kernels() {
                            let out_ok = k.output_set.iter().filter(|&&b| answer.bit_get(b as usize)).count() * 2 >= k.output_set.len();
                            if !out_ok {
                                continue;
                            }
                            let mut missing = [0usize; 3];
                            let mut reads_memory = false;
                            let mut matched = 0usize;
                            for (wi, m) in k.input_words(k.input_set.last().map_or(0, |&b| b as usize / 64 + 1)).into_iter().enumerate() {
                                let w = k.input_idx + wi;
                                let f = w / frame;
                                let slot = if f == 0 { 0 } else if f <= mid_frames { 1 } else { 2 };
                                if slot == 1 && m != 0 {
                                    reads_memory = true;
                                }
                                let on = input.as_words().get(w).copied().unwrap_or(0) & m;
                                matched += on.count_ones() as usize;
                                missing[slot] += (m & !on).count_ones() as usize;
                            }
                            if !reads_memory {
                                continue;
                            }
                            let ratio = matched as f32 / k.threshold.max(1) as f32;
                            let rel = (k.stats.hits as f32 + 1.0) / (k.stats.hits as f32 + k.stats.misses as f32 + 2.0);
                            // prefer matching kernels, then the most reliable
                            let better = best.map_or(true, |(r, _, br, _)| {
                                let (m, bm) = (ratio >= 1.0, r >= 1.0);
                                (m, rel, ratio) > (bm, br, r)
                            });
                            if better {
                                best = Some((ratio, missing, rel, k.stats.hits as u32 + k.stats.misses as u32));
                            }
                        }
                        cover[0] += 1;
                        if let Some((r, miss, rel, uses)) = best {
                            cover[1] += 1;
                            cover[2] += (r * 100.0) as usize;
                            for i in 0..3 {
                                cover[3 + i] += miss[i];
                            }
                            cover[6] += (rel * 100.0) as usize;
                            cover[7] += uses as usize;
                        }
                    }
                    if s.held_out && std::env::var("DIAG").is_ok() {
                        let kd = &mut kdiag[right as usize];
                        kd[6] += 1;
                        match column.l23.winner() {
                            None => kd[0] += 1,
                            Some(k) => {
                                // which input frames the kernel's mask covers
                                let frame = BITS / 64;
                                let mut per = [0usize; 3]; // current word, memory frames, previous word
                                for (wi, m) in k.input_words(k.input_set.last().map_or(0, |&b| b as usize / 64 + 1)).into_iter().enumerate() {
                                    let f = (k.input_idx + wi) / frame;
                                    let slot = if f == 0 { 0 } else if f <= mid_frames { 1 } else { 2 };
                                    per[slot] += m.count_ones() as usize;
                                }
                                kd[1] += (per[1] > 0) as usize;
                                for i in 0..3 {
                                    kd[2 + i] += per[i];
                                }
                                kd[5] += ((1000 * column.l23.confidence().unwrap_or(0) as u64) >> 16) as usize;
                            }
                        }
                    }
                    if s.held_out {
                        let d = &mut diag[right as usize];
                        for i in 0..3 {
                            d[i] += pending_diag[i];
                        }
                        d[3] += 1;
                    }
                    if s.held_out {
                        let k = match rolled_surname {
                            None => 0,
                            Some(true) => 1,
                            Some(false) => 2,
                        };
                        by_surname[k].0 += right as usize;
                        by_surname[k].1 += 1;
                    }
                    let r = if s.held_out { &mut held } else { &mut seen };
                    r.0 += right as usize;
                    r.1 += 1;
                    if s.held_out && belief_q() && s.words[s.answer_at - 1] == "a" {
                        if let Some(i) = NEW_NAMES.iter().position(|n| *n == s.words[s.answer_at - 3]) {
                            let w = index[NEW_NAMES[i]];
                            let isa = rel.relation_for(&[index["is"], index["a"]]);
                            let belief = isa.and_then(|r| rel.bayes.believed(&(r, w, 0, 1)).map(|v| rel.belief(w, r, 0, 1, v)));
                            let sure = match (unknown_learned, belief) {
                                (_, None) => false,
                                (true, Some(_)) => decisiveness(&rel, w, isa.unwrap()).is_some_and(|(_, c)| unknown_bg.value(&unknown_code(c, true)) > unknown_bg.value(&unknown_code(c, false))),
                                (false, Some(b)) => b > ONE / 2,
                            };
                            let u = &mut unknown_tally[i];
                            u[0] += 1;
                            if belief_unknown && !sure {
                                u[3] += 1;
                            } else if right {
                                u[1] += 1;
                            } else {
                                u[2] += 1;
                            }
                        }
                    }
                    if s.held_out && belief_q() {
                        let fam_q = s.words[s.answer_at - 1] == "a";
                        let q = &mut belief_tally[fam_q as usize];
                        q.0 += right as usize;
                        q.1 += 1;
                        // place questions per new name (the name stands right before "went")
                        if let (false, Some(i)) = (fam_q, NEW_NAMES.iter().position(|n| s.words[..s.answer_at].contains(n))) {
                            belief_names[i].0 += right as usize;
                            belief_names[i].1 += 1;
                        }
                    }
                    if s.held_out {
                        let h = &mut held_halves[(s_i - TRAIN >= TEST / 2) as usize];
                        h.0 += right as usize;
                        h.1 += 1;
                    }
                    if task == Task::Season && saccade.is_some() {
                        sacc_wording[(s.held_out && new_wording()) as usize].2 += right as usize;
                    }
                    if task == Task::Books {
                        book_bins[book_bin].0 += 1;
                        book_bins[book_bin].1 += right as usize;
                    }
                    if task == Task::Season {
                        let b = match season_distance {
                            0 => 0,
                            1 => 1,
                            2..=3 => 2,
                            4..=7 => 3,
                            8..=15 => 4,
                            _ => 5,
                        };
                        season_bins[b].0 += 1;
                        season_bins[b].1 += right as usize;
                    }
                    let c = mix_conf.unwrap_or_else(|| column.confidence());
                    let b = CONF_BANDS.iter().filter(|&&e| c >= e).count();
                    calib[b].0 += 1;
                    calib[b].1 += right as usize;
                    calib[b].2 += to_f32(c) as f64; // report
                    if speak && !replaying {
                        // the cortex speaks its answer into the output buffer, and hears it
                        // in place of the page's word
                        let said = if motor_speech {
                            // the basal ganglia release speech or hold it; the motor area says it
                            let cb = CONF_BANDS.iter().filter(|&&e| c >= e).count();
                            let nov = if cortex_fam_on {
                                exposure_band(&word_count, &ids, if learned_bound { seg_start } else { s.words[..=t].iter().rposition(|x| *x == ".").map_or(0, |i| i + 1) }, t)
                            } else if kernel_fam_on {
                                sentence_familiarity(&cortex_fam, &ids, if learned_bound { seg_start } else { s.words[..=t].iter().rposition(|x| *x == ".").map_or(0, |i| i + 1) }, t)
                            } else {
                                fam_band
                            };
                            let sc = speak_ctx(cb, nov, mix_agreed);
                            let cands = [speak_code(sc, 0), speak_code(sc, 1)];
                            let go = speak_bg.select(&cands, None::<&mut StdRng>) == Some(1);
                            sp_stats[cb][0] += 1;
                            let nb = (nov.min(7) / 2) as usize;
                            sp_agree[mix_agreed as usize][0] += 1;
                            sp_novel[nb][0] += 1;
                            let word = if go {
                                let (w, p) = motor_say(&motor, &tract, &out);
                                let r = (w == Some(next)) as usize;
                                sp_stats[cb][1] += 1;
                                sp_stats[cb][2] += r;
                                sp_agree[mix_agreed as usize][1] += 1;
                                sp_agree[mix_agreed as usize][2] += r;
                                sp_novel[nb][1] += 1;
                                sp_novel[nb][2] += r;
                                eff_pred[t + 1] = p;
                                w
                            } else {
                                None
                            };
                            speech.speak(word, c, Some(next), s.held_out as u8)
                        } else {
                            speech.speak(enc.decode(&out), c, Some(next), s.held_out as u8)
                        };
                        if s.held_out && speech.score(Some(1)).2 % 80 == 1 {
                            let q_start = if learned_bound { seg_start } else { s.words[..=t].iter().rposition(|x| *x == ".").map_or(0, |i| i + 1) };
                            transcripts.push(format!("{:?} -> said {:?} (page: {})", &s.words[q_start..=t], said.map_or("unknown", |w| vocab[w]), vocab[next]));
                        }
                        let gap = std::env::var("SILENCE").map_or(false, |v| v == "gap");
                        story_altered = said.is_some() || gap;
                        match said {
                            // silence (SILENCE unset): the network says nothing and hears the
                            // answer from the page, as the other speaker gives it in a dialogue
                            None if !gap => {}
                            Some(heard) => {
                                ids[t + 1] = heard;
                                s.words[t + 1] = vocab[heard];
                                inner[t + 1] = true;
                                efference[t + 1] = Some(heard);
                            }
                            None => {
                                // SILENCE=gap: silence is no word; the answer's position is
                                // taken out of the stream, and the story goes on
                                ids.remove(t + 1);
                                s.words.remove(t + 1);
                                page_marks.remove(t + 1);
                                inner.remove(t + 1);
                                inner_code.remove(t + 1);
                                efference.remove(t + 1);
                                eff_pred.remove(t + 1);
                            }
                        }
                    }
                }
                // SPEECH=motor, training: at a question the basal ganglia choose to speak or not;
                // the page's answer follows and rewards the choice
                if motor_speech && !testing && !replaying && t + 1 == s.answer_at {
                    let c = mix_conf.unwrap_or_else(|| column.confidence());
                    let cb = CONF_BANDS.iter().filter(|&&e| c >= e).count();
                    let nov = if cortex_fam_on {
                        exposure_band(&word_count, &ids, if learned_bound { seg_start } else { s.words[..=t].iter().rposition(|x| *x == ".").map_or(0, |i| i + 1) }, t)
                    } else if kernel_fam_on {
                        sentence_familiarity(&cortex_fam, &ids, if learned_bound { seg_start } else { s.words[..=t].iter().rposition(|x| *x == ".").map_or(0, |i| i + 1) }, t)
                    } else {
                        fam_band
                    };
                    let sc = speak_ctx(cb, nov, mix_agreed);
                    let cands = [speak_code(sc, 0), speak_code(sc, 1)];
                    let a = speak_bg.select(&cands, Some(&mut speak_rng)).unwrap_or(0);
                    sp_train[0] += 1;
                    let r = if a == 1 {
                        sp_train[1] += 1;
                        if motor_say(&motor, &tract, &out).0 == Some(next) { ONE as i32 } else { -speak_penalty }
                    } else {
                        0
                    };
                    speak_bg.reward_candidate(&cands[a], r, &mut speak_rng);
                }
                // RECITE: past the cue, the network says the next word and hears it
                if let (true, Some(cue_len)) = (reciting, recite) {
                    if t + 1 >= cue_len && t + 1 < ids.len() {
                        if recite_plan && t + 1 == cue_len && !(testing && bind_lesion) {
                            if let Some(hc) = &bind_hc {
                                let n_prev = bind_list.len().saturating_sub(bind_sentence_pairs.len());
                                let mut idx: Vec<usize> = bind_sentence_pairs.iter().flat_map(|&(w, c)| sparse_binding(&enc.codes[w], w, c, false)).collect();
                                idx.extend(bind_list[..n_prev].iter().flat_map(|&(w, c)| sparse_binding(&enc.codes[w], w, c, true)));
                                idx.sort_unstable();
                                idx.dedup();
                                // the story just read: the most recent episode, by recency; its
                                // first event is the cue's, so the plan starts after the event
                                // the cue's words match best; else, the episode the cue recalls
                                let recent = hc.recent_episode(32);
                                let cue_words: Vec<usize> = truth_ids[..cue_len].to_vec();
                                let words_of_ev = |ev: &Vec<usize>| ev.iter().map(|&i| i % 4096).filter(|&w| w < vocab.len()).collect::<Vec<usize>>();
                                let start = recent
                                    .iter()
                                    .enumerate()
                                    .max_by_key(|(i, ev)| (words_of_ev(ev).iter().filter(|w| cue_words.contains(w)).count(), std::cmp::Reverse(*i)))
                                    .map_or(0, |(i, _)| i + 1);
                                let events: Vec<Vec<usize>> = if recent.is_empty() { hc.sequence_from(&idx, 16) } else { recent[start.min(recent.len())..].to_vec() };
                                for ev in events {
                                    plan.extend(ev.iter().map(|&i| i % 4096).filter(|&w| w < vocab.len()));
                                    if plan.last().map_or(true, |&w| vocab[w] != ".") {
                                        plan.push(index["."]);
                                    }
                                }
                            }
                        }
                        let k = t + 1 - cue_len;
                        let cortex = enc.decode(&out).unwrap_or(index["."]);
                        let said = match plan.get(k) {
                            Some(&p) => {
                                plan_stats[0] += 1;
                                let expect = column.l23.peek_union(&input, BITS);
                                let fits = expect.count_ones() == 0 || enc.codes[p].as_words().iter().zip(expect.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24;
                                plan_stats[1] += fits as usize;
                                if fits { p } else { cortex }
                            }
                            None => cortex,
                        };
                        let truth = truth_ids[t + 1];
                        // motor speech: what is meant is said through the motor area and the tract
                        let said = if motor_speech {
                            let (w, p) = motor_say(&motor, &tract, &enc.codes[said]);
                            eff_pred[t + 1] = p;
                            w.unwrap_or(index["."])
                        } else {
                            said
                        };
                        recite_stats[0] += 1;
                        recite_stats[1] += (said == truth) as usize;
                        recited.push(said);
                        // altered feedback: what is heard may differ from what was said
                        let heard = if self_noise > 0 && chance(&mut noise_rng, self_noise) { (said + 1 + noise_rng.gen_range(0..vocab.len() - 1)) % vocab.len() } else { said };
                        // RECITE_DRY=1 (a control): speak, but hear the story's word
                        let heard = if std::env::var("RECITE_DRY").is_ok() { truth } else { heard };
                        ids[t + 1] = heard;
                        s.words[t + 1] = vocab[heard];
                        inner[t + 1] = true;
                        efference[t + 1] = Some(said);
                    }
                }
                // the question's recall contained the answer: good credit for that episode
                // (whether or not the still-learning predictor used it)
                let recall_had_answer = input.as_words().len() >= 2 * (BITS / 64)
                    && input.as_words()[BITS / 64..2 * (BITS / 64)].iter().zip(enc.codes[next].as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24;
                if !testing && policy == Policy::Consolidate && t + 1 == s.answer_at && recall_had_answer {
                    if let Some(id) = last_recall_id {
                        match replay_mode.as_str() {
                            "tagged" => {
                                memory.tag(id);
                                if let Some(c) = &last_recall_cue {
                                    tag_cues.insert(id, c.clone());
                                }
                                tagged_or_replayed += 1;
                            }
                            "awake" => {
                                if let (Some(ep), Some(cue)) = (memory.get_by_id(id).cloned(), last_recall_cue.clone()) {
                                    // keyed on the question's own cue (what the prefrontal
                                    // query asked for), not on the episode's rarest bits
                                    let mut content = memory.novel(&ep, habituation);
                                    for (r, &c) in content.as_words_mut().iter_mut().zip(cue.as_words()) {
                                        *r &= !c;
                                    }
                                    if cue.count_ones() > 0 && content.count_ones() > 0 {
                                        let mut o = BitVector::new(BITS, Some(0));
                                        semantic.process_predictive(&cue, &mut o);
                                        semantic.feedback(&cue, &content, &mut sleep_rng);
                                        tagged_or_replayed += 1;
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                // the learned gate's outcome: how well the higher area predicted the next word
                if let Some((code, asked)) = mg_pending.take() {
                    let r = area.column.outcome(&enc.codes[next]) as i32 - if asked { gate_cost } else { 0 };
                    mg_bg.reward_candidate(&code, r, &mut bg_rng);
                }
                // L5 → basal ganglia: the column's outcome for this prediction
                // L5 attribution for the L6 gate, read before L2/3 learns: per channel passed,
                // did the prediction read its frame and come true?
                // the top-down frame's L4 index, for L5 attribution
                let td_frame = if hier_early { 1 } else { 1 + base_frames };
                // the gate's learning signal: by default the column read the top-down frame and
                // came true (use); HIER_GATE_SIGNAL=area: the higher area's own prediction came
                // true (its L5 outcome: the source's reliability, whoever the column followed)
                let td_used = hier_gate_step.is_some()
                    && if hier_gate_area { area.column.outcome(&enc.codes[next]) >= Q_HALF } else { column.outcome_via(td_frame, &enc.codes[next]) >= Q_HALF };
                let l6_used: Vec<bool> = match &l6_step {
                    Some((_, passed)) => passed.iter().enumerate().map(|(c, &p)| p && column.outcome_via(1 + early_shift + c, &enc.codes[next]) >= Q_HALF).collect(),
                    None => Vec::new(),
                };
                let l5 = if l5_used { column.outcome_via(choice_frame, &enc.codes[next]) } else { column.outcome(&enc.codes[next]) };
                if !testing && t + 1 == s.answer_at {
                    if let Policy::Pfc { learned: true } = policy {
                        // dopamine: the recall that working memory cued contained the answer
                        // (local), or the column predicted the answer (L5)
                        let r = if l5_reward { l5 as i32 } else if recall_had_answer { ONE as i32 } else { 0 };
                        pfc_gate.reward(r, &mut bg_rng);
                        l5_sum[0] += to_f32(l5) as f64; // report
                        l5_sum[1] += 1.0;
                        pfc_rewards += recall_had_answer as usize;
                        pfc_questions += 1;
                    }
                }
                last_recall_id = None;
                last_recall_cue = None;
                if testing {
                    // the fast inhibitory loop keeps running when slow learning is off. It
                    // learns from what is heard next: the page's word, or, where the network
                    // spoke, its own word as heard (so it learns only from a real mismatch
                    // between what was said and what came back)
                    column.fast_inhibit(&enc.codes[ids[t + 1]]);
                }
                if !testing {
                    // PROF: [2] evaluation, diagnostics, rewards
                    prof[2] += prof_t.elapsed().as_secs_f64();
                    prof_t = std::time::Instant::now();
                    // the cortex learns from the page only: not from an internal step's word,
                    // which is its own (or memory's) prediction
                    let page = !inner[t + 1];
                    if page {
                        column.learn(&input, &enc.codes[next], &mut rng);
                        if let Some(l) = layer5.as_mut() {
                            l.learn(&input, &enc.codes[next], &mut rng);
                        }
                        if let Some(l) = primed5.as_mut() {
                            l.learn(&input, &enc.codes[next], &mut rng);
                        }
                        if let Some(l) = primed23.as_mut() {
                            l.learn(&input, &enc.codes[next], &mut rng);
                        }
                        for l in bit23.iter_mut() {
                            l.learn(&input, &enc.codes[next], &mut rng);
                        }
                        if let Some(l) = bit5.as_mut() {
                            l.learn(row5.as_ref().unwrap_or(&input), &enc.codes[next], &mut rng);
                        }
                        if let Some(sc) = slow.as_mut() {
                            sc.feedback(&input, &enc.codes[next], &mut slow_rng);
                        }
                        // the climbing fibre: the cerebellum learns what came next, fast
                        if let (Some(cb), Some(x)) = (cerebellum.as_mut(), cb_input.take()) {
                            cb.learn(&x, &enc.codes[next], &mut cb_rng);
                        }
                    }
                    // the top-down go/no-go's reward (HIER_SURPRISE=learned): consulting is worth
                    // what the frame changed, against the column's own prediction without it
                    // (one look-up): right where it would be wrong 1, wrong where it would be
                    // right 0, no difference one half less the compute; skipping, one half
                    // the routing record (ROUTE): one channel in turn, passed whole against removed
                    if let (Some((_, ch, slots)), false) = (route_step.as_ref(), route_rerun) {
                        if page && !ch.is_empty() {
                            // every slot at once, from the same match
                            let key = ((if t > 0 { ids[t - 1] } else { vocab.len() }) * (vocab.len() + 1) + ids[t]) as u64 * 64;
                            let with = column.l23.peek(&input).and_then(|o| enc.decode(&o)) == Some(next);
                            for k in 1..=*slots {
                                let in_slot: Vec<usize> = ch.iter().map(|(c, _)| *c).filter(|c| route_order.iter().position(|x| x == c).unwrap_or(route_order.len()).min(slots - 1) + 1 == k).collect();
                                if in_slot.is_empty() {
                                    continue;
                                }
                                let without = column.l23.peek_shallow(&input, k).and_then(|o| enc.decode(&o)) == Some(next);
                                if with != without {
                                    for c in in_slot {
                                        mix.record(ROUTE_SRC + c as u8, key, with);
                                        let e = route_score.entry(c).or_insert((0, 0));
                                        if with { e.0 += 1 } else { e.1 += 1 }
                                    }
                                }
                            }
                        }
                        route_step = None;
                    }
                    if let Some((word, ch, slots)) = route_step.take() {
                        if page && !ch.is_empty() {
                            let j = route_turn % ch.len();
                            route_turn += 1;
                            let key = ((if t > 0 { ids[t - 1] } else { vocab.len() }) * (vocab.len() + 1) + ids[t]) as u64 * 64;
                            let mut shares: Vec<u64> = ch.iter().map(|(c, _)| (2 * mix.rate(ROUTE_SRC + *c as u8, key) as u64).min(ONE as u64)).collect();
                            shares[j] = ONE as u64;
                            let right = |skip: Option<usize>| {
                                let row = route_row(&word, &ch, &shares, &route_order, slots, skip);
                                column.l23.peek(&BitVector::from_words(row)).and_then(|o| enc.decode(&o)) == Some(next)
                            };
                            let (with, without) = (right(None), right(Some(j)));
                            if with != without {
                                let c = ch[j].0;
                                mix.record(ROUTE_SRC + c as u8, key, with);
                                let e = route_score.entry(c).or_insert((0, 0));
                                if with { e.0 += 1 } else { e.1 += 1 }
                            }
                        }
                    }
                    // the frame's counterfactual record (HIER_TRUST_GATE=cf): full frame against none
                    if let (Some(full), Some((at, n))) = (td_full.take(), td_span) {
                        if page {
                            let with_frame = |f: Option<&BitVector>| {
                                let mut w = input.as_words().to_vec();
                                match f {
                                    Some(f) => w[at..at + n].copy_from_slice(f.as_words()),
                                    None => w[at..at + n].iter_mut().for_each(|x| *x = 0),
                                }
                                column.l23.peek(&BitVector::from_words(w)).and_then(|o| enc.decode(&o)) == Some(next)
                            };
                            let (with, without) = (with_frame(Some(&full)), with_frame(None));
                            if with != without {
                                let p = if t > 0 { ids[t - 1] } else { vocab.len() };
                                let key = (p * (vocab.len() + 1) + ids[t]) as u64 * 64;
                                let bucket = CONF_BANDS.iter().filter(|&&e| area.column.confidence() >= e).count() as u64;
                                mix.record(TRUST_CF, key + bucket, with);
                            }
                        }
                    }
                    if hier_trace_len > 0 && page {
                        // credit over time: this step's outcome, to every eligible choice
                        if let Some(c) = hier_pending.take() {
                            hier_trace.push_back((c, hier_consult, 0, 0, ONE, 0));
                        }
                        let right = (enc.decode(&out) == Some(next)) as u64;
                        let w = if t + 1 == s.answer_at { hier_answer_weight } else { 1 };
                        for e in hier_trace.iter_mut() {
                            e.2 += e.4 as u64 * right * w;
                            e.3 += e.4 as u64 * w;
                            e.4 = ((e.4 as u64 * hier_decay as u64) >> 16) as Q16;
                            e.5 += 1;
                        }
                        while hier_trace.front().map_or(false, |e| e.5 >= hier_trace_len) {
                            let (c, consulted, sum, weight, _, _) = hier_trace.pop_front().unwrap();
                            let credit = ((sum << 16) / weight.max(1)) as i32 - if consulted { hier_cost as i32 } else { 0 };
                            hier_bg.reward_candidate(&c, credit.max(0), &mut bg_rng);
                        }
                    }
                    if let Some(c) = hier_pending.take() {
                        let r = if hier_consult {
                            let with = enc.decode(&out) == Some(next);
                            let without = td_span.map_or(with, |(at, n)| {
                                let mut w = input.as_words().to_vec();
                                w[at..at + n].iter_mut().for_each(|x| *x = 0);
                                column.l23.peek(&BitVector::from_words(w)).and_then(|o| enc.decode(&o)) == Some(next)
                            });
                            match (with, without) {
                                (true, false) => ONE as i32,
                                (false, true) => 0,
                                _ => (ONE / 2) as i32 - hier_cost as i32,
                            }
                        } else {
                            (ONE / 2) as i32
                        };
                        hier_bg.reward_candidate(&c, r.max(0), &mut bg_rng);
                    }
                    // predictive coding: the higher area learns the lower column's residual,
                    // the words it failed to predict (HIER_RESIDUAL=0: every word)
                    if let Some(ctx) = hier_gate_step.take() {
                        hier_gate.learn(0, &ctx, td_used, &mut rng);
                    }
                    if let (Some(a), Some(x), true) = (assoc.as_mut(), assoc_in.take(), page) {
                        a.learn(&x, &enc.codes[next], &mut rng);
                    }
                    if let Some(hin) = hier_in.take() {
                        if page && (!hier_residual || column.surprise(&enc.codes[next]) >= Q_HALF) {
                            area.learn(&hin, &enc.codes[next], &mut rng);
                            for (u, x) in upper.iter_mut().zip(upper_in.iter_mut()) {
                                if let Some(x) = x.take() {
                                    u.learn(&x, &enc.codes[next], &mut rng);
                                }
                            }
                        }
                    }
                    // PROF: [3] L2/3 learning
                    prof[3] += prof_t.elapsed().as_secs_f64();
                    prof_t = std::time::Instant::now();
                    // dopamine: did the followed item's recall contain what came next?
                    if policy == Policy::LearnedGate && enc.decode(&out) != Some(next) {
                        // surprise: discover and score routes that would have relayed `next`
                        route_scores.observe_surprise(&column.l6, &enc.codes[next], 3, 8);
                    }
                    if let (Policy::L6Gate { gated: true }, Some((ctx, passed))) = (policy, l6_step.take()) {
                        // corticothalamic Hebbian update: an open channel is strengthened if
                        // the prediction read its frame and came true (L5 attribution)
                        for (c, &p) in passed.iter().enumerate() {
                            if p {
                                l6_gate.learn(c, &ctx, l6_used[c], &mut rng);
                            }
                        }
                    }
                    if let Some(rel) = gate_pending.take() {
                        let hit = rel.as_words().iter().zip(enc.codes[next].as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24;
                        gate_bg.reward(if l5_reward { l5 as i32 } else if hit { ONE as i32 } else { 0 }, &mut bg_rng);
                        l5_sum[0] += to_f32(l5) as f64; // report
                        l5_sum[1] += 1.0;
                    }
                    if let Some(hop2) = bg_pending.take() {
                        let hit = hop2.as_words().iter().zip(enc.codes[next].as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24;
                        bg.reward(if l5_reward { l5 as i32 } else if hit { ONE as i32 } else { 0 }, &mut bg_rng);
                        l5_sum[0] += to_f32(l5) as f64; // report
                        l5_sum[1] += 1.0;
                    }
                }
                bg_pending = None;
                gate_pending = None;
            }

            // the event ends here: at "." (default), or where the boundary cell fires
            let boundary_now = if learned_bound {
                let fired = bcell.observe(&set_bits(&enc.codes[ids[t]])) || t + 1 == ids.len();
                if testing && !replaying {
                    bound_stats[0] += fired as usize;
                    bound_stats[1] += (fired && ids[t] == full_stop) as usize;
                    bound_stats[2] += (ids[t] == full_stop) as usize;
                }
                fired
            } else {
                ids[t] == full_stop
            };
            // the event's words: [seg_lo, seg_hi) (the "." itself excluded by default)
            let (seg_lo, seg_hi) = if learned_bound { (seg_start, t + 1) } else { (s.words[..t].iter().rposition(|w| *w == ".").map_or(0, |p| p + 1), t) };
            if boundary_now && hier && bound_detect {
                if testing {
                    let opening = seg_lo == 0;
                    story_openings += opening as usize;
                    if bound_fired {
                        if opening {
                            bound_hits.0 += 1;
                        } else {
                            bound_hits.1 += 1;
                        }
                    }
                }
                bound_fired = false;
            }
            if boundary_now {
                query_ec = None;
                // one-shot: the whole sentence (or its unpredicted part) is one episode.
                // Persist: test questions are not stored, or the first anchor question
                // would leak its own answer to every later one.
                let question = sentence.as_words().iter().zip(enc.codes[index["where"]].as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24;
                if !replaying && !(task == Task::Persist && testing && question) {
                    if hier && std::env::var("MEM_CONTEXT").is_ok() {
                        let mut e = if predictive_novelty { surprising.clone() } else { sentence.clone() };
                        e.or_mut(&area.state(&surprising));
                        memory.store(&e);
                    } else {
                        memory.store(if predictive_novelty { &surprising } else { &sentence });
                    }
                }
                if let (Some(dg), Some(ca3), false) = (&dg, &mut ca3, replaying) {
                    // Encode the novel part: content shared by most episodes ("went to the")
                    // would otherwise dominate the dentate gyrus, give every episode the same
                    // code, and swamp recall.
                    let x = set_bits(&if predictive_novelty { surprising.clone() } else { memory.novel(&sentence, Q_04) });
                    if !x.is_empty() {
                        ca3.store(&x, &dg.separate(&x));
                    }
                }
                sentence = BitVector::new(BITS, Some(0));
                if hippo_self && bind && !bind_sentence.is_empty() {
                    if ((!testing || recite_plan || self::question() || store_test) && !replaying) || (reciting && self_store) {
                        if let Some(hc) = &mut bind_hc {
                            hc.set_source(reciting as u8);
                            let ev = event_vec(&bind_sentence, &bind_prev, ctx_offset);
                            let content = event_vec(&bind_sentence, &BitVector::new(BITS, Some(0)), ctx_offset);
                            let mut context = bind_prev.clone();
                            context.rotl_mut(ctx_offset);
                            if std::env::var("SELFDIAG").is_ok() && s.words[..=t].iter().any(|w| NEW_NAMES.contains(w)) {
                                let sent: Vec<String> = bind_list[bind_list.len() - bind_sentence.len().min(bind_list.len())..].iter().map(|(w, sl)| format!("{}@{}", vocab[*w], sl)).collect();
                                let nov = hc.novelty(&set_bits(&ev));
                                let r = hc.recall(&set_bits(&ev));
                                eprintln!("  SELFSTORE {:?}: bindings {:?}, {} bits, novelty {}, recall of itself before storing: {} bits", &s.words[..=t], sent, ev.count_ones(), nov, r.ec.len());
                            }
                            if sparse_bind {
                                let n_prev = bind_list.len().saturating_sub(bind_sentence_pairs.len());
                                let mut content_idx: Vec<usize> = bind_sentence_pairs.iter().flat_map(|&(w, c)| sparse_binding(&enc.codes[w], w, c, false)).collect();
                                // the held item, stored with this event if attached
                                let start = if learned_bound { seg_start } else { s.words[..t].iter().rposition(|w| *w == ".").map_or(0, |p| p + 1) };
                                let attach = match (held_item, qattach_learned) {
                                    (Some((hw, _)), true) if !ids[start..seg_hi].contains(&hw) && seg_hi > start + 1 => {
                                        // the choice's code is the sentence itself (its words'
                                        // codes, shifted per action) plus a per-action bias, so
                                        // what is learned carries over to sentences sharing words
                                        let mut bag = BitVector::new(BITS, Some(0));
                                        for &w in &ids[start..seg_hi] {
                                            bag.or_mut(&enc.codes[w]);
                                        }
                                        let cands: Vec<BitVector> = (0..2)
                                            .map(|a| {
                                                let mut v = bag.clone();
                                                v.rotl_mut(a * BITS / 2);
                                                v.or_mut(&step_code(7000, a));
                                                v
                                            })
                                            .collect();
                                        let a = step_bg.select(&cands, if !testing { Some(&mut bg_rng) } else { None }).unwrap_or(0);
                                        attach_pending.push((cands[a].clone(), a == 1, None, hw));
                                        (a == 1).then_some(hw)
                                    }
                                    _ => None,
                                };
                                if let Some(hw) = attach {
                                    content_idx.extend(sparse_binding(&enc.codes[hw], hw, HELD_FIELD, false));
                                    hold_stats[2] += (testing && s.held_out) as usize;
                                }
                                let context_idx: Vec<usize> = bind_list[..n_prev].iter().flat_map(|&(w, c)| sparse_binding(&enc.codes[w], w, c, true)).collect();
                                hc.store_split(&content_idx, &context_idx, &set_bits(if replay_gen { &ev } else { &content }));
                                if ne_on && sent_surprises >= 2 {
                                    // NE: the salient event is strengthened (stored again)
                                    hc.store_split(&content_idx, &context_idx, &set_bits(if replay_gen { &ev } else { &content }));
                                    ne_stats[0] += 1;
                                }
                                if let (true, Some(r), Some(last)) = (qattach_learned && attach.is_some(), hc.last_row(), attach_pending.last_mut()) {
                                    last.2 = Some(r);
                                }
                                if let Some(r) = hc.last_row() {
                                    row_narrator.insert(r, narrator + 1);
                                    if reinstate && hier {
                                        row_cortex.insert(r, area.state(&BitVector::new(BITS, Some(0))));
                                    }
                                }
                                // the cortical state this event was read in (the higher area's
                                // slow state at the sentence's start), kept per row for replay
                                if let (true, Some(r)) = (infer_reps.is_some() && hier, hc.last_row()) {
                                    row_state.insert(r, set_bits(&area.state(&BitVector::new(BITS, Some(0)))).into_iter().map(|b| b as u32).collect());
                                    let start = if learned_bound { seg_start } else { s.words[..t].iter().rposition(|w| *w == ".").map_or(0, |p| p + 1) };
                                    row_prefix.insert(r, ids[..start].to_vec());
                                }
                            } else {
                                hc.store_event(&set_bits(&content), &set_bits(&context));
                            }
                            hc.advance_time();
                            hc.set_source(0);
                        }
                    }
                    for b in &bind_sentence {
                        bind_prev.or_mut(b);
                    }
                }
                bind_sentence.clear();
                bind_sentence_pairs.clear();
                if proposals_on && !testing && !replaying && !proposals.is_empty() {
                    // what is read confirms or contradicts the proposals of this season
                    let sent: Vec<usize> = s.words[seg_lo..seg_hi].iter().map(|w| index[w]).collect();
                    if let (Some(season), Some(&first)) = (s.words.iter().find_map(|w| SEASONS.iter().position(|x| x == w)), sent.first()) {
                        for ((ps, pw), e) in proposals.iter_mut() {
                            if *ps != season || pw.first() != Some(&first) || pw.len() != sent.len() {
                                continue;
                            }
                            let diff = pw.iter().zip(&sent).filter(|(a, b)| a != b).count();
                            if diff == 0 {
                                e.1 += 1;
                            } else if diff == 1 {
                                e.2 += 1;
                            }
                        }
                    }
                }
                if rel_reps.is_some() && !testing && !replaying {
                    let fact: Vec<usize> = s.words[seg_lo..seg_hi].iter().rev().map(|w| index[w]).collect();
                    rel.observe_from(&fact.into_iter().rev().collect::<Vec<_>>(), narrator + 1);
                }
                if semantic_reps.is_some() && !testing && !replaying && !hippo_self {
                    sem_buf.push(s.words[seg_lo..seg_hi].iter().rev().map(|w| index[w]).collect());
                }
                completed_sentence = false;
                rolled = 0;
                if hier {
                    area.end_sentence(&surprising);
                    if let Some(a) = assoc.as_mut() {
                        a.end_sentence(&surprising);
                    }
                    for u in upper.iter_mut() {
                        u.end_sentence(&surprising);
                    }
                }
                surprising = BitVector::new(BITS, Some(0));
                // word frequencies (per sentence), for read-back's "rare"
                sentence_count += !replaying as u32;
                if testing {
                    let first_of_story = seg_lo == 0;
                    bound_log.push((first_of_story, sent_surprise / sent_words.max(1) as f32, first_surprise));
                }
                sent_surprise = 0.0;
                sent_words = 0;
                sent_surprises = 0;
                for w in &s.words[seg_lo..seg_hi] {
                    word_count[index[w]] += !replaying as u32;
                }
                // read-back: say back the most recent rare word held, hear it
                if hier {
                    if let Some(mode) = readback.as_deref() {
                        let n = upper.len();
                        let empty = BitVector::new(BITS, Some(0));
                        let rare = |w: usize| ((word_count[w] as u64) << 16) < readback_rare as u64 * sentence_count as u64;
                        let words_of = |bv: &BitVector| -> Option<usize> { enc.decode(bv) };
                        let mut heard: Vec<usize> = Vec::new();
                        // areas that read back: the top one, or all; index n = area 2
                        let which: Vec<usize> = if mode == "all" { (0..=n).collect() } else { vec![if n > 0 { n - 1 } else { n }] };
                        for k in which {
                            let a = if k == n { &mut area } else { &mut upper[k] };
                            // the target: the most recent rare word in the window
                            let target = a.recent_words().filter_map(|c| words_of(c)).find(|&w| rare(w));
                            let Some(target) = target else { continue };
                            // the probe: the cue "." in the sentence frame (end of sentence: say
                            // back) and the area's state; an area reading from above gets an
                            // empty frame there
                            let cue = &enc.codes[if learned_bound { ids[t] } else { full_stop }];
                            let x = if (k == n && n > 0 && !chain_mix) || (k + 1 < n && !chain_mix) {
                                a.input_with(cue, &empty, Some(&empty))
                            } else {
                                a.input_with(cue, &empty, None)
                            };
                            let said = words_of(&a.predict(&x));
                            if testing {
                                readback_stats.0 += 1;
                                readback_stats.1 += (said == Some(target)) as usize;
                            } else if said != Some(target) {
                                a.learn(&x, &enc.codes[target], &mut sleep_rng);
                            }
                            if let Some(w) = said {
                                heard.push(w);
                            }
                        }
                        // hearing itself: the spoken words are surprising input of the next
                        // sentence, for every area's window
                        for w in heard {
                            surprising.or_mut(&enc.codes[w]);
                            area.note_word(&enc.codes[w]);
                            for u in upper.iter_mut() {
                                u.note_word(&enc.codes[w]);
                            }
                        }
                    }
                }
            }
            if boundary_now {
                seg_start = t + 1;
            }
            prev = Some(ids[t]);
        }
        if policy == Policy::Consolidate && !testing && !memory.is_empty() && chance(&mut sleep_rng, replay_prob) {
            // sleep: replay stored episodes into the cortical semantic store
            let boost = if replay_mode == "tagged" { tag_boost } else { 0 };
            for _ in 0..replays {
                let i = memory.sample_replay(&mut sleep_rng, boost).unwrap();
                let ep = memory.get(i).unwrap().clone();
                // a tagged episode is replayed under the cue of the question that tagged it
                let cue = tag_cues.get(&memory.id_of(i)).cloned().unwrap_or_else(|| memory.rarest(&ep, Q_TENTH, rarity_ratio));
                if cue.count_ones() == 0 {
                    continue;
                }
                let mut content = memory.novel(&ep, habituation);
                for (r, &c) in content.as_words_mut().iter_mut().zip(cue.as_words()) {
                    *r &= !c;
                }
                if content.count_ones() == 0 {
                    continue;
                }
                let mut out = BitVector::new(BITS, Some(0));
                semantic.process_predictive(&cue, &mut out);
                semantic.feedback(&cue, &content, &mut sleep_rng);
            }
        }
        if let (true, Some(cue_len)) = (reciting, recite) {
            let content = |w: usize| (word_count[w] as u64) * 20 < sentence_count as u64 && vocab[w] != ".";
            let mut want: Vec<usize> = truth_ids[cue_len.min(truth_ids.len())..].iter().copied().filter(|&w| content(w)).collect();
            want.sort_unstable();
            want.dedup();
            recite_stats[2] += want.len();
            recite_stats[3] += want.iter().filter(|w| recited.contains(w)).count();
            if recite_sample.len() < 2 && s.held_out {
                recite_sample.push(format!(
                    "story: {} | retold: {} {}",
                    truth_ids.iter().map(|&w| vocab[w]).collect::<Vec<_>>().join(" "),
                    truth_ids[..cue_len.min(truth_ids.len())].iter().map(|&w| vocab[w]).collect::<Vec<_>>().join(" "),
                    recited.iter().map(|&w| vocab[w]).collect::<Vec<_>>().join(" ")
                ));
            }
        }
        if reciting {
            // a retelling can stop mid-sentence: its unfinished sentence must not run into
            // the next story's first sentence (a page always ends with ".")
            sentence = BitVector::new(BITS, Some(0));
            surprising = BitVector::new(BITS, Some(0));
            bind_sentence.clear();
            bind_sentence_pairs.clear();
            sent_surprise = 0.0;
            sent_words = 0;
            sent_surprises = 0;
        }
        if !reciting {
            if !story_altered {
                boundary_carry = Some((prev, slot_prev, role_now, expect_prev.clone()));
            }
            prev_altered = story_altered;
        }
        if let Some((p, sp, rn, ep)) = saved_carry {
            prev = p;
            slot_prev = sp;
            role_now = rn;
            expect_prev = ep;
        }
        if let Some(c) = saved_reading {
            area.restore_context(&c[0]);
            for (u, x) in upper.iter_mut().zip(&c[1..]) {
                u.restore_context(x);
            }
        }
        prev_replaying = replaying;
        }
    }
    let pct = |r: (usize, usize)| 100.0 * r.0 as f64 / r.1.max(1) as f64;
    if l5_sum[1] > 0.0 {
        eprintln!("  L5 seed {seed} ({}): mean training outcome at rewarded steps {:.3} over {}", if l5_reward { "reward = L5" } else { "local reward" }, l5_sum[0] / l5_sum[1], l5_sum[1]);
    }
    if let Policy::Pfc { learned } = policy {
        let mut rows: Vec<(&&str, &(usize, usize))> = pfc_loads.iter().filter(|(_, (l, _))| *l > 0).collect();
        rows.sort_by(|a, b| (b.1 .0 * 1000 / b.1 .1.max(1)).cmp(&(a.1 .0 * 1000 / a.1 .1.max(1))));
        let shown: Vec<String> = rows.iter().take(12).map(|(w, (l, d))| format!("{w} {:.0}%", 100.0 * *l as f64 / (*d).max(1) as f64)).collect();
        eprintln!("  PFC seed {seed} ({}): words loaded into working memory at test (share of occurrences): {}", if learned { "learned gate" } else { "load names" }, shown.join(", "));
        if learned {
            let vals: Vec<String> = ["mary", "went", "to", "kitchen", ".", "cat", "where", "person", "?"]
                .iter()
                .map(|w| format!("{w} load {:.2} keep {:.2}", pfc_gate.value(Gate::Load, index[w]), pfc_gate.value(Gate::Keep, index[w])))
                .collect();
            eprintln!("  PFC seed {seed}: training rewards {pfc_rewards}/{pfc_questions}; values: {}", vals.join("; "));
        }
    }
    if policy == Policy::Consolidate {
        eprintln!(
            "  CONSOLIDATE seed {seed} ({replay_mode}): memory frame from the cortex at {from_cortex} test answers; semantic kernels {}; {} question tags / awake replays",
            semantic.len(),
            tagged_or_replayed
        );
    }
    if let Policy::L6Gate { gated } = policy {
        let names: Vec<String> = routes.iter().map(|r| format!("({},{})", r.query_lag, r.value_offset)).chain(["memory".to_string()]).collect();
        let at = |k: usize| -> String {
            names.iter().zip(&l6_open_at_answer[k]).map(|(n, &c)| format!("{n} {:.0}%", 100.0 * c as f64 / l6_answers[k].max(1) as f64)).collect::<Vec<_>>().join(", ")
        };
        eprintln!(
            "  L6 seed {seed} ({}{}): channels passed per word at test {:.2} of {}; open at test answers: {}",
            if gated { "gated" } else { "all open" },
            if l6_pair { ", context = current + previous" } else { "" },
            l6_open_sum as f64 / l6_words.max(1) as f64,
            names.len(),
            if task == Task::Give { format!("\"what did X give ?\" {}; other questions {}", at(1), at(0)) } else { at(0) }
        );
    }
    if matches!(policy, Policy::ThalamicGate | Policy::LearnedGate) {
        let total = gate_chosen_at_answer.values().sum::<usize>().max(1) as f64;
        let mut shares: Vec<(String, usize)> = gate_chosen_at_answer.into_iter().collect();
        shares.sort_by(|a, b| b.1.cmp(&a.1));
        let shares: Vec<String> = shares.iter().map(|(n, c)| format!("{n} {:.0}%", 100.0 * *c as f64 / total)).collect();
        let pool: Vec<String> = gate_routes.iter().map(|r| format!("({},{})", r.query_lag, r.value_offset)).collect();
        eprintln!("  GATE seed {seed} routes in pool [{}]; channel released at test answers: {}", pool.join(" "), shares.join(", "));
    }
    if std::env::var("DIAG").is_ok() {
        let c = cover[1].max(1) as f64;
        eprintln!(
            "  CDIAG seed {seed} wrong held-out: n={} with a memory-reading kernel for the answer {:.0}%, best match/threshold {:.2}, its missing bits current/memory/previous {:.1}/{:.1}/{:.1}, its reliability {:.2} over {:.0} uses",
            cover[0],
            100.0 * cover[1] as f64 / cover[0].max(1) as f64,
            cover[2] as f64 / 100.0 / c,
            cover[3] as f64 / c,
            cover[4] as f64 / c,
            cover[5] as f64 / c,
            cover[6] as f64 / 100.0 / c,
            cover[7] as f64 / c
        );
        for (label, kd) in [("right", kdiag[1]), ("wrong", kdiag[0])] {
            let n = kd[6].max(1) as f64;
            let fired = (kd[6] - kd[0]).max(1) as f64;
            eprintln!(
                "  KDIAG seed {seed} held-out {label}: n={} no winner {:.0}%, winner reads memory {:.0}%, mask bits current/memory/previous {:.1}/{:.1}/{:.1}, reliability {:.2}",
                kd[6],
                100.0 * kd[0] as f64 / n,
                100.0 * kd[1] as f64 / fired,
                kd[2] as f64 / fired,
                kd[3] as f64 / fired,
                kd[4] as f64 / fired,
                kd[5] as f64 / 1000.0 / fired
            );
        }
        for (label, d) in [("right", diag[1]), ("wrong", diag[0])] {
            let n = d[3].max(1) as f64;
            eprintln!(
                "  DIAG seed {seed} held-out {label}: n={} answer bits {:.1}/32, recall size {:.0} bits, whole words {:.2}",
                d[3],
                d[0] as f64 / n,
                d[1] as f64 / n,
                d[2] as f64 / n
            );
        }
    }
    test_secs = phase_start.elapsed().as_secs_f64();
    if dump.is_none() {
        // COST: what the column stores, densely as now and as sparse indices
        let kernels = column.l23.kernels();
        if column.l23.canon_reused() > 0 {
            eprintln!("  CANON seed {seed}: {} growths found an identical kernel already present", column.l23.canon_reused());
        }
        let (fl, fh, fp) = column.l23.frame_memo_stats();
        if fl > 0 {
            eprintln!("  FRAME_MEMO seed {seed}: {fl} frame lookups: {:.0}% hits, {:.0}% patched, {:.0}% rebuilt", 100.0 * fh as f64 / fl as f64, 100.0 * fp as f64 / fl as f64, 100.0 * (fl - fh - fp) as f64 / fl as f64);
        }
        let (lookups, hits) = column.l23.memo_stats();
        if lookups > 0 {
            eprintln!("  MEMO seed {seed}: {hits} of {lookups} interpretations served from the memo ({:.0}%)", 100.0 * hits as f64 / lookups as f64);
        }
        if !upper.is_empty() {
            let parts: Vec<String> = upper
                .iter()
                .zip(&upper_has_answer)
                .enumerate()
                .map(|(i, (u, h))| format!("area {} (window {} sentences) {} kernels, held the answer at {:.1}%", i + 3, u.span(), u.column.l23.live(), 100.0 * *h as f64 / TEST as f64))
                .collect();
            eprintln!("  CHAIN seed {seed}: {}", parts.join("; "));
        }
        if hier_trust_gate.is_some() {
            let share = |a: [usize; 2]| 100.0 * a[1] as f64 / (a[0] + a[1]).max(1) as f64;
            if hier_trust_gate.as_deref() == Some("cf") {
                let mean = |i: usize| 100.0 * cf_share_sum[i] as f64 / (trust_passed[i][1].max(1) as f64 * ONE as f64);
                eprintln!("  TRUSTGATE seed {seed}: mean share of the top-down frame passed {:.1}% in training, {:.1}% at test; scaled down at {} training and {} test steps", mean(0), mean(1), cf_scaled[0], cf_scaled[1]);
            } else {
                eprintln!("  TRUSTGATE seed {seed}: top-down frame passed at {:.0}% of training steps, {:.0}% of test steps", share(trust_passed[0]), share(trust_passed[1]));
            }
        }
        if hc_ec {
            eprintln!("  HC_EC seed {seed}: entorhinal feedback at {} steps ({} at test), mean share passed {:.0}%", hc_ec_stats[0], hc_ec_stats[2], 100.0 * hc_ec_stats[1] as f64 / (hc_ec_stats[0].max(1) as f64 * ONE as f64));
        }
        if let Some(cb) = cerebellum.as_ref() {
            eprintln!("  LEARNING seed {seed}: slow cortex {} kernels, cerebellum {} kernels; at test answers the cerebellum proposed a word at {} and was right at {} ({:.1}%)", column.l23.live(), cb.live(), cb_stats[0], cb_stats[1], 100.0 * cb_stats[1] as f64 / cb_stats[0].max(1) as f64);
        }
        if let Some(sc) = slow.as_ref() {
            eprintln!("  SLOWCORTEX seed {seed}: {} kernels (the column {}); at test answers it proposed a word at {} and was right at {} ({:.1}%)", sc.live(), column.l23.live(), slow_stats[0], slow_stats[1], 100.0 * slow_stats[1] as f64 / slow_stats[0].max(1) as f64);
        }
        if route_on {
            let score: Vec<String> = route_order.iter().map(|c| { let (f, b) = route_score.get(c).copied().unwrap_or((0, 0)); format!("{c}: +{f} −{b}") }).collect();
            eprintln!("  ROUTE seed {seed}: final order (channel: fixes, breaks) [{}]; re-routings: {}; hippocampus routed at {hc_routed} steps", score.join(", "), if route_log.is_empty() { "none".to_string() } else { route_log.join("; ") });
        }
        if hier_grow {
            eprintln!(
                "  GROW seed {seed}: {} area(s) above the column at the end{}; {}",
                1 + upper.len() - bud_live as usize,
                if bud_live { " and a bud" } else { "" },
                if grow_log.is_empty() { "no promotions or prunings".to_string() } else { grow_log.join("; ") }
            );
        }
        if std::env::var("BOUNDDIAG").is_ok() && !bound_log.is_empty() {
            let stats = |first: bool, f: &dyn Fn(&(bool, f32, f32)) -> f32| -> String {
                let mut v: Vec<f32> = bound_log.iter().filter(|e| e.0 == first).map(f).collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                if v.is_empty() {
                    return "-".into();
                }
                let q = |p: f32| v[((v.len() - 1) as f32 * p) as usize];
                format!("n {} p10 {:.2} median {:.2} p90 {:.2}", v.len(), q(0.1), q(0.5), q(0.9))
            };
            eprintln!(
                "  BOUNDDIAG seed {seed}: sentence mean surprise: story-opening {} | other {}; first word: story-opening {} | other {}",
                stats(true, &|e| e.1),
                stats(false, &|e| e.1),
                stats(true, &|e| e.2),
                stats(false, &|e| e.2)
            );
        }
        if saccade.is_some() && sacc_stats[3] > 0 {
            eprintln!(
                "  SACCADE seed {seed}: at test, regressions to the previous sentence {} and to the page top / a landmark {} ({:.2} per story); {:.1}% of words were re-reads; a regression just before {:.1}% of answers, re-reading the first sentence at {:.1}%",
                sacc_stats[0],
                sacc_stats[1],
                (sacc_stats[0] + sacc_stats[1]) as f64 / TEST as f64,
                100.0 * sacc_stats[2] as f64 / (sacc_stats[2] + sacc_stats[3]) as f64,
                100.0 * sacc_stats[4] as f64 / TEST as f64,
                100.0 * sacc_hit_first as f64 / TEST as f64
            );
        }
        if replay_gen {
            eprintln!("  REPLAY_GEN seed {seed}: {} free-settling replays, {} decoded, {} (input, target) pairs given to sleep generalisation", gen_stats[0], gen_stats[1], gen_stats[2]);
        }
        if proposals_on {
            // truth from the world's rule: the name's family and the season decide the place
            let truth = |season: usize, w: &[usize]| -> Option<bool> {
                let name = vocab[*w.first()?];
                let place = vocab[*w.last()?];
                let f = match NEW_NAMES.iter().position(|n| *n == name) {
                    Some(i) => family_of(i, true),
                    None => family_of(NAMES.iter().position(|n| *n == name)?, false),
                };
                PLACES.contains(&place).then(|| PLACES[family_place(f, season)] == place)
            };
            let mut rows: [(usize, usize, usize); 5] = [(0, 0, 0); 5]; // (n, true, judged) for all, corroborated, confirmed, contradicted, validated
            let mut open: Vec<String> = Vec::new();
            for ((season, w), e) in &proposals {
                let tr = truth(*season, w);
                let support = e.5.iter().map(|&f| bind_hc.as_ref().map_or(1, |h| h.testimony(f)) as usize).sum::<usize>().min(e.0.len());
                let cred = rel_reps.and(rel.relation_for(&[index["is"], index["a"]])).map(|isa| proposal_credibility(&rel, isa, w, &e.6, &e.0, &row_narrator));
                let corroborated = cred.map_or(support >= proposal_support, |c| c >= proposal_min);
                let valid = (e.1 > 0 || corroborated) && e.2 == 0;
                for (k, on) in [true, corroborated, e.1 > 0, e.2 > 0, valid].into_iter().enumerate() {
                    if on {
                        rows[k].0 += 1;
                        rows[k].1 += (tr == Some(true)) as usize;
                        rows[k].2 += tr.is_some() as usize;
                    }
                }
                if !valid && e.2 == 0 && open.len() < 6 {
                    open.push(format!("{} {} (credibility {:.2}; independent premises {support}: {} testimonies of the fact, {} source events)", SEASONS[*season], w.iter().map(|&x| vocab[x]).collect::<Vec<_>>().join(" "), to_f32(cred.unwrap_or(0)), e.5.iter().map(|&f| bind_hc.as_ref().map_or(1, |h| h.testimony(f))).sum::<u32>(), e.0.len()));
                }
            }
            let names = ["made", "corroborated", "confirmed by reading", "contradicted by reading", "validated (replayed)"];
            eprintln!(
                "  PROPOSALS seed {seed}: {}",
                (0..5).map(|k| format!("{} {} ({:.0}% true of {} judged)", names[k], rows[k].0, 100.0 * rows[k].1 as f64 / rows[k].2.max(1) as f64, rows[k].2)).collect::<Vec<_>>().join("; ")
            );
            eprintln!("  PROPOSALS seed {seed}: open, to investigate: {}", open.join("; "));
        }
        if self_store || source_tag {
            eprintln!(
                "  SOURCE seed {seed} ({}): at test questions the hippocampus recalled a world event {} times, the network's own words {} times",
                if source_tag { "recall about the world: world events only" } else { "untagged recall" },
                source_stats[0],
                source_stats[1]
            );
        }
        if let Some(k) = recite {
            eprintln!(
                "  RECITE seed {seed}: cue {k} words; {:.1}% of {} words right in place; content words said {:.1}% of {}",
                100.0 * recite_stats[1] as f64 / recite_stats[0].max(1) as f64,
                recite_stats[0],
                100.0 * recite_stats[3] as f64 / recite_stats[2].max(1) as f64,
                recite_stats[2]
            );
            if recite_plan {
                eprintln!("  RECITE seed {seed}: the hippocampus planned {} steps; the planned word fit the cortex's expectation and was spoken at {:.1}%", plan_stats[0], 100.0 * plan_stats[1] as f64 / plan_stats[0].max(1) as f64);
            }
            for x in &recite_sample {
                eprintln!("  RECITE seed {seed}: {x}");
            }
        }
        if motor_speech {
            let (ki, kf, nb) = motor.stats();
            let say_right = (0..vocab.len()).filter(|&w| motor_say(&motor, &tract, &enc.codes[w]).0 == Some(w)).count();
            eprintln!(
                "  SPEECH seed {seed}: babbled {nb} commands (inverse {ki}, forward {kf} kernels); says {say_right} of {} words right; training questions {}, spoken at {:.1}%",
                vocab.len(),
                sp_train[0],
                100.0 * sp_train[1] as f64 / sp_train[0].max(1) as f64
            );
            let bands: Vec<String> = (0..5)
                .map(|b| {
                    let [n, sp, r] = sp_stats[b];
                    format!("band {b}: spoke {:.0}% of {n}, {:.1}% right", 100.0 * sp as f64 / n.max(1) as f64, 100.0 * r as f64 / sp.max(1) as f64)
                })
                .collect();
            let values: Vec<String> = (0..5).map(|b| format!("{:.2}/{:.2}", to_f32(speak_bg.value(&speak_code(speak_ctx(b, 7, true), 0))), to_f32(speak_bg.value(&speak_code(speak_ctx(b, 7, true), 1))))).collect();
            if cortex_fam_on || kernel_fam_on {
                let f = |n: &str| index.get(n).map_or(0, |&w| if cortex_fam_on { word_count[w] } else { cortex_fam.get(w).copied().unwrap_or(0) });
                eprintln!(
                    "  SPEECH seed {seed}: cortical familiarity ({}) at the end of the test: new names {}; trained names {}",
                    if cortex_fam_on { "exposure" } else { "kernels keyed on the word" },
                    NEW_NAMES.iter().map(|n| format!("{n} {}", f(n))).collect::<Vec<_>>().join(", "),
                    ["john", "mary", "anna", "daniel"].iter().map(|n| format!("{n} {}", f(n))).collect::<Vec<_>>().join(", ")
                );
            }
            let part = |x: [usize; 3]| format!("spoke {:.0}% of {}, {:.1}% right", 100.0 * x[1] as f64 / x[0].max(1) as f64, x[0], 100.0 * x[2] as f64 / x[1].max(1) as f64);
            eprintln!(
                "  SPEECH seed {seed} (context: {}): sources agreed: {}; disagreed: {}; by novelty band (0 = newest): {}",
                if speak_ctx_cfg.is_empty() { "confidence" } else { speak_ctx_cfg.as_str() },
                part(sp_agree[1]),
                part(sp_agree[0]),
                (0..4).map(|b| format!("{b}: {}", part(sp_novel[b]))).collect::<Vec<_>>().join("; ")
            );
            eprintln!("  SPEECH seed {seed}: at test questions, by the mix's confidence band: {}; value silent/speak per band {}", bands.join(", "), values.join(" "));
        }
        if efference_on || recite.is_some() || speak {
            eprintln!(
                "  EFFERENCE seed {seed} ({}): {} own words heard; {} altered, {} of those caught as surprising; {} unaltered own words were surprising",
                if efference_on { "copy on" } else { "copy off" },
                eff_stats[0],
                eff_stats[1],
                eff_stats[2],
                eff_stats[3]
            );
        }
        if speak {
            let line = |tag: Option<u8>| {
                let (sp, r, n) = speech.score(tag);
                format!("{} questions, spoken {:.1}%, right {:.1}% of spoken, {:.1}% of all", n, 100.0 * sp as f64 / n.max(1) as f64, 100.0 * r as f64 / sp.max(1) as f64, 100.0 * r as f64 / n.max(1) as f64)
            };
            eprintln!("  SPEAK seed {seed}: held out: {}; trained: {}", line(Some(1)), line(Some(0)));
            let curve: Vec<String> = [0.0, 0.5, 0.7, 0.8, 0.9]
                .iter()
                .map(|&th| {
                    let (sp, r, n) = speech.score_at(q16(th), Some(1));
                    format!(">= {th}: {:.0}% answered, {:.1}% right", 100.0 * sp as f64 / n.max(1) as f64, 100.0 * r as f64 / sp.max(1) as f64)
                })
                .collect();
            eprintln!("  SPEAK seed {seed}: held out, by threshold: {}", curve.join("; "));
            // what the wrong answers were: the end of the sentence ("."), another place, or else
            let (mut dot, mut place, mut other) = (0, 0, 0);
            for sp in speech.said().iter().filter(|x| x.tag == 1 && x.word.is_some() && x.word != x.truth) {
                let w = sp.word.unwrap();
                if vocab[w] == "." {
                    dot += 1;
                } else if PLACES.contains(&vocab[w]) {
                    place += 1;
                } else {
                    other += 1;
                }
            }
            eprintln!("  SPEAK seed {seed}: held-out wrong answers: \".\" {dot}, another place {place}, another word {other}");
            for tr in &transcripts {
                eprintln!("  SPEAK seed {seed}: {tr}");
            }
        }
        if gate_learned || sparse_hc {
            let bands: Vec<String> = (0..5).map(|b| format!("band {b}: asked {:.1}% of {}", 100.0 * mg_stats[b][1] as f64 / mg_stats[b][0].max(1) as f64, mg_stats[b][0])).collect();
            eprintln!("  GATE seed {seed}: at test, by the column's own confidence band: {}; hippocampal answers withheld {}", bands.join(", "), hc_withheld);
        }
        if cue_ctl {
            let pct = |a: usize, b: usize| if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 };
            let tr: usize = cue_train.iter().sum();
            let te: usize = cue_stats.iter().map(|x| x[0]).sum();
            let names = ["as is", "walk", "content only", "focus"];
            eprintln!(
                "  CUE seed {seed}: training choices {}; at test {}",
                (0..4).map(|a| format!("{} {:.1}%", names[a], pct(cue_train[a], tr))).collect::<Vec<_>>().join(", "),
                (0..4).map(|a| format!("{} {:.1}% ({:.1}% gave the next word)", names[a], pct(cue_stats[a][0], te), pct(cue_stats[a][1], cue_stats[a][0]))).collect::<Vec<_>>().join(", ")
            );
            // the learned policy: the preferred edit per (confidence band, familiarity band)
            let mut pol = String::new();
            for c in 0..5 {
                pol.push_str(&format!(" conf{c}:"));
                for f in 0..8 {
                    let best = (0..4).max_by_key(|&a| cue_bg.value(&cue_code(c * 8 + f, a))).unwrap();
                    pol.push(['A', 'W', 'C', 'F'][best]);
                }
            }
            eprintln!("  CUE seed {seed}: policy (per familiarity band 0..7; A as is, W walk, C content only, F focus){pol}");
        }
        if graded {
            eprintln!("  GRADED seed {seed}: {} steps with a semantic-store answer in the mix", graded_stats[0]);
        }
        if cooperate {
            eprintln!("  COOPERATE seed {seed}: {} uncertain steps answered from the semantic store ({} in held-out test stories)", coop_stats[0], coop_stats[1]);
        }
        if infer_reps.is_some() {
            eprintln!("  INFER seed {seed}: {} inferred events ({} as word pairs taught to the higher area; {} words read as replay stories)", infer_stats[0], infer_stats[1], replay_words);
        }
        if hippo_self {
            eprintln!(
                "  HIPPO_SELF seed {seed}: {} cue-free replays, {} decoded with content, {} of those cued by a new name, {} tagged events marked consolidated",
                sem_hstats[0], sem_hstats[1], sem_hstats[2], sem_hstats[3]
            );
        }
        eprintln!(
            "  MEMORY seed {seed}: training text {} bytes ({} words); hippocampus {} bytes; list memory {} bytes ({} episodes)",
            train_text_bytes,
            train_words,
            bind_hc.as_ref().map_or(0, |h| h.memory_bytes()),
            bind_mem.bytes(),
            bind_mem.len()
        );
        if let Some(hc) = &bind_hc {
            let r = hc.report();
            if !r.is_empty() {
                eprintln!("  CIRCUIT seed {seed}: {r}");
            }
        }
        if let Some(hc) = &bind_hc {
            let (hits, all, nov_sum, nov_n) = hc.stats();
            eprintln!(
                "  HIPPO seed {seed}: {} episodes stored, mean novelty {:.2}; recalls {} ({:.0}% answered from the cache, no change in the cue)",
                hc.len(),
                to_f32((nov_sum / nov_n.max(1) as u64) as Q16),
                all,
                100.0 * hits as f64 / all.max(1) as f64
            );
        }
        if rollout_loop {
            eprintln!("  LOOP seed {seed}: at test {} fed-back vectors, {} exactly one word's code, {} a blend of several words", loop_stats[0], loop_stats[1], loop_stats[2]);
        }
        if step_learned {
            let src = ["slot memory", "semantic store", "higher area", "column"];
            let parts: Vec<String> = (0..4).filter(|&i| step_stats[i][0] > 0).map(|i| format!("{} {}/{}", src[i], step_stats[i][1], step_stats[i][0])).collect();
            eprintln!("  STEP seed {seed}: at test, steps taken / offered by source: {}", parts.join(", "));
            // learned values: start a rollout vs read on, per source and definite expectation
            // (averaged over confidence bands and novelty)
            let mut vals = Vec::new();
            for (i, name) in src.iter().enumerate() {
                for def in 0..2 {
                    let (mut a, mut b) = (0.0, 0.0);
                    for cb in 0..3 {
                        for nov in 0..2 {
                            let ctx = def + 2 * cb + 12 * nov + 24 * i;
                            a += to_f32(step_bg.value(&step_code(ctx, 0)));
                            b += to_f32(step_bg.value(&step_code(ctx, 1)));
                        }
                    }
                    vals.push(format!("{name}{} {:.2}/{:.2}", if def == 1 { " (definite)" } else { "" }, a / 6.0, b / 6.0));
                }
            }
            eprintln!("  STEP seed {seed}: value of looking again / reading on: {}", vals.join(", "));
        }
        if rel_reps.is_some() {
            let (k, n) = rel.stats();
            let fr: Vec<String> = rel.frames().iter().take(16).map(|f| f.iter().map(|&w| vocab[w]).collect::<Vec<_>>().join(" ")).collect();
            let names: Vec<String> = NEW_NAMES
                .iter()
                .filter_map(|n| index.get(n))
                .map(|&w| format!("{}: {:?}", vocab[w], rel.about(&enc.codes, w).iter().map(|a| format!("{}>{}", a.0, vocab[a.3])).collect::<Vec<_>>()))
                .collect();
            eprintln!("  REL seed {seed}: {} facts parsed, {} relations, {} kernels from {} replays; frames {:?}", rel_stats[0], rel.frames().len(), k, n, fr);
            eprintln!("  REL seed {seed}: about the new names: {}", names.join("; "));
            if let Some(k) = narrators {
                let isa = rel.relation_for(&[index["is"], index["a"]]);
                let beliefs: Vec<String> = NEW_NAMES
                    .iter()
                    .filter_map(|n| index.get(n))
                    .map(|&w| {
                        let c = isa.map(|r| rel.claims(w, r, 0, 1)).unwrap_or_default();
                        format!("{}: {}", vocab[w], c.iter().map(|&(v, src, b)| format!("{} by {} ({:.2})", vocab[v], src, to_f32(b))).collect::<Vec<_>>().join(", "))
                    })
                    .collect();
                eprintln!(
                    "  TRUST seed {seed}: {} lies told; trust per narrator: {}; claims about the new names: {}",
                    lies_told,
                    (1..=k as u16).map(|n| format!("{n}{} {:.2} (on families {:.2})", if n as usize == k { " (liar)" } else { "" }, to_f32(rel.trust(n)), to_f32(isa.map_or(0, |r| rel.bayes.trust_in(n, r as u64))))).collect::<Vec<_>>().join(", "),
                    beliefs.join("; ")
                );
                // the believed family of each new name, the relation store's answer, and the
                // truth
                let verdicts: Vec<String> = NEW_NAMES
                    .iter()
                    .enumerate()
                    .filter_map(|(i, n)| index.get(n).map(|&w| (i, w)))
                    .map(|(i, w)| {
                        let truth = SURNAMES[family_of(i, true)];
                        let believed = isa.and_then(|r| rel.bayes.believed(&(r, w, 0, 1))).map_or("-", |v| vocab[v]);
                        let stored = isa.and_then(|r| rel.ask(&enc.codes, w, r, 0, 1)).map_or("-", |v| vocab[v]);
                        format!("{}: believed {believed}, store {stored}, true {truth}", vocab[w])
                    })
                    .collect();
                eprintln!("  BELIEF seed {seed}: {}", verdicts.join("; "));
            }
            for fam in ["jones", "smith"] {
                if let Some(&w) = index.get(fam) {
                    let a: Vec<String> = rel.about(&enc.codes, w).iter().map(|&(r, i, j, _)| format!("{r}:{i}>{j} {:?}", rel.ask_all(&enc.codes, w, r, i, j).iter().map(|&x| vocab[x]).collect::<Vec<_>>())).collect();
                    eprintln!("  REL seed {seed}: about {fam}: {}", a.join("; "));
                }
            }
            eprintln!("  REL seed {seed}: {} relations of relations learned ({:?}), {} facts inferred and replayed", rel.rules().len(), rel.rules().iter().take(6).map(|r| format!("{} = {:?} then {:?} ({}/{})", r.relation, r.first, r.second, r.confirmed, r.applicable)).collect::<Vec<_>>(), rel.inferred());
        }
        if semantic_reps.is_some() {
            eprintln!(
                "  SEMANTIC seed {seed}: {} kernels from {} replays; rollout steps taken from the store {} ({} in held-out stories)",
                sem_store.len(),
                sem_replays,
                sem_used[0],
                sem_used[1]
            );
            if sem_frame {
                let mut fr: Vec<String> = frame_names.values().map(|fb| fb.iter().map(|&(w, c)| format!("{}@{}", vocab[w], c)).collect::<Vec<_>>().join(" ")).collect();
                fr.sort();
                let per: Vec<usize> = sem_frames.values().map(|v| v.len()).collect();
                eprintln!(
                    "  FRAMES seed {seed}: {} frames learned, {} cues, {:.2} frames per cue; e.g. {:?}",
                    frame_names.len(),
                    per.len(),
                    per.iter().sum::<usize>() as f64 / per.len().max(1) as f64,
                    fr.iter().take(12).collect::<Vec<_>>()
                );
            }
        }
        if complete.is_some() {
            let mut cw: Vec<_> = complete_words.iter().collect();
            cw.sort_by(|a, b| b.1.cmp(a.1));
            eprintln!(
                "  COMPLETE seed {seed}: held-out answers right: no surname supplied {}/{}, right family {}/{}, wrong family {}/{}; completions in training {}, at test {} (trained names) and {} (held out); held out: {:?}",
                by_surname[0].0,
                by_surname[0].1,
                by_surname[1].0,
                by_surname[1].1,
                by_surname[2].0,
                by_surname[2].1,
                complete_stats[0],
                complete_stats[1],
                complete_stats[2],
                cw.iter().filter(|x| NEW_NAMES.iter().any(|n| x.0.contains(n))).take(16).collect::<Vec<_>>()
            );
        }
        if question() {
            eprintln!("  QERR seed {seed}: held-out answers: right {}, the family's place for another season {}, the other family's place for this season {}, other {}, no place {}", q_err[0], q_err[1], q_err[2], q_err[3], q_err[4]);
        }
        if ne_on {
            eprintln!("  NE seed {seed}: {} salient sentences stored twice; mean exploration in training {:.2}, final {:.2}", ne_stats[0], ne_stats[2] as f64 / (ne_stats[1].max(1) as f64 * ONE as f64), to_f32(step_bg.explore));
        }
        if ach_on {
            if bind_lesion {
                eprintln!("  ACH seed {seed}: hippocampus lesioned at test (no recall, no reinstatement, no update); mean level in the last training stories {:.2}", to_f32(ach));
            } else {
                eprintln!("  ACH seed {seed}: mean level at test {:.2}", ach_stats[1] as f64 / (ach_stats[0].max(1) as f64 * ONE as f64));
            }
        }
        if reinstate && bind_lesion {
            eprintln!("  REINSTATE seed {seed}: hippocampus lesioned at test: reinstatement in training only");
        } else if reinstate {
            eprintln!("  REINSTATE seed {seed}: at test, a cortical state reinstated at {} steps (mean {} bits of context with it)", reinstate_stats[0], reinstate_stats[2] / reinstate_stats[0].max(1));
        }
        if qhold.is_some() {
            eprintln!("  QHOLD seed {seed}: in held-out test stories, an item held at {} answers ({} the stranger); {} events stored with it, {} of them recalled for the answer; queries {} (an event found {}, holding a surname {}); held: {:?}", hold_stats[0], hold_stats[1], hold_stats[2], hold_stats[3], query_stats[0], query_stats[1], query_stats[2], { let mut v: Vec<_> = held_words.iter().collect(); v.sort_by(|a, b| b.1.cmp(a.1)); v.into_iter().take(8).collect::<Vec<_>>() });
        }
        if inner_speech {
            eprintln!("  INNER seed {seed}: at test, {} surprises where the network could speak, {} spoken to itself ({} in held-out stories)", inner_stats[0], inner_stats[1], inner_stats[2]);
        }
        for (name, l) in [("L23", bit23.as_ref()), ("L5", bit5.as_ref())] {
            if let Some(l) = l {
                eprintln!(
                    "  {name} seed {seed}: bitwise, {} committed cells; context threshold {:.2}; bursts {}, spikes {}, nothing {}; commitments {}, freed {}",
                    l.committed(), to_f32(l.threshold()), l.stats[0], l.stats[1], l.stats[2], l.stats[3], l.stats[4]
                );
            }
        }
        if let Some(l) = primed23.as_ref() {
            eprintln!(
                "  L23 seed {seed}: primed, {} committed cells; context threshold {:.2}; bursts {}, spikes {}, nothing {}; the old kernels predicted {} of {} steps; commitments {}, freed {}",
                l.committed(), to_f32(l.threshold()), l.stats[0], l.stats[1], l.stats[2], l23_stats[1], l23_stats[0] + l23_stats[1], l.stats[3], l.stats[4]
            );
        }
        if let Some(l) = primed5.as_ref() {
            eprintln!(
                "  L5 seed {seed}: primed, {} committed cells; context threshold {:.2} (SST {:.2}, VIP {:.2}); bursts won {}, spikes {}, nothing fired {}; commitments {}, freed {}",
                l.committed(), to_f32(l.threshold()), to_f32(l.interneuron_state().0), to_f32(l.interneuron_state().1), l.stats[0], l.stats[1], l.stats[2], l.stats[3], l.stats[4]
            );
        }
        if let Some(l) = layer5.as_ref() {
            eprintln!("  L5 seed {seed}: {} cells; a burst decided {} predictions, none at {}; grown {}, recycled {}", l.cells(), l.stats[0], l.stats[1], l.stats[2], l.stats[3]);
        }
        if let Some((th, passed, failed)) = column.l23.burst_stats() {
            eprintln!("  COMPETE seed {seed}: burst threshold {:.2}; a burst won {passed} predictions, nothing burst at {failed}", to_f32(th));
        }
        if burst_gate {
            let mut b: Vec<(u8, f32)> = src_burst.iter().map(|(&k, &v)| (k, (to_f32(v) * 100.0).round() / 100.0)).collect();
            b.sort_by_key(|x| x.0);
            eprintln!(
                "  GATE seed {seed}: threshold {:.2}; at test answers {} gated, {} passed, {} to the cerebellum, {} ungated; burst rates {:?}",
                to_f32(gate_theta), gate_stats[0], gate_stats[1], gate_stats[2], gate_stats[3], b
            );
        }
        if std::env::var("COMPETE").map_or(false, |v| v == "evidence") {
            eprintln!("  COMPETE seed {seed}: learned depth gains {:?}", column.l23.depth_gains().iter().map(|&g| (to_f32(g) * 100.0).round() / 100.0).collect::<Vec<_>>());
        }
        if std::env::var("OWNDIAG").is_ok() {
            eprintln!("  OWNDIAG seed {seed}: at {} test answers the column decoded a word at {} and output any bits at {}; matched kernels {} (a place {}, the answer {}); some matched kernel proposes the answer at {} answers", own_diag[0], own_diag[1], own_diag[2], own_diag_k[0], own_diag_k[1], own_diag_k[2], own_diag_k[3]);
        }
        if let Some(a) = assoc.as_ref() {
            eprintln!("  ASSOC seed {seed}: {} kernels; {} replays taught it; at test answers it proposed a word at {} and was right at {}", a.column.l23.kernels().len(), assoc_stats[2], assoc_stats[0], assoc_stats[1]);
        }
        if da_on {
            eprintln!("  DA seed {seed}: {} traces, mean dopamine {:.2}; {} chosen for replay over all sleeps", da_stats[0], da_stats[1] as f64 / (da_stats[0].max(1) as f64 * ONE as f64), da_stats[2]);
        }
        if consolidate.is_some() {
            eprintln!("  CONSOLIDATE seed {seed}: {} replays of novel episodes to the higher area; {} to the column (sleep-gated)", replayed, sleep_column);
        }
        if mixing && held_halves[0].1 > 0 {
            eprintln!(
                "  HALVES seed {seed}: held-out answers right in the first half of the test {:.1}%, second half {:.1}%",
                100.0 * held_halves[0].0 as f64 / held_halves[0].1 as f64,
                100.0 * held_halves[1].0 as f64 / held_halves[1].1.max(1) as f64
            );
        }
        if bind {
            eprintln!(
                "  BIND seed {seed}: {} slot cells, {} episodes; at test the slot memory answered {} of {} answers ({:.1}% of those right); held-out answers: {:.1}% right of {}",
                roles.used(),
                bind_mem.len(),
                bind_stats.0,
                TEST,
                100.0 * bind_stats.1 as f64 / bind_stats.0.max(1) as f64,
                100.0 * bind_new.1 as f64 / bind_new.0.max(1) as f64,
                bind_new.0
            );
        }
        if role_mode.as_deref() == Some("cells") {
            let mut cells: Vec<(&usize, u32)> = role_words.iter().map(|(c, m)| (c, m.values().sum())).collect();
            cells.sort_by(|a, b| b.1.cmp(&a.1));
            let parts: Vec<String> = cells
                .iter()
                .take(8)
                .map(|(c, n)| {
                    let mut ws: Vec<(&usize, &u32)> = role_words[c].iter().collect();
                    ws.sort_by(|a, b| b.1.cmp(a.1));
                    let top: Vec<String> = ws.iter().take(5).map(|(w, k)| format!("{} {}", vocab[**w], k)).collect();
                    format!("cell {c} ({n}): {}", top.join(", "))
                })
                .collect();
            eprintln!("  ROLES seed {seed}: {} cells recruited; what filled each cell's slot at test: {}", roles.used(), parts.join(" | "));
        }
        if learned_bound {
            eprintln!(
                "  EVENT_BOUNDARY seed {seed}: at test, {} boundaries, {} of them at \".\" ({:.1}%); {:.1}% of the {} periods closed an event",
                bound_stats[0],
                bound_stats[1],
                100.0 * bound_stats[1] as f64 / bound_stats[0].max(1) as f64,
                100.0 * bound_stats[1] as f64 / bound_stats[2].max(1) as f64,
                bound_stats[2]
            );
        }
        if saccade.is_some() && new_wording() {
            let f = |k: usize| {
                let (n, l, r) = sacc_wording[k];
                format!("{:.1}% right, looked back at the season before {:.1}% of {n} answers", 100.0 * r as f64 / n.max(1) as f64, 100.0 * l as f64 / n.max(1) as f64)
            };
            eprintln!("  WORDING seed {seed}: trained wording: {}; new wording: {}", f(0), f(1));
        }
        if bound_detect {
            eprintln!(
                "  BOUNDARY seed {seed}: at test, boundaries detected at {} of {} story openings ({:.1}%) and at {} other sentences",
                bound_hits.0,
                story_openings,
                100.0 * bound_hits.0 as f64 / story_openings.max(1) as f64,
                bound_hits.1
            );
        }
        if readback.is_some() && readback_stats.0 > 0 {
            eprintln!(
                "  READBACK seed {seed}: at test, said back the most recent rare word held at {:.1}% of {} read-backs",
                100.0 * readback_stats.1 as f64 / readback_stats.0 as f64,
                readback_stats.0
            );
        }
        if task == Task::Books {
            let names = ["first session", "back to back", "after 1-2 others", "after 3+ others"];
            let parts: Vec<String> = book_bins
                .iter()
                .zip(names)
                .map(|(b, n)| if b.0 > 0 { format!("{n}: {:.0}% of {}", 100.0 * b.1 as f64 / b.0 as f64, b.0) } else { format!("{n}: -") })
                .collect();
            eprintln!("  BOOKS seed {seed}: accuracy by session: {}; at test \"@open\" chose keep {} / reset {} / reinstate {}", parts.join(", "), ctx_chosen[0], ctx_chosen[1], ctx_chosen[2]);
        }
        if task == Task::Season {
            let names = ["0", "1", "2-3", "4-7", "8-15", "16+"];
            let parts: Vec<String> = season_bins
                .iter()
                .zip(names)
                .map(|(b, n)| if b.0 > 0 { format!("{n}: {:.0}% of {}", 100.0 * b.1 as f64 / b.0 as f64, b.0) } else { format!("{n}: -") })
                .collect();
            eprintln!("  SEASON seed {seed}: accuracy by filler stories since the season was announced: {}", parts.join(", "));
        }
        if hier && area.column.l23.slept_general() != (0, 0) {
            let (m, rj) = area.column.l23.slept_general();
            eprintln!("  SLEEPGEN seed {seed}: the higher area formed {m} general rules during sleep; {rj} candidates failed the replay test");
        }
        if hier && area.column.l23.spawned() > 0 {
            eprintln!("  SPAWN seed {seed}: the higher area spawned {} general kernels ({} copies skipped as already covered)", area.column.l23.spawned(), area.column.l23.spawn_subsumed());
        }
        if hier {
            eprintln!(
                "  HIER seed {seed}: higher area {} kernels; its top-down frame held the answer at {:.1}% of test answers; top-down passed the gate at {:.1}% of answers",
                area.column.l23.live(),
                100.0 * topdown_has_answer as f64 / TEST as f64,
                100.0 * topdown_passed_at_answer as f64 / TEST as f64
            );
        }
        if column.l23.expected_steps() > 0 {
            eprintln!(
                "  SURPRISE seed {seed}: {} of {} training words were expected (winner right): confirmed only",
                column.l23.expected_steps(),
                train_words
            );
        }
        if sleeps > 0 || column.l23.gated_growth() > 0 {
            eprintln!(
                "  SLEEP seed {seed}: {sleeps} sleeps pruned {slept_pruned} and merged {slept_merged} kernels; growth suppressed by the uncertainty gate {} times; {} live kernels",
                column.l23.gated_growth(),
                column.l23.live()
            );
        }
        let dense: usize = kernels.iter().map(|k| k.connection_bytes()).sum();
        let set: usize = kernels.iter().map(|k| k.input_set.len() + k.output_set.len()).sum();
        let sparse = set * 2 + kernels.len() * 8; // u16 bit indices + offsets, threshold, 2 × u8 counters
        eprintln!(
            "  COST seed {seed}: {} live kernels; connections stored {:.2} MB, {} set bits ({:.2} MB as u16 indices, + {:.2} MB inverted index); {} episodes × {} B; train {:.1} µs/word over {} words, test {:.1} µs/word",
            column.l23.live(),
            dense as f64 / 1e6,
            set,
            sparse as f64 / 1e6,
            (set * 4) as f64 / 1e6,
            memory.len(),
            BITS / 8,
            1e6 * train_secs / train_words.max(1) as f64,
            train_words,
            1e6 * test_secs / test_words.max(1) as f64
        );
    }
    if mixing && mix_stats[0] > 0 {
        let pct = |r: usize, n: usize| if n > 0 { format!("{:.1}%", 100.0 * r as f64 / n as f64) } else { "-".to_string() };
        eprintln!(
            "  MIX seed {seed}: test answers {}: column alone {} right, mixed {} right; mix changed the column's answer at {} ({} of those right); all sources agreed at {} ({} right); {} reliability entries",
            mix_stats[0],
            pct(mix_stats[1], mix_stats[0]),
            pct(mix_stats[2], mix_stats[0]),
            pct(mix_stats[3], mix_stats[0]),
            pct(mix_stats[4], mix_stats[3]),
            pct(mix_stats[5], mix_stats[0]),
            pct(mix_stats[6], mix_stats[5]),
            mix.len()
        );
    }
    {
        // CALIB: does confidence predict correctness? Accuracy per confidence bucket, the
        // expected calibration error (answer-weighted |accuracy − mean confidence|), and
        // accuracy / coverage if the column abstained below 0.8
        let n: usize = calib.iter().map(|b| b.0).sum();
        if n > 0 {
            let names = ["<.5", ".5-.7", ".7-.8", ".8-.9", ">=.9"];
            let parts: Vec<String> = calib
                .iter()
                .zip(names)
                .map(|(b, name)| if b.0 == 0 { format!("{name} -") } else { format!("{name} {:.0}% of {}", 100.0 * b.1 as f64 / b.0 as f64, b.0) })
                .collect();
            let ece: f64 = calib.iter().filter(|b| b.0 > 0).map(|b| (b.1 as f64 - b.2).abs()).sum::<f64>() / n as f64;
            let (cn, cr) = calib[3..].iter().fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
            eprintln!(
                "  CALIB seed {seed}: {}; ECE {:.3}; answering only at >= .8: {:.1}% right, {:.1}% answered",
                parts.join(", "),
                ece,
                if cn > 0 { 100.0 * cr as f64 / cn as f64 } else { 0.0 },
                100.0 * cn as f64 / n as f64
            );
        }
    }
    if ask > 0 {
        let (open, searches, mean) = curiosity.stats();
        let kind = |w: usize| if NEW_NAMES.contains(&vocab[w]) { 0 } else if PRACTICE_NAMES.contains(&vocab[w]) { 1 } else { 2 };
        let mut by_kind = [0usize; 3];
        for &(w, _) in &asked_log {
            by_kind[kind(w)] += 1;
        }
        let new_asked: Vec<String> = asked_log.iter().filter(|x| kind(x.0) == 0).map(|&(w, sl)| format!("{} (sleep {sl})", vocab[w])).collect();
        let ctxs: Vec<String> = curiosity.learned().iter().map(|&(c, n, g)| format!("belief {}/8 lead {}/16: {} asked, gain {:.2}, policy value {:.2}", c / 8, c % 8, n, to_f32(g), to_f32(curiosity.learned_value(c)))).collect();
        let (stops, spent) = curiosity.spending();
        eprintln!(
            "  CURIOSITY seed {seed}: compute {} replays ({:.1} per question); energy spent {:.2} reserves, stops chosen {}, energy left {:.2}",
            work_total,
            work_total as f64 / searches.max(1) as f64,
            spent as f64 / ONE as f64,
            stops,
            to_f32(energy)
        );
        eprintln!(
            "  CURIOSITY seed {seed}: {} questions asked ({} about new names, {} practice, {} trained names), mean gain {:.2}, {} open at the end; new names asked: {}; learned value by context: {}",
            searches,
            by_kind[0],
            by_kind[1],
            by_kind[2],
            to_f32(mean),
            open,
            new_asked.join(", "),
            ctxs.join("; ")
        );
    }
    if belief_q() && narrators.is_some() {
        let parts: Vec<String> = NEW_NAMES
            .iter()
            .zip(unknown_tally)
            .map(|(n, u)| {
                let p = |x: usize| 100.0 * x as f64 / u[0].max(1) as f64;
                format!("{n}: right {:.0}%, wrong {:.0}%, unknown {:.0}%", p(u[1]), p(u[2]), p(u[3]))
            })
            .collect();
        eprintln!("  UNKNOWN seed {seed}: family questions {}", parts.join("; "));
        if unknown_learned {
            let bands: Vec<String> = (0..64)
                .filter(|&b| quiz_stats[b][0] > 0)
                .map(|b| {
                    let q = quiz_stats[b];
                    format!(
                        "belief {}/8 lead {}/16: {} quizzes, answered {:.0}% ({:.0}% of those right), now {}",
                        b / 8,
                        b % 8,
                        q[0],
                        100.0 * q[1] as f64 / q[0] as f64,
                        100.0 * q[2] as f64 / q[1].max(1) as f64,
                        if unknown_bg.value(&unknown_code(b, true)) > unknown_bg.value(&unknown_code(b, false)) { "answers" } else { "says unknown" }
                    )
                })
                .collect();
            eprintln!("  UNKNOWN seed {seed}: learned go/no-go by belief and lead: {}", bands.join("; "));
            let ctx: Vec<String> = NEW_NAMES
                .iter()
                .filter_map(|n| index.get(n).map(|&w| (n, w)))
                .filter_map(|(n, w)| rel.relation_for(&[index["is"], index["a"]]).and_then(|r| decisiveness(&rel, w, r)).map(|(_, c)| format!("{n}: belief {}/8 lead {}/16", c / 8, c % 8)))
                .collect();
            eprintln!("  UNKNOWN seed {seed}: the new names' contexts: {}", ctx.join("; "));
            if let Some(isa) = rel.relation_for(&[index["is"], index["a"]]) {
                let pn: Vec<String> = PRACTICE_NAMES
                    .iter()
                    .enumerate()
                    .filter_map(|(j, n)| index.get(n).map(|&w| (j, n, w)))
                    .map(|(j, n, w)| {
                        let c = rel.claims(w, isa, 0, 1);
                        format!("{n} (kind {}, true {}): {}", j % 3, SURNAMES[practice_truth[j]], c.iter().map(|&(v, src, b)| format!("{} by {} {:.2}", vocab[v], src, to_f32(b))).collect::<Vec<_>>().join(", "))
                    })
                    .collect();
                eprintln!("  UNKNOWN seed {seed}: practice claims: {}", pn.join("; "));
            }
        }
    }
    if belief_q() {
        let pc = |x: (usize, usize)| 100.0 * x.0 as f64 / x.1.max(1) as f64;
        eprintln!(
            "  BELIEF_Q seed {seed}: new names' place questions {:.1}% of {} ({}), family questions {:.1}% of {}",
            pc(belief_tally[0]),
            belief_tally[0].1,
            NEW_NAMES.iter().zip(belief_names).map(|(n, x)| format!("{n} {:.0}%", pc(x))).collect::<Vec<_>>().join(", "),
            pc(belief_tally[1]),
            belief_tally[1].1
        );
    }
    if hier_learned {
        let pc = |x: [usize; 2]| 100.0 * x[1] as f64 / (x[0] + x[1]).max(1) as f64;
        eprintln!("  HIERGATE seed {seed}: higher area consulted at {:.0}% of training steps, {:.0}% of test steps", pc(hier_choices[0]), pc(hier_choices[1]));
    }
    if hc_surprise || hier_surprise {
        eprintln!(
            "  GATED seed {seed}: hippocampal recalls reused {} of {}; higher-area steps reused {} of {}",
            gated_steps[0],
            gated_steps[0] + gated_steps[1],
            gated_steps[2],
            gated_steps[2] + gated_steps[3]
        );
    }
    if dump.is_none() {
        let total: f64 = prof.iter().sum();
        let names = ["L4 assembly (recall, relays, gates)", "L2/3 prediction", "evaluation and rewards", "L2/3 learning", "storage and after-learning", "observe / surprise / bag"];
        let parts: Vec<String> = names.iter().zip(prof).map(|(n, p)| format!("{n} {:.0}%", 100.0 * p / total)).collect();
        eprintln!("  PROF seed {seed}: {:.0} s in the word loop: {}", total, parts.join(", "));
    }
    Outcome { seen: pct(seen), held_out: pct(held), recall: 100.0 * recall_has_answer as f64 / TEST as f64, places: recall_places as f64 / TEST as f64 }
}

/// An environment setting as an integer ratio, read from its text with no float in
/// between: "p/q", or a decimal such as "0.5" (= 5/10) or "0.75" (= 75/100).
fn ratio_env(name: &str) -> Option<(u16, u16)> {
    let v = std::env::var(name).ok()?;
    if let Some((p, q)) = v.split_once('/') {
        return Some((p.trim().parse().ok()?, q.trim().parse().ok()?));
    }
    let (int, frac) = v.split_once('.').unwrap_or((v.as_str(), ""));
    let den = 10u16.checked_pow(frac.len() as u32)?;
    let int: u16 = if int.is_empty() { 0 } else { int.parse().ok()? };
    let frac: u16 = if frac.is_empty() { 0 } else { frac.parse().ok()? };
    Some((int.checked_mul(den)?.checked_add(frac)?, den))
}

/// Familiarity of a binding: the full hippocampus's own count when it runs on its own
/// (HIPPO_SELF), else the list store's.
fn fam_count(hc: &Option<Box<dyn EpisodicCircuit>>, mem: &EpisodicMemory, own: bool, b: &BitVector) -> u64 {
    match (own, hc) {
        (true, Some(h)) => h.familiarity(&set_bits(b)),
        _ => mem.count_of(b),
    }
}

/// SPARSE_BIND: the hippocampus's input index of a binding: word code `code` in slot
/// `slot`, in the slot's own 8,192-bit field (context bindings in a second set of fields),
/// so a bit belongs to essentially one binding.
const SPARSE_FIELDS: usize = 64;
/// QHOLD: the field of the item held in working memory (a content field no sentence reaches).
const HELD_FIELD: usize = SPARSE_FIELDS - 1;

fn engram_mode() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("HIPPO").map_or(false, |v| v == "engram"))
}

fn sparse_binding(code: &BitVector, w: usize, slot: usize, context: bool) -> Vec<usize> {
    // HIPPO=engram: one id per binding (slot × 4096 + word); the story context is the
    // store's place code, not stored bindings
    if engram_mode() {
        // the story's earlier bindings are stored and cued too, as ids of their own
        // (context slot fields); ENGRAM_NO_CONTEXT=1: content only (the place code alone)
        if context && std::env::var("ENGRAM_NO_CONTEXT").is_ok() {
            return Vec::new();
        }
        return vec![((slot % SPARSE_FIELDS) + if context { SPARSE_FIELDS } else { 0 }) * 4096 + w];
    }
    let field = (slot % SPARSE_FIELDS) + if context { SPARSE_FIELDS } else { 0 };
    set_bits(code).into_iter().map(|b| field * BITS + b).collect()
}

/// Familiarity of the binding (`code` in `slot`, rotated form `b`): from the sparse binding
/// space when SPARSE_BIND is on, else as `fam_count`.
fn fam_binding(hc: &Option<Box<dyn EpisodicCircuit>>, mem: &EpisodicMemory, own: bool, sparse: bool, b: &BitVector, code: &BitVector, w: usize, slot: usize) -> u64 {
    match (sparse && own, hc) {
        (true, Some(h)) => h.familiarity(&sparse_binding(code, w, slot, false)),
        _ => fam_count(hc, mem, own, b),
    }
}

/// Episodes stored so far (the full hippocampus's count when it runs on its own).
fn mem_size(hc: &Option<Box<dyn EpisodicCircuit>>, mem: &EpisodicMemory, own: bool) -> usize {
    match (own, hc) {
        (true, Some(h)) => h.len(),
        _ => mem.len(),
    }
}

/// An event (HIPPO_SELF): the current sentence's bindings, plus the story's earlier
/// bindings as context, held at a separate rotation (a lateral-EC context code), so they
/// cue and disambiguate but do not answer a slot's readout.
fn event_vec(sentence: &[BitVector], prev: &BitVector, ctx_offset: usize) -> BitVector {
    let mut e = BitVector::new(prev.bit_len(), Some(0));
    for b in sentence {
        e.or_mut(b);
    }
    let mut c = prev.clone();
    c.rotl_mut(ctx_offset);
    e.or_mut(&c);
    e
}

fn set_bits(bv: &BitVector) -> Vec<usize> {
    (0..bv.bit_len()).filter(|&b| bv.bit_get(b)).collect()
}

fn main() {
    let seeds: u64 = std::env::var("SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let seed_start: u64 = std::env::var("SEED_START").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let tasks: Vec<Task> = match std::env::var("TASK").as_deref() {
        Ok("short") => vec![Task::Short],
        Ok("long") => vec![Task::Long],
        Ok("varied") => vec![Task::Varied],
        Ok("twohop") => vec![Task::TwoHop],
        Ok("persist") => vec![Task::Persist],
        Ok("topic") => vec![Task::Topic],
        Ok("give") => vec![Task::Give],
        Ok("elim") => vec![Task::Elim],
        Ok("habit") => vec![Task::Habit],
        Ok("season") => vec![Task::Season],
        Ok("books") => vec![Task::Books],
        _ => vec![Task::Short, Task::Long, Task::Varied],
    };
    println!("answer accuracy on {TEST} test stories after {TRAIN} training stories (predictor learning off at test); chance 1/6");
    if std::env::var("NOVELTY").map_or(false, |v| v == "prediction") {
        println!("NOVELTY=prediction: CA1-style comparator; store, cue and read out only what the predictor did not predict");
    }
    for task in tasks {
        let fact_settings: &[usize] = if matches!(task, Task::TwoHop | Task::Persist | Task::Topic | Task::Give | Task::Elim | Task::Habit | Task::Season | Task::Books) { &[0] } else { &[2, 3] };
        for &max_facts in fact_settings {
            println!();
            if task == Task::Habit {
                println!("Habit stories: \"in the morning .\" / \"at night .\", 1-3 fillers, \"X went to the\" -> X's place for that time of day");
            } else if task == Task::Elim {
                println!("Elim stories: \"is it the P ? no .\" for 5 places, then \"is it the\" -> the 6th");
            } else if task == Task::Give {
                println!("Give stories: 2-3 \"X gave the O to Y .\"; \"what did X give ?\" -> O or \"who got the O ?\" -> Y");
            } else if task == Task::Topic {
                println!("Topic stories: \"X went to the P . <1-3 distractor sentences> where is the person ?\"");
            } else if task == Task::Persist {
                println!("Persist stories: anchors (bill, fred, julie) stated only in the first {} training stories; \"held-out\" = anchor questions at test", anchor_stories());
            } else if task == Task::TwoHop {
                println!("TwoHop stories: 2-3 people, 1-2 moves each, 1-2 objects picked up; \"where is the O ?\"");
            } else {
                println!("{task:?} stories, 1-{max_facts} facts, question about a random one");
            }
            let policies: Vec<Policy> = match std::env::var("POLICIES").as_deref() {
                Ok("loop") => vec![Policy::NoMemory, Policy::Episodic, Policy::Loop(1), Policy::Loop(2), Policy::Branch(3)],
                Ok("branch") => vec![Policy::Branch(3)],
                Ok("select") => vec![Policy::Select],
                Ok("bg") => vec![Policy::Loop(2), Policy::Branch(3), Policy::Select],
                Ok("twohop_learned") => vec![Policy::Branch(3), Policy::Select],
                Ok("gate") => vec![Policy::FixedRelay, Policy::Episodic, Policy::ThalamicGate],
                Ok("learned_gate") => vec![Policy::LearnedGate],
                Ok("thalamic_gate") => vec![Policy::ThalamicGate],
                Ok("l6") => vec![Policy::L6Gate { gated: false }, Policy::L6Gate { gated: true }],
                Ok("l6_gated") => vec![Policy::L6Gate { gated: true }],
                Ok("l6_open") => vec![Policy::L6Gate { gated: false }],
                Ok("consolidate") => vec![Policy::NoMemory, Policy::Episodic, Policy::Consolidate],
                Ok("consolidate_only") => vec![Policy::Consolidate],
                Ok("pfc") => vec![Policy::NoMemory, Policy::Episodic, Policy::Pfc { learned: false }, Policy::Pfc { learned: true }],
                Ok("pfc_learned") => vec![Policy::Pfc { learned: true }],
                Ok("episodic") => vec![Policy::Episodic],
                Ok("nomemory") => vec![Policy::NoMemory],
                Ok("ca1") => vec![Policy::Episodic, Policy::Loop(2), Policy::Branch(3), Policy::Ca3 { cells: 16384, k: 32, settle: 2 }],
                Ok("ca3_high") => vec![Policy::Ca3 { cells: 16384, k: 32, settle: 2 }],
                Ok("ca3_big") => vec![Policy::Ca3 { cells: 16384, k: 32, settle: 2 }, Policy::Ca3 { cells: 16384, k: 32, settle: 0 }],
                Ok("ca3") => vec![
                    Policy::Episodic,
                    Policy::Ca3 { cells: 1024, k: 64, settle: 2 },
                    Policy::Ca3 { cells: 16384, k: 32, settle: 2 },
                    Policy::Ca3 { cells: 16384, k: 32, settle: 0 },
                ],
                _ => vec![Policy::NoMemory, Policy::FixedRelay, Policy::Episodic],
            };
            let jobs: Vec<(Policy, u64)> = policies.iter().flat_map(|&p| (seed_start..seed_start + seeds).map(move |seed| (p, seed))).collect();
            let mut all = run_all(&jobs, task, max_facts).into_iter();
            for policy in policies {
                let runs: Vec<Outcome> = all
                    .by_ref()
                    .take(seeds as usize)
                    .map(|(o, out, err)| {
                        std::print!("{out}");
                        std::eprint!("{err}");
                        o
                    })
                    .collect();
                let mean = |f: fn(&Outcome) -> f64| runs.iter().map(f).sum::<f64>() / runs.len() as f64;
                println!(
                    "  {:<40} seen pairs {:5.1}%   held-out pairs {:5.1}%   answer in recall {:5.1}%   places in recall {:4.2}   held-out runs [{}]",
                    format!("{policy:?}"),
                    mean(|o| o.seen),
                    mean(|o| o.held_out),
                    mean(|o| o.recall),
                    mean(|o| o.places),
                    runs.iter().map(|o| format!("{:.0}", o.held_out)).collect::<Vec<_>>().join(" ")
                );
            }
        }
    }
}

/// A word's code bound to a role (rotated by the role's offset, as the slot bindings), or
/// the code itself without one.
fn sem_bind(code: &BitVector, slot: Option<usize>) -> BitVector {
    sem_bind_in(code, slot, None)
}

/// As `sem_bind`, the role keyed on a frame too (its offset added to the role's).
fn sem_bind_in(code: &BitVector, slot: Option<usize>, frame: Option<u64>) -> BitVector {
    let mut b = code.clone();
    if let Some(c) = slot {
        b.rotl_mut(bind_offset(c, frame, code.bit_len()));
    }
    b
}

fn bind_offset(c: usize, frame: Option<u64>, bits: usize) -> usize {
    let f = frame.map_or(0, |f| (f.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 20) as usize % bits);
    (((c + 1) * 2_654_435_761usize) % bits + f) % bits
}

/// The role a word holds in the current sentence's bindings.
fn slot_in(pairs: &[(usize, usize)], w: usize) -> Option<usize> {
    pairs.iter().rev().find(|x| x.0 == w).map(|x| x.1)
}

/// The semantic store's answer for word `w` (in role `slot` when relations are directed).
/// Untyped: the content as stored. Typed, `words`: every role (in each of the frames `w`
/// was stored with, when keyed on frames) unbound and the words found in it kept, as
/// plain codes; typed, not `words`: the role-bound content as is.
#[allow(clippy::too_many_arguments)]
fn sem_read(store: &KernelClass<SimpleKernel>, codes: &[BitVector], w: usize, slot: Option<usize>, typed: u8, roles: usize, words: bool, frames: &HashMap<usize, Vec<u64>>, rel: Option<(&RelationStore, usize)>, sentence: &[(usize, usize)]) -> Option<BitVector> {
    if let Some((rel, hops)) = rel {
        return rel_answer(rel, codes, w, sentence, hops);
    }
    if typed > 1 && slot.is_none() {
        return None;
    }
    let out = store.peek(&sem_bind(&codes[w], if typed > 1 { slot } else { None }))?;
    if typed == 0 || !words {
        return Some(out);
    }
    let fs: Vec<Option<u64>> = match frames.get(&w) {
        Some(v) if !v.is_empty() => v.iter().map(|&f| Some(f)).collect(),
        _ => vec![None],
    };
    let mut bag = BitVector::new(out.bit_len(), Some(0));
    for f in fs {
        for c in 0..roles {
            let mut u = out.clone();
            u.rotr_mut(bind_offset(c, f, out.bit_len()));
            for code in codes {
                if code.as_words().iter().zip(u.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 28 {
                    bag.or_mut(code);
                }
            }
        }
    }
    (bag.count_ones() > 0).then_some(bag)
}

/// The relation store's answer about word `w`: the relation the sentence's other words
/// name, if `w` has it, else all it knows about `w`; the answers' codes OR-ed (plain).
fn rel_answer(rel: &RelationStore, codes: &[BitVector], w: usize, sentence: &[(usize, usize)], hops: usize) -> Option<BitVector> {
    let all = rel.about(codes, w);
    if all.is_empty() {
        return None;
    }
    let query: Vec<usize> = sentence.iter().map(|x| x.0).filter(|&x| x != w).collect();
    let named = rel.relation_for(&query);
    let pick: Vec<usize> = match named.filter(|r| all.iter().any(|a| a.0 == *r)) {
        Some(r) => all.iter().filter(|a| a.0 == r).map(|a| a.3).collect(),
        None => all.iter().map(|a| a.3).collect(),
    };
    let mut out = BitVector::new(codes[0].bit_len(), Some(0));
    for &a in &pick {
        out.or_mut(&codes[a]);
        if hops > 1 {
            // the second hop: forward steps from the answer (an earlier filler to a later one)
            for (r, i, j, _) in rel.about(codes, a) {
                if i < j {
                    for b in rel.ask_all(codes, a, r, i, j) {
                        if b != w {
                            out.or_mut(&codes[b]);
                        }
                    }
                }
            }
        }
    }
    Some(out)
}

/// Say what the cortex means (a heard-word code) through the motor area and the vocal tract:
/// the word the tract produced (None for a garbled or missing command), and the forward
/// model's prediction of its sound (the efference copy).
fn motor_say(motor: &MotorArea, tract: &VocalTract, meant: &BitVector) -> (Option<usize>, Option<BitVector>) {
    match motor.plan(meant) {
        Some(cmd) => (tract.articulate(&cmd), motor.predict(&cmd)),
        None => (None, None),
    }
}

/// Cortical familiarity of each word: how many of the column's kernels are keyed on it,
/// i.e. sample their current-word (frame 0) bits mostly from its code.
fn cortical_familiarity(kernels: &[SimpleKernel], codes: &[BitVector], bits: usize) -> Vec<u32> {
    let mut fam = vec![0u32; codes.len()];
    for k in kernels {
        // the kernel's sampled bits in frame 0 (the current word)
        let b0: Vec<usize> = k.input_set.iter().map(|&b| b as usize).filter(|&b| b < bits).collect();
        if b0.len() < 4 {
            continue;
        }
        for (w, code) in codes.iter().enumerate() {
            let o = b0.iter().filter(|&&b| code.bit_get(b)).count();
            if o * 2 >= b0.len() {
                fam[w] += 1;
            }
        }
    }
    fam
}

/// The current sentence's least familiar word (by cortical familiarity), as a band
/// (log2 of its count, at most 7).
fn sentence_familiarity(fam: &[u32], ids: &[usize], start: usize, t: usize) -> u64 {
    ids[start..=t].iter().map(|&w| fam.get(w).copied().unwrap_or(0)).min().map_or(7, |c| (32 - c.leading_zeros() as u64).min(7))
}

/// Novelty from exposure: the band of the current sentence's least exposed word (sentences
/// read that held it): under 16 → 0, under 64 → 2, under 256 → 4, more → 6 (the go/no-go
/// halves it into four bands).
fn exposure_band(count: &[u32], ids: &[usize], start: usize, t: usize) -> u64 {
    let c = ids[start..=t].iter().map(|&w| count.get(w).copied().unwrap_or(0)).min().unwrap_or(u32::MAX);
    match c {
        0..=15 => 0,
        16..=63 => 2,
        64..=255 => 4,
        _ => 6,
    }
}

/// A proposal's credibility from the Bayes module: the belief in its fact (its first word,
/// the new name, is a member of the partner's family) times the best credibility among the
/// narrators of its source events.
fn proposal_credibility(rel: &RelationStore, isa: usize, words: &[usize], partners: &[usize], sources: &[u32], narrator: &HashMap<u32, u16>) -> u32 {
    let Some(&name) = words.first() else { return 0 };
    let fact = partners.iter().map(|&f| rel.belief(name, isa, 0, 1, f)).max().unwrap_or(0);
    let told = sources.iter().map(|r| narrator.get(r).map_or(0, |&n| rel.bayes.credibility(n))).max().unwrap_or(0);
    ((fact as u64 * told as u64) >> 16) as u32
}

type Proposals = HashMap<(usize, Vec<usize>), (Vec<u32>, u32, u32, bool, Vec<Vec<usize>>, Vec<u32>, Vec<usize>)>;

/// Judge the proposals not yet replayed: validated when confirmed by a reading, or credible
/// (with the relation store: the Bayes module's belief in the fact times the source
/// narrator's credibility, at least `min`; without it: at least `support_min` independent
/// premises), and never contradicted. Returns the stories that replay the newly validated
/// ones, one per source context.
fn validate_proposals(
    proposals: &mut Proposals,
    rel: Option<(&RelationStore, usize)>,
    hc: &dyn EpisodicCircuit,
    narrator: &HashMap<u32, u16>,
    vocab: &[&'static str],
    min: Q16,
    support_min: usize,
) -> Vec<Story> {
    let mut stories = Vec::new();
    let mut keys: Vec<(usize, Vec<usize>)> = proposals.keys().cloned().collect();
    keys.sort();
    for k in keys {
        let e = proposals.get_mut(&k).unwrap();
        if e.3 {
            continue;
        }
        let corroborated = match rel {
            Some((r, isa)) => proposal_credibility(r, isa, &k.1, &e.6, &e.0, narrator) >= min,
            None => e.5.iter().map(|&f| hc.testimony(f) as usize).sum::<usize>().min(e.0.len()) >= support_min,
        };
        if (e.1 > 0 || corroborated) && e.2 == 0 {
            // replayed in each of its sources' contexts, as inferred events are
            e.3 = true;
            for prefix in &e.4 {
                let mut words: Vec<&'static str> = prefix.iter().map(|&w| vocab[w]).collect();
                words.extend(k.1.iter().map(|&w| vocab[w]));
                words.push(".");
                let answer_at = words.len().saturating_sub(2);
                stories.push(Story { words, answer_at, held_out: false });
            }
        }
    }
    stories
}

/// The routed L4 row (ROUTE): `[word | slot 1 | … | slot k]`. Each channel (source id, content)
/// is bound to its source (rotated by a per-source offset), passes the share of its bits
/// given (a fixed subset by bit position), and goes to the slot its place in `order` gives
/// it; channels past the last slot are ORed into it. `skip` leaves one channel out.
fn route_row(word: &BitVector, channels: &[(usize, BitVector)], shares: &[u64], order: &[usize], slots: usize, skip: Option<usize>) -> Vec<u64> {
    let fw = word.as_words().len();
    let mut row = vec![0u64; fw * (1 + slots)];
    row[..fw].copy_from_slice(word.as_words());
    if slots == 0 {
        return row;
    }
    for (j, (c, content)) in channels.iter().enumerate() {
        if skip == Some(j) || content.count_ones() == 0 {
            continue;
        }
        let mut v = content.clone();
        if route_bind() {
            v.rotl_mut(((*c + 1) * ROUTE_ROT) % (fw * 64));
        }
        let share = shares[j];
        let slot = order.iter().position(|x| x == c).unwrap_or(order.len()).min(slots - 1);
        let dst = &mut row[fw * (1 + slot)..fw * (2 + slot)];
        for (wi, (d, w)) in dst.iter_mut().zip(v.as_words()).enumerate() {
            let mut w = *w;
            if share < ONE as u64 {
                let mut keep = 0u64;
                for b in 0..64 {
                    let h = ((wi * 64 + b) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 48;
                    if h < share {
                        keep |= 1 << b;
                    }
                }
                w &= keep;
            }
            *d |= w;
        }
    }
    row
}

/// The column's L4 row from a current word and its middle frames: the hand layout
/// (`assemble`), or under ROUTE the routed one, each frame and the previous input a channel
/// (source = its index in the hand layout) passed whole.
/// `row` with `frame` appended as one more frame.
fn with_frame(row: &BitVector, frame: &BitVector) -> BitVector {
    let mut w = row.as_words().to_vec();
    w.extend_from_slice(frame.as_words());
    BitVector::from_words(w)
}

fn l4_row(column: &CorticalColumn, current: &BitVector, frames: &[BitVector], route: Option<&[usize]>) -> BitVector {
    match route {
        None => column.assemble(current, frames),
        Some(order) => {
            let mut ch: Vec<(usize, BitVector)> = frames.iter().enumerate().map(|(i, f)| (i + 1, f.clone())).collect();
            ch.push((frames.len() + 1, column.previous()));
            let shares = vec![ONE as u64; ch.len()];
            BitVector::from_words(route_row(current, &ch, &shares, order, frames.len() + 1, None))
        }
    }
}

/// ROUTE_BIND=1: bind each channel to its source by a rotation. Off by default: a rotated
/// word no longer matches the output code, so the column cannot copy it (a new name read
/// from memory or from above); the slot, fixed after the critical period, already says
/// where a channel came from.
fn route_bind() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ROUTE_BIND").is_ok())
}

/// Slots the routed row adds beyond the hand row (ROUTE_EXTRA, default 1 with HC_ROUTE).
fn route_extra_slots() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        if std::env::var("ROUTE").is_err() {
            // HC_EC: the entorhinal feedback slot; INNER_SLOT: the heard slot
            return std::env::var("HC_EC").is_ok() as usize + std::env::var("INNER_SLOT").is_ok() as usize;
        }
        let default = std::env::var("HC_ROUTE").is_ok() as usize + std::env::var("LEARNING").map_or(false, |v| v == "three") as usize + std::env::var("QQUERY").is_ok() as usize;
        std::env::var("ROUTE_EXTRA").ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    })
}

/// GENOME=<file>: the network is a genome (`genomes/*.gen`), built by the grammar and run by
/// its own update loop (`Reader`); this harness only supplies the stories and scores the
/// answers. Every number the network needs is in the genome, none in settings.
fn run_genome(task: Task, max_facts: usize, seed: u64) -> Outcome {
    let mut rng = StdRng::seed_from_u64(seed);
    let vocab = task_vocab(task);
    let index: HashMap<&str, usize> = vocab.iter().enumerate().map(|(i, w)| (*w, i)).collect();
    let enc = Encoder::new(vocab.len(), BITS, 32, &mut rng);
    let stream = |k: u64| StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ k.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    let mut story_rng = stream(1);
    let season_len: usize = std::env::var("SEASON_LEN").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
    let path = std::env::var("GENOME").expect("GENOME=<file>");
    let mut text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("GENOME {path}: {e}"));
    // MUTATE_STACK=<op>:<k>: the k-th single mutation of the stack text (number, delete,
    // insert, swap); MUTATE=<op>:<k>: of the gene list (nudge, rewire, add, duplicate,
    // toggle). GENES=1: run through the gene list (implied by MUTATE).
    let mutant = |v: &str| -> (String, u64) {
        let (op, k) = v.split_once(':').unwrap_or((v, "0"));
        (op.to_string(), k.parse().unwrap_or(0))
    };
    if let Ok(v) = std::env::var("MUTATE_STACK") {
        let (op, k) = mutant(&v);
        let (t, d) = neurocomp::program::genes::mutate_stack_text(&text, &op, &mut StdRng::seed_from_u64(k));
        eprintln!("  MUTANT seed {seed}: stack {d}");
        text = t;
    }
    let fail = |e: String| -> Outcome {
        eprintln!("  MUTANT seed {seed}: the genome does not build: {e}");
        Outcome { seen: 0.0, held_out: 0.0, recall: 0.0, places: 0.0 }
    };
    let genome = match neurocomp::program::Genome::parse(&text) {
        Ok(g) => g,
        Err(e) => return fail(e),
    };
    let genes = std::env::var("GENES").is_ok() || std::env::var("MUTATE").is_ok();
    let mut reader = if genes {
        let mut list = match neurocomp::program::GeneList::from_genome(&genome) {
            Ok(l) => l,
            Err(e) => return fail(e),
        };
        if let Ok(v) = std::env::var("MUTATE") {
            let (op, k) = mutant(&v);
            let d = list.mutate(&op, &mut StdRng::seed_from_u64(k));
            eprintln!("  MUTANT seed {seed}: genes {d}");
        }
        let (net, schedule) = list.build(seed ^ 0x67e9_0e5e);
        neurocomp::program::Reader::from_network(net, schedule, seed ^ 0x67e9_0e5e)
    } else {
        neurocomp::program::Reader::new(&genome, seed ^ 0x67e9_0e5e)
    };
    if neurocomp::program::Module::n_outputs(&reader.net) == 0 {
        return fail("no output".into());
    }
    let (mut seen, mut held) = ((0usize, 0usize), (0usize, 0usize));
    for s_i in 0..TRAIN + TEST {
        let testing = s_i >= TRAIN;
        if s_i == TRAIN {
            reader.start_test();
        }
        let held_out = testing && s_i % 2 == 1;
        let s = match task {
            Task::Season => {
                let d = story_rng.gen_range(0..season_len);
                season_story_with(&mut story_rng, d, held_out, None, None)
            }
            Task::Habit => habit_story(&mut story_rng, held_out),
            Task::Elim => elim_story(&mut story_rng, held_out),
            Task::Give => give_story(&mut story_rng, held_out),
            Task::Topic => topic_story(&mut story_rng, held_out),
            Task::Persist => persist_story(&mut story_rng, s_i, held_out, testing),
            Task::Books => panic!("GENOME: the books task is not supported yet"),
            _ => story(&mut story_rng, task, max_facts, held_out),
        };
        let ids: Vec<usize> = s.words.iter().map(|w| index[w]).collect();
        for t in 0..ids.len() {
            let p = reader.read(&enc.codes[ids[t]]).clone();
            if testing && t + 1 == s.answer_at {
                let right = enc.decode(&p) == Some(ids[t + 1]);
                let r = if s.held_out { &mut held } else { &mut seen };
                r.0 += right as usize;
                r.1 += 1;
            }
            if s.words[t] == "." {
                reader.end_sentence();
            }
        }
        reader.end_story();
    }
    eprintln!("  GENOME seed {seed}: {} kernels\n{}", neurocomp::program::Module::kernels(&reader.net), neurocomp::program::Module::describe(&reader.net, 2).trim_end());
    let pct = |r: (usize, usize)| 100.0 * r.0 as f64 / r.1.max(1) as f64;
    Outcome { seen: pct(seen), held_out: pct(held), recall: 0.0, places: 0.0 }
}
