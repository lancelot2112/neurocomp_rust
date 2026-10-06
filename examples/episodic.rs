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

use std::collections::HashMap;

use common::Encoder;
use neurocomp::bitvec::BitVector;
use neurocomp::kernel::{GrowthConfig, KernelClass, SimpleKernel};
use neurocomp::program::{Autoassociative, BasalGanglia, Ca3FloatMemory, Ca3Memory, CorticalColumn, CorticothalamicGate, DentateGyrus, SourceMix, HigherArea, EpisodicMemory, Gate, PfcGate, RelayChannel, RouteScores, Thalamus, WorkingMemory};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

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

/// Season task: `n`'s place in season `s` (distinct per season for every name).
fn season_place(n: usize, s: usize) -> usize {
    (2 * n + [0, 1, 3, 4][s] + 1) % PLACES.len()
}

fn season_story(rng: &mut StdRng, distance: usize, held_out: bool) -> Story {
    let season = rng.gen_range(0..SEASONS.len());
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
    let n = rng.gen_range(0..NAMES.len());
    words.extend([NAMES[n], "went", "to", "the"]);
    let answer_at = words.len();
    words.extend([PLACES[season_place(n, season)], "."]);
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

fn run(policy: Policy, task: Task, max_facts: usize, seed: u64) -> Outcome {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vocab: Vec<&str> = vec![
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
    if task == Task::Season {
        vocab.extend(SEASONS);
        vocab.push("came");
    }
    let index: HashMap<&str, usize> = vocab.iter().enumerate().map(|(i, w)| (*w, i)).collect();
    let enc = Encoder::new(vocab.len(), BITS, 32, &mut rng);

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
    let replay_prob: f64 = std::env::var("REPLAY_PROB").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    // REPLAY_MODE: random (default) | tagged (a question whose recall predicted the answer
    // tags that episode; sleep replay favours tags) | awake (that episode is replayed into
    // cortex at once, at the question: prefrontal-driven retrieval)
    let replay_mode = std::env::var("REPLAY_MODE").unwrap_or_else(|_| "random".into());
    let tag_boost: u32 = std::env::var("TAG_BOOST").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let mut last_recall_id: Option<usize> = None; // hippocampal episode recalled this step
    let mut last_recall_cue: Option<BitVector> = None; // the cue that recalled it
    // tagged replay: the question's cue, stored with the tag, keys the sleep replay
    let mut tag_cues: HashMap<usize, BitVector> = HashMap::new();
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
    let mut pfc_gate = PfcGate::new(BITS, pfc_trace, 0.9, seed + 11);
    let mut pfc_loads = HashMap::<&str, (usize, usize)>::new(); // word -> (loads, decisions) at test
    let (mut pfc_rewards, mut pfc_questions) = (0usize, 0usize); // training
    let names_set: std::collections::HashSet<usize> = NAMES.iter().map(|n| index[n]).collect();
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
            ca3.set_readout_fraction(env("CA3_READOUT", 0.5));
            (Some(DentateGyrus::new(BITS, cells, fan_in, k, seed + 100)), Some(ca3))
        }
        _ => (None, None),
    };
    let mut sentence = BitVector::new(BITS, Some(0)); // bag of the current sentence so far
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
    l6_gate.weaken = std::env::var("L6_WEAKEN").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let (mut l6_open_sum, mut l6_words) = (0usize, 0usize);
    // [question kind][channel]: kind 1 = "what did X give ?", else 0
    let mut l6_open_at_answer = vec![vec![0usize; routes.len() + 1]; 2];
    let mut l6_answers = [0usize; 2];
    let mut gate_chosen_at_answer: HashMap<String, usize> = HashMap::new();
    // LearnedGate: route discovery and the current pool
    let mut route_scores = RouteScores::default();
    let mut gate_routes: Vec<RelayChannel> = if policy == Policy::ThalamicGate { routes.clone() } else { Vec::new() };
    // CA1-style comparator (NOVELTY=prediction): store, cue and read out only what the
    // predictor failed to predict, instead of frequency habituation.
    let predictive_novelty = std::env::var("NOVELTY").map_or(false, |v| v == "prediction");
    let mut surprising = BitVector::new(BITS, Some(0)); // unpredicted bits of the sentence so far
    let predicted_share: f32 = std::env::var("PREDICTED_SHARE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
    // recalled content: drop bits in more than 40% of episodes (all kept with prediction novelty)
    let habituation = if predictive_novelty { 1.0 } else { 0.4 };
    // which recalled item to cue with: the rarest among stored items (an IDF-like
    // specificity); RARITY=all cues with everything unpredicted
    let rarity_ratio = if std::env::var("RARITY").map_or(false, |v| v == "all") { f32::INFINITY } else { 1.5 };
    let min_overlap = 8; // floor for the recall threshold

    let mut class: KernelClass<SimpleKernel> = KernelClass::predictive(GrowthConfig {
        max_kernels: 100_000,
        frame_words: BITS / 64,
        max_frames: mid_frames + 2,
        sample_bits: 16,
        match_fraction: 0.8,
        surprise_fraction: 0.5,
        // GENERALIZE=f: drop silent inputs of near-matching kernels that would have been
        // right (synapse-level credit), after GENERALIZE_AFTER misses; off by default
        generalize: std::env::var("GENERALIZE").ok().and_then(|v| v.parse().ok()),
        generalize_after: std::env::var("GENERALIZE_AFTER").ok().and_then(|v| v.parse().ok()).unwrap_or(1),
    });

    // TRUST=f: depth only outranks reliability among kernels at least f reliable
    class.set_trust_floor(ratio_env("TRUST"));
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
    // the cortical column: L4 input assembly, L2/3 predictor (`class`), L5 prediction /
    // confidence / surprise, L6 context (`th`, whose match rules the thalamus gates)
    let mut column = CorticalColumn::new(BITS, class, th);
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
    let make_area = |span: usize, with_above: bool| {
        let mut c: KernelClass<SimpleKernel> = KernelClass::predictive(GrowthConfig {
            max_kernels: 100_000,
            frame_words: BITS / 64,
            max_frames: if std::env::var("HIER_SEPARATE").is_ok() { 1 + span } else { 2 } + with_above as usize,
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
        // HIER_GROW_TRUST=p/q: growth trust floor in the higher area (did not help)
        c.set_growth_trust(ratio_env("HIER_GROW_TRUST"));
        // HIER_SLEEP=1: sleep compacts the higher area too (half the cost, a few points less
        // on habit). No uncertainty gate on its growth: the gate stops growth when no input
        // frame carries the target, but the higher area's target (the column's residual) is
        // never in its input, by design
        if std::env::var("HIER_SLEEP").is_ok() {
            c.set_replay(std::env::var("REPLAY_LEN").ok().and_then(|v| v.parse().ok()).unwrap_or(512));
        }
        let mut a = HigherArea::new(BITS, c, span);
        // HIER_SEPARATE=1: one slow-state frame per recent sentence (worse: deep kernels at
        // varying lags); default: one frame with all recent surprises
        a.separate = std::env::var("HIER_SEPARATE").is_ok();
        // HIER_FOCUS=1: attention to one remembered word at growth (did not help)
        a.focus = std::env::var("HIER_FOCUS").is_ok();
        a
    };
    let mut area = make_area(hier_span, hier_levels > 1 && !chain_mix);
    let mut upper: Vec<HigherArea> = (1..hier_levels).map(|j| make_area(hier_span.pow(j as u32 + 1), j + 1 < hier_levels && !chain_mix)).collect();
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
    let bound_kind: f64 = std::env::var("BOUNDARY_KIND").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
    // context signature per word: the words seen just before (< V) and just after (V + w)
    let mut word_ctx: Vec<std::collections::HashSet<usize>> = vec![std::collections::HashSet::new(); vocab.len()];
    // test sentences with a detected boundary: (story-opening, other); story-opening sentences
    let mut bound_hits = (0usize, 0usize);
    let mut story_openings = 0usize;
    // READBACK=top|all: self-supervised read-back. At each sentence end, an area says back
    // the most recent rare word its window holds (rare: seen in fewer than READBACK_RARE of
    // the sentences so far, default 0.02). Its target is that word, from its own input; it
    // learns from the mismatch (training only), and what it says is heard: it joins the
    // next sentence's surprising words, so it re-enters every area's window (rehearsal)
    let readback = std::env::var("READBACK").ok();
    let readback_rare: f64 = std::env::var("READBACK_RARE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.02);
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
    let hier_gate_conf: Option<f32> = std::env::var("HIER_GATE_CONF").ok().and_then(|v| v.parse().ok());
    let hier_gate_warmup: usize = std::env::var("HIER_GATE_WARMUP").ok().and_then(|v| v.parse().ok()).unwrap_or(1000);
    let mut hier_gate = CorticothalamicGate::new(BITS, 1, seed + 31);
    hier_gate.weaken = std::env::var("HIER_GATE_WEAKEN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.25);
    let mut hier_gate_step: Option<BitVector> = None;
    let mut topdown_passed_at_answer = 0usize;
    // HIER_EARLY=1: the top-down frame right after the current word
    let hier_early = std::env::var("HIER_EARLY").is_ok();
    let mut topdown_has_answer = 0usize;
    let mut prev: Option<usize> = None;
    let (mut seen, mut held) = ((0usize, 0usize), (0usize, 0usize));
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
    let mut season_bins = [(0usize, 0usize); 6];
    // COST: wall time and words for training and test
    let (mut train_secs, mut test_secs, mut train_words, mut test_words) = (0f64, 0f64, 0usize, 0usize);
    let mut phase_start = std::time::Instant::now();
    let mut prof = [0f64; 6];
    let sleep_every: Option<usize> = std::env::var("SLEEP_EVERY").ok().and_then(|v| v.parse().ok());
    let (mut sleeps, mut slept_pruned, mut slept_merged) = (0usize, 0usize, 0usize);
    let mut prof_t = std::time::Instant::now();
    for s_i in 0..TRAIN + TEST {
        let testing = s_i >= TRAIN;
        // SLEEP_EVERY=n: an offline sleep pass for the column every n training stories
        if !testing && s_i > 0 && sleep_every.map_or(false, |n| s_i % n == 0) {
            let (p, m) = column.l23.sleep();
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
            gate_routes = route_scores.top(8, 2.0);
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
        }
        let s = if task == Task::Season {
            season_distance = rng.gen_range(0..season_len);
            season_story(&mut rng, season_distance, testing && s_i % 2 == 1)
        } else if task == Task::Habit {
            habit_story(&mut rng, testing && s_i % 2 == 1)
        } else if task == Task::Elim {
            elim_story(&mut rng, testing && s_i % 2 == 1)
        } else if task == Task::Give {
            give_story(&mut rng, testing && s_i % 2 == 1)
        } else if task == Task::Topic {
            topic_story(&mut rng, testing && s_i % 2 == 1)
        } else if task == Task::Persist {
            persist_story(&mut rng, s_i, testing && s_i % 2 == 1, testing)
        } else {
            story(&mut rng, task, max_facts, testing && s_i % 2 == 1)
        };
        if let Some(out) = dump.as_mut() {
            use std::io::Write;
            writeln!(out, "{}\t{}\t{}\t{}", if testing { "test" } else { "train" }, s.held_out as u8, s.answer_at, s.words.join(" ")).unwrap();
            continue;
        }
        if testing {
            test_words += s.words.len();
        } else {
            train_words += s.words.len();
        }
        let ids: Vec<usize> = s.words.iter().map(|w| index[w]).collect();
        if hier && hier_reset {
            area.clear();
            for u in upper.iter_mut() {
                u.clear();
            }
        }
        for t in 0..ids.len() {
            // PROF: [4] storage and everything after learning (from the previous word)
            prof[4] += prof_t.elapsed().as_secs_f64();
            prof_t = std::time::Instant::now();
            let code = &enc.codes[ids[t]];
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
            let share = 1.0 - column.surprise(code);
            if sent_words == 0 {
                first_surprise = 1.0 - share;
            }
            sent_surprise += 1.0 - share;
            sent_words += 1;
            if share < predicted_share {
                surprising.or_mut(code);
                if hier {
                    area.note_word(code);
                    for u in upper.iter_mut() {
                        u.note_word(code);
                    }
                }
            }
            if std::env::var("TRACE_SHARE").is_ok() && testing && s_i < TRAIN + 3 {
                eprint!("{}:{share:.2} ", s.words[t]);
                if t + 1 == ids.len() {
                    eprintln!();
                }
            }
            let cue_source = if predictive_novelty { &surprising } else { &sentence };
            if let Policy::Pfc { learned } = policy {
                let load = if learned {
                    let explore = if testing { None } else { Some(&mut rng) };
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
                let mut words = code.as_words().to_vec();
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
                        let cue = memory.rarest(cue_source, 0.1, rarity_ratio);
                        if cue.count_ones() > 0 {
                            let need = ((cue.count_ones() as f32 * 0.7) as u32).max(min_overlap);
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
                            let explore = if testing { None } else { Some(&mut rng) };
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
                        let cue = memory.rarest(cue_source, 0.1, rarity_ratio);
                        let mut recalled = None;
                        if cue.count_ones() > 0 {
                            let need = ((cue.count_ones() as f32 * 0.7) as u32).max(min_overlap);
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
                            let explore = if testing { None } else { Some(&mut rng) };
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
                        let cue = memory.rarest(cue_source, 0.1, rarity_ratio);
                        let chain = memory.recall_chain(&cue, hops, habituation, 0.1, rarity_ratio);
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
                        let cue = memory.rarest(cue_source, 0.1, rarity_ratio);
                        let (mut hop1, mut hop2) = (BitVector::new(BITS, Some(0)), BitVector::new(BITS, Some(0)));
                        let need = ((cue.count_ones() as f32 * 0.7).ceil() as u32).max(1);
                        if cue.count_ones() > 0 {
                            if let Some((id, ep)) = memory.recall_excluding(&cue, need, &[]) {
                                hop1 = memory.novel(ep, habituation);
                                for (c, &u) in hop1.as_words_mut().iter_mut().zip(cue.as_words()) {
                                    *c &= !u;
                                }
                                let items = memory.items(&hop1, 16);
                                let explore = if testing { None } else { Some(&mut rng) };
                                if let Some(i) = bg.select(&items, explore) {
                                    let item = &items[i];
                                    let need = ((item.count_ones() as f32 * 0.7).ceil() as u32).max(1);
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
                        let cue = memory.rarest(cue_source, 0.1, rarity_ratio);
                        for frame in memory.recall_branches(&cue, b, habituation, 16) {
                            words.extend_from_slice(frame.as_words());
                        }
                    }
                    Policy::Ca3 { .. } => {
                        let cue = memory.rarest(cue_source, 0.1, rarity_ratio);
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
                            if strength > 0.0 {
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
                        let cue = if matches!(policy, Policy::Pfc { .. }) { wm.content() } else { memory.rarest(cue_source, 0.1, rarity_ratio) };
                        let mut recalled = BitVector::new(BITS, Some(0));
                        if cue.count_ones() > 0 {
                            // recall needs most of the cue to be present in the episode
                            let need = ((cue.count_ones() as f32 * 0.7) as u32).max(min_overlap);
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
                if hier {
                    // the chain, top down: each upper area predicts from its window and the
                    // prediction of the area above it
                    let mut above: Option<BitVector> = None;
                    for i in (0..upper.len()).rev() {
                        let hin_u = upper[i].input_with(&sentence, &surprising, if chain_mix { None } else { above.as_ref() });
                        let p = upper[i].predict(&hin_u);
                        upper_pred[i] = (p.count_ones() > 0).then(|| p.clone());
                        if testing && t + 1 == s.answer_at {
                            upper_has_answer[i] += (p.as_words().iter().zip(enc.codes[ids[t + 1]].as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24) as usize;
                        }
                        above = Some(p);
                        upper_in[i] = Some(hin_u);
                    }
                    let hin = area.input_with(&sentence, &surprising, if chain_mix { None } else { above.as_ref() });
                    let td = area.predict(&hin);
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
                        let explore = if testing { None } else { Some(&mut rng) };
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
                    let frame = if passed { td } else { BitVector::new(BITS, Some(0)) };
                    if hier_early {
                        let at = BITS / 64; // right after the current word
                        words.splice(at..at, frame.as_words().iter().copied());
                    } else {
                        words.extend_from_slice(frame.as_words());
                    }
                    hier_in = Some(hin);
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
                let input = BitVector::from_words(words);

                let mut out = BitVector::new(BITS, Some(0));
                out.or_mut(column.predict(&input));
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
                // MIX: every source votes for its words with its reliability as the weight
                let mut mix_conf: Option<f32> = None;
                if mixing {
                    let prev = if t > 0 { ids[t - 1] } else { vocab.len() };
                    let ctx = (prev * (vocab.len() + 1) + ids[t]) as u64 * 8;
                    let bucket = |c: f32| [0.5f32, 0.7, 0.8, 0.9].iter().filter(|&&e| c >= e).count() as u64;
                    let words_of = |bv: &BitVector| -> Vec<usize> {
                        (0..vocab.len()).filter(|&i| enc.codes[i].as_words().iter().zip(bv.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24).collect()
                    };
                    // (source, key, proposed words): 0 column, 1 memory, 2 top-down
                    let mut proposals: Vec<(u8, u64, Vec<usize>)> = Vec::new();
                    let own = enc.decode(&out);
                    if let Some(w) = own {
                        proposals.push((0, ctx + bucket(column.confidence()), vec![w]));
                    }
                    let mw = words_of(&mem_src);
                    if !mw.is_empty() {
                        proposals.push((1, ctx + mw.len().min(3) as u64, mw));
                    }
                    if let Some(td) = td_src.as_ref() {
                        let tw = words_of(td);
                        if !tw.is_empty() {
                            proposals.push((2, ctx + bucket(area.column.confidence()), tw));
                        }
                    }
                    // the upper areas of the chain, one source each
                    for (i, p) in upper_pred.iter().enumerate() {
                        if let Some(p) = p {
                            let uw = words_of(p);
                            if !uw.is_empty() {
                                proposals.push((3 + i as u8, ctx + bucket(upper[i].column.confidence()), uw));
                            }
                        }
                    }
                    let votes: Vec<(usize, u32)> = proposals.iter().flat_map(|(src, key, ws)| ws.iter().map(|&w| (w, mix.weight(*src, *key))).collect::<Vec<_>>()).collect();
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
                    }
                    if !testing {
                        for (src, key, ws) in &proposals {
                            for &w in ws {
                                mix.record(*src, *key, w == next);
                            }
                        }
                    }
                }
                if testing && t + 1 == s.answer_at {
                    let right = enc.decode(&out) == Some(next);
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
                                kd[5] += (1000.0 * column.l23.confidence().unwrap_or(0.0)) as usize;
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
                    let r = if s.held_out { &mut held } else { &mut seen };
                    r.0 += right as usize;
                    r.1 += 1;
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
                    let b = [0.5f32, 0.7, 0.8, 0.9].iter().filter(|&&e| c >= e).count();
                    calib[b].0 += 1;
                    calib[b].1 += right as usize;
                    calib[b].2 += c as f64;
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
                                        semantic.feedback(&cue, &content, &mut rng);
                                        tagged_or_replayed += 1;
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
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
                    && if hier_gate_area { area.column.outcome(&enc.codes[next]) >= 0.5 } else { column.outcome_via(td_frame, &enc.codes[next]) >= 0.5 };
                let l6_used: Vec<bool> = match &l6_step {
                    Some((_, passed)) => passed.iter().enumerate().map(|(c, &p)| p && column.outcome_via(1 + early_shift + c, &enc.codes[next]) >= 0.5).collect(),
                    None => Vec::new(),
                };
                let l5 = if l5_used { column.outcome_via(choice_frame, &enc.codes[next]) } else { column.outcome(&enc.codes[next]) };
                if !testing && t + 1 == s.answer_at {
                    if let Policy::Pfc { learned: true } = policy {
                        // dopamine: the recall that working memory cued contained the answer
                        // (local), or the column predicted the answer (L5)
                        let r = if l5_reward { l5 } else { recall_had_answer as u32 as f32 };
                        pfc_gate.reward(r, &mut rng);
                        l5_sum[0] += l5 as f64;
                        l5_sum[1] += 1.0;
                        pfc_rewards += recall_had_answer as usize;
                        pfc_questions += 1;
                    }
                }
                last_recall_id = None;
                last_recall_cue = None;
                if testing {
                    // the fast inhibitory loop keeps running when slow learning is off
                    column.fast_inhibit(&enc.codes[next]);
                }
                if !testing {
                    // PROF: [2] evaluation, diagnostics, rewards
                    prof[2] += prof_t.elapsed().as_secs_f64();
                    prof_t = std::time::Instant::now();
                    column.learn(&input, &enc.codes[next], &mut rng);
                    // predictive coding: the higher area learns the lower column's residual,
                    // the words it failed to predict (HIER_RESIDUAL=0: every word)
                    if let Some(ctx) = hier_gate_step.take() {
                        hier_gate.learn(0, &ctx, td_used, &mut rng);
                    }
                    if let Some(hin) = hier_in.take() {
                        if !hier_residual || column.surprise(&enc.codes[next]) >= 0.5 {
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
                        gate_bg.reward(if l5_reward { l5 } else { hit as u32 as f32 }, &mut rng);
                        l5_sum[0] += l5 as f64;
                        l5_sum[1] += 1.0;
                    }
                    if let Some(hop2) = bg_pending.take() {
                        let hit = hop2.as_words().iter().zip(enc.codes[next].as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24;
                        bg.reward(if l5_reward { l5 } else { hit as u32 as f32 }, &mut rng);
                        l5_sum[0] += l5 as f64;
                        l5_sum[1] += 1.0;
                    }
                }
                bg_pending = None;
                gate_pending = None;
            }

            if ids[t] == full_stop && hier && bound_detect {
                // fact conflicts: this sentence's rare surprising words against the rare words
                // each area still holds
                let rare = |w: usize| sentence_count > 50 && (word_count[w] as f64) < readback_rare * sentence_count as f64;
                let same_kind = |a: usize, b: usize| {
                    let (x, y) = (&word_ctx[a], &word_ctx[b]);
                    let inter = x.intersection(y).count();
                    let union = x.union(y).count();
                    union > 0 && inter as f64 >= bound_kind * union as f64
                };
                let new_facts: Vec<usize> = (0..vocab.len())
                    .filter(|&w| rare(w) && enc.codes[w].as_words().iter().zip(surprising.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24)
                    .collect();
                let mut fired = false;
                for &w in &new_facts {
                    let conflicts = |c: &BitVector| enc.decode(c).map_or(false, |h| h != w && rare(h) && same_kind(h, w));
                    fired |= area.forget_through(conflicts);
                    for u in upper.iter_mut() {
                        fired |= u.forget_through(conflicts);
                    }
                }
                if testing {
                    let opening = !s.words[..t].contains(&".");
                    story_openings += opening as usize;
                    if fired {
                        if opening {
                            bound_hits.0 += 1;
                        } else {
                            bound_hits.1 += 1;
                        }
                    }
                }
            }
            if ids[t] == full_stop {
                // one-shot: the whole sentence (or its unpredicted part) is one episode.
                // Persist: test questions are not stored, or the first anchor question
                // would leak its own answer to every later one.
                let question = sentence.as_words().iter().zip(enc.codes[index["where"]].as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>() >= 24;
                if !(task == Task::Persist && testing && question) {
                    memory.store(if predictive_novelty { &surprising } else { &sentence });
                }
                if let (Some(dg), Some(ca3)) = (&dg, &mut ca3) {
                    // Encode the novel part: content shared by most episodes ("went to the")
                    // would otherwise dominate the dentate gyrus, give every episode the same
                    // code, and swamp recall.
                    let x = set_bits(&if predictive_novelty { surprising.clone() } else { memory.novel(&sentence, 0.4) });
                    if !x.is_empty() {
                        ca3.store(&x, &dg.separate(&x));
                    }
                }
                sentence = BitVector::new(BITS, Some(0));
                if hier {
                    area.end_sentence(&surprising);
                    for u in upper.iter_mut() {
                        u.end_sentence(&surprising);
                    }
                }
                surprising = BitVector::new(BITS, Some(0));
                // word frequencies (per sentence), for read-back's "rare"
                sentence_count += 1;
                if testing {
                    let first_of_story = !s.words[..t].contains(&".");
                    bound_log.push((first_of_story, sent_surprise / sent_words.max(1) as f32, first_surprise));
                }
                sent_surprise = 0.0;
                sent_words = 0;
                for w in s.words[..=t].iter().rev().skip(1).take_while(|w| **w != ".") {
                    word_count[index[w]] += 1;
                }
                // read-back: say back the most recent rare word held, hear it
                if hier {
                    if let Some(mode) = readback.as_deref() {
                        let n = upper.len();
                        let empty = BitVector::new(BITS, Some(0));
                        let rare = |w: usize| (word_count[w] as f64) < readback_rare * sentence_count as f64;
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
                            let cue = &enc.codes[full_stop];
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
                                a.learn(&x, &enc.codes[target], &mut rng);
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
            prev = Some(ids[t]);
        }
        if policy == Policy::Consolidate && !testing && !memory.is_empty() && rng.gen_bool(replay_prob) {
            // sleep: replay stored episodes into the cortical semantic store
            let boost = if replay_mode == "tagged" { tag_boost } else { 0 };
            for _ in 0..replays {
                let i = memory.sample_replay(&mut rng, boost).unwrap();
                let ep = memory.get(i).unwrap().clone();
                // a tagged episode is replayed under the cue of the question that tagged it
                let cue = tag_cues.get(&memory.id_of(i)).cloned().unwrap_or_else(|| memory.rarest(&ep, 0.1, rarity_ratio));
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
                semantic.feedback(&cue, &content, &mut rng);
            }
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
        if task == Task::Season {
            let names = ["0", "1", "2-3", "4-7", "8-15", "16+"];
            let parts: Vec<String> = season_bins
                .iter()
                .zip(names)
                .map(|(b, n)| if b.0 > 0 { format!("{n}: {:.0}% of {}", 100.0 * b.1 as f64 / b.0 as f64, b.0) } else { format!("{n}: -") })
                .collect();
            eprintln!("  SEASON seed {seed}: accuracy by filler stories since the season was announced: {}", parts.join(", "));
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
        _ => vec![Task::Short, Task::Long, Task::Varied],
    };
    println!("answer accuracy on {TEST} test stories after {TRAIN} training stories (predictor learning off at test); chance 1/6");
    if std::env::var("NOVELTY").map_or(false, |v| v == "prediction") {
        println!("NOVELTY=prediction: CA1-style comparator; store, cue and read out only what the predictor did not predict");
    }
    for task in tasks {
        let fact_settings: &[usize] = if matches!(task, Task::TwoHop | Task::Persist | Task::Topic | Task::Give | Task::Elim | Task::Habit | Task::Season) { &[0] } else { &[2, 3] };
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
            for policy in policies {
                let runs: Vec<Outcome> = (seed_start..seed_start + seeds).map(|seed| run(policy, task, max_facts, seed)).collect();
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
