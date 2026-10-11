//! hold_ladder: can the striatum learn when to hold an episode's start and when to reinstate it?
//!
//! A small world shaped like the read/tell task, without the cortex:
//! - a story of 16 new words (event boundaries at words 0, 4, 8 and 12);
//! - an instruction: "now" (also a boundary), "read" or "tell", "it";
//! - three waiting steps, then six task steps.
//!
//! **The two gates** are channels of the striatum, decided as in `readtell`:
//! - at an event boundary, *hold* keeps the hippocampal context of that moment;
//! - at any step with something held and no retrieval yet, *reinstate* restores the held
//!   context and starts retrieval.
//!
//! **What follows from them:**
//! - In "tell", task word j is right only if the context held is this story's start and
//!   retrieval began after the story, so the recall sequence lines up with the task.
//! - In "read", the book gives every word, whatever the gates did.
//! - Retrieval mode stops storing, as in `readtell`: words heard while retrieving are not stored.
//! - A new hold ends retrieval.
//!
//! **The striatum sees:**
//! - the working-memory content (the verb, from when it is heard);
//! - the sound now and the one before (silence before the story: the marker of a new source);
//! - the held context's cells;
//! - the recall arriving.
//!
//! **The rungs:**
//! - `end`: one reward at the episode's end (the sum).
//! - `word`: a reward at each right task word (as in `readtell`).
//! - `retag`: plus the held content's tag re-armed when it is used (PBWM).
//! - `theta`: plus storing goes on while retrieving (encoding and retrieval alternate within
//!   each theta cycle).
//! - `no marker`: as `theta`, but the story follows another sound, not a pause.
//!
//! Run: `cargo run --release --example hold_ladder` (`HL_TRAIN`, default 3000; `HL_SEEDS`,
//! default 3).

use neurocomp::fixed::ONE;
use neurocomp::program::{MaskStriatum, Striatum};

/// The striatum under test: popcount cells (`Striatum`) or masks (`MaskStriatum`).
enum Bg {
    Cells(Striatum),
    Masks(MaskStriatum),
}

impl Bg {
    fn begin_groups(&mut self, s: &[Vec<usize>], learn: bool, rng: &mut StdRng) {
        match self {
            Bg::Cells(x) => x.begin_groups(s, learn, rng),
            Bg::Masks(x) => x.begin_groups(s, learn),
        }
    }
    fn choose(&mut self, ch: usize, n: usize, rng: Option<&mut StdRng>) -> usize {
        match self {
            Bg::Cells(x) => x.choose(ch, n, rng),
            Bg::Masks(x) => x.choose(ch, n, rng.is_some()),
        }
    }
    fn choose_among(&mut self, ch: usize, cands: &[Vec<Vec<usize>>], rng: Option<&mut StdRng>) -> Option<usize> {
        match self {
            Bg::Cells(x) => x.choose_among(ch, cands, rng),
            Bg::Masks(x) => x.choose_among(ch, cands, rng.is_some()),
        }
    }
    fn maintain(&mut self, ch: usize) {
        match self {
            Bg::Cells(x) => x.maintain(ch),
            Bg::Masks(x) => x.maintain(ch),
        }
    }
    fn retag(&mut self, ch: usize) {
        match self {
            Bg::Cells(x) => x.retag(ch),
            Bg::Masks(x) => x.retag(ch),
        }
    }
    fn reward(&mut self, r: i32) {
        match self {
            Bg::Cells(x) => x.reward(r),
            Bg::Masks(x) => x.reward(r),
        }
    }
    fn end(&mut self, learn: bool, rng: &mut StdRng) {
        match self {
            Bg::Cells(x) => x.end(learn, rng),
            Bg::Masks(x) => x.end(learn),
        }
    }
    fn value(&mut self) -> i32 {
        match self {
            Bg::Cells(x) => x.debug_now(0).0,
            Bg::Masks(x) => x.value(),
        }
    }
    fn last_delta(&self) -> i32 {
        match self {
            Bg::Cells(x) => x.last_delta,
            Bg::Masks(x) => x.last_delta,
        }
    }
    fn report(&self) -> String {
        match self {
            Bg::Cells(x) => format!("cells {:?}, mean |dopamine| {:.3}", x.sizes(), x.stats.1 as f64 / x.stats.0.max(1) as f64 / ONE as f64),
            Bg::Masks(x) => format!("{} bit swaps, mean |dopamine| {:.3}", x.swaps, x.stats.1 as f64 / x.stats.0.max(1) as f64 / ONE as f64),
        }
    }
}
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const L: usize = 16;
const T: usize = 6;
const W: usize = 4096;
const CH_HOLD: usize = 0;
const CH_REINSTATE: usize = 1;

#[derive(Clone, Copy)]
struct Rung {
    name: &'static str,
    end_only: bool,
    retag: bool,
    theta: bool,
    marker: bool,
    /// reinstating is possible while retrieving (it starts the sequence over)
    restart: bool,
    /// diagnosis: the hold is given (at the story's start only), only reinstating is learned
    hold_given: bool,
    /// diagnosis: reinstating is given (at the last waiting step), only holding is learned
    reinstate_given: bool,
    /// the striatum sees the scene: the teacher's voice as such (what stays the same while a
    /// teacher speaks) and a page in view, not only the words
    voice: bool,
    /// the striatum has time cells (steps since the last boundary and since the last hold)
    time: bool,
    /// bookmarks: every boundary is kept (the last `MARKS`), and on every step the striatum may
    /// go back to one of them (offered by the sound at it and the one before), as readtell's
    /// index rows; no hold gate
    marks: bool,
    /// what going back costs (effort, in sixteenths of a reward)
    effort: i32,
}

const MARKS: usize = 6;
const CH_MARK: usize = 2;

fn pattern(rng: &mut StdRng, n: usize, range: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).map(|_| rng.gen_range(0..range)).collect();
    v.sort_unstable();
    v.dedup();
    v
}

#[derive(Default, Clone, Copy)]
struct Tally {
    episodes: usize,
    holds: usize,
    start_held: usize,
    tell: usize,
    read: usize,
    reinstated_tell: usize,
    reinstated_read: usize,
    reinstated_story: usize,
    tell_right: usize,
}

thread_local! {
    static DEBUG: std::cell::RefCell<[usize; 20]> = const { std::cell::RefCell::new([0; 20]) };
    static DELTA: std::cell::RefCell<Vec<(i64, i64)>> = std::cell::RefCell::new(vec![(0, 0); 128]);
}

struct World {
    vocab: Vec<Vec<usize>>,
    silence: Vec<usize>,
    now: Vec<usize>,
    verbs: [Vec<usize>; 2],
    it: Vec<usize>,
    /// the teacher's voice (the same while the teacher speaks)
    voice: Vec<usize>,
    /// a page in view (the same for every book)
    page: Vec<usize>,
    /// time cells' patterns (ten bands since a boundary, ten since a hold) and the counts
    times: Vec<Vec<usize>>,
    since_b: usize,
    since_h: usize,
    /// bookmarks: (episode, position, features)
    marks: Vec<(usize, usize, Vec<Vec<usize>>)>,
    /// (episode, position, context cells) of what is held
    held: Option<(usize, usize, Vec<usize>)>,
    retrieving: bool,
    /// recall steps since reinstating
    k: usize,
    last: Vec<usize>,
    step_no: usize,
}

impl World {
    fn new(rng: &mut StdRng) -> Self {
        let vocab = (0..300).map(|_| pattern(rng, 16, W)).collect();
        Self {
            vocab,
            silence: pattern(rng, 16, W),
            now: pattern(rng, 16, W),
            verbs: [pattern(rng, 16, W), pattern(rng, 16, W)],
            it: pattern(rng, 16, W),
            voice: pattern(rng, 16, W),
            page: pattern(rng, 16, W),
            times: (0..20).map(|_| pattern(rng, 16, W)).collect(),
            since_b: 0,
            since_h: 0,
            marks: Vec::new(),
            held: None,
            retrieving: false,
            k: 0,
            last: Vec::new(),
            step_no: 0,
        }
    }
}

/// The striatum's state: working memory, the sound now, the one before, the held context, the
/// recall arriving (each group in its own range).
fn state(wm: &[usize], heard: &[usize], before: &[usize], held: &[usize], recall: &[usize], voice: &[usize], since_b: &[usize], since_h: &[usize]) -> Vec<Vec<usize>> {
    [wm, heard, before, held, recall, voice, since_b, since_h].iter().enumerate().map(|(g, v)| v.iter().map(|&b| b + g * 65536).collect()).collect()
}

fn episode(st: &mut Bg, w: &mut World, ep: usize, rung: Rung, learn: bool, rng: &mut StdRng, t: &mut Tally) {
    let tell = rng.gen_bool(0.5);
    let story: Vec<usize> = (0..L).map(|_| rng.gen_range(0..w.vocab.len())).collect();
    let ctx: Vec<Vec<usize>> = (0..=L).map(|_| pattern(rng, 32, 65536)).collect();
    let mut stored = [false; L];
    let mut wm: Vec<usize> = Vec::new();
    let mut total = 0;
    let mut reinstated_after_cue = false;
    t.episodes += 1;
    w.step_no = 0;
    t.tell += tell as usize;
    t.read += !tell as usize;
    // what comes before the story: a pause, or (without the marker) a word
    w.last = if rung.marker { w.silence.clone() } else { w.vocab[rng.gen_range(0..w.vocab.len())].clone() };
    // one step of the loop; returns the recalled position (if retrieving this episode's story)
    // phase: 0 story, 1 instruction, 2 wait, 3 task, 4 the last waiting step
    let mut step = |st: &mut Bg, w: &mut World, stored: &[bool; L], heard: Vec<usize>, wm: &[usize], boundary: Option<usize>, phase: u8, rng: &mut StdRng, t: &mut Tally| -> Option<usize> {
        let held_ctx = w.held.as_ref().map_or(Vec::new(), |h| h.2.clone());
        let recalled = if w.retrieving && (phase == 0 || phase == 3) {
            let r = w.held.as_ref().and_then(|h| (h.0 == ep && h.1 + w.k < L && stored[h.1 + w.k]).then_some(h.1 + w.k));
            w.k += 1;
            r
        } else {
            None
        };
        let recall = recalled.map_or(Vec::new(), |p| w.vocab[story[p]].clone());
        // the scene: the teacher's voice while the teacher speaks; a page while a book is open
        let voice = if !rung.voice {
            Vec::new()
        } else if phase == 3 && !tell {
            w.page.clone()
        } else if heard != w.silence {
            w.voice.clone()
        } else {
            Vec::new()
        };
        // time cells: the band of steps since the last event boundary, and since the last hold
        let band = |d: usize| [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7].get(d).copied().unwrap_or(8 + (d >= 24) as usize);
        let (since_b, since_h) = if rung.time {
            (w.times[band(w.since_b)].clone(), w.times[10 + band(w.since_h)].clone())
        } else {
            (Vec::new(), Vec::new())
        };
        let s = state(wm, &heard, &w.last, &held_ctx, &recall, &voice, &since_b, &since_h);
        w.since_b += 1;
        w.since_h += 1;
        if boundary.is_some() {
            w.since_b = 0;
        }
        st.begin_groups(&s, learn, rng);
        if learn && std::env::var("HL_DEBUG").is_ok() {
            DELTA.with(|d| {
                let mut d = d.borrow_mut();
                let k = (tell as usize) * 64 + w.step_no.min(63);
                d[k].0 += if std::env::var("HL_DEBUG").unwrap() == "v" { st.value() as i64 } else { st.last_delta() as i64 };
                d[k].1 += 1;
            });
        }
        w.step_no += 1;
        if rung.marks {
            // every boundary is a bookmark; on every step the striatum may go back to one
            if let Some(pos) = boundary {
                if w.marks.len() >= MARKS {
                    w.marks.remove(0);
                }
                let f = vec![heard.iter().map(|&b| b + 10 * 65536).collect(), w.last.iter().map(|&b| b + 11 * 65536).collect()];
                w.marks.push((ep, pos, f));
            }
            if !w.marks.is_empty() && (!w.retrieving || rung.restart) {
                let cands: Vec<Vec<Vec<usize>>> = w.marks.iter().map(|m| m.2.clone()).collect();
                let explore = if learn { Some(&mut *rng) } else { None };
                if let Some(i) = st.choose_among(CH_MARK, &cands, explore) {
                    let (e, pos, _) = w.marks[i];
                    w.held = Some((e, pos, ctx[pos.min(L)].clone()));
                    w.retrieving = true;
                    w.k = 0;
                    st.reward(-rung.effort * (ONE as i32 / 16));
                    t.holds += (e == ep && pos == 0) as usize;
                    if phase == 0 {
                        t.reinstated_story += 1;
                    }
                }
            }
            w.last = heard;
            return recalled;
        }
        // the two gates are separate loops: holding at a boundary, then reinstating
        if let Some(pos) = boundary {
            let explore = if learn { Some(&mut *rng) } else { None };
            let hold = if rung.hold_given { pos == 0 } else { st.choose(CH_HOLD, 2, explore) == 1 };
            if hold {
                w.held = Some((ep, pos, ctx[pos.min(L)].clone()));
                w.retrieving = false;
                st.maintain(CH_HOLD);
                w.since_h = 0;
                t.holds += 1;
            }
        }
        let explore = if learn { Some(&mut *rng) } else { None };
        let may = w.held.is_some() && (!w.retrieving || rung.restart);
        let reinstate = if rung.reinstate_given { phase == 4 && may && tell } else { may && st.choose(CH_REINSTATE, 2, explore) == 1 };
        if reinstate {
            w.retrieving = true;
            w.k = 0;
            if rung.retag && learn {
                st.retag(CH_HOLD);
            }
            if phase == 0 {
                t.reinstated_story += 1;
            }
        }
        w.last = heard;
        recalled
    };
    for i in 0..L {
        // storing (stopped in retrieval mode, unless encoding alternates with it)
        if !w.retrieving || rung.theta {
            stored[i] = true;
        }
        let b = (i % 4 == 0).then_some(i);
        let heard = w.vocab[story[i]].clone();
        step(st, w, &stored, heard, &wm, b, 0, rng, t);
    }
    let instr = [w.now.clone(), w.verbs[tell as usize].clone(), w.it.clone()];
    for (j, x) in instr.into_iter().enumerate() {
        if j == 1 {
            wm = w.verbs[tell as usize].clone();
        }
        let r0 = w.retrieving;
        step(st, w, &stored, x, &wm, (j == 0).then_some(L), 1, rng, t);
        reinstated_after_cue |= !r0 && w.retrieving;
    }
    for k in 0..3 {
        let r0 = w.retrieving;
        step(st, w, &stored, w.silence.clone(), &wm, None, if k == 2 { 4 } else { 2 }, rng, t);
        reinstated_after_cue |= !r0 && w.retrieving;
    }
    if w.held.as_ref().map_or(false, |h| h.0 == ep && h.1 == 0) {
        t.start_held += 1;
    }
    if std::env::var("HL_DEBUG").is_ok() && learn {
        let p = w.held.as_ref().map(|h| if h.0 == ep { h.1 as i64 } else { -1 }).unwrap_or(-2);
        DEBUG.with(|d| d.borrow_mut()[(p + 2) as usize] += 1);
        if ep % 500 == 499 {
            DEBUG.with(|d| {
                eprintln!("ep {ep}: held at task time (none, older, 0, 4, 8, 12, now): {:?}", [0, 1, 2, 6, 10, 14, 18].map(|i| d.borrow()[i]));
                d.borrow_mut().iter_mut().for_each(|x| *x = 0);
            });
            DELTA.with(|d| {
                let mut d = d.borrow_mut();
                for tl in 0..2 {
                    let v: Vec<String> = (0..29).map(|i| { let x = d[tl * 64 + i]; format!("{:+.2}", x.0 as f64 / x.1.max(1) as f64 / ONE as f64) }).collect();
                    eprintln!("  dopamine by step ({}): {}", if tl == 1 { "tell" } else { "read" }, v.join(" "));
                }
                d.iter_mut().for_each(|x| *x = (0, 0));
            });
        }
    }
    for j in 0..T {
        let r0 = w.retrieving;
        let heard = if tell { w.silence.clone() } else { w.vocab[story[j]].clone() };
        let recalled = step(st, w, &stored, heard, &wm, None, 3, rng, t);
        if j == 0 {
            reinstated_after_cue |= !r0 && w.retrieving;
        }
        let right = if tell { recalled == Some(j) } else { true };
        if tell && right {
            t.tell_right += 1;
        }
        if right {
            total += ONE as i32;
            if !rung.end_only {
                st.reward(ONE as i32);
            }
        }
    }
    if tell {
        t.reinstated_tell += reinstated_after_cue as usize;
    } else {
        t.reinstated_read += reinstated_after_cue as usize;
    }
    if rung.end_only {
        st.reward(total);
    }
    st.end(learn, rng);
}

/// A striatum by name: `cells` (with HL_STRIATUM choosing its parts), or `masks` with options
/// after a colon: `d1`..`d3` (τ = 2^−n), `flat` (no novelty), `nohold`, `taps`, `sticky`.
fn make(spec: &str, seed: u64) -> Bg {
    if let Some(opts) = spec.strip_prefix("masks") {
        let mut m = MaskStriatum::new(2048, 64, 7 + seed);
        for d in 1..=4 {
            if opts.contains(&format!("d{d}")) {
                m.drop_shift = d;
            }
        }
        m.novelty = !opts.contains("flat");
        m.hold = !opts.contains("nohold");
        m.taps = opts.contains("taps");
        m.sticky = opts.contains("sticky");
        m.push_pull = !opts.contains("swap");
        m.go_ltd = !opts.contains("noltd");
        m.low_value = !opts.contains("calm");
        m.conj = opts.contains("conj");
        if opts.contains("p1") {
            m.pause_min = 1;
        }
        return Bg::Masks(m);
    }
    let mut st = Striatum::new(1 << 16);
    // HL_STRIATUM: which of the striatum's newer parts are on (default all)
    if let Ok(v) = std::env::var("HL_STRIATUM") {
        st.prune = v.contains("prune,") || v.ends_with("prune");
        st.prune_critic = v.contains("pcritic");
        st.mixed = v.contains("mixed");
        st.pause_min = if v.contains("tan") { 2 } else { 1 };
        st.gamma_div = if v.contains("gamma") { 8 } else { 0 };
        st.critic_td0 = v.contains("td0");
        st.d2_affinity = v.contains("d2");
    }
    Bg::Cells(st)
}

fn run(spec: &str, rung: Rung, seed: u64, n_train: usize) -> Tally {
    let mut rng = StdRng::seed_from_u64(100 + seed);
    let mut st = make(spec, seed);
    let mut w = World::new(&mut rng);
    let mut t = Tally::default();
    for ep in 0..n_train {
        episode(&mut st, &mut w, ep, rung, true, &mut rng, &mut t);
    }
    if seed == 0 {
        eprintln!("{spec} {}: {}", rung.name, st.report());
    }
    let mut t = Tally::default();
    for ep in n_train..n_train + 400 {
        episode(&mut st, &mut w, ep, rung, false, &mut rng, &mut t);
    }
    t
}

/// A contextual bandit: 16 contexts (64 bits each; contexts 2i and 2i+1 share half their bits
/// but want different actions), 4 actions, a reward for the context's best action. Each trial
/// replaces a quarter of the context's bits with random ones. Returns the share of best choices
/// per block of 500 trials, and in a test without exploration.
fn bandit(spec: &str, seed: u64, n: usize) -> (Vec<f64>, f64) {
    let mut rng = StdRng::seed_from_u64(500 + seed);
    let mut st = make(spec, seed);
    let mut ctx: Vec<Vec<usize>> = Vec::new();
    for i in 0..16 {
        let v: Vec<usize> = if i % 2 == 1 {
            let twin: &Vec<usize> = &ctx[i - 1];
            twin[..32].iter().copied().chain((0..32).map(|_| rng.gen_range(0..1 << 20))).collect()
        } else {
            (0..64).map(|_| rng.gen_range(0..1 << 20)).collect()
        };
        ctx.push(v);
    }
    let best: Vec<usize> = (0..16).map(|i| if i % 2 == 1 { 1 + (rng.gen_range(0..3) + 1) % 4 } else { 1 + rng.gen_range(0..4) }).collect();
    let noisy = |i: usize, rng: &mut StdRng| -> Vec<Vec<usize>> { vec![ctx[i].iter().map(|&b| if rng.gen_range(0..4) == 0 { rng.gen_range(0..1 << 20) } else { b }).collect()] };
    let mut curve = Vec::new();
    let mut right = 0;
    for t in 0..n {
        let i = rng.gen_range(0..16);
        let s = noisy(i, &mut rng);
        st.begin_groups(&s, true, &mut rng);
        let c = st.choose(0, 5, Some(&mut rng));
        let ok = c == best[i];
        right += ok as usize;
        st.reward(if ok { ONE as i32 } else { 0 });
        st.end(true, &mut rng);
        if t % 500 == 499 {
            curve.push(right as f64 / 500.0);
            right = 0;
        }
    }
    let mut test = 0;
    for _ in 0..400 {
        let i = rng.gen_range(0..16);
        let s = noisy(i, &mut rng);
        st.begin_groups(&s, false, &mut rng);
        test += (st.choose(0, 5, None) == best[i]) as usize;
        st.end(false, &mut rng);
    }
    (curve, test as f64 / 400.0)
}

fn main() {
    let n_train: usize = std::env::var("HL_TRAIN").ok().and_then(|v| v.parse().ok()).unwrap_or(3000);
    let seeds: u64 = std::env::var("HL_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let specs: Vec<String> = std::env::var("HL_IMPLS").unwrap_or_else(|_| "cells,masks".into()).split(',').map(|x| x.to_string()).collect();
    if std::env::var("HL_MODE").map_or(false, |v| v == "bandit") {
        // the contextual bandit, for each striatum, in parallel
        let n: usize = std::env::var("HL_BANDIT").ok().and_then(|v| v.parse().ok()).unwrap_or(3000);
        let jobs: Vec<(usize, u64)> = (0..specs.len()).flat_map(|r| (0..seeds).map(move |s| (r, s))).collect();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let results = std::sync::Mutex::new(vec![(vec![0.0; n / 500], 0.0); specs.len()]);
        std::thread::scope(|sc| {
            for _ in 0..4 {
                sc.spawn(|| loop {
                    let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&(r, seed)) = jobs.get(j) else { break };
                    let (c, t) = bandit(&specs[r], seed, n);
                    let mut res = results.lock().unwrap();
                    for (a, b) in res[r].0.iter_mut().zip(c) {
                        *a += b / seeds as f64;
                    }
                    res[r].1 += t / seeds as f64;
                });
            }
        });
        println!("contextual bandit: 16 contexts (twins share half their bits, want different actions), 4 actions, a quarter of each context's bits noise; {seeds} seeds; best action chosen per block of 500 trials, then a test without exploration (chance 25%)");
        for (spec, (c, t)) in specs.iter().zip(results.into_inner().unwrap()) {
            println!("  {:<22} {}  test {:.1}%", spec, c.iter().map(|x| format!("{:5.1}%", 100.0 * x)).collect::<Vec<_>>().join(" "), 100.0 * t);
        }
        return;
    }
    let base = Rung { name: "", end_only: false, retag: false, theta: false, marker: true, restart: false, hold_given: false, reinstate_given: false, voice: false, time: false, marks: false, effort: 0 };
    let rungs = [
        Rung { name: "end", end_only: true, ..base },
        Rung { name: "word", ..base },
        Rung { name: "retag", retag: true, ..base },
        Rung { name: "theta", retag: true, theta: true, ..base },
        Rung { name: "restart", retag: true, theta: true, restart: true, ..base },
        Rung { name: "hold.given", retag: true, theta: true, restart: true, hold_given: true, ..base },
        Rung { name: "reinst.given", retag: true, theta: true, restart: true, reinstate_given: true, ..base },
        Rung { name: "scene", retag: true, theta: true, restart: true, voice: true, ..base },
        Rung { name: "scene+rg", retag: true, theta: true, restart: true, voice: true, reinstate_given: true, ..base },
        Rung { name: "time", retag: true, theta: true, restart: true, voice: true, time: true, ..base },
        Rung { name: "time+rg", retag: true, theta: true, restart: true, voice: true, time: true, reinstate_given: true, ..base },
        Rung { name: "marks", theta: true, restart: true, marks: true, ..base },
        Rung { name: "marks+eff1", theta: true, restart: true, marks: true, effort: 1, ..base },
        Rung { name: "marks+eff4", theta: true, restart: true, marks: true, effort: 4, ..base },
        Rung { name: "marks+eff4+t", theta: true, restart: true, marks: true, effort: 4, time: true, voice: true, ..base },
        Rung { name: "no marker", retag: true, theta: true, restart: true, voice: true, time: true, marker: false, ..base },
    ];
    let only: Option<Vec<String>> = std::env::var("HL_RUNGS").ok().map(|v| v.split(',').map(|x| x.to_string()).collect());
    let rungs: Vec<Rung> = rungs.into_iter().filter(|r| only.as_ref().map_or(true, |o| o.iter().any(|x| x == r.name))).collect();
    println!("hold ladder: {n_train} training episodes, then 400 test episodes (no exploration), {seeds} seeds; percentages are test means");
    println!(
        "  {:<14} {:>6} {:>11} {:>13} {:>13} {:>17} {:>10}",
        "rung", "holds", "start held", "reinst. tell", "reinst. read", "reinst. in story", "tell right"
    );
    // all (striatum, rung, seed) runs, four at a time
    let nr = rungs.len();
    let jobs: Vec<(usize, u64)> = (0..specs.len() * nr).flat_map(|r| (0..seeds).map(move |s| (r, s))).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(vec![Tally::default(); specs.len() * nr]);
    std::thread::scope(|sc| {
        for _ in 0..4 {
            sc.spawn(|| loop {
                let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(&(r, seed)) = jobs.get(j) else { break };
                let t = run(&specs[r / nr], rungs[r % nr], seed, n_train);
                let mut res = results.lock().unwrap();
                let sum = &mut res[r];
                sum.episodes += t.episodes;
                sum.holds += t.holds;
                sum.start_held += t.start_held;
                sum.tell += t.tell;
                sum.read += t.read;
                sum.reinstated_tell += t.reinstated_tell;
                sum.reinstated_read += t.reinstated_read;
                sum.reinstated_story += t.reinstated_story;
                sum.tell_right += t.tell_right;
            });
        }
    });
    let pc = |a: usize, b: usize| 100.0 * a as f64 / b.max(1) as f64;
    for (r, sum) in results.into_inner().unwrap().into_iter().enumerate() {
        if r % nr == 0 {
            println!("{}:", specs[r / nr]);
        }
        let rung = &rungs[r % nr];
        println!(
            "  {:<14} {:>6.2} {:>10.1}% {:>12.1}% {:>12.1}% {:>16.1}% {:>9.1}%",
            rung.name,
            sum.holds as f64 / sum.episodes as f64,
            pc(sum.start_held, sum.episodes),
            pc(sum.reinstated_tell, sum.tell),
            pc(sum.reinstated_read, sum.read),
            pc(sum.reinstated_story, sum.episodes),
            pc(sum.tell_right, sum.tell * T),
        );
    }
}
