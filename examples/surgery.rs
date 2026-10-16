//! How far an evolved genome is from learning from reward: hand edits to it, scored on the
//! repeat rung (the maze with no stem, the food on one side all lifetime, no cue), on fresh worlds.
//! A measure only: an edited genome is never fed back to evolution.
//!
//! `SRG_FILE` (checkpoint.txt), `SRG_ISLAND` (switching), `SRG_TOP` (8: genomes scanned from the
//! top), `SRG_RANK` (the one to edit; default the best scanned), `SRG_WORLDS` (48), `SRG_TRIALS`
//! (200), `SRG_FROM` (70,000: the first world seed), `SRG_LAG` (repeat lags tried, 1-6), `SRG_INPLACE`, `SRG_RATION`, `SRG_FOOD` (as the
//! run's). Edits, each scored on its own and stacked:
//! - `seed`: the grammar's hand-written act learner alone (`TagGenome::act_seed`), a control;
//! - `aff`: a channel copying the affect port (a wire), which a gate can name;
//! - `learner`: with `aff`, a population from the eye to the hand, growing kernels that repeat the
//!   act made `lag` steps back, learning only while the affect copy is on;
//! - `fade`: the learner's repeat kernels fading at chance 2^-fade a step (2 or 4);
//! - `babbler`: a tonic population from the eye into the hand, firing at random;
//! - `gatehand`: with `aff`, every existing population into the hand or act made such a learner
//!   (grow, repeat, gate on the affect copy): property changes only, no new population.

use neurocomp::evolve::tags::{Pop, TagGenome, Token, AFFECT, EYE, HAND};
use neurocomp::evolve::{live_lesson, Lesson, Physics, Task};
use neurocomp::world::observe;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// The repeat probe's mean over `worlds`, and the share of turns repeated after a stroke and after
/// none.
fn score(g: &TagGenome, worlds: &[u64], trials: usize, lesson: Lesson) -> (i32, String) {
    let (mut sum, mut ws, mut wn, mut ls, mut ln) = (0i64, 0usize, 0usize, 0usize, 0usize);
    for &s in worlds {
        let (ev, _) = live_lesson(g, Task::Maze(trials), Physics::Varied, s, 6, lesson);
        sum += observe::repeat(&ev) as i64;
        let turns: Vec<(Option<usize>, bool)> = ev.iter().filter(|e| e.scene == neurocomp::world::SceneKind::Maze).filter_map(|e| e.chose.map(|c| (e.moved, c))).collect();
        for w in turns.windows(2) {
            let same = (w[0].0 == w[1].0) as usize;
            if w[0].1 { ws += same; wn += 1 } else { ls += same; ln += 1 }
        }
    }
    let pct = |a: usize, b: usize| if b == 0 { "-".to_string() } else { format!("{}%", a * 100 / b) };
    ((sum / worlds.len().max(1) as i64) as i32, format!("repeats {} after a stroke, {} after none", pct(ws, wn), pct(ls, ln)))
}

/// A channel tag as far as possible (Hamming) from every declared one.
fn free_tag(g: &TagGenome) -> u8 {
    let tags: Vec<u8> = g.tokens.iter().filter_map(|t| if let Token::Channel { tag, .. } = t { Some(*tag) } else { None }).collect();
    (0..=255u8).max_by_key(|&t| tags.iter().map(|&x| (t ^ x).count_ones()).min().unwrap_or(8)).unwrap()
}

/// The affect copy: a channel and a wire into it from the affect port. Returns its tag.
fn add_aff(g: &mut TagGenome) -> u8 {
    let tag = free_tag(g);
    g.tokens.insert(0, Token::Channel { tag, width: 0, hold: false });
    let id = g.next_id;
    g.next_id += 1;
    let w = Pop { from_port: true, cover: true, ..Pop::wire(id, AFFECT, tag) };
    g.tokens.insert(1, Token::Pop(w));
    tag
}

fn add_learner(g: &mut TagGenome, aff: u8, lag: u8, fade: u8) {
    let id = g.next_id;
    g.next_id += 1;
    let p = Pop { from_port: true, to_port: true, cover: true, tile: 6, grow: true, repeat: true, repeat_lag: lag, gate: Some((aff, true)), ..Pop::new(id, EYE, HAND) };
    let p = if fade > 0 { Pop { fade, ..p } } else { p };
    g.tokens.push(Token::Pop(p));
}

/// A babbler: tonic kernels from the eye into the hand, firing at random (chance 2^-3 a step).
fn add_babbler(g: &mut TagGenome) {
    let id = g.next_id;
    g.next_id += 1;
    g.tokens.push(Token::Pop(Pop { from_port: true, to_port: true, cover: true, tile: 6, count: 8, tonic: 3, ..Pop::new(id, EYE, HAND) }));
}

/// Every kernel population into an output port made a reward learner. Returns how many.
fn gate_hand(g: &mut TagGenome, aff: u8, lag: u8) -> usize {
    let mut n = 0;
    for t in &mut g.tokens {
        if let Token::Pop(p) = t {
            if p.to_port && !p.wire {
                p.grow = true;
                p.repeat = true;
                p.repeat_lag = lag;
                p.gate = Some((aff, true));
                n += 1;
            }
        }
    }
    n
}

fn main() {
    neurocomp::program::neurocomp::set_in_place(env("SRG_INPLACE", 1u8) == 1);
    let ration: u64 = env("SRG_RATION", 256);
    neurocomp::evolve::set_energy(ration, ration * env("SRG_FOOD", 32u64));
    let path = std::env::var("SRG_FILE").unwrap_or_else(|_| "checkpoint.txt".into());
    let island = std::env::var("SRG_ISLAND").unwrap_or_else(|_| "switching".into());
    let top: usize = env("SRG_TOP", 8);
    let n: u64 = env("SRG_WORLDS", 48);
    let trials: usize = env("SRG_TRIALS", 200);
    let lags: Vec<u8> = std::env::var("SRG_LAG").ok().map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_else(|| (1..=6).collect());
    let from: u64 = env("SRG_FROM", 70_000);
    let worlds: Vec<u64> = (from..from + n).collect();
    let lesson = Lesson { maze_len: 0, maze_fixed: true, maze_nocue: true, ..Lesson::default() };
    let text = std::fs::read_to_string(&path).expect("checkpoint");
    let body = text.split("island ").skip(1).find(|p| p.split_whitespace().next() == Some(island.as_str())).expect("island");
    let (head, body) = body.split_once('\n').unwrap();
    let gs = TagGenome::from_text(body).expect("genomes");
    println!("{head}; repeat rung, {n} fresh worlds, {trials} trials (goal 300)");
    let mut best = (i32::MIN, 0usize);
    for (k, g) in gs.iter().take(top).enumerate() {
        let (s, how) = score(g, &worlds, trials, lesson);
        println!("  rank {k}: repeat {s:5}  ({} populations; {how})", g.populations());
        if s > best.0 {
            best = (s, k);
        }
    }
    let k: usize = env("SRG_RANK", best.1);
    let base = &gs[k];
    println!("editing rank {k} ({} populations, {} channels):", base.populations(), base.channels());
    let (s, how) = score(&TagGenome::act_seed(), &worlds, trials, lesson);
    println!("  control, the hand-written act learner alone: repeat {s:5}  ({how})");
    {
        let mut g = base.clone();
        add_aff(&mut g);
        let (s, how) = score(&g, &worlds, trials, lesson);
        println!("  + aff (1 channel, 1 wire): repeat {s:5}  ({how})");
    }
    for &lag in &lags {
        let mut g = base.clone();
        let aff = add_aff(&mut g);
        let changed = gate_hand(&mut g, aff, lag);
        let (s, how) = score(&g, &worlds, trials, lesson);
        println!("  + aff + gatehand lag {lag} ({changed} populations changed, 3 properties each): repeat {s:5}  ({how})");
    }
    for &lag in &lags {
        let mut g = base.clone();
        let aff = add_aff(&mut g);
        add_learner(&mut g, aff, lag, 0);
        let (s, how) = score(&g, &worlds, trials, lesson);
        println!("  + aff + learner lag {lag} (1 channel, 2 populations): repeat {s:5}  ({how})");
    }
    for &lag in &lags {
        for fade in [2u8, 4] {
            let mut g = base.clone();
            let aff = add_aff(&mut g);
            add_learner(&mut g, aff, lag, fade);
            let (s, how) = score(&g, &worlds, trials, lesson);
            println!("  + aff + learner lag {lag}, fading 1/2^{fade} a step (1 channel, 2 populations): repeat {s:5}  ({how})");
        }
        let mut g = base.clone();
        let aff = add_aff(&mut g);
        add_learner(&mut g, aff, lag, 0);
        add_babbler(&mut g);
        let (s, how) = score(&g, &worlds, trials, lesson);
        println!("  + aff + learner lag {lag} + babbler (1 channel, 3 populations): repeat {s:5}  ({how})");
    }
}
