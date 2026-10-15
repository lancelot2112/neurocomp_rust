//! What an island's genomes do on the maze's repeat rung (no stem, the food on one side all
//! lifetime, no cue): per lifetime, how many trials end in a turn, how many turns were stroked,
//! and how often a turn repeats the last one after a stroke and after none. Tells whether a stuck
//! rung is missing choice (no turns) or learning from reward (turns, but no repeat).
//! `MZC_FILE` (checkpoint.txt), `MZC_ISLAND` (maze), `MZC_TOP` (4), `MZC_WORLDS` (24),
//! `MZC_INPLACE` (0), `MZC_TRIALS` (200).

use neurocomp::evolve::tags::TagGenome;
use neurocomp::evolve::{live_lesson, Lesson, Physics, Task};
use neurocomp::world::SceneKind;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    neurocomp::program::neurocomp::set_in_place(env("MZC_INPLACE", 0u8) == 1);
    let path = std::env::var("MZC_FILE").unwrap_or_else(|_| "checkpoint.txt".into());
    let island = std::env::var("MZC_ISLAND").unwrap_or_else(|_| "maze".into());
    let top: usize = env("MZC_TOP", 4);
    let n: u64 = env("MZC_WORLDS", 24);
    let trials: usize = env("MZC_TRIALS", 200);
    let text = std::fs::read_to_string(&path).expect("checkpoint");
    let lesson = Lesson { maze_len: 0, maze_fixed: true, maze_nocue: true, ..Lesson::default() };
    for part in text.split("island ").skip(1) {
        let (head, body) = part.split_once('\n').unwrap_or((part, ""));
        if head.split_whitespace().next() != Some(island.as_str()) {
            continue;
        }
        println!("{head}: per lifetime ({n} worlds, {trials} trials): turned, stroked, repeat after a stroke / after none");
        for (k, g) in TagGenome::from_text(body).expect("genomes").iter().take(top).enumerate() {
            let (mut turned, mut stroked, mut tr) = (0usize, 0usize, 0usize);
            let (mut ws, mut wn, mut ls, mut ln) = (0usize, 0usize, 0usize, 0usize);
            for s in 0..n {
                let (ev, _) = live_lesson(g, Task::Maze(trials), Physics::Varied, 60_000 + s, 6, lesson);
                tr += ev.iter().filter_map(|e| e.maze.map(|m| m.0)).collect::<std::collections::BTreeSet<_>>().len();
                let turns: Vec<(Option<usize>, bool)> = ev.iter().filter(|e| e.scene == SceneKind::Maze).filter_map(|e| e.chose.map(|c| (e.moved, c))).collect();
                turned += turns.len();
                stroked += turns.iter().filter(|t| t.1).count();
                for w in turns.windows(2) {
                    let same = (w[0].0 == w[1].0) as usize;
                    if w[0].1 { ws += same; wn += 1 } else { ls += same; ln += 1 }
                }
            }
            let pct = |a: usize, b: usize| if b == 0 { "-".to_string() } else { format!("{}%", a * 100 / b) };
            println!("  rank {k}: turned {} of trials, stroked {} of turns, repeat {} after a stroke / {} after none", pct(turned, tr), pct(stroked, turned), pct(ws, wn), pct(ls, ln));
        }
    }
}
