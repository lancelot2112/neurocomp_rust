//! A mind in a world: speech play, mouse play, then lessons (read it, tell it), with every reward
//! the mind's own. The world gives none; an observer outside both scores what happened.
//!
//! Environment: SEED (0), LS_SPEECH (600 steps of speech play), LS_PLAY (300 mouse-play
//! episodes), LS_TRAIN (3000 lessons), LS_TEST (300 lessons), LS_PRACTICE (0.25: the share of
//! training lessons where the teacher listens while the child does the task), LS_SLEEP (10:
//! lessons between sleeps), LS_SPONT (2: a primitive of each effector is spontaneously active 1
//! step in this many).
//!
//! At test the teacher only listens (no demonstration, no correction), and smiles at right
//! words as always; the mind does not know it is being tested, and keeps learning.

use neurocomp::program::mind::{Config, Mind};
use neurocomp::program::world::{Acts, Book, Lesson, Phase, Task, Who, World};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

#[derive(Default, Clone, Copy)]
struct Tally {
    lessons: usize,
    words: usize,
    right: usize,
    silent: usize,
    said_story: usize,
    said_book: usize,
    opened: usize,
}

fn main() {
    let seed: u64 = env("SEED", 0);
    let n_speech: usize = env("LS_SPEECH", 600);
    let n_play: usize = env("LS_PLAY", 300);
    let n_train: usize = env("LS_TRAIN", 3000);
    let n_test: usize = env("LS_TEST", 300);
    let practice: f64 = env("LS_PRACTICE", 0.25);
    let sleep_every: usize = env("LS_SLEEP", 10);
    let mut rng = StdRng::seed_from_u64(seed ^ 0xC0FE);
    let mut world = World::new(seed);
    let mut mind = Mind::with_config(world.body(), seed, Config::from_env());
    mind.spontaneous = env("LS_SPONT", 1);
    let t0 = std::time::Instant::now();
    let mut acts = Acts::default();
    // the valuations summed (own effect, response, smile, effort), per stage
    let mut valued = [[0i64; 4]; 4];
    // per stage: (steps, goals taken, acts released from a goal, acts, imagined, recall reached the cortex)
    let mut inner = [[0usize; 6]; 4];
    macro_rules! run {
        ($stage:expr) => {{
            let start = world.events.len();
            while !world.done() {
                let senses = world.step(&acts);
                acts = mind.step(&senses);
                for k in 0..4 {
                    valued[$stage][k] += mind.report.valued[k] as i64;
                }
                let r = &mind.report;
                let x = &mut inner[$stage];
                x[0] += 1;
                x[1] += r.goals_taken;
                x[2] += r.from_goal as usize;
                x[3] += (r.spoke as usize) + (r.moved as usize);
                x[4] += r.inward as usize;
                x[5] += r.recalled as usize;
            }
            start
        }};
    }
    // speech play: a continuous stream; a caregiver comes and goes
    if n_speech > 0 {
        world.speech_play(n_speech);
        let start = run!(0);
        // said aloud, by halves and by whether a caregiver answered anything in the window
        let ev = &world.events[start..];
        let half = ev.len() / 2;
        let aloud = |e: &[neurocomp::program::world::Event]| e.iter().filter(|x| x.said.iter().any(|s| s.0 == Who::Child)).count();
        let replies = ev.iter().filter(|x| x.said.iter().any(|s| s.0 == Who::Caregiver)).count();
        println!("speech play: {} steps; said aloud on {} steps in the first half, {} in the second; caregiver replies {}", ev.len(), aloud(&ev[..half]), aloud(&ev[half..]), replies);
    }
    // mouse play
    if n_play > 0 {
        let mut opened = [0usize; 2];
        let mut moves = [0usize; 2];
        // (clicks on a book, clicks) by half
        let mut clicks = [(0usize, 0usize); 2];
        for k in 0..n_play {
            world.mouse_play();
            let start = run!(1);
            let h = (2 * k >= n_play) as usize;
            opened[h] += world.events[start..].iter().filter(|e| e.opened).count();
            moves[h] += world.events[start..].iter().filter(|e| e.moved.is_some()).count();
            for e in &world.events[start..] {
                if e.moved == Some(neurocomp::program::world::CLICK) {
                    clicks[h].0 += e.opened as usize;
                    clicks[h].1 += 1;
                }
            }
        }
        println!("mouse play: {n_play} episodes; books opened {} then {} (first, second half); hand moves {} then {}; clicks on a book {} of {} then {} of {}", opened[0], opened[1], moves[0], moves[1], clicks[0].0, clicks[0].1, clicks[1].0, clicks[1].1);
    }
    // lessons
    let books = [Book::Same, Book::Other, Book::Closed];
    let mut res = [[Tally::default(); 3]; 2];
    let mut train = [(0usize, 0usize, 0usize); 2];
    // per phase (story, instruction, wait, task by teacher, task by child, pause): (steps, child aloud, hand moves, smiles)
    let mut by_phase = [[0usize; 4]; 6];
    let mut steps = 0usize;
    for lesson in 0..n_train + n_test {
        let testing = lesson >= n_train;
        let read = rng.gen_bool(0.5);
        let book = if read { books[rng.gen_range(0..2)] } else { books[rng.gen_range(0..3)] };
        let task = if read { Task::Read } else { Task::Tell };
        let child_does = testing || rng.gen_bool(practice);
        world.lesson(Lesson { task, book, child_does, demonstrate: !testing, correct: !testing });
        let start = run!(if testing { 3 } else { 2 });
        let ev = &world.events[start..];
        // the observer: the words wanted in the task and what the child said
        let mut t = Tally { lessons: 1, ..Default::default() };
        steps += ev.len();
        for e in ev {
            let k = match e.phase {
                Phase::Story => 0,
                Phase::Instruction => 1,
                Phase::Wait => 2,
                Phase::Task(false) => 3,
                Phase::Task(true) => 4,
                _ => 5,
            };
            by_phase[k][0] += 1;
            by_phase[k][1] += e.said.iter().any(|s| s.0 == Who::Child) as usize;
            by_phase[k][2] += e.moved.is_some() as usize;
            by_phase[k][3] += e.stroked as usize;
            if let (Phase::Task(true), Some(w)) = (e.phase, e.want) {
                t.words += 1;
                match e.said.iter().find(|s| s.0 == Who::Child) {
                    Some(&(_, c)) => t.right += (c == w) as usize,
                    None => t.silent += 1,
                }
            }
            t.opened += (e.phase == Phase::Wait && e.opened) as usize;
        }
        if testing {
            let r = &mut res[!read as usize][books.iter().position(|b| *b == book).unwrap()];
            r.lessons += 1;
            r.words += t.words;
            r.right += t.right;
            r.silent += t.silent;
            r.opened += t.opened;
        } else if child_does {
            let k = (lesson * 2 >= n_train) as usize;
            train[k].0 += t.words;
            train[k].1 += t.right;
            train[k].2 += t.silent;
        }
        if !testing && sleep_every > 0 && (lesson + 1) % sleep_every == 0 {
            mind.sleep(3);
        }
        if !testing && (lesson + 1) % 500 == 0 {
            let names = ["story", "instr", "wait", "task-T", "task-C", "pause"];
            let line: Vec<String> = (0..6).map(|k| format!("{} {}/{}/{}/{}", names[k], by_phase[k][0], by_phase[k][1], by_phase[k][2], by_phase[k][3])).collect();
            println!("  phases (steps/aloud/moves/touches): {}; {:.0} steps per lesson", line.join(", "), steps as f64 / 500.0);
            by_phase = [[0usize; 4]; 6];
            steps = 0;
            println!(
                "  lesson {:5}: practice words right {:.1}%, silent {:.1}% (this half so far); hippocampus rows {} ({:.0} s)",
                lesson + 1,
                100.0 * train[(lesson * 2 >= n_train) as usize].1 as f64 / train[(lesson * 2 >= n_train) as usize].0.max(1) as f64,
                100.0 * train[(lesson * 2 >= n_train) as usize].2 as f64 / train[(lesson * 2 >= n_train) as usize].0.max(1) as f64,
                mind.hc.mem.rows(),
                t0.elapsed().as_secs_f64()
            );
        }
    }
    println!("practice: words right {:.1}% then {:.1}%, silent {:.1}% then {:.1}% (first, second half of training)", 100.0 * train[0].1 as f64 / train[0].0.max(1) as f64, 100.0 * train[1].1 as f64 / train[1].0.max(1) as f64, 100.0 * train[0].2 as f64 / train[0].0.max(1) as f64, 100.0 * train[1].2 as f64 / train[1].0.max(1) as f64);
    println!("test ({n_test} lessons): words right / silent / book opened while waiting");
    for (i, name) in ["read", "tell"].iter().enumerate() {
        for (j, b) in ["same", "other", "closed"].iter().enumerate() {
            let r = res[i][j];
            if r.lessons == 0 {
                continue;
            }
            println!("  {name} book {b:<6}: {:5.1}% right, {:5.1}% silent, opened in {} of {} lessons", 100.0 * r.right as f64 / r.words.max(1) as f64, 100.0 * r.silent as f64 / r.words.max(1) as f64, r.opened, r.lessons);
        }
    }
    let names = ["speech play", "mouse play", "training", "test"];
    for (k, v) in valued.iter().enumerate() {
        println!("  dopamine from {:<11}: own effect {:.1}, response {:.1}, touch {:.1}, effort {:.1}", names[k], v[0] as f64 / 65536.0, v[1] as f64 / 65536.0, v[2] as f64 / 65536.0, v[3] as f64 / 65536.0);
    }
    for (k, x) in inner.iter().enumerate() {
        println!("  {:<11}: {} steps; goals taken per step {:.2}; acts {} ({} from a goal); imagined {}; recall reached the cortex on {:.0}% of steps", names[k], x[0], x[1] as f64 / x[0].max(1) as f64, x[3], x[2], x[4], 100.0 * x[5] as f64 / x[0].max(1) as f64);
    }
    println!("loops: {} columns with a loop", mind.report.loops);
    let (c, o, b, r, a) = mind.hc.stats;
    println!("hippocampus: {} rows, {c} completions ({o} reached the cortex), {b} boundaries, {r} relearned awake, {a} asleep; {:.0} s", mind.hc.mem.rows(), t0.elapsed().as_secs_f64());
}
