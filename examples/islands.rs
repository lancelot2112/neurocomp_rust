//! Islands: populations pressed on different capacities, trading circuits.
//!
//! Each island is the tag genome from a naked start, evolving by epsilon-lexicase on its own
//! probes (`observe::probes`):
//! - switching: respond, tell, discriminate, reverse, win-stay (serial reversal);
//! - recall: respond, remember (a light gone a few steps ago);
//! - reading: copy (say the token shown on the eye a few steps before) and echo (say the token
//!   the teacher said a few steps before: the stepping stone, sound to voice with no glyph);
//! - generalist: all ten (surprise on a cycle of 3 broken one time in 8);
//! - prediction (`ISL_PREDICTION=1`): surprise, clicking after a token that broke the stream's
//!   cycle;
//! - maze (`ISL_MAZE=1`): a T-maze, turning at the junction to the side of the food the cue shows;
//! - screen (`ISL_SCREEN=1`): move a cursor over a grid of cells to the target glyph and click;
//! - shapes (`ISL_SHAPES=1`): respond, tell and discriminate with the lights distorted into
//!   shapes, a fresh exemplar every trial.
//! Every `ISL_EVERY` generations each island takes `ISL_PLASMIDS` plasmids (one population of a
//! donor with the circuit of any channel it touches, made private; or the population alone if it
//! only reads and writes ports) from the best genomes of other islands into its new children, so a circuit grown under one pressure is tried under another. Each plasmid is
//! recorded (from, to, the circuit) in `ISL_LOG`; at each exchange every island's best genome is
//! scored on every probe on held-out worlds (does a circuit carry over?).
//!
//! Every `ISL_REPORT` generations, per island: its population's means on its own probes (training
//! worlds), its best genome's probes on held-out worlds, its work per step and which ports its
//! genomes read and write.
//!
//! `ISL_POP` (16 a island), `ISL_ELITE` (4), `ISL_GENS` (300), `ISL_LIVES` (12 training worlds),
//! `ISL_HELD` (12), `ISL_TRIALS` (200), `ISL_GAP` (6), `ISL_EVERY` (20), `ISL_PLASMIDS` (2),
//! `ISL_REPORT` (10), `ISL_LOG` (plasmids.txt), `ISL_SEED` (7), `ISL_THRIFT` (1).
//!
//! Curricula (`ISL_LADDERS=1`): an island can climb a ladder of stages, each its own probes, its
//! own world (`Lesson`: tokens and how often the teacher speaks) and a goal; it moves to the next
//! stage once its population's training mean meets the goal for `hold` generations in a row
//! (`evolve_islands`). Switching climbs respond, tell, win-stay, discriminate (no swaps), then one
//! swap, then three; recall clicks after a light gone 1, 2, 3 (the plain task), then 4 steps
//! before the window. Reading's: babble, then echo with 2, 4 and 8 tokens
//! with the teacher on every token, then the teacher fading (6, 4, 2, 1 in 8) so the glyph must
//! carry more of the work. `ISL_LADDER_<island>` replaces an island's ladder: stages separated by
//! `;`, each `name key=value ...` with keys `probes` (probe names joined by `+`), `tokens`,
//! `teach` (eighths), `period` and `noise` (surprise's stream), `blur` and `shift` (shapes), `maze`, `gone` and `symbol` (the T-maze), `sw`, `sh` and `lit` (the screen), `swaps` (of the paying light in a reversal lifetime), `delay` (steps from the
//! delayed light to the response window), `goal` (`probe@score`, joined by `+`) and `hold` (3). The cross-island table
//! at each exchange is always in the plain world, on every probe.
//!
//! Checkpoints (also every `ISL_CKPT` generations, if set): at every exchange each island's whole population is written to `ISL_CHECKPOINT`
//! (checkpoint.txt, as text); `ISL_RESUME=<file>` starts every island named in it from its saved
//! population and generation (islands not in it start naked), so a change to the world or the
//! probes can be injected into a running experiment: stop it, change, resume.

use std::io::Write;

use neurocomp::evolve::tags::{TagGenome, Token};
use neurocomp::evolve::{evolve_islands, probe_all_in, probe_means, Island, Lesson, Physics, Stage, ALL_PROBES};
use neurocomp::world::observe::{ProbeRow, PROBES};
use rand::rngs::StdRng;
use rand::Rng;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn row(m: &ProbeRow, only: &[usize]) -> String {
    (0..m.len()).map(|p| if only.contains(&p) { format!("{:>7}", m[p]) } else { format!("{:>7}", ".") }).collect::<Vec<_>>().join("")
}

/// A probe by (the start of) its name.
fn probe_named(n: &str) -> usize {
    PROBES.iter().position(|p| p.starts_with(n)).unwrap_or_else(|| panic!("no probe named {n}"))
}

/// A ladder from its text (see the header).
fn ladder(text: &str) -> Vec<Stage> {
    text.split(';').filter(|x| !x.trim().is_empty()).map(|st| {
        let mut it = st.split_whitespace();
        let name = it.next().unwrap_or("stage").to_string();
        let mut stage = Stage { name, probes: vec![], lesson: Lesson::default(), goal: vec![], hold: 3 };
        for kv in it {
            let (k, v) = kv.split_once('=').expect("key=value");
            match k {
                "probes" => stage.probes = v.split('+').map(probe_named).collect(),
                "tokens" => stage.lesson.tokens = v.parse().expect("tokens"),
                "teach" => stage.lesson.teach = v.parse().expect("teach"),
                "swaps" => stage.lesson.swaps = v.parse().expect("swaps"),
                "delay" => stage.lesson.delay = v.parse().expect("delay"),
                "period" => stage.lesson.period = v.parse().expect("period"),
                "noise" => stage.lesson.noise = v.parse().expect("noise"),
                "blur" => stage.lesson.blur = v.parse().expect("blur"),
                "shift" => stage.lesson.shift = v.parse().expect("shift"),
                "maze" => stage.lesson.maze_len = v.parse().expect("maze"),
                "gone" => stage.lesson.cue_gone = v == "1",
                "symbol" => stage.lesson.symbol = v == "1",
                "sw" => stage.lesson.screen_w = v.parse().expect("sw"),
                "sh" => stage.lesson.screen_h = v.parse().expect("sh"),
                "lit" => stage.lesson.screen_lit = v.parse().expect("lit"),
                "hold" => stage.hold = v.parse().expect("hold"),
                "goal" => stage.goal = v.split('+').map(|g| {
                    let (p, x) = g.split_once('@').expect("probe@score");
                    (probe_named(p), x.parse().expect("score"))
                }).collect(),
                _ => panic!("unknown key {k}"),
            }
        }
        stage
    }).collect()
}

/// Reading's ladder: a varied voice, then echoing (sound to voice, no glyph) with more and more
/// tokens, then the teacher fading so the glyph must carry more of the work.
const READING_LADDER: &str = "babble probes=babble tokens=2 teach=8 goal=babble@768; \
    echo2 probes=babble+echo tokens=2 teach=8 goal=echo@150; \
    echo4 probes=babble+echo tokens=4 teach=8 goal=echo@150; \
    echo8 probes=babble+echo tokens=8 teach=8 goal=echo@150; \
    fade6 probes=echo+copy tokens=8 teach=6 goal=copy@100; \
    fade4 probes=echo+copy tokens=8 teach=4 goal=copy@100; \
    fade2 probes=echo+copy tokens=8 teach=2 goal=copy@100; \
    read probes=echo+copy tokens=8 teach=1";

/// Switching's ladder: see a light, tell the two apart, repeat what was stroked, prefer the
/// paying light, then follow one swap, then three.
const SWITCHING_LADDER: &str = "respond probes=respond swaps=0 goal=respond@100; \
    tell probes=respond+tell swaps=0 goal=tell@100; \
    winstay probes=respond+tell+win-stay swaps=0 goal=win-stay@80; \
    discrim probes=respond+tell+win-stay+discriminate swaps=0 goal=discriminate@100; \
    swap1 probes=respond+tell+win-stay+discriminate+reverse swaps=1 goal=reverse@80; \
    swap3 probes=respond+tell+win-stay+discriminate+reverse swaps=3";

/// Recall's ladder: click after a light gone 1 step before the window, then 2, then 3 (the plain
/// task), then 4.
const RECALL_LADDER: &str = "lag1 probes=respond+remember delay=1 goal=remember@150; \
    lag2 probes=respond+remember delay=2 goal=remember@150; \
    lag3 probes=respond+remember delay=3 goal=remember@150; \
    lag4 probes=respond+remember delay=4";

/// Prediction's ladder: a stream of tokens in a cycle, one in eight replaced by another; the
/// program is scored on clicking after a token that broke the cycle (surprise), with no code to
/// learn for any token. A break in a cycle of k is a token unlike the one k steps back, so each
/// stage asks for a longer reach into the past; the last ones need the two tokens before (a cycle
/// of 6 over 3 tokens repeats tokens) and a longer cycle.
const PREDICTION_LADDER: &str = "cycle1 probes=surprise period=1 noise=1 goal=surprise@200; \
    cycle2 probes=surprise period=2 noise=1 goal=surprise@200; \
    cycle3 probes=surprise period=3 noise=1 goal=surprise@200; \
    cycle4 probes=surprise period=4 noise=1 goal=surprise@200; \
    order2 probes=surprise period=6 tokens=3 noise=1 goal=surprise@150; \
    cycle8 probes=surprise period=8 noise=1";

/// Shapes' ladder: the two lights become shapes, a fresh exemplar every trial (each line moved
/// within its word with chance `blur` in 8, the whole pattern rotated by up to `shift` lines,
/// which carries it into other words and tiles), and the program must still tell them apart;
/// then prefer the paying one. No swaps, so only the shapes are hard.
const SHAPES_LADDER: &str = "clean probes=respond+tell swaps=0 goal=tell@150; \
    blur1 probes=respond+tell swaps=0 blur=1 goal=tell@150; \
    blur2 probes=respond+tell swaps=0 blur=2 goal=tell@150; \
    shift8 probes=respond+tell swaps=0 blur=1 shift=8 goal=tell@120; \
    shift64 probes=respond+tell swaps=0 blur=2 shift=64 goal=tell@100; \
    paying probes=respond+tell+discriminate swaps=0 blur=2 shift=16";

/// Maze's ladder: a T-maze, stroked at the food's arm. Turn to the side the cue is on with no stem,
/// then walk a stem of 2 first, then with the cue only at the stem's foot (remember the side), a
/// longer stem, then a cue that is a light's code (which side it means is learned from the
/// stroke), then that code only at the foot. The hand's codes are drawn each lifetime, so even
/// the first stage is learned within a lifetime, not wired in.
const MAZE_LADDER: &str = "turn probes=maze maze=0 goal=maze@200; \
    walk probes=maze maze=2 goal=maze@200; \
    gone probes=maze maze=2 gone=1 goal=maze@150; \
    long probes=maze maze=4 gone=1 goal=maze@150; \
    symbol probes=maze maze=0 symbol=1 goal=maze@150; \
    symbolgone probes=maze maze=2 gone=1 symbol=1";

/// Screen's ladder: move a cursor to the glyph on the screen and click it (stroked on the target).
/// Two cells side by side, then four in a row, then a 2 x 2 grid (up and down too), then 4 x 2;
/// then a distractor beside the target (which glyph is the target is learned from the stroke),
/// then three distractors on the full screen.
const SCREEN_LADDER: &str = "two probes=screen sw=2 sh=1 lit=1 goal=screen@300; \
    row probes=screen sw=4 sh=1 lit=1 goal=screen@250; \
    square probes=screen sw=2 sh=2 lit=1 goal=screen@250; \
    full probes=screen sw=4 sh=2 lit=1 goal=screen@200; \
    pick probes=screen sw=2 sh=2 lit=2 goal=screen@150; \
    crowd probes=screen sw=4 sh=2 lit=4";

/// The world the generalist and the cross-island table use: the plain world, with a cycle of 3
/// broken one time in 8 for surprise (the plain stream has no pattern to break).
fn plain() -> Lesson {
    Lesson { period: 3, noise: 1, ..Lesson::default() }
}

fn main() {
    let pop: usize = env("ISL_POP", 16);
    let elite: usize = env("ISL_ELITE", 4);
    let gens: usize = env("ISL_GENS", 300);
    let lives: u64 = env("ISL_LIVES", 12);
    let held_n: u64 = env("ISL_HELD", 12);
    let trials: usize = env("ISL_TRIALS", 200);
    let gap: usize = env("ISL_GAP", 6);
    let every: usize = env("ISL_EVERY", 20);
    let plasmids: usize = env("ISL_PLASMIDS", 2);
    let report_every: usize = env("ISL_REPORT", 10);
    let seed: u64 = env("ISL_SEED", 7);
    let thrift = env("ISL_THRIFT", 1u8) == 1;
    let log_path = std::env::var("ISL_LOG").unwrap_or_else(|_| "plasmids.txt".into());
    let resume = std::env::var("ISL_RESUME").ok();
    let mut log = if resume.is_some() {
        std::fs::OpenOptions::new().create(true).append(true).open(&log_path).expect("plasmid log")
    } else {
        std::fs::File::create(&log_path).expect("plasmid log")
    };
    let ck_path = std::env::var("ISL_CHECKPOINT").unwrap_or_else(|_| "checkpoint.txt".into());
    let ckpt: usize = env("ISL_CKPT", 0);
    let train: Vec<u64> = (0..lives).collect();
    let held: Vec<u64> = (10_000..10_000 + held_n).collect();
    let mut islands = vec![
        Island::plain("switching", vec![0, 1, 2, 3, 5]),
        Island::plain("recall", vec![0, 4]),
        Island::plain("reading", vec![6, 7]),
        Island { name: "generalist".into(), stages: vec![Stage { name: "all".into(), probes: ALL_PROBES.to_vec(), lesson: plain(), goal: vec![], hold: 0 }] },
    ];
    // (`ISL_PREDICTION=1`: a fifth island, prediction, on the surprise probe; it starts naked, or
    // from a checkpoint that has it)
    if env("ISL_PREDICTION", 0u8) == 1 {
        islands.push(Island { name: "prediction".into(), stages: vec![Stage { name: "all".into(), probes: vec![9], lesson: plain(), goal: vec![], hold: 0 }] });
    }
    // (`ISL_MAZE=1`: a T-maze; starts naked or from a checkpoint)
    if env("ISL_MAZE", 0u8) == 1 {
        islands.push(Island { name: "maze".into(), stages: vec![Stage { name: "all".into(), probes: vec![10], lesson: plain(), goal: vec![], hold: 0 }] });
    }
    // (`ISL_SCREEN=1`: a screen with a cursor; starts naked or from a checkpoint)
    if env("ISL_SCREEN", 0u8) == 1 {
        islands.push(Island { name: "screen".into(), stages: vec![Stage { name: "all".into(), probes: vec![11], lesson: plain(), goal: vec![], hold: 0 }] });
    }
    // (`ISL_SHAPES=1`: shapes, telling two distorted lights apart; starts naked or from a checkpoint)
    if env("ISL_SHAPES", 0u8) == 1 {
        let lesson = Lesson { swaps: 0, blur: 2, shift: 16, ..Lesson::default() };
        islands.push(Island { name: "shapes".into(), stages: vec![Stage { name: "all".into(), probes: vec![0, 1, 2], lesson, goal: vec![], hold: 0 }] });
    }
    // (`ISL_LADDERS=1`: the default ladders; `ISL_LADDER_<island>`: one's own)
    let ladders = env("ISL_LADDERS", 0u8) == 1;
    for isl in islands.iter_mut() {
        if let Ok(text) = std::env::var(format!("ISL_LADDER_{}", isl.name)) {
            isl.stages = ladder(&text);
        } else if ladders {
            match isl.name.as_str() {
                "reading" => isl.stages = ladder(READING_LADDER),
                "switching" => isl.stages = ladder(SWITCHING_LADDER),
                "recall" => isl.stages = ladder(RECALL_LADDER),
                "prediction" => isl.stages = ladder(PREDICTION_LADDER),
                "shapes" => isl.stages = ladder(SHAPES_LADDER),
                "maze" => isl.stages = ladder(MAZE_LADDER),
                "screen" => isl.stages = ladder(SCREEN_LADDER),
                _ => {}
            }
        }
    }
    let tokens = (0..4).map(|i| Token::Channel { tag: (0x11u8).wrapping_mul(i as u8 + 1) ^ 0x5A, width: 3, hold: false }).collect();
    let naked = TagGenome { tokens, next_id: 1 };
    // starting populations: naked, or a checkpoint's (by island name)
    let mut start: Vec<Vec<TagGenome>> = vec![vec![naked.clone()]; islands.len()];
    let mut first = 0usize;
    let mut stages = vec![0usize; islands.len()];
    if let Some(path) = &resume {
        let text = std::fs::read_to_string(path).expect("checkpoint");
        for part in text.split("island ").skip(1) {
            let (head, body) = part.split_once('\n').unwrap_or((part, ""));
            let mut h = head.split_whitespace();
            let name = h.next().unwrap_or("");
            let g: usize = h.nth(1).and_then(|x| x.parse().ok()).unwrap_or(0);
            // (and its stage, if it was climbing a ladder: `stage n`)
            let st: usize = h.nth(1).and_then(|x| x.parse().ok()).unwrap_or(0);
            if let Some(i) = islands.iter().position(|isl| isl.name == name) {
                start[i] = TagGenome::from_text(body).expect("checkpoint genomes");
                first = first.max(g + 1);
                stages[i] = st.min(islands[i].stages.len() - 1);
                println!("resumed {name} from {path}: {} genomes, generation {g}, stage {} ({})", start[i].len(), stages[i], islands[i].stages[stages[i]].name);
            }
        }
    }
    println!("islands: {} of {pop}, elite {elite}, {lives} worlds, {held_n} held out, {trials} trials, gaps 0-{gap}, exchange every {every} ({plasmids} plasmids an island), thrift {thrift}",
        islands.iter().map(|i| i.name.as_str()).collect::<Vec<_>>().join(", "));
    let names: String = PROBES.iter().map(|p| format!("{:>7}", &p[..p.len().min(6)])).collect();
    for isl in islands.iter().filter(|i| i.stages.len() > 1) {
        println!("{} climbs: {}", isl.name, isl.stages.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(" > "));
    }
    let mut was = stages.clone();
    // a plasmid: one circuit of the donor (a random declared channel and what touches it)
    let transfer = |recipient: &TagGenome, donor: &TagGenome, rng: &mut StdRng| -> (TagGenome, String) {
        if donor.populations() == 0 {
            return (recipient.clone(), "(nothing: the donor has no population)".into());
        }
        let mut child = recipient.clone();
        let k = rng.gen_range(0..donor.populations());
        let moved = child.take_population_plasmid(donor, k, rng);
        let text = TagGenome { tokens: moved, next_id: 0 }.describe();
        (child, text)
    };
    evolve_islands(&islands, start, stages, first, pop, trials, Physics::Varied, &train, gap, elite, gens, seed, thrift, every, plasmids, transfer, |g, r| {
        for (from, to, what) in &r.moved {
            let _ = writeln!(log, "generation {g}: {} -> {}\n{what}", islands[*from].name, islands[*to].name);
        }
        // a checkpoint at every exchange, every `ISL_CKPT` generations, and at the end: each
        // island's population, best first
        if (every > 0 && g % every == 0) || (ckpt > 0 && g % ckpt == 0) || g + 1 == gens {
            let mut ck = String::new();
            for (i, isl) in islands.iter().enumerate() {
                ck += &format!("island {} generation {g} stage {}\n", isl.name, r.stage[i]);
                for x in &r.ranked[i] {
                    ck += &x.to_text();
                }
            }
            let _ = std::fs::write(&ck_path, ck);
        }
        // every generation, one line: each climbing island's goal probes against their goals
        let goals: Vec<String> = islands.iter().enumerate().filter(|(_, isl)| isl.stages.len() > 1).filter_map(|(i, isl)| {
            let st = &isl.stages[r.stage[i]];
            if st.goal.is_empty() {
                return None;
            }
            let g: Vec<String> = st.goal.iter().map(|&(p, x)| format!("{} {}/{x}", PROBES[p], r.means[i][p])).collect();
            Some(format!("{} {} {} (met {}/{})", isl.name, st.name, g.join(" "), r.met[i], st.hold.max(1)))
        }).collect();
        if !goals.is_empty() {
            println!("{g:>4}  goals: {}", goals.join("; "));
        }
        // an island that met its stage's goal: it is scored on the next stage from this generation
        for (i, isl) in islands.iter().enumerate() {
            if r.stage[i] != was[i] {
                println!("{g:>4}  {} moves to stage {} ({}): goal met", isl.name, r.stage[i], isl.stages[r.stage[i]].name);
                was[i] = r.stage[i];
            }
        }
        if !r.moved.is_empty() {
            println!("{g:>4}  plasmids moved: {}", r.moved.iter().map(|(f, t, w)| format!("{}->{} ({} populations)", islands[*f].name, islands[*t].name, w.lines().filter(|l| l.contains("->")).count())).collect::<Vec<_>>().join(", "));
            // every island's best on every probe, held out: does a circuit carry over?
            println!("      every island's best on every probe, held out:{}", "");
            println!("      {:<11}{names}", "");
            for (i, isl) in islands.iter().enumerate() {
                let h = probe_all_in(vec![r.ranked[i][0].clone()], trials, Physics::Varied, &held, gap, &ALL_PROBES, plain());
                println!("      {:<11}{}", isl.name, row(&probe_means(&h[0].probes), &ALL_PROBES));
            }
        }
        if g % report_every == 0 || g + 1 == gens {
            println!("{g:>4}  {:<11}{names}   work   size  ports", "");
            for (i, isl) in islands.iter().enumerate() {
                let best = &r.ranked[i][0];
                let st = &isl.stages[r.stage[i]];
                let h = probe_all_in(vec![best.clone()], trials, Physics::Varied, &held, gap, &st.probes, st.lesson);
                let (rd, wr) = best.ports();
                let at = if isl.stages.len() > 1 { format!("  [stage {} {}, {} generations]", r.stage[i], st.name, r.held[i]) } else { String::new() };
                println!("      {:<11}{}  {:>5}  {:>5}  reads {} writes {}  (population, training){at}", isl.name, row(&r.means[i], &st.probes), r.work[i], best.tokens.len(), rd.join("+"), wr.join("+"));
                println!("      {:<11}{}  (its best, held out)", "", row(&probe_means(&h[0].probes), &st.probes));
            }
        }
    });
    println!("plasmids recorded in {log_path}");
}
