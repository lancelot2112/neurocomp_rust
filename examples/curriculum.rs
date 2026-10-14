//! Evolution through a curriculum: smaller tasks first, the population carried from each stage to
//! the next, and the best always also checked on the original task (serial reversal, the paying
//! light swapping every 100 trials of a 400-trial lifetime) on held-out lifetimes.
//!
//! Stages (`CUR_STAGES`, comma-separated `name:generations`):
//! - `instrumental`: click while the light is on (one light, paying every time it shows);
//! - `delayed`: the light flashes once; a click three to five steps later pays;
//! - `reversal1`: serial reversal with one swap per lifetime (every 200 trials);
//! - `reversal`: the original (every 100 trials).
//!
//! `CUR_SCORE=phase` (reversal stages score a lifetime's worst phase), `CUR_LIVES_REV` (lifetimes a
//! genome on reversal stages), `CUR_CHANNELS` (empty channels in a naked tag genome),
//! `CUR_SHOW=1` prints the current best network under each row.
//! `CUR_GENOME=edges|tags|stack`, `CUR_START=context|bare|naked` (the context circuit; a bare start,
//! a single tonic population on the act; or naked, an empty genome), `CUR_TRIALS`, `CUR_LIVES`, `CUR_HELD`, `CUR_POP`, `CUR_ELITE`,
//! `CUR_EVERY` (held-out check interval), `CUR_SEED`.

use neurocomp::evolve::genes::GeneGenome;
use neurocomp::evolve::tags::{TagGenome, Token};
use neurocomp::evolve::{evolve_population, fitness, fitness_detail, Evolvable, Genome, Physics, Task};
use neurocomp::world::Conditioning;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn task(name: &str, trials: usize) -> Task {
    // (`CUR_SCORE=phase`: reversal stages score each lifetime's worst phase)
    let worst = std::env::var("CUR_SCORE").map_or(false, |v| v == "phase");
    match name {
        "instrumental" => Task::Conditioning(Conditioning::Instrumental, trials),
        "delayed" => Task::Conditioning(Conditioning::Delayed, trials),
        "reversal1" => Task::Reversal { trials, period: trials / 2, worst_phase: worst },
        _ => Task::Reversal { trials, period: trials / 4, worst_phase: worst },
    }
}

fn run<G: Evolvable>(start: G, size: impl Fn(&G) -> usize, describe: impl Fn(&G) -> String) {
    let trials: usize = env("CUR_TRIALS", 400);
    let lives: u64 = env("CUR_LIVES", 16);
    let held_n: u64 = env("CUR_HELD", 16);
    let (pop, elite, every, seed) = (env("CUR_POP", 48usize), env("CUR_ELITE", 12usize), env("CUR_EVERY", 10usize), env("CUR_SEED", 7u64));
    let stages = std::env::var("CUR_STAGES").unwrap_or_else(|_| "instrumental:20,reversal1:20,reversal:30".into());
    let original = Task::Reversal { trials, period: trials / 4, worst_phase: false };
    let original_worst = Task::Reversal { trials, period: trials / 4, worst_phase: true };
    // (`CUR_LIVES_REV`: lifetimes a genome on reversal stages; default CUR_LIVES)
    let lives_rev: u64 = env("CUR_LIVES_REV", lives);
    let show = env("CUR_SHOW", 0u8) == 1;
    let train_rev: Vec<u64> = (0..lives_rev).collect();
    let train: Vec<u64> = (0..lives).collect();
    let held: Vec<u64> = (10_000..10_000 + held_n).collect();
    // held out: the stage's score, then the original's (pooled and worst phase); and, on the stage,
    // the discrimination pooled over every held-out trial and how many lifetimes never clicked
    let check = |g: &G, t: Task| {
        let (h, r, silent) = fitness_detail(g, t, Physics::Varied, &held);
        (h, fitness(g, original, Physics::Varied, &held).0, fitness(g, original_worst, Physics::Varied, &held).0, r.discrimination(), silent)
    };
    let mut population: Vec<G> = vec![start; pop];
    println!("pop {pop}, elite {elite}, {lives} lifetimes a genome, {held_n} held out, {trials} trials a lifetime; stages {stages}");
    println!("{:<13} {:>4}  {:>5} {:>5}  {:>5}  {:>11}  {:>14}  {:>14}  {:>10}  {:>6}", "stage", "gen", "best", "mean", "size", "held:stage", "held:original", "held:orig-worst", "held:disc", "silent");
    let (s0, o0, w0, d0, q0) = check(&population[0], task(stages.split(',').next().unwrap().split(':').next().unwrap(), trials));
    println!("{:<13} {:>4}  {:>5} {:>5}  {:>5}  {s0:>11}  {o0:>14}  {w0:>14}  {d0:>10}  {q0:>3}/{}", "start", "-", "", "", size(&population[0]), held.len());
    for (k, st) in stages.split(',').enumerate() {
        let mut it = st.split(':');
        let name = it.next().unwrap().to_string();
        let gens: usize = it.next().and_then(|g| g.parse().ok()).unwrap_or(20);
        let t = task(&name, trials);
        let tr = if name.starts_with("reversal") { &train_rev } else { &train };
        population = evolve_population(population, t, Physics::Varied, tr, elite, gens, seed + k as u64, |g, r| {
            if g % every == 0 || g + 1 == gens {
                let (h, o, w, d, q) = check(&r.best, t);
                println!("{name:<13} {g:>4}  {:>5} {:>5}  {:>5}  {h:>11}  {o:>14}  {w:>14}  {d:>10}  {q:>3}/{}", r.best_fitness, r.mean_fitness, size(&r.best), held.len());
                // (`CUR_SHOW=1`: the current best network under each row)
                if show {
                    print!("{}", describe(&r.best));
                }
            }
        });
    }
    let (_, o, w, _, _) = check(&population[0], original);
    println!("final best: held out on the original {o}/1024 pooled, {w}/1024 worst phase");
    print!("{}", describe(&population[0]));
}

fn main() {
    let start = std::env::var("CUR_START").unwrap_or_default();
    let bare = start == "bare";
    let naked = start == "naked";
    match std::env::var("CUR_GENOME").unwrap_or_else(|_| "edges".into()).as_str() {
        "tags" => {
            println!("curriculum, tag genome from {}", if bare { "a bare start" } else { "the context circuit" });
            if naked {
                // (`CUR_CHANNELS=n`: the empty string plus n empty channels, room for populations
                // to resolve to instead of crowding onto the senses and the act)
                let n: usize = env("CUR_CHANNELS", 0);
                println!("(naked: an empty string, {n} empty channels)");
                let tokens = (0..n).map(|i| Token::Channel { tag: (0x11u8).wrapping_mul(i as u8 + 1) ^ 0x5A, width: 3, hold: false }).collect();
                return run(TagGenome { tokens, next_id: 1 }, |g| g.tokens.len(), |g| g.describe());
            }
            run(if bare { TagGenome::bare_seed() } else { TagGenome::context_seed() }, |g| g.tokens.len(), |g| g.describe());
        }
        "stack" => {
            println!("curriculum, stack genome from an empty program");
            run(Genome { code: vec![] }, |g| g.code.len(), |_| String::new());
        }
        _ => {
            println!("curriculum, edge genome from {}", if bare { "a bare start" } else { "the context circuit" });
            if naked {
                println!("(naked: no edges)");
                return run(GeneGenome { edges: vec![], next_id: 2 }, |g| g.edges.len(), |_| String::new());
            }
            run(if bare { GeneGenome::act_seed(8) } else { GeneGenome::context_seed(8) }, |g| g.edges.len(), |_| String::new());
        }
    }
}
