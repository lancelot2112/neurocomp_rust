//! Evolve graph programs in the world and print what comes out.
//!
//! `EVO_TASK=instrumental|delayed|reversal`, `EVO_TRIALS` (trials a lifetime), `EVO_SEEDS`
//! (world seeds a genome is scored on), `EVO_POP`, `EVO_ELITE`, `EVO_GENS`, `EVO_SEED`;
//! `EVO_PHYSICS=fixed|varied` (default varied), `EVO_TEST` (held-out world seeds the best genome is checked on at the end).

use neurocomp::evolve::{evolve, fitness, Physics, Task};
use neurocomp::world::Conditioning;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let kind = match std::env::var("EVO_TASK").unwrap_or_default().as_str() {
        "delayed" => Conditioning::Delayed,
        "reversal" => Conditioning::Reversal,
        _ => Conditioning::Instrumental,
    };
    let trials: usize = env("EVO_TRIALS", 80);
    let nseeds: u64 = env("EVO_SEEDS", 3);
    let task = Task::Conditioning(kind, trials);
    let seeds: Vec<u64> = (0..nseeds).collect();
    // (`EVO_PHYSICS=fixed`: the same codes every lifetime; else new codes each lifetime)
    let physics = if std::env::var("EVO_PHYSICS").map_or(false, |v| v == "fixed") { Physics::Fixed(7) } else { Physics::Varied };
    let (pop, elite, gens, seed) = (env("EVO_POP", 48), env("EVO_ELITE", 8), env("EVO_GENS", 60), env("EVO_SEED", 1u64));
    println!("evolve: {kind:?}, {trials} trials a lifetime, {nseeds} world seeds, {physics:?} physics, pop {pop}, elite {elite}, {gens} generations");
    println!("gen  best  mean  paying-clicked  other-clicked  len kern syn thr out op");
    let best = evolve(task, physics, &seeds, pop, elite, gens, seed, |g, r| {
        let b = &r.best;
        let s = &r.responding;
        if g % 5 == 0 || g + 1 == gens {
            println!("{g:>3}  {:>4}  {:>4}  {:>7}/{:<6}  {:>6}/{:<6}  {:>3} {:>4} {:>3} {:>3} {:>3} {:?}",
                r.best_fitness, r.mean_fitness, s.paying_clicked, s.paying_trials, s.other_clicked, s.other_trials,
                b.code.len(), b.kernels, b.syn, b.threshold, b.out, b.op);
        }
    });
    let test: u64 = env("EVO_TEST", 5);
    let held: Vec<u64> = (1000..1000 + test).collect();
    let (f, r) = fitness(&best, task, physics, &held);
    println!("best on {test} held-out world seeds: discrimination {f}/1024, paying clicked {}/{}, other clicked {}/{}", r.paying_clicked, r.paying_trials, r.other_clicked, r.other_trials);
    println!("best program: {:?}", best.code);
}
