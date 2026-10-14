//! Evolution starting from the hand-written context circuit (gene list): a population of copies
//! of `GeneGenome::context_seed`, scored on serial reversal (discrimination over a lifetime, new
//! codes each lifetime), then the best checked on held-out lifetimes against the seed itself.
//!
//! `SEED_START=context|babble` (the full circuit, or only its act path), `SEED_TRIALS` (a
//! lifetime; the paying light swaps every quarter), `SEED_SEEDS`, `SEED_POP`, `SEED_ELITE`,
//! `SEED_GENS`, `SEED_TEST`.

use neurocomp::evolve::genes::GeneGenome;
use neurocomp::evolve::{evolve_from, fitness, Physics, Task};
use neurocomp::world::Conditioning;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let start_name = std::env::var("SEED_START").unwrap_or_else(|_| "context".into());
    let mut start = GeneGenome::context_seed(8);
    if start_name == "babble" {
        start.edges.retain(|e| e.src == 0 && (e.dst == 7 || e.dst == 8) || e.src == 8 || e.src == 10);
    }
    let trials: usize = env("SEED_TRIALS", 400);
    let nseeds: u64 = env("SEED_SEEDS", 3);
    let (pop, elite, gens) = (env("SEED_POP", 24usize), env("SEED_ELITE", 6usize), env("SEED_GENS", 20usize));
    let task = Task::Conditioning(Conditioning::Reversal, trials);
    let seeds: Vec<u64> = (0..nseeds).collect();
    let held: Vec<u64> = (1000..1000 + env("SEED_TEST", 5u64)).collect();
    let (f0, _) = fitness(&start, task, Physics::Varied, &held);
    println!("evolve from the {start_name} seed: reversal, {trials} trials a lifetime, {nseeds} lifetimes a genome, pop {pop}, elite {elite}, {gens} generations");
    println!("the seed on held-out lifetimes: {f0}/1024");
    println!("gen  best  mean  nodes edges");
    let best = evolve_from(vec![start.clone(); pop], task, Physics::Varied, &seeds, elite, gens, 7, |g, r| {
        println!("{g:>3}  {:>4}  {:>4}  {:>5} {:>5}", r.best_fitness, r.mean_fitness, r.best.nodes.len(), r.best.edges.len());
    });
    let (f, r) = fitness(&best, task, Physics::Varied, &held);
    println!("best on held-out lifetimes: {f}/1024 (seed {f0}); paying clicked {}/{}, other clicked {}/{}", r.paying_clicked, r.paying_trials, r.other_clicked, r.other_trials);
    let added: Vec<_> = best.edges.iter().filter(|e| !start.edges.iter().any(|s| s == *e)).collect();
    let removed: Vec<_> = start.edges.iter().filter(|s| !best.edges.iter().any(|e| e == *s)).collect();
    println!("edges changed or added: {}, removed or changed from the seed: {}", added.len(), removed.len());
    for e in added.iter().take(8) {
        println!("  + {:?} {}->{} lag {} {:?} grow {} gate {:?} new {} wide {} repeat {}/{} kernels {}", e.kind, e.src, e.dst, e.lag, e.op, e.grow, e.gate, e.new_code, e.wide, e.repeat, e.repeat_lag, e.kernels);
    }
}
