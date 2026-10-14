//! Evolve edge genomes in the world's serial reversal, with selection that can mean something:
//! many lifetimes per genome (each with new codes), the population scored in parallel, and the
//! best checked every few generations on held-out lifetimes it was never selected on.
//!
//! `EVO_START=context|act|noise` (the hand-written context circuit, its act path alone, or a single
//! noise edge with babblers and nothing else), `EVO_TRIALS` (a lifetime; the paying light swaps
//! every quarter), `EVO_LIVES` (lifetimes a genome is scored on), `EVO_HELD` (held-out lifetimes),
//! `EVO_POP`, `EVO_ELITE`, `EVO_GENS`, `EVO_EVERY` (held-out check interval), `EVO_SEED`.

use neurocomp::evolve::genes::{EdgeGene, GeneGenome, INPUT, OUTPUT};
use neurocomp::evolve::{evolve_from, fitness, Physics, Task};
use neurocomp::kernel::KernelOp;
use neurocomp::program::graph::EdgeKind;
use neurocomp::world::Conditioning;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn start(name: &str) -> GeneGenome {
    match name {
        "act" => GeneGenome::act_seed(8),
        "noise" => {
            // spontaneous activity into babbling kernels on the act; everything else must evolve
            let mut n = EdgeGene::new(100, 2, 2, EdgeKind::Noise, KernelOp::Or);
            n.spec.noise = 8;
            n.spec.out_words = 1;
            let mut b = EdgeGene::new(101, 2, OUTPUT, EdgeKind::Kernels, KernelOp::Or);
            b.spec.start_kernels = 8;
            let _ = INPUT;
            GeneGenome { edges: vec![n, b], next_id: 102 }
        }
        _ => GeneGenome::context_seed(8),
    }
}

fn main() {
    let name = std::env::var("EVO_START").unwrap_or_else(|_| "context".into());
    let trials: usize = env("EVO_TRIALS", 400);
    let lives: u64 = env("EVO_LIVES", 16);
    let held_n: u64 = env("EVO_HELD", 16);
    let (pop, elite, gens, every, seed) = (env("EVO_POP", 32usize), env("EVO_ELITE", 8usize), env("EVO_GENS", 40usize), env("EVO_EVERY", 10usize), env("EVO_SEED", 7u64));
    let task = Task::Conditioning(Conditioning::Reversal, trials);
    let train: Vec<u64> = (0..lives).collect();
    let held: Vec<u64> = (10_000..10_000 + held_n).collect();
    let g0 = start(&name);
    let (h0, _) = fitness(&g0, task, Physics::Varied, &held);
    println!("evolve edges from '{name}': reversal, {trials} trials a lifetime, {lives} lifetimes a genome, {held_n} held out, pop {pop}, elite {elite}, {gens} generations");
    println!("start: {} edges, held out {h0}/1024", g0.edges.len());
    println!("gen  best  mean  edges  held-out");
    let best = evolve_from(vec![g0.clone(); pop], task, Physics::Varied, &train, elite, gens, seed, |g, r| {
        let h = if g % every == 0 || g + 1 == gens { format!("{}", fitness(&r.best, task, Physics::Varied, &held).0) } else { String::new() };
        println!("{g:>3}  {:>4}  {:>4}  {:>5}  {h:>8}", r.best_fitness, r.mean_fitness, r.best.edges.len());
    });
    let (h, r) = fitness(&best, task, Physics::Varied, &held);
    println!("best, held out: {h}/1024 (start {h0}); paying clicked {}/{}, other clicked {}/{}", r.paying_clicked, r.paying_trials, r.other_clicked, r.other_trials);
    let kinds = |k: EdgeKind| best.edges.iter().filter(|e| e.spec.kind == k).count();
    println!("best: {} edges ({} wires, {} kernel edges, {} noise), {} growing, {} gated, {} repeat", best.edges.len(), kinds(EdgeKind::Wire), kinds(EdgeKind::Kernels), kinds(EdgeKind::Noise),
        best.edges.iter().filter(|e| e.spec.grow).count(), best.edges.iter().filter(|e| e.gate.is_some()).count(), best.edges.iter().filter(|e| e.spec.repeat).count());
}
