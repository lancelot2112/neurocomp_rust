//! Evolve tag-addressed genomes (`src/evolve/tags.rs`) in the world's serial reversal: many
//! lifetimes per genome (new codes each), scored in parallel, the best checked on held-out
//! lifetimes every few generations.
//!
//! `EVO_START=act|bare` (the act path, or a single tonic population on the act), `EVO_TRIALS`,
//! `EVO_LIVES`, `EVO_HELD`, `EVO_POP`, `EVO_ELITE`, `EVO_GENS`, `EVO_EVERY`, `EVO_SEED`.

use neurocomp::evolve::tags::{TagGenome, Token};
use neurocomp::evolve::{evolve_from, fitness, Physics, Task};
use neurocomp::world::Conditioning;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let name = std::env::var("EVO_START").unwrap_or_else(|_| "act".into());
    let trials: usize = env("EVO_TRIALS", 400);
    let lives: u64 = env("EVO_LIVES", 16);
    let held_n: u64 = env("EVO_HELD", 16);
    let (pop, elite, gens, every, seed) = (env("EVO_POP", 32usize), env("EVO_ELITE", 8usize), env("EVO_GENS", 40usize), env("EVO_EVERY", 10usize), env("EVO_SEED", 7u64));
    let task = Task::Conditioning(Conditioning::Reversal, trials);
    let train: Vec<u64> = (0..lives).collect();
    let held: Vec<u64> = (10_000..10_000 + held_n).collect();
    let g0 = if name == "bare" { TagGenome::bare_seed() } else { TagGenome::act_seed() };
    let (h0, _) = fitness(&g0, task, Physics::Varied, &held);
    println!("evolve tags from '{name}': reversal, {trials} trials a lifetime, {lives} lifetimes a genome, {held_n} held out, pop {pop}, elite {elite}, {gens} generations");
    println!("start: {} tokens, held out {h0}/1024", g0.tokens.len());
    println!("gen  best  mean  tokens  held-out");
    let best = evolve_from(vec![g0.clone(); pop], task, Physics::Varied, &train, elite, gens, seed, |g, r| {
        let h = if g % every == 0 || g + 1 == gens { format!("{}", fitness(&r.best, task, Physics::Varied, &held).0) } else { String::new() };
        println!("{g:>3}  {:>4}  {:>4}  {:>6}  {h:>8}", r.best_fitness, r.mean_fitness, r.best.tokens.len());
    });
    let (h, r) = fitness(&best, task, Physics::Varied, &held);
    println!("best, held out: {h}/1024 (start {h0}); paying clicked {}/{}, other clicked {}/{}", r.paying_clicked, r.paying_trials, r.other_clicked, r.other_trials);
    let chans = best.tokens.iter().filter(|t| matches!(t, Token::Channel { .. })).count();
    println!("best: {} tokens ({chans} channels, {} populations)", best.tokens.len(), best.tokens.len() - chans);
    for t in &best.tokens {
        match t {
            Token::Channel { tag, width, .. } => println!("  channel {tag:08b} width 2^{width}"),
            Token::Motion { tag, conveyor, wave } => println!("  motion {tag:08b} conveyor {conveyor} wave {wave}"),
            Token::Pop(p) => println!("  pop {:08b}->{:08b} {} {:?} lag {} field 2^{}@{} count {} fan {}/{} q{} tonic {} grow {} new {} repeat {}/{} gate {:?}",
                p.from, p.to, if p.wire { "wire" } else { "kernels" }, p.op, p.lag, p.field, p.at, p.count, p.fan_in, p.fan_out, p.quarters, p.tonic, p.grow, p.new_code, p.repeat, p.repeat_lag, p.gate),
        }
    }
}
