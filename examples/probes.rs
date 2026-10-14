//! Evolving on graded probes instead of one task score.
//!
//! The tag genome from a naked start (an empty string and `PRB_CHANNELS` empty channels) evolves
//! by epsilon-lexicase selection on five probes (`observe::probes`), each a small capacity on a
//! smooth scale in parts per 1024:
//! - respond: clicks while a light is up minus clicks with none (a blind babbler scores 0);
//! - tell: clicks differently to the two lights, either way round;
//! - discriminate: more clicks on the paying light (the task);
//! - reverse: the worst phase of a serial reversal (it follows every swap);
//! - remember: more clicks after a light that is gone than after none.
//! Each (world seed, probe) pair is a lexicase case, so a genome best at any one capacity in a
//! few worlds can be a parent. The probes are read by the observer from the event log; the
//! network never sees them.
//!
//! Every `PRB_EVERY` generations: the best genome's probe means on its training worlds and on
//! held-out worlds, the population's means, and the genome size; `PRB_SHOW=1` prints the network.
//!
//! `PRB_POP` (32), `PRB_ELITE` (8), `PRB_GENS` (100), `PRB_LIVES` (16 training worlds),
//! `PRB_HELD` (16), `PRB_TRIALS` (200 a lifetime), `PRB_CHANNELS` (4), `PRB_EVERY` (10),
//! `PRB_SEED` (7), `PRB_SHOW` (0).

use neurocomp::evolve::tags::{TagGenome, Token};
use neurocomp::evolve::{evolve_lexicase, probe_all, probe_means, Physics};
use neurocomp::world::observe::PROBES;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn row(m: &[i32; 5]) -> String {
    m.iter().map(|x| format!("{x:>7}")).collect::<Vec<_>>().join("")
}

fn main() {
    let pop: usize = env("PRB_POP", 32);
    let elite: usize = env("PRB_ELITE", 8);
    let gens: usize = env("PRB_GENS", 100);
    let lives: u64 = env("PRB_LIVES", 16);
    let held_n: u64 = env("PRB_HELD", 16);
    let trials: usize = env("PRB_TRIALS", 200);
    let channels: usize = env("PRB_CHANNELS", 4);
    let every: usize = env("PRB_EVERY", 10);
    let seed: u64 = env("PRB_SEED", 7);
    let show = env("PRB_SHOW", 0u8) == 1;
    let train: Vec<u64> = (0..lives).collect();
    let held: Vec<u64> = (10_000..10_000 + held_n).collect();
    let tokens = (0..channels).map(|i| Token::Channel { tag: (0x11u8).wrapping_mul(i as u8 + 1) ^ 0x5A, width: 3, hold: false }).collect();
    let start = TagGenome { tokens, next_id: 1 };
    println!("probes: tag genome from a naked start ({channels} empty channels); pop {pop}, elite {elite}, {lives} worlds, {held_n} held out, {trials} trials a lifetime");
    let names: String = PROBES.iter().map(|p| format!("{:>7}", &p[..p.len().min(6)])).collect();
    println!("{:>4}  {:>5}  {:<35}  {:<35}  {:<35}", "gen", "size", "best, training", "best, held out", "population mean");
    println!("{:>4}  {:>5}  {names}  {names}  {names}", "", "");
    let last = evolve_lexicase(vec![start; pop], trials, Physics::Varied, &train, elite, gens, seed, |g, r| {
        if g % every == 0 || g + 1 == gens {
            let h = probe_all(vec![r.best.clone()], trials, Physics::Varied, &held);
            let hm = probe_means(&h[0].0);
            println!("{g:>4}  {:>5}  {}  {}  {}", r.best.tokens.len(), row(&r.best_means), row(&hm), row(&r.pop_means));
            if show {
                print!("{}", r.best.describe());
            }
        }
    });
    let h = probe_all(vec![last[0].clone()], trials, Physics::Varied, &held);
    println!("final best, held out: {}", row(&probe_means(&h[0].0)));
    print!("{}", last[0].describe());
}
