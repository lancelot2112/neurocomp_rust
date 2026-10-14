//! A baseline for the conditioning scores: what does a network that cannot see score?
//!
//! A single tonic population on the act reading the self channel (it fires at random, reading
//! nothing that tells the lights apart) and a covering population on the senses with no learning that matters. Scored
//! on the curriculum's training seeds (0..24) and held-out seeds (10000..10016), and on a large
//! set (20000..20000+`BASE_N`), per task. A network that cannot see should score about 0 on all.
//!
//! Each cell: score / pooled discrimination (silence aside).
//!
//! `BASE_N` (default 256), `BASE_TONIC` (default 3: fires 1 step in 8).

use neurocomp::evolve::tags::{Pop, TagGenome, Token, ACT, SELF, SENSES};
use neurocomp::evolve::{fitness, Physics, Task};
use neurocomp::world::Conditioning;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let n: u64 = env("BASE_N", 256);
    let tonic: u8 = env("BASE_TONIC", 3);
    let blind = TagGenome { tokens: vec![Token::Pop(Pop { count: 8, tonic, ..Pop::new(1, SELF, ACT) })], next_id: 2 };
    let seeing = TagGenome { tokens: vec![Token::Pop(Pop { count: 8, tonic, cover: true, tile: 6, ..Pop::new(1, SENSES, ACT) })], next_id: 2 };
    let sets: [(&str, Vec<u64>); 3] = [("train 0..24", (0..24).collect()), ("held 10000..16", (10_000..10_016).collect()), ("large", (20_000..20_000 + n).collect())];
    let tasks = [
        ("instrumental", Task::Conditioning(Conditioning::Instrumental, 400)),
        ("delayed", Task::Conditioning(Conditioning::Delayed, 400)),
        ("reversal", Task::Reversal { trials: 400, period: 100, worst_phase: false }),
        ("reversal worst", Task::Reversal { trials: 400, period: 100, worst_phase: true }),
    ];
    println!("{:<8} {:<15} {:>14} {:>14} {:>14}", "genome", "task", sets[0].0, sets[1].0, sets[2].0);
    for (gname, g) in [("blind", &blind), ("seeing", &seeing)] {
        for (tname, t) in tasks {
            // the score (silent lifetimes count -4096), and the discrimination pooled over every
            // trial of the set (silence aside)
            let s: Vec<String> = sets.iter().map(|(_, seeds)| { let (f, r) = fitness(g, t, Physics::Varied, seeds); format!("{f}/{}", r.discrimination()) }).collect();
            println!("{gname:<8} {tname:<15} {:>14} {:>14} {:>14}", s[0], s[1], s[2]);
        }
    }
}
