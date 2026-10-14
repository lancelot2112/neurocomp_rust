//! Why do sense readers not survive selection? The probes and work per step of a few one- or
//! two-population genomes, over `PC_N` worlds (default 64), gaps of 0-6 steps between trials.

use neurocomp::evolve::tags::{Pop, TagGenome, Token, ACT, EYE, HAND, SELF, TOUCH};
use neurocomp::evolve::{probe_all, probe_means, Physics};
use neurocomp::kernel::KernelOp;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let n: u64 = env("PC_N", 64);
    let c = 0x3C;
    let ch = Token::Channel { tag: c, width: 3, hold: false };
    let to_hand = |p: Pop| Pop { to: HAND, to_port: true, ..p };
    let genomes: Vec<(&str, TagGenome)> = vec![
        ("tonic clicker on an empty channel", TagGenome { tokens: vec![ch.clone(), Token::Pop(to_hand(Pop { count: 4, tonic: 4, ..Pop::new(1, c, HAND) }))], next_id: 2 }),
        ("tonic clicker on the self port", TagGenome { tokens: vec![Token::Pop(to_hand(Pop { count: 4, tonic: 4, ..Pop::new(1, SELF, HAND) }))], next_id: 2 }),
        ("eye -> hand, cover, 4 kernels", TagGenome { tokens: vec![Token::Pop(to_hand(Pop { count: 4, cover: true, from: EYE, from_port: true, ..Pop::new(1, EYE, HAND) }))], next_id: 2 }),
        ("eye -> hand, cover, growing", TagGenome { tokens: vec![Token::Pop(to_hand(Pop { count: 4, cover: true, grow: true, new_code: true, from: EYE, from_port: true, ..Pop::new(1, EYE, HAND) }))], next_id: 2 }),
        ("eye -> hand, cover, growing, tonic", TagGenome { tokens: vec![Token::Pop(to_hand(Pop { count: 4, tonic: 4, cover: true, grow: true, new_code: true, from: EYE, from_port: true, ..Pop::new(1, EYE, HAND) }))], next_id: 2 }),
        ("eye -> hand growing + a clicker on self", TagGenome { tokens: vec![
            Token::Pop(to_hand(Pop { count: 4, cover: true, grow: true, from: EYE, from_port: true, ..Pop::new(1, EYE, HAND) })),
            Token::Pop(to_hand(Pop { count: 4, tonic: 3, ..Pop::new(2, SELF, HAND) })),
        ], next_id: 3 }),
        ("touch -> hand, cover, 4 kernels", TagGenome { tokens: vec![Token::Pop(to_hand(Pop { count: 4, cover: true, from: TOUCH, from_port: true, op: KernelOp::Or, ..Pop::new(1, TOUCH, HAND) }))], next_id: 2 }),
    ];
    let _ = ACT;
    let seeds: Vec<u64> = (20_000..20_000 + n).collect();
    println!("{:<38} {:>7} {:>7} {:>7} {:>7} {:>7} {:>6}", "genome", "respond", "tell", "discr", "reverse", "rememb", "work");
    for (name, g) in genomes {
        let r = probe_all(vec![g], 200, Physics::Varied, &seeds, 6);
        let m = probe_means(&r[0].probes);
        println!("{name:<38} {:>7} {:>7} {:>7} {:>7} {:>7} {:>6}", m[0], m[1], m[2], m[3], m[4], r[0].work);
    }
}
