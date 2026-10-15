//! What the echo2 rung can give: hand-written policies that hear only the ear and speak only
//! with the voice, in reading's echo2 world (2 tokens, the teacher saying every one), on many
//! fresh worlds.
//! - echoer: speaks on odd steps only, learning each voice code's sound as the lines its ear
//!   always holds when it says that code (its own voice is heard, mixed with the teacher's);
//!   on even steps it listens, and on the next step says the code whose sound it heard;
//! - stubborn: always says the first code;
//! - random: says a random code every step.
//! Prints echo (and babble): the mean, and the spread of a 12-world mean. `ECC_WORLDS` (48),
//! `ECC_STEPS` (200), `ECC_TOKENS` (2).

use neurocomp::bitvec::BitVector;
use neurocomp::world::observe::probes;
use neurocomp::world::{channel, World, EAR};
use rand::{Rng, SeedableRng};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

#[derive(Clone, Copy, PartialEq)]
enum Policy {
    Echoer,
    Stubborn,
    Random,
}

/// One lifetime: (echo, babble).
fn lifetime(seed: u64, policy: Policy, steps: usize, tokens: usize) -> (i32, i32) {
    let p = seed.wrapping_mul(0xA24B_AED4_963E_E407);
    let mut w = World::with_physics(p, seed, tokens);
    w.teacher = true;
    w.teacher_rate = 8;
    let mut r = rand::rngs::StdRng::seed_from_u64(seed ^ 0x5E9);
    let toks: Vec<usize> = (0..steps).map(|_| r.gen_range(0..tokens)).collect();
    w.sequence_marked(&toks, &vec![false; steps]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0xEC0);
    let mut senses = BitVector::new(neurocomp::world::SENSE_BITS, Some(0));
    // each code's sound: the ear's lines every time it was said (None: not said yet)
    let mut sound: Vec<Option<Vec<u64>>> = vec![None; tokens];
    let (mut said_last, mut t) = (None::<usize>, 0usize);
    while !w.done() {
        let ear = channel(&senses, EAR).as_words().to_vec();
        if let Some(k) = said_last {
            sound[k] = Some(match &sound[k] { None => ear.clone(), Some(s) => s.iter().zip(&ear).map(|(a, b)| a & b).collect() });
        }
        let say = match policy {
            Policy::Stubborn => Some(0),
            Policy::Random => Some(rng.gen_range(0..tokens)),
            Policy::Echoer => {
                if t % 2 == 1 {
                    // the code whose learned sound the ear held a step ago (heard when silent),
                    // else a code not learned yet
                    let heard = sound.iter().position(|s| s.as_ref().is_some_and(|s| s.iter().any(|&x| x != 0) && s.iter().zip(&ear).all(|(a, b)| a & !b == 0)));
                    heard.or_else(|| sound.iter().position(|s| s.is_none())).or(Some(rng.gen_range(0..tokens)))
                } else {
                    None
                }
            }
        };
        let a = w.act(say, None);
        senses = w.step(&a);
        said_last = say;
        t += 1;
    }
    let row = probes(&[], 1, &[], &w.events, &[], &[], &[]);
    (row[7], row[8])
}

fn main() {
    let n: u64 = env("ECC_WORLDS", 48);
    let steps: usize = env("ECC_STEPS", 200);
    let tokens: usize = env("ECC_TOKENS", 2);
    println!("{tokens} tokens, {steps} steps, the teacher on every token, {n} worlds: echo and babble (mean; spread of a 12-world mean)");
    for (name, policy) in [("echoer", Policy::Echoer), ("stubborn", Policy::Stubborn), ("random", Policy::Random)] {
        let rows: Vec<(i32, i32)> = (0..n).map(|s| lifetime(50_000 + s, policy, steps, tokens)).collect();
        let stat = |f: &dyn Fn(&(i32, i32)) -> i32| {
            let xs: Vec<f64> = rows.iter().map(|r| f(r) as f64).collect();
            let mean = xs.iter().sum::<f64>() / xs.len() as f64;
            let sd = (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (xs.len() as f64 - 1.0).max(1.0)).sqrt();
            format!("{mean:6.0} ±{:.0}", sd / 12f64.sqrt())
        };
        println!("  {name:<9} echo {}   babble {}", stat(&|r| r.0), stat(&|r| r.1));
    }
}
