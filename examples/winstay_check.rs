//! What the win-stay rung can give: hand-written policies that see only what a network senses
//! (the eye and the affect channel), scored in the world of switching's and shapes' win-stay
//! rung (serial reversal with no swaps, 200 trials, gaps 0-6), on many fresh worlds.
//! - learner: clicks a light unless its last click went unstroked (win-stay, lose-shift), so
//!   it learns within the lifetime which light pays;
//! - one light: always clicks the first light it saw (ignores strokes);
//! - random: clicks either light half the time.
//! Prints each policy's win-stay and discriminate: the mean and the spread of a 12-world mean.
//! `WSC_WORLDS` (48), `WSC_TRIALS` (200), `WSC_GAP` (6).

use neurocomp::bitvec::BitVector;
use neurocomp::world::observe::probes;
use neurocomp::world::{channel, Conditioning, World, AFFECT, CLICK, EYE};
use rand::{Rng, SeedableRng};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

#[derive(Clone, Copy)]
enum Policy {
    Learner,
    OneLight,
    Random,
}

/// One lifetime: (win-stay, discriminate).
fn lifetime(seed: u64, policy: Policy, trials: usize, gap: usize) -> (i32, i32) {
    let p = seed.wrapping_mul(0xA24B_AED4_963E_E407);
    let mut w = World::with_physics(p, seed, 8);
    w.gap_max = gap;
    w.reversal_period = trials;
    w.conditioning(Conditioning::Reversal, trials);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0x5157);
    let mut senses = BitVector::new(neurocomp::world::SENSE_BITS, Some(0));
    // what the policy remembers: each light seen, and how its last click went
    let mut lights: Vec<(Vec<u64>, Option<bool>)> = Vec::new();
    let mut first: Option<Vec<u64>> = None;
    let mut clicked: Option<usize> = None;
    let mut shown_before = false;
    let mut coin = false;
    while !w.done() {
        let eye = channel(&senses, EYE).as_words().to_vec();
        let shown = eye.iter().any(|&x| x != 0);
        let stroked = channel(&senses, AFFECT).count_ones() > 0;
        if stroked {
            if let Some(k) = clicked.take() {
                lights[k].1 = Some(true);
            }
        }
        // a new light: the last trial's click, if it went unstroked, counts as a loss
        if shown && !shown_before {
            if let Some(k) = clicked.take() {
                lights[k].1 = Some(false);
            }
            coin = rng.gen_bool(0.5);
        }
        // (click on a light's second step: its first comes before the response window)
        let second = shown && shown_before;
        shown_before = shown;
        let mut hand = None;
        if second && clicked.is_none() {
            let k = lights.iter().position(|l| l.0 == eye).unwrap_or_else(|| {
                lights.push((eye.clone(), None));
                lights.len() - 1
            });
            first.get_or_insert(eye.clone());
            let click = match policy {
                Policy::Learner => lights[k].1 != Some(false),
                Policy::OneLight => first.as_ref() == Some(&eye),
                Policy::Random => coin,
            };
            if click {
                hand = Some(CLICK);
                clicked = Some(k);
            }
        }
        let a = w.act(None, hand);
        senses = w.step(&a);
    }
    let row = probes(&w.events, trials, &[], &[], &[], &[], &[]);
    (row[5], row[2])
}

fn main() {
    let n: u64 = env("WSC_WORLDS", 48);
    let trials: usize = env("WSC_TRIALS", 200);
    let gap: usize = env("WSC_GAP", 6);
    println!("{trials} trials, gaps 0-{gap}, {n} worlds: win-stay and discriminate (mean; spread of a 12-world mean)");
    for (name, policy) in [("learner", Policy::Learner), ("one light", Policy::OneLight), ("random", Policy::Random)] {
        let rows: Vec<(i32, i32)> = (0..n).map(|s| lifetime(50_000 + s, policy, trials, gap)).collect();
        let stat = |f: &dyn Fn(&(i32, i32)) -> i32| {
            let xs: Vec<f64> = rows.iter().map(|r| f(r) as f64).collect();
            let mean = xs.iter().sum::<f64>() / xs.len() as f64;
            let sd = (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (xs.len() as f64 - 1.0).max(1.0)).sqrt();
            format!("{mean:6.0} ±{:.0}", sd / 12f64.sqrt())
        };
        println!("  {name:<10} win-stay {}   discriminate {}", stat(&|r| r.0), stat(&|r| r.1));
    }
}
