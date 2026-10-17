//! The feel probe's ceiling for strategies that do not know which hand code is the click: the
//! feel world (a stroke one step in eight, whatever is done) driven directly by a policy, over 48
//! worlds with codes drawn afresh in each, as the islands' are (`Physics::Varied`).
//! `cargo run --release --example feelcheck`
use neurocomp::world::{channel, observe, World, AFFECT, CLICK};

/// the hand's moves (`world::MOVES`)
const MOVES: usize = 5;
use rand::{Rng, SeedableRng};

fn main() {
    let names = ["nothing", "a random move after a stroke", "one move kept for the lifetime, after a stroke", "the click after a stroke (knows the click)"];
    for (k, name) in names.iter().enumerate() {
        let mut sum = 0i64;
        let mut worst = i32::MAX;
        let mut best = i32::MIN;
        for seed in 0..48u64 {
            let mut w = World::with_physics(seed.wrapping_mul(0xA24B_AED4_963E_E407), seed, 8);
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0xF00D);
            let kept = rng.gen_range(0..MOVES);
            w.feel(1000);
            let mut since = usize::MAX;
            let mut act = w.act(None, None);
            for _ in 0..1000 {
                let senses = w.step(&act);
                since = if channel(&senses, AFFECT).count_ones() > 0 { 0 } else { since.saturating_add(1) };
                let mv = if since < 3 {
                    match k { 1 => Some(rng.gen_range(0..MOVES)), 2 => Some(kept), 3 => Some(CLICK), _ => None }
                } else {
                    None
                };
                act = w.act(None, mv);
            }
            let s = observe::feel(&w.events);
            sum += s as i64;
            worst = worst.min(s);
            best = best.max(s);
        }
        println!("{name}: feel mean {} (worst {worst}, best {best}) of the goal 300", sum / 48);
    }
}
