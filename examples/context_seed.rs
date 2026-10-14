//! The hand-written context circuit (gene list, `GeneGenome::context_seed`) living in the world's
//! serial reversal: one of two lights is on for three steps each trial; a click while it is on (or
//! the step after) is followed by a stroke when that light is the one paying now; the paying light
//! swaps every `CTX_PERIOD` trials. New codes each lifetime (varied physics).
//!
//! Per phase: the observer's discrimination (parts per 1024: click rate on paying trials minus on
//! the others), and from inside the network (a probe; nothing goes back in): how often the context
//! vector C changed to a different code, and how many distinct codes it held.
//!
//! `CTX_VARIANT=seed|nocontext|babble` (`nocontext`: C does not reach the act node; `babble`: only
//! the babbling and stroke-repeat path), `CTX_PERIOD`, `CTX_PHASES`, `CTX_SEEDS`, `CTX_BABBLERS`.

use std::collections::HashSet;

use neurocomp::bitvec::bithistory::AdvanceMode;
use neurocomp::bitvec::BitVector;
use neurocomp::evolve::genes::GeneGenome;
use neurocomp::evolve::Evolvable;
use neurocomp::world::{self, observe, Conditioning, World};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

pub fn variant(name: &str, babblers: u8) -> GeneGenome {
    let mut g = GeneGenome::context_seed(babblers);
    match name {
        // C does not reach the act node
        "nocontext" => g.edges.retain(|e| !(e.src == 4 && e.dst == 8)),
        // only the act path: the eye channel to babblers and stroke-repeat births
        "babble" => g.edges.retain(|e| e.src == 0 && (e.dst == 7 || e.dst == 8) || e.src == 8 || e.src == 10),
        _ => {}
    }
    g
}

fn main() {
    let name = std::env::var("CTX_VARIANT").unwrap_or_else(|_| "seed".into());
    let period: usize = env("CTX_PERIOD", 100);
    let phases: usize = env("CTX_PHASES", 6);
    let seeds: u64 = env("CTX_SEEDS", 5);
    let babblers: u8 = env("CTX_BABBLERS", 8);
    let g = variant(&name, babblers);
    println!("context seed ({name}): reversal every {period} trials, {phases} phases, {seeds} lifetimes, {babblers} babblers");
    println!("phase  discrimination  paying-clicked  other-clicked  C-changes  C-codes");
    let mut rows = vec![(0i64, 0u32, 0u32, 0u32, 0u32, 0u32, 0usize); phases];
    for seed in 0..seeds {
        let mut w = World::with_physics(seed.wrapping_mul(0xA24B_AED4_963E_E407), seed, 8);
        w.reversal_period = period;
        w.conditioning(Conditioning::Reversal, period * phases);
        let mut net = g.build(&w, seed ^ 0x5EED);
        let c_node = 4.min(net.nodes.len() - 1);
        let mut act = BitVector::new(world::ACT_BITS, Some(0));
        let mut last_c: Vec<u64> = Vec::new();
        let mut codes: Vec<HashSet<Vec<u64>>> = vec![HashSet::new(); phases];
        let mut changes = vec![0u32; phases];
        while !w.done() {
            let senses = w.step(&act);
            let ph = w.events.last().and_then(|e| e.trial).map_or(0, |t| (t.0 / period).min(phases - 1));
            net.tick(AdvanceMode::Clear, 0);
            net.write_input(&senses, 0);
            act = net.read_output().clone();
            if name != "babble" {
                let c = net.read_node(c_node).as_words().to_vec();
                if c.iter().any(|&x| x != 0) {
                    if !last_c.is_empty() && c != last_c {
                        changes[ph] += 1;
                    }
                    codes[ph].insert(c.clone());
                    last_c = c;
                }
            }
        }
        for ph in 0..phases {
            let evs: Vec<_> = w.events.iter().filter(|e| e.trial.map_or(false, |t| t.0 / period == ph)).cloned().collect();
            let r = observe::responding(&evs);
            let row = &mut rows[ph];
            row.0 += r.discrimination() as i64;
            row.1 += r.paying_clicked;
            row.2 += r.paying_trials;
            row.3 += r.other_clicked;
            row.4 += r.other_trials;
            row.5 += changes[ph];
            row.6 += codes[ph].len();
        }
    }
    for (ph, r) in rows.iter().enumerate() {
        println!("{ph:>5}  {:>14}  {:>7}/{:<6}  {:>6}/{:<6}  {:>9}  {:>7}", r.0 / seeds as i64, r.1, r.2, r.3, r.4, r.5 / seeds as u32, r.6 / seeds as usize);
    }
}
