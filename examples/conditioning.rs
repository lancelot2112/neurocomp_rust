//! Conditioning probes: can the mind learn what an external cue is worth, and act on a cue to
//! earn a smile, with every reward its own? A fresh mind, a caregiver there throughout, trials of
//! 10 steps; the world gives no reward, an observer scores.
//!
//! COND=pavlov (default): a tone (CS+) is followed by a smile two steps later; another tone (CS−)
//!   by nothing. Read: the critic's value and the dopamine error at the tone, CS+ against CS−, and
//!   the dopamine error at the smile (it should shrink as the tone comes to predict it).
//! COND=instr: a light for three steps on half the trials; a click while it is on gets a smile.
//!   Read: clicks per trial in the window, light against no light.
//! COND=delayed: a light for one step; a click three to five steps later gets a smile.
//!
//! COND_TRIALS (600); SEED (0); the mind's switches (MIND_HC, MIND_TOPO, MIND_AMY, MIND_PAUSE,
//! MIND_CONTINGENCY, MIND_IMAGINE_VALUE, MIND_OWN, MIND_RESPONSE: 0 turns one off).

use neurocomp::program::mind::{Config, Mind};
use neurocomp::program::world::{Acts, Conditioning, World, CLICK};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let seed: u64 = env("SEED", 0);
    let trials: usize = env("COND_TRIALS", 600);
    let kind = match std::env::var("COND").unwrap_or_default().as_str() {
        "instr" => Conditioning::Instrumental,
        "delayed" => Conditioning::Delayed,
        _ => Conditioning::Pavlov,
    };
    let cfg = Config::from_env();
    println!("{kind:?}, {trials} trials, seed {seed}; {cfg:?}");
    let mut world = World::new(seed);
    let mut mind = Mind::with_config(world.body(), seed, cfg);
    world.conditioning(kind, trials);
    let mut acts = Acts::default();
    let block = 100;
    // per block, per cue (false, true): (value at the cue step, dopamine at the cue step, cue
    // steps, dopamine at the smile step, smile steps, clicks in the window, trials, smiles)
    let mut stats = [[(0i64, 0i64, 0usize, 0i64, 0usize, 0usize, 0usize, 0usize); 2]; 64];
    // the amygdala's prediction at the smile's step (step 4), per block and cue
    let mut amy = [[0i64; 2]; 64];
    let mut trial = 0usize;
    let mut pauses = 0usize;
    let mut sizes = (0usize, 0usize, 0usize, 0usize);
    while !world.done() {
        let senses = world.step(&acts);
        acts = mind.step(&senses);
        let ev = world.events.last().unwrap().clone();
        let Some((t, cue)) = ev.cond else { continue };
        if t == 0 && world.events.len() > 1 {
            trial += 1;
        }
        let b = (trial / block).min(63);
        let x = &mut stats[b][cue as usize];
        let r = &mind.report;
        pauses += r.paused as usize;
        let (on, att) = mind.amy.trace_sizes();
        sizes.0 += 1;
        sizes.1 += on;
        sizes.2 += att;
        sizes.3 += (r.spoke || r.moved) as usize;
        // the cue is sensed at step 2
        if t == 2 {
            x.0 += r.value as i64;
            x.1 += r.delta as i64;
            x.2 += 1;
        }
        if t == 4 {
            amy[b][cue as usize] += r.amy_predicted as i64;
        }
        if ev.smiled {
            x.3 += r.delta as i64;
            x.4 += 1;
            x.7 += 1;
        }
        let window = match kind {
            Conditioning::Pavlov => 3..=5,
            Conditioning::Instrumental => 3..=5,
            Conditioning::Delayed => 5..=7,
        };
        if window.contains(&t) && ev.moved == Some(CLICK) {
            x.5 += 1;
        }
        if t == 9 {
            x.6 += 1;
        }
    }
    let f = |v: i64, n: usize| v as f64 / n.max(1) as f64 / 65536.0;
    for (b, s) in stats.iter().enumerate() {
        if s[0].6 + s[1].6 == 0 {
            continue;
        }
        let (no, yes) = (s[0], s[1]);
        match kind {
            Conditioning::Pavlov => println!(
                "  trials {:4}-{:4}: at the tone, value CS+ {:+.3} CS- {:+.3}, dopamine CS+ {:+.3} CS- {:+.3}; dopamine at the smile {:+.3}; amygdala's prediction there CS+ {:+.3} CS- {:+.3}",
                b * block,
                (b + 1) * block,
                f(yes.0, yes.2),
                f(no.0, no.2),
                f(yes.1, yes.2),
                f(no.1, no.2),
                f(yes.3, yes.4),
                f(amy[b][1], yes.6),
                f(amy[b][0], no.6)
            ),
            _ => println!(
                "  trials {:4}-{:4}: clicks in the window per trial, cue {:.2} no cue {:.2}; smiles per cue trial {:.2}; value at the cue {:+.3} (no cue {:+.3})",
                b * block,
                (b + 1) * block,
                yes.5 as f64 / yes.6.max(1) as f64,
                no.5 as f64 / no.6.max(1) as f64,
                yes.7 as f64 / yes.6.max(1) as f64,
                f(yes.0, yes.2),
                f(no.0, no.2)
            ),
        }
    }
    println!("amygdala cells (appetitive, aversive, pause): {:?}; orbitofrontal {:?}", mind.amy.cells(), mind.ofc.cells());
    println!("cholinergic pauses: {pauses} steps; amygdala updates {}; per step: onsets {:.1}, attended {:.1}, acts {:.2}", mind.amy.stats.0, sizes.1 as f64 / sizes.0 as f64, sizes.2 as f64 / sizes.0 as f64, sizes.3 as f64 / sizes.0 as f64);
}
