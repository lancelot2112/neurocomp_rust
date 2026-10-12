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
    // sleep every this many trials (0: never)
    let sleep_every: usize = env("COND_SLEEP", 50);
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
    // steps, dopamine at the touch step, touch steps, clicks in the window, trials, smiles)
    let mut stats = [[(0i64, 0i64, 0usize, 0i64, 0usize, 0usize, 0usize, 0usize); 2]; 64];
    // the amygdala's prediction at the smile's step (step 4), per block and cue
    let mut amy = [[0i64; 2]; 64];
    let mut trial = 0usize;
    let mut pauses = 0usize;
    let mut sizes = (0usize, 0usize, 0usize, 0usize);
    let mut tonic_sum = 0usize;
    // per block: (dips while paused, bursts while paused, sum of dips, sum of bursts, steps)
    let mut da = [(0u64, 0u64, 0i64, 0i64, 0u64); 64];
    let mut cells_at: Vec<[(usize, usize); 2]> = Vec::new();
    // diagnosis: how much the eye's and the seen-acts area's codes change when the light comes on
    // (overlap of the step-2 code with the step-1 code, on light trials), summed per block
    let mut light_seen = [(0u64, 0u64, 0u64, 0u64); 64];
    let mut prev_eye: Option<neurocomp::bitvec::BitVector> = None;
    while !world.done() {
        let senses = world.step(&acts);
        {
            use neurocomp::program::mind::{overlap, ACTS, EYE};
            let code = mind.brain.encode_only(EYE, &senses.eye);
            if let (Some(p), Some((2, true))) = (prev_eye.as_ref(), world.events.last().and_then(|e| e.cond)) {
                let b = (trial / block).min(63);
                light_seen[b].0 += overlap(&code, p) as u64;
                light_seen[b].1 += mind.brain.encode_only(ACTS, &senses.eye).count_ones() as u64;
                light_seen[b].2 += 1;
                light_seen[b].3 += code.count_ones() as u64;
            }
            prev_eye = Some(code);
        }
        acts = mind.step(&senses);
        let ev = world.events.last().unwrap().clone();
        let Some((t, cue)) = ev.cond else { continue };
        if t == 0 && world.events.len() > 1 {
            trial += 1;
            if sleep_every > 0 && trial % sleep_every == 0 {
                mind.sleep(3);
                if trial % block == 0 {
                    let (n, h) = mind.brain.sense_hits();
                    println!(
                        "  sleep at trial {trial}: cortical cells {}, sensory predictions right {:.1}%; compacted so far {} merged, {} pruned; hippocampal rows {}, consolidated {}",
                        mind.brain.cortical_cells(),
                        100.0 * h as f64 / n.max(1) as f64,
                        mind.compacted.0,
                        mind.compacted.1,
                        mind.hc.mem.rows(),
                        mind.hc.consolidated
                    );
                }
            }
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
        tonic_sum += r.tonic as usize;
        {
            let b = (trial / block).min(63);
            let d = &mut da[b];
            d.4 += 1;
            if r.delta < 0 {
                d.2 += r.delta as i64;
                d.0 += r.paused as u64;
            } else {
                d.3 += r.delta as i64;
                d.1 += r.paused as u64;
            }
            if t == 9 && trial % block == block - 1 {
                cells_at.push(mind.motor_cells(neurocomp::program::mind::HAND));
            }
        }
        // the cue is sensed at step 2
        if t == 2 {
            x.0 += r.value as i64;
            x.1 += r.delta as i64;
            x.2 += 1;
        }
        if t == 4 {
            amy[b][cue as usize] += r.amy_predicted as i64;
        }
        if ev.stroked {
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
                "  trials {:4}-{:4}: at the tone, value CS+ {:+.3} CS- {:+.3}, dopamine CS+ {:+.3} CS- {:+.3}; dopamine at the touch {:+.3}; amygdala's prediction there CS+ {:+.3} CS- {:+.3}",
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
                "  trials {:4}-{:4}: clicks in the window per trial, cue {:.2} no cue {:.2}; touches per cue trial {:.2}; value at the cue {:+.3} (no cue {:+.3})",
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
    for (b, d) in da.iter().enumerate().filter(|x| x.1 .4 > 0) {
        let c = cells_at.get(b).copied().unwrap_or_default();
        println!(
            "  block {b}: dopamine per step: dips {:+.4}, bursts {:+.4}; at a pause: {} dips, {} bursts; the hand's loop go/no-go cells {}/{}, its gate {}/{}",
            d.2 as f64 / d.4 as f64 / 65536.0,
            d.3 as f64 / d.4 as f64 / 65536.0,
            d.0,
            d.1,
            c[0].0,
            c[0].1,
            c[1].0,
            c[1].1
        );
    }
    for (b, x) in light_seen.iter().enumerate().filter(|x| x.1 .2 > 0) {
        println!("  block {b}: the light comes on: the eye's code has {:.1} cells, {:.1} of them from the step before; the seen-acts area codes {:.1} cells", x.3 as f64 / x.2 as f64, x.0 as f64 / x.2 as f64, x.1 as f64 / x.2 as f64);
    }
    println!("amygdala cells (appetitive, aversive, pause, baseline): {:?}; orbitofrontal {:?}; what an act usually brings {:+.3}", mind.amy.cells(), mind.ofc.cells(), mind.report.rate_act as f64 / 65536.0);
    println!("cholinergic pauses: {pauses} steps; amygdala updates {}; per step: onsets {:.1}, attended {:.1}, acts {:.2}, tonic noise {:.2}", mind.amy.stats.0, sizes.1 as f64 / sizes.0 as f64, sizes.2 as f64 / sizes.0 as f64, sizes.3 as f64 / sizes.0 as f64, tonic_sum as f64 / sizes.0 as f64);
}
