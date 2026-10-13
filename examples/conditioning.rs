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
use std::collections::HashMap;
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
        "reversal" => Conditioning::Reversal,
        _ => Conditioning::Pavlov,
    };
    let cfg = Config::from_env();
    println!("{kind:?}, {trials} trials, seed {seed}; {cfg:?}");
    let mut world = World::new(seed);
    let mut mind = Mind::with_config(world.body(), seed, cfg);
    world.reversal_period = env("COND_PERIOD", 500);
    world.conditioning(kind, trials);
    // reversal: per block, (clicks in the window with the paying light, trials with it, clicks
    // with the other light, trials with it)
    let mut rev = [(0usize, 0usize, 0usize, 0usize); 64];
    let mut a_pays = true;
    let given_context = std::env::var("COND_GIVEN_CONTEXT").map_or(false, |v| v == "1");
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
    let mut frus = [(0i64, 0u64); 64];
    // per block: (dips while paused, bursts while paused, sum of dips, sum of bursts, steps)
    let mut da = [(0u64, 0u64, 0i64, 0i64, 0u64); 64];
    let mut cells_at: Vec<[(usize, usize); 2]> = Vec::new();
    // diagnosis: how much the eye's and the seen-acts area's codes change when the light comes on
    // (overlap of the step-2 code with the step-1 code, on light trials), summed per block
    let mut light_seen = [(0u64, 0u64, 0u64, 0u64); 64];
    let mut prev_eye: Option<neurocomp::bitvec::BitVector> = None;
    // diagnosis: per block, per trial step: appetitive cells removed (negative error), recruited
    let mut churn = [[(0u64, 0u64); 10]; 64];
    let mut last_churn = (0u64, 0u64, 0u64);
    // the trace's items at age 2 at the touch's step (CS+), summed, and steps
    let mut age2 = [(0u64, 0u64); 64];
    // the ear's onsets at the CS+ tone: how many, and how many shared with the last CS+ tone's
    let mut tone = [(0u64, 0u64, 0u64); 64];
    let mut last_tone: Vec<u32> = Vec::new();
    // drift: per light step (step 2) of each trial, per panel, its active bits (the trial, the
    // cue, the panels); at each block's end, each panel's overlap with the same cue's code the
    // trial before, 50 and 100 trials before
    let mut seen_at: Vec<(usize, bool, Vec<(String, Vec<usize>)>)> = Vec::new();
    let mut prev_sel: Vec<(usize, u64)> = Vec::new();
    let drift_panels = ["eye", "top_in", "top_driven", "top_l4", "top_l23", "top_burst", "wm", "ofc_z", "ofc_held", "ofc_phase", "amy_value", "ctx"];
    // the phase diagnostic (reversal): per block, which phase it was in (A pays) and, per panel and
    // light, how often each bit was on at the light, and on how many trials
    type Freq = std::collections::HashMap<usize, u32>;
    let mut phase_blocks: Vec<(bool, Vec<([Freq; 2], [u32; 2])>)> = Vec::new();
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
        if t == 2 {
            let mut p: Vec<(String, Vec<usize>)> = mind.snapshot().into_iter().filter(|(n, _)| drift_panels.contains(n)).map(|(n, b)| (n.to_string(), b)).collect();
            p.push(("ctx".into(), mind.hand_view().0.map_or(Vec::new(), |v| v.0)));
            seen_at.push((trial, cue, p));
        }
        if t == 0 {
            a_pays = (trial / world.reversal_period.max(1)) % 2 == 0;
            // the choice side alone (a diagnosis): the context is given, one per phase of the
            // reversal (`COND_GIVEN_CONTEXT=1`, with `MIND_PHASE=1`)
            if given_context {
                mind.give_context(if a_pays { 0 } else { 1 });
            }
        }
        if t == 0 && world.events.len() > 1 {
            trial += 1;
            // progress, as each block ends: clicks per trial in the window (cue / none), or the
            // amygdala's prediction (CS+ / CS-)
            if trial % block == 0 {
                let b = trial / block - 1;
                let (no, yes) = (stats[b][0], stats[b][1]);
                let r = |a: usize, n: usize| a as f64 / n.max(1) as f64;
                match kind {
                    Conditioning::Pavlov => eprintln!("progress {kind:?} seed {seed} trials {trial}: amygdala CS+ {:+.2} CS- {:+.2}", amy[b][1] as f64 / yes.6.max(1) as f64 / 65536.0, amy[b][0] as f64 / no.6.max(1) as f64 / 65536.0),
                    Conditioning::Reversal => eprintln!("progress {kind:?} seed {seed} trials {trial}: paying light {:.2}, the other {:.2}", r(rev[b].0, rev[b].1), r(rev[b].2, rev[b].3)),
                    _ => eprintln!("progress {kind:?} seed {seed} trials {trial}: clicks cue {:.2} none {:.2}", r(yes.5, yes.6), r(no.5, no.6)),
                }
                // drift: mean overlap (shared / larger) with the same cue's code 1, 50, 100 trials back
                {
                    let sim = |a: &[usize], b: &[usize]| -> f64 {
                        let n = a.len().max(b.len());
                        if n == 0 { return 1.0; }
                        a.iter().filter(|x| b.binary_search(x).is_ok()).count() as f64 / n as f64
                    };
                    let mut line = String::new();
                    for name in drift_panels {
                        let mut acc = [(0.0f64, 0usize); 3];
                        for (i, (tr, cue, p)) in seen_at.iter().enumerate().filter(|(_, x)| x.0 + block > trial && x.0 < trial) {
                            let Some(now) = p.iter().find(|x| x.0 == name).map(|x| { let mut v = x.1.clone(); v.sort_unstable(); v }) else { continue };
                            for (k, back) in [1usize, 50, 100].iter().enumerate() {
                                let then = seen_at[..i].iter().rev().find(|x| x.1 == *cue && x.0 + back <= *tr);
                                if let Some((_, _, q)) = then {
                                    if let Some(old) = q.iter().find(|x| x.0 == name) {
                                        let mut o = old.1.clone();
                                        o.sort_unstable();
                                        acc[k].0 += sim(&now, &o);
                                        acc[k].1 += 1;
                                    }
                                }
                            }
                        }
                        // and against the other cue's code at the latest trial before (cue vs none)
                        let mut cross = (0.0f64, 0usize);
                        for (i, (_, cue, p)) in seen_at.iter().enumerate().filter(|(_, x)| x.0 + block > trial && x.0 < trial) {
                            let Some(now) = p.iter().find(|x| x.0 == name).map(|x| { let mut v = x.1.clone(); v.sort_unstable(); v }) else { continue };
                            if let Some((_, _, q)) = seen_at[..i].iter().rev().find(|x| x.1 != *cue) {
                                if let Some(old) = q.iter().find(|x| x.0 == name) {
                                    let mut o = old.1.clone();
                                    o.sort_unstable();
                                    cross.0 += sim(&now, &o);
                                    cross.1 += 1;
                                }
                            }
                        }
                        let f = |a: (f64, usize)| if a.1 == 0 { "-".to_string() } else { format!("{:.2}", a.0 / a.1 as f64) };
                        line += &format!(" {name} {}/{}/{} x{}", f(acc[0]), f(acc[1]), f(acc[2]), f(cross));
                    }
                    // the fate of the last block's light-selective top layer 4 cells: how many still
                    // fire on half the light trials, how many now fire on a tenth or more of the
                    // others, how many fire on neither
                    {
                        let mut freq: [HashMap<usize, usize>; 2] = [HashMap::new(), HashMap::new()];
                        let mut n = [0usize; 2];
                        for (_, cue, p) in seen_at.iter().filter(|x| x.0 + block > trial && x.0 < trial) {
                            n[*cue as usize] += 1;
                            if let Some(x) = p.iter().find(|x| x.0 == "top_driven") {
                                for &b in &x.1 {
                                    *freq[*cue as usize].entry(b).or_default() += 1;
                                }
                            }
                        }
                        let on_cue = |b: &usize| freq[1].get(b).copied().unwrap_or(0) * 2 >= n[1].max(1);
                        let on_none = |b: &usize| freq[0].get(b).copied().unwrap_or(0) * 10 > n[0];
                        let (mut kept, mut smeared, mut silent, mut replaced) = (0, 0, 0, 0);
                        let born = |b: usize| mind.brain.full.as_ref().and_then(|f| f.l4.as_ref()).map_or(0, |p| p.born_at(b));
                        for (b, at) in &prev_sel {
                            if born(*b) != *at {
                                replaced += 1;
                                continue;
                            }
                            match (on_cue(b), on_none(b)) {
                                (true, false) => kept += 1,
                                (_, true) => smeared += 1,
                                (false, false) => silent += 1,
                            }
                        }
                        line += &format!(" fate {}: kept {kept} smeared {smeared} silent {silent} replaced {replaced}", prev_sel.len());
                        prev_sel = freq[1].keys().copied().filter(|b| on_cue(b) && !on_none(b)).map(|b| (b, born(b))).collect();
                    }
                    // selective bits: on at least 0.4 more often on cue trials than on the others, per
                    // panel, over this block
                    for name in ["eye", "top_in", "top_driven", "top_l23", "ctx"] {
                        let mut freq: [HashMap<usize, usize>; 2] = [HashMap::new(), HashMap::new()];
                        let mut n = [0usize; 2];
                        for (_, cue, p) in seen_at.iter().filter(|x| x.0 + block > trial && x.0 < trial) {
                            n[*cue as usize] += 1;
                            if let Some(x) = p.iter().find(|x| x.0 == name) {
                                for &b in &x.1 {
                                    *freq[*cue as usize].entry(b).or_default() += 1;
                                }
                            }
                        }
                        // on clearly more often with the cue: P(on | cue) less P(on | none) at least 0.4
                        let sel = freq[1].iter().filter(|&(b, &f)| f as f64 / n[1].max(1) as f64 - freq[0].get(b).copied().unwrap_or(0) as f64 / n[0].max(1) as f64 >= 0.4).count();
                        line += &format!(" sel_{name} {sel}");
                    }
                    // the striatal context's overlap between the two lights, by the source of its
                    // regions: (shared bits, bits on) summed over pairs of consecutive A/B trials
                    {
                        let mut by: std::collections::BTreeMap<usize, (usize, usize)> = Default::default();
                        let recent: Vec<&(usize, bool, Vec<(String, Vec<usize>)>)> = seen_at.iter().filter(|x| x.0 + block > trial && x.0 < trial).collect();
                        for w in recent.windows(2) {
                            if w[0].1 == w[1].1 {
                                continue;
                            }
                            let get = |p: &Vec<(String, Vec<usize>)>| p.iter().find(|x| x.0 == "ctx").map_or(Vec::new(), |x| x.1.clone());
                            let (a, b) = (get(&w[0].2), get(&w[1].2));
                            for &x in a.iter().chain(b.iter()) {
                                by.entry(mind.striatal_source(x / 64).unwrap_or(9999)).or_default().1 += 1;
                            }
                            for &x in a.iter().filter(|x| b.contains(x)) {
                                by.entry(mind.striatal_source(x / 64).unwrap_or(9999)).or_default().0 += 2;
                            }
                        }
                        line += " | ctx overlap by source:";
                        for (src, (sh, on)) in by.iter().filter(|(_, v)| v.1 > 0) {
                            line += &format!(" {src}:{:.2}({on})", *sh as f64 / *on as f64);
                        }
                        // learned attention: each source's mean discrimination over its regions
                        let mut att: std::collections::BTreeMap<usize, (u64, u64)> = Default::default();
                        for r in 0..mind.striatal_regions().1 {
                            if let (Some(src), Some(d)) = (mind.striatal_source(r), mind.attention_of(r)) {
                                let e = att.entry(src).or_default();
                                e.0 += d;
                                e.1 += 1;
                            }
                        }
                        if !att.is_empty() {
                            line += " | attention by source:";
                            for (src, (d, n)) in &att {
                                line += &format!(" {src}:{:.2}", *d as f64 / *n as f64 / 65536.0);
                            }
                        }
                    }
                    // phase: bits that code which phase it is (which light pays), whichever light is
                    // on. Against the latest block of the other phase, per light: bits on at least 0.4
                    // more often now (raw); the same against the block a phase earlier of this phase
                    // (control: drift alone); and bits on in both blocks of this phase and at least
                    // 0.4 less in the other phase between them, or the reverse (consistent: a phase code)
                    if matches!(kind, Conditioning::Reversal) {
                        let phase_a = ((trial - block) / world.reversal_period.max(1)) % 2 == 0;
                        let mut per: Vec<([Freq; 2], [u32; 2])> = Vec::new();
                        for name in drift_panels {
                            let mut f: [Freq; 2] = [Freq::new(), Freq::new()];
                            let mut n = [0u32; 2];
                            for (_, cue, p) in seen_at.iter().filter(|x| x.0 + block > trial && x.0 < trial) {
                                n[*cue as usize] += 1;
                                if let Some(x) = p.iter().find(|x| x.0 == name) {
                                    for &b in &x.1 {
                                        *f[*cue as usize].entry(b).or_default() += 1;
                                    }
                                }
                            }
                            per.push((f, n));
                        }
                        phase_blocks.push((phase_a, per));
                        let now = phase_blocks.len() - 1;
                        let other = (0..now).rev().find(|&i| phase_blocks[i].0 != phase_a);
                        let earlier = other.and_then(|o| (0..o).rev().find(|&i| phase_blocks[i].0 == phase_a));
                        let control = now.checked_sub(1).filter(|&i| phase_blocks[i].0 == phase_a);
                        if let Some(o) = other {
                            line += " | phase bits (raw/control/consistent):";
                            for (k, name) in drift_panels.iter().enumerate() {
                                let rate = |i: usize, l: usize, b: usize| -> f64 {
                                    let (f, n) = &phase_blocks[i].1[k];
                                    f[l].get(&b).copied().unwrap_or(0) as f64 / n[l].max(1) as f64
                                };
                                let bits = |i: usize, l: usize| -> Vec<usize> { phase_blocks[i].1[k].0[l].keys().copied().collect() };
                                let differ = |a: usize, b: usize| -> usize {
                                    (0..2).map(|l| {
                                        let mut all = bits(a, l);
                                        all.extend(bits(b, l));
                                        all.sort_unstable();
                                        all.dedup();
                                        all.iter().filter(|&&x| (rate(a, l, x) - rate(b, l, x)).abs() >= 0.4).count()
                                    }).sum::<usize>() / 2
                                };
                                let raw = differ(now, o);
                                let ctrl = control.map_or("-".to_string(), |c| differ(now, c).to_string());
                                let cons = earlier.map_or("-".to_string(), |e| {
                                    ((0..2).map(|l| {
                                        let mut all = bits(now, l);
                                        all.extend(bits(o, l));
                                        all.extend(bits(e, l));
                                        all.sort_unstable();
                                        all.dedup();
                                        all.iter().filter(|&&x| {
                                            let (a, b, c) = (rate(now, l, x), rate(o, l, x), rate(e, l, x));
                                            (a - b >= 0.4 && c - b >= 0.4) || (b - a >= 0.4 && b - c >= 0.4)
                                        }).count()
                                    }).sum::<usize>() / 2).to_string()
                                });
                                line += &format!(" {name} {raw}/{ctrl}/{cons}");
                            }
                        }
                    }
                    let mut sz = [(0usize, 0usize, 0usize); 2];
                    for (_, cue, p) in seen_at.iter().filter(|x| x.0 + block > trial && x.0 < trial) {
                        let n = |k: &str| p.iter().find(|x| x.0 == k).map_or(0, |x| x.1.len());
                        let s = &mut sz[*cue as usize];
                        s.0 += n("top_driven");
                        s.1 += n("top_l4");
                        s.2 += 1;
                    }
                    let r = |s: (usize, usize, usize)| format!("{:.0}/{:.0}", s.0 as f64 / s.2.max(1) as f64, s.1 as f64 / s.2.max(1) as f64);
                    line += &format!(" | driven/l4 bits: cue {} none {}", r(sz[1]), r(sz[0]));
                    {
                        let mut ctx_bits = (0usize, 0usize);
                        for (_, _, p) in seen_at.iter().filter(|x| x.0 + block > trial && x.0 < trial) {
                            if let Some(x) = p.iter().find(|x| x.0 == "ctx") {
                                ctx_bits.0 += x.1.len();
                                ctx_bits.1 += 1;
                            }
                        }
                        // density: each panel's bits on over its width, averaged over this block
                        let widths = [("eye", 2048.0), ("top_driven", 4096.0), ("top_l23", 4096.0), ("wm", 4096.0), ("ofc_z", 4096.0), ("ctx", 16384.0)];
                        let mut dens = String::new();
                        for (name, w) in widths {
                            let mut a = (0usize, 0usize);
                            for (_, _, p) in seen_at.iter().filter(|x| x.0 + block > trial && x.0 < trial) {
                                if let Some(x) = p.iter().find(|x| x.0 == name) {
                                    a.0 += x.1.len();
                                    a.1 += 1;
                                }
                            }
                            dens += &format!(" {name} {:.1}%", 100.0 * a.0 as f64 / a.1.max(1) as f64 / w);
                        }
                        line += &format!(" | density{dens}");
                        let (owned, regions) = mind.striatal_regions();
                        line += &format!(" | ctx bits {:.0}, regions owned {owned} of {regions}", ctx_bits.0 as f64 / ctx_bits.1.max(1) as f64);
                    }
                    eprintln!("progress {kind:?} seed {seed} trials {trial}: drift (1/50/100 back, x other cue){line}");
                }
                let (used, rec, cyc, why) = mind.brain.full_l4();
                eprintln!("progress {kind:?} seed {seed} trials {trial}: top layer 4 cells in use {used} (recruited {rec}: salient {}, recurring {}, new {}; recycled {cyc}; orbitofrontal recycled {}); phase switches {}, context {}", why.0, why.1, why.2, mind.ofc_recycled, mind.phase_switches, mind.phase_context());
            }
            a_pays = (trial / world.reversal_period.max(1)) % 2 == 0;
            if sleep_every > 0 && trial % sleep_every == 0 {
                mind.sleep(3);
                if trial % block == 0 {
                    let (n, h) = mind.brain.sense_hits();
                    println!(
                        "  sleep at trial {trial}: cortical cells {}, sensory predictions right {:.1}%; compacted so far {} merged, {} pruned, {} generalized; hippocampal rows {}, consolidated {}",
                        mind.brain.cortical_cells(),
                        100.0 * h as f64 / n.max(1) as f64,
                        mind.compacted.0,
                        mind.compacted.1,
                        mind.brain.generalized,
                        mind.hc.mem.rows(),
                        mind.hc.consolidated
                    );
                }
            }
        }
        let b = (trial / block).min(63);
        {
            let c = mind.amy.churn;
            churn[b][t].0 += c.1 - last_churn.1;
            churn[b][t].1 += c.0 - last_churn.0;
            last_churn = c;
            if t == 2 && cue {
                let ear: Vec<u32> = mind.amy.trace_at(0).iter().copied().filter(|&f| (2048..4096).contains(&f)).collect();
                tone[b].0 += ear.len() as u64;
                tone[b].1 += ear.iter().filter(|f| last_tone.contains(f)).count() as u64;
                tone[b].2 += 1;
                last_tone = ear;
            }
            if t == 4 && cue {
                age2[b].0 += mind.amy.ages().get(2).copied().unwrap_or(0) as u64;
                age2[b].1 += 1;
            }
        }
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
            frus[b].0 += r.frustration as i64;
            frus[b].1 += 1;
        }
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
            Conditioning::Instrumental | Conditioning::Reversal => 3..=5,
            Conditioning::Delayed => 5..=7,
        };
        if window.contains(&t) && ev.moved == Some(CLICK) {
            x.5 += 1;
        }
        if t == 9 {
            x.6 += 1;
        }
        if kind == Conditioning::Reversal {
            let pays = cue == a_pays;
            let r = &mut rev[b];
            if window.contains(&t) && ev.moved == Some(CLICK) {
                if pays {
                    r.0 += 1;
                } else {
                    r.2 += 1;
                }
            }
            if t == 9 {
                if pays {
                    r.1 += 1;
                } else {
                    r.3 += 1;
                }
            }
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
            Conditioning::Reversal => println!(
                "  trials {:4}-{:4}: clicks in the window per trial, the paying light {:.2}, the other {:.2} ({} pays)",
                b * block,
                (b + 1) * block,
                rev[b].0 as f64 / rev[b].1.max(1) as f64,
                rev[b].2 as f64 / rev[b].3.max(1) as f64,
                if (b * block / world.reversal_period.max(1)) % 2 == 0 { "A" } else { "B" }
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
    for (b, c) in churn.iter().enumerate().filter(|x| x.1.iter().any(|y| y.0 + y.1 > 0)) {
        println!("  block {b}: appetitive cells removed / recruited by trial step: {}; the trace's items two steps back at the touch {:.1}; the tone's ear onsets {:.1}, {:.1} of them as last time", c.iter().map(|(r, a)| format!("{r}/{a}")).collect::<Vec<_>>().join(" "), age2[b].0 as f64 / age2[b].1.max(1) as f64, tone[b].0 as f64 / tone[b].2.max(1) as f64, tone[b].1 as f64 / tone[b].2.max(1) as f64);
    }
    println!("amygdala cells (appetitive, aversive, pause, baseline): {:?}; orbitofrontal {:?}; what an act usually brings {:+.3}", mind.amy.cells(), mind.ofc.cells(), mind.report.rate_act as f64 / 65536.0);
    println!("frustration per block: {}", frus.iter().filter(|x| x.1 > 0).map(|x| format!("{:.3}", x.0 as f64 / x.1 as f64 / 65536.0)).collect::<Vec<_>>().join(" "));
    {
        use neurocomp::program::area::ColumnId;
        let full = mind.brain.loads_of(ColumnId::Assoc(mind.brain.pairs.len()));
        let o = mind.ofc_area.col.loads;
        println!("claustrum: {} of {} proposals held; held loads (surprise, gate): full association {:?}, orbitofrontal {:?}", mind.claustrum.stats.1, mind.claustrum.stats.0, full, o);
        let n = mind.brain.pairs.len();
        if let Some(f) = mind.brain.full.as_ref() {
            println!("held allowance: full association {} bits (share {:.3}, grown {} times), orbitofrontal {} bits (share {:.3}, grown {} times); layer 4 widths {} and {}", f.held_cap, f.rho as f64 / 65536.0, f.resized.0, mind.ofc_area.col.held_cap, mind.ofc_area.col.rho as f64 / 65536.0, mind.ofc_area.col.resized.0, f.l4.as_ref().map_or(0, |p| p.cells()), mind.ofc_area.col.l4.as_ref().map_or(0, |p| p.cells()));
        }
        let _ = n;
    }
    let o = &mind.ofc_area;
    println!("orbitofrontal area: cells (layer 2/3, outcome) {:?}; outcomes expected right {} of {}; holds {}; splits {}", o.cells(), o.stats.1, o.stats.0, o.stats.2, o.stats.3);
    println!("cholinergic pauses: {pauses} steps; amygdala updates {}; per step: onsets {:.1}, attended {:.1}, acts {:.2}, tonic noise {:.2}", mind.amy.stats.0, sizes.1 as f64 / sizes.0 as f64, sizes.2 as f64 / sizes.0 as f64, sizes.3 as f64 / sizes.0 as f64, tonic_sum as f64 / sizes.0 as f64);
}
