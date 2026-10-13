//! A generic column's held layer 2/3 population on a synthetic reversal, alone: does it hold a
//! state through a phase, let go at the swap, and come back to the same state when the phase
//! comes back? Nothing task-specific: the column sees random sparse codes, one step for the light
//! shown (A or B) and one for what followed (paid or not, the same code whichever light), and
//! predicts each step from the last (its input and what it holds). The paying light pays with
//! probability `P` and swaps every `PERIOD` acts.
//!
//! Per swap: acts until the held population first falls silent, whether the state it settles on
//! is the one that phase had before (most of its bits), and releases beyond the first.
//! Within phases: how often the outcome was predicted right.
//!
//! `COL_P`, `COL_PERIOD`, `COL_PHASES`, `COL_SEEDS`, `COL_MODE=wrong|life` (held until wrong, or
//! the fixed persistence).

use neurocomp::bitvec::BitVector;
use neurocomp::program::column::Column;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const BITS: usize = 2048;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn code(rng: &mut StdRng, k: usize) -> BitVector {
    let mut v = BitVector::new(BITS, Some(0));
    while (v.count_ones() as usize) < k {
        v.bit_set(rng.gen_range(0..BITS));
    }
    v
}

fn overlap(a: &BitVector, b: &BitVector) -> u32 {
    a.as_words().iter().zip(b.as_words()).map(|(x, y)| (x & y).count_ones()).sum()
}

fn main() {
    let p: f64 = env("COL_P", 0.5);
    let period: u32 = env("COL_PERIOD", 250);
    let phases: usize = env("COL_PHASES", 8);
    let seeds: u64 = env("COL_SEEDS", 10);
    let wrong = std::env::var("COL_MODE").map_or(true, |v| v == "wrong");
    println!("column alone: {}, pay {p}, {period} acts a phase, {seeds} seeds", if wrong { "held until wrong" } else { "fixed persistence" });
    println!("swap  detected  median-acts  missed  back  false-releases  outcome-right");
    let mut rows: Vec<(Vec<u32>, u32, u32, u32, u64, u64)> = vec![(Vec::new(), 0, 0, 0, 0, 0); phases];
    for seed in 0..seeds {
        let mut rng = StdRng::seed_from_u64(seed);
        let lights = [code(&mut rng, 32), code(&mut rng, 32)];
        let outcomes = [code(&mut rng, 32), code(&mut rng, 32)];
        let mut col = Column::agranular(BITS, 4096);
        col.set_recurrence(2);
        col.superpose = true;
        col.hold_until_wrong = wrong;
        let mut end_state: [Option<BitVector>; 2] = [None, None];
        for ph in 0..phases {
            let kind = ph % 2;
            let releases_before = col.releases;
            let mut detected: Option<u32> = None;
            let (mut right, mut checked) = (0u64, 0u64);
            for t in 0..period {
                let light = rng.gen_range(0..2usize);
                let paid = light == kind && rng.gen_bool(p);
                let before = col.releases;
                col.sense_code(&lights[light], None, true, &mut rng);
                let expected = col.expected.clone();
                let out = &outcomes[paid as usize];
                col.sense_code(out, None, true, &mut rng);
                if let Some(e) = expected {
                    checked += 1;
                    let other = &outcomes[1 - paid as usize];
                    right += (overlap(&e, out) > overlap(&e, other)) as u64;
                }
                if col.releases > before && detected.is_none() {
                    detected = Some(t);
                }
            }
            let r = &mut rows[ph];
            r.4 += right;
            r.5 += checked;
            if ph > 0 {
                match detected {
                    Some(t) => r.0.push(t),
                    None => r.1 += 1,
                }
                r.3 += (col.releases - releases_before).saturating_sub(1) as u32;
                if let Some(prev) = &end_state[kind] {
                    let now = col.held();
                    if now.count_ones() > 0 && overlap(prev, now) * 2 >= prev.count_ones().min(now.count_ones()) as u32 {
                        r.2 += 1;
                    }
                }
            }
            end_state[kind] = Some(col.held().clone());
        }
    }
    for (ph, r) in rows.iter().enumerate() {
        let mut d = r.0.clone();
        d.sort();
        let med = d.get(d.len() / 2).map_or("-".to_string(), |x| x.to_string());
        let acc = if r.5 > 0 { format!("{:.2}", r.4 as f64 / r.5 as f64) } else { "-".into() };
        if ph == 0 {
            println!("   0         -            -       -     -               -  {acc:>13}");
        } else {
            println!("{ph:>4}  {:>8}  {med:>11}  {:>6}  {:>4}  {:>14}  {acc:>13}", d.len(), r.1, r.2, r.3);
        }
    }
}
