//! Does a reader of the match vector hold the context through a reversal?
//!
//! **Task.** Each trial: a light (A or B, at random), then (with `REV_CLICK=1`) a click, the same
//! whichever light, then the outcome (paid or not). The paying light pays half the time; the
//! other never does. The paying light swaps every `REV_PERIOD` trials. Every code sits in one
//! 64-bit word of a 512-bit step vector (topographic).
//!
//! **Layer 1 (prediction).** Kernels read the last two steps (two lag slots, a 1024-bit context)
//! and predict the next step. They count the lines that are on, firing at 3/4 of their own mask;
//! they do not hold each other down, so a context followed by either of two things predicts both.
//! A context gets one kernel for each thing that has followed it: when nothing that matched
//! predicted the step, a kernel is born on the context, writing that step.
//!
//! **The match vector M.** Every kernel that matched writes its small code (4 lines in one word);
//! if what it predicted came, its big code (those lines plus 4 more). `REV_CODES=match`: small
//! codes only (no confirmation); `REV_CODES=confirm`: big codes only, written only when right;
//! default `nested`.
//!
//! **Layer 2 (the reader, a harness rule for now).** Context assemblies over M, as the earlier
//! branch's holder: each has the lines of M it expects (the OR of its predictors' outputs). An
//! assembly explains a step when fewer than `REV_RES` lines of M are unexpected (default 4: the
//! extra lines of one big code). A young assembly (fewer than 16 explained steps) learns what it
//! does not explain. A settled one is challenged only by a rival that explains the step when it
//! does not: the rival gains drive, which leaks 1 in 16 steps; the reader switches when a rival's
//! drive reaches bitlen(explained). A step nobody explains seeds a candidate assembly.
//!
//! Per swap: detected (switched before the next swap), median trials to switch, back (switched
//! to the assembly this kind of phase had before), new (to a new assembly), false alarms (switches
//! after the first, per phase; and any in the first phase).
//!
//! `REV_P`, `REV_PERIOD`, `REV_PHASES`, `REV_SEEDS`, `REV_CLICK`, `REV_CODES`, `REV_RES`.

use neurocomp::bitvec::BitVector;
use neurocomp::kernel::{KernelOp, SimpleKernel};
use neurocomp::world::clustered_code;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

const STEP: usize = 512;
const MBITS: usize = 4096;

#[derive(Clone, Copy, PartialEq)]
enum Codes {
    Nested,
    Match,
    Confirm,
}

fn pop(v: &[u64]) -> u32 {
    v.iter().map(|w| w.count_ones()).sum()
}

/// Layer 1: predictors over the last two steps, writing the match vector.
struct Predict {
    kernels: Vec<SimpleKernel>,
    prev: [BitVector; 2],
    codes: Codes,
}

impl Predict {
    fn context(&self) -> BitVector {
        let mut w = self.prev[0].as_words().to_vec();
        w.extend_from_slice(self.prev[1].as_words());
        BitVector::from_words(w)
    }

    /// One step: predict it from the context, check, write M, learn; then the step enters the
    /// window.
    fn step(&mut self, now: &BitVector, rng: &mut StdRng) -> BitVector {
        let ctx = self.context();
        let mut m = BitVector::new(MBITS, Some(0));
        let mut any_right = false;
        for k in &mut self.kernels {
            if k.agreement(&ctx) >= k.threshold {
                let right = k.output_mask.as_words() == now.as_words();
                any_right |= right;
                match self.codes {
                    Codes::Nested => {
                        k.write_match(&mut m);
                        k.confirm(right, Some(&mut m));
                    }
                    Codes::Match => k.write_match(&mut m),
                    Codes::Confirm => k.confirm(right, Some(&mut m)),
                }
            }
        }
        let lines = ctx.count_ones();
        if !any_right && lines > 0 {
            let thr = (3 * lines).div_ceil(4);
            let mut k = SimpleKernel::new(ctx.clone(), 0, now.clone(), 0, thr, KernelOp::Or);
            let (small, extra) = SimpleKernel::nested_code(MBITS, 4, 4, rng);
            k.match_code = Some(small);
            k.confirm_mask = Some(extra);
            self.kernels.push(k);
        }
        self.prev[1] = std::mem::replace(&mut self.prev[0], now.clone());
        m
    }
}

/// Layer 2: context assemblies over M (the earlier branch's holder, as a harness rule).
struct Reader {
    expects: Vec<Vec<u64>>,
    explained: Vec<u8>,
    drive: Vec<u8>,
    active: Option<usize>,
    res: u32,
}

impl Reader {
    fn unexpected(&self, l: usize, m: &[u64]) -> u32 {
        pop(&m.iter().zip(&self.expects[l]).map(|(x, e)| x & !e).collect::<Vec<_>>())
    }

    fn learn(&mut self, l: usize, m: &[u64]) {
        for (e, x) in self.expects[l].iter_mut().zip(m) {
            *e |= x;
        }
    }

    fn new_assembly(&mut self, m: &[u64], drive: u8) -> usize {
        self.expects.push(m.to_vec());
        self.explained.push(0);
        self.drive.push(drive);
        self.expects.len() - 1
    }

    fn enter(&mut self, l: usize) {
        self.active = Some(l);
        self.drive.iter_mut().for_each(|d| *d = 0);
    }

    fn hold(&mut self, m: &BitVector, rng: &mut StdRng) -> usize {
        let m = m.as_words();
        if pop(m) == 0 {
            return self.active.unwrap_or(0);
        }
        let n = self.expects.len();
        let e: Vec<bool> = (0..n).map(|l| self.unexpected(l, m) < self.res).collect();
        let Some(a) = self.active else {
            let l = self.new_assembly(m, 0);
            self.enter(l);
            return l;
        };
        let strength = 32 - (self.explained[a] as u32).leading_zeros() as u8;
        let young = self.explained[a] < 16;
        if e[a] {
            self.explained[a] = self.explained[a].saturating_add(1);
        } else if young {
            self.learn(a, m);
        } else if !(0..n).any(|l| l != a && e[l]) {
            // nobody explains it: a candidate assembly, with this step as its first drive
            self.new_assembly(m, 1);
        }
        let leak = rng.gen_range(0..16) == 0;
        let mut winner: Option<usize> = None;
        for l in 0..self.expects.len() {
            if l == a {
                continue;
            }
            if e.get(l).copied().unwrap_or(false) && !e[a] {
                self.drive[l] = self.drive[l].saturating_add(1);
            }
            if leak {
                self.drive[l] = self.drive[l].saturating_sub(1);
            }
            if !young && self.drive[l] >= strength && winner.map_or(true, |w| self.drive[l] > self.drive[w]) {
                winner = Some(l);
            }
        }
        if let Some(l) = winner {
            self.enter(l);
        }
        self.active.unwrap()
    }
}

fn main() {
    let p: f64 = env("REV_P", 0.5);
    let period: usize = env("REV_PERIOD", 500);
    let phases: usize = env("REV_PHASES", 8);
    let seeds: u64 = env("REV_SEEDS", 10);
    let click_on = std::env::var("REV_CLICK").is_ok();
    let res: u32 = env("REV_RES", 4);
    let codes = match std::env::var("REV_CODES").unwrap_or_default().as_str() {
        "match" => Codes::Match,
        "confirm" => Codes::Confirm,
        _ => Codes::Nested,
    };
    let name = match codes { Codes::Nested => "nested", Codes::Match => "match only", Codes::Confirm => "confirm only" };
    println!("reversal reader: {name} codes, {}, pay {p}, period {period}, {seeds} seeds, residue < {res}", if click_on { "light, click, outcome" } else { "light, outcome" });
    println!("swap  detected  median-trials  missed  back  new  false-alarms");
    // per phase: (trials to switch, missed, back, new, false alarms)
    let mut rows: Vec<(Vec<usize>, u32, u32, u32, u32)> = vec![(Vec::new(), 0, 0, 0, 0); phases];
    let mut first_phase_switches = 0u32;
    let mut l1_kernels = 0usize;
    let mut assemblies = 0usize;
    for seed in 0..seeds {
        let mut rng = StdRng::seed_from_u64(seed);
        let lights = [clustered_code(STEP, 16, &mut rng), clustered_code(STEP, 16, &mut rng)];
        let outcome = [clustered_code(STEP, 16, &mut rng), clustered_code(STEP, 16, &mut rng)];
        let click = clustered_code(STEP, 16, &mut rng);
        let zero = BitVector::new(STEP, Some(0));
        let mut pred = Predict { kernels: Vec::new(), prev: [zero.clone(), zero.clone()], codes };
        let mut reader = Reader { expects: Vec::new(), explained: Vec::new(), drive: Vec::new(), active: None, res };
        let mut end_of: [Option<usize>; 2] = [None, None];
        for ph in 0..phases {
            let kind = ph % 2;
            let before = reader.active;
            let n_before = reader.expects.len();
            let mut switched: Vec<(usize, usize)> = Vec::new();
            for t in 0..period {
                let light = rng.gen_range(0..2usize);
                let paid = light == kind && rng.gen_bool(p);
                let mut steps = vec![lights[light].clone()];
                if click_on {
                    steps.push(click.clone());
                }
                steps.push(outcome[paid as usize].clone());
                for s in &steps {
                    let was = reader.active;
                    let m = pred.step(s, &mut rng);
                    let now = reader.hold(&m, &mut rng);
                    if was.is_some() && was != Some(now) {
                        switched.push((t, now));
                        if std::env::var("REV_DEBUG").is_ok() && seed == 0 && ph == 2 && switched.len() < 30 {
                            let mw = m.as_words();
                            let un: Vec<u32> = (0..reader.expects.len()).map(|l| reader.unexpected(l, mw)).collect();
                            eprintln!("t {t} light {light} paid {paid} step-pop {} {:?} -> {now}; unexpected per assembly {:?}; explained {:?}", m.count_ones(), was, un, reader.explained);
                        }
                    }
                }
            }
            let r = &mut rows[ph];
            if ph == 0 {
                // settling in: switches after the first 50 trials are false alarms
                first_phase_switches += switched.iter().filter(|s| s.0 >= 50).count() as u32;
            } else {
                match switched.first() {
                    Some(&(t, to)) => {
                        r.0.push(t);
                        r.2 += (end_of[kind].is_some() && Some(to) == end_of[kind] && Some(to) != before) as u32;
                        r.3 += (to >= n_before) as u32;
                    }
                    None => r.1 += 1,
                }
                r.4 += switched.len().saturating_sub(1) as u32;
                if std::env::var("REV_DEBUG2").is_ok() && kind == 0 {
                    let to: Vec<usize> = switched.iter().map(|s| s.1).collect();
                    eprintln!("seed {seed} phase {ph}: {} switches, between {:?}", to.len(), &to[..to.len().min(16)]);
                }
            }
            end_of[kind] = reader.active;
        }
        l1_kernels += pred.kernels.len();
        assemblies += reader.expects.len();
    }
    for (ph, r) in rows.iter().enumerate().skip(1) {
        let mut d = r.0.clone();
        d.sort();
        let med = d.get(d.len() / 2).map_or("-".to_string(), |x| x.to_string());
        println!("{ph:>4}  {:>8}  {med:>13}  {:>6}  {:>4}  {:>3}  {:>12}", d.len(), r.1, r.2, r.3, r.4);
    }
    println!("first-phase false alarms {first_phase_switches}; mean layer-1 kernels {}, mean assemblies {}", l1_kernels / seeds as usize, assemblies / seeds as usize);
}
