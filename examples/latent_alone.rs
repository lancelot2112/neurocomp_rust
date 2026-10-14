//! The latent-cause attractor alone, on a synthetic reversal: two lights and a background, one
//! light shown per trial; acting at the paying light pays with probability `P`, else nothing.
//! The paying light swaps every `PERIOD` trials. The worths per context are learned as in the
//! mind (toward each outcome at 1/(2 + outcomes learned from), in the active context), and the
//! attractor decides the context from the outcomes alone.
//!
//! How the agent samples matters (a learner that stops acting at an unpaid light stops hearing
//! that it pays again), so two policies: `random` acts at half the trials whatever it has
//! learned; `greedy` acts at a light in proportion to its worth in the active context (at least
//! a tenth of the time).
//!
//! Per swap it reports the trials until the first silencing, where the attractor settled
//! (back on the context that phase had before, a new one, or a wrong old one), and silencings
//! beyond the first in a phase (false alarms).
//!
//! `LATENT_MODEL=attractor|circuit` (the rate-test attractor, or the recurrent assembly stopped by
//! inhibition, `assembly.rs`), `LATENT_SCALE=fixed|volatile`, `LATENT_POLICY=random|greedy`, `LATENT_P` (0..1),
//! `LATENT_PERIOD`, `LATENT_PHASES`, `LATENT_SEEDS`, `LATENT_DEBUG=1`.

use neurocomp::bitvec::BitVector;
use neurocomp::fixed::ONE;
use neurocomp::program::area::Pool;
use neurocomp::program::assembly::Assembly;
use neurocomp::program::kernel_cells::{Holder, Kernel};
use neurocomp::program::latent::{LatentCause, Scale};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};

const BITS: usize = 3;
const BACKGROUND: usize = 2;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

#[derive(Default, Clone)]
struct Swap {
    detect: Vec<u32>,
    missed: u32,
    back: u32,
    new: u32,
    wrong: u32,
    false_alarms: u32,
}

fn main() {
    let scale = if std::env::var("LATENT_SCALE").map_or(true, |v| v == "volatile") { Scale::Volatile } else { Scale::Fixed };
    let greedy = std::env::var("LATENT_POLICY").map_or(false, |v| v == "greedy");
    let p: f64 = env("LATENT_P", 0.5);
    let period: u32 = env("LATENT_PERIOD", 500);
    let phases: usize = env("LATENT_PHASES", 8);
    let seeds: u64 = env("LATENT_SEEDS", 20);
    let debug = std::env::var("LATENT_DEBUG").is_ok();
    let kern = std::env::var("LATENT_MODEL").map_or(false, |v| v == "kernel");
    let circuit = kern || std::env::var("LATENT_MODEL").map_or(false, |v| v == "circuit");
    // per phase index: the outcome of the swap into it
    let mut per: Vec<Swap> = vec![Swap::default(); phases];
    let mut first_phase_false = 0u32;
    let from: u64 = env("LATENT_FROM", 0);
    for seed in from..from + seeds {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut lc = LatentCause::new(BITS, scale);
        lc.debug = debug;
        let mut cc = Assembly::holding(env("LATENT_N", 256), env("LATENT_K", 16), 1024, seed ^ 0x5EED);
        cc.debug = debug;
        // (`LATENT_MODEL=kernel`: kernel cells separate the input and a holder of kernel
        // assemblies holds; `kernel_cells.rs`)
        // (`LATENT_HIST=h`: the kernel's cells may look up to h inputs back: conjunctions in time)
        let hist: usize = env("LATENT_HIST", 0);
        let mut ksep = Kernel::new(1024, 1024, 4, 8, seed ^ 0xC0DE).with_history(hist);
        // (`LATENT_CONJ=1`: only the conjunction cells send to the holder)
        ksep.conj_only = std::env::var("LATENT_CONJ").is_ok();
        let mut kh = Holder::new(1024, seed ^ 0xF00D);
        let noise: usize = env("LATENT_NOISE", 0);
        let seq = std::env::var("LATENT_SEQ").is_ok();
        // (`LATENT_TRACE=p`: a trace, rotated and thinned at 2^-p a step, is what the kernel codes)
        let trace_p: u32 = env("LATENT_TRACE", 0);
        let mut trace = vec![0u64; 16];
        // (the thinning is a fixed mask, the same every step: each bit kept unless all of p
        // fixed random words have it set, so the same recent sequence leaves the same trace)
        let keep: Vec<u64> = {
            let mut r = StdRng::seed_from_u64(0x7EA5);
            (0..16).map(|_| { let mut d = !0u64; for _ in 0..trace_p.max(1) { d &= r.next_u64(); } !d }).collect()
        };
        let mut prev = BitVector::new(1024, Some(0));
        kh.debug = debug;
        // the circuit's input: layer 4's code of what came together (the light shown and what
        // followed), a random sparse code per conjunction, as layer 4 recruits one for each
        // (`LATENT_SHARED=1`: a conjunction's code shares half its bits with the other outcome
        // at the same light, as layer 4's codes of inputs in common overlap)
        let shared = std::env::var("LATENT_SHARED").is_ok();
        let mut sparse = |k: usize| {
            let mut v = BitVector::new(1024, Some(0));
            while (v.count_ones() as usize) < k {
                v.bit_set(rng.gen_range(0..1024));
            }
            v
        };
        let cue = [sparse(16), sparse(16)];
        // (`LATENT_L4=1`: the circuit reads layer 4's code: a pool over two sources, the light
        // and what followed (paid or not, the same code whichever light), recruiting conjunction
        // cells as new inputs need them)
        let l4_on = std::env::var("LATENT_L4").is_ok();
        let outcome = [sparse(16), sparse(16)];
        let click = sparse(16);
        let mut l4 = Pool::new(2, 1024, 1024, 32, 2, seed);
        // (`LATENT_SEP=1`: the codes pass through pattern separation first)
        let sep_on = std::env::var("LATENT_SEP").is_ok();
        let mut sep = Assembly::coding(1024, 16, 1024, 32, seed);
        let codes: Vec<BitVector> = (0..4)
            .map(|i| {
                if shared {
                    let mut v = sparse(16);
                    for (w, c) in v.as_words_mut().iter_mut().zip(cue[i / 2].as_words()) {
                        *w |= c;
                    }
                    v
                } else {
                    sparse(32)
                }
            })
            .collect();
        let distract: Vec<BitVector> = (0..noise).map(|_| sparse(32)).collect();
        // (layer 4 develops first: the separator settles on the inputs the world can give, in
        // random order, before the task and the circuit begin)
        if sep_on {
            // (only what the first phase can give: A paid, A unpaid, B unpaid; B paid is first
            // met at the first swap)
            for _ in 0..2000 {
                let i = [0usize, 1, 2][rng.gen_range(0..3usize)];
                sep.code(&codes[i], true);
            }
        }
        let mut worth: Vec<Vec<i32>> = vec![vec![-1; BITS]];
        let mut learned: Vec<Vec<u16>> = vec![vec![0; BITS]];
        // which context each kind of phase (A pays / B pays) last settled on
        let mut home: [Option<usize>; 2] = [None, None];
        for ph in 0..phases {
            let kind = ph % 2;
            let resets_before = if kern { kh.resets } else if circuit { cc.resets() } else { lc.resets };
            let mut detected: Option<u32> = None;
            for t in 0..period {
                let light = rng.gen_range(0..2usize);
                let ctx = if kern { kh.now } else if circuit { cc.now() } else { lc.now };
                let act = if greedy {
                    let v = worth.get(ctx).map_or(-1, |w| w[light]);
                    let share = if v < 0 { 0.5 } else { (v as f64 / ONE as f64).max(0.1) };
                    rng.gen_bool(share.min(1.0))
                } else {
                    rng.gen_bool(0.5)
                };
                if !act {
                    continue;
                }
                let got = if light == kind && rng.gen_bool(p) { ONE as i32 } else { 0 };
                let bits = [light, BACKGROUND];
                let before = if kern { kh.resets } else if circuit { cc.resets() } else { lc.resets };
                if kern {
                    // (`LATENT_NOISE=n`: before each act, half the time, one of n other inputs,
                    // the same in every phase, as the area's other sources come and go)
                    if noise > 0 && rng.gen_bool(0.5) {
                        let d = &distract[rng.gen_range(0..noise)];
                        let born = ksep.births;
                        let (code, _) = ksep.code(d, true);
                        kh.hold(&code, ksep.births > born);
                    }
                    let born = ksep.births;
                    // (`LATENT_SEQ=1`: the act is a sequence of two steps, the light and then what
                    // followed, paid or not, the same code whichever light; the kernel's input is
                    // each step with the step before it, rotated (a transition))
                    let ev = if seq && trace_p > 0 {
                        // the cheapest recursion: one trace, each step rotated by a fixed amount,
                        // thinned by a fixed mask (each old bit dropped where the mask says, a share
                        // 2^-p), and the new input ORed in; the kernel codes the trace
                        let mut step = |now: &BitVector, trace: &mut Vec<u64>, rng: &mut StdRng| -> BitVector {
                            let mut t = vec![0u64; 16];
                            for b in 0..1024 {
                                if trace[b / 64] >> (b % 64) & 1 == 1 {
                                    let d = (b + 97) % 1024;
                                    t[d / 64] |= 1 << (d % 64);
                                }
                            }
                            for (w, k) in t.iter_mut().zip(&keep) {
                                *w &= k;
                            }
                            for (w, x) in t.iter_mut().zip(now.as_words()) {
                                *w |= x;
                            }
                            *trace = t.clone();
                            BitVector::from_words(t)
                        };
                        for first in [cue[light].clone(), click.clone()] {
                            let v = step(&first, &mut trace, &mut rng);
                            let b0 = ksep.births;
                            let (c1, _) = ksep.code(&v, true);
                            kh.hold(&c1, ksep.births > b0);
                        }
                        let out = outcome[(got > 0) as usize].clone();
                        step(&out, &mut trace, &mut rng)
                    } else if seq && hist > 0 {
                        // light, then the click (the same whichever light), then what followed;
                        // each coded as it is, the kernel's history holding what came before
                        for first in [cue[light].clone(), click.clone()] {
                            let b0 = ksep.births;
                            let (c1, _) = ksep.code(&first, true);
                            kh.hold(&c1, ksep.births > b0);
                        }
                        outcome[(got > 0) as usize].clone()
                    } else if seq {
                        let step = |now: &BitVector, before: &BitVector| -> BitVector {
                            let mut v = now.clone();
                            for b in (0..1024).filter(|&b| before.bit_get(b)) {
                                v.bit_set((b + 512) % 1024);
                            }
                            v
                        };
                        let first = step(&cue[light], &prev);
                        let b0 = ksep.births;
                        let (c1, _) = ksep.code(&first, true);
                        kh.hold(&c1, ksep.births > b0);
                        let out = &outcome[(got > 0) as usize];
                        let second = step(out, &cue[light]);
                        prev = out.clone();
                        second
                    } else {
                        codes[light * 2 + (got > 0) as usize].clone()
                    };
                    let born = if seq { ksep.births } else { born };
                    let (code, _) = ksep.code(&ev, true);
                    if debug {
                        eprintln!("event light {light} paid {} code {:?} new {}", got > 0, (0..1024).filter(|&b| code.bit_get(b)).take(6).collect::<Vec<_>>(), ksep.births > born);
                    }
                    kh.hold(&code, ksep.births > born);
                } else if circuit {
                    if l4_on {
                        let code = l4.learn(&[(0, &cue[light]), (1, &outcome[(got > 0) as usize])]);
                        cc.hold(&code);
                    } else if sep_on {
                        // (the critical period closed: layer 4 codes, it no longer learns;
                        // LATENT_SEP_OPEN=1 keeps it learning)
                        let code = sep.code(&codes[light * 2 + (got > 0) as usize], true);
                        cc.hold(&code);
                    } else {
                        cc.hold(&codes[light * 2 + (got > 0) as usize]);
                    }
                } else {
                    lc.step(&worth, &learned, &bits, got);
                }
                if (if kern { kh.resets } else if circuit { cc.resets() } else { lc.resets }) > before && detected.is_none() {
                    detected = Some(t);
                }
                // the active context learns the outcome (locking in)
                let c = if kern { kh.now } else if circuit { cc.now() } else { lc.now };
                while worth.len() <= c {
                    worth.push(vec![-1; BITS]);
                    learned.push(vec![0; BITS]);
                }
                for &b in &bits {
                    let (v, n) = (&mut worth[c][b], &mut learned[c][b]);
                    *v = if *v < 0 { got } else { *v + (got - *v) / (2 + *n as i32) };
                    *n = n.saturating_add(1);
                }
            }
            let resets = if kern { kh.resets } else if circuit { cc.resets() } else { lc.resets } - resets_before;
            let (now, contexts) = if kern { (kh.now, kh.contexts()) } else if circuit { (cc.now(), cc.contexts()) } else { (lc.now, lc.contexts) };
            let s = &mut per[ph];
            if ph == 0 {
                first_phase_false += resets;
            } else {
                match detected {
                    Some(t) => s.detect.push(t),
                    None => s.missed += 1,
                }
                s.false_alarms += resets.saturating_sub(1);
                if detected.is_some() {
                    match home[kind] {
                        Some(h) if h == now => s.back += 1,
                        _ if home[1 - kind] == Some(now) || (home[kind].is_some() && home[kind] != Some(now) && now < contexts - 1) => s.wrong += 1,
                        _ => s.new += 1,
                    }
                }
            }
            home[kind] = Some(now);
        }
    }
    println!(
        "latent alone: {}, policy {}, pay {p}, period {period}, {seeds} seeds; false silencings in the first phase {first_phase_false}",
        if kern { "kernel".to_string() } else if circuit { "circuit".to_string() } else { format!("attractor {scale:?}") },
        if greedy { "greedy" } else { "random" }
    );
    println!("swap  detected  median-trials  missed  back  new  wrong  false-alarms");
    for (ph, s) in per.iter().enumerate().skip(1) {
        let mut d = s.detect.clone();
        d.sort();
        let med = d.get(d.len() / 2).map_or("-".to_string(), |x| x.to_string());
        println!("{ph:>4}  {:>8}  {med:>13}  {:>6}  {:>4}  {:>3}  {:>5}  {:>12}", s.detect.len(), s.missed, s.back, s.new, s.wrong, s.false_alarms);
    }
}
