//! Byte prediction on raw text with two-mask kernels at several scales over a sliding window.
//!
//! - **Window** `W`: 8192 bits, the last 1024 bytes, newest in byte 0. Each step it shifts by
//!   one byte and the new byte is written into byte 0.
//! - **Kernels**: a care mask over the last `s` bytes (`s` = 1, 2, 4, ... 1024: an 8-bit, 16-bit,
//!   ... 8192-bit window) and the pattern it expects there, ones and zeros. It fires when its
//!   agreeing lines meet `BYTES_Q` quarters of its mask. Firing, it writes its byte (the one that
//!   followed when it was born) into the prediction byte. The best match goes first, and kernels
//!   whose masks overlap are held down (so a longer matching context wins: back-off).
//! - **Births** (a harness rule for now, labelled): on a wrong or missing prediction a kernel is
//!   born on the current window at twice the scale of the kernel that predicted (one byte if
//!   none did), writing the byte that came.
//!
//! Baselines: the commonest byte; a direct lookup of the longest exactly matching context among
//! the same scales (what exact kernels should reproduce).
//!
//! `BYTES_FILE` (default data/alice.txt), `BYTES_N` (bytes read), `BYTES_Q` (1-4, default 4:
//! exact), `BYTES_HELD=1` (held-down kernels learn anti-Hebbian).
//!
//! `BYTES_SPARSE=1`: a sparsifying first layer. Layer 1 holds exact 8-bit kernels on the newest
//! byte, born as each new byte value appears, each writing its own random code of `BYTES_K`
//! lines (default 8) in a 64-bit slot. Those codes slide through a sparse window `S` (one word a
//! byte, 128 bytes), and the predicting kernels read `S` instead of the raw window: they count
//! the lines that are on, at scales of 1, 2, 4, ... 128 slots, firing at `BYTES_Q` quarters.
//!
//! `BYTES_CONFIRM=1`: the third mask. Every kernel that matched (fired or held down) checks its
//! own prediction: confirmed, it writes
//! its confirmation line into the confirmation vector and counts up; else it counts down. A wrong
//! winner also gets a sibling at its own scale writing the byte that came (as well as the kernel
//! one scale up), so a context followed by several bytes holds one kernel for each, and the
//! most often confirmed wins among equal matches.

use std::collections::HashMap;

use neurocomp::bitvec::BitVector;
use neurocomp::kernel::{KernelClass, KernelOp, SimpleKernel};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

const BITS: usize = 8192;
const MAX_SCALE: usize = 1024;

fn slide(w: &mut BitVector, byte: u8) {
    let words = w.as_words_mut();
    let mut carry = byte as u64;
    for x in words.iter_mut() {
        let out = *x >> 56;
        *x = (*x << 8) | carry;
        carry = out;
    }
}

/// The care mask for the last `s` bytes, and the pattern now in that window.
fn window(w: &BitVector, s: usize) -> (BitVector, BitVector) {
    let nwords = (8 * s).div_ceil(64);
    let mut care = vec![u64::MAX; nwords];
    if 8 * s < 64 {
        care[0] = (1u64 << (8 * s)) - 1;
    }
    let pat: Vec<u64> = w.as_words()[..nwords].iter().zip(&care).map(|(a, m)| a & m).collect();
    (BitVector::from_words(care), BitVector::from_words(pat))
}

fn main() {
    let path = std::env::var("BYTES_FILE").unwrap_or_else(|_| "data/alice.txt".into());
    let n: usize = env("BYTES_N", 20000);
    let q: usize = env("BYTES_Q", 4);
    let held = std::env::var("BYTES_HELD").is_ok();
    let confirm = std::env::var("BYTES_CONFIRM").is_ok();
    // the match vector: each kernel's small code when it matches, its big code (small plus
    // extra lines) when its prediction came true
    let mut confirmations = BitVector::new(BITS, Some(0));
    let sparse = std::env::var("BYTES_SPARSE").is_ok();
    let fuzzy_depth = std::env::var("BYTES_FUZZYDEPTH").is_ok();
    // (`BYTES_EXACTCONFIRM=1`: only kernels that matched exactly check their prediction)
    let exact_confirm = std::env::var("BYTES_EXACTCONFIRM").is_ok();
    let or_slot = std::env::var("BYTES_ORSLOT").is_ok();
    let no_hebb = std::env::var("BYTES_NOHEBB").is_ok();
    let (mut wins, mut wins_right, mut fuzzy_scale) = ([0u32; 2], [0u32; 2], [0u32; 12]);
    let kbits: usize = env("BYTES_K", 8);
    let mut layer1: KernelClass<SimpleKernel> = KernelClass::with_kernels(Vec::new());
    let mut sw = BitVector::new(BITS, Some(0));
    let max_scale = if sparse { BITS / 64 } else { MAX_SCALE };
    let text = std::fs::read(&path).expect("corpus file");
    let text = &text[..n.min(text.len())];
    println!("bytes: {path}, {} bytes, threshold {q}/4, held-down learn {}, confirmation {}, {}", text.len(), if held { "on" } else { "off" }, if confirm { "on" } else { "off" }, if sparse { format!("sparse first layer ({kbits} of 64 lines a byte)") } else { "raw window".into() });

    let mut rng = StdRng::seed_from_u64(1);
    let mut class: KernelClass<SimpleKernel> = KernelClass::with_kernels(Vec::new());
    let mut scale: Vec<usize> = Vec::new();
    let mut w = BitVector::new(BITS, Some(0));
    // baselines
    let mut freq = [0u32; 256];
    let mut ctx: HashMap<(usize, Vec<u8>), [u32; 256]> = HashMap::new();
    let chunk = (text.len() / 10).max(1);
    let (mut hit, mut base1, mut base2, mut seen) = (0u32, 0u32, 0u32, 0u32);
    let mut fired_any = 0u32;
    println!("{:>7}  {:>8}  {:>8}  {:>9}  {:>8}  {:>7}", "bytes", "kernels", "common%", "longest%", "fired%", "right%");
    for (t, &b) in text.iter().enumerate() {
        // the kernels predict the next byte from the window
        let mut p = BitVector::new(64, Some(0));
        let input = if sparse { &sw } else { &w };
        let fired = class.process(input, &mut p, 0, 0, true, &mut rng);
        let winner = fired.first().copied();
        // the winner writes the slot: its own byte (`BYTES_ORSLOT=1`: the OR of every kernel that
        // fired, as the prediction vector holds it)
        let pred = winner.map(|k| if or_slot { (p.as_words()[0] & 0xFF) as u8 } else { (class.kernels()[k].output_mask.as_words()[0] & 0xFF) as u8 });
        hit += (pred == Some(b)) as u32;
        if let Some(k) = winner {
            let kk = &class.kernels()[k];
            let exact = kk.agreement(input) == kk.input_mask.count_ones();
            let i = exact as usize;
            wins[i] += 1;
            wins_right[i] += (pred == Some(b)) as u32;
            if !exact { fuzzy_scale[scale[k].trailing_zeros() as usize] += 1; }
        }
        if confirm {
            // every kernel that matched (fired or held down) sees whether its own prediction came
            // true: confirmed, it writes its third mask and counts up; else it counts down
            confirmations.bit_clear_all();
            for k in class.kernels_mut().iter_mut() {
                let a = k.agreement(input);
                if a >= k.threshold && (!exact_confirm || a == k.input_mask.count_ones()) {
                    let right = (k.output_mask.as_words()[0] & 0xFF) as u8 == b;
                    k.write_match(&mut confirmations);
                    k.confirm(right, Some(&mut confirmations));
                }
            }
        }
        fired_any += winner.is_some() as u32;
        // baselines: the commonest byte so far; the longest context seen before (its commonest next)
        let common = (0..256).max_by_key(|&c| (freq[c], 255 - c)).unwrap() as u8;
        base1 += (common == b && t > 0) as u32;
        let mut longest: Option<u8> = None;
        let mut s = 1;
        while s <= MAX_SCALE.min(t) {
            let key = (s, text[t - s..t].to_vec());
            if let Some(f) = ctx.get(&key) {
                longest = Some((0..256).max_by_key(|&c| (f[c], 255 - c)).unwrap() as u8);
            } else {
                break;
            }
            s *= 2;
        }
        base2 += (longest == Some(b)) as u32;
        freq[b as usize] += 1;
        let mut s = 1;
        while s <= MAX_SCALE.min(t) {
            ctx.entry((s, text[t - s..t].to_vec())).or_insert([0; 256])[b as usize] += 1;
            s *= 2;
        }
        // births on a wrong or missing prediction (harness rule): one scale up from the winner
        // the next scale up from the widest kernel that matched exactly (a fuzzy match is not
        // context known; `BYTES_FUZZYDEPTH=1`: from the winner, fuzzy or not)
        let exact_scale = fired.iter().filter(|&&f| { let k = &class.kernels()[f]; k.agreement(input) == k.input_mask.count_ones() }).map(|&f| scale[f]).max();
        let from = if fuzzy_depth { winner.map(|k| scale[k]) } else { exact_scale };
        let mut scales: Vec<usize> = vec![from.map_or(1, |s| (s * 2).min(max_scale)).min(t)];
        if confirm {
            if let Some(k) = winner {
                // a sibling at the winner's own scale for the byte that came, unless one exists
                let s0 = scale[k];
                let exists = fired.iter().any(|&f| {
                    let k = &class.kernels()[f];
                    scale[f] == s0 && (k.output_mask.as_words()[0] & 0xFF) as u8 == b && k.agreement(input) == k.input_mask.count_ones()
                });
                if !exists {
                    scales.push(s0);
                }
            }
        }
        if pred != Some(b) && t > 0 && b != 0 {
          for s in scales {
            if s >= 1 {
                let mut k = if sparse {
                    // a sparse kernel: the lines on in the last s slots
                    let care = BitVector::from_words(sw.as_words()[..s].to_vec());
                    let thr = (q * care.count_ones()).div_ceil(4).max(1);
                    SimpleKernel::new(care, 0, BitVector::from_words(vec![b as u64]), 0, thr, KernelOp::Or)
                } else {
                    let (care, pat) = window(&w, s);
                    let lines = care.count_ones();
                    let thr = (q * lines).div_ceil(4).max(1);
                    SimpleKernel::with_pattern(care, pat, 0, BitVector::from_words(vec![b as u64]), 0, thr, KernelOp::Or)
                };
                k.learn_held = held;
                // (`BYTES_NOHEBB=1`: no Hebbian move when it fires)
                k.learn_fire = !no_hebb;
                // its nested match code: 4 lines when it matches, 4 more when it was right
                let (small, extra) = SimpleKernel::nested_code(BITS, 4, 4, &mut rng);
                k.match_code = Some(small);
                k.confirm_mask = Some(extra);
                class.add(k);
                scale.push(s);
            }
          }
        }
        slide(&mut w, b);
        if sparse {
            // layer 1 codes the newest byte: an exact 8-bit kernel per byte value, born with its
            // own random code when none fires
            let newest = BitVector::from_words(vec![b as u64]);
            let mut code = BitVector::new(64, Some(0));
            let f = layer1.process(&newest, &mut code, 0, 0, true, &mut rng);
            if f.is_empty() {
                use rand::seq::SliceRandom;
                let lines: Vec<usize> = (0..64).collect();
                let pick: Vec<usize> = lines.choose_multiple(&mut rng, kbits).copied().collect();
                let out = BitVector::from_bits(&pick, 64);
                code = out.clone();
                let mut k = SimpleKernel::with_pattern(BitVector::from_words(vec![0xFF]), newest.clone(), 0, out, 0, 8, KernelOp::Or);
                k.learn_held = false;
                layer1.add(k);
            }
            let words = sw.as_words_mut();
            for i in (1..words.len()).rev() {
                words[i] = words[i - 1];
            }
            words[0] = code.as_words()[0];
        }
        seen += 1;
        if (t + 1) % chunk == 0 || t + 1 == text.len() {
            let pc = |x: u32| 100.0 * x as f64 / seen as f64;
            println!("{:>7}  {:>8}  {:>8.1}  {:>9.1}  {:>8.1}  {:>7.1}", t + 1, class.len(), pc(base1), pc(base2), pc(fired_any), pc(hit));
            hit = 0; base1 = 0; base2 = 0; seen = 0; fired_any = 0;
        }
    }
    println!("winners: exact {} (right {}), fuzzy {} (right {}); fuzzy winners by scale (1, 2, 4, ...): {:?}", wins[1], wins_right[1], wins[0], wins_right[0], fuzzy_scale);
}
