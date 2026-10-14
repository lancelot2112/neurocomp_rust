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
    let text = std::fs::read(&path).expect("corpus file");
    let text = &text[..n.min(text.len())];
    println!("bytes: {path}, {} bytes, threshold {q}/4, held-down learn {}", text.len(), if held { "on" } else { "off" });

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
        let fired = class.process(&w, &mut p, 0, 0, true, &mut rng);
        let winner = fired.first().copied();
        let pred = winner.map(|_| (p.as_words()[0] & 0xFF) as u8);
        hit += (pred == Some(b)) as u32;
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
        if pred != Some(b) && t > 0 && b != 0 {
            let s = winner.map_or(1, |k| (scale[k] * 2).min(MAX_SCALE)).min(t);
            if s >= 1 {
                let (care, pat) = window(&w, s);
                let lines = care.count_ones();
                let thr = (q * lines).div_ceil(4).max(1);
                let mut k = SimpleKernel::with_pattern(care, pat, 0, BitVector::from_words(vec![b as u64]), 0, thr, KernelOp::Or);
                k.learn_held = held;
                class.add(k);
                scale.push(s);
            }
        }
        slide(&mut w, b);
        seen += 1;
        if (t + 1) % chunk == 0 || t + 1 == text.len() {
            let pc = |x: u32| 100.0 * x as f64 / seen as f64;
            println!("{:>7}  {:>8}  {:>8.1}  {:>9.1}  {:>8.1}  {:>7.1}", t + 1, class.len(), pc(base1), pc(base2), pc(fired_any), pc(hit));
            hit = 0; base1 = 0; base2 = 0; seen = 0; fired_any = 0;
        }
    }
}
