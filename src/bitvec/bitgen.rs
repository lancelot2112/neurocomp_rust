//! Bit generation utilities with controllable sparsity and clustering,
//! implemented directly on BitVector using bit-level utilities.
//!
//! - bits: total number of bits
//! - bit_sets: target number of 1s (sparsity)
//! - cluster in [0, 1]: 0 = i.i.d. Bernoulli, 1 = very sticky (long runs)
//! - FillMode::Bernoulli: approximate target (fast)
//! - FillMode::ExactK: adjust to exactly bit_sets after sampling

use rand::seq::SliceRandom;
use rand::Rng;

use crate::bitvec::BitVector;
//use crate::bitvec::bits::Bits; // assumes bits.rs exposes a Bits trait (get/set/clear)
//use crate::bitvec::bitcount::BitCount; // assumes bitcount.rs exposes count_ones()

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillMode {
    Bernoulli,
    ExactK,
}

#[derive(Clone, Debug)]
pub struct BitGenConfig {
    pub bits: usize,
    pub bit_sets: usize,
    pub cluster: f64, // [0,1]
    pub mode: FillMode,
}

impl BitGenConfig {
    pub fn new(bits: usize, bit_sets: usize) -> Self {
        Self {
            bits,
            bit_sets,
            cluster: 0.0,
            mode: FillMode::Bernoulli,
        }
    }
    pub fn with_cluster(mut self, cluster: f64) -> Self {
        self.cluster = cluster;
        self
    }
    pub fn exact(mut self) -> Self {
        self.mode = FillMode::ExactK;
        self
    }
}

pub fn generate(cfg: &BitGenConfig) -> BitVector {
    let mut rng = rand::thread_rng();
    generate_with_rng(cfg, &mut rng)
}

pub fn generate_with_rng<R: Rng + ?Sized>(cfg: &BitGenConfig, rng: &mut R) -> BitVector {
    let bits = cfg.bits;
    if bits == 0 {
        return BitVector::new(0, Some(0));
    }

    let k = cfg.bit_sets.min(bits);
    let p1 = (k as f64) / (bits as f64);
    let s = cfg.cluster.clamp(0.0, 0.999_999); // avoid singular transitions

    // Transition probs for 2-state Markov chain with stationary p1 and "stickiness" s:
    // P(1->1)=a, P(0->0)=b
    let a = s + (1.0 - s) * p1;
    let b = s + (1.0 - s) * (1.0 - p1);

    // Start with all zeros
    let mut bv = BitVector::new(bits, Some(0));

    // Initial bit ~ Bernoulli(p1)
    let mut state_one = rng.gen_bool(p1);
    if state_one {
        bv.bit_set(0);
    }

    // Emit remaining bits
    for i in 1..bits {
        let u = rng.r#gen::<f64>();
        if state_one {
            state_one = u < a; // stay 1 with prob a
        } else {
            state_one = !(u < (1.0 - b)); // stay 0 with prob b
        }
        if state_one {
            bv.bit_set(i);
        }
    }

    if matches!(cfg.mode, FillMode::ExactK) {
        exact_adjust(&mut bv, k, rng);
    }

    bv
}

// Adjust to exactly k ones by flipping random bits.
fn exact_adjust<R: Rng + ?Sized>(bv: &mut BitVector, k: usize, rng: &mut R) {
    let bits = bv.bit_len();
    let mut ones_now = bv.count_ones();
    if ones_now == k {
        return;
    }

    // Collect candidate indices (only what we need to flip).
    if ones_now > k {
        // Too many ones: randomly clear (ones_now - k) 1-bits.
        let mut idx_ones = Vec::with_capacity(ones_now);
        for i in 0..bits {
            if bv.bit_get(i) {
                idx_ones.push(i);
            }
        }
        let to_clear = ones_now - k;
        idx_ones.shuffle(rng);
        for &i in idx_ones.iter().take(to_clear) {
            bv.bit_clear(i);
        }
    } else {
        // Too few ones: randomly set (k - ones_now) 0-bits.
        let mut idx_zeros = Vec::with_capacity(bits - ones_now);
        for i in 0..bits {
            if !bv.bit_get(i) {
                idx_zeros.push(i);
            }
        }
        let to_set = k - ones_now;
        idx_zeros.shuffle(rng);
        for &i in idx_zeros.iter().take(to_set) {
            bv.bit_set(i);
        }
    }

    // Optional sanity: update local count (not strictly needed)
    ones_now = bv.count_ones();
    debug_assert_eq!(ones_now, k, "ExactK adjustment failed: got {}", ones_now);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn bernoulli_density_close() {
        let bits = 50_000;
        let k = 5_000; // 10%
        let cfg = BitGenConfig::new(bits, k).with_cluster(0.0);
        let mut rng = StdRng::seed_from_u64(123);
        let bv = generate_with_rng(&cfg, &mut rng);

        let frac = (bv.count_ones() as f64) / (bits as f64);
        assert!((frac - 0.10).abs() < 0.015, "frac={}", frac);
    }

    #[test]
    fn exact_mode_matches_k() {
        let bits = 8192 + 7;
        let k = 777;
        let cfg = BitGenConfig::new(bits, k).with_cluster(0.7).exact();
        let bv = generate(&cfg);
        assert_eq!(bv.count_ones(), k);
    }

    #[test]
    fn clustering_effect() {
        let bits = 40_000;
        let k = 8_000; // 20%
        let mut rng = StdRng::seed_from_u64(42);

        let bv_low = generate_with_rng(&BitGenConfig::new(bits, k).with_cluster(0.0), &mut rng);
        let bv_high = generate_with_rng(&BitGenConfig::new(bits, k).with_cluster(0.95), &mut rng);

        // Estimate P(1->1) transition
        let p11_low = p11(&bv_low);
        let p11_high = p11(&bv_high);
        assert!(p11_high > p11_low, "p11_high={} <= p11_low={}", p11_high, p11_low);
    }

    fn p11(bv: &BitVector) -> f64 {
        let mut t1 = 0usize;
        let mut t11 = 0usize;
        let mut prev = bv.bit_get(0);
        for i in 1..bv.bit_len() {
            let cur = bv.bit_get(i);
            if prev {
                t1 += 1;
                if cur { t11 += 1; }
            }
            prev = cur;
        }
        if t1 == 0 { 0.0 } else { (t11 as f64) / (t1 as f64) }
    }
}