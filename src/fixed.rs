//! Fixed-point and integer helpers: the model computes with bits and integers only.
//!
//! Fractions (probabilities, rates, thresholds) are `Q16` integers: 0..=65,536 stands
//! for 0..=1. A configuration value given as a float is converted once, at the boundary,
//! with [`q16`]. Every per-step computation then uses integer arithmetic: products are
//! shifted back down, ratios are formed as `num · ONE / den`, and random choices compare a
//! uniform integer draw with a `Q16` probability ([`chance`]).
//!
//! The library holds itself to this: a test (`tests::no_floats_in_per_step_code`) scans
//! `src/` and fails on any float outside test code and lines marked `// float:` (a
//! configuration conversion or a report-only readout).

use rand::Rng;

/// A fraction in 1/65,536 units: `ONE` = 1.0.
pub type Q16 = u32;

/// 1.0 in `Q16`.
pub const ONE: Q16 = 1 << 16;

/// Converts a configuration fraction to `Q16` (clamped to 0..=1). Configuration only.
pub fn q16(x: f64) -> Q16 { // float: config
    (x.clamp(0.0, 1.0) * ONE as f64).round() as Q16 // float: config
}

/// Converts a configuration value to `Q16` without the upper clamp (gains and ratios
/// may exceed 1; saturates at `u32::MAX`). Configuration only.
pub fn q16x(x: f64) -> Q16 { // float: config
    if x.is_infinite() && x > 0.0 { u32::MAX } else { (x.max(0.0) * ONE as f64).round().min(u32::MAX as f64) as Q16 } // float: config
}

/// The `Q16` value as a float, for reports only.
pub fn to_f32(q: Q16) -> f32 { // float: report
    q as f32 / ONE as f32 // float: report
}

/// True with probability `p` (in `Q16`): one uniform 64-bit draw compared with
/// `p · 2^48`. This consumes the generator exactly as a Bernoulli draw does (one `u64`,
/// none when `p` ≥ 1), so switching a float probability to its `Q16` value keeps the
/// random stream aligned.
pub fn chance<R: Rng + ?Sized>(rng: &mut R, p: Q16) -> bool {
    if p >= ONE {
        return true;
    }
    rng.next_u64() < (p as u64) << 48
}

/// `n · p`, rounded down (`p` in `Q16`).
pub fn mul_floor(n: u64, p: Q16) -> u64 {
    (n * p as u64) >> 16
}

/// `n · p`, rounded up (`p` in `Q16`).
pub fn mul_ceil(n: u64, p: Q16) -> u64 {
    (n * p as u64 + (ONE as u64 - 1)) >> 16
}

/// `num / den` in `Q16` (0 when `den` is 0), rounded to the nearest unit, so an exact
/// fraction such as 4/5 lands on the same value as the constant for 0.8.
pub fn ratio(num: u64, den: u64) -> Q16 {
    if den == 0 {
        0
    } else {
        (((num << 16) + den / 2) / den).min(u32::MAX as u64) as Q16
    }
}

/// `num / den`, rounded to the nearest integer (`den` > 0).
pub fn div_round(num: u64, den: u64) -> u64 {
    (num + den / 2) / den
}

/// `0.5^(1/h)` in `Q16`, for an integer-valued half-life `h` given in eighths
/// (`h8 = 8·h`), by bisection on the integer power. Used once per configuration.
pub fn half_life_survival(h8: u32) -> Q16 {
    // find p such that p^h ≈ 1/2, i.e. p^(h8) ≈ 2^-8, in Q16, by bisection
    let target = ONE as u64 >> 8; // 2^-8 in Q16
    let pow = |p: u64, e: u32| -> u64 {
        let mut r = ONE as u64;
        for _ in 0..e {
            r = (r * p) >> 16;
        }
        r
    };
    let (mut lo, mut hi) = (0u64, ONE as u64);
    for _ in 0..20 {
        let mid = (lo + hi) / 2;
        if pow(mid, h8.max(1)) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi as Q16
}

/// 2^(n/d) in `Q16` units of 1 (e.g. `exp2_frac(3, 8)` ≈ 1.297 · ONE), for 0 ≤ n < d,
/// by integer square roots of a fixed table: exact to about 1e-4.
pub fn exp2_frac(n: u32, d: u32) -> Q16 {
    // 2^(n/d) = (2^n)^(1/d): bisection on x^d = 2^n in Q16
    let target = (ONE as u64) << n.min(16);
    let pow = |x: u64, e: u32| -> u64 {
        let mut r = ONE as u64;
        for _ in 0..e {
            r = (r * x) >> 16;
        }
        r
    };
    let (mut lo, mut hi) = (ONE as u64, 2 * ONE as u64);
    for _ in 0..24 {
        let mid = (lo + hi) / 2;
        if pow(mid, d.max(1)) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi as Q16
}

/// ⌊log2 n⌋ for n ≥ 1 (0 for n = 0).
pub fn log2_floor(n: u64) -> u32 {
    63 - n.max(1).leading_zeros()
}

/// Integer square root (⌊√n⌋).
pub fn isqrt(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let mut x = 1u64 << ((64 - n.leading_zeros()) / 2 + 1);
    loop {
        let y = (x + n / x) / 2;
        if y >= x {
            return x;
        }
        x = y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn fixed_point_helpers() {
        assert_eq!(q16(0.5), ONE / 2);
        assert_eq!(mul_floor(10, q16(0.25)), 2);
        assert_eq!(mul_ceil(10, q16(0.25)), 3);
        assert_eq!(ratio(1, 4), ONE / 4);
        assert_eq!(isqrt(99), 9);
        assert_eq!(isqrt(100), 10);
        assert_eq!(log2_floor(1), 0);
        assert_eq!(log2_floor(1024), 10);
        // 0.5^(1/4) ≈ 0.8409
        let p = half_life_survival(32);
        assert!((p as i64 - 55109).abs() < 40, "{p}");
        // 2^(1/2) ≈ 1.4142
        let e = exp2_frac(1, 2);
        assert!((e as i64 - 92682).abs() < 40, "{e}");
        let mut rng = StdRng::seed_from_u64(1);
        let hits = (0..10_000).filter(|_| chance(&mut rng, q16(0.3))).count();
        assert!((2_800..3_200).contains(&hits), "{hits}");
    }

    /// The library computes with bits and integers: no float may appear in `src/`
    /// outside test code, the float reference store (`Ca3FloatMemory`), and lines marked
    /// `// float:` (a configuration conversion or a report-only readout).
    #[test]
    fn no_floats_in_per_step_code() {
        let mut bad = Vec::new();
        let mut stack = vec![std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/src"))];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.extension().map_or(true, |e| e != "rs") {
                    continue;
                }
                let text = std::fs::read_to_string(&path).unwrap();
                let mut in_float_store = false;
                for (n, line) in text.lines().enumerate() {
                    if line.trim_start().starts_with("#[cfg(test)]") {
                        break;
                    }
                    // the deliberate float reference store, kept for comparison
                    if line.contains("pub struct Ca3FloatMemory") || line.contains("impl Ca3FloatMemory") || line.contains("impl Autoassociative for Ca3FloatMemory") {
                        in_float_store = true;
                    }
                    if in_float_store {
                        if line.starts_with('}') {
                            in_float_store = false;
                        }
                        continue;
                    }
                    let code = line.split("//").next().unwrap_or("");
                    let has_float = ["f32", "f64", ".powf(", ".powi(", ".sqrt()", ".ln()", ".exp2()", ".log2()"].iter().any(|t| code.contains(t));
                    if has_float && !line.contains("// float:") {
                        bad.push(format!("{}:{}: {}", path.display(), n + 1, line.trim()));
                    }
                }
            }
        }
        assert!(bad.is_empty(), "floats in per-step code:\n{}", bad.join("\n"));
    }
}
