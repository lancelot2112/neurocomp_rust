//! Deterministic hash maps and sets.
//!
//! `std::collections::HashMap` seeds its hasher randomly per process, so iteration order
//! changes from run to run. Where that order decides a result (kernel matching, sleep
//! merging and pruning, tallies), two runs at the same seed would differ. These aliases use
//! a fixed-key hasher, so a run is a function of its seed alone. Construct them with
//! `default()` (or `collect()`), not `new()`.

use std::collections::hash_map::DefaultHasher;
use std::hash::BuildHasherDefault;

/// The fixed-key hasher (SipHash with zero keys).
pub type FixedState = BuildHasherDefault<DefaultHasher>;
/// A `HashMap` with deterministic iteration order for a given insertion history.
pub type HashMap<K, V> = std::collections::HashMap<K, V, FixedState>;
/// A `HashSet` with deterministic iteration order for a given insertion history.
pub type HashSet<T> = std::collections::HashSet<T, FixedState>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iteration_order_is_fixed() {
        let order = || -> Vec<u32> {
            let m: HashMap<u32, u32> = (0..1000).map(|i| (i * 7919 % 10007, i)).collect();
            m.keys().copied().collect()
        };
        assert_eq!(order(), order());
    }
}
