//! The Bayes module: how far to believe each source and each claim.
//!
//! Nothing is fully trusted, neither the world nor the network itself. Every piece of
//! knowledge is a *claim*: a value for a key, made by a source (a narrator, the network's
//! own words, one of its proposals). Claims on one key with different values conflict.
//!
//! - **A many-valued key is not a conflict.** If one source gives a key several values
//!   ("al's children"), the values are compatible; a conflict is different sources giving
//!   different values, each one value.
//! - **Trust and belief are estimated together** (truth discovery, after TruthFinder):
//!   - a value's belief is the summed trust of the sources that claim it;
//!   - a source's trust is the mean share of belief its claims win, over the keys at least
//!     two claims were made on (a lone claim says nothing about its source), with one win
//!     and one loss as a prior;
//!   - eight rounds settle both. Integers only (`Q16`).
//! - **The believed value** of a key is the one with the most belief.
//! - **Not yet a full posterior.** The belief is a trust-weighted vote, a linear stand-in
//!   for the Bayesian posterior, which would add each source's log-odds
//!   (log t / (1 − t)) instead of its trust; a source that is almost always right would
//!   then outweigh several weak ones. That is the natural next step.

use std::hash::Hash;

use crate::det::HashMap;
use crate::fixed::ONE;

pub struct Bayes<K> {
    claims: HashMap<K, Vec<(usize, u16)>>,
    trust: HashMap<u16, u32>,
    /// Learn the sources' trust (true), or count every source the same: a plain vote.
    pub use_trust: bool,
}

impl<K: Hash + Eq + Clone> Default for Bayes<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq + Clone> Bayes<K> {
    pub fn new() -> Self {
        Self { claims: HashMap::default(), trust: HashMap::default(), use_trust: true }
    }

    /// `source` claims `value` for `key`. Returns false if it had already.
    pub fn claim(&mut self, key: K, value: usize, source: u16) -> bool {
        let c = self.claims.entry(key).or_default();
        if c.contains(&(value, source)) {
            return false;
        }
        c.push((value, source));
        true
    }

    /// Estimate the sources' trust from every contested key (see the module doc).
    pub fn resolve(&mut self) {
        let keys: Vec<&Vec<(usize, u16)>> = self.claims.values().filter(|c| c.len() >= 2 && !many_valued(c)).collect();
        let mut trust: HashMap<u16, u32> = HashMap::default();
        for c in &keys {
            for &(_, s) in c.iter() {
                trust.insert(s, ONE / 2);
            }
        }
        for _ in 0..if self.use_trust { 8 } else { 0 } {
            let mut credit: HashMap<u16, (u64, u64)> = HashMap::default();
            for c in &keys {
                let total: u64 = c.iter().map(|x| trust[&x.1] as u64).sum::<u64>().max(1);
                for &(v, s) in c.iter() {
                    let belief: u64 = c.iter().filter(|x| x.0 == v).map(|x| trust[&x.1] as u64).sum();
                    let e = credit.entry(s).or_default();
                    e.0 += (belief << 16) / total;
                    e.1 += 1;
                }
            }
            for (s, (sum, n)) in credit {
                // prior: one claim won, one lost
                trust.insert(s, ((sum + ONE as u64 / 2) / (n + 1)) as u32);
            }
        }
        self.trust = trust;
    }

    /// The believed value of `key`: the one with the most summed trust (unknown sources
    /// count half; ties to the lower value). None if nothing was claimed, or the key is
    /// many-valued.
    pub fn believed(&self, key: &K) -> Option<usize> {
        let c = self.claims.get(key)?;
        if many_valued(c) {
            return None;
        }
        let mut best: Option<(u64, usize)> = None;
        for &(v, _) in c {
            let b: u64 = c.iter().filter(|x| x.0 == v).map(|x| self.trust(x.1) as u64).sum();
            if best.map_or(true, |(bb, bv)| b > bb || (b == bb && v < bv)) {
                best = Some((b, v));
            }
        }
        best.map(|x| x.1)
    }

    /// A source's trust (`Q16`; half for a source never contested).
    pub fn trust(&self, source: u16) -> u32 {
        self.trust.get(&source).copied().unwrap_or(ONE / 2)
    }

    /// The claims on `key`: (value, source, the value's share of all belief on the key).
    pub fn claims(&self, key: &K) -> Vec<(usize, u16, u32)> {
        let Some(c) = self.claims.get(key) else { return Vec::new() };
        let t = |s: u16| self.trust(s) as u64;
        let total: u64 = c.iter().map(|x| t(x.1)).sum::<u64>().max(1);
        c.iter().map(|&(v, s)| (v, s, ((c.iter().filter(|x| x.0 == v).map(|x| t(x.1)).sum::<u64>() << 16) / total) as u32)).collect()
    }

    /// The keys in conflict (different sources, different values; not many-valued).
    pub fn contested(&self) -> Vec<K> {
        self.claims.iter().filter(|(_, c)| !many_valued(c) && c.iter().any(|x| x.0 != c[0].0)).map(|(k, _)| k.clone()).collect()
    }
}

/// A key one source gave several values: a many-valued relation, not a conflict.
pub fn many_valued(c: &[(usize, u16)]) -> bool {
    c.iter().any(|a| c.iter().any(|b| a.1 == b.1 && a.0 != b.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reliable_source_breaks_a_tie_with_an_unreliable_one() {
        let mut b: Bayes<u32> = Bayes::new();
        // sources 1 and 2 agree on keys 0..10; source 3 contradicts them on all of them
        for k in 0..10 {
            b.claim(k, 1, 1);
            b.claim(k, 1, 2);
            b.claim(k, 2, 3);
        }
        // key 99: only source 1 and source 3, one each
        b.claim(99, 5, 1);
        b.claim(99, 6, 3);
        b.resolve();
        assert!(b.trust(3) < b.trust(1));
        assert_eq!(b.believed(&99), Some(5));
        // a plain vote cannot tell (the tie goes to the lower value either way, so check
        // with the liar on the lower value)
        let mut v: Bayes<u32> = Bayes::new();
        v.use_trust = false;
        for k in 0..10 {
            v.claim(k, 1, 1);
            v.claim(k, 1, 2);
            v.claim(k, 2, 3);
        }
        v.claim(99, 6, 1);
        v.claim(99, 5, 3);
        v.resolve();
        assert_eq!(v.believed(&99), Some(5), "the vote follows the tie rule, not the source");
        b.claim(98, 6, 1);
        b.claim(98, 5, 3);
        b.resolve();
        assert_eq!(b.believed(&98), Some(6), "trust follows the reliable source");
    }

    #[test]
    fn one_source_with_several_values_is_not_a_conflict() {
        let mut b: Bayes<u32> = Bayes::new();
        b.claim(0, 1, 1);
        b.claim(0, 2, 1);
        b.resolve();
        assert_eq!(b.believed(&0), None);
        assert!(b.contested().is_empty());
    }
}
