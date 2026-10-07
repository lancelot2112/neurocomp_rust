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
//! - **Belief is isolated behind one rule** (`BeliefRule`), so it can be swapped:
//!   - `Full`: everything claimed is believed (belief 1), every source fully credible;
//!   - `Vote`: a value's share of the claims; sources not modelled (credibility 1);
//!   - `Graded`: a value's share of the trust (a trust-weighted vote);
//!   - `Posterior`: Bayesian. Each source is right with probability t (its trust); a
//!     value's odds are the product of its sources' odds t / (1 − t), and its belief is
//!     odds / (1 + Σ odds over the claimed values), the 1 standing for "none of these".
//!     A single claim is believed as far as its source is trusted; agreeing sources
//!     multiply. Computed exactly in integers: multiplied through by Π (1 − t), it is a
//!     ratio of products of `Q16` numbers.
//! - **The believed value** of a key is the one with the most belief.

use std::hash::Hash;

use crate::det::HashMap;
use crate::fixed::ONE;

/// How belief is computed from claims (see the module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeliefRule {
    Full,
    Vote,
    Graded,
    Posterior,
}

impl BeliefRule {
    /// From a name: "full", "vote", "graded", "posterior" (default: graded).
    pub fn named(name: &str) -> Self {
        match name {
            "full" => Self::Full,
            "vote" => Self::Vote,
            "posterior" => Self::Posterior,
            _ => Self::Graded,
        }
    }
}

pub struct Bayes<K> {
    claims: HashMap<K, Vec<(usize, u16)>>,
    trust: HashMap<u16, u32>,
    pub rule: BeliefRule,
}

impl<K: Hash + Eq + Clone> Default for Bayes<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq + Clone> Bayes<K> {
    pub fn new() -> Self {
        Self { claims: HashMap::default(), trust: HashMap::default(), rule: BeliefRule::Graded }
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
        let learn = matches!(self.rule, BeliefRule::Graded | BeliefRule::Posterior);
        for _ in 0..if learn { 8 } else { 0 } {
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

    /// The believed value of `key`: the one with the most belief (ties to the lower
    /// value). None if nothing was claimed, or the key is many-valued.
    pub fn believed(&self, key: &K) -> Option<usize> {
        let c = self.claims.get(key)?;
        if many_valued(c) {
            return None;
        }
        let mut best: Option<(u32, usize)> = None;
        for &(v, _) in c {
            let b = self.belief(key, v);
            if best.map_or(true, |(bb, bv)| b > bb || (b == bb && v < bv)) {
                best = Some((b, v));
            }
        }
        best.map(|x| x.1)
    }

    /// How far `value` is believed for `key` (`Q16`), by the rule. 0 if not claimed; for a
    /// many-valued key, as if each value were claimed alone.
    pub fn belief(&self, key: &K, value: usize) -> u32 {
        let Some(c) = self.claims.get(key) else { return 0 };
        if !c.iter().any(|x| x.0 == value) {
            return 0;
        }
        let c: Vec<(usize, u16)> = if many_valued(c) { c.iter().copied().filter(|x| x.0 == value).collect() } else { c.clone() };
        match self.rule {
            BeliefRule::Full => ONE,
            BeliefRule::Vote => ((c.iter().filter(|x| x.0 == value).count() as u64 * ONE as u64) / c.len() as u64) as u32,
            BeliefRule::Graded => {
                let t = |s: u16| self.trust(s) as u64;
                let total: u64 = c.iter().map(|x| t(x.1)).sum::<u64>().max(1);
                ((c.iter().filter(|x| x.0 == value).map(|x| t(x.1)).sum::<u64>() << 16) / total) as u32
            }
            BeliefRule::Posterior => {
                // each claim's source right with probability t: multiplied through by
                // Π (1 − t), value u's odds become Π_{s says u} t · Π_{s says other} (1 − t)
                let t = |s: u16| self.trust(s).clamp(1, ONE - 1) as u64;
                let mut values: Vec<usize> = c.iter().map(|x| x.0).collect();
                values.sort_unstable();
                values.dedup();
                let weight = |u: usize| c.iter().fold(ONE as u64, |acc, &(v, s)| (acc * if v == u { t(s) } else { ONE as u64 - t(s) }) >> 16);
                let none = c.iter().fold(ONE as u64, |acc, &(_, s)| (acc * (ONE as u64 - t(s))) >> 16);
                let total: u64 = none + values.iter().map(|&u| weight(u)).sum::<u64>();
                ((weight(value) << 16) / total.max(1)) as u32
            }
        }
    }

    /// How credible a source is (`Q16`): its trust where the rule models trust, else fully.
    pub fn credibility(&self, source: u16) -> u32 {
        match self.rule {
            BeliefRule::Full | BeliefRule::Vote => ONE,
            BeliefRule::Graded | BeliefRule::Posterior => self.trust(source),
        }
    }

    /// A source's trust (`Q16`; half for a source never contested).
    pub fn trust(&self, source: u16) -> u32 {
        self.trust.get(&source).copied().unwrap_or(ONE / 2)
    }

    /// The claims on `key`: (value, source, the value's belief).
    pub fn claims(&self, key: &K) -> Vec<(usize, u16, u32)> {
        let Some(c) = self.claims.get(key) else { return Vec::new() };
        c.iter().map(|&(v, s)| (v, s, self.belief(key, v))).collect()
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
        v.rule = BeliefRule::Vote;
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
    fn the_posterior_multiplies_agreeing_sources_and_a_single_claim_is_its_trust() {
        let mut b: Bayes<u32> = Bayes::new();
        b.rule = BeliefRule::Posterior;
        // build trust: sources 1 and 2 agree on keys 0..20, source 3 contradicts on 0..10
        for k in 0..20 {
            b.claim(k, 1, 1);
            b.claim(k, 1, 2);
        }
        for k in 0..10 {
            b.claim(k, 2, 3);
        }
        b.claim(50, 7, 1); // a single claim
        b.claim(51, 7, 1); // two agreeing sources
        b.claim(51, 7, 2);
        b.resolve();
        let t1 = b.trust(1);
        let single = b.belief(&50, 7);
        assert!(single.abs_diff(t1) <= ONE / 100, "single claim {single} vs trust {t1}");
        assert!(b.belief(&51, 7) > single, "agreeing sources raise belief");
        // graded gives a lone claim full belief; the posterior only its source's trust
        let mut g: Bayes<u32> = Bayes::new();
        g.claim(50, 7, 1);
        g.resolve();
        assert_eq!(g.belief(&50, 7), ONE);
        // full trusts everything
        let mut f: Bayes<u32> = Bayes::new();
        f.rule = BeliefRule::Full;
        f.claim(9, 1, 1);
        f.claim(9, 2, 3);
        f.resolve();
        assert_eq!((f.belief(&9, 1), f.belief(&9, 2)), (ONE, ONE));
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
