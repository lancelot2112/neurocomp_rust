//! Typed relations, navigable by relation code: a cortical semantic store whose entries are
//! (entity, relation, direction) → filler, learned from plain sentences.
//!
//! - **Relations are learned, not given.** A fact's *neighbours* are the facts read of
//!   the same length that agree with it in all but at most two positions ("lucy is a
//!   jones" ~ "tom is a smith"). The positions where most neighbours agree are its
//!   *frame* (what every fact of the kind shares: "_ 's father is _"); the rest, in reading
//!   order, are its *fillers* (the entities: "tom", "bob"). A relation is a frame: its
//!   words at their positions; its code is its index. Word frequency does not decide it:
//!   a family name can be frequent everywhere and still be a filler here.
//! - **Binding is a permutation.** For fillers i ≠ j of a fact of relation r, the key is
//!   filler i's code rotated by an offset hashed from (r, i, j), and the value is filler
//!   j's code. "tom 's father is bob" stores tom ↦(father, 0→1) bob and bob ↦(father,
//!   1→0) tom. Relations of one word that fill the same position do not collide: tom's
//!   father and tom's mother are different rotations of tom.
//! - **The store is a predictive kernel class**, trained by replay (`consolidate`), like
//!   the semantic store: facts are buffered while read and replayed `reps` times.
//! - **Navigation:** `ask(tom, father, 0, 1)` gives bob; `follow(tom, &[father, father])`
//!   gives bob's father. The relation for a query is found from its words
//!   (`relation_for(["father"])`): the frame sharing the most of them.

use rand::Rng;

use crate::bitvec::BitVector;
use crate::det::HashMap;
use crate::kernel::{GrowthConfig, KernelClass, SimpleKernel};

pub struct RelationStore {
    bits: usize,
    store: KernelClass<SimpleKernel>,
    /// Distinct facts read, and their buckets: (length, two positions left out, the other
    /// words) → facts. Facts in one bucket agree in all but those two positions.
    facts: Vec<Vec<usize>>,
    seen: HashMap<Vec<usize>, usize>,
    buckets: HashMap<(usize, usize, usize, u64), Vec<usize>>,
    /// Facts read since the last consolidation.
    buffer: Vec<Vec<usize>>,
    /// Learned relations: each a frame (its length and its (position, word)s).
    frames: Vec<(usize, Vec<(usize, usize)>)>,
    /// Per entity: the (relation, from, to) keys stored with it.
    links: HashMap<usize, Vec<(usize, usize, usize)>>,
    replays: usize,
}

/// At most this many neighbours are compared per fact.
const MAX_NEIGHBOURS: usize = 256;

impl RelationStore {
    pub fn new(bits: usize) -> Self {
        Self {
            bits,
            store: KernelClass::predictive(GrowthConfig {
                max_kernels: 20_000,
                frame_words: bits / 64,
                max_frames: 1,
                sample_bits: 16,
                match_fraction: 0.8,
                surprise_fraction: 0.5,
                generalize: None,
                generalize_after: 1,
            }),
            facts: Vec::new(),
            seen: HashMap::default(),
            buckets: HashMap::default(),
            buffer: Vec::new(),
            frames: Vec::new(),
            links: HashMap::default(),
            replays: 0,
        }
    }

    fn bucket_keys(fact: &[usize]) -> Vec<(usize, usize, usize, u64)> {
        let n = fact.len();
        let mut keys = Vec::new();
        for i in 0..n {
            for j in i..n {
                let h = (0..n).filter(|&p| p != i && p != j).fold(0xcbf2_9ce4_8422_2325u64, |h, p| (h ^ ((p as u64) << 32 | fact[p] as u64)).wrapping_mul(0x100_0000_01b3));
                keys.push((n, i, j, h));
            }
        }
        keys
    }

    /// Read a fact (word ids in order): keep it for the next replay, and among the facts
    /// whose shape it is compared with.
    pub fn observe(&mut self, fact: &[usize]) {
        if fact.len() < 3 {
            return;
        }
        if !self.seen.contains_key(fact) {
            let id = self.facts.len();
            self.seen.insert(fact.to_vec(), id);
            for k in Self::bucket_keys(fact) {
                self.buckets.entry(k).or_default().push(id);
            }
            self.facts.push(fact.to_vec());
        }
        self.buffer.push(fact.to_vec());
    }

    /// A fact's (frame, fillers in order), by the facts read so far; None unless it has
    /// at least two neighbours, a frame and at least two fillers. Frame positions are those
    /// where more than half of its neighbours (same length, all but two positions equal)
    /// have the same word.
    pub fn parse(&self, fact: &[usize]) -> Option<((usize, Vec<(usize, usize)>), Vec<usize>)> {
        let me = self.seen.get(fact).copied();
        let mut nb: Vec<usize> = Vec::new();
        for k in Self::bucket_keys(fact) {
            if let Some(b) = self.buckets.get(&k) {
                nb.extend(b.iter().copied().filter(|&f| Some(f) != me && self.facts[f].as_slice() != fact));
            }
        }
        nb.sort_unstable();
        nb.dedup();
        nb.truncate(MAX_NEIGHBOURS);
        if nb.len() < 2 {
            return None;
        }
        let (mut frame, mut fillers) = (Vec::new(), Vec::new());
        for (p, &w) in fact.iter().enumerate() {
            let agree = nb.iter().filter(|&&f| self.facts[f][p] == w).count();
            if agree * 2 > nb.len() {
                frame.push((p, w));
            } else if !fillers.contains(&w) {
                fillers.push(w);
            }
        }
        (!frame.is_empty() && fillers.len() >= 2).then_some(((fact.len(), frame), fillers))
    }

    fn relation_of(&mut self, frame: (usize, Vec<(usize, usize)>)) -> usize {
        match self.frames.iter().position(|f| *f == frame) {
            Some(r) => r,
            None => {
                self.frames.push(frame);
                self.frames.len() - 1
            }
        }
    }

    /// The binding offset of relation `r` from filler position `from` to `to`.
    fn offset(&self, r: usize, from: usize, to: usize) -> usize {
        let mut h = (r as u64) << 16 ^ (from as u64) << 8 ^ to as u64 ^ 0x9E37_79B9_7F4A_7C15;
        h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 31;
        1 + (h as usize) % (self.bits - 1)
    }

    fn key(&self, code: &BitVector, r: usize, from: usize, to: usize) -> BitVector {
        let mut k = code.clone();
        k.rotl_mut(self.offset(r, from, to));
        k
    }

    /// Sleep: the facts read since the last call are parsed (by the facts read so far) and
    /// replayed `reps` times, shuffled; every ordered pair of fillers is a binding.
    /// Returns the facts that parsed as relations.
    pub fn consolidate<R: Rng>(&mut self, codes: &[BitVector], reps: usize, rng: &mut R) -> usize {
        let facts = std::mem::take(&mut self.buffer);
        let mut pairs: Vec<(BitVector, usize, BitVector)> = Vec::new(); // (key, entity, value)
        let mut parsed = 0;
        for fact in &facts {
            let Some((frame, fillers)) = self.parse(fact) else { continue };
            parsed += 1;
            let r = self.relation_of(frame);
            for i in 0..fillers.len() {
                for j in 0..fillers.len() {
                    if i != j {
                        pairs.push((self.key(&codes[fillers[i]], r, i, j), fillers[i], codes[fillers[j]].clone()));
                        let l = self.links.entry(fillers[i]).or_default();
                        if !l.contains(&(r, i, j)) {
                            l.push((r, i, j));
                        }
                    }
                }
            }
        }
        let mut order: Vec<usize> = (0..pairs.len()).collect();
        for _ in 0..reps {
            for i in (1..order.len()).rev() {
                order.swap(i, rng.gen_range(0..=i));
            }
            for &i in &order {
                let (key, _, value) = &pairs[i];
                let mut out = BitVector::new(self.bits, Some(0));
                self.store.process_predictive(key, &mut out);
                self.store.feedback(key, value, rng);
                self.replays += 1;
            }
        }
        parsed
    }

    /// The relation whose frame shares the most of `words` (fewest other words on a tie).
    pub fn relation_for(&self, words: &[usize]) -> Option<usize> {
        (0..self.frames.len())
            .map(|r| (self.frames[r].1.iter().filter(|w| words.contains(&w.1)).count(), r))
            .filter(|x| x.0 > 0)
            .max_by_key(|&(shared, r)| (shared, std::cmp::Reverse(self.frames[r].1.len())))
            .map(|x| x.1)
    }

    /// The store's answer, as a code, for `entity` through relation `r` from filler
    /// position `from` to `to`.
    pub fn ask_code(&self, codes: &[BitVector], entity: usize, r: usize, from: usize, to: usize) -> Option<BitVector> {
        self.store.peek(&self.key(&codes[entity], r, from, to))
    }

    /// As `ask_code`, read out as the word whose code it holds most of (at least half).
    pub fn ask(&self, codes: &[BitVector], entity: usize, r: usize, from: usize, to: usize) -> Option<usize> {
        let out = self.ask_code(codes, entity, r, from, to)?;
        decode(codes, &out)
    }

    /// Navigate a path of relations, each from the first filler to the second ("tom" →
    /// father → father: tom's grandfather). None where a step has no answer.
    pub fn follow(&self, codes: &[BitVector], entity: usize, path: &[usize]) -> Option<usize> {
        path.iter().try_fold(entity, |e, &r| self.ask(codes, e, r, 0, 1))
    }

    /// Everything stored about `entity`: (relation, from, to, answer).
    pub fn about(&self, codes: &[BitVector], entity: usize) -> Vec<(usize, usize, usize, usize)> {
        self.links
            .get(&entity)
            .map(|l| l.iter().filter_map(|&(r, i, j)| self.ask(codes, entity, r, i, j).map(|a| (r, i, j, a))).collect())
            .unwrap_or_default()
    }

    /// The relations' frames (their words in order), by relation code.
    pub fn frames(&self) -> Vec<Vec<usize>> {
        self.frames.iter().map(|f| f.1.iter().map(|x| x.1).collect()).collect()
    }

    /// A relation's frame as a template: its length and its (position, word)s.
    pub fn template(&self, r: usize) -> &(usize, Vec<(usize, usize)>) {
        &self.frames[r]
    }

    /// (kernels, replays)
    pub fn stats(&self) -> (usize, usize) {
        (self.store.len(), self.replays)
    }
}

fn decode(codes: &[BitVector], out: &BitVector) -> Option<usize> {
    let ov = |c: &BitVector| c.as_words().iter().zip(out.as_words()).map(|(a, b)| (a & b).count_ones()).sum::<u32>();
    (0..codes.len()).map(|w| (ov(&codes[w]), w)).filter(|&(o, w)| o * 2 >= codes[w].count_ones() as u32 && o > 0).max().map(|x| x.1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    const BITS: usize = 2048;

    fn codes(n: usize, rng: &mut StdRng) -> Vec<BitVector> {
        let all: Vec<usize> = (0..BITS).collect();
        (0..n).map(|_| BitVector::from_bits(&all.choose_multiple(rng, 32).copied().collect::<Vec<_>>(), BITS)).collect()
    }

    /// A family tree read as sentences: "<child> 's father is <man>", "<child> 's mother
    /// is <woman>", with a few "<x> likes <y>" facts mixed in.
    fn family() -> (Vec<&'static str>, Vec<Vec<usize>>, Vec<(usize, usize, usize)>) {
        let words = vec![
            "'s", "father", "mother", "is", "likes", // frame words
            "al", "ann", "bob", "bea", "cy", "cat", "dan", "dee", "ed", "eve", "fred", "fay", "gus", "gem", "hal", "hun",
        ];
        let id = |s: &str| words.iter().position(|w| *w == s).unwrap();
        // (child, father, mother)
        let tree = [
            ("bob", "al", "ann"),
            ("cy", "al", "ann"),
            ("dee", "bob", "bea"),
            ("ed", "bob", "bea"),
            ("fay", "cy", "cat"),
            ("fred", "cy", "cat"),
            ("gus", "ed", "eve"),
            ("gem", "ed", "eve"),
            ("hal", "fred", "dee"),
            ("hun", "fred", "dee"),
            ("dan", "ed", "fay"),
        ];
        let mut facts = Vec::new();
        let mut truth = Vec::new();
        for (c, f, m) in tree {
            facts.push(vec![id(c), id("'s"), id("father"), id("is"), id(f)]);
            facts.push(vec![id(c), id("'s"), id("mother"), id("is"), id(m)]);
            truth.push((id(c), id(f), id(m)));
        }
        for (a, b) in [("al", "ann"), ("cy", "cat"), ("gus", "gem")] {
            facts.push(vec![id(a), id("likes"), id(b)]);
        }
        (words, facts, truth)
    }

    fn learned() -> (Vec<&'static str>, Vec<BitVector>, RelationStore, Vec<(usize, usize, usize)>) {
        let mut rng = StdRng::seed_from_u64(3);
        let (words, facts, truth) = family();
        let codes = codes(words.len(), &mut rng);
        let mut s = RelationStore::new(BITS);
        for f in &facts {
            s.observe(f);
        }
        s.consolidate(&codes, 20, &mut rng);
        (words, codes, s, truth)
    }

    #[test]
    fn relations_are_learned_from_fact_shapes() {
        let (words, _, s, _) = learned();
        let id = |x: &str| words.iter().position(|w| *w == x).unwrap();
        let names: Vec<Vec<&str>> = s.frames().iter().map(|f| f.iter().map(|&w| words[w]).collect()).collect();
        assert!(s.relation_for(&[id("father")]).is_some(), "frames {names:?}");
        assert_ne!(s.relation_for(&[id("father")]), s.relation_for(&[id("mother")]), "frames {names:?}");
    }

    #[test]
    fn name_and_relation_give_the_filler() {
        let (words, codes, s, truth) = learned();
        let id = |x: &str| words.iter().position(|w| *w == x).unwrap();
        let father = s.relation_for(&[id("father")]).unwrap();
        let mother = s.relation_for(&[id("mother")]).unwrap();
        for &(c, f, m) in &truth {
            assert_eq!(s.ask(&codes, c, father, 0, 1), Some(f), "{}'s father", words[c]);
            // the same entity in the same position, another relation: no collision
            assert_eq!(s.ask(&codes, c, mother, 0, 1), Some(m), "{}'s mother", words[c]);
        }
        // and backwards: whose father is al? one of his children
        let kid = s.ask(&codes, id("al"), father, 1, 0);
        assert!(kid == Some(id("bob")) || kid == Some(id("cy")), "{:?}", kid.map(|k| words[k]));
    }

    #[test]
    fn relations_chain() {
        let (words, codes, s, _) = learned();
        let id = |x: &str| words.iter().position(|w| *w == x).unwrap();
        let father = s.relation_for(&[id("father")]).unwrap();
        let mother = s.relation_for(&[id("mother")]).unwrap();
        // hal's father is fred, fred's father is cy, cy's father is al
        assert_eq!(s.follow(&codes, id("hal"), &[father, father]), Some(id("cy")));
        assert_eq!(s.follow(&codes, id("hal"), &[father, father, father]), Some(id("al")));
        // hal's mother is dee, dee's father is bob
        assert_eq!(s.follow(&codes, id("hal"), &[mother, father]), Some(id("bob")));
        // al has no father stated
        assert_eq!(s.follow(&codes, id("hal"), &[father, father, father, father]), None);
    }
}
