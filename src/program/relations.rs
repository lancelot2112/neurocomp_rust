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
//! - **A frame word can also be an entity** (`lift`): "jones" is a filler of "X is a Y"
//!   and part of the frame "X jones went to the Y". A fact is then read twice: as parsed,
//!   and with its weak frame positions (under 3/4 of its neighbours agree) and its frame
//!   words that are entities elsewhere lifted into the fillers ("X _ went to the Y":
//!   john, jones, hallway). So jones → (went, 1→2) → the places joneses went, and lucy →
//!   jones → place can be followed.
//! - **Relations of relations are learned at sleep.** For each stated fact (x, r, z),
//!   every two-step path x →s1→ y →s2→ z through the store is counted. A path that
//!   gives the stated filler for at least three quarters of the r-facts it applies to (and
//!   at least twice) becomes a rule r = s1 ∘ s2 ("grandfather = father ∘ father"). Each
//!   rule then infers r for the entities that have the path but no stated r, and those
//!   inferred facts are replayed into the store like stated ones, so `ask` answers them
//!   directly (generative replay; the cortex learns what it was never told).
//! - **No source is fully trusted.** Every fact is a *claim* by a source (a narrator, the
//!   network itself, a proposal). Claims about the same thing (entity, relation,
//!   direction) with different values conflict. At sleep, the sources' trust and the
//!   claims' belief are estimated together (truth discovery, as TruthFinder): a value's
//!   belief is the summed trust of the sources claiming it; a source's trust is the share
//!   of belief its claims win, over the facts at least two sources spoke about. Only the
//!   believed value of a conflict is consolidated; the others stay recorded as claims.
//! - **Navigation:** `ask(tom, father, 0, 1)` gives bob; `follow(tom, &[father, father])`
//!   gives bob's father. The relation for a query is found from its words
//!   (`relation_for(["father"])`): the frame sharing the most of them.

use rand::Rng;

use crate::bitvec::BitVector;
use crate::det::HashMap;
use crate::fixed::ONE;
use crate::kernel::{GrowthConfig, KernelClass, SimpleKernel};

pub struct RelationStore {
    bits: usize,
    store: KernelClass<SimpleKernel>,
    /// Distinct facts read, and their buckets: (length, two positions left out, the other
    /// words) → facts. Facts in one bucket agree in all but those two positions.
    facts: Vec<Vec<usize>>,
    seen: HashMap<Vec<usize>, usize>,
    buckets: HashMap<(usize, usize, usize, u64), Vec<usize>>,
    /// Facts read since the last consolidation, with their source.
    buffer: Vec<(Vec<usize>, u16)>,
    /// Claims: (relation, entity, from, to) → the (value, source) pairs claimed.
    claims: HashMap<(usize, usize, usize, usize), Vec<(usize, u16)>>,
    /// Each source's trust (`Q16`), from the last resolution.
    trust: HashMap<u16, u32>,
    /// Learn the sources' trust (true), or count every source the same: a plain vote.
    pub use_trust: bool,
    /// Learned relations: each a frame (its length and its (position, word)s).
    frames: Vec<(usize, Vec<(usize, usize)>)>,
    /// Per entity: the (relation, from, to) keys stored with it.
    links: HashMap<usize, Vec<(usize, usize, usize)>>,
    /// Stated binary facts (relation, first filler, second filler).
    stated: Vec<(usize, usize, usize)>,
    /// Learned compositions: relation = step ∘ step, with (confirmations, applicable).
    rules: Vec<Rule>,
    /// Words seen as fillers (entities).
    entities: std::collections::BTreeSet<usize>,
    /// Lift frame words that are entities elsewhere into fillers (a second reading of the
    /// fact, kept beside the first).
    pub lift: bool,
    inferred: usize,
    replays: usize,
}

/// A step through the store: (relation, from position, to position).
pub type Step = (usize, usize, usize);

/// A learned relation of relations: `relation` = `first` then `second`.
#[derive(Clone, Debug, PartialEq)]
pub struct Rule {
    pub relation: usize,
    pub first: Step,
    pub second: Step,
    /// Stated facts the path reproduced, and stated facts it applied to.
    pub confirmed: usize,
    pub applicable: usize,
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
            claims: HashMap::default(),
            trust: HashMap::default(),
            use_trust: true,
            frames: Vec::new(),
            links: HashMap::default(),
            stated: Vec::new(),
            rules: Vec::new(),
            entities: Default::default(),
            lift: false,
            inferred: 0,
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
        self.observe_from(fact, 0);
    }

    /// Read a fact claimed by `source` (who said it).
    pub fn observe_from(&mut self, fact: &[usize], source: u16) {
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
        self.buffer.push((fact.to_vec(), source));
    }

    /// A fact's (frame, fillers in order), by the facts read so far; None unless it has
    /// at least two neighbours, a frame and at least two fillers. Frame positions are those
    /// where more than half of its neighbours (same length, all but two positions equal)
    /// have the same word.
    pub fn parse(&self, fact: &[usize]) -> Option<((usize, Vec<(usize, usize)>), Vec<usize>)> {
        let fp = self.frame_positions(fact)?;
        Self::reading(fact, &fp.iter().map(|x| x.0).collect::<Vec<_>>())
    }

    /// The frame positions of a fact and whether each is strong (at least 3/4 of its
    /// neighbours agree there); None with fewer than two neighbours.
    fn frame_positions(&self, fact: &[usize]) -> Option<Vec<(usize, bool)>> {
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
        Some(
            (0..fact.len())
                .filter_map(|p| {
                    let agree = nb.iter().filter(|&&f| self.facts[f][p] == fact[p]).count();
                    (agree * 2 > nb.len()).then_some((p, agree * 4 >= nb.len() * 3))
                })
                .collect(),
        )
    }

    /// The fact read with these frame positions: (frame, fillers in order), if it has a
    /// frame and at least two fillers.
    fn reading(fact: &[usize], frame_pos: &[usize]) -> Option<((usize, Vec<(usize, usize)>), Vec<usize>)> {
        let (mut frame, mut fillers) = (Vec::new(), Vec::new());
        for (p, &w) in fact.iter().enumerate() {
            if frame_pos.contains(&p) {
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
        let batch = std::mem::take(&mut self.buffer);
        let facts: Vec<Vec<usize>> = batch.iter().map(|x| x.0.clone()).collect();
        let source_of: HashMap<Vec<usize>, Vec<u16>> = batch.iter().fold(HashMap::default(), |mut m, (f, src)| {
            let e: &mut Vec<u16> = m.entry(f.clone()).or_default();
            if !e.contains(src) {
                e.push(*src);
            }
            m
        });
        let mut pairs: Vec<(BitVector, usize, BitVector)> = Vec::new(); // (key, entity, value)
        let mut parsed = 0;
        // the first reading of each fact (its frame: positions where most neighbours agree)
        let fps: Vec<(Vec<usize>, Vec<(usize, bool)>)> = facts.iter().filter_map(|f| self.frame_positions(f).map(|fp| (f.clone(), fp))).collect();
        let mut all: Vec<((usize, Vec<(usize, usize)>), Vec<usize>, Vec<u16>)> = Vec::new();
        for (fact, fp) in &fps {
            if let Some((frame, fillers)) = Self::reading(fact, &fp.iter().map(|x| x.0).collect::<Vec<_>>()) {
                self.entities.extend(fillers.iter().copied());
                all.push((frame, fillers, source_of.get(fact).cloned().unwrap_or_default()));
                parsed += 1;
            }
        }
        // the second reading (`lift`): frame positions that are weak (under 3/4 of the
        // neighbours agree: "jones" among joneses and smiths) or hold a word that is an
        // entity elsewhere ("jones" in "X jones went to the Y") read as fillers too
        if self.lift {
            for (fact, fp) in &fps {
                let kept: Vec<usize> = fp.iter().filter(|&&(p, strong)| strong && !self.entities.contains(&fact[p])).map(|x| x.0).collect();
                if kept.len() < fp.len() {
                    if let Some(r) = Self::reading(fact, &kept) {
                        parsed += all.iter().all(|a| a.1 != r.1) as usize;
                        self.entities.extend(r.1.iter().copied());
                        all.push((r.0, r.1, source_of.get(fact).cloned().unwrap_or_default()));
                    }
                }
            }
        }
        let mut batch_claims: Vec<((usize, usize, usize, usize), usize)> = Vec::new();
        for (frame, fillers, sources) in all {
            let r = self.relation_of(frame);
            if !self.stated.contains(&(r, fillers[0], fillers[1])) {
                self.stated.push((r, fillers[0], fillers[1]));
            }
            for i in 0..fillers.len() {
                for j in 0..fillers.len() {
                    if i != j {
                        let c = self.claims.entry((r, fillers[i], i, j)).or_default();
                        for &src in &sources {
                            if !c.contains(&(fillers[j], src)) {
                                c.push((fillers[j], src));
                            }
                        }
                        batch_claims.push(((r, fillers[i], i, j), fillers[j]));
                        let l = self.links.entry(fillers[i]).or_default();
                        if !l.contains(&(r, i, j)) {
                            l.push((r, i, j));
                        }
                    }
                }
            }
        }
        // conflicts: trust and belief estimated together; what is replayed is what is believed
        self.resolve();
        let mut seen_keys: std::collections::BTreeSet<(usize, usize, usize, usize)> = Default::default();
        pairs.clear();
        for (k, v) in batch_claims {
            let w = self.believed(k).unwrap_or(v);
            if seen_keys.insert(k) || w == v {
                pairs.push((self.key(&codes[k.1], k.0, k.2, k.3), k.1, codes[w].clone()));
            }
        }
        // a conflict whose verdict changed with the trust is replayed too
        let contested: Vec<(usize, usize, usize, usize)> = self.claims.iter().filter(|(_, c)| !many_valued(c) && c.iter().any(|x| x.0 != c[0].0)).map(|(k, _)| *k).collect();
        for k in contested {
            if seen_keys.insert(k) {
                if let Some(w) = self.believed(k) {
                    pairs.push((self.key(&codes[k.1], k.0, k.2, k.3), k.1, codes[w].clone()));
                }
            }
        }
        self.replay(&pairs, reps, rng);
        // relations of relations: learned from what the store now answers, then the facts
        // they infer are replayed too
        self.learn_rules(codes);
        let mut inferred: Vec<(BitVector, usize, BitVector)> = Vec::new();
        for rule in self.rules.clone() {
            let ents: Vec<usize> = self.links.iter().filter(|(_, l)| l.contains(&rule.first)).map(|(&e, _)| e).collect();
            for x in ents {
                if self.links.get(&x).is_some_and(|l| l.contains(&(rule.relation, 0, 1))) {
                    continue;
                }
                let Some(z) = self.ask(codes, x, rule.first.0, rule.first.1, rule.first.2).and_then(|y| self.ask(codes, y, rule.second.0, rule.second.1, rule.second.2)) else { continue };
                if z == x {
                    continue;
                }
                inferred.push((self.key(&codes[x], rule.relation, 0, 1), x, codes[z].clone()));
                self.links.entry(x).or_default().push((rule.relation, 0, 1));
                self.inferred += 1;
            }
        }
        self.replay(&inferred, reps, rng);
        parsed
    }

    fn replay<R: Rng>(&mut self, pairs: &[(BitVector, usize, BitVector)], reps: usize, rng: &mut R) {
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
    }

    /// Count, for every stated fact (x, r, z), the two-step paths x → y → z' through the
    /// store; keep as rules the paths that reproduce z for at least three quarters of the
    /// r-facts they apply to, and at least twice.
    fn learn_rules(&mut self, codes: &[BitVector]) {
        let mut counts: HashMap<(usize, Step, Step), (usize, usize)> = HashMap::default();
        for &(r, x, z) in &self.stated {
            let Some(l1) = self.links.get(&x) else { continue };
            for &s1 in l1.iter().filter(|s| s.1 < 2 && s.2 < 2 && s.0 != r) {
                let Some(y) = self.ask(codes, x, s1.0, s1.1, s1.2).filter(|&y| y != x && y != z) else { continue };
                let Some(l2) = self.links.get(&y) else { continue };
                for &s2 in l2.iter().filter(|s| s.1 < 2 && s.2 < 2 && s.0 != r) {
                    let Some(z2) = self.ask(codes, y, s2.0, s2.1, s2.2).filter(|&z2| z2 != y) else { continue };
                    let c = counts.entry((r, s1, s2)).or_default();
                    c.0 += (z2 == z) as usize;
                    c.1 += 1;
                }
            }
        }
        let mut rules: Vec<Rule> = counts
            .into_iter()
            .filter(|(_, (ok, n))| *ok >= 2 && ok * 4 >= n * 3)
            .map(|((relation, first, second), (confirmed, applicable))| Rule { relation, first, second, confirmed, applicable })
            .collect();
        rules.sort_by(|a, b| b.confirmed.cmp(&a.confirmed).then((a.relation, a.first, a.second).cmp(&(b.relation, b.first, b.second))));
        self.rules = rules;
    }

    /// Truth discovery over the claims that at least two sources made: a value's belief is
    /// the summed trust of its sources; a source's trust is the mean share of belief its
    /// claims win (with one win and one loss as a prior). Eight rounds, integers only.
    fn resolve(&mut self) {
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

    /// The believed value of a claim key: the one with the most summed trust (unknown
    /// sources count half), None if nothing was claimed.
    fn believed(&self, k: (usize, usize, usize, usize)) -> Option<usize> {
        let c = self.claims.get(&k)?;
        if many_valued(c) {
            return None;
        }
        let t = |s: u16| self.trust.get(&s).copied().unwrap_or(ONE / 2) as u64;
        let mut best: Option<(u64, usize)> = None;
        for &(v, _) in c {
            let b: u64 = c.iter().filter(|x| x.0 == v).map(|x| t(x.1)).sum();
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

    /// The claims about `entity` through relation `r` from `from` to `to`: (value, source,
    /// the value's belief in `Q16` of all belief on this key).
    pub fn claims(&self, entity: usize, r: usize, from: usize, to: usize) -> Vec<(usize, u16, u32)> {
        let Some(c) = self.claims.get(&(r, entity, from, to)) else { return Vec::new() };
        let t = |s: u16| self.trust(s) as u64;
        let total: u64 = c.iter().map(|x| t(x.1)).sum::<u64>().max(1);
        c.iter().map(|&(v, s)| (v, s, ((c.iter().filter(|x| x.0 == v).map(|x| t(x.1)).sum::<u64>() << 16) / total) as u32)).collect()
    }

    /// The relations of relations learned so far.
    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }

    /// Facts inferred by rules and replayed into the store.
    pub fn inferred(&self) -> usize {
        self.inferred
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

    /// Every answer the store holds for `entity` through relation `r` from `from` to `to`
    /// (one-to-many relations: all of a country's cities): the union of what every
    /// matching kernel predicts, read out as the words it holds at least 3/4 of.
    pub fn ask_all(&self, codes: &[BitVector], entity: usize, r: usize, from: usize, to: usize) -> Vec<usize> {
        let out = self.store.peek_union(&self.key(&codes[entity], r, from, to), self.bits);
        (0..codes.len())
            .filter(|&w| {
                let o: u32 = codes[w].as_words().iter().zip(out.as_words()).map(|(a, b)| (a & b).count_ones()).sum();
                o > 0 && o * 4 >= codes[w].count_ones() as u32 * 3
            })
            .collect()
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

/// A key one source gave several values ("al's children"): a many-valued relation, not a
/// conflict between sources.
fn many_valued(c: &[(usize, u16)]) -> bool {
    c.iter().any(|a| c.iter().any(|b| a.1 == b.1 && a.0 != b.0))
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
            "'s", "father", "mother", "is", "likes", "grandfather", // frame words
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
        // grandfathers (father's father) stated for some grandchildren only
        let father_of = |c: usize| tree.iter().find(|t| id(t.0) == c).map(|t| id(t.1));
        for c in ["dee", "ed", "fay", "fred", "gus", "hal"] {
            let gf = father_of(id(c)).and_then(father_of).unwrap();
            facts.push(vec![id(c), id("'s"), id("grandfather"), id("is"), gf]);
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
    fn grandfather_is_learned_as_father_of_father() {
        let (words, codes, s, truth) = learned();
        let id = |x: &str| words.iter().position(|w| *w == x).unwrap();
        let father = s.relation_for(&[id("father")]).unwrap();
        let gf = s.relation_for(&[id("grandfather")]).unwrap();
        assert!(s.rules().iter().any(|r| r.relation == gf && r.first == (father, 0, 1) && r.second == (father, 0, 1)), "rules {:?}", s.rules());
        // never told: gem, hun, dan; the store answers from inferred replay
        let father_of = |c: usize| truth.iter().find(|t| t.0 == c).map(|t| t.1);
        for c in ["gem", "hun", "dan"] {
            let want = father_of(id(c)).and_then(father_of);
            assert_eq!(s.ask(&codes, id(c), gf, 0, 1), want, "{c}'s grandfather");
        }
    }

    /// Depth: six family lines of eight generations. Fathers are stated for everyone;
    /// grandfathers and great-grandfathers for half of those who have one. Rules build on
    /// rules across sleeps: grandfather from father, then great-grandfather from those.
    #[test]
    fn compositions_build_on_compositions() {
        let mut rng = StdRng::seed_from_u64(11);
        let (lines, gens) = (6usize, 8usize);
        let frame = ["'s", "father", "grandfather", "greatgrandfather", "is"];
        let n_words = frame.len() + lines * gens;
        let person = |l: usize, g: usize| frame.len() + l * gens + g; // g = 0 is the root
        let codes = codes(n_words, &mut rng);
        let mut s = RelationStore::new(BITS);
        let (mut held_gf, mut held_ggf) = (Vec::new(), Vec::new());
        for l in 0..lines {
            for g in 1..gens {
                s.observe(&[person(l, g), 0, 1, 4, person(l, g - 1)]);
                if g >= 2 {
                    if (l + g) % 2 == 0 {
                        s.observe(&[person(l, g), 0, 2, 4, person(l, g - 2)]);
                    } else {
                        held_gf.push((person(l, g), person(l, g - 2)));
                    }
                }
                if g >= 3 {
                    if (l + g) % 2 == 1 {
                        s.observe(&[person(l, g), 0, 3, 4, person(l, g - 3)]);
                    } else {
                        held_ggf.push((person(l, g), person(l, g - 3)));
                    }
                }
            }
        }
        s.consolidate(&codes, 20, &mut rng);
        s.consolidate(&codes, 20, &mut rng); // a second night: rules over inferred facts too
        let gf = s.relation_for(&[2]).unwrap();
        let ggf = s.relation_for(&[3]).unwrap();
        let right = |r: usize, held: &[(usize, usize)]| held.iter().filter(|&&(x, z)| s.ask(&codes, x, r, 0, 1) == Some(z)).count();
        assert_eq!(right(gf, &held_gf), held_gf.len(), "rules {:?}", s.rules());
        assert_eq!(right(ggf, &held_ggf), held_ggf.len(), "rules {:?}", s.rules());
        // and plain chains of fathers, as deep as the lines go
        let father = s.relation_for(&[1]).unwrap();
        for depth in 1..gens {
            let ok = (0..lines).filter(|&l| s.follow(&codes, person(l, gens - 1), &vec![father; depth]) == Some(person(l, gens - 1 - depth))).count();
            assert_eq!(ok, lines, "father x{depth}");
        }
    }

    /// Nothing is about families: a directed graph over abstract tokens. "X r1 Y" and
    /// "X r2 to Y" are two relations of different shapes; edges are directed (forward and
    /// inverse answers differ) and one-to-many (a source has several targets).
    #[test]
    fn a_generic_directed_graph() {
        let mut rng = StdRng::seed_from_u64(5);
        let (r1, r2, to) = (0usize, 1usize, 2usize);
        let n = 40;
        let node = |i: usize| 3 + i;
        let codes = codes(3 + n, &mut rng);
        let mut s = RelationStore::new(BITS);
        // r1: node i → node (i * 7 + 3) % n (a function); r2: node i → nodes i+1, i+2 (one-to-many)
        let f = |i: usize| (i * 7 + 3) % n;
        for i in 0..n {
            if f(i) != i {
                s.observe(&[node(i), r1, node(f(i))]);
            }
            for d in [1, 2] {
                s.observe(&[node(i), r2, to, node((i + d) % n)]);
            }
        }
        s.consolidate(&codes, 20, &mut rng);
        let a = s.relation_for(&[r1]).unwrap();
        let b = s.relation_for(&[r2]).unwrap();
        assert_ne!(a, b);
        let mut fwd = 0;
        let mut many = 0;
        for i in (0..n).filter(|&i| f(i) != i) {
            fwd += (s.ask(&codes, node(i), a, 0, 1) == Some(node(f(i)))) as usize;
            let all = s.ask_all(&codes, node(i), b, 0, 1);
            many += (all.contains(&node((i + 1) % n)) && all.contains(&node((i + 2) % n))) as usize;
        }
        let m = (0..n).filter(|&i| f(i) != i).count();
        assert_eq!(fwd, m, "forward r1");
        assert!(many * 10 >= m * 9, "one-to-many r2: both targets for {many} of {m}");
        // direction: the inverse of r1 answers the source, not the target
        let i = 4;
        assert_eq!(s.ask(&codes, node(f(i)), a, 1, 0), Some(node(i)));
        assert_ne!(s.ask(&codes, node(i), a, 0, 1), s.ask(&codes, node(i), a, 1, 0));
    }

    /// "X is a F" and "X F went to the P": the family word is a filler of the first and
    /// in the frame of the second. Lifted, the store links a family to its places, and a
    /// name stated only as "lucy is a jones" reaches the joneses' places in two steps.
    #[test]
    fn a_frame_word_is_also_an_entity() {
        let mut rng = StdRng::seed_from_u64(2);
        let words = ["is", "a", "went", "to", "the", "jones", "smith", "hall", "yard", "attic", "cellar", "lucy", "ann", "bo", "cy", "di", "ed", "flo", "gil"];
        let id = |x: &str| words.iter().position(|w| *w == x).unwrap();
        let codes = codes(words.len(), &mut rng);
        let mut s = RelationStore::new(BITS);
        s.lift = true;
        let fam = [("ann", "jones"), ("bo", "jones"), ("cy", "jones"), ("di", "smith"), ("ed", "smith"), ("flo", "smith"), ("gil", "jones")];
        let places = |f: &str| if f == "jones" { ["hall", "yard"] } else { ["attic", "cellar"] };
        for (n, f) in fam {
            s.observe(&[id(n), id("is"), id("a"), id(f)]);
            for p in places(f) {
                s.observe(&[id(n), id(f), id("went"), id("to"), id("the"), id(p)]);
            }
        }
        s.observe(&[id("lucy"), id("is"), id("a"), id("jones")]);
        s.consolidate(&codes, 20, &mut rng);
        let is_a = s.relation_for(&[id("is"), id("a")]).unwrap();
        let went = (0..s.frames().len()).find(|&r| s.template(r).1.iter().all(|x| words[x.1] != "jones" && words[x.1] != "smith") && s.frames()[r].contains(&id("went"))).expect("a lifted relation");
        assert_eq!(s.ask(&codes, id("lucy"), is_a, 0, 1), Some(id("jones")));
        // the lifted relation's fillers: (name, family, place); family → place is 1 → 2
        let mut got = s.ask_all(&codes, id("jones"), went, 1, 2);
        got.sort_unstable();
        assert_eq!(got, vec![id("hall"), id("yard")], "{:?}", got.iter().map(|&w| words[w]).collect::<Vec<_>>());
        let fam_of_lucy = s.ask(&codes, id("lucy"), is_a, 0, 1).unwrap();
        let mut two = s.ask_all(&codes, fam_of_lucy, went, 1, 2);
        two.sort_unstable();
        assert_eq!(two, vec![id("hall"), id("yard")]);
    }

    /// No source is fully trusted: three honest narrators each state half the facts, a
    /// fourth states all of them and lies in 3 of 4. Trust is learned from the conflicts,
    /// and the believed fact is the true one, also where only one honest narrator and the
    /// liar spoke (a tie a vote cannot break).
    #[test]
    fn trust_resolves_conflicts_between_sources() {
        let mut rng = StdRng::seed_from_u64(4);
        let fams = 4usize;
        let n = 40usize;
        let (is, a) = (0usize, 1usize);
        let fam = |f: usize| 2 + f;
        let person = |i: usize| 2 + fams + i;
        let codes = codes(2 + fams + n, &mut rng);
        let truth = |i: usize| (i * 7 + 1) % fams;
        let mut s = RelationStore::new(BITS);
        let mut ties = Vec::new();
        for i in 0..n {
            let mut honest = 0;
            for src in 1..=3u16 {
                if rng.gen_range(0..2) == 0 {
                    s.observe_from(&[person(i), is, a, fam(truth(i))], src);
                    honest += 1;
                }
            }
            let lie = rng.gen_range(0..4) < 3;
            let f = if lie { (truth(i) + 1 + rng.gen_range(0..fams - 1)) % fams } else { truth(i) };
            s.observe_from(&[person(i), is, a, fam(f)], 9);
            if honest == 1 && lie {
                ties.push(i);
            }
        }
        s.consolidate(&codes, 20, &mut rng);
        assert!(s.trust(9) < s.trust(1) && s.trust(9) < s.trust(2) && s.trust(9) < s.trust(3), "trust {} {} {} liar {}", s.trust(1), s.trust(2), s.trust(3), s.trust(9));
        let r = s.relation_for(&[is, a]).unwrap();
        let right = (0..n).filter(|&i| s.ask(&codes, person(i), r, 0, 1) == Some(fam(truth(i)))).count();
        // facts only the liar stated cannot be checked: count the others
        let checkable = (0..n).filter(|&i| s.claims(person(i), r, 0, 1).iter().any(|c| c.1 != 9)).count();
        assert!(right * 10 >= checkable * 9, "right {right} of {checkable} checkable");
        assert!(!ties.is_empty());
        for &i in &ties {
            assert_eq!(s.ask(&codes, person(i), r, 0, 1), Some(fam(truth(i))), "tie for person {i}: {:?}", s.claims(person(i), r, 0, 1));
        }
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
