//! Episodic autoassociative memory: one-shot storage of sparse patterns and
//! recall from partial cues.
//!
//! Each stored *episode* is a sparse bit pattern (e.g. the union of the word
//! codes of one sentence). Storing is a single write, with no gradual learning.
//! Recall takes a cue pattern and returns the stored episode whose bits overlap
//! the cue most, the most recent one on ties: content-addressed retrieval with
//! a hard max, i.e. attention over stored episodes (Ramsauer et al. 2020).
//!
//! Recall is event-based: an inverted index maps each bit to the stored episodes that
//! contain it, so a cue touches only the episodes sharing its bits (as the predictor's
//! kernels are reached through their input bits) instead of comparing it with every
//! episode at full width. Overlaps, and so recalls, are exactly those of the full scan.
//!
//! Habituation: the memory counts how often each bit occurs across stored
//! episodes. `novel` keeps only bits that are rare, so content shared by most
//! episodes (function words) neither drives recall nor fills the output.

use crate::fixed::{mul_ceil, mul_floor, ratio, Q16, ONE};

/// 0.7 in `Q16`: the share of a cue's bits an episode must contain to be recalled in a chain.
const SEVEN_TENTHS: Q16 = 45875;
use crate::bitvec::BitVector;

pub struct EpisodicMemory {
    bits: usize,
    capacity: usize,
    episodes: Vec<BitVector>, // oldest first
    priority: Vec<u32>,       // replay priority tags, parallel to `episodes`
    bit_counts: Vec<u32>,     // how many stored episodes (ever) had each bit
    stored: u32,
    first_id: usize,          // id of episodes[0]; ids stay stable as old episodes are forgotten
    index: Vec<Vec<u32>>,     // bit -> ids of the stored episodes that contain it
}

impl EpisodicMemory {
    pub fn new(bits: usize, capacity: usize) -> Self {
        Self { bits, capacity, episodes: Vec::new(), priority: Vec::new(), bit_counts: vec![0; bits], stored: 0, first_id: 0, index: vec![Vec::new(); bits] }
    }

    /// Tag episode `id` (as returned by `recall_excluding`) for prioritised replay, e.g.
    /// because a question recalled it and the recall predicted the answer.
    pub fn tag(&mut self, id: usize) {
        if id >= self.first_id {
            if let Some(p) = self.priority.get_mut(id - self.first_id) {
                *p = p.saturating_add(1);
            }
        }
    }

    /// The episode with id `id` (as returned by `recall_excluding`), if still held.
    pub fn get_by_id(&self, id: usize) -> Option<&BitVector> {
        id.checked_sub(self.first_id).and_then(|i| self.episodes.get(i))
    }

    /// Id of the `i`-th held episode.
    pub fn id_of(&self, i: usize) -> usize {
        self.first_id + i
    }

    /// Replay priority of the `i`-th held episode.
    pub fn priority(&self, i: usize) -> u32 {
        self.priority.get(i).copied().unwrap_or(0)
    }

    /// Pick an episode to replay: index `i` with weight 1 + `boost` × its tags
    /// (`boost` 0 = uniform).
    pub fn sample_replay<R: rand::Rng + ?Sized>(&self, rng: &mut R, boost: u32) -> Option<usize> {
        if self.episodes.is_empty() {
            return None;
        }
        let total: u64 = self.priority.iter().map(|&p| 1 + boost as u64 * p as u64).sum();
        let mut x = rng.gen_range(0..total);
        for (i, &p) in self.priority.iter().enumerate() {
            let w = 1 + boost as u64 * p as u64;
            if x < w {
                return Some(i);
            }
            x -= w;
        }
        Some(self.episodes.len() - 1)
    }

    /// The `i`-th episode currently held (0 = oldest), e.g. for replay.
    pub fn get(&self, i: usize) -> Option<&BitVector> {
        self.episodes.get(i)
    }

    /// Store one episode in a single shot (the oldest is forgotten at capacity).
    pub fn store(&mut self, episode: &BitVector) {
        let id = (self.first_id + self.episodes.len()) as u32;
        for b in set_bits(episode) {
            if b < self.bits {
                self.bit_counts[b] += 1;
                self.index[b].push(id);
            }
        }
        self.stored += 1;
        self.episodes.push(episode.clone());
        self.priority.push(0);
        if self.episodes.len() > self.capacity {
            let old = self.episodes.remove(0);
            let old_id = self.first_id as u32;
            for b in set_bits(&old) {
                if b < self.bits {
                    // ids are appended in order, so the oldest is at the front
                    if self.index[b].first() == Some(&old_id) {
                        self.index[b].remove(0);
                    } else {
                        self.index[b].retain(|&x| x != old_id);
                    }
                }
            }
            self.priority.remove(0);
            self.first_id += 1;
        }
    }

    /// Only the bits of `pattern` that occurred in at most `max_fraction` of the
    /// stored episodes (habituation to frequent content). `max_fraction` in `Q16`.
    pub fn novel(&self, pattern: &BitVector, max_fraction: Q16) -> BitVector {
        let mut out = BitVector::new(self.bits, Some(0));
        // count ≤ max(stored · fraction, 1), all scaled by ONE
        let limit = (self.stored as u64 * max_fraction as u64).max(ONE as u64);
        for b in set_bits(pattern) {
            if b < self.bits && ((self.bit_counts[b] as u64) << 16) <= limit {
                out.bit_set(b);
            }
        }
        out
    }

    /// Only the rarest content of `pattern`: bits stored at most `ratio` times as
    /// often as the `percentile` rarest of its previously seen bits (bits never
    /// stored count as rarest). Taking a low percentile rather than the single
    /// rarest bit keeps a whole rare word even when a few of its bits are shared
    /// with other words. "where is peter ?" keeps *peter*.
    /// `percentile` and `ratio` in `Q16` (`ratio` may exceed `ONE`).
    pub fn rarest(&self, pattern: &BitVector, percentile: Q16, ratio: Q16) -> BitVector {
        let bits = set_bits(pattern);
        let mut seen: Vec<u32> = bits.iter().map(|&b| self.bit_counts.get(b).copied().unwrap_or(0)).filter(|&c| c > 0).collect();
        let mut out = BitVector::new(self.bits, Some(0));
        if seen.is_empty() {
            return out;
        }
        seen.sort_unstable();
        let reference = seen[mul_floor((seen.len() - 1) as u64, percentile.min(ONE)) as usize];
        let limit = reference as u64 * ratio as u64; // scaled by ONE
        for b in bits {
            let c = self.bit_counts.get(b).copied().unwrap_or(0);
            if b < self.bits && (c == 0 || (c as u64) << 16 <= limit) {
                out.bit_set(b);
            }
        }
        out
    }

    /// Mean fraction of stored episodes containing each bit of `pattern`, in `Q16`.
    pub fn frequency(&self, pattern: &BitVector) -> Q16 {
        let bits = set_bits(pattern);
        if bits.is_empty() || self.stored == 0 {
            return 0;
        }
        let sum: u64 = bits.iter().map(|&b| self.bit_counts.get(b).copied().unwrap_or(0) as u64).sum();
        ratio(sum, bits.len() as u64 * self.stored as u64)
    }

    /// Mean number of stored episodes containing each bit of `pattern` (rounded).
    pub fn count_of(&self, pattern: &BitVector) -> u64 {
        let bits = set_bits(pattern);
        if bits.is_empty() {
            return 0;
        }
        let sum: u64 = bits.iter().map(|&b| self.bit_counts.get(b).copied().unwrap_or(0) as u64).sum();
        (sum + bits.len() as u64 / 2) / bits.len() as u64
    }

    /// The stored episode overlapping `cue` the most (at least `min_overlap`
    /// bits), the most recent one on ties.
    pub fn recall(&self, cue: &BitVector, min_overlap: u32) -> Option<&BitVector> {
        self.recall_excluding(cue, min_overlap, &[]).map(|(_, e)| e)
    }

    /// Rarity-weighted recall: each active cue bit votes for the episodes containing it with
    /// weight ≈ log2(stored episodes / episodes containing the bit), from integer bit lengths
    /// (at least 1). A rare shared item (a name seen once) then outweighs several common
    /// ones (filler words shared by many episodes). Returns the best episode scoring at
    /// least `min_score`, the most recent on ties.
    pub fn recall_rare(&self, cue: &BitVector, min_score: u64) -> Option<&BitVector> {
        self.recall_rare_scored(cue, min_score).map(|(_, e)| e)
    }

    /// `recall_rare` with the winning score (the recall's strength).
    pub fn recall_rare_scored(&self, cue: &BitVector, min_score: u64) -> Option<(u64, &BitVector)> {
        let bitlen = |x: u64| 64 - x.leading_zeros();
        let total = bitlen(self.stored as u64 + 1);
        let mut scores = vec![0u64; self.episodes.len()];
        for (wi, &w) in cue.as_words().iter().enumerate() {
            let mut w = w;
            while w != 0 {
                let b = wi * 64 + w.trailing_zeros() as usize;
                w &= w - 1;
                if let Some(ids) = self.index.get(b) {
                    let c = self.bit_counts.get(b).copied().unwrap_or(0) as u64;
                    let weight = total.saturating_sub(bitlen(c + 1)).max(1) as u64;
                    for &id in ids {
                        scores[id as usize - self.first_id] += weight;
                    }
                }
            }
        }
        let mut best: Option<(u64, usize)> = None;
        for (i, &o) in scores.iter().enumerate() {
            if o >= min_score && o > 0 && best.map_or(true, |(bo, _)| o >= bo) {
                best = Some((o, i));
            }
        }
        best.map(|(o, i)| (o, &self.episodes[i]))
    }

    /// Like `recall`, skipping episodes whose ids are in `exclude`; returns (id, episode).
    pub fn recall_excluding(&self, cue: &BitVector, min_overlap: u32, exclude: &[usize]) -> Option<(usize, &BitVector)> {
        // event-based: each active cue bit votes for the episodes that contain it
        let mut counts = vec![0u32; self.episodes.len()];
        for (wi, &w) in cue.as_words().iter().enumerate() {
            let mut w = w;
            while w != 0 {
                let b = wi * 64 + w.trailing_zeros() as usize;
                w &= w - 1;
                if let Some(ids) = self.index.get(b) {
                    for &id in ids {
                        counts[id as usize - self.first_id] += 1;
                    }
                }
            }
        }
        let mut best: Option<(u32, usize)> = None;
        for (i, &o) in counts.iter().enumerate() {
            if o == 0 || exclude.contains(&(self.first_id + i)) {
                continue;
            }
            if o >= min_overlap && best.map_or(true, |(bo, _)| o >= bo) {
                best = Some((o, i));
            }
        }
        best.map(|(_, i)| (self.first_id + i, &self.episodes[i]))
    }

    /// Big-loop recall (EC -> hippocampus -> EC -> ...): recall from `cue`, then use
    /// the rarest new content of what came back as the next cue, for up to `hops`
    /// hops. Episodes already recalled are skipped (inhibition of return) and content
    /// already used as a cue is removed, so each hop moves on. Returns each hop's new
    /// (habituated) content; stops early when nothing new is recalled.
    ///
    /// "where is the ball ?" -> hop 1: "mary picked up the ball" -> *mary* ->
    /// hop 2: "mary went to the kitchen" -> *kitchen*.
    /// `habituation`, `percentile` and `ratio` in `Q16`.
    pub fn recall_chain(&self, cue: &BitVector, hops: usize, habituation: Q16, percentile: Q16, ratio: Q16) -> Vec<BitVector> {
        let mut out = Vec::new();
        let mut visited = Vec::new();
        let mut used = cue.clone(); // everything cued so far
        let mut cue = cue.clone();
        for _ in 0..hops {
            let need = (mul_ceil(cue.count_ones() as u64, SEVEN_TENTHS) as u32).max(1);
            if cue.count_ones() == 0 {
                break;
            }
            let Some((id, ep)) = self.recall_excluding(&cue, need, &visited) else { break };
            visited.push(id);
            let mut content = self.novel(ep, habituation);
            for (c, &u) in content.as_words_mut().iter_mut().zip(used.as_words()) {
                *c &= !u;
            }
            if content.count_ones() == 0 {
                break;
            }
            cue = self.rarest(&content, percentile, ratio);
            for (u, &c) in used.as_words_mut().iter_mut().zip(cue.as_words()) {
                *u |= c;
            }
            out.push(content);
        }
        out
    }

    /// Split `pattern` into item-like groups: bits with the same stored count
    /// (with sparse codes, one word's bits share a count). Groups smaller than
    /// `min_bits` (collision debris) are dropped; rarest groups first.
    pub fn items(&self, pattern: &BitVector, min_bits: usize) -> Vec<BitVector> {
        let mut by_count: std::collections::BTreeMap<u32, Vec<usize>> = std::collections::BTreeMap::new();
        for b in set_bits(pattern) {
            if b < self.bits {
                by_count.entry(self.bit_counts[b]).or_default().push(b);
            }
        }
        by_count.into_values().filter(|bits| bits.len() >= min_bits).map(|bits| BitVector::from_bits(&bits, self.bits)).collect()
    }

    /// Branching big loop: hop 1 from `cue` as in `recall_chain`, then a separate
    /// hop-2 recall from each of the `branches` rarest items of hop 1's new content
    /// (like several attention heads). Returns `[hop 1, branch 1, ..]`; a reader that
    /// learns (the predictor) decides which branch carries the answer. Missing
    /// recalls are empty frames, so positions stay fixed.
    /// `habituation` in `Q16`.
    pub fn recall_branches(&self, cue: &BitVector, branches: usize, habituation: Q16, min_item_bits: usize) -> Vec<BitVector> {
        let empty = BitVector::new(self.bits, Some(0));
        let mut out = vec![empty.clone(); branches + 1];
        if cue.count_ones() == 0 {
            return out;
        }
        let need = (mul_ceil(cue.count_ones() as u64, SEVEN_TENTHS) as u32).max(1);
        let Some((id, ep)) = self.recall_excluding(cue, need, &[]) else { return out };
        let mut hop1 = self.novel(ep, habituation);
        for (c, &u) in hop1.as_words_mut().iter_mut().zip(cue.as_words()) {
            *c &= !u;
        }
        for (b, item) in self.items(&hop1, min_item_bits).into_iter().take(branches).enumerate() {
            let need = (mul_ceil(item.count_ones() as u64, SEVEN_TENTHS) as u32).max(1);
            if let Some((_, ep2)) = self.recall_excluding(&item, need, &[id]) {
                let mut content = self.novel(ep2, habituation);
                for (c, (&u, &i)) in content.as_words_mut().iter_mut().zip(cue.as_words().iter().zip(item.as_words())) {
                    *c &= !(u | i);
                }
                out[b + 1] = content;
            }
        }
        out[0] = hop1;
        out
    }

    /// Bytes held: the stored episodes (dense bit vectors), the bit counts and the index.
    pub fn bytes(&self) -> usize {
        self.episodes.len() * self.bits.div_ceil(8) + 4 * self.bit_counts.len() + self.index.iter().map(|p| 4 * p.len() + 24).sum::<usize>()
    }

    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }
}

fn overlap(a: &BitVector, b: &BitVector) -> u32 {
    a.as_words().iter().zip(b.as_words()).map(|(x, y)| (x & y).count_ones()).sum()
}

fn set_bits(bv: &BitVector) -> Vec<usize> {
    let mut out = Vec::new();
    for (wi, &w) in bv.as_words().iter().enumerate() {
        let mut w = w;
        while w != 0 {
            out.push(wi * 64 + w.trailing_zeros() as usize);
            w &= w - 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn rarity_weighted_recall_prefers_a_rare_shared_item() {
        let bits = 1024;
        let word = |i: usize| BitVector::from_bits(&(i * 32..i * 32 + 32).collect::<Vec<_>>(), bits);
        let or = |ws: &[usize]| {
            let mut v = BitVector::new(bits, Some(0));
            for &w in ws {
                v.or_mut(&word(w));
            }
            v
        };
        let mut m = EpisodicMemory::new(bits, 100);
        // common words 0, 1, 2 in most episodes; the rare word 9 in one
        for i in 0..20 {
            m.store(&or(&[0, 1, 2, 10 + i % 5]));
        }
        m.store(&or(&[9, 20]));
        // cue: two common words and the rare one; plain recall prefers the common pair,
        // rarity-weighted recall the episode with the rare word
        let cue = or(&[0, 1, 9]);
        assert!(m.recall(&cue, 1).unwrap().bit_get(32));
        assert!(m.recall_rare(&cue, 1).unwrap().bit_get(9 * 32));
    }


    #[test]
    fn indexed_recall_matches_a_full_scan() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let bits = 512;
        let mut mem = EpisodicMemory::new(bits, 20);
        let random = |rng: &mut rand::rngs::StdRng, n: usize| {
            let v: Vec<usize> = (0..n).map(|_| rng.gen_range(0..bits)).collect();
            BitVector::from_bits(&v, bits)
        };
        for step in 0..200 {
            mem.store(&random(&mut rng, 24)); // wraps capacity many times
            let cue = random(&mut rng, 12);
            let need = rng.gen_range(1..4);
            let exclude = if step % 3 == 0 { vec![mem.first_id + 5] } else { vec![] };
            // brute force, as recall used to be
            let mut best: Option<(u32, usize)> = None;
            for (i, e) in mem.episodes.iter().enumerate() {
                if exclude.contains(&(mem.first_id + i)) {
                    continue;
                }
                let o = overlap(e, &cue);
                if o >= need && best.map_or(true, |(bo, _)| o >= bo) {
                    best = Some((o, i));
                }
            }
            let want = best.map(|(_, i)| mem.first_id + i);
            assert_eq!(mem.recall_excluding(&cue, need, &exclude).map(|(id, _)| id), want, "step {step}");
        }
    }
    use super::*;

    fn sym(i: usize) -> BitVector {
        BitVector::from_bits(&[i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3], 256)
    }

    fn bag(words: &[usize]) -> BitVector {
        let mut out = BitVector::new(256, Some(0));
        for &w in words {
            out.or_mut(&sym(w));
        }
        out
    }

    #[test]
    fn tagged_episodes_are_replayed_more() {
        use rand::SeedableRng;
        let mut m = EpisodicMemory::new(64, 10);
        for i in 0..4 {
            m.store(&sym(i));
        }
        let (id, _) = m.recall_excluding(&sym(2), 4, &[]).unwrap();
        for _ in 0..5 {
            m.tag(id);
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let picks = (0..1000).filter(|_| m.sample_replay(&mut rng, 4) == Some(2)).count();
        assert!(picks > 700, "tagged episode picked {picks}/1000"); // weight 21 of 24
    }

    #[test]
    fn recalls_the_most_recent_episode_sharing_the_cue() {
        // 1 = mary, 2 = john, 10 = went/to/the, 20 = kitchen, 21 = garden, 22 = office
        let mut m = EpisodicMemory::new(256, 16);
        m.store(&bag(&[1, 10, 20])); // mary went to the kitchen
        m.store(&bag(&[2, 10, 21])); // john went to the garden
        m.store(&bag(&[1, 10, 22])); // mary went to the office (later)
        let got = m.recall(&sym(1), 4).unwrap();
        assert_eq!(got.as_words(), bag(&[1, 10, 22]).as_words());
        assert_eq!(m.recall(&sym(2), 4).unwrap().as_words(), bag(&[2, 10, 21]).as_words());
        assert!(m.recall(&sym(5), 4).is_none());
    }

    #[test]
    fn habituation_removes_content_shared_by_most_episodes() {
        let mut m = EpisodicMemory::new(256, 16);
        for (name, place) in [(1, 20), (2, 21), (3, 22), (4, 23)] {
            m.store(&bag(&[name, 10, place])); // "went to the" (10) in every episode
        }
        let ep = m.recall(&sym(3), 4).unwrap().clone();
        let novel = m.novel(&ep, crate::fixed::q16(0.5));
        assert_eq!(novel.as_words(), bag(&[3, 22]).as_words()); // 10 is gone
    }

    #[test]
    fn rarest_keeps_the_most_informative_content() {
        let mut m = EpisodicMemory::new(256, 64);
        // 10, 11 occur everywhere ("where is"); names 1..4 once each
        for name in 1..=4 {
            m.store(&bag(&[10, 11, name]));
            m.store(&bag(&[10, 11]));
        }
        let cue = m.rarest(&bag(&[10, 11, 3]), crate::fixed::q16(0.1), crate::fixed::ONE * 3 / 2);
        assert_eq!(cue.as_words(), sym(3).as_words());
    }

    #[test]
    fn big_loop_chains_two_hops_without_revisiting() {
        // 1 mary, 2 john, 30 ball, 31 picked-up, 10 went-to-the, 20 kitchen, 21 garden
        let mut m = EpisodicMemory::new(256, 64);
        for _ in 0..3 {
            m.store(&bag(&[40, 10])); // background episodes make "went to the" (10)
            m.store(&bag(&[41, 31])); // and "picked up" (31) frequent, as in real stories
        }
        m.store(&bag(&[1, 31, 30])); // mary picked up the ball
        m.store(&bag(&[2, 10, 21])); // john went to the garden
        m.store(&bag(&[1, 10, 20])); // mary went to the kitchen
        let hops = m.recall_chain(&sym(30), 2, crate::fixed::q16(0.4), crate::fixed::q16(0.1), crate::fixed::ONE * 3 / 2);
        assert_eq!(hops.len(), 2);
        assert_eq!(hops[0].as_words(), sym(1).as_words(), "hop 1 brings back mary");
        assert_eq!(hops[1].as_words(), sym(20).as_words(), "hop 2 brings back kitchen");
    }
}