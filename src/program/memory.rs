//! Episodic autoassociative memory: one-shot storage of sparse patterns and
//! recall from partial cues.
//!
//! Each stored *episode* is a sparse bit pattern (e.g. the union of the word
//! codes of one sentence). Storing is a single write, with no gradual learning.
//! Recall takes a cue pattern and returns the stored episode whose bits overlap
//! the cue most, the most recent one on ties: content-addressed retrieval with
//! a hard max, i.e. attention over stored episodes (Ramsauer et al. 2020).
//!
//! Habituation: the memory counts how often each bit occurs across stored
//! episodes. `novel` keeps only bits that are rare, so content shared by most
//! episodes (function words) neither drives recall nor fills the output.

use crate::bitvec::BitVector;

pub struct EpisodicMemory {
    bits: usize,
    capacity: usize,
    episodes: Vec<BitVector>, // oldest first
    bit_counts: Vec<u32>,     // how many stored episodes (ever) had each bit
    stored: u32,
}

impl EpisodicMemory {
    pub fn new(bits: usize, capacity: usize) -> Self {
        Self { bits, capacity, episodes: Vec::new(), bit_counts: vec![0; bits], stored: 0 }
    }

    /// Store one episode in a single shot (the oldest is forgotten at capacity).
    pub fn store(&mut self, episode: &BitVector) {
        for b in set_bits(episode) {
            if b < self.bits {
                self.bit_counts[b] += 1;
            }
        }
        self.stored += 1;
        self.episodes.push(episode.clone());
        if self.episodes.len() > self.capacity {
            self.episodes.remove(0);
        }
    }

    /// Only the bits of `pattern` that occurred in at most `max_fraction` of the
    /// stored episodes (habituation to frequent content).
    pub fn novel(&self, pattern: &BitVector, max_fraction: f32) -> BitVector {
        let mut out = BitVector::new(self.bits, Some(0));
        let limit = (self.stored as f32 * max_fraction).max(1.0);
        for b in set_bits(pattern) {
            if b < self.bits && (self.bit_counts[b] as f32) <= limit {
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
    pub fn rarest(&self, pattern: &BitVector, percentile: f32, ratio: f32) -> BitVector {
        let bits = set_bits(pattern);
        let mut seen: Vec<u32> = bits.iter().map(|&b| self.bit_counts.get(b).copied().unwrap_or(0)).filter(|&c| c > 0).collect();
        let mut out = BitVector::new(self.bits, Some(0));
        if seen.is_empty() {
            return out;
        }
        seen.sort_unstable();
        let reference = seen[((seen.len() - 1) as f32 * percentile.clamp(0.0, 1.0)) as usize];
        let limit = reference as f32 * ratio;
        for b in bits {
            let c = self.bit_counts.get(b).copied().unwrap_or(0);
            if b < self.bits && (c == 0 || c as f32 <= limit) {
                out.bit_set(b);
            }
        }
        out
    }

    /// Mean fraction of stored episodes containing each bit of `pattern`.
    pub fn frequency(&self, pattern: &BitVector) -> f32 {
        let bits = set_bits(pattern);
        if bits.is_empty() || self.stored == 0 {
            return 0.0;
        }
        bits.iter().map(|&b| self.bit_counts.get(b).copied().unwrap_or(0) as f32).sum::<f32>() / (bits.len() as f32 * self.stored as f32)
    }

    /// The stored episode overlapping `cue` the most (at least `min_overlap`
    /// bits), the most recent one on ties.
    pub fn recall(&self, cue: &BitVector, min_overlap: u32) -> Option<&BitVector> {
        let mut best: Option<(u32, usize)> = None;
        for (i, e) in self.episodes.iter().enumerate() {
            let o = overlap(e, cue);
            if o >= min_overlap && best.map_or(true, |(bo, _)| o >= bo) {
                best = Some((o, i));
            }
        }
        best.map(|(_, i)| &self.episodes[i])
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
        let novel = m.novel(&ep, 0.5);
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
        let cue = m.rarest(&bag(&[10, 11, 3]), 0.1, 1.5);
        assert_eq!(cue.as_words(), sym(3).as_words());
    }
}