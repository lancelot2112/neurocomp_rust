//! An index memory: one row per episode, recall by winner-take-all. This is the bit
//! version of a modern Hopfield network (Ramsauer et al. 2020) and of hippocampal
//! indexing theory (Teyler & DiScenna 1986).
//!
//! **The structure.** Each memory gets its own row: an index unit that holds its pattern
//! explicitly, as a list of active input indices (`keys`) and the pattern to give back
//! (`out`).
//! - **Store:** a new row (one write, no gradual learning).
//! - **Recall:** the row whose keys best match the cue wins, and its `out` is emitted.
//!   Winner-take-all is the softmax at zero temperature.
//! - **Capacity:** storage grows by one row per episode. The number of patterns that stay
//!   separable grows exponentially with the code's width, because each row competes as a
//!   whole and crosstalk never builds up in shared weights.
//!
//! **Event-based recall.** An inverted index maps each input index to the rows that
//! contain it. Each cue index adds its weight to the counters of those rows, and the best
//! row wins, so work scales with the cue's spikes, not with rows × width.
//! - **Weighting:** a cue index used by n rows votes 1/n (inverse document frequency,
//!   from the reciprocal table, no division). A common binding then barely moves the
//!   vote.
//! - **Cap:** indices in more than `cap` rows are skipped, since their vote is near zero
//!   and they cost the most.
//!
//! **Sequences.** Each row points to the row stored after it (until `end_sequence`), so
//! one cue can play out a whole episode (`recall_sequence`).
//!
//! **Forgetting.** Each row has a strength: set high on write (higher for a novel episode),
//! bumped on each recall, and lowered by one every `period` stores (every
//! `period / 4` once consolidated). It is computed lazily from the time of the last touch.
//! A row at zero is gone, and a periodic sweep removes it from the index. Rehearsed
//! memories survive and unused ones fade.
//!
//! **Replay.** Offline, rows are sampled by priority, their strength × an external
//! weight (e.g. how badly the cortex predicts them). Each is emitted with its successor
//! chain.
//!
//! **Consolidation.** Once the cortex reconstructs a row's content on its own,
//! `mark_consolidated` lets it decay four times faster.

use std::cell::{Cell, RefCell};

use rand::{Rng, RngCore};

use super::hippocampal_circuit::{EpisodicCircuit, Recall};
use crate::fixed::{div_round, ratio, recip32, Q16, ONE};

#[derive(Clone, Debug)]
pub struct IndexConfig {
    /// Cue indices in more than this many live rows are skipped at recall.
    pub cap: usize,
    /// Weight each cue index by 1/n (n = rows containing it); else every index counts 1.
    pub inverse: bool,
    /// The winner must share at least this many indices with the cue.
    pub min_overlap: u32,
    /// Strength on write, plus `novelty_bonus` × novelty; capped at 255.
    pub write_strength: u8,
    pub novelty_bonus: u8,
    /// Strength added on each recall.
    pub bump: u8,
    /// Stores per unit of strength lost (consolidated rows: a quarter of this).
    pub period: u32,
    /// Link each stored row to the next one (sequences).
    pub link: bool,
    /// The place code (0: off): each episode (between `end_sequence` calls) has its own
    /// `place_bits` input indices, drawn from `place_pool` ids starting at `place_base`.
    /// Stored events include the current place's indices, and so does every cue: place
    /// cells are more input, and the inverse weighting makes them count where this episode
    /// has events (they are rare) and nowhere else (a new place matches nothing).
    pub place_bits: usize,
    pub place_pool: usize,
    pub place_base: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self { cap: 4096, inverse: true, min_overlap: 16, write_strength: 128, novelty_bonus: 127, bump: 16, period: 64, link: true, place_bits: 0, place_pool: 4096, place_base: 0 }
    }
}

struct Row {
    keys: Vec<u32>,
    out: Vec<usize>,
    strength: Cell<u8>,
    /// Store clock at the last change of `strength`.
    touched: Cell<u32>,
    next: Option<u32>,
    consolidated: Cell<bool>,
    dead: bool,
}

pub struct IndexMemory {
    pub cfg: IndexConfig,
    rows: Vec<Row>,
    postings: Vec<Vec<u32>>,
    now: u32,
    last: Option<u32>,
    live: usize,
    evicted: usize,
    tags: Vec<(Vec<u32>, Vec<usize>)>,
    novelty_sum: (u64, usize),
    scores: RefCell<(Vec<u64>, Vec<u32>, Vec<u32>)>, // per-row score, overlap, touched rows
    recalls: Cell<usize>,
    work: Cell<u64>,
    /// The current episode's number (the place code's seed).
    place: u64,
}

impl IndexMemory {
    pub fn new(cfg: IndexConfig) -> Self {
        Self {
            cfg,
            rows: Vec::new(),
            postings: Vec::new(),
            now: 0,
            last: None,
            live: 0,
            evicted: 0,
            tags: Vec::new(),
            novelty_sum: (0, 0),
            scores: RefCell::new((Vec::new(), Vec::new(), Vec::new())),
            recalls: Cell::new(0),
            work: Cell::new(0),
            place: 0,
        }
    }

    /// The current place's input indices (empty when the place code is off).
    fn place_ids(&self) -> Vec<usize> {
        let mut v: Vec<usize> = (0..self.cfg.place_bits as u64)
            .map(|j| {
                let h = (self.place.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (j + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9)).rotate_left(17);
                self.cfg.place_base + (h % self.cfg.place_pool.max(1) as u64) as usize
            })
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// `cue` with the current place's indices.
    fn with_place(&self, cue: &[usize]) -> Vec<usize> {
        if self.cfg.place_bits == 0 {
            return cue.to_vec();
        }
        let mut c = cue.to_vec();
        c.extend(self.place_ids());
        c
    }

    /// Every live row sharing an index with `cue` (place included), with its weighted score.
    fn scored(&self, cue: &[usize]) -> Vec<(u64, u32)> {
        let cue = self.with_place(cue);
        let mut acc: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
        for &i in &cue {
            let n = self.n(i);
            if n == 0 || n > self.cfg.cap {
                continue;
            }
            let w = if self.cfg.inverse { recip32(n as u64) >> 16 } else { ONE as u64 };
            for &r in &self.postings[i] {
                *acc.entry(r).or_default() += w;
            }
        }
        acc.into_iter().filter(|&(r, _)| self.strength(&self.rows[r as usize]) > 0).map(|(r, s)| (s, r)).collect()
    }

    /// A row's strength now (0 = forgotten).
    fn strength(&self, r: &Row) -> u8 {
        if r.dead {
            return 0;
        }
        let period = if r.consolidated.get() { (self.cfg.period / 4).max(1) } else { self.cfg.period.max(1) };
        let lost = (self.now - r.touched.get()) / period;
        r.strength.get().saturating_sub(lost.min(255) as u8)
    }

    fn n(&self, i: usize) -> usize {
        self.postings.get(i).map_or(0, |p| p.len())
    }

    /// The best live row for `cue`: (row, weighted score, overlap), and the cue's own
    /// weight (what a row containing all of it would score).
    fn best(&self, cue: &[usize]) -> (Option<(u32, u64, u32)>, u64) {
        let mut guard = self.scores.borrow_mut();
        let (score, overlap, touched) = &mut *guard;
        if score.len() < self.rows.len() {
            score.resize(self.rows.len(), 0);
            overlap.resize(self.rows.len(), 0);
        }
        let mut own = 0u64;
        let mut work = 0u64;
        for &i in cue {
            let n = self.n(i);
            if n > self.cfg.cap {
                continue;
            }
            let w = if self.cfg.inverse { recip32(n.max(1) as u64) >> 16 } else { ONE as u64 };
            // an index no row holds counts fully in the cue's own weight: it is what is new
            own += w;
            if n == 0 {
                continue;
            }
            for &r in &self.postings[i] {
                let ri = r as usize;
                if score[ri] == 0 && overlap[ri] == 0 {
                    touched.push(r);
                }
                score[ri] += w;
                overlap[ri] += 1;
            }
            work += n as u64;
        }
        self.work.set(self.work.get() + work);
        let mut best: Option<(u32, u64, u32)> = None;
        for &r in touched.iter() {
            let ri = r as usize;
            let (s, o) = (score[ri], overlap[ri]);
            score[ri] = 0;
            overlap[ri] = 0;
            if o < self.cfg.min_overlap || self.strength(&self.rows[ri]) == 0 {
                continue;
            }
            // highest score; ties to the most recent row
            if best.map_or(true, |(br, bs, _)| s > bs || (s == bs && r > br)) {
                best = Some((r, s, o));
            }
        }
        touched.clear();
        (best, own)
    }

    fn recall_row(&self, cue: &[usize], bump: bool) -> Recall {
        let cue = self.with_place(cue);
        let (best, own) = self.best(&cue);
        let Some((r, s, o)) = best else { return Recall::default() };
        let row = &self.rows[r as usize];
        if bump {
            let now = self.strength(row);
            row.strength.set(now.saturating_add(self.cfg.bump));
            row.touched.set(self.now);
        }
        Recall { ec: row.out.clone(), strength: o * 16, ca1_match: ratio(s, own.max(1)).min(ONE), ca3: vec![r], ca1: vec![r] }
    }

    fn emit(&self, r: u32) -> Recall {
        match self.rows.get(r as usize) {
            Some(row) if self.strength(row) > 0 => Recall { ec: row.out.clone(), strength: row.keys.len() as u32 * 16, ca1_match: ONE, ca3: vec![r], ca1: vec![r] },
            _ => Recall::default(),
        }
    }

    /// The row stored after `row` in its sequence (empty if none, or forgotten).
    pub fn successor(&self, row: u32) -> Recall {
        match self.rows.get(row as usize).and_then(|r| r.next) {
            Some(n) => self.emit(n),
            None => Recall::default(),
        }
    }

    /// Reconsolidation: give `row` a new output (what it points to), as when a recalled
    /// memory is re-stored with what the cortex now completes it to. Its keys are unchanged.
    pub fn reconsolidate(&mut self, row: u32, out: &[usize]) {
        if let Some(r) = self.rows.get_mut(row as usize) {
            if !r.dead {
                let mut o = out.to_vec();
                o.sort_unstable();
                o.dedup();
                r.out = o;
            }
        }
    }

    /// The successor of `row` and how many of its keys `cue` shares (None if it has none, or
    /// it is forgotten).
    pub fn successor_match(&self, row: u32, cue: &[usize]) -> Option<(u32, u32)> {
        let n = self.rows.get(row as usize)?.next?;
        let r = &self.rows[n as usize];
        if self.strength(r) == 0 {
            return None;
        }
        Some((n, cue.iter().filter(|&&c| r.keys.binary_search(&(c as u32)).is_ok()).count() as u32))
    }

    /// The live rows sharing an index with `cue`, with their weighted scores, best first (at
    /// most `max`): the whole completion, not only its winner.
    pub fn matches(&self, cue: &[usize], max: usize) -> Vec<(u32, u64)> {
        let mut v: Vec<(u32, u64)> = self.scored(cue).into_iter().map(|(sc, r)| (r, sc)).collect();
        v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(b.0.cmp(&a.0)));
        v.truncate(max);
        v
    }

    /// The keys of `row` (empty if forgotten): its context, and whatever else it was stored with.
    pub fn keys_of(&self, row: u32) -> &[u32] {
        self.rows.get(row as usize).filter(|r| !r.dead).map_or(&[], |r| &r.keys[..])
    }

    /// What `row` points to now (empty if forgotten).
    pub fn out_of(&self, row: u32) -> &[usize] {
        self.rows.get(row as usize).filter(|r| !r.dead).map_or(&[], |r| &r.out[..])
    }

    /// Rows stored so far (live or forgotten).
    pub fn rows(&self) -> usize {
        self.rows.len()
    }

    /// Recall from `cue`, then follow the successor pointers: up to `max` patterns.
    pub fn recall_sequence(&self, cue: &[usize], max: usize) -> Vec<Recall> {
        let first = self.recall_row(cue, true);
        let mut out = Vec::new();
        let mut next = first.ca3.first().and_then(|&r| self.rows[r as usize].next);
        if first.ec.is_empty() {
            return out;
        }
        out.push(first);
        while let Some(r) = next {
            if out.len() >= max {
                break;
            }
            let e = self.emit(r);
            if e.ec.is_empty() {
                break;
            }
            out.push(e);
            next = self.rows[r as usize].next;
        }
        out
    }

    /// Replay: one row sampled with probability ∝ strength × `weight(row)` (e.g. the
    /// cortex's error on it), with its successor chain (up to `max`).
    pub fn replay_prioritized(&self, rng: &mut dyn RngCore, weight: impl Fn(u32) -> u64, max: usize) -> Vec<Recall> {
        let pri: Vec<u64> = self.rows.iter().enumerate().map(|(i, r)| self.strength(r) as u64 * weight(i as u32)).collect();
        let total: u64 = pri.iter().sum();
        if total == 0 {
            return Vec::new();
        }
        let mut x = rng.gen_range(0..total);
        let mut pick = 0;
        for (i, &p) in pri.iter().enumerate() {
            if x < p {
                pick = i;
                break;
            }
            x -= p;
        }
        let mut out = vec![self.emit(pick as u32)];
        let mut next = self.rows[pick].next;
        while let (Some(r), true) = (next, out.len() < max) {
            let e = self.emit(r);
            if e.ec.is_empty() {
                break;
            }
            out.push(e);
            next = self.rows[r as usize].next;
        }
        out
    }

    /// Remove forgotten rows from the index.
    fn sweep(&mut self) {
        let dead: Vec<bool> = self.rows.iter().map(|r| self.strength(r) == 0).collect();
        let mut n = 0;
        for (r, &d) in self.rows.iter_mut().zip(&dead) {
            if d && !r.dead {
                r.dead = true;
                r.keys = Vec::new();
                r.out = Vec::new();
                n += 1;
            }
        }
        if n > 0 {
            for p in self.postings.iter_mut() {
                p.retain(|&r| !dead[r as usize]);
            }
            self.evicted += n;
            self.live -= n;
        }
    }

    /// Live rows, rows evicted, and cue-index postings visited per recall.
    pub fn report(&self) -> (usize, usize, u64) {
        (self.live, self.evicted, self.work.get() / self.recalls.get().max(1) as u64)
    }
}

impl EpisodicCircuit for IndexMemory {
    fn recall(&self, cue: &[usize]) -> Recall {
        self.recalls.set(self.recalls.get() + 1);
        self.recall_row(cue, true)
    }

    fn familiarity(&self, bits: &[usize]) -> u64 {
        if bits.is_empty() {
            return 0;
        }
        div_round(bits.iter().map(|&b| self.n(b) as u64).sum(), bits.len() as u64)
    }

    fn novelty(&self, x: &[usize]) -> Q16 {
        let r = self.recall_row(x, false);
        if r.ca3.is_empty() { ONE } else { ONE - r.ca1_match }
    }

    fn store(&mut self, x: &[usize]) -> Q16 {
        self.store_split(x, &[], x)
    }

    fn store_event(&mut self, content: &[usize], context: &[usize]) -> Q16 {
        let mut all: Vec<usize> = content.iter().chain(context).copied().collect();
        all.sort_unstable();
        all.dedup();
        self.store_split(content, context, &all)
    }

    fn store_split(&mut self, content: &[usize], context: &[usize], out: &[usize]) -> Q16 {
        if content.is_empty() {
            return 0;
        }
        let mut keys: Vec<u32> = content.iter().chain(context).map(|&i| i as u32).collect();
        let all: Vec<usize> = { let mut a: Vec<usize> = keys.iter().map(|&i| i as usize).collect(); a.sort_unstable(); a.dedup(); a };
        let novelty = self.novelty(&all);
        keys.extend(self.place_ids().into_iter().map(|i| i as u32));
        keys.sort_unstable();
        keys.dedup();
        self.novelty_sum.0 += novelty as u64;
        self.novelty_sum.1 += 1;
        let unseen: Vec<usize> = content.iter().copied().filter(|&b| self.n(b) == 0).collect();
        self.now += 1;
        let id = self.rows.len() as u32;
        if unseen.len() >= 16 && self.tags.len() < 4096 {
            self.tags.push((vec![id], unseen));
        }
        let strength = (self.cfg.write_strength as u64 + ((self.cfg.novelty_bonus as u64 * novelty as u64) >> 16)).min(255) as u8;
        let max = *keys.last().unwrap() as usize;
        if max >= self.postings.len() {
            self.postings.resize_with(max + 1, Vec::new);
        }
        for &k in &keys {
            self.postings[k as usize].push(id);
        }
        let mut out = out.to_vec();
        out.sort_unstable();
        out.dedup();
        self.rows.push(Row { keys, out, strength: Cell::new(strength), touched: Cell::new(self.now), next: None, consolidated: Cell::new(false), dead: false });
        if self.cfg.link {
            if let Some(l) = self.last {
                self.rows[l as usize].next = Some(id);
            }
            self.last = Some(id);
        }
        self.live += 1;
        if self.now % (self.cfg.period.max(1) * 16).max(64) == 0 {
            self.sweep();
        }
        novelty
    }

    fn advance_time(&mut self) {}

    fn end_sequence(&mut self) {
        self.last = None;
        self.place += 1;
    }

    fn last_row(&self) -> Option<u32> {
        (!self.rows.is_empty()).then(|| self.rows.len() as u32 - 1)
    }

    fn row_here(&self, row: u32) -> bool {
        let ids = self.place_ids();
        match (self.rows.get(row as usize), ids.first()) {
            (Some(r), Some(&p)) => r.keys.binary_search(&(p as u32)).is_ok(),
            _ => false,
        }
    }

    fn recall_here_all(&self, cue: &[usize]) -> (Vec<u32>, Vec<usize>) {
        let mut rows: Vec<u32> = self.scored(cue).into_iter().map(|x| x.1).filter(|&r| self.row_here(r)).collect();
        rows.sort_unstable_by(|a, b| b.cmp(a));
        let mut words: Vec<usize> = rows.iter().flat_map(|&r| self.rows[r as usize].out.clone()).collect();
        words.sort_unstable();
        words.dedup();
        (rows, words)
    }

    fn recall_soft(&self, cue: &[usize], _bonus: usize, max_rows: usize) -> (Vec<u32>, Vec<usize>) {
        // the place indices are part of the cue: this episode's events lead by their weight
        let sc = self.scored(cue);
        let Some(top) = sc.iter().map(|x| x.0).max() else { return (Vec::new(), Vec::new()) };
        let mut rows: Vec<u32> = sc.iter().filter(|x| x.0 * 4 >= top * 3).map(|x| x.1).collect();
        rows.sort_unstable_by(|a, b| b.cmp(a));
        rows.truncate(max_rows.max(1));
        let mut words: Vec<usize> = rows.iter().flat_map(|&r| self.rows[r as usize].out.clone()).collect();
        words.sort_unstable();
        words.dedup();
        (rows, words)
    }

    fn mark_consolidated(&self, start: &[u32]) {
        for &r in start {
            if let Some(row) = self.rows.get(r as usize) {
                if !row.consolidated.get() {
                    // keep its current strength; from now on it fades four times faster
                    row.strength.set(self.strength(row));
                    row.touched.set(self.now);
                    row.consolidated.set(true);
                }
            }
        }
    }

    fn take_tags(&mut self) -> Vec<(Vec<u32>, Vec<usize>)> {
        std::mem::take(&mut self.tags)
    }

    fn replay_from(&self, start: &[u32]) -> Recall {
        start.first().map_or_else(Recall::default, |&r| self.emit(r))
    }

    fn replay(&self, rng: &mut dyn RngCore) -> Recall {
        self.replay_prioritized(rng, |_| 1, 1).into_iter().next().unwrap_or_default()
    }

    fn len(&self) -> usize {
        self.live
    }

    fn stats(&self) -> (usize, usize, u64, usize) {
        (0, self.recalls.get(), self.novelty_sum.0, self.novelty_sum.1)
    }
    /// Keys as u32, outputs as indices (usize), the row header, and the postings.
    fn memory_bytes(&self) -> usize {
        let rows: usize = self.rows.iter().map(|r| 4 * r.keys.len() + 8 * r.out.len() + 24).sum();
        let postings: usize = self.postings.iter().map(|p| 4 * p.len() + 24).sum();
        rows + postings
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    fn word(w: usize) -> Vec<usize> {
        let mut rng = StdRng::seed_from_u64(w as u64 + 77);
        (0..2048).collect::<Vec<_>>().choose_multiple(&mut rng, 16).copied().collect()
    }

    fn episode(ws: &[usize]) -> Vec<usize> {
        let mut x: Vec<usize> = ws.iter().flat_map(|&w| word(w)).collect();
        x.sort_unstable();
        x.dedup();
        x
    }

    fn overlap(a: &[usize], b: &[usize]) -> usize {
        a.iter().filter(|x| b.contains(x)).count()
    }

    /// The place code: the same partial cue recalls this episode's event, not an earlier
    /// episode's similar one, and a new place matches nothing of its own.
    #[test]
    fn place_cells_prefer_this_episode() {
        let cfg = IndexConfig { place_bits: 16, place_base: 1 << 14, min_overlap: 4, ..IndexConfig::default() };
        let mut m = IndexMemory::new(cfg);
        let old = episode(&[1, 2, 3]); // "lucy went bedroom" in an earlier story
        m.store(&old);
        m.end_sequence();
        let here = episode(&[1, 4, 5]); // "lucy came" in this story
        m.store(&here);
        let r = m.recall(&episode(&[1, 2])); // the cue matches the old event's words better
        assert_eq!(r.ec, here, "this episode's event wins on its place cells");
        assert!(m.row_here(r.ca3[0]));
        m.end_sequence();
        let r = m.recall(&episode(&[1, 2]));
        assert_eq!(r.ec, old, "in a new place, content decides");
        assert!(!m.row_here(r.ca3[0]));
    }

    /// The one-shot test of the hippocampal circuit: one rare episode among many common
    /// ones, recalled from a cue that mixes common words with the rare name.
    #[test]
    fn recalls_a_one_shot_episode_among_common_ones() {
        let mut m = IndexMemory::new(IndexConfig::default());
        let mut rng = StdRng::seed_from_u64(9);
        for i in 0..351 {
            if i == 300 {
                m.store(&episode(&[0, 1, 50, 51]));
                continue;
            }
            let mut ws: Vec<usize> = (0..6).collect::<Vec<_>>().choose_multiple(&mut rng, 3).copied().collect();
            ws.push(10 + rng.gen_range(0..6));
            m.store(&episode(&ws));
        }
        let r = m.recall(&episode(&[0, 1, 50]));
        assert_eq!(overlap(&r.ec, &word(51)), 16);
        // novelty: a stored episode is familiar, one with new words is not
        assert!(m.novelty(&episode(&[0, 1, 50, 51])) < ONE / 10);
        assert!(m.novelty(&episode(&[0, 1, 60, 61])) > ONE / 3);
    }

    #[test]
    fn a_cue_plays_out_its_sequence() {
        let mut m = IndexMemory::new(IndexConfig::default());
        let story: Vec<Vec<usize>> = (0..6).map(|i| episode(&[100 + 2 * i, 101 + 2 * i])).collect();
        for e in &story {
            m.store(e);
        }
        m.end_sequence();
        m.store(&episode(&[300, 301]));
        let seq = m.recall_sequence(&word(102), 10);
        assert_eq!(seq.len(), 5, "from the second event to the end of its sequence");
        for (r, e) in seq.iter().zip(&story[1..]) {
            assert_eq!(&r.ec, e);
        }
    }

    #[test]
    fn unused_memories_fade_and_rehearsed_ones_survive() {
        let cfg = IndexConfig { write_strength: 20, novelty_bonus: 0, bump: 20, period: 10, min_overlap: 8, ..IndexConfig::default() };
        let mut m = IndexMemory::new(cfg);
        let a = episode(&[1, 2]);
        let b = episode(&[3, 4]);
        m.store(&a);
        m.store(&b);
        for i in 0..400 {
            m.store(&episode(&[1000 + 2 * i, 1001 + 2 * i]));
            if i % 50 == 0 {
                m.recall(&word(1)); // rehearse a
            }
        }
        assert_eq!(m.recall(&word(1)).ec, a, "rehearsed");
        assert!(m.recall(&word(3)).ec.is_empty(), "unused: forgotten");
        assert!(m.report().1 > 0, "forgotten rows are evicted");
    }

    #[test]
    fn consolidated_rows_fade_faster() {
        let cfg = IndexConfig { write_strength: 40, novelty_bonus: 0, bump: 0, period: 10, min_overlap: 8, ..IndexConfig::default() };
        let mut m = IndexMemory::new(cfg);
        m.store(&episode(&[1, 2]));
        m.store(&episode(&[3, 4]));
        m.mark_consolidated(&[0]);
        for i in 0..200 {
            m.store(&episode(&[1000 + 2 * i, 1001 + 2 * i]));
        }
        assert!(m.recall(&word(1)).ec.is_empty(), "consolidated: gone after 4 × 10 × 40 / 4 stores");
        assert!(!m.recall(&word(3)).ec.is_empty(), "not consolidated: still held");
    }
}
