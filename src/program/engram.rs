//! An engram store: an index memory over compact rows of binding ids and grid phases.
//!
//! Each row is one episode:
//! - **what:** a few sorted binding ids (a word in a slot is one id), not bit fields;
//! - **where:** one phase per grid module, the place (here, position in the stream of
//!   stories) at which it was stored;
//! - **strength** and **last touch** (event clock), for lazy forgetting;
//! - **time:** append order, so "the next event of the same episode" is the next row with
//!   the same place, and no successor pointer or context field is needed.
//!
//! **Two indexes.**
//! - **What:** a posting list per binding id, giving the rows that contain it. Its
//!   length is M·K/N on average, so sparse ids keep it short.
//! - **Where:** a hash from the phase tuple to its rows, so exact-place lookup is O(1).
//!
//! **Event handlers.**
//! - **Move** (`move_by`): add a displacement to the clock and recompute the G phases.
//!   O(G), and touches no rows.
//! - **Recall:** cue ids are processed rarest first. Each one walks its posting list and
//!   adds its weight (1/n) to a counter per row, through a touched-list so the reset is
//!   cheap. The walk stops once the leader cannot be overtaken by the remaining weight.
//!   Rows from the current place get a fixed bonus, so location disambiguates similar
//!   content.
//! - **Store (a theta cycle):** the first half recalls with the event as cue. In the second
//!   half, if a row holds exactly this content, it is strengthened and touched (no
//!   duplicate is stored). Otherwise a new row is appended and indexed, which costs O(K).
//!
//! **Forgetting is lazy.** Effective strength is strength − (now − last touch) / τ,
//! computed on access, with no global tick. The ring buffer's write head is the evictor:
//! when it reaches a row that is still strong, the row gets a second chance (it is copied
//! to the head and the head moves on, at most `second_chances` times per write);
//! otherwise it is overwritten. Posting entries of overwritten rows are tombstones,
//! dropped when their list is next written.
//!
//! **Replay.** At `begin_sleep`, rows are ordered by strength × the cortex's error on
//! them, and replays pop from that order. The harness reports the error after each
//! replay (`report_error`); unknown rows count as fully unknown. Replayed events are
//! flagged for training the cortex and are not stored again.

use std::cell::{Cell, RefCell};

use rand::RngCore;

use super::hippocampal_circuit::{EpisodicCircuit, Recall};
use crate::det::HashMap;
use crate::fixed::{div_round, ratio, recip32, Q16, ONE};

/// What a theta cycle does with an event whose content a row already holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dedup {
    /// Strengthen that row wherever it was stored (one row per distinct content).
    Any,
    /// Strengthen it and move it to the current place (it was last seen here).
    Move,
    /// Only a row at the current place counts as the same event; elsewhere, store anew.
    Place,
}

#[derive(Clone, Debug)]
pub struct EngramConfig {
    /// Ring capacity (rows).
    pub capacity: usize,
    /// Grid modules: each module's phase is ⌊clock / scale⌋ mod 256.
    pub scales: Vec<u32>,
    /// Weight each cue id by 1/n (n = rows holding it).
    pub inverse: bool,
    /// The winner must share at least this many ids with the cue.
    pub min_overlap: u32,
    /// Score bonus (in units of one fully weighted id, `Q16`) for rows from the current
    /// place.
    pub where_bonus: Q16,
    pub write_strength: u8,
    pub novelty_bonus: u8,
    pub bump: u8,
    /// Events per unit of strength lost (τ).
    pub tau: u32,
    /// A row at or above this strength gets a second chance at the write head.
    pub keep: u8,
    pub second_chances: usize,
    /// New ids needed to tag an event as novel.
    pub tag_min: usize,
    /// Inference from every statement of a fact: a stored world event that holds a content
    /// word held by at most this many rows is a fact to infer from, and that word can be
    /// the inference's new word N. 1 (default): only words never seen before (the first
    /// statement only).
    pub infer_rare: usize,
    pub dedup: Dedup,
    /// Make the current place's rows candidates even when they share no id with the cue.
    pub seed_place: bool,
    /// Ids are field × `word_space` + word; fields below `content_fields` are content
    /// (above: context). Used to index rows by word, in any slot, for the walk.
    pub word_space: usize,
    pub content_fields: usize,
    /// Inferred events also in a variant with the partner word dropped (`infer_from`).
    pub infer_drop: bool,
    /// Recall walks one step (`recall_walk`) when the winning row was reached through a
    /// cue id held by at most `walk_rare` rows.
    pub walk: bool,
    pub walk_rare: usize,
}

impl Default for EngramConfig {
    fn default() -> Self {
        Self {
            capacity: 1 << 16,
            scales: vec![1, 4, 16, 64],
            inverse: true,
            min_overlap: 1,
            where_bonus: ONE,
            write_strength: 128,
            novelty_bonus: 127,
            bump: 16,
            tau: 64,
            keep: 32,
            second_chances: 8,
            tag_min: 1,
            infer_rare: 1,
            dedup: Dedup::Move,
            seed_place: false,
            word_space: 4096,
            content_fields: 64,
            walk: false,
            walk_rare: 2,
            infer_drop: true,
        }
    }
}

#[derive(Clone)]
struct Row {
    serial: u32,
    what: Vec<u32>,
    /// The content ids in the order they were read.
    order: Vec<u32>,
    phase: Vec<u8>,
    /// `place_key(&phase)`, kept with the row (recall compares it on every visit).
    place: u64,
    out: Vec<usize>,
    time: u32,
    /// The cortex's last reported error on this row (`Q16`; `ONE` = unknown).
    err: Cell<Q16>,
    /// Where the event came from (source memory): 0 the world (read or heard), 1 the
    /// network itself (said, retold).
    source: u8,
    /// Testimonies: in how many different episodes the event was stated (a repeat in the
    /// same episode is not another testimony).
    testimony: Cell<u16>,
}

/// A row's fields that recall checks on every visit, kept apart from the row (one small
/// array, indexed like the ring) so scoring touches a few bytes per posting, not the row.
struct Hot {
    /// The row's serial; `u32::MAX` for an empty slot.
    serial: u32,
    strength: Cell<u8>,
    touched: Cell<u32>,
    source: u8,
    /// The row's `place` key.
    place: u64,
}

impl Hot {
    fn empty() -> Self {
        Self { serial: u32::MAX, strength: Cell::new(0), touched: Cell::new(0), source: 0, place: 0 }
    }
}

pub struct EngramStore {
    pub cfg: EngramConfig,
    ring: Vec<Option<Row>>,
    hot: Vec<Hot>,
    head: u32,
    serial: u32,
    clock: u32,
    phase: Vec<u8>,
    now: u32,
    what: Vec<Vec<u32>>,
    /// Word → rows holding it in any content slot (for the walk).
    words: Vec<Vec<u32>>,
    walks: Cell<usize>,
    /// New facts since the last `infer`: (row, its new ids).
    facts: Vec<(u32, Vec<usize>)>,
    last_row: Cell<Option<u32>>,
    /// The rows written or strengthened in the current episode (sequence), in order, and
    /// in the previous one.
    episode_rows: RefCell<Vec<u32>>,
    prev_episode_rows: Vec<u32>,
    /// The source the next stored events are tagged with, and the sources recall may
    /// return (a bit per source; all by default).
    source_now: u8,
    recall_mask: Cell<u8>,
    /// Rows stored from sources other than the world (0: rarity needs no filtering).
    others: usize,
    place: HashMap<u64, Vec<u32>>,
    live: usize,
    evicted: usize,
    deduped: usize,
    tags: Vec<(Vec<u32>, Vec<usize>)>,
    novelty_sum: (u64, usize),
    order: RefCell<Vec<u32>>,
    cursor: Cell<usize>,
    scratch: RefCell<(Vec<u64>, Vec<u32>, Vec<u32>, Vec<bool>)>,
    recalls: Cell<usize>,
    work: Cell<u64>,
}

fn place_key(phase: &[u8]) -> u64 {
    phase.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &p| (h ^ p as u64).wrapping_mul(0x100_0000_01b3))
}

impl EngramStore {
    pub fn new(cfg: EngramConfig) -> Self {
        let phase = vec![0; cfg.scales.len()];
        Self {
            ring: vec![None; cfg.capacity.max(1)],
            hot: (0..cfg.capacity.max(1)).map(|_| Hot::empty()).collect(),
            head: 0,
            serial: 0,
            clock: 0,
            phase,
            now: 0,
            what: Vec::new(),
            words: Vec::new(),
            walks: Cell::new(0),
            facts: Vec::new(),
            last_row: Cell::new(None),
            episode_rows: RefCell::new(Vec::new()),
            prev_episode_rows: Vec::new(),
            source_now: 0,
            recall_mask: Cell::new(u8::MAX),
            others: 0,
            place: HashMap::default(),
            live: 0,
            evicted: 0,
            deduped: 0,
            tags: Vec::new(),
            novelty_sum: (0, 0),
            order: RefCell::new(Vec::new()),
            cursor: Cell::new(0),
            scratch: RefCell::new((Vec::new(), Vec::new(), Vec::new(), Vec::new())),
            recalls: Cell::new(0),
            work: Cell::new(0),
            cfg,
        }
    }

    /// Move: add `d` to the clock and recompute each module's phase. O(G).
    pub fn move_by(&mut self, d: u32) {
        self.clock = self.clock.wrapping_add(d);
        for (p, &s) in self.phase.iter_mut().zip(&self.cfg.scales) {
            *p = ((self.clock / s.max(1)) % 256) as u8;
        }
    }

    fn slot(&self, serial: u32) -> usize {
        serial as usize % self.ring.len()
    }

    /// The live row with this serial, if it is still in the ring.
    fn row(&self, serial: u32) -> Option<&Row> {
        let k = self.slot(serial);
        if self.live_at(k, serial) { self.ring[k].as_ref() } else { None }
    }

    /// The row with this serial is in the ring and not faded (strength > 0), from the hot
    /// fields alone: (now − touched) / τ < strength, without the division.
    fn live_at(&self, k: usize, serial: u32) -> bool {
        let h = &self.hot[k];
        h.serial == serial && ((self.now - h.touched.get()) as u64) < h.strength.get() as u64 * self.cfg.tau.max(1) as u64
    }

    /// The hot fields of the row with this serial, if it is live.
    fn hot_live(&self, serial: u32) -> Option<&Hot> {
        let k = self.slot(serial);
        self.live_at(k, serial).then(|| &self.hot[k])
    }

    fn hot(&self, r: &Row) -> &Hot {
        &self.hot[self.slot(r.serial)]
    }

    fn strength(&self, r: &Row) -> u8 {
        let h = self.hot(r);
        let lost = (self.now - h.touched.get()) / self.cfg.tau.max(1);
        h.strength.get().saturating_sub(lost.min(255) as u8)
    }

    /// Strengthen a row on recall or restatement.
    fn bump_row(&self, r: &Row) {
        let h = self.hot(r);
        h.strength.set(self.strength(r).saturating_add(self.cfg.bump));
        h.touched.set(self.now);
    }

    fn n(&self, id: usize) -> usize {
        let Some(p) = self.what.get(id) else { return 0 };
        if self.others == 0 || self.recall_mask.get() == u8::MAX {
            return p.len();
        }
        // source memory: rarity among the rows recall may return only
        p.iter().filter(|&&s| self.hot_live(s).is_some_and(|h| self.recall_mask.get() >> h.source & 1 == 1)).count()
    }

    /// Rows holding word `w` in any content slot (among the sources recall may return).
    fn word_rows(&self, w: usize) -> usize {
        let Some(p) = self.words.get(w) else { return 0 };
        if self.others == 0 || self.recall_mask.get() == u8::MAX {
            return p.len();
        }
        p.iter().filter(|&&s| self.hot_live(s).is_some_and(|h| self.recall_mask.get() >> h.source & 1 == 1)).count()
    }

    fn weight(&self, n: usize) -> u64 {
        if self.cfg.inverse { recip32(n.max(1) as u64) >> 16 } else { ONE as u64 }
    }

    /// The best row for `cue` at the current place: (serial, score, overlap), and the
    /// cue's own weight.
    fn best(&self, cue: &[usize]) -> (Option<(u32, u64, u32)>, u64) {
        let mut ids: Vec<(usize, usize)> = cue.iter().map(|&i| (self.n(i), i)).collect();
        ids.sort_unstable(); // rarest first
        let own: u64 = ids.iter().map(|&(n, _)| self.weight(n)).sum();
        let mut remaining = ids.iter().filter(|x| x.0 > 0).map(|&(n, _)| self.weight(n)).sum::<u64>();
        let here = place_key(&self.phase);
        let bonus = |h: &Hot| if h.place == here { self.cfg.where_bonus as u64 } else { 0 };
        let mut guard = self.scratch.borrow_mut();
        let (score, overlap, touched, local) = &mut *guard;
        if score.len() < self.ring.len() {
            score.resize(self.ring.len(), 0);
            overlap.resize(self.ring.len(), 0);
            local.resize(self.ring.len(), false);
        }
        let (mut first, mut second) = (0u64, 0u64);
        let mut work = 0u64;
        // the current place's rows are candidates whatever they share with the cue (an
        // exact-place lookup in the where-index): they start at the place bonus
        if self.cfg.seed_place && self.cfg.where_bonus > 0 {
            if let Some(rows) = self.place.get(&here) {
                for &s in rows {
                    let Some(h) = self.hot_live(s) else { continue };
                    if self.recall_mask.get() >> h.source & 1 == 0 {
                        continue;
                    }
                    let k = self.slot(s);
                    if score[k] == 0 {
                        touched.push(s);
                        score[k] = bonus(h);
                        local[k] = true;
                    }
                }
            }
        }
        // MaxScore: once even the best a row not yet touched could reach (the remaining
        // weight and the place bonus) is below the runner-up, no such row can change the
        // leader, the runner-up or the answer (both only rise), so the posting lists are
        // left and only the rows still in reach are scored, by looking each remaining id up
        // in the row. Exact when every touched row is eligible (min_overlap <= 1).
        let mut reach: Option<Vec<u32>> = None;
        for &(n, id) in &ids {
            if n == 0 {
                continue;
            }
            // early exit: the leader holds the threshold and nothing can overtake it
            if first > 0 && first >= second + remaining + self.cfg.where_bonus as u64 {
                break;
            }
            if reach.is_none() && self.cfg.min_overlap <= 1 && touched.len() > 1 && second > 0 && remaining + (self.cfg.where_bonus as u64) < second {
                reach = Some(touched.iter().copied().filter(|&s| score[self.slot(s)] + remaining >= second).collect());
            }
            let w = self.weight(n);
            remaining -= w;
            // a row's score rises by w; the leader and runner-up follow (a lone row is not
            // its own runner-up)
            let bump = |k: usize, multi: bool, score: &mut Vec<u64>, overlap: &mut Vec<u32>, lead: &mut (u64, u64)| {
                score[k] += w;
                overlap[k] += 1;
                let v = score[k];
                if v > lead.0 {
                    if multi {
                        lead.1 = lead.1.max(lead.0);
                    }
                    lead.0 = v;
                } else if v > lead.1 {
                    lead.1 = v;
                }
            };
            let mut lead = (first, second);
            if let Some(rows) = reach.as_mut() {
                // rows in serial order, as in the posting lists
                for &s in rows.iter() {
                    let k = self.slot(s);
                    if self.ring[k].as_ref().is_some_and(|r| r.what.binary_search(&(id as u32)).is_ok()) {
                        bump(k, true, score, overlap, &mut lead);
                    }
                }
                (first, second) = lead;
                work += rows.len() as u64;
                rows.retain(|&s| score[self.slot(s)] + remaining >= second);
                continue;
            }
            for &s in &self.what[id] {
                let k = self.slot(s);
                if !self.live_at(k, s) {
                    continue;
                }
                let h = &self.hot[k];
                if self.recall_mask.get() >> h.source & 1 == 0 {
                    continue;
                }
                if score[k] == 0 {
                    touched.push(s);
                    score[k] = bonus(h);
                }
                bump(k, touched.len() > 1, score, overlap, &mut lead);
            }
            (first, second) = lead;
            work += n as u64;
        }
        self.work.set(self.work.get() + work);
        let mut best: Option<(u32, u64, u32)> = None;
        for &s in touched.iter() {
            let k = self.slot(s);
            let (v, o, here) = (score[k], overlap[k], local[k]);
            score[k] = 0;
            overlap[k] = 0;
            local[k] = false;
            if o < self.cfg.min_overlap && !here {
                continue;
            }
            if best.map_or(true, |(bs, bv, _)| v > bv || (v == bv && s > bs)) {
                best = Some((s, v, o));
            }
        }
        touched.clear();
        (best, own)
    }

    fn recall_row(&self, cue: &[usize], bump: bool) -> Recall {
        self.recall_best(self.best(cue), bump)
    }

    /// Recall from `best`'s answer for a cue (computed once, reused by the walk).
    fn recall_best(&self, (best, own): (Option<(u32, u64, u32)>, u64), bump: bool) -> Recall {
        let Some((s, v, o)) = best else { return Recall::default() };
        let r = self.row(s).unwrap();
        if bump {
            self.bump_row(r);
        }
        Recall { ec: r.out.clone(), strength: o * 512, ca1_match: ratio(v.min(own), own.max(1)).min(ONE), ca3: vec![s], ca1: vec![s] }
    }

    /// The words of a row's content ids (each once).
    fn content_words(&self, what: &[u32]) -> Vec<usize> {
        let ws = self.cfg.word_space.max(1);
        let mut v: Vec<usize> = what.iter().map(|&i| i as usize).filter(|&i| i / ws < self.cfg.content_fields).map(|i| i % ws).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// The best row (other than `exclude`) for context `ids` (slot-specific), required
    /// `cue_words` and bonus `words` (both matched in any content slot), each weighted 1/n:
    /// (serial, score, overlap). A row must hold at least `need` of the cue words: the
    /// bonus words only choose among rows that answer the cue.
    fn best_mixed(&self, ids: &[usize], cue_words: &[usize], words: &[usize], exclude: u32, need: u32) -> Option<(u32, u64, u32)> {
        let mut guard = self.scratch.borrow_mut();
        let (score, overlap, touched, _) = &mut *guard;
        if score.len() < self.ring.len() {
            score.resize(self.ring.len(), 0);
            overlap.resize(self.ring.len(), 0);
        }
        let empty = Vec::new();
        // overlap counts the cue words in its high bits, so `need` tests them
        let lists = ids
            .iter()
            .map(|&i| (self.what.get(i).unwrap_or(&empty), 1u32))
            .chain(cue_words.iter().map(|&w| (self.words.get(w).unwrap_or(&empty), 1u32 << 16)))
            .chain(words.iter().map(|&w| (self.words.get(w).unwrap_or(&empty), 1u32)));
        for (list, unit) in lists {
            if list.is_empty() {
                continue;
            }
            let w = self.weight(list.len());
            for &s in list {
                if s == exclude || self.row(s).is_none() {
                    continue;
                }
                let k = self.slot(s);
                if score[k] == 0 && overlap[k] == 0 {
                    touched.push(s);
                }
                score[k] += w;
                overlap[k] += unit;
            }
        }
        let mut best: Option<(u32, u64, u32)> = None;
        for &s in touched.iter() {
            let k = self.slot(s);
            let (v, o) = (score[k], overlap[k]);
            score[k] = 0;
            overlap[k] = 0;
            let (content, all) = (o >> 16, (o >> 16) + (o & 0xFFFF));
            if self.row(s).map_or(true, |r| self.recall_mask.get() >> r.source & 1 == 0) {
                continue;
            }
            if content >= need && all >= self.cfg.min_overlap && best.map_or(true, |(bs, bv, _)| v > bv || (v == bv && s > bs)) {
                best = Some((s, v, all));
            }
        }
        touched.clear();
        best
    }

    /// Recall with one step of a walk (the big loop): if the winning row is from another
    /// episode and was reached through a rare content id of the cue (held by at most
    /// `walk_rare` rows as a word, in any slot: e.g. a name stated once in another story),
    /// that id is a bridge. The row's other words replace it in the cue, matched in any
    /// content slot, and the cue recalls again: "tom went to the" → "tom is a smith" →
    /// {is, a, smith} + "went to the" + the story's context → a smiths' event in this
    /// season. Otherwise, as `recall`.
    pub fn recall_walk(&self, cue: &[usize], bump: bool) -> Recall {
        let found = self.best(cue);
        let Some((s, _, _)) = found.0 else { return Recall::default() };
        let r1 = self.row(s).unwrap();
        // a bridge: a rare content id of the cue, through a row from another episode (a
        // fact stated elsewhere)
        let ws = self.cfg.word_space.max(1);
        let elsewhere = r1.phase != self.phase;
        let bridge = cue
            .iter()
            .copied()
            .filter(|&i| {
                elsewhere
                    && i / ws < self.cfg.content_fields
                    && r1.what.binary_search(&(i as u32)).is_ok()
                    // rare as a word, in any slot (slot-specific ids fragment common words)
                    && self.word_rows(i % ws) <= self.cfg.walk_rare
            })
            .min_by_key(|&i| self.word_rows(i % ws));
        let Some(b) = bridge else { return self.recall_best(found, bump) };
        let bw = b % ws;
        // context ids stay slot-specific; the rest of the cue's content, as words in any
        // slot, must still be answered (at least half of it)
        let ctx: Vec<usize> = cue.iter().copied().filter(|&i| i / ws >= self.cfg.content_fields).collect();
        let cue_ids: Vec<u32> = cue.iter().map(|&i| i as u32).filter(|&i| i as usize != b).collect();
        let cue_words: Vec<usize> = self.content_words(&cue_ids).into_iter().filter(|&w| w != bw).collect();
        if cue_words.is_empty() {
            return self.recall_best(found, bump);
        }
        let words: Vec<usize> = self.content_words(&r1.what).into_iter().filter(|&w| w != bw && !cue_words.contains(&w)).collect();
        let need = (cue_words.len() as u32).div_ceil(2);
        let Some((s2, _, o2)) = self.best_mixed(&ctx, &cue_words, &words, s, need) else { return self.recall_best(found, bump) };
        self.walks.set(self.walks.get() + 1);
        let r2 = self.row(s2).unwrap();
        if bump {
            self.bump_row(r2);
        }
        Recall { ec: r2.out.clone(), strength: o2 * 512, ca1_match: 0, ca3: vec![s2], ca1: vec![s, s2] }
    }

    /// Inferred events from a new fact (generative replay): the walk run forward. The fact
    /// row (e.g. "lucy is a jones") links its new word N (a word no other row holds) to a
    /// partner F: among the rows of the same schema (sharing all but two content ids, "X is
    /// a Y ."), the fact's most variable word (a filler: jones, not the frame: is, a).
    /// Rows of that schema are not sources (they are facts of the same kind). Each row of another episode that holds F ("john jones went
    /// to the hallway", with that story's context) becomes an inferred event: its content in
    /// reading order with the word in N's slot replaced by N (or N put first), and F
    /// dropped ("lucy went to the hallway"); and the same keeping F. Returns up to
    /// `max_rows` source rows' events, most recent first, as (content ids in order, context
    /// ids, source row, fact row, partner word): the event, its two premises, and the word
    /// that links them (the fact's filler F).
    pub fn infer_from(&self, fact: u32, new: &[usize], max_rows: usize) -> Vec<(Vec<usize>, Vec<usize>, u32, u32, usize)> {
        let ws = self.cfg.word_space.max(1);
        let Some(r) = self.row(fact) else { return Vec::new() };
        let count = |w: usize| self.word_rows(w);
        // N: a new word (held by this row alone), not just a known word in a new slot
        let Some(&n_id) = new.iter().filter(|&&i| i / ws < self.cfg.content_fields && count(i % ws) <= self.cfg.infer_rare.max(1)).min_by_key(|&&i| count(i % ws)) else { return Vec::new() };
        let (n_word, n_slot) = (n_id % ws, n_id / ws);
        // the fact's schema: rows that share all but two of its content words, in any slot
        // ("X is a Y ."); slot ids vary with the learned roles, words do not
        let fact_words = self.content_words(&r.what);
        let mut shared: HashMap<u32, u32> = HashMap::default();
        for &w in &fact_words {
            for &t in self.words.get(w).map_or(&[][..], |l| &l[..]) {
                if t != fact && self.row(t).is_some() {
                    *shared.entry(t).or_default() += 1;
                }
            }
        }
        let need = (fact_words.len() as u32).saturating_sub(2).max(2);
        let similar: Vec<u32> = shared.iter().filter(|e| *e.1 >= need).map(|e| *e.0).collect();
        // F, the partner: the fact's most variable word across its schema (a filler like
        // "jones", not a frame word like "is" that every such row has), held elsewhere too
        let share = |w: usize| similar.iter().filter(|&&t| self.row(t).map_or(false, |x| self.content_words(&x.what).contains(&w))).count();
        let partner = fact_words
            .iter()
            .copied()
            .filter(|&w| w != n_word && count(w) > 1 && share(w) * 10 < similar.len().max(1) * 9)
            .min_by_key(|&w| (share(w), count(w)));
        let Some(f) = partner else { return Vec::new() };
        // sources: rows of other episodes holding F, outside the fact's own schema
        let mut sources: Vec<u32> = self.words.get(f).map_or(Vec::new(), |l| l.iter().copied().filter(|&t| t != fact && !similar.contains(&t)).collect());
        sources.sort_unstable_by(|a, b| b.cmp(a));
        let mut out = Vec::new();
        for t in sources.into_iter().filter_map(|t| self.row(t).map(|x| (t, x))).filter(|(_, x)| x.phase != r.phase).take(max_rows) {
            let row = t.1;
            let context: Vec<usize> = row.what.iter().map(|&i| i as usize).filter(|&i| i / ws >= self.cfg.content_fields).collect();
            for keep_partner in [false, true] {
                if !keep_partner && !self.cfg.infer_drop {
                    continue;
                }
                let mut seq: Vec<usize> = Vec::new();
                let mut placed = false;
                for &i in &row.order {
                    let i = i as usize;
                    let w = i % ws;
                    if i / ws == n_slot && !placed {
                        seq.push(n_id);
                        placed = true;
                    } else if w == f && !keep_partner {
                        continue;
                    } else {
                        seq.push(i);
                    }
                }
                if !placed {
                    seq.insert(0, n_id);
                }
                out.push((seq, context.clone(), t.0, fact, f));
            }
        }
        out
    }

    /// Walks taken.
    pub fn walks(&self) -> usize {
        self.walks.get()
    }

    fn emit(&self, s: u32) -> Recall {
        match self.row(s) {
            Some(r) => Recall { ec: r.out.clone(), strength: r.what.len() as u32 * 512, ca1_match: ONE, ca3: vec![s], ca1: vec![s] },
            None => Recall::default(),
        }
    }

    /// The next row of the same episode (same place, later time), if any.
    pub fn successor(&self, s: u32) -> Option<u32> {
        let r = self.row(s)?;
        let rows = self.place.get(&r.place)?;
        rows.iter().copied().filter_map(|t| self.row(t).map(|x| (x.time, t))).filter(|&(time, _)| time > r.time).min().map(|x| x.1)
    }

    /// Recall from `cue`, then play the episode forward: up to `max` rows.
    pub fn recall_sequence(&self, cue: &[usize], max: usize) -> Vec<Recall> {
        let first = self.recall_row(cue, true);
        let Some(&s) = first.ca3.first() else { return Vec::new() };
        let mut out = vec![first];
        let mut next = self.successor(s);
        while let (Some(t), true) = (next, out.len() < max) {
            out.push(self.emit(t));
            next = self.successor(t);
        }
        out
    }

    /// Append a row at the write head (evicting or giving second chances), and index it.
    fn append(&mut self, mut row: Row, strength: u8) -> u32 {
        let mut chances = 0;
        loop {
            let k = self.head as usize % self.ring.len();
            let keep = match &self.ring[k] {
                Some(old) => {
                    let st = self.strength(old);
                    st > 0 && st >= self.cfg.keep && chances < self.cfg.second_chances
                }
                None => false,
            };
            if keep {
                // second chance: the strong row is copied forward to a new serial at this
                // slot, and the head moves on
                chances += 1;
                let mut old = self.ring[k].take().unwrap();
                old.serial = self.serial;
                self.hot[k].serial = old.serial;
                self.serial += 1;
                for &id in &old.what {
                    self.what[id as usize].push(old.serial);
                }
                for w in self.content_words(&old.what) {
                    self.words[w].push(old.serial);
                }
                self.place.entry(old.place).or_default().push(old.serial);
                self.ring[k] = Some(old);
                self.head += 1;
                continue;
            }
            if let Some(old) = &self.ring[k] {
                if self.strength(old) > 0 {
                    self.evicted += 1;
                }
                self.live -= 1;
            }
            row.serial = self.serial;
            self.serial += 1;
            for &id in &row.what {
                let id = id as usize;
                if id >= self.what.len() {
                    self.what.resize_with(id + 1, Vec::new);
                }
                // drop tombstones while the list is being written
                let ring = &self.ring;
                let len = ring.len();
                self.what[id].retain(|&s| ring[s as usize % len].as_ref().map_or(false, |r| r.serial == s));
                self.what[id].push(row.serial);
            }
            for w in self.content_words(&row.what) {
                if w >= self.words.len() {
                    self.words.resize_with(w + 1, Vec::new);
                }
                let ring = &self.ring;
                let len = ring.len();
                self.words[w].retain(|&s| ring[s as usize % len].as_ref().map_or(false, |r| r.serial == s));
                self.words[w].push(row.serial);
            }
            let key = row.place;
            let ring = &self.ring;
            let len = ring.len();
            let bucket = self.place.entry(key).or_default();
            bucket.retain(|&s| ring[s as usize % len].as_ref().map_or(false, |r| r.serial == s));
            bucket.push(row.serial);
            let s = row.serial;
            self.hot[k] = Hot { serial: s, strength: Cell::new(strength), touched: Cell::new(self.now), source: row.source, place: row.place };
            self.ring[k] = Some(row);
            self.head += 1;
            self.live += 1;
            return s;
        }
    }

    /// Live rows, rows evicted, events stored by strengthening an existing row, and
    /// posting entries walked per recall.
    pub fn report(&self) -> (usize, usize, usize, u64) {
        (self.live, self.evicted, self.deduped, self.work.get() / self.recalls.get().max(1) as u64)
    }
}

impl EpisodicCircuit for EngramStore {
    fn recall(&self, cue: &[usize]) -> Recall {
        self.recalls.set(self.recalls.get() + 1);
        if self.cfg.walk { self.recall_walk(cue, true) } else { self.recall_row(cue, true) }
    }

    fn sequence_from(&self, cue: &[usize], max: usize) -> Vec<Vec<usize>> {
        let first = self.recall_row(cue, false);
        let Some(&s) = first.ca3.first() else { return Vec::new() };
        let mut out = Vec::new();
        let mut next = self.successor(s);
        while let (Some(t), true) = (next, out.len() < max) {
            if let Some(r) = self.row(t) {
                out.push(r.order.iter().map(|&i| i as usize).collect());
            }
            next = self.successor(t);
        }
        out
    }

    fn recent_episode(&self, max: usize) -> Vec<Vec<usize>> {
        // the episode being read, else the one before (a new sequence may just have begun)
        let cur = self.episode_rows.borrow();
        let rows: &[u32] = if cur.is_empty() { &self.prev_episode_rows } else { &cur };
        rows.iter().take(max).filter_map(|&t| self.row(t)).map(|r| r.order.iter().map(|&i| i as usize).collect()).collect()
    }

    fn set_source(&mut self, source: u8) {
        self.source_now = source.min(7);
    }

    fn set_recall_sources(&self, mask: u8) {
        self.recall_mask.set(mask);
    }

    fn row_source(&self, row: u32) -> Option<u8> {
        self.row(row).map(|r| r.source)
    }

    fn testimony(&self, row: u32) -> u32 {
        self.row(row).map_or(0, |r| r.testimony.get() as u32)
    }

    fn recall_peek(&self, cue: &[usize]) -> Recall {
        if self.cfg.walk { self.recall_walk(cue, false) } else { self.recall_row(cue, false) }
    }

    fn recall_as(&self, cue: &[usize], walk: bool) -> Recall {
        self.recalls.set(self.recalls.get() + 1);
        if walk { self.recall_walk(cue, true) } else { self.recall_row(cue, true) }
    }

    fn familiarity(&self, ids: &[usize]) -> u64 {
        if ids.is_empty() {
            return 0;
        }
        div_round(ids.iter().map(|&i| self.n(i) as u64).sum(), ids.len() as u64)
    }

    fn novelty(&self, x: &[usize]) -> Q16 {
        let r = self.recall_row(x, false);
        if r.ca3.is_empty() { ONE } else { ONE - r.ca1_match }
    }

    fn store(&mut self, x: &[usize]) -> Q16 {
        self.store_split(x, &[], x)
    }

    fn store_event(&mut self, content: &[usize], context: &[usize]) -> Q16 {
        self.store_split(content, context, content)
    }

    /// A theta cycle: recall with the event, then strengthen the row that holds exactly
    /// these ids, or append a new row. Context ids, if any, are stored with the content.
    fn store_split(&mut self, content: &[usize], context: &[usize], out: &[usize]) -> Q16 {
        if content.is_empty() {
            return 0;
        }
        self.now += 1;
        // what: the content ids, and any context ids the caller supplies
        let mut what: Vec<u32> = content.iter().chain(context).map(|&i| i as u32).collect();
        what.sort_unstable();
        what.dedup();
        let ids: Vec<usize> = what.iter().map(|&i| i as usize).collect();
        // an event merges only with a stored event of the same source
        let mask = self.recall_mask.replace(1 << self.source_now);
        let (best, own) = self.best(&ids);
        self.recall_mask.set(mask);
        let novelty = best.map_or(ONE, |(_, v, _)| ONE - ratio(v.min(own), own.max(1)).min(ONE));
        self.novelty_sum.0 += novelty as u64;
        self.novelty_sum.1 += 1;
        if let Some((s, _, _)) = best {
            let (same, here, old) = {
                let r = self.row(s).unwrap();
                let here = r.phase == self.phase;
                let same = r.what == what && (here || self.cfg.dedup != Dedup::Place);
                if same {
                    self.bump_row(r);
                    if !here {
                        r.testimony.set(r.testimony.get().saturating_add(1));
                    }
                }
                (same, here, r.place)
            };
            if same {
                self.deduped += 1;
                self.last_row.set(Some(s));
                self.episode_rows.borrow_mut().push(s);
                if !here && self.cfg.dedup == Dedup::Move {
                    // last seen here: re-index the row under the current place
                    let new = place_key(&self.phase);
                    let k = self.slot(s);
                    let (phase, now) = (self.phase.clone(), self.now);
                    if let Some(row) = self.ring[k].as_mut() {
                        row.phase = phase;
                        row.place = new;
                        row.time = now;
                    }
                    self.hot[k].place = new;
                    if let Some(b) = self.place.get_mut(&old) {
                        b.retain(|&t| t != s);
                    }
                    self.place.entry(new).or_default().push(s);
                }
                return novelty;
            }
        }
        let unseen: Vec<usize> = content.iter().copied().filter(|&i| self.n(i) == 0).collect();
        let strength = (self.cfg.write_strength as u64 + ((self.cfg.novelty_bonus as u64 * novelty as u64) >> 16)).min(255) as u8;
        let mut out = out.to_vec();
        out.sort_unstable();
        out.dedup();
        let order: Vec<u32> = content.iter().map(|&i| i as u32).collect();
        let row = Row {
            serial: 0,
            what,
            order,
            phase: self.phase.clone(),
            place: place_key(&self.phase),
            out,
            time: self.now,
            err: Cell::new(ONE),
            source: self.source_now,
            testimony: Cell::new(1),
        };
        self.others += (self.source_now != 0) as usize;
        let s = self.append(row, strength);
        self.last_row.set(Some(s));
        self.episode_rows.borrow_mut().push(s);
        // the network's own words are not new facts about the world
        if self.source_now == 0 && unseen.len() >= self.cfg.tag_min && self.tags.len() < 4096 {
            self.facts.push((s, unseen.clone()));
            self.tags.push((vec![s], unseen));
        } else if self.source_now == 0 && self.cfg.infer_rare > 1 && self.facts.len() < 4096 {
            // a later statement about a rare word ("lucy is a smith", after "lucy is a
            // jones"): a fact to infer from too
            let ws = self.cfg.word_space.max(1);
            let rare: Vec<usize> = content.iter().copied().filter(|&i| i / ws < self.cfg.content_fields && self.word_rows(i % ws) <= self.cfg.infer_rare).collect();
            if !rare.is_empty() {
                self.facts.push((s, rare));
            }
        }
        novelty
    }

    fn advance_time(&mut self) {}

    /// A story ended: move one step on the grid (the next story is a new place).
    fn recall_here(&self, cue: &[usize]) -> Option<(u32, Vec<usize>)> {
        let here = place_key(&self.phase);
        let mut best: Option<(u32, usize)> = None;
        for &s in self.place.get(&here)? {
            let Some(r) = self.row(s).filter(|r| r.place == here) else { continue };
            let o = cue.iter().filter(|&&i| r.what.binary_search(&(i as u32)).is_ok()).count();
            if o > 0 && best.map_or(true, |(bs, bo)| o > bo || (o == bo && s > bs)) {
                best = Some((s, o));
            }
        }
        let (s, _) = best?;
        Some((s, self.content_words(&self.row(s)?.what)))
    }

    fn recall_here_all(&self, cue: &[usize]) -> (Vec<u32>, Vec<usize>) {
        let here = place_key(&self.phase);
        let (mut rows, mut words) = (Vec::new(), Vec::new());
        for &s in self.place.get(&here).map_or(&[][..], |v| &v[..]) {
            let Some(r) = self.row(s).filter(|r| r.place == here) else { continue };
            if cue.iter().any(|&i| r.what.binary_search(&(i as u32)).is_ok()) {
                rows.push(s);
                words.extend(self.content_words(&r.what));
            }
        }
        words.sort_unstable();
        words.dedup();
        (rows, words)
    }

    fn end_sequence(&mut self) {
        self.prev_episode_rows = std::mem::take(&mut *self.episode_rows.borrow_mut());
        self.move_by(1);
    }

    fn mark_consolidated(&self, start: &[u32]) {
        self.report_error(start, 0);
    }

    fn report_error(&self, start: &[u32], err: Q16) {
        for &s in start {
            if let Some(r) = self.row(s) {
                r.err.set(err);
            }
        }
    }

    fn last_row(&self) -> Option<u32> {
        self.last_row.get()
    }

    fn infer(&mut self, max_rows: usize) -> Vec<(Vec<usize>, Vec<usize>, u32, u32, usize)> {
        let facts = std::mem::take(&mut self.facts);
        facts.iter().flat_map(|(r, new)| self.infer_from(*r, new, max_rows)).collect()
    }

    fn begin_sleep(&mut self) {
        let mut rows: Vec<(u64, u32)> = self
            .ring
            .iter()
            .flatten()
            .filter(|r| self.strength(r) > 0)
            .map(|r| (self.strength(r) as u64 * r.err.get().max(1) as u64, r.serial))
            .collect();
        rows.sort_unstable_by(|a, b| b.cmp(a));
        *self.order.borrow_mut() = rows.into_iter().map(|x| x.1).collect();
        self.cursor.set(0);
    }

    fn take_tags(&mut self) -> Vec<(Vec<u32>, Vec<usize>)> {
        std::mem::take(&mut self.tags)
    }

    fn replay_from(&self, start: &[u32]) -> Recall {
        start.first().map_or_else(Recall::default, |&s| self.emit(s))
    }

    /// The next row by priority (strength × cortex error) since `begin_sleep`.
    fn replay(&self, _rng: &mut dyn RngCore) -> Recall {
        let order = self.order.borrow();
        if order.is_empty() {
            return Recall::default();
        }
        let i = self.cursor.get();
        self.cursor.set(i + 1);
        self.emit(order[i % order.len()])
    }

    fn len(&self) -> usize {
        self.live
    }

    fn stats(&self) -> (usize, usize, u64, usize) {
        (0, self.recalls.get(), self.novelty_sum.0, self.novelty_sum.1)
    }
    /// Live rows (ids as u32, phases, outputs as indices, header), the what postings and
    /// the place index.
    fn report(&self) -> String {
        let (live, evicted, deduped, work) = EngramStore::report(self);
        format!("engram: {live} rows, {evicted} evicted, {deduped} events stored by strengthening a row, {work} postings walked per recall, {} walks", self.walks.get())
    }

    fn memory_bytes(&self) -> usize {
        let rows: usize = self.ring.iter().flatten().map(|r| 4 * r.what.len() + r.phase.len() + 8 * r.out.len() + 24).sum();
        let what: usize = self.what.iter().map(|p| 4 * p.len() + 24).sum();
        let place: usize = self.place.values().map(|p| 4 * p.len() + 32).sum();
        rows + what + place
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Binding id of word `w` in slot `s`.
    fn b(w: usize, s: usize) -> usize {
        s * 4096 + w
    }

    fn event(ws: &[usize]) -> Vec<usize> {
        ws.iter().enumerate().map(|(s, &w)| b(w, s)).collect()
    }

    #[test]
    fn recalls_and_deduplicates() {
        let mut m = EngramStore::new(EngramConfig::default());
        for i in 0..500 {
            m.store(&event(&[1, 2, 10 + i % 7]));
        }
        assert_eq!(m.len(), 7, "repeats strengthen their row instead of storing again");
        m.store(&event(&[1, 50, 51]));
        let r = m.recall(&event(&[1, 50]));
        assert_eq!(r.ec, event(&[1, 50, 51]));
    }

    #[test]
    fn source_memory_separates_what_was_read_from_what_was_said() {
        let mut m = EngramStore::new(EngramConfig::default());
        m.store(&event(&[1, 2, 3])); // read: "lucy went to the hall"
        m.set_source(1);
        m.store(&event(&[1, 2, 4])); // said: "lucy went to the yard"
        m.store(&event(&[1, 2, 3])); // said, the same as read: a row of its own
        m.set_source(0);
        assert_eq!(m.len(), 3, "an event merges only with an event of the same source");
        // untagged recall may return the network's own words
        let any = m.recall_peek(&event(&[1, 2, 4]));
        assert_eq!(any.ec, event(&[1, 2, 4]));
        assert_eq!(m.row_source(any.ca3[0]), Some(1));
        // reality monitoring: recall about the world returns only what was read
        m.set_recall_sources(1);
        let world = m.recall_peek(&event(&[1, 2, 4]));
        assert_eq!(world.ec, event(&[1, 2, 3]));
        assert_eq!(m.row_source(world.ca3[0]), Some(0));
        // and recall of one's own words returns only those
        m.set_recall_sources(2);
        assert_eq!(m.row_source(m.recall_peek(&event(&[1, 2])).ca3[0]), Some(1));
    }

    #[test]
    fn recall_here_finds_the_item_in_this_episode_only() {
        let mut m = EngramStore::new(EngramConfig::default());
        m.store(&event(&[5, 1, 2])); // an earlier story: the item with another fact
        m.end_sequence();
        m.store(&event(&[3, 4])); // this story: unrelated
        m.store(&event(&[5, 6, 7])); // this story: the item with its fact
        m.store(&event(&[8, 9]));
        let (row, words) = m.recall_here(&event(&[5])).expect("found here");
        assert_eq!(m.row(row).unwrap().what, event(&[5, 6, 7]).iter().map(|&i| i as u32).collect::<Vec<_>>());
        assert!(!words.is_empty());
        let (rows, _) = m.recall_here_all(&event(&[5]));
        assert_eq!(rows.len(), 1, "only this story's event with the item");
        m.end_sequence();
        assert!(m.recall_here(&event(&[5])).is_none(), "a new story has nothing bound to the item yet");
    }

    #[test]
    fn place_disambiguates_similar_content() {
        let mut m = EngramStore::new(EngramConfig::default());
        m.store(&event(&[1, 2, 3])); // "it is autumn" in story 0
        m.end_sequence();
        m.store(&event(&[1, 2, 4])); // "it is winter" in story 1
        m.end_sequence();
        m.store(&event(&[1, 2, 5])); // story 2
        let mut q = EngramStore::new(EngramConfig::default());
        q.store(&event(&[1, 2, 3]));
        q.end_sequence();
        q.store(&event(&[1, 2, 4]));
        q.end_sequence();
        q.store(&event(&[7, 8, 9]));
        // in story 2, "it is ..." recalls story 2's own fact
        assert_eq!(m.recall(&event(&[1, 2])).ec, event(&[1, 2, 5]));
        // with no fact here, the most recent other one
        assert_eq!(q.recall(&event(&[1, 2])).ec, event(&[1, 2, 4]));
    }

    #[test]
    fn an_episode_plays_forward() {
        let mut m = EngramStore::new(EngramConfig::default());
        for e in 0..5 {
            m.store(&event(&[100 + e, 200 + e]));
        }
        m.end_sequence();
        m.store(&event(&[300, 301]));
        let seq = m.recall_sequence(&[b(101, 0)], 10);
        assert_eq!(seq.len(), 4);
        assert_eq!(seq[3].ec, event(&[104, 204]));
    }

    #[test]
    fn the_write_head_evicts_weak_rows_and_keeps_strong_ones() {
        let cfg = EngramConfig { capacity: 64, tau: 4, keep: 32, write_strength: 40, novelty_bonus: 0, bump: 64, ..EngramConfig::default() };
        let mut m = EngramStore::new(cfg);
        m.store(&event(&[1, 2]));
        for i in 0..300 {
            m.store(&event(&[1000 + i, 2000 + i]));
            if i % 20 == 0 {
                m.recall(&event(&[1, 2])); // rehearse
            }
        }
        assert_eq!(m.recall(&event(&[1, 2])).ec, event(&[1, 2]), "rehearsed row survives many laps");
        assert!(m.recall(&event(&[1000, 2000])).ec.is_empty(), "an old unrehearsed row is gone");
        assert!(m.len() <= 64);
    }

    #[test]
    fn replay_follows_strength_times_error() {
        let mut m = EngramStore::new(EngramConfig { novelty_bonus: 0, ..EngramConfig::default() });
        for e in 0..3 {
            m.store(&event(&[10 + e, 20 + e]));
        }
        let rows: Vec<u32> = (0..3).map(|e| m.recall(&event(&[10 + e])).ca3[0]).collect();
        m.report_error(&rows[0..1], ONE / 10); // the cortex knows row 0
        m.report_error(&rows[1..2], ONE);
        m.report_error(&rows[2..3], ONE / 2);
        m.begin_sleep();
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(0);
        let order: Vec<u32> = (0..3).map(|_| m.replay(&mut rng).ca3[0]).collect();
        assert_eq!(order, vec![rows[1], rows[2], rows[0]]);
    }
    /// "tom is a smith" stated once; smiths go to the kitchen in autumn and the garden in
    /// winter (other stories); "tom went to the" in an autumn story walks through "tom".
    #[test]
    fn a_walk_composes_a_stated_fact_with_a_rule() {
        let ctx = |w: usize| b(w, 64); // a context binding: the season, stated earlier
        let (tom, is, a, smith, john, went, to, the, kitchen, garden, autumn, winter, dot) = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
        let mut m = EngramStore::new(EngramConfig { walk: true, ..EngramConfig::default() });
        for i in 0..20 {
            let (season, place) = if i % 2 == 0 { (autumn, kitchen) } else { (winter, garden) };
            let mut e = event(&[john, smith, went, to, the, place, dot]);
            e.push(ctx(season));
            m.store(&e);
            m.end_sequence();
        }
        let mut st = event(&[tom, is, a, smith, dot]);
        st.push(ctx(winter));
        m.store(&st);
        m.end_sequence();
        // the question: tom went to the ___, in an autumn story
        let mut q = event(&[tom, went, to, the]);
        q.push(ctx(autumn));
        let r = m.recall(&q);
        assert!(r.ec.contains(&b(kitchen, 5)), "walked to a smiths' autumn event: {:?}", r.ec);
        assert_eq!(m.walks(), 1);
        // without the walk, the one row holding "tom" wins and has no place in it
        let plain = m.recall_row(&q, false);
        assert!(!plain.ec.iter().any(|&i| i % 4096 == kitchen || i % 4096 == garden));
    }
    #[test]
    fn a_new_fact_yields_inferred_events() {
        let ctx = |w: usize| b(w, 64);
        let (tom, is, a, smith, john, went, to, the, kitchen, autumn, dot) = (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13);
        let mut m = EngramStore::new(EngramConfig::default());
        for _ in 0..3 {
            let mut e = event(&[john, smith, went, to, the, kitchen, dot]);
            e.push(ctx(autumn));
            m.store_split(&event(&[john, smith, went, to, the, kitchen, dot]), &[ctx(autumn)], &e);
            m.end_sequence();
        }
        m.take_tags();
        m.facts.clear();
        m.store(&event(&[tom, is, a, smith, dot]));
        let inferred = m.infer(8);
        // tom took john's slot; smith dropped (and kept, in the second variant)
        assert!(inferred.iter().any(|(seq, c, _, _, _)| seq == &vec![b(tom, 0), b(went, 2), b(to, 3), b(the, 4), b(kitchen, 5), b(dot, 6)] && c == &vec![ctx(autumn)]), "{inferred:?}");
        assert!(inferred.iter().any(|(seq, _, _, _, _)| seq.contains(&b(smith, 1))));
    }
}
