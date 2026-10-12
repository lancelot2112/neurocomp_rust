//! The hippocampus as an index that always completes (after Teyler & DiScenna's index theory,
//! Howard & Kahana's temporal context model and Hasselmo's acetylcholine modes).
//!
//! - **One row per event,** grown as needed (`IndexMemory`). A row's keys are the context in
//!   force and the content (the cells the cortex represents: each area's own, at its slot); it
//!   points to those cells.
//! - **The context drifts** with what the cortex represents: each step a few of its 32 cells are
//!   replaced by cells set from the represented cells (min-hash, so a drifted assembly moves it
//!   almost the same way); a learned event boundary (`BoundaryCell`, from the cortex's surprise)
//!   moves it by half.
//! - **Always completing:** each step recall offers what comes next. Nothing new represented, the
//!   offer holds; the cortex represents what was offered, the sequence moves on (the row's
//!   successor); it represents something else, the content is completed (among the rows whose
//!   content matches about as well as the best, not one offered in the last few steps, the one
//!   whose context is closest, then the most recent), a jump that reinstates that row's context.
//!   Following, the context blends a quarter toward the row's own.
//! - **Acetylcholine sets the mode:** attending to something from outside raises it (encoding:
//!   the offer is withheld from the cortex); it decays each step, and turning inward halves it
//!   (retrieval: the offer reaches the cortex). Storing goes on in both, except for one's own
//!   speech (corollary discharge).
//! - **Reconsolidation:** a recalled row whose pointed cells the cortex then completes (at least
//!   half kept) is relearned to the cells as they are now.

use rand::RngCore;

use crate::bitvec::BitVector;
use crate::fixed::ONE;
use crate::program::boundary::BoundaryCell;
use crate::program::hippocampal_circuit::EpisodicCircuit;
use crate::program::index_memory::{IndexConfig, IndexMemory};

/// The context's cells are drawn from this many.
pub const CTX_POOL: usize = 1 << 16;
/// The content keys come after the context's ids in the key space.
pub const CONTENT_BASE: usize = CTX_POOL + 4096;
/// Context cells.
const SLOTS: usize = 32;

fn ones(v: &BitVector) -> Vec<usize> {
    let mut out = Vec::new();
    for (wi, &x) in v.as_words().iter().enumerate() {
        let mut x = x;
        while x != 0 {
            out.push(wi * 64 + x.trailing_zeros() as usize);
            x &= x - 1;
        }
    }
    out
}

fn mix(x: u64) -> u64 {
    let mut y = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    y ^= y >> 29;
    y = y.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    y ^ (y >> 31)
}

pub struct IndexHc {
    pub mem: IndexMemory,
    ctx: Vec<usize>,
    boundary: BoundaryCell,
    last_recalled: Option<u32>,
    refractory: Vec<u32>,
    /// each row's context slot by slot as stored, and whether a boundary followed it
    ordered: std::collections::HashMap<u32, (Vec<usize>, bool)>,
    stored_now: Option<u32>,
    just_stored: Option<u32>,
    blend_rng: u64,
    to_reconsolidate: Option<u32>,
    /// acetylcholine (`Q16`)
    pub ach: u32,
    /// what the cortex represents now (the content half of the input)
    content: Vec<usize>,
    /// the state at the last step (an unchanged state is nothing new)
    last_state: Vec<usize>,
    /// the width of what the index points to
    out_bits: usize,
    /// (completions, of which reached the cortex), boundaries, rows relearned, relearned asleep
    pub stats: (usize, usize, usize, usize, usize),
}

impl IndexHc {
    /// `out_bits`: the width of the cortical cells pointed to (every area's slot).
    pub fn new(out_bits: usize, seed: u64) -> Self {
        let mut h = seed ^ 0x4242;
        let ctx = (0..SLOTS)
            .map(|_| {
                h = mix(h);
                (h % CTX_POOL as u64) as usize
            })
            .collect();
        let cfg = IndexConfig { min_overlap: 12, tiers: true, ..Default::default() };
        Self {
            mem: IndexMemory::new(cfg),
            ctx,
            boundary: BoundaryCell::new(out_bits, 3, 80),
            last_recalled: None,
            refractory: Vec::new(),
            ordered: std::collections::HashMap::new(),
            stored_now: None,
            just_stored: None,
            blend_rng: 0x9E37_79B9_7F4A_7C15 ^ seed,
            to_reconsolidate: None,
            ach: 0,
            content: Vec::new(),
            last_state: Vec::new(),
            out_bits,
            stats: (0, 0, 0, 0, 0),
        }
    }

    /// `n` of the context's cells are replaced, each set by one represented cell and the cell it
    /// replaces (the `n` cells with the smallest hash act).
    fn drift(&mut self, cells: &[usize], n: usize, salt: u64) {
        let mut acting: Vec<(u64, usize)> = cells.iter().map(|&c| (mix(c as u64 ^ salt.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)), c)).collect();
        acting.sort_unstable();
        let len = self.ctx.len() as u64;
        for &(h, c) in acting.iter().take(n) {
            let slot = (h % len) as usize;
            self.ctx[slot] = (mix(c as u64 ^ (self.ctx[slot] as u64).rotate_left(21) ^ salt) % CTX_POOL as u64) as usize;
        }
    }

    fn remember_offered(&mut self, r: u32) {
        self.refractory.push(r);
        if self.refractory.len() > 4 {
            self.refractory.remove(0);
        }
    }

    fn blend_from(&mut self, stored: &[usize], all: bool) {
        if stored.len() != self.ctx.len() {
            return;
        }
        if all {
            self.ctx = stored.to_vec();
            return;
        }
        for i in 0..self.ctx.len() {
            self.blend_rng ^= self.blend_rng << 13;
            self.blend_rng ^= self.blend_rng >> 7;
            self.blend_rng ^= self.blend_rng << 17;
            if self.blend_rng & 3 == 0 {
                self.ctx[i] = stored[i];
            }
        }
    }

    /// Retrieval mode: acetylcholine low.
    pub fn retrieving(&self) -> bool {
        self.ach < ONE / 2
    }

    /// One completion: what the index offers the cortex now (empty while encoding).
    fn complete(&mut self) -> BitVector {
        let content_keys: Vec<usize> = self.content.iter().map(|&c| CONTENT_BASE + c).collect();
        let need = (content_keys.len() / 2).max(1);
        let has = |mem: &IndexMemory, r: u32| -> usize {
            let k = mem.keys_of(r);
            content_keys.iter().filter(|&&c| k.binary_search(&(c as u32)).is_ok()).count()
        };
        let expected = self.last_recalled;
        let (now, jumped): (Option<u32>, bool) = if content_keys.is_empty() {
            (None, false)
        } else if let Some(e) = expected.filter(|&e| has(&self.mem, e) >= need) {
            (Some(e), false)
        } else {
            let cands: Vec<(u32, u64)> = self.mem.matches(&content_keys, 32).into_iter().filter(|&(r, _)| Some(r) != self.just_stored && !self.refractory.contains(&r)).collect();
            let top = cands.first().map_or(0, |x| x.1);
            let ctx: std::collections::HashSet<u32> = self.ctx.iter().map(|&c| c as u32).collect();
            let best = cands
                .iter()
                .filter(|&&(_, sc)| sc * 4 >= top * 3)
                .max_by_key(|&&(r, _)| (self.mem.keys_of(r).iter().filter(|k| ctx.contains(k)).count(), r))
                .map(|x| x.0);
            (best, best.is_some())
        };
        let out_row = match now {
            Some(n) => self.mem.successor(n).ca3.first().copied(),
            None => expected,
        };
        self.stats.0 += 1;
        if out_row != expected {
            if let Some(o) = out_row {
                self.remember_offered(o);
            }
        }
        self.last_recalled = out_row;
        self.to_reconsolidate = out_row;
        let mut v = BitVector::new(self.out_bits, Some(0));
        if !self.retrieving() {
            return v;
        }
        let src = if jumped { now } else { out_row };
        if let Some((stored, _)) = src.and_then(|r| self.ordered.get(&r)).cloned() {
            self.blend_from(&stored, jumped);
        }
        self.stats.1 += 1;
        if let Some(o) = out_row {
            for &b in self.mem.out_of(o).iter().filter(|&&b| b < self.out_bits) {
                v.bit_set(b);
            }
        }
        v
    }

    /// One step, the only way in: the entorhinal input is what the cortex holds now, every area's
    /// cells (`state`: sensed, or imagined and fed back top-down). It is never told what was
    /// heard. The rest is neuromodulation and the cortex's own signals:
    /// - `outward` (some area's input changed from outside: attention to the world): acetylcholine
    ///   rises, and the state is stored as an event;
    /// - `own` (the voice's efference copy marks the sound as one's own: corollary discharge):
    ///   represented, never stored, and it does not raise acetylcholine;
    /// - neither, with a new state (imagined, meant): attention is inward, acetylcholine falls;
    /// - `surprise` (`Q16`, the cortex's): the boundary cell learns from it.
    /// An unchanged state is nothing new: the offer holds. Returns what the index offers the
    /// cortex (empty while encoding).
    pub fn step(&mut self, state: &BitVector, outward: bool, own: bool, surprise: u32) -> BitVector {
        self.ach -= self.ach / 4;
        let cells_v = ones(state);
        let new = cells_v != self.last_state;
        self.last_state = cells_v.clone();
        self.just_stored = None;
        if !new || cells_v.is_empty() {
            self.content = Vec::new();
            return self.complete();
        }
        self.boundary.learn(surprise);
        if let Some(r) = self.to_reconsolidate.take() {
            let old = self.mem.out_of(r).to_vec();
            let kept = old.iter().filter(|c| cells_v.binary_search(c).is_ok()).count();
            if !old.is_empty() && kept * 2 >= old.len() && old != cells_v {
                self.mem.reconsolidate(r, &cells_v);
                self.stats.3 += 1;
            }
        }
        let external = outward && !own;
        if external {
            self.ach += (ONE - self.ach.min(ONE)) / 2;
        } else if !own {
            self.ach /= 2;
        }
        self.content = cells_v.clone();
        if external {
            let mut keys = self.ctx.clone();
            keys.extend(cells_v.iter().map(|c| CONTENT_BASE + c));
            let id = self.mem.rows() as u32;
            self.mem.store_split(&keys, &[], &cells_v);
            if self.mem.rows() as u32 > id {
                self.ordered.insert(id, (self.ctx.clone(), false));
                self.stored_now = Some(id);
                self.just_stored = Some(id);
            }
        }
        self.drift(&cells_v, 4, 1);
        if self.boundary.observe(&cells_v) {
            self.stats.2 += 1;
            self.drift(&cells_v, SLOTS / 2, 2);
            if let Some(e) = self.stored_now.and_then(|id| self.ordered.get_mut(&id)) {
                e.1 = true;
            }
        }
        self.stored_now = None;
        self.complete()
    }

    /// Sleep: chains of rows replayed by strength (`chains` of up to 30). `complete` is the
    /// cortex completing a pointer (what it then represents); a row whose cells it keeps at least
    /// half of is relearned to the completion.
    pub fn replay(&mut self, rng: &mut dyn RngCore, chains: usize, mut complete: impl FnMut(&BitVector) -> Vec<usize>) {
        for _ in 0..chains {
            let chain = self.mem.replay_prioritized(rng, |_| 1, 30);
            for r in chain {
                let (Some(&row), false) = (r.ca3.first(), r.ec.is_empty()) else { continue };
                let mut frame = BitVector::new(self.out_bits, Some(0));
                for &b in r.ec.iter().filter(|&&b| b < self.out_bits) {
                    frame.bit_set(b);
                }
                let now = complete(&frame);
                let kept = r.ec.iter().filter(|c| now.binary_search(c).is_ok()).count();
                if kept * 2 >= r.ec.len() && now != r.ec {
                    self.mem.reconsolidate(row, &now);
                    self.stats.4 += 1;
                }
            }
        }
    }
}
