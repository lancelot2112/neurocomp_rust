//! The full hippocampal circuit: entorhinal cortex, dentate gyrus, CA3, CA2, CA1 and the
//! subiculum, with the connections between them.
//!
//! `Ca3Memory` (in `hippocampus`) is the autoassociative core alone: the dentate gyrus
//! code is CA3's code, and CA3 reads straight back to the entorhinal cortex. This module
//! wires the rest of the loop (Amaral & Witter 1989; Lisman & Otmakhova 2001; Hasselmo
//! 2006):
//!
//! - **EC layer II → DG** (perforant path to the granule cells): the `DentateGyrus`
//!   expansion separates each episode into a sparse granule code.
//! - **DG → CA3, mossy fibres:** each granule cell has one fixed "detonator" target in
//!   CA3. At storage the granule code *chooses* the CA3 cells, so similar episodes get
//!   different CA3 codes. Mossy fibres are not used at recall (encoding mode only).
//! - **EC layer II → CA3** (perforant path, learned): drives CA3 from a cue at recall.
//!   Its synapses are scaled presynaptically: an EC input written by n episodes drives
//!   CA3 with its weights divided by about n (a right shift by log2 n), as heterosynaptic
//!   depression and synaptic scaling would. A common binding then counts little per
//!   target and a rare one much, so the cue's rare part finds its episode.
//! - **CA3 ↔ CA3** (recurrent collaterals, learned): pattern completion by settling.
//! - **EC layer III → CA1** (temporoammonic path, fixed): a random expansion with
//!   k-winners-take-all gives CA1 the *current* input's code.
//! - **CA3 → CA1** (Schaffer collaterals, learned): CA1 is taught to map each CA3
//!   attractor to its episode's CA1 code, so at recall it decodes what CA3 completed.
//! - **CA1 comparator:** at recall, the CA1 code from CA3 (what memory predicts) is
//!   compared with the CA1 code from EC III (what is actually there). The overlap is a
//!   match signal; its complement is novelty.
//! - **Novelty-gated encoding:** before an episode is stored, it is recalled; the CA1
//!   mismatch sets the write strength (as acetylcholine and dopamine from the VTA scale
//!   plasticity for novel input). A novel one-shot episode is written strongly, a
//!   familiar one weakly, so a single exposure can still win recall against bindings
//!   stored thousands of times (whose weights saturate).
//! - **CA2** (temporal context): a small population whose code drifts with time
//!   (`advance_time`). CA2 → CA1 is learned at storage, so at recall the current CA2
//!   state biases CA1 toward recent episodes (`ca2_weight`, 0 = off).
//! - **CA1 → subiculum → EC layer V** (learned): the output. The subiculum passes CA1's
//!   readout on with its strength; EC V is where recalled episodes leave the hippocampus.
//! - **Replay** (`replay`): sharp-wave-ripple-like. CA3 starts from a random set of cells,
//!   settles through its recurrent weights into a stored attractor, and the episode is
//!   read out through CA1 and the subiculum, with no cue.
//! - **Event-based recall:** a recall whose cue equals the previous one returns the
//!   cached result (no pathway is driven); every pathway's drive reads only the rows of
//!   active cells.
//!
//! All weights are bit-sliced counters with lazy plane-shift decay (`Pathway`), as in
//! `Ca3Memory`.

use std::cell::RefCell;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use super::hippocampus::{top_k, DentateGyrus, Pathway};
use crate::bitvec::BitVector;
use crate::fixed::{div_round, ratio, recip32, Q16, ONE};

/// Sizes and rates of the circuit.
#[derive(Clone, Debug)]
pub struct HippocampusConfig {
    pub ec_bits: usize,
    pub dg_cells: usize,
    pub dg_fan_in: usize,
    /// Active granule cells per episode (and so at most this many CA3 cells).
    pub dg_k: usize,
    pub ca3_cells: usize,
    pub ca1_cells: usize,
    pub ca1_fan_in: usize,
    pub ca1_k: usize,
    pub ca2_cells: usize,
    pub ca2_k: usize,
    /// Fraction of CA2's active cells replaced per `advance_time` (`Q16`).
    pub ca2_drift: Q16,
    /// Weight of CA2's drive onto CA1 at recall, in quarters of the Schaffer drive (0 = off).
    pub ca2_weight: u32,
    pub settle: usize,
    /// Per-store decay of every weight (as in `Ca3Memory`); converted once to a half-life.
    pub decay: f32, // float: config
    /// Write strength = base × (1 + novelty_gain × novelty), capped at the counter maximum
    /// (`Q16`, may exceed `ONE`).
    pub novelty_gain: Q16,
    /// EC V readout keeps the bits scoring at least this fraction of the best (`Q16`).
    pub readout_fraction: Q16,
    /// Presynaptic scaling on the CA3 recurrent and Schaffer pathways too (not only the
    /// perforant path): a CA3 cell written by n events drives its targets with its weights
    /// shifted right by log2 n, so hub cells do not pull every cue into one attractor.
    pub scale_all: bool,
    /// The output space (CA1 → subiculum → EC V), when it differs from the input space:
    /// e.g. a compact code the cortex reads, while the input is a large sparse binding
    /// space. 0 = the same as `ec_bits`.
    pub out_bits: usize,
    /// Hashed projections (DG, EC III → CA1) for a large, sparse input space: each active
    /// input bit drives `fan_out` cells.
    pub hashed_fan_out: Option<usize>,
    /// Homeostatic centering of the CA3 drive from the cue: each CA3 cell keeps the
    /// fraction of events it was written in (a running mean over about two half-lives)
    /// and has its share of the cue's total drive subtracted, total × rate / Σ rates,
    /// before the k winners are chosen. The subtracted shares add up to the total drive,
    /// so crosstalk has mean zero; much-used cells no longer win by bulk. No division.
    pub center: bool,
    pub seed: u64,
}

impl HippocampusConfig {
    pub fn new(ec_bits: usize, seed: u64) -> Self {
        Self {
            ec_bits,
            dg_cells: 8192,
            dg_fan_in: 300,
            dg_k: 32,
            ca3_cells: 4096,
            ca1_cells: 4096,
            ca1_fan_in: 300,
            ca1_k: 32,
            ca2_cells: 1024,
            ca2_k: 16,
            ca2_drift: 3277, // 0.05
            ca2_weight: 0,
            settle: 2,
            decay: 0.999,
            novelty_gain: 3 * ONE,
            readout_fraction: ONE / 2,
            scale_all: false,
            out_bits: 0,
            hashed_fan_out: None,
            center: false,
            seed,
        }
    }
}

/// What one recall produced.
#[derive(Clone, Debug, Default)]
pub struct Recall {
    /// The EC layer V pattern read out (active bits).
    pub ec: Vec<usize>,
    /// The best readout score (0: nothing recalled).
    pub strength: u32,
    /// CA1 comparator: overlap of the CA1 code from CA3 with the CA1 code of the cue
    /// itself (from EC III), as a fraction of `ca1_k` (`Q16`).
    pub ca1_match: Q16,
    pub ca3: Vec<u32>,
    pub ca1: Vec<u32>,
}

pub struct Hippocampus {
    pub cfg: HippocampusConfig,
    dg: DentateGyrus,
    /// Granule cell → its CA3 target (mossy fibre).
    mossy: Vec<u32>,
    /// EC layer III → CA1 (temporoammonic), as a fixed expansion.
    temporo: DentateGyrus,
    perforant: Pathway,
    /// Episodes that wrote each EC input's perforant row (for presynaptic scaling).
    perforant_writes: Vec<u32>,
    /// Events that wrote each CA3 cell's recurrent and Schaffer rows (`scale_all`).
    ca3_writes: Vec<u32>,
    recurrent: Pathway,
    schaffer: Pathway,
    ca2_ca1: Pathway,
    output: Pathway,
    out_bits: usize,
    ca2_state: Vec<u32>,
    rng: StdRng,
    half_life: u32,
    stores: u32,
    planes: usize,
    cache: RefCell<Option<(Vec<usize>, Recall)>>,
    /// Novelty of the latest stored episode, and the running (sum, count), in `Q16`.
    pub last_novelty: Q16,
    pub novelty_sum: (u64, usize),
    /// Write amount per phase of a halving period (as in `Ca3Memory`).
    amounts: Vec<u32>,
    /// Novelty tags: CA3 codes of events that contained something never stored before (at
    /// least a word's worth of content inputs no event had written), kept until `take_tags`
    /// (synaptic tagging; the events sleep replays first).
    tags: Vec<(Vec<u32>, Vec<usize>)>,
    /// Recalls answered from the cache (events with no change), and all recalls.
    pub cache_hits: std::cell::Cell<(usize, usize)>,
    /// Each CA3 cell's running write rate (32 fractional bits; `center`).
    ca3_rate: Vec<u64>,
}

impl Hippocampus {
    pub fn new(cfg: HippocampusConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let (dg, temporo) = match cfg.hashed_fan_out {
            Some(f) => (
                DentateGyrus::hashed(cfg.dg_cells, f, cfg.dg_k, cfg.seed.wrapping_add(1)),
                DentateGyrus::hashed(cfg.ca1_cells, f, cfg.ca1_k, cfg.seed.wrapping_add(2)),
            ),
            None => (
                DentateGyrus::new(cfg.ec_bits, cfg.dg_cells, cfg.dg_fan_in, cfg.dg_k, cfg.seed.wrapping_add(1)),
                DentateGyrus::new(cfg.ec_bits, cfg.ca1_cells, cfg.ca1_fan_in, cfg.ca1_k, cfg.seed.wrapping_add(2)),
            ),
        };
        let out_bits = if cfg.out_bits == 0 { cfg.ec_bits } else { cfg.out_bits };
        let mossy = (0..cfg.dg_cells).map(|_| rng.gen_range(0..cfg.ca3_cells) as u32).collect();
        let all: Vec<u32> = (0..cfg.ca2_cells as u32).collect();
        let mut ca2_state: Vec<u32> = all.choose_multiple(&mut rng, cfg.ca2_k).copied().collect();
        ca2_state.sort_unstable();
        let half_life = super::hippocampus::half_life_of(cfg.decay);
        Self {
            perforant: Pathway::new(cfg.ec_bits, cfg.ca3_cells),
            perforant_writes: vec![0; cfg.ec_bits],
            ca3_writes: vec![0; cfg.ca3_cells],
            recurrent: Pathway::new(cfg.ca3_cells, cfg.ca3_cells),
            schaffer: Pathway::new(cfg.ca3_cells, cfg.ca1_cells),
            ca2_ca1: Pathway::new(cfg.ca2_cells, cfg.ca1_cells),
            output: Pathway::new(cfg.ca1_cells, out_bits),
            out_bits,
            dg,
            mossy,
            temporo,
            ca2_state,
            rng,
            half_life,
            stores: 0,
            planes: 7,
            cache: RefCell::new(None),
            last_novelty: 0,
            novelty_sum: (0, 0),
            amounts: super::hippocampus::write_amounts(half_life),
            cache_hits: std::cell::Cell::new((0, 0)),
            tags: Vec::new(),
            ca3_rate: vec![0; cfg.ca3_cells],
            cfg,
        }
    }

    fn epoch(&self) -> u32 {
        self.stores / self.half_life
    }

    /// Number of episodes stored.
    pub fn len(&self) -> usize {
        self.stores as usize
    }

    pub fn is_empty(&self) -> bool {
        self.stores == 0
    }

    /// The CA3 code a stored episode gets: the mossy-fibre targets of its granule cells.
    /// With `scale_all`, each EC input drives the granule cells less the more episodes it
    /// has been in (1 / 2^⌊log2 n⌋, as on the perforant path), so the novel part of an
    /// input chooses its code (pattern separation of events that share common words).
    fn ca3_code(&self, x: &[usize]) -> Vec<u32> {
        let granules = if self.cfg.scale_all || self.cfg.hashed_fan_out.is_some() {
            let writes = &self.perforant_writes;
            self.dg.separate_weighted(x, |b| ONE >> (32 - writes[b].max(1).leading_zeros() - 1))
        } else {
            self.dg.separate(x)
        };
        let mut c: Vec<u32> = granules.iter().map(|&g| self.mossy[g as usize]).collect();
        c.sort_unstable();
        c.dedup();
        c
    }

    /// EC III → CA1: CA1's code for an input. With hashed projections (a sparse input
    /// space, where a bit's write count is its binding's familiarity), each input drives
    /// CA1 less the more familiar it is, so an event's new content, not its shared
    /// context, sets its CA1 code.
    fn ca1_code(&self, x: &[usize]) -> Vec<u32> {
        if self.cfg.hashed_fan_out.is_some() {
            let writes = &self.perforant_writes;
            self.temporo.separate_weighted(x, |b| ONE >> (32 - writes.get(b).copied().unwrap_or(0).max(1).leading_zeros() - 1))
        } else {
            self.temporo.separate(x)
        }
    }

    /// A CA3 pathway's drive from `active` cells (presynaptically scaled with `scale_all`).
    fn ca3_drive(&self, w: &Pathway, active: &[usize]) -> Vec<u32> {
        if self.cfg.scale_all {
            let writes = &self.ca3_writes;
            w.drive_scaled(active, self.epoch(), |i| 32 - writes[i].max(1).leading_zeros() - 1)
        } else {
            w.drive(active, self.epoch())
        }
    }

    /// CA3 settled from a drive, through the recurrent weights.
    /// `from_cue` is in `Q16` units of counter weight.
    fn settle(&self, from_cue: &[u64], k: usize) -> Vec<u32> {
        let mut c = top_k(from_cue.iter().copied(), k);
        for _ in 0..self.cfg.settle {
            let active: Vec<usize> = c.iter().map(|&j| j as usize).collect();
            let rec = self.ca3_drive(&self.recurrent, &active);
            c = top_k(rec.iter().zip(from_cue).map(|(&r, &f)| ((r as u64) << 16) + f), k);
        }
        c
    }

    /// CA3 → CA1 (+ CA2 → CA1) → subiculum → EC V, and the CA1 comparator against `cue`.
    fn read_out(&self, c: Vec<u32>, cue: Option<&[usize]>) -> Recall {
        if c.is_empty() {
            return Recall::default();
        }
        let active: Vec<usize> = c.iter().map(|&j| j as usize).collect();
        let mut drive = self.ca3_drive(&self.schaffer, &active);
        if self.cfg.ca2_weight > 0 {
            let ca2: Vec<usize> = self.ca2_state.iter().map(|&j| j as usize).collect();
            for (d, t) in drive.iter_mut().zip(self.ca2_ca1.drive(&ca2, self.epoch())) {
                *d += t * self.cfg.ca2_weight / 4;
            }
        }
        let a = top_k(drive.iter().map(|&v| v as u64), self.cfg.ca1_k);
        let ca1_match = match cue {
            Some(cue) if !a.is_empty() => {
                let a_cue = self.ca1_code(cue);
                ratio(a.iter().filter(|j| a_cue.binary_search(j).is_ok()).count() as u64, self.cfg.ca1_k as u64)
            }
            _ => 0,
        };
        let a_idx: Vec<usize> = a.iter().map(|&j| j as usize).collect();
        let out = self.output.drive(&a_idx, self.epoch());
        let best = out.iter().copied().max().unwrap_or(0);
        if best == 0 {
            return Recall { ca3: c, ca1: a, ca1_match, ..Default::default() };
        }
        let floor = best as u64 * self.cfg.readout_fraction as u64; // scaled by ONE
        let ec = (0..self.out_bits.min(out.len())).filter(|&b| (out[b] as u64) << 16 >= floor).collect();
        Recall { ec, strength: best, ca1_match, ca3: c, ca1: a }
    }

    fn recall_uncached(&self, cue: &[usize]) -> Recall {
        if self.stores == 0 || cue.is_empty() {
            return Recall::default();
        }
        let writes = &self.perforant_writes;
        let mut from_cue: Vec<u64> =
            self.perforant.drive_scaled(cue, self.epoch(), |i| 32 - writes[i].max(1).leading_zeros() - 1).into_iter().map(|v| (v as u64) << 16).collect();
        if self.cfg.center {
            let total: u64 = from_cue.iter().sum();
            let rates: u64 = self.ca3_rate.iter().sum::<u64>() >> 16; // Σ rates, in Q16
            if rates > 0 {
                let per = recip32(rates); // 2^32 / (Σ rates · 2^16)
                for (h, &r) in from_cue.iter_mut().zip(&self.ca3_rate) {
                    // total × (r / 2^32) / (rates / 2^16) = total × r × per / 2^48
                    let share = ((total as u128 * r as u128 * per as u128) >> 48) as u64;
                    *h = h.saturating_sub(share);
                }
            }
        }
        let c = self.settle(&from_cue, self.cfg.dg_k);
        self.read_out(c, Some(cue))
    }

    /// Recall from a partial EC (layer II / III) cue. A cue equal to the previous one is an
    /// event with no change: the cached result is returned.
    pub fn recall(&self, cue: &[usize]) -> Recall {
        let (hits, all) = self.cache_hits.get();
        if let Some((last, r)) = self.cache.borrow().as_ref() {
            if last.as_slice() == cue {
                self.cache_hits.set((hits + 1, all + 1));
                return r.clone();
            }
        }
        self.cache_hits.set((hits, all + 1));
        let r = self.recall_uncached(cue);
        *self.cache.borrow_mut() = Some((cue.to_vec(), r.clone()));
        r
    }

    /// Familiarity of an EC pattern: the mean number of stored episodes that wrote its
    /// inputs (rounded), from the perforant path's own write counts, the same counts its
    /// presynaptic scaling uses. Perirhinal-like item familiarity, with no separate store.
    pub fn familiarity(&self, bits: &[usize]) -> u64 {
        if bits.is_empty() {
            return 0;
        }
        let sum: u64 = bits.iter().map(|&b| self.perforant_writes.get(b).copied().unwrap_or(0) as u64).sum();
        div_round(sum, bits.len() as u64)
    }

    /// The CA1 comparator's novelty for a full episode: 1 − the match between what memory
    /// completes it to and what it is, in `Q16`. `ONE` when nothing is stored.
    pub fn novelty(&self, x: &[usize]) -> Q16 {
        let r = self.recall_uncached(x);
        if r.ca1.is_empty() {
            ONE
        } else {
            ONE - r.ca1_match.min(ONE)
        }
    }

    /// Store one episode (EC active bits), with novelty-gated strength. Returns its novelty.
    pub fn store(&mut self, x: &[usize]) -> Q16 {
        self.store_event(x, &[])
    }

    /// Store an event: `content` (what happened) and `context` (where in the story; e.g.
    /// lateral-EC context bits). The dentate gyrus separates the content alone, so events
    /// that share a context still get distinct CA3 codes; the context is associated with
    /// that code through the perforant path and the readout, so it cues and is recalled.
    /// Novelty is judged on the whole pattern. Returns the novelty.
    pub fn store_event(&mut self, content: &[usize], context: &[usize]) -> Q16 {
        let mut all: Vec<usize> = content.iter().chain(context).copied().collect();
        all.sort_unstable();
        all.dedup();
        self.store_split(content, context, &all)
    }

    /// Store an event whose input (EC II / III: `content` and `context`, in the input
    /// space) differs from what the readout should give back (`out`, in the output space,
    /// EC V). Returns the novelty.
    pub fn store_split(&mut self, content: &[usize], context: &[usize], out: &[usize]) -> Q16 {
        if content.is_empty() {
            return 0;
        }
        let mut all: Vec<usize> = content.iter().chain(context).copied().collect();
        all.sort_unstable();
        all.dedup();
        let x = &all[..];
        let novelty = self.novelty(x);
        // a novelty tag: at least 16 content inputs no event has written yet (in a sparse
        // binding space, a new binding)
        let unseen: Vec<usize> = content.iter().copied().filter(|&b| self.perforant_writes.get(b).copied().unwrap_or(0) == 0).collect();
        self.last_novelty = novelty;
        self.novelty_sum.0 += novelty as u64;
        self.novelty_sum.1 += 1;
        self.stores += 1;
        let (epoch, planes) = (self.epoch(), self.planes);
        let base = self.amounts[(self.stores % self.half_life) as usize] as u64;
        let factor = ONE as u64 + ((self.cfg.novelty_gain as u64 * novelty as u64) >> 16); // 1 + gain · novelty, Q16
        let amount = div_round(base * factor, ONE as u64).min((1u64 << planes) - 1) as u32;
        let c = self.ca3_code(content);
        if self.cfg.center {
            // running mean over min(stores, 2^s), s ≈ log2 of two half-lives
            let s = 32 - (2 * self.half_life).leading_zeros();
            let step = recip32((self.stores as u64).min(1u64 << s)) as i128;
            let mut j = 0;
            for (cell, r) in self.ca3_rate.iter_mut().enumerate() {
                let hit = j < c.len() && c[j] as usize == cell;
                if hit {
                    j += 1;
                }
                let x: i128 = if hit { 1 << 32 } else { 0 };
                *r = (*r as i128 + (((x - *r as i128) * step) >> 32)).clamp(0, 1 << 32) as u64;
            }
        }
        if unseen.len() >= 16 && self.tags.len() < 4096 {
            self.tags.push((c.clone(), unseen));
        }
        let a = self.ca1_code(x);
        let c_mask = BitVector::from_bits(&c.iter().map(|&j| j as usize).collect::<Vec<_>>(), self.cfg.ca3_cells);
        let a_mask = BitVector::from_bits(&a.iter().map(|&j| j as usize).collect::<Vec<_>>(), self.cfg.ca1_cells);
        let x_mask = BitVector::from_bits(out, self.out_bits);
        for &i in x {
            self.perforant.strengthen(i, &c_mask, amount, planes, epoch);
            self.perforant_writes[i] = self.perforant_writes[i].saturating_add(1);
        }
        for &i in &c {
            self.ca3_writes[i as usize] = self.ca3_writes[i as usize].saturating_add(1);
            let mut others = c_mask.clone();
            others.bit_clear(i as usize);
            self.recurrent.strengthen(i as usize, &others, amount, planes, epoch);
            self.schaffer.strengthen(i as usize, &a_mask, amount, planes, epoch);
        }
        for &j in &a {
            self.output.strengthen(j as usize, &x_mask, amount, planes, epoch);
        }
        if self.cfg.ca2_weight > 0 {
            for &i in &self.ca2_state.clone() {
                self.ca2_ca1.strengthen(i as usize, &a_mask, amount, planes, epoch);
            }
        }
        *self.cache.borrow_mut() = None;
        novelty
    }

    /// CA2's temporal context drifts: a fraction of its active cells is replaced.
    pub fn advance_time(&mut self) {
        let n = (div_round(self.cfg.ca2_drift as u64 * self.cfg.ca2_k as u64, ONE as u64) as usize).max(1);
        for _ in 0..n {
            let i = self.rng.gen_range(0..self.ca2_state.len());
            loop {
                let c = self.rng.gen_range(0..self.cfg.ca2_cells) as u32;
                if !self.ca2_state.contains(&c) {
                    self.ca2_state[i] = c;
                    break;
                }
            }
        }
        self.ca2_state.sort_unstable();
        *self.cache.borrow_mut() = None;
    }

    /// The novelty tags since the last call: each event's CA3 code (for prioritised replay)
    /// and the content inputs that were new when it was stored (what the tag is for).
    pub fn take_tags(&mut self) -> Vec<(Vec<u32>, Vec<usize>)> {
        std::mem::take(&mut self.tags)
    }

    /// Replay a given CA3 code (e.g. a novelty tag): the tagged pattern is reinstated as it
    /// was stored and read out through CA1 and the subiculum. (Letting it settle first
    /// lets the recurrent weights pull it into the attractor of some common event.)
    pub fn replay_from(&self, start: &[u32]) -> Recall {
        if self.stores == 0 || start.is_empty() {
            return Recall::default();
        }
        self.read_out(start.to_vec(), None)
    }

    /// Replay (sharp-wave ripple): CA3 starts from random cells, settles into an attractor
    /// through its recurrent weights, and the episode is read out through CA1 and the
    /// subiculum, with no cue.
    pub fn replay<R: Rng>(&self, rng: &mut R) -> Recall {
        if self.stores == 0 {
            return Recall::default();
        }
        let k = self.cfg.dg_k;
        let all: Vec<u32> = (0..self.cfg.ca3_cells as u32).collect();
        let mut c: Vec<u32> = all.choose_multiple(rng, k).copied().collect();
        c.sort_unstable();
        for _ in 0..self.cfg.settle + 3 {
            let active: Vec<usize> = c.iter().map(|&j| j as usize).collect();
            let rec = self.ca3_drive(&self.recurrent, &active);
            let next = top_k(rec.iter().map(|&v| v as u64), k);
            if next.is_empty() {
                break;
            }
            c = next;
        }
        self.read_out(c, None)
    }
}

/// What the episodic harness needs from a hippocampal circuit, so different circuits
/// (`Hippocampus`, `PhaseHippocampus`) can stand in for each other. Inputs are active
/// input indices; recall and replay give EC V patterns (`Recall::ec`).
pub trait EpisodicCircuit {
    fn recall(&self, cue: &[usize]) -> Recall;
    /// Recall without side effects on the store (no strengthening of what is recalled):
    /// for cues the network made itself, e.g. while retelling. Default: as `recall`.
    fn recall_peek(&self, cue: &[usize]) -> Recall {
        self.recall(cue)
    }
    /// Recall with the walk on or off for this one call (a cue controller's choice).
    /// Default: as `recall` (circuits without a walk).
    fn recall_as(&self, cue: &[usize], _walk: bool) -> Recall {
        self.recall(cue)
    }
    fn familiarity(&self, bits: &[usize]) -> u64;
    fn novelty(&self, x: &[usize]) -> Q16;
    fn store(&mut self, x: &[usize]) -> Q16;
    fn store_event(&mut self, content: &[usize], context: &[usize]) -> Q16;
    fn store_split(&mut self, content: &[usize], context: &[usize], out: &[usize]) -> Q16;
    fn advance_time(&mut self);
    /// A sequence (story) ended: the next stored event does not follow the last one.
    fn end_sequence(&mut self) {}
    /// The cortex now reconstructs these stored events on its own (`replay_from` codes).
    fn mark_consolidated(&self, _start: &[u32]) {}
    /// The cortex's error (`Q16`) when these stored events were last replayed to it.
    fn report_error(&self, _start: &[u32], _err: Q16) {}
    /// An offline period starts: replay order may be set now.
    fn begin_sleep(&mut self) {}
    /// Inferred events from the facts stored since the last call (generative replay): each
    /// (content ids in reading order, context ids, the source row). Default: none.
    fn infer(&mut self, _max_rows: usize) -> Vec<(Vec<usize>, Vec<usize>, u32)> {
        Vec::new()
    }
    /// The episode played forward from the event `cue` recalls: up to `max` following
    /// events, each as its content ids in reading order (sequence replay, e.g. to retell a
    /// story). Default: none (circuits without event order).
    fn sequence_from(&self, _cue: &[usize], _max: usize) -> Vec<Vec<usize>> {
        Vec::new()
    }
    /// The most recent episode (the one the latest stored event belongs to), its events in
    /// time order, each as content ids in reading order: recall by recency, e.g. to retell
    /// the story just read. Default: none.
    fn recent_episode(&self, _max: usize) -> Vec<Vec<usize>> {
        Vec::new()
    }
    /// The row the latest store wrote or strengthened (for a caller that keeps something
    /// per row, e.g. the cortical state it was stored in).
    fn last_row(&self) -> Option<u32> {
        None
    }
    fn take_tags(&mut self) -> Vec<(Vec<u32>, Vec<usize>)>;
    fn replay_from(&self, start: &[u32]) -> Recall;
    fn replay(&self, rng: &mut dyn rand::RngCore) -> Recall;
    fn len(&self) -> usize;
    /// (recalls answered from the cache, all recalls, novelty sum in `Q16`, stores judged)
    fn stats(&self) -> (usize, usize, u64, usize);
    /// Bytes the memory's state occupies (weights or rows, and their indexes).
    fn memory_bytes(&self) -> usize {
        0
    }
    /// A one-line report of circuit-specific counts.
    fn report(&self) -> String {
        String::new()
    }
}

impl EpisodicCircuit for Hippocampus {
    fn recall(&self, cue: &[usize]) -> Recall {
        Hippocampus::recall(self, cue)
    }
    fn familiarity(&self, bits: &[usize]) -> u64 {
        Hippocampus::familiarity(self, bits)
    }
    fn novelty(&self, x: &[usize]) -> Q16 {
        Hippocampus::novelty(self, x)
    }
    fn store(&mut self, x: &[usize]) -> Q16 {
        Hippocampus::store(self, x)
    }
    fn store_event(&mut self, content: &[usize], context: &[usize]) -> Q16 {
        Hippocampus::store_event(self, content, context)
    }
    fn store_split(&mut self, content: &[usize], context: &[usize], out: &[usize]) -> Q16 {
        Hippocampus::store_split(self, content, context, out)
    }
    fn advance_time(&mut self) {
        Hippocampus::advance_time(self)
    }
    fn take_tags(&mut self) -> Vec<(Vec<u32>, Vec<usize>)> {
        Hippocampus::take_tags(self)
    }
    fn replay_from(&self, start: &[u32]) -> Recall {
        Hippocampus::replay_from(self, start)
    }
    fn replay(&self, rng: &mut dyn rand::RngCore) -> Recall {
        let mut r = rng;
        Hippocampus::replay(self, &mut r)
    }
    fn len(&self) -> usize {
        Hippocampus::len(self)
    }
    fn stats(&self) -> (usize, usize, u64, usize) {
        let (h, a) = self.cache_hits.get();
        (h, a, self.novelty_sum.0, self.novelty_sum.1)
    }
    fn memory_bytes(&self) -> usize {
        self.perforant.bytes() + self.recurrent.bytes() + self.schaffer.bytes() + self.ca2_ca1.bytes() + self.output.bytes() + 4 * (self.perforant_writes.len() + self.ca3_writes.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BITS: usize = 2048;

    /// Word i's code: 16 bits from a fixed random draw.
    fn word(i: usize) -> Vec<usize> {
        let mut rng = StdRng::seed_from_u64(i as u64 + 77);
        let all: Vec<usize> = (0..BITS).collect();
        all.choose_multiple(&mut rng, 16).copied().collect()
    }

    fn episode(words: &[usize]) -> Vec<usize> {
        let mut x: Vec<usize> = words.iter().flat_map(|&w| word(w)).collect();
        x.sort_unstable();
        x.dedup();
        x
    }

    fn overlap(a: &[usize], b: &[usize]) -> usize {
        a.iter().filter(|x| b.contains(x)).count()
    }

    fn small(novelty_gain: Q16) -> Hippocampus {
        let mut cfg = HippocampusConfig::new(BITS, 5);
        cfg.dg_cells = 4096;
        cfg.ca3_cells = 2048;
        cfg.ca1_cells = 2048;
        cfg.dg_fan_in = 100;
        cfg.ca1_fan_in = 100;
        cfg.novelty_gain = novelty_gain;
        Hippocampus::new(cfg)
    }

    #[test]
    fn recalls_a_stored_episode_from_part_of_it() {
        let mut h = small(3 * ONE);
        for e in 0..20 {
            h.store(&episode(&[e * 3 + 100, e * 3 + 101, e * 3 + 102]));
        }
        let target = episode(&[130, 131, 132]);
        let r = h.recall(&episode(&[130, 131]));
        assert!(overlap(&r.ec, &word(132)) >= 12, "completion should bring back the missing word");
        assert!(overlap(&r.ec, &target) * 10 >= target.len() * 8);
    }

    #[test]
    fn novelty_falls_with_repetition() {
        let mut h = small(3 * ONE);
        let x = episode(&[1, 2, 3]);
        let first = h.store(&x);
        h.store(&x);
        let third = h.store(&x);
        assert_eq!(first, ONE);
        assert!(third < ONE / 2, "a repeated episode should be familiar (novelty {third})");
    }

    /// The failure case of experiment 45: one episode with a rare word, against many
    /// episodes made of common words. The cue mixes common words with the rare one.
    #[test]
    fn a_novel_one_shot_episode_wins_recall_against_common_ones() {
        let run = |gain: Q16| {
            let mut h = small(gain);
            let mut rng = StdRng::seed_from_u64(9);
            for _ in 0..300 {
                let mut ws: Vec<usize> = (0..6).collect::<Vec<_>>().choose_multiple(&mut rng, 3).copied().collect();
                ws.push(10 + rng.gen_range(0..6));
                h.store(&episode(&ws));
            }
            // the one-shot episode: common words 0, 1, a rare name 50 and its family 51
            h.store(&episode(&[0, 1, 50, 51]));
            for _ in 0..50 {
                let mut ws: Vec<usize> = (0..6).collect::<Vec<_>>().choose_multiple(&mut rng, 3).copied().collect();
                ws.push(10 + rng.gen_range(0..6));
                h.store(&episode(&ws));
            }
            let r = h.recall(&episode(&[0, 1, 50]));
            overlap(&r.ec, &word(51))
        };
        let gated = run(3 * ONE);
        eprintln!("one-shot family bits recalled: novelty-gated {gated}, ungated {}", run(0));
        assert!(gated >= 12, "with novelty-gated encoding the rare episode's family should be recalled ({gated} of 16 bits)");
    }

    /// The one-shot test with homeostatic centering.
    #[test]
    fn one_shot_recall_with_centering() {
        for center in [true] {
            let mut cfg = HippocampusConfig::new(BITS, 5);
            (cfg.dg_cells, cfg.ca3_cells, cfg.ca1_cells, cfg.dg_fan_in, cfg.ca1_fan_in) = (4096, 2048, 2048, 100, 100);
            cfg.center = center;
            let mut h = Hippocampus::new(cfg);
            let mut rng = StdRng::seed_from_u64(9);
            for i in 0..351 {
                if i == 300 {
                    h.store(&episode(&[0, 1, 50, 51]));
                    continue;
                }
                let mut ws: Vec<usize> = (0..6).collect::<Vec<_>>().choose_multiple(&mut rng, 3).copied().collect();
                ws.push(10 + rng.gen_range(0..6));
                h.store(&episode(&ws));
            }
            let r = h.recall(&episode(&[0, 1, 50]));
            assert!(overlap(&r.ec, &word(51)) >= 12, "center {center}: {} of 16 family bits", overlap(&r.ec, &word(51)));
        }
    }

    #[test]
    fn replay_reads_out_a_stored_episode() {
        let mut h = small(3 * ONE);
        let eps: Vec<Vec<usize>> = (0..10).map(|e| episode(&[e * 2 + 200, e * 2 + 201])).collect();
        for e in &eps {
            h.store(e);
        }
        let mut rng = StdRng::seed_from_u64(3);
        let mut found = 0;
        for _ in 0..20 {
            let r = h.replay(&mut rng);
            if eps.iter().any(|e| overlap(&r.ec, e) * 10 >= e.len() * 8) {
                found += 1;
            }
        }
        assert!(found >= 10, "most replays should settle into a stored episode ({found} of 20)");
    }

    #[test]
    fn an_unchanged_cue_is_answered_from_the_cache() {
        let mut h = small(3 * ONE);
        h.store(&episode(&[1, 2, 3]));
        let cue = episode(&[1, 2]);
        let a = h.recall(&cue);
        let b = h.recall(&cue);
        assert_eq!(a.ec, b.ec);
        assert_eq!(h.cache_hits.get(), (1, 2));
    }

    #[test]
    fn ca2_drifts() {
        let mut cfg = HippocampusConfig::new(BITS, 1);
        cfg.ca2_weight = 4;
        let mut h = Hippocampus::new(cfg);
        let before = h.ca2_state.clone();
        for _ in 0..40 {
            h.advance_time();
        }
        let kept = before.iter().filter(|c| h.ca2_state.contains(c)).count();
        assert!(kept < before.len() / 2, "after 40 steps most of CA2's code should have changed");
    }
}
