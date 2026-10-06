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
use crate::fixed::{div_round, ratio, Q16, ONE};

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
    recurrent: Pathway,
    schaffer: Pathway,
    ca2_ca1: Pathway,
    output: Pathway,
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
    /// Recalls answered from the cache (events with no change), and all recalls.
    pub cache_hits: std::cell::Cell<(usize, usize)>,
}

impl Hippocampus {
    pub fn new(cfg: HippocampusConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let dg = DentateGyrus::new(cfg.ec_bits, cfg.dg_cells, cfg.dg_fan_in, cfg.dg_k, cfg.seed.wrapping_add(1));
        let temporo = DentateGyrus::new(cfg.ec_bits, cfg.ca1_cells, cfg.ca1_fan_in, cfg.ca1_k, cfg.seed.wrapping_add(2));
        let mossy = (0..cfg.dg_cells).map(|_| rng.gen_range(0..cfg.ca3_cells) as u32).collect();
        let all: Vec<u32> = (0..cfg.ca2_cells as u32).collect();
        let mut ca2_state: Vec<u32> = all.choose_multiple(&mut rng, cfg.ca2_k).copied().collect();
        ca2_state.sort_unstable();
        let half_life = ((0.5f32.ln() / cfg.decay.clamp(0.01, 0.999_9).ln()).round() as u32).max(1); // float: config
        Self {
            perforant: Pathway::new(cfg.ec_bits, cfg.ca3_cells),
            perforant_writes: vec![0; cfg.ec_bits],
            recurrent: Pathway::new(cfg.ca3_cells, cfg.ca3_cells),
            schaffer: Pathway::new(cfg.ca3_cells, cfg.ca1_cells),
            ca2_ca1: Pathway::new(cfg.ca2_cells, cfg.ca1_cells),
            output: Pathway::new(cfg.ca1_cells, cfg.ec_bits),
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
    fn ca3_code(&self, x: &[usize]) -> Vec<u32> {
        let mut c: Vec<u32> = self.dg.separate(x).iter().map(|&g| self.mossy[g as usize]).collect();
        c.sort_unstable();
        c.dedup();
        c
    }

    /// CA3 settled from a drive, through the recurrent weights.
    fn settle(&self, from_cue: &[u32], k: usize) -> Vec<u32> {
        let mut c = top_k(from_cue.iter().map(|&v| v as u64), k);
        for _ in 0..self.cfg.settle {
            let active: Vec<usize> = c.iter().map(|&j| j as usize).collect();
            let rec = self.recurrent.drive(&active, self.epoch());
            c = top_k(rec.iter().zip(from_cue).map(|(r, f)| (r + f) as u64), k);
        }
        c
    }

    /// CA3 → CA1 (+ CA2 → CA1) → subiculum → EC V, and the CA1 comparator against `cue`.
    fn read_out(&self, c: Vec<u32>, cue: Option<&[usize]>) -> Recall {
        if c.is_empty() {
            return Recall::default();
        }
        let active: Vec<usize> = c.iter().map(|&j| j as usize).collect();
        let mut drive = self.schaffer.drive(&active, self.epoch());
        if self.cfg.ca2_weight > 0 {
            let ca2: Vec<usize> = self.ca2_state.iter().map(|&j| j as usize).collect();
            for (d, t) in drive.iter_mut().zip(self.ca2_ca1.drive(&ca2, self.epoch())) {
                *d += t * self.cfg.ca2_weight / 4;
            }
        }
        let a = top_k(drive.iter().map(|&v| v as u64), self.cfg.ca1_k);
        let ca1_match = match cue {
            Some(cue) if !a.is_empty() => {
                let a_cue = self.temporo.separate(cue);
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
        let ec = (0..self.cfg.ec_bits.min(out.len())).filter(|&b| (out[b] as u64) << 16 >= floor).collect();
        Recall { ec, strength: best, ca1_match, ca3: c, ca1: a }
    }

    fn recall_uncached(&self, cue: &[usize]) -> Recall {
        if self.stores == 0 || cue.is_empty() {
            return Recall::default();
        }
        let writes = &self.perforant_writes;
        let from_cue = self.perforant.drive_scaled(cue, self.epoch(), |i| 32 - writes[i].max(1).leading_zeros() - 1);
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
        if x.is_empty() {
            return 0;
        }
        let novelty = self.novelty(x);
        self.last_novelty = novelty;
        self.novelty_sum.0 += novelty as u64;
        self.novelty_sum.1 += 1;
        self.stores += 1;
        let (epoch, planes) = (self.epoch(), self.planes);
        let base = self.amounts[(self.stores % self.half_life) as usize] as u64;
        let factor = ONE as u64 + ((self.cfg.novelty_gain as u64 * novelty as u64) >> 16); // 1 + gain · novelty, Q16
        let amount = div_round(base * factor, ONE as u64).min((1u64 << planes) - 1) as u32;
        let c = self.ca3_code(x);
        let a = self.temporo.separate(x);
        let c_mask = BitVector::from_bits(&c.iter().map(|&j| j as usize).collect::<Vec<_>>(), self.cfg.ca3_cells);
        let a_mask = BitVector::from_bits(&a.iter().map(|&j| j as usize).collect::<Vec<_>>(), self.cfg.ca1_cells);
        let x_mask = BitVector::from_bits(x, self.cfg.ec_bits);
        for &i in x {
            self.perforant.strengthen(i, &c_mask, amount, planes, epoch);
            self.perforant_writes[i] = self.perforant_writes[i].saturating_add(1);
        }
        for &i in &c {
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
            let rec = self.recurrent.drive(&active, self.epoch());
            let next = top_k(rec.iter().map(|&v| v as u64), k);
            if next.is_empty() {
                break;
            }
            c = next;
        }
        self.read_out(c, None)
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
