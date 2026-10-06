//! A phase-coded hippocampal circuit: the circuit of `hippocampal_circuit` with integer
//! complex (phasor) weights, for a phase-bound binding space.
//!
//! **The binding space.** A binding (word w in slot s) is w's bits at slot s's phase.
//! Input index i = p · `field` + b stands for input bit b at phase p of `in_phases`. So
//! the same word in two slots uses the same input rows at different phases: binding by
//! phase, instead of by a field of its own (the sparse binding space of experiment 49).
//! - The index layout matches the sparse space's (field · bits + bit), so the harness's
//!   decoding (index mod bits = the bit) is unchanged.
//!
//! **The populations** (CA3, CA1, EC V) are phasor memories, as `PhaseAssociate`
//! ([`modules`](super::modules)): each active cell carries a phase.
//! - **Write:** w_ij += amount · e^{i(θ_j − φ_i)}.
//! - **Read:** h_j = Σ_i w_ij · e^{iφ_i}. A cell's score is |h_j|².
//! - **CA3 and CA1 phases:** each stored event's cells get phases from a fixed hash of
//!   the event.
//! - **EC V:** the output bits sit at phase 0.
//! - **Cancellation:** terms from unrelated events arrive at unrelated phases and largely
//!   cancel, so no centering is needed. A common word in many slots arrives at many
//!   phases, and its rows no longer pull every cue into the common events' attractor.
//!
//! **The rest is as in `Hippocampus` (hashed-projection mode):**
//! - the dentate gyrus separates the content, weighted toward novel inputs; mossy fibres
//!   pick the CA3 cells;
//! - EC III → CA1 gives the comparator's code;
//! - CA3 settles through its recurrent weights;
//! - novelty-gated write strength; novelty tags; tagged and free replay;
//! - weights halve every `half_life` stores (lazily, per row).
//!
//! Weights are kept per row in a hash map of the targets written (sparse); all arithmetic
//! is integer: phases from an integer (cos, sin) table, products shifted.

use std::cell::{Cell, RefCell};

use rand::seq::SliceRandom;
use rand::{Rng, RngCore};

use super::hippocampal_circuit::{EpisodicCircuit, HippocampusConfig, Recall};
use super::hippocampus::{half_life_of, mix64, top_k, write_amounts, DentateGyrus};
use crate::det::HashMap;
use crate::fixed::{div_round, isqrt, phasors, ratio, recip32, Q16, ONE};

/// (a + ib)(c + id) / 2^14 (c + id a table phasor).
fn rot(a: (i32, i32), t: (i32, i32)) -> (i64, i64) {
    let (a0, a1, t0, t1) = (a.0 as i64, a.1 as i64, t.0 as i64, t.1 as i64);
    ((a0 * t0 - a1 * t1) >> 14, (a0 * t1 + a1 * t0) >> 14)
}

/// One pathway: for each source row, the complex weights onto the targets it has
/// written, and the epoch the row was last brought up to date (decay by halving).
struct PhasePath {
    rows: Vec<Option<(HashMap<u32, (i32, i32)>, u32)>>,
}

impl PhasePath {
    fn new() -> Self {
        Self { rows: Vec::new() }
    }

    /// Row `i` gains `amount` · e^{i(θ_j − φ)} on each target (j, e^{iθ_j}); `conj` =
    /// e^{−iφ}.
    fn write(&mut self, i: usize, conj: (i32, i32), targets: &[(u32, (i32, i32))], amount: u32, epoch: u32) {
        if i >= self.rows.len() {
            self.rows.resize_with(i + 1, || None);
        }
        let (row, at) = self.rows[i].get_or_insert_with(|| (HashMap::default(), epoch));
        let shift = epoch - *at;
        if shift > 0 {
            row.retain(|_, w| {
                *w = (w.0 >> shift.min(31), w.1 >> shift.min(31));
                *w != (0, 0) && *w != (-1, -1) && *w != (0, -1) && *w != (-1, 0)
            });
            *at = epoch;
        }
        for &(j, t) in targets {
            let z = rot(t, conj);
            // scale: table 2^14 → 2^6 per unit of amount
            let e = row.entry(j).or_insert((0, 0));
            e.0 += ((z.0 * amount as i64) >> 8) as i32;
            e.1 += ((z.1 * amount as i64) >> 8) as i32;
        }
    }

    /// Adds row `i`'s weights, rotated by e^{iφ} (`ph`) and scaled by `weight` (`Q16`),
    /// into `h`.
    fn read(&self, i: usize, ph: (i32, i32), weight: u64, epoch: u32, h: &mut [(i64, i64)]) {
        let Some(Some((row, at))) = self.rows.get(i) else { return };
        let shift = (epoch - at).min(31);
        for (&j, &w) in row {
            let z = rot((w.0 >> shift, w.1 >> shift), ph);
            if let Some(hj) = h.get_mut(j as usize) {
                hj.0 += (z.0 * weight as i64) >> 16;
                hj.1 += (z.1 * weight as i64) >> 16;
            }
        }
    }
}

fn mag2(h: (i64, i64)) -> u64 {
    ((h.0 as i128 * h.0 as i128 + h.1 as i128 * h.1 as i128) >> 8).min(u64::MAX as i128) as u64
}

pub struct PhaseHippocampus {
    pub cfg: HippocampusConfig,
    /// Bits per phase field (the word-code width).
    field: usize,
    in_phases: usize,
    phases: usize,
    /// 1/n weighting of each cue input by its write count (default on: a binding at a
    /// fixed slot phase arrives at the same phase in every event, so its rows add up
    /// coherently across the word's bits, and only 1/n keeps common bindings from
    /// dominating; without it a one-shot binding recalls 1 of 16 bits, with it 16).
    pub inverse: bool,
    in_table: Vec<(i32, i32)>,
    table: Vec<(i32, i32)>,
    dg: DentateGyrus,
    temporo: DentateGyrus,
    mossy: Vec<u32>,
    perforant: PhasePath,
    recurrent: PhasePath,
    schaffer: PhasePath,
    output: PhasePath,
    writes: Vec<u32>,
    out_bits: usize,
    half_life: u32,
    amounts: Vec<u32>,
    stores: u32,
    tags: Vec<(Vec<u32>, Vec<usize>)>,
    novelty_sum: (u64, usize),
    cache: RefCell<Option<(Vec<usize>, Recall)>>,
    cache_hits: Cell<(usize, usize)>,
}

type Code = Vec<(u32, usize)>; // (cell, phase)

impl PhaseHippocampus {
    /// `cfg.ec_bits` = `in_phases` × the field width; `phases`: the phase resolution of
    /// CA3 and CA1. Uses `cfg`'s cells, k, fan-out (`hashed_fan_out`, default 300),
    /// settle, decay, novelty gain, readout fraction, output width and seed.
    pub fn new(cfg: HippocampusConfig, in_phases: usize, phases: usize) -> Self {
        let in_phases = in_phases.max(1);
        let field = cfg.ec_bits / in_phases;
        let fan = cfg.hashed_fan_out.unwrap_or(300);
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(cfg.seed);
        let mossy = (0..cfg.dg_cells).map(|_| rng.gen_range(0..cfg.ca3_cells) as u32).collect();
        let half_life = half_life_of(cfg.decay);
        let out_bits = if cfg.out_bits == 0 { field } else { cfg.out_bits };
        Self {
            dg: DentateGyrus::hashed(cfg.dg_cells, fan, cfg.dg_k, cfg.seed.wrapping_add(1)),
            temporo: DentateGyrus::hashed(cfg.ca1_cells, fan, cfg.ca1_k, cfg.seed.wrapping_add(2)),
            mossy,
            perforant: PhasePath::new(),
            recurrent: PhasePath::new(),
            schaffer: PhasePath::new(),
            output: PhasePath::new(),
            writes: vec![0; cfg.ec_bits],
            out_bits,
            half_life,
            amounts: write_amounts(half_life),
            stores: 0,
            tags: Vec::new(),
            novelty_sum: (0, 0),
            cache: RefCell::new(None),
            cache_hits: Cell::new((0, 0)),
            field,
            in_phases,
            phases: phases.max(1),
            inverse: true,
            in_table: phasors(in_phases),
            table: phasors(phases.max(1)),
            cfg,
        }
    }

    fn epoch(&self) -> u32 {
        self.stores / self.half_life
    }

    fn n(&self, i: usize) -> u32 {
        self.writes.get(i).copied().unwrap_or(0)
    }

    /// An input's drive weight in the dentate gyrus and CA1 codes: 1 / 2^⌊log2 n⌋ (as in
    /// the hashed `Hippocampus`), so an event's new content chooses its codes.
    fn novelty_weight(&self, i: usize) -> u32 {
        ONE >> (31 - self.n(i).max(1).leading_zeros())
    }

    /// Phases for an event's cells, from a fixed hash of the event's inputs.
    fn phased(&self, cells: &[u32], x: &[usize], salt: u64) -> Code {
        let key = x.iter().fold(self.cfg.seed ^ salt, |h, &b| mix64(h ^ b as u64));
        cells.iter().map(|&c| (c, (mix64(key ^ c as u64) % self.phases as u64) as usize)).collect()
    }

    fn ca3_code(&self, content: &[usize]) -> Code {
        let g = self.dg.separate_weighted(content, |b| self.novelty_weight(b));
        let mut c: Vec<u32> = g.iter().map(|&g| self.mossy[g as usize]).collect();
        c.sort_unstable();
        c.dedup();
        self.phased(&c, content, 0xCA3)
    }

    fn ca1_code(&self, x: &[usize]) -> Code {
        let a = self.temporo.separate_weighted(x, |b| self.novelty_weight(b));
        self.phased(&a, x, 0xCA1)
    }

    /// The `k` cells with the largest |h|², each at the table phase nearest arg h.
    fn select(&self, h: &[(i64, i64)], k: usize) -> Code {
        let winners = top_k(h.iter().map(|&z| mag2(z)), k);
        winners.into_iter().map(|j| (j, self.nearest(h[j as usize]))).collect()
    }

    fn nearest(&self, z: (i64, i64)) -> usize {
        (0..self.phases).max_by_key(|&p| z.0 * self.table[p].0 as i64 + z.1 * self.table[p].1 as i64).unwrap_or(0)
    }

    fn targets(&self, code: &Code) -> Vec<(u32, (i32, i32))> {
        code.iter().map(|&(c, p)| (c, self.table[p])).collect()
    }

    /// Drive from a population code through `path` (rows = its cells, at their phases).
    fn drive(&self, path: &PhasePath, code: &Code, n: usize) -> Vec<(i64, i64)> {
        let mut h = vec![(0i64, 0i64); n];
        for &(c, p) in code {
            path.read(c as usize, self.table[p], ONE as u64, self.epoch(), &mut h);
        }
        h
    }

    fn cue_drive(&self, cue: &[usize]) -> Vec<(i64, i64)> {
        let mut h = vec![(0i64, 0i64); self.cfg.ca3_cells];
        for &i in cue {
            let (b, p) = (i % self.field, (i / self.field).min(self.in_phases - 1));
            let w = if self.inverse { recip32(self.n(i).max(1) as u64) >> 16 } else { ONE as u64 };
            self.perforant.read(b, self.in_table[p], w, self.epoch(), &mut h);
        }
        h
    }

    fn settle(&self, from_cue: &[(i64, i64)], steps: usize) -> Code {
        let mut c = self.select(from_cue, self.cfg.dg_k);
        for _ in 0..steps {
            let rec = self.drive(&self.recurrent, &c, self.cfg.ca3_cells);
            let total: Vec<(i64, i64)> = rec.iter().zip(from_cue).map(|(r, f)| (r.0 + f.0, r.1 + f.1)).collect();
            c = self.select(&total, self.cfg.dg_k);
        }
        c
    }

    fn read_out(&self, c: Code, cue: Option<&[usize]>) -> Recall {
        if c.is_empty() {
            return Recall::default();
        }
        let a = self.select(&self.drive(&self.schaffer, &c, self.cfg.ca1_cells), self.cfg.ca1_k);
        let ca1: Vec<u32> = a.iter().map(|x| x.0).collect();
        let ca1_match = match cue {
            Some(cue) if !a.is_empty() => {
                let own: Vec<u32> = self.ca1_code(cue).iter().map(|x| x.0).collect();
                ratio(ca1.iter().filter(|j| own.contains(j)).count() as u64, self.cfg.ca1_k as u64)
            }
            _ => 0,
        };
        let out = self.drive(&self.output, &a, self.out_bits);
        let scores: Vec<u64> = out.iter().map(|&z| mag2(z)).collect();
        let best = scores.iter().copied().max().unwrap_or(0);
        let ca3 = c.iter().map(|&(j, p)| j * self.phases as u32 + p as u32).collect();
        if best == 0 {
            return Recall { ca3, ca1, ca1_match, ..Default::default() };
        }
        // |h| ≥ fraction · best  ⇔  |h|² · 2^32 ≥ fraction² · best²
        let f = self.cfg.readout_fraction as u128;
        let ec = (0..self.out_bits).filter(|&b| (scores[b] as u128) << 32 >= f * f * best as u128).collect();
        let strength = (isqrt(best) >> 2) as u32;
        Recall { ec, strength, ca1_match, ca3, ca1 }
    }

    fn recall_uncached(&self, cue: &[usize]) -> Recall {
        if self.stores == 0 || cue.is_empty() {
            return Recall::default();
        }
        let h = self.cue_drive(cue);
        let c = self.settle(&h, self.cfg.settle);
        self.read_out(c, Some(cue))
    }
}

impl EpisodicCircuit for PhaseHippocampus {
    fn recall(&self, cue: &[usize]) -> Recall {
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

    fn familiarity(&self, bits: &[usize]) -> u64 {
        if bits.is_empty() {
            return 0;
        }
        div_round(bits.iter().map(|&b| self.n(b) as u64).sum(), bits.len() as u64)
    }

    fn novelty(&self, x: &[usize]) -> Q16 {
        let r = self.recall_uncached(x);
        if r.ca1.is_empty() { ONE } else { ONE - r.ca1_match.min(ONE) }
    }

    fn store(&mut self, x: &[usize]) -> Q16 {
        self.store_split(x, &[], x)
    }

    fn store_event(&mut self, content: &[usize], context: &[usize]) -> Q16 {
        let mut all: Vec<usize> = content.iter().chain(context).copied().collect();
        all.sort_unstable();
        all.dedup();
        let out: Vec<usize> = all.iter().map(|&i| i % self.field).collect();
        self.store_split(content, context, &out)
    }

    fn store_split(&mut self, content: &[usize], context: &[usize], out: &[usize]) -> Q16 {
        if content.is_empty() {
            return 0;
        }
        let mut all: Vec<usize> = content.iter().chain(context).copied().filter(|&i| i < self.writes.len()).collect();
        all.sort_unstable();
        all.dedup();
        let novelty = self.novelty(&all);
        let unseen: Vec<usize> = content.iter().copied().filter(|&b| self.n(b) == 0).collect();
        self.novelty_sum.0 += novelty as u64;
        self.novelty_sum.1 += 1;
        self.stores += 1;
        let epoch = self.epoch();
        let base = self.amounts[(self.stores % self.half_life) as usize] as u64;
        let factor = ONE as u64 + ((self.cfg.novelty_gain as u64 * novelty as u64) >> 16);
        let amount = div_round(base * factor, ONE as u64).min(127) as u32;
        let c = self.ca3_code(content);
        if unseen.len() >= 16 && self.tags.len() < 4096 {
            self.tags.push((c.iter().map(|&(j, p)| j * self.phases as u32 + p as u32).collect(), unseen));
        }
        let a = self.ca1_code(&all);
        let (ct, at) = (self.targets(&c), self.targets(&a));
        for &i in &all {
            let (b, p) = (i % self.field, (i / self.field).min(self.in_phases - 1));
            let conj = (self.in_table[p].0, -self.in_table[p].1);
            self.perforant.write(b, conj, &ct, amount, epoch);
            self.writes[i] = self.writes[i].saturating_add(1);
        }
        for &(j, p) in &c {
            let conj = (self.table[p].0, -self.table[p].1);
            let others: Vec<(u32, (i32, i32))> = ct.iter().copied().filter(|t| t.0 != j).collect();
            self.recurrent.write(j as usize, conj, &others, amount, epoch);
            self.schaffer.write(j as usize, conj, &at, amount, epoch);
        }
        let mut out_b: Vec<usize> = out.iter().copied().filter(|&b| b < self.out_bits).collect();
        out_b.sort_unstable();
        out_b.dedup();
        let out_t: Vec<(u32, (i32, i32))> = out_b.iter().map(|&b| (b as u32, self.table[0])).collect();
        for &(j, p) in &a {
            let conj = (self.table[p].0, -self.table[p].1);
            self.output.write(j as usize, conj, &out_t, amount, epoch);
        }
        *self.cache.borrow_mut() = None;
        novelty
    }

    fn advance_time(&mut self) {}

    fn take_tags(&mut self) -> Vec<(Vec<u32>, Vec<usize>)> {
        std::mem::take(&mut self.tags)
    }

    fn replay_from(&self, start: &[u32]) -> Recall {
        if self.stores == 0 || start.is_empty() {
            return Recall::default();
        }
        let c: Code = start.iter().map(|&x| (x / self.phases as u32, (x % self.phases as u32) as usize)).collect();
        self.read_out(c, None)
    }

    fn replay(&self, rng: &mut dyn RngCore) -> Recall {
        if self.stores == 0 {
            return Recall::default();
        }
        let all: Vec<u32> = (0..self.cfg.ca3_cells as u32).collect();
        let mut c: Code = all.choose_multiple(rng, self.cfg.dg_k).map(|&j| (j, 0)).collect();
        for x in c.iter_mut() {
            x.1 = rng.gen_range(0..self.phases);
        }
        for _ in 0..self.cfg.settle + 3 {
            let next = self.select(&self.drive(&self.recurrent, &c, self.cfg.ca3_cells), self.cfg.dg_k);
            if next.is_empty() {
                break;
            }
            c = next;
        }
        self.read_out(c, None)
    }

    fn len(&self) -> usize {
        self.stores as usize
    }

    fn stats(&self) -> (usize, usize, u64, usize) {
        let (h, a) = self.cache_hits.get();
        (h, a, self.novelty_sum.0, self.novelty_sum.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    const FIELD: usize = 2048;
    const IN_PHASES: usize = 16;

    /// Word w's 16 bits.
    fn word(w: usize) -> Vec<usize> {
        let mut rng = StdRng::seed_from_u64(w as u64 + 77);
        (0..FIELD).collect::<Vec<_>>().choose_multiple(&mut rng, 16).copied().collect()
    }

    /// Words bound to slots by phase: slot s → phase s.
    fn bound(ws: &[(usize, usize)]) -> Vec<usize> {
        let mut x: Vec<usize> = ws.iter().flat_map(|&(w, s)| word(w).into_iter().map(move |b| s * FIELD + b)).collect();
        x.sort_unstable();
        x.dedup();
        x
    }

    fn circuit() -> PhaseHippocampus {
        let mut cfg = HippocampusConfig::new(IN_PHASES * FIELD, 5);
        cfg.ca3_cells = 2048;
        cfg.ca1_cells = 2048;
        cfg.dg_cells = 4096;
        cfg.out_bits = FIELD;
        cfg.hashed_fan_out = Some(100);
        PhaseHippocampus::new(cfg, IN_PHASES, 16)
    }

    fn overlap(a: &[usize], b: &[usize]) -> usize {
        a.iter().filter(|x| b.contains(x)).count()
    }

    #[test]
    fn recalls_a_one_shot_binding_among_common_events() {
        let mut h = circuit();
        let mut rng = StdRng::seed_from_u64(9);
        // common events: words 0..6 in slots 0..3, plus a filler in slot 3
        for _ in 0..300 {
            let mut ws: Vec<(usize, usize)> = (0..6).collect::<Vec<_>>().choose_multiple(&mut rng, 3).enumerate().map(|(s, &w)| (w, s)).collect();
            ws.push((10 + rng.gen_range(0..6), 3));
            let x = bound(&ws);
            let out: Vec<usize> = x.iter().map(|&i| i % FIELD).collect();
            h.store_split(&x, &[], &out);
        }
        // the one-shot event: name 50 in slot 1, family 51 in slot 3
        let x = bound(&[(0, 0), (50, 1), (1, 2), (51, 3)]);
        let out: Vec<usize> = x.iter().map(|&i| i % FIELD).collect();
        h.store_split(&x, &[], &out);
        for _ in 0..50 {
            let mut ws: Vec<(usize, usize)> = (0..6).collect::<Vec<_>>().choose_multiple(&mut rng, 3).enumerate().map(|(s, &w)| (w, s)).collect();
            ws.push((10 + rng.gen_range(0..6), 3));
            let x = bound(&ws);
            let out: Vec<usize> = x.iter().map(|&i| i % FIELD).collect();
            h.store_split(&x, &[], &out);
        }
        let r = h.recall(&bound(&[(0, 0), (50, 1), (1, 2)]));
        assert!(overlap(&r.ec, &word(51)) >= 12, "the one-shot family comes back ({} of 16 bits)", overlap(&r.ec, &word(51)));
        assert!(h.take_tags().len() >= 1, "the new name is tagged");
    }

    #[test]
    fn the_same_word_in_another_slot_is_a_different_binding() {
        let mut h = circuit();
        // "a b c" and "c b a": the same words, opposite slots, different completions
        let e1 = bound(&[(1, 0), (2, 1), (3, 2), (4, 3)]);
        let e2 = bound(&[(3, 0), (2, 1), (1, 2), (5, 3)]);
        for e in [&e1, &e2] {
            let out: Vec<usize> = e.iter().map(|&i| i % FIELD).collect();
            h.store_split(e, &[], &out);
        }
        let r1 = h.recall(&bound(&[(1, 0), (2, 1), (3, 2)]));
        let r2 = h.recall(&bound(&[(3, 0), (2, 1), (1, 2)]));
        assert!(overlap(&r1.ec, &word(4)) >= 12 && overlap(&r1.ec, &word(5)) < 8, "slot order 1: {} / {}", overlap(&r1.ec, &word(4)), overlap(&r1.ec, &word(5)));
        assert!(overlap(&r2.ec, &word(5)) >= 12 && overlap(&r2.ec, &word(4)) < 8, "slot order 2: {} / {}", overlap(&r2.ec, &word(5)), overlap(&r2.ec, &word(4)));
    }
}
