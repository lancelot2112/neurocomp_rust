//! Dentate gyrus and CA3: pattern separation by sparse expansion, and a
//! Hebbian autoassociative store with recurrent pattern completion.
//!
//! Follows the classic hippocampal memory model (Marr 1971; Treves & Rolls
//! 1994). An episode arrives as an entorhinal (EC) pattern `x`:
//! - **Dentate gyrus** (`DentateGyrus`): a fixed random expansion. Each of many
//!   granule cells samples `fan_in` random EC bits; the `k` cells with the most
//!   active inputs win (k-winner-take-all). The code is sparse and conjunctive,
//!   so episodes that share most of their words still get largely different codes
//!   (pattern separation).
//! - **CA3** (`Ca3Memory`): storing links EC→CA3 (`x`→`c`), CA3→CA3 (`c`↔`c`,
//!   recurrent collaterals) and CA3→EC (`c`→`x`) with Hebbian weights. Recall
//!   drives CA3 from a partial EC cue directly (perforant path; the dentate gyrus is used for
//!   storage, not retrieval), lets the recurrent weights settle the CA3 code for a
//!   few steps (pattern completion, a self-reference loop), and reads EC back out.
//!   Weights are bit-sliced counters (`SlicedCounter`, one row per source cell) and
//!   decay by halving, a plane shift, every few stores (a palimpsest), so recent
//!   episodes dominate recall. All arithmetic is on integers and bit planes.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::bitvec::{BitVector, SlicedCounter};

pub struct DentateGyrus {
    pub cells: usize,
    pub k: usize,
    inputs_to_cells: Vec<Vec<u32>>, // EC bit -> granule cells sampling it
}

impl DentateGyrus {
    pub fn new(input_bits: usize, cells: usize, fan_in: usize, k: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut inputs_to_cells = vec![Vec::new(); input_bits];
        let all: Vec<usize> = (0..input_bits).collect();
        for cell in 0..cells {
            for &b in all.choose_multiple(&mut rng, fan_in.min(input_bits)) {
                inputs_to_cells[b].push(cell as u32);
            }
        }
        Self { cells, k, inputs_to_cells }
    }

    /// The sparse code for active input bits `x`: the `k` granule cells with the
    /// most active inputs (ties broken by cell index).
    pub fn separate(&self, x: &[usize]) -> Vec<u32> {
        let mut drive = vec![0u16; self.cells];
        for &b in x {
            if let Some(cells) = self.inputs_to_cells.get(b) {
                for &c in cells {
                    drive[c as usize] += 1;
                }
            }
        }
        top_k(drive.iter().map(|&d| d as f32), self.k)
    }
}

/// One pathway's weights in bits: for each source cell, a row of bit-sliced counters
/// (one small counter per target), allocated when first written. Decay is a plane
/// shift (halving) every `half_life` stores, applied lazily: each row remembers the
/// epoch it was last normalized, and reads skip the planes that would have been shifted
/// out (floor(v / 2^s) = Σ_{p ≥ s} bit_p · 2^(p − s)).
struct Pathway {
    rows: Vec<Option<(SlicedCounter, u32)>>,
    targets: usize,
}

impl Pathway {
    fn new(sources: usize, targets: usize) -> Self {
        Self { rows: (0..sources).map(|_| None).collect(), targets }
    }

    /// Add `amount` to row `i`'s counters under `mask` (after bringing the row up to `epoch`).
    fn strengthen(&mut self, i: usize, mask: &BitVector, amount: u32, planes: usize, epoch: u32) {
        let targets = self.targets;
        let (row, at) = self.rows[i].get_or_insert_with(|| (SlicedCounter::new(targets, planes, 0), epoch));
        row.shift_down((epoch - *at) as usize);
        *at = epoch;
        for p in 0..planes {
            if amount >> p & 1 == 1 {
                row.add_power(mask, p);
            }
        }
    }

    /// Summed (decayed) weights from the active `sources` onto every target.
    fn drive(&self, sources: &[usize], epoch: u32) -> Vec<u32> {
        let mut out = vec![0u32; self.targets];
        for &i in sources {
            let Some(Some((row, at))) = self.rows.get(i) else { continue };
            let shift = (epoch - at) as usize;
            for p in shift..row.planes() {
                let w = 1u32 << (p - shift);
                for (wi, &word) in row.plane(p).as_words().iter().enumerate() {
                    let mut word = word;
                    while word != 0 {
                        out[wi * 64 + word.trailing_zeros() as usize] += w;
                        word &= word - 1;
                    }
                }
            }
        }
        out
    }

    /// Number of nonzero counters (stored synapses), as of their last normalization.
    fn synapses(&self) -> usize {
        self.rows
            .iter()
            .flatten()
            .map(|(row, _)| {
                let mut any = BitVector::new(self.targets, Some(0));
                for p in 0..row.planes() {
                    any.or_mut(row.plane(p));
                }
                any.count_ones()
            })
            .sum()
    }
}

/// One pathway as a delay line: for each source cell, one bit plane per recent store
/// ("written `age` stores ago"), in a ring indexed by store number. Decay is a pure
/// shift: advancing the ring ages every plane at once, and planes older than the ring
/// expire. A weight is Σ over its writes of `weights[age]`, an exact integer table
/// (e.g. 1024·decay^age), so nothing is rounded away between writes.
struct RingPathway {
    rows: Vec<Option<(Vec<BitVector>, Vec<u32>)>>, // (planes, store number each holds)
    targets: usize,
}

impl RingPathway {
    fn new(sources: usize, targets: usize) -> Self {
        Self { rows: (0..sources).map(|_| None).collect(), targets }
    }

    fn strengthen(&mut self, i: usize, mask: &BitVector, store: u32, len: usize) {
        let targets = self.targets;
        let (planes, stamps) = self.rows[i].get_or_insert_with(|| (vec![BitVector::new(targets, Some(0)); len], vec![u32::MAX; len]));
        let slot = store as usize % len;
        if stamps[slot] != store {
            planes[slot] = BitVector::new(targets, Some(0)); // the slot's old plane has expired
            stamps[slot] = store;
        }
        planes[slot].or_mut(mask);
    }

    fn drive(&self, sources: &[usize], store: u32, weights: &[u32]) -> Vec<u32> {
        let mut out = vec![0u32; self.targets];
        for &i in sources {
            let Some(Some((planes, stamps))) = self.rows.get(i) else { continue };
            for (plane, &at) in planes.iter().zip(stamps) {
                if at == u32::MAX || store.wrapping_sub(at) as usize >= weights.len() {
                    continue;
                }
                let w = weights[(store - at) as usize];
                for (wi, &word) in plane.as_words().iter().enumerate() {
                    let mut word = word;
                    while word != 0 {
                        out[wi * 64 + word.trailing_zeros() as usize] += w;
                        word &= word - 1;
                    }
                }
            }
        }
        out
    }
}

enum Weights {
    /// Bit-sliced counters, halved every `half_life` stores.
    Counters(Pathway),
    /// Delay line of age planes with an exact weight per age.
    Ring(RingPathway),
}

pub struct Ca3Memory {
    pub k: usize,
    /// Stores per halving of every weight (decay ≈ 0.5^(1/half_life) per store).
    pub half_life: u32,
    pub settle_steps: usize,
    /// Readout keeps EC bits scoring at least this fraction of the best score
    /// (higher = sharper clean-up toward the single strongest memory).
    pub readout_fraction: f32,
    planes: usize,
    /// Delay-line weight per age (empty when using counters).
    age_weights: Vec<u32>,
    ec_to_ca3: Weights,
    ca3_to_ca3: Weights,
    ca3_to_ec: Weights,
    ec_bits: usize,
    ca3_cells: usize,
    stores: u32,
}

impl Ca3Memory {
    /// `decay` per stored episode is approximated by halving every
    /// round(ln 0.5 / ln decay) stores. Weights are 7-plane bit-sliced counters
    /// (0..127). Within a halving period, the n-th store adds 16·2^(n/half_life)
    /// (16 up to 31), so later stores outweigh earlier ones exactly as continuous decay
    /// would; each write survives about 5 halvings.
    pub fn new(ec_bits: usize, ca3_cells: usize, k: usize, decay: f32, settle_steps: usize) -> Self {
        let half_life = ((0.5f32.ln() / decay.clamp(0.01, 0.999).ln()).round() as u32).max(1);
        Self {
            k,
            half_life,
            settle_steps,
            readout_fraction: 0.5,
            planes: 7,
            age_weights: Vec::new(),
            ec_to_ca3: Weights::Counters(Pathway::new(ec_bits, ca3_cells)),
            ca3_to_ca3: Weights::Counters(Pathway::new(ca3_cells, ca3_cells)),
            ca3_to_ec: Weights::Counters(Pathway::new(ca3_cells, ec_bits)),
            ec_bits,
            ca3_cells,
            stores: 0,
        }
    }

    /// The same store with delay-line weights: each write goes into the plane for the
    /// current store, weighted round(1024·decay^age) when read, until decay^age < 0.02.
    pub fn new_delay_line(ec_bits: usize, ca3_cells: usize, k: usize, decay: f32, settle_steps: usize) -> Self {
        let mut m = Self::new(ec_bits, ca3_cells, k, decay, settle_steps);
        let d = decay.clamp(0.01, 0.999);
        m.age_weights = (0..).map(|a| d.powi(a)).take_while(|&w| w >= 0.02).map(|w| (1024.0 * w).round() as u32).collect();
        m.ec_to_ca3 = Weights::Ring(RingPathway::new(ec_bits, ca3_cells));
        m.ca3_to_ca3 = Weights::Ring(RingPathway::new(ca3_cells, ca3_cells));
        m.ca3_to_ec = Weights::Ring(RingPathway::new(ca3_cells, ec_bits));
        m
    }

    /// Bit-native delay line: `planes` age planes per row whose weights are powers of
    /// two, newest = 2^(planes−1). Each synapse's planes are then literally the binary
    /// digits of its weight (its write history as a shift register, newest write the most
    /// significant bit): writing is OR into the newest plane, decay is the ring advancing
    /// (a shift, halving every weight per store), and nothing is ever rounded.
    pub fn new_shift_register(ec_bits: usize, ca3_cells: usize, k: usize, planes: usize, settle_steps: usize) -> Self {
        let mut m = Self::new_delay_line(ec_bits, ca3_cells, k, 0.5, settle_steps);
        m.age_weights = (0..planes).map(|a| 1u32 << (planes - 1 - a)).collect();
        m
    }

    fn write(w: &mut Weights, i: usize, mask: &BitVector, amount: u32, planes: usize, epoch: u32, store: u32, len: usize) {
        match w {
            Weights::Counters(p) => p.strengthen(i, mask, amount, planes, epoch),
            Weights::Ring(r) => r.strengthen(i, mask, store, len),
        }
    }

    fn read(&self, w: &Weights, sources: &[usize]) -> Vec<u32> {
        match w {
            Weights::Counters(p) => p.drive(sources, self.epoch()),
            Weights::Ring(r) => r.drive(sources, self.stores, &self.age_weights),
        }
    }

    fn epoch(&self) -> u32 {
        self.stores / self.half_life
    }

    /// Store one episode: EC pattern `x` (active bits) with CA3 code `c`
    /// (e.g. from the dentate gyrus).
    pub fn store(&mut self, x: &[usize], c: &[u32]) {
        self.stores += 1;
        let (epoch, planes) = (self.epoch(), self.planes);
        let phase = (self.stores % self.half_life) as f32 / self.half_life as f32;
        let amount = (16.0 * 2f32.powf(phase)).round() as u32;
        let c_idx: Vec<usize> = c.iter().map(|&j| j as usize).collect();
        let (store, len) = (self.stores, self.age_weights.len().max(1));
        let c_mask = BitVector::from_bits(&c_idx, self.ca3_cells);
        let x_mask = BitVector::from_bits(x, self.ec_bits);
        for &i in x {
            Self::write(&mut self.ec_to_ca3, i, &c_mask, amount, planes, epoch, store, len);
        }
        for &i in &c_idx {
            let mut others = c_mask.clone();
            others.bit_clear(i);
            Self::write(&mut self.ca3_to_ca3, i, &others, amount, planes, epoch, store, len);
            Self::write(&mut self.ca3_to_ec, i, &x_mask, amount, planes, epoch, store, len);
        }
    }

    /// Recall from a partial EC cue: drive CA3 from the cue, settle through the
    /// recurrent weights, read EC out. Returns the EC bits scoring at least
    /// `readout_fraction` of the best score, and that best score (0 if nothing was recalled).
    pub fn recall(&self, cue: &[usize], ec_bits: usize) -> (Vec<usize>, f32) {
        let from_cue = self.read(&self.ec_to_ca3, cue);
        let mut c = top_k(from_cue.iter().map(|&v| v as f32), self.k);
        for _ in 0..self.settle_steps {
            let active: Vec<usize> = c.iter().map(|&j| j as usize).collect();
            let rec = self.read(&self.ca3_to_ca3, &active);
            c = top_k(rec.iter().zip(&from_cue).map(|(r, f)| (r + f) as f32), self.k);
        }
        let active: Vec<usize> = c.iter().map(|&j| j as usize).collect();
        let out = self.read(&self.ca3_to_ec, &active);
        let best = out.iter().copied().max().unwrap_or(0);
        if best == 0 {
            return (Vec::new(), 0.0);
        }
        let floor = self.readout_fraction * best as f32;
        ((0..ec_bits.min(out.len())).filter(|&b| out[b] as f32 >= floor).collect(), best as f32)
    }

    /// Number of stored synapses (all three pathways; counters only, else 0).
    pub fn synapses(&self) -> usize {
        [&self.ec_to_ca3, &self.ca3_to_ca3, &self.ca3_to_ec]
            .iter()
            .map(|w| if let Weights::Counters(p) = w { p.synapses() } else { 0 })
            .sum()
    }
}

/// Weight with lazy exponential decay: (value at `t`, `t`).
type Synapses = crate::det::HashMap<u32, (f32, u32)>;

/// The original float version of `Ca3Memory` (f32 weights, exact exponential decay),
/// kept for comparison.
pub struct Ca3FloatMemory {
    pub k: usize,
    pub decay: f32,
    pub prune_below: f32,
    pub settle_steps: usize,
    /// Readout keeps EC bits scoring at least this fraction of the best score
    /// (higher = sharper clean-up toward the single strongest memory).
    pub readout_fraction: f32,
    ec_to_ca3: Vec<Synapses>,
    ca3_to_ca3: Vec<Synapses>,
    ca3_to_ec: Vec<Synapses>,
    now: u32,
}

impl Ca3FloatMemory {
    pub fn new(ec_bits: usize, ca3_cells: usize, k: usize, decay: f32, settle_steps: usize) -> Self {
        Self {
            k,
            decay,
            prune_below: 0.02,
            settle_steps,
            readout_fraction: 0.5,
            ec_to_ca3: vec![crate::det::HashMap::default(); ec_bits],
            ca3_to_ca3: vec![crate::det::HashMap::default(); ca3_cells],
            ca3_to_ec: vec![crate::det::HashMap::default(); ca3_cells],
            now: 0,
        }
    }

    fn effective(&self, (w, t): (f32, u32)) -> f32 {
        w * self.decay.powi((self.now - t) as i32)
    }

    fn strengthen(syn: &mut Synapses, target: u32, now: u32, decay: f32) {
        let e = syn.entry(target).or_insert((0.0, now));
        let current = e.0 * decay.powi((now - e.1) as i32);
        *e = (current + 1.0, now);
    }

    /// Store one episode: EC pattern `x` (active bits) with CA3 code `c`
    /// (e.g. from the dentate gyrus).
    pub fn store(&mut self, x: &[usize], c: &[u32]) {
        self.now += 1;
        let (now, decay) = (self.now, self.decay);
        for &i in x {
            for &j in c {
                Self::strengthen(&mut self.ec_to_ca3[i], j, now, decay);
            }
        }
        for &i in c {
            for &j in c {
                if i != j {
                    Self::strengthen(&mut self.ca3_to_ca3[i as usize], j, now, decay);
                }
            }
            for &j in x {
                Self::strengthen(&mut self.ca3_to_ec[i as usize], j as u32, now, decay);
            }
        }
        if self.now % 50 == 0 {
            self.prune();
        }
    }

    fn prune(&mut self) {
        let (now, decay, eps) = (self.now, self.decay, self.prune_below);
        for layer in [&mut self.ec_to_ca3, &mut self.ca3_to_ca3, &mut self.ca3_to_ec] {
            for syn in layer.iter_mut() {
                syn.retain(|_, &mut (w, t)| w * decay.powi((now - t) as i32) >= eps);
            }
        }
    }

    fn drive(&self, layer: &[Synapses], from: &[usize], size: usize) -> Vec<f32> {
        let mut out = vec![0f32; size];
        for &i in from {
            if let Some(syn) = layer.get(i) {
                for (&j, &wt) in syn {
                    out[j as usize] += self.effective(wt);
                }
            }
        }
        out
    }

    /// Recall from a partial EC cue: drive CA3 from the cue, settle through the
    /// recurrent weights, read EC out. Returns the EC bits scoring at least
    /// `readout_fraction` of the best score, and that best score (0 if nothing was recalled).
    pub fn recall(&self, cue: &[usize], ec_bits: usize) -> (Vec<usize>, f32) {
        let ca3_cells = self.ca3_to_ca3.len();
        let from_cue = self.drive(&self.ec_to_ca3, cue, ca3_cells);
        let mut c = top_k(from_cue.iter().copied(), self.k);
        for _ in 0..self.settle_steps {
            let active: Vec<usize> = c.iter().map(|&j| j as usize).collect();
            let rec = self.drive(&self.ca3_to_ca3, &active, ca3_cells);
            c = top_k(rec.iter().zip(&from_cue).map(|(r, f)| r + f), self.k);
        }
        let active: Vec<usize> = c.iter().map(|&j| j as usize).collect();
        let out = self.drive(&self.ca3_to_ec, &active, ec_bits);
        let best = out.iter().cloned().fold(0f32, f32::max);
        if best <= 0.0 {
            return (Vec::new(), 0.0);
        }
        ((0..ec_bits).filter(|&b| out[b] >= self.readout_fraction * best).collect(), best)
    }

    /// Number of stored synapses (all three pathways).
    pub fn synapses(&self) -> usize {
        [&self.ec_to_ca3, &self.ca3_to_ca3, &self.ca3_to_ec].iter().map(|l| l.iter().map(|s| s.len()).sum::<usize>()).sum()
    }
}

/// Common interface of the CA3 stores, so experiments can swap them.
pub trait Autoassociative {
    fn store(&mut self, x: &[usize], c: &[u32]);
    fn recall(&self, cue: &[usize], ec_bits: usize) -> (Vec<usize>, f32);
    fn set_readout_fraction(&mut self, f: f32);
}

impl Autoassociative for Ca3Memory {
    fn store(&mut self, x: &[usize], c: &[u32]) {
        Ca3Memory::store(self, x, c)
    }
    fn recall(&self, cue: &[usize], ec_bits: usize) -> (Vec<usize>, f32) {
        Ca3Memory::recall(self, cue, ec_bits)
    }
    fn set_readout_fraction(&mut self, f: f32) {
        self.readout_fraction = f;
    }
}

impl Autoassociative for Ca3FloatMemory {
    fn store(&mut self, x: &[usize], c: &[u32]) {
        Ca3FloatMemory::store(self, x, c)
    }
    fn recall(&self, cue: &[usize], ec_bits: usize) -> (Vec<usize>, f32) {
        Ca3FloatMemory::recall(self, cue, ec_bits)
    }
    fn set_readout_fraction(&mut self, f: f32) {
        self.readout_fraction = f;
    }
}

/// Indices of the `k` largest positive values (ties by index).
fn top_k(values: impl Iterator<Item = f32>, k: usize) -> Vec<u32> {
    let mut v: Vec<(f32, u32)> = values.enumerate().filter(|(_, x)| *x > 0.0).map(|(i, x)| (x, i as u32)).collect();
    v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    let mut out: Vec<u32> = v.into_iter().take(k).map(|(_, i)| i).collect();
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(i: usize) -> Vec<usize> {
        (i * 8..i * 8 + 8).collect()
    }

    fn episode(words: &[usize]) -> Vec<usize> {
        words.iter().flat_map(|&w| word(w)).collect()
    }

    #[test]
    fn dentate_gyrus_separates_overlapping_episodes() {
        let dg = DentateGyrus::new(512, 4096, 64, 20, 1);
        // two episodes sharing 4 of 5 words
        let a = dg.separate(&episode(&[1, 2, 3, 4, 5]));
        let b = dg.separate(&episode(&[1, 2, 3, 4, 6]));
        let shared = a.iter().filter(|c| b.contains(c)).count();
        // inputs overlap 80%, codes much less
        assert_eq!(a.len(), 20);
        assert!(shared < 16, "codes too similar: {shared}/20 shared");
    }

    #[test]
    fn ca3_completes_the_most_recent_episode_from_a_partial_cue() {
        let dg = DentateGyrus::new(512, 4096, 64, 20, 1);
        let mut m = Ca3Memory::new(512, 4096, 20, 0.9, 2);
        // mary(1) went(10) kitchen(20); john(2) went(10) garden(21); mary(1) went(10) office(22)
        for ep in [episode(&[1, 10, 20]), episode(&[2, 10, 21]), episode(&[1, 10, 22])] {
            m.store(&ep, &dg.separate(&ep));
        }
        let (out, strength) = m.recall(&word(1), 512);
        assert!(strength > 0.0);
        let has = |w: usize| word(w).iter().all(|b| out.contains(b));
        assert!(has(22) && !has(20), "should recall the recent mary episode (office)");
        let (out, _) = m.recall(&word(2), 512);
        assert!(word(21).iter().all(|b| out.contains(b)));
    }

    #[test]
    fn shift_register_ca3_completes_the_most_recent_episode() {
        let dg = DentateGyrus::new(512, 4096, 64, 20, 1);
        let mut m = Ca3Memory::new_shift_register(512, 4096, 20, 11, 2);
        for ep in [episode(&[1, 10, 20]), episode(&[2, 10, 21]), episode(&[1, 10, 22])] {
            m.store(&ep, &dg.separate(&ep));
        }
        let (out, _) = m.recall(&word(1), 512);
        let has = |w: usize| word(w).iter().all(|b| out.contains(b));
        assert!(has(22) && !has(20));
    }

    #[test]
    fn delay_line_ca3_completes_the_most_recent_episode() {
        let dg = DentateGyrus::new(512, 4096, 64, 20, 1);
        let mut m = Ca3Memory::new_delay_line(512, 4096, 20, 0.9, 2);
        for ep in [episode(&[1, 10, 20]), episode(&[2, 10, 21]), episode(&[1, 10, 22])] {
            m.store(&ep, &dg.separate(&ep));
        }
        let (out, _) = m.recall(&word(1), 512);
        let has = |w: usize| word(w).iter().all(|b| out.contains(b));
        assert!(has(22) && !has(20));
    }
}
