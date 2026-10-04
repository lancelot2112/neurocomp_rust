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
//!   Weights decay by `decay` per stored episode (a palimpsest), so recent episodes
//!   dominate recall; entries that decay below `prune_below` are removed.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

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

/// Weight with lazy exponential decay: (value at `t`, `t`).
type Synapses = HashMap<u32, (f32, u32)>;

pub struct Ca3Memory {
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

impl Ca3Memory {
    pub fn new(ec_bits: usize, ca3_cells: usize, k: usize, decay: f32, settle_steps: usize) -> Self {
        Self {
            k,
            decay,
            prune_below: 0.02,
            settle_steps,
            readout_fraction: 0.5,
            ec_to_ca3: vec![HashMap::new(); ec_bits],
            ca3_to_ca3: vec![HashMap::new(); ca3_cells],
            ca3_to_ec: vec![HashMap::new(); ca3_cells],
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
}
