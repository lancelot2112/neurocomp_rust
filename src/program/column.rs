//! One cortical column for every area (`Column`): sensory, motor and association areas differ
//! only in what they are connected to, as in the cortex (Mountcastle's canonical circuit).
//!
//! - **Layer 4** (`Pool`): recodes the input, recruiting cells as new inputs need them. Motor
//!   cortex is agranular (no layer 4): there the input passes as it is.
//! - **Layer 2/3** (`BitCells`): predicts what the column's layer 4 will receive next, from its
//!   input, its own previous activity (recurrence) and context on the tuft (from above). In a
//!   motor column, with a command as its input, its prediction is what the command will be
//!   sensed as: the forward model, and the efference copy.
//! - **Layer 5** (`BitCells`): the column's output. In a motor column it maps what is meant (a
//!   sensory goal) to the command that produces it: the inverse model.
//!
//! - **Persistence** (`set_persistence`): layer 2/3's activity outlasts its input, each bit of
//!   the trace clearing at random with probability 2^-τ a step, τ the area's own (none in the
//!   senses, longer up the hierarchy: intrinsic timescales grow from sensory to prefrontal
//!   cortex, Murray et al. 2014). The trace sits on the column's tuft and goes out with its
//!   activity; a loop through the basal ganglia and thalamus can refresh it (`refresh`), so what
//!   is kept is chosen.
//!
//! - **A held population** (`set_recurrence`; prefrontal and association areas, none in the
//!   senses). Driving input (layer 4) alone decides what the area codes now; beside it, cells
//!   that sustain themselves hold a state, as prefrontal cells keep firing for a cue while
//!   others answer what is there now. The column's code is the two together (in silence, the
//!   held state alone). Items are superposed in the one held vector (sparse codes barely
//!   overlap): what layer 2/3 did not expect of the input (graded surprise: acetylcholine's
//!   switch from recurrent completion to feedforward encoding, Hasselmo) joins what is held as
//!   an item of at most 32 bits, as does the driven code when the column's loop opens the gate
//!   (`refresh`: the basal ganglia and thalamus); at most 128 bits are held (about four items).
//!   Each step a random sixteenth drops out (noise), and an autoassociative map, which learns
//!   every driven code and every item held, restores each learned pattern still mostly present;
//!   what nothing restores fades. A held bit is restored only for `life` steps after it was
//!   loaded (the area's persistence; then adaptation lets it go), so what is held is recent.
//!
//! Both layers learn the same way: a confirmed prediction strengthens the cells that made it;
//! where nothing predicted the target, a cell is grown for the moment (its tuft on the
//! context, silent synapses on the input, the target as its output).

use crate::bitvec::BitVector;
use crate::program::area::Pool;
use crate::program::bitcells::BitCells;

fn frames(fs: &[&BitVector]) -> BitVector {
    let mut w = Vec::new();
    for f in fs {
        w.extend_from_slice(f.as_words());
    }
    BitVector::from_words(w)
}

fn overlap(a: &BitVector, b: &BitVector) -> u32 {
    a.as_words().iter().zip(b.as_words()).map(|(x, y)| (x & y).count_ones()).sum()
}

/// A layer of primed cells learning a map, input (and context) → target: rows are
/// [input (basal) | context (tuft) | extra input (basal)].
pub struct Map {
    bits: usize,
    cells: BitCells,
    cap: usize,
}

impl Map {
    pub fn new(bits: usize, cap: usize) -> Self {
        let mut cells = BitCells::new(bits / 64, 16, 0);
        cells.set_meta(true);
        cells.set_tag_capture(true);
        cells.set_clustered(true);
        Self { bits, cells, cap }
    }

    fn row(&self, input: &BitVector, ctx: Option<&BitVector>, extra: Option<&BitVector>) -> BitVector {
        let zero = BitVector::new(self.bits, Some(0));
        // no context (or an empty one): the input itself is the context
        let ctx = ctx.filter(|c| c.count_ones() > 0).unwrap_or(input);
        frames(&[input, ctx, extra.unwrap_or(&zero)])
    }

    /// The layer's prediction for this input, if any cell fires.
    pub fn predict(&mut self, input: &BitVector, ctx: Option<&BitVector>, extra: Option<&BitVector>) -> Option<BitVector> {
        if input.count_ones() == 0 {
            return None;
        }
        let row = self.row(input, ctx, extra);
        self.cells.predict(&row, self.bits).map(|x| x.0)
    }

    /// Learn that `target` follows this input: predict, learn from the outcome, and where the
    /// prediction missed, grow a cell for the moment. Returns whether the prediction was right
    /// (at least half the target's bits).
    pub fn learn<R: rand::Rng + ?Sized>(&mut self, input: &BitVector, ctx: Option<&BitVector>, extra: Option<&BitVector>, target: &BitVector, rng: &mut R) -> bool {
        if input.count_ones() == 0 || target.count_ones() == 0 {
            return false;
        }
        let row = self.row(input, ctx, extra);
        let p = self.cells.predict(&row, self.bits).map(|x| x.0);
        let right = p.as_ref().map_or(false, |p| overlap(p, target) * 2 >= target.count_ones() as u32);
        self.cells.learn(&row, target, rng);
        if !right {
            self.cells.grow_cell(&row, target, self.cap, rng);
        }
        right
    }

    /// As `predict`, with whether the winner burst (its tuft agreed).
    pub fn predict_full(&mut self, input: &BitVector, ctx: Option<&BitVector>, extra: Option<&BitVector>) -> Option<(BitVector, bool)> {
        if input.count_ones() == 0 {
            return None;
        }
        let row = self.row(input, ctx, extra);
        self.cells.predict(&row, self.bits).map(|x| (x.0, x.2))
    }

    /// Every cell that fired at the latest prediction: the layer's competing proposals.
    pub fn proposals(&self) -> Vec<(BitVector, bool)> {
        self.cells.proposals(self.bits)
    }

    /// The column's goal now: of the cells that fired, the one whose output has been worth most
    /// when followed (its learned worth; ties to a burst, then the layer's own winner order).
    /// Returns (cell, output, burst).
    pub fn goal(&self) -> Option<(usize, BitVector, bool)> {
        self.cells.fired_cells(self.bits).into_iter().enumerate().max_by_key(|(i, (_, _, b, w))| (*w, *b, std::cmp::Reverse(*i))).map(|(_, (c, o, b, _))| (c, o, b))
    }

    /// Dopamine for cell `c`, whose output was followed as a goal.
    pub fn reinforce(&mut self, c: usize, delta: i32) {
        self.cells.add_worth(c, delta);
    }

    /// The latest winner's learning progress (see `BitCells::progress`).
    pub fn progress(&self) -> i32 {
        self.cells.progress()
    }

    /// The cells that fired at the latest prediction.
    pub fn last_fired(&self) -> Vec<usize> {
        self.cells.last_fired()
    }

    pub fn committed(&self) -> usize {
        self.cells.committed()
    }

    /// Sleep's compaction of this layer (see `BitCells::compact`): (merged, pruned).
    pub fn compact(&mut self, idle: u64) -> (usize, usize) {
        self.cells.compact(idle)
    }
}

/// A column: layer 4 (or none: agranular), layer 2/3, layer 5.
pub struct Column {
    bits: usize,
    pub l4: Option<Pool>,
    pub l23: Map,
    pub l5: Map,
    /// layer 2/3's own previous activity, projected onto the row
    rec: BitVector,
    /// the latest input as layer 4 coded it
    pub code: BitVector,
    /// what layer 2/3 expects next (in layer 4's code)
    pub expected: Option<BitVector>,
    /// the input and context layer 2/3 predicted from, to learn when the next input comes
    last: Option<(BitVector, BitVector)>,
    /// (predictions checked, right)
    pub hits: (u64, u64),
    /// how many of the latest code's bits were expected
    pub last_hit: u32,
    /// the persistent trace of layer 2/3's activity, and its decay (each bit clears with
    /// probability 2^-tau a step; tau 0: no persistence)
    trace: BitVector,
    tau: u32,
    /// recurrent self-excitation: the autoassociative map, the gain, and the gate (open: this
    /// step the input becomes the state)
    auto: Map,
    gain: u32,
    gate: bool,
    /// superposed expectation: layer 2/3 expects every next input its context supports (the
    /// union of the cells that burst, or of all that fired when none did, up to four), not only
    /// its winner's, so an alternative it has seen follow is not a surprise
    pub superpose: bool,
    /// the held population's state, and each held bit's age (steps since it was loaded)
    held: BitVector,
    age: Vec<u16>,
    /// steps a held bit is restored by completion before adaptation lets it go (the area's
    /// persistence)
    pub life: u16,
    /// (loads on surprise, loads by the gate) for diagnosis
    pub loads: (u64, u64),
}

impl Column {
    /// A granular column (sensory or association): layer 4 is `l4`.
    pub fn granular(bits: usize, l4: Pool, cap: usize) -> Self {
        Self::make(bits, Some(l4), cap)
    }

    /// An agranular column (motor): no layer 4.
    pub fn agranular(bits: usize, cap: usize) -> Self {
        Self::make(bits, None, cap)
    }

    fn make(bits: usize, l4: Option<Pool>, cap: usize) -> Self {
        let z = BitVector::new(bits, Some(0));
        Self { bits, l4, l23: Map::new(bits, cap), l5: Map::new(bits, cap), rec: z.clone(), code: z.clone(), expected: None, last: None, hits: (0, 0), last_hit: 0, trace: z, tau: 0, auto: Map::new(bits, cap), gain: 0, gate: false, superpose: false, held: BitVector::new(bits, Some(0)), age: vec![0; bits], life: 8, loads: (0, 0) }
    }

    /// Persistence: each bit of the trace clears with probability 2^-`tau` a step (0: none).
    pub fn set_persistence(&mut self, tau: u32) {
        self.tau = tau;
    }

    /// A held population (`gain` 2 or more; 0: none).
    pub fn set_recurrence(&mut self, gain: u32) {
        self.gain = gain;
    }

    /// What the held population holds.
    pub fn held(&self) -> &BitVector {
        &self.held
    }

    /// The code from layer 4's input (`input`, the driven code) and the held population (see the
    /// module notes): the autoassociative map learns the driven code; the held state is loaded
    /// from it on surprise or by the gate, else maintained by completion; the code is both.
    fn recur<R: rand::Rng + ?Sized>(&mut self, input: &BitVector, learn: bool, rng: &mut R) -> BitVector {
        if self.gain < 2 {
            return input.clone();
        }
        let n_in = input.count_ones() as u32;
        let gate = std::mem::take(&mut self.gate);
        if learn && n_in > 0 {
            self.auto.learn(input, None, None, input, rng);
        }
        // maintained: each held bit ages; one past the area's persistence (`life`) adapts and
        // goes at random (half a step), and is not restored. The rest lose a random sixteenth
        // (noise), and the autoassociative map restores each learned pattern still mostly held
        // by them, the restored bits as old as the youngest of their pattern; what nothing
        // restores fades. Items held together in one sparse vector are completed together.
        if self.held.count_ones() > 0 {
            let life = self.life;
            let mut young = self.held.clone();
            let mut old = BitVector::new(self.bits, Some(0));
            for b in 0..self.bits {
                if self.held.bit_get(b) {
                    self.age[b] = self.age[b].saturating_add(1);
                    if self.age[b] > life {
                        young.bit_clear(b);
                        if rng.gen_range(0..2) == 0 {
                            old.bit_set(b);
                        }
                    }
                }
            }
            let mut noisy = young.clone();
            for x in noisy.as_words_mut() {
                *x &= !(rng.next_u64() & rng.next_u64() & rng.next_u64() & rng.next_u64());
            }
            let mut held = noisy.clone();
            if noisy.count_ones() > 0 && self.auto.predict(&noisy, None, None).is_some() {
                for (p, _) in self.auto.proposals() {
                    // a pattern is restored only where most of it is still held (and young)
                    if overlap(&p, &noisy) * 2 >= p.count_ones() as u32 {
                        let youngest = (0..self.bits).filter(|&b| p.bit_get(b) && noisy.bit_get(b)).map(|b| self.age[b]).min().unwrap_or(0);
                        for b in 0..self.bits {
                            if p.bit_get(b) && !held.bit_get(b) {
                                held.bit_set(b);
                                self.age[b] = youngest;
                            }
                        }
                    }
                }
            }
            for (w, y) in held.as_words_mut().iter_mut().zip(old.as_words()) {
                *w |= y;
            }
            self.held = held;
        }
        // a new item joins what is held (superposed: sparse codes barely overlap), at most `K`
        // bits of it: on the gate, what drives the area now; on surprise, what layer 2/3 did not
        // expect (acetylcholine). The held population keeps at most `HELD` bits (about four
        // items): beyond that, bits go at random
        let item: Option<BitVector> = if n_in == 0 {
            None
        } else if gate {
            self.loads.1 += 1;
            Some(Self::sparse(input.as_words(), 32, rng))
        } else {
            let novel: Vec<u64> = match self.expected.as_ref() {
                Some(e) => input.as_words().iter().zip(e.as_words()).map(|(a, b)| a & !b).collect(),
                None => input.as_words().to_vec(),
            };
            let n_novel: u32 = novel.iter().map(|w| w.count_ones()).sum();
            (n_novel >= 4).then(|| {
                self.loads.0 += 1;
                Self::sparse(&novel, 32, rng)
            })
        };
        if let Some(it) = item {
            // the map learns the item as it is held: what is held often becomes an attractor
            if learn {
                self.auto.learn(&it, None, None, &it, rng);
            }
            for b in 0..self.bits {
                if it.bit_get(b) {
                    self.age[b] = 0;
                }
            }
            let others: Vec<u64> = self.held.as_words().iter().zip(it.as_words()).map(|(h, i)| h & !i).collect();
            let keep = Self::sparse(&others, 128usize.saturating_sub(it.count_ones() as usize), rng);
            self.held = BitVector::from_words(keep.as_words().iter().zip(it.as_words()).map(|(a, b)| a | b).collect());
        }
        BitVector::from_words(input.as_words().iter().zip(self.held.as_words()).map(|(a, b)| a | b).collect())
    }

    /// At most `k` of the bits of `words`, chosen at random.
    fn sparse<R: rand::Rng + ?Sized>(words: &[u64], k: usize, rng: &mut R) -> BitVector {
        let mut on: Vec<usize> = Vec::new();
        for (w, &x) in words.iter().enumerate() {
            let mut x = x;
            while x != 0 {
                on.push(w * 64 + x.trailing_zeros() as usize);
                x &= x - 1;
            }
        }
        let mut v = BitVector::from_words(vec![0; words.len()]);
        for i in 0..k.min(on.len()) {
            let j = rng.gen_range(i..on.len());
            on.swap(i, j);
            v.bit_set(on[i]);
        }
        v
    }

    /// The trace decays one step.
    fn decay<R: rand::Rng + ?Sized>(&mut self, rng: &mut R) {
        if self.tau == 0 {
            for w in self.trace.as_words_mut() {
                *w = 0;
            }
            return;
        }
        for w in self.trace.as_words_mut() {
            let mut clear = !0u64;
            for _ in 0..self.tau {
                clear &= rng.next_u64();
            }
            *w &= !clear;
        }
    }

    /// The column's loop chose: with a held population, its gate opens (this step's driven code
    /// is loaded); with a trace, its activity joins the trace (or, with `replace`, becomes it).
    pub fn refresh(&mut self, replace: bool) {
        if self.gain > 0 {
            self.gate = true;
            return;
        }
        let rec = self.rec.clone();
        for (w, r) in self.trace.as_words_mut().iter_mut().zip(rec.as_words()) {
            *w = if replace { *r } else { *w | r };
        }
    }

    /// The persistent trace.
    pub fn trace(&self) -> &BitVector {
        &self.trace
    }

    fn project(&self, cells: &[usize]) -> BitVector {
        let mut v = BitVector::new(self.bits, Some(0));
        for &c in cells {
            for i in 0..4u64 {
                let mut h = (c as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
                h ^= h >> 31;
                v.bit_set((h % self.bits as u64) as usize);
            }
        }
        v
    }

    /// Sense (sources, as for a pool): layer 4 codes the input; layer 2/3 learns that this code
    /// followed what it last predicted from, then predicts the next from this code, its own
    /// activity and the tuft's context `ctx` (from above). Returns the code.
    pub fn sense<R: rand::Rng + ?Sized>(&mut self, input: &[(usize, &BitVector)], ctx: Option<&BitVector>, learn: bool, rng: &mut R) -> BitVector {
        let code = match self.l4.as_mut() {
            Some(p) if learn => p.learn(input),
            Some(p) => p.encode(input),
            None => input.first().map_or_else(|| BitVector::new(self.bits, Some(0)), |x| x.1.clone()),
        };
        self.sense_code(&code, ctx, learn, rng)
    }

    /// As `sense`, for a code already made (layer 4 bypassed: its code, or an input that is
    /// itself sparse).
    pub fn sense_code<R: rand::Rng + ?Sized>(&mut self, code: &BitVector, ctx: Option<&BitVector>, learn: bool, rng: &mut R) -> BitVector {
        let code = if self.gain > 0 { self.recur(code, learn, rng) } else { code.clone() };
        let code = &code;
        self.code = code.clone();
        self.last_hit = self.expected.as_ref().map_or(0, |e| overlap(e, &self.code));
        if self.code.count_ones() > 0 {
            if let Some(e) = self.expected.as_ref() {
                self.hits.0 += 1;
                self.hits.1 += (overlap(e, &self.code) * 2 >= self.code.count_ones() as u32) as u64;
            }
            if learn {
                if let Some((inp, c)) = self.last.take() {
                    let code = self.code.clone();
                    self.l23.learn(&inp, Some(&c), None, &code, rng);
                }
            }
            // the tuft: context from above, the column's own previous activity and its trace
            let mut c = self.rec.clone();
            for (w, y) in c.as_words_mut().iter_mut().zip(self.trace.as_words()) {
                *w |= y;
            }
            if let Some(x) = ctx {
                for (w, y) in c.as_words_mut().iter_mut().zip(x.as_words()) {
                    *w |= y;
                }
            }
            self.expected = self.l23.predict(&self.code, Some(&c), None);
            if self.superpose {
                if let Some(e) = self.expected.as_mut() {
                    let props = self.l23.proposals();
                    let burst = props.iter().any(|p| p.1);
                    for (p, _) in props.iter().filter(|p| p.1 || !burst).take(4) {
                        for (w, y) in e.as_words_mut().iter_mut().zip(p.as_words()) {
                            *w |= y;
                        }
                    }
                }
            }
            self.rec = self.project(&self.l23.last_fired());
            self.last = Some((self.code.clone(), c));
            // the trace: what fired stays on, fading
            if self.tau > 0 {
                self.decay(rng);
                let rec = self.rec.clone();
                for (w, r) in self.trace.as_words_mut().iter_mut().zip(rec.as_words()) {
                    *w |= r;
                }
            }
        } else {
            self.expected = None;
            self.last = None;
            if self.tau > 0 {
                self.decay(rng);
            }
        }
        self.code.clone()
    }

    /// Layer 2/3's recent activity (projected) and its persistent trace, as feedback to the
    /// tufts of the areas below and to the striatum.
    pub fn activity(&self) -> BitVector {
        if self.tau == 0 {
            return self.rec.clone();
        }
        BitVector::from_words(self.rec.as_words().iter().zip(self.trace.as_words()).map(|(a, b)| a | b).collect())
    }

    /// Motor: the command for what is meant (layer 5, the inverse model).
    pub fn command_for(&mut self, meant: &BitVector) -> Option<BitVector> {
        self.l5.predict(meant, None, None)
    }

    /// Motor: what a command will be sensed as (layer 2/3 with the command as input, the
    /// forward model; its prediction is the efference copy).
    pub fn predict_result(&mut self, command: &BitVector) -> Option<BitVector> {
        self.l23.predict(command, None, None)
    }

    /// Motor: a command and what it produced (babbling, or any act): layer 2/3 learns the
    /// result from the command, layer 5 the command from the result. Returns the forward
    /// model's error before learning (`Q16`: the share of predicted and sensed bits that do
    /// not match; all of it if nothing was predicted).
    pub fn observe<R: rand::Rng + ?Sized>(&mut self, command: &BitVector, sensed: &BitVector, rng: &mut R) -> u32 {
        let one = crate::fixed::ONE;
        let err = match self.l23.predict(command, None, None) {
            Some(p) => {
                let total = p.count_ones() as u32 + sensed.count_ones() as u32;
                if total == 0 {
                    0
                } else {
                    one - 2 * overlap(&p, sensed) * one / total
                }
            }
            None => one,
        };
        self.l23.learn(command, None, None, sensed, rng);
        self.l5.learn(sensed, None, None, command, rng);
        err
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    const BITS: usize = 2048;
    const K: usize = 32;

    fn rand_code(rng: &mut StdRng) -> BitVector {
        let all: Vec<usize> = (0..BITS).collect();
        BitVector::from_bits(&all.choose_multiple(rng, K).copied().collect::<Vec<_>>(), BITS)
    }

    #[test]
    fn a_motor_column_learns_to_say_and_to_predict_by_babbling() {
        let mut rng = StdRng::seed_from_u64(1);
        let n = 60;
        let sounds: Vec<BitVector> = (0..n).map(|_| rand_code(&mut rng)).collect();
        let commands: Vec<BitVector> = (0..n).map(|_| rand_code(&mut rng)).collect();
        let mut col = Column::agranular(BITS, 4096);
        assert!(col.command_for(&sounds[0]).is_none());
        let mut order: Vec<usize> = (0..n).collect();
        for _ in 0..10 {
            order.shuffle(&mut rng);
            for &w in &order {
                col.observe(&commands[w], &sounds[w], &mut rng);
            }
        }
        let (mut said, mut predicted) = (0, 0);
        for w in 0..n {
            said += col.command_for(&sounds[w]).map_or(false, |c| overlap(&c, &commands[w]) * 2 >= K as u32) as usize;
            predicted += col.predict_result(&commands[w]).map_or(false, |p| overlap(&p, &sounds[w]) * 2 >= K as u32) as usize;
        }
        assert!(said * 100 >= n * 95, "said right {said} of {n}");
        assert!(predicted * 100 >= n * 95, "predicted {predicted} of {n}");
    }

    #[test]
    fn a_sensory_column_predicts_what_comes_next() {
        let mut rng = StdRng::seed_from_u64(2);
        let seq: Vec<BitVector> = (0..6).map(|_| rand_code(&mut rng)).collect();
        let mut col = Column::granular(BITS, Pool::new(1, BITS, BITS, K, 1, 2), 4096);
        for _ in 0..30 {
            for x in &seq {
                col.sense(&[(0, x)], None, true, &mut rng);
            }
        }
        // the codes, then whether each one's prediction is the next one's code
        let codes: Vec<BitVector> = seq.iter().map(|x| col.l4.as_ref().unwrap().encode(&[(0, x)])).collect();
        let mut right = 0;
        for i in 0..seq.len() {
            col.sense(&[(0, &seq[i])], None, false, &mut rng);
            let next = &codes[(i + 1) % seq.len()];
            right += col.expected.as_ref().map_or(false, |e| overlap(e, next) * 2 >= K as u32) as usize;
        }
        assert!(right >= 5, "predicted the next input for {right} of 6");
    }

    #[test]
    fn the_held_population_keeps_a_surprising_input_until_the_next_surprise() {
        let mut rng = StdRng::seed_from_u64(3);
        let (a, b, scene) = (rand_code(&mut rng), rand_code(&mut rng), rand_code(&mut rng));
        let silent = BitVector::new(BITS, Some(0));
        let mut col = Column::agranular(BITS, 4096);
        col.set_recurrence(3);
        // a familiar scene, now and then A or B for a step (as the delayed probe's light)
        for i in 0..40 {
            for _ in 0..4 {
                col.sense_code(&scene, None, true, &mut rng);
            }
            col.sense_code(if i % 2 == 0 { &a } else { &b }, None, true, &mut rng);
        }
        for _ in 0..4 {
            col.sense_code(&scene, None, true, &mut rng);
        }
        // A, unexpected, is held while the expected scene drives the area, and in silence
        col.sense_code(&a, None, true, &mut rng);
        for step in 0..6 {
            let x = if step % 2 == 0 { &scene } else { &silent };
            let c = col.sense_code(x, None, true, &mut rng);
            assert!(overlap(col.held(), &a) >= 28, "step {step}: {} of A held", overlap(col.held(), &a));
            assert!(overlap(&c, &a) >= 28, "step {step}: A in the code");
        }
        // the gate loads what drives the area now
        col.refresh(false);
        col.sense_code(&b, None, true, &mut rng);
        assert!(overlap(col.held(), &b) >= 28, "B {} after the gate", overlap(col.held(), &b));
    }
}
