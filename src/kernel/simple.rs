use crate::bitvec::{ones_of, select_nth_bit, BitVector};
use crate::kernel::class::{KernelTrait, KernelOp, KernelContext};



#[derive(Clone)]
/// A simple “dendrite” kernel:
/// - Reads input ∧ input_mask, counts ones
/// - If count >= threshold, applies `op` of output_mask at `dest_word_idx` into `output`
pub struct SimpleKernel {
    pub input_mask: BitVector,
    pub input_idx: usize,  // word index in input where input_mask[0] lands
    pub output_mask: BitVector,
    pub output_idx: usize, // word index in output where output_mask[0] lands
    pub threshold: usize,     // fire when (input & input_mask).ones() >= threshold
    pub op: KernelOp,
    /// search when it does not fire: a near miss (at least half the threshold) moves one silent
    /// synapse onto an active line; a window with nothing on drifts one word left or right
    /// (chance 1/64 a step)
    pub search: bool,
    /// a two-mask kernel: the value it expects on each of its lines (`input_mask` is then its
    /// care mask). It counts the cared lines that agree with the pattern, ones and zeros alike,
    /// so it can match dense input (raw bytes). None: it counts cared lines that are on.
    pub pattern: Option<BitVector>,
    /// learn anti-Hebbian when matched but held down (on by default)
    pub learn_held: bool,
    /// learn Hebbian when it fires (on by default)
    pub learn_fire: bool,
    /// its match code: a small code written into a match vector when it matches
    pub match_code: Option<BitVector>,
    /// the extra lines of its confirmed code: when what it predicted came true, it writes its
    /// match code plus these into the same vector (a bigger code). A reader of that vector sees
    /// nothing (not this context), the small code (this context, and its prediction failed:
    /// surprise bound to context), or the big code (this context, and it came true).
    pub confirm_mask: Option<BitVector>,
    /// how often its prediction came true lately: up one when confirmed, down one when not
    /// (a u8, saturating); breaks ties between equally good matches
    pub confirmed: u8,
    /// steps since it last fired (saturating)
    pub age: u8,
    /// activity: 0, it fires only when its input matches (responsive); k > 0, it also fires
    /// spontaneously with chance 2^-k a step (tonic), writing its output without learning
    pub tonic: u8,
    /// a sparse kernel: the mask words it reads (offsets into `input_mask`). Matching, inhibition
    /// and learning then visit only these words, so a kernel spread across a wide field costs its
    /// lines, not the field; its lines move only within these words. None: every mask word.
    pub words: Option<Vec<u32>>,
    /// how many lines its mask has (learning moves lines, so this stays fixed); counted once
    lines: usize,
    /// a tile kernel: per mask word, the lines it may use (its tile); its lines move only there
    pub room: Option<Vec<u64>>,
    /// a tile kernel's tile: (first bit, bits)
    pub tile_at: Option<(usize, usize)>,
}

impl SimpleKernel {
    pub fn new(input_mask: BitVector, input_idx: usize, output_mask: BitVector, output_idx: usize, threshold: usize, op: KernelOp) -> Self {
        Self { input_mask, input_idx, output_mask, output_idx, threshold, op, search: false, pattern: None, learn_held: true, learn_fire: true, match_code: None, confirm_mask: None, confirmed: 0, age: u8::MAX, tonic: 0, words: None, lines: 0, room: None, tile_at: None }.counted()
    }

    /// A tile kernel: it reads bits `start .. start + bits` of its input (a tile, which may
    /// straddle words), with `lines` (absolute bit indices inside the tile) as its mask; its
    /// lines move only within the tile.
    pub fn tile(start: usize, bits: usize, lines: &[usize], output_mask: BitVector, output_idx: usize, threshold: usize, op: KernelOp) -> Self {
        let (w0, w1) = (start / 64, (start + bits.max(1) - 1) / 64);
        let n = w1 - w0 + 1;
        let mut mask = BitVector::new(n * 64, Some(0));
        for &b in lines {
            mask.bit_set(b - w0 * 64);
        }
        let room = (w0..=w1).map(|w| tile_word(start, bits, w)).collect();
        let mut k = Self::new(mask, w0, output_mask, output_idx, threshold, op);
        k.words = Some((0..n as u32).collect());
        k.room = Some(room);
        k.tile_at = Some((start, bits));
        k
    }

    /// Its lines as bit indices of its input.
    pub fn lines_abs(&self) -> Vec<usize> {
        self.input_mask.as_words().iter().enumerate().flat_map(|(j, &w)| ones_of(w).into_iter().map(move |b| (self.input_idx + j) * 64 + b)).collect()
    }

    /// A copy of this tile kernel on the tile starting at `start` (its lines at the same places
    /// inside the tile), writing `output_mask` at `output_idx`; it starts unconfirmed.
    pub fn moved_to(&self, start: usize, output_idx: usize) -> Self {
        let Some((s0, bits)) = self.tile_at else { return self.clone() };
        let lines: Vec<usize> = self.lines_abs().into_iter().map(|b| b - s0 + start).collect();
        let mut k = Self::tile(start, bits, &lines, self.output_mask.clone(), output_idx, self.threshold, self.op);
        k.tonic = self.tonic;
        k.learn_held = self.learn_held;
        k.learn_fire = self.learn_fire;
        k
    }

    /// The lines of word `w` inside bits `start .. start + bits`.
    pub fn tile_word(start: usize, bits: usize, w: usize) -> u64 {
        tile_word(start, bits, w)
    }

    /// Whether it can do anything on an input with no lines on: fire spontaneously (tonic), match
    /// expected zeros (a two-mask kernel), drift (search), or fire with a zero threshold.
    pub fn acts_on_nothing(&self) -> bool {
        self.tonic > 0 || self.pattern.is_some() || self.search || self.threshold == 0
    }

    /// Count its mask's lines once.
    fn counted(mut self) -> Self {
        self.lines = self.input_mask.count_ones();
        self
    }

    /// A two-mask kernel: `care` lines, expecting `pattern` on them.
    pub fn with_pattern(care: BitVector, pattern: BitVector, input_idx: usize, output_mask: BitVector, output_idx: usize, threshold: usize, op: KernelOp) -> Self {
        let mut k = Self::new(care, input_idx, output_mask, output_idx, threshold, op);
        k.pattern = Some(pattern);
        k
    }

    /// Its strength given its match count: the share of its lines that agree (in 1/4096), then
    /// its width, then how often it was confirmed.
    fn strength_of(&self, count: usize) -> usize {
        let care = self.care().max(1);
        let share = count * 4096 / care;
        (share << 24) | (care.min(65535) << 8) | self.confirmed as usize
    }

    /// Make it sparse: from now on it visits only the mask words that have lines.
    pub fn sparsify(&mut self) {
        let ws = self.input_mask.as_words().iter().enumerate().filter(|(_, m)| **m != 0).map(|(w, _)| w as u32).collect();
        self.words = Some(ws);
    }

    /// Input word `w` of its mask (0 past the input's end).
    #[inline]
    fn at(&self, input: &BitVector, w: u32) -> u64 {
        input.as_words().get(self.input_idx + w as usize).copied().unwrap_or(0)
    }

    /// How many lines its mask has.
    #[inline]
    pub fn care(&self) -> usize {
        self.lines
    }

    /// A sparse kernel's learning move: one line from a candidate `src(a, m)` to a candidate
    /// `dst(a, m)` (a the input word, m the mask word), both within its words.
    fn sparse_move<R: rand::Rng + ?Sized>(&mut self, input: &BitVector, src: impl Fn(u64, u64) -> u64, dst: impl Fn(u64, u64) -> u64, rng: &mut R) {
        let Some(ws) = &self.words else { return };
        let m = self.input_mask.as_words();
        let room = |w: u32| self.room.as_ref().map_or(u64::MAX, |r| r[w as usize]);
        let cand: Vec<(u32, u64, u64)> = ws.iter().map(|&w| { let a = self.at(input, w); (w, src(a, m[w as usize]), dst(a, m[w as usize]) & room(w)) }).collect();
        let pick = |sel: &dyn Fn(&(u32, u64, u64)) -> u64, rng: &mut R| -> Option<(usize, u32)> {
            let n: u32 = cand.iter().map(|c| sel(c).count_ones()).sum();
            if n == 0 { return None; }
            let mut k = rng.gen_range(0..n);
            for c in &cand {
                let x = sel(c);
                if k < x.count_ones() {
                    return Some((c.0 as usize, select_nth_bit(x, k as usize) as u32));
                }
                k -= x.count_ones();
            }
            None
        };
        let (Some((sw, sb)), Some((dw, db))) = (pick(&|c| c.1, rng), pick(&|c| c.2, rng)) else { return };
        let m = self.input_mask.as_words_mut();
        m[sw] &= !(1u64 << sb);
        m[dw] |= 1u64 << db;
    }

    /// How many of its lines agree with the input (on lines that are on; or, with a pattern,
    /// cared lines whose value matches).
    pub fn agreement(&self, input: &BitVector) -> usize {
        if let Some(ws) = &self.words {
            let m = self.input_mask.as_words();
            return match &self.pattern {
                None => ws.iter().map(|&w| (self.at(input, w) & m[w as usize]).count_ones() as usize).sum(),
                Some(p) => { let q = p.as_words(); ws.iter().map(|&w| (!(self.at(input, w) ^ q[w as usize]) & m[w as usize]).count_ones() as usize).sum() }
            };
        }
        match &self.pattern {
            None => input.mask_and_count(self.input_idx, &self.input_mask, |a, m| a & m),
            Some(p) => {
                let x = input.as_slice(self.input_idx, self.input_mask.word_len());
                x.iter().zip(self.input_mask.as_words()).zip(p.as_words()).map(|((a, m), q)| (!(a ^ q) & m).count_ones() as usize).sum()
            }
        }
    }

    /// Move one cared line: from a line where the input `agrees` (true) or disagrees (false)
    /// with the pattern, to an uncared line of the same words, whose expected value is set so
    /// that it now disagrees (anti-Hebbian) or agrees (Hebbian) with the input.
    fn move_pattern_line<R: rand::Rng + ?Sized>(&mut self, input: &BitVector, from_agreeing: bool, rng: &mut R) {
        let Some(p) = self.pattern.as_mut() else { return };
        let x = input.as_slice(self.input_idx, self.input_mask.word_len()).to_vec();
        let m = self.input_mask.as_words().to_vec();
        let q = p.as_words().to_vec();
        let pick = |words: Vec<u64>, rng: &mut R| -> Option<usize> {
            let n: u32 = words.iter().map(|w| w.count_ones()).sum();
            if n == 0 { return None; }
            let mut k = rng.gen_range(0..n);
            for (i, w) in words.iter().enumerate() {
                let c = w.count_ones();
                if k < c {
                    return Some(i * 64 + select_nth_bit(*w, k as usize));
                }
                k -= c;
            }
            None
        };
        let agree: Vec<u64> = x.iter().zip(&m).zip(&q).map(|((a, m), q)| !(a ^ q) & m).collect();
        let from: Vec<u64> = agree.iter().zip(&m).map(|(g, m)| if from_agreeing { *g } else { !g & m }).collect();
        // a sparse kernel's lines stay within its words
        let mine = |i: usize| self.words.as_ref().map_or(true, |ws| ws.contains(&(i as u32)));
        let room = |i: usize| self.room.as_ref().map_or(u64::MAX, |r| r[i]);
        let free: Vec<u64> = m.iter().enumerate().map(|(i, m)| if mine(i) { !m & room(i) } else { 0 }).collect();
        let (Some(a), Some(b)) = (pick(from, rng), pick(free, rng)) else { return };
        self.input_mask.bit_clear(a);
        self.input_mask.bit_set(b);
        let xb = x[b / 64] >> (b % 64) & 1 == 1;
        // Hebbian: expect what is there; anti-Hebbian: expect the opposite
        if xb == from_agreeing { p.bit_clear(b) } else { p.bit_set(b) }
    }

    pub fn default(output_bit: usize) -> Self {
        Self {
            input_mask: BitVector::new(64, Some(0xF0F0_F0F0_F0F0_F0F0)),
            input_idx: 0,
            output_mask: BitVector::from_bits(&[output_bit&63],1),
            output_idx: output_bit >> 6,
            threshold: 16,
            op: KernelOp::Or,
            search: false,
            pattern: None,
            learn_held: true,
            learn_fire: true,
            match_code: None,
            confirm_mask: None,
            confirmed: 0,
            age: u8::MAX,
            tonic: 0,
            words: None,
            lines: 32,
            room: None,
            tile_at: None,
        }
    }

    #[inline]
    fn input_stats(&self, ctx: &KernelContext, count: Option<usize>) -> InputStats {
        let count = count.unwrap_or_else(|| self.agreement(ctx.input));
        
        //The firing threshold is a function of "temperature" (as in simulated annealing)
        // - Higher temperature means we are more likely to fire (lower threshold)
        // It's also a function of the phase (to introduce oscillations)
        let firing_thresh = if ctx.temperature > 0 {
            self.threshold.saturating_sub(ctx.temperature as usize) + (ctx.phase & 0x10) as usize
        } else {
            self.threshold + (ctx.phase & 0x10) as usize
        };
        InputStats { threshold:firing_thresh, count }
    }

    #[inline]
    fn is_inhibited(&self, ctx: &KernelContext, _stats: &InputStats) -> bool {
        // held down when more than half of our mask overlaps lines a fired kernel took
        let taken = match &self.words {
            Some(ws) => { let m = self.input_mask.as_words(); ws.iter().map(|&w| (self.at(ctx.inhibit, w) & m[w as usize]).count_ones() as usize).sum() }
            None => ctx.inhibit.mask_and_count(self.input_idx, &self.input_mask, |a, m| a & m),
        };
        2 * taken > self.care()
    }

    #[inline]
    fn search_remap<R: rand::Rng + ?Sized>(&mut self, ctx: &KernelContext, _stats: &InputStats, rng: &mut R) {
        // Example placeholder: move one connected+active to inactive+unconnected
        // (use your mask_move_random_* helpers)
        // If inhibited we SEARCH for a new pattern to connect to
        //  eg. [SEARCH] CLEAR a CONNECTED + ACTIVE bit and choose a random INACTIVE + UNCONNECTED bit to set.
        if self.words.is_some() {
            return self.sparse_move(ctx.input, |a, m| a & m, |a, m| !a & !m, rng);
        }
        ctx.input.mask_move_random_connected_and_set(self.input_idx, &mut self.input_mask, rng);
    }

    #[inline]
    fn strengthen<R: rand::Rng + ?Sized>(&mut self, ctx: &KernelContext, _stats: &InputStats, rng: &mut R,) {
        // Eg: move one connected+inactive to active+unconnected
        //If not inhibited we STRENGTHEN our active pattern
        // eg. [STRENGTHEN] CLEAR a CONNECTED + INACTIVE bit and choose a random ACTIVE + UNCONNECTED bit to SET.
        if self.words.is_some() {
            return self.sparse_move(ctx.input, |a, m| !a & m, |a, m| a & !m, rng);
        }
        ctx.input.mask_move_random_connected_and_not_set(self.input_idx, &mut self.input_mask, rng);
    }

    #[inline]
    fn apply_output(&self, out: &mut BitVector) {
        let op = self.op;
        let f = move |a: u64, b: u64| op.word(a, b);
        out.mask_mut(self.output_idx, &self.output_mask, f);
    }
}

impl SimpleKernel {
    /// Extinction: it acted (fired) and what should follow did not. One synapse moves off the
    /// lines it read then (`then`, the input of that time) onto a silent line (anti-Hebbian).
    pub fn fade<R: rand::Rng + ?Sized>(&mut self, then: &BitVector, rng: &mut R) {
        if self.words.is_some() {
            return self.sparse_move(then, |a, m| a & m, |a, m| !a & !m, rng);
        }
        then.mask_move_random_connected_and_set(self.input_idx, &mut self.input_mask, rng);
    }

    /// A small code and its extra lines for a nested match code: `small` random lines in one
    /// random word of a `bits`-wide match vector, and `extra` more lines in the same word.
    pub fn nested_code<R: rand::Rng + ?Sized>(bits: usize, small: usize, extra: usize, rng: &mut R) -> (BitVector, BitVector) {
        use rand::seq::SliceRandom;
        let w = rng.gen_range(0..(bits / 64).max(1));
        let lines: Vec<usize> = (0..64).collect();
        let picked: Vec<usize> = lines.choose_multiple(rng, (small + extra).min(64)).map(|b| w * 64 + b).collect();
        let (a, b) = picked.split_at(small.min(picked.len()));
        (BitVector::from_bits(a, bits), BitVector::from_bits(b, bits))
    }

    /// It matched: its small code goes into the match vector.
    pub fn write_match(&self, matches: &mut BitVector) {
        if let Some(m) = &self.match_code {
            matches.mask_mut(0, m, |a, b| a | b);
        }
    }

    /// Its prediction was checked. Confirmed, it writes its big code (match code plus extra
    /// lines) into `matches` (if given) and counts up; else it counts down.
    pub fn confirm(&mut self, right: bool, matches: Option<&mut BitVector>) {
        if right {
            self.confirmed = self.confirmed.saturating_add(1);
            if let Some(c) = matches {
                self.write_match(c);
                if let Some(m) = &self.confirm_mask {
                    c.mask_mut(0, m, |a, b| a | b);
                }
            }
        } else {
            self.confirmed = self.confirmed.saturating_sub(1);
        }
    }

    /// Output learning for a kernel that cancels (writes by XOR or clear), after the tick, with
    /// `dest` its destination as the tick left it: each line it wrote that is still on (its
    /// cancel failed: what it predicted did not come) moves to a line of the
    /// destination still on that it does not write. It learns to cancel what is left; its output
    /// keeps its number of lines. Each failed line moves with chance 2^-chance_shift (1/16 by
    /// the grammar's starting value): slower than success,
    /// so a prediction that comes true half the time is kept while one that never comes fades.
    pub fn learn_output<R: rand::Rng + ?Sized>(&mut self, dest: &BitVector, chance_shift: u8, rng: &mut R) {
        let words = self.output_mask.word_len();
        let d = dest.as_slice(self.output_idx, words);
        let o = self.output_mask.as_words();
        let mut failed: Vec<usize> = Vec::new();
        let mut left: Vec<usize> = Vec::new();
        for w in 0..words {
            let (f, l) = (o[w] & d[w], d[w] & !o[w]);
            failed.extend(ones_of(f).into_iter().map(|b| w * 64 + b));
            left.extend(ones_of(l).into_iter().map(|b| w * 64 + b));
        }
        for b in failed {
            if left.is_empty() {
                break;
            }
            if rng.gen_range(0..1u32 << chance_shift.min(31)) == 0 {
                let i = rng.gen_range(0..left.len());
                let to = left.swap_remove(i);
                self.output_mask.bit_clear(b);
                self.output_mask.bit_set(to);
            }
        }
    }
}

/// The lines of word `w` inside bits `start .. start + bits`.
pub fn tile_word(start: usize, bits: usize, w: usize) -> u64 {
    let (lo, hi) = (start.max(w * 64), (start + bits).min(w * 64 + 64));
    if lo >= hi {
        return 0;
    }
    let n = hi - lo;
    (if n == 64 { u64::MAX } else { (1u64 << n) - 1 }) << (lo - w * 64)
}

pub struct InputStats {
    pub threshold: usize, 
    pub count: usize,
}

impl KernelTrait for SimpleKernel {
    fn try_fire<R: rand::Rng + ?Sized>(
        &mut self,
        ctx: &KernelContext,
        out: &mut BitVector,
        rng: &mut R,
    ) -> bool {
        self.try_fire_counted(ctx, None, out, rng)
    }

    fn rank(&self, input: &BitVector) -> (usize, Option<usize>) {
        let count = self.agreement(input);
        (self.strength_of(count), Some(count))
    }

    fn try_fire_counted<R: rand::Rng + ?Sized>(
        &mut self,
        ctx: &KernelContext,
        count: Option<usize>,
        out: &mut BitVector,
        rng: &mut R,
    ) -> bool {
        let stats = self.input_stats(ctx, count);
        if stats.count < stats.threshold {
            // tonic: a spontaneous firing (no input, so nothing to learn)
            if self.tonic > 0 && rng.gen_range(0..1u32 << self.tonic.min(31)) == 0 && !self.is_inhibited(ctx, &stats) {
                self.apply_output(out);
                self.age = 0;
                return true;
            }
            if self.search {
                if stats.count > 0 && 2 * stats.count >= stats.threshold {
                    self.strengthen(ctx, &stats, rng);
                } else if stats.count == 0 && rng.gen_range(0..64) == 0 {
                    let last = ctx.input.word_len().saturating_sub(self.input_mask.word_len());
                    self.input_idx = if rng.gen_bool(0.5) { self.input_idx.saturating_sub(1) } else { (self.input_idx + 1).min(last) };
                }
            }
            return false;
        }
        if self.is_inhibited(ctx, &stats) {
            if self.learn_held {
                if self.pattern.is_some() {
                    self.move_pattern_line(ctx.input, true, rng);
                } else {
                    self.search_remap(ctx, &stats, rng);
                }
            }
            return false;
        }
        if !self.learn_fire {
        } else if self.pattern.is_some() {
            self.move_pattern_line(ctx.input, false, rng);
        } else {
            self.strengthen(ctx, &stats, rng);
        }
        debug_assert_eq!(self.lines, self.input_mask.count_ones(), "learning keeps a mask's lines");
        self.apply_output(out);
        self.age = 0;
        true
    }

    /// The share of its lines that agree (in 1/4096), then its width, then how often it was
    /// confirmed: a better match is a larger share; among equal shares, the wider (more
    /// context), then the more often right.
    fn strength(&self, input: &BitVector) -> usize {
        self.strength_of(self.agreement(input))
    }

    fn mark_inhibit(&self, inhibit: &mut BitVector) {
        if let Some(ws) = &self.words {
            let m = self.input_mask.as_words();
            let iw = inhibit.as_words_mut();
            for &w in ws {
                if let Some(x) = iw.get_mut(self.input_idx + w as usize) { *x |= m[w as usize]; }
            }
            return;
        }
        inhibit.mask_mut(self.input_idx, &self.input_mask, |a, m| a | m);
    }

    fn clear_inhibit(&self, inhibit: &mut BitVector) {
        let iw = inhibit.as_words_mut();
        match &self.words {
            Some(ws) => {
                for &w in ws {
                    if let Some(x) = iw.get_mut(self.input_idx + w as usize) { *x = 0; }
                }
            }
            None => {
                let end = (self.input_idx + self.input_mask.word_len()).min(iw.len());
                if self.input_idx < end { iw[self.input_idx..end].fill(0); }
            }
        }
    }

    fn word_range(&self) -> (usize, usize) {
        let start = self.output_idx;
        let end = start + self.output_mask.word_len();
        (start, end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitvec::BitVector;

    #[test]
    fn match_and_confirmation_write_nested_codes_to_one_vector() {
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(3);
        let mut k = SimpleKernel::new(BitVector::from_words(vec![0xFF]), 0, BitVector::from_words(vec![1]), 0, 8, KernelOp::Or);
        let (small, extra) = SimpleKernel::nested_code(512, 4, 4, &mut rng);
        k.match_code = Some(small.clone());
        k.confirm_mask = Some(extra.clone());
        // matched: the small code
        let mut m = BitVector::new(512, Some(0));
        k.write_match(&mut m);
        assert_eq!(m.count_ones(), 4);
        // its prediction failed: nothing more, and it counts down (from 0: stays 0)
        k.confirm(false, Some(&mut m));
        assert_eq!((m.count_ones(), k.confirmed), (4, 0));
        // confirmed: the big code, a superset of the small one, in the same word
        let mut c = BitVector::new(512, Some(0));
        k.confirm(true, Some(&mut c));
        assert_eq!((c.count_ones(), k.confirmed), (8, 1));
        let both: usize = c.as_words().iter().zip(small.as_words()).map(|(a, b)| (a & b).count_ones() as usize).sum();
        assert_eq!(both, 4);
        assert_eq!(c.as_words().iter().filter(|&&w| w != 0).count(), 1);
    }

    #[test]
    fn a_sparse_kernel_matches_and_learns_like_a_dense_one_within_its_words() {
        // lines in words 0 and 1000 of a 1024-word field
        let mut mask = BitVector::new(1024 * 64, Some(0));
        for b in [1, 2, 3, 4, 64_000, 64_001, 64_002, 64_003] { mask.bit_set(b); }
        let mut input = BitVector::new(1024 * 64, Some(0));
        for b in [1, 2, 3, 64_000, 64_001, 64_005] { input.bit_set(b); }
        let dense = SimpleKernel::new(mask.clone(), 0, BitVector::from_words(vec![1]), 0, 4, KernelOp::Or);
        let mut sparse = SimpleKernel::new(mask, 0, BitVector::from_words(vec![1]), 0, 4, KernelOp::Or);
        sparse.sparsify();
        assert_eq!(sparse.words.as_deref(), Some(&[0u32, 1000][..]));
        assert_eq!(dense.agreement(&input), 5);
        assert_eq!(sparse.agreement(&input), 5);
        assert_eq!(sparse.care(), 8);
        // Hebbian on firing: a line moves from an off line to an on line, in its own words
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(1);
        let inhibit = BitVector::new(1024 * 64, Some(0));
        let mut out = BitVector::new(64, Some(0));
        assert!(sparse.try_fire(&KernelContext { input: &input, inhibit: &inhibit, temperature: 0, phase: 0 }, &mut out, &mut rng));
        assert_eq!(sparse.care(), 8);
        assert_eq!(sparse.agreement(&input), 6);
        let m = sparse.input_mask.as_words();
        assert_eq!(m.iter().enumerate().filter(|(w, x)| **x != 0 && *w != 0 && *w != 1000).count(), 0);
    }

    #[test]
    fn a_tile_kernel_straddles_words_and_learns_inside_its_tile() {
        // a 64-bit tile from bit 32: the top half of word 0 and the bottom half of word 1
        assert_eq!(tile_word(32, 64, 0), 0xFFFF_FFFF_0000_0000);
        assert_eq!(tile_word(32, 64, 1), 0x0000_0000_FFFF_FFFF);
        assert_eq!(tile_word(32, 64, 2), 0);
        let mut k = SimpleKernel::tile(32, 64, &[40, 41, 70, 71], BitVector::from_words(vec![1]), 0, 3, KernelOp::Or);
        assert_eq!((k.input_idx, k.care()), (0, 4));
        let mut input = BitVector::new(256, Some(0));
        for b in [40, 41, 70, 72, 73, 10, 200] { input.bit_set(b); }
        assert_eq!(k.agreement(&input), 3);
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(1);
        let inhibit = BitVector::new(256, Some(0));
        let mut out = BitVector::new(64, Some(0));
        for _ in 0..8 {
            k.try_fire(&KernelContext { input: &input, inhibit: &inhibit, temperature: 0, phase: 0 }, &mut out, &mut rng);
        }
        // Hebbian moves went to on lines inside the tile (72, 73), never to 10 or 200
        assert_eq!(k.agreement(&input), 4);
        let m = k.input_mask.as_words();
        assert_eq!(m[0] & !tile_word(32, 64, 0), 0);
        assert_eq!(m[1] & !tile_word(32, 64, 1), 0);
    }

    #[test]
    fn a_tile_kernel_moves_to_another_tile_keeping_its_shape() {
        let k = SimpleKernel::tile(32, 64, &[40, 41, 70], BitVector::from_words(vec![1]), 0, 2, KernelOp::Xor);
        let m = k.moved_to(128, 3);
        assert_eq!(m.lines_abs(), vec![136, 137, 166]);
        assert_eq!((m.tile_at, m.output_idx, m.threshold, m.care()), (Some((128, 64)), 3, 2, 3));
    }

    #[test]
    fn fires_when_threshold_met() {
        // input has ones at bits 0..=7 (first byte)
        let input = BitVector::from_words(vec![0xFF]);
        // input mask checks those same bits
        let in_mask = BitVector::from_words(vec![0xFF]);

        // output starts zero
        let mut output = BitVector::new(64, Some(0));
        // output mask sets upper nibble of first byte
        let out_mask = BitVector::from_words(vec![0xF0]);

        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(1);
        let ctx = KernelContext {
            input: &input,
            inhibit: &BitVector::new(64, Some(0)),
            temperature: 0,
            phase: 0,
        };

        let mut k = SimpleKernel::new(in_mask, 0, out_mask, 0, 8, KernelOp::Or);
        k.try_fire(&ctx, &mut output, &mut rng);

        // Expect upper nibble set
        assert_eq!(output.as_words()[0] & 0xFF, 0xF0);
    }

    #[test]
    fn does_not_fire_below_threshold() {
        let input = BitVector::from_words(vec![0x0F]); // 4 ones
        let in_mask = BitVector::from_words(vec![0xFF]);

        let mut output = BitVector::new(64, Some(0));
        let out_mask = BitVector::from_words(vec![0x0F]);

        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(1);
        let ctx = KernelContext {
            input: &input,
            inhibit: &BitVector::new(64, Some(0)),
            temperature: 0,
            phase: 0,
        };

        let mut k = SimpleKernel::new(in_mask, 0, out_mask, 0, 5, KernelOp::Or);
        k.try_fire(&ctx, &mut output, &mut rng);

        assert_eq!(output.as_words()[0] & 0xFF, 0x00);
    }

    #[test]
    fn clear_mode_works() {
        let input = BitVector::from_words(vec![0xFFFF_FFFF_FFFF_FFFF]);
        let in_mask = BitVector::from_words(vec![0xFFFF_FFFF_FFFF_FFFF]);

        let mut output = BitVector::from_words(vec![0xFFFF]);
        let out_mask = BitVector::from_words(vec![0x0F0F]);

        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(1);
        let ctx = KernelContext {
            input: &input,
            inhibit: &BitVector::new(64, Some(0)),
            temperature: 0,
            phase: 0,
        };

        let mut k = SimpleKernel::new(in_mask, 0, out_mask, 0, 1, KernelOp::Clear);
        k.try_fire(&ctx, &mut output, &mut rng);

        assert_eq!(output.as_words()[0] & 0xFFFF, 0xF0F0);
    }

    #[test]
    fn reacts_to_input_update() {
        let mut input = BitVector::from_words(vec![0x0F]); // 4 ones
        let in_mask = BitVector::from_words(vec![0xFF]);

        let mut output = BitVector::new(64, Some(0));
        let out_mask = BitVector::from_words(vec![0x0F]);

        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(1);

        let mut k = SimpleKernel::new(in_mask, 0, out_mask, 0, 5, KernelOp::Or);
        let inhibit = BitVector::new(64, Some(0));
        k.try_fire(&KernelContext {
            input: &input,
            inhibit: &inhibit,
            temperature: 0,
            phase: 0,
        }, &mut output, &mut rng);

        assert_eq!(output.as_words()[0] & 0xFF, 0x00);

        input.bit_set(4); // now 5 ones (fires >= threshold of 5)
        assert_eq!(input.count_ones(), 5);

        k.try_fire(
            &KernelContext {
                input: &input,
                inhibit: &inhibit,
                temperature: 0,
                phase: 0,
            }, 
            &mut output,
            &mut rng);
        assert_eq!(output.as_words()[0] & 0xFF, 0x0F);

        output.bit_clear_all();
        assert_eq!(output.count_ones(), 0);
        input.bit_set(5); // now 6 ones
        assert_eq!(input.count_ones(), 6);
        k.try_fire(
            &KernelContext {
                input: &input,
                inhibit: &inhibit,
                temperature: 0,
                phase: 0,
            }, 
            &mut output,
            &mut rng);
        assert_eq!(output.as_words()[0] & 0xFF, 0x0F);
    }
}