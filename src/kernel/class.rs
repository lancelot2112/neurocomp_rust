use crate::bitvec::BitVector;
use crate::common::config;

/// How a write lands in its destination word: `d` is the destination word, `m` what is written
/// (a kernel's output mask, or a wire's copied word). Every op is one cheap word operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelOp {
    Or,
    And,
    Xor,
    Clear, // a & !b
    Nand,  // !(d & m)
    Nor,   // !(d | m)
    /// the destination word becomes what is written (an overwrite: a held vector stores a new
    /// value in one write, where `Or` would pile it onto the old)
    Set,
    /// integer add / subtract, in lanes of 2^k bits (k = 3..6: 8, 16, 32, 64), wrapping in a lane
    Add(u8),
    Sub(u8),
    /// what is written is shifted or rotated by this many bits (0..63), then ORed in: a word
    /// carried from a vector's previous frame, shifted, makes a sliding window
    Shl(u8),
    Shr(u8),
    Rotl(u8),
    Rotr(u8),
}

/// The op choice (`set_op_choice`): ops drawn without the redundant `Rotr`, and a population's op
/// re-drawn by its own mutation (in the tag grammar), so an op such as `clear` or `xor` is tried
/// far more often than when it changes only as one of a population's many properties. Off by
/// default.
static OP_CHOICE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_op_choice(on: bool) {
    OP_CHOICE.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn op_choice() -> bool {
    OP_CHOICE.load(std::sync::atomic::Ordering::Relaxed)
}

/// The top bit of each lane of 2^k bits.
fn lane_tops(k: u8) -> u64 {
    match k.clamp(3, 6) {
        3 => 0x8080_8080_8080_8080,
        4 => 0x8000_8000_8000_8000,
        5 => 0x8000_0000_8000_0000,
        _ => 0x8000_0000_0000_0000,
    }
}

impl KernelOp {
    /// The destination word after the write.
    #[inline]
    pub fn word(self, d: u64, m: u64) -> u64 {
        match self {
            KernelOp::Or => d | m,
            KernelOp::And => d & m,
            KernelOp::Xor => d ^ m,
            KernelOp::Clear => d & !m,
            KernelOp::Nand => !(d & m),
            KernelOp::Nor => !(d | m),
            KernelOp::Set => m,
            KernelOp::Add(k) => {
                if k >= 6 { return d.wrapping_add(m); }
                let h = lane_tops(k);
                ((d & !h).wrapping_add(m & !h)) ^ ((d ^ m) & h)
            }
            KernelOp::Sub(k) => {
                if k >= 6 { return d.wrapping_sub(m); }
                let h = lane_tops(k);
                ((d | h).wrapping_sub(m & !h)) ^ ((d ^ !m) & h)
            }
            KernelOp::Shl(n) => d | (m << (n & 63)),
            KernelOp::Shr(n) => d | (m >> (n & 63)),
            KernelOp::Rotl(n) => d | m.rotate_left((n & 63) as u32),
            KernelOp::Rotr(n) => d | m.rotate_right((n & 63) as u32),
        }
    }

    /// As text: `or`, `add:4`, `rotl:13`, ... (read back by `parse`).
    pub fn text(self) -> String {
        match self {
            KernelOp::Or => "or".into(),
            KernelOp::And => "and".into(),
            KernelOp::Xor => "xor".into(),
            KernelOp::Clear => "clear".into(),
            KernelOp::Nand => "nand".into(),
            KernelOp::Nor => "nor".into(),
            KernelOp::Set => "set".into(),
            KernelOp::Add(k) => format!("add:{k}"),
            KernelOp::Sub(k) => format!("sub:{k}"),
            KernelOp::Shl(n) => format!("shl:{n}"),
            KernelOp::Shr(n) => format!("shr:{n}"),
            KernelOp::Rotl(n) => format!("rotl:{n}"),
            KernelOp::Rotr(n) => format!("rotr:{n}"),
        }
    }

    /// The op `text` wrote.
    pub fn parse(s: &str) -> Option<Self> {
        let (name, arg) = s.split_once(':').map_or((s, None), |(a, b)| (a, b.parse::<u8>().ok()));
        Some(match (name, arg) {
            ("or", _) => KernelOp::Or,
            ("and", _) => KernelOp::And,
            ("xor", _) => KernelOp::Xor,
            ("clear", _) => KernelOp::Clear,
            ("nand", _) => KernelOp::Nand,
            ("nor", _) => KernelOp::Nor,
            ("set", _) => KernelOp::Set,
            ("add", Some(k)) => KernelOp::Add(k),
            ("sub", Some(k)) => KernelOp::Sub(k),
            ("shl", Some(n)) => KernelOp::Shl(n),
            ("shr", Some(n)) => KernelOp::Shr(n),
            ("rotl", Some(n)) => KernelOp::Rotl(n),
            ("rotr", Some(n)) => KernelOp::Rotr(n),
            _ => return None,
        })
    }

    /// A random op, with a random lane width or amount where it takes one (with `set_op_choice`
    /// on, never `Rotr`: `rotr:n` is `rotl:64-n`, so its draw goes to the other ops).
    pub fn random<R: rand::Rng + ?Sized>(rng: &mut R) -> Self {
        let n: u8 = rng.gen_range(0..64);
        let k: u8 = rng.gen_range(3..7);
        if op_choice() {
            return match rng.gen_range(0..12) {
                0 => KernelOp::Or,
                1 => KernelOp::And,
                2 => KernelOp::Xor,
                3 => KernelOp::Clear,
                4 => KernelOp::Nand,
                5 => KernelOp::Nor,
                6 => KernelOp::Add(k),
                7 => KernelOp::Sub(k),
                8 => KernelOp::Shl(n),
                9 => KernelOp::Shr(n),
                10 => KernelOp::Rotl(n),
                _ => KernelOp::Set,
            };
        }
        match rng.gen_range(0..13) {
            0 => KernelOp::Or,
            1 => KernelOp::And,
            2 => KernelOp::Xor,
            3 => KernelOp::Clear,
            4 => KernelOp::Nand,
            5 => KernelOp::Nor,
            6 => KernelOp::Add(k),
            7 => KernelOp::Sub(k),
            8 => KernelOp::Shl(n),
            9 => KernelOp::Shr(n),
            10 => KernelOp::Rotl(n),
            11 => KernelOp::Set,
            _ => KernelOp::Rotr(n),
        }
    }

    /// The same op with its lane width or amount nudged (others unchanged).
    pub fn nudge(self, up: bool) -> Self {
        let a = |n: u8| if up { (n + 1) & 63 } else { n.wrapping_sub(1) & 63 };
        let l = |k: u8| if up { (k + 1).min(6) } else { k.saturating_sub(1).max(3) };
        match self {
            KernelOp::Add(k) => KernelOp::Add(l(k)),
            KernelOp::Sub(k) => KernelOp::Sub(l(k)),
            KernelOp::Shl(n) => KernelOp::Shl(a(n)),
            KernelOp::Shr(n) => KernelOp::Shr(a(n)),
            KernelOp::Rotl(n) => KernelOp::Rotl(a(n)),
            KernelOp::Rotr(n) => KernelOp::Rotr(a(n)),
            o => o,
        }
    }
}

#[cfg(test)]
mod veto_tests {
    use super::*;
    use crate::kernel::SimpleKernel;

    #[test]
    fn an_inhibit_input_holds_down_a_kernel_on_its_lines() {
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(1);
        let input = BitVector::from_bits(&[1, 2, 3, 4], 64);
        let kernel = || SimpleKernel::new(BitVector::from_bits(&[1, 2, 3, 4], 64), 0, BitVector::from_words(vec![1]), 0, 3, KernelOp::Or);
        let fires = |veto: &[u64], rng: &mut rand::rngs::StdRng| {
            let mut class = KernelClass::with_kernels(vec![kernel()]);
            class.set_veto(veto);
            let mut out = BitVector::new(64, Some(0));
            class.process(&input, &mut out, 0, 0, false, rng).len()
        };
        assert_eq!(fires(&[], &mut rng), 1);
        // another vector on three of its four lines: held down
        assert_eq!(fires(&[0b11100], &mut rng), 0);
        // on one of them: it fires
        assert_eq!(fires(&[0b10], &mut rng), 1);
        // with the overlap rule at 3 quarters, three of four lines still let it fire; at 1, two
        // of four hold it down
        let with = |veto: &[u64], q: u8, rng: &mut rand::rngs::StdRng| {
            let mut class = KernelClass::with_kernels(vec![kernel()]);
            class.set_veto(veto);
            class.set_inhibit_quarters(q);
            let mut out = BitVector::new(64, Some(0));
            class.process(&input, &mut out, 0, 0, false, rng).len()
        };
        assert_eq!(with(&[0b11100], 3, &mut rng), 1);
        assert_eq!(with(&[0b00110], 1, &mut rng), 0);
        assert_eq!(with(&[0b00110], 2, &mut rng), 1);
    }
}

#[cfg(test)]
mod op_tests {
    use super::KernelOp::*;

    #[test]
    fn every_op_is_one_word_operation() {
        let (d, m) = (0b1100u64, 0b1010u64);
        assert_eq!(Or.word(d, m), 0b1110);
        assert_eq!(And.word(d, m), 0b1000);
        assert_eq!(Xor.word(d, m), 0b0110);
        assert_eq!(Clear.word(d, m), 0b0100);
        assert_eq!(Nand.word(d, m), !0b1000);
        assert_eq!(Nor.word(d, m), !0b1110);
        // lanes wrap on their own: 0xFF + 1 in a byte lane does not carry into the next byte
        assert_eq!(Add(3).word(0x00FF, 0x0001), 0x0000);
        assert_eq!(Add(4).word(0x00FF, 0x0001), 0x0100);
        assert_eq!(Sub(3).word(0x0000, 0x0001), 0x00FF);
        assert_eq!(Sub(6).word(5, 7), 5u64.wrapping_sub(7));
        // shifts and rotations move what is written, then OR it in
        assert_eq!(Shl(8).word(0x01, 0xAB), 0xAB01);
        assert_eq!(Shr(4).word(0, 0xF0), 0x0F);
        assert_eq!(Rotl(4).word(0, 0xF000_0000_0000_0000), 0xF);
        assert_eq!(Rotr(4).word(0, 0xF), 0xF000_0000_0000_0000);
    }
}

pub struct KernelClassTemperature {
    pub max: i16,
    pub min: i16,
    pub current: i16,
}

impl KernelClassTemperature {
    pub fn default() -> Self {
        let kdef = &config().kernel;
        Self {
            min: kdef.temperature_min,
            max: kdef.temperature_max,
            current: kdef.temperature_initial,
        }
    }

    pub fn new(min: i16, max: i16, initial: i16) -> Self {
        assert!(min <= initial && initial <= max);
        Self { min, max, current: initial }
    }

    pub fn incr(&mut self) {
        if self.current < self.max {
            self.current += 1;
        }
    }

    pub fn decr(&mut self) {
        if self.current > self.min {
            self.current -= 1;
        }
    }
}

// Light, borrow-based context passed to each spawned kernel
pub struct KernelContext<'a> {
    pub input: &'a BitVector,
    pub inhibit: &'a BitVector,
    pub temperature: i16,
    pub phase: u16,
    /// a heat input's pull on every threshold this step, in 256ths of the kernel's own threshold:
    /// positive lowers it (to at least one line), negative raises it; 0 leaves it
    pub heat: i16,
    /// how much of a kernel's mask taken lines must cover to hold it down, in quarters: held down
    /// when they cover more than this many quarters (2: more than half; 4: never). 0 reads as 2
    pub inhibit_quarters: u8,
    /// (`program::neurocomp::set_write_compete`) the words of the destination that kernels of
    /// earlier (better-matched) populations wrote this step (a claimed word is all ones): a kernel
    /// more than `inhibit_quarters` quarters of whose output words are claimed is held down, as
    /// one whose input lies on taken lines is. Empty: no competition
    pub taken_out: &'a [u64],
}

pub trait KernelTrait {
    fn try_fire<R: rand::Rng + ?Sized>(&mut self, ctx: &KernelContext, out: &mut BitVector, rng: &mut R) -> bool;
    fn word_range(&self) -> (usize, usize);
    /// Mark the input lines this kernel takes when it fires (for overlap inhibition).
    fn mark_inhibit(&self, inhibit: &mut BitVector);
    /// Clear what `mark_inhibit` marked (the words it touched), so the class can reuse the
    /// vector next step without clearing all of it. Default: clear everything.
    fn clear_inhibit(&self, inhibit: &mut BitVector) {
        inhibit.as_words_mut().fill(0);
    }
    /// How many of its lines agree with the input (the better match goes first).
    fn strength(&self, input: &BitVector) -> usize;
    /// Its strength and its match count, from one pass over its mask (the count is handed back
    /// to `try_fire_counted`, so the match is computed once a step). Default: no count.
    fn rank(&self, input: &BitVector) -> (usize, Option<usize>) {
        (self.strength(input), None)
    }
    /// `try_fire` with its match count already known (from `rank`, on the same input).
    fn try_fire_counted<R: rand::Rng + ?Sized>(&mut self, ctx: &KernelContext, _count: Option<usize>, out: &mut BitVector, rng: &mut R) -> bool {
        self.try_fire(ctx, out, rng)
    }
    /// It matched this step but was held down because its output lay on lines another
    /// population had taken (`KernelContext::taken_out`); reading it clears it. Default: never.
    fn take_lost(&mut self) -> bool {
        false
    }
    /// Mark the destination lines it writes (for the competition between populations).
    fn mark_output(&self, _taken: &mut [u64]) {}
}

pub struct KernelClass<K: KernelTrait> {
    active_kernels: Vec<K>,
    temperature: KernelClassTemperature,
    /// reused each step: the lines fired kernels took (cleared word by word after), and the
    /// order kernels go in
    inhibit: BitVector,
    order: Vec<(usize, usize, Option<usize>)>,
    /// this step's heat (`KernelContext::heat`), set by the runtime from the population's heat
    /// input before it runs
    heat: i16,
    /// this step's inhibit input (`set_veto`): lines another vector had on, taken before any
    /// kernel fires, so a kernel whose mask lies more than half on them is held down
    veto: Vec<u64>,
    /// (`KernelContext::inhibit_quarters`) this population's overlap rule
    inhibit_quarters: u8,
    /// this step's taken destination lines (`KernelContext::taken_out`); empty: none
    taken: Vec<u64>,
    /// this step: kernels that matched and lost to another population's output, and the best
    /// share (in 1/4096 of its own lines) of the kernels that fired
    lost: u32,
    best: usize,
}

impl<K: KernelTrait> KernelClass<K> {
    pub fn default() -> Self {
        Self { active_kernels: Vec::new(), temperature: KernelClassTemperature::new(0, 0, 0), inhibit: BitVector::from_words(Vec::new()), order: Vec::new(), heat: 0, veto: Vec::new(), inhibit_quarters: 2, taken: Vec::new(), lost: 0, best: 0 }
    }

    pub fn with_kernels(kernels: Vec<K>) -> Self {
        Self { active_kernels: kernels, temperature: KernelClassTemperature::new(0, 0, 0), inhibit: BitVector::from_words(Vec::new()), order: Vec::new(), heat: 0, veto: Vec::new(), inhibit_quarters: 2, taken: Vec::new(), lost: 0, best: 0 }
    }

    pub fn add(&mut self, k: K) {
        self.active_kernels.push(k);
    }

    /// Run all kernels; kernels may OR/XOR/AND/CLEAR into `output`. A kernel that fires marks
    /// its input mask as taken: a later kernel whose mask overlaps the taken lines by more than
    /// half is held down (and, if it matched, learns anti-Hebbian).
    pub fn process_all<R: rand::Rng + ?Sized>(&mut self, input: &BitVector, output: &mut BitVector, phase: u16, adj_temperature: i16, rng: &mut R) -> Vec<usize> {
        self.process(input, output, phase, adj_temperature, true, rng)
    }

    /// As `process_all`; with `inhibit` off, kernels do not hold each other down.
    pub fn process<R: rand::Rng + ?Sized>(&mut self, input: &BitVector, output: &mut BitVector, phase: u16, adj_temperature: i16, inhibit_on: bool, rng: &mut R) -> Vec<usize> {
        if self.inhibit.word_len() != input.word_len() {
            self.inhibit = BitVector::new(input.bit_len(), Some(0));
        }
        let mut fired = Vec::new();
        // (an inhibit input) its lines are taken before any kernel fires
        let vetoed = !self.veto.is_empty();
        if vetoed {
            let iw = self.inhibit.as_words_mut();
            let (n, v) = (iw.len(), self.veto.len());
            if n > 0 {
                for j in 0..n.max(v) {
                    iw[j % n] |= self.veto[j % v];
                }
            }
        }
        // the better-matched kernels go first, so where masks overlap the best match fires and
        // the others are held down (ties: the older kernel)
        let mut order = std::mem::take(&mut self.order);
        order.clear();
        order.extend(self.active_kernels.iter().enumerate().map(|(i, k)| { let (s, c) = k.rank(input); (s, i, c) }));
        order.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        for &(_, i, count) in &order {
            let k = &mut self.active_kernels[i];
            let ctx = KernelContext {
                input,
                inhibit: &self.inhibit,
                temperature: self.temperature.current + adj_temperature,
                phase,
                heat: self.heat,
                inhibit_quarters: self.inhibit_quarters,
                taken_out: &self.taken,
            };
            if k.try_fire_counted(&ctx, count, output, rng) {
                if inhibit_on {
                    k.mark_inhibit(&mut self.inhibit);
                }
                fired.push(i);
            } else if k.take_lost() {
                self.lost += 1;
            }
        }
        self.best = order.iter().filter(|o| fired.contains(&o.1)).map(|o| o.0 >> 24).max().unwrap_or(0);
        self.order = order;
        if inhibit_on {
            for &i in &fired {
                self.active_kernels[i].clear_inhibit(&mut self.inhibit);
            }
        }
        if vetoed {
            self.inhibit.as_words_mut().fill(0);
        }
        fired
    }

    pub fn kernels(&self) -> &[K] { &self.active_kernels }
    pub fn kernels_mut(&mut self) -> &mut [K] { &mut self.active_kernels }

    pub fn len(&self) -> usize { self.active_kernels.len() }
    /// This step's heat (`KernelContext::heat`).
    pub fn set_heat(&mut self, heat: i16) {
        self.heat = heat;
    }

    /// The population's overlap rule (`KernelContext::inhibit_quarters`).
    pub fn set_inhibit_quarters(&mut self, q: u8) {
        self.inhibit_quarters = q;
    }

    /// This step's inhibit input: the words of the vector that holds this population down, folded
    /// onto its input in order (wrapping where the widths differ). Empty: none.
    pub fn set_veto(&mut self, words: &[u64]) {
        self.veto.clear();
        self.veto.extend_from_slice(words);
    }

    /// This step's taken destination lines (`KernelContext::taken_out`); empty: none.
    pub fn set_taken(&mut self, words: &[u64]) {
        self.taken.clear();
        self.taken.extend_from_slice(words);
    }

    /// Kernels held down last step by another population's output (cleared on reading), and the
    /// best share of its fired kernels then.
    pub fn take_lost(&mut self) -> u32 {
        std::mem::take(&mut self.lost)
    }

    pub fn best_share(&self) -> usize {
        self.best
    }

    /// Mark the destination lines the given kernels write.
    pub fn mark_outputs(&self, fired: &[usize], taken: &mut [u64]) {
        for &i in fired {
            self.active_kernels[i].mark_output(taken);
        }
    }

    pub fn clear(&mut self) { self.active_kernels.clear() }
    pub fn is_empty(&self) -> bool { self.active_kernels.is_empty() }
}