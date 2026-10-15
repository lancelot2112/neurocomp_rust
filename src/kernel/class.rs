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

    /// A random op, with a random lane width or amount where it takes one.
    pub fn random<R: rand::Rng + ?Sized>(rng: &mut R) -> Self {
        let n: u8 = rng.gen_range(0..64);
        let k: u8 = rng.gen_range(3..7);
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
}

pub struct KernelClass<K: KernelTrait> {
    active_kernels: Vec<K>,
    temperature: KernelClassTemperature,
    /// reused each step: the lines fired kernels took (cleared word by word after), and the
    /// order kernels go in
    inhibit: BitVector,
    order: Vec<(usize, usize, Option<usize>)>,
}

impl<K: KernelTrait> KernelClass<K> {
    pub fn default() -> Self {
        Self { active_kernels: Vec::new(), temperature: KernelClassTemperature::new(0, 0, 0), inhibit: BitVector::from_words(Vec::new()), order: Vec::new() }
    }

    pub fn with_kernels(kernels: Vec<K>) -> Self {
        Self { active_kernels: kernels, temperature: KernelClassTemperature::new(0, 0, 0), inhibit: BitVector::from_words(Vec::new()), order: Vec::new() }
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
            };
            if k.try_fire_counted(&ctx, count, output, rng) {
                if inhibit_on {
                    k.mark_inhibit(&mut self.inhibit);
                }
                fired.push(i);
            }
        }
        self.order = order;
        if inhibit_on {
            for &i in &fired {
                self.active_kernels[i].clear_inhibit(&mut self.inhibit);
            }
        }
        fired
    }

    pub fn kernels(&self) -> &[K] { &self.active_kernels }
    pub fn kernels_mut(&mut self) -> &mut [K] { &mut self.active_kernels }

    pub fn len(&self) -> usize { self.active_kernels.len() }
    pub fn is_empty(&self) -> bool { self.active_kernels.is_empty() }
}