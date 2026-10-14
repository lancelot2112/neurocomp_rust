//! A tag-addressed grammar: the genome is a string of tokens, and every string builds a network.
//!
//! - `Channel { tag, width }`: a bit vector of 2^width words.
//! - `Pop { from, to, .. }`: a population of kernels (or a wire) reading the channel whose tag is
//!   nearest `from` and writing the one nearest `to`, with every property inline.
//!
//! The senses and the act are channels that always exist (tags `SENSES` and `ACT`). A tag always
//! resolves to the nearest channel (Hamming distance; ties to the earlier one), so there are no
//! dangling references: inserting, deleting, moving or copying any part of the string still makes a
//! network. Wiring changes gradually (a flipped tag bit may move a population to a neighbouring
//! channel, or change nothing), and dead ends (channels nobody reads) are kept as raw material.
//!
//! Two density knobs per population:
//! - **connectivity**: `fan_in` lines per kernel, drawn from its `field` (2^field source words, from
//!   word `at`);
//! - **activity**: `tonic` 0 fires only on a match (responsive); k > 0 also fires spontaneously with
//!   chance 2^-k a step (tonic: babbling is a tonic population on the act).

use petgraph::graph::Graph;
use petgraph::Directed;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::bitvec::BitVector;
use crate::kernel::{KernelClass, KernelGroup, KernelOp, SimpleKernel};
use crate::program::graph::{EdgeKind, EdgeSpec, NodeSpec};
use crate::program::neurocomp::{EdgeInfo, RuntimeNetwork};
use crate::world::{self, World};

/// The widest channel: 2^11 words (131,072 bits). Fields and spreads go as wide.
pub const MAX_WIDTH: u8 = 11;
/// The widest tile: 2^17 bits (a whole widest channel).
pub const MAX_TILE: u8 = 17;

/// How far a mutation moves an offset: one word, or a whole field (so a wide population can step
/// to its neighbour's lines in one move).
fn hop(scale: u8, rng: &mut StdRng) -> u16 {
    if rng.gen_bool(0.5) { 1 } else { 1u16 << scale.min(MAX_WIDTH) }
}

use super::Evolvable;

/// The senses' tag.
pub const SENSES: u8 = 0x00;
/// The act's tag.
pub const ACT: u8 = 0xFF;

/// A population: kernels (or a wire) from one channel to another, with all its properties.
#[derive(Clone, Debug, PartialEq)]
pub struct Pop {
    /// a permanent id: keys its random stream (a copy gets a new one)
    pub id: u32,
    pub from: u8,
    pub to: u8,
    /// a wire copies what it reads (no kernels)
    pub wire: bool,
    pub op: KernelOp,
    /// frames back it reads, and how many frames it ORs together
    pub lag: u8,
    pub span: u8,
    /// its field: 2^field source words from word `at` (wrapping)
    pub at: u16,
    pub field: u8,
    /// where it writes: from destination word `place` (wrapping), over 2^spread words
    pub place: u16,
    pub spread: u8,
    /// starting kernels; lines each reads (fan-in) and writes (fan-out); threshold in quarters
    pub count: u8,
    pub fan_in: u8,
    pub fan_out: u8,
    pub quarters: u8,
    /// activity: 0 responsive; k > 0 tonic, firing spontaneously with chance 2^-k
    pub tonic: u8,
    /// its kernels do not hold each other down
    pub share: bool,
    /// births where its destination is left uncovered; with a fresh code into an empty one
    pub grow: bool,
    pub new_code: bool,
    /// births repeating the act `repeat_lag` frames back, read on what came before it
    pub repeat: bool,
    pub repeat_lag: u8,
    /// learns only while the channel nearest this tag had lines on (true) or none, `gate_lag` back
    pub gate: Option<(u8, bool)>,
    pub gate_lag: u8,
    /// births: synapses, threshold quarters, fewest and most output lines, cap; fresh code size
    pub birth_syn: u8,
    pub birth_quarters: u8,
    pub out_min: u8,
    pub out_max: u8,
    pub cap: u16,
    pub code_lines: u8,
    /// output learning chance 2^-out_learn; extinction chance 2^-fade
    pub out_learn: u8,
    pub fade: u8,
    /// a wire's rotation, in bits
    pub rotate: u16,
    /// sparse kernels: each visits only the words its lines are in
    pub sparse: bool,
    /// full coverage: it reads its whole source (no `at`/`field`) in tiles of 2^tile bits
    /// every half tile, its kernels are born where input is not yet read, and each writes at the
    /// same relative place in its destination (no `place`/`spread`). A wire copies the whole
    /// source onto the destination.
    pub cover: bool,
    pub tile: u8,
}

impl Pop {
    /// A population with the grammar's starting values.
    pub fn new(id: u32, from: u8, to: u8) -> Self {
        let p = EdgeSpec::plain();
        Self {
            id, from, to, wire: false, op: KernelOp::Or, lag: 1, span: 1, at: 0, field: 0, place: 0, spread: 0,
            count: 0, fan_in: p.start_syn, fan_out: p.start_out, quarters: p.start_quarters, tonic: 0, share: false,
            grow: false, new_code: false, repeat: false, repeat_lag: 1, gate: None, gate_lag: 1, birth_syn: p.birth_syn,
            birth_quarters: p.birth_quarters, out_min: p.out_min, out_max: p.out_max, cap: p.cap, code_lines: p.code_lines,
            out_learn: p.out_learn, fade: p.fade, rotate: 0, sparse: false, cover: false, tile: 6,
        }
    }

    pub fn wire(id: u32, from: u8, to: u8) -> Self {
        Self { wire: true, ..Self::new(id, from, to) }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    /// a channel: 2^width words. `hold`: it starts each step with what it held the step before
    /// (only writes that clear take lines away); otherwise it starts each step empty.
    Channel { tag: u8, width: u8, hold: bool },
    Pop(Pop),
}

#[derive(Clone, Debug, PartialEq)]
pub struct TagGenome {
    pub tokens: Vec<Token>,
    pub next_id: u32,
}


/// What the string's channels resolve to.
struct Channels {
    /// (tag, words), the senses first and the act last
    list: Vec<(u8, usize)>,
    /// whether each holds its lines from step to step
    hold: Vec<bool>,
}

impl Channels {
    fn of(tokens: &[Token]) -> Self {
        let mut list = vec![(SENSES, world::SENSE_BITS / 64)];
        let mut hold = vec![false];
        for t in tokens {
            if let Token::Channel { tag, width, hold: h } = t {
                list.push((*tag, 1usize << (*width).min(MAX_WIDTH)));
                hold.push(*h);
            }
        }
        list.push((ACT, world::ACT_BITS / 64));
        hold.push(false);
        Self { list, hold }
    }

    /// The channel nearest a tag (ties: the earlier); `writable` skips the senses.
    fn resolve(&self, tag: u8, writable: bool) -> usize {
        (0..self.list.len())
            .filter(|&i| !(writable && i == 0))
            .min_by_key(|&i| ((self.list[i].0 ^ tag).count_ones(), i))
            .unwrap()
    }
}

fn sparse(bits: usize, k: usize, rng: &mut StdRng) -> BitVector {
    let all: Vec<usize> = (0..bits).collect();
    BitVector::from_bits(&all.choose_multiple(rng, k.min(bits)).copied().collect::<Vec<_>>(), bits)
}

impl TagGenome {
    fn pops(&self) -> impl Iterator<Item = &Pop> {
        self.tokens.iter().filter_map(|t| if let Token::Pop(p) = t { Some(p) } else { None })
    }

    /// The genome as text, one token a line: each channel with its width, and each population
    /// with the channels its tags resolve to (S the senses, A the act, Cn the n-th channel) and
    /// its properties.
    pub fn describe(&self) -> String {
        let ch = Channels::of(&self.tokens);
        let last = ch.list.len() - 1;
        let name = |i: usize| if i == 0 { "S".to_string() } else if i == last { "A".to_string() } else { format!("C{i}") };
        let mut out = String::new();
        let mut c = 0;
        for t in &self.tokens {
            match t {
                Token::Channel { tag, width, hold } => {
                    c += 1;
                    out += &format!("C{c} channel {tag:08b}  {} bits{}\n", 64usize << (*width).min(MAX_WIDTH), if *hold { "  holds" } else { "" });
                }
                Token::Pop(p) => {
                    let (a, b) = (ch.resolve(p.from, false), ch.resolve(p.to, true));
                    out += &format!(
                        "   {} -> {} {} {:?} lag {}{} {}{}{}{}{}{}{}{}\n",
                        name(a), name(b), if p.wire { "wire" } else { "kernels" }, p.op, p.lag,
                        if p.span > 1 { format!(" span {}", p.span) } else { String::new() },
                        if p.cover {
                            if p.wire { "whole".to_string() } else { format!("cover, tiles of {} bits", 1usize << p.tile.clamp(3, MAX_TILE)) }
                        } else {
                            format!("field {}w@{} -> {}w@{}", 1usize << p.field.min(MAX_WIDTH), p.at, 1usize << p.spread.min(MAX_WIDTH), p.place)
                        },
                        if p.wire { String::new() } else { format!(" n{} in{} out{} q{}/4", p.count, p.fan_in, p.fan_out, p.quarters) },
                        if p.tonic > 0 { format!(" tonic 1/{}", 1u32 << p.tonic) } else { String::new() },
                        if p.share { " share" } else { "" },
                        if p.grow { format!(" grow(syn {}, out {}-{}{}{})", p.birth_syn, p.out_min, p.out_max, if p.new_code { ", new" } else { "" }, if p.repeat { format!(", repeat {}", p.repeat_lag) } else { String::new() }) } else { String::new() },
                        p.gate.map_or(String::new(), |(g, on)| format!(" gate {}{}", if on { "" } else { "!" }, name(ch.resolve(g, false)))),
                        if p.sparse { " sparse" } else { "" },
                        if p.rotate > 0 { format!(" rot {}", p.rotate) } else { String::new() },
                    );
                }
            }
        }
        out
    }

    /// The runtime graph: one node per channel (senses first, act last), one edge per population.
    pub fn graph(&self) -> Graph<NodeSpec, EdgeSpec, Directed> {
        let ch = Channels::of(&self.tokens);
        let n = ch.list.len();
        // history depth: the deepest frame read from each channel
        let mut depth = vec![1usize; n];
        let mut edges: Vec<(usize, usize, EdgeSpec)> = Vec::new();
        for p in self.pops() {
            let (a, b) = (ch.resolve(p.from, false), ch.resolve(p.to, true));
            let lag = (p.lag as usize).max(1);
            let span = (p.span as usize).max(1);
            depth[a] = depth[a].max(lag + span - 1);
            if p.repeat {
                depth[a] = depth[a].max(lag + p.repeat_lag as usize + 1);
                depth[b] = depth[b].max(p.repeat_lag as usize);
            }
            let gate = p.gate.map(|(t, on)| (ch.resolve(t, false), on));
            if let Some((g, _)) = gate {
                depth[g] = depth[g].max((p.gate_lag as usize).max(1));
            }
            let field = 1usize << p.field.min(MAX_WIDTH);
            edges.push((a, b, EdgeSpec {
                input_read_back: lag,
                op: p.op,
                kind: if p.wire { EdgeKind::Wire } else { EdgeKind::Kernels },
                rotate: p.rotate,
                grow: p.grow,
                inhibit: !p.share,
                gate,
                new_code: p.new_code,
                // births sample across the whole field when it is wider than a word
                wide: field > 1 && !p.cover,
                span,
                window: if p.cover { None } else { Some((p.at as usize, field)) },
                dst_word: if p.cover { 0 } else { p.place as usize },
                repeat: p.repeat,
                repeat_lag: (p.repeat_lag as usize).max(1),
                start_kernels: p.count,
                start_syn: p.fan_in,
                start_quarters: p.quarters,
                start_out: p.fan_out,
                birth_syn: p.birth_syn,
                birth_quarters: p.birth_quarters,
                out_min: p.out_min,
                out_max: p.out_max,
                cap: p.cap,
                code_lines: p.code_lines,
                out_learn: p.out_learn,
                fade: p.fade,
                gate_lag: (p.gate_lag as usize).max(1),
                salt: p.id,
                out_words: if p.cover { 0 } else { 1 << p.spread.min(MAX_WIDTH) },
                tile: if p.cover && !p.wire { p.tile.clamp(3, MAX_TILE) } else { 0 },
                tonic: p.tonic,
                sparse: p.sparse,
                ..EdgeSpec::plain()
            }));
        }
        let mut g: Graph<NodeSpec, EdgeSpec, Directed> = Graph::new();
        let idx: Vec<_> = (0..n)
            .map(|i| {
                let name = if i == 0 { Some("input".to_string()) } else if i == n - 1 { Some("output".to_string()) } else { None };
                g.add_node(NodeSpec { name, bits: ch.list[i].1 * 64, history_depth: depth[i], noise: 0, hold: ch.hold[i] })
            })
            .collect();
        for (a, b, spec) in edges {
            g.add_edge(idx[a], idx[b], spec);
        }
        g
    }

    /// Split: the population at token `at` writes into a new channel one tag bit (`bit`) from
    /// its destination, as wide, and a wire carries that channel on to the old destination one
    /// frame later (a stage inserted on the path).
    pub fn split(&mut self, at: usize, bit: u8) {
        let Some(Token::Pop(p)) = self.tokens.get(at) else { return };
        let ch = Channels::of(&self.tokens);
        let (old, words) = ch.list[ch.resolve(p.to, true)];
        let width = words.trailing_zeros() as u8;
        let new = old ^ (1u8 << (bit & 7));
        let carry = Pop { field: width, spread: width, ..Pop::wire(self.next_id, new, old) };
        self.next_id += 1;
        if let Token::Pop(p) = &mut self.tokens[at] {
            p.to = new;
        }
        self.tokens.insert(at, Token::Channel { tag: new, width, hold: false });
        self.tokens.insert(at + 2, Token::Pop(carry));
    }

    /// Self-loop: the population at token `at` writes back into the channel it reads (a carry,
    /// a clock, a counter, held state). Not on the senses, which nothing writes.
    pub fn self_loop(&mut self, at: usize) {
        let ch = Channels::of(&self.tokens);
        if let Some(Token::Pop(p)) = self.tokens.get_mut(at) {
            let src = ch.resolve(p.from, false);
            if src != 0 {
                p.to = ch.list[src].0;
            }
        }
    }

    fn random_pop(&mut self, rng: &mut StdRng) -> Pop {
        let id = self.next_id;
        self.next_id += 1;
        // a new population is born active: some kernels, a random field and fan-in, sometimes
        // tonic, sometimes growing (an inert gene would need several more mutations to matter)
        let mut p = Pop::new(id, rng.r#gen(), rng.r#gen());
        p.wire = rng.gen_bool(0.25);
        p.sparse = rng.gen_bool(0.5);
        // full coverage, at a random scale from a byte to two words
        p.cover = true;
        p.tile = rng.gen_range(3..=7);
        // any of the ops, evenly (OR is one of twelve)
        p.op = KernelOp::random(rng);
        p.count = rng.gen_range(1..=8);
        p.field = rng.gen_range(0..4);
        p.at = rng.r#gen();
        p.fan_in = 1 << rng.gen_range(2..5);
        p.tonic = if rng.gen_range(0..3) == 0 { rng.gen_range(1..5) } else { 0 };
        p.grow = rng.gen_bool(0.3);
        p.repeat = p.grow && rng.gen_bool(0.3);
        p.repeat_lag = rng.gen_range(1..6);
        if rng.gen_range(0..3) == 0 {
            p.gate = Some((rng.r#gen(), rng.gen_bool(0.5)));
        }
        p
    }

    /// The act path (experiment 06 starting point): a tonic population babbling into the act, and
    /// a growing population repeating acts a stroke followed, on what the eye saw before them.
    pub fn act_seed() -> Self {
        let ch = (world::CH_BITS / 64) as u16;
        let (eye, aff) = (0x0F, 0xF0);
        let mut t = vec![Token::Channel { tag: eye, width: 3, hold: false }, Token::Channel { tag: aff, width: 3, hold: false }];
        let mut w = Pop::wire(1, SENSES, eye);
        w.at = world::EYE as u16 * ch;
        w.field = 3;
        t.push(Token::Pop(w));
        let mut w = Pop::wire(2, SENSES, aff);
        w.at = world::AFFECT as u16 * ch;
        w.field = 3;
        t.push(Token::Pop(w));
        let mut b = Pop::new(3, eye, ACT);
        b.count = 8;
        b.tonic = 3;
        t.push(Token::Pop(b));
        let mut r = Pop::new(4, eye, ACT);
        r.field = 3;
        r.grow = true;
        r.repeat = true;
        r.repeat_lag = 4;
        r.gate = Some((aff, true));
        t.push(Token::Pop(r));
        Self { tokens: t, next_id: 5 }
    }

    /// The context circuit as tokens: the start that learned with the edge genome (experiment 06).
    /// Channels (tags far apart, so they resolve exactly):
    /// - Y: eye and affect of the last three steps, side by side;
    /// - Z: conjunction codes on Y (fresh codes born where nothing matches);
    /// - R: Z less what the context C expects (expectations learned while young);
    /// - C: the context, held by a self-wire, cleared by surprises when settled, restored by
    ///   what a context expected, set to a fresh code when empty;
    /// - N, YG: C's onset, held 32 frames ("young");
    /// - AFF: the affect channel; A: the eye beside C.
    ///
    /// Acts: a tonic population babbling into the act, and acts a stroke followed repeated on what
    /// A held before them (fading when no stroke follows).
    pub fn context_seed() -> Self {
        let ch = (world::CH_BITS / 64) as u16;
        let (y, z, r, c, n, yg, aff, a) = (0x33u8, 0x55, 0x66, 0x99, 0xAA, 0xCC, 0xF0, 0x0F);
        let mut t = vec![
            Token::Channel { tag: y, width: 6, hold: false },
            Token::Channel { tag: z, width: 3, hold: false },
            Token::Channel { tag: r, width: 3, hold: false },
            Token::Channel { tag: c, width: 3, hold: false },
            Token::Channel { tag: n, width: 3, hold: false },
            Token::Channel { tag: yg, width: 3, hold: false },
            Token::Channel { tag: aff, width: 3, hold: false },
            Token::Channel { tag: a, width: 4, hold: false },
        ];
        let mut id = 1u32;
        let mut pop = |p: Pop, t: &mut Vec<Token>| { t.push(Token::Pop(Pop { id, ..p })); id += 1; };
        for lag in 1..=3u8 {
            for (k, chan) in [world::EYE, world::AFFECT].into_iter().enumerate() {
                pop(Pop { lag, at: chan as u16 * ch, field: 3, place: ((lag - 1) as u16 * 2 + k as u16) * ch, ..Pop::wire(0, SENSES, y) }, &mut t);
            }
        }
        pop(Pop { field: 6, spread: 3, grow: true, new_code: true, ..Pop::new(0, y, z) }, &mut t);
        pop(Pop { field: 3, ..Pop::wire(0, z, r) }, &mut t);
        pop(Pop { field: 3, op: KernelOp::Clear, grow: true, share: true, gate: Some((yg, true)), ..Pop::new(0, c, r) }, &mut t);
        pop(Pop { field: 3, spread: 3, op: KernelOp::Clear, grow: true, gate: Some((yg, false)), ..Pop::new(0, r, c) }, &mut t);
        pop(Pop { field: 3, grow: true, gate: Some((yg, true)), ..Pop::new(0, r, c) }, &mut t);
        pop(Pop { field: 3, grow: true, new_code: true, ..Pop::new(0, r, c) }, &mut t);
        pop(Pop { field: 3, ..Pop::wire(0, c, c) }, &mut t);
        pop(Pop { field: 3, ..Pop::wire(0, c, n) }, &mut t);
        pop(Pop { field: 3, lag: 2, op: KernelOp::Clear, ..Pop::wire(0, c, n) }, &mut t);
        pop(Pop { field: 3, span: 32, ..Pop::wire(0, n, yg) }, &mut t);
        pop(Pop { at: world::AFFECT as u16 * ch, field: 3, ..Pop::wire(0, SENSES, aff) }, &mut t);
        pop(Pop { at: world::EYE as u16 * ch, field: 3, ..Pop::wire(0, SENSES, a) }, &mut t);
        pop(Pop { field: 3, place: ch, ..Pop::wire(0, c, a) }, &mut t);
        pop(Pop { count: 8, tonic: 3, ..Pop::new(0, a, ACT) }, &mut t);
        pop(Pop { field: 4, grow: true, repeat: true, repeat_lag: 4, gate: Some((aff, true)), ..Pop::new(0, a, ACT) }, &mut t);
        Self { tokens: t, next_id: id }
    }

    /// A bare start: one tonic population on the act, nothing else.
    pub fn bare_seed() -> Self {
        let mut b = Pop::new(1, SENSES, ACT);
        b.count = 8;
        b.tonic = 3;
        Self { tokens: vec![Token::Pop(b)], next_id: 2 }
    }
}

/// A population's starting kernels, from its own properties and its own random stream: each reads
/// `fan_in` lines across the field (the window) and writes `fan_out` lines in one word of its
/// spread (into the act: one of the body's motor primitives); tonic ones also fire spontaneously.
pub fn start_kernels(e: &EdgeInfo, primitives: &[BitVector]) -> KernelGroup {
    let sp = e.spec;
    // drawn fresh each lifetime: the genome sets the kind of cell, not the cell
    let mut r = StdRng::seed_from_u64((sp.salt as u64 + 1).wrapping_mul(0x2545_F491_4F6C_DD1D) ^ e.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let src_words = (e.src_bits / 64).max(1);
    let field = sp.window.map_or(src_words, |(o, n)| { let o = o % src_words; n.max(1).min(src_words - o) });
    let dst_words = (e.dst_bits / 64).max(1);
    let from = sp.dst_word % dst_words;
    let span = (sp.out_words as usize).max(1).min(dst_words - from);
    let motor = |r: &mut StdRng| {
        let p = primitives.choose(r).unwrap();
        let w = p.as_words().iter().position(|&x| x != 0).unwrap_or(0);
        (BitVector::from_words(vec![p.as_words()[w]]), w, KernelOp::Or)
    };
    let kernels = (0..sp.start_kernels)
        .map(|_| {
            let mut k = if sp.tile > 0 {
                // full coverage: a random tile, lines inside it, output at the same relative place
                let bits = src_words * 64;
                let t = (1usize << sp.tile.min(20)).min(bits);
                let stride = (t / 2).max(1);
                let tiles = if t >= bits { 1 } else { (bits - t) / stride + 1 };
                let start = r.gen_range(0..tiles) * stride;
                let all: Vec<usize> = (start..start + t).collect();
                let syn: Vec<usize> = all.choose_multiple(&mut r, (sp.start_syn as usize).min(t)).copied().collect();
                let (out, out_word, op) = if e.to_output && !primitives.is_empty() {
                    motor(&mut r)
                } else {
                    (sparse(64, sp.start_out as usize, &mut r), (start * dst_words / src_words).min(dst_words - 1), sp.op)
                };
                let threshold = (sp.start_quarters as usize * syn.len()).div_ceil(4).max(1);
                SimpleKernel::tile(start, t, &syn, out, out_word, threshold, op)
            } else {
                let syn = sparse(field * 64, sp.start_syn as usize, &mut r);
                let (out, out_word, op) = if e.to_output && !primitives.is_empty() {
                    motor(&mut r)
                } else {
                    (sparse(64, sp.start_out as usize, &mut r), from + r.gen_range(0..span), sp.op)
                };
                let threshold = (sp.start_quarters as usize * syn.count_ones()).div_ceil(4).max(1);
                let mut k = SimpleKernel::new(syn, 0, out, out_word, threshold, op);
                if sp.sparse {
                    k.sparsify();
                }
                k
            };
            k.tonic = sp.tonic;
            k
        })
        .collect();
    KernelGroup::from_classes(vec![KernelClass::with_kernels(kernels)])
}

impl Evolvable for TagGenome {
    fn build(&self, w: &World, seed: u64) -> RuntimeNetwork {
        let g = self.graph();
        let body = w.body();
        let primitives: Vec<BitVector> = (0..body.voice.len())
            .map(|i| w.act(Some(i), None))
            .chain((0..body.hand.len()).map(|i| w.act(None, Some(i))))
            .collect();
        RuntimeNetwork::from_graph_with(&g, world::SENSE_BITS, world::ACT_BITS, seed, |e: &EdgeInfo| start_kernels(e, &primitives))
    }

    fn mutate(&self, rng: &mut StdRng) -> Self {
        let mut g = self.clone();
        let n = g.tokens.len();
        match rng.gen_range(0..14) {
            // a new token
            0 => {
                let t = if rng.gen_bool(0.3) { Token::Channel { tag: rng.r#gen(), width: rng.gen_range(0..=MAX_WIDTH), hold: rng.gen_bool(0.25) } } else { Token::Pop(g.random_pop(rng)) };
                let at = rng.gen_range(0..=n);
                g.tokens.insert(at, t);
            }
            // a token removed
            1 if n > 0 => {
                let at = rng.gen_range(0..n);
                g.tokens.remove(at);
            }
            // a segment copied elsewhere; sometimes privately (channels declared inside get fresh
            // tags, and the copy's references to them follow, so its inner wiring is its own)
            2 | 3 if n > 0 => {
                let a = rng.gen_range(0..n);
                let len = rng.gen_range(1..=4.min(n - a));
                let mut seg: Vec<Token> = g.tokens[a..a + len].to_vec();
                if rng.gen_bool(0.5) {
                    let mut map: Vec<(u8, u8)> = Vec::new();
                    for t in &mut seg {
                        if let Token::Channel { tag, .. } = t {
                            let new: u8 = rng.r#gen();
                            map.push((*tag, new));
                            *tag = new;
                        }
                    }
                    for t in &mut seg {
                        if let Token::Pop(p) = t {
                            for (old, new) in &map {
                                if p.from == *old { p.from = *new }
                                if p.to == *old { p.to = *new }
                                if let Some((gt, on)) = p.gate { if gt == *old { p.gate = Some((*new, on)) } }
                            }
                        }
                    }
                }
                for t in &mut seg {
                    if let Token::Pop(p) = t {
                        p.id = g.next_id;
                        g.next_id += 1;
                    }
                }
                let at = rng.gen_range(0..=n);
                for (k, t) in seg.into_iter().enumerate() {
                    g.tokens.insert(at + k, t);
                }
            }
            // a segment moved
            4 if n > 1 => {
                let a = rng.gen_range(0..n);
                let len = rng.gen_range(1..=4.min(n - a));
                let seg: Vec<Token> = g.tokens.drain(a..a + len).collect();
                let at = rng.gen_range(0..=g.tokens.len());
                for (k, t) in seg.into_iter().enumerate() {
                    g.tokens.insert(at + k, t);
                }
            }
            // a stage split into a population's path, or a population turned back on its source
            12 | 13 if g.pops().count() > 0 => {
                let pops: Vec<usize> = (0..n).filter(|&i| matches!(g.tokens[i], Token::Pop(_))).collect();
                let at = *pops.choose(rng).unwrap();
                if rng.gen_bool(0.5) { g.split(at, rng.gen_range(0..8)) } else { g.self_loop(at) }
            }
            // one property of one token
            _ if n > 0 => {
                let at = rng.gen_range(0..n);
                let up = rng.gen_bool(0.5);
                let step = |v: u8, lo: u8, hi: u8| if up { v.saturating_add(1).min(hi) } else { v.saturating_sub(1).max(lo) };
                let bit = 1u8 << rng.gen_range(0..8);
                match &mut g.tokens[at] {
                    Token::Channel { tag, width, hold } => match rng.gen_range(0..3) {
                        0 => *tag ^= bit,
                        1 => *width = step(*width, 0, MAX_WIDTH),
                        _ => *hold = !*hold,
                    },
                    Token::Pop(p) => match rng.gen_range(0..33) {
                        0 => p.from ^= bit,
                        1 => p.to ^= bit,
                        2 => p.wire = !p.wire,
                        3 => p.op = if rng.gen_bool(0.5) { KernelOp::random(rng) } else { p.op.nudge(up) },
                        4 => p.lag = step(p.lag, 1, 8),
                        5 => p.span = if up { (p.span * 2).min(64) } else { (p.span / 2).max(1) },
                        6 => { let d = hop(p.field, rng); p.at = if up { p.at.wrapping_add(d) } else { p.at.wrapping_sub(d) } }
                        7 => p.field = step(p.field, 0, MAX_WIDTH),
                        8 => { let d = hop(p.spread, rng); p.place = if up { p.place.wrapping_add(d) } else { p.place.wrapping_sub(d) } }
                        9 => p.spread = step(p.spread, 0, MAX_WIDTH),
                        10 => p.count = step(p.count, 0, 32),
                        11 => p.fan_in = step(p.fan_in, 1, 64),
                        12 => p.fan_out = step(p.fan_out, 1, 64),
                        13 => p.quarters = step(p.quarters, 1, 4),
                        14 => p.tonic = step(p.tonic, 0, 12),
                        15 => p.share = !p.share,
                        16 => p.grow = !p.grow,
                        17 => p.new_code = !p.new_code,
                        18 => p.repeat = !p.repeat,
                        19 => p.repeat_lag = step(p.repeat_lag, 1, 8),
                        20 => p.gate = match p.gate { Some((t, on)) => if rng.gen_bool(0.3) { None } else if rng.gen_bool(0.5) { Some((t ^ bit, on)) } else { Some((t, !on)) }, None => Some((rng.r#gen(), rng.gen_bool(0.5))) },
                        21 => p.gate_lag = step(p.gate_lag, 1, 8),
                        22 => p.birth_syn = step(p.birth_syn, 1, 64),
                        23 => p.birth_quarters = step(p.birth_quarters, 1, 4),
                        24 => p.out_min = step(p.out_min, 1, 64),
                        25 => p.out_max = step(p.out_max, 1, 64),
                        26 => p.code_lines = step(p.code_lines, 1, 64),
                        27 => p.out_learn = step(p.out_learn, 0, 8),
                        28 => p.fade = step(p.fade, 0, 8),
                        29 => p.rotate = if up { p.rotate.wrapping_add(61) } else { p.rotate.wrapping_sub(61) },
                        30 => p.sparse = !p.sparse,
                        31 => p.cover = !p.cover,
                        _ => p.tile = step(p.tile, 3, MAX_TILE),
                    },
                }
            }
            _ => g.tokens.push(Token::Pop(g.clone().random_pop(rng))),
        }
        g
    }

    fn size(&self) -> usize {
        self.tokens.len()
    }

    /// One point: this string's front, the other's back (tags carry the wiring across).
    fn crossover(&self, other: &Self, rng: &mut StdRng) -> Self {
        let a = rng.gen_range(0..=self.tokens.len());
        let b = rng.gen_range(0..=other.tokens.len());
        TagGenome { tokens: self.tokens[..a].iter().chain(other.tokens[b..].iter()).cloned().collect(), next_id: self.next_id.max(other.next_id) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolve::{live, Physics, Task};
    use crate::world::Conditioning;

    #[test]
    fn a_tag_resolves_to_the_nearest_channel() {
        let ch = Channels::of(&[Token::Channel { tag: 0b1010_0000, width: 0, hold: false }, Token::Channel { tag: 0b0000_0101, width: 1, hold: false }]);
        assert_eq!(ch.resolve(0b1010_0001, false), 1);
        assert_eq!(ch.resolve(0b0000_0111, false), 2);
        assert_eq!(ch.resolve(SENSES, false), 0);
        // nothing writes the senses: the nearest writable channel instead
        assert_ne!(ch.resolve(SENSES, true), 0);
        assert_eq!(ch.resolve(ACT, true), 3);
    }

    #[test]
    fn any_string_builds_a_network_that_lives() {
        let mut rng = StdRng::seed_from_u64(9);
        let mut g = TagGenome::act_seed();
        for i in 0..400u64 {
            g = g.mutate(&mut rng);
            let _ = g.graph();
            if i % 20 == 0 {
                let _ = live(&g, Task::Conditioning(Conditioning::Reversal, 6), Physics::Varied, i);
            }
        }
        // and an empty string is a network too
        let _ = live(&TagGenome { tokens: vec![], next_id: 0 }, Task::Conditioning(Conditioning::Reversal, 4), Physics::Varied, 1);
    }

    /// A sliding window from the grammar alone: a channel carrying its previous frame shifted
    /// left a byte, and the newest byte written in. No window is built into the runtime.
    #[test]
    fn a_sliding_window_is_two_populations() {
        use crate::bitvec::bithistory::AdvanceMode;
        let w = 0x5A;
        let mut into = Pop::wire(1, SENSES, w);
        into.field = 0;
        let mut carry = Pop::wire(2, w, w);
        carry.op = KernelOp::Shl(8);
        let g = TagGenome { tokens: vec![Token::Channel { tag: w, width: 0, hold: false }, Token::Pop(into), Token::Pop(carry)], next_id: 3 };
        let world = World::new(1, 4);
        let mut net = g.build(&world, 1);
        let mut last = 0u64;
        for b in [0x11u64, 0x22, 0x33, 0x44] {
            net.tick(AdvanceMode::Clear, 0);
            net.write_input(&BitVector::from_words(vec![b]), 0);
            last = net.read_node(1).as_words()[0];
        }
        // the window holds the bytes before the newest, newest lowest (it reads the senses a frame back)
        assert_eq!(last, 0x11_22_33);
    }

    #[test]
    fn the_context_seed_resolves_every_tag_exactly() {
        let g = TagGenome::context_seed();
        let ch = Channels::of(&g.tokens);
        for p in g.pops() {
            let (a, b) = (ch.resolve(p.from, false), ch.resolve(p.to, true));
            assert_eq!(ch.list[a].0, p.from);
            assert_eq!(ch.list[b].0, p.to);
        }
        let _ = live(&g, Task::Conditioning(Conditioning::Reversal, 8), Physics::Varied, 1);
    }

    #[test]
    fn split_inserts_a_stage_and_self_loop_closes_one() {
        let (a, b) = (0x0Fu8, 0xF0u8);
        let p = Pop { count: 2, ..Pop::new(1, a, b) };
        let mut g = TagGenome { tokens: vec![Token::Channel { tag: a, width: 2, hold: false }, Token::Channel { tag: b, width: 5, hold: false }, Token::Pop(p)], next_id: 2 };
        g.split(2, 0);
        // a new channel 0xF1 as wide as B; the population writes it; a wire carries it on to B
        assert_eq!(g.tokens[2], Token::Channel { tag: 0xF1, width: 5, hold: false });
        let Token::Pop(p) = &g.tokens[3] else { panic!() };
        assert_eq!((p.from, p.to), (a, 0xF1));
        let Token::Pop(w) = &g.tokens[4] else { panic!() };
        assert!(w.wire && w.from == 0xF1 && w.to == b && w.field == 5);
        assert_eq!(g.graph().edge_count(), 2);
        g.self_loop(3);
        let Token::Pop(p) = &g.tokens[3] else { panic!() };
        assert_eq!(p.to, a);
        // a population reading the senses is left alone
        let mut s = TagGenome { tokens: vec![Token::Pop(Pop::new(1, SENSES, a))], next_id: 2 };
        s.self_loop(0);
        assert_eq!(s.tokens[0], Token::Pop(Pop::new(1, SENSES, a)));
    }

    #[test]
    fn a_held_channel_keeps_its_lines_until_something_clears_them() {
        // H copies the senses' word 0 (OR); a CLEAR wire takes off whatever is on in the senses'
        // word 1. Shown a code once, a held H keeps it with no carry wire; an all-on word 1
        // clears it. A plain H loses it the next step.
        let run = |hold: bool| -> Vec<u32> {
            let h = 0x3C;
            let set = Pop { field: 0, ..Pop::wire(1, SENSES, h) };
            let clear = Pop { at: 1, field: 0, op: KernelOp::Clear, ..Pop::wire(2, SENSES, h) };
            let g = TagGenome { tokens: vec![Token::Channel { tag: h, width: 0, hold }, Token::Pop(set), Token::Pop(clear)], next_id: 3 };
            let mut net = g.build(&World::new(1, 4), 1);
            let mut seen = Vec::new();
            for t in 0..6 {
                let mut words = vec![0u64; world::SENSE_BITS / 64];
                if t == 0 { words[0] = 0xF0F0 }
                if t == 4 { words[1] = u64::MAX }
                net.tick(crate::bitvec::bithistory::AdvanceMode::Clear, 0);
                net.write_input(&BitVector::from_words(words), 0);
                seen.push(net.read_node(1).count_ones() as u32);
            }
            seen
        };
        // wires read one frame back, so H sees step t's input at step t+1
        assert_eq!(run(true), vec![0, 8, 8, 8, 8, 0]);
        assert_eq!(run(false), vec![0, 8, 0, 0, 0, 0]);
    }

    #[test]
    fn a_covering_population_finds_input_wherever_it_lands() {
        // one growing population over the whole senses, tiles of 64 bits: shown a code in a
        // different word each lifetime, it is born on that code and fires on it again
        for word in [0usize, 5, 17, 30] {
            let c = 0x3C;
            let p = Pop { cover: true, tile: 6, grow: true, new_code: true, ..Pop::new(1, SENSES, c) };
            let g = TagGenome { tokens: vec![Token::Channel { tag: c, width: 3, hold: false }, Token::Pop(p)], next_id: 2 };
            let mut net = g.build(&World::new(1, 4), word as u64);
            let mut on = vec![0u64; world::SENSE_BITS / 64];
            on[word] = 0x0F0F_0000_0000_F0F0;
            let mut seen = Vec::new();
            for t in 0..8 {
                let x = if t % 2 == 0 { on.clone() } else { vec![0u64; world::SENSE_BITS / 64] };
                net.tick(crate::bitvec::bithistory::AdvanceMode::Clear, 0);
                net.write_input(&BitVector::from_words(x), 0);
                seen.push(net.read_node(1).count_ones());
            }
            // the code shown at even steps is read at odd ones (lag 1): born at step 1, then the
            // kernel fires at each odd step
            assert!(seen[3] > 0 && seen[5] > 0 && seen[7] > 0, "word {word}: {seen:?}");
            assert_eq!((seen[2], seen[4], seen[6]), (0, 0, 0), "word {word}: {seen:?}");
            let n = net.edges[0].group.kernel_count();
            assert!(n >= 1 && n <= 3, "word {word}: {n} kernels");
        }
    }

    #[test]
    fn a_tonic_population_babbles_and_a_responsive_one_does_not() {
        let t = Task::Conditioning(Conditioning::Instrumental, 30);
        let mut g = TagGenome::bare_seed();
        let acts = |g: &TagGenome| live(g, t, Physics::Varied, 3).iter().filter(|e| e.moved.is_some() || e.said.is_some()).count();
        assert!(acts(&g) > 0);
        if let Token::Pop(p) = &mut g.tokens[0] {
            p.tonic = 0;
        }
        assert_eq!(acts(&g), 0);
    }
}
