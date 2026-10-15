//! A tag-addressed grammar: the genome is a string of tokens, and every string builds a network.
//!
//! - `Channel { tag, width }`: a bit vector of 2^width words.
//! - `Pop { from, to, .. }`: a population of kernels (or a wire) reading the channel whose tag is
//!   nearest `from` and writing the one nearest `to`, with every property inline.
//!
//! Tags name the genome's own channels only: a tag resolves to the nearest declared channel
//! (Hamming distance; ties to the earlier one). The world's side has ports with a tag space of
//! their own: a population with `from_port` reads the input port nearest its `from` tag (all the
//! senses, the eye, ear, touch or affect, or the self channel), and one with `to_port` writes the
//! output port nearest its `to` (the whole act, the voice or the hand). A population with no
//! declared channel to resolve to is left out until one exists. Inserting, deleting, moving or
//! copying any part of the string still makes a network. Wiring changes gradually (a flipped tag bit may move a population to a neighbouring
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
/// The self channel's tag: what the network senses of itself (read-only; see
/// `RuntimeNetwork::write_self`).
pub const SELF: u8 = 0x80;
/// The self channel's width: 4 words.
pub const SELF_BITS: usize = 256;

/// Input ports: where a population reading from outside (`from_port`) can read, each with its own
/// tag in a tag space of its own (a population's `from` tag picks the nearest): all the senses,
/// one sense channel, or the self channel.
pub const EYE: u8 = 0x07;
pub const EAR: u8 = 0x38;
pub const TOUCH: u8 = 0xC1;
pub const AFFECT: u8 = 0xE0;
const INPUTS: [(u8, &str); 6] = [(SENSES, "S"), (EYE, "eye"), (EAR, "ear"), (TOUCH, "touch"), (AFFECT, "affect"), (SELF, "M")];
/// Output ports: where a population writing outward (`to_port`) can write: the whole act, the
/// voice or the hand.
pub const VOICE: u8 = 0x1F;
pub const HAND: u8 = 0xF8;
const OUTPUTS: [(u8, &str); 3] = [(ACT, "A"), (VOICE, "voice"), (HAND, "hand")];

/// Where an endpoint lands: a node, its first word and how many words (a port may be part of a
/// node, as the eye is part of the senses).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Place {
    node: usize,
    off: usize,
    len: usize,
    /// the whole node
    whole: bool,
}

/// A population: kernels (or a wire) from one channel to another, with all its properties.
#[derive(Clone, Debug, PartialEq)]
pub struct Pop {
    /// a permanent id: keys its random stream (a copy gets a new one)
    pub id: u32,
    /// where it reads: with `from_port`, the input port nearest `from` (senses, one sense, self);
    /// otherwise the declared channel nearest `from` (tags name the genome's own channels only)
    pub from: u8,
    pub from_port: bool,
    /// where it writes: with `to_port`, the output port nearest `to` (act, voice, hand); otherwise
    /// the declared channel nearest `to`
    pub to: u8,
    pub to_port: bool,
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
    /// an execution gate: it runs only while the channel nearest this tag had lines on (true)
    /// or none (false) a frame ago; paused, it reads, writes and learns nothing and costs nothing
    pub pause: Option<(u8, bool)>,
    /// kernel replication: 0 off; k > 0, a kernel confirmed 2^k times (net) is copied, to a
    /// neighbouring tile (`replicate_shift`) or in place
    pub replicate: u8,
    pub replicate_shift: bool,
}

impl Pop {
    /// A population with the grammar's starting values.
    pub fn new(id: u32, from: u8, to: u8) -> Self {
        let p = EdgeSpec::plain();
        Self {
            id, from, from_port: from == SENSES || from == SELF, to, to_port: to == ACT, wire: false, op: KernelOp::Or, lag: 1, span: 1, at: 0, field: 0, place: 0, spread: 0,
            count: 0, fan_in: p.start_syn, fan_out: p.start_out, quarters: p.start_quarters, tonic: 0, share: false,
            grow: false, new_code: false, repeat: false, repeat_lag: 1, gate: None, gate_lag: 1, birth_syn: p.birth_syn,
            birth_quarters: p.birth_quarters, out_min: p.out_min, out_max: p.out_max, cap: p.cap, code_lines: p.code_lines,
            out_learn: p.out_learn, fade: p.fade, rotate: 0, sparse: false, cover: false, tile: 6, pause: None, replicate: 0, replicate_shift: false,
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
    /// (tag, words) per node: the senses first, then the declared channels, the self channel, the
    /// act last
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
        list.push((SELF, SELF_BITS / 64));
        hold.push(false);
        list.push((ACT, world::ACT_BITS / 64));
        hold.push(false);
        Self { list, hold }
    }

    /// The self channel's index (just before the act).
    fn self_index(&self) -> usize {
        self.list.len() - 2
    }

    fn act_index(&self) -> usize {
        self.list.len() - 1
    }

    /// The declared channel nearest a tag (ties: the earlier), if any is declared.
    fn internal(&self, tag: u8) -> Option<usize> {
        (1..self.self_index()).min_by_key(|&i| ((self.list[i].0 ^ tag).count_ones(), i))
    }

    fn whole(&self, node: usize) -> Place {
        Place { node, off: 0, len: self.list[node].1, whole: true }
    }

    /// The input port nearest a tag.
    fn input(&self, tag: u8) -> Place {
        let k = (0..INPUTS.len()).min_by_key(|&i| ((INPUTS[i].0 ^ tag).count_ones(), i)).unwrap();
        match k {
            0 => self.whole(0),
            5 => self.whole(self.self_index()),
            c => {
                let ch = world::CH_BITS / 64;
                let sense = [world::EYE, world::EAR, world::TOUCH, world::AFFECT][c - 1];
                Place { node: 0, off: sense * ch, len: ch, whole: false }
            }
        }
    }

    /// The output port nearest a tag.
    fn output(&self, tag: u8) -> Place {
        let k = (0..OUTPUTS.len()).min_by_key(|&i| ((OUTPUTS[i].0 ^ tag).count_ones(), i)).unwrap();
        let a = self.act_index();
        let ch = world::CH_BITS / 64;
        match k {
            0 => self.whole(a),
            1 => Place { node: a, off: world::VOICE * ch, len: ch, whole: false },
            _ => Place { node: a, off: world::HAND * ch, len: ch, whole: false },
        }
    }

    /// Where a population reads, if anywhere.
    fn from(&self, p: &Pop) -> Option<Place> {
        if p.from_port { Some(self.input(p.from)) } else { self.internal(p.from).map(|i| self.whole(i)) }
    }

    /// Where a population writes, if anywhere.
    fn to(&self, p: &Pop) -> Option<Place> {
        if p.to_port { Some(self.output(p.to)) } else { self.internal(p.to).map(|i| self.whole(i)) }
    }

    /// A short name for a place.
    fn name(&self, at: Place) -> String {
        if at.node == 0 {
            if at.whole { return "S".into() }
            let ch = world::CH_BITS / 64;
            return ["eye", "ear", "touch", "affect"][at.off / ch].into();
        }
        if at.node == self.self_index() { return "M".into() }
        if at.node == self.act_index() {
            if at.whole { return "A".into() }
            return if at.off == world::VOICE * world::CH_BITS / 64 { "voice".into() } else { "hand".into() };
        }
        format!("C{}", at.node)
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
        let gname = |t: u8| ch.internal(t).map_or("-".to_string(), |i| format!("C{i}"));
        let mut out = String::new();
        let mut c = 0;
        for t in &self.tokens {
            match t {
                Token::Channel { tag, width, hold } => {
                    c += 1;
                    out += &format!("C{c} channel {tag:08b}  {} bits{}\n", 64usize << (*width).min(MAX_WIDTH), if *hold { "  holds" } else { "" });
                }
                Token::Pop(p) => {
                    let (Some(a), Some(b)) = (ch.from(p), ch.to(p)) else {
                        out += "   (a population with no channel to read or write)\n";
                        continue;
                    };
                    out += &format!(
                        "   {} -> {} {} {:?} lag {}{} {}{}{}{}{}{}{}{}\n",
                        ch.name(a), ch.name(b), if p.wire { "wire" } else { "kernels" }, p.op, p.lag,
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
                        p.gate.map_or(String::new(), |(g, on)| format!(" gate {}{}", if on { "" } else { "!" }, gname(g)))
                            + &p.pause.map_or(String::new(), |(g, on)| format!(" runs while {}{}", if on { "" } else { "!" }, gname(g))),
                        if p.sparse { " sparse".to_string() } else { String::new() }
                            + &if p.replicate > 0 { format!(" replicate at {} ({})", 1u32 << p.replicate.min(7), if p.replicate_shift { "to a neighbour tile" } else { "in place" }) } else { String::new() },
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
            let (Some(a), Some(b)) = (ch.from(p), ch.to(p)) else { continue };
            let lag = (p.lag as usize).max(1);
            let span = (p.span as usize).max(1);
            depth[a.node] = depth[a.node].max(lag + span - 1);
            if p.repeat {
                depth[a.node] = depth[a.node].max(lag + p.repeat_lag as usize + 1);
                depth[b.node] = depth[b.node].max(p.repeat_lag as usize);
            }
            let gate = p.gate.and_then(|(t, on)| ch.internal(t).map(|i| (i, on)));
            if let Some((g, _)) = gate {
                depth[g] = depth[g].max((p.gate_lag as usize).max(1));
            }
            let pause = p.pause.and_then(|(t, on)| ch.internal(t).map(|i| (i, on)));
            let field = 1usize << p.field.min(MAX_WIDTH);
            // what it reads: all of its port or channel (cover), or a field placed inside it
            let window = if p.cover {
                if a.whole { None } else { Some((a.off, a.len)) }
            } else {
                let at = p.at as usize % a.len.max(1);
                Some((a.off + at, field.min(a.len - at)))
            };
            // where it writes: inside its port (a part of the act), or its channel
            let place = if p.cover { 0 } else { p.place as usize % b.len.max(1) };
            edges.push((a.node, b.node, EdgeSpec {
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
                window,
                dst_word: b.off + place,
                dst_len: if b.whole { 0 } else { b.len },
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
                out_words: if p.cover { 0 } else { (1usize << p.spread.min(MAX_WIDTH)).min(if b.whole { usize::MAX } else { b.len - place }) as u16 },
                tile: if p.cover && !p.wire { p.tile.clamp(3, MAX_TILE) } else { 0 },
                tonic: p.tonic,
                sparse: p.sparse,
                pause,
                replicate: p.replicate,
                replicate_shift: p.replicate_shift,
                ..EdgeSpec::plain()
            }));
        }
        let mut g: Graph<NodeSpec, EdgeSpec, Directed> = Graph::new();
        let idx: Vec<_> = (0..n)
            .map(|i| {
                let name = if i == 0 { Some("input".to_string()) } else if i == n - 1 { Some("output".to_string()) } else if i == n - 2 { Some("self".to_string()) } else { None };
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
        let Some(b) = ch.to(p) else { return };
        let width = b.len.max(1).trailing_zeros() as u8;
        let (old, old_port) = (p.to, p.to_port);
        let new = old ^ (1u8 << (bit & 7));
        // the carry copies the whole new channel on to the old destination (inside its port)
        let carry = Pop { cover: true, from_port: false, to: old, to_port: old_port, ..Pop::wire(self.next_id, new, old) };
        self.next_id += 1;
        if let Token::Pop(p) = &mut self.tokens[at] {
            p.to = new;
            p.to_port = false;
        }
        self.tokens.insert(at, Token::Channel { tag: new, width, hold: false });
        self.tokens.insert(at + 2, Token::Pop(carry));
    }

    /// A circuit: the declared channel at `channel` (its index among the channels, 1-based as
    /// `Channels` numbers them) and every population that reads it, writes it, or is gated or
    /// paused on it. Returns token indices (the channel's first).
    pub fn circuit(&self, channel: usize) -> Vec<usize> {
        let ch = Channels::of(&self.tokens);
        let mut c = 0;
        let mut out = Vec::new();
        for (i, t) in self.tokens.iter().enumerate() {
            if let Token::Channel { .. } = t {
                c += 1;
                if c == channel {
                    out.insert(0, i);
                }
            }
        }
        if out.is_empty() {
            return out;
        }
        for (i, t) in self.tokens.iter().enumerate() {
            if let Token::Pop(p) = t {
                let here = |t: u8| ch.internal(t) == Some(channel);
                let touches = (!p.from_port && here(p.from))
                    || (!p.to_port && here(p.to))
                    || p.gate.is_some_and(|(g, _)| here(g))
                    || p.pause.is_some_and(|(g, _)| here(g));
                if touches {
                    out.push(i);
                }
            }
        }
        out
    }

    /// The tokens of a circuit (`circuit(channel)`) made private: the channel gets the fresh tag
    /// `fresh`, every reference that resolved to it now names `fresh` exactly, and populations
    /// get new ids from `next_id`. What the circuit reads and writes outside stays as it was.
    fn private_circuit(&self, channel: usize, fresh: u8, next_id: &mut u32) -> Vec<Token> {
        self.private_tokens(&self.circuit(channel), channel, fresh, next_id)
    }

    /// `private_circuit` on some of the circuit's tokens (`at`: indices, the channel's first).
    fn private_tokens(&self, at: &[usize], channel: usize, fresh: u8, next_id: &mut u32) -> Vec<Token> {
        let ch = Channels::of(&self.tokens);
        at.iter()
            .copied()
            .map(|i| match &self.tokens[i] {
                Token::Channel { width, hold, .. } => Token::Channel { tag: fresh, width: *width, hold: *hold },
                Token::Pop(p) => {
                    let mut q = p.clone();
                    let here = |t: u8| ch.internal(t) == Some(channel);
                    if !p.from_port && here(p.from) { q.from = fresh }
                    if !p.to_port && here(p.to) { q.to = fresh }
                    if let Some((g, on)) = p.gate { if here(g) { q.gate = Some((fresh, on)) } }
                    if let Some((g, on)) = p.pause { if here(g) { q.pause = Some((fresh, on)) } }
                    q.id = *next_id;
                    *next_id += 1;
                    Token::Pop(q)
                }
            })
            .collect()
    }

    /// A tag no declared channel (nor the senses, act or self channel) has.
    fn fresh_tag(&self, rng: &mut StdRng) -> u8 {
        let ch = Channels::of(&self.tokens);
        loop {
            let t: u8 = rng.r#gen();
            if ch.list.iter().all(|c| c.0 != t) {
                return t;
            }
        }
    }

    /// Circuit copy (gene duplication): the circuit around a declared channel, copied whole with
    /// a private channel, appended. With `rewire`, the copy's populations that read from outside
    /// read another channel instead (the same machinery on other data).
    pub fn copy_circuit(&mut self, channel: usize, rewire: Option<(u8, bool)>, rng: &mut StdRng) {
        let fresh = self.fresh_tag(rng);
        let mut next = self.next_id;
        let mut copy = self.private_circuit(channel, fresh, &mut next);
        if copy.is_empty() {
            return;
        }
        if let Some((src, port)) = rewire {
            for t in &mut copy {
                if let Token::Pop(p) = t {
                    if p.to == fresh && !p.to_port && (p.from != fresh || p.from_port) {
                        p.from = src;
                        p.from_port = port;
                    }
                }
            }
        }
        self.next_id = next;
        self.tokens.extend(copy);
    }

    /// Plasmid transfer: the circuit around one of `donor`'s declared channels, made private,
    /// appended to this genome. What it reads and writes outside resolves among this genome's
    /// channels (the senses and the act always match).
    pub fn take_plasmid(&mut self, donor: &TagGenome, channel: usize, rng: &mut StdRng) {
        let fresh = self.fresh_tag(rng);
        let mut next = self.next_id.max(donor.next_id);
        let plasmid = donor.private_circuit(channel, fresh, &mut next);
        if plasmid.is_empty() {
            return;
        }
        self.next_id = next;
        self.tokens.extend(plasmid);
    }

    /// The ports its populations read and write (each named once): "eye", "hand", ...
    pub fn ports(&self) -> (Vec<String>, Vec<String>) {
        let ch = Channels::of(&self.tokens);
        let (mut r, mut w) = (Vec::new(), Vec::new());
        for p in self.pops() {
            if let (Some(a), Some(b)) = (ch.from(p), ch.to(p)) {
                if p.from_port && !r.contains(&ch.name(a)) {
                    r.push(ch.name(a));
                }
                if p.to_port && !w.contains(&ch.name(b)) {
                    w.push(ch.name(b));
                }
            }
        }
        (r, w)
    }

    /// Its blind control: every population that reads the senses (or one sense) reads the self
    /// channel instead, so the same circuit runs with nothing from the world.
    pub fn blind(&self) -> TagGenome {
        let ch = Channels::of(&self.tokens);
        let mut g = self.clone();
        for t in &mut g.tokens {
            if let Token::Pop(p) = t {
                if p.from_port && ch.input(p.from).node == 0 {
                    p.from = SELF;
                }
            }
        }
        g
    }

    /// A plasmid around one of `donor`'s populations (the `k`-th): the circuit of a declared
    /// channel it reads, writes or is gated by, made private; or, when it touches none (it
    /// reads and writes ports), the population alone, with a new id. Appended to this genome;
    /// returns the plasmid's tokens as moved (empty if the donor has no population).
    pub fn take_population_plasmid(&mut self, donor: &TagGenome, k: usize, rng: &mut StdRng) -> Vec<Token> {
        self.take_population_plasmid_upto(donor, k, 0, rng)
    }

    /// `take_population_plasmid`, with a circuit of more than `max` populations (0: no limit)
    /// cut to the chosen population and its nearest neighbours in the circuit (genome order),
    /// `max` of them with the channel: a small mobile unit, not a busy channel's whole tangle.
    pub fn take_population_plasmid_upto(&mut self, donor: &TagGenome, k: usize, max: usize, rng: &mut StdRng) -> Vec<Token> {
        let pops: Vec<usize> = (0..donor.tokens.len()).filter(|&i| matches!(donor.tokens[i], Token::Pop(_))).collect();
        if pops.is_empty() {
            return vec![];
        }
        let chosen = pops[k % pops.len()];
        let Token::Pop(p) = &donor.tokens[chosen] else { unreachable!() };
        let ch = Channels::of(&donor.tokens);
        let touched = [(!p.from_port).then_some(p.from), (!p.to_port).then_some(p.to), p.gate.map(|g| g.0), p.pause.map(|g| g.0)]
            .into_iter()
            .flatten()
            .find_map(|t| ch.internal(t));
        let before = self.tokens.len();
        match touched {
            Some(c) if max > 0 && donor.circuit(c).len() > max + 1 => {
                let circ = donor.circuit(c);
                let members = &circ[1..];
                let at = members.iter().position(|&i| i == chosen).unwrap_or(0);
                let lo = at.saturating_sub((max - 1) / 2).min(members.len() - max);
                let keep: Vec<usize> = std::iter::once(circ[0]).chain(members[lo..lo + max].iter().copied()).collect();
                let fresh = self.fresh_tag(rng);
                let mut next = self.next_id.max(donor.next_id);
                let moved = donor.private_tokens(&keep, c, fresh, &mut next);
                self.next_id = next;
                self.tokens.extend(moved);
            }
            Some(c) => self.take_plasmid(donor, c, rng),
            None => {
                let mut q = p.clone();
                q.id = self.next_id.max(donor.next_id);
                self.next_id = q.id + 1;
                self.tokens.push(Token::Pop(q));
            }
        }
        self.tokens[before..].to_vec()
    }

    /// The genome as text, one token a line (`next_id` first), read back by `from_text`: a
    /// checkpoint can be saved and an evolution resumed from it.
    pub fn to_text(&self) -> String {
        let b = |x: bool| x as u8;
        let opt = |o: Option<(u8, bool)>| o.map_or("-".to_string(), |(t, on)| format!("{t}:{}", b(on)));
        let mut out = format!("genome {}\n", self.next_id);
        for t in &self.tokens {
            match t {
                Token::Channel { tag, width, hold } => out += &format!("channel {tag} {width} {}\n", b(*hold)),
                Token::Pop(p) => {
                    out += &format!(
                        "pop id={} from={} from_port={} to={} to_port={} wire={} op={} lag={} span={} at={} field={} place={} spread={} count={} fan_in={} fan_out={} quarters={} tonic={} share={} grow={} new_code={} repeat={} repeat_lag={} gate={} gate_lag={} birth_syn={} birth_quarters={} out_min={} out_max={} cap={} code_lines={} out_learn={} fade={} rotate={} sparse={} cover={} tile={} pause={} replicate={} replicate_shift={}\n",
                        p.id, p.from, b(p.from_port), p.to, b(p.to_port), b(p.wire), p.op.text(), p.lag, p.span, p.at, p.field, p.place, p.spread,
                        p.count, p.fan_in, p.fan_out, p.quarters, p.tonic, b(p.share), b(p.grow), b(p.new_code), b(p.repeat), p.repeat_lag,
                        opt(p.gate), p.gate_lag, p.birth_syn, p.birth_quarters, p.out_min, p.out_max, p.cap, p.code_lines, p.out_learn, p.fade,
                        p.rotate, b(p.sparse), b(p.cover), p.tile, opt(p.pause), p.replicate, b(p.replicate_shift)
                    );
                }
            }
        }
        out
    }

    /// The genomes in `text` (as `to_text` wrote them, one after another).
    pub fn from_text(text: &str) -> Result<Vec<TagGenome>, String> {
        let mut out: Vec<TagGenome> = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            let bad = |what: &str| format!("line {}: {what}: {line}", n + 1);
            let mut w = line.split_whitespace();
            match w.next() {
                None => continue,
                Some("genome") => out.push(TagGenome { tokens: vec![], next_id: w.next().and_then(|x| x.parse().ok()).ok_or_else(|| bad("next id"))? }),
                Some("channel") => {
                    let v: Vec<u8> = w.map(|x| x.parse().map_err(|_| bad("channel"))).collect::<Result<_, _>>()?;
                    let [tag, width, hold] = v[..] else { return Err(bad("channel")) };
                    out.last_mut().ok_or_else(|| bad("before any genome"))?.tokens.push(Token::Channel { tag, width, hold: hold == 1 });
                }
                Some("pop") => {
                    let kv: std::collections::HashMap<&str, &str> = w.filter_map(|x| x.split_once('=')).collect();
                    let get = |k: &str| kv.get(k).copied().ok_or_else(|| bad(k));
                    macro_rules! num { ($k:expr) => { get($k)?.parse().map_err(|_| bad($k))? }; }
                    let flag = |k: &str| -> Result<bool, String> { Ok(get(k)? == "1") };
                    let opt = |k: &str| -> Result<Option<(u8, bool)>, String> {
                        let v = get(k)?;
                        if v == "-" { return Ok(None) }
                        let (t, on) = v.split_once(':').ok_or_else(|| bad(k))?;
                        Ok(Some((t.parse().map_err(|_| bad(k))?, on == "1")))
                    };
                    let p = Pop {
                        id: num!("id"), from: num!("from"), from_port: flag("from_port")?, to: num!("to"), to_port: flag("to_port")?,
                        wire: flag("wire")?, op: KernelOp::parse(get("op")?).ok_or_else(|| bad("op"))?, lag: num!("lag"), span: num!("span"),
                        at: num!("at"), field: num!("field"), place: num!("place"), spread: num!("spread"), count: num!("count"),
                        fan_in: num!("fan_in"), fan_out: num!("fan_out"), quarters: num!("quarters"), tonic: num!("tonic"),
                        share: flag("share")?, grow: flag("grow")?, new_code: flag("new_code")?, repeat: flag("repeat")?,
                        repeat_lag: num!("repeat_lag"), gate: opt("gate")?, gate_lag: num!("gate_lag"), birth_syn: num!("birth_syn"),
                        birth_quarters: num!("birth_quarters"), out_min: num!("out_min"), out_max: num!("out_max"), cap: num!("cap"),
                        code_lines: num!("code_lines"), out_learn: num!("out_learn"), fade: num!("fade"), rotate: num!("rotate"),
                        sparse: flag("sparse")?, cover: flag("cover")?, tile: num!("tile"), pause: opt("pause")?,
                        replicate: num!("replicate"), replicate_shift: flag("replicate_shift")?,
                    };
                    out.last_mut().ok_or_else(|| bad("before any genome"))?.tokens.push(Token::Pop(p));
                }
                Some(_) => return Err(bad("unknown line")),
            }
        }
        Ok(out)
    }

    /// How many populations it has.
    pub fn populations(&self) -> usize {
        self.pops().count()
    }

    /// How many channels are declared.
    pub fn channels(&self) -> usize {
        self.tokens.iter().filter(|t| matches!(t, Token::Channel { .. })).count()
    }

    /// Self-loop: the population at token `at` writes back into the channel it reads (a carry,
    /// a clock, a counter, held state). Not on the senses, which nothing writes.
    pub fn self_loop(&mut self, at: usize) {
        let ch = Channels::of(&self.tokens);
        if let Some(Token::Pop(p)) = self.tokens.get_mut(at) {
            if p.from_port {
                return;
            }
            if let Some(i) = ch.internal(p.from) {
                p.to = ch.list[i].0;
                p.to_port = false;
            }
        }
    }

    fn random_pop(&mut self, rng: &mut StdRng) -> Pop {
        let id = self.next_id;
        self.next_id += 1;
        // a new population is born active: some kernels, a random field and fan-in, sometimes
        // tonic, sometimes growing (an inert gene would need several more mutations to matter)
        let mut p = Pop::new(id, rng.r#gen(), rng.r#gen());
        // reading from outside (a sense or the self channel) or a channel; writing outward (the
        // act, voice or hand) or a channel
        p.from_port = rng.gen_bool(0.4);
        p.to_port = rng.gen_bool(0.4);
        p.wire = rng.gen_bool(0.25);
        p.sparse = rng.gen_bool(0.5);
        // full coverage, at a random scale from a byte to two words
        p.cover = true;
        p.tile = rng.gen_range(3..=7);
        // sometimes replicating its confirmed kernels
        if rng.gen_range(0..3) == 0 {
            p.replicate = rng.gen_range(2..=5);
            p.replicate_shift = rng.gen_bool(0.5);
        }
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
    // the body's moves and tokens inside its port (the hand's, the voice's, or all of them)
    let (lo, hi) = if sp.dst_len > 0 { (from, from + sp.dst_len) } else { (0, dst_words) };
    let mine: Vec<&BitVector> = primitives.iter().filter(|p| p.as_words().iter().position(|&x| x != 0).is_some_and(|w| (lo..hi).contains(&w))).collect();
    let motor = |r: &mut StdRng| {
        let p = mine.choose(r).unwrap();
        let w = p.as_words().iter().position(|&x| x != 0).unwrap_or(0);
        (BitVector::from_words(vec![p.as_words()[w]]), w, KernelOp::Or)
    };
    let kernels = (0..sp.start_kernels)
        .map(|_| {
            let mut k = if sp.tile > 0 {
                // full coverage: a random tile, lines inside it, output at the same relative place
                let bits = field * 64;
                let t = (1usize << sp.tile.min(20)).min(bits);
                let stride = (t / 2).max(1);
                let tiles = if t >= bits { 1 } else { (bits - t) / stride + 1 };
                let start = r.gen_range(0..tiles) * stride;
                let all: Vec<usize> = (start..start + t).collect();
                let syn: Vec<usize> = all.choose_multiple(&mut r, (sp.start_syn as usize).min(t)).copied().collect();
                let (out, out_word, op) = if e.to_output && !mine.is_empty() {
                    motor(&mut r)
                } else {
                    let len = if sp.dst_len > 0 { sp.dst_len.min(dst_words - from) } else { dst_words };
                    let base = if sp.dst_len > 0 { from } else { 0 };
                    (sparse(64, sp.start_out as usize, &mut r), base + (start * len / field.max(1)).min(len - 1), sp.op)
                };
                let threshold = (sp.start_quarters as usize * syn.len()).div_ceil(4).max(1);
                SimpleKernel::tile(start, t, &syn, out, out_word, threshold, op)
            } else {
                let syn = sparse(field * 64, sp.start_syn as usize, &mut r);
                let (out, out_word, op) = if e.to_output && !mine.is_empty() {
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
        let mut net = RuntimeNetwork::from_graph_held(&g, world::SENSE_BITS, world::ACT_BITS, seed, w.in_place, |e: &EdgeInfo| start_kernels(e, &primitives));
        net.count_reads = w.count_reads;
        net
    }

    fn mutate(&self, rng: &mut StdRng) -> Self {
        let mut g = self.clone();
        let n = g.tokens.len();
        match rng.gen_range(0..16) {
            // a new token (two chances in sixteen: new populations are the raw material)
            0 | 15 => {
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
            // a circuit copied whole (gene duplication), half the time reading another channel
            14 if g.channels() > 0 => {
                let c = rng.gen_range(1..=g.channels());
                let rewire = if rng.gen_bool(0.5) { Some((rng.r#gen(), rng.gen_bool(0.5))) } else { None };
                g.copy_circuit(c, rewire, rng);
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
                    Token::Pop(p) => match rng.gen_range(0..38) {
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
                        32 => p.tile = step(p.tile, 3, MAX_TILE),
                        33 => p.replicate = step(p.replicate, 0, 7),
                        34 => p.replicate_shift = !p.replicate_shift,
                        36 => p.from_port = !p.from_port,
                        37 => p.to_port = !p.to_port,
                        _ => p.pause = match p.pause { Some((t, on)) => if rng.gen_bool(0.3) { None } else if rng.gen_bool(0.5) { Some((t ^ bit, on)) } else { Some((t, !on)) }, None => Some((rng.r#gen(), rng.gen_bool(0.5))) },
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

    /// Half the time a plasmid: one whole circuit of the other genome, made private and appended
    /// (so a working circuit moves intact); otherwise one point: this string's front, the
    /// other's back (tags carry the wiring across).
    fn crossover(&self, other: &Self, rng: &mut StdRng) -> Self {
        if other.populations() > 0 && rng.gen_bool(0.5) {
            let mut g = self.clone();
            let k = rng.gen_range(0..other.populations());
            g.take_population_plasmid(other, k, rng);
            return g;
        }
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
    fn tags_name_channels_and_ports_have_their_own_tags() {
        let ch = Channels::of(&[Token::Channel { tag: 0b1010_0000, width: 0, hold: false }, Token::Channel { tag: 0b0000_0101, width: 1, hold: false }]);
        // an ordinary tag resolves among the declared channels only
        assert_eq!(ch.internal(0b1010_0001), Some(1));
        assert_eq!(ch.internal(0b0000_0111), Some(2));
        // 0x00 is two bits from both channels (ties go to the earlier): just a channel tag here
        assert_eq!(ch.internal(SENSES), Some(1), "the senses are not in the channels' tag space");
        assert_eq!(Channels::of(&[]).internal(0x42), None);
        // ports: each sense, the self channel, and parts of the act, by their own tags
        let w = world::CH_BITS / 64;
        assert_eq!(ch.input(SENSES), Place { node: 0, off: 0, len: 4 * w, whole: true });
        assert_eq!(ch.input(EYE), Place { node: 0, off: world::EYE * w, len: w, whole: false });
        assert_eq!(ch.input(AFFECT), Place { node: 0, off: world::AFFECT * w, len: w, whole: false });
        assert_eq!(ch.input(SELF).node, ch.self_index());
        assert_eq!(ch.output(ACT), Place { node: ch.act_index(), off: 0, len: 2 * w, whole: true });
        assert_eq!(ch.output(HAND), Place { node: ch.act_index(), off: world::HAND * w, len: w, whole: false });
        // every input port is reachable from some tag, and so is every output port
        let mut seen_in = std::collections::HashSet::new();
        let mut seen_out = std::collections::HashSet::new();
        for t in 0..=255u8 {
            seen_in.insert(ch.name(ch.input(t)));
            seen_out.insert(ch.name(ch.output(t)));
        }
        assert_eq!((seen_in.len(), seen_out.len()), (6, 3));
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
            let (a, b) = (ch.from(p).unwrap(), ch.to(p).unwrap());
            assert_eq!(ch.list[a.node].0, p.from);
            assert_eq!(ch.list[b.node].0, p.to);
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
        assert!(w.wire && w.from == 0xF1 && w.to == b && w.cover && !w.from_port && !w.to_port);
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
    fn in_place_a_network_builds_its_own_delay_and_latch() {
        use crate::bitvec::bithistory::AdvanceMode;
        let (a, b, e, c) = (0x3C, 0xC3, 0x5A, 0xA5);
        // a delay line: B takes A before A takes the senses' word 0 (genome order), so B is A a
        // step late; both overwrite (Set), so a value passes through and is gone
        let a_to_b = Pop { cover: true, op: KernelOp::Set, ..Pop::wire(1, a, b) };
        let in_a = Pop { field: 0, op: KernelOp::Set, ..Pop::wire(2, SENSES, a) };
        // a latch: E is the senses' word 1 now; C takes word 0 only while E is on, else keeps it
        let in_e = Pop { at: 1, field: 0, op: KernelOp::Set, ..Pop::wire(3, SENSES, e) };
        let latch = Pop { field: 0, op: KernelOp::Set, pause: Some((e, true)), ..Pop::wire(4, SENSES, c) };
        let ch = |t: u8| Token::Channel { tag: t, width: 0, hold: false };
        let g = TagGenome { tokens: vec![ch(a), ch(b), ch(e), ch(c), Token::Pop(a_to_b), Token::Pop(in_a), Token::Pop(in_e), Token::Pop(latch)], next_id: 5 };
        let mut w = World::new(1, 4);
        w.in_place = true;
        let mut net = g.build(&w, 1);
        let (x1, x2) = (0xF0F0u64, 0x0F0Fu64);
        let mut rows = Vec::new();
        for t in 0..6 {
            let mut words = vec![0u64; world::SENSE_BITS / 64];
            words[0] = [x1, 0, x2, 0, 0, 0][t];
            words[1] = [u64::MAX, 0, 0, 0, 0, 0][t];
            net.tick(AdvanceMode::Clear, 0);
            net.write_input(&BitVector::from_words(words), 0);
            rows.push([1, 2, 4].map(|n| net.read_node(n).as_words()[0]));
        }
        // (the senses arrive a step after they are shown)
        let a_seen: Vec<u64> = rows.iter().map(|r| r[0]).collect();
        let b_seen: Vec<u64> = rows.iter().map(|r| r[1]).collect();
        let c_seen: Vec<u64> = rows.iter().map(|r| r[2]).collect();
        assert_eq!(a_seen, vec![0, x1, 0, x2, 0, 0]);
        assert_eq!(b_seen, vec![0, 0, x1, 0, x2, 0]);
        // the latch took x1 while enabled and kept it through x2
        assert_eq!(c_seen, vec![0, x1, x1, x1, x1, x1]);
        // no history: every held channel is one frame
        assert!((1..=4).all(|n| net.nodes[n].frames() == 1));
    }

    #[test]
    fn in_place_every_channel_holds_in_one_frame_and_order_matters() {
        use crate::bitvec::bithistory::AdvanceMode;
        let (h, k) = (0x3C, 0xC3);
        let set = Pop { field: 0, ..Pop::wire(1, SENSES, h) };
        let clear = Pop { at: 1, field: 0, op: KernelOp::Clear, ..Pop::wire(2, SENSES, h) };
        // K copies H (lag 1): held in place, it sees what H holds now, written earlier this step
        let copy = Pop { cover: true, ..Pop::wire(3, h, k) };
        let g = TagGenome { tokens: vec![Token::Channel { tag: h, width: 0, hold: false }, Token::Channel { tag: k, width: 0, hold: false }, Token::Pop(set), Token::Pop(clear), Token::Pop(copy)], next_id: 4 };
        let run = |in_place: bool| -> (Vec<u32>, Vec<u32>, usize) {
            let mut w = World::new(1, 4);
            w.in_place = in_place;
            let mut net = g.build(&w, 1);
            let (mut hs, mut ks) = (Vec::new(), Vec::new());
            for t in 0..6 {
                let mut words = vec![0u64; world::SENSE_BITS / 64];
                if t == 0 { words[0] = 0xF0F0 }
                if t == 4 { words[1] = u64::MAX }
                net.tick(AdvanceMode::Clear, 0);
                net.write_input(&BitVector::from_words(words), 0);
                hs.push(net.read_node(1).count_ones() as u32);
                ks.push(net.read_node(2).count_ones() as u32);
            }
            (hs, ks, net.nodes[1].frames())
        };
        // held in place: H keeps the code (no hold flag, no carry) until the clearing write, in
        // one frame; K has it the same step H gets it, and keeps it after H is cleared
        let (hs, ks, frames) = run(true);
        assert_eq!(hs, vec![0, 8, 8, 8, 8, 0]);
        assert_eq!(ks, vec![0, 8, 8, 8, 8, 8]);
        assert_eq!(frames, 1);
        // the old runtime: H loses it the next step, K sees it a step late, two frames each
        let (hs, ks, frames) = run(false);
        assert_eq!(hs, vec![0, 8, 0, 0, 0, 0]);
        assert_eq!(ks, vec![0, 0, 8, 0, 0, 0]);
        assert_eq!(frames, 2);
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
    fn only_a_genome_silent_in_every_lifetime_dies() {
        use crate::evolve::{fitness_detail, Physics, Task};
        use crate::world::{observe, Conditioning};
        let task = Task::Conditioning(Conditioning::Instrumental, 40);
        let empty = TagGenome { tokens: vec![], next_id: 1 };
        let (f, _, silent) = fitness_detail(&empty, task, Physics::Varied, &[1, 2, 3]);
        assert_eq!((f, silent), (observe::SILENT, 3));
        // a tonic population on the act clicks in some lifetimes: no death, silent ones count 0
        let babble = TagGenome { tokens: vec![Token::Pop(Pop { count: 8, tonic: 1, ..Pop::new(1, SELF, ACT) })], next_id: 2 };
        let (f, _, silent) = fitness_detail(&babble, task, Physics::Varied, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(silent < 8 && f > -1024, "{f} {silent}");
    }

    #[test]
    fn a_paused_population_does_nothing_and_costs_nothing() {
        // C copies the senses, with an execution gate on G (which nothing writes, so G is empty):
        // "runs while G" never runs, "runs while !G" always does
        let run = |on: bool| -> (usize, u64) {
            let (c, gt) = (0x3C, 0xC3);
            let copy = Pop { pause: Some((gt, on)), count: 4, ..Pop::new(1, SENSES, c) };
            let g = TagGenome { tokens: vec![Token::Channel { tag: c, width: 3, hold: false }, Token::Channel { tag: gt, width: 0, hold: false }, Token::Pop(copy)], next_id: 2 };
            let mut net = g.build(&World::new(1, 4), 1);
            let mut lines = 0;
            for _ in 0..6 {
                net.tick(crate::bitvec::bithistory::AdvanceMode::Clear, 0);
                net.write_input(&BitVector::new(world::SENSE_BITS, Some(u64::MAX)), 0);
                lines += net.read_node(1).count_ones();
            }
            (lines, net.work)
        };
        assert_eq!(run(true), (0, 0));
        let (lines, work) = run(false);
        assert!(lines > 0 && work >= 4 * 5, "{lines} {work}");
    }

    #[test]
    fn the_self_channel_reports_work_size_and_time() {
        // C copies the self channel (a wire reads it a frame back); a population of 4 kernels on
        // the senses gives it some work to report
        let c = 0x3C;
        let look = Pop { cover: true, ..Pop::wire(1, SELF, c) };
        let busy = Pop { count: 4, ..Pop::new(2, SENSES, ACT) };
        let g = TagGenome { tokens: vec![Token::Channel { tag: c, width: 2, hold: false }, Token::Pop(look), Token::Pop(busy)], next_id: 3 };
        let ch = Channels::of(&g.tokens);
        assert_eq!(ch.input(SELF).node, ch.self_index());
        assert!((0..=255u8).all(|t| ch.output(t).node != ch.self_index()), "nothing writes the self channel");
        let mut net = g.build(&World::new(1, 4), 1);
        for _ in 0..6 {
            net.tick(crate::bitvec::bithistory::AdvanceMode::Clear, 0);
        }
        let w = net.read_node(1).as_words().to_vec();
        // the frame read was written at step 4 (0-based), about step 3: work 4 words wired (the
        // senses are empty, so the population is idle and costs nothing) -> 3 lines; 4 kernels
        // -> 3 lines; 4 steps so far; 2 edges ran -> 2 lines
        assert_eq!(w, vec![0b111, 0b111, 4, 0b11]);
    }

    #[test]
    fn a_circuit_copies_whole_and_moves_between_genomes() {
        // channel R (0x3C): the eye copied in, predictions XORed out of it, and R wired to the act
        let r = 0x3C;
        let copy = Pop { field: 3, ..Pop::wire(1, SENSES, r) };
        let predict = Pop { lag: 2, op: KernelOp::Xor, grow: true, ..Pop::new(2, SENSES, r) };
        let out = Pop::wire(3, r, ACT);
        let other = Pop::new(4, SENSES, ACT);
        let g = TagGenome { tokens: vec![Token::Channel { tag: r, width: 3, hold: false }, Token::Pop(copy), Token::Pop(predict), Token::Pop(out), Token::Pop(other)], next_id: 5 };
        assert_eq!(g.circuit(1), vec![0, 1, 2, 3], "R and the three populations that touch it");
        let mut rng = StdRng::seed_from_u64(1);
        let mut d = g.clone();
        d.copy_circuit(1, None, &mut rng);
        assert_eq!((d.channels(), d.tokens.len(), d.graph().edge_count()), (2, 9, 7));
        // the copy is private: its populations name its own channel, with fresh ids
        let Token::Channel { tag: fresh, .. } = d.tokens[5] else { panic!() };
        assert_ne!(fresh, r);
        let pops: Vec<&Pop> = d.tokens[6..].iter().map(|t| if let Token::Pop(p) = t { p } else { panic!() }).collect();
        assert!(pops[0].to == fresh && pops[1].to == fresh && pops[2].from == fresh);
        assert!(pops.iter().all(|p| p.id >= 5));
        // a plasmid: the circuit moves into a genome that has none of it
        let mut h = TagGenome { tokens: vec![Token::Pop(Pop::new(1, SENSES, ACT))], next_id: 2 };
        h.take_plasmid(&g, 1, &mut rng);
        assert_eq!((h.channels(), h.tokens.len(), h.graph().edge_count()), (1, 5, 4));
    }

    #[test]
    fn confirmed_predicting_kernels_replicate() {
        // the residue circuit (an eye copy ORed into R, XOR predictions from two frames back) on a
        // repeating cycle: its kernels come true, and with replication they copy themselves
        let run = |replicate: u8| -> usize {
            let r = 0x3C;
            let copy = Pop { field: 3, ..Pop::wire(1, SENSES, r) };
            let predict = Pop { lag: 2, field: 3, op: KernelOp::Xor, grow: true, share: true, replicate, cap: 64, ..Pop::new(2, SENSES, r) };
            let g = TagGenome { tokens: vec![Token::Channel { tag: r, width: 3, hold: false }, Token::Pop(copy), Token::Pop(predict)], next_id: 3 };
            let mut net = g.build(&World::new(1, 4), 1);
            let mut rng = StdRng::seed_from_u64(2);
            let codes: Vec<BitVector> = (0..3).map(|_| world::clustered_code(world::CH_BITS, 16, &mut rng)).collect();
            for t in 0..300 {
                net.tick(crate::bitvec::bithistory::AdvanceMode::Clear, 0);
                net.write_input(&codes[t % 3], 0);
            }
            net.edges[1].group.kernel_count()
        };
        let (plain, copied) = (run(0), run(2));
        assert!(copied > plain, "{plain} -> {copied}");
    }

    #[test]
    fn a_port_reads_one_sense_and_writes_one_effector() {
        // C gets the eye only (a covering wire from the eye port); the hand gets C (a covering wire
        // into the hand port); the voice is never touched
        let c = 0x3C;
        let see = Pop { cover: true, from: EYE, from_port: true, ..Pop::wire(1, EYE, c) };
        let act = Pop { cover: true, to: HAND, to_port: true, ..Pop::wire(2, c, HAND) };
        let g = TagGenome { tokens: vec![Token::Channel { tag: c, width: 3, hold: false }, Token::Pop(see), Token::Pop(act)], next_id: 3 };
        let mut net = g.build(&World::new(1, 4), 1);
        let w = world::CH_BITS / 64;
        let mut senses = vec![u64::MAX; world::SENSE_BITS / 64];
        for x in &mut senses[world::EYE * w..(world::EYE + 1) * w] {
            *x = 0;
        }
        senses[world::EYE * w] = 0b1011;
        for _ in 0..3 {
            net.tick(crate::bitvec::bithistory::AdvanceMode::Clear, 0);
            net.write_input(&BitVector::from_words(senses.clone()), 0);
        }
        // C holds the eye's word (nothing from the other senses, which were all on)
        assert_eq!(net.read_node(1).count_ones(), 3);
        let a = net.read_output().as_words().to_vec();
        assert_eq!(a[world::HAND * w], 0b1011);
        assert!(a[world::VOICE * w..(world::VOICE + 1) * w].iter().all(|&x| x == 0));
    }

    #[test]
    fn a_population_plasmid_moves_its_circuit_or_itself() {
        let r = 0x3C;
        let copy = Pop { field: 3, ..Pop::wire(1, SENSES, r) };
        let out = Pop::wire(2, r, ACT);
        let reflex = Pop { count: 4, cover: true, from: EYE, from_port: true, to: HAND, to_port: true, ..Pop::new(3, EYE, HAND) };
        let donor = TagGenome { tokens: vec![Token::Channel { tag: r, width: 3, hold: false }, Token::Pop(copy), Token::Pop(out), Token::Pop(reflex)], next_id: 4 };
        let mut rng = StdRng::seed_from_u64(1);
        // a population touching R brings R's circuit (R and both its populations)
        let mut g = TagGenome { tokens: vec![], next_id: 1 };
        let moved = g.take_population_plasmid(&donor, 0, &mut rng);
        assert_eq!((moved.len(), g.channels(), g.populations()), (3, 1, 2));
        // a population on ports only moves alone, with a new id
        let mut h = TagGenome { tokens: vec![], next_id: 1 };
        let moved = h.take_population_plasmid(&donor, 2, &mut rng);
        assert_eq!((moved.len(), h.channels(), h.populations()), (1, 0, 1));
        let Token::Pop(p) = &h.tokens[0] else { panic!() };
        assert!(p.from_port && p.to_port && p.id >= 4);
    }

    #[test]
    fn a_capped_plasmid_takes_the_chosen_population_and_its_neighbours() {
        let r = 0x3C;
        // five populations touch R; the third (k = 2) is chosen
        let mut t = vec![Token::Channel { tag: r, width: 3, hold: false }];
        for id in 1..=5 {
            t.push(Token::Pop(Pop { count: id as u8, ..Pop::wire(id, SENSES, r) }));
        }
        let donor = TagGenome { tokens: t, next_id: 6 };
        let mut rng = StdRng::seed_from_u64(1);
        let mut g = TagGenome { tokens: vec![], next_id: 1 };
        let moved = g.take_population_plasmid_upto(&donor, 2, 3, &mut rng);
        // the channel and three populations: the chosen one (count 3) and one either side
        let counts: Vec<u8> = moved.iter().filter_map(|t| if let Token::Pop(p) = t { Some(p.count) } else { None }).collect();
        assert_eq!((g.channels(), counts), (1, vec![2, 3, 4]));
        // no limit: the whole circuit
        let mut h = TagGenome { tokens: vec![], next_id: 1 };
        assert_eq!(h.take_population_plasmid_upto(&donor, 2, 0, &mut rng).len(), 6);
    }

    #[test]
    fn a_blind_control_reads_the_self_channel_for_the_senses() {
        let r = 0x3C;
        let look = Pop { from_port: true, ..Pop::wire(1, EYE, r) };
        let out = Pop::wire(2, r, ACT);
        let g = TagGenome { tokens: vec![Token::Channel { tag: r, width: 3, hold: false }, Token::Pop(look), Token::Pop(out)], next_id: 3 };
        assert_eq!(g.ports().0, vec!["eye".to_string()]);
        let b = g.blind();
        assert_eq!(b.ports().0, vec!["M".to_string()]);
        // the channel's own reader is untouched
        assert_eq!(b.tokens[2], g.tokens[2]);
    }

    #[test]
    fn a_genome_survives_text_and_back() {
        let mut rng = StdRng::seed_from_u64(11);
        let mut gs = vec![TagGenome::context_seed(), TagGenome::act_seed()];
        let mut g = TagGenome { tokens: vec![], next_id: 1 };
        for _ in 0..300 {
            g = g.mutate(&mut rng);
        }
        gs.push(g);
        let text: String = gs.iter().map(|g| g.to_text()).collect();
        let back = TagGenome::from_text(&text).unwrap();
        assert_eq!(back, gs);
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
