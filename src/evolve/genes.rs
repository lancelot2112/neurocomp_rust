//! The genome as a list of edges.
//!
//! An edge gene names its source and destination vectors by id and carries all of its own
//! properties. There are no node genes: a vector exists because edges name it, and what it is
//! follows from them:
//! - its **width** is the widest span written into it (an edge writes from `dst_word` over its
//!   span: a wire, what it reads; kernels and noise, `out_words`);
//! - its **history depth** is the deepest frame any edge, gate or repeat reads from it.
//!
//! Vector 0 is the world's senses (nothing writes it); vector 1 is the act (the world's act
//! width). Every edge has a permanent id, which also keys its random stream, so adding or removing
//! one leaves the others alone. The genome compiles to the same runtime graph as the stack grammar.

use std::collections::{BTreeMap, BTreeSet};

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

use super::Evolvable;

/// The senses.
pub const INPUT: u32 = 0;
/// The act.
pub const OUTPUT: u32 = 1;
/// The widest a vector may grow, in words.
/// The widest vector: 2048 words (131,072 bits).
const MAX_WORDS: usize = 2048;

#[derive(Clone, Debug, PartialEq)]
pub struct EdgeGene {
    pub id: u32,
    pub src: u32,
    pub dst: u32,
    /// learns only while this vector had lines on (true) or none (false), `spec.gate_lag` back
    pub gate: Option<(u32, bool)>,
    /// everything else this edge is (its gate field is filled from `gate` when compiled)
    pub spec: EdgeSpec,
}

impl EdgeGene {
    pub fn new(id: u32, src: u32, dst: u32, kind: EdgeKind, op: KernelOp) -> Self {
        Self { id, src, dst, gate: None, spec: EdgeSpec { kind, op, salt: id, ..EdgeSpec::plain() } }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeneGenome {
    pub edges: Vec<EdgeGene>,
    /// the next permanent id (for edges and for new vectors)
    pub next_id: u32,
}


fn sparse(bits: usize, k: usize, rng: &mut StdRng) -> BitVector {
    let all: Vec<usize> = (0..bits).collect();
    BitVector::from_bits(&all.choose_multiple(rng, k.min(bits)).copied().collect::<Vec<_>>(), bits)
}

impl GeneGenome {
    /// Every vector the edges name, the senses and the act included.
    pub fn vectors(&self) -> BTreeSet<u32> {
        let mut v: BTreeSet<u32> = [INPUT, OUTPUT].into_iter().collect();
        for e in &self.edges {
            v.insert(e.src);
            v.insert(e.dst);
            if let Some((g, _)) = e.gate {
                v.insert(g);
            }
        }
        v
    }

    /// Each vector's width in words, from the edges that write it (a fixed point: a wire spans
    /// what it reads, which may itself be derived).
    pub fn widths(&self) -> BTreeMap<u32, usize> {
        let mut w: BTreeMap<u32, usize> = self.vectors().into_iter().map(|v| (v, 1)).collect();
        w.insert(INPUT, world::SENSE_BITS / 64);
        w.insert(OUTPUT, world::ACT_BITS / 64);
        for _ in 0..64 {
            let mut changed = false;
            for e in &self.edges {
                if e.dst == INPUT || e.dst == OUTPUT {
                    continue;
                }
                let span = match e.spec.kind {
                    EdgeKind::Wire => {
                        let src = w[&e.src];
                        e.spec.window.map_or(src, |(o, n)| { let o = o % src.max(1); n.max(1).min(src - o) })
                    }
                    _ => (e.spec.out_words as usize).max(1),
                };
                let need = (e.spec.dst_word + span).min(MAX_WORDS);
                if need > w[&e.dst] {
                    w.insert(e.dst, need);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        w
    }

    /// Each vector's history depth: the deepest frame read from it.
    pub fn depths(&self) -> BTreeMap<u32, usize> {
        let mut d: BTreeMap<u32, usize> = self.vectors().into_iter().map(|v| (v, 1)).collect();
        let mut need = |v: u32, k: usize| {
            let e = d.get_mut(&v).unwrap();
            *e = (*e).max(k);
        };
        for e in &self.edges {
            let s = &e.spec;
            need(e.src, s.input_read_back.max(1) + s.span.max(1) - 1);
            if s.repeat {
                need(e.src, s.input_read_back.max(1) + s.repeat_lag + 1);
                need(e.dst, s.repeat_lag);
            }
            if let Some((g, _)) = e.gate {
                need(g, s.gate_lag.max(1));
            }
        }
        d
    }

    /// The runtime graph: the senses first, the other vectors by id, the act last.
    pub fn graph(&self) -> Graph<NodeSpec, EdgeSpec, Directed> {
        let widths = self.widths();
        let depths = self.depths();
        let mut order: Vec<u32> = vec![INPUT];
        order.extend(self.vectors().into_iter().filter(|&v| v != INPUT && v != OUTPUT));
        order.push(OUTPUT);
        let index = |id: u32| order.iter().position(|&v| v == id).unwrap();
        let mut g: Graph<NodeSpec, EdgeSpec, Directed> = Graph::new();
        let idx: Vec<_> = order
            .iter()
            .map(|&v| {
                let name = match v { INPUT => Some("input".to_string()), OUTPUT => Some("output".to_string()), _ => None };
                g.add_node(NodeSpec { name, bits: widths[&v] * 64, history_depth: depths[&v], noise: 0 })
            })
            .collect();
        for e in &self.edges {
            if e.dst == INPUT {
                continue;
            }
            let mut spec = e.spec.clone();
            spec.gate = e.gate.map(|(v, on)| (index(v), on));
            spec.salt = e.id;
            g.add_edge(idx[index(e.src)], idx[index(e.dst)], spec);
        }
        g
    }

    fn any_vector(&self, rng: &mut StdRng, as_dst: bool) -> u32 {
        let v: Vec<u32> = self.vectors().into_iter().filter(|&v| !(as_dst && v == INPUT)).collect();
        *v.choose(rng).unwrap()
    }

    /// The context circuit, hand-written as edges (experiment 05). The vectors:
    /// - 2, Y: eye and affect of the last three steps, side by side;
    /// - 3, Z: conjunction codes on Y (born with new codes when nothing matches);
    /// - 4, R: Z less what the context expects;
    /// - 5, C: the context, held by a self-wire. Kernels on R clear it (leave: learned when
    ///   settled), restore a context that expected this (recognise: learned while young), or set
    ///   a new code when C is empty (enter);
    /// - 6, N and 7, Yg: C's onset, held 32 frames ("young");
    /// - 8, Aff: the affect channel;
    /// - 9, A: the eye channel and C side by side;
    /// - 10: spontaneous activity, driving babbling kernels into the act; acts a stroke followed
    ///   are repeated on what A held before them, and fade when no stroke follows.
    pub fn context_seed(babblers: u8) -> Self {
        let ch = world::CH_BITS / 64;
        let (y, z, r, c, n, yg, aff, a, spont) = (2, 3, 4, 5, 6, 7, 8, 9, 10);
        let mut edges: Vec<EdgeGene> = Vec::new();
        let mut id = 100;
        let mut add = |src: u32, dst: u32, kind: EdgeKind, op: KernelOp, f: &dyn Fn(&mut EdgeGene)| {
            let mut e = EdgeGene::new(id, src, dst, kind, op);
            f(&mut e);
            edges.push(e);
            id += 1;
        };
        use EdgeKind::{Kernels, Noise, Wire};
        use KernelOp::{Clear, Or};
        for lag in 1..=3usize {
            for (k, chan) in [world::EYE, world::AFFECT].into_iter().enumerate() {
                add(INPUT, y, Wire, Or, &|e| {
                    e.spec.input_read_back = lag;
                    e.spec.window = Some((chan * ch, ch));
                    e.spec.dst_word = ((lag - 1) * 2 + k) * ch;
                });
            }
        }
        add(y, z, Kernels, Or, &|e| { e.spec.grow = true; e.spec.new_code = true; e.spec.wide = true; e.spec.out_words = ch as u16; });
        add(z, r, Wire, Or, &|_| {});
        add(c, r, Kernels, Clear, &|e| { e.spec.grow = true; e.spec.inhibit = false; e.gate = Some((yg, true)); });
        add(r, c, Kernels, Clear, &|e| { e.spec.grow = true; e.gate = Some((yg, false)); e.spec.out_words = ch as u16; });
        add(r, c, Kernels, Or, &|e| { e.spec.grow = true; e.gate = Some((yg, true)); });
        add(r, c, Kernels, Or, &|e| { e.spec.grow = true; e.spec.new_code = true; });
        add(c, c, Wire, Or, &|_| {});
        add(c, n, Wire, Or, &|_| {});
        add(c, n, Wire, Clear, &|e| e.spec.input_read_back = 2);
        add(n, yg, Wire, Or, &|e| e.spec.span = 32);
        add(INPUT, aff, Wire, Or, &|e| e.spec.window = Some((world::AFFECT * ch, ch)));
        add(INPUT, a, Wire, Or, &|e| e.spec.window = Some((world::EYE * ch, ch)));
        add(c, a, Wire, Or, &|e| e.spec.dst_word = ch);
        add(spont, spont, Noise, Or, &|e| { e.spec.noise = 8; e.spec.out_words = 1; });
        add(spont, OUTPUT, Kernels, Or, &|e| e.spec.start_kernels = babblers);
        add(a, OUTPUT, Kernels, Or, &|e| {
            e.spec.grow = true;
            e.spec.wide = true;
            e.spec.repeat = true;
            e.spec.repeat_lag = 4;
            e.gate = Some((aff, true));
        });
        Self { edges, next_id: id }
    }

    /// Only the act path of the context seed: spontaneous babbling, and acts repeated where a
    /// stroke followed them (on the eye channel), fading where none did.
    pub fn act_seed(babblers: u8) -> Self {
        let mut g = Self::context_seed(babblers);
        g.edges.retain(|e| matches!((e.src, e.dst), (INPUT, 8) | (INPUT, 9) | (10, _) | (9, OUTPUT)));
        g
    }
}

/// An edge's starting kernels, made from its own properties and its own random stream: each
/// reads one word of what the edge reads and writes one word of its span (into the act: one of
/// the body's motor primitives).
pub fn start_kernels(e: &EdgeInfo, primitives: &[BitVector]) -> KernelGroup {
    let sp = e.spec;
    let mut r = StdRng::seed_from_u64((sp.salt as u64 + 1).wrapping_mul(0x2545_F491_4F6C_DD1D));
    let dst_words = (e.dst_bits / 64).max(1);
    let from = sp.dst_word.min(dst_words - 1);
    let span = if sp.out_words == 0 { dst_words - from } else { (sp.out_words as usize).min(dst_words - from) };
    let kernels = (0..sp.start_kernels)
        .map(|_| {
            let syn = sparse(64, sp.start_syn as usize, &mut r);
            let src_words = (e.src_bits / 64).max(1);
            let read = sp.window.map_or(src_words, |(o, n)| { let o = o % src_words; n.max(1).min(src_words - o) });
            let in_word = r.gen_range(0..read);
            let (out, out_word, op) = if e.to_output && !primitives.is_empty() {
                let p = primitives.choose(&mut r).unwrap();
                let w = p.as_words().iter().position(|&x| x != 0).unwrap_or(0);
                (BitVector::from_words(vec![p.as_words()[w]]), w, KernelOp::Or)
            } else {
                (sparse(64, sp.start_out as usize, &mut r), from + r.gen_range(0..span), sp.op)
            };
            let threshold = (sp.start_quarters as usize * sp.start_syn as usize).div_ceil(4).max(1);
            SimpleKernel::new(syn, in_word, out, out_word, threshold, op)
        })
        .collect();
    KernelGroup::from_classes(vec![KernelClass::with_kernels(kernels)])
}

impl Evolvable for GeneGenome {
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
        let fresh = |g: &mut GeneGenome| {
            g.next_id += 1;
            g.next_id - 1
        };
        match rng.gen_range(0..10) {
            // a new edge (its source or destination may be a new vector)
            0 => {
                let src = if rng.gen_range(0..4) == 0 { fresh(&mut g) } else { g.any_vector(rng, false) };
                let dst = if rng.gen_range(0..4) == 0 { fresh(&mut g) } else { g.any_vector(rng, true) };
                let kind = *[EdgeKind::Wire, EdgeKind::Kernels, EdgeKind::Kernels, EdgeKind::Noise].choose(rng).unwrap();
                let id = fresh(&mut g);
                let op = if rng.gen_bool(0.5) { KernelOp::Or } else { KernelOp::random(rng) };
                let mut e = EdgeGene::new(id, src, dst, kind, op);
                // born active: some kernels, a random fan-in, sometimes tonic or growing
                e.spec.start_kernels = rng.gen_range(1..=8);
                e.spec.start_syn = 1 << rng.gen_range(2..5);
                e.spec.tonic = if rng.gen_range(0..3) == 0 { rng.gen_range(1..5) } else { 0 };
                e.spec.grow = rng.gen_bool(0.3);
                e.spec.wide = rng.gen_bool(0.5);
                if rng.gen_bool(0.5) {
                    e.spec.window = Some((rng.gen_range(0..32), 1 << rng.gen_range(0..4)));
                }
                if kind == EdgeKind::Noise {
                    e.spec.noise = 4;
                    e.spec.out_words = 1;
                }
                g.edges.push(e);
            }
            // a copy of an edge (a new id: its own stream)
            1 if !g.edges.is_empty() => {
                let mut e = g.edges.choose(rng).unwrap().clone();
                e.id = fresh(&mut g);
                g.edges.push(e);
            }
            // a group copied: every edge touching one vector, onto a fresh vector (a module
            // duplicated; its outside connections shared, the vector its own)
            9 if !g.edges.is_empty() => {
                let vs: Vec<u32> = g.vectors().into_iter().filter(|&v| v != INPUT && v != OUTPUT).collect();
                if let Some(&v) = vs.choose(rng) {
                    let nv = fresh(&mut g);
                    let group: Vec<EdgeGene> = g.edges.iter().filter(|e| e.src == v || e.dst == v).cloned().collect();
                    for mut e in group {
                        e.id = fresh(&mut g);
                        if e.src == v { e.src = nv }
                        if e.dst == v { e.dst = nv }
                        g.edges.push(e);
                    }
                }
            }
            // an edge removed
            2 if !g.edges.is_empty() => {
                let i = rng.gen_range(0..g.edges.len());
                g.edges.remove(i);
            }
            // one end of an edge moved (possibly to a new vector)
            3 if !g.edges.is_empty() => {
                let i = rng.gen_range(0..g.edges.len());
                let v = if rng.gen_range(0..4) == 0 { fresh(&mut g) } else { g.any_vector(rng, false) };
                if rng.gen_bool(0.5) { g.edges[i].src = v } else if v != INPUT { g.edges[i].dst = v }
            }
            // one property of an edge changed
            _ if !g.edges.is_empty() => {
                let vectors: Vec<u32> = g.vectors().into_iter().collect();
                let i = rng.gen_range(0..g.edges.len());
                let e = &mut g.edges[i];
                let up = rng.gen_bool(0.5);
                let step = |v: u8, lo: u8, hi: u8| if up { v.saturating_add(1).min(hi) } else { v.saturating_sub(1).max(lo) };
                let stepu = |v: usize, lo: usize, hi: usize| if up { (v + 1).min(hi) } else { v.saturating_sub(1).max(lo) };
                let which = rng.gen_range(0..31);
                if which == 11 {
                    e.gate = if e.gate.is_some() && rng.gen_bool(0.5) { None } else { vectors.choose(rng).map(|&v| (v, rng.gen_bool(0.5))) };
                }
                let s = &mut e.spec;
                match which {
                    0 => s.op = if rng.gen_bool(0.5) { KernelOp::random(rng) } else { s.op.nudge(up) },
                    1 => s.kind = *[EdgeKind::Wire, EdgeKind::Kernels, EdgeKind::Noise].choose(rng).unwrap(),
                    2 => s.input_read_back = stepu(s.input_read_back, 1, 8),
                    3 => s.span = if up { (s.span * 2).min(64) } else { (s.span / 2).max(1) },
                    4 => s.rotate = if up { s.rotate.wrapping_add(61) } else { s.rotate.wrapping_sub(61) },
                    5 => s.grow = !s.grow,
                    6 => s.inhibit = !s.inhibit,
                    7 => s.new_code = !s.new_code,
                    8 => s.wide = !s.wide,
                    9 => s.repeat = !s.repeat,
                    10 => s.repeat_lag = stepu(s.repeat_lag, 1, 8),
                    11 => {}
                    12 => s.gate_lag = stepu(s.gate_lag, 1, 8),
                    13 => s.window = if s.window.is_some() && rng.gen_bool(0.3) { None } else { let (o, n) = s.window.unwrap_or((0, 8)); Some(if up { (o + 1, n) } else { (o.saturating_sub(1), n) }) },
                    14 => s.window = s.window.map(|(o, n)| (o, if up { (n * 2).min(MAX_WORDS) } else { (n / 2).max(1) })),
                    15 => s.dst_word = stepu(s.dst_word, 0, MAX_WORDS - 1),
                    16 => s.out_words = if up { (s.out_words * 2).max(1).min(MAX_WORDS as u16) } else { s.out_words / 2 },
                    17 => s.noise = if up { (s.noise * 2).max(1).min(64) } else { s.noise / 2 },
                    18 => s.start_kernels = step(s.start_kernels, 0, 32),
                    19 => s.start_syn = step(s.start_syn, 1, 64),
                    20 => s.start_quarters = step(s.start_quarters, 1, 4),
                    21 => s.start_out = step(s.start_out, 1, 64),
                    22 => s.birth_syn = step(s.birth_syn, 1, 64),
                    23 => s.birth_quarters = step(s.birth_quarters, 1, 4),
                    24 => s.out_min = step(s.out_min, 1, 64),
                    25 => s.out_max = step(s.out_max, 1, 64),
                    26 => s.code_lines = step(s.code_lines, 1, 64),
                    27 => s.out_learn = step(s.out_learn, 0, 8),
                    28 => s.fade = step(s.fade, 0, 8),
                    29 => s.tonic = step(s.tonic, 0, 12),
                    _ => s.cap = if up { s.cap.saturating_mul(2).min(4096) } else { (s.cap / 2).max(1) },
                }
            }
            _ => {}
        }
        g
    }

    fn size(&self) -> usize {
        self.edges.len()
    }

    /// Each edge (by id) from one parent or the other; an edge only one parent has, half the time.
    fn crossover(&self, other: &Self, rng: &mut StdRng) -> Self {
        let mut edges: Vec<EdgeGene> = Vec::new();
        for e in &self.edges {
            match other.edges.iter().find(|o| o.id == e.id) {
                Some(o) => edges.push(if rng.gen_bool(0.5) { e.clone() } else { o.clone() }),
                None => if rng.gen_bool(0.5) { edges.push(e.clone()) },
            }
        }
        for o in &other.edges {
            if !self.edges.iter().any(|e| e.id == o.id) && rng.gen_bool(0.5) {
                edges.push(o.clone());
            }
        }
        GeneGenome { edges, next_id: self.next_id.max(other.next_id) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectors_take_their_shape_from_the_edges() {
        let g = GeneGenome::context_seed(8);
        let w = g.widths();
        // Y holds six channel windows side by side: 48 words
        assert_eq!(w[&2], 48);
        // A holds the eye beside the context: 8 + 8 words
        assert_eq!(w[&9], 16);
        // the spontaneous vector is one word
        assert_eq!(w[&10], 1);
        let d = g.depths();
        // N reads C two frames back; Yg reads N over 32 frames; the senses are read 3 back
        assert!(d[&5] >= 2 && d[&6] >= 32 && d[&INPUT] >= 3);
        let graph = g.graph();
        assert_eq!(graph.node_weight(0.into()).unwrap().name.as_deref(), Some("input"));
        assert_eq!(graph.node_weight(petgraph::graph::NodeIndex::new(graph.node_count() - 1)).unwrap().name.as_deref(), Some("output"));
    }

    #[test]
    fn every_mutant_lives() {
        use crate::evolve::{live, Physics, Task};
        use crate::world::Conditioning;
        let mut rng = StdRng::seed_from_u64(11);
        let mut g = GeneGenome::context_seed(8);
        for i in 0..300 {
            g = g.mutate(&mut rng);
            if i % 10 == 0 {
                let _ = live(&g, Task::Conditioning(Conditioning::Reversal, 8), Physics::Varied, i);
            }
        }
    }

    #[test]
    fn mutation_never_writes_the_senses_and_always_compiles() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut g = GeneGenome::context_seed(8);
        for _ in 0..2000 {
            g = g.mutate(&mut rng);
            assert!(g.edges.iter().all(|e| e.dst != INPUT));
            let w = g.widths();
            assert!(w.values().all(|&x| (1..=MAX_WORDS).contains(&x)));
            let _ = g.graph();
        }
    }
}
