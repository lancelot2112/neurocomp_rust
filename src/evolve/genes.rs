//! The genome as a list of genes: nodes and edges with permanent ids.
//!
//! The stack grammar (`GraphProgram`) cannot wire a re-entrant circuit without junk edges: every
//! edge goes out from the cursor or in from the input, the stack offers only push, dup and swap,
//! and a gate must name a node made before it. Here each node and edge has a permanent id; an
//! edge names its endpoints (and its gate) by id, so adding or removing a gene leaves the rest of
//! the wiring alone (`wiki/evolution.md`). It compiles to the same runtime graph.

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Input,
    Output,
    Hidden,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NodeGene {
    pub id: u32,
    pub role: Role,
    /// width (hidden nodes; input and output take the world's widths)
    pub bits: usize,
    /// random lines switched on each step (spontaneous activity)
    pub noise: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EdgeGene {
    pub id: u32,
    pub src: u32,
    pub dst: u32,
    /// frames back it reads (1: the frame before)
    pub lag: usize,
    /// frames it ORs together, from `lag` back
    pub span: usize,
    /// only these words of the source (offset, count)
    pub window: Option<(usize, usize)>,
    /// a wire lands its copy from this destination word
    pub dst_word: usize,
    pub kind: EdgeKind,
    pub op: KernelOp,
    /// a wire's rotation, in steps of 61 bits
    pub rotate: u16,
    pub grow: bool,
    /// kernels hold each other down where their masks overlap
    pub inhibit: bool,
    /// learns only while this node had lines on (true) or none (false)
    pub gate: Option<(u32, bool)>,
    pub new_code: bool,
    pub wide: bool,
    pub repeat: bool,
    /// frames back of the act a repeat birth copies
    pub repeat_lag: usize,
    /// random kernels it starts with
    pub kernels: u8,
}

impl EdgeGene {
    pub fn wire(id: u32, src: u32, dst: u32) -> Self {
        Self { id, src, dst, lag: 1, span: 1, window: None, dst_word: 0, kind: EdgeKind::Wire, op: KernelOp::Or, rotate: 0, grow: false, inhibit: true, gate: None, new_code: false, wide: false, repeat: false, repeat_lag: 1, kernels: 0 }
    }

    pub fn kernels(id: u32, src: u32, dst: u32, op: KernelOp) -> Self {
        Self { kind: EdgeKind::Kernels, op, ..Self::wire(id, src, dst) }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeneGenome {
    pub nodes: Vec<NodeGene>,
    pub edges: Vec<EdgeGene>,
    /// synapses of a starting kernel, its threshold in quarters, its output lines
    pub syn: u8,
    pub quarters: u8,
    pub out: u8,
    /// the starting kernels' random stream
    pub seed: u64,
    pub next_id: u32,
}

const OPS: [KernelOp; 4] = [KernelOp::Or, KernelOp::And, KernelOp::Xor, KernelOp::Clear];

fn sparse(bits: usize, k: usize, rng: &mut StdRng) -> BitVector {
    let all: Vec<usize> = (0..bits).collect();
    BitVector::from_bits(&all.choose_multiple(rng, k.min(bits)).copied().collect::<Vec<_>>(), bits)
}

impl GeneGenome {
    /// The runtime graph: the input node first, hidden nodes by id, the output node last.
    pub fn graph(&self) -> Graph<NodeSpec, EdgeSpec, Directed> {
        let mut order: Vec<&NodeGene> = self.nodes.iter().filter(|n| n.role == Role::Input).collect();
        let mut hidden: Vec<&NodeGene> = self.nodes.iter().filter(|n| n.role == Role::Hidden).collect();
        hidden.sort_by_key(|n| n.id);
        order.extend(hidden);
        order.extend(self.nodes.iter().filter(|n| n.role == Role::Output));
        let index = |id: u32| order.iter().position(|n| n.id == id);
        let mut g: Graph<NodeSpec, EdgeSpec, Directed> = Graph::new();
        let idx: Vec<_> = order
            .iter()
            .map(|n| {
                let out_reach = self.edges.iter().filter(|e| e.src == n.id).map(|e| (e.lag + e.span - 1).max(if e.repeat { e.lag + e.repeat_lag + 1 } else { 1 })).max().unwrap_or(1);
                let in_reach = self.edges.iter().filter(|e| e.dst == n.id && e.repeat).map(|e| e.repeat_lag).max().unwrap_or(1);
                let depth = out_reach.max(in_reach).max(1);
                let name = match n.role { Role::Output => Some("output".to_string()), Role::Input => Some("input".to_string()), Role::Hidden => None };
                g.add_node(NodeSpec { name, bits: n.bits, history_depth: depth, noise: n.noise })
            })
            .collect();
        for e in &self.edges {
            let (Some(a), Some(b)) = (index(e.src), index(e.dst)) else { continue };
            let gate = e.gate.and_then(|(id, on)| index(id).map(|i| (i, on)));
            g.add_edge(idx[a], idx[b], EdgeSpec {
                input_read_back: e.lag.max(1),
                threshold: 1,
                op: e.op,
                kind: e.kind,
                rotate: e.rotate,
                grow: e.grow,
                inhibit: e.inhibit,
                gate,
                new_code: e.new_code,
                wide: e.wide,
                span: e.span.max(1),
                window: e.window,
                dst_word: e.dst_word,
                repeat: e.repeat,
                repeat_lag: e.repeat_lag.max(1),
            });
        }
        g
    }

    fn hidden_ids(&self) -> Vec<u32> {
        self.nodes.iter().filter(|n| n.role == Role::Hidden).map(|n| n.id).collect()
    }

    fn any_node(&self, rng: &mut StdRng, as_dst: bool) -> u32 {
        let ids: Vec<u32> = self.nodes.iter().filter(|n| !(as_dst && n.role == Role::Input)).map(|n| n.id).collect();
        *ids.choose(rng).unwrap()
    }

    /// The context circuit, hand-written as genes (experiment 05):
    /// - Y: the senses of the last three steps, side by side (wires at lags 1-3, rotated apart);
    /// - Z: conjunction codes (kernels on all of Y, born with new codes when nothing matches);
    /// - R: Z less what the context C expects (expectation kernels clear their lines);
    /// - C: the context, held by a self-wire. Kernels on R clear it (leave: born when settled),
    ///   restore a context that expected this (recognise: born while young), or set a new code
    ///   when C is empty (enter);
    /// - N, Yg: C's onset, held 32 frames: "young", the gate for learning expectations;
    /// - Aff: the affect channel; A: the eye channel beside C;
    /// - A to the act: random babbling kernels, and kernels born when a stroke follows an act,
    ///   repeating that act on what came before it.
    pub fn context_seed(babblers: u8) -> Self {
        let ch = world::CH_BITS / 64;
        let nodes = vec![
            NodeGene { id: 0, role: Role::Input, bits: world::SENSE_BITS, noise: 0 },
            NodeGene { id: 1, role: Role::Hidden, bits: world::SENSE_BITS, noise: 0 }, // Y
            NodeGene { id: 2, role: Role::Hidden, bits: 512, noise: 0 },               // Z
            NodeGene { id: 3, role: Role::Hidden, bits: 512, noise: 0 },               // R
            NodeGene { id: 4, role: Role::Hidden, bits: 512, noise: 0 },               // C
            NodeGene { id: 5, role: Role::Hidden, bits: 512, noise: 0 },               // N
            NodeGene { id: 6, role: Role::Hidden, bits: 512, noise: 0 },               // Yg
            NodeGene { id: 7, role: Role::Hidden, bits: 512, noise: 0 },               // Aff
            NodeGene { id: 8, role: Role::Hidden, bits: 1024, noise: 0 },              // A
            NodeGene { id: 9, role: Role::Output, bits: world::ACT_BITS, noise: 0 },
            NodeGene { id: 10, role: Role::Hidden, bits: 64, noise: 8 },              // spontaneous
        ];
        let mut edges = Vec::new();
        let mut id = 100;
        let mut push = |e: EdgeGene, edges: &mut Vec<EdgeGene>| edges.push(e);
        // Y: the eye and affect channels of the last three steps, side by side (one channel
        // per eight words: step 1 eye at word 0, affect at 8; step 2 at 16 and 24; ...)
        for lag in 1..=3usize {
            for (k, chan) in [world::EYE, world::AFFECT].into_iter().enumerate() {
                let mut e = EdgeGene::wire(id, 0, 1);
                e.lag = lag;
                e.window = Some((chan * ch, ch));
                e.dst_word = ((lag - 1) * 2 + k) * ch;
                push(e, &mut edges);
                id += 1;
            }
        }
        // Z: conjunction codes
        let mut e = EdgeGene::kernels(id, 1, 2, KernelOp::Or);
        e.grow = true;
        e.new_code = true;
        e.wide = true;
        push(e, &mut edges);
        id += 1;
        // R = Z ...
        push(EdgeGene::wire(id, 2, 3), &mut edges);
        id += 1;
        // ... less what C expects (learned while young)
        let mut e = EdgeGene::kernels(id, 4, 3, KernelOp::Clear);
        e.grow = true;
        e.inhibit = false;
        e.gate = Some((6, true));
        push(e, &mut edges);
        id += 1;
        // leave: clear C on what it does not expect (learned when settled)
        let mut e = EdgeGene::kernels(id, 3, 4, KernelOp::Clear);
        e.grow = true;
        e.gate = Some((6, false));
        push(e, &mut edges);
        id += 1;
        // recognise: restore the context that expected this (learned while young)
        let mut e = EdgeGene::kernels(id, 3, 4, KernelOp::Or);
        e.grow = true;
        e.gate = Some((6, true));
        push(e, &mut edges);
        id += 1;
        // enter: a new context code when C is empty
        let mut e = EdgeGene::kernels(id, 3, 4, KernelOp::Or);
        e.grow = true;
        e.new_code = true;
        push(e, &mut edges);
        id += 1;
        // hold C
        push(EdgeGene::wire(id, 4, 4), &mut edges);
        id += 1;
        // N = C(t-1) and not C(t-2): onset; Yg = N over the last 32 frames: young
        push(EdgeGene::wire(id, 4, 5), &mut edges);
        id += 1;
        let mut e = EdgeGene::wire(id, 4, 5);
        e.lag = 2;
        e.op = KernelOp::Clear;
        push(e, &mut edges);
        id += 1;
        let mut e = EdgeGene::wire(id, 5, 6);
        e.span = 32;
        push(e, &mut edges);
        id += 1;
        // Aff: the affect channel
        let mut e = EdgeGene::wire(id, 0, 7);
        e.window = Some((world::AFFECT * ch, ch));
        push(e, &mut edges);
        id += 1;
        // A: the eye channel and C side by side
        let mut e = EdgeGene::wire(id, 0, 8);
        e.window = Some((world::EYE * ch, ch));
        push(e, &mut edges);
        id += 1;
        let mut e = EdgeGene::wire(id, 4, 8);
        e.dst_word = ch;
        push(e, &mut edges);
        id += 1;
        // acts: babbling kernels on spontaneous activity, and acts repeated where a stroke
        // followed them (born on what A held before)
        let mut e = EdgeGene::kernels(id, 10, 9, KernelOp::Or);
        e.kernels = babblers;
        push(e, &mut edges);
        id += 1;
        let mut e = EdgeGene::kernels(id, 8, 9, KernelOp::Or);
        e.grow = true;
        e.wide = true;
        e.repeat = true;
        // a click is felt as a stroke four steps later on Aff's previous frame
        e.repeat_lag = 4;
        e.gate = Some((7, true));
        push(e, &mut edges);
        id += 1;
        Self { nodes, edges, syn: 4, quarters: 2, out: 16, seed: 1, next_id: id }
    }
}

impl Evolvable for GeneGenome {
    fn build(&self, w: &World, seed: u64) -> RuntimeNetwork {
        let g = self.graph();
        let body = w.body();
        let primitives: Vec<BitVector> = (0..body.voice.len())
            .map(|i| w.act(Some(i), None))
            .chain((0..body.hand.len()).map(|i| w.act(None, Some(i))))
            .collect();
        // starting kernels per edge, in the graph's edge order (edges with a missing end are left out)
        let counts: Vec<u8> = self.edges.iter().filter(|e| self.nodes.iter().any(|n| n.id == e.src) && self.nodes.iter().any(|n| n.id == e.dst)).map(|e| e.kernels).collect();
        RuntimeNetwork::from_graph_with(&g, world::SENSE_BITS, world::ACT_BITS, seed, |e: &EdgeInfo| {
            let n = counts.get(e.index).copied().unwrap_or(0);
            let mut r = StdRng::seed_from_u64(self.seed ^ (e.index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let kernels = (0..n)
                .map(|_| {
                    let syn = sparse(64, self.syn as usize, &mut r);
                    // the words the edge actually reads (a window narrows the source)
                    let src_words = (e.src_bits / 64).max(1);
                    let read = e.spec.window.map_or(src_words, |(o, n)| n.min(src_words.saturating_sub(o)).max(1));
                    let in_word = r.gen_range(0..read);
                    let (out, out_word, op) = if e.to_output {
                        let p = primitives.choose(&mut r).unwrap();
                        let w = p.as_words().iter().position(|&x| x != 0).unwrap_or(0);
                        (BitVector::from_words(vec![p.as_words()[w]]), w, KernelOp::Or)
                    } else {
                        (sparse(64, self.out as usize, &mut r), r.gen_range(0..(e.dst_bits / 64).max(1)), e.spec.op)
                    };
                    let threshold = (self.quarters as usize * self.syn as usize).div_ceil(4).max(1);
                    SimpleKernel::new(syn, in_word, out, out_word, threshold, op)
                })
                .collect();
            KernelGroup::from_classes(vec![KernelClass::with_kernels(kernels)])
        })
    }

    fn mutate(&self, rng: &mut StdRng) -> Self {
        let mut g = self.clone();
        match rng.gen_range(0..12) {
            // a new hidden node, wired in from one node and out to another
            0 => {
                let id = g.next_id;
                g.next_id += 1;
                g.nodes.push(NodeGene { id, role: Role::Hidden, bits: 512, noise: 0 });
                let (a, b) = (g.any_node(rng, false), g.any_node(rng, true));
                g.edges.push(EdgeGene::wire(g.next_id, a, id));
                g.edges.push(EdgeGene::kernels(g.next_id + 1, id, b, KernelOp::Or));
                g.next_id += 2;
            }
            // remove a hidden node and its edges
            1 => {
                if let Some(&id) = g.hidden_ids().choose(rng) {
                    g.nodes.retain(|n| n.id != id);
                    g.edges.retain(|e| e.src != id && e.dst != id);
                    for e in &mut g.edges {
                        if e.gate.map_or(false, |(n, _)| n == id) {
                            e.gate = None;
                        }
                    }
                }
            }
            // a new edge
            2 => {
                let (a, b) = (g.any_node(rng, false), g.any_node(rng, true));
                let mut e = if rng.gen_bool(0.5) { EdgeGene::wire(g.next_id, a, b) } else { EdgeGene::kernels(g.next_id, a, b, *OPS.choose(rng).unwrap()) };
                e.op = *OPS.choose(rng).unwrap();
                g.edges.push(e);
                g.next_id += 1;
            }
            // remove an edge
            3 if !g.edges.is_empty() => {
                let i = rng.gen_range(0..g.edges.len());
                g.edges.remove(i);
            }
            // rewire one end of an edge
            4 if !g.edges.is_empty() => {
                let i = rng.gen_range(0..g.edges.len());
                if rng.gen_bool(0.5) { g.edges[i].src = g.any_node(rng, false) } else { g.edges[i].dst = g.any_node(rng, true) }
            }
            // change one setting of an edge
            5..=9 if !g.edges.is_empty() => {
                let i = rng.gen_range(0..g.edges.len());
                let hidden = g.hidden_ids();
                let e = &mut g.edges[i];
                match rng.gen_range(0..14) {
                    0 => e.op = *OPS.choose(rng).unwrap(),
                    1 => e.kind = if e.kind == EdgeKind::Wire { EdgeKind::Kernels } else { EdgeKind::Wire },
                    2 => e.lag = if rng.gen_bool(0.5) { (e.lag + 1).min(8) } else { e.lag.saturating_sub(1).max(1) },
                    3 => e.span = if rng.gen_bool(0.5) { (e.span * 2).min(64) } else { (e.span / 2).max(1) },
                    4 => e.rotate = e.rotate.wrapping_add(1) % 8,
                    5 => e.grow = !e.grow,
                    6 => e.inhibit = !e.inhibit,
                    7 => e.new_code = !e.new_code,
                    8 => e.wide = !e.wide,
                    9 => e.repeat = !e.repeat,
                    10 => e.gate = if e.gate.is_some() && rng.gen_bool(0.5) { None } else { hidden.choose(rng).map(|&n| (n, rng.gen_bool(0.5))) },
                    11 => e.kernels = if rng.gen_bool(0.5) { e.kernels.saturating_add(1).min(32) } else { e.kernels.saturating_sub(1) },
                    12 if e.repeat => e.repeat_lag = if rng.gen_bool(0.5) { (e.repeat_lag + 1).min(8) } else { e.repeat_lag.saturating_sub(1).max(1) },
                    _ => e.dst_word = (if rng.gen_bool(0.5) { e.dst_word + 1 } else { e.dst_word.saturating_sub(1) }) % 32,
                }
            }
            10 if rng.gen_bool(0.5) => g.quarters = (g.quarters as i8 + if rng.gen_bool(0.5) { 1 } else { -1 }).clamp(1, 4) as u8,
            10 => {
                let ids = g.hidden_ids();
                if let Some(&id) = ids.choose(rng) {
                    let n = g.nodes.iter_mut().find(|n| n.id == id).unwrap();
                    n.noise = if rng.gen_bool(0.5) { (n.noise * 2).max(1).min(n.bits / 2) } else { n.noise / 2 };
                }
            }
            _ => g.seed = rng.r#gen(),
        }
        g
    }

    fn size(&self) -> usize {
        self.nodes.len() + self.edges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_context_seed_compiles_with_input_first_and_output_last() {
        let g = GeneGenome::context_seed(8).graph();
        assert_eq!(g.node_count(), 11);
        assert_eq!(g.node_weight(0.into()).unwrap().name.as_deref(), Some("input"));
        assert_eq!(g.node_weight(10.into()).unwrap().name.as_deref(), Some("output"));
        // N reads C two frames back, Yg reads N over 32 frames
        assert!(g.node_weight(4.into()).unwrap().history_depth >= 2);
        assert!(g.node_weight(5.into()).unwrap().history_depth >= 32);
    }

    #[test]
    fn mutation_keeps_edges_off_the_input_node() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut g = GeneGenome::context_seed(8);
        for _ in 0..500 {
            g = g.mutate(&mut rng);
            assert!(g.edges.iter().all(|e| e.dst != 0));
            let _ = g.graph();
        }
    }
}
