use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Directed;

use crate::bitvec::{ones_of, select_nth_bit, BitVector};
use crate::bitvec::bithistory::{AdvanceMode, BitVecHistory};
use crate::kernel::class::KernelOp;
use crate::kernel::group::KernelGroup;
use crate::kernel::SimpleKernel;
use crate::program::graph::{EdgeKind, EdgeSpec, GraphProgram, NodeSpec, ProgramDefaults};
use rand::seq::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::Rng;

/// A runnable network: node histories plus per-edge kernel groups.
/// For now, each edge also carries a simple default kernel used for execution
/// until KernelClass execution is wired in.
pub struct RuntimeNetwork {
    pub nodes: Vec<BitVecHistory>,
    pub edges: Vec<RuntimeEdgeGroup>,
    /// each node's own random stream (its spontaneous activity)
    pub node_rngs: Vec<StdRng>,
    /// the act node
    pub output: usize,
    /// spontaneous lines per step, per node
    pub noise: Vec<usize>,
    /// per node: it holds its lines from step to step instead of starting empty
    pub hold: Vec<bool>,
}

/// What an edge reads: frames `read_back .. read_back + span` of a node ORed together, only the
/// words of its window (the offset wraps; at least one word). Costs the window, not the vector.
fn view(node: &BitVecHistory, read_back: usize, span: usize, window: Option<(usize, usize)>) -> BitVector {
    let first = node.get_frame(read_back);
    let n = first.word_len().max(1);
    let (off, len) = match window {
        Some((o, k)) => { let o = o % n; (o, k.max(1).min(n - o)) }
        None => (0, n),
    };
    let mut w: Vec<u64> = first.as_words()[off..off + len].to_vec();
    for f in 1..span.max(1) {
        for (a, b) in w.iter_mut().zip(&node.get_frame(read_back + f).as_words()[off..off + len]) {
            *a |= b;
        }
    }
    BitVector::from_words(w)
}

/// The word with the most lines on, and how many.
fn busiest_word(words: &[u64]) -> (usize, u32) {
    words.iter().enumerate().map(|(i, w)| (i, w.count_ones())).max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))).unwrap_or((0, 0))
}

fn apply(d: &mut [u64], t: &[u64], op: KernelOp) {
    for (a, b) in d.iter_mut().zip(t) {
        *a = op.word(*a, *b);
    }
}

/// What an edge's kernel factory is told about the edge it fills.
pub struct EdgeInfo<'a> {
    pub index: usize,
    pub src: usize,
    pub dst: usize,
    pub src_bits: usize,
    pub dst_bits: usize,
    /// the destination is the output node
    pub to_output: bool,
    pub spec: &'a EdgeSpec,
    /// the lifetime's seed (so starting cells can differ from lifetime to lifetime)
    pub seed: u64,
}

/// Per-edge runtime unit: group of kernel classes plus a default kernel.
pub struct RuntimeEdgeTarget {
    pub idx: usize, // target node index
    pub read_back: usize, // how many frames back to read from
    pub span: usize,      // frames read (ORed), from read_back back
}
pub struct RuntimeEdgeGroup {
    pub src: RuntimeEdgeTarget,
    pub dst: usize,
    pub group: KernelGroup,
    pub kind: EdgeKind,
    pub op: KernelOp,
    pub rotate: u16,
    pub grow: bool,
    /// kernels hold each other down where their masks overlap
    pub inhibit: bool,
    /// it learns only while this node had lines on (true) or none (false) a frame ago
    pub gate: Option<(usize, bool)>,
    /// births into an empty destination write a fresh random code
    pub new_code: bool,
    /// births sample synapses across the whole source
    pub wide: bool,
    /// read only these source words
    pub window: Option<(usize, usize)>,
    /// a wire lands its copy from this destination word
    pub dst_word: usize,
    /// births write what the destination held a frame ago
    pub repeat: bool,
    pub repeat_lag: usize,
    /// all of this edge's own parameters (births, rates, its gate's lag, ...)
    pub spec: EdgeSpec,
    /// this edge's own random stream
    pub rng: StdRng,
    /// births are allowed (not on edges into the act node, whose kernels write the body's
    /// primitives)
    pub can_grow: bool,
}

impl RuntimeEdgeGroup {
    /// Kernels on this edge.
    pub fn group_len(&self) -> usize {
        self.group.kernel_count()
    }
}

impl RuntimeNetwork {
    /// Build a runnable network from a GraphProgram.
    pub fn from_program(prog: &GraphProgram, defaults: &ProgramDefaults) -> Self {
        let g = prog.build_graph(defaults);
        Self::from_graph(&g)
    }

    /// Build a runnable network from a Graph<NodeSpec, EdgeSpec>.
    pub fn from_graph(g: &Graph<NodeSpec, EdgeSpec, Directed>) -> Self {
        // Materialize node histories
        let mut nodes: Vec<BitVecHistory> = Vec::with_capacity(g.node_count());
        for ni in g.node_indices() {
            let ns = g.node_weight(ni).expect("node weight");
            let bits = ns.bits;
            // one more frame than the deepest read, so a read never lands on the frame being
            // written
            let ring_len = ns.history_depth.max(1) + 1;
            let init = BitVector::new(bits, Some(0));
            nodes.push(BitVecHistory::new(init, ring_len));
        }

        // Compile edges to runtime edge groups
        let mut edges: Vec<RuntimeEdgeGroup> = Vec::with_capacity(g.edge_count());
        for e in g.edge_references() {
            let src = e.source().index();
            let dst = e.target().index();
            let es = e.weight();

            //let src_bits = g.node_weight(NodeIndex::new(src)).unwrap();
            //let dst_bits = g.node_weight(NodeIndex::new(dst)).unwrap();

            //let input_mask = ones_mask(src_bits);
            //let output_mask = ones_mask(dst_bits);

            edges.push(RuntimeEdgeGroup {
                src: RuntimeEdgeTarget {
                    idx: src,
                    read_back: es.input_read_back,
                    span: es.span.max(1),
                },
                dst,
                group: KernelGroup::default(), // placeholder; later: pick group per edge
                kind: es.kind,
                op: es.op,
                rotate: es.rotate,
                grow: es.grow,
                inhibit: es.inhibit,
                gate: es.gate,
                new_code: es.new_code,
                wide: es.wide,
                window: es.window,
                dst_word: es.dst_word,
                repeat: es.repeat,
                repeat_lag: es.repeat_lag,
                spec: es.clone(),
                rng: StdRng::seed_from_u64((es.salt as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                can_grow: true,
            });
        }

        let output = g.node_indices().find(|n| g.node_weight(*n).unwrap().name.as_deref() == Some("output")).map_or(g.node_count() - 1, |n| n.index());
        let noise = g.node_indices().map(|n| g.node_weight(n).unwrap().noise).collect();
        let node_rngs = (0..g.node_count()).map(|i| StdRng::seed_from_u64(i as u64 ^ 0xA0A0)).collect();
        let hold = g.node_indices().map(|n| g.node_weight(n).unwrap().hold).collect();
        Self { nodes, edges, node_rngs, output, noise, hold }
    }

    /// Build from a graph with the input and output nodes resized (to a world's senses and
    /// acts) and each edge's kernels made by `factory`; the network's random stream is seeded
    /// with `seed`.
    pub fn from_graph_with<F>(g: &Graph<NodeSpec, EdgeSpec, Directed>, in_bits: usize, out_bits: usize, seed: u64, mut factory: F) -> Self
    where
        F: FnMut(&EdgeInfo) -> KernelGroup,
    {
        // the act node: the one named "output" (else the last)
        let last = g.node_indices().find(|n| g.node_weight(*n).unwrap().name.as_deref() == Some("output")).map_or(g.node_count() - 1, |n| n.index());
        let bits_of = |i: usize| if i == 0 { in_bits } else if i == last { out_bits } else { g.node_weight(NodeIndex::new(i)).unwrap().bits };
        let mut nodes: Vec<BitVecHistory> = Vec::with_capacity(g.node_count());
        for ni in g.node_indices() {
            let ns = g.node_weight(ni).expect("node weight");
            let ring_len = ns.history_depth.max(1) + 1;
            nodes.push(BitVecHistory::new(BitVector::new(bits_of(ni.index()), Some(0)), ring_len));
        }
        let mut edges: Vec<RuntimeEdgeGroup> = Vec::with_capacity(g.edge_count());
        for (index, e) in g.edge_references().enumerate() {
            let (src, dst) = (e.source().index(), e.target().index());
            let info = EdgeInfo { index, src, dst, src_bits: bits_of(src), dst_bits: bits_of(dst), to_output: dst == last, spec: e.weight(), seed };
            edges.push(RuntimeEdgeGroup {
                src: RuntimeEdgeTarget { idx: src, read_back: e.weight().input_read_back.max(1), span: e.weight().span.max(1) },
                dst,
                group: factory(&info),
                kind: e.weight().kind,
                op: e.weight().op,
                rotate: e.weight().rotate,
                grow: e.weight().grow,
                inhibit: e.weight().inhibit,
                gate: e.weight().gate,
                new_code: e.weight().new_code,
                wide: e.weight().wide,
                window: e.weight().window,
                dst_word: e.weight().dst_word,
                repeat: e.weight().repeat,
                repeat_lag: e.weight().repeat_lag.max(1),
                spec: e.weight().clone(),
                rng: StdRng::seed_from_u64(seed ^ (e.weight().salt as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                can_grow: dst != last || e.weight().repeat,
            });
        }
        let noise = g.node_indices().map(|n| g.node_weight(n).unwrap().noise).collect();
        let node_rngs = (0..g.node_count()).map(|i| StdRng::seed_from_u64(seed ^ (i as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03))).collect();
        let hold = g.node_indices().map(|n| g.node_weight(n).unwrap().hold).collect();
        Self { nodes, edges, node_rngs, output: last, noise, hold }
    }

    /// Advance histories and process all edges once.
    pub fn tick(&mut self, mode: AdvanceMode, adj_temperature: i16) {
        // Advance all nodes first so writes go into a fresh current frame
        // (a node that holds starts from its last frame: only clearing writes take lines away)
        for (n, &hold) in self.nodes.iter_mut().zip(&self.hold) {
            n.advance(if hold && matches!(mode, AdvanceMode::Clear) { AdvanceMode::CopyPrev } else { mode });
        }

        // Execute edges in their order (the genome's): writes into a vector land in that order,
        // whatever their ops. Each edge reads its source as it was `read_back` (at least 1)
        // frames ago, never the frame being written, so edges into different vectors do not
        // depend on each other.
        let Self { nodes, edges, node_rngs, noise, .. } = self;
        // spontaneous activity: random lines on, in the nodes that have it (each its own stream)
        for (i, &k) in noise.iter().enumerate() {
            let bits = nodes[i].current().bit_len();
            for _ in 0..k {
                let b = node_rngs[i].gen_range(0..bits.max(1));
                nodes[i].current_mut().bit_set(b);
            }
        }
        let mut fired: Vec<Vec<usize>> = vec![Vec::new(); edges.len()];
        let mut seen: Vec<Option<BitVector>> = vec![None; edges.len()];
        {
            for (i, eg) in edges.iter_mut().enumerate() {
                let mut src_view = view(&nodes[eg.src.idx], eg.src.read_back, eg.src.span, eg.window);
                let dst_curr = nodes[eg.dst].current_mut();
                match eg.kind {
                    EdgeKind::Noise => {
                        // spontaneous lines in the destination's span, each touching one word
                        let dw = dst_curr.word_len().max(1);
                        let from = eg.spec.dst_word % dw;
                        let n = if eg.spec.out_words == 0 { dw - from } else { (eg.spec.out_words as usize).min(dw - from) };
                        for _ in 0..eg.spec.noise {
                            let b = from * 64 + eg.rng.gen_range(0..n * 64);
                            let d = &mut dst_curr.as_words_mut()[b / 64];
                            *d = eg.op.word(*d, 1u64 << (b % 64));
                        }
                    }
                    EdgeKind::Wire => {
                        if eg.rotate > 0 {
                            let n = src_view.bit_len();
                            src_view.rotr_mut(eg.rotate as usize % n.max(1));
                        }
                        // the copy lands from `dst_word`, wrapping round the destination; the op
                        // touches only the words it lands on (a wider source folds in by OR)
                        let dw = dst_curr.word_len().max(1);
                        let src = src_view.as_words();
                        if src.len() <= dw {
                            let d = dst_curr.as_words_mut();
                            for (j, w) in src.iter().enumerate() {
                                let k = (j + eg.dst_word) % dw;
                                d[k] = eg.op.word(d[k], *w);
                            }
                        } else {
                            let mut t = vec![0u64; dw];
                            for (j, w) in src.iter().enumerate() {
                                t[(j + eg.dst_word) % dw] |= w;
                            }
                            apply(dst_curr.as_words_mut(), &t, eg.op);
                        }
                    }
                    EdgeKind::Kernels => {
                        fired[i] = eg.group.process(&src_view, dst_curr, 0, adj_temperature, eg.inhibit, &mut eg.rng).into_iter().flatten().collect();
                        seen[i] = Some(src_view);
                    }
                }
            }
        }
        // After the tick: cancelling kernels learn their outputs, and growing edges give birth
        // where their destination was left uncovered
        for (i, eg) in edges.iter_mut().enumerate() {
            let sp = eg.spec.clone();
            let rng = &mut eg.rng;
            if eg.kind != EdgeKind::Kernels {
                continue;
            }
            // kernels count the steps since they fired
            if let Some(class) = eg.group.classes_mut().first_mut() {
                for k in class.kernels_mut() {
                    if !fired[i].is_empty() || k.age < u8::MAX {
                        k.age = k.age.saturating_add(1);
                    }
                }
                for &k in &fired[i] {
                    class.kernels_mut()[k].age = 0;
                }
            }
            // a gate: it learns only while the gate node had lines on (or none) a frame ago
            if let Some((g, on)) = eg.gate {
                let open = (nodes[g].get_frame(sp.gate_lag.max(1)).count_ones() > 0) == on;
                // extinction on a repeat edge: a kernel that acted `repeat_lag` steps ago, with
                // the gate still shut, fades (chance 1/4)
                if !open && eg.repeat {
                    let then = view(&nodes[eg.src.idx], eg.src.read_back + eg.repeat_lag, 1, eg.window);
                    if let Some(class) = eg.group.classes_mut().first_mut() {
                        for k in class.kernels_mut() {
                            if k.age as usize == eg.repeat_lag && rng.gen_range(0..1u32 << sp.fade.min(31)) == 0 {
                                k.fade(&then, rng);
                            }
                        }
                    }
                }
                if !open {
                    continue;
                }
            }
            let Some(class) = eg.group.classes_mut().first_mut() else { continue };
            if matches!(eg.op, KernelOp::Xor | KernelOp::Clear) {
                let dest = nodes[eg.dst].current();
                for &k in &fired[i] {
                    class.kernels_mut()[k].learn_output(dest, sp.out_learn, rng);
                }
            }
            if !(eg.grow && eg.can_grow) || class.len() >= sp.cap as usize {
                continue;
            }
            // a repeat birth reads its source as it was when the repeated act was made
            let older;
            let src = if eg.repeat && eg.repeat_lag > 1 {
                older = view(&nodes[eg.src.idx], eg.src.read_back + eg.repeat_lag, 1, eg.window);
                &older
            } else {
                seen[i].as_ref().unwrap()
            };
            let ones = ones_of;
            let dest = nodes[eg.dst].current();
            if sp.tile > 0 {
                // full coverage: a kernel is born in the tile (2^tile bits, every half tile) with
                // the most active input lines no firing kernel reads, on those lines, writing at
                // the same relative place in the destination
                let sw = src.as_words();
                let mut covered = vec![0u64; sw.len()];
                for &k in &fired[i] {
                    let kk = &class.kernels()[k];
                    for (j, m) in kk.input_mask.as_words().iter().enumerate() {
                        if let Some(c) = covered.get_mut(kk.input_idx + j) { *c |= m; }
                    }
                }
                let bits = src.bit_len();
                let t = (1usize << sp.tile.min(20)).min(bits).max(1);
                let stride = (t / 2).max(1);
                let tiles = if t >= bits { 1 } else { (bits - t) / stride + 1 };
                let count = |s: usize| -> u32 { (s / 64..=(s + t - 1) / 64).map(|w| (sw[w] & !covered[w] & SimpleKernel::tile_word(s, t, w)).count_ones()).sum() };
                let (start, n) = (0..tiles).map(|j| (j * stride, count(j * stride))).max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))).unwrap_or((0, 0));
                if n == 0 {
                    continue;
                }
                let lines: Vec<usize> = (start / 64..=(start + t - 1) / 64)
                    .flat_map(|w| ones(sw[w] & !covered[w] & SimpleKernel::tile_word(start, t, w)).into_iter().map(move |b| w * 64 + b))
                    .collect();
                let syn: Vec<usize> = lines.choose_multiple(rng, (sp.birth_syn as usize).min(lines.len())).copied().collect();
                let threshold = (sp.birth_quarters as usize * syn.len()).div_ceil(4).max(1);
                let (out, dw) = if eg.repeat {
                    let prev = nodes[eg.dst].get_frame(eg.repeat_lag);
                    let (dw, dn) = busiest_word(prev.as_words());
                    if (dn as usize) < sp.out_min as usize {
                        continue;
                    }
                    (ones(prev.as_words()[dw]), dw)
                } else {
                    let dwords = dest.word_len().max(1);
                    let dw = (start * dwords / sw.len().max(1)).min(dwords - 1);
                    let wrote = fired[i].iter().map(|&k| &class.kernels()[k]).filter(|kk| kk.output_idx == dw).fold(0u64, |a, kk| a | kk.output_mask.as_words()[0]);
                    let left = ones(dest.as_words()[dw] & !wrote);
                    if left.len() >= sp.out_min as usize {
                        (left.choose_multiple(rng, (sp.out_max as usize).min(left.len())).copied().collect(), dw)
                    } else {
                        let all: Vec<usize> = (0..64).collect();
                        (all.choose_multiple(rng, sp.code_lines as usize).copied().collect(), dw)
                    }
                };
                let out_mask = BitVector::from_bits(&out, 64);
                if eg.new_code {
                    let d = nodes[eg.dst].current_mut();
                    apply(&mut d.as_words_mut()[dw..dw + 1], out_mask.as_words(), eg.op);
                }
                class.add(SimpleKernel::tile(start, t, &syn, out_mask, dw, threshold, eg.op));
                continue;
            }
            // what it writes: a fresh random code into an empty destination (`new_code`), or the
            // destination word with the most lines its firing kernels left uncovered
            let (out, dw) = if eg.repeat {
                // what the destination held a frame ago (the act just made)
                let prev = nodes[eg.dst].get_frame(eg.repeat_lag);
                let (dw, dn) = busiest_word(prev.as_words());
                if (dn as usize) < sp.out_min as usize {
                    continue;
                }
                (ones(prev.as_words()[dw]), dw)
            } else if eg.new_code {
                if (dest.count_ones() as usize) >= sp.out_min as usize {
                    continue;
                }
                let dw = rng.gen_range(0..dest.word_len().max(1));
                let lines: Vec<usize> = (0..64).collect();
                (lines.choose_multiple(rng, sp.code_lines as usize).copied().collect::<Vec<usize>>(), dw)
            } else {
                // the destination word with the most lines no firing kernel wrote (word by word, no
                // copy of the destination)
                let mut covered: Vec<(usize, u64)> = Vec::new();
                for &k in &fired[i] {
                    let kk = &class.kernels()[k];
                    for (j, m) in kk.output_mask.as_words().iter().enumerate() {
                        covered.push((kk.output_idx + j, *m));
                    }
                }
                let cover = |w: usize| covered.iter().filter(|c| c.0 == w).fold(0u64, |a, c| a | c.1);
                let (dw, dn) = dest.as_words().iter().enumerate().filter(|(_, x)| **x != 0)
                    .map(|(w, x)| (w, (x & !cover(w)).count_ones()))
                    .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))).unwrap_or((0, 0));
                if (dn as usize) < sp.out_min as usize {
                    continue;
                }
                let lines = ones(dest.as_words()[dw] & !cover(dw));
                (lines.choose_multiple(rng, (sp.out_max as usize).min(lines.len())).copied().collect(), dw)
            };
            // what it reads: the source word with the most active lines, or (`wide`) all of the
            // source's active lines
            let (syn_mask, sw) = if eg.wide {
                // k of its n active lines, drawn by rank (the same draw as choosing from the list
                // of active lines, without making the list): each rank is found by walking word
                // popcounts and selecting the bit within the word
                let words = src.as_words();
                let n: usize = words.iter().map(|x| x.count_ones() as usize).sum();
                if n == 0 {
                    continue;
                }
                let mut syn: Vec<usize> = Vec::with_capacity(sp.birth_syn as usize);
                for mut r in rand::seq::index::sample(rng, n, (sp.birth_syn as usize).min(n)).into_iter() {
                    for (w, x) in words.iter().enumerate() {
                        let c = x.count_ones() as usize;
                        if r < c {
                            syn.push(w * 64 + select_nth_bit(*x, r));
                            break;
                        }
                        r -= c;
                    }
                }
                (BitVector::from_bits(&syn, src.bit_len()), 0)
            } else {
                let (sw, sn) = busiest_word(src.as_words());
                if sn == 0 {
                    continue;
                }
                let active = ones(src.as_words()[sw]);
                let syn: Vec<usize> = active.choose_multiple(rng, (sp.birth_syn as usize).min(active.len())).copied().collect();
                (BitVector::from_bits(&syn, 64), sw)
            };
            let threshold = (sp.birth_quarters as usize * syn_mask.count_ones()).div_ceil(4).max(1);
            let out_mask = BitVector::from_bits(&out, 64);
            let _ = dest;
            if eg.new_code {
                // a new code fires as it is born (the input that recruited it is here now)
                let d = nodes[eg.dst].current_mut();
                apply(&mut d.as_words_mut()[dw..dw + 1], out_mask.as_words(), eg.op);
            }
            let mut k = SimpleKernel::new(syn_mask, sw, out_mask, dw, threshold, eg.op);
            if sp.sparse {
                k.sparsify();
            }
            class.add(k);
        }
    }

    /// Read-only access to current frame of a node (for inspection).
    pub fn node_current(&self, idx: usize) -> &BitVector {
        self.nodes[idx].current()
    }

    /// Mutable access to current frame of a node (e.g., feed inputs).
    pub fn node_current_mut(&mut self, idx: usize) -> &mut BitVector {
        self.nodes[idx].current_mut()
    }

    pub fn write_input(&mut self, input: &BitVector, word_offset: usize) {
        let node = self.node_current_mut(0);
        node.mask_mut(word_offset, input, |a, b| a | b);
    }

    pub fn read_output(&self) -> &BitVector {
        self.node_current(self.output)
    }

    pub fn read_node(&self, node_idx: usize) -> &BitVector {
        self.node_current(node_idx)
    }

    pub fn write_node(&mut self, input: &BitVector, node_idx: usize, word_offset: usize) {
        let node = self.node_current_mut(node_idx);
        node.mask_mut(word_offset, input, |a, b| a | b);
    }

    pub fn overwrite_node(&mut self, input: &BitVector, node_idx: usize, word_offset: usize) {
        let node = self.node_current_mut(node_idx);
        node.mask_mut(word_offset, input, |_, b| b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::KernelClass;
    use crate::program::graph::GraphInstruction as I;

    /// The residue circuit, written in the graph language: the input wired (OR) into R, and a
    /// second edge from the input one frame back whose kernels cancel (XOR) and are born where R
    /// is left on. On a repeating sequence R goes to nothing.
    #[test]
    fn residue_circuit_cancels_a_repeating_sequence() {
        let prog = GraphProgram { code: vec![I::EdgeWire(), I::ReadInput(), I::EdgeLag(), I::EdgeXor(), I::EdgeGrow(), I::EdgeShare()] };
        let g = prog.build_graph(&ProgramDefaults { default_bits: 512, ..ProgramDefaults::default() });
        let mut net = RuntimeNetwork::from_graph_with(&g, 512, 512, 1, |_| KernelGroup::from_classes(vec![KernelClass::with_kernels(vec![])]));
        let mut rng = StdRng::seed_from_u64(2);
        let codes: Vec<BitVector> = (0..5).map(|_| crate::world::clustered_code(512, 16, &mut rng)).collect();
        let (mut early, mut late) = (0, 0);
        for t in 0..200 {
            net.tick(AdvanceMode::Clear, 0);
            net.write_input(&codes[t % 5], 0);
            let r = net.read_node(1).count_ones();
            if t < 10 { early += r } else if t >= 150 { late += r }
        }
        assert!(early >= 40, "the first steps are unpredicted ({early})");
        assert_eq!(late, 0, "a learned cycle leaves no residue");
        assert_eq!(net.edges.iter().map(|e| e.group_len()).sum::<usize>(), 5, "one kernel per transition");
    }
    use crate::program::graph::{GraphProgram};

    #[test]
    fn materialize_with_groups_and_tick() {
        // Build: input -> N0 (implicit in program)->output, then tick once
        let prog = GraphProgram::new();
        let mut defaults = ProgramDefaults::default();
        defaults.default_threshold = 1; // make sure kernel fires easily for test

        let mut net = RuntimeNetwork::from_program(&prog, &defaults);

        // Seed input: set 8 low bits
        net.write_input(
            &BitVector::from_words(vec![0xFF]),
            0,
        );
        // One tick: CopyPrev then process edges with default kernel
        net.tick(AdvanceMode::CopyPrev, 16);

        // N0 (node 1) should have fired, with high temperature
        let output = net.read_output();
        assert_eq!(output.count_ones(), 1);
    }

    #[test]
    fn simple_network_learns_static_pattern() {
        // Build: input -> N0 (implicit in program)->output, then tick once
        let prog = GraphProgram::new();
        let defaults = ProgramDefaults::default();
        
        let mut net = RuntimeNetwork::from_program(&prog, &defaults);

        //build vocab
        //need at least 16 bits for the mask to fire (threshold is 16)
        let a = BitVector::from_words(vec![0b01010101010101010101010101010101]);
        net.write_input(&a, 0);
        
        //show that it doesn't respond yet
        net.tick(AdvanceMode::CopyPrev, 0);
        assert_eq!(net.read_output().count_ones(),0);

        //start at a temperature of 16 that forces it to fire
        //and then gradually reduce temperature down
        for i in 0..16 {
            net.tick(AdvanceMode::CopyPrev, 16 - i);
            assert_eq!(net.read_output().count_ones(),1, "Failed at temperature adjustment {}",16 - i);
            net.overwrite_node(&BitVector::new(64, Some(0)), 1, 0); //clear output after first tick
            assert_eq!(net.read_output().count_ones(),0);
        }
        //fire with 0 temperature and show that the pattern has learned
        net.tick(AdvanceMode::CopyPrev, 0);

        //show that the input is preserved
        let input = net.read_node(0);
        assert_eq!(input.count_ones(), 16);
        //show that the kernel has fired to output
        let output = net.read_output();
        assert_eq!(output.count_ones(), 1);

    }
}