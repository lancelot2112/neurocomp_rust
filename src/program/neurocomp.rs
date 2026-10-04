use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Directed;

use crate::bitvec::BitVector;
use crate::bitvec::bithistory::{AdvanceMode, BitVecHistory};
use crate::kernel::class::KernelOp;
use crate::kernel::group::KernelGroup;
use crate::program::graph::{EdgeSpec, GraphProgram, NodeSpec, ProgramDefaults};

/// A runnable network: node histories plus per-edge kernel groups.
/// For now, each edge also carries a simple default kernel used for execution
/// until KernelClass execution is wired in.
pub struct RuntimeNetwork {
    pub nodes: Vec<BitVecHistory>,
    pub edges: Vec<RuntimeEdgeGroup>,
}

/// Per-edge runtime unit: group of kernel classes plus a default kernel.
pub struct RuntimeEdgeTarget {
    pub idx: usize, // target node index
    pub read_back: usize, // how many frames back to read from
    pub frames: usize, // consecutive frames read from read_back on, concatenated most-recent first
}
pub struct RuntimeEdgeGroup {
    pub src: RuntimeEdgeTarget,
    pub dst: usize,
    pub group: KernelGroup,
    pub last_input: Option<BitVector>, // what the group read on the last tick (for `learn`)
}

impl RuntimeNetwork {
    /// Build a runnable network from a GraphProgram.
    pub fn from_program(prog: &GraphProgram, defaults: &ProgramDefaults) -> Self {
        let g = prog.build_graph(defaults);
        Self::from_graph(&g)
    }

    /// Build a runnable network from a Graph<NodeSpec, EdgeSpec>.
    pub fn from_graph(g: &Graph<NodeSpec, EdgeSpec, Directed>) -> Self {
        // Every frame an outgoing edge reads must exist in the ring without wrapping
        // onto the current frame (index 0).
        let mut deepest_read = vec![0usize; g.node_count()];
        for e in g.edge_references() {
            let es = e.weight();
            let d = &mut deepest_read[e.source().index()];
            *d = (*d).max(es.input_read_back + es.input_frames.max(1) - 1);
        }

        // Materialize node histories
        let mut nodes: Vec<BitVecHistory> = Vec::with_capacity(g.node_count());
        for ni in g.node_indices() {
            let ns = g.node_weight(ni).expect("node weight");
            let bits = ns.bits;
            let ring_len = ns.history_depth.max(deepest_read[ni.index()] + 1).max(2);
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
                    frames: es.input_frames.max(1),
                },
                dst,
                group: KernelGroup::default(), // placeholder; later: pick group per edge
                last_input: None,
            });
        }

        Self { nodes, edges }
    }

    /// Advance histories and process all edges once.
    pub fn tick(&mut self, mode: AdvanceMode, adj_temperature: i16) {
        // Advance all nodes first so writes go into a fresh current frame
        for n in &mut self.nodes {
            n.advance(mode);
        }

        // Execute edges
        for eg in &mut self.edges {
            // Snapshot source at read_back frames
            let src_view = read_frames(&self.nodes[eg.src.idx], eg.src.read_back, eg.src.frames);
            let dst_curr = self.nodes[eg.dst].current_mut();

            eg.group.process_all(&src_view, dst_curr, 0, adj_temperature); // phase=0 for now
            eg.last_input = Some(src_view);
        }
    }

    /// Predictive learning step: tell every edge into `dst` what `dst` should have
    /// held after the last tick (e.g. the next input). Call after `tick`, before the next one.
    pub fn learn(&mut self, dst: usize, target: &BitVector) {
        let mut rng = rand::thread_rng();
        for eg in self.edges.iter_mut().filter(|eg| eg.dst == dst) {
            if let Some(input) = &eg.last_input {
                eg.group.feedback(input, target, &mut rng);
            }
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
        self.node_current(self.nodes.len() - 1)
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

/// Read `frames` consecutive history frames starting at `read_back`,
/// concatenated most-recent first.
fn read_frames(hist: &BitVecHistory, read_back: usize, frames: usize) -> BitVector {
    if frames <= 1 {
        return hist.snapshot(read_back);
    }
    let mut words = Vec::new();
    for f in 0..frames {
        words.extend_from_slice(hist.get_frame(read_back + f).as_words());
    }
    BitVector::from_words(words)
}

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn history_ring_covers_multi_frame_reads() {
        let defaults = ProgramDefaults::default();
        let mut g = GraphProgram::new().build_graph(&defaults);
        let e = g.edge_indices().next().unwrap(); // input -> output
        g.edge_weight_mut(e).unwrap().input_frames = 3;
        let net = RuntimeNetwork::from_graph(&g);
        // reading frames 1..=3 needs a ring of 4 so frame 3 doesn't wrap onto the current frame
        assert!(net.nodes[0].get_frame(3).bit_len() > 0);
        assert_eq!(net.edges[0].src.frames, 3);
    }

    #[test]
    fn learns_to_predict_next_input() {
        use crate::kernel::{GrowthConfig, KernelClass, KernelGroup};
        let defaults = ProgramDefaults { default_bits: 64, ..ProgramDefaults::default() };
        let mut net = RuntimeNetwork::from_program(&GraphProgram::new(), &defaults);
        let mut group = KernelGroup::new();
        group.add_class(KernelClass::predictive(GrowthConfig { sample_bits: 8, ..GrowthConfig::default() }));
        net.edges[0].group = group;
        let out = net.nodes.len() - 1;

        let seq = [0xFFu64, 0xFF00, 0xFF_0000]; // A B C A B C ...
        let mut correct = 0;
        for t in 0..30 {
            net.write_input(&BitVector::from_words(vec![seq[t % 3]]), 0);
            net.tick(AdvanceMode::Clear, 0);
            let next = BitVector::from_words(vec![seq[(t + 1) % 3]]);
            if t >= 3 && net.read_node(out).as_words()[0] == seq[(t + 1) % 3] {
                correct += 1;
            }
            net.learn(out, &next);
        }
        assert_eq!(correct, 27);
    }
}