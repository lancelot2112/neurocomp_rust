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
}
pub struct RuntimeEdgeGroup {
    pub src: RuntimeEdgeTarget,
    pub dst: usize,
    pub group: KernelGroup,
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
            let ring_len = ns.history_depth.max(2);
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
                },
                dst,
                group: KernelGroup::default(), // placeholder; later: pick group per edge
            });
        }

        Self { nodes, edges }
    }

    /// Advance histories and process all edges once.
    pub fn tick(&mut self, mode: AdvanceMode) {
        // Advance all nodes first so writes go into a fresh current frame
        for n in &mut self.nodes {
            n.advance(mode);
        }

        // Execute edges
        for eg in &mut self.edges {
            // Snapshot source at read_back frames
            let src_view = self.nodes[eg.src.idx].get_frame(eg.src.read_back);
            let dst_curr = self.nodes[eg.dst].current_mut();

            eg.group.process_all(&src_view, dst_curr, 0); // phase=0 for now
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
}

fn ones_mask(bits: usize) -> BitVector {
    BitVector::new(bits, Some(u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::graph::{GraphInstruction, GraphProgram};

    #[test]
    fn materialize_with_groups_and_tick() {
        // Build: input -> N0 (implicit in program)->output, then tick once
        let prog = GraphProgram::new();
        let defaults = ProgramDefaults::default();

        let mut net = RuntimeNetwork::from_program(&prog, &defaults);

        // Seed input: set 8 low bits
        {
            let inp = net.node_current_mut(0);
            let mask = BitVector::from_words(vec![0xFF]);
            inp.mask_mut(0, &mask, |a, b| a | b);
        }

        // One tick: CopyPrev then process edges with default kernel
        net.tick(AdvanceMode::CopyPrev);

        // N0 (node 1) should have received a write (default threshold is 1)
        let n0_word0 = net.node_current(1).as_lswords().next().unwrap_or(0);
        assert_ne!(n0_word0, 0);
    }

    fn simple_network_learns_static_pattern() {
        // Build: input -> N0 (implicit in program)->output, then tick once
        let prog = GraphProgram::new();
        let defaults = ProgramDefaults::default();
        
        let mut net = RuntimeNetwork::from_program(&prog, &defaults);
    }
}