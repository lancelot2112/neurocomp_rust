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
pub struct RuntimeEdgeGroup {
    pub src: usize,
    pub dst: usize,
    pub group: KernelGroup,
    pub default: DefaultEdge,
}

/// A simple default kernel: if popcount(input & mask) >= threshold,
/// apply op with output_mask to destination current frame.
pub struct DefaultEdge {
    pub threshold: usize,
    pub op: KernelOp,
    pub input_mask: BitVector,
    pub output_mask: BitVector,
    pub read_back: usize, // 0=current, 1=prev1, 2=prev2
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
            let ring_len = ns.history_depth.max(3);
            let init = BitVector::new(bits, Some(0));
            nodes.push(BitVecHistory::new(init, ring_len));
        }

        // Compile edges to runtime edge groups
        let mut edges: Vec<RuntimeEdgeGroup> = Vec::with_capacity(g.edge_count());
        for e in g.edge_references() {
            let src = e.source().index();
            let dst = e.target().index();
            let es = e.weight();

            let src_bits = g.node_weight(NodeIndex::new(src)).unwrap().bits;
            let dst_bits = g.node_weight(NodeIndex::new(dst)).unwrap().bits;

            let input_mask = ones_mask(src_bits);
            let output_mask = ones_mask(dst_bits);

            edges.push(RuntimeEdgeGroup {
                src,
                dst,
                group: KernelGroup::default(), // placeholder; later: pick group per edge
                default: DefaultEdge {
                    threshold: es.threshold,
                    op: es.op,
                    input_mask,
                    output_mask,
                    read_back: 1, // read previous frame by default
                },
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
        for eg in &self.edges {
            // TODO: If eg.group is non-empty, dispatch KernelClasses here.
            // For now, always execute the simple default kernel:
            let d = &eg.default;

            // Snapshot source at read_back frames
            let src_bv = self.nodes[eg.src].snapshot(d.read_back);

            // Count ones in (src & input_mask) using existing API
            let ones = src_bv.mask_and_count(0, &d.input_mask, |a, m| a & m);

            if ones >= d.threshold {
                // Apply op with output_mask into destination current frame using existing API
                let dst = self.nodes[eg.dst].current_mut();
                match d.op {
                    KernelOp::Or => dst.mask_mut_or(0, &d.output_mask),
                    KernelOp::And => dst.mask_mut_and(0, &d.output_mask),
                    KernelOp::Xor => dst.mask_mut_xor(0, &d.output_mask),
                    KernelOp::Clear => dst.mask_mut_clear(0, &d.output_mask),
                }
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::graph::{GraphInstruction, GraphProgram};

    #[test]
    fn materialize_with_groups_and_tick() {
        // Build: input -> N0 (implicit in program), then tick once
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
}