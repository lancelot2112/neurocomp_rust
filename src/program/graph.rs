// A program for generating a graph of bitvectors (node) and kernels (edge)
// - Each node will have either a Bitvector or a Bivector history with some number of historical frames
// - Each edge will have a kernel class connecting source to destination via "bit ops on high dimensional vectors"
// - Instructions will define how to build the graph structure and assign kernels to edges
// - Graph instructions will use a stack based cursor to "carve" out a directed graph structure
// - If the cursor stays on a node then the history depth increases by one
// - If the cursor moves to a new node then a new node is created with history depth 1 (only current frame)

use petgraph::graph::{EdgeIndex, Graph, NodeIndex};
use petgraph::Directed;
use crate::kernel::class::KernelOp;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeSpec {
    pub name: Option<String>,
    pub bits: usize,
    pub history_depth: usize, // frames (conceptual) for this node
    /// random lines switched on each step (spontaneous activity)
    pub noise: usize,
}

impl NodeSpec {
    pub fn from_defaults(defaults: &ProgramDefaults) -> Self {
        Self {
            name: None,
            bits: defaults.default_bits,
            history_depth: defaults.default_history_depth,
            noise: 0,
        }
    }
}

impl Default for NodeSpec {
    fn default() -> Self {
        Self {
            name: None,
            bits: 128,
            history_depth: 1,
            noise: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphInstruction {
    // Explicit sequence prediction nodes (creates a class of kenrnels with two inputs (t and t-1) and output to current output)
    Stay(), // stay on current node, increase history depth 

    // Simple input output nodes
    Create(), // create a new node and move to it 
    Pop(), // pop index from stack and move to that node creating an edge
    ReadInput(), // create an edge from current node directly to the input (do not move cursor)
    WriteOutput(), // create an edge from current node directly to the output (do not move cursor)?

    // Bitvector operations
    Double(), // Increase number of bits in current node's bitvector by doubling size 
    Half(), // Decrease number of bits in current node's bitvector by halving size

    // Stack operations
    Push(), // push current node index onto stack
    Swap(), // swap top two indices on stack
    Dup(), // duplicate top index on stack
    Nop(), // no operation... leaves potential site for future mutation while leaving the rest constant

    // LOOP
    Jump(), // jump back to the last loop instruction and disable this jump
    Mark(), // mark the start of a loop section

    // Edge settings: each changes the last edge made (no-op if there is none)
    EdgeOr(),     // its writes set lines
    EdgeAnd(),    // its writes keep only their lines
    EdgeXor(),    // its writes toggle lines (compare, cancel)
    EdgeClear(),  // its writes clear lines (AND-NOT)
    EdgeWire(),   // it copies its source whole (with its op) instead of holding kernels
    EdgeRotate(), // a wire rotates what it copies by a fixed step more
    EdgeLag(),    // it reads its source one frame further back
    EdgeGrow(),   // kernels are born on it where its destination is left uncovered
    EdgeShare(),  // its kernels do not hold each other down (all that match fire)
    EdgeGate(),   // pop a node: it learns (births, output learning) only while that node had lines on
    EdgeGateNot(),// pop a node: it learns only while that node had no lines on
    EdgeNew(),    // births into an empty destination write a fresh random code
    EdgeWide(),   // births sample synapses across the whole source, not one word
    EdgeSpan(),   // it reads the OR of twice as many past frames
    Noise(),      // the current node gets spontaneous activity: random lines each step (doubles)
    EdgeNand(),   // its writes are NAND
    EdgeNor(),    // its writes are NOR
    EdgeAdd(),    // its writes add, in byte lanes
    EdgeSub(),    // its writes subtract, in byte lanes
    EdgeShl(),    // what it writes is shifted left by 8, then ORed in
    EdgeShr(),    // ... shifted right by 8
    EdgeRotl(),   // ... rotated left by 8
    EdgeRotr(),   // ... rotated right by 8
    EdgeWider(),  // its op's lane width or amount grows (lanes 8 to 64 bits; amounts 0 to 63)
    EdgeKernels(),  // the last edge starts with more kernels (0, 1, 2, 4, ... 32)
    EdgeSynapses(), // its starting kernels have more synapses (4, 8, 16, 32, then 2)
    EdgeQuarter(),  // its starting kernels need one more quarter of their synapses (1 to 4, round)
}

/// What an edge carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    /// a group of kernels (masks over the source, writing masks into the destination)
    Kernels,
    /// the source frame itself, rotated by `rotate` steps, written into the destination
    Wire,
    /// spontaneous activity: `noise` random lines a step, written into the destination's span
    Noise,
}

/// Instructions for defining available kernel classes
/*
pub enum ClassInstruction {
    WindowIncr(), // increase candidate window size by 64 bits
    WindowDecr(), // decrease candidate window size by 64 bits
    ThresholdIncr(), // increase default firing threshold by 1
    ThresholdDecr(), // decrease default firing threshold by 1

    Nop(), // no operation... leaves potential site for future mutation while leaving the rest constant
}
*/

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EdgeSpec {
    pub input_read_back: usize, // how many frames back to read from source node
    pub threshold: usize,
    pub op: KernelOp,
    pub kind: EdgeKind,
    /// a wire's rotation, in steps
    pub rotate: u16,
    /// kernels are born on uncovered destination lines
    pub grow: bool,
    /// kernels whose masks overlap hold each other down (off: all that match fire)
    pub inhibit: bool,
    /// it learns only while this node had lines on (true) or none (false) a frame ago
    pub gate: Option<(usize, bool)>,
    /// births into an empty destination write a fresh random code
    pub new_code: bool,
    /// births sample synapses across the whole source
    pub wide: bool,
    /// frames read (ORed), from `input_read_back` back
    pub span: usize,
    /// read only these words of the source (offset, count): a channel or a slice
    pub window: Option<(usize, usize)>,
    /// a wire lands its copy starting at this word of the destination (side by side: concat)
    pub dst_word: usize,
    /// births write what the destination held a frame ago (repeat the last act)
    pub repeat: bool,
    /// how many frames back the repeated act is (the birth reads its source as many frames
    /// further back, when that act was made)
    pub repeat_lag: usize,
    /// everything below is this edge's own; there is nothing network-wide
    /// kernels it starts with, their synapses (in one source word), threshold (quarters of their
    /// synapses) and output lines (in one destination word)
    pub start_kernels: u8,
    pub start_syn: u8,
    pub start_quarters: u8,
    pub start_out: u8,
    /// births: synapses, threshold (quarters), fewest and most output lines, most kernels
    pub birth_syn: u8,
    pub birth_quarters: u8,
    pub out_min: u8,
    pub out_max: u8,
    pub cap: u16,
    /// lines of a fresh code (`new_code` births)
    pub code_lines: u8,
    /// output learning of cancelling kernels: a failed line moves with chance 2^-out_learn
    pub out_learn: u8,
    /// extinction on repeat edges: chance 2^-fade a step
    pub fade: u8,
    /// frames back the gate node is read
    pub gate_lag: usize,
    /// this edge's random stream
    pub salt: u32,
    /// destination words it writes, from `dst_word` (0: all of the destination); starting
    /// kernels and noise land there
    pub out_words: u16,
    /// a noise edge's random lines a step
    pub noise: u16,
    /// its starting kernels' activity: 0 responsive; k > 0 tonic (chance 2^-k a step)
    pub tonic: u8,
}

impl EdgeSpec {
    /// The grammar's starting values for every per-edge parameter (each edge then carries its
    /// own copy).
    pub fn plain() -> Self {
        Self {
            input_read_back: 1,
            threshold: 16,
            op: KernelOp::Or,
            kind: EdgeKind::Kernels,
            rotate: 0,
            grow: false,
            inhibit: true,
            gate: None,
            new_code: false,
            wide: false,
            span: 1,
            window: None,
            dst_word: 0,
            repeat: false,
            repeat_lag: 1,
            start_kernels: 0,
            start_syn: 4,
            start_quarters: 2,
            start_out: 16,
            birth_syn: 16,
            birth_quarters: 3,
            out_min: 8,
            out_max: 64,
            cap: 512,
            code_lines: 16,
            out_learn: 4,
            fade: 2,
            gate_lag: 1,
            salt: 0,
            out_words: 0,
            noise: 0,
            tonic: 0,
        }
    }
}

impl EdgeSpec {
    pub fn from_defaults(defaults: &ProgramDefaults) -> Self {
        Self {
            input_read_back: defaults.default_history_depth, // default to reading snapshot of previous frame
            threshold: defaults.default_threshold,
            op: defaults.default_op,
            ..EdgeSpec::plain()
        }
    }
}

impl Default for EdgeSpec {
    fn default() -> Self {
        Self::plain()
    }
}

#[derive(Clone, Debug)]
pub struct ProgramDefaults {
    pub default_bits: usize,      // starting bits for new nodes
    pub min_bits: usize,          // clamp Half() to this lower bound
    pub default_threshold: usize, // default threshold for edges
    pub default_history_depth: usize, // default history depth for new nodes
    pub default_op: KernelOp,       // default op for edges
}

impl Default for ProgramDefaults {
    fn default() -> Self {
        Self {
            default_bits: 256,
            min_bits: 8,
            default_threshold: 16,
            default_history_depth: 1,
            default_op: KernelOp::Or,
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct GraphProgram {
    pub code: Vec<GraphInstruction>,
}

impl GraphProgram {
    pub fn new() -> Self {
        Self { code: Vec::new() }
    }

    pub fn add_instr(mut self, instr: GraphInstruction) -> Self {
        self.code.push(instr);
        self
    }

    pub fn push(self) -> Self {
        self.add_instr(GraphInstruction::Push())
    }
    pub fn pop(self) -> Self {
        self.add_instr(GraphInstruction::Pop())
    }
    pub fn stay(self) -> Self {
        self.add_instr(GraphInstruction::Stay())
    }
    pub fn create(self) -> Self {
        self.add_instr(GraphInstruction::Create())
    }
    pub fn read_input(self) -> Self {
        self.add_instr(GraphInstruction::ReadInput())
    }
    pub fn write_output(self) -> Self {
        self.add_instr(GraphInstruction::WriteOutput())
    }
    pub fn double(self) -> Self {
        self.add_instr(GraphInstruction::Double())
    }
    pub fn half(self) -> Self {
        self.add_instr(GraphInstruction::Half())
    }
    pub fn swap(self) -> Self {
        self.add_instr(GraphInstruction::Swap())
    }
    pub fn dup(self) -> Self {
        self.add_instr(GraphInstruction::Dup())
    }
    pub fn nop(self) -> Self {
        self.add_instr(GraphInstruction::Nop())
    }
    pub fn mark(self) -> Self {
        self.add_instr(GraphInstruction::Mark())
    }
    pub fn jump(self) -> Self {
        self.add_instr(GraphInstruction::Jump())
    }


    /// Build a petgraph from the program.
    /// Semantics:
    /// - Starts with an implicit node (history=1, bits=defaults.default_bits) and empty stack.
    /// - Stay(): increment current node's history_depth by 1.
    /// - Create(): create a new node (history=1, bits=defaults.default_bits) and move cursor to it.
    /// - Pop(): pop a node index from the stack, add directed edge current -> popped with default edge spec, then move cursor to popped node.
    /// - Double(): current.bits *= 2.
    /// - Half(): current.bits = max(min_bits, current.bits / 2), floor at min_bits.
    /// - Push(): push current index onto stack.
    /// - Swap(), Dup(), Nop(): stack/flow ops as named.
    pub fn build_graph(&self, defaults: &ProgramDefaults) -> Graph<NodeSpec, EdgeSpec, Directed> {
        let mut graph: Graph<NodeSpec, EdgeSpec, Directed> = Graph::new();
        // Start with one implicit node so programs can begin with ops without Create().
        let mut curr = graph.add_node(NodeSpec::from_defaults(&defaults));
        #[allow(unused_assignments)]
        let mut last_edge: EdgeIndex = EdgeIndex::end();
        let mut stack: Vec<NodeIndex> = Vec::new();

        // Node 0 is always the "input" node. A bitvector with an edge is created off this
        // to start from. when there's a non-zero program length.  Otherwise we just connect 
        // the input to the output directly with an edge.
        let input_node = curr;
        if self.code.len() > 0 {
            curr = graph.add_node(NodeSpec::from_defaults(&defaults));
            last_edge = graph.add_edge(
                input_node,
                curr,
                EdgeSpec::from_defaults(&defaults),
            );
        }

        // Create the output vector node to make it available to connect to
        let output_node = graph.add_node(NodeSpec { name: Some("output".into()), ..NodeSpec::from_defaults(&defaults) });

        //Execute with a manual program counter so we can do jumps and marks
        let mut code = self.code.clone(); // so we can disable a jump after using it (replace with NOP in the clone)
        let mut pc: usize = 0;
        let mut last_mark: Option<usize> = None; // track last mark for jumps (all previous marks are superseded)

        while pc < code.len() {
            let instr = &code[pc];
            match instr {
                GraphInstruction::Stay() => {
                    {
                        let n = graph.node_weight_mut(curr).expect("current node must exist");
                        n.history_depth = n.history_depth.saturating_add(1);
                    }
                    last_edge = graph.add_edge(
                        curr,
                        curr,
                        EdgeSpec {
                            input_read_back: graph.node_weight(curr).unwrap().history_depth - 1, // each self-loop reads one frame further back, the first the previous frame
                            ..EdgeSpec::from_defaults(&defaults)
                        },
                    );
                }
                GraphInstruction::Create() => {
                    let new_node = graph.add_node(NodeSpec::from_defaults(&defaults));
                    last_edge = graph.add_edge(
                        curr,
                        new_node,
                        EdgeSpec::from_defaults(&defaults),
                    );
                    // move cursor
                    curr = new_node;
                }
                GraphInstruction::Pop() => {
                    if let Some(target) = stack.pop() {
                        if target == curr {
                            // self-loop increases history depth
                            {
                                let n = graph.node_weight_mut(curr).expect("current node must exist");
                                n.history_depth = n.history_depth.saturating_add(1);
                            }
                            last_edge = graph.add_edge(
                                curr,
                                curr,
                                EdgeSpec {
                                    input_read_back: graph.node_weight(curr).unwrap().history_depth - 1, // each self-loop reads one frame further back, the first the previous frame
                                    ..EdgeSpec::from_defaults(&defaults)
                                },
                            );
                        } else {
                            // add edge: curr -> target
                            last_edge = graph.add_edge(
                                curr,
                                target,
                                EdgeSpec::from_defaults(&defaults),
                            );
                        }
                        // move cursor
                        curr = target;
                    } //else: no-op on empty stack
                }
                GraphInstruction::ReadInput() => {
                    // add edge: input_node -> curr
                    last_edge = graph.add_edge(
                        input_node,
                        curr,
                        EdgeSpec::from_defaults(&defaults),
                    );
                    // do not move cursor
                }
                GraphInstruction::WriteOutput() => {
                    // add edge: curr -> output_node
                    last_edge = graph.add_edge(
                        curr,
                        output_node,
                        EdgeSpec::from_defaults(&defaults),
                    );
                    // do not move cursor
                }
                GraphInstruction::Double() => {
                    let n = graph.node_weight_mut(curr).expect("current node must exist");
                    n.bits = n.bits.saturating_mul(2);
                }
                GraphInstruction::Half() => {
                    let n = graph.node_weight_mut(curr).expect("current node must exist");
                    n.bits = (n.bits / 2).max(defaults.min_bits);
                }
                GraphInstruction::Push() => {
                    stack.push(curr);
                }
                GraphInstruction::Swap() => {
                    let len = stack.len();
                    if len >= 2 { //if there aren't two elements just do nothing
                        stack.swap(len - 1, len - 2);
                    }
                }
                GraphInstruction::Dup() => {
                    if !stack.is_empty() { // if stack is empty just do nothing
                        let top = *stack.last().unwrap();
                        stack.push(top);
                    }
                }
                GraphInstruction::Jump() => {
                    // disable this jump for future iterations by replacing with NOP
                    code[pc] = GraphInstruction::Nop();
                    if let Some(mark_pos) = last_mark {
                        // jump back to last mark
                        pc = mark_pos;
                    }
                }
                GraphInstruction::Mark() => {
                    last_mark = Some(pc);
                }
                GraphInstruction::Nop() => {}
                GraphInstruction::Noise() => {
                    let n = graph.node_weight_mut(curr).expect("current node must exist");
                    n.noise = (n.noise * 2).max(1).min(n.bits / 2);
                }
                GraphInstruction::EdgeOr() | GraphInstruction::EdgeAnd() | GraphInstruction::EdgeXor() | GraphInstruction::EdgeClear()
                | GraphInstruction::EdgeWire() | GraphInstruction::EdgeRotate() | GraphInstruction::EdgeLag() | GraphInstruction::EdgeGrow()
                | GraphInstruction::EdgeShare() | GraphInstruction::EdgeGate() | GraphInstruction::EdgeGateNot()
                | GraphInstruction::EdgeNew() | GraphInstruction::EdgeWide() | GraphInstruction::EdgeSpan()
                | GraphInstruction::EdgeKernels() | GraphInstruction::EdgeSynapses() | GraphInstruction::EdgeQuarter()
                | GraphInstruction::EdgeNand() | GraphInstruction::EdgeNor() | GraphInstruction::EdgeAdd() | GraphInstruction::EdgeSub()
                | GraphInstruction::EdgeShl() | GraphInstruction::EdgeShr() | GraphInstruction::EdgeRotl() | GraphInstruction::EdgeRotr()
                | GraphInstruction::EdgeWider() => {
                    // a gate takes its node off the stack (nothing happens without one)
                    let popped = if matches!(instr, GraphInstruction::EdgeGate() | GraphInstruction::EdgeGateNot()) { stack.pop() } else { None };
                    if last_edge != EdgeIndex::end() {
                        let src = graph.edge_endpoints(last_edge).unwrap().0;
                        let e = graph.edge_weight_mut(last_edge).unwrap();
                        match instr {
                            GraphInstruction::EdgeOr() => e.op = KernelOp::Or,
                            GraphInstruction::EdgeAnd() => e.op = KernelOp::And,
                            GraphInstruction::EdgeXor() => e.op = KernelOp::Xor,
                            GraphInstruction::EdgeClear() => e.op = KernelOp::Clear,
                            GraphInstruction::EdgeNand() => e.op = KernelOp::Nand,
                            GraphInstruction::EdgeNor() => e.op = KernelOp::Nor,
                            GraphInstruction::EdgeAdd() => e.op = KernelOp::Add(3),
                            GraphInstruction::EdgeSub() => e.op = KernelOp::Sub(3),
                            GraphInstruction::EdgeShl() => e.op = KernelOp::Shl(8),
                            GraphInstruction::EdgeShr() => e.op = KernelOp::Shr(8),
                            GraphInstruction::EdgeRotl() => e.op = KernelOp::Rotl(8),
                            GraphInstruction::EdgeRotr() => e.op = KernelOp::Rotr(8),
                            GraphInstruction::EdgeWider() => e.op = e.op.nudge(true),
                            GraphInstruction::EdgeWire() => e.kind = EdgeKind::Wire,
                            GraphInstruction::EdgeRotate() => e.rotate = e.rotate.wrapping_add(61),
                            GraphInstruction::EdgeKernels() => e.start_kernels = (e.start_kernels * 2).max(1).min(32),
                            GraphInstruction::EdgeSynapses() => e.start_syn = if e.start_syn >= 32 { 2 } else { e.start_syn * 2 },
                            GraphInstruction::EdgeQuarter() => e.start_quarters = e.start_quarters % 4 + 1,
                            GraphInstruction::EdgeGrow() => e.grow = true,
                            GraphInstruction::EdgeShare() => e.inhibit = false,
                            GraphInstruction::EdgeGate() => e.gate = popped.map(|n| (n.index(), true)),
                            GraphInstruction::EdgeGateNot() => e.gate = popped.map(|n| (n.index(), false)),
                            GraphInstruction::EdgeNew() => e.new_code = true,
                            GraphInstruction::EdgeWide() => e.wide = true,
                            GraphInstruction::EdgeSpan() => {
                                e.span = (e.span * 2).min(64);
                                let reach = e.input_read_back + e.span - 1;
                                let n = graph.node_weight_mut(src).unwrap();
                                n.history_depth = n.history_depth.max(reach);
                            }
                            _ => {
                                // lag: read one frame further back; the source keeps enough frames
                                e.input_read_back += 1;
                                let rb = e.input_read_back + e.span - 1;
                                let n = graph.node_weight_mut(src).unwrap();
                                n.history_depth = n.history_depth.max(rb);
                            }
                        }
                    }
                }
            }
            pc += 1;

        }

        // As a final step connect the current node to the output node!
        // This ensures there is always a path to output.
        last_edge = graph.add_edge(
            curr,
            output_node,
            EdgeSpec::from_defaults(&defaults),
        );

        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_program_connects_in_to_out() {
        let prog = GraphProgram::new();
        let g = prog.build_graph(&ProgramDefaults::default());
        assert_eq!(g.node_count(), 2); // input, output
        let n = g.node_weight(1.into()).unwrap();
        assert_eq!(n.bits, 256);
        assert_eq!(n.history_depth, 1);
    }

    #[test]
    fn create_and_connect_with_stack() {
        let defaults = ProgramDefaults {
            default_bits: 128,
            min_bits: 16,
            default_threshold: 4,
            default_history_depth: 1,
            default_op: KernelOp::Xor,
        };
        // Flow:
        // input -> N0
        // implicit N0, Push() -> [N0]
        // Create() -> curr=N0
        // Push() -> [N0, N1]
        // Pop() -> edge N1->N1, curr=N1 (increase history depth at node eg. self-loop) [N0]
        // Pop() -> edge N1->N0, curr=N0 []
        // Output edge N0 -> output
        let prog = GraphProgram::new()
            .push()
            .create()
            .push()
            .pop()
            .pop();

        let g = prog.build_graph(&defaults);
        assert_eq!(g.node_count(), 4); // input, N0, N1, output
        assert_eq!(g.edge_count(), 5); // input->N0, N0->N1, N1(t-1), N1->N0, N0->output

        // Validate edge specs are defaults
        for e in g.edge_weights() {
            assert_eq!(e.threshold, 4);
            assert_eq!(e.op, KernelOp::Xor);
        }
    }

    #[test]
    fn stay_increases_history_double_half_bits() {
        //input -> N0
        let prog = GraphProgram::new()
            .stay()
            .double()
            .half()
            .half(); // force clamp
        //N0 -> output

        let defaults = ProgramDefaults {
            default_bits: 64,
            min_bits: 32,
            default_threshold: 1,
            default_history_depth: 1,
            default_op: KernelOp::Or,
        };
        let g = prog.build_graph(&defaults);
        assert_eq!(g.node_count(), 3); // input, N0, output
        let n = g.node_weight(1.into()).unwrap();
        // Start 64, Double=128, Half=64, Half=32 (clamped to 32) ... 
        // doesn't make sense with current bitvector needed 64 bits min
        // but should be allowed for future flexibility
        assert_eq!(n.bits, 32);
        assert_eq!(n.history_depth, 2);
    }

    #[test]
    fn stack_swap_dup() {
        // Build 3 nodes and connect them via stack ops
        //input -> N0
        let prog = GraphProgram::new()
            .push()   // [N0]
            .create() // N1
            .push()   // [N0, N1]
            .create() // N2
            .swap()   // [N1, N0]
            .dup()   // [N1, N0, N0]
            .pop()    // edge N2->N0, curr=N1 [N1, N0]
            .pop();   // edge N0->N0, curr=N0
        // N0 -> output
        let g = prog.build_graph(&ProgramDefaults::default());
        assert_eq!(g.node_count(), 5); // input, N0, N1, N2, output
        assert_eq!(g.edge_count(), 6); // input->N0, N0->N1, N1->N2, N2->N0, N0(t-1) N0->output
    }

    #[test]
    fn mark_jump_runs_block_twice_then_disables() {
        // input -> N0 (implicit)
        // create -> N1 (edge N0->N1)
        // mark
        // create -> N2 (edge N1->N2)
        // jump  -> rewinds to after mark, disables jump
        // create -> N3 (edge N2->N3)
        // final -> N3 -> output
        let prog = GraphProgram::new()
            .create()
            .mark()
            .create()
            .jump();

        let g = prog.build_graph(&ProgramDefaults::default());
        // Nodes: input, N0, N1, N2, N3, output
        assert_eq!(g.node_count(), 6);
        // Edges: input->N0, N0->N1, N1->N2, N2->N3, N3->output
        assert_eq!(g.edge_count(), 5);
    }

    #[test]
    fn jump_without_mark_is_noop() {
        let prog = GraphProgram::new()
            .jump()     // no mark yet -> no-op
            .create();  // create one node

        let g = prog.build_graph(&ProgramDefaults::default());
        // input, N0, N1, output
        assert_eq!(g.node_count(), 4);
        // input->N0, N0->N1, N1->output
        assert_eq!(g.edge_count(), 3);
    }
}