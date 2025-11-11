// A program for generating a graph of bitvectors (node) and kernels (edge)
// - Each node will have either a Bitvector or a Bivector history with some number of historical frames
// - Each edge will have a kernel class connecting source to destination via "bit ops on high dimensional vectors"
// - Instructions will define how to build the graph structure and assign kernels to edges
// - Graph instructions will use a stack based cursor to "carve" out a directed graph structure
// - If the cursor stays on a node then the history depth increases by one
// - If the cursor moves to a new node then a new node is created with history depth 1 (only current frame)

use petgraph::graph::{Graph, NodeIndex};
use petgraph::Directed;
use crate::kernel::class::KernelOp;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeSpec {
    pub name: Option<String>,
    pub bits: usize,
    pub history_depth: usize, // frames (conceptual) for this node
}

impl NodeSpec {
    pub fn from_defaults(defaults: &ProgramDefaults) -> Self {
        Self {
            name: None,
            bits: defaults.default_bits,
            history_depth: defaults.default_history_depth,
        }
    }
}

impl Default for NodeSpec {
    fn default() -> Self {
        Self {
            name: None,
            bits: 128,
            history_depth: 1,
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
    // room for future kernel-class parameters (windows, remapping, etc.)
}

impl EdgeSpec {
    pub fn from_defaults(defaults: &ProgramDefaults) -> Self {
        Self {
            input_read_back: 1, // default to reading snapshot of previous frame
            threshold: defaults.default_threshold,
            op: defaults.default_op,
        }
    }
}

impl Default for EdgeSpec {
    fn default() -> Self {
        Self {
            input_read_back: 1, // default to reading snapshot of previous frame
            threshold: 16,
            op: KernelOp::Or,
        }
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
        let mut stack: Vec<NodeIndex> = Vec::new();

        // Node 0 is always the "input" node. A bitvector with an edge is created off this
        // to start from.
        let input_node = curr;
        curr = graph.add_node(NodeSpec::from_defaults(&defaults));
        graph.add_edge(
            input_node,
            curr,
            EdgeSpec::from_defaults(&defaults),
        );

        // Create the output vector node to make it available to connect to
        let output_node = graph.add_node(NodeSpec::from_defaults(&defaults));

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
                    graph.add_edge(
                        curr,
                        curr,
                        EdgeSpec {
                            input_read_back: graph.node_weight(curr).unwrap().history_depth, // self-loop reads previous frame
                            threshold: defaults.default_threshold,
                            op: defaults.default_op,
                        },
                    );
                }
                GraphInstruction::Create() => {
                    let new_node = graph.add_node(NodeSpec::from_defaults(&defaults));
                    graph.add_edge(
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
                            graph.add_edge(
                                curr,
                                curr,
                                EdgeSpec {
                                    input_read_back: graph.node_weight(curr).unwrap().history_depth, // self-loop reads previous frame
                                    threshold: defaults.default_threshold,
                                    op: defaults.default_op,
                                },
                            );
                        } else {
                            // add edge: curr -> target
                            graph.add_edge(
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
                    graph.add_edge(
                        input_node,
                        curr,
                        EdgeSpec::from_defaults(&defaults),
                    );
                    // do not move cursor
                }
                GraphInstruction::WriteOutput() => {
                    // add edge: curr -> output_node
                    graph.add_edge(
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
            }
            pc += 1;

        }

        // As a final step connect the current node to the output node!
        // This ensures there is always a path to output.
        graph.add_edge(
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
    fn default_program_starts_with_one_node() {
        let prog = GraphProgram::new();
        let g = prog.build_graph(&ProgramDefaults::default());
        assert_eq!(g.node_count(), 3); // input, N0, output
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
        assert_eq!(g.edge_count(), 4); // input->N0, N0->N1, N1++ (not an edge), N1->N0, N0->output

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
        assert_eq!(g.edge_count(), 5); // input->N0, N0->N1, N1->N2, N2->N0, N0++ N0->output
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