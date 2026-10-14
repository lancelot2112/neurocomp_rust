# 02. The residue circuit is expressible in the graph language

**Question.** Can prediction-and-residue be written with general primitives, rather than as
a special module?

**Primitives added** (all general):
- **Edge instructions:** `EdgeOr/And/Xor/Clear` (write op), `EdgeWire` (copy the source whole),
  `EdgeRotate`, `EdgeLag` (read a frame further back), `EdgeGrow` (births on uncovered
  destination lines) and `EdgeShare` (no mutual inhibition on this edge).
- **Write order within a tick:** set (OR), then keep (AND), then toggle or clear (XOR, AND-NOT).
- **Output learning for cancelling kernels:** a toggled line still on moves (chance 1/16) to a
  line still on that the kernel does not write.

**Program.** `[EdgeWire, ReadInput, EdgeLag, EdgeXor, EdgeGrow, EdgeShare]`:
- R = input(t) OR'd in;
- kernels on input(t-1) XOR their predictions into R, and are born where R is left on.

`examples/residue.rs`, 5 seeds.

| Sequence | Residue (bits a step, late) | Kernels |
|---|---|---|
| fixed cycle of 6 | 0.0 from step 100 | 6, one per transition |
| branching restart, with inhibition | 2.3–2.8 | 129 and growing |
| branching restart, `EdgeShare` | 2.4–2.9 (the floor is about 2.7) | 9 |

**Meaning.**
- The circuit takes six instructions, with no special module.
- Predictors for alternatives must not inhibit each other, or the held-down one keeps being
  reborn. This is now an evolvable edge setting, not a rule.
- Failure learning at 1/16 per line, not 1/2 (lesson 13).
- Evolution from scratch did not find this circuit (experiment 01's later runs). No task yet
  rewards it.
