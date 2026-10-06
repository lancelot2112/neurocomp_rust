# 51 · Networks of kernels: modules, recursion, and a grammar that builds them

**Question.** Experiments 40–50 built a lot of machinery, but much of it is assembled by
hand in [`examples/episodic.rs`](../../examples/episodic.rs), about 3,600 lines of glue.
The early graph grammar ([`src/program/graph.rs`](../../src/program/graph.rs)) can only
wire nodes and kernel edges; none of the later components can be expressed in it. Can the
system instead be built from the base kernels, composed into networks, with networks
composed into larger networks? And can a grammar express those architectures so they can
be mutated and evolved?

**Code.** [`src/program/modules.rs`](../../src/program/modules.rs), plus
`BitVector::EMPTY`. Tests: `cargo test --release --lib modules`. Nothing existing changed
behaviour; the regression suite is untouched.

## The design

### One interface: `Module`
A module has bit-vector input ports and output ports and ticks once per step.
- **Methods:** `tick(inputs, ctx)`, `output(port)`, `sleep()`, `reset()` (a story
  boundary: activity cleared, learning kept), `kernels()`, `describe()`.
- **Ports carry only bit vectors.** An unconnected port reads as empty (all zeros).

### Leaves: the base kernels
| Leaf | What it is | Brain role it plays |
|---|---|---|
| `Predictor` | a predictive `KernelClass`. Inputs `[input, teach]`, outputs `[prediction, surprise]` | L2/3, higher areas, semantic store, CA3 heteroassociation |
| `BitOp` | a `KernelOp` (or, and, xor, clear = a & !b) over two inputs | integration, gating, inhibition |
| `Delay` | one frame of history | L6 context, recurrence |
| `Concat(n)` | frames side by side | L4 assembly |
| `Separate` | a hashed expansion with k winners (`DentateGyrus::hashed`) | dentate gyrus |

**Learning is local and travels on wires.** A `Predictor` learns at each tick that
whatever arrives on its `teach` port should have followed its previous input. Then it
predicts from its current input.
- No outside call to `learn` and no backpropagation: what a kernel learns from is part of
  the wiring.
- With learning off (at test) only fast inhibition runs, as in `CorticalColumn`.
- Its `surprise` output is the bitwise prediction error: `teach & !previous prediction`.

### Composition: `Network` is a `Module`
A network holds children and wires. Each child input reads one of:
- a network input;
- another child's output.

The network then exports chosen outputs as its own. Because a network is a module, it can
be a child of another network, to any depth.

**Timing is fixed and deterministic.** Children tick in placement order.
- A wire from an earlier child carries this tick's value.
- A wire from the same or a later child carries last tick's value.

Every loop therefore has a one-tick delay, as recurrent cortex does.

### The grammar: `NetOp`, `Genome`
A stack language builds networks. Signals go on a stack.
- **`In(i)`** pushes a network input.
- **`Place(prim)`** pops a leaf's inputs and pushes its outputs.
- **`Sub(k)`** does the same with an earlier definition in the genome. This is the
  recursion.
- **`Feedback`** pushes a forward reference. **`Close`** binds the latest open one to
  the signal on top, which makes loops.
- **`Out`** exports a signal.
- **`Dup`, `Swap`, `Drop`, `Over`, `Nop`** are stack operations.

A `Genome` is a list of definitions. Each may place the ones before it, so recursion
always terminates.

**Every instruction sequence builds a valid network.** Missing signals become unconnected
ports, and out-of-range or forward `Sub` references are no-ops. Genomes can therefore be
mutated freely, as the original graph grammar intended.

**A cortical column, written in the grammar** (`column_code`):
```
In(0) Dup Delay Concat(2) In(0) Predict Swap Out Out
```
That is: L6 = `Delay(x)`; L4 = `[x, previous]`; L2/3 = a `Predictor` taught by the next x.
It exports the prediction and the surprise.

**A two-level hierarchy** (`hierarchy_genome`) is built from a column with a top-down port,
placed by `Sub`:
- **State:** the lower column's surprise, OR-accumulated over the story through a
  `Delay` loop. This is the slow state of `HigherArea`
  ([24](24-cortical-hierarchy.md)).
- **Higher area:** a `Predictor` reading `[x, state]`, taught by x.
- **Lower column:** reads x and the higher prediction as its top-down frame. Its
  surprise feeds the state through a `Feedback`.

## Results (`cargo test --release --lib modules`)

| Test | Result |
|---|---|
| The grammar's column vs `CorticalColumn`, same seed, 300 steps | **bit-identical predictions at every step** |
| A X Y → B vs C X Y → D (the next word depends on the word three back): a column alone | 49 / 100 right at Y (chance) |
| Same task: a hierarchy of two columns, built by `Sub` | **100 / 100** |
| One-shot sequence memory (`Separate` → `Predictor`, taught by the next code, prediction fed back): replay 8 items from the first after one exposure | **7 / 7 steps exact** |
| 50 random genomes (3 nested definitions, random instructions, every leaf type) | all build, run, sleep and reset without error |

## Findings
1. **The column is a composition of base kernels, exactly.**
   - The grammar's column reproduces `CorticalColumn` bit for bit (without relayed
     frames).
   - The class that hand-wired L4/L6 around a kernel class is one instruction sequence.
2. **Recursion works and pays.**
   - The column alone cannot predict past an ambiguous two-word context (49%).
   - Placing it inside a network with a slower area over its own surprise gives 100%.
   - That network is eighteen instructions, one of them a `Sub`. The same structure as experiment
     24, but expressed rather than hand-written.
3. **Sequence memory from base kernels.**
   - A predictor taught by its own next input, over a separated code, is a one-shot
     heteroassociative memory.
   - Closing its output back to its input replays the sequence. This is the CA3
     sequence memory that experiment [50](50-replay-for-rule-extraction.md) found missing,
     built from the existing kernel rather than new machinery.
   - Not yet tested: whether replay through it teaches the higher area's rules.
4. **The grammar is closed under mutation.** Any genome builds and runs. Architecture
   search can work on genomes without a repair step.

## What is composed, and what is not yet

| Component | As a network of base kernels? |
|---|---|
| Cortical column (L2/3, L4, L6 previous frame) | **yes**, bit-identical |
| Higher area (state of surprises, top-down frame) | **yes** in structure. The current `HigherArea` also fades its state and judges surprise by Q16 probability. A fading state needs a decay leaf (e.g. a stochastic `Clear`). |
| Semantic store | **yes**: it is a `Predictor` (cue → content) |
| CA3 sequence memory | **yes** (new) |
| Dentate gyrus | **yes**, as the `Separate` leaf |
| CA3 autoassociation, Schaffer, perforant paths with presynaptic scaling | not yet. These use a second learning rule (Hebbian bit-sliced counters, `Pathway`), not the predictive kernel. It should become a second learning leaf, `Associate`. |
| CA1 comparator, novelty gating, tags | not yet: they need `Associate` plus scalar signals |
| Role cells (competitive Hebbian) | not yet: a `Compete` leaf, or a `Separate` with learned weights |
| Relay channels (content-addressed match over L6) | not yet: a `Match` leaf over a `Delay` chain |
| Thalamic mix, basal ganglia, L6 gating, confidence | not yet. These carry Q16 scalars (confidence, reward, value), which ports do not. |
| The episodic harness (`examples/episodic.rs`) | not migrated |

**Open design choice: scalars on ports.** The gating systems trade numbers (confidence,
reward, precision), not patterns. There are two options:
- add a scalar port type;
- carry them as population codes (e.g. a thermometer of k active bits), so everything
  stays a bit vector.

The second keeps the "everything is bits" rule and lets kernels read them. It costs
resolution.

## Next
1. **`Associate`:** the Hebbian pathway as a learning leaf, with presynaptic scaling.
   Then express CA3 autoassociation and CA1 → EC readout as networks, and rebuild
   `Hippocampus` as a genome, checking parity against `hippocampal_circuit.rs` the way the
   column was checked here.
2. **Scalar signals** as population codes, so the mix, basal ganglia and gates can be
   leaves wired like the rest.
3. **Sequence replay into the higher area:** the experiment 50 test, with the sequence
   memory here supplying the replay.
4. **Migrate the episodic harness** one component at a time, keeping the regression suite
   at parity. The goal is an architecture that is one genome.
5. **Then search:** mutate genomes and select on the regression tasks.

## Biology
- **Canonical microcircuit:** cortex repeats one local circuit, and areas are copies of
  it wired into hierarchies (Douglas & Martin 2004; Bastos et al. 2012). Here a column is
  one definition, and areas are copies of it placed by `Sub`.
- **Local learning with a teaching input:** apical dendrites receive a teaching or
  expectation signal separate from the basal drive (Larkum 2013; Guerguiev et al. 2017).
  This plays the role of the `teach` port.
- **Developmental programs:** the brain's wiring is generated by a compact program, not
  stored. This is the genomic bottleneck (Zador 2019), and the reason for a grammar.
