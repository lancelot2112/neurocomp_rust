# 05. The context circuit in the grammar, as a seed for evolution

**Question.** Can the context reader (experiment 04's harness holder, now deleted) be written
in the network's own grammar, and used as a starting point for evolution? What holes does
writing it expose?

## Holes found, and what fills them

| Hole | Fix |
|---|---|
| The stack grammar cannot wire a re-entrant circuit: edges only go out from the cursor or in from the input, the stack has only push/dup/swap, and every move makes an edge, so feedback and gates need junk edges | **gene list** (`src/evolve/genes.rs`): nodes and edges with permanent ids, edges naming their ends (as `wiki/evolution.md` recommended). Compiles to the same graph |
| (bug) the act was read from the last node, but `build_graph` makes the output node third: any program with `Create` acted from a hidden node | output found by name |
| (bug) `Stay`'s self-loop read two frames back, so a held vector flickered | reads the previous frame |
| plasticity could not be switched by another vector (no third factor) | `EdgeGate` / `EdgeGateNot`: learn only while a node had lines on / none |
| births could only copy uncovered lines, so nothing new could be named | `EdgeNew`: births into an empty destination write a fresh code, firing at birth |
| births read one word, so no conjunction across codes in different words | `EdgeWide` |
| no reading over time | `EdgeSpan`: OR of several past frames |
| no channel selection, no side-by-side placement | edge `window` (source words) and `dst_word` (landing word) |
| no spontaneous activity, so nothing to explore with | node `noise` (`Noise()` in the stack grammar) |
| no way to repeat the act a stroke followed | `repeat` births copy the act from `repeat_lag` frames back, reading the source as it was then |
| no extinction: acts not followed by a stroke never weakened (perseveration) | on a repeat edge, a kernel that acted `repeat_lag` steps ago with the gate shut moves a synapse off (chance 1/4); kernels keep a `u8` age |
| (bug) starting kernels ignored an edge's window | sized to what the edge reads |

## The seed

`GeneGenome::context_seed`, 11 nodes and 21 edges:
- **Y:** eye and affect of the last three steps.
- **Z:** conjunction codes on Y (`grow`, `new`, `wide`).
- **R:** Z, less what the context C expects. Expectations are learned while young, gated by Yg.
- **C:** held by a self-wire, and changed by three kinds of kernel on R:
  - **leave:** clear C, learned when settled;
  - **recognise:** restore a context, learned while young;
  - **enter:** a new code when C is empty.
- **Yg:** C's onset held for 32 frames ("young").
- **A:** the eye and C side by side.
- **Acts:** 8 babblers on spontaneous activity, plus `repeat` births gated by the affect channel,
  with extinction.

## Results

`examples/context_seed.rs`, serial reversal in the world: the paying light swaps every 100
trials, new codes each lifetime, 5 lifetimes. Discrimination is in parts per 1024; phase 0 is
light A paying.

| Variant | phase 0 | phase 1 | phases 2–5 | C changes per phase |
|---|---|---|---|---|
| seed, before extinction | 258 | −308 | +231, −347, +223, −405 (perseverates on A) | 5–22 |
| seed, with extinction | 230 | −83 | −1 to −63 | 4–10 |
| without C reaching the acts | 260 | 87 | −53 to 56 | 3–6 |
| act path only (babble + repeat + extinction) | 265 | **228** | −140 to 47 | — |

Evolution from the seed (`examples/evolve_seed.rs`): reversal, 400 trials a lifetime, 4 lifetimes
per genome, population 24, 40 generations.

| Start | Seed, held out | Best, training | Best, held out (5 lifetimes) |
|---|---|---|---|
| context seed | 22 | 182 | 51 |
| act path only | 37 | 177 | 9 |

## Meaning

- **The circuit can be written**, but only in the gene list. The stack grammar's cursor and
  stack cannot express feedback without junk edges.
- **The act path learns:** a stroke after an act recruits a kernel repeating that act on what
  came before it (258–472 in the first phase). Extinction is needed to reverse at all. The act
  path alone reverses once (228), then degrades.
- **The context part does not yet do its job.** C changes 4–10 times per phase, but not at the
  swaps, and routing C into the acts makes reversal worse, not better. Evolution, given the
  seed, mostly took the circuit apart.
- **Evolution's gains do not hold out.** Training fitness rose to about 180, held out stayed
  within noise (9–51). Four lifetimes per genome are too few to select on reversal: one
  lifetime's discrimination moves by hundreds. More lifetimes per genome, or longer lifetimes,
  come first.
- **Open holes:**
  - birth parameters (synapses, threshold, minimum output) are global, not per edge;
  - a gate reads its node one frame back only;
  - extinction exists only on repeat edges;
  - the absorb-versus-switch decision of the context circuit is still not right.

## Update: no globals

Every parameter that was shared across the network or genome is now a field of each edge (or
node), with the grammar's starting value copied into every new gene.

**What moved onto each edge:**
- starting kernels: count, synapses, threshold quarters, output lines;
- births: synapses, threshold quarters, fewest and most output lines, cap;
- new-code size;
- learning rates (output learning 2^-k, extinction 2^-k), given as exponents;
- the gate's lag;
- rotation, now in bits;
- a random stream keyed by the edge's permanent id.

**Elsewhere:**
- Nodes keep their own noise streams.
- The stack genome is now only its program. Three new instructions set an edge's starting
  kernels: `EdgeKernels`, `EdgeSynapses`, `EdgeQuarter`.

**Effect on the seed.** Same circuit; different random streams for the babbling kernels (now per
edge):
- Phase 0 went from 230 to 22, and the act-path control from 265 to 98. The babblers now click on
  about 60% of trials, which buries what is learned.
- One random wiring moving discrimination by 200 points is the noise that makes selection on a
  few lifetimes meaningless (above).
- The residue circuit is unchanged.

**Left as rule constants, not genes:**
- one Hebbian move per firing;
- the overlap-inhibition rule (more than half of the smaller mask);
- the write-phase order;
- the near-miss search drift (an option that is off).

The base kernel class still initialises a "temperature" from `common::config`. It is 0 and
unused, and should go.
