# 52 · A Hebbian leaf, and the hippocampus as a genome

**Question.** [51](51-networks-of-kernels.md) expressed the cortical column in the
network grammar, but not the hippocampus. Its pathways learn by a second rule, Hebbian
counters, rather than the predictive kernel's surprise-driven growth. If that rule becomes
a base kernel too, can the full hippocampal circuit
([46](46-full-hippocampus.md)) be written as a genome and reproduce `Hippocampus` exactly?

**Code.**
- [`src/program/modules.rs`](../../src/program/modules.rs):
  - leaves `Associate`, `Scatter`, `Gate` and `Separate::table`;
  - the instruction `NetOp::Pick`;
  - `hippocampus_genome`, `theta_store`, `theta_recall`.
- [`src/program/hippocampus.rs`](../../src/program/hippocampus.rs): `Pathway` rows grow
  on demand; `half_life_of`.
- [`examples/superposition.rs`](../../examples/superposition.rs): the capacity
  measurements.
- The math is in [superposition and clean-up](../concepts/superposition-and-clean-up.md).
- Regression suite: all 14 entries unchanged.

## The leaves
**`Associate`, the second base kernel:** a population of cells with Hebbian input
pathways.
- **Ports:** inputs `[teach, gain, pre_0, …]`, output `[activity]`.
- **Read:** drive = the sum over pathways of the rows of the active input bits. The
  population keeps the k most driven cells, or every cell within a fraction of the best.
  With a recurrent pathway it repeats this from its own activity (settling).
- **Write:** when `teach` has bits, the population is clamped to `teach` and every
  pathway's active rows gain `amount` on those cells.
  - `amount` = base × (1 + gain × novelty).
  - Novelty is the popcount of the `gain` input over `gain_k`, so a scalar travels as a
    population code, still bits.
- **Options:**
  - per-pathway scaling of much-written rows: none; a counter shift as in `Hippocampus`;
    or an exact 1/n;
  - centering, which subtracts each cell's chance drive before the read-out.

**`Scatter`:** a fixed one-to-one random projection (mossy fibres: one CA3 target per
granule cell).

**`Gate`:** passes its input while a control input has any bit. It is the encoding switch
(acetylcholine-like).

**`Separate::table`:** the dentate gyrus with a fixed random fan-in table.

## The hippocampus genome (`hippocampus_genome`, 29 instructions)

| Part | Built from |
|---|---|
| EC II → DG | `SeparateTable` |
| DG → CA3 (mossy fibres) | `Scatter`, gated by `encode` → CA3's teaching code |
| CA3 | `Associate`: perforant path (shift-scaled) + recurrent settling |
| EC III → CA1 | `SeparateTable` |
| CA1 | `Associate` over the Schaffer collaterals, taught by the EC III code |
| CA1 → subiculum → EC V | `Associate`, fraction read-out, taught by `out` |
| Comparator | `BitOp(Clear)`: EC III cells that recall did not reproduce |
| Novelty-gated encoding | the comparator's output, fed back as every population's `gain` |

**One event takes two ticks, a theta cycle** (`theta_store`):
1. **Retrieval half** (`encode` empty): the cue is recalled and the comparator measures
   what was not recognised.
2. **Encoding half** (`encode` set): every population is clamped to its teaching code and
   writes. Its gain is the retrieval half's novelty, arriving on the feedback wire with
   its one-tick delay.

This is Hasselmo's account of theta: retrieval and encoding alternate within each cycle,
switched by acetylcholine (Hasselmo, Bodelón & Wyble 2002). The circuit's "recall first,
then write by novelty" falls out of the grammar's timing rule. It needs no special code.

## Results

| Test | Result |
|---|---|
| Genome vs `Hippocampus` on the 351 events of the one-shot test: novelty at every store | **identical at every event** |
| Recall from half-cues every 50 events | **identical** |
| The one-shot fact ("0 1 50" → family 51) after 50 later events | **identical, the family's bits recalled** |
| Random genomes with every leaf, including `Associate` | all build and run |

**Superposition capacity of one `Associate`** (from the concept page; right = at least 80%
of the target recovered):

| codes | events | counts | shift-scaled (as `Hippocampus`) | 1/n + centered |
|---|---|---|---|---|
| random | 4,000 | 73% | **0%** | **100%** |
| language-like | 1,000 | 0% | 81% | **89%** |
| language-like | 2,000 | 2% | 33% | **56%** |

## Findings
1. **The hippocampus is a network of base kernels, exactly.** With one more learning leaf,
   the circuit of experiments 46–49 reproduces `Hippocampus` bit for bit:
   - pattern separation;
   - mossy-fibre detonation;
   - autoassociation with settling;
   - the Schaffer and readout pathways;
   - the CA1 comparator;
   - novelty-gated encoding.

   The system's two learning rules are now its two base kernels:
   - **surprise-driven growth** (`Predictor`): cortex, the semantic store, sequence
     links;
   - **Hebbian counting** (`Associate`): the hippocampal pathways.
2. **Scalars can travel as bits.** Novelty crosses the circuit as the set of unmatched
   CA1 cells, and the population reads its popcount. Here that is exact, because the
   CA1 code has exactly k cells. This settles the open design choice of 51 for gain
   signals.
3. **The mode switch is a gate on the teaching wires,** and the order of operations is
   the grammar's timing rule. Encoding versus retrieval needs no new mechanism.
4. **A weakness in the current hippocampus shows up in isolation.** Its presynaptic
   scaling shifts each counter before summing, so a single write rounds to zero once an
   input is common. On random codes at 4,000 events it collapses to 0%. An exact 1/n
   (multiply the row's sum) together with centering holds 100%, and is the best variant
   on language-like codes too. The episodic runs still use the shift for parity. The 1/n
   + centered pathway is the candidate replacement, to be tried through the genome.

## Not in the genome yet
- **CA2's drift, novelty tags, tagged replay and free replay** (a random CA3 start).
  Replay needs a way to inject a start state, for example a `Gate` on CA3's teach
  wire.
- **The sparse binding space** of 49: hashed separation, novelty-weighted dentate gyrus
  and CA1 codes (`separate_weighted`), and split content and context. The novelty
  weighting is an `Associate`-like read with per-input weights, and should become an
  option of `Separate`.
- **The episodic harness** still calls `Hippocampus` directly.

## Next
1. Add the missing pieces (replay via an injected start, tags as a gated store of novel
   CA3 codes, weighted separation). Then switch the harness's hippocampus to the genome,
   at regression parity.
2. Try `Scale::Inverse` + centering in the hippocampus genome on the experiment 49 task.
   It should push back the attractor of common events that limits recall.
3. Then the remaining systems (role cells, relay matching, the mix, the basal ganglia) as
   leaves or genomes, using population codes for their scalars.

## Biology
- **Theta phases of encoding and retrieval:** Hasselmo, Bodelón & Wyble 2002; Hasselmo
  2006 (acetylcholine sets the mode).
- **Mossy fibre "detonator" synapses** impose the dentate gyrus's code on CA3 at encoding
  (Henze, Wittner & Buzsáki 2002; Treves & Rolls 1992).
- **Synaptic scaling and heterosynaptic plasticity** keep much-used synapses from
  dominating (Turrigiano 2008; Chistiakova et al. 2014).
