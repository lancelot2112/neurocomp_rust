# The network as a genome

**Goal.** The reading network should be described, not hand-wired: one genome that
specifies the initial configuration (which circuits exist and how they connect) and the
update loop (what happens at each step and each clock event), built from a small set of
base kernels, the same for every task. The harness then only supplies stories and scores
answers. This is the roadmap's "grown, not designed" item and rule 5 of the
[connection audit](connection-plausibility.md#rules-kept-from-here-on).

## The format

[`src/program/modules.rs`](../../src/program/modules.rs):
- **Structure.** A `Genome` is a library of definitions written in a stack language
  (`NetOp`): push inputs, place base kernels (`Prim`) or earlier definitions (`Sub`),
  close loops (`Feedback` / `Close`), expose outputs. The last definition is the whole
  architecture. Any instruction sequence builds a valid network, so genomes can be
  mutated ([51](../experiments/51-networks-of-kernels.md)). `Zero` reserves an empty
  frame (a slot of a `Concat`).
- **Learning rules as genes.** `KernelSpec`: generalisation, the surprise gate, canonical
  kernels, memo, the uncertainty growth gate, slow-learning probability, sticky synapses,
  copy growth, growth trust, trust floor, replay length: what the harness set from
  environment variables. `Prim::PredictWith` places a predictor with a spec.
- **The update loop.** Every word, the network ticks once, with slow learning on until
  the test. Besides that, the genome's `Schedule` says three things: clear activity at the
  end of a sentence, clear it at the end of a story, sleep every n stories. Learning
  signals travel on wires (a predictor's teaching port), so nothing outside calls "learn".

[`src/program/reader.rs`](../../src/program/reader.rs): `Reader` runs a genome in four
calls: `read(word)` (one tick; returns the prediction), `end_sentence()`, `end_story()`,
`start_test()`. It knows nothing about the task.

[`examples/episodic.rs`](../../examples/episodic.rs): `GENOME=column` runs the same
stories through a genome instead of the hand-built network and prints the same report, so
any genome can be compared with a suite entry.

## Parity so far

| Genome | Harness setting it replaces | Result |
|---|---|---|
| `column`: `[word, empty slot, previous]` → predictor with the harness's genes, reset at each story, sleep every `SLEEP_EVERY` stories | `POLICIES=nomemory` without `HIER` or `MIX` | **Identical**: same kernels (101) and same answers on the give task, seeds 0 and 1 (8.4 / 8.8%, 10.0 / 8.2%); season and habit 0% in both (a column alone cannot carry the context) |

## Migration plan: what each harness part needs

Each step adds the primitives it needs, expresses one system as a definition, and checks
parity with the harness setting it replaces before going on.

| Step | System | Primitives to add | Parity target |
|---|---|---|---|
| 1 | Column | `PredictWith`, `KernelSpec`, `Zero`, `Schedule`, `Reader` (done) | nomemory column (done) |
| 2 | Higher area | a sentence-end input port (the world's clock as a wire, so modules respond by wiring, not by hooks); `Bag` (OR of the sentence so far, cleared at a sentence end); `Window(n)` (OR of the last n sentences' contents, advanced at a sentence end); `Surprised` (word-level: did the prediction miss the input, as a flag); `Gate` exists (teach only on the column's residual) | `HIER=1`, one area, no mix |
| 3 | Thalamic mix | `Mix(n)`: n proposal ports and a context port; each source's record per context; outputs the combined word and its confidence; taught by the next input | `MIX=1` |
| 4 | Routing | `Share`: a channel scaled by its record (same-pass credit from depth-limited predictions) | `ROUTE=1` |
| 5 | Hippocampus | `Engram` (store, recall by cue, walk, replay at sleep, source tags) wrapping the engram store; `Roles` (role cells); `Decode` (CA1: recall → word codes); a `Replay` action in the schedule | `HIPPO=engram`, `BIND=1`, `HC_EC=1` |
| 6 | Basal ganglia | `Select` (go / no-go over candidate codes, eligibility traces, reward port) | learned stepping, answer-or-unknown |
| 7 | Cerebellum, slow cortex | none: `PredictWith` with fast and slow specs | `LEARNING=three` |
| 8 | Relation store, Bayes, speech | wrapped as primitives first, each marked as a specification to be replaced by circuits | the relation, belief and speech entries |
| 9 | One genome for all tasks | the whole suite from a single genome; then search over mutations | the suite within noise |

**What stays outside the genome:** the world (stories, questions, the teacher), the word
codes (the senses), and the score. Everything the network does with them belongs in it.
