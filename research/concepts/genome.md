# The network as a genome

**Goal.** The reading network should be described, not hand-wired: one genome that
specifies the initial configuration (which circuits exist and how they connect) and the
update loop (what happens at each step and each clock event), built from a small set of
base kernels, the same for every task. The harness then only supplies stories and scores
answers. This is the roadmap's "grown, not designed" item and rule 5 of the
[connection audit](connection-plausibility.md#rules-kept-from-here-on).

## The format

A genome is a text file ([`genomes/`](../../genomes)) read by `Genome::parse` and built by a
stack machine ([`src/program/modules.rs`](../../src/program/modules.rs)). Nothing the
network needs is set in code or in environment variables: every size, threshold and
learning rule is a number in the genome.

- **Definitions.** `def <name> <inputs>` … `end`. The last definition is the whole
  network; `sub:<name>` places an earlier one inside it.
- **Two stacks.** Numbers go on the number stack (`8192`); signals go on the signal stack
  (`in:0` is the word, `in:1` the sentence clock). `dup swap drop over pick:n` move
  signals; `feedback` … `close` make a loop (reading a later module gives its value from the
  last step); `out` exposes a signal; `zero` is an empty slot.
- **Kernels.** `make:<kind>` places a base kernel, taking its numbers from the number stack
  and then its signal inputs: `8192 3 16 make:predict` is a predictor of 8,192 bits reading
  3 frames, sampling 16 bits a frame. Kinds: `predict`, `concat`, `delay`, `gate`, `and`,
  `or`, `xor`, `clear`, `bag`, `window`, `surprise`.
- **Genes.** `set:<gene>` takes a number into the definition's learning rule, used by the
  predictors it places next (`1 set:surprise_gate`, `100000 set:max_kernels`), and, in the
  top definition, its update loop (`1 set:reset_at_story`, `500 set:sleep_every`).
  Fractions are integers over 65,536.
- **The update loop.** Every word the network ticks once. The world's clock reaches it two
  ways: as a wire (input 1 is on at the first word after a sentence ended, so a module that
  keeps a sentence's content knows to move on) and as the schedule's resets and sleeps.

[`src/program/reader.rs`](../../src/program/reader.rs): `Reader` runs a genome in four calls,
`read(word)`, `end_sentence()`, `end_story()`, `start_test()`. It knows nothing about the
task.

[`examples/episodic.rs`](../../examples/episodic.rs): `GENOME=genomes/<file>.gen` runs the
harness's stories through a genome instead of the hand-built network and prints the same
report, so any genome can be compared with a suite entry.

## The genomes so far

- [`column.gen`](../../genomes/column.gen): `[word | empty slot | previous word]` →
  predictor. **Identical** to the harness's column (`POLICIES=nomemory`, no higher area,
  no mix): the same kernels and the same answers (give task, seeds 0 and 1: 8.4 / 8.8%,
  10.0 / 8.2%).
- [`hierarchy.gen`](../../genomes/hierarchy.gen): the column with a higher area (experiment
  24). `surprise` (the column's word-level surprise: the word's share of its last
  prediction × its confidence, both brought back through loops) feeds a `window` of the
  last four sentences' surprising words; with the sentence so far (`bag`) it is the area's
  input; the area is taught only by surprising words and its prediction is the column's
  top-down slot. Against the harness's `HIER=1` (one area, no mix), habit, seeds 0 and 1:
  harness 79.8 / 84.4% and 69.4 / 68.2% (seen / held-out), genome 91.2 / 93.0% and
  71.6 / 74.2%. **Close, not identical.** The known difference: at a story's end the genome
  clears the column's activity (its previous word and last prediction); the harness carries
  them into the next story (the column grows 592 kernels against the harness's 420).
  Season is 0% in both (one area, no story boundary: as in 25).
- Finding the gap showed a hidden constant: the kernel class's default cap of 1,024
  kernels, which the harness overrode. The cap and the match and surprise fractions are
  now genes, set in every genome file.

## Migration plan: what each harness part needs

Each step adds the primitives it needs, expresses one system as a definition, and checks
parity with the harness setting it replaces before going on.

| Step | System | Primitives to add | Parity target |
|---|---|---|---|
| 1 | Column | numbers, genes, `make:predict`, `zero`, the text format, `Reader` (done) | nomemory column: identical |
| 2 | Higher area | the sentence clock as input 1; `bag`, `window`, `surprise`; a predictor's confidence as a third output (done) | `HIER=1`, one area, no mix: close (above) |
| 3 | Thalamic mix | `Mix(n)`: n proposal ports and a context port; each source's record per context; outputs the combined word and its confidence; taught by the next input | `MIX=1` |
| 4 | Routing | `Share`: a channel scaled by its record (same-pass credit from depth-limited predictions) | `ROUTE=1` |
| 5 | Hippocampus | `Engram` (store, recall by cue, walk, replay at sleep, source tags) wrapping the engram store; `Roles` (role cells); `Decode` (CA1: recall → word codes); a `Replay` action in the schedule | `HIPPO=engram`, `BIND=1`, `HC_EC=1` |
| 6 | Basal ganglia | `Select` (go / no-go over candidate codes, eligibility traces, reward port) | learned stepping, answer-or-unknown |
| 7 | Cerebellum, slow cortex | none: `PredictWith` with fast and slow specs | `LEARNING=three` |
| 8 | Relation store, Bayes, speech | wrapped as primitives first, each marked as a specification to be replaced by circuits | the relation, belief and speech entries |
| 9 | One genome for all tasks | the whole suite from a single genome; then search over mutations | the suite within noise |

**What stays outside the genome:** the world (stories, questions, the teacher), the word
codes (the senses), and the score. Everything the network does with them belongs in it.
