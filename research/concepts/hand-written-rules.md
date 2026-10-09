# Hand-written rules in the harness: an audit

**Why.** The question act of [89](../experiments/89-questions-as-inner-speech.md) restated
"the person is a smith" as "autumn is a jones". No part of the network proposed it. It came
from an edit rule written in the harness ("put the open item in place of the first k
words"), applied to an item picked by a familiarity count. The network had only chosen
among options written for it. This page lists where else the harness decides things the
network should learn, so they can be replaced rather than built on.

The harness ([`examples/episodic.rs`](../../examples/episodic.rs)) has 242 options. Each part
of it falls into one of five kinds:

| Kind | What it is | Policy |
|---|---|---|
| **Environment** | the world: stories, tasks, teacher, narrators, word codes (the senses), scoring | stays outside the network |
| **Architecture** | which circuits exist and how they connect; learning rules | designed, but expressed as genome primitives and wiring |
| **Scaffolding** | a decision inside the network written as a rule | to be replaced by a learned circuit, then removed |
| **Symbolic shortcut** | the network's vectors turned into word ids, counts or tables keyed by word | to be replaced by vector operations on codes |
| **Oracle** | a rule that uses the right answer or a label the network cannot see | only as a reference for measuring; never a default, always named as such |

## Oracles (references only)

| Option | What it knows | Status |
|---|---|---|
| `SACCADE=oracle` | looks back to the page top exactly at the answer | reference for `SACCADE=learned` (29) |
| `QUESTION_ACT=1` | asks about "the stranger", from the name lists | reference for the learned ask (89) |
| `QUESTION_SUBJECT`, `QUESTION_CONTROL` | the restatement's subject set by hand | controls (89) |

## Scaffolding: decisions written as rules

Each row says what the rule decides, what should decide it instead, and the step of the
[genome migration](genome.md#migration-plan-what-each-harness-part-needs) where it goes.

| Rule (option) | What is hand-written | Learned or plausible replacement | Step |
|---|---|---|---|
| **The rollout** (`COMPLETE=rollout*`, 44–62) | when to recall (a definite expectation contradicted), from which source in a fixed order (slot memory, semantic store, higher area, column), insertion into the word slot | inner speech: the network says its own integrated prediction into the loop ([88](../experiments/88-inner-speech.md)); a learned go/no-go for when | 6 |
| **One-word completion** (`COMPLETE=test`) | insert memory's word once per sentence, at a skipped slot | the same | 6 |
| **Question act** (`QUESTION_ACT`, 89) | ask at sentence ends about the least familiar word; restate by replacing the first k words | the open item kept active as context, bound by the hippocampus with what is read ([plan](#the-first-rebuild-the-question-act)) | 5–6 |
| **Inner speech choice** (`INNER_SAY=recall`, 88) | say memory's word if it fits, else the semantic store's, else the prediction | the integrated prediction, recall already feeding it as context | 3–5 |
| **Definite expectation** (rollout, `INNER_WHEN`) | count the words whose code overlaps the expectation by ≥ 24 bits; "definite" = exactly one | a confidence signal from the column (its winner's reliability or margin), as L5 already gives | 1 |
| **"Fits the kind"** (`CLASS_READ`, rollout, inner speech) | a memory word is kept if it overlaps the expectation by ≥ 24 bits | a thalamic AND of the vectors (the closed loop's gate, `ROLLOUT_LOOP`), no decoding | 3 |
| **Familiarity bands** (`BIND_FAM`, `FAMILIARITY`, `CUE_CTL`, the question act) | the hippocampus's stored-binding counts per (word, slot), log-binned | a novelty signal from the hippocampus's own match (CA1 comparator) or from the cortex's (perirhinal) | 5 |
| **Rarity** (`BIND_RARE`, `RARITY`, `READBACK_RARE`, semantic cue) | word frequency tables; the sentence's rarest word chosen as cue | habituation in the cue path (`BIND_HAB` already does it as a vector rule) | 5 |
| **Consolidation selection** (`CONSOLIDATE`) | which stories leave a trace (novel names at the answer), when to replay | replay tagged by surprise at encoding (`REPLAY_TAGGED`), sleep-gated plasticity | 5, 7 |
| **Semantic store reading** (`SEMANTIC`, `SEM_TYPED`) | the cue is the rarest word; the slot is computed; typed relations | a learned kernel class over the cue (`REL_COMPLETE` is a start) | 8 |
| **Relation store, Bayes** (`REL`, `BELIEF`) | relations and belief rules as algorithms | already marked as specifications (connection audit) | 8 |
| **Source ranking in the mix** (`MIX`) | sources vote with per-context reliabilities: learned, but the vote is over decoded words | the mix over vectors (each source's share of bits, as routing) | 3 |
| **Sentence-end triggers** | storage, resets, consolidation traces fire on the "." token | a boundary cell learning from surprise where events end (`EVENT_BOUNDARY=learned`, [97](../experiments/97-learned-competition-frames-boundaries.md)): 90% of its boundaries on the "." untold; costs relations and story boundary | 2 (built, not default) |
| **Fixed row layout** (column input) | [word, memory frames, top-down, previous] in fixed places | learned routing (`ROUTE`, 87), not yet the default | 4 |
| **Policies** (`POLICIES`) | memory reaches the column by a fixed policy (relay, gate, select…) | routing and the thalamic mix | 3–4 |

## Symbolic shortcuts

The network's vectors are turned into word ids in about 30 places. Lookup tables are keyed by
word ids in about 13 more. The main ones:
- **Context keys:** the mix's reliabilities, the trust gates, routing records and every basal
  ganglia context are keyed by `(previous word, current word)` ids × bands, so a context is a
  table row, not an activity pattern. A plausible context is a vector (the column's state, the
  role cell), as the basal ganglia already accept codes as candidates.
- **Decoding inside mechanisms:** `enc.decode` turns an output into a word that is then voted,
  compared or inserted. Decoding is fine for the report; inside a mechanism it hides a
  winner-take-all over the whole vocabulary that the network would need as a circuit.
- **Named tokens:** ".", "where", "is" are recognised by identity in a few places (sentence
  ends, persist questions, the question act's hand rule).

## What is not scaffolding

- **Architecture:** the column's kernels, L5 confidence, the higher area, the hippocampus
  (engram store, role cells, CA1 decoding), the basal ganglia, the cerebellum, the loops
  (efference copy, phonological loop), sleep and replay. They are designed and should be
  primitives in the genome, as the column already is.
- **Learning rules** (one-shot growth, slow growth probability, eligibility traces, the reward
  baseline) are genes.
- **The environment:** tasks, their words, the teacher, the score.

## Rule 6

Added to the [connection audit](connection-plausibility.md#rules-kept-from-here-on):
**no new decisions written as rules.** A new behaviour is a primitive plus learning. A
hand-set rule is allowed only as a labelled reference (an oracle or a specification) that the
learned version is measured against. It never becomes a default, and is never reported as
the network's own behaviour.

## The first rebuild: the question act

Without the restate operator and the oracle:
- **Holding.** An item the hippocampus finds novel (its match is weak: CA1's comparator) stays
  active in a working-memory slot of the column's row for the following sentences. Holding
  costs; a go/no-go decides, from outcomes, whether to keep or drop it.
- **Binding.** The hippocampus stores each sentence with what is active, the held item
  included, so "the person is a smith" is stored *with* lucy. Nothing is rewritten.
- **Using.** At the question, the held item's events are recalled as usual, and the fact is
  said back by inner speech only if the network's own prediction carries it.
- **Measured** against no holding, on the same task, with no oracle. The task stays: it is the
  environment.

Built in [90](../experiments/90-holding-an-open-question.md): negative. Recall at the
question matches the sentence ("lucy went to the …" from earlier stories), not the held
item's facts.

Removed after [91](../experiments/91-learned-working-memory-hold.md): every oracle and hand
rule of the question and hold line (`QUESTION_ACT` and its operators and subject controls,
`QHOLD=oracle|novel`, `QATTACH=oracle|all`, the take-over gate, the word-id key for attach).
What is left is learned, apart from the query trigger.

**The column's winner ranking** belongs on this list too: depth first (context frames in the
hand-set row layout), then reliability. It is the bottleneck of consolidation under three
learning systems ([94](../experiments/94-sleep-gated-consolidation.md), addendum 3): the
right candidate matches at 92% of answers and loses on that ranking. A hand-set
specificity rule made it worse; the replacement should be learned.


Learned replacements were tried in [97](../experiments/97-learned-competition-frames-boundaries.md):
a vote with a learned gain per depth (mixed), learned routing of the row (worse), and a
boundary cell for the period trigger (finds the sentence untold; costs two entries). The next
step is a burst gate: the row split into apical (context) and basal (input) frames, and a
source or kernel trusted when the two agree and the input confirms it.

The burst gate was built in [98](../experiments/98-burst-gate.md): the column's row split into
basal and apical frames, kernels reading both win by a trace of confirmed coincidences, and
the thalamic mix weighs sources by burst rates instead of word-keyed tables. It is not yet a
replacement: a rate per source loses the context the tables held. Bursts should vote per
moment.

Two-compartment layer 5 cells ([99](../experiments/99-two-compartment-layer5.md)) are the first
learned replacement for the winner ranking that gains on average under three systems (+2.8):
a cell bursts on its input and its context together, and a burst overrides the ranked habit.

**Default since [103](../experiments/103-primed-layer5-becomes-default.md):** primed layer 5 with
SST/VIP interneurons. Where a two-compartment cell bursts (its input arrives while its context
primes it), its prediction overrides the hand-ranked L2/3 winner; the context threshold is set
by the inhibitory circuit, not a rule. The ranking still decides where nothing bursts.
