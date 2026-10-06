# 43 · Learned stepping: the basal ganglia decide when to look again

**Question.** The internal rollout of [40](40-family-stated-once.md)–[42](42-semantic-store.md)
("tom went to the" → "tom [is a smith] went to the") starts by a hand-set rule: the page
contradicts a *definite* expectation. The [roadmap](../roadmap.md) (stage 3) calls for the
basal ganglia to choose between "answer" and "look again", rewarded for being right and
charged a small cost per step. Can that choice be learned? And should the sources of a
step's word be *mixed*, like the sources of an answer ([24](24-cortical-hierarchy.md)),
instead of taken in a fixed order?

**Code.** In [`examples/episodic.rs`](../../examples/episodic.rs): `STEP=learned`,
`STEP_TEST_LEARN`, `STEP_COST`, `SEMANTIC_MIX`, `ROLLOUT_MIX`, `FAMILY_SHORT`, and the
`STEP` report. The settings are the `learned-stepping` entry of
[`scripts/regress.tsv`](../../scripts/regress.tsv).

## Learned stepping
- **Where a choice is offered:** wherever the page contradicts the column's expectation
  (the next word is not one it expects), and some source offers a word of the expected
  kind. There is no "definite" gate: every contradiction is a choice point.
- **The choice:** a basal-ganglia selector ([15](15-basal-ganglia-selector.md)), with two
  candidates per context, "look again" (start a rollout) and "read on".
- **The context** is coarse and general:
  - whether the expectation was definite (one word);
  - the column's confidence, in three bands;
  - whether the sentence is novel (familiarity band);
  - which source offers the word: slot memory, semantic store, higher area or column.
- **A started rollout runs on** until the page fits again, as before. A first version
  let the basal ganglia decide every step of a rollout. A three-step chain then needed
  three right choices in three different contexts, all credited by one answer, and it
  learned poorly.
- **Reward:** every choice in a story is rewarded at its answer: 1 if right, else 0,
  minus `STEP_COST` (0.05) for looking again. Each candidate's value moves toward the
  rewards it received, a per-context bandit.
- **When it learns:** in training, and at test with `STEP_TEST_LEARN`, the way the mix
  keeps learning at test ([36](36-slot-binding-memory.md)).

## Two fixes on the way
- **The cortex learns from the page only.** When rollouts also ran in training, the
  column and higher area learned an internal step's word as if read. They learned their
  own predictions, and trained names fell 2–4 points. Now no cortical learning happens
  when the target word is an internal step.
- **The mix does not score sources on internal steps.** Their "next word" is the
  network's own, so scoring would be self-confirming. This changed experiment 40's
  regression figures by a fraction of a point (67.2 → 67.6% trained, 41.6 → 41.8% held
  out), re-recorded.

## Results (seeds 0 / 1 / 2; family stated once; semantic store and answer-trace consolidation of [42](42-semantic-store.md))

| | Held-out right | Family supplied correctly | Trained names |
|---|---|---|---|
| No rollout | 19 / 15 / 43% | – | 68% |
| Hand-set trigger, at test (42) | **69 / 69 / 60%** | 98 / 100 / 97% | 67% |
| Hand-set trigger, also in training | 51 / 53 / 54% | 95 / 98 / 100% | 67% |
| Learned, choice at every step, training only | 23 / 15 / 27% | – | 66% |
| Learned, choice at rollout start, training only | 24 / 50 / 43% | 97 / 96 / 100% | 65% |
| Learned, choice at rollout start, also at test | 50 / 52 / 52% | 98 / 96 / 100% | 66% |
| **Learned, rollouts and learning at test** | **57 / 54 / 55%** | 100 / 100 / 100% | 68% |
| Hippocampus off: hand-set trigger | 67 / 67 / 61% | 100 / 100 / 100% | 66% |
| **Hippocampus off: learned, at test** | **61 / 56 / 62%** | 100 / 100 / 100% | 65% |

What it learned (rollouts at test, hippocampus off): it accepted the semantic store's
offers 86–92% of the time, the column's 12–15%. Intact, it accepted the store's 55–88%
and the slot memory's 18–50%.

**Mixing** (hand-set trigger unless stated):

| | Held-out right, intact | Held out, hippocampus off |
|---|---|---|
| Fixed order: slot memory → semantic store → higher area → column | 69 / 69 / 60% | 67 / 67 / 61% |
| Semantic store also votes in the answer mix (`SEMANTIC_MIX`) | 69 / 69 / 60% (identical) | – |
| Rollout word from the mix (`ROLLOUT_MIX`) | 47 / 69 / 59% | 45 / 47 / 40% |
| Learned stepping + rollout word from the mix | 39 / 60 / 59% | 41 / 36 / 38% |

## Findings
1. **The choice of when to look again can be learned, and it recovers most of the hand
   rule's benefit.**
   - With the hippocampus off, learned stepping reaches 56–62% against the hand trigger's
     61–67%, from 15–43% with no rollout.
   - It learned which offers to trust: the semantic store's almost always, the column's
     guesses rarely.
2. **But it learns mostly at test.** Trained on training stories alone, it reaches 15–50%.
   - Training holds almost no situation where looking again pays: trained names always
     come with their surname, and only the three new names lack one.
   - Making a quarter of trained questions omit the surname (`FAMILY_SHORT=0.25`) did not
     change that (45–59%). The column simply learned to answer "mary went to the"
     directly.
   - A learner needs experience where thinking helps. This world has little.
3. **Starting is the right unit of choice.** Deciding each step separately failed
   (15–27%): a chain is only rewarded whole. Deciding to start and letting the rollout run
   until the page fits again works. "Look again" is one act, not three.
4. **Internal steps must not teach the cortex.** Rollouts during training cost held-out
   accuracy (51–54% against 60–69% when they run only at test), even with that learning
   switched off. They change what the slot memory stores and what the higher area holds.
   Thinking and reading are best kept apart in what they write.
5. **Mixing the rollout's sources failed, for a principled reason.**
   - The mix learns each source's reliability by checking its word against the next
     page word.
   - Inside a rollout there is no page word, so the semantic store never earns weight
     there, and the column's confident guess outvotes it. With the hippocampus off,
     held-out answers fall to 38–47%.
   - The fixed order works because it encodes "specific memory before generic guess".
     A mix would need reliabilities credited from the final answer, as the basal ganglia
     are.
6. **The semantic store as an answer voter changes nothing here.** For "tom" it holds
   {is, a, smith}. None of those is a place, so it never has a word of the expected kind
   at the answer. Its use is inside the chain, supplying the family.

## Should the sources be mixed?
In principle yes. That is how the answer is formed, and the thalamic mix is the
architecture's general rule for combining sources ([24](24-cortical-hierarchy.md)). But a
mix is only as good as its reliabilities, and those come from checking each source against
what happens next. Inside a chain of thought nothing happens next on the page, so the
reliabilities must come from the outcome of the whole chain.

The next step is outcome-credited reliabilities for inner steps: each source that supplied
a rollout word gets credit or blame when the answer is scored. This is the same signal the
basal ganglia learn from. Until then, the fixed order (episodic memory, then semantic,
then the cortex's own guess) is the better rule.

## The semantic store in the brain
The semantic store is not entorhinal.
- **Entorhinal cortex** is the gateway between the hippocampus and neocortex.
  - Its medial part supplies structural codes (here the slot cells of
    [36](36-slot-binding-memory.md)).
  - Its lateral part supplies content and temporal context (here the fading state of
    [35](35-fading-state-and-entorhinal-codes.md)).
  - Replay passes through it on the way out to cortex. It is the conduit, not the store.
- **The store's counterpart is the anterior temporal lobe**, the "hub" of semantic memory
  (Patterson, Nestor & Rogers 2007). It holds what is known about an entity across
  episodes. Semantic dementia, which damages it, removes that knowledge while episodic
  memory for recent events can be spared, the reverse of hippocampal amnesia.
- **Perirhinal cortex**, next to the entorhinal, holds item-level codes and familiarity:
  is this item known? The familiarity bands of [36](36-slot-binding-memory.md) are
  perirhinal-like.

So a fact flows: episode (hippocampus) → replay through entorhinal cortex → semantic
store (anterior temporal) → read back during thought by the rollout.

## Next
- **Outcome-credited reliabilities for inner steps,** so the rollout's sources can be
  mixed.
- **A world where thinking pays in training:** multi-step questions (roadmap stage 6),
  so the basal ganglia learn to look again before the test.
- **A "give up" option:** abstention (roadmap stage 1) as a third choice beside "look
  again" and "read on".

## Biology
- **Cortico-basal ganglia loops gate cognitive actions** as they gate movements (Frank,
  Loughry & O'Reilly 2001; O'Reilly & Frank 2006). Whether to retrieve, hold or answer is
  selected and reinforced by dopamine.
- **The value of computation:** thinking has a cost, and it is worth it only where it
  changes the outcome (metareasoning; Lieder & Griffiths 2017). The step cost here is
  that cost.
- **Semantic memory and its hub:** Patterson, Nestor & Rogers 2007; Lambon Ralph et al.
  2017. **Perirhinal familiarity:** Brown & Aggleton 2001.
