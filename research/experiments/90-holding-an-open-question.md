# 90 · Holding an open question, self-taught: a negative result and what blocks it

**Question.** [89](89-questions-as-inner-speech.md)'s question act was hand-written: an edit
rule rewrote sentences, and its gain turned out to be an artefact (the
[audit](../concepts/hand-written-rules.md) that followed lists such rules; rule 6 forbids new
ones). This experiment rebuilds the act from the network's own signals, with no rewriting
and no oracle. Can holding a novel item in working memory, and storing it with what is read
next, bind a stranger's fact so that it shapes the answer?

**Task.** As in 89: "winter came . lucy came . … the person is a smith . … mary is a jones
. … lucy went to the ___". The stranger's family is random in every story. Training uses
fresh strangers (`QUESTION_POOL=300`) and test uses the new names. The engram store's
settings are the engram-store entry's, without `SCHEMA_K`. Events are stored at test too.

**Code** ([`examples/episodic.rs`](../../examples/episodic.rs)): `QHOLD`, `QATTACH`,
`HELD_FIELD`.
- **Novelty:** the hippocampus's own novelty for each word's binding as it is bound (1 − the
  recall match), not a count.
- **Holding:** a binding more novel than the one held takes working memory. The reference
  is `QHOLD=novel` (take it at novelty ≥ 3/4). `QHOLD=learned`: the basal ganglia choose
  hold or ignore per novelty band, rewarded at the answer.
- **Cueing:** the held item is active, so it joins every recall cue (in a field of its own).
- **Binding:** at a sentence's end the held item may be stored with that sentence's event.
  `QATTACH=all` attaches to every later sentence. `QATTACH=learned`: the basal ganglia
  choose per sentence, keyed by its first two words (a symbolic shortcut, listed in the
  audit). Credit is tagged by recall: the attached event recalled for the answer gets the
  outcome.
- **Use:** recall returns to the column as entorhinal context (`HC_EC`), so nothing is said
  or rewritten.
- `QHOLD=oracle` holds the stranger: a labelled reference, used only to measure recall.
- `SELFDIAG` prints the cue and the recalled event at the answer.

## Results

Seeds 0 and 1, held-out (89's baseline without `HC_EC`: 68.2):

| Arm | Held-out | Item held at the answer | Attached events recalled for the answer |
|---|---|---|---|
| `HC_EC`, no holding | 53.8 / 30.2 | | |
| hold if novel, attach to all | 54.8 / 44.2 | 4 / 3 of 500 (the stranger) | 0 |
| hold if novel, learned attach | 55.4 / 27.4 | 3 / 3 | 0 |
| learned hold, learned attach | 41.6 / 49.0 | 500 / 500, never the stranger | 2 / 2 |
| *with story context (place bonus 2, current story's rows always candidates):* | | | |
| no holding | 23.4 / 22.4 | | |
| oracle hold (reference), learned attach | 25.2 / 23.2 | 500 (the stranger) | 83 / 0 |
| learned hold, learned attach | 25.2 / 21.8 | 500, never the stranger | 176 / 292 |

## Findings
1. **Recall answers the sentence, not the question.** At "lucy went to the", the
   hippocampus recalls the event most like that sentence:
   - without story context, an earlier test story's "lucy went to the bedroom" (all four cue
     words match), with another season's and family's place;
   - with the place bonus, this story's "lucy came .".

   Neither holds the fact, so a fact bound to lucy is never the one recalled. This blocked 89
   too. It also explains the weak, seed-dependent `HC_EC` baseline: the column has been
   reading old answers.
2. **Story context alone makes it worse (23%).** The suite's answers lean on recalled past
   stories with the same question, and the bonus replaces those with the current story's
   mentions.
3. **Novelty is the wrong signal for "open".** The new names recur through the test, and test
   events are stored, so a name is familiar after its first story. What should be held is an
   item unresolved *in this story*, not one never seen.
4. **The learned hold holds the wrong thing.** The first item taken blocks later ones (a
   newcomer must be more novel), so it holds an item from the opening sentences and never the
   stranger.
5. **Attaching to everything binds nothing.** With the held item on every later sentence,
   several events tie on it. Recall then picks the one that also names the item ("lucy came").

Nothing becomes a default. The hand-set act of 89 and its operators stay as labelled
references only.

## What it needs
- **A query, not a sentence, as the cue.** At the question, the held item alone (with this
  story's context) should cue recall, as a "who is lucy?" query, separate from the
  sentence-matching recall that predicts the next word. This is pattern completion from
  the item, which the engram walk already does for a bridge word.
- **"Open" as unresolved in this story.** An item with no fact bound to it yet in the
  current story, measured by recalling with the item and this story's place, not by
  lifetime novelty.
- **A takeover rule that lets a newer open item in**, such as decay of what is held, learned
  like the rest.
- **Context keys as vectors, not word ids**, so the attach choice generalises (the audit's
  symbolic shortcut).

## Addendum: the held item as its own recall query

**Build** (`QQUERY`, [`engram.rs`](../../src/program/engram.rs) `recall_here`,
`recall_here_all`). When the held item is read again ("lucy went …"), it cues the
hippocampus on its own. The cue is the item (its held-field id) plus the context (the
current story's place). The words of what it finds join the entorhinal feedback for the
rest of the sentence, and the events found share the answer's credit.
- It is context-dependent retrieval: only the current story's events are searched, so
  lucy's events from earlier stories cannot compete.
- Mechanically it is autoassociative completion: the item and the fact were stored in one
  event, and the item alone completes it. Functionally it is a paired association.
- `QQUERY=all` returns the blend of every event bound with the item here, since a cue
  matched by several traces completes to their blend.
- `QQUERY_FULL` passes the answer whole, not scaled by the cortex's uncertainty.
- Labelled references: `QHOLD=oracle` holds the stranger; `QATTACH=oracle` attaches only the
  stranger's sentence.

**Results** (held-out, per seed):

| Hold / attach / query | Held-out | What the query found |
|---|---|---|
| `HC_EC`, no holding | 53.8 / 30.2 / 67.4 | |
| oracle hold, learned attach, query | 54.8 / 29.2 | nothing: the learned attach never attaches |
| oracle hold, attach all, query (latest event) | 67.0 / 67.2 | the latest attached event; a surname in 8% |
| learned hold, learned attach, query | 66.0 / 59.6 | the hold never takes the stranger, so no query |
| oracle hold, learned attach, blended query | 67.8 / 67.2 | nothing: never attaches |
| oracle hold, attach all, blended query | 55.0 / 49.8 | every event: both families' surnames |
| **oracle hold, oracle attach, query** (upper bound) | **64.0 / 64.0 / 53.6** | the stranger's fact, 500 of 500 |
| the same, answer passed whole | 60.0 / 60.4 / 50.6 | the same |

1. **Retrieval works.** With the right event attached, the item-and-context query finds it
   in every story.
2. **Use is now the bottleneck.** Given the right fact in its context, the column answers
   right about 60% of the time, below the 68% the network reaches without entorhinal
   feedback (where it answers from votes and the rollout). Passing the fact whole does not
   help. The column does not learn to turn a surname in the entorhinal slot into that
   family's place. This is the same weakness experiment 87 found for `HC_EC` (worse wherever
   memory must carry the answer).
3. **The learned attach collapses to never attaching.** Attaching pays only once the column
   uses what is queried, and the column learns to use it only if attaching happens.
4. **`HC_EC` alone varies between runs** (30–67% on the same seeds across builds), so
   two-seed differences on it are not readable.

**Next.** The column must learn to read memory's context. Two concrete ways:
- give the queried answer its own learned channel (routing, [87](87-three-learning-systems-and-routing.md)),
  so kernels grow on it rather than on a slot shared with the sentence recall;
- consolidation: replay "lucy … smith → the smith place" so the cortex learns the
  combination.

Then the learned attach has something to learn from. The cue can also become a soft
conjunction (item match plus a same-story bonus, not a gate), so a story that says nothing
about the item falls back to what is known about it from elsewhere.

## Addendum 2: soft retrieval and a learned channel for the answer

**Build.**
- **Soft retrieval** (`QQUERY=soft`, `recall_soft` in [`engram.rs`](../../src/program/engram.rs)):
  the cue is the item and the context as a weighted match. Every event holding the item
  scores its overlap with the cue, plus a bonus of 2 if it was stored in the current story.
  The top-scoring events (the latest 8) are blended. If the story bound the item, its events
  win; if the story says nothing about it, what the item was bound to elsewhere answers. The
  hard filter of `recall_here` returned nothing in that case. A unit test covers both cases.
- **A learned channel** (`QQUERY` under `ROUTE`): the query's answer is a routed channel of
  its own (`Q_CHANNEL`), with its share and slot learned like every other channel's
  ([87](87-three-learning-systems-and-routing.md)), instead of a share of the entorhinal slot.

**Results.** Three seeds, held-out, learned routing, `QUESTION_POOL=300`:

| Hold / attach / query | Held-out | Mean | The query channel's record (seed 0: fixes / breaks) |
|---|---|---|---|
| routing, no holding | 69.2 / 66.6 / 69.0 | 68.3 | |
| oracle hold, oracle attach, query channel | 69.0 / 66.8 / 66.2 | 67.3 | +487 / −195, ranked last of 3 |
| the same, soft retrieval | 69.0 / 66.8 / 66.2 | 67.3 | the same |
| oracle hold, learned attach, soft query channel | 69.8 / 67.4 / 67.8 | 68.3 | +227 / −98; seeds 0 and 2 never attach, seed 1 attaches (661 events, all found by the query) |

1. **Routing alone is stable** (66.6–69.2%) where the entorhinal path varied from 30 to 67%,
   so this is the better base for the comparison.
2. **The right fact in a learned channel still does not reach the answer.** The query finds
   the stranger's fact in all 500 stories. Routing credits the channel a little (it fixes
   more predictions than it breaks), but held-out answers do not move (67.3 vs 68.3).
3. **Soft and hard retrieval give identical results here**, as they should on this task:
   every story binds the stranger, so the same-story events always win. Soft retrieval
   matters only where a story says nothing about the item. This task does not test that.
4. **The learned attach attaches on one seed of three.** On seed 1 it attached 661 events,
   all found by the query, and that seed still scored 67.4. On the other two seeds it never
   attaches: with nothing gained from the query, there is nothing to learn.

**Where the gap is.** Retrieval and routing work; the column does not turn a surname in a
side channel into the family's place. For trained names it never had to: in "mary smith went
to the", the surname reaches the answer through the higher area's sentence context (with the
season), not through a slot of the column's row. So the queried fact probably belongs where
the cortex already keeps context: the higher area's working context, as hippocampal output
reaches association cortex rather than the primary area. That is the next build to test,
with the oracle hold and attach as the upper bound first.

## Addendum 3: the answer in the higher area, and what the task was really testing

**Build** (`QAREA`): the queried event is reinstated in the higher areas' sentence context,
as `COOPERATE` already does with the semantic store's content. It is not sent to the
column's row. The hippocampus's output reaches association cortex through the entorhinal
cortex, so this is a wiring choice, not a rule. A diagnostic (`QERR`) splits held-out
answers by what they got wrong, and `QHOLD` reports what was held.

**Results** (learned routing, soft query, three seeds; routing alone: 68.3):

| Hold / attach | Held-out | Mean | What happened |
|---|---|---|---|
| oracle / oracle (upper bound) | 70.2 / 67.6 / 67.6 | 68.5 | the stranger's fact reinstated in every story: no gain |
| oracle / learned | 69.4 / 23.0 / 24.0 | | collapses on two seeds: with little attached here, soft retrieval falls back to lucy's facts from earlier stories, with other families |
| **learned / learned** | **76.8 / 75.0 / 72.8** | **74.9** | holds the **season** word in every story (no stranger, no query) |
| learned hold only, no attach (seed 0) | 69.2 | | holding alone does nothing |

Held-out errors on seed 0 (500 answers):

| Version | Right | Right family, wrong season | Wrong family |
|---|---|---|---|
| routing alone | 346 | 154 | 0 |
| oracle hold and attach, answer in the area | 351 | 149 | 0 |
| learned hold and attach | 384 | 113 | 3 |

1. **The task never tested binding.** Without any question act, the network never names the
   other family's place: it already finds the stranger's family. Most likely this is a
   shortcut in the task's design: "the person is a …" always carries the stranger's fact.
   Every binding mechanism of 89–90 could only help with something that was not wrong.
   That explains why none of them showed a gain, whatever the path.
2. **The real deficit is the season**, stated once at the story's start. Every error is the
   family's place for another season.
3. **The self-taught hold found that** (but see [91](91-learned-working-memory-hold.md):
   with the hand-set take-over gate and word-id key removed, it finds it on one seed of
   five, and the mean falls). Learning from the answer alone, the basal ganglia
   hold the season word and store it with chosen later sentences, and season errors fall
   from 154 to 113. The gain (+6.6 points) holds on all three seeds. Holding without
   storing does nothing, so the gain comes through the hippocampus: later events carry the
   season, and recall brings it back near the question. This is the first behaviour in
   88–90 that the network chose for itself and that helped.
4. **Soft retrieval's fallback is a liability when names recur with different facts.** An
   item that this story binds weakly recalls what it meant in other stories.

**Next.**
- Confirm the learned hold at five seeds and on suite entries where long-range context
  matters (story boundary, saccades, season distance).
- For binding itself, a task that cannot be solved by such a shortcut. For example, two
  strangers in a story, each named in a fact that only its position or order distinguishes.
- Report the hold's choice in the context terms of the audit: it is keyed by novelty band,
  not by word, so it generalises.

