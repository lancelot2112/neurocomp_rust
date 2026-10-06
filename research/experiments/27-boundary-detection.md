# 27 · Detecting context boundaries: when a fact is contradicted, a new story has begun

**Question.** In [26](26-context-and-readback.md), telling the areas where each story
starts (`HIER_RESET`) took the chain of three higher areas from 9–23% to 86–98% on the
season task. Can the network find those boundaries itself?

**Code.**
- `HigherArea::forget_through` in [`src/program/cortex.rs`](../../src/program/cortex.rs).
- In [`examples/episodic.rs`](../../examples/episodic.rs): `BOUNDARY`, `BOUNDARY_KIND`,
  the word context signatures, the `BOUNDARY` report, and the `BOUNDDIAG` surprise
  diagnostic.

Run: `COPY_GROW=1 GROW_TRUST=0.5 STICKY=255 GENERALIZE=0.5 GENERALIZE_AFTER=1
TRUST_AT_TEST=0.5 GROW_GATE=3 SLEEP_EVERY=500 SURPRISE_GATE=1 MEMO=1 CANON=1 TASK=season
POLICIES=nomemory HIER=1 HIER_LEVELS=3 HIER_CHAIN=mix MIX=1 BOUNDARY=1 cargo run
--release --example episodic`.

## Surprise alone does not mark the boundary
Event segmentation theory puts boundaries where prediction error spikes (Zacks et al.
2007). `BOUNDDIAG` measured the column's surprise on each test sentence:

| Task | Story-opening sentence: mean surprise | Other sentences | First word, story-opening | First word, other |
|---|---|---|---|---|
| Season | 0.34 | median 0.33 (p90 0.34) | 1.00 | median 0.35, p90 1.00 |
| Topic | median 0.54 | median 0.34 | 1.00 | median 1.00 |
| Habit | median 0.50 | median 0.34 | 1.00 | median 1.00 |

A story's first word is always fully unexpected, but so are the first words of many
other sentences (names, "then", "it"). On the season task the opening sentence is no
more surprising than any other. A threshold on surprise would fire all the time or
never.

## A boundary is a contradicted fact
What does change at a new story is that a fact is replaced: a new season while the old
one is still held. The rule:
- **A fact** is a rare word: seen in fewer than 2% of sentences so far (as for
  read-back), and surprising when it arrives.
- **Two facts are of the same kind** when the words seen just before and just after them
  mostly coincide (Jaccard ≥ 0.5). The signatures are learned online from the input:
  - every season follows "." and precedes "came";
  - every name precedes "went";
  - every place follows "the" and precedes ".".

  No categories are given.
- **A conflict.** When a fact arrives while an area's window holds a different fact of
  the same kind, the boundary lies just after the older fact. The area forgets the
  sentence holding it and every older sentence, and keeps the newer ones
  (`forget_through`). Placing the boundary as late as the evidence allows keeps
  everything that may belong to the new story.
- **When.** It is checked as each word arrives, not at the end of the sentence. A first
  version checked at the sentence end: the previous story's question ("mary went to the
  garden") then stayed in the windows until after this story's answer had been
  predicted, and the stale place misled the areas (21 / 53 / 62%). Checked at the word,
  the new name "john" contradicts "mary", and the old question is gone before "went to
  the".

## Results (held-out; seeds 0 / 1 / 2)

| Season task (0–31 filler stories) | No boundary | Boundary given (26) | **Boundary detected** |
|---|---|---|---|
| 1 higher area | 3 / 11 / 8% | 18 / 20 / 29% | 7 / 17 / 21% |
| 2 higher areas | – | 36 / 54 / 56% | 30 / 56 / 54% |
| 3 higher areas | 9 / 23 / 14% | 86 / 96 / 98% | **91 / 90 / 99%** |

With 3 higher areas and detection, by distance: 82–86% at 0 filler stories, 85–97% at
1, 91–96% at 2–3, 91–98% at 4–7, 91–99% at 8–15, 88–100% at 16+.

**Where it fires** (3 higher areas, at test):
- At 73–76% of story openings: the new season contradicts the old one. In the rest, no
  area still held the old season, so nothing stale was left.
- At about 1,150 other sentences (about one per story). Presumably most are the question
  sentence, where the new name or place contradicts the last story's.

**Other tasks,** with the higher area on (1 level, as in 24): habit 79 / 82 / 80%,
habit + memory 69 / 80 / 80%, topic 100 / 96 / 95%, give 99.6 / 99 / 100%, two-hop
94 / 81 / 62%, varied 100%. All are the same as without detection within a point. The
detector never fired there: their facts are not rare enough, or never contradicted
within reach.

**Cost.** The signatures are one small set per word, and the check runs only for rare
surprising words. Measured test time with 3 higher areas was 156–213 µs/word, against
120–130 with the given boundary. The runs shared the machine 4 at a time, so part of
that gap may be load.

## Findings
1. **The network finds its own story boundaries, as well as being told.** With three
   areas, detected boundaries give 90–99%, against 86–98% given. The detector does not
   look for stories at all. It notices that a fact it holds has been contradicted by
   a newer fact of the same kind, and drops the old context.
2. **A boundary is better placed late than early.** Forgetting through the old fact,
   rather than everything, keeps whatever came after it. Here that is the new story's
   season when the new name arrives.
3. **Timing matters as much as the rule.** The same rule checked at sentence ends
   reached only 21–62%: a stale context that survives until after the answer is as bad
   as no boundary.
4. **Limits.**
   - "Rare" is a fixed 2% threshold, relative to this task's frequencies.
   - Kinds come from immediate neighbours only, so words that share contexts by
     accident would merge.
   - A story whose facts never conflict with the last one, for example the same season
     twice, keeps its old context. That is harmless when the facts agree, but other
     stale words then survive too.

   Learning both thresholds is the obvious next step.

## Biology
- **Event boundaries** are where the current event model fails (Zacks et al. 2007;
  Kurby & Zacks 2008). Here the failure is of the model's facts, not of word-level
  prediction: a schema-level prediction error.
- **Updating working memory:** a new value of the same attribute replaces the old one,
  and older bindings are cleared. The prefrontal–basal ganglia gating account of
  O'Reilly & Frank 2006 describes the same update.
- **Boundaries separate memories:** items across a boundary are remembered as belonging
  to different events, and their order is harder to recall (DuBrow & Davachi 2013).
  Here, forgetting through the old fact is that separation, in each area's window.
