# 64 · Graded gating: both stores always answer

**Question.** In [63](63-routes-cooperate.md) the semantic store was consulted only where
the column was unsure (a binary gate at half reliability). The hippocampus is already
cued on every word, and its candidate is weighed in the source mix by the column's
confidence band. If the semantic store also always answers, and the column's confidence
grades how much each answer counts, does that beat the binary gate?

**Code.** [`examples/episodic.rs`](../../examples/episodic.rs):
- `GRADED=1`: the semantic store posts a candidate to the source mix at every step. The
  candidate is the higher area's prediction with the store's content for the sentence's
  least familiar word added to its context (peeked: no side effects). It is keyed by the
  column's confidence band, as the hippocampus's candidate now is. Each candidate has keys
  of its own (a context has room for only 8 offsets; the first version shared them; with
  separate keys the intact run gave the same figures).
- `GRADED=enrich`: the store's content always joins the higher area's context, each of
  its bits with probability 1 − the column's confidence (a fixed hash per bit and step:
  a sure column lets almost nothing in).

## Results (family stated once; seeds 0 / 1 / 2; semantic store + walk on; rollout off)

| Gate | Replay (62) | Hippocampus at test | Held out | Trained |
|---|---|---|---|---|
| binary, where the column is unsure (63) | 50× | lesioned | **56 / 62 / 60%** | 58.1% |
| graded, both in the mix | none | lesioned | 33 / 62 / 34% | 59.8% |
| graded, both in the mix | 50× | lesioned | 45 / 39 / 57% | 65.1% |
| graded enrichment | none | lesioned | 34 / 21 / 44% | 60.6% |
| graded enrichment | 50× | lesioned | 43 / 24 / 48% | 63.6% |
| binary (63) | 50× | intact | **88 / 93 / 86%** | 87.7% |
| graded, both in the mix | 50× | intact | 88 / 90 / 71% | 71.1% |
| graded enrichment | 50× | intact | 87 / 89 / 83% | 89.0% |

## Findings
1. **Graded gating is worse than the binary gate, both ways.**
   - Lesioned, the best graded figure is 47% (binary 59.5%).
   - Intact, the trained names fall too (71% in the mix).
2. **In the mix, the semantic candidate's reliability is learned on trained stories.**
   There the column is sure and the candidate is redundant, so its weight stays low where
   it would matter (new names). It answered at about 285,000 steps per seed, almost all
   where it adds nothing.
3. **Enrichment floods the context, as surprise-gated consultation did in 63.** Even at
   high confidence some of the store's bits get in at every step, and "dog" → "the ran
   away" is noise for the higher area. The threshold is doing real work: the cortex
   should ask memory *at* uncertainty, not a little everywhere.
4. **What the brain suggests instead:** hippocampal output is weighted by prediction
   error, but the weighting acts on a recall that the cue already made specific. Here the
   semantic store's answer is unspecific (a bag for a word), so grading its strength does
   not make it relevant. That points to the cue (65) and to the store's content (66).

Not kept as a default; the flags stay for comparison.
