# 91 · A learned working-memory hold, without oracles or hand rules

**Question.** In [90](90-holding-an-open-question.md) (addendum 3), the learned version of the
hold chose by itself to hold the story's season and store it with later sentences. That
gained 6.6 points on the question task over three seeds. Two parts of it were still written
by hand:
- a new item could take over working memory only if it was more novel than the held one;
- the attach choice was keyed by the sentence's first two word ids (a table).

Once those parts and every oracle are removed, so that the network is a flexible learner, does
the gain hold at five seeds, and does it carry over to the suite's hippocampus entries?

**What was removed** ([rule 6](../concepts/hand-written-rules.md#rule-6)):
- the oracle hold and attach of 90 (`QHOLD=oracle`, `QATTACH=oracle`);
- the fixed rules (`QHOLD=novel`, `QATTACH=all`);
- 89's question act with its restate operator, oracle ask and subject controls
  (`QUESTION_ACT`, `QUESTION_SUBJECT`, `QUESTION_CONTROL`);
- the take-over gate;
- the word-id key.

**What is learned** (`QHOLD=learned QATTACH=learned`, [`examples/episodic.rs`](../../examples/episodic.rs)):
- **Hold:** at every bound word, the basal ganglia choose to take it into working memory or
  not. They see the hippocampus's novelty for its binding and the novelty of what is held
  (bands), and are rewarded at the story's answer less STEP_COST.
- **Attach:** at each sentence's end, they choose to store the held item with that sentence's
  event or not. The choice's code is the sentence's own word codes (shifted per action) plus
  a per-action bias, so what is learned for one sentence carries over to sentences that share
  words. Credit is tagged by recall: the attached event recalled for the answer gets the
  outcome.

Still hand-written: the query trigger ("the held item read again"). It is not exercised here.

## Results

**The question task**, learned routing, five seeds, held-out:

| | Per seed | Mean | What was held |
|---|---|---|---|
| routing alone | 69.2 / 66.6 / 69.0 / 68.6 / 69.6 | 68.6 | |
| learned hold and attach | 32.6 / 62.4 / **92.0** / 55.6 / 60.0 | 60.5 | seed 0: "the" in every story; seeds 1–4: the season |
| *(90, with the take-over gate and word-id key, three seeds)* | 76.8 / 75.0 / 72.8 | 74.9 | the season |

Errors by kind, out of 500 (right / right family wrong season / other family): seed 0 163 / 328 / 7;
seed 2 **460 / 34 / 6**; seed 3 278 / 220 / 2.

**The suite's hippocampus entries**, five seeds, held-out, recorded mean → with the learned hold:

| Entry | Recorded | Learned hold |
|---|---|---|
| hippocampus teaches cortex | 60.9 | 55.8 |
| index hippocampus | 66.0 | 55.4 |
| engram store | 67.4 | 65.8 |
| engram walk | 66.8 | 63.5 |
| engram walk only | 87.0 | **62.9** |
| inference replay | 59.4 | 60.2 |
| inference read | 45.6 | 44.4 |
| cooperate | 54.0 | 49.8 |
| relations | 62.8 | 64.4 |
| speak | 63.1 | 64.5 |
| speech motor | 62.9 | 64.4 |
| belief decides | 82.4 | 82.0 |

## Findings
1. **The flexible learner can find an excellent policy, but it does so unreliably.** On one
   seed it holds the season and attaches it well, and season errors fall from about 150 to
   34 (92% held-out). On another it locks onto "the" (33%). Three seeds hold the season but
   attach badly and end below the baseline. The mean falls to 60.5.
2. **The gain of 90 depended on the take-over gate.** The gate let only the story's first
   item take working memory, and the first word is the season. It narrowed the choice by
   hand. Removed, every bound word (about 20 a story) is a hold decision sharing one reward.
   That credit is too thin to find the policy reliably.
3. **On the suite, holding and attaching interfere with memory.** Storing the held item with
   events changes what recall returns. The engram walk alone loses 24 points and the index
   hippocampus 11. Nothing gains beyond noise.
4. Nothing becomes a default.

## Next
Better credit, not new rules:
- **Recall-tagged credit for the hold, as for the attach.** A hold is credited when the held
  item was in the cue of the recall that served the answer. Otherwise it gets only its cost.
  That gives each hold decision its own outcome instead of a share of the story's.
- **Fewer decisions by structure, not by a hand-picked rule.** Working memory is offered the
  most surprising word per sentence (the column's own surprise picks it). The choice to take
  it is learned. This is the bottleneck attention gives, not a decision about which word.
- **Attaching must not damage recall.** Store the held item as context (the store's context
  ids, which recall does not require to match), not as content.
