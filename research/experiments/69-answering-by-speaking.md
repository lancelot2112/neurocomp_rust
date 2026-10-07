# 69 · An output buffer: answering by speaking, with abstention

**Question.** So far the network only listens: its "answer" is the next-word prediction
scored against the page. Can the cortex write its answer to an output buffer instead,
hear its own word in place of the page's, and say "unknown" when unsure? This is the
first step of the [output plan](../concepts/output-and-self-supervision.md); speaking is
committing to one word, the sparsest message the network can send.

**Code.**
- [`src/program/speech.rs`](../../src/program/speech.rs), `OutputBuffer`:
  - `speak(word, confidence, truth, tag)`: writes the word, or "unknown" (None) under the
    threshold;
  - `score` and `score_at(threshold)`: accuracy and coverage, at the threshold used or at
    any other (so one run gives the whole accuracy–coverage curve);
  - unit test `speaks_abstains_and_scores`.
- [`examples/episodic.rs`](../../examples/episodic.rs): `SPEAK=1`, `SPEAK_MIN` (default
  0).
  - At a test question the network speaks the source mix's word, with the mix's
    confidence (else the column's).
  - The page's answer is not read: the spoken word comes back as the next input in its
    place, marked as an internal step (nothing learns from it as the world's word).
    "unknown" is heard as ".".
  - The `SPEAK` report gives accuracy, coverage, the curve, the kinds of wrong answers,
    and sampled transcripts.
- Regression entry `speak`. Every earlier entry is unchanged.

## Results (family stated once; seeds 0 / 1 / 2; cooperation + replay + relation store)

| Setting | Hippocampus | Held out, all questions | Trained |
|---|---|---|---|
| reading the answer (67) | lesioned | 65 / 65 / 64% | 62.5% |
| **speaking the answer** | lesioned | **65 / 65 / 65%** | 62.4% |
| speaking the answer | intact | 86 / 93 / 91% | 88.8% |

Accuracy against coverage for held-out questions, hippocampus lesioned (from the same
runs, scored at each threshold):

| Speak only at confidence ≥ | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| 0 (always) | 100% answered, 64.6% right | 100%, 65.0% | 100%, 64.8% |
| 0.5 | 94%, 67.4% | 99%, 65.7% | 97%, 66.7% |
| 0.8 | 76%, 77.7% | 90%, 70.6% | 65%, 86.8% |
| 0.9 | 65%, 84.1% | 64%, 88.2% | 59%, 91.2% |

With `SPEAK_MIN=0.9` (actually abstaining): 65 / 64 / 58% answered, 84 / 87 / 93% of those
right.

Wrong answers, held out, lesioned, speaking always: another place 128 / 158 / 152, "."
23 / 8 / 16, another word 26 / 9 / 8 (of 500 questions per seed).

Transcripts (seed 0): `["tom", "went", "to", "the"] -> said "hallway" (page: hallway)`;
`["lucy", "went", "to", "the"] -> said "bedroom" (page: bathroom)`.

## Findings
1. **The network answers by speaking as well as by predicting the page** (64.8% against
   64.4%), and hears its own answer without harm.
2. **Abstention works from the confidence alone, with no training.** Answering only at
   0.9 or above keeps about 60% of the questions at 84–91% right, against 65% when it
   always answers.
3. **Its errors are mostly the wrong place,** a place of the family in another season or
   of the other family, not a refusal (".") or a word of the wrong kind. The class is
   right; the binding to the season is what fails.
4. **With the hippocampus intact, the confidence does not separate right from wrong:**
   almost every answer is spoken at 0.9 or above, and 86–93% are right. The mix is
   overconfident when memory agrees with itself; abstention is useful only where the
   cortex answers alone.

## Next
- **The efference copy:** a `self` frame in L4 while the network's own word comes back,
  so the predicted input is not surprising, and a mismatch between what was meant and
  what was heard is a real error.
- **Recitation:** retell a story from a cue, each spoken word driving the next.
- **Read-back as training:** regenerate each sentence or window and learn from the
  mismatch with what was read, for every area ([25](25-area-chain.md),
  [26](26-context-and-readback.md)).
- **Calibrate the mix when memory is involved,** so the intact network can abstain too.
