# 72 · What the speak/stay-silent choice should see: novelty and agreement

**Question.** In [71](71-speech-routing.md) the basal ganglia's choice to speak saw only the
source mix's confidence band, learned on questions about trained names, and kept speaking
at middle confidence where new names were mostly wrong. Given two more signals (how new
the question is, and whether the sources agree), does it learn to stay silent where it
would be wrong?

**Code.** [`examples/episodic.rs`](../../examples/episodic.rs):
- `SPEAK_CTX=novelty|agree|novelty,agree` (with `SPEECH=motor`): the go/no-go's context is
  the confidence band × the novelty band of the sentence's least familiar word (four
  bands) × whether every source in the mix supports the chosen word.
- Novelty from the hippocampus's counts (default), or `FAMILIARITY=cortex`: from exposure,
  the number of sentences read that held the word (replays not counted). Exposure is the
  usual model of perirhinal familiarity, which survives a hippocampal lesion.
- `FAMILIARITY=kernels` (kept as a failed measure): the number of column kernels keyed on
  the word.
- The go/no-go has random draws of its own, so that what it sees does not change the
  rest of the training run.
- Silence: the network says nothing, its turn is scored, and it then hears the answer
  from the page, as the other speaker gives it in a dialogue (`SILENCE=gap`: no word).

## Results (family stated once; seeds 0 / 1 / 2; cooperation + replay + relation store; motor speech)

Held-out (new-name) questions. Every row trains the same network; only the go/no-go
differs (test answers when always speaking: 86 / 93 / 91% intact, 65 / 65 / 65% lesioned).

| Hippocampus | Go/no-go sees | Answered | Right of answered |
|---|---|---|---|
| intact | confidence | 100 / 100 / 100% | 85.8 / 93.2 / 90.8% |
| intact | + hippocampal novelty + agreement | 100 / 42 / 50% | 85.8 / **97.6 / 97.6%** |
| intact | + cortical novelty (exposure) + agreement | 87 / 78 / 81% | 84.2 / **93.8 / 91.9%** |
| lesioned | confidence | 94 / 98 / 97% | 67.4 / 66.0 / 66.7% |
| lesioned | + agreement | 100 / 100 / 97% | 64.6 / 65.0 / 66.7% |
| lesioned | + cortical novelty + agreement | 92 / 70 / 74% | 68.4 / 60.2 / 59.2% |
| *fixed threshold 0.9 (69)* | lesioned | 65 / 64 / 58% | 84 / 87 / 93% |

When the sources agreed (intact): 97–98% right; when they disagreed: 79–83%.

Cortical familiarity at the end of the test: new names 151–182 sentences, trained names
729–795. The kernel count was backwards: new names had 36–39 kernels keyed on them,
trained names 14–25 (surprise-driven growth and 50 replayed readings grow kernels for
exactly the new names).

## Findings
1. **With the hippocampus, novelty and agreement let the go/no-go abstain where it
   should.** On two seeds it speaks on 42–50% of new-name questions and is right on 97.6%
   of them. On seed 0 it learned nothing (it speaks always): the newest band never
   occurs in training, so its values stay at their start.
2. **Agreement is the most informative single signal:** answers are 97–98% right when
   every source supports them, 79–83% when they do not.
3. **Cortical familiarity (exposure) works with the hippocampus intact,** more gently:
   78–87% answered, 92–94% right on two seeds. Exposure rises during the test (each new
   name is read in about 170 stories), so new names move from the newest band to the
   middle ones as they are read about.
4. **Without the hippocampus, neither signal helps.**
   - Agreement never fires: the lesioned mix's sources (column, higher area, relation
     store, class vote) never all support one word.
   - Exposure novelty makes it worse on two seeds (59–60% right): its newest bands have
     no training questions, so their values are untrained.
   - The fixed threshold of 69 remains the better abstention without the hippocampus.
5. **The basal ganglia learn only in contexts that training visits.** New names are, by
   definition, rare in training; a context that never appears there keeps its initial
   value. Calibration for the new needs either training questions about once-seen items,
   or a prior that treats unvisited contexts as risky.
6. **Two mistakes found on the way:**
   - kernels keep their inputs as a sparse bit list, so the first kernel-familiarity
     measure read 0 everywhere;
   - a shared random generator made each go/no-go variant a different training run. A
     seed that trained badly looked like "silence damages later answers"; it was not.

## Next
- Give unvisited contexts a cautious prior, or train the go/no-go on questions about
  facts stated once.
- A lesioned agreement signal: agreement among the cortical sources that remain.
