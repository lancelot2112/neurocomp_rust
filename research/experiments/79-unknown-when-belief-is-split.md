# 79 · "Unknown" when belief is split

**Question.** With belief deciding the answer ([78](78-belief-decides-the-answer.md)),
some conflicts cannot be settled: two sources of equal standing disagree and nothing else
tells them apart. The right answer there is "unknown". Can the network say so from its own
belief, without saying it where trust does settle the conflict?

**Code.** [`examples/episodic.rs`](../../examples/episodic.rs):
- `UNDECIDED=1` (with `NARRATOR_SPLIT`): the last new name (sam, truly a smith) is stated
  once by the first honest narrator (smith) and once by the second honest narrator, wrongly
  (jones). The honest narrators earn nearly the same trust (0.71 and 0.76), so nothing
  can settle sam's family. Tom and lucy keep their honest-against-liar conflicts of 78,
  which trust can settle.
- `BELIEF_UNKNOWN=1`: at a question that names a relation the relation store holds for
  the word ("sam is a"), the answer is "unknown" unless the believed value is believed
  more than half, i.e. more likely than every alternative together (for the posterior,
  "none of these" included). The threshold is the rule's own scale, not a tuned number.
- `UNKNOWN` report: per new name, the share of family questions answered right, answered
  wrong, and answered "unknown".

## Results (as 78; sam's conflict between two honest narrators; seeds 0 / 1 / 2)

Belief per rule (trust: honest 0.71 and 0.76, liar 0.48, every seed):

| Rule | tom: smith (honest) vs jones (liar) | sam: smith (narrator 1) vs jones (narrator 2) |
|---|---|---|
| full | 1.00 vs 1.00 | 1.00 vs 1.00 |
| vote | 0.50 vs 0.50 | 0.50 vs 0.50 |
| graded | **0.59** vs 0.41 | 0.49 vs **0.51** (wrong side, just over half) |
| posterior | **0.54** vs 0.23 | 0.39 vs 0.44 (neither over half) |

Family questions, right / wrong / unknown:

| Rule | tom | lucy | sam (undecidable) |
|---|---|---|---|
| full | 65–99 / 1–35 / 0% | 0–15 / 85–100 / 0% | 72–91 / 9–28 / 0% |
| vote | 0 / 0 / **100%** | 0 / 0 / **100%** | 0 / 0 / 100% |
| graded | 100 / 0 / 0% | 90–98 / 2–10 / 0% | 1–54 / 46–99 / 0% |
| posterior | **100 / 0 / 0%** | **90–98 / 2–10 / 0%** | **0 / 0 / 100%** |

## Findings
1. **The posterior says "unknown" exactly where it should:** never on tom and lucy, whose
   conflicts trust settles, and on every question about sam, whose two honest sources
   disagree (0.44 against 0.39, neither more likely than not).
2. **Graded belief cannot abstain:** it shares belief only among the claimed values, so
   in any two-way conflict one side is at or over one half. Sam's wrong side gets 0.51,
   from a trust difference of 0.71 against 0.76 that says nothing about who is right.
   Graded then answers wrong on 46–99% of sam's questions.
3. **A vote knows only that it cannot tell:** every 1:1 conflict is 0.50, so it answers
   "unknown" to everything, including the two conflicts trust could settle.
4. **Trusting everyone never abstains** and answers lucy wrong 85–100% of the time.
5. The difference is the posterior's "none of these" term: belief is weighed against the
   chance that every source is wrong, so two weak or equal sources leave every value
   under one half, while one trusted source against a distrusted one does not.

## Next
- Learn the threshold as a go/no-go (the basal ganglia's answer-or-abstain choice of
  [72](72-go-no-go-signals.md)), rewarded for right answers and for "unknown" where the
  truth was not settled, instead of the fixed one half.
- Seek evidence on an "unknown": the open question as curiosity (a proposal to check), as
  in [74](74-proposals-and-premises.md).
- Proposals and the network's own claims as sources with earned trust.
