# 80 · Answer or "unknown", learned as a go/no-go

**Question.** In [79](79-unknown-when-belief-is-split.md) the network said "unknown" when
its best value was believed no more than one half, a fixed threshold that only the
posterior used well. Can the choice to answer or abstain be learned from practice instead,
for any belief rule, as a basal-ganglia go/no-go rewarded by whether answering paid off?

**Code.** [`examples/episodic.rs`](../../examples/episodic.rs):
- `PRACTICE=1`: 18 practice names, met only in training. Each has a true family and a
  conflict of one of three kinds: the first honest narrator (true) against the liar (a
  lie); the first honest narrator (true) against the second (wrong once: undecidable); a
  single honest statement. Each statement falls in a story told by the right narrator,
  before the new names' phase. Families are balanced within each kind (below).
- `BELIEF_UNKNOWN=learned`: at each sleep, after the relation store consolidates, the
  network is quizzed on every practice name it has claims about (`QUIZ_REPS`, 4). The
  go/no-go (`BasalGanglia`, one candidate per (context, action)) answers with the
  believed family or says "unknown", and the world reveals the truth: answering right is
  worth 1, answering wrong 0, "unknown" one half. Answering wins a context once its
  answers there are right more often than not.
- **The context is how strongly and how decisively the value is believed:** the band of
  the best value's belief (eighths) × the band of its lead over the runner-up (sixteenths,
  up to 7/16). The first version used the belief band alone (below).
- At test, the learned choice replaces the fixed one half for the new names' family
  questions.
- `UNKNOWN` report: the learned choice per context, the new names' contexts, the practice
  claims.

## Results (as 79: tom and lucy honest against the liar, sam two honest narrators; seeds 0 / 1 / 2)

The practice conflicts between honest narrators lower their trust (0.59 and 0.63; the
liar 0.48), so beliefs sit lower than in 79. Under the posterior, honest against the liar
is now 0.43 against 0.28; honest against honest 0.41 against 0.35.

Family questions, right / wrong / unknown:

| Rule | tom | lucy | sam (undecidable) |
|---|---|---|---|
| full | 0 / 0 / 100% | 0 / 0 / 100% | 0 / 0 / 100% |
| vote | 0 / 0 / 100% | 0 / 0 / 100% | 0 / 0 / 100% |
| graded | **94–100 / 0–6 / 0%** | **87–98 / 2–13 / 0%** | **0 / 0 / 100%** |
| posterior | **95–100 / 0–5 / 0%** | **82–98 / 2–18 / 0%** | **0 / 0 / 100%** |

What the go/no-go learned (seed 0; answered / of those right in practice):

| Rule | Context of an honest-against-liar conflict | Context of an honest-against-honest one | single honest claim |
|---|---|---|---|
| full | belief 7/8, lead 0: "unknown" (both kinds share it: 45% answered, 63% right) | same context | belief 7/8, lead 7/16: answers (100% right) |
| vote | belief 4/8, lead 0: "unknown" (shared: 45% answered, 63% right) | same context | belief 7/8, lead 7/16: answers |
| graded | belief 4/8, lead 1/16: **answers** (90% answered, 100% right) | belief 4/8, lead 0: "unknown" (6% answered, 0% right) | belief 7/8, lead 7/16: answers |
| posterior | belief 3/8, lead 2/16: **answers** (90%, 100% right) | belief 3/8, lead 0–1/16: "unknown" (3–11% answered, 0% right) | belief 4–5/8, lead 7/16: answers |

The new names land in the contexts their practice counterparts taught: tom and lucy in the
"answer" context, sam in the "unknown" one, under graded and posterior alike.

## Findings
1. **Answer or "unknown" can be learned from practice,** and with graded or posterior
   belief it is right where it answers (82–100%) and abstains exactly on the undecidable
   name (100% "unknown" on sam, every seed).
2. **Decisiveness is what separates settleable from undecidable, not belief alone.**
   With the belief band alone, the posterior put honest-against-liar (0.43) and
   honest-against-honest (0.41) in one band: half of it always right, half always wrong
   (the second honest narrator, slightly more trusted, was the one who erred). It learned
   "unknown" for both, and abstained on tom and lucy too. The lead over the runner-up
   (0.15 against 0.06) separates them.
3. **Graded belief now works as well as the posterior.** It cannot abstain with a fixed
   one half (79: sam's wrong side 0.51), but its lead (0.10 against 0.04) carries the
   difference, and the learned choice uses it.
4. **Full and vote learn that they cannot tell:** every conflict, settleable or not, is
   the same context to them (full: belief 1 with no lead; vote: 0.50 with no lead), right
   about as often as not, so they say "unknown" to every conflict, and answer only facts
   no one disputed. Better than answering wrong (79: full answered lucy wrong 85–100%),
   but it gives up tom and lucy.
5. **A frame-learning weakness, found on the way.** With families assigned at random, most
   practice statements said "smith". The relation store learns a fact's frame from its
   neighbours (positions where most agree), so "smith" became part of the frame ("X is a
   smith"), those facts no longer parsed, and every new name kept only its "jones" claim.
   Balancing the families fixed this run; the frame learner should not treat a filler's
   majority value as frame (next).
6. **Trusting everything is expensive:** full and vote validate nearly every proposal
   (288–942 per seed, 50–60% true), and each is replayed 50 times as a story (2.8 million
   extra words on one seed), so these runs take 30–60 minutes against a few for graded
   and posterior.

## Next
- Frames that survive a lopsided filler: a position whose values are the fillers of the
  same relation elsewhere is a filler, whatever its share.
- An "unknown" that prompts a search: the curiosity module ([81](81-curiosity.md)).
- Cap or weigh proposal replay, so a rule that validates everything does not dominate the
  run.
