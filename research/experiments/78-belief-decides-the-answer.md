# 78 · A task where the believed fact decides the answer

**Question.** The Bayes module learns whom to trust and resolves conflicting facts
([75](75-bayes-module.md)–[77](77-infer-from-every-statement.md)), but on the reading task
the answers never depended on it: the questions asked a new name's place, and the network
answered from the stories' patterns. When the question asks the contested fact itself, and
a vote cannot settle the conflict, does the believed value become the answer, and do the
belief rules then differ?

**Code.**
- [`examples/episodic.rs`](../../examples/episodic.rs):
  - `BELIEF_Q=1` (with `FAMILY_STATED`): half the held-out questions about a new name ask
    its family ("tom is a ___"), the other half its place. Reported apart (`BELIEF_Q`).
  - `NARRATOR_SPLIT=1` (with `NARRATORS`, `LIAR`): the statements about each new name
    alternate between the liar, who always lies in them, and the first (honest) narrator,
    so each new name's family is a 1:1 conflict (`SCHEMA_K=2`: one statement each). Which
    comes first alternates by name. Trust can only come from the other facts: the trained
    names' families, where the liar lies with probability `LIAR` among honest narrators.
  - `BELIEF` report: per new name, the believed family, the relation store's answer, and
    the truth. `BQDIAG=1`: each source's vote at family questions.
- Two fixes that made the store's answer reach the answer at all (`SEMANTIC_MIX=1`: the
  relation store votes in the source mix with its own learned reliability):
  - its vote was kept only if the column expected a word of that kind, read from
    `expect_prev`, which is only updated while the hippocampus is intact (here it is
    lesioned at test). It is now read fresh (from the column's cached response);
  - a question that names a relation the store holds for the word ("tom is a": family)
    asks for that relation's answer, of its kind by construction. That answer is now kept
    even where the column, sure of another word, does not expect it.
- [`src/program/bayes.rs`](../../src/program/bayes.rs), a fix: a key was many-valued (not
  a conflict) as soon as one source gave it two values. A liar who lies 3 times in 4 gives
  "mary" both families over training, so every trained-name conflict was set aside as
  many-valued, and trust stayed at 0.50 for every narrator. A key is now many-valued only
  when more than half its sources give it several values ("al's children"); one source
  contradicting itself among single-valued ones is a conflict. Unit test
  `a_source_that_contradicts_itself_among_honest_ones_is_a_conflict`. The relation-store
  entries of the suite (67, 69, 71) are unchanged.

## Results (relation store + cooperation + proposals; hippocampus lesioned; three narrators, liar p = 0.75, new names stated once by each side; seeds 0 / 1 / 2)

Learned trust, every seed: honest narrators 0.71 and 0.76, the liar **0.48**. The rules
that model trust believe the true family of all three new names; `full` and `vote` see
three 1:1 ties and break them to the same word (smith), which is wrong for lucy.

| Rule | Belief, tom (smith, honest) vs (jones, liar) | Believed families right | Family questions | New-name questions overall |
|---|---|---|---|---|
| full | 1.00 vs 1.00 | 2 of 3 (tie) | 58 / 29 / 64% | 45.6% |
| vote | 0.50 vs 0.50 | 2 of 3 (tie) | 65 / 36 / 68% | 46.7% |
| graded | 0.60 vs 0.40 | **3 of 3** | **100 / 98 / 99.6%** | **73.3%** |
| posterior | 0.56 vs 0.21 | **3 of 3** | **100 / 100 / 99.2%** | **81.2%** |

Before the store's vote could reach the answer (same runs, graded): family questions
43 / 32 / 61%. The column alone said "jones" for every new name; the higher area said
"is" or "went".

Place questions about new names are unchanged by the wiring (graded 68 / 14 / 61%,
posterior 68 / 56 / 65%): they need the family carried on to the family's place rule,
which this path does not do. The difference between graded and posterior there comes
from training: the rules validate different proposals, so the cortex learns from
different replays.

## Findings
1. **Where the believed fact is asked, belief decides the answer:** 98–100% with graded or
   posterior belief, against 29–68% when every source is trusted or counted equally.
2. **Trust is what settles a 1:1 conflict.** The vote is a tie on every new name; trust
   learned from other facts (the liar 0.48 against 0.71–0.76) breaks it the right way on
   all three. Under a tie, `full` and `vote` answer by an arbitrary order, right where it
   happens to fall on the truth.
3. **Two wiring faults hid the module.** The relation store's vote was dropped by a stale
   expectation and by a kind filter that let the column veto an answer to a question it
   had not understood. With both fixed, the store's answer wins the mix where it is the
   only source that knows.
4. **A self-contradicting source is not a many-valued relation.** The old rule exempted
   any key one source gave two values, which is exactly what a liar who lies some of the
   time does; trust could then never be learned from the trained names' facts.
5. **The place questions still do not use the belief:** "tom went to the" needs the
   family and then the family's place. That is a two-step composition (family → place
   rule), the next step for this task.

## Next
- Carry the believed family into place questions (a relational hop from the name to its
  family's place).
- Proposals and the network's own claims as sources with earned trust.
- An "unknown" answer when belief is split (a near-tie under the posterior).
