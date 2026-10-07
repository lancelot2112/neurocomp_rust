# 76 · Swapping the belief rule: what graded belief buys

**Question.** With belief isolated behind one rule in the Bayes module
([75](75-bayes-module.md)), and the network's inferences validated by the module instead
of by counting rows ([74](74-proposals-and-premises.md)): what does graded belief buy over
trusting everything, and does a Bayesian posterior differ from a trust-weighted vote?

**Code.**
- [`src/program/bayes.rs`](../../src/program/bayes.rs): `BeliefRule`, chosen by `BELIEF=`.
  The rest of the system asks the module only how far a value is believed (`belief`) and
  how credible a source is (`credibility`).

  | Rule | Belief in a value | Source credibility |
  |---|---|---|
  | `full` | 1 if anyone claims it | 1 |
  | `vote` | its share of the claims | 1 |
  | `graded` (default) | its share of the trust | learned trust |
  | `posterior` | its sources' odds t / (1 − t) multiplied, normalised against "none of these" (exact in integers) | learned trust |

  Unit test: under the posterior a lone claim is believed as far as its source is trusted,
  agreeing sources raise belief; graded gives a lone claim full belief; full believes both
  sides of a conflict.
- [`examples/episodic.rs`](../../examples/episodic.rs): a proposal's credibility is the
  belief in its fact ("lucy is a jones") times the credibility of the narrator who told its
  source event; validated at `PROPOSAL_MIN` (0.5), or when confirmed by reading, unless
  contradicted. Inference returns the partner word (the family) linking the two premises;
  the harness records each stored event's narrator. Proposals are judged again after the
  relation store consolidates at each sleep, so facts of the last stretch count.

## Results (cooperation + replay + relation store + proposals; hippocampus lesioned; seeds 0 / 1 / 2)

**Three honest narrators, each new name's family stated once.** Every rule validates the
same 24 proposals (all true; credibility 0.93 under graded and posterior), and the answers
are identical: **65 / 65 / 60%** (63.3%) under full, vote, graded and posterior.

**One narrator lying 3 times in 4, families stated three times.** Only the first statement
about a new name yields proposals.

| Rule | Seed 0 | Seed 1 | Seed 2 | Held out |
|---|---|---|---|---|
| full | no proposals | **8 validated, 0% true** | 8 validated, 100% true | 57 / **50** / 55% |
| vote | no proposals | 8 rejected (0% true) | 8 validated, 100% true | 57 / **64** / 55% |
| graded | no proposals | 8 rejected (0% true) | 8 validated, 100% true | 57 / **64** / 55% |
| posterior * | no proposals | 8 rejected (0% true) | 8 validated, 100% true | 57 / 61 / 62% |

\* the posterior run's training followed another random path (159 / 203 lies against
172 / 210), so its figures are not a controlled comparison.

Belief in the new names' family facts (seed 1):

| Rule | lucy: jones (2 honest) vs smith (liar) | tom: smith (1 honest) vs jones (liar) |
|---|---|---|
| full | 1.00 vs 1.00 | 1.00 vs 1.00 |
| vote | 0.67 vs 0.33 | 0.50 vs 0.50 (tie) |
| graded | 0.84 vs 0.16 | 0.73 vs 0.27 |
| posterior | 0.84 vs 0.05 | 0.66 vs 0.09 |

## Findings
1. **Trusting everything lets a lie become knowledge.** Under `full`, the liar's statement
   about lucy (heard first) became a premise, eight false inferences were validated and
   taught to the cortex, and that seed's new-name answers fell from 64 to 50%. Any graded
   rule rejected them.
2. **When every source is honest, grading costs nothing:** all rules validate the same true
   inferences and give the same answers.
3. **Vote and graded differ only on ties.** Here the false premise had two honest
   statements against it, so a vote sufficed. Trust is what decides a 1:1 conflict (tom:
   0.73 against 0.27 graded; a 0.50 tie under a vote).
4. **The posterior is the most decided where sources agree and the most sceptical of a
   doubted source:** lucy jones 0.84 against smith 0.05 (graded: 0.16); a lone claim gets
   its source's trust, not full belief.
5. **The task still leans little on belief:** the effect shows on one seed of three, the
   one where the lie came first. A task where conflicting statements decide the answer
   directly would measure it better.

## Next
- Infer from every statement of a fact, not only the first (so a true later statement can
  correct proposals built on an earlier lie).
- A task where the believed fact decides the answer.
- Proposals and the network's own claims as sources with earned trust (needs proposals
  the world can later check).
