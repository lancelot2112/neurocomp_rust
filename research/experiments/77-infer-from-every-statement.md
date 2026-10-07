# 77 · Inferring from every statement of a fact

**Question.** Inference started only from the first statement about a new name (a word
never seen before). When a liar's statement came first, every inference about that name
rested on the lie, and the true statements that followed produced none
([74](74-proposals-and-premises.md), [76](76-belief-rules.md)). If every statement about a
rare word is a fact to infer from, do the true inferences appear beside the false ones, and
does the Bayes module pick the true ones?

**Code.**
- [`src/program/engram.rs`](../../src/program/engram.rs): `EngramConfig::infer_rare` (k).
  A stored world event that holds a content word held by at most k rows is queued as a fact
  to infer from, and that word can be the inference's new word N (the rarest such word).
  k = 1 (default): only words never seen before, as before.
- [`examples/episodic.rs`](../../examples/episodic.rs): `INFER_EVERY=k`.
- A restatement worded exactly as a stored fact still only strengthens that row (one more
  testimony); its inferences would be the same.

## Results (cooperation + replay + relation store + proposals; hippocampus lesioned; three narrators; `INFER_EVERY=4`; seeds 0 / 1 / 2)

| World | Belief | Proposals made (true) | Validated (true) | Held out |
|---|---|---|---|---|
| honest, stated once | graded | 24 / 24 / 24 (100%) | 24 / 24 / 24 (100%) | 65 / 65 / 60% |
| liar, stated 3× | **full** | 40 / 48 / 32 (60 / 50 / 75%) | **all** (60 / 50 / 75% true) | 56 / 45 / 59% |
| liar, stated 3× | **graded** | 40 / 48 / 32 (60 / 50 / 75%) | **8 / 24 / 24 (100% true)** | 34 / 61 / 72% * |

\* see finding 4.

Open proposals under graded (seed 0, the closest conflict): "winter tom went to the bedroom
(credibility 0.35: tom smith 0.52 against jones 0.48, narrator trust 0.67)". Seed 1:
"spring lucy went to the kitchen (credibility 0.12)": an inference from the liar's "lucy is
a smith", rejected.

## Findings
1. **Every statement now yields its inferences,** the liar's and the honest narrators'
   alike: 32–48 proposals per seed, of which 50–75% are true.
2. **Graded belief validates only true ones** (8, 24 and 24 per seed, 100% true), and
   rejects every inference built on a lie. Trusting everything validates all of them:
   25–50% of what it teaches the cortex is false.
3. **The true inferences about a lied-about name now get through** (seeds 1 and 2: all of
   them), where before only the lie's inferences existed. On seed 0 the conflict about tom
   is nearly tied (0.52 against 0.48, trust barely learned), so his true inferences stay
   open (credibility 0.35 < 0.5): graded belief abstains where it cannot tell.
4. **Accuracy is not a clean comparison here.** These runs followed different random paths
   in training (the liar told 202 lies in this seed-0 run against 194 before), and per-seed
   accuracy swings widely (34 to 72% under graded). The harness shares one random generator
   among many subsystems, so a change in what is replayed changes the stories that follow.
   The validation counts (findings 1–3) are exact; the accuracy figures are one draw each.
5. **With honest sources nothing changes** (63.3%, as in 76).

## Next
- **Independent random streams per subsystem** (stories, replay, sleep, each store), so
  that a change in one part leaves the rest of the run identical and accuracy differences
  can be attributed.
- A lower validation bar for near-ties, or more statements per name, so that abstention is
  not the only outcome when trust is barely learned.
