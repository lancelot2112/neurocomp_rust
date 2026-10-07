# 74 · The network's inferences as proposals: validated before they count

**Question.** The hippocampus composes events it never read ("lucy went to the hallway",
in winter: [61](61-inferred-replay.md)), and replay taught them to the cortex as if read
([62](62-replay-as-reading.md)). Thoughts should not count as facts until validated. If
inferences are held as proposals, validated by the world or by independent evidence, and
only validated ones are taught, what is kept and what is lost?

**Code.** [`examples/episodic.rs`](../../examples/episodic.rs), `PROPOSALS=1`
(`PROPOSAL_SUPPORT`, default 2); [`src/program/engram.rs`](../../src/program/engram.rs).
- **Each inferred event is a proposal**, keyed by (season, words), stored in the
  hippocampus tagged as such (source 2; with `SOURCE_TAG=1` recall for answering never
  returns it).
- **Its evidence:**
  - *confirmed:* a sentence later read in a story of the same season is the proposal;
  - *contradicted:* the same name and season, another place;
  - *independent premises:* an inference has two premises, the fact ("lucy is a jones")
    and the source event ("daniel jones went to the hallway"); inference now returns
    both. The support is the weaker of the fact's testimonies (in how many episodes it
    was stated: rows count them) and the number of distinct source events. Derivations
    that all rest on one fact stated once are not independent.
  - (A first version counted source events only; then every proposal looked corroborated.)
- **Validated** = confirmed, or support ≥ 2, and never contradicted. Only validated
  proposals are replayed to the cortex, in each of their sources' contexts.
- **Open proposals** (neither validated nor contradicted) are listed as what to
  investigate next.
- The report scores each class against the world's rule (family and season decide the
  place).

## Results (seeds 0 / 1 / 2; hippocampus lesioned at test)

| Setting | Proposals made | True | Corroborated | Held out |
|---|---|---|---|---|
| walk only, every inference replayed (62) | – | – | – | 50 / 43 / 50% |
| walk only, proposals (support = source events) | 24 | 100% | 24 | 26 / 31 / 38% * |
| walk only, proposals (independent premises) | 24 | 100% | 0 | 23 / 17 / 25% |
| relation store, every inference replayed (67) | – | – | – | 65 / 65 / 64% |
| relation store, proposals (independent premises) | 24 | 100% | 0 | 61 / 58 / 64% |
| same, family stated 3× by 3 narrators | 0 / 8 / 8 | 100% | 0 | 61 / 60 / 55% |
| same, one narrator lying 3 times in 4 | 0 / 8 / 8 | 0 / 100% | 0 | 57 / 61 / 69% |

\* replayed in one source context each; replaying in every source's context was the fix,
but premise counting then left nothing to replay.

Open proposals (seed 0): "winter tom went to the bedroom (1 testimony of the fact, 10
source events)"; "summer sam smith went to the office (1, 12)"; "winter lucy went to the
hallway (1, 7)".

## Findings
1. **Counting independent premises is strict, and in this world nothing passes.** Every
   proposal about a new name rests on its one statement; ten source events do not make
   it more certain. With nothing validated, nothing is replayed, and the cortex is back to
   what it had without inferred replay (walk only 22%, relation store 61%).
2. **The inferences were all true** (where the family was stated truthfully). The price
   of not trusting unvalidated thoughts is the knowledge they would have given.
3. **Repeated testimony is invisible to the hippocampus as built.** Stated three times,
   the fact still showed one testimony: the three statements are stored as different
   rows (their learned slots differ between stories), and only the first statement, while
   the name is new, triggers inference. A false first statement (by the liar) yields
   false proposals (0% true on seed 1) that the true later statements never correct.
4. **What validation should use instead:** the Bayes module ([75](75-bayes-module.md))
   already knows how many narrators claimed "lucy is a jones", and how far each is
   trusted. A proposal's premise credibility should be its fact's belief there, and its
   support the trust of independent claimants, not row counts in the hippocampus.

## Next
- Premise credibility from the Bayes module; proposals as claims by a "self" source whose
  trust is earned by confirmations.
- Open proposals as curiosity: priority for replay and for attention when a matching
  story is read.
