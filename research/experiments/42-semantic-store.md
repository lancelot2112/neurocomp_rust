# 42 · A semantic store: "tom is a smith" consolidated into the cortex

**Question.** In [41](41-consolidating-the-stated-family.md), sleep replay strengthened
the family rule, but the new member's family ("tom is a smith", stated once) stayed in the
hippocampus. With the hippocampus off, the rollout of [40](40-family-stated-once.md)
guessed one family for everyone. The diagnosis: the cortex had only next-word
predictors, and no store that, given "tom", activates "smith". Experiment
[17](17-consolidation.md) built such a store (cue → content, trained by replay) for the
older memory-frame setup. Wired into the current one, does it carry the family when the
hippocampus is gone?

**Code.** In [`examples/episodic.rs`](../../examples/episodic.rs): `SEMANTIC=r`,
`SEMANTIC_RANDOM=n` (control), and the `SEMANTIC` report. The settings are the
`semantic-store` entry of [`scripts/regress.tsv`](../../scripts/regress.tsv).

## The semantic store
- **The store:** a separate predictive kernel class with one frame, as in 17. It sees a
  cue word and predicts content words. It learns only from replay, never from reading.
- **What the hippocampus keeps:** every training sentence, as an episode, until the next
  sleep.
- **Sleep replay** (every 500 stories, and the night before the test). The *novel*
  sentences are replayed: those with a word seen in under 1% of sentences so far. They
  are interleaved with as many randomly drawn others, r rounds.
  - Each replayed sentence's rarest word cues the codes of its other words:
    "tom" → {is, a, smith}.
  - A replay is one learning step for the store: it grows a kernel if surprised, as
    every kernel class does.
- **Use:** at a rollout step ([40](40-family-stated-once.md)), the store is queried
  with the current sentence's rarest word. The word source order is the hippocampal slot
  memory, then the semantic store, then the higher area, then the column. As always, only
  a word of the kind the column expects here is taken. At "tom is a ?" the column expects
  {smith, jones}, and the store's {is, a, smith} gives smith.
- **The lesion** (`BIND_LESION`) switches off the slot memory at test, after training and
  its sleeps. That is Tse's design: lesion after consolidation.

## Results (seeds 0 / 1 / 2; family stated once; rollout at test)

| | Held-out right | Family supplied correctly | Trained names |
|---|---|---|---|
| Hippocampus off, no consolidation (41) | 29 / 42 / 29% | 54 / 68 / 38% | 62% |
| **Hippocampus off, semantic store (r = 3)** | **49 / 53 / 57%** | **100 / 95 / 100%** | 63% |
| Hippocampus off, semantic store (r = 1) | 51 / 53 / 58% | 100 / 95 / 100% | 63% |
| Hippocampus off, answer-trace consolidation (41) | 44 / 46 / 41% | 64 / 68 / 65% | 65% |
| **Hippocampus off, semantic store + answer-trace consolidation** | **67 / 67 / 61%** | **100 / 100 / 100%** | 66% |
| Intact, no consolidation (41) | 42 / 51 / 28% | 73 / 91 / 39% | 65% |
| Intact, semantic store | 51 / 51 / 55% | 96 / 93 / 96% | 65% |
| No-schema group, hippocampus off, semantic store | 17 / 19 / 19% | 100 / 100 / 100% | 16% |

The control, replay without the novelty priority (`SEMANTIC_RANDOM=n`, hippocampus off):

| Sentences replayed per sleep, drawn at random | Replays in all | Family supplied correctly | Held-out right |
|---|---|---|---|
| Novelty-tagged (above, r = 1) | 6 | 100 / 95 / 100% | 51 / 53 / 58% |
| 6 at random | 108 | 60 / 34 / 70% | 34 / 25 / 36% |
| 200 at random | 3,600 | 47 / 40 / 56% | 28 / 32 / 39% |
| 2,000 at random (half of all) | 36,000 | 100 / 74 / 69% | 60 / 48 / 34% |

## Findings
1. **The family is now cortical.** With the hippocampus switched off, the rollout
   supplies the right family 95–100% of the time, against 38–68% before. That earlier
   figure was a constant guess, two-thirds right because two of three new members are
   smiths. Held-out answers rise to 49–58% (from 29–42%).
   - This is Tse's 48-hour result in this network: a schema-consistent fact, learned in
     one trial, no longer needs the hippocampus after sleep.
2. **The two consolidations add up.** The semantic store supplies the family. Answer-trace
   replay ([41](41-consolidating-the-stated-family.md)) strengthens the rule that turns a
   family into a place. Together, with the hippocampus off, held-out answers reach
   61–67%. That is as good as the intact, consolidated network of 41 (42–67%), and about
   what trained names score (66%).
3. **The schema is still what makes the fact useful.** The no-schema group consolidates
   the family just as well (100%) but still answers at chance (17–19%). Knowing tom is a
   smith helps only if the smith rule exists.
4. **Replay must favour what is new.** With the same budget drawn at random, the store
   mostly misses the one statement about tom and the family stays a guess (34–70%).
   Replaying half of all sentences catches it more often (69–100%), but other sentences'
   associations compete. Novelty-tagged replay needs one round of 6 replays.
   - This matches the tagging of novel, salient experience for preferential replay
     (dopamine-gated tagging, Lisman & Grace 2005; reward- and novelty-biased replay).
5. **The cue is the rarest word, and the class filter does the rest.** The store answers
   "what went with tom": {is, a, smith}. It holds no structure: it does not know smith
   is the family. The column's expectation picks the right member at each step: "is"
   after "tom", "smith" after "a". The schema gives the kind, the store gives which one.
   That is the same division of labour as the hippocampal readout of
   [37](37-schema-supports-episode.md), now in the cortex.

## Limits
- **A small world makes novelty easy.** Here "a word in under 1% of sentences" picks out
  exactly the new-name statements: the store learned from 6 to 18 replays and holds 5–6
  kernels. In real text, many sentences hold a rare word, and the store would need
  capacity and interference control (sleep pruning, as for the column).
- **The cue is one word.** "tom" alone keys the store. Two facts about tom from different
  sentences would compete for the same cue. A conjunction of cues, or several kernels
  per cue, is needed beyond one fact per entity.
- **It is a rollout source only.** It does not yet vote in the precision-weighted mix.

## Next
- **The store as a mix source** with its own learned reliability, so it can also help
  where no rollout runs.
- **Learned stepping** (roadmap stage 3): let the basal ganglia choose when to roll out,
  rewarded by the answer, instead of the definite-expectation trigger.
- **A harder world for the store:** several facts per entity, and real text.

## Biology
- **Schema-dependent consolidation:** Tse et al. 2007, 2011. A hippocampal lesion 48 hours
  after one-trial learning of a schema-consistent pair spared it.
- **Semantic memory's hub:** the anterior temporal lobes bind what is known about an
  entity (Patterson, Nestor & Rogers 2007; Lambon Ralph et al. 2017).
- **Prioritised replay:** novel and rewarded experiences are replayed preferentially
  (Lisman & Grace 2005; Singer & Frank 2009; Mattar & Daw 2018). Unprioritised replay is
  what the random control shows: the rare important episode is mostly missed.
