# Errata: earlier conclusions corrected by later work

A pass over all 85 experiment pages (2026-10-08). Pages are lab notebook entries and keep
what was believed at the time; corrections are listed here, and the larger ones are also
marked on the page itself with a "Correction (wiki pass)" note. Status of every page, by
theme: [overview](OVERVIEW.md).

## Errors fixed on the pages

| Page | What it said | Correction |
|---|---|---|
| [10](experiments/10-route-pool-inhibition.md) | Kernel-gate row with unfilled placeholders (`KERNEL_SHORT_3` …), "5-seed run pending" | The run was never completed; the row is marked abandoned (0%, one seed). The page's link to an "open-ended route discovery" section of 09 points to a section that does not exist |
| [12](experiments/12-dentate-gyrus-ca3.md) and README | "99% held-out (1–3 facts) vs list memory 97.5%" | A comparison across builds, withdrawn on the page itself; in one build the CA3 version scored 99.4/99.5/100/100% after the predictor fixes, the list memory 100%. Finding "pattern separation matters as load grows" (93.5 vs 96.1) is reversed by the same page's bit version (90.3 vs 89.1): withdrawn |
| [24](experiments/24-cortical-hierarchy.md) | Mixing "helps where it should and nowhere hurts" | Its table has habit seed 1 82 → 80, topic seed 2 95 → 94, early-placement topic seed 2 59 → 51. "Recommended L2/3 setting" (probation) was removed from the code later |
| [33](experiments/33-generalisation-during-sleep.md) | "65.3 against 65.0 on average" | 65.7 (62/70/65) against 65.3 (66/57/73); the conclusion stands |
| [57](experiments/57-memory-vs-text.md) and README | Text search "67.6% held out, beating every memory here" | It stored the test stories; corrected, 17% held out ([58](experiments/58-a-story-is-a-graph.md)). The banner on 57 said so; the README row and finding 1 did not |
| [81](experiments/81-curiosity.md) and README | "The learned policy kept tom and lucy at 91–100%" | First runs only; with asking during practice (the addendum) it lost lucy on seed 1 and both on seed 2. "The network balances cost by itself" holds only when questions are cheap: at 0.25 a question or more it never reaches the question that matters |
| [83](experiments/83-compute-only-where-needed.md), README, roadmap | Learned gate: "relation and hippocampus entries lose 10–47 points" | The 47 is sleep generalisation; the relation and hippocampus entries lose about 7–21 (sleep generalisation, which lost 47, is neither) |
| [31](experiments/31-role-cells-and-transfer.md) | Role-frame transfer "12–40% on new names" | 2–48% per its table (40/2/12 cells, 48/5/22 raw) |
| [89](experiments/89-questions-as-inner-speech.md) and README | "Holding a gap open and restating its answer binds the fact: 68 → 91%"; the wrong-name control "shows the gain is the binding" | The hand-set ask had opened its question on the season word, so the restatement was "winter came is a smith". With the stranger as subject: 68.8% (no gain). The gain was the season placed beside the right family (87.5%), and the control was not a control for binding |
| [89](experiments/89-questions-as-inner-speech.md)–[90](experiments/90-holding-an-open-question.md) | The question task tests binding a stranger's fact | No held-out answer names the other family's place, even with no question act; every error is the season. The task's family part is solved by a shortcut, so the binding results of 89–90 say nothing about binding |
| Consolidation replay (`CONSOLIDATE`, 38–49 and the entries recorded with it) | Replay taught the higher area what the trace answers | Learning credits the last prediction's matches, and replay learned without predicting the replayed input first: it was credited against the last awake step's matches, and skipped when that stale winner predicted the answer. Fixed as an option (`REPLAY_PREDICT`, [94](experiments/94-sleep-gated-consolidation.md)); the recorded figures carry the bug |
| [connection audit](concepts/connection-plausibility.md) | `HIER_UP=both` "plausible as information"; 83's learned gate "a second run"; `COOPERATE` "semantic store" | `HIER_UP=both` failed (the rotated burst code stops copying, like `ROUTE_BIND`); only 83's first addendum used a second run, the eligibility-trace version does not; the suite cooperates with the relation store since 67 |

## Claims within seed noise

Kernel growth is path-dependent: scaling the top-down frame at 0.2% of training steps moved
one seed by 22 points ([85](experiments/85-top-down-as-a-trusted-witness.md)), and a
deterministic re-run of 40's settings in 41 differed by up to 12 points on one seed. So:

| Page | Claim | Status |
|---|---|---|
| [02](experiments/02-predictive-growth.md) | "Beats every n-gram held out" | 81.9 against 79.2 on a 73-character sentence, one run: about 2 characters |
| [07](experiments/07-top-down-bias.md) | The top-down bias works; gating removes the harm | +0.3 and 0.1–0.3 point differences, one run. 85 finds top-down helps as context |
| [14](experiments/14-ca1-comparator.md) | Comparator fix 64 → 93% | One seed, unreplicated |
| [15](experiments/15-basal-ganglia-selector.md) | "One learned choice beats several unlearned branches" | 84.5 vs 80.8 and 89.1 vs 84.5 on 5 seeds with wide ranges |
| [19](experiments/19-l5-shared-reward.md) | Attribution fixes the hop-2 selector | Rests on seed 1's swing; the mean is lower with L5 (81 vs 85) |
| [21](experiments/21-several-routes.md) | L6 "better than all six open" | Seed 2 98 vs 99: matches or beats |
| [23](experiments/23-compaction.md) | The surprise gate "seems to help" two-hop | Seed 2 71.8 → 81.6, within noise; tie-break fix changed baselines mid-page |
| [24](experiments/24-cortical-hierarchy.md) | Two-hop "slightly better"; arbitration and mixing gains | One seed (94.4 vs 93.0); 3–5 points on three seeds |
| [29](experiments/29-saccades.md) | "88–99.6% at every distance"; a boundary "changes little" | D = 0 is 78–96%; one seed fell to 44% with the boundary |
| [32](experiments/32-general-and-specific.md) | Spawning "passes all three tests" | Gains of 2–4 points on three seeds |
| [36](experiments/36-slot-binding-memory.md)–[38](experiments/38-consolidation-of-one-shot-episodes.md) | One-shot lifts | Wide three-seed spreads; seed-level drops hidden in ranges (37: 48 → 39; 38: "trained names rise to 67–86%" against 71–82% without) |
| [40](experiments/40-family-stated-once.md) | Rollout +6 points; "repeat runs differ by at most a point" | 41's re-run of the same settings differs by up to 12 points; the supplied-family split is the stronger evidence |
| [43](experiments/43-learned-stepping.md), [44](experiments/44-closed-loop.md), [59](experiments/59-engram-walk.md), [61](experiments/61-inferred-replay.md), [63](experiments/63-routes-cooperate.md) | Gains of 1–5 points | Within noise ([47](experiments/47-integer-only.md) already noted this) |
| [54](experiments/54-phase-hippocampus-backed-out.md), [55](experiments/55-index-memory.md) | Centering "slightly positive"; "forgetting helps" | One seed each |
| [66](experiments/66-typed-relations.md), [67](experiments/67-relation-store-in-reading.md), [68](experiments/68-lifted-frames-and-sparse-gating.md), [82](experiments/82-fact-completion.md) | Roles +3.7; relations over the bag +5; two hops +4; completion +3 | Parity is the safe reading |
| [72](experiments/72-go-no-go-signals.md) | Novelty and agreement let the go/no-go abstain (97.6% right) | Two seeds; seed 0 learned nothing |
| [75](experiments/75-bayes-module.md), [76](experiments/76-belief-rules.md) | Seed 0's weak trust "where the lies fell"; `full` costs 14 points | Both predate 78's many-valued fix; the 14 points is one seed with path divergence (77). The validation counts hold |
| [83](experiments/83-compute-only-where-needed.md), [85](experiments/85-top-down-as-a-trusted-witness.md) | Gate gains of +3 to +5; the counterfactual gate is "the right measure" | Within noise; the counterfactual gate is about neutral and used a second run |

## Reinterpretations

- **The column learns like a cerebellum, not a cortex.** One-shot growth on a miss, each
  output taught by the next input. Pages that call a kernel store "the cortex" or say it
  "plays the neocortex" ([17](experiments/17-consolidation.md), [24](experiments/24-cortical-hierarchy.md),
  [66](experiments/66-typed-relations.md), [67](experiments/67-relation-store-in-reading.md))
  describe a fast learner. See the [connection audit](concepts/connection-plausibility.md#learning-systems)
  and the roadmap's three learning systems.
- **Top-down is context, not a witness.** The higher area's word is usually wrong (it
  learns only the column's misses), but its frame helps the column generalise
  ([85](experiments/85-top-down-as-a-trusted-witness.md)). 07's reading of top-down as a
  bias, and 24's gates on it, are read in that light.
- **A memory that answers while reading is a crutch.** Answers it supplies stop being
  surprising, so nothing else learns them; with the hippocampus lesioned at test nothing
  knows them. 55's and 56's lower lesioned trained-name figures fit this, beside the
  replay-bias explanation they give; 63's mapping section endorsed posting recall into the
  column on every word.
- **Rotating a code to tag its source stops copying.** New names are answered by copying
  a word from an input slot; a rotated word no longer matches its output code (the routing
  work after 85: held out 67 → 48% from the rotation alone).
- **"Frame" has two senses.** In the relation pages (66–68, 82) a frame is a sentence
  template ("X is a Y"); in the kernel and top-down pages it is an input slot of a
  kernel's row. 82 uses both on one page.
- **Costs charged to learned policies collapsed them.** 68's learned gate (cost 0.05),
  81's cost policy (0.25 and 0.5 a question) and 83's first gate (`HIER_COST`) failed or
  never reached the target with a cost; 83's later suites and 85 ran at cost 0. A cost has
  to be learned against a reward that arrives later, which these did not have.
- **Recall-at-test numbers are not consolidation.** 55's 95% and 60's 91–95% are answers
  the hippocampus supplied at test; lesioned, 60 is 17–25%.

## Inconsistencies between pages (minor)

- 22 vs 23: elimination fan-out 17,247 vs 17,273 kernels.
- 24 vs 32: habit under higher-area pruning "0–11%" vs 42/45/41%.
- 36: the slot memory's "shown once" accuracy given three ways (28/31–33/69–76; 28/38/58;
  28/36/76).
- 46 vs 48 vs 47: the full-hippocampus row quoted as 66/67/26 and 65/67/26; 47's 62.4 does
  not match 46's seed 0.
- 49 vs 54: family 100/67/87 vs 99/67/87.
- 53 finding 2: "100% vs 93%"; the table's best count variant is 94%.
- 56 finding 3: "60.5% lesioned" trained names is not in its table.
- 58 says 13 used dense codes; 13 used 11's sparse codes.
- 60 finding 4 attributes the cortex-alone 29–49% to the semantic store's consolidation;
  that row has no semantic store.
- 64, 65, 67, 68, 75: averages misquoted by 0.2–0.3 points (59.3 not 59.5; 78.3 not 78.5;
  60.7 and 64.7 not 60.9 and 64.4; 65.3 not 65.5; 57.3 not 57.1).
- 66 finding 2 calls the frequency-gap rule "the better parse"; 67 showed it parses
  "jones" as frame.
- 71: band percentages over 1,000 questions per seed are read as new-name accuracy; 69 has
  500 held-out questions.
- 78 vs 79: tom's belief 0.60/0.40 and 0.56/0.21 vs 0.59/0.41 and 0.54/0.23 with the same
  trust.
- Old figures (60's 92%, 63's 59.5%, 62's 48%, 55, 49, 46) differ from the current suite
  (84.0, 56.8, 44.2, 66.3, 61.9, 41.4): quote them as history.
