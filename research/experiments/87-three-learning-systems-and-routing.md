# 87 · Three learning systems, learned routing, and a plausible hippocampal path (five seeds)

**Question.** Three changes for biological plausibility, each an option, judged on the
five-seed suite (the noise of [85](85-top-down-as-a-trusted-witness.md) made three seeds too
few):
- **three learning systems** (`LEARNING=three`): the column's L2/3 becomes a slow cortex
  (a miss grows a kernel only with probability 1/4, with near-miss generalisation), the fast
  one-shot rule moves to a `Cerebellum` that reads a copy of the column's input and returns
  through the thalamus as a vote, and the column's output (and every surprise read from it)
  is the integrated prediction ([coupling](../concepts/connection-plausibility.md#coupling-who-learns-from-which-error));
- **learned input routing** (`ROUTE=1`): every frame of the column's row is a channel whose
  share of bits is learned from the same pass (the kernels that do not read a slot give the
  prediction without it), slot order fixed after a critical period, codes not rotated;
- **the hippocampus as entorhinal feedback only** (`HC_EC=1`): recall reaches the column as
  context in its own slot, passed in proportion to the cortex's uncertainty; no vote, no
  rollout insertion.

**Code.** [`src/program/cerebellum.rs`](../../src/program/cerebellum.rs),
`KernelClass::set_growth_probability`, `CorticalColumn::set_output`, `peek_shallow`;
[`examples/episodic.rs`](../../examples/episodic.rs) (`LEARNING`, `SLOW_P`, `ROUTE`,
`ROUTE_CRITICAL`, `HC_EC`).

## Results (held-out, five-seed means; recorded → option)

| Entry | Recorded | Three systems | Routing |
|---|---|---|---|
| story boundary | 94.8 | **98.4** | 88.2 |
| saccades | 96.4 | **98.0** | 92.0 |
| role transfer | 67.2 | **73.4** | 64.1 |
| sleep generalisation | 74.7 | 77.6 | 72.8 |
| schema advantage | 61.3 | **69.8** | 63.7 |
| learned stepping | 54.5 | **64.4** | 57.3 |
| superposed evidence | 57.2 | **65.2** | 58.1 |
| engram walk | 66.8 | **76.2** | 68.6 |
| slot memory | 19.3 | 10.6 | **30.9** |
| family stated | 38.0 | 23.1 | 40.8 |
| family consolidated | 52.0 | 20.2 | 54.2 |
| semantic store | 63.1 | 17.9 | 64.5 |
| full hippocampus | 38.0 | 37.2 | 42.5 |
| hippocampus teaches cortex | 60.9 | 18.1 | 45.6 |
| index hippocampus | 66.0 | 18.2 | 56.8 |
| engram store | 67.4 | 22.0 | 67.8 |
| engram walk only | 87.0 | 73.8 | **91.2** |
| inference replay | 59.4 | 38.2 | **66.1** |
| inference read | 45.6 | 22.8 | 50.0 |
| cooperate | 54.0 | 35.6 | **62.0** |
| relations, speak, motor speech | 62.8–63.1 | 62.6–62.9 | 65.7–65.8 |
| belief decides | 82.4 | 82.1 | 80.9 |

Seen pairs rise under three systems on nearly every entry (about +10).

**The hippocampus as entorhinal feedback** (seeds 0 and 1 only, against the same seeds):
speech motor 64.8 / 66.8 → 39.0 / 55.0, index hippocampus 67.0 / 63.8 → 52.6 / 55.6 and
hippocampus teaches cortex 67.4 / 49.8 → 66.8 / 40.8 (all three lesion the hippocampus at
test); slot memory 19.4 / 14.4 → 18.8 / 33.4, family consolidated 57.0 / 52.8 → 63.0 / 54.4,
engram walk 69.4 / 59.0 → 61.2 / 54.8.

## Findings
1. **Three learning systems split the suite in two.** Where the answer depends on context the
   network can learn from many stories (story boundary, saccades, role transfer, schemas,
   learned stepping, superposed evidence, the engram walk), they gain 3–10 points, with
   story boundary and saccades at 98%. Where the answer depends on facts met once and carried
   by consolidation (the slot, family, semantic, index and engram-store entries), they
   collapse to 18–23%. The slow cortex grows too little in 3,000 stories to absorb what
   replay teaches it, and the cerebellum memorises the stories it has seen (seen +10).
2. **Learned routing is the more even change:** better on ten entries (slot memory +12,
   inference replay +7, cooperate +8, engram walk only +4, relations +3), worse on the context
   tasks (story boundary −7, saccades −4) and two hippocampus entries (−9, −15).
3. **The plausible hippocampal path removes the crutch the implausible ones hid.** Memory as
   context only is worse wherever the hippocampus is lesioned at test: the cortex had been
   getting answers through the vote and the inserted word, and nothing now carries them into
   it.
4. **The three results point at one gap: consolidation.** A plausible hippocampus, a slow
   cortex and replay that teaches it are each right in kind, and together they fail where
   facts met once must move into the cortex. Replay strong enough, and timed, to move them
   is the next step, together with the workspace (a recalled answer broadcast so that every
   system learns from it).
5. None becomes a default. With five seeds, effects of 5+ points on these entries are now
   readable; per-seed spreads are still wide (cooperate 11–53% under three systems).

## Addendum: recall at test repaired (five seeds)

The diagnosis of the collapse (task 27) found that consolidation itself worked (replay
reached the semantic store and the higher area) but recall at test did not: the rollout read
the slow L2/3 alone, whose expectation was nearly always empty, so it never started (0
completions). Two fixes: the rollout reads the integrated expectation (the slow cortex's and
the cerebellum's candidates), and recall starts where the page contradicts the integrated
**winning** prediction, the union of candidates only filtering what may be offered
(commits 4b29863, 850fa0d). `LEARNING=three SLOW_P=0.25`, five seeds, held-out:

| Entry | Recorded | Three, before | Three, recall fixed |
|---|---|---|---|
| story boundary / saccades | 94.8 / 96.4 | 98.4 / 98.0 | 98.4 / 98.0 |
| role transfer | 67.2 | 73.4 | 73.4 |
| schema advantage | 61.3 | 69.8 | **74.3** |
| learned stepping | 54.5 | 64.4 | 50.1 |
| superposed evidence | 57.2 | 65.2 | 54.1 |
| engram walk | 66.8 | 76.2 | 43.2 |
| slot memory | 19.3 | 10.6 | 11.5 |
| family stated / consolidated | 38.0 / 52.0 | 23.1 / 20.2 | 41.2 / 40.6 |
| semantic store | 63.1 | 17.9 | 36.4 |
| full hippocampus | 38.0 | 37.2 | 36.2 |
| hippocampus teaches cortex | 60.9 | 18.1 | 44.8 |
| index hippocampus | 66.0 | 18.2 | 31.4 |
| engram store | 67.4 | 22.0 | 39.0 |
| engram walk only | 87.0 | 73.8 | 84.6 |
| inference replay / read | 59.4 / 45.6 | 38.2 / 22.8 | 52.2 / 23.6 |
| cooperate | 54.0 | 35.6 | 43.2 |
| relations / speak / motor speech | 62.8 / 63.1 / 62.9 | 62.6–62.9 | 60.0 / 59.8 / 60.2 |
| belief decides | 82.4 | 82.1 | **85.2** |

- **The consolidation entries recover about half their loss** (family consolidated 20 → 41,
  hippocampus teaches cortex 18 → 45, engram store 22 → 39), still 10–35 points under the
  default. The slow cortex stays small (66–80 kernels; 0% alone at test), so the facts come
  back through the cerebellum and memory, not the cortex: consolidation proper is still
  missing.
- **Recall that fires everywhere costs the context entries.** Under three systems the
  integrated winner is definite at most steps, so the rollout inserts 5,000–6,000 words per
  test (none before the fix), and learned stepping (−14), superposed evidence
  (−11) and the engram walk (−33) lose what they gained when recall was silent. The trigger,
  a hand-written rule, is the problem; the next step replaces it (experiment 88, inner speech).

## Next
- Consolidation for the slow cortex: more replay, interleaved, and replay that runs through
  the cerebellum-to-cortex path (the fast learner teaching the slow one), measured on the
  lesioned entries.
- The workspace and brake (roadmap).
- Three systems with routing and the entorhinal path together, once consolidation works.
