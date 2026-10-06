# 54 · A phase-bound hippocampus, and 1/n and centering in the circuit: backed out

**Question.** In an isolated memory, phase-coded inputs cancelled crosstalk, and slot
phases with 1/n weighting beat every count variant on language-like codes
([53](53-phase-codes-and-centering.md)). Two follow-ups on the experiment 49 task, where
the hippocampus runs on its own and teaches the cortex:
1. **A phase-bound binding space.** A binding (word in slot) is the word's bits at the
   slot's phase, replacing a field of its own per slot. CA3, CA1 and EC V are phasor
   populations.
2. **1/n weighting and homeostatic centering** on the existing counts circuit.

**Settings.** The `hippocampus-teaches-cortex` entry of
[`scripts/regress.tsv`](../../scripts/regress.tsv): 16,384 CA3 cells, sparse binding
space, tagged replay, and the hippocampus lesioned at test (`BIND_LESION=1`). Seeds
0 / 1 / 2.

## Results

| Hippocampus | Held-out right | Family right | Train µs/word |
|---|---|---|---|
| **Counts circuit, as in 49** | **64 / 59 / 53%** | 99 / 67 / 87% | **2,190–2,390** |
| Counts, + homeostatic centering on CA3 | 64 / 59 / 54% | 100 / 67 / 90% | 2,310–2,400 |
| Counts, + exact 1/n on the perforant path | 65 / 40 / 54% | 99 / 66 / 90% | 8,390–9,090 |
| Counts, + 1/n + centering (2 seeds) | (not finished) | 63 / 66% | 8,320–8,920 |
| Counts, slot fields → 64 shared random phases (control) | 63 / 52 / 55% | 100 / 100 / 89% | 2,240–2,440 |
| **Phase circuit** (phasor CA3 / CA1 / EC V, 64 slot phases, 1/n) | 67 / 42 / (stopped)% | 97 / 68% | **8,120–8,270** |

For comparison, training cost of the earlier memories on the same machine (regression
logs):

| Memory | Train µs/word |
|---|---|
| List memory + harness sentence buffer ([42](42-semantic-store.md), [45](45-superposed-thought-and-ca3.md)) | ~230 |
| Full circuit, story episodes ([46](46-full-hippocampus.md)) | ~390 |
| Full circuit on its own, sparse binding space ([49](49-sparse-binding-space.md)) | ~2,260 |
| Phase circuit (this page) | ~8,100 |

## Findings
1. **The phase circuit is worse and 3.5× slower, so it is backed out.**
   - Seed 0 gained 3 points (67 vs 64%) and seed 1 lost 17 (42 vs 59%).
   - Each recall reads complex weights from hash-map rows, about 8 ms per word in
     training. The counts circuit's bit-sliced counters take 2.3 ms.
   - `PhaseHippocampus`, `HIPPO=phase` and `PHASE_FIELDS` are removed. The
     `EpisodicCircuit` trait the harness now uses to hold a circuit stays: the regression
     entries are unchanged by it.
2. **Exact 1/n on the perforant path does not help in the circuit either.**
   - It broke seed 1 (40%) and ran almost 4× slower.
   - Its novelty fell from 0.92 to 0.57, which suggests the weighted cue recalls familiar
     events more readily and so writes new ones more weakly.
   - In isolation (53) it fixed the shift's rounding at high load. Here the shift's
     coarse weighting is apparently not the limit. Removed.
3. **Homeostatic centering is neutral to slightly positive, at no cost:** seed 2 held
   out 53 → 54%, family 87 → 90%, seeds 0 and 1 unchanged. It stays as an option
   (`HIPPO_CENTER=1`), off by default.
4. **The slot layout tells something.** Squeezing the 128 slot fields into 64 shared,
   randomly assigned fields (counts, no phases) fixed seed 1's family (67 → 100%) but
   cost held-out accuracy on seeds 1 and 2.
   - Seed 1's family failure, undiagnosed since 49, therefore depends on which fields
     its bindings land in: a collision or near-collision in the sparse space, not a
     capacity limit.
   - This is a lead for that diagnosis, not a fix.
5. **The lesson for phase codes:** the isolated measurement (53) scored recall of
   stored targets from full cues. Here the hippocampus has to:
   - recall from partial cues;
   - judge novelty through the comparator;
   - tag new events and replay them;
   - and do it all at 15,000+ events, with the cortex learning from it.

   The phase circuit's novelty signal also changed (0.69–0.73 vs 0.91). The gain in
   isolated crosstalk did not survive into the system.

## The cost of the self-contained hippocampus
Running the hippocampus on its own (49) costs about 10× the list memory's training time
(2.3 ms vs 0.23 ms per word):
- event storage on every sentence, through five pathways of 16,384 cells;
- a recall per word;
- no cache hits, because every word changes the cue.

Before adding to it, the obvious savings are:
- recall only when a word was surprising (as the event-based fast path of
  [23](23-compaction.md) does for the column);
- smaller CA1 / EC V populations;
- reusing the settled CA3 state between words of a sentence.

## Biology
Phase coding and theta-gamma nesting are real (O'Keefe & Recce 1993; Lisman & Jensen
2013). This result says only that this phase circuit, with its hashed phases and
phasor weights, did not beat counts on this task at this cost.
