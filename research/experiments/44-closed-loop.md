# 44 · Closing the loop: the network runs on its own output

**Question.** In [40](40-family-stated-once.md)–[43](43-learned-stepping.md) an internal
step was made by the harness:
1. the offering source's output was *decoded* into a word;
2. the word was checked against the expected kind, one vocabulary word at a time;
3. it was inserted into the stream and *re-encoded* as that word's clean code.

A brain has no such decode step. Its prediction is a pattern of activity, and a thought
is that pattern fed back as input. Does the rollout still work when the network's own
output vector is what it hears next?

**Code.** In [`examples/episodic.rs`](../../examples/episodic.rs): `ROLLOUT_LOOP` and the
`LOOP` report. The settings are the `closed-loop` entry of
[`scripts/regress.tsv`](../../scripts/regress.tsv): experiment 43's learned stepping, plus
`ROLLOUT_LOOP=1`.

## The closed loop
- **The gate is a bitwise AND, not a word lookup.** Each source's raw output vector is
  ANDed with the column's expectation (its superposed continuations). This is a
  thalamic gate: only what the cortex expects here passes.
  - Slot memory: the unbound episode, before any decoding.
  - Semantic store: its output for the sentence's rarest word.
  - Column: its own prediction.
- **The first source whose gated vector keeps a word's worth of bits (24) is fed back**
  as the next input. The order is slot memory, semantic store, higher area, column.
- **"Definite" is a bit count:** the expectation holds at most 1.5 words' worth of bits
  (48). Before, it was "exactly one decoded word".
- **The rest is unchanged:**
  - the choice point (the page's word overlaps the expectation by under 24 bits);
  - stopping when the page fits again;
  - the basal ganglia's choice to start (43).
- **Bookkeeping only:** a word is still decoded from the fed-back vector, for reports and
  the harness's context keys. The network never sees it.

## Results (seeds 0 / 1 / 2; family stated once; semantic store, answer-trace consolidation)

| | Held-out right | Family correct | Trained names |
|---|---|---|---|
| Hand trigger, decode and re-encode (42) | 69 / 69 / 60% | 98 / 100 / 97% | 67% |
| Hand trigger, closed loop | 69 / 69 / 60% (identical) | 98 / 100 / 97% | 67% |
| Hippocampus off, hand trigger, decode and re-encode | 67 / 67 / 61% | 100% | 66% |
| Hippocampus off, hand trigger, closed loop | 67 / 67 / 61% (identical) | 100% | 66% |
| Learned stepping, decode and re-encode (43) | 57 / 54 / 55% | 100% | 68% |
| **Learned stepping, closed loop** | **61 / 60 / 58%** | 99–100% | 70% |
| Hippocampus off, learned stepping, either | 61 / 56 / 62% (identical) | 100% | 65% |

What was fed back (at test):

| | Fed-back vectors | Exactly one word's code | A blend of several words |
|---|---|---|---|
| Hand trigger | 1,500–1,597 | 68–91% | 0–2 |
| Hippocampus off, hand trigger | 1,500–1,597 | 79–100% | 0 |
| Learned stepping | 1,301–1,507 | 62–78% | 21–414 |

## Findings
1. **Decoding was not doing any work.** With the hand trigger, the closed loop gives
   exactly the same answers as decode-and-re-encode, intact and lesioned, on every seed.
   - Yet 9–32% of the fed-back vectors were not a clean word code. They were part of one
     (24–31 of its 32 bits, the rest gated out).
   - The column matches on a fraction of a kernel's bits (80%), so a partial pattern
     drives it as the full one does.
   - The step from "a source's output" to "the next input" needs no symbols.
2. **Blends appear, and they don't hurt.** With learned stepping, the slot memory's raw
   readout sometimes passes the gate with two expected words in it: up to 414 blends
   on one seed. Held-out answers came out slightly higher (58–61% against 54–57%), though
   that is a small difference on three seeds. A superposed thought ("smith or jones")
   is fed forward and the column resolves it, or carries both.
3. **With the hippocampus off, nothing changes either.** The semantic store's gated
   output is a single word: exactly its code 79–100% of the time, otherwise part of it.
   Its output for "tom" is {is, a, smith}, and only one of those is expected at each
   step.
4. **One more piece is now the network's own.** An internal step is the network's output
   fed back through a gate. Still hand-set: where a choice point is, the order of the
   sources, and the cap of 4 steps
   ([learned vs hand-coded](../concepts/architecture-map.md#learned-vs-hand-coded-after-44)).

## Next
- **Sources in parallel, not in order.** With the loop closed, all gated sources could be
  OR-ed into one fed-back vector, a superposition, and the column resolves it. This is
  mixing without per-source reliabilities at the inner step. [43](43-learned-stepping.md)
  showed why those reliabilities cannot be learned from the page there.
- **The basal ganglia reading a cortical state** (the expectation plus a confidence code)
  instead of hand-picked features.
- **Novelty from the memory's own familiarity signal,** gating both replay and stepping.

## Biology
- **Thought as simulated perception:** imagery and inner speech reuse the sensory pathways
  that perception drives (Hesslow 2002; Pearson et al. 2015). The prediction is fed back
  through the same cortex that would have received the input.
- **Thalamic gating of internal loops:** cortico-thalamo-cortical loops gated by the
  reticular nucleus and basal ganglia select what is re-entered (Crick 1984; Halassa &
  Kastner 2017). The bitwise AND with the expectation is that gate in its simplest form.
- **Superposed predictions:** cortex represents several candidate continuations at once
  and resolves them over time (e.g. Kok, Mostert & de Lange 2017 on predicted stimuli in
  V1).
