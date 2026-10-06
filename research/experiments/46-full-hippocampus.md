# 46 · The full hippocampal circuit

**Question.** Until now the hippocampus has been built in pieces:
- [12](12-dentate-gyrus-ca3.md): a dentate gyrus and a Hebbian CA3, in bits;
- [14](14-ca1-comparator.md): a CA1 comparator, written in the harness;
- [17](17-consolidation.md): replay into a cortical store.

None of them had the whole circuit's wiring. In [45](45-superposed-thought-and-ca3.md),
the DG + CA3 core could not recall a fact seen once ("tom is a smith") against bindings
stored thousands of times. Does the full circuit, with its own novelty signal, work as
the network's hippocampus?

**Code.** [`src/program/hippocampal_circuit.rs`](../../src/program/hippocampal_circuit.rs)
(`Hippocampus`, `HippocampusConfig`, `Recall`), with 6 unit tests. `Pathway::drive_scaled`
is in [`src/program/hippocampus.rs`](../../src/program/hippocampus.rs). In
[`examples/episodic.rs`](../../examples/episodic.rs): `HIPPO=full`, `HIPPO_GAIN`,
`HIPPO_CA2`, and the `HIPPO` report. The regression entry is `full-hippocampus` (without
the semantic store, so the hippocampus must carry the fact).

## The circuit
| Part / connection | Here | Learned? |
|---|---|---|
| EC layer II → dentate gyrus | Random expansion (8,192 granule cells, fan-in 300), k-winners-take-all (32): pattern separation | Fixed |
| DG → CA3, mossy fibres | Each granule cell has one "detonator" target in CA3. At storage, the granule code chooses the episode's CA3 cells | Fixed; storage only |
| EC II → CA3, perforant path | Drives CA3 from a cue at recall, with **presynaptic scaling** (below) | Hebbian |
| CA3 ↔ CA3, recurrent collaterals | 4,096 cells; recall settles for 2 steps (pattern completion) | Hebbian |
| EC layer III → CA1, temporoammonic path | Random expansion to 4,096 CA1 cells, k = 32: CA1's code for the current input | Fixed |
| CA3 → CA1, Schaffer collaterals | Maps each CA3 attractor to its episode's CA1 code: CA1 decodes what CA3 completed | Hebbian |
| CA1 comparator | Overlap of CA1-from-CA3 (what memory predicts) with CA1-from-EC III (what is there): match, and novelty = 1 − match | – |
| **Novelty-gated encoding** | Before storing, the episode is recalled. Its novelty multiplies the write strength by 1 + 3 × novelty (as acetylcholine / dopamine scale plasticity) | – |
| CA2 | 1,024 cells, 16 active, a code that drifts with time (5% per story). CA2 → CA1 is learned at storage and biases recall toward recent episodes | Hebbian; off by default (below) |
| CA1 → subiculum → EC layer V | The readout: recalled episodes leave the hippocampus here, with their strength | Hebbian |
| Replay | Sharp-wave-ripple-like: CA3 starts from random cells, settles into an attractor, and is read out through CA1 with no cue | – |
| Event-based recall | A cue equal to the previous one returns the cached recall. Every pathway reads only the rows of active cells | – |

All weights are bit-sliced counters with lazy plane-shift decay (halving about every 690
stores), as in [12](12-dentate-gyrus-ca3.md).

### Presynaptic scaling
The unit test of the [45](45-superposed-thought-and-ca3.md) failure case stores 300
episodes of common words, then one episode with a rare name and its family, then 50 more
common ones. It then cues with two common words and the name.
- **Neither mechanism alone recalls the family:** 1 of its 16 bits, with novelty gating
  alone or with scaling alone.
- **The cue's common words were the problem.** Their perforant synapses were written in
  hundreds of episodes and saturate, so they drive every common episode's CA3 cells.
- **The fix:** an EC input written by n episodes drives CA3 with its weights shifted
  right by log2 n. A much-used input counts little per target, as heterosynaptic
  depression and synaptic scaling make it. This is the rarity weighting of
  [36](36-slot-binding-memory.md), now a property of synapses.
- **Both together recall all 16 bits.** The novel trace is written strongly, and its
  rare cue reaches it.

## Results (seeds 0 / 1 / 2; family stated once; evidence superposition, consolidation; hand trigger unless stated)

| Hippocampus | Semantic store | Held-out right | Family correct |
|---|---|---|---|
| List memory, rarity-weighted search (36–45) | no | 67 / 53 / 42% | 95 / 69 / 66% |
| DG + CA3 core (45) | no | 36 / 47 / 42% | 54 / 68 / 63% (a guess) |
| **Full circuit** | no | **66 / 67 / 26%** | **100 / 100 / 34%** |
| Full circuit, novelty gating off | no | 29 / 46 / 25% | 29 / 68 / 32% |
| Full circuit, learned stepping | no | 62 / 50 / 41% | 99 / 100 / 29% |
| List memory | yes | 69 / 69 / 60% | 99 / 100 / 99% |
| DG + CA3 core | yes | 64 / 68 / 57% | 77 / 100 / 92% |
| **Full circuit** | yes | **67 / 67 / 55%** | **100 / 100 / 100%** |
| Full circuit, CA2 recency bias on (weight 1) | yes | 57 / 48 / 35% | 36 / 69 / 34% |
| Full circuit, learned stepping | yes | 62 / 52 / 56% | 100 / 100 / 100% |
| Full circuit, lesioned at test | yes | 67 / 67 / 61% | 100% |

Trained names: 65–70% throughout.

The circuit's own statistics (without the semantic store):
- 3,000 episodes stored; mean novelty at storage 0.89–0.92;
- about 97,000 recalls, 75% answered from the cache;
- 0.45–0.58 ms per word, against 0.25–0.32 for the list memory and 2.3–3.1 for the
  DG + CA3 core.

## Findings
1. **The full circuit recalls a one-shot fact, and the DG + CA3 core did not.** Without
   the semantic store, it supplies the right family 100% of the time on seeds 0 and 1,
   where the core only guessed (54–68%). Held-out answers are as good as the
   hand-written list memory's on average (53% and 54%), with no explicit search: the
   rarity weighting is in the synapses.
2. **Novelty-gated encoding is what makes it work.** Switched off, the family is a guess
   again (29–68%).
   - The CA1 comparator's mismatch decides how strongly an episode is written. Without
     it, the one statement about tom is as faint as any story, and the common bindings'
     saturated weights win.
3. **Seed 2 fails the same way for every memory.** Tom is called a jones and lucy a
   smith.
   - The episode is a whole story: "tom is a smith" plus a question about someone else,
     whose surname is bound in the same "surname" slot.
   - Recall finds the right story, but the slot holds two surnames, and the class filter
     picks one.
   - The list memory is weak on this seed too (66%). The fix is finer episodes (one per
     sentence or event), not a better store.
4. **CA2 as a recency bias hurts here** (35–57% against 55–67%). The statement about tom
   is up to 600 stories old, and a drifting time code favours recent episodes over it.
   CA2 is built and tested, but off by default. Its use is for tasks where *when*
   matters: what happened last time.
5. **Event-based recall pays.** Three in four recalls repeat the previous cue: the story's
   bindings change only at a surprising word. The cache makes the full circuit about 5×
   faster than the core without it.
6. **The semantic store still matters.** With it, all three seeds get the family right.
   With the hippocampus lesioned at test, the answers are unchanged (61–67%): sleep had
   already moved the fact into the cortex.

## What is still missing
- **EC as a learned layer.** EC is the word codes plus slot rotation. There is no learned
  EC II / III / V population, and EC V's output is not yet fed back into the cortex
  (the readout goes to the harness's slot unbinding).
- **Replay from the circuit to the cortex.** `replay` works (unit test: most replays
  settle into a stored episode), but the semantic store is still fed from the harness's
  sentence buffer. Feeding it from the hippocampus's own replay is the next step.
- **Episodes per event, not per story** (finding 3).
- **CA1's match signal as the familiarity signal.** The familiarity bands still come from
  the list store's counts (perirhinal-like). CA1's mismatch is computed but not yet used
  by the mix or the basal ganglia.

## Biology
- **The circuit:** Amaral & Witter 1989; trisynaptic (EC II → DG → CA3 → CA1) and
  monosynaptic (EC III → CA1) paths. CA1 compares them (Lisman & Otmakhova 2001; Hasselmo
  & Schnell 1994).
- **Mossy fibres as detonators** that impose sparse CA3 codes at encoding, while the
  perforant path cues recall (McNaughton & Morris 1987; Treves & Rolls 1992).
- **Novelty and encoding:** CA1 mismatch → VTA dopamine → stronger, lasting plasticity for
  new episodes (Lisman & Grace 2005). Acetylcholine sets encoding mode (Hasselmo 2006).
- **Synaptic scaling and heterosynaptic depression** keep heavily used inputs from
  dominating (Turrigiano 2008; Chistiakova et al. 2014).
- **CA2:** time and social memory; a slowly drifting population code (Mankin et al.
  2015; Hitti & Siegelbaum 2014).
- **Sharp-wave ripples:** CA3-initiated replay read out through CA1 and the subiculum to
  cortex (Buzsáki 2015).
