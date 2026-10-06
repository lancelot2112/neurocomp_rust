# 45 · Superposed thought without per-source weights, and the learned hippocampus in the loop

**Questions.**
1. With the loop closed ([44](44-closed-loop.md)), an internal step can feed back a
   *superposition* of what every source offers, instead of the first source in a fixed
   order. Can the column resolve it, with no per-source reliabilities (which cannot be
   learned inside a thought, [43](43-learned-stepping.md))?
2. The slot ⊗ content memory of [36](36-slot-binding-memory.md)–[44](44-closed-loop.md)
   recalls by an explicit, rarity-weighted overlap search over a list of episodes. Does
   the learned hippocampus of [12](12-dentate-gyrus-ca3.md) work in its place: a dentate
   gyrus, and a Hebbian CA3 with recurrent pattern completion?

**Code.** In [`examples/episodic.rs`](../../examples/episodic.rs): `ROLLOUT_SUPER`
(`=1` or `=evidence`) and `HIPPO=ca3` (with `HIPPO_CELLS`, `HIPPO_K`, `HIPPO_SETTLE`,
`HIPPO_DECAY`, `HIPPO_HAB`). The superposition settings are the `superposed-evidence`
entry of [`scripts/regress.tsv`](../../scripts/regress.tsv).

## 1. Superposed thought
- **`ROLLOUT_SUPER=1`:** every source's output, gated by the column's expectation (a
  bitwise AND), is OR-ed into one fed-back vector. The sources are the slot memory, the
  semantic store, the higher area and the column's own prediction.
- **`ROLLOUT_SUPER=evidence`:** only the *evidence* sources are superposed: slot memory,
  semantic store and higher area. The column's own prediction is fed back only when none
  of them offers anything.

| Held-out right (seeds 0 / 1 / 2) | Intact, hand trigger | Intact, learned stepping | Hippocampus off, hand | Hippocampus off, learned |
|---|---|---|---|---|
| First source in fixed order (44) | 69 / 69 / 60% | 61 / 60 / 58% | 67 / 67 / 61% | 61 / 56 / 62% |
| All sources superposed | 52 / 55 / 49% | 43 / 43 / 51% | 50 / 49 / 50% | 52 / 40 / 45% |
| **Evidence sources superposed** | **69 / 69 / 60%** | **67 / 58 / 62%** | **67 / 67 / 61%** | **61 / 56 / 62%** |

Blends fed back (evidence, learned stepping): 43–321 per seed.

### Findings
1. **Superposing everything fails, and the reason is a double-counted prior.**
   - At the surname step the column's own guess ("jones") is OR-ed with the semantic
     store's "smith". The blend {smith, jones} goes forward, and the family rule cannot
     choose.
   - The bits are binary, so a blend carries no strengths. Each part counts the same.
   - But the column's prediction *is* the expectation the gate already applies: it is the
     prior. Adding it again counts the prior as if it were evidence.
2. **Superposing only the evidence works, with no order and no per-source weights.**
   - With the hand trigger it gives exactly the fixed order's answers, intact and
     lesioned.
   - With learned stepping it is as good or better (58–67% against 58–61%).
   - When two evidence sources disagree, both go forward, and the column resolves them
     against its own context.
   - This replaces the hand-set order with one principled rule: the prior gates, and
     evidence is superposed. It is a bitwise Bayesian combination: the prior selects what
     is possible here, and the evidence selects within it.

## 2. The learned hippocampus in the loop (`HIPPO=ca3`)
- **Storage:** each training story's slot ⊗ content episode, the same vector the list
  memory stores, is separated by the dentate gyrus. That is a random expansion to 8,192
  cells with k-winners-take-all, k = 32. It is stored in CA3 along three pathways:
  EC→CA3, CA3↔CA3 and CA3→EC. The weights are Hebbian, bit-sliced and decaying (halving
  every ~690 stores).
- **Recall:** the story's bindings so far drive CA3. The CA3 code settles for 2 steps
  through the recurrent weights (pattern completion), and CA3→EC reads the episode back
  out. The rest is unchanged: unbinding by the expected slot, the class filter, the
  closed loop.
- **The list memory still runs alongside,** but only for the familiarity statistics
  (perirhinal-like) and the cue habituation. It no longer answers.

| Seeds 0 / 1 / 2 (evidence superposition, semantic store, consolidation) | Held-out right | Family correct | Slot memory right on held-out answers |
|---|---|---|---|
| List memory, rarity-weighted recall | 69 / 69 / 60% | 99 / 100 / 99% | 35 / 18 / 24% |
| **CA3, habituated cue (0.3)** | **64 / 68 / 57%** | 77 / 100 / 92% | 24 / 28 / 26% |
| CA3, whole-story cue | 64 / 67 / 57% | 77 / 100 / 93% | 24 / 28 / 26% |
| CA3, learned stepping | 57 / 58 / 53% | 73 / 100 / –% | – |
| **CA3, no semantic store** | **36 / 47 / 42%** | **54 / 68 / 63%** | 26 / 28 / 27% |

The CA3 runs are about 10× slower: 2.3–3.1 ms per word, against 0.25–0.32 ms.

### Findings
3. **The learned hippocampus does not recall the one-shot statement.** Without the
   semantic store, the family it supplies (54–68%) is the cortex's constant "smith"
   guess. The list memory supplied 73–91% in [40](40-family-stated-once.md).
   - **Why:** CA3 sums Hebbian weights. "tom" was stored once, with weights of 16–31 on
     its 32 CA3 cells. Common bindings ("the" at the sentence start, the season) were
     stored in thousands of episodes, and their weights saturate (127) on many cells.
   - The cue's common bits drive a generic attractor, the blend of typical stories, not
     tom's episode.
   - The list memory avoided this with its explicit rarity weighting. Habituating the cue
     (leaving out bindings in over 30% of episodes) changed nothing measurable.
4. **With the semantic store, the system still answers new members** (57–68%). Sleep
   moved the fact into the cortex before the test, so the weak hippocampal recall
   matters less. This is the network's own version of "consolidated memories survive
   hippocampal damage".
5. **What CA3 lacks is a novelty-gated encoding strength.** In the brain, novelty raises
   acetylcholine and dopamine at encoding. That strengthens the plasticity for a new
   item and suppresses the recurrent collaterals, so a single exposure can make a strong
   trace (Hasselmo 2006; Lisman & Grace 2005). Here every store writes with the same
   strength.
   - The next step is a store whose strength grows with the episode's novelty (the
     perirhinal familiarity band). The CA1 comparator could then gate recall.

## The hippocampus we have, and what is missing
| Part | Here | Status |
|---|---|---|
| Entorhinal cortex (EC) | Sparse word codes, with slot rotation for structure; habituation; the fading state (lateral EC), slot cells (medial EC) | Codes, not a learned layer; no layer II / III / V split |
| Dentate gyrus | `DentateGyrus`: random expansion + k-WTA (pattern separation) | Built ([12](12-dentate-gyrus-ca3.md)); used for storage only |
| Mossy fibres (DG → CA3) | The DG code *is* the CA3 code at storage | No separate "detonator" pathway; not used at recall |
| Perforant path (EC → CA3) | Drives CA3 at recall | Built |
| CA3 recurrent collaterals | Hebbian, bit-sliced, settling for pattern completion | Built |
| CA3 → EC | Direct readout | A shortcut: in the brain it goes through CA1 |
| CA1 (Schaffer collaterals from CA3, temporoammonic input from EC layer III) | A prediction-error comparator ([14](14-ca1-comparator.md)) in an older pipeline | Partial; not in this pipeline |
| CA2 | – | Not modelled |
| Subiculum → EC layer V → neocortex | – | Not modelled (replay is fed to the cortex by the harness) |
| Event-based processing | – | Not done: the event-based fast path of [23](23-compaction.md) is the cortex's kernel matching, not the hippocampus |

## Next
- **Novelty-gated encoding** in CA3: store strength from the familiarity band, so a
  one-shot episode can win recall.
- **The full circuit:**
  - EC layers II / III / V;
  - DG → CA3 mossy fibres at storage, EC → CA3 at recall;
  - CA3 → CA1 and EC III → CA1, with CA1 as the comparator and decoder;
  - CA1 → subiculum → EC V → cortex as the replay output.
  This replaces the CA3 → EC shortcut and the harness-fed replay.
- **Event-based recall:** only the active cells' rows are read (already true for the
  pathways' drive), plus skipping recall when nothing in the cue changed. This brings
  CA3 back near the list memory's speed.

## Biology
- **Prior and evidence:** predictive coding treats top-down predictions as priors that
  gate and weight bottom-up evidence (Friston 2005). Superposing evidence within the
  prior's support is the binary analogue.
- **Novelty and encoding:** acetylcholine sets the hippocampus to encoding mode,
  suppressing recurrent recall (Hasselmo 2006). Dopamine from the VTA, triggered by
  novelty, enables long-lasting plasticity for new items (Lisman & Grace 2005).
- **The trisynaptic and monosynaptic paths:** EC II → DG → CA3 → CA1, and EC III → CA1
  direct. CA1 compares them (Lisman & Otmakhova 2001). Output runs CA1 → subiculum →
  EC V (Witter et al. 2000).
