# Hippocampal-formation functions: what we have and what's missing

The episodic memory in [11](../experiments/11-episodic-memory.md) covers only part of
what the hippocampus and entorhinal cortex (EC) do. This page maps each function to its
counterpart (or absence) in this codebase.

| Function | Biology | Here | Gap / how to add |
|---|---|---|---|
| **Pattern separation (expansion)** | Dentate gyrus: EC inputs fan out to many more, very sparsely active granule cells, so similar inputs get distinct codes ([Marr 1971; Treves & Rolls 1994; Yassa & Stark 2011](../related-work.md#hippocampus-and-entorhinal-cortex)) | By hand: sparse random word codes (8192 bits / 32 active; see [sparse codes](sparse-codes-and-collisions.md)). An episode is the OR of its words. | Learned expansion of *combinations*: random projection of the episode into a large space + winner-take-few, so overlapping episodes ("mary … kitchen" / "mary … garden") get distinct conjunctive codes. Test: recall accuracy as stored facts per name grow. |
| **Pattern completion** | CA3 recurrent collaterals act as an autoassociative attractor | `EpisodicMemory::recall`: the stored episode with the best overlap (a list searched by overlap) | Hebbian weight matrix (Willshaw / sparse Hopfield) so completion is a network operation with measurable capacity. |
| **Self-reference loops** | CA3→CA3 settling; the big loop EC→DG→CA3→CA1→EC re-enters retrieved content as the next cue | None: one recall per word | Feed the recalled episode back as the next cue. Test: two-hop questions (bAbI task 2). |
| **Match / novelty** | CA1 compares recall with direct EC input; mismatch drives encoding vs retrieval (theta-phase switching, [Hasselmo 2005](../related-work.md#hippocampus-and-entorhinal-cortex)) | Partial: habituation (`novel`, `rarest`); surprise drives kernel growth ([surprise-driven growth](surprise-driven-growth.md)) | A match/mismatch signal that gates storing vs recalling. |
| **Place / grid / graph (structure) codes** | EC grid cells give a relational scaffold; hippocampal cells bind structure to content ([Hafting et al. 2005; O'Keefe & Dostrovsky 1971; TEM: Whittington et al. 2020](../related-work.md#hippocampus-and-entorhinal-cortex)) | Crude: thalamic routes' fixed offsets (q, v) ([09](../experiments/09-thalamic-attention.md)) | A learned role/position code bound to content (XOR or conjunction). Whittington et al. 2022 show transformer position encodings ≈ the EC structure code and attention ≈ hippocampal binding, so this is the most direct bridge to transformers. Test: sentences a bag of words can't disambiguate ("mary saw john in the kitchen"). |
| **Event boundaries** | Episodes are cut at prediction-error peaks ([Zacks et al. 2007](../related-work.md#hippocampus-and-entorhinal-cortex)) | By hand: "." ends an episode | Surprise-based segmentation from [04](../experiments/04-word-segmentation.md). |
| **Replay / consolidation** | Hippocampal replay trains neocortex slowly (complementary learning systems, [McClelland, McNaughton & O'Reilly 1995](../related-work.md#hippocampus-and-entorhinal-cortex)) | None: fast store and slow predictor are separate | Replay stored episodes through the predictor's growth rules between stories. |
| **Indexing** | The hippocampus stores an index to distributed cortical patterns ([Teyler & DiScenna 1986](../related-work.md#hippocampus-and-entorhinal-cortex)) | Episodes store the word codes themselves | Store a pointer (sparse index code) bound to cortical codes; pairs naturally with expansion. |

## Is the dentate gyrus code random or semantic?
Biologically: driven by entorhinal input that *is* meaningful (lateral EC: objects /
content; medial EC: grid / structure), through a broad, quasi-random projection, then
sparsified by competition so that **similar inputs get dissimilar codes**. It depends on
content but deliberately discards similarity; similarity lives in EC and cortex, and the
dentate gyrus keeps memories from colliding. Plasticity (including adult neurogenesis)
tunes it somewhat. Here, `DentateGyrus` is a fixed random projection + k-WTA over
*random* word codes, so nothing on the memory path is semantic yet. Next: feed it the
learned syntax/topic codes from [05](../experiments/05-syntax.md)/[06](../experiments/06-meaning.md),
so recall can generalize by meaning while episodes stay separated.

## Superposition, counts and phase
- **Superimposed history with counts.** Bundle past frames into one count vector
  (optionally decaying) instead of separate frames: it keeps how often and how
  recently each feature was active, which a readout can normalize by (fixes frequent
  words drowning out distinctive ones).
- **Rotate by age, then superimpose:** history = Σ rot^lag(frame). Rotating back by k
  recovers the frame k steps ago as the strongest component. One vector holds an ordered
  sequence: a position code (cf. transformer position encodings, the EC structure code).
  Uses `rotl_mut`/`rotr_mut` and popcount already in `bitvec`. Capacity is limited by
  superposition noise.
- **Phase.** Theta–gamma multiplexing keeps competing items apart in time instead of
  blending them ([Lisman & Jensen 2013](../related-work.md#hippocampus-and-entorhinal-cortex));
  encode vs retrieve phases ([Hasselmo 2005](../related-work.md#hippocampus-and-entorhinal-cortex));
  phasor binding and resonator networks factor superpositions iteratively
  ([Plate 1995](../related-work.md#sparse-distributed-representations);
  [Frady & Sommer 2020](../related-work.md#hippocampus-and-entorhinal-cortex)). In bits: serial
  recall with inhibition of return, which is also the big loop.

## Thalamic links (not yet modelled)
Nucleus reuniens (prefrontal ↔ hippocampus) and anterior thalamus (Papez circuit). The
natural analogue here is making memory recall one more candidate route in the
[experiment 10](../experiments/10-route-pool-inhibition.md) pool, so the learned gate
decides when a recall gets through.

## Suggested order
1. Dentate-gyrus-style expansion + Hebbian completion weights (interference test), with
   a count-normalized readout and clean-up.
2. Big-loop recurrence as serial recall with inhibition of return (multi-hop test).
3. Rotation-superimposed history as a structure/order code (word-order test), then
   learned roles (TEM-style).
4. Semantic (learned) codes on the memory path.
5. Surprise-cut boundaries and replay, later.

See also [variable binding](variable-binding.md) and [open questions](../open-questions.md).
