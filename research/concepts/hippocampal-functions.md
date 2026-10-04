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

## Suggested order
1. Dentate-gyrus-style expansion + Hebbian completion weights (interference test).
2. Big-loop recurrence (multi-hop test).
3. Learned structure/role code (TEM-style) bound to content (word-order-ambiguity test).
4. Surprise-cut boundaries and replay, later.

See also [variable binding](variable-binding.md) and [open questions](../open-questions.md).
