# Hippocampal-formation functions: what we have and what's missing

The episodic memory in [11](../experiments/11-episodic-memory.md) covers only part of
what the hippocampus and entorhinal cortex (EC) do. This page maps each function to its
counterpart (or absence) in this codebase.

> **Status (wiki pass, 2026-10-08).** Written early (after 11–14). Since then the
> hippocampus gained slot binding ([36](../experiments/36-slot-binding-memory.md)), the
> full EC–DG–CA3–CA1 circuit ([46](../experiments/46-full-hippocampus.md)), a sparse
> binding space ([49](../experiments/49-sparse-binding-space.md)), the index and engram
> stores ([55](../experiments/55-index-memory.md), [56](../experiments/56-engram-store.md)),
> the walk ([59](../experiments/59-engram-walk.md)), generative replay
> ([62](../experiments/62-replay-as-reading.md)) and source tags
> ([73](../experiments/73-source-memory.md)). For the current state see the
> [overview](../OVERVIEW.md#the-network-today); for each pathway's biology, the
> [connection audit](connection-plausibility.md#the-hippocampus). The "what we have"
> column below is historical.

| Function | Biology | Here | Gap / how to add |
|---|---|---|---|
| **Pattern separation (expansion)** | Dentate gyrus: EC inputs fan out to many more, very sparsely active granule cells, so similar inputs get distinct codes ([Marr 1971; Treves & Rolls 1994; Yassa & Stark 2011](../related-work.md#hippocampus-and-entorhinal-cortex)) | By hand: sparse random word codes (8192 bits / 32 active; see [sparse codes](sparse-codes-and-collisions.md)). An episode is the OR of its words. | Learned expansion of *combinations*: random projection of the episode into a large space + winner-take-few, so overlapping episodes ("mary … kitchen" / "mary … garden") get distinct conjunctive codes. Test: recall accuracy as stored facts per name grow. |
| **Pattern completion** | CA3 recurrent collaterals act as an autoassociative attractor | `EpisodicMemory::recall`: the stored episode with the best overlap (a list searched by overlap) | Hebbian weight matrix (Willshaw / sparse Hopfield) so completion is a network operation with measurable capacity. |
| **Self-reference loops** | CA3→CA3 settling; the big loop EC→DG→CA3→CA1→EC re-enters retrieved content as the next cue | CA3 settling in [12](../experiments/12-dentate-gyrus-ca3.md); chained and branching recall in [13](../experiments/13-big-loop.md) | Hop count and which item to follow are hand-set; see [learned hops](#can-hops-be-learned). |
| **Match / novelty** | CA1 compares recall with direct EC input; mismatch drives encoding vs retrieval (theta-phase switching, [Hasselmo 2005](../related-work.md#hippocampus-and-entorhinal-cortex)) | Partial: habituation (`novel`, `rarest`) is a frequency proxy; surprise drives kernel growth ([surprise-driven growth](surprise-driven-growth.md)) | A CA1 comparator; see [CA1 and subiculum](#the-unmodelled-sections-ca1-and-subiculum). |
| **Place / grid / graph (structure) codes** | EC grid cells give a relational scaffold; hippocampal cells bind structure to content ([Hafting et al. 2005; O'Keefe & Dostrovsky 1971; TEM: Whittington et al. 2020](../related-work.md#hippocampus-and-entorhinal-cortex)) | Crude: thalamic routes' fixed offsets (q, v) ([09](../experiments/09-thalamic-attention.md)) | A learned role/position code bound to content (XOR or conjunction). Whittington et al. 2022 show transformer position encodings ≈ the EC structure code and attention ≈ hippocampal binding, so this is the most direct bridge to transformers. Test: sentences a bag of words can't disambiguate ("mary saw john in the kitchen"). |
| **Event boundaries** | Episodes are cut at prediction-error peaks ([Zacks et al. 2007](../related-work.md#hippocampus-and-entorhinal-cortex)) | By hand: "." ends an episode | Surprise-based segmentation from [04](../experiments/04-word-segmentation.md). |
| **Replay / consolidation** | Hippocampal replay trains neocortex slowly (complementary learning systems, [McClelland, McNaughton & O'Reilly 1995](../related-work.md#hippocampus-and-entorhinal-cortex)) | Replay into a cortical semantic store: [17](../experiments/17-consolidation.md) (facts survive after the hippocampus forgets them) | Slow, interleaved cortical learning; prioritised replay; updating facts. |
| **Indexing** | The hippocampus stores an index to distributed cortical patterns ([Teyler & DiScenna 1986](../related-work.md#hippocampus-and-entorhinal-cortex)) | Episodes store the word codes themselves | Store a pointer (sparse index code) bound to cortical codes; pairs naturally with expansion. |

## An index to the cortex, not a store of vectors

**What the literature says.**
- **Index theory** (Teyler & DiScenna 1986; Teyler & Rudy 2007). The hippocampus stores a
  sparse *index* of an experience: a CA3 pattern linked to the distributed cortical pattern
  that was active at encoding. It does not store the experience itself. Retrieval completes
  the index, and the index *reinstates* the cortical pattern. The content of a memory is
  generated in the cortex; the hippocampus brings it back.
- **Evidence of reinstatement.** Recall reactivates the sensory regions active at encoding
  (Wheeler, Petersen & Buckner 2000). Single neurons that fired during an episode fire again
  when it is freely recalled (Gelbard-Sagiv et al. 2008).
- **Re-experiencing needs it.** Reliving a past event, as opposed to knowing that it
  happened, needs the hippocampus (Tulving's autonoetic consciousness).
- **The input is the cortical state, summarised.** The hippocampus does not receive raw
  sensory vectors. Its input comes through the entorhinal cortex, which collects converging,
  already processed output of association cortex: perirhinal cortex for items and objects,
  parahippocampal cortex for place and context. Its output returns by the same route, so
  what it stores and returns is a compressed summary of the cortical state.

**What ours does.**
- The engram store keeps word-in-slot binding ids and the story's earlier bindings: a
  near-symbolic transcript of the sentence, not an index into the cortical state.
- Recall returns words, which the harness decodes and places somewhere: votes, the
  rollout, an entorhinal slot.
- The cortical state at encoding is kept in two places, but neither is used when reading:
  - `MEM_CONTEXT` stores the higher area's slow state with each episode;
  - generative replay keeps the state per row (`row_state`).

**Why it matters.** In [90](../experiments/90-holding-an-open-question.md)–[91](../experiments/91-learned-working-memory-hold.md)
every error was the story's season, stated once at the start. With an index and
reinstatement, recalling any event of this story would bring back the cortical state it was
stored in, the season context included. The learned hold that chose to keep the season was
a workaround for that missing reinstatement.

**To build.**
- Each event stored with its entorhinal summary of the cortical state: the higher area's
  slow state plus the words.
- On recall, that state reinstated into the higher area as context (the path of
  [90](../experiments/90-holding-an-open-question.md)'s `QAREA`), not decoded into words.

## Acetylcholine and norepinephrine: modes of the hippocampus

**Acetylcholine** (medial septum and diagonal band, which also pace theta) switches between
encoding and retrieval (Hasselmo; Hasselmo, Bodelón & Wyble 2002):
- **High acetylcholine (active exploration, attention):** encoding. It suppresses the
  recurrent CA3 connections that complete patterns, leaves the entorhinal input largely
  intact, and enhances plasticity. New input dominates and is stored, instead of being
  absorbed by what it resembles.
- **Low acetylcholine (quiet rest, slow-wave sleep):** retrieval and consolidation. The
  recurrent connections dominate, giving pattern completion, sharp-wave ripples and replay
  to cortex. Within a theta cycle, encoding and retrieval are also thought to alternate by
  phase.
- **The loop:** CA1's comparator (recalled vs arriving) signals novelty to the septum through
  the lateral septum. Novelty raises acetylcholine (encode), familiarity lowers it
  (retrieve), so the hippocampus sets its own mode.

**Norepinephrine** (locus coeruleus, dense to the dentate gyrus and CA3):
- **Phasic bursts on novel or surprising events** strengthen plasticity (β-adrenergic LTP).
  The locus coeruleus also releases dopamine in the hippocampus (Takeuchi et al. 2016;
  Kempadoo et al. 2016). So a salient event around an ordinary one rescues the ordinary
  memory (synaptic tagging and capture).
- **Network reset** (Bouret & Sara 2005): a burst interrupts ongoing processing so the
  networks reorganise around the new event. This is the roadmap's "brake".
- **Adaptive gain** (Aston-Jones & Cohen 2005): phasic mode supports focused exploitation,
  high tonic mode supports exploration and disengagement.

**Against ours.**

| Function | Ours |
|---|---|
| Encoding vs retrieval mode | built as an option (`ACH`, [93](../experiments/93-neuromodulated-hippocampus.md)): novelty of the episode from the reinstating retrieval gates reinstatement. Before it, every step stored and recalled at full strength ([90](../experiments/90-holding-an-open-question.md): recall matched the sentence and pulled in old stories; [92](../experiments/92-reinstatement.md): reinstatement always at full strength) |
| Mode from CA1's mismatch | partly: the hippocampus's novelty is computed (the learned hold uses it) but sets no mode |
| Low acetylcholine: replay | present (sleep replay, consolidation) |
| Salience tag strengthening storage | partly: `REPLAY_TAGGED` (replay priority), growth on surprise; storage strength is not modulated |
| Reset on surprise | missing (the brake) |
| Adaptive exploration | missing: the basal ganglia explore at a fixed 10%, and in [91](../experiments/91-learned-working-memory-hold.md) a rarely rewarded hold was extinguished |

## Novelty should come from prediction, not frequency
Habituation (`novel`) and the rarity cue (`rarest`) force novelty from per-bit
frequency counts. That is why they need sparse random codes, fragile constants, and
why [13](../experiments/13-big-loop.md) couldn't tell a name from "picked up": frequency
is a proxy for "informative". The cortical predictor already computes the real
quantity: **how badly it predicted each word** (`target_probability`, surprise).
"went to the" is predictable after a name; the name and the place aren't. Proposed
upstream step:
- **Encode by prediction error:** store (and cue with) the parts of a sentence the
  predictor failed to predict. This is novelty-weighted encoding as in biology, where
  CA1 mismatch and neuromodulators (acetylcholine, dopamine) gate storage
  ([Hasselmo 2005](../related-work.md#hippocampus-and-entorhinal-cortex);
  [Lisman & Grace 2005](../related-work.md#hippocampus-and-entorhinal-cortex)).
- **Category / role codes from cortex:** names, places and objects cluster in the learned
  syntax codes of [05](../experiments/05-syntax.md). Binding a role tag ("entity") to each
  word would let the loop follow *the entity* rather than *the rarest word*. This is the
  lateral-EC "what" input, and with a position/structure code it is the medial-EC input
  (TEM).
- **Credit-learned cueing:** the predictor's credit readouts (`credited_inputs`) say which
  recalled bits it used for a correct answer; those can train what to cue with.

## Can hops be learned?
Yes, the same way the thalamic routes were. Treat "recall again, cued by item *i*" as a
route in the [10](../experiments/10-route-pool-inhibition.md) pool:
- **Which item to follow:** a `KernelGate` over [item code | context] → RIGHT/WRONG,
  trained in hindsight: once the answer is known, the branch whose recall contained it
  is RIGHT (the hindsight route proposals of [09](../experiments/09-thalamic-attention.md)).
  Branch(3) already does this implicitly: the predictor picks the branch.
- **How many hops (halting):** keep recalling while the predictor's confidence
  (`peek_scored`) stays low; stop when it is confident or nothing new is recalled
  (cf. adaptive computation time, [Graves 2016](../related-work.md#binding-and-attention)).
- Prior art: end-to-end memory networks learn multi-hop attention over stored sentences
  on exactly these bAbI tasks ([Sukhbaatar et al. 2015; Weston et al. 2015](../related-work.md#binding-and-attention)).
  Biologically, prefrontal cortex steers retrieval through nucleus reuniens, and
  cortex–basal ganglia–thalamus loops select which "action" (here, which recall) runs.

## The unmodelled sections: CA1 and subiculum
We have the dentate gyrus and CA3 ([12](../experiments/12-dentate-gyrus-ca3.md)); our CA3
reads straight back to EC. The real output path is CA3 → **CA1** → **subiculum** → EC and
beyond.
- **CA1: comparator and decoder.** It gets CA3's recall (Schaffer collaterals) *and* the
  current EC input directly (temporoammonic path). Match vs mismatch is a novelty signal
  that switches the circuit between encoding and retrieval ([Hasselmo 2005](../related-work.md#hippocampus-and-entorhinal-cortex);
  [Lisman & Grace 2005](../related-work.md#hippocampus-and-entorhinal-cortex)). It also
  re-maps CA3's arbitrary sparse code back into EC's format (heteroassociative decoding)
  and carries time/sequence cells. Here: `recalled AND NOT predicted` would replace
  frequency habituation, and the same mismatch would decide whether to store.
- **Subiculum: output hub and gate.** The main hippocampal output, projecting to EC,
  prefrontal cortex, nucleus accumbens, and via the fornix to the mammillary bodies →
  **anterior thalamus**, with nucleus reuniens closing a loop from prefrontal cortex.
  It has boundary-vector and head-direction-related cells and bursting output. It is the
  hippocampus → thalamus link: the natural home for "recall as a gated route".
- (CA2 is small; mainly social memory and timing. Not needed here.)

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
1. ~~Dentate-gyrus expansion + Hebbian completion~~ ([12](../experiments/12-dentate-gyrus-ca3.md)).
2. ~~Big-loop recurrence~~ ([13](../experiments/13-big-loop.md)); next: learned hops
   (gate over which item to follow + confidence halting).
2b. CA1 comparator: novelty = recalled/current input the predictor didn't predict,
   replacing frequency habituation.
3. Rotation-superimposed history as a structure/order code (word-order test), then
   learned roles (TEM-style).
4. Semantic (learned) codes on the memory path.
5. Surprise-cut boundaries and replay, later.

See also [variable binding](variable-binding.md) and [open questions](../open-questions.md).
