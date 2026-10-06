# Open questions and next steps

Ordered roughly by how much they matter for "reading like a transformer".

## Current (after [23](experiments/23-compaction.md))
The staged plan these feed into is in the [roadmap](roadmap.md).

- **Remaining per-word cost.** After event-based recall
  ([23](experiments/23-compaction.md#5-event-based-hippocampal-recall)), L2/3 matching is
  25–50% and learning 24–38%; kernels are now stored sparsely
  ([23](experiments/23-compaction.md#6-sparse-kernel-storage)). Whole-input memoisation is built
  ([23 §7](experiments/23-compaction.md#7-memoised-interpretation-after-hashlife)): exact,
  but it hits only 4–21%. The per-frame memo is exact but not a net
  win ([23 §8](experiments/23-compaction.md#8-per-frame-memo-hashlifes-sub-nodes-exact-but-not-a-net-win));
  eager per-kernel updates might fix that. Canonical kernels (hash-consing) are built and
  a clear win ([23 §9](experiments/23-compaction.md#9-canonical-kernels-hashlifes-hash-consed-nodes)).
  Chunking was tried and removed as premature
  ([23 §10](experiments/23-compaction.md#10-chunking-hashlifes-time-skipping--removed-from-the-code)): at most
  35% faster answering, limited by how much of the text is habit (1–24% here). Also the dense
  8,192-bit frame copies L4 still builds every word.
- **Why the surprise gate leaves more kernels on elimination** (2,909 against 1,193).
- **Sparse mask storage.** Kernel masks are stored at full input width: 5–50 MB where
  0.3–2.3 MB is needed.
- **Mixing sources** ([24](experiments/24-cortical-hierarchy.md#precision-weighted-mixing-instead-of-switching)):
  L2/3 ranking rules and a switching selector were tried and removed. Precision-weighted
  mixing (`SourceMix`) adds 3–5 points on habit + memory and costs nothing elsewhere. But
  the column's copy kernels already mix most of what it could, and the sources are not
  independent (the column copies them), which makes the mixed confidence overconfident.
  Next: vote only with frames the winner did not read.
- **The chain of areas** ([25](experiments/25-area-chain.md), [26](experiments/26-context-and-readback.md)):
  with a context boundary at each story, three higher areas reach 86–98% on the season
  task, each area extending the reach as its window predicts. Open:
  - ~~Detect boundaries~~: done in [27](experiments/27-boundary-detection.md). A rare
    fact contradicted by a newer one of the same kind gives 90–99%, as good as being
    told. Surprise spikes do not work here. Open: learn the "rare" and "same kind"
    thresholds.
  - **Recency within a story:** a decaying state (each bit survives a sentence with
    probability p).
  - **Order:** position-bound codes.
  - **An event-driven clock:** cost is 2–2.5× with three higher areas.
- **Read-back** ([26](experiments/26-context-and-readback.md)): areas learn to say back
  the fact they hold (80–97%). As rehearsal it extends a single area's reach, but it hurts
  a chain that already holds the fact. Use it where an area must hold something beyond
  its window, and test it with recitation and answering.
- **Literal knowledge** ([30](experiments/30-cortex-driven-saccades.md),
  [31](experiments/31-role-cells-and-transfer.md)): an unseen wording gives 0%.
  - **New names in a role-level rule transfer fully** through learned generalisation
    (pruning unused inputs).
  - **But that pruning breaks rules where identity matters.**
  - **Role cells emerge** but do not help as an extra frame.

  Spawning a general copy beside the specific kernel ([32](experiments/32-general-and-specific.md))
  keeps the transfer and lifts habit to 86–89%. Spawning after 3 confirmations, covered
  only by as-reliable kernels, passes all three tests (transfer, name rule, habit), at
  42,000 kernels. Generalising from replay offline
  ([33](experiments/33-generalisation-during-sleep.md)) passes all three at the default
  size and speed. The schema test ([34](experiments/34-schema-test.md)) fails: new
  name–place pairs are not learned from 1–4 exposures, with or without a schema or
  episodic memory. Next:
  - **a clean context code.** Recency (a fading state, [35](experiments/35-fading-state-and-entorhinal-codes.md))
    is not enough. It must be structural: a learned slot code (medial EC / TEM) with
    episodes bound as slot ⊗ content and recalled by slot;
  - **then a learned copy-from-memory procedure** that carries over to new names.

  Slot ⊗ content memory ([36](experiments/36-slot-binding-memory.md)) learns new pairs
  in one exposure (28–76%), and a setting slot emerges. But the schema group's mix keeps
  trusting its cortex on new names. Familiarity-gated arbitration that keeps learning at
  test brings new names to 28–52% after 2–4 exposures. But the no-schema group learns
  them as well or better. Filtering the memory's readout by the cortex's class expectation
  ([37](experiments/37-schema-supports-episode.md)) lifts new names to 28–44% after one
  exposure, and trained names to 71–82%. Next:
  - **a test where new items follow the learned structure,** for a real schema advantage;
  - **consolidation:** replaying novel episodes' gist in sleep
    ([38](experiments/38-consolidation-of-one-shot-episodes.md)) moves some pairs into the
    cortex after 4 exposures (lesioned 13–31%), little after 1. Interleaved replay feeding
    generalisation makes it steadier (22–27%).
  - **A schema advantage** ([39](experiments/39-schema-advantage.md)): new members of a
    known family are answered with no exposure (55–75% vs 10–28% without the schema),
    from the cortex (54–72% with the hippocampus lesioned).
  - **A family stated once** ([40](experiments/40-family-stated-once.md)): the cortex
    rolls out its expectation ("tom [is a smith] went"), memory supplies the family
    (right 74–91%, lesioned 34–70%), and the cortex's rule answers (45–55% given the
    right family). Next: learn when to step.
  - **Consolidating the stated family** ([41](experiments/41-consolidating-the-stated-family.md)):
    replay strengthens the family rule (held out 40→54% intact) but does not move
    "tom → smith" into the cortex. The column and higher area only predict next words,
    and the higher area's top-down prediction at the surname step is still "is". Next: a
    semantic (associative) area that consolidation can write "tom ~ smith" into.
  - **A semantic store** ([42](experiments/42-semantic-store.md)): 17's cue → content store,
    fed by novelty-prioritised replay, carries the family with the hippocampus off
    (95–100% right; held out 61–67% with answer-trace consolidation, as good as intact).
    Random replay with the same budget misses it. Open: several facts per entity, real
    text, and the store as a mix source.
  - **Learned stepping** ([43](experiments/43-learned-stepping.md)): the basal ganglia
    learn when to look again (56–62% with the hippocampus off, hand rule 61–67%).
    Open: training that rewards thinking, and reliabilities for inner steps credited from
    the outcome, so rollout sources can be mixed.
- **Active reading** ([29](experiments/29-saccades.md)): learned saccades with one
  higher area reach 88–99.6% on the season task, with no story boundaries and at half
  the cost of the three-area chain. Open:
  - **Targets:** learned from a page index of landmarks (90–97%). Done.
  - **Too many regressions:** confidence in the context did not help. A higher cost finds
    about one regression per story on some seeds only. Needed: finer value resolution
    in the selector.
  - **Skipping** confident words.
  - **The books task** with reaching plus looking back.
- **Actions as context** ([28](experiments/28-reading-with-actions.md)): reinstating
  a book's windows when it is opened lifts interleaved reading from chance to 41–54%.
  The ceiling comes from crowded windows (every earlier question's words) and from
  contradiction detection over-firing within a book. Next:
  - **Reaching for a book:** the network chooses the action that brings back the
    context a question needs.
  - **States held per kind,** newest kept.
  - **The audit of hand-supplied pieces** in 28: rarity and kind thresholds, sentence
    units, decoding to word identities, and the keyed context store.
- **Grow areas by need** ([roadmap](roadmap.md)): a shadow bud above the top area,
  learning its residual, promoted when it predicts that residual above chance and pruned
  when not. The number of areas would then follow the task.
- **Output** ([plan](concepts/output-and-self-supervision.md)): answering by speaking
  with abstention, recitation, and read-back with an efference copy ("self" frame).
- **Basal ganglia and L6 together** on the same relays (select by reward among what L6
  lets through).
- **Two-hop's weak seed.** The selector picks well (answer in recall 86–89%), but the
  predictor misses on seed 2 (72–77%).

## Earlier list (items 1, 4 and parts of 7 have since been addressed: held-out binding
[11](experiments/11-episodic-memory.md)–[21](experiments/21-several-routes.md), growth
gating [23](experiments/23-compaction.md))

1. **Variable binding / content-addressed retrieval.** The 0% on held-out bAbI-style
   pairs ([06B](experiments/06-meaning.md)) is the main architectural gap. Try fast
   one-shot binding kernels, XOR binding, or copy kernels
   ([variable binding](concepts/variable-binding.md)). Success criterion: held-out-pair
   accuracy well above chance (17%).
2. **Generalization across similar symbols.** Predictive kernels match exact bits, so
   "the cat sat" teaches nothing about "the dog sat". Feed the learned syntactic and
   semantic codes ([05](experiments/05-syntax.md), [06A](experiments/06-meaning.md))
   into the predictor instead of random word codes. Expect next-word accuracy to beat
   the n-gram baselines, which have the same blind spot.
3. **Fallback when nothing matches.** The back-off n-gram always predicts; the network
   sometimes predicts nothing ([03](experiments/03-book-scale-char-prediction.md),
   [05a](experiments/05-syntax.md)). Options: a depth-0 "unigram" kernel per output, or
   lower-tolerance partial matches.
4. **Growth efficiency at word level.** About 1.3 kernels grown per token. Gate growth on
   repeated surprise in the same context (a counter), not every miss.
5. **Better segmentation.** Add a backward predictor (word starts are predictable
   right-to-left), or iterate segmentation → lexicon → re-segmentation over the book.
   Compare against Bayesian models ([Goldwater et al. 2009](related-work.md#word-segmentation)).
6. **One pipeline instead of separate examples.** Letters → words (stage 2) → syntax
   codes (3) → topical codes (4) → predictor, all online in one `RuntimeNetwork`.
7. **Causal credit for hidden layers.** In the long-gap task
   ([08](experiments/08-credit-assignment.md)), three-factor credit reaches 44% against
   100% for an oracle. Try ablation credit, surprise-timed eligibility, and
   credit-guided growth ([credit assignment](concepts/credit-assignment.md)).
8. **A better higher layer to make top-down pay off.** Top-down bias is worth up to
   +40 points with a perfect word layer and +0.3 with today's
   ([07](experiments/07-top-down-bias.md)). Items 1–3 above are what would improve
   the word layer.
9. **Closing the Hebbian vs count-vector gap** (75 vs 82.5% POS, about 58 vs 71% semantic):
   larger masks with more moves, or weighting moves by surprise.
