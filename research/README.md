# neurocomp research wiki

A running lab notebook for the question **"can this network learn to read the way a
transformer does?"** Each experiment page records the setup, the numbers, what we
concluded, and what changed in the code because of it. Concept pages collect ideas
that cut across experiments. Add to it as you go: new experiment pages get the next
number, and every claim should point at the code or command that reproduces it.

## Where we are (headline results)

| Stage | Test | Network | Best baseline | Page |
|---|---|---|---|---|
| 0 | Next char, stock network, tiny text | bigram level at best (~61%) | 6-gram 92% | [01](experiments/01-stock-network-reading.md) |
| 1 | Next char, tiny text, surprise-driven growth | 89% seen / 82% held-out | 6-gram 92% / best n-gram 79% held-out | [02](experiments/02-predictive-growth.md) |
| 1 | Next char, *Alice* (142K chars, read once) | 56.9% | back-off n-gram 59.0%, best fixed n-gram 53.0% | [03](experiments/03-book-scale-char-prediction.md) |
| 2 | Word boundaries with no spaces | boundary F1 63.9, token F1 30.8 | transitional probability 55.9 / 21.1 | [04](experiments/04-word-segmentation.md) |
| 3 | Next word, Brown (100K tokens) | 9.1% | back-off n-gram 11.1% | [05](experiments/05-syntax.md) |
| 3 | Part-of-speech induction (1000 words) | 74.9% NN agreement | chance 24.6%, count vectors 82.5% | [05](experiments/05-syntax.md) |
| 4 | Semantic groups (92 words, 12 groups) | 52–60% | chance 7.4%, count vectors 70.7% | [06](experiments/06-meaning.md) |
| 4 | Fact binding, held-out name/place pairs | **0%** (100% on seen pairs) | chance 17% | [06](experiments/06-meaning.md) |
| – | Top-down bias: word layer → char layer (60K chars) | 53.4% (+0.3) | oracle word layer 92.6% | [07](experiments/07-top-down-bias.md) |
| – | Long-gap memory: what should hidden units hold? | ablation credit + guided growth 67% | oracle 100%, Hebbian-like 25% | [08](experiments/08-credit-assignment.md) |
| – | Binding via thalamic relay, held-out pairs | oracle route **100%**; learned (hindsight proposals + gradual generalization) **82%** | no relay 0% | [09](experiments/09-thalamic-attention.md) |
| – | Binding by inhibition over a route pool (1–2 facts) | context gate 100% (original), value gate 100% (long) | 4 channel slots 75% / 44.5% | [10](experiments/10-route-pool-inhibition.md) |
| – | Binding by one-shot episodic memory, variable sentence shapes | **98%** held-out | best fixed routes 33–34% | [11](experiments/11-episodic-memory.md) |
| – | Same, with dentate-gyrus expansion + Hebbian CA3 store | **99%** held-out (1–3 facts) | list memory 97.5% | [12](experiments/12-dentate-gyrus-ca3.md) |
| – | Predictor fixes from diagnostics (copy credit, credit-guided growth, trust before depth) | list memory **100%**; CA3 **99.4–100%** on every seed and load | 98.7% / 90–99% before, seeds down to 75% | [12](experiments/12-dentate-gyrus-ca3.md) |
| – | Two-hop questions ("where is the ball?") by big-loop recall | branching recall **80.8%** | single-cue chain 5.5%, one hop 0% | [13](experiments/13-big-loop.md) |
| – | Storing what the predictor didn't predict (CA1 comparator) | 93% / 87% held-out (1 seed) | frequency habituation 98% | [14](experiments/14-ca1-comparator.md) |
| – | Two-hop questions, learned choice of what to follow (basal ganglia) | **89.1%** held-out (5 seeds, bit-sliced counters) | branching recall 80.8% | [15](experiments/15-basal-ganglia-selector.md) |
| – | Basal ganglia choose among thalamic channels (learned routes + memory recall) | **100%** every seed, memory released at every answer | routes only 36–41% | [16](experiments/16-thalamic-gate-memory-channel.md) |
| – | Facts the hippocampus has overwritten, recalled from cortex after replay (consolidation) | **100%** every seed | hippocampus only 0–37% (guessing) | [17](experiments/17-consolidation.md) |
| – | Small replay budget: replay chosen by questions (tagged or awake) | **100%** every seed | random replay 34–66% | [17](experiments/17-consolidation.md#prioritised-replay-questions-decide-what-is-consolidated) |
| – | Question names no one: recall cued by a working-memory slot, loaded by a learned basal-ganglia gate | **100%** every seed | cue at the question 15–17%; trace-credited gate 16–21% | [18](experiments/18-prefrontal-working-memory.md) |
| – | One shared reward from cortical layer 5 (the column's outcome, credited only if the prediction read the choice) for every selector | gates **100%** every seed; hop-2 selector 91 / 69 / 82% | each selector's own answer key: 100% / 86 / 83 / 86%; unattributed outcome: down to 18% / 32% | [19](experiments/19-l5-shared-reward.md) |
| – | Cortical layer 6 gates the thalamic relays (per context, learned from use, no reward) | **100%** every seed, 0.04–0.19 channels per word, 2.4× faster | all channels open: 100%, 3.5 channels per word | [20](experiments/20-l6-corticothalamic-gating.md) |
| – | Different questions need different relays: L6 opens several routes at once, per context (with a warm-up and slow weakening) | **100 / 100 / 98%** held-out, ~1 channel per word | all relays open 91 / 88 / 99%; basal ganglia (one channel) 55–60% | [21](experiments/21-several-routes.md) |
| – | Elimination ("which place hasn't been named?") by a fast-learning inhibitory loop in L2/3, gated by reliability in integers | **100%** every seed, varied stories still 100% | no inhibition 14–17% (chance); ungated: varied falls to 78–84% | [22](experiments/22-fast-inhibition.md) |
| – | Compaction: event-based fast path, uncertainty-gated growth, sleep (downscale, prune, merge by replay) | **5–14× fewer kernels**, accuracy kept on every task; answering 5–62 µs/word (from 160–3,600) with surprise-gated learning, event-based recall, sparse storage and canonical kernels | transformer 16–100 µs/word, but 25–77% on held-out binding where we get 72–100% | [23](experiments/23-compaction.md), [comparison](concepts/brain-transformer-comparison.md) |
| – | A cortical hierarchy: a higher area (sentence + slow state of past surprises) learns the column's errors and feeds back a top-down frame | **81 / 82 / 80%** on a task needing story-level context | column alone 0%, episodic memory 49–52% | [24](experiments/24-cortical-hierarchy.md) |
| – | A chain of areas (windows of 4, 16, 64 sentences), each voting in a precision-weighted mix: how far back a fact can be used | each area extends the reach (4–7 stories back: 5 → 18 → 23%), but accuracy stays low | as frames into the area below: worse (0%) | [25](experiments/25-area-chain.md) |
| – | **Swapping the belief rule** (full / vote / graded / posterior) behind one interface; proposals validated by the module's belief in their fact × their source's credibility | **trusting everything validated 8 false inferences built on a lie and cost that seed 14 points (64 → 50%); every graded rule rejected them** | with honest sources all rules are identical; the effect shows only where the lie came first (one seed of three) | [76](experiments/76-belief-rules.md) |
| – | **The Bayes module:** facts are claims by sources; source trust and claim belief estimated together; conflicts resolved by trust | **a liar's trust is learned from conflicts alone (0.27 / 0.43 against 0.73–0.83 on two seeds); 1:1 conflicts a vote cannot break resolve to the truth** | answers unchanged (this task does not lean on the believed fact); seed 0 barely separates | [75](experiments/75-bayes-module.md) |
| – | **Inferences as proposals:** held until confirmed by the world or supported by independent premises (the weaker of the fact's testimonies and its source events) | 24 proposals per run, all true when the premise was | nothing is corroborated (each new fact stated once; repeated testimony invisible to the store), so nothing is replayed: walk only falls to 22% | [74](experiments/74-proposals-and-premises.md) |
| – | **Source memory:** hippocampal events tagged world / self / proposal; recall and its rarity statistics limited by source | **storing the network's own retellings costs nothing when tagged (89.7%)**; untagged, 35–47% of recalls return its own words and answers fall to 61% | the tag alone was not enough: one's own words changed the store's rarity counts | [73](experiments/73-source-memory.md) |
| – | **The go/no-go sees novelty and agreement:** whether to speak depends on the confidence band × the question's novelty (hippocampal counts, or cortical exposure) × whether every source supports the word | **intact: speaks on 42–50% of new-name questions, 97.6% right (two seeds); answers the sources agree on are 97–98% right** | one seed learns nothing (the newest band never occurs in training); without the hippocampus neither signal helps, and the fixed threshold stays better | [72](experiments/72-go-no-go-signals.md) |
| – | **Speech routed as in the brain:** a vocal tract with its own motor codes, a motor area (inverse + forward models) learned by babbling, a basal-ganglia go/no-go to speak, the forward model's prediction as the efference copy | **every word said after babbling; recitation 89–100% in place through the motor route; the basal ganglia learn silence in the lowest confidence bands** | they still speak at middle bands where new names are 15–36% right (trained on trained names): 61–70% right of answered, below the fixed 0.9 threshold | [71](experiments/71-speech-routing.md) |
| – | **The efference copy and recitation:** own words marked as own (no surprise; a mismatch caught); a story retold from its first words, each spoken word driving the next | **hippocampus plans, cortex speaks: 93–98% of words right in place, 86–94% of content words; the copy catches every altered word** | the cortex alone falls into "the dog ran away" (14–16%, content 1–5%); storing test stories to retell interferes with later answers (90 → 74%) | [70](experiments/70-efference-copy-and-recitation.md) |
| – | **An output buffer: answering by speaking:** at a question the cortex writes the mix's word (or "unknown") to an output buffer and hears its own word in place of the page's | **as good as reading the answer (64.8%, lesioned); abstaining under 0.9 confidence: ~60% answered at 84–91% right** | errors are mostly the wrong place; intact, the mix is overconfident (no useful abstention) | [69](experiments/69-answering-by-speaking.md) |
| – | **Frame words as entities; sparse gating of both stores:** facts read twice (weak or entity frame words lifted into fillers); the hippocampus's answer sent only where the column is unsure; a learned gate tried | **lifting finds "X _ went to the Y" (a family → its places); two hops raise trained names 62.5 → 66.7%; the hippocampus withholds ~60% of its answers with no loss (89.7%)** | new names barely move (65%); the learned gate fails (46%: dense reward, as in 65) | [68](experiments/68-lifted-frames-and-sparse-gating.md) |
| – | **The relation store in reading:** facts parsed by sentence shape (positions where same-shape facts agree = frame), answers as plain fillers in place of the semantic bag; relations of relations learned at sleep (rules that reproduce stated facts) and their inferred facts replayed | **hippocampus lesioned, cooperation + replay: new names 64.4% (65 / 65 / 64%), the best consolidated figure (bag 59.5%); intact 89.8%; grandfather = father ∘ father learned and answered for grandchildren never told** | frequency-based parsing failed (family names frequent everywhere); no relation compositions in this task's world | [67](experiments/67-relation-store-in-reading.md) |
| – | **Typed relations:** facts stored role-bound (and frame-keyed) in the semantic store; a general `RelationStore` learns relations from word statistics (frame = words above the fact's largest frequency gap) and binds fillers by permutation | **rollout, hippocampus lesioned: new names 60 → 63.7% (roles); the relation store answers name + relation (father, mother, inverse, chains) in tests** | role-bound bits given unread to the higher area collapse cooperation (27%); a directed key misses (43%); the harness's frame rule is crude | [66](experiments/66-typed-relations.md) |
| – | **A cue controller:** the basal ganglia choose how the hippocampus is cued (as is, walk, content only, focus), rewarded by recalling the next word | 78.5% at best (learning at test, no walk cost) | the hand-set walk is better (92%); the next-word reward does not see where cueing matters | [65](experiments/65-cue-controller.md) |
| – | **Graded gating:** the semantic store always answers (in the mix, or leaking into the context in proportion to the column's doubt) | lesioned 47% at best | the binary gate of 63 is better (59.5% lesioned, 89% intact) | [64](experiments/64-graded-gating.md) |
| – | **Routes that cooperate:** the semantic store is consulted where the column is unsure of the next word (its own prediction under half reliable), and its content joins the higher area's context; no rollout trigger | **with replay, hippocampus lesioned: new names back from 46% to 59.5%; identical with rollout off (the hand-set choice point is no longer needed)** | consulting on every surprising word hurts (47%); the routes do not add up (semantic 60, replay 48, both 59.5) | [63](experiments/63-routes-cooperate.md) |
| – | **Replay as reading:** inferred events replayed as short stories (the source story's opening + the inferred sentence) through the same reading steps, as training, storing nothing | **hippocampus lesioned: new names 22 → 48% (50 / 43 / 50%) at 50 readings (pairs: 30%); trained names 65 → 67%** | conflicts with the semantic store + rollout route (60 → 46%); with the hippocampus intact 92 → 88% | [62](experiments/62-replay-as-reading.md) |
| – | **Inferred replay:** in sleep the engram store composes events never read (new fact + the schema filler + other stories' rows: "lucy went to the hallway") and the higher area grows kernels from them, keyed on the new word and the state shared across sources | **hippocampus lesioned: new names 22 → 30% (walk only), 60 → 63% with semantic store + rollout; trained names unchanged** | `area.learn` harmed (blame, live-window masks); the season often leaves the area's state; inferred kernels are never corrected | [61](experiments/61-inferred-replay.md) |
| – | **The walk alone** (semantic store and rollout completion off) | **new names, family stated once: 91 / 95 / 91% (no walk 21 / 17 / 25%; with rollout on, 57–64%); trained names 90%** | rollout completion masked the walk; the relation does not reach the cortex (lesioned: chance); next: inferred replay from walks | [60](experiments/60-walk-alone.md) |
| – | **The walk, built into the engram store:** a rare word in the cue bridges to a row from another story; its words then choose among rows that still answer the rest of the cue (matched as words in any slot) | **held out 64 / 63 / 59% (no walk 61 / 64 / 57%); recall changes 25.6% of answers; matches the text graph walk on seed 0 (64 vs 66%)** | most of the composition was already done by the semantic store; 5,000–11,000 walks per run, mostly early in training | [59](experiments/59-engram-walk.md) |
| – | **A story is a graph:** words ∈ sentences ∈ stories, no parser; one hop of a walk replaces a rare word by what it was stated with ("tom" → "tom is a smith") | **family stated once: graph walk 66% held out, text lookup 17%, network 61–64%; trained 85%** | the walk is outside the network; relation types and operators not yet learned | [58](experiments/58-a-story-is-a-graph.md) |
| – | **Memory size vs the text, and a dumb text search** | **the memories hold 6×–575× the raw text (319 KB; 75 KB as word ids, 17 KB gzipped): counts circuit 183 MB, index memory 69 MB, engram 15.5 MB; text search weighting past stories by shared rare words: 92.8% trained, 67.6% held out, beating every memory here** | rows copy the story context into every event; on this task the hippocampus models are expensive lookup, and the network has to show composition, not storage | [57](experiments/57-memory-vs-text.md) |
| – | **An engram store:** rows of binding ids + grid phases, posting and place indexes, theta-cycle dedup, ring-buffer eviction, replay by strength × cortex error | **as fast as the list memory (220–440 µs/word, counts circuit 2,300); with context ids, recall changes 21% of test answers (81% right), trained names 89.5% intact; held out 56–64%; family 97–100% every seed** | a place code cannot replace story-context content on this task; an additive place bonus swamps 1/n-weighted content | [56](experiments/56-engram-store.md) |
| – | **An index memory:** one row per episode (modern Hopfield at zero temperature / hippocampal indexing), event-based recall over an inverted index, successors, lazy strength-based forgetting, consolidation marking | **100% of 16,000 random events in isolation (Hebbian: 0%); on the experiment 49 task, lesioned: 62 / 64 / 65% held out (counts circuit 64 / 59 / 53%), seed 1's family fixed (67 → 100%); intact: trained names 95% (67%); 1.6 vs 2.3 ms/word** | trained names with the hippocampus lesioned 64% vs 67%; rows carry the whole story context; no cortex-error priority yet | [55](experiments/55-index-memory.md) |
| – | **A phase-bound hippocampus, and 1/n / centering in the circuit** (the experiment 49 task) | **backed out: the phase circuit 67 / 42% held out vs 64 / 59% and 3.5× slower (8.1 vs 2.3 ms/word); exact 1/n broke seed 1 (40%) and ran 4× slower; homeostatic centering neutral (64 / 59 / 54% vs 64 / 59 / 53%), kept as an option** | the self-contained hippocampus already costs 10× the list memory (2.3 vs 0.23 ms/word); seed 1's family failure depends on the slot layout | [54](experiments/54-phase-hippocampus-backed-out.md) |
| – | **Integer phase codes and centering without division:** 1/n from a reciprocal table, centering by a per-cell running rate (a homeostatic threshold), and `PhaseAssociate` (integer complex weights) | **homeostatic centering = exact division at every load; phase-coded inputs cancel crosstalk: 100% at 8,000 random events (centered counts 93%); words at slot phases + 1/n: 82% / 50% at 2,000 / 4,000 language-like events (best counts 57% / 13%)** | phase only on the output does not help; exact phase recovery is the weak part; not yet in the hippocampus genome | [53](experiments/53-phase-codes-and-centering.md), [math](concepts/superposition-and-clean-up.md) |
| – | **A Hebbian leaf and the hippocampus as a genome:** `Associate` (Hebbian population: pathways, settling, novelty gain as a population code), the circuit as 29 instructions, encoding and retrieval as the two halves of a theta cycle | **bit-identical to `Hippocampus` (novelty at every store, every recall, the one-shot fact); 1/n scaling + centering holds 100% where the shift-scaled pathway falls to 0% (random codes, 4,000 events)** | CA2, tags, replay and the sparse binding space not yet in the genome; harness not switched | [52](experiments/52-associate-and-hippocampus-genome.md), [math](concepts/superposition-and-clean-up.md) |
| – | **Networks of kernels:** a `Module` interface, base-kernel leaves (predictor with a teaching port, bit op, delay, concat, separate), networks as modules (recursive), and a stack grammar with sub-definitions and loops | **the grammar's column is bit-identical to `CorticalColumn`; a hierarchy of two columns 100% where a column alone is at chance (49%); one-shot sequence replay 7/7 from base kernels; every random genome builds and runs** | the Hebbian pathways, gates and scalar signals are not yet leaves; the episodic harness is not migrated | [51](experiments/51-networks-of-kernels.md) |
| – | **Hippocampal replay for rule extraction:** free-settling (gist) and cued replays feed the higher area's sleep generalisation instead of its waking buffer | **no help: new family members 19–32% (waking buffer 52–59%, none 17–32%)** | replayed events are unordered sets, not the area's own sequential inputs; next: sequence memory in CA3 | [50](experiments/50-replay-for-rule-extraction.md) |
| – | **A sparse binding space:** each (word, slot) binding owns its inputs (1M sparse bits); hashed DG / CA1 projections, novelty-weighted codes, novelty tags with their new inputs, tagged replay into the semantic store | **the hippocampus remembers, tags and teaches the cortex on its own: held out 65/59/59% intact, 64/59/53% lesioned (glued system 69/69/60, 67/67/61); family 100/67/87% from the cortex alone** | the binding space is assigned, not learned; seed 1 partly fails; answer-trace consolidation still harness-recorded | [49](experiments/49-sparse-binding-space.md) |
| – | **The hippocampus on its own:** familiarity from the circuit's own counts, events (one per sentence, with a context code), the dentate gyrus separating content, scaling on every CA3 pathway, replay from the circuit into the semantic store | **runs without the list store; events fix seed 2 but lose seed 0 (family 68/100/61% vs 99/100/34%); cue-free replay never reaches a one-shot fact (0 of 7,200)** | the 8,192-bit binding space is too crowded (~12 bindings per bit) for per-synapse novelty; next: a sparse binding space | [48](experiments/48-hippocampus-on-its-own.md) |
| – | **Integer only:** every per-step computation in bits and integers (`Q16` fixed point, integer random draws), enforced by a test that scans `src/` for floats | **10 of 13 regression entries identical, 3 within 0.2–4 points; random draws aligned with the old stream** | config conversions, reports and the float reference store stay float (marked) | [47](experiments/47-integer-only.md) |
| – | **The full hippocampal circuit:** EC → DG → CA3 (mossy fibres), EC → CA3 (perforant, presynaptically scaled), CA3 recurrent, EC III → CA1 and CA3 → CA1 with the CA1 comparator, novelty-gated encoding, CA2, CA1 → subiculum → EC V, replay, event-based recall | **recalls the one-shot family (100/100/34%) where DG + CA3 guessed; as good as the list memory on average, with no explicit search; 75% of recalls cached** | seed 2 fails for every memory (story-level episodes); EC not a learned layer; replay not yet feeding the cortex; CA2 recency hurts here | [46](experiments/46-full-hippocampus.md) |
| – | **Superposed thought; the learned hippocampus in the loop:** evidence sources OR-ed within the expectation (no order, no per-source weights); DG + CA3 recall in place of the list memory | **evidence superposition matches the fixed order (69/69/60%), learned stepping 58–67%; CA3 alone does not recall the one-shot fact (family 54–68%, a guess), the semantic store carries it (57–68%)** | CA3 lacks novelty-gated encoding; no CA1 / subiculum / EC layers in the loop; 10× slower | [45](experiments/45-superposed-thought-and-ca3.md) |
| – | **Closing the loop:** an internal step feeds the source's output vector, gated by the expectation (bitwise AND), back as input; no decode / re-encode | **identical answers with the hand trigger (69/69/60%); learned stepping 58–61% (from 54–57%); 9–38% of fed-back vectors are partial or blended codes** | choice points, source order and step cap still hand-set | [44](experiments/44-closed-loop.md) |
| – | **Learned stepping:** the basal ganglia choose whether to start a rollout ("look again") or read on, rewarded by the answer minus a step cost; source mixing for rollout words tested | **hippocampus off: 56–62% held out (hand rule 61–67%, none 15–43%); trusts the semantic store's offers, rarely the column's** | learns mostly at test (training offers little where thinking pays); mixing rollout sources fails without outcome-credited reliabilities | [43](experiments/43-learned-stepping.md) |
| – | **A semantic store:** the cue → content store of 17, fed by novelty-prioritised sleep replay, read by the rollout | **with the hippocampus off, the right family 95–100% (from 38–68%); held out 49–58%, 61–67% with answer-trace consolidation too (as good as intact)** | learned from 6–18 replays in a small world; one cue word per fact; not yet a mix source | [42](experiments/42-semantic-store.md) |
| – | **Consolidating the stated family:** answer, step and association replay; the higher area as a rollout source | **consolidation strengthens the rule (held out 40→54% intact, 33→44% lesioned) but not the link: lesioned, the family is a constant guess** | the cortex has no associative store ("tom ~ smith"), only next-word predictors | [41](experiments/41-consolidating-the-stated-family.md) |
| – | **"Tom is a smith":** a family stated once, questions name only the first name; the cortex rolls out its expectation, memory fills the family | **right family 74–91% (lesioned 34–70%); held out 35–51% vs 19–26% without completion; given the right family the cortex's rule answers 45–55%** | the trigger is surprise, not a learned choice; the stated family is not yet consolidated | [40](experiments/40-family-stated-once.md) |
| – | **A schema advantage:** new members of a known family (the family decides the place) | **answered with no exposure: 55–75% (schema) vs 10–28% (no schema); survives a hippocampal lesion (54–72%)** | one-trial learning of a family stated once is the next test | [39](experiments/39-schema-advantage.md) |
| – | Consolidation: novel episodes' gist replayed to the higher area in sleep; hippocampal lesion at test | after 4 exposures the lesioned cortex answers 13–31% of new names (8–18% without consolidation); after 1 exposure little | interleaved replay with generalisation makes it steadier (22–27% on every seed); one-exposure consolidation still out of reach | [38](experiments/38-consolidation-of-one-shot-episodes.md) |
| – | The schema supports the episode: the cortex's class expectation filters the memory's readout | **new names after one exposure 28–44% (from 16–28%), after four 39–66%; trained names up to 71–82%** | not yet better than the no-schema group (36% vs 44% mean after one exposure) | [37](experiments/37-schema-supports-episode.md) |
| – | Slot ⊗ content memory (after TEM): learned slot cells, words bound by slot rotation, rarity-weighted recall by slot | **one-exposure learning works in the store: new names 28–76% after one exposure** (0–3% unseen); a setting slot emerges unsupervised | with familiarity-gated arbitration that keeps learning at test, the answer reaches 28–52% after 2–4 exposures (memory alone 40–70%); the schema does not yet speed learning | [36](experiments/36-slot-binding-memory.md) |
| – | A fading state (drifting temporal context, as in lateral entorhinal cortex) for the higher areas | small gains (season 12–28% without boundaries; schema test new names 14–24%) | recency is not relevance: the code needed is structural (medial-EC / TEM) | [35](experiments/35-fading-state-and-entorhinal-codes.md) |
| – | The schema test (after Tse et al. 2007): new name–place pairs seen 1, 2 or 4 times, with and without a learned schema, with and without (context-bound) episodic memory | **not learned: new names 5–18%**, no better than never seen | one-shot kernels are keyed on incidental filler; recall is not context-specific | [34](experiments/34-schema-test.md) |
| – | Generalisation during sleep: general rules formed from replay and tested on it before they are kept; specifics kept | **new names 63–78% = known; name rule no loss; habit 81–85%, at the default size and speed** | sleep's merge on top costs 1–12 points | [33](experiments/33-generalisation-during-sleep.md) |
| – | General and specific kernels side by side: generalisation spawns a general copy, the specific kernel stays | spawning after 3 confirmations, covered only by as-reliable kernels: **new names = known names (61–81%), name rule 69–73% (no loss), habit 82–85%** | 42,000 kernels (8× slower); sleep compacts 20× but costs 5–15 points | [32](experiments/32-general-and-specific.md) |
| – | Learned roles: role cells from competitive Hebbian learning on the column's expectations; transfer to names never seen in training | role cells find "sentence start", "noun after the" and more, unsupervised. **Generalisation by pruning gives full transfer: new names 61–72% = known names 65–71%**, with 8× fewer kernels | the same pruning breaks name-dependent rules (21–30%) | [31](experiments/31-role-cells-and-transfer.md) |
| – | Cortex-driven saccades (the column's possible continuations as the basal ganglia's context) and a transfer test with an unseen question wording | 86–99% on the trained wording; new wording **0%**, even with a perfect look-back | the network's knowledge is keyed on word identities: the baseline for schemas | [30](experiments/30-cortex-driven-saccades.md) |
| – | Active reading: the basal ganglia choose saccades (read on, look back to the previous sentence or the page top); the page is external memory | **99.6 / 88 / 97%** with one higher area, no boundaries needed, half the cost of the three-area chain | reading straight through 3–11% | [29](experiments/29-saccades.md) |
| – | Reading with actions: "@open book" reinstates that book's saved context; three books read in interleaved sessions | 41–54% (reinstate + contradiction detection), chance 25–33% without | limited by crowded windows, not by the actions | [28](experiments/28-reading-with-actions.md) |
| – | Story boundaries detected by the network: a rare fact contradicted by a newer one of the same kind (kinds learned from neighbouring words) | **91 / 90 / 99%**, as good as being told (86 / 96 / 98%) | surprise spikes do not mark story starts here | [27](experiments/27-boundary-detection.md) |
| – | Same chain with a context boundary at each new story; self-supervised read-back (say back the fact you hold, hear it) | **86 / 96 / 98%**, and 87–100% with the fact 16+ stories back | no boundary 9–23%; read-back helps one area (3–11 → 27–33%), not the chain | [26](experiments/26-context-and-readback.md) |

**Short version.** Local growth rules driven by surprise turn the network into a
competent variable-order sequence memory (comparable to PPM-style n-gram back-off
with ~5x less storage), and the repo's Hebbian mask rule learns usable syntactic and
semantic word categories once the codes are sparse enough. What is missing for
"reading like a transformer" is **variable binding / content-addressed retrieval**:
the network memorizes combinations it has seen and cannot answer about new ones
([concept page](concepts/variable-binding.md)).

**Top-down feedback and credit assignment** ([07](experiments/07-top-down-bias.md),
[08](experiments/08-credit-assignment.md)). Bias from a higher layer works
mechanically: a perfect word layer lifts character prediction from 53% to 93%. But it
only helps as much as the higher layer knows. With today's word layer it adds 0.3
points. A credit-assignment operation *is* needed once a layer must supply useful
features to another. Activity-driven (Hebbian) choice fails, and naive credit is
fooled by co-active inputs. Ablation (counterfactual) credit plus credit-guided growth
gets 67% of the way where the oracle gets 100%.

**Attention via a thalamic relay** ([09](experiments/09-thalamic-attention.md)).
Routing "what followed the earlier occurrence of this word" into the predictor (an
induction head in thalamic form) takes held-out fact binding from 0% to 100%. Learning
*which* route to use is the bottleneck. Hindsight proposals ("which route would have
carried what I failed to predict?") find it, and gradual synapse-level credit ("drop
inputs that keep being irrelevant when you're right") stops the network memorizing
names: 82% on held-out pairs, all learned, with nothing task-specific put in by hand.

## Pages

Experiments (chronological)
1. [Stock network on a reading task](experiments/01-stock-network-reading.md)
2. [Predictive learning with surprise-driven growth](experiments/02-predictive-growth.md)
3. [Book-scale character prediction](experiments/03-book-scale-char-prediction.md)
4. [Word segmentation and recognition without spaces](experiments/04-word-segmentation.md)
5. [Syntax: next-word prediction and part-of-speech induction](experiments/05-syntax.md)
6. [Meaning: topical similarity and fact binding](experiments/06-meaning.md)
7. [Top-down bias from a higher layer](experiments/07-top-down-bias.md)
8. [Credit assignment for a hidden layer](experiments/08-credit-assignment.md)
9. [Thalamus-like relay as attention](experiments/09-thalamic-attention.md)
10. [Attention by inhibition: a route pool with learned gating](experiments/10-route-pool-inhibition.md)
11. [Episodic autoassociative memory](experiments/11-episodic-memory.md)
12. [Dentate gyrus expansion and a Hebbian CA3 store](experiments/12-dentate-gyrus-ca3.md)
13. [Big-loop recurrence: chaining recalls for two-hop questions](experiments/13-big-loop.md)
14. [A CA1-style comparator: store what wasn't predicted](experiments/14-ca1-comparator.md)
15. [A basal-ganglia selector: learning which recalled item to follow](experiments/15-basal-ganglia-selector.md)
16. [Memory recall as a thalamic channel, chosen by the basal ganglia](experiments/16-thalamic-gate-memory-channel.md)
17. [Consolidation: hippocampal replay into a cortical semantic store](experiments/17-consolidation.md)
18. [Prefrontal working memory: a gated slot that cues recall](experiments/18-prefrontal-working-memory.md)
19. [Layer 5 as the shared reward: one dopamine signal for every selector](experiments/19-l5-shared-reward.md)
20. [Layer 6 corticothalamic gating: cortex learns which relays to let through](experiments/20-l6-corticothalamic-gating.md)
21. [Several routes needed: L6 opens more than one relay where each is used](experiments/21-several-routes.md)
22. [A fast-learning inhibitory loop in L2/3](experiments/22-fast-inhibition.md)
23. [Compaction: an event-based fast path, uncertainty-gated growth and sleep](experiments/23-compaction.md)
24. [A cortical hierarchy: a higher area that predicts the column's errors](experiments/24-cortical-hierarchy.md)
25. [A chain of cortical areas: does each area reach further back in time?](experiments/25-area-chain.md)
26. [Context boundaries and read-back: keeping the right facts in reach](experiments/26-context-and-readback.md)
27. [Detecting context boundaries: when a fact is contradicted, a new story has begun](experiments/27-boundary-detection.md)
28. [Reading with actions: opening a book brings back its context](experiments/28-reading-with-actions.md)
29. [Active reading: look back instead of holding everything](experiments/29-saccades.md)
30. [Cortex-driven saccades, and a first transfer test](experiments/30-cortex-driven-saccades.md)
31. [Learned roles and transfer to new names](experiments/31-role-cells-and-transfer.md)
32. [General and specific kernels side by side](experiments/32-general-and-specific.md)
33. [Generalisation during sleep: general rules formed offline from replay](experiments/33-generalisation-during-sleep.md)
34. [The schema test: can a new fact be learned in one exposure? (not yet)](experiments/34-schema-test.md)
35. [A fading state, and what the entorhinal cortex would add](experiments/35-fading-state-and-entorhinal-codes.md)
36. [Slot ⊗ content memory: one-exposure learning works in the hippocampus, not yet in the answer](experiments/36-slot-binding-memory.md)
37. [The schema supports the episode: class-level predictions](experiments/37-schema-supports-episode.md)
38. [Consolidating one-shot episodes into the cortex (partly)](experiments/38-consolidation-of-one-shot-episodes.md)
39. [A schema advantage: new members of a known family](experiments/39-schema-advantage.md)
40. ["Tom is a smith": one statement, completed from memory, applied by the cortex](experiments/40-family-stated-once.md)
41. [Consolidating the stated family: the rule strengthens, the link does not move](experiments/41-consolidating-the-stated-family.md)
42. [A semantic store: "tom is a smith" consolidated into the cortex](experiments/42-semantic-store.md)
43. [Learned stepping: the basal ganglia decide when to look again](experiments/43-learned-stepping.md)
44. [Closing the loop: the network runs on its own output](experiments/44-closed-loop.md)
45. [Superposed thought without per-source weights, and the learned hippocampus in the loop](experiments/45-superposed-thought-and-ca3.md)
46. [The full hippocampal circuit](experiments/46-full-hippocampus.md)
47. [Integer only: the model computes with bits and integers](experiments/47-integer-only.md)
48. [The hippocampus on its own: what works, and the binding space that blocks it](experiments/48-hippocampus-on-its-own.md)
49. [A sparse binding space: the hippocampus remembers, tags and teaches on its own](experiments/49-sparse-binding-space.md)
50. [Hippocampal replay for rule extraction: neither prototypes nor events teach the rule yet](experiments/50-replay-for-rule-extraction.md)
51. [Networks of kernels: modules, recursion, and a grammar that builds them](experiments/51-networks-of-kernels.md)
52. [A Hebbian leaf, and the hippocampus as a genome](experiments/52-associate-and-hippocampus-genome.md)
53. [Integer phase codes, and centering without division](experiments/53-phase-codes-and-centering.md)
54. [A phase-bound hippocampus, and 1/n and centering in the circuit: backed out](experiments/54-phase-hippocampus-backed-out.md)
55. [An index memory: one row per episode, winner-take-all recall](experiments/55-index-memory.md)
56. [An engram store: rows of binding ids and grid phases](experiments/56-engram-store.md)
57. [How big is a memory, compared with the text? And a dumb text search](experiments/57-memory-vs-text.md)
58. [A story is a graph: one hop of a walk composes what lookup cannot](experiments/58-a-story-is-a-graph.md)
59. [The walk, built into the engram store](experiments/59-engram-walk.md)
60. [The walk alone: 92% where the stated fact meets the rule, and none of it in cortex](experiments/60-walk-alone.md)
61. [Inferred replay: teaching the cortex what the walk composes](experiments/61-inferred-replay.md)
62. [Replay as reading: inferred stories read through the same steps](experiments/62-replay-as-reading.md)
63. [Routes that cooperate through the cortex's uncertainty](experiments/63-routes-cooperate.md)
64. [Graded gating: both stores always answer](experiments/64-graded-gating.md)
65. [A cue controller: the basal ganglia choose how the hippocampus is cued](experiments/65-cue-controller.md)
66. [Typed relations: role- and frame-bound facts, navigable by relation code](experiments/66-typed-relations.md)
67. [The relation store in reading, and relations of relations](experiments/67-relation-store-in-reading.md)
68. [Frame words as entities, and sparse gating of both stores](experiments/68-lifted-frames-and-sparse-gating.md)
69. [An output buffer: answering by speaking, with abstention](experiments/69-answering-by-speaking.md)
70. [The efference copy and recitation](experiments/70-efference-copy-and-recitation.md)
71. [Speech routed as in the brain: motor area, basal-ganglia go/no-go, forward-model efference copy](experiments/71-speech-routing.md)
72. [What the speak/stay-silent choice should see: novelty and agreement](experiments/72-go-no-go-signals.md)
73. [Source memory: what was read versus what the network said](experiments/73-source-memory.md)
74. [The network's inferences as proposals: validated before they count](experiments/74-proposals-and-premises.md)
75. [The Bayes module: source trust, and conflicts resolved by it](experiments/75-bayes-module.md)
76. [Swapping the belief rule: what graded belief buys](experiments/76-belief-rules.md)

Concepts
- [Brain, this network, and a transformer: function vs speed and memory](concepts/brain-transformer-comparison.md)
- [Surprise-driven growth and recycling](concepts/surprise-driven-growth.md)
- [Superposition and clean-up: the math of a Hebbian population](concepts/superposition-and-clean-up.md)
- [The Hebbian mask rule](concepts/hebbian-mask-rule.md)
- [Sparse codes and collisions](concepts/sparse-codes-and-collisions.md)
- [Variable binding: the gap to transformers](concepts/variable-binding.md)
- [Relational memory: a learned, directed relation graph, and sparse communication with the cortex](concepts/relational-memory.md)
- [Epistemics: what the network believes, and why (source memory, proposals, trust)](concepts/epistemics.md)
- [Top-down bias](concepts/top-down-bias.md)
- [Credit assignment](concepts/credit-assignment.md)
- [Hippocampal-formation functions: what we have and what's missing](concepts/hippocampal-functions.md)
- [Basal ganglia and cerebellum: two more credit-assignment loops](concepts/basal-ganglia-and-cerebellum.md)
- [Probability in bits: what is bitwise now, and how to keep it that way](concepts/probability-in-bits.md)
- [Architecture map: which brain systems we model, and how they connect](concepts/architecture-map.md)
- [Output and self-supervision: speaking, and hearing yourself speak](concepts/output-and-self-supervision.md)

Reference
- [Bugs found and fixed](bugs-and-fixes.md)
- [Related work and reading list](related-work.md)
- [Open questions and next steps](open-questions.md)
- [Roadmap: towards reliable higher-order thinking](roadmap.md)

## Reproducing

```sh
./scripts/fetch_corpora.sh                       # Alice, Bryant stories, tagged Brown -> data/
cargo test                                       # unit tests (kernels, growth, runtime)
cargo run --release --example read_text          # experiments 01-02 (seconds)
cargo run --release --example read_corpus        # experiment 03 (~12 min for all configs)
cargo run --release --example segment_words      # experiment 04 (~3 min)
cargo run --release --example syntax             # experiment 05 (~2 min)
cargo run --release --example meaning            # experiment 06 (~2 min)
cargo run --release --example topdown            # experiment 07 (~4 min)
cargo run --release --example credit             # experiment 08 (~15 min for all configs; pass 20000 for the long run)
cargo run --release --example thalamus           # experiments 09-10 (~15 min per policy set; see POLICIES)
cargo run --release --example episodic           # experiments 11-14 (~35 min; POLICIES=ca3 for 12; TASK=twohop POLICIES=loop for 13; NOVELTY=prediction for 14)
```

Learning uses `rand::thread_rng()` inside the kernels, so numbers move by a point or
two between runs; the pages quote single runs unless noted.
