# Roadmap: towards reliable higher-order thinking

Where the project goes after the hierarchy ([24](experiments/24-cortical-hierarchy.md)).
"Reliable higher-order thinking" is taken to mean three abilities the network does not
yet have:
- knowing how reliable each of its sources is, and saying so;
- chaining internal steps, each checked before the next;
- applying a rule to cases it has never seen.

Each stage below names what to build, the brain system it follows, and the test that
says it worked. The stages are ordered so that each one rests on the one before it: a
reasoning loop that runs on badly calibrated confidence only compounds its errors.

## 1. Trustworthy reliability (in progress)
**Why.** Every later stage trusts the kernels' hit rates: to choose sources, to stop
thinking, to abstain. Under surprise-gated learning only the winner was scored on a
correct guess, so a kernel that did not win was scored only on the winner's misses, and
its rate was biased.

**Build.**
- **Score every matching kernel on every step** (`SCORE_ALL`, done).
- **Probation for new kernels** (`PROBATION=0.8`, done). A new kernel persists, matches
  and is scored, but changes an answer only after its rate has reached 0.8 once (three
  hits without a miss), or where no proven kernel matches.
- **A calibration report** (`CALIB`, done): accuracy per confidence bucket and the
  expected calibration error. "0.8 confident" should be right about 80% of the time.
- **Abstention:** below a confidence threshold, answer "unknown". Score accuracy when
  answering and coverage, not only accuracy.

**Brain.** Synapses consolidate only after repeated success (synaptic tagging and
capture, Frey & Morris 1997). Confidence signals in orbitofrontal and parietal cortex
track the probability of being right (Kepecs et al. 2008; Kiani & Shadlen 2009).

**Test.** ECE below 0.05 on every task, with no loss of accuracy against the default.

**Status** ([24](experiments/24-cortical-hierarchy.md#scoring-every-kernel-probation-and-calibration)):
scoring every kernel, probation and the calibration report are built.
- `SCORE_ALL=1 PROBATION=0.8 PROBATION_BEAT=1` matches the default everywhere with fewer
  kernels.
- The absolute floor alone lifts habit to 93–95% but breaks memory copying.
- Answering only at confidence 0.8 or above already gives 93–100% accuracy on the memory
  tasks, but not on habit (70–86%).
- Memory-derived answers are underconfident (topic ECE 0.21). Abstention is not built.

## 2. Learned source arbitration
**Why.** Memory, top-down and the column's own context compete inside the kernel
ranking, so frame order or kernel age decides between them
([24](experiments/24-cortical-hierarchy.md#a-thalamic-gate-on-the-top-down-frame)).

**Build.** A basal-ganglia selector per context that chooses which frame to trust. It is
rewarded by the L5 outcome ([19](experiments/19-l5-shared-reward.md), with the
advantage baseline), and the thalamic gate applies its choice
([20](experiments/20-l6-corticothalamic-gating.md)). Arbitration then becomes an
explicit, inspectable decision instead of a side effect of growth.

**Brain.** Precision weighting in predictive coding (Feldman & Friston 2010): pulvinar
and the reticular nucleus scale each stream by its expected reliability; the basal
ganglia select among competing cortical inputs.

**Test.** Habit + memory above 85% on every seed, with topic, two-hop and give unchanged.

**Status** ([24](experiments/24-cortical-hierarchy.md#precision-weighted-mixing-instead-of-switching)):
the switching selector was built and removed. Its replacement is precision-weighted
mixing (`SourceMix`, `MIX=1`): every source votes with its learned reliability per
context and per-question confidence.
- Habit + memory 76 → 79% with no loss elsewhere.
- The column's own copy kernels already do most of the mixing.
- Next: make sources independent, by voting only with frames the column's winner did not
  read, so agreement is not counted twice.

### Learned input routing: no fixed slots (proposed)
**Why.** Each cortical kernel reads its input as a row of fixed-width slots, and the
column's row is laid out by hand: [current word | memory | top-down | previous input]. What
is learned is only which slots a kernel reads (it grows one slot deeper when surprised).
Which source sits in which slot, and that a source gets a slot at all, is set in the
harness. The gates on the top-down channel ([83](experiments/83-compute-only-where-needed.md),
[85](experiments/85-top-down-as-a-trusted-witness.md)) only scale what goes into a slot
someone else placed. And a new area grown by need ([84](experiments/84-areas-grow-by-need.md))
cannot reach the column's input at all; it can only vote in the mix.

**Build.**
- **Sources as relays, not slots.** Every input to the column (the current word, memory,
  each higher area, the relation store, the previous input) is a relay channel into L4. A
  new source, such as a promoted area, is one more channel, with no change to the column.
- **Superposed and bound, not positioned.** Each channel's content is bound to its source
  by a fixed per-source rotation (as the relation store binds a word to its position) and
  the channels are ORed into one input of fixed width. A kernel samples bits, so it keys on
  "this word, from this source" wherever it came from, and the input does not widen as
  sources are added.
- **Thalamic gain per channel and context.** The thalamus sets each channel's share (a
  fixed subset of its bits, as in 85's scaling), learned from its counterfactual record:
  the column's prediction with the channel against without it. The L6 gating of
  [20](experiments/20-l6-corticothalamic-gating.md)–[21](experiments/21-several-routes.md)
  and the route pool of [10](experiments/10-route-pool-inhibition.md) did this for memory
  routes; here it covers every input.
- **Risk.** Superposition raises density: many channels ORed together blur. Gains that
  close unhelpful channels keep it sparse; a cap on active bits (k-winners) is the fallback.

**Brain.** L4 receives converging thalamic relays, and the higher-order thalamus (pulvinar)
routes and scales cortical signals between areas by context (Saalmann et al. 2012), so what
an area reads is set by thalamic gain rather than by fixed wiring alone.

**Test.** The suite at parity with the fixed layout; a promoted area (84) reaching the
column with no change to its input width; story boundary growing from one area with its
new areas read by the column, not only voting.

### Three learning systems (proposed)
**Why.** The brain combines learners of different speeds and teachers: the
**hippocampus** stores single events fast and in detail; the **cortex** learns slowly and
interleaved, so it generalises without overwriting (complementary learning systems,
McClelland, McNaughton & O'Reilly 1995); the **cerebellum** learns from errors, each output
with its own teacher; the **basal ganglia** learn from reward (Doya 2000). Here the
hippocampus (the engram store) and the basal ganglia (the selectors) match. The cortex and
the cerebellum do not: the column's kernels grow in one shot on every miss, each output
corrected by the next word, which is the cerebellum's rule, and slow generalisation is
bolted on at sleep. That fits two things the runs keep showing: a change at 0.2% of training
steps moves a seed by 20 points (every early event shapes what follows), and seen pairs beat
held-out ones on the memory entries (a fast learner memorises).

**Build.**
- **Cerebellum:** the current fast kernels, kept as the precise corrector, one more source
  of the column's prediction.
- **Cortex:** a separate slow learner reading the same input: learning is stochastic and
  rare (a miss grows a kernel only with a small probability, so a context must recur before
  it is learned, as stochastic synapses with low transition probabilities learn slowly,
  Amit & Fusi 1994), with near-miss generalisation, from waking and replay alike.
- **Hippocampus:** as now, with its output as entorhinal feedback (below) and replay
  priority from a prediction-error tag set at encoding.
- **How they combine:** each is a source in the thalamic mix, weighed by its record per
  context; the slow cortex should win where a context is common, the hippocampus where it
  is new, the fast kernels where they are reliable.

**Test.** Held-out (new names) at least as good as now, seen pairs no better than held-out
by much; the suite at parity; smaller spread across seeds.

### A workspace and a brake (proposed)
**Why.** Most arbitration between the systems is unconscious and goes by reliability,
which the thalamic mix already does. What is missing is what the brain adds when that
arbitration is uncertain, conflicted or wrong:
- **No shared picture.** Each module keeps its own state; no single winning interpretation
  is broadcast back to all of them as the context they share and learn from.
- **No salience filter.** Routing gains turn down unhelpful channels, but nothing decides
  what deserves the whole network's attention.
- **No brake.** Speech has an efference copy, so the network notices its own words, but
  nothing cancels an answer already under way when a conflict is detected.

This may also bear on the crutch problem ([connection audit](concepts/connection-plausibility.md#coupling-who-learns-from-which-error)):
a recalled answer that is broadcast becomes what every system learns from, instead of
silently removing the surprise that would have taught them.

**Build.**
- **Ignition.** When the mix's evidence for one interpretation clears a threshold (learned
  per context, as the answer-or-unknown go/no-go is), it is broadcast as one shared frame
  to the column, the higher areas, the hippocampus's cue and the basal ganglia, and held
  for a few words. Below it, nothing is broadcast and the systems run on their own.
- **Salience.** What may ignite is gated by surprise and by the record of each source
  (the routing shares), so frequent, predicted content never takes the workspace.
- **Brake.** When the sources disagree strongly, or the efference copy shows a mismatch, a
  stop (learned like the go/no-go, but fast) cancels the answer being spoken; the network
  says "unknown" or re-reads instead.
- **Gain.** Expected uncertainty (acetylcholine-like) and unexpected change
  (noradrenaline-like) scale the learning rate and the channels' shares, instead of fixed
  constants.

**Brain.** The global workspace: prefrontal and parietal networks with long-range links
and thalamic coordination, whose sudden sustained "ignition" correlates with conscious
access (Baars 1988; Dehaene & Changeux 2011). The salience network (anterior insula,
dorsal anterior cingulate) and the thalamic reticular nucleus select and suppress.
Prefrontal cortex holds the goal and biases every area toward it (Miller & Cohen 2001).
The stop circuit, right inferior frontal gyrus to the subthalamic nucleus (the basal
ganglia's hyperdirect pathway), halts an action under way within about 200 ms, triggered
by conflict signals from the anterior cingulate (Aron et al. 2014; Botvinick et al. 2001).
Neuromodulators set gain and learning rate (Yu & Dayan 2005).

**Test.** Fewer confident errors (answers given and wrong); the crutch effect smaller
(an entry trained with the hippocampus routed in, tested with it lesioned); held-out
accuracy at least as good; the brake's stops mostly on answers that would have been wrong.

### Grown, not designed (proposed)
**Why.** The reading network is wired by hand: a harness of about 6,500 lines and 267
settings, a list of settings per task ([audit](concepts/connection-plausibility.md#evolution-and-development-how-the-architecture-arises)).
The genome grammar of [51](experiments/51-networks-of-kernels.md)–[52](experiments/52-associate-and-hippocampus-genome.md)
can express the column and the hippocampus exactly, but builds nothing the suite runs.

**Build.**
- Express the suite's systems as genomes: column, higher area, hippocampus, cerebellum,
  thalamic routing, basal-ganglia go/no-go, each a definition built from base kernels.
- One genome for the reading network, the same for every task; what differs between tasks
  is only the data. Growth (areas by need, kernels on surprise, routing in a critical
  period) does the rest.
- Then search: mutate genomes, score them on the suite, keep what improves it, as
  evolution would. The hand-written modules (truth discovery, the relation parser, the
  curiosity ranking) become targets a searched circuit must match.

**Test.** One genome scoring within noise of the hand-tuned settings on every entry.

### Relations from structure, not a parser (proposed)
**Why.** The relation store finds a fact's relation by counting neighbours over exact
positions and composes rules by counting paths: hand-written algorithms over kernels
([audit](concepts/connection-plausibility.md#relations-how-the-network-learns-who-is-what-to-whom)).
The hippocampus already does the plausible thing: role codes bound to content.

**Build.**
- Key facts by the role cells' codes (structure) instead of the parser's frames, and
  answer by completion (the kernel class of [82](experiments/82-fact-completion.md),
  made cheaper) instead of lookup.
- Compose by walking structure (the engram walk) and by generative replay, with the
  rule-counting kept only as the specification the learned version is checked against.
- Trust learned as each source's record (as the thalamic mix learns each internal
  source's), with the truth-discovery algorithm as the specification.

**Test.** The relation and belief entries at parity with the parser; the family-tree
compositions (grandfather = father ∘ father) learned.

## 3. Thinking in steps
**Why.** Two-hop chains one recall into another, but the chain is fixed by hand. Reasoning
is a sequence of internal steps whose number depends on the question.

**Build.** An internal loop:
1. L5 predicts.
2. If the prediction is confident, answer.
3. If not, the prefrontal gate ([18](experiments/18-prefrontal-working-memory.md)) holds
   it as a cue; memory or the higher area is queried again, and the column re-predicts.
4. Stop when confident, or give up after N steps.

The basal ganglia choose "answer", "look again" or "give up", rewarded for being right
and charged a small cost per step, so the network learns how much thinking each question
deserves.

**Brain.** Cortico-basal ganglia-thalamo-cortical loops gate working memory updates
(O'Reilly & Frank 2006); evidence accumulates to a bound before a decision (Gold &
Shadlen 2007).

**Test.** Three- and four-step chains at the accuracy of two-hop today, with the number
of steps used tracking the number needed.

**Status** ([40](experiments/40-family-stated-once.md)): a first internal step. When the
page contradicts a definite expectation, the network continues that expectation inside,
with memory supplying the specifics, until the page fits again ("tom [is a smith] went to
the"). Its length follows the need, not a hand-set chain. It is triggered by surprise; the
basal ganglia choice of answer / look again / give up is not built.

**Update** ([43](experiments/43-learned-stepping.md)): "look again" vs "read on" is now a
basal-ganglia choice at each contradiction, rewarded by the answer minus a step cost. It
recovers most of the hand rule's benefit (hippocampus off: 56–62% against 61–67%) but
learns mostly at test, since training holds little where thinking pays. Mixing the
rollout's sources needs reliabilities credited from the outcome: there is no page word to
check an inner step against. Still to come: "give up", and tasks that need several steps.

## 4. Abstraction: rules over roles
**Why.** Held-out binding works because answers are copied from memory. Rules ("X gave Y
to Z, so Z has Y") are still learned per word, so they do not transfer to new names.

**Build.** A role layer in the higher area or the prefrontal slots that binds role to
filler (giver, object, receiver). Kernels key on roles, so a rule learned with "john"
applies to "mary". See [variable binding](concepts/variable-binding.md).

**Brain.** Prefrontal and parietal codes that abstract over items (structural
representations: Whittington et al. 2020; Bernardi et al. 2020).

**Test.** A rule trained with one set of names, tested with names never seen in that
rule, at 90% or more.

**Status** ([39](experiments/39-schema-advantage.md)): a schema advantage. New members
of a known family are answered with no exposure (55–75% vs 10–28% without the schema), and
the knowledge is cortical (it survives a hippocampal lesion). The path there ran through
[30](experiments/30-cortex-driven-saccades.md)–[38](experiments/38-consolidation-of-one-shot-episodes.md).

**Status** ([30](experiments/30-cortex-driven-saccades.md)): the first transfer test.
- An unseen question wording gives 0% even with a perfect look-back: every kernel is
  keyed on word identities.
- Cortex-driven saccade contexts (the column's superposed continuations) help the policy
  only partly.
- This is the baseline the schema work has to move.

**Update** ([31](experiments/31-role-cells-and-transfer.md)):
- Role categories emerge from competitive Hebbian learning on the column's expectations.
- Learned generalisation (pruning inputs that did not matter) gives full transfer to new
  names when the rule is role-level, but breaks rules where identity matters.
- Next: keep general and specific kernels side by side, arbitrated by reliability.

**Update** ([32](experiments/32-general-and-specific.md)): general and specific kernels
now live side by side.
- New names transfer fully, and habit improves to 86–89%.
- The copies are too many and too similar; the name-dependent rule is mixed.
- Next: spawn after repeated evidence, then the one-exposure schema test.

**Update** ([33](experiments/33-generalisation-during-sleep.md)): general rules are now
formed offline from replay and tested on it. They give transfer to new names, no loss on
name-dependent rules, and better habit, at the default size and speed. Next: the
one-exposure schema test.

**Update** ([34](experiments/34-schema-test.md)): the schema test fails. New pairs are
not learned in 1–4 exposures.
- One-shot cortical kernels are keyed on incidental filler.
- Recall is not specific to the context, because the context state is a bag of filler.
- The missing piece is a clean context code: recency or the role of "the setting".

**Update** ([35](experiments/35-fading-state-and-entorhinal-codes.md)): a fading state
(temporal context, as in lateral EC) gives only small gains: recency is not relevance.
The code needed is structural, as in medial EC and the Tolman-Eichenbaum Machine: learned
slot codes bound to content in the hippocampus, recalled by slot.

**Update** ([36](experiments/36-slot-binding-memory.md)): built.
- A setting slot emerges.
- Slot ⊗ content episodes with rarity-weighted recall learn new name–place pairs in one
  exposure (28–76%).
- The answer does not use them yet in the schema group. Next: familiarity-gated
  arbitration between cortex and hippocampus.

### Relational memory (after [67](experiments/67-relation-store-in-reading.md))
- **Done:**
  - a learned, directed relation graph (`RelationStore`) in place of the semantic bag;
  - relations of relations at sleep;
  - the best consolidated figure on the season task: 64.4% for new names with the
    hippocampus lesioned.
  - See [relational memory](concepts/relational-memory.md).
- **Sparse communication is the rule that held:** answers sent only where the column is
  unsure, and only the filler. Graded or always-on alternatives (64) and a learned cue
  controller (65) were worse and are not defaults.
- **Next:**
  1. one-step rules (merge differently worded relations) and transitive relations,
     judged by contradiction;
  2. ~~let a frame word be an entity~~ (done, [68](experiments/68-lifted-frames-and-sparse-gating.md));
  3. a cue controller that reacts to the first recall and is rewarded at the answer.

### Output and self-supervision (after [71](experiments/71-speech-routing.md))
- **Done:**
  - an output buffer;
  - answering by speaking, with abstention ([69](experiments/69-answering-by-speaking.md));
  - the efference copy and recitation ([70](experiments/70-efference-copy-and-recitation.md));
  - speech routed as in the brain ([71](experiments/71-speech-routing.md)).
- **Next:**
  1. ~~a go/no-go that sees novelty and agreement between memory and cortex~~ (done,
     [72](experiments/72-go-no-go-signals.md));
  2. read-back as training for the area chain
     ([plan](concepts/output-and-self-supervision.md)).

### Epistemics (after [75](experiments/75-bayes-module.md))
- **Done:**
  - source memory in the hippocampus ([73](experiments/73-source-memory.md));
  - inferences held as proposals ([74](experiments/74-proposals-and-premises.md));
  - the Bayes module: source trust, conflicts resolved by it ([75](experiments/75-bayes-module.md)).
  - See [epistemics](concepts/epistemics.md).
  - belief isolated behind a swappable rule (full / vote / graded / posterior); proposals
    validated by belief × credibility ([76](experiments/76-belief-rules.md)).
  - inference from every statement of a fact ([77](experiments/77-infer-from-every-statement.md)).
  - a task where the believed fact decides the answer: family questions 98–100% with
    graded or posterior belief, 29–68% trusting everyone or counting votes
    ([78](experiments/78-belief-decides-the-answer.md)).
  - "unknown" when belief is split: the posterior abstains on every question about a name
    two equally trusted sources dispute, and never where trust settles the conflict
    ([79](experiments/79-unknown-when-belief-is-split.md)).
  - answer or "unknown" learned as a basal-ganglia go/no-go from practice quizzes, per
    belief × lead over the runner-up ([80](experiments/80-learned-answer-or-unknown.md)).
  - curiosity: open questions ranked by value of information, or picked by the network's
    own basal-ganglia policy rewarded by the information gained; asking a teacher settles
    the undecidable name ([81](experiments/81-curiosity.md)).
- **Next:**
  1. ~~independent random streams per subsystem~~ (done, see Infrastructure);
  2. ~~a task where the believed fact decides the answer~~ (done, [78](experiments/78-belief-decides-the-answer.md));
  3. ~~carry the believed family into place questions~~ (already carried by cooperation:
     lucy's place answers 1–15% under a wrong belief, 36–68% under a right one; the limit
     is holding the season, [78](experiments/78-belief-decides-the-answer.md));
  4. proposals and self-generated claims as sources with earned trust;
  5. ~~"unknown" when belief is split~~ (done: the posterior abstains exactly on the
     undecidable conflict, [79](experiments/79-unknown-when-belief-is-split.md));
  6. ~~learn the answer-or-abstain threshold as a go/no-go~~ (done: from practice, per
     belief × lead; graded and posterior answer tom and lucy and abstain on sam,
     [80](experiments/80-learned-answer-or-unknown.md));
  7. ~~seek evidence on an "unknown": a curiosity module~~ (built: asking a teacher,
     by a fixed ranking or the network's own learned policy; a question asked is settled,
     [81](experiments/81-curiosity.md)); with practice the learned policy settles the
     undecidable name on every seed, and a cost policy balances questions against energy
     by itself; trust is per topic. Next: saving energy for later, a go/no-go that
     generalises without leaking, replay and attention as searches;
  8. ~~frames that survive a lopsided filler~~ (done: a position where a known filler
     recurs is a slot, unit test; relation-store entries within margin).
- **The relation store's learned parts, and what is still an algorithm over word ids.**
  The binding map (entity, relation → value) is a predictive kernel class over rotated
  bit codes. Frame induction (neighbour counting), relation codes (an index into a list),
  binding offsets (a hash), rule learning (counting compositions) and the Bayes module
  (integer bookkeeping over claims) are discrete algorithms over word ids. To make them the
  network's own representations:
  1. ~~frames from code overlap~~ (tried: predictability alone cannot tell a relation word
     from a value, [82](experiments/82-fact-completion.md));
  2. ~~binding offsets from a hash~~ → **fact completion** (done, `REL_COMPLETE`: a general
     kernel class completes the fact; as good or better, 3–4× the cost, not yet default,
     [82](experiments/82-fact-completion.md)); next: key claims, rules and topics by the
     question with its blank, and drop the frames;
  3. claims and trust in bit-sliced counters (as the basal ganglia hold values), so belief
     is per-bit arithmetic over codes (approximate, where the integer posterior is exact).
- **The goal is continuous learning.** Experiments switch learning off at test so a run
  is measured cleanly, and that stays for now. The network itself is meant to learn all the
  time, with no train/test split: belief, trust, the go/no-go and the curiosity policy
  keep updating from whatever it reads, asks and is told.

## 5. A third level and a slower clock
**Tried ([83](experiments/83-compute-only-where-needed.md)):** the higher area only where the
column needs help (surprised, or unsure of the next word). It helps some tasks a lot (slot
memory 16 → 41%) and kills others (story boundary 0%): when top-down helps must be learned
per context (a go/no-go charged for compute), not set by a rule. A learned go/no-go,
rewarded by what blanking the frame changes in the next prediction, skips 70–90% of steps
and keeps the context tasks, but loses about 7–21 points on the relation and
hippocampus entries (and 47 on sleep generalisation): one step's counterfactual undervalues a frame whose worth arrives later.
Eligibility traces credited mostly by the answer, with a margin before skipping, beat
always consulting on 14 of 25 entries but save only ~10% and still collapse on single
seeds, because the gate also decides what the area and column learn. Next: learn on every
training step and gate only the use, and keep the area's context running while its
prediction is skipped.

**Build.**
- Run the higher area only on the column's surprises, or once per sentence. This removes
  its cost and gives each level a different timescale.
- Add a third level over story structure: goals, where events happen, who knows what.
  That is where "why" and "what next" questions become answerable.

**Brain.** Timescales lengthen up the cortical hierarchy (Hasson et al. 2008; Murray et
al. 2014).

**Test.** Cost on memory tasks back to within 1.5× of no hierarchy; a question needing
story-level structure answered above 80%.

**Status** ([25](experiments/25-area-chain.md)): a chain of three higher areas (windows
of 4, 16 and 64 sentences) is built. Each area extends the reach on the season task,
but only as a source in the mix, and accuracy stays low (9–38%).
- Needed: recency in the windows, and more teaching signal for the upper areas.
- The planned source of that signal is output with self-supervised read-back
  ([plan](concepts/output-and-self-supervision.md)). It comes before the event-driven
  clock.

**Update** ([26](experiments/26-context-and-readback.md)): with a context boundary at
each story, three higher areas reach 86–98% and hold a fact 16+ stories back. Context,
not depth, was the missing piece. Next: detect boundaries, a decaying state for recency,
and growth by need.

**Update** ([27](experiments/27-boundary-detection.md)): the network now finds story
boundaries itself, from contradicted facts (90–99%).

**Update** ([28](experiments/28-reading-with-actions.md)): the reader's own actions
(opening a book) mark and reinstate context, which lifts interleaved reading from chance
to 41–54%. Next: let the network choose the action (reach for the book a question
needs), and hold states per kind so a restored context is not crowded. 28 also audits
what is still hand-supplied.

**Update** ([29](experiments/29-saccades.md)): active reading changes the picture.
With learned saccades the page is external memory, and one higher area reaches 88–99.6%
where three areas and context boundaries were needed before. Depth is still needed for
what is not on the page (a gist, or a book read days ago), but looking back is the cheaper
first resort. Next: confidence-driven regressions, a spatial index of the page, and
skipping.

**Built ([84](experiments/84-areas-grow-by-need.md)):** from one area, story boundary grows exactly three (windows 4, 16, 64 sentences) and prunes the fourth, 85–96% against 8–17% at one; other tasks over-grow (2–4 areas, no gain), so promotion should also weigh compute or answers.

**Growing areas by need (the proposal).** Instead of fixing the number of areas, grow
one where the top area cannot contain its surprise:
- **A bud.** The top area keeps a candidate area above it, with a window 4× longer. The
  bud runs only on the top area's residual (event-driven, so it is cheap) and votes in the
  mix in shadow, without affecting answers.
- **Promotion.** If the bud's learned reliability on that residual rises clearly above
  chance (its `SourceMix` weight), the surprise was structure out of reach, not noise,
  and the bud becomes a full area with a bud of its own.
- **Pruning.** If it stays at chance, the surprise is irreducible (random names, say),
  and the bud is pruned. This is kernel growth and probation one level up.
- **Brain.** Adult cortex does not grow new areas. It recruits and repurposes existing
  cortex for new skills; reading, for example, recruits the visual word form area
  (Dehaene & Cohen 2007). So a bud is uncommitted cortex being recruited.
- **Test.** On the season task the chain should grow to about three higher areas, and
  on topic or give it should stay at one or none.

## 6. A ladder of harder tasks
Each new stage is measured on accuracy, calibration and abstention, plus cost against
the transformer baseline ([comparison](concepts/brain-transformer-comparison.md)):
- three- and four-step chains;
- transitive inference (a > b, b > c, so a > c);
- negation ("not in the kitchen");
- counting;
- contradiction between two stories;
- questions whose right answer is "unknown".

Then real text again, where the earlier stages (03–05) left off.

## Infrastructure: repeatable runs and a regression suite (done)
These are not research stages. They keep earlier results trustworthy as the code grows.
- **Exactly repeatable runs.** Rust's `HashMap` seeds its hasher randomly per process,
  and iteration order in kernel matching, sleep merging and pruning, and tallies moved
  results by about a point between runs at the same seed
  ([40](experiments/40-family-stated-once.md)). Every map and set in the library and the
  experiment harness now uses `neurocomp::det` (a fixed-key hasher, `src/det.rs`), and two
  runs at one seed print identical output.
  - The fixed order is a different sample than earlier runs: seed 0 of experiment 40
    moved from 35 to 42% held out. Figures in the experiment pages are from the old,
    random order, one draw among several. Figures recorded from now on are exact.
- **An experiment regression suite.** `cargo test` checks the building blocks (121 unit
  tests in `src/`). [`scripts/regress.sh`](../scripts/regress.sh) reruns the key
  experiments at their recorded settings (full length) on three seeds, each (entry, seed)
  a parallel job. It compares the mean answer accuracy on trained and held-out items with
  [`scripts/regress.tsv`](../scripts/regress.tsv) and fails on a drop of the mean of more
  than `MARGIN` (3) points, printing each seed's held-out figure beside it. One seed is
  too noisy a check (below). `QUICK=1` runs seed 0 only, reported without pass or fail,
  for a fast look while developing. Since runs are deterministic, it also reports any change within
  the margin. `--record` writes new figures after an intended change.
  - **Entries:** 24, from story boundary (26) to motor speech (71); the table lists them
    with their pages and settings.
  - **Run it before every commit** that touches `src/` or the harness.
  - Re-recorded after intended changes: the frame fix (a recurring filler is a slot) moved
    the relation-store entries within the margin; trust per topic then moved
    belief-decides (78) past it on seen answers (68.1 → 63.7, held-out 80.7 → 81.7), back
    near its first recording (61.9). Add an entry with
    each new experiment.
- **Independent random streams per subsystem** (after [77](experiments/77-infer-from-every-statement.md)).
  Stories, the basal ganglia, sleep (replay order, consolidation, sleep learning), the
  relation store and altered feedback each draw from their own stream split off the seed;
  `rng` is left to setup and the cortex's waking learning. Switching a subsystem on, or
  changing what it does, no longer changes the stories or the rest of the run.
  - Every regression figure moved (a different draw) and the table was re-recorded.
  - Seed-0 figures are single draws from a wide spread. Checked on three seeds before and
    after the change: full hippocampus 60 / 50 / 36% before, 33 / 45 / 47% after; slot
    memory 38 / 15 / 22% before, 19 / 14 / 15% after; story boundary 98 / 87 / 97% after
    (page: 86 / 96 / 98%). The old seed 0 was often the lucky one, so figures in earlier
    pages are best read as ranges.
- **Speed: every system caches its response** (after [77](experiments/77-infer-from-every-statement.md)).
  Everything runs on the CPU, one thread per run; the suite runs its (entry, seed) jobs in
  parallel. Stack samples of the full model (motor speech, [71](experiments/71-speech-routing.md),
  seed 0) showed where a run's time goes: hippocampal recall about 58% of samples, the
  higher area's prediction about 20%, the column's look-ups (`peek`) about 15%.
  - **Kernel classes cache their response.** A class keeps the matched kernels for the
    last 8 inputs it was asked about, valid until its wiring changes (growth, pruning,
    sleep). The winner and the outputs are still read live from the matched kernels, so
    a change in a kernel's record needs no invalidation. Output identical; about 1%
    faster, since most asks within a step are for different inputs.
  - **Recall reads hot fields only.** A row's serial, strength, last touch, source and
    place key sit in their own small array, so scoring a posting no longer loads the row,
    and liveness ((now − touched) / τ < strength) is checked without a division.
  - **The walk reuses its scoring.** Recall by walk scored the cue, and, finding no
    bridge, scored the same cue again for a plain recall; it now passes its result on.
    Postings walked per recall: 26 273 → 13 454.
  - Together: the run went from 217 to 141 s (training 566 → 369 µs per word, test
    7.35 → 4.77 ms per word). The three-seed suite: every entry the same.
  - **Recall stops walking when the answer is settled (MaxScore).** The leader's and
    runner-up's scores only rise, and the runner-up's final value does not depend on
    the order rows are scored in. So once even the best an untouched row could still
    reach (the remaining cue weight plus the place bonus) is below the runner-up, no
    untouched row can change the answer. Recall then leaves the posting lists and scores
    only the rows still in reach, by looking each remaining id up in the row, dropping
    rows as they fall out of reach. Exact (when every touched row is eligible,
    `min_overlap` ≤ 1). Postings walked per recall: 13 454 → 1 006.
  - **Kernel classes keep running match counts.** A class keeps its last input's match
    counts and, while its wiring is unchanged, updates them for the bits that changed
    instead of fanning every active bit out again (a change bigger than the input
    recounts from zero). The higher area's input (`[lead | slow state | sentence bag]`)
    changes by a few bits a word. Look-ups go through the same counts, so the column's
    own step on an input it was just asked about counts nothing new. Exact.
    - The per-frame memo of [23](experiments/23-compaction.md), tried on the higher area
      instead: exact but slower (144 vs 120 s), backed out.
  - **Runs in parallel.** The harness runs its (policy, seed) runs on `THREADS` threads
    (default: the machine's cores), each run's printing buffered and printed in run
    order. Each run depends only on its seed, so the output is identical for any thread
    count (checked: 1 vs 3 threads, three seeds). The suite already runs its (entry,
    seed) jobs in parallel. A single run stays on one thread: each word's step depends
    on the last one's learning, and splitting a step's few hundred microseconds across
    threads costs more than it saves.
  - Profiled run, all of it: 217 → 84 s (training 566 → 237 µs per word, test 7.35 →
    2.77 ms per word), output identical.
  - **The three-seed suite on the final build: every entry the same** (all 24, every
    seed's figure equal to the recorded one). The suite (72 runs on 4 cores) took 912 s,
    against 1 789 s after the first round of changes (response cache, hot fields, the
    walk's reuse).
  - A GPU is a poor fit: the work is sparse and event-driven (index fan-out, a few
    matched kernels, early exits), not dense matrix products.
- **Integer only** ([47](experiments/47-integer-only.md)). Every per-step computation is
  bitwise or integer (`Q16` fixed point, `src/fixed.rs`). A unit test fails on any float
  in `src/` outside test code and marked configuration or report lines.
- **Networks of kernels and a grammar** ([51](experiments/51-networks-of-kernels.md), in
  progress). `src/program/modules.rs` is a `Module` interface with base-kernel leaves.
  Networks are modules, so they nest, and a stack grammar (`Genome`, `NetOp`) builds them.
  The column and a hierarchy of columns are expressed so far.
  - **Done ([52](experiments/52-associate-and-hippocampus-genome.md)):** a Hebbian
    `Associate` leaf, and the hippocampus as a genome at bit parity.
  - **Next:** replay, tags and the sparse binding space in the genome; switch the
    harness to it. (1/n, centering and a phase-bound binding space were tried in the
    circuit and backed out or left off: [54](experiments/54-phase-hippocampus-backed-out.md).)
  - **Cost first:** the self-contained hippocampus trains at 2.3 ms/word against the
    list memory's 0.23. Recall only on surprise, and reuse CA3's settled state within a
    sentence, before adding more to it.
  - **Then:** scalar signals as population codes, and the episodic harness migrated one
    component at a time.
  - **The goal:** the whole architecture is one genome that can be mutated and searched.
