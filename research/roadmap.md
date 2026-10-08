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
     [81](experiments/81-curiosity.md)); next: more experience for the learned policy, a
     go/no-go that generalises to unpractised states, replay and attention as searches;
  8. frames that survive a lopsided filler (found in 80).

## 5. A third level and a slower clock
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

**Growing areas by need (proposed).** Instead of fixing the number of areas, grow
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
  - **Run it before every commit** that touches `src/` or the harness. Add an entry with
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
