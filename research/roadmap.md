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
