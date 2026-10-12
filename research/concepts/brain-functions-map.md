# The brain's main functions, and which our model has

A study, written 2026-10-12, after the World/Mind split ([108](../experiments/108-world-and-mind.md)).
For each of the brain's main functions it records:
- the substrate in the brain;
- what the integrated mind (`src/program/mind.rs`, run by `examples/lessons.rs` and
  `examples/conditioning.rs`) has of it;
- what exists elsewhere in the repository but is not wired into that mind (earlier experiments'
  harnesses);
- the evidence that it works.

Status:
- **Present:** in the mind, and shown to work by a probe or a task.
- **Built:** in the mind, but not shown to work, or shown not to.
- **Elsewhere:** built and tested in an earlier harness, not part of the mind.
- **Partial:** a simplified stand-in.
- **Absent.**

## 1. Perception

| Function | Brain | Our model | Status |
|---|---|---|---|
| Sensory feature learning (sparse codes for what is sensed) | Primary and secondary sensory cortex, layer 4 | Each sensing area's layer 4 `Pool` recruits cells for new inputs; the senses arrive as sparse random codes from the world, not learned from raw signals | Present (codes given, recoding learned) |
| Prediction of the next input | Cortical layers 2/3 (predictive coding) | Every column's layer 2/3 (`BitCells`) predicts its next input; surprise is the share not predicted | Present |
| Multisensory association | Association cortex | Pairwise association columns grow from co-activity (`pair_after`), a full area over them | Present ([96](../experiments/96-association-area.md), [107](../experiments/107-read-aloud-or-retell.md)) |
| Hierarchy of areas, feedback down it | Felleman & Van Essen; corticocortical feedback | Two levels (pairs, full); feedback from association columns to the senses' tufts; recall cascades down by layer 4 back-projection | Partial (two levels) |
| Invariance, object constancy | Ventral stream (IT) | None beyond what sparse overlap gives | Absent |
| Motion, change detection | MT/MST, superior colliculus | The seen-acts area takes the eye's onsets (the motion stream) | Partial |
| Raw signal processing (pixels, sound spectra) | Retina, cochlea, thalamus | The world hands over symbolic codes | Absent (by design) |

## 2. Attention

| Function | Brain | Our model | Status |
|---|---|---|---|
| Bottom-up salience | Superior colliculus, pulvinar | Novel features attended; the own-effect (agency) signal | Partial |
| Attention to predictive cues (associability) | Central amygdala, basal forebrain (Mackintosh, Pearce–Hall) | The amygdala's attention bit vector: set by surprise, value and novelty, fading 1/32 per step; only attended cues reach the striatum | Built (attention collapsed to 4–5 features per step in one probe) |
| Top-down attention, selection among inputs | Prefrontal and parietal cortex, thalamic reticular nucleus | The thalamic relay weighs sources by learned reliability in context; the basal ganglia gate which columns' goals are taken in | Partial |
| Attention as encoding versus retrieval mode | Acetylcholine (Hasselmo) | Acetylcholine rises with input from outside and falls when attention turns inward; it sets the hippocampus's mode | Present |

## 3. Memory

| Function | Brain | Our model | Status |
|---|---|---|---|
| Working memory, gated | Prefrontal cortex with basal ganglia gating (PBWM) | One slot (`WorkingMemory`), loaded by each association column's loop gate; on the tufts as context | Built (one slot) |
| Episodic memory: one-shot storage, completion | Hippocampus (index theory, CA3) | Index (`IndexHc`) over the association columns' codes: stores events from outside, completes, follows sequences, jumps on content, reinstates context; acetylcholine modes; reconsolidation | Present (recall right 9–17% in the read/tell tasks; [107](../experiments/107-read-aloud-or-retell.md)) |
| Temporal context, event boundaries | Entorhinal cortex (temporal context model), boundary cells | Drifting context with min-hash updates; a learned boundary cell from surprise | Present |
| Consolidation into cortex during sleep | Hippocampal replay, slow-wave sleep | The mind's `sleep()` replays chains and relearns rows to what the cortex completes now; the cortex is not taught by replay. Elsewhere: replay taught the cortex facts the hippocampus had lost (100%, [17](../experiments/17-consolidation.md); [36](../experiments/36-slot-binding-memory.md)–[62](../experiments/62-replay-as-reading.md); sleep-gated plasticity [94](../experiments/94-sleep-gated-consolidation.md)) | Partial in the mind, Elsewhere in full |
| Semantic memory, facts and relations | Anterior temporal lobe, distributed cortex | Elsewhere: relation store (typed facts, rules composed at sleep, [66](../experiments/66-typed-relations.md)–[67](../experiments/67-relation-store-in-reading.md)); variable binding ([09](../experiments/09-thalamic-attention.md)–[16](../experiments/16-thalamic-gate-memory-channel.md)) | Elsewhere |
| Procedural memory, habits | Dorsolateral striatum, cerebellum | Sticky habit bits in the striatum's masks; motor areas' inverse and forward models | Partial |
| Priming, familiarity | Perirhinal cortex | Familiarity counts in the index (novelty bonus in storage strength) | Partial |

## 4. Learning systems

| Function | Brain | Our model | Status |
|---|---|---|---|
| Self-supervised prediction | Cortex | Columns learn the next input; layer 5 learns goals toward what the effectors' senses received | Present |
| Error-corrective learning (supervised, timing) | Cerebellum (climbing fibres) | `CerebellarCircuit` predicts from the ear and the pons; its output is a source in the mix and on the tufts | Built (its contribution untested in the mind) |
| Reinforcement learning, TD | Basal ganglia, dopamine (Schultz) | `MaskStriatum`: mask-cell critic and actors, TD error with γ = 7/8, eligibility over 8 steps; the critic on the limbic territory | Present (Pavlovian transfer: dopamine at CS+ +0.49 against CS− +0.01, [108](../experiments/108-world-and-mind.md)) |
| Fast cue → outcome learning | Basolateral amygdala | Bit-cell amygdala: appetitive and aversive value cells on an age-tagged onset trace, Rescorla–Wagner shared error | Present (CS+ +0.95 against CS− +0.00) |
| Act + context → outcome learning | Orbitofrontal and prelimbic cortex, dorsomedial striatum | The same bit-cell learner over cues and the efference copy (`ofc`); values to the critic and the motor loops | Built (not yet tested; the instrumental probe has failed in every earlier version) |
| Agency, which acts have effects | Short-latency dopamine to sensory events (Redgrave & Gurney) | The forward model's error on its own act trains the motor loops' last choices, not the critic | Built |
| Plasticity gating by cue | Cholinergic interneurons' pause | Pause cells learned on cues before large errors open actor plasticity | Present (pauses on 4–40% of steps, learned) |
| Metaplasticity, homeostasis | Synaptic scaling, intrinsic excitability | Self-calibrating thresholds and rates in `BitCells` (`set_meta`), layer 4 settling | Partial |

## 5. Action

| Function | Brain | Our model | Status |
|---|---|---|---|
| Action selection | Basal ganglia loops (Alexander, DeLong & Strick) | A loop per column, created as columns grow; per motor area a loop and a motor gate; each loop sees its own territory | Present (the loops act; the instrumental probe fails) |
| Release versus withhold (go/no-go, inner versus outer) | Direct and indirect pathways, subthalamic nucleus | The motor gate releases or withholds; withheld acts are imagined. The STN hold is off in the mind | Partial |
| Motor primitives, pattern generators | Brainstem and spinal CPGs | The genome's primitives (syllables, mouse moves), one per effector spontaneously active each step | Present (babbling holds up in play) |
| Inverse and forward models | Motor and premotor cortex, cerebellum | Each motor area's agranular column: layer 5 inverse, layer 2/3 forward; learned from every act | Present |
| Efference copy, corollary discharge | Motor cortex collaterals; auditory suppression of own speech | One's own sound marked by the voice's efference copy (not stored, no acetylcholine); the efference copy to the motor loops and the orbitofrontal learner | Present |
| Imitation, mirror mapping | Premotor and parietal mirror areas; the dorsal speech stream | Seen acts evoke the felt act through the association columns; heard words evoke the command that says them (the imitation route) | Present ([107](../experiments/107-read-aloud-or-retell.md): 63 of 63 words after speech play) |
| Motor timing, sequencing | Cerebellum, supplementary motor area | Only the relay's and the columns' step-to-step predictions | Partial |
| Vigor, effort | Dopamine tone, striatum | Effort cost removed; Pavlovian–instrumental transfer lowers the motor brake with cue value | Built |

## 6. Valuation, emotion, drives

| Function | Brain | Our model | Status |
|---|---|---|---|
| Primary rewards and drives | Hypothalamus, brainstem (hunger, thirst, temperature, pain) | None; the only innate values are social (a smile, a contingent reply) and its own effect | Absent |
| Social reward | Ventral striatum, oxytocin systems | An innately recognised smile; a reply, paid by a contingency detector (Watson) | Present |
| Fear and defence, aversive learning | Amygdala, periaqueductal grey | Aversive value cells exist; nothing in the world is aversive | Built (unused) |
| Interoception, bodily state | Insula, hypothalamus | None | Absent |
| Emotion as global state (mood, stress) | Amygdala, hypothalamic–pituitary–adrenal axis | None | Absent |
| Curiosity, learning progress | Locus coeruleus, dopamine, anterior cingulate | Learning progress of the columns' goal cells loosens the striatum's choices (`extra_noise`) | Partial |

## 7. Executive function and decision

| Function | Brain | Our model | Status |
|---|---|---|---|
| Goal-directed versus habitual control | Dorsomedial versus dorsolateral striatum; prefrontal cortex | One kind of loop; habits only as sticky bits; act outcomes from the orbitofrontal learner | Partial |
| Model-based planning, simulation | Prefrontal cortex, hippocampal sweeps (Johnson & Redish) | Withheld acts are imagined through the forward model; the motor gate sees the amygdala's value of the imagined outcome. No search over sequences | Partial |
| Inhibition, stopping | Right inferior frontal gyrus, subthalamic nucleus | The release brake, lowered or raised by cue value; no stop signal | Partial |
| Conflict and error monitoring | Anterior cingulate | None (elsewhere: confidence and "unknown" answers, [80](../experiments/80-learned-answer-or-unknown.md)) | Absent in the mind, Elsewhere |
| Task sets, rules, instructions held | Prefrontal cortex, PBWM | The instruction loaded into working memory, if a loop gate learns to | Built (one slot) |
| Metacognition, confidence | Prefrontal cortex | Elsewhere: questions as inner speech, holding an open question ([88](../experiments/88-inner-speech.md)–[91](../experiments/91-learned-working-memory-hold.md)) | Elsewhere |

## 8. Neuromodulation

| Modulator | Brain role | Our model | Status |
|---|---|---|---|
| Dopamine | Reward prediction error, vigor, salience | One TD error to all loops; a separate agency signal to the motor loops | Present (simplified: one scalar) |
| Acetylcholine | Encoding versus retrieval; attention; striatal pauses | Hippocampal mode; the cholinergic pause | Present |
| Noradrenaline | Arousal, unexpected uncertainty, explore versus exploit (locus coeruleus) | Curiosity noise stands in for the locus coeruleus's tonic mode | Partial |
| Serotonin | Patience, temporal discounting, aversive prediction | None (a fixed γ) | Absent |
| Arousal and sleep–wake (orexin, brainstem) | States of the whole brain | `sleep()` called by the driver every 10 lessons | Partial (scheduled from outside) |

## 9. Space and time

| Function | Brain | Our model | Status |
|---|---|---|---|
| Place, grid and head-direction cells; path integration | Hippocampus, entorhinal cortex | None in the mind (elsewhere: place cells in the index, [95](../experiments/95-circuit-place-and-dopamine.md)) | Absent in the mind, Elsewhere |
| Interval timing | Striatum, cerebellum | The age-tagged onset trace (microstimuli, 4 steps) | Partial |
| Circadian rhythm | Suprachiasmatic nucleus | None | Absent |

## 10. Social cognition and language

| Function | Brain | Our model | Status |
|---|---|---|---|
| Detecting others, faces, voices | Fusiform face area, superior temporal sulcus | A presence sense (someone there, attending, smiling) | Partial |
| Others' goals and beliefs (theory of mind) | Temporoparietal junction, medial prefrontal cortex | None | Absent |
| Joint attention | Superior temporal sulcus, frontal eye fields | None (the teacher's cursor is seen, not followed as a pointer) | Absent |
| Speech perception and production | Superior temporal gyrus, Broca's area, the dorsal and ventral streams | Ear and voice areas, the ear–voice association (dorsal stream), babbling with a caregiver | Present (babbling; imitation) |
| Word learning, reading, retelling | Language network | The lessons: listening, the instruction, reading and telling | Built (silent or at chance so far) |
| Syntax, composition | Left inferior frontal and temporal cortex | Only sequence prediction and recall; elsewhere: syntax by prediction ([05](../experiments/05-syntax.md)), roles ([36](../experiments/36-slot-binding-memory.md)) | Partial, Elsewhere |

## 11. Imagination and the default mode

| Function | Brain | Our model | Status |
|---|---|---|---|
| Mental imagery | Feedback to sensory cortex (Kosslyn, Pearson) | Imagined results arrive top-down at the next step, never through layer 4 | Present |
| Recall as re-experience | Hippocampal reinstatement down the hierarchy | Recall in the association columns, cascading to the senses by back-projection | Built (new; untested) |
| Inner speech | Inner speech network | A word withheld from the voice runs inwardly through the forward model | Present |
| Prospection, the default mode network | Medial prefrontal cortex, posterior cingulate, hippocampus | None as a mode; the mind always runs one pipeline | Absent |

## 12. Development and maintenance

| Function | Brain | Our model | Status |
|---|---|---|---|
| Growth of areas and connections by experience | Development, critical periods | Association columns grow by co-activity; cells recruited on demand; loops grow with columns | Present |
| Pruning, forgetting | Synaptic pruning, decay | Stalest cells replaced at capacity; index rows forgotten unused | Partial |
| Critical periods | Plasticity windows | Layer 4 settling with experience | Partial |
| Sleep stages (slow-wave replay, REM) | Thalamocortical oscillations | One replay routine | Partial |

## What this says

**The mind is strongest where the project began:**
- prediction;
- one-shot memory and its modes;
- the selection of actions by learned, local, bitwise rules;
- the newest valuation parts: fast cue value, the critic's transfer to the cue, learned pauses.

**Built but not integrated.** Several functions shown earlier live in other harnesses:
- consolidation into cortex;
- semantic relations;
- metacognition and "unknown" answers;
- place.

The integrated mind uses the hippocampus only as an index with replay that relearns its own rows.

**The gaps that matter most for the current tasks**, in the order they block the lessons:
1. **Acting on a cue** (goal-directed control). The instrumental probe has failed in every version.
   The orbitofrontal split and the efference copy are the current attempt; goal-directed versus
   habitual loops may be needed.
2. **Working memory beyond one slot** that a loop gate fills. The instruction ("read" or "tell")
   must be held across a wait and a task; the probes have not tested holding at all. The delayed
   probe is the closest, and it fails.
3. **Joint attention and social learning beyond imitation.** The teacher demonstrates on their own
   cursor; following another's attention and acts as pointers is what children use to learn what
   words and acts refer to.
4. **Drives.** With only social and agency values, everything the mind does is for a smile or for
   an effect. A drive (some state that builds and is relieved) would give it a reason to persist
   with an act.
5. **Consolidation into cortex,** so what the hippocampus holds becomes the cortex's own (shown
   elsewhere, [17](../experiments/17-consolidation.md), [94](../experiments/94-sleep-gated-consolidation.md)).

**Absent, and not needed for the present tasks:** primary drives, interoception, mood, circadian
rhythm, space and navigation, theory of mind, invariant object recognition, raw sensory
processing.
