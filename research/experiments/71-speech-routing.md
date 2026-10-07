# 71 · Speech routed as in the brain: motor area, basal-ganglia go/no-go, forward-model efference copy

**Question.** Speaking was a shortcut: the harness decoded the source mix's word, wrote it
to the output buffer, and set a flag for the efference copy
([69](69-answering-by-speaking.md), [70](70-efference-copy-and-recitation.md)). In the
brain, a word is planned in association/premotor cortex, released by the basal ganglia
through the thalamus, executed by motor cortex, and a copy of the command predicts the
sound that comes back through the ears. Routed that way, does speaking still work, and
does the basal ganglia's go/no-go learn when to stay silent?

**Code.**
- [`src/program/motor.rs`](../../src/program/motor.rs):
  - `VocalTract` (the world): each word has a motor code of its own, unrelated to the code
    it is heard as; driving the tract with a command says the word it matches, or nothing.
  - `MotorArea`: an inverse model (sound → command: how to say what is meant) and a
    forward model (command → expected sound), both predictive kernel classes, learned by
    babbling (random commands, each heard).
  - Unit tests: after babbling, ≥ 95% of words are said right and their sound predicted;
    a garbled command says nothing.
- [`examples/episodic.rs`](../../examples/episodic.rs): `SPEECH=motor` (with `SPEAK` and/or
  `RECITE`).
  - **Babble** before reading (`SPEECH_BABBLE` rounds, default 10).
  - **Plan:** the cortex's evidence for the word (the source mix's choice, a sensory code)
    goes to the motor area, which plans a command; the tract says it.
  - **Select:** at a question the basal ganglia choose speak or stay silent, per the mix's
    confidence band. They learn in training, where the page's answer follows: speaking
    right +1, speaking wrong −1 (`SPEAK_PENALTY`), silence 0.
  - **Efference copy:** the forward model's prediction of the sound, compared with the word
    heard (no flag).
  - **Reafference:** what the tract said is the next input, through the ordinary input path.
  - Recitation goes through the same route.

## Results (family stated once; seeds 0 / 1 / 2; cooperation + replay + relation store)

Babbling: 630 commands, 63 inverse and 63 forward kernels; all 63 words said right.

| Setting | Hippocampus | Answered (held out) | Right of answered | Right of all |
|---|---|---|---|---|
| shortcut, always speak (69) | lesioned | 100% | 64.6 / 65.0 / 64.8% | same |
| shortcut, threshold 0.9 (69) | lesioned | 65 / 64 / 58% | 84 / 87 / 93% | – |
| **motor route, basal ganglia decide** | lesioned | 88 / 100 / 95% | 69.6 / 69.0 / 60.8% | 61.4 / 69.0 / 58.0% |
| motor route, basal ganglia decide | intact | 100% | 85.4 / 90.2 / 88.4% | same |

What the basal ganglia learned (lesioned, at test, by the mix's confidence band):

| Band | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| 0 (< .5) | silent (75) | spoke, 0% right (14) | silent (24) |
| 1 (.5–.7) | silent (12) | spoke, 17% (18) | spoke, 13% (39) |
| 2 (.7–.8) | spoke, 27% (144) | – | spoke, 15% (174) |
| 3 (.8–.9) | spoke, 29% (163) | spoke, 24% (283) | spoke, 36% (174) |
| 4 (≥ .9) | spoke, 85% (606) | spoke, 90% (685) | spoke, 89% (589) |

Recitation through the motor route (hippocampus plans): 88.9 / 95.0 / 100% of words
right in place, content words 78.7 / 89.9 / 99.8% (shortcut: 96.1 / 97.5 / 93.0%).

Efference copy from the forward model: none of the network's own spoken words were
surprising when heard as predicted.

## Findings
1. **Speech works through a motor route learned by babbling.** The inverse model says
   every word; the forward model's prediction serves as the efference copy; reafference
   comes back through the ordinary input. Recitation keeps 89–100% of words in place.
2. **The basal ganglia learn to stay silent only where training showed failure.** They
   are silent in the lowest bands on two seeds, but keep speaking in bands 2–3, where
   new names are right only 15–36% of the time.
   - The values are learned on training questions, about trained names, where the same
     bands are right far more often. Confidence means something different for a new name.
   - So the learned gate answers more (88–100%) and is right less (61–70%) than the fixed
     threshold of 0.9 (58–65% answered, 84–93% right).
3. **With the hippocampus intact, every answer is spoken** (bands 3–4 only), 85–90% right,
   as in 69: the mix is sure whenever memory agrees.
4. **The route is now the brain's in shape:** association cortex plans, basal ganglia
   release, motor area executes, forward model predicts, sensory path hears. What is still
   simplified:
   - the plan is the mix's word, not a learned premotor sequence;
   - the go/no-go is per confidence band only;
   - the motor area is not a column with layers; the forward model's prediction is
     compared at the word, not as a frame into L4.

## Next
- **Give the go/no-go the signals that separate new from known:** the sentence's
  familiarity (novelty), and whether memory and cortex agree. A new name with a confident
  mix should still be suspect.
- **Learn in training from questions about new names** (the facts stated once arrive in
  training; questions about them could too), so the bands are calibrated where it matters.
- **Read-back as training:** regenerate each sentence and learn from the mismatch.
