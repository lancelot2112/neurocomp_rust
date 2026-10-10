# 107 · Read aloud or retell: two tasks learned from a teacher, chosen by a spoken instruction

**Question.** Can the plausible loop learn two tasks, reading a book aloud and retelling a story
from memory, from a teacher's examples alone, with a spoken instruction deciding which? Nothing
may say which source to use, and nothing may bypass the network: every word said is planned by
the cortex and spoken through the learned motor path.

**Driver:** [`examples/readtell.rs`](../../examples/readtell.rs). Seed 0, single runs.

## The task
Each trial has three parts:
1. **Listening.** The teacher tells a short story S (3–6 sentences, 59 words in all: e.g. "sue
   went to the shop . she saw a fox . the fox was hungry . then sue went home .").
2. **The instruction.** A book is opened at the same story, at another story P, or stays
   closed. The teacher says "now read it" or "now tell it".
3. **The task.** The teacher does it, word by word:
   - *read* says the book's words;
   - *tell* says the heard story, whatever the book shows.

The network hears and sees what a child would, and at each word learns to plan the word the
teacher says. When the book shows another story, the eye and memory propose different words,
and only the instruction says which one is wanted.

**At test** (300 new stories, cortex learning off) the network does the task itself: it plans,
speaks, and hears its own words. The test asks whether each word said was the story's or the
book's. The same test is repeated with the prefrontal content removed before the task.

## The network
- **Eye and ear are separate inputs** with their own codes, and each has its own layer 4.
  Reading must be learned as a mapping from the visual code to the spoken one; there is no
  route from the eye to speech that skips the cortex.
- **The hippocampus** stores the story as it is heard: each word as content, with the moment's
  context. Recall returns to the cortex as an input (entorhinal cortex → cortex).
- **The prefrontal cortex** holds whatever a basal-ganglia gate chooses to load from what is
  heard (`PfcGate`, after PBWM). The gate's credit is whether the words spoken while it held
  that item were right. What it holds reaches the cells' tuft (layer 1) and the thalamic
  relay's context.
- **The rest is the prose loop:**
  - recurrent layer 2/3: basal input is eye, ear, recall and its own past activity; the tuft
    gets the prefrontal content and the cerebellum;
  - layer 5;
  - the cerebellar circuit;
  - the thalamic relay;
  - speech through the motor area's inverse model (learned by babbling) and the vocal tract.

## Two problems found and fixed
1. **Cells were one-off conjunctions.** A committing cell took its 16 synapses at random from
   every active input at once (eye, ear, recall, recurrent activity), so it fired only when that
   exact mix came back. On a 500-trial run, the prefrontal content made no difference.

   `BitCells::set_clustered`: a committing cell takes its synapses from one input frame, as
   synapses from one pathway cluster on a dendritic branch (Kleindienst et al. 2011, Takahashi
   et al. 2012). Cells then read one pathway each. With this, removing the prefrontal content
   halved accuracy (500 trials).

   The option is on in this driver and off by default elsewhere; the suite is unchanged.

2. **The hippocampus forgot how to recall.** The hippocampus scales each input's drive down by
   how often that input has been written. Word bits and a reused context were written thousands
   of times, and their drive rounded to zero. Recall of the next word (while telling) fell from
   54% after 100 trials to 8% after 3,000. Four remedies left it at 10–19% after 1,500 trials:
   faster decay, CA3 centering, scaling on every pathway, and both.

   **The fix: a sparse conjunctive context.** The context is 64 of 65,536 bits, set through a
   fixed random projection by the story and the last two sounds heard, as entorhinal conjunctive
   cells would give. Before anyone speaks, the sound heard is "silence". A conjunction is rarely
   repeated, so it keeps its drive. With the teacher's words heard, recall is right on 95% of
   telling steps (87% at test, where the network's own words are the cue).

## Results (3,000 training trials)
Each cell is the share of words said that were right (said the story's / the book's word).

| Asked | Book | Learned gate | Gate fixed to the instruction | Learned gate, tell apart | Fixed gate, tell apart |
|---|---|---|---|---|---|
| read | same story | 77.7 | 91.3 | 76.1 | **97.5** |
| read | another story (never heard) | 69.8 | 85.3 | 69.4 | **88.8** |
| tell | same story | 80.8 | 88.7 | 71.5 | 67.6 |
| tell | another story | 42.6 (book 61.7) | 45.8 (book 62.6) | 40.7 (book 56.5) | 43.6 (book 47.4) |
| tell | closed | 31.7 | 10.3 | **33.1** | 13.1 |

- *Gate fixed to the instruction* is a diagnosis: the instruction word is always loaded, with
  no gate.
- *Tell apart* (`RT_TELL=apart`): in training the teacher retells only with the book closed or
  at another story; the test keeps all three.

**With the prefrontal content removed** before the task, reading falls by 13–35 points with the
fixed gate (97.5 → 62.5, 88.8 → 53.6) and by 9–13 with the learned gate. Retelling with the book
closed falls from 31.7 to 11.4 and from 33.1 to 9.1 with the learned gate.

**The learned gate:**
- holds "read" (value of loading 0.78–1.00 where it learned) but never "tell" (0.06–0.15);
- in the run with full recall it held nothing in particular (read 0.15, tell 0.12; "it" 0.41).

**While the teacher speaks** (the last 500 training trials), the plan is right on 73–83% of words
when telling with another book open, and 75–79% with the book closed. At test, where the network
hears its own words, these fall to 41–44% and 10–33%.

## Findings
1. **Reading aloud is learned through the cortex,** from the teacher's examples: 88.8% on books
   never heard and 97.5% on the heard story (fixed gate, tell apart). It depends on the
   instruction held in the prefrontal cortex: without it, reading falls to 54–63%.
2. **Retelling works only partly.** With the book closed, a third of the words are right with the
   learned gate. With another book open, the network still says the book's word more often than
   the story's (47–62% against 41–46%). The training change (tell apart) cut the book intrusions
   (62.6 → 47.4 with the fixed gate) but did not raise the story's words.
3. **The gap is speaking, not knowing.** While the teacher speaks, the plan is right on 73–83% of
   retelling words. When the network hears its own words, one mistake changes the hippocampal cue
   and the next recall, and the errors compound. Training only ever heard the teacher's correct
   words, never its own mistakes.
4. **The learned gate learns to hold "read" but not "tell".** Retelling pays less, so holding
   "tell" is never credited above the baseline. Holding nothing, the network falls back on its
   default, which is to read whatever the eye sees.
5. **The hippocampus needs conjunctive context codes** to keep recalling over thousands of
   episodes. A reused context and single words lose their drive under its presynaptic scaling.

## Practice in its own voice, and a time code
Learned gate, `RT_TELL=apart`, 3,000 trials. Each cell is the share of words right; for "tell"
with another story open, the story's word / the book's word.

- **Practice** (`RT_PRACTICE=0.5`): in half the training tasks the network speaks and hears its
  own words, with the teacher's word as the target. As built, that target is never heard: the
  correction is an invisible oracle (see below).
- **Time code** (`RT_HC_TIME=1`): 32 more bits in the hippocampal cue, from the story and the
  position in the telling, as time cells give. A cue whose last words were wrong still matches
  in time.

| Asked | Book | Neither | Practice | Time code | Both |
|---|---|---|---|---|---|
| read | same story | 76.1 | 77.7 | 80.2 | 58.3 |
| read | another story | 69.4 | 75.1 | 60.3 | 53.1 |
| tell | same story | 71.5 | 69.7 | 80.3 | 57.2 |
| tell | another story | 40.7 / 56.5 | 37.8 / 61.4 | **51.1** / 55.0 | 36.0 / 46.2 |
| tell | closed | 33.1 | 16.2 | **48.9** | 44.0 |
| | recall right at test | 86% | 65% | **97%** | 91% |
| | gate's value of loading "tell" | 0.06 | 0.08 | **0.32** | 0.25 |

- **The time code is what helps retelling:** with the book closed, 33 → 49% of words right. With
  another book open, the story's word now comes about as often as the book's (51 / 55). Recall at
  test reaches 97%, and the gate starts to value holding "tell" (0.32). Reading another book
  falls (69 → 60): the remembered story now competes with the eye.
- **Practice hurts.** Retelling with the book closed falls to 16%. Its own wrong words gave the
  hippocampus wrong cues during training (recall right 68% instead of 95%), and the cortex learned
  from those. With the time code, practice cost 4–22 points everywhere.

**Not plausible as built.** Several triggers here are set by the driver, not learned or
heard:
- **The practice correction is never heard:** an invisible oracle. The plausible version: the
  teacher says the correct word aloud, and the network learns from hearing it.
- **The hippocampus stores only while listening.** It should store everything, gated by
  novelty.
- **The eye moves one word per step,** in lockstep with the teacher. It should be driven by
  saccades from the learned control loop.
- **The driver starts each story's context and counts the time code.** These should come from
  learned event boundaries, with time cells that reset at a boundary.
- **The driver ends an invented story** at "home ." or 40 words. The network should stop by a
  learned go/no-go on speaking.
- **The prefrontal gate keeps its values per word number,** where it should be keyed by the
  heard pattern.

## Next
- **Replace the hand-set triggers above,** starting with a heard correction in practice.
- **Free invention** (`RT_MAKE=1`, "now make one"): built and smoke-tested; full run pending.
- **Credit for the gate per instruction.** Compare the reward with what the same trial type
  usually earns, not the overall average, so holding "tell" is credited.
- **Seeds 1–4** for every column.
