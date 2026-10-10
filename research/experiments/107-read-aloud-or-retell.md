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
- ~~**The practice correction is never heard**~~ (replaced: a heard correction).
- ~~**The hippocampus stores only while listening**~~ (replaced: it stores everything).
- **The eye moves one word per step,** in lockstep with the teacher. It should be driven by
  saccades from the learned control loop.
- **The driver starts each story's context and counts the time code.** These should come from
  learned event boundaries, with time cells that reset at a boundary.
- ~~**The driver ends an invented story**~~ (replaced: a learned stop, by saying silence).
- **The prefrontal gate keeps its values per word number,** where it should be keyed by the
  heard pattern.

## Replacing three hand-set triggers, and free invention
Every run here: learned gate, time code, `RT_TELL=apart`, 3,000 trials, and "now make one" in a
quarter of the trials (`RT_MAKE=1`).

**The three replacements:**
1. **A heard correction** (`RT_CORRECT=heard`). In practice, a wrong word is followed by the
   teacher saying the right one aloud. The network hears its own word, then the correction, and
   learns only from what it hears; the gate is credited only when no correction came.
2. **The hippocampus stores everything it hears** (`RT_HC_STORE=all`): the instruction, the
   task, its own speech and the corrections. Its own novelty gating sets the strength.
3. **A learned stop** (`RT_STOP=learned`). Silence is a sound the vocal tract makes (closing
   the mouth), learned by babbling like the words. The teacher's silence after the last word is
   heard, so the network learns to predict the end of an utterance. An invented story ends when
   the network says silence.

**Free invention** (`RT_MAKE=1`): after "now make one" the teacher makes up a new story. At test
the network speaks freely, with noise on the relay's weights (30%), and each story is judged:
- **grammar:** every sentence is one of the seven forms;
- **story shape:** it starts as a story begins and ends "then N went home .";
- **coherence:** one name, its pronoun, one animal;
- **novelty:** never heard in training.

**Storing everything broke the hippocampus at first.** It triples the stores, and the
hippocampus scaled each input's drive by its *lifetime* write count. By 3,000 trials the context
bits had lost their drive: recall while telling was 0% and every invented story was silence.

The fix (`HippocampusConfig::decay_writes`): the write counts halve every half-life of stores, as
the weights do, so familiarity is recent use. It is off by default in the library and on in this
driver.

| Asked | Book | A: heard correction | C: + store all, learned stop | D: as C, no practice |
|---|---|---|---|---|
| read | same story | 72.5 | 64.9 | **75.6** |
| read | another story | **65.0** | 50.6 | 56.8 |
| tell | another story (story's / book's word) | 44.6 / 60.7 | 45.4 / 47.6 | **52.7** / 55.3 |
| tell | closed | 42.8 | 44.6 | **52.0** |
| | recall right at test | 96% | 95% | 97% |
| | tell closed, prefrontal content removed | 35.9 | 18.6 | 21.9 |
| invention | ended by itself | 93% | 28% | 92% |
| | sentences grammatical | 17% | 12% | 19% |
| | well formed, coherent, new | 1.4% | 0% | 0% |

C and D use the decaying write counts. B and the first C and D, without them, are not shown:
the hippocampus failed as described.

**Invented stories** (D, the first six):
> … the owl . he found a hat . fox . went home .
> … went home .
> tom went home .
> … went to the shop . he saw a fish . he saw a fish . he saw a owl … carrot a fish . he saw a fish . he found a fish . he saw a fish .

From A (practice with the heard correction):
> bob went to the park . … saw a fox . she saw a fox . she then bob went home .

**Findings:**
- **The heard correction removes practice's harm.** Retelling with the book closed is 42.8% with
  it, against 16.2% with the unheard oracle. Practice still does not beat no practice (D: 52.0).
- **Storing everything works once familiarity decays.** With the learned stop it gives the best
  retelling yet: D, 52.0% with the book closed and 52.7% story words against 55.3% book words
  with another book open. The instruction carries it: without the prefrontal content, retelling
  falls to 22%.
- **The learned stop works.** In A and D, 92–93% of invented stories end on their own, by the
  network saying silence. In C, 28%.
- **Invention is still poor.** Most stories are the shortest form ("tom went home .") or loop on
  one sentence ("he saw a fish ."). Only 12–19% of sentences are grammatical, and no story was
  well formed, coherent and new. A blank ("…") often replaces the name: the plan for the first
  word is weak.

## The hippocampus as an index, with learned event boundaries
The circuit above stored a copy of each word's sound and gave it back, and the driver started
each story's context and counted positions. It now defaults to an index (`RT_HC=index`;
`RT_HC=circuit` restores the circuit), built on `IndexMemory` (hippocampal indexing theory,
Teyler & DiScenna; the bit form of a modern Hopfield network). The index never holds content.

**Rows:**
- **One row per event, grown as needed.** Rows are forgotten when unused. Separable capacity
  grows exponentially with the code's width, because each row competes as a whole.
- **A row points to a cortical assembly:** the ear's layer 4 cells that fired when the sound was
  heard. Recall reinstates them, layer 2/3 completes the assembly, and the cortex says the word.
- **A row's keys are the context in force.**

**Context (the temporal context model, Howard & Kahana 2002):**
- **Driven by layer 4.** Each sound replaces 4 of the context's 32 cells, each set by one layer 4
  cell: the 4 with the smallest hash (min-hash). A layer 4 response that drifted by a cell or two
  then moves the context almost the same way.
- **Learned event boundaries.** A boundary cell (`BoundaryCell`) learns where the cortex's
  surprise rises; at a boundary, half the context is replaced. It fires about 9 times a trial,
  mostly near sentence ends.
- **Pauses are heard as silence,** a natural boundary.

**Order and recall:**
- **Each row links to the next.**
- **Sequence bias (CA3):** recall expects the successor of the row just recalled, unless another
  row matches the context better by 12 of 32 cells.
- **Retrieval mode (Hasselmo):** while a reinstated episode keeps predicting what is heard,
  nothing new is stored, so the network's own retelling does not overwrite the memory it reads.
  A mismatch, which is novelty, returns it to encoding.

**Prefrontal gates.** Two learned gates, keyed by the last two sounds:
- one holds the context where an episode begins (at a boundary);
- one reinstates the held context (on any sound).

Their dopamine is the hippocampus's own comparator: did the row it recalled predict the next
sound?

**Keeping up with drift** (the cortex's layer 4 keeps learning):
- **Reconsolidation:** when the cortex completes a recalled pointer to an assembly that keeps at
  least half the pointed cells, the row is relearned to the assembly as it is now.
- **Sleep replay:** every 10 trials, 3 chains of up to 30 rows are reinstated; the cortex
  completes each, and the row is re-pointed (Káli & Dayan 2004). The cortex does not learn from
  replay yet.
- **Layer 4 settles** (`Layer4::set_settling`): a cell's chance to move a synapse halves with
  each doubling of its wins past 64.

**Bugs found on the way:**
- the context was first driven by a hash of layer 4's whole response, so one changed cell gave
  a different context (fixed with the min-hash);
- reconsolidation consumed the pointer recall needed to follow the links: only 122 of 6,727
  recalls knew where they were;
- the network's own wrong words, stored during a retelling, became the best match (fixed with
  retrieval mode);
- exploratory reinstatements left the hippocampus in retrieval mode, so new stories were not
  stored (fixed by ending retrieval mode on a mismatch).

**Results** (3,000 trials, with invention; learned gates against the start held and reinstated
by the driver, `RT_IX_HOLD=oracle`, a diagnosis):

| | Learned gates | Oracle hold | Circuit (D above) |
|---|---|---|---|
| recall right at test | 11% | **89%** (99–100% at 400 trials) | 97% |
| tell, book closed | 0.1 | 36.4 | **52.0** |
| tell, another book (story's / book's) | 28.9 / 43.5 | 40.8 / 73.9 | 52.7 / 55.3 |
| read, same story | 56.3 | 58.2 | 75.6 |
| invention: ended by itself | 14% | 22% | 92% |

**Findings:**
- **The index works when the start is found:** 89–100% recall from pointers to assemblies, with
  graded context, links, retrieval mode and reconsolidation keeping up with layer 4's drift.
- **The cortex uses it less well than the circuit's copy of the content:** 36% against 52% with
  the book closed. Another open book now wins more often (74% of words from the book).
- **The learned gates do not find when to hold and reinstate.** The hippocampal match also
  rewards predicting the predictable: the hold gate learned to hold the start of "now tell it"
  (value 0.38) and of common phrases ("the fox", 0.61), not the start of the story.
- **Invention lost its learned stop:** stories end by themselves only 14–22% of the time, against
  92% with the circuit.

## Indexing a higher area: where eye and ear converge
In the brain the hippocampus talks to association cortex, not to primary sensory layers. Here
the index pointed to the ear's layer 4: the lowest level, tied to one sense and the quickest to
drift.

**The association area** (`Net::assoc_of`) is a competitive layer (k-winners-take-all, learned)
over the eye's and the ear's layer 4. When reading aloud, the word seen and the word heard arrive
together, and its cells grow synapses on both, so either sense alone comes to evoke the same
assembly. The index now points to these assemblies, and the context is driven by them. Layer 4
is no longer indexed.

**A rule needed for convergence** (`Layer4::set_pathways`): a winner moves a synapse away only if
its own input pathway was active. Without it, listening (the ear alone) moved every cell's eye
synapses to the ear, and the pairings learned while reading were unlearned: an absent sense was
taken as evidence against it. With it, after 400 trials a word seen and heard share 10.9 of 32
cells (3.3 without).

**Results** (3,000 trials, with invention):

| | Learned gates | Oracle hold | Index on layer 4, oracle (above) |
|---|---|---|---|
| a word seen and heard share (of 32 cells) | 12.8 | 13.2 | – |
| words sharing at least half | 34% | 41% | – |
| recall right at test | 12% | **86%** | 89% |
| tell, book closed | 1.2 (silent 77%) | 15.3 (silent 68%) | 36.4 |
| tell, another book (story's / book's) | 31.3 / 62.7 | 34.8 / 57.3 | 40.8 / 73.9 |
| read, another book | **80.4** | 67.8 | 56.3 |

**Findings:**
- **The higher area can be indexed:** recall from its assemblies is as good as from layer 4 (86%
  against 89%).
- **It is only partly cross-modal yet.** A word seen and heard share about 13 of 32 cells.
- **Reading improved,** to 80% on an unheard book with the learned gates.
- **Retelling with the book closed fell** (36 → 15% with the oracle), mostly to silence. The
  network says "silence", the learned end of an utterance, on two thirds of those words, though it
  plans the right word 66% of the time while the teacher speaks.
- **The learned gates still do not find the story's start** (recall 12%).

## Why retelling fell silent, and the fix
Retelling with the book closed was two-thirds silence. The traces showed the network starting
with "… home ." and then "silence", though recall was right for the first words: the cortex was
not using recall. Most of the time recall was noise. While a new story is heard, the
hippocampus has nothing ahead of it to recall, so on nearly half the training steps its output
was a wrong guess, and the cortex learned to ignore it.

**Fixes:**
1. **Recall reaches the cortex only in retrieval mode.** In encoding mode its output is
   suppressed (Hasselmo), so what arrives is a memory being retrieved, never a guess
   (`RT_HC_OUT=always` restores the old behaviour).
2. **The network's own speech is not novelty** (corollary discharge damps the response to
   self-made sounds), so a slip of its own does not end retrieval; only what others say can.
3. **A step where nothing was said is heard as nothing,** not as the unsaid plan
   (`RT_SILENT=efference` restores the old behaviour).

The first fix carries it (3,000 trials, start held by the driver):

| | Before | After |
|---|---|---|
| tell, book closed | 15.3 (silent 68%) | **73.9** (silent 4%) |
| tell, another book (story's / book's) | 34.8 / 57.3 | **54.1** / 51.2 |
| read, another book | 67.8 | 69.1 |

> story: bob went to the shop . he saw a owl . the owl was big . bob gave the owl a cake . then bob went home .
> said:  bob went to the school . he saw a fox . the school . she . bob went to the a shell . then bob went home .

This is the best retelling so far: 52% with the circuit that held a copy of each word. With
another book open the remembered story now wins.

**The learned gates still fail:** retelling 1%, silent 90%. Their values barely differ between
word pairs (0.1–0.25), and they reinstate at random, about 13 times a trial, so retrieval mode
is rarely entered at the right moment.

**Invented stories no longer end by themselves** (0%). Recall no longer reaches the cortex
during invention, and the story runs on to the 40-word limit.

## Reaching for the book
A book used to be handed over open. Now, by default, it stays shut until the network reaches for
it (`RT_REACH=learned`):
- **The reach is a motor act,** a basal-ganglia choice between waiting and reaching, keyed by the
  last two sounds heard.
- **The world answers it like a retrieved file:** the book, from its first word, now on the eye.
- **The reach is an event boundary** for the hippocampus. The context moves on by half, and the
  hold gate may hold the new book's start.
- **Nothing says when to reach:** "read it" and "tell it" are just sounds, and a book open at the
  wrong time intrudes on a retelling.

**The credit had to be an action's own value.** The prefrontal gate's rule credits only the last
load against an average over all trials, and it learned the reverse: always reach after "tell
it", never after "read it". Retelling trials score well whatever the reach does, so whatever was
chosen there gained. With each choice's value in its context moving toward the outcomes it
brought (an eligibility trace over the trial's decisions, the trial's share of words right as
the reward, no global baseline), it learned the task.

**Results** (3,000 trials, with invention):

| | Learned gates | Start held by the driver (oracle) |
|---|---|---|
| reached after "read it" | **100%, at once** | 100%, at once |
| reached after "tell it" | **10–16%** | 100%, after a word |
| value of reaching after "read it" / "tell it" | 0.81 / 0.56 | 0.81 / 0.74 |
| read, same story / another book | 66.0 / 66.7 | 71.4 / 70.3 |
| tell, book closed | 0.5 (silent 92%) | **61.8** |
| tell, another book (story's / book's) | 1.2 / 0.3 | 47.1 / 22.1 |

**Findings:**
- **The reach is learned from the outcome:** it reaches to read and mostly waits to retell. The
  same credit, each action's own value from what it brought, is what the hippocampal gates lack.
- **With retrieval triggered (oracle), memory now beats an open book:** 47% of words from the
  story against 22% from the book.
- **The learned hippocampal gates still never enter retrieval** (recall 4%), so learned retelling
  is silent.
- **Invented stories run to the 40-word limit** (ended by themselves 0–4%).

## A striatum that learns when to give credit
Until here every gate had a hand-written credit rule: what counts as reward, when it comes, and
which choice gets it. Each gate needed its own fix, and the hippocampal hold, whose payoff comes
30 words later, never learned. All four gates now share one striatum
([`src/program/striatum.rs`](../../src/program/striatum.rs)), an actor-critic that learns credit
timing by temporal-difference learning (Schultz, Dayan & Montague 1997).

**The striatum:**
- **State is a pattern, not a word number:** what the prefrontal cortex holds, the association
  area's assembly for the last sound, and mode signals (book in view, retrieving, something
  held). Each bit has its own synapses, so similar states share what they learned.
- **One critic, one dopamine signal:** δ = reward + γ·V(next state) − V(state), with γ = 0.97,
  applied through eligibility traces (λ = 0.8).
- **The only reward is a word that comes out right.**
- **The gates are actor channels:**
  - reach for the book;
  - hold where an episode begins;
  - reinstate what is held;
  - load what was heard into working memory.

  A unit test carries a payoff back three steps to the choice that earned it.

**The teacher now waits** (up to three pauses) before starting, and the network may reach during
the wait. Before this, exploration tried reaching at random points mid-task, where a book opened
at its first word disagrees with a teacher already at word 18, so reaching looked worse than
never reaching.

**Two bugs on the way:**
- **Learning too slow:** eligibility was divided by the number of active bits, while values are
  means, so learning was about 70 times too slow.
- **Reinstating at every sound:** reinstating while already retrieving only starts the sequence
  over. It is now allowed only when not retrieving.

**Results** (3,000 trials, with invention):

| | All learned | Start held by the driver (oracle) |
|---|---|---|
| reach after "read it" / "tell it" | 100% / 100% | 100% / 100% |
| read, same story / another book | **73.7 / 74.6** | 76.0 / 73.7 |
| read, prefrontal content removed | 13.7 | 65.9 |
| tell, book closed | 0.0 (silent 100%) | 65.0 |
| tell, another book (story's / book's) | 35.4 / 63.8 | 53.2 / 56.5 |

With reinstating only when not retrieving (all learned): reading 64.2 / 62.0, retelling still
silent; reinstatements fell from 125,207 to 64,614 over the run, holds rose from 1,356 to 2,782.

**Findings:**
- **Reading is the best yet, and depends on the instruction:** 74% of words right with every
  gate learned, and 14% without the prefrontal content.
- **The reach is learned from the outcome,** but now it reaches under "tell" too. Opening another
  book costs a retelling little when the cortex can lean on memory.
- **The hippocampal gates are not learned yet.** The striatum reinstates far too often, and a
  mismatch while listening ends retrieval, so it reinstates again. Recall stays near 1%.

## A cost for reinstating, and a shelf of books
**The cost emerges; it is not written in.** Reinstating already has a real price: the
hippocampus stops taking in what is new, and the old story's recall reaches the cortex while a
new one is heard. That price shows only if correct predictions count all the time. So the
reward is now every heard word the cortex predicted, while listening, during the instruction
and in the task (`RT_REWARD=all`). Dopamine neurons signal sensory prediction errors too
(Takahashi et al. 2017; Gardner et al. 2018).

Result (3,000 trials): reinstatements fell from 64,614 to 10,598, holds rose from 2,782 to
9,407, and reading rose to 77–79%. Learned retelling is still silent: it now holds at many
boundaries (overwriting the story's start), and reinstates at the wrong moments.

**The shelf** (`RT_SHELF=4`, default):
- **Books at four places,** now and then one replaced by a new book. Each is about a different
  animal.
- **The find task:** in a quarter of the trials the teacher says "now find the <animal>", waits
  three pauses, then reads that book. The network may reach to a place (a striatum channel over
  the places), and the book there is what its eye sees. A wrong place means the eye and the
  teacher disagree, and the words come out wrong.
- **The index binds books to places.** While a book from a place is in view, rows are keyed by
  the context, the place cells, and the content: the association area's assembly for what is
  *seen*, the words at that place. (Keying by what was heard bound the requested story to
  whatever place was reached.)
- **Only the entorhinal cortex cues the hippocampus.** `IndexHc::ec` is the context, the place
  and the content now. A first version also cued it from working memory directly, and that was
  removed. "Where was this seen?" is the place component of the completion: every matching
  episode votes for its place with its match score, and episodes with no place vote nothing.
  Taking only the best match failed, because that was the instruction just heard, which has no
  place.
- **The recalled place joins what the striatum sees,** and TD credits a reach by the reading
  that follows.

| (3,000 trials) | |
|---|---|
| hippocampus recalls the right place for the animal just heard | 36.9% (chance 25%) |
| reaches the right place at test | **35.4%** (chance 25%) |
| words read right in find trials | 41.3% |
| working memory holds the animal at the wait | 3.6% |
| a word seen and heard share (of 32 cells) | 11.3 |

**Findings:**
- **Above chance, but not far.** Where-recall is limited by the association area's cross-modal
  convergence: the heard "fox" must match the seen "fox", and they share only 11 of 32 cells.
- **The striatum uses the recalled place about as well as the hippocampus supplies it**
  (35% reach right, 37% recall right).
- **Working memory almost never holds the animal.** Finding works without it, because the cue
  is what is active in the cortex just after the word is heard.
- **Reading fell to 58%** with a quarter of the trials given to finding.

## Closer to the brain: outcome reward, acetylcholine, and a reuniens-like request
**Where credit comes from in the brain:**
- **Most cortical prediction learning is local.** A cell whose feedforward and apical inputs
  coincide fires a burst, and bursts drive LTP, lone spikes LTD (burst-dependent plasticity;
  Larkum; Payeur et al. 2021; in CA1, plateau-driven BTSP, Bittner et al. 2017). The cortical
  cells here already learn this way.
- **Dopamine is a three-factor signal mainly for choices,** at corticostriatal synapses.

**How the prefrontal cortex reaches the hippocampus:**
- through the entorhinal cortex, and through the nucleus reuniens of the thalamus, which
  coordinates retrieval (Vertes; Dolleman-Van der Weel et al. 2019);
- hardly at all directly (a sparse anterior cingulate → CA3/CA1 path, Rajasethupathy et al.
  2015).

The hippocampus projects straight back to the medial prefrontal cortex.

**Changes:**
- **The striatum's reward is the outcome only:** words said right in the task.
  `RT_REWARD=all` restores rewarding every predicted word.
- **Acetylcholine for novelty:**
  - the CA1 comparator's mismatch with what others say raises it (CA1 → septum; Hasselmo &
    Schnell; Meeter et al.); it decays by a quarter each step;
  - the prefrontal decision to reinstate becomes a standing request (the reuniens route),
    withdrawn when a new episode's start is held;
  - the hippocampus retrieves while the request stands and acetylcholine is low, and encodes
    otherwise;
  - a mismatch now pauses retrieval instead of cancelling it.
- **No discounting within a trial** (a finite episode). Mode flags and the recalled place weigh
  24 bits each, and the sound before the last joins what the striatum sees.

**Results** (3,000 trials, with invention and the shelf):

| | All learned | Start held by the driver (oracle) |
|---|---|---|
| tell, book closed | 0.0 (silent) | **72.7** |
| tell, same story | 0.0 | **83.4** |
| tell, another book (story's / book's) | 0.0 | **73.2 / 31.2** |
| read | 0.0 (no reach) | 0.0 (no reach) |

**Findings:**
- **Retrieval with acetylcholine is the best retelling yet.** With the start held, the network
  retells 73% with the book closed and 83% with the same story open. With another story open,
  the remembered story now wins clearly (73% against 31%). A mismatch pauses retrieval instead
  of ending it.
- **With only outcomes rewarded, the striatum stopped reaching for the book,** though reaching
  helps reading (60.5% of words against 54.0% without). The reason: at the wait it cannot tell
  "read it" from "tell it". It sees the pause and "it"; the verb is three sounds back, and the
  prefrontal cortex holds it only about 5% of the time. Under "tell", an opened book intrudes,
  so on average waiting wins. Holding the verb would pay only once reaching is valued, and
  reaching is valued only when the verb is held.
- **Next:** give the striatum the cortex's ongoing context, as corticostriatal input carries
  (layer 2/3 and layer 5 activity), with each input group weighed by its own mean so a large
  group does not drown the rest. The instruction would then be visible at the wait without
  first having to be held in working memory.

## The striatum sees the cortex's ongoing activity
**Changes:**
- **Corticostriatal input:** the striatum sees layer 2/3's and layer 5's fired cells, besides
  the prefrontal content, the last two sounds, the mode signals and the recalled place.
- **Each input group counts by its own mean** (`Striatum::begin_groups`). A state's value and an
  action's preference are sums of group means, so the large cortical groups do not drown the
  small signals.
- **The cortex now learns during the instruction:** the next sound, then the teacher's pause.
  Before this it never learned there, so layer 2/3 was silent at the wait (0 bits).

**What was found** (1,500 trials, each change tested in turn):
- **A switched-on signal looked like a loss.** Giving an "off" mode signal cells of its own let
  those cells carry the baseline value, so switching "book in view" on read as a drop. Only
  signals that are on now have cells.
- **The dopamine right after a reach stays negative,** −2.4 to −6.3 under "read", though reaching
  helps reading (68.2% of words against 55.8%). It stays negative with a longer eligibility trace
  (λ = 0.95 and 1.0) too.
- **The cortex's activity does not hold the instruction.** At the wait, layer 2/3 fires 15 bits.
  Their overlap between two "read" trials is 0.97, and between a "read" and a "tell" trial 0.95:
  its activity is the last word, "it". So the striatum cannot tell the two apart, and on balance
  waiting wins.

**Where this leaves it:** the instruction can reach the striatum at the wait only through
something that holds it: the prefrontal cortex. But the striatum's learned gate holds the verb
only about 5% of the time, and holding it would pay only once reaching does. A plausible way
out is that the prefrontal cortex updates on salient, surprising input, gated by phasic dopamine
(Braver & Cohen 2000). In "now read it" the verb is the unpredictable word, "now" and "it" are
not, so a surprise-gated update would hold the verb with no gate to learn first.

## A bitvec striatum
**Why it was replaced.** The actor-critic's learning rule was generic, but most of what fed it
was designed for this task:
- what it saw: hand-picked input groups and three invented mode flags;
- the actions;
- when it decided;
- the reward and where an episode ends;
- integer weights averaged per group.

Each fix to it (the flag encoding, the bits per flag, the group means) made it more so.

**The striatum, as it is wired:**
- **It is the basal ganglia's input nucleus.** Layer 5 of nearly all cortex and the thalamus
  project to it. Its spiny cells (about 95%) are bistable and fire only on many coincident
  inputs.
- **Two pathways per action:** D1 cells, the direct "go" pathway, and D2 cells, the indirect
  "no-go" pathway. They converge on GPi/SNr, which keep the thalamus inhibited until an action is
  released.
- **Striosomes project to the dopamine neurons** (the critic); the matrix is the actor.
- **Learning is three-factor:** a synaptic tag (0.3–2 s, Yagishita et al. 2014) that dopamine
  converts. A burst gives D1 synapses LTP and D2 synapses LTD; a dip the reverse.

**The bitvec version** ([`src/program/striatum.rs`](../../src/program/striatum.rs)):
- **Spiny cells are popcount units:** 12 synapses on input bits, firing at 8 active (the up
  state). A cell is recruited for a state by sampling one active input pathway (topographic
  input).
- **Per action, a D1 and a D2 population.** The drive is D1 firing minus D2 firing, plus a little
  noise; the strongest wins, and the default wins ties.
- **Striosome cells are the critic:** value is the number firing, each worth half a word.
  Dopamine is reward + V(next) − V(state), in integers.
- **Learning is structural.** Recent states and choices stay eligible for 8 steps, credit
  halving per step back. A burst recruits striosome and go cells and removes no-go cells that
  fired; a dip the reverse. The number changed follows |δ|.
- **Tests:** a delayed payoff, and opposite actions learned in two contexts.

**Input is only real activity:**
- working memory;
- the last two sounds;
- layer 2/3 and layer 5;
- the eye's layer 4 ("a book in view" is visual input);
- the hippocampus's output ("retrieving" is recall arriving);
- the held context and the recalled place cells.

The hand-made flags are gone.

**Results** (3,000 trials, everything learned):

| | |
|---|---|
| read, same story / another book | 69.5 / 65.2 |
| reaches after "read it" / "tell it" | 100% / 100% |
| tell, book closed | 3.8 (silent 95%) |
| tell, another book (story's / book's) | 34.3 / 58.8 |
| working memory holds the animal (find) | 72.4% |
| find: right place | 22.2% (chance 25%) |
| striatum cells | 4,071 striosome, 23,102 go, 15,422 no-go |

**Findings:**
- **Reaching is learned again.** The dopamine right after a reach is positive (+1.28 under
  "read"), where the weighted critic gave −2.4 to −6.3.
- **Working memory now holds what was asked for:** the animal, 72% of the time, against about 5%
  before.
- **Too much reaching:** it reaches under "tell" as well, so another book intrudes on a retelling.
- **Retelling and finding are still not learned:** the hippocampal gates hold and reinstate at
  the wrong moments (recall 0%), and the recalled place is at chance.
- **The striatum churns:** about 1.8 million cells recruited and as many removed. Populations sit
  at their cap, and the run is about twice as slow.

## The rest of the loop as bit cells
**Before this,** only the striatum was cells. Choosing an action was a function (the strongest
D1 − D2, plus noise), and dopamine was a formula.

**Now every stage is a population** ([`src/program/striatum.rs`](../../src/program/striatum.rs)).
Each population has 32 cells with their own thresholds, so its response is a count of cells
firing.
- **Fast-spiking interneurons:** feedforward inhibition. They scale a channel's D1 and D2 counts
  so the total is at most 32.
- **GPe** (per action): tonically active (24 of 32), inhibited by the action's D2 cells.
- **STN** (per channel): a tonic baseline, inhibited by GPe. The cortex excites it through the
  hyperdirect path when more than one option is driven at once, which is conflict.
- **GPi/SNr** (per action):
  - tonically active and noisy (±4 cells, the exploration);
  - excited by STN;
  - inhibited by the action's D1 cells and by GPe.

  An action is released when its GPi falls at least 3 cells below tonic. The most released
  action wins. Conflict raises every GPi, so the loop waits.
- **Dopamine neurons:** 32 cells, 8 firing at baseline.
  - Reward and the next state's value excite them; the current state's striosome cells inhibit
    them.
  - Each cell above or below baseline is an eighth of a reward. A burst can rise far; a dip can
    only fall to silence.
- **Cholinergic interneurons (TANs):** they pause at phasic dopamine. Eligibility tags become
  lasting changes only during a pause.
- **The tag fades** to four-fifths per step back, and each change happens with probability ¼.

**The first run failed, and the failure was informative:**
- "Wait" was an action with its own go cells.
- Small value errors kept rewarding whatever was chosen, which was usually waiting.
- Once normalisation drove wait's GPi to zero, noise could never release the reach. The network
  never reached, and reading fell to 26%.
- **The fix:** in the brain, waiting is not an action but the absence of a release. Tonic GPi
  keeps the thalamus shut, there is no efference copy, and nothing is tagged. Now action 0 is
  that default: it has no cells and is never learned. Only a released action carries
  eligibility.

**Results** (3,000 trials, everything learned; before → now):

| | bitvec striatum | full loop |
|---|---|---|
| read, same story / another book | 69.5 / 65.2 | 61.5 / 66.3 |
| reaches after "read it" / "tell it" | 100% / 100% | 100% / 100% |
| tell, book closed | 3.8 (silent 95%) | 0.0 (silent 97%) |
| tell, another book (story's / book's) | 34.3 / 58.8 | 31.0 / 58.7 |
| working memory holds the animal (find) | 72.4% | 61.7% |
| find: right place | 22.2% | 10.5% (19 trials; chance 25%) |
| striatal cells (striosome / go / no-go) | 4,071 / 23,102 / 15,422 | 4,096 / 19,465 / 1,168 |
| cells recruited | about 1.8 million | 102,607 |

**Findings:**
- **Behaviour is about the same, and the churn is gone.** About 18 times fewer cells were changed.
  TANs paused at 46% of dopamine events. Changes come at the tag's rate, and the default is no
  longer learned, so most no-go cells were never needed.
- **The STN brake does not stop reaching under "tell".** Only one option (reach) is ever
  driven, so there is no conflict for the hyperdirect path to see. The reach still pays: the
  dopamine right after a reach is +0.06 under "read" and +0.02 under "tell". So the striatum has
  no reason to hold back. What would stop it is a cost the outcome reflects: under "tell", an
  open book should make the retelling worse, and the reward should show it.
- **The probe "preference after 'read it'" reads 0.** It builds a state from the instruction
  alone, which fires no cells. The real states at the wait carry layer 2/3 and 5 activity.
- **Retelling and finding are unchanged.** They are limited by the hippocampal gates (recall
  about 0%), not by the striatum.

## Next
- **Learn when to hold and reinstate.** The gates need credit for what holding makes possible
  later (retelling), not for predicting the predictable.
- **Let the cortex learn from replay,** and find why retelling falls silent with the index.
- **The hippocampal gates under TD:** reinstating should cost something (it stops the hippocampus
  taking in what is new), and the hold needs the boundary that begins a story to be distinct in
  the state.
- **Better cross-modal convergence** in the association area, the limit on finding a book by name.
- **Credit for the gate per instruction.** Compare the reward with what the same trial type
  usually earns, not the overall average, so holding "tell" is credited.
- **Seeds 1–4** for every column.
