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

## Holding and reinstating: a test ladder, bookmarks, and actions that grow
**The question.** The learned gates hold and reinstate at the wrong moments: about 11 holds and
11 reinstatements per trial, so recall stays at 0%. Is the striatum missing input, or credit?

### A test ladder
[`examples/hold_ladder.rs`](../../examples/hold_ladder.rs) is the read/tell task without the
cortex:
- a 16-word story with event boundaries at words 0, 4, 8 and 12;
- the instruction ("now", the verb, "it");
- three waiting steps, then six task words.

Hold and reinstate are striatal channels, decided as in readtell. Each rung adds one thing.
Results are means over 2 seeds, 3,000 episodes, then 400 test episodes:

| rung | what it adds | holds / episode | start held | tell words right |
|---|---|---|---|---|
| end | one reward at the episode's end | 0.00 | 0% | 0% |
| word | a reward per right task word | 0.00 | 0% | 0% |
| retag | the hold's tag re-armed when its content is used (PBWM) | 0.01 | 0% | 0% |
| theta | storing goes on while retrieving | 0.01 | 0% | 0% |
| restart | reinstating may start the sequence over | 0.02 | 0.2% | 0% |
| **hold given** | the hold made by the driver; reinstating learned | 1.00 | 100% | **100%** |
| reinstate given | reinstating made by the driver under "tell"; holding learned | 0.00 | 0% | 0% |
| scene / time | the teacher's voice and a page as input; time cells | 0–1.2 | 0% | 0–3% |

**Findings:**
- **Reinstating is learnable from per-word dopamine.** With the hold given, it happens after the
  instruction 99% of the time and almost never mid-story, and every retold word is right. Its
  payoff comes 1–4 steps later.
- **Holding is not learnable in any form tried,** even with reinstating given. Its payoff comes
  about 25 steps later, and only if no later boundary overwrites it.
  - The credit for it is about (⅘)^25 ≈ 0.004.
  - Exploring finds it about ⅕ × (⅘)^4 ≈ 8% of the time.
  - Noise credit at "now", the same sound every episode and just before the verb that predicts
    reward, wins instead.
- **What the striatum lacked, found on the way** (now in
  [`src/program/striatum.rs`](../../src/program/striatum.rs)):
  - **Discounting (⅞) and a TD(0) critic.** A steady reward flow gave a burst on every step
    (δ = 1 + V − V), which credited whatever gate had fired.
  - **Dips read by D2 affinity.** A dip used to stop at silence (−1) while bursts reach +3, so
    releasing was favoured on average. Dips now reach as far as bursts, read by the
    high-affinity D2 receptors.
  - **A pause threshold for the cholinergic interneurons:** a quarter of a reward. Small tonic
    drift (−V/8 on every step of a story) no longer punishes every action.
  - **Pruning of unused critic cells.** The critic sat at its cap, full of cells for one-off
    words. Pruning the actor pools broke learning, so it stays off there.
  - **Stable phase context.** With only content as input, the critic cannot value "reading,
    step k". It needs the voice, a page in view and elapsed time, and cells that combine two
    pathways.
  - **Conflict means close competitors.** STN now counts options driven within 2 cells of the
    strongest, not every option driven. With many candidates the old brake released nothing.

### A regression found: the closed book
In the closed-book condition a reach still "succeeded": the world handed over an empty book.
The reach was an event boundary that moved the context by half. The oracle reinstated the
story's start *before* the wait, so the reach erased it. Every oracle run since reaching was
learned (100%) was broken this way.

**The fix corrects the world:**
- a closed book cannot be grasped;
- the oracle reinstates as the task begins.

| oracle (hold and reinstate given), 1,500 trials | tell, book closed | recall right |
|---|---|---|
| storing during retrieval (theta) | 18.8 / 27.0% | 32% |
| retrieval stops storing | **61.0 / 54.5%** | **57%** |

Storing during retrieval files the teacher's retelling under the reinstated context, and those
rows then win recall. It is now off by default (`RT_THETA=on` restores it).

### Bookmarks instead of a hold
The hold is better as the hippocampus's job: people anchor memories at event onsets, and recall
of a narrative starts at boundaries. Two versions were built.
- **The strongest anchor** (`RT_ANCHOR=strongest`): each boundary keeps its context with its
  strength (how far the boundary cell's expected surprise rose above threshold).
  - It fails: strength is about the same everywhere (+0.11 in the story, +0.105 after the task).
  - The boundary cell fires on the input *before* an unpredictable stretch (the pause), not at
    the story's first word.
- **Bookmarks chosen when needed** (`RT_ANCHOR=choose`, default): the last 16 boundaries are
  kept. The striatum may go back to one at any step, or none.
  - The choice is made by a new `Striatum::choose_among`, "cortex proposes, basal ganglia
    dispose". Each candidate is scored by one shared pair of go/no-go pools on the state
    together with the candidate's own features:
    - its age (time cells);
    - its strength;
    - the sound before it and the sound at it.
  - A test learns which feature to pick per context on fresh candidates it has never seen.
  - A prefrontal request to retrieve now lapses after 32 steps unless renewed. Without that,
    retrieval mode never ended and storing stopped after a few hundred rows.

| 1,500 trials | read, same / another book | tell, book closed | recall right | find: right place |
|---|---|---|---|---|
| bookmarks chosen | 83.6 / 80.7% | 2.0% | 0.2% | 40% (15 trials) |

Reading is the best yet, but the bookmark of the story's start is not yet found: recall stays
near 0%.

### Actions that grow, with curiosity
- **A repertoire** ([`src/program/repertoire.rs`](../../src/program/repertoire.rs)):
  - The motor cortex proposes motor commands (babbling), and the basal ganglia choose among
    them with `choose_among`.
  - Each command is offered with its own bits and with what the motor area predicts it will do
    (its efference copy), so an action is chosen by its expected effect.
  - Half of all new proposals vary the command worth the most so far. A full repertoire drops
    the command worth the least.
- **What a command does is learned by the existing motor area** (`MotorArea::observe`): its
  forward model, motor command → what the senses report, on the same predictive kernels as the
  rest of the cortex.
- **Curiosity is learning progress:** a slow average of a command's prediction error minus a
  fast one, beyond the error's usual variation.
  - In a test, a learnable effect pays about 1.9 rewards over its first 10 tries, then nothing
    (boredom).
  - A random effect pays about 0.3 in all, and nothing later.
  - Each command costs ¹⁄₁₆ of a reward of effort.
- **The world has a body.** One field of a command's bits points the arm at one of four places;
  another closes the hand. A grasp toward a place with a book opens it. The senses report
  proprioception, the grasp, and a page when one comes into view.

**Result: it does not learn yet.** In 1,500 trials the basal ganglia released a command only 82
times, the book was never opened, and reading fell to 0–14%. It is opt-in (`RT_MOTOR=on`) until
it learns.

### Audit: the cortex is already motor, and what was built beside the network
**How the network acts already.** readtell speaks through the cortex:
- The cortex's bit cells (layer 2/3 and layer 5, [`BitCells`](../../src/program/bitcells.rs))
  predict the next sound: the plan.
- The motor area ([`MotorArea`](../../src/program/motor.rs)) turns that plan into a motor
  command. Its inverse model is learned by babbling, on the same predictive kernels as the rest
  of the cortex.
- The vocal tract says the word. The basal ganglia only decide go or no-go.

So an action is already *the cortex predicting its sensory consequence* (ideomotor theory;
active inference: a motor command is a proprioceptive prediction the body fulfils). Reaching can
use the same route, as the sketch in "The way back into the network" below describes.

**What the last round built beside the network instead,** and the counterpart in the network for
each:

| built | what it is | in the network already |
|---|---|---|
| motor repertoire (`repertoire.rs`) | a list of random commands, each with scalar averages (fast and slow error, deviation, worth) | the cortex's own predictions are the commands; new actions are new cells, which `BitCells` grows on its own |
| curiosity | scalar averages of one forward model's error per command | the cortex's prediction error is a bit overlap (plan against input), and each cell's confirmed or contradicted fires are its own record of it |
| bookmarks (`Mark`) | a list of 16 copied contexts, with age read from a hand-written table of bands | the index rows already hold each moment's context; how far the drifting context has moved from a row's is its age, in bits |
| anchor strength | one integer from the boundary cell | the same cell (kept) |
| `choose_among` | the striatum scoring hashed id lists of candidates | right in kind (cortex proposes, basal ganglia dispose); the candidates should be what the cortex or hippocampus actually has active, not lists made for them |

**Why it went this way.** Each was the quickest way to test an idea. Each added a second copy of
something the network does already: a second motor system, a second memory of past contexts, a
second error signal. Those copies carry scalar bookkeeping rather than bitvec populations.

**The way back into the network:**
1. **Actions through the cortex.**
   - The arm gets a motor area like the vocal tract's, learned by babbling, and the body reports
     proprioception as a sense.
   - The cortex learns, like any sequence, that after "read it" a page comes into view.
   - Its prediction goes through the arm's inverse model, and the basal ganglia gate it.
   - What can be done is whatever the cortex can predict, so the set of actions grows with it.
2. **Curiosity from the cortex's own error:** learning progress is the fall of the plan's
   mismatch with what comes, in bits, where the network acts.
3. **Bookmarks are index rows:** at the time of need, the candidates are the rows the current
   cue recalls, offered by their own bits (context, content). Nothing extra is copied.

The repertoire stays opt-in (`RT_MOTOR=on`) and bookmarks stay the default only until these
replace them.

### Actions through the cortex: a mouse
**What replaced the side structures:**
- **A mouse is the network's body** (relative moves and a click). The screen is a row of
  places: the desk with the story's book (unless it is closed), then the shelf.
  - The eye sees where the cursor is.
  - A mouse command is felt as its motion, sensed like a sound.
  - The mouse's motor area is learned by babbling before the trials, as speech is.
- **The cortex runs at the waiting steps** and learns what it senses next.
  - Every prediction a layer-5 cell makes that is a mouse motion is a candidate
    (`BitCells::proposals`). The basal ganglia release one or none, and the motor area turns it
    into a command.
  - While learning, the mouse also moves on its own now and then (motor babbling).
- **Curiosity** is the progress of the cortex's own misses there: a 64-bit shift register of
  misses, the long-run rate minus the last eight.
- **Bookmarks are index rows.** Pauses are stored like any sound. When one is needed, the
  hippocampus, cued with "a pause" in the current context, offers the rows that match. Each row
  is offered by its own keys: context, content, and the context it still shares with now.
- **Removed:** the motor repertoire, the body function, the bookmark list, the anchors, and the
  hand-made reach and place channels.

**Results (1,500 trials): not learned yet, and the default is broken.**

| | read, same / another book | tell, book closed | notes |
|---|---|---|---|
| all learned | 6.8 / 3.2% (silent 90%) | 1.8% | bookmark reinstated about 34 times a trial; 494 books opened; find: right place 45% (11 trials) |
| hold and reinstate given (oracle) | 2.4 / 2.4% | 46.7 / 55.2% | 61% of recalls right; about 4,900 mouse releases, almost all "right", so the cursor leaves the desk |

- With the start held, retelling works (55% with the book closed).
- Reading needs the click on the desk, and the learned mouse does not make it reliably.
- With everything learned, retrieval is re-entered on almost every step, and the network falls
  silent.

### A mask striatum
[`src/program/mask_striatum.rs`](../../src/program/mask_striatum.rs) is the basal ganglia in
popcount and masking only.
- **Values.** Each action has a go mask and a no-go mask over a context of fixed popcount (64 of
  2,048 bits, an equal min-hash share from each input pathway). Its value is
  `q = popcount(C & Mgo & R1) − popcount(C & Mno & R2)`.
  - Go and no-go start as the same random mask, so chance overlap cancels exactly and q
    starts at 0.
- **Selection:** a tonic brake, and an STN hold when the runner-up is within a margin of the
  winner.
- **Exploration** is dropout, drawn apart for the two pathways so that the noise does not
  cancel. Its rate τ doubles where the state's value is at or below zero (low tonic dopamine)
  or where most of the context is new.
- **The critic** is a mask pair scored the same way, learned TD(0) with discounting.
- **Learning is three-factor and bitwise.** A burst potentiates the go mask on the context's
  bits and depresses the no-go mask there (push-pull, so the pair keeps its density); a dip
  does the reverse. Options:
  - the swap rule (each mask at constant density, bits outside the context cleared);
  - sticky habit bits;
  - log-spaced eligibility taps;
  - conjunctive context bits (pairs of inputs from two pathways).

**A contextual bandit** (`HL_MODE=bandit`): 16 contexts, where twins share half their bits but
want different actions; 4 actions; a quarter of each context's bits are noise on each trial; 5
seeds. The best action is chosen per block of 500 trials, then in a test without exploration
(chance 25%):

| | first block | last block | test |
|---|---|---|---|
| cells (current striatum) | 46.2% | 95.2% | 98.1% |
| masks, swap rule | 30.2% | 78.8% | 83.2% |
| **masks, push-pull** | 46.4% | 95.2% | **98.4%** |
| masks, no STN hold | **60.6%** | 95.0% | 98.2% |
| **masks, sticky habit bits** | 48.4% | **98.4%** | **99.8%** |
| masks, conjunctive bits and swap rule | 17.7% | 80.8% | 86.1% |

**The hold ladder** (3 seeds, retold words right):

| rung | cells | masks, push-pull | masks, conjunctive bits and swap |
|---|---|---|---|
| hold given (reinstating learned) | **100%** | 16.7% | 81.9% |
| reinstating given (holding learned) | 0% (start held 0%) | 0% | 10.7% (start held 10.2%) |
| every other rung | ≤ 3% | ≤ 3% | ≤ 2% |

**Findings:**
- **Push-pull, not the swap rule.** The swap rule clears bits outside the context, and in a
  shared mask those are other contexts' learned bits, so learning one context erases others
  (83% against 98%). A burst that strengthens D1 and weakens D2 on the same synapses changes
  nothing else.
- **Starting go and no-go as the same mask matters.** With independent random masks each
  context starts with fixed, random preferences that dropout cannot overcome.
- **Tonic dopamine as temperature is needed.** An action that was never rewarded in a context,
  but gets value there through shared bits, is never corrected. The critic expects nothing
  there, so there is no dip. Doubling τ where the value is at or below zero finds the better
  action.
- **The STN hold slows early learning on the bandit** (46% against 61% in the first block). Its
  value is in conflicts the bandit does not test.
- **Sticky habit bits give the best bandit result** (99.8%).
- **On sequential gating the cells still win.** With the hold given, masks keep reinstating at
  every step, so recall restarts and only the first word is right. A mask is linear over
  context bits. The held context's bits stay the same from the instruction through the task,
  so "reinstate" weight learned on them carries over to steps where it is wrong. The cells are
  small conjunctions. Conjunctive context bits with the swap rule recover most of it (82%), and
  are the only variant that ever holds the story's start (10%).
- **Neither learns holding.** That stays the hippocampus's job (bookmarks), as the earlier
  ladder found.

### The mask striatum in readtell
The chosen setup (`RT_STRIATUM=masks`, now the default; `RT_STRIATUM=cells` for the old one):
- no STN hold;
- sticky habit bits;
- pair features (conjunctive context bits);
- the swap rule.

The mask striatum keeps the current state, so it takes the same calls as the cell striatum.

**Before it went in, two habit changes:**
- **Habits form more slowly.** A confirmed bit becomes sticky in three stages, as in synaptic
  tagging and capture: a first tag, a second tag, then sticky. Each step happens with
  probability ⅛ per confirming burst, so a habit takes about 24 bursts. A bit cleared on the way
  loses its tags.
- **A dip loosens a contradicted habit.** Sticky go bits in the context drop back to the second
  tag with probability ⅛, as the cortex's cell layers unstick a contradicted prediction.
  Without this a habit can never be undone: under the swap rule a dip only grows the no-go
  mask.

| this setup | bandit test | hold ladder: hold given, words right |
|---|---|---|
| habits in one step (⅛) | 95.3% | 16.7% |
| habits in three steps | 97.4% | 16.7% |
| three steps, and dips loosen habits | 92.3% | 69.1% |
| (no sticky bits, for reference) | 86.1% | 81.9% |

**readtell, 1,500 trials, one seed:**

| | read, same / another book | tell, book closed | bookmarks reinstated | books opened |
|---|---|---|---|---|
| cells, all learned | 6.8 / 3.2% | 1.8% | about 34 a trial | 494 |
| masks (habits in one step), all learned | 0.5 / 0.0% | 2.1% | 99,773 (every step) | 391 |
| masks (habits slower, loosened), all learned | 13.2 / 13.6% | 1.7% | 68,035 | 267 |
| cells, hold and reinstate given | 2.4 / 2.4% | 46.7 / 55.2% | — | 133 |
| masks (habits in one step), given | 13.7 / 18.6%; second test 46.8 / 55.1% | 57.1 / 59.7% | — | 413 |
| masks (habits slower, loosened), given | 0.0 / 0.0% | 52.2 / 52.3% | — | 284 |

**Findings:**
- **With the start given, masks retell as well as cells or better** (52–60% with the book
  closed), with recall right about 60% of the time.
- **Masks used the mouse more evenly** (left, right and click about equally, where the cells
  almost only moved right). In one run they read 47–55% on the second test. In the next run,
  with habits slower, the book was never opened at test. Opening the book by mouse is not yet
  reliable.
- **With everything learned, nothing works yet.** The bookmark channel releases on almost
  every step, so retrieval restarts continually and the network falls silent. Faster habits
  made this worse (every step); slower ones only reduced it.
- One seed per row: the differences in reading are within what one seed can swing.

### Bookmarks dropped; what breaks, found by ablation; the teacher shows the mouse
**Why bookmarks went.** The striatum had to learn *which* past moment to go back to and *when*.
On the ladder (a bookmark rung: every boundary kept, offered on every step):
- the masks went back on 8–16 of 16 story steps, whatever the effort cost or the features (pair
  features included);
- the cells never went back.

So the choice of *where* moved to the hippocampus. The striatum keeps one gate, *retrieve now*,
and the hippocampus, cued with a pause in the current context, goes back to the row it recalls
best.

**Four seeds, everything learned (masks): it fails.** Reading 0–12%, retelling with the book
closed 0–6%, silent 47–100%. The retrieve gate fired about 24 times a trial, and recall was 0%.

**Ablations (one seed each) show two separate faults:**

| | read (same / another book) | tell, book closed | recall right |
|---|---|---|---|
| book open from the start, masks | 56 / 66% | 1.5% | 0% |
| book open from the start, cells | 56 / 60% | 4.0% | 0.1% |
| book open, oracle retrieval | 49 / 57% | 54 / 65% | 61% |
| book open, gate may not restart retrieval (masks) | **72 / 64%** | 5–13% | 0.2% |
| book open, driver retrieves the recalled pause at the task's start | **78 / 76%** | 0–5% | 0.3% |

1. **The mouse broke reading.** With the book open from the start, reading is back for both
   striata.
2. **Restarting made the gate fire on every step:** each firing sent recall back to the start.
   Without restarting it fires about twice a trial. Restarting is now off by default.
3. **The recalled pause is the wrong one.** Even retrieving exactly when the task begins, recall
   is 0.3%. By then the context has drifted away from every old pause, so the most recent one
   wins: the pause *after* the story, whose link leads to the instruction. "The most recent
   pause" is the wrong rule.

**The teacher shows the mouse** (`RT_TEACH=mouse`, default). Speech is learned from the teacher's
words; the mouse had no teacher.
- In training, if a book is to be opened ("read", "find") and the network has not opened it by
  the time the teacher needs the remaining steps, the teacher moves the cursor to it and clicks,
  a step at a time.
- The network sees the cursor move and senses each motion as it would its own, so the cortex
  learns the sequence like the teacher's words.
- It is the teacher's act: the basal ganglia are not tagged.
- The teacher now waits long enough to show it from anywhere (3 + 4 pauses when there is a book
  to open).
- The striatum now also sees the cursor at the wait. It did not before, so it could not learn
  "go to the desk, then click".

| taught mouse, 1,500 trials | read (first / second test) | tell, book closed | book opened at test (read) |
|---|---|---|---|
| masks, seed 0 | 33 / 18%; 0 / 0% | 0% | 43 / 29% |
| masks, seed 1 | 15 / 4%; 8 / 7% | 6–8% | 20 / 0% |
| masks, oracle retrieval | 20 / 27%; 63 / 66% | 52 / 60% | 36 / 44% |
| **cells** | **61 / 66%; 70 / 72%** | 0–7% | **100 / 100%** |

**Findings:**
- **Taught, the cell striatum learns the mouse.** It opens the book every time for reading, and
  reading is back to 61–72%.
- **The mask striatum does not yet.** It opens the book 0–44% of the time, and with a book open
  it is often silent.
- **Retrieval is still not learned** with either striatum: the rule for where to go back to
  points at the wrong pause.

### The teacher reads from a book; imagining the opening cues recall
**Why the pause went.** The pause cue was chosen by hand, not learned by the hippocampus, and it
was the wrong cue: the most recent pause is the one after the story. The trigger should be how
the episode began, as the cortex represents it: here, the opening of the book. Imagining someone
opening it is the cortex producing that same pattern without acting.

**What was built:**
- **The teacher reads the story from a book** (`RT_STORY=book`, default). The book is opened with
  a click the network sees; the hippocampus stores it like any event, and the story's words
  follow, linked. The network sees each word as it is read.
- **Imagine or do** (`RT_RETRIEVE=imagine`, default). The cortex's layer 5 predictions of a mouse
  motion are offered to two channels:
  - *do* sends one to the mouse's motor area;
  - *imagine* sends one to the hippocampus as a content cue.
- **Three repairs to recall, found by tracing it:**
  - **Cue by what is imagined alone.** With the current context in the cue too, the context's
    rare bits outweighed a content that every episode shares, and recall went to whatever was
    stored last. That also sank the pause cue. Cued by content alone, every opening matches as
    well, and the most recent wins: this story's.
  - **Refractoriness.** The rows recalled in the last 4 steps do not win again, so recall moves
    on along the links instead of repeating the opening.
  - **Retrieved context** (the temporal context model). While retrieving, the recalled row's
    context becomes the current one, so the context walks forward with the story. Before, the
    opening kept winning because nothing moved the context during a retelling.
- The pause cue and pause rows are kept only as `RT_RETRIEVE=pause`.

**Results (1,500 trials, one seed per row):**

| | tell, book closed (two test tables) | recall right | read (two test tables) | book opened for reading |
|---|---|---|---|---|
| driver imagines the click at the task's start | 40.8 / 47.5% | 32% | 22–28 / 15–16% | 0% |
| **masks, seed 0, all learned** | **9.3 / 42.4%** | 20% | 42–49 / 26–34% | 75–83% |
| masks, seed 1, all learned | 0 / 0% (silent) | 14% | 18–30 / 0% | 0% |
| cells, all learned | 20.1 / 15.5% | 25% | 28–33 / 11–17% | 0% |

**Findings:**
- **The first learned retelling with the book closed:** 42% (masks, seed 0) and 15–20% (cells).
  The network imagines the opening of the book, and that cue lands on the story's start. Nothing
  says which cue to use; the imagine channel learns it.
- **Before the three repairs, the same cue gave 0.2%** even when the driver imagined it at the
  right moment. The cue was right; recall was not following it.
- **Not reliable yet.** Masks on seed 1 fell silent on the second test. The cells learned
  retelling but stopped opening the book (they mostly moved left). The mouse and the imagine
  channel share the same proposals, and how they settle varies by seed.
- **Reading is lower than with the book given** (26–49% against 72–78%). Opening by mouse is
  still part of every reading trial.

### The context moves with what the cortex represents
**The question.** Why did the context stay static? It moved only when something was heard, so
it followed the input stream rather than the cortex's state. When the network was silent, or
recalled or planned a word without saying it, the context froze. The "retrieved context" fix
(copying the recalled row's context) patched this from the hippocampal side.

**Now the context moves every step with the association area's assembly of what the cortex
represents:**
- what is heard or seen;
- when nothing is, its plan (a word recalled or meant, said or not);
- while retrieving, the recalled assembly: the cells that were active when that row was stored.

The network's own speech, heard back while retrieving, is predicted (corollary discharge) and
does not move it again. The patch is gone.

**Making the replay exact.** With the recalled assembly driving the context, retrieval replays
the episode's own context trajectory, but only if the replay matches what happened at storing:
- each row keeps its context slot by slot (the index keeps it only as a set, and each slot's new
  value depends on the old one);
- the context moves with the row just passed, as it did then;
- each row also keeps whether an event boundary followed it, and the replay makes the same jump.

Traced: after the opening, the context matches each next row exactly (32 of 32) for four steps,
then 28 and 18 of 32. Reconsolidation re-points some rows, which may explain the drop; that has
not been checked. The best match by context is the next row at every step, so the links and the
context agree.

**Results (1,500 trials, one seed per row; two test tables: with the instruction held, then
with the prefrontal content removed):**

| | tell, book closed | recall right | read | book opened for reading |
|---|---|---|---|---|
| driver imagines the click at the task's start | 41.6 / 51.2% | 28% | 0–6% | 0% |
| **cells, all learned** | **45.2 / 35.8%** | 23% | 26–35% | 0% |
| masks, seed 0, all learned | 12.3 / 0% (silent on the second table) | 13% | 0–15% | 8–25% |
| masks, seed 1, all learned | 24.4 / 0% (silent on the second table) | 12% | 0–36% | 40–73% |

**Findings:**
- **With cells, learned retelling doubled:** 45% with the book closed (from 20%), close to the
  driver-timed cue (42–51%).
- **With masks it got worse,** and on the second test table (prefrontal content removed) the
  network falls silent on every seed.
- **Doing and imagining still trade off.** The cells retell well but stopped opening the book;
  the masks open it more often but retell less.

### An audit, checked against the code
An outside audit of the design, written against a one-paragraph summary in bit notation, was
checked claim by claim against the code:

| Claim | In the code | Verdict |
|---|---|---|
| pair bits are ≈ a\|b, not conjunctive | pair bits are a hash of the pair, so (a,b) and (a,c) give unrelated bits | does not apply: the summary's notation was wrong |
| no go LTD on a dip | true under the swap rule (readtell's): a dip only grew the no-go mask | **valid: fixed** |
| drift erases the context in silence; its size is not fixed | the slot form is kept (4 of 32 slots), and there is no drift on an empty assembly | does not apply (a formula in the summary) |
| the plan feeds the association area and back, the area learns from predictions, own words count twice | the plan moves only the hippocampal context, without learning, on silent steps; own speech is not counted twice | mostly does not apply |
| without context reinstatement, recall loses the episode | the recalled row's ordered context is reinstated at the start and replayed; it slips over time | partly: a **partial blend added** |
| cortex sticky bits never clear | contradicted bursts unstick (`bitcells.rs`), and losing cells prune | does not apply; growth is not density-paired (true) |
| the index grows forever, with no eviction | rows lose strength and are forgotten (about 78,000 of 90,000); sleep replay; the cortex recycles failing cells | mostly does not apply |
| "a step" is undefined, so drift per step changes meaning | fair | valid, open |
| curiosity is unread and global | it is a reward at the waiting steps, but global | partly valid, open |
| per-pathway permutations; rarity weights as power-of-two tiers via LZCNT | one salt (pathway ids are disjoint); a reciprocal table | good suggestions, open |

**Fixes made:**
- **Go LTD.** Under the swap rule a dip now also moves the go mask away from the context, at
  constant density: it clears picks of `M_go & C & ~sticky` and sets as many of `~M_go & ~C`.
  - Bandit test: 92.3% → **98.8%**.
  - Hold ladder, hold given: 69% → **76%** of words retold right.
- **Partial context reinstatement on recall.** While retrieving, each slot takes the recalled
  row's stored value with probability ¼, so the context stays in the episode as the replay
  slips.

**readtell after the fixes (1,500 trials, one seed per row; two test tables):**

| | tell, book closed | recall right | read | silent |
|---|---|---|---|---|
| driver imagines the click | 45.7 / 59.8% | 31% | 0–48% | 19–23% (tell) |
| **cells** | **38.8 / 35.2%** | 25% | 30–42% | 21–39% |
| masks, seed 0 | 3.0 / 2.4% | 9% | 4–11% | 79–100% |
| masks, seed 1 | 0.7 / 0.0% | 9% | 0–1% | 43–100% |

**Findings:**
- Go LTD fixes the bandit and the ladder, but **not the masks in readtell**: the network is
  silent on most steps.
- With cells, learned retelling stays at 35–39% with the book closed.
- The masks' silence is a different fault, not yet found. Next: trace which gate (working memory
  load, imagine, act) drives it.

### Tracing the masks' silence
Each of the three channels in use was handed to the cell striatum in turn (`RT_CELLS_FOR`), the
rest staying on masks (1,500 trials, seed 0; two test tables):

| channels on cells | tell, book closed | read | silent, second table |
|---|---|---|---|
| none (masks everywhere) | 3 / 2% | 4–11% | 83–100% |
| working-memory load | 20 / 0% | 0–51% | 100% |
| imagine | 10 / 0% | 7–27% | 64–100% |
| act (the mouse) | 21 / 4% | 31–39% | 18–92% |
| **act and imagine** | **31 / 49%** | **31–47%** | **17–29%** |
| act and imagine, masks' brake at 1 | 35 / 40% | 16–32% | 17–28% |

**The masks' mouse channel causes the silence:**
1. The masks release the mouse far less often (about 865–1,400 times in training, against about
   3,100–3,500 for cells).
2. So the teacher opens the book far more often. A demonstration does not credit the network's
   own striatum, so it never learns to open the book itself.
3. At test there is no teacher, the book stays shut, and with nothing to read the network says
   nothing.

**Why the masks release less.** They start at q = 0 exactly, and dropout noise rarely reaches the
brake. Lowering the brake to 1 or 0 did not help: releases stayed around 1,400 and the second
table stayed silent.

**Why it fails more deeply: the masks do not grow.**
- The cell striatum recruits a new go or no-go cell whenever a burst or dip finds none that
  covers the situation, up to 4,096 per pool.
- The mask striatum has one fixed go mask and one fixed no-go mask per action (2,048 bits at
  density ⅛), over a 64-bit context shared by nine pathways (about 7 bits each, so the cursor's
  place gets about 7).

Everything the mouse channel must learn (cursor place × instruction × waiting step ×
candidate) goes into one mask pair, and new learning overwrites old. That fits a 16-context
bandit, not readtell. A mask that grows is several masks per action, each voting when
`|C & M_i| ≥ θ`, which is a striatal cell with a large receptive field. The two designs meet:
the cell striatum's growth with the mask striatum's learning rules.

### A mask striatum that grows
**Grow mode** (`MaskStriatum::grow`; readtell `RT_MASK_GROW=1`): the cell striatum's growth with
the mask striatum's learning rules.
- **Pools of mask-cells.** Each action has a pool of go cells and one of no-go cells, and the
  critic has a positive and a negative pool. Each cell is a 16-bit mask over the context that
  fires when at least 10 of its bits are in it: a striatal cell with a large receptive field.
  q = go cells firing − no-go cells firing.
- **A burst:** go cells that fired tighten toward the context by one bit and consolidate (slow
  habits); n new go cells are recruited from the context (n from |δ|, as in the cell striatum);
  up to n no-go cells that fired move away from it.
- **A dip** does the reverse: go cells that fired move away (go LTD) and their habits loosen; n
  no-go cells are recruited.
- **A full pool** (2,048 cells) drops, of a few sampled, the cell unused longest.
- **Exploration** is GPi's tonic noise (±3 per option). Dropout would silence threshold cells:
  keeping half the bits leaves almost no cell with 10 of its 16.

**Three bugs found on the way:**
- **Recruiting only "if too few fired"** capped a value at about n cells (¼ reward): the critic
  never learned a full reward. Like the cell striatum, every burst now recruits n.
- **A swap that did not preserve density.** When a mask's bits all lay inside the context there
  was nothing to remove, but a bit was still added. Each burst then grew the mask by one bit,
  until a cell covered the shared pathway and fired in every context (pools of 400+ cells firing
  everywhere). Swaps now add only as many bits as they remove. The plain masks had the same
  drift, which is the density problem the audit warned of.
- **Dropout in grow mode** made training choices near random (60% on the bandit while the test
  was 100%).

| contextual bandit (5 seeds) | first block | last block | test |
|---|---|---|---|
| cells | 46% | 95% | 98.1% |
| masks (no hold, sticky, pairs, swap) | 47% | 96% | 97.8% |
| growing masks, sticky | 53% | 90% | 91.4% |

| hold ladder, hold given (3 seeds) | reinstated after the instruction | reinstated mid-story | words retold right |
|---|---|---|---|
| cells | 99.4% | 0.7% | 100% |
| masks | 3.6% | 420% | 75.6% |
| **growing masks, sticky** | **100%** | **0%** | 73.2% |

**readtell, growing masks, all learned, 4 seeds (1,500 trials; the first test table, with the
instruction held):**

| seed | read (same / another book) | tell, book closed | recall right | book opened at test |
|---|---|---|---|---|
| 0 | 47.8 / 45.7% | 38.9% | 24% | 100% |
| 1 | 5.0 / 8.6% | 2.9% | 4% | 0% |
| 2 | 50.6 / 49.3% | 46.8% | 20% | 100% |
| 3 | **57.6 / 56.6%** | **54.2%** | 22% | 100% |

**Findings:**
- **The best fully learned results so far.** Three of four seeds open the book every time at test,
  read 46–58% and retell 39–54% with the book closed. The mouse problem is gone: the pools grow
  to hold cursor place × instruction × step, where one fixed mask pair could not.
- **Seed 1 fails.** Its mouse mostly moves right, and the cursor wanders off the book.
- **With the prefrontal content removed (the second test table) it falls silent** (84–100%),
  where the cells did not. This is not yet understood.

### Audit items, continued
- **Per-pathway permutations.** Each input pathway now has its own min-hash permutation, so their
  picks are not correlated.
- **Rarity weights as power-of-two tiers.** The index's vote for a cue index used by n rows is
  2^−⌊log₂ n⌋ (a leading-zero count and a shift) instead of 1/n from a reciprocal table
  (`IndexConfig::tiers`; readtell's default, `RT_HC_TIERS=0` for 1/n).
- **Curiosity without a divide, per context.** Progress is the misses 9–16 steps back minus
  those of the last 8. Each context has its own register (the working memory's content and the
  cursor's place, hashed into 64), so learning in one situation is not averaged with others.
- **A step is one represented event:** a heard or seen word, a pause, a mouse motion, or while
  retrieving, a recalled word. The context drifts once per step.
- **Context at several timescales** (`RT_CTX_BANKS=1`). Besides the fast bank (32 slots, 4 per
  step, a half-life of about 5 steps), two slower banks (16 and 32 slots, one per step: about
  11 and 22 steps). Event boundaries jump only the fast bank, so the slow banks carry the episode
  across boundaries. The replay uses the same per-bank drift. On 4 seeds (growing masks, first
  test table) it does not change the overall picture: three seeds work either way, but a
  different one fails.

  | seed | without banks: read / tell, book closed | with banks: read / tell, book closed |
  |---|---|---|
  | 0 | 47.8–45.7% / 38.9% | 51.8–42.9% / 33.6% |
  | 1 | 5.0–8.6% / 2.9% | **43.5–46.3% / 34.9%** |
  | 2 | 50.6–49.3% / 46.8% | 51.9–23.6% / 42.4% |
  | 3 | 57.6–56.6% / 54.2% | 0 / 0% (silent) |

  The failures are seed-specific: one seed's mouse wanders, or the network falls silent. They
  are not caused by the context's timescale.
- **Top-k recall** (`RT_HC_K`). When recall does not follow a link, the k best matches vote and
  the output keeps the bits most of them share (a majority). Not yet measured.

### Why one seed in four failed: superstition
**The wandering mouse was not curiosity.** Seeds 1 and 3 pressed LEFT thousands of times (about
4,800 against about 750 in working seeds) and never opened the book at test. Curiosity was the
first suspect, but it is not the cause:

| curiosity | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|
| as reward | works | wanders | works | works |
| as reward, registers start empty | works | wanders | works | wanders |
| off | reads, no reach at test | works | reads, no reach at test | wanders |
| as noise (below) | works | wanders | works | wanders |

**The count that found it.** In read trials, with the cursor already on the desk book (where
only a click helps):

| | steps | CLICK proposed by the cortex | LEFT proposed | CLICK released | LEFT released |
|---|---|---|---|---|---|
| seed 0, training | 246 | 225 | 85 | 221 | 3 |
| seed 0, test | 18 | 18 | 0 | 18 | 0 |
| seed 3, training | 1,412 | 595 | 1,394 | 8 | 1,386 |
| seed 3, test | 152 | 0 | 152 | 0 | 152 |

LEFT on the book moved nothing, but the motion was still sensed. Then the teacher, needing the
remaining steps, clicked and the book opened. The burst that followed credited the network's own
tagged LEFT presses (the teacher's act is not tagged): a superstition, as in Skinner's pigeons.
Two locks followed:
- **The striatum:** go cells for LEFT on the book grew and became habits. Even when the cortex
  proposed CLICK, it was released 8 times in 595.
- **The cortex:** it learns what follows what. It learned that LEFT follows LEFT on the book (the
  network's own presses), and stopped proposing CLICK. The basal ganglia only choose among the
  cortex's proposals, so by test CLICK was never on offer.

Seed 0 happened to click before the teacher stepped in, so the superstition never formed.

**Two fixes, each tested on seeds 1 and 3:**
- **The world, made honest** (`RT_BLOCKED`, now the default): a move against the screen's edge
  moves nothing, and nothing is sensed. The cortex no longer learns that LEFT follows on the
  book, so CLICK stays on offer.
- **The teacher's act, tagged as if done** (`RT_OBSERVE=1`, the mirror system's view): the
  burst credits the observed click, not the earlier LEFT presses.

| | seed 1: read / tell, book closed | seed 3: read / tell, book closed |
|---|---|---|
| neither (noise curiosity) | 18–21% / 21%, no reach | 0–2% / 13%, no reach |
| `RT_OBSERVE=1` | 41–27% / 34% | 45–51% / 51% |
| `RT_BLOCKED=1` | 45–33% / 42%, reaches 100% | 50–43% / 27%, reaches 100% |

Both stop the wandering (left presses back to about 800). The blocked move is the default
because it only corrects the world.

### Curiosity as noise
Curiosity paid as reward competes with the task: it pays the mouse to go where predictions are
improving, whether or not that helps. Now (`RT_CURIOSITY=noise`, the default) the same progress
signal (misses 9–16 steps back minus those of the last 8, for the current context) adds up to ±2
to the growing masks' GPi noise (±3) for the mouse and imagine choices, and pays nothing. It
loosens a habit a little where there is something to learn; it does not choose what replaces it.
`RT_CURIOSITY=reward` keeps the old form; `0` turns it off. As noise, seeds 0 and 2 open the book at
test (with curiosity off they did not); seeds 1 and 3 still wandered before the blocked-move fix.

Biology suggests the source should be the cortex, not a table in the driver. Random exploration
is linked to the locus coeruleus's tonic mode (a noise gain); what drives it is surprise and
uncertainty tracked in the anterior cingulate and orbitofrontal cortex. Directed exploration
(trying the uncertain option on purpose) is frontopolar, and advance information does excite
dopamine neurons. The cortex here already marks its own surprise: a predictive kernel bursts when
its prediction fails. A candidate: a slow running rate of cortical bursts as the noise gain,
damped while reward is coming. Not built.

### Silence without the instruction: no input at all
A trace of each step's inputs to the cortex, in the test with the prefrontal content removed:

| input | bits |
|---|---|
| eye | 0: the book stays shut (the mouse never reaches, even when asked to read) |
| ear | 0: nothing said yet |
| prefrontal | 0: removed |
| hippocampal recall | 0: nothing cues it |

For one or two steps, leftover activity in the cerebellum and layer 2/3 gives a guess; then the
plan is empty. Reaching, imagining the opening (which cues recall) and so the whole chain are
learned in the instruction's context; without it nothing starts. This is arguably right (no task,
nothing to do). Whether the story just heard should be enough of a cue is a design question.

### Top-k recall, and the new defaults
Defaults now: growing masks (`RT_MASK_GROW=0` for one mask per pathway), blocked edge moves,
curiosity as noise. Top-k recall (`RT_HC_K=3`: the 3 best matches vote, a majority keeps the
bits) against top-1, first test table:

| seed | top-1: read same–other / tell closed / recall right | top-3: read same–other / tell closed / recall right |
|---|---|---|
| 0 | 49.7–28.2% / 30.0% / 19.4% | 36.7–32.5% / 24.7% / 21.7% |
| 1 | 44.8–32.9% / 41.7% / 23.3% | 42.1–45.3% / 27.8% / 24.3% |
| 2 | 37.2–28.7% / 28.5% / 23.5% | 38.5–45.1% / 64.3% (28 words) / 24.1% |
| 3 | 49.7–43.1% / 26.6% / 20.8% | 56.7–58.2% / 37.9% / 20.6% |

No seed wanders now, and all four read and tell at 25–60%. But with top-1, seeds 0 and 2 did not
reach for the book at test (seed 0 also with top-3): those read scores come from hearing the
teacher read, not from seeing the page. Top-3 recall is right on slightly more telling steps
(about +1 point); on what is said, it is mixed, within seed-to-seed noise. Top-1 stays the default.
With the instruction removed, both are 85–100% silent (above).

### Superstition again: a click on nothing, and the edge
**Seeds 0 and 2 stopped reaching at test** with blocked edge moves. Logging where the cursor sat
and what was proposed there: at test it stayed at a shelf place (left there by the last find
trial) and clicked, every step (seed 0: 119 of 119; seed 2: 140 of 140), though LEFT was
proposed (60 times on seed 0). A click on nothing was still sensed, and the teacher's eventual
move and click credited it: the LEFT superstition in another form.

**A click that opens nothing now senses nothing too** (part of the default blocked world). Seed 0:
reaches 100%, reads 46–51%, tells with the book closed 33%. Seed 2 slid into a third form: at
the right edge it pressed RIGHT (79 of 84 test steps, LEFT proposed 76 times), and its speech
collapsed (3–5% right, 50–70% silent).

**Two credit fixes that did not hold:**

| | seed 0 | seed 2 |
|---|---|---|
| blocked clicks + the teacher's act tagged (`RT_OBSERVE=1`) | reads 53–59%, reaches 100% | silent; the cortex proposes no mouse act (84 of 84 steps) |
| blocked clicks + the teacher seen, not felt (`RT_TEACH_SEEN=1`) + the check (`RT_CHECK=1`) | silent; no proposals at place 4 (161 of 161) | RIGHT at the edge (133); reads 26–28% |

- **Seeing the teacher instead of feeling it**, without a mirror, removed the demonstrations
  from what the cortex learns as motion: CLICK was proposed on the book 12 times in 148.
- **The check trigger:** when a reward follows the network's own acts, each is marked due a
  check; the next time it is done, the forward model's expected effect is compared with what
  comes, and no effect at all is a dip. About 820 checks per seed, 150–170 found no effect, but
  the RIGHT habit at the edge survived.

**Why speech dies with the mouse** (trace of seed 2, telling with the book closed): the
prefrontal content is there and recall gives 32 bits, but it is the story's last word (".")
at every step; the cortex makes no plan from it. Imagining the opening is what takes recall
back to the story's start, and imagining uses the cortex's mouse proposals: when those die, so
does imagining, so recall is stuck at the end and the network is silent. The mouse failure and
the speech failure are one failure.

### Cortical curiosity
Curiosity's noise now comes from the cortex itself (`RT_CURIOSITY=cortex`, the default). Each
layer 5 cell keeps its own record of its last 16 predictions as winner (hit or miss, one bit
each). When it predicts, its learning progress (misses 9–16 back minus those of the last 8,
less one) is the extra GPi noise, 0 to ±2. The per-context miss registers hashed in the driver
are gone from the default. `RT_LC_DAMP=1` adds the locus coeruleus's brake: no extra noise while
the critic expects reward. The noise is small (+0.09 on average). On 4 seeds (before blocked
clicks): seeds 0 and 1 work (reach 100%, tell closed 34–37%); seeds 2 and 3 click at a shelf
place at test.

### An always-completing hippocampus
`RT_HC_AUTO=1`: no cue and no request. Every step the entorhinal input (the context, and the
association area's assembly for whatever the cortex represents: heard, seen, said, meant or
imagined) is completed to its best stored row (not the one just written: that was the other half
of the theta cycle). The row that followed it is the output, and the context is pulled half way
toward the completed row's own. Acetylcholine alone sets the mode: high (after a mismatch), the
output is suppressed (encoding); low, it reaches the cortex (retrieval). Writing goes on in both,
except for its own speech. Imagining the opening just makes the cortex represent it, and the
hippocampus completes it. A 60-trial check runs; not yet measured fully. A concern: the context's
rare bits may outweigh an imagined content seen in every episode (why the earlier cue used
content alone).

### An area per device, and association that grows
`src/program/area.rs`:
- **`Brain::attach(name, Sensor | Motor | SensorMotor)`** gives a device its own cortical area: a
  sensory side (its own layer 4, a pool that recruits cells as new inputs need them: `k` cells per
  new input, a cell fires when 10 of its 16 synapses are on) and a motor side (inverse and
  forward models). Issuing a command keeps the efference copy; what is sensed next is compared
  with it (the act's effect error).
- **Pairwise association areas** grow for two areas active together often enough (8 steps);
  their cells take half their synapses from each, and fire on either half alone (8 of 16), so
  one sense evokes the pair's code.
- **A full association area** grows once there are two pairwise areas, conjoining them. The
  output is the highest level that is active (the full area recruits only when two pairs are
  active together; before the fix it returned nothing, and a word seen and heard shared 0 of 32
  cells; now 28.1, against 16.5 with the old association layer).
- **A synapse moves only within its own source, and only if that source was active,** so a pair's
  cell keeps both halves when one sense comes alone (as layer 4's pathways rule).
- **The mirror** (`Brain::mirror`): one side of a pair evokes the cells that conjoined it with the
  other; their synapses on the other side project back to that area's layer 4 code, and that
  code back to its input (`Pool::backproject`, by vote). Biology: mirror neurons are learned from
  correlated experience (Heyes; Keysers & Perrett): acting, one sees one's own hand, and the visual
  (STS) and motor (premotor) representations are linked by Hebbian learning.

**In readtell** (`RT_AREAS=1`): eye, ear, hand (sensorimotor), voice (motor) and **seen acts** (the
STS stream: what an act looks like, wherever it happens). The hand's felt motion is its own basal
frame in layers 2/3 and 5 (the mouse no longer borrows the ear). The voice babbles in its own area;
saying a word keeps its efference copy, compared with what the ear hears. Own acts, seen and felt,
grow the association of seen acts with the hand; with `RT_TEACH_SEEN=1` the teacher's act, only
seen, is mirrored to the felt motion, and the cortex takes that as what came. In a 100-trial
check, 126 of 157 of the teacher's acts were mirrored to the right felt motion (the misses come
before own acts have built the link). The mirror needed its own visual stream: with the eye's
whole scene (the cursor's place and the act) as one code, own acts at one place did not carry over
to the teacher's at another (15 of 119).

Before the association fix, the area port worked on seed 0 (reads 34–46%, tells closed 38%,
reaches 100%) and failed on seed 1 (RIGHT at the edge, silent); with the voice in its own area too,
both seeds pressed RIGHT and were silent (the voice itself: 46 words said, none heard other than
its efference copy expected).

### The areas after the association fix, the mirror, and the teacher's own cursor
Seeds 0 and 2, first test table; "reach" is the share of test trials in which the network opened
the book itself.

| | seed 0: read / tell closed / reach | seed 2: read / tell closed / reach | the mirror right |
|---|---|---|---|
| areas | 6% / 14% / 0 (RIGHT at the edge, 112 of 119) | 38–39% / 2.5% / 91–100% | — |
| areas + the teacher seen, mirrored | silent / 0 (RIGHT at the edge) | silent / 0 (RIGHT at the edge, 133) | 98% (2,448 of 2,489; 2,484 of 2,514) |
| areas + the teacher's own cursor | 1% / 5% / 0 (LEFT on the book) | 13% / 2% / 100% | 98% (1,439 of 1,469); 96% (1,110 of 1,152) |

- **The association fix holds:** a word seen and the same word heard share 26.5–28.3 of 32 cells.
- **The mirror works:** 96–98% of the teacher's acts evoke the right felt motion. But mirroring the
  teacher's acts while they still move the network's own cursor does not stop the RIGHT habit:
  the teacher's completion still credits the network's useless presses.
- **The teacher's own cursor** (`RT_TEACHER_CURSOR=1`): the teacher shows how on their own copy of
  the book, with a cursor of their own; the network's cursor stays put and its book opens only
  when it clicks it itself, so reward follows only its own acts. The eye sees both cursors (a
  cursor looks the same whoever's it is), and the teacher's act is seen and mirrored. Seed 2
  reached on every test trial, the first time this seed did so with areas; seed 0 kept a LEFT
  habit on the book (CLICK never proposed there at test). The network opened its own book far
  less often in training (255 and 585 times, against about 900 when the teacher opened it), so it
  saw the page less: reading fell.
- **Hippocampal recall is broken with areas:** it held the wanted word on 0–1.6% of telling
  steps (about 20% before), so telling with the book closed fails even where the mouse works.
  Not yet diagnosed.
- **The voice's corollary discharge:** in every run, every word said was heard as its efference
  copy expected (908 and 1,332 words where it spoke).

### One column for every area
Are the association areas cortex? Until now, no. The cortex was three mechanisms: layer 4 pools
(encoding only: the senses and the association areas), one shared pair of layers 2/3 and 5 (the
only prediction, for every modality at once), and predictive kernels for the motor maps. The
cortex is largely one six-layer circuit everywhere (Mountcastle); areas differ in their
connections: sensory cortex has a thick layer 4, motor cortex almost none (agranular: layer 5
drives the muscles, and the forward prediction is the column predicting the sensory result),
association cortex is the same circuit fed by other areas, predicting and sending feedback.

`src/program/column.rs`: one `Column` for every area.
- **Layer 4:** the growing pool (none in an agranular motor column).
- **Layer 2/3:** primed bit cells predicting the column's next input from its input, its own
  previous activity and context on the tuft (from above). With a command as its input, its
  prediction is what the command will be sensed as: the forward model, and the efference copy.
- **Layer 5:** the output; in a motor column, what is meant → the command (the inverse model).
- Both layers grow a cell for the moment where nothing predicted the target.
- **Tests:** a motor column learns by babbling to say and to predict 60 words (95%+; the kernels
  did the same); a sensory column predicts the next input of a sequence; association columns
  predict the next scene of a story read aloud (at least 7 in 8).

**Ported** (`RT_COLUMNS=1`, which implies areas): the voice's and the hand's motor sides are
agranular columns (`Motor::Column`), learned by babbling; association areas are columns whose
layer 4 is the conjunction pool and whose layer 2/3 predicts their next code. In a 100-trial check
the association columns predicted the next scene 59% of the time, the voice said 46 words (all
heard as expected), and 60 of the teacher's 73 acts were mirrored right. Still to port: the senses
as full columns, and the shared layers 2/3 and 5 replaced by the top association column, with
feedback down as tuft context.

### Acts as acts; the columns ported; the top column
All with the teacher's own cursor, seeds 0 and 2, first test table.

| | seed 0: read / tell closed / recall right | seed 2: read / tell closed / recall right | own book openings (training) |
|---|---|---|---|
| columns (motor and association) | 2–3% / 0% / 1.6% | 1–4% / 20% / 0% | 264; 152 |
| areas + acts as acts | 3–10% / 10% / **12.7%** | 0% / 0% / 0.1% | 542; 165 |
| top association column (senses as columns) | 5–6% / 9% / 5.4% | 1–4% / 3% / 0.6% | 530; 230 |

- **Acts as acts.** With areas, the teacher's opening click had still gone in through the ear (the
  old shortcut), and an imagined act was coded through the ear's area, whose thresholded cells
  mostly gave no code: 319 of 5,138 imaginings found a row. Now an act, seen, done or imagined,
  is the association of the hand (how it feels) and seen acts (how it looks), and the teacher's
  click is seen. Every imagining finds a row; recall on seed 0 rose from 1.6% to 12.7%.
- **The top association column** (`RT_TOP=1`): the senses are columns (layer 2/3 predicting the
  next code, the top's activity as tuft feedback), and one top column replaces the shared layers
  2/3 and 5: input the scene and the hippocampus's return, tuft the prefrontal content and the
  cerebellum; layer 2/3 predicts the next scene (53% right), layer 5 sends the goal. It reached
  the book on 0–29% of test trials and was silent half the time.
- **The common failure is the mouse.** With the teacher on their own cursor the network opens
  its own book rarely (150–540 times in 1,500 trials), and a LEFT habit forms on the book (seed
  2, areas: LEFT released 1,228 times there in training, CLICK 16). A guess: each wait is
  followed by the story, whose rewards (words predicted right) do not depend on the wait's acts,
  yet reach back to them. Being tested (`RT_WAIT_END=1`: the wait is its own credit episode).
- **The mirror** stays accurate (97–99%), and the voice was never heard other than its
  efference copy expected.

### What the biology says about the layout, and streams
The cortex is largely one circuit (Mountcastle; Douglas and Martin), with layer 4 for input,
feedback onto layer 1's tufts, and agranular motor cortex. Where the model differed:
- **Feedforward runs from layer 2/3** of the lower area to layer 4 of the next (Felleman and Van
  Essen), not from layer 4.
- **Every area's layer 5** reaches the striatum, the pons and the higher-order thalamus (the
  transthalamic route, Sherman and Guillery), not only the top's.
- **No single top:** parallel streams, ventral (what: objects, word forms, meaning) and dorsal
  (where and how: sensorimotor), with several apexes.
- **Speech is planned in a dorsal auditory–motor stream** (Hickok and Poeppel: an area at the
  parietal–temporal junction, Broca's, premotor); reading's meeting place of seen and heard
  words is the visual word form area.
- **The entorhinal cortex is not an association area of every area:** it takes the highest
  association areas (perirhinal: what, into lateral EC; parahippocampal and retrosplenial:
  where, into medial EC; prefrontal, cingulate, insula), and smell directly; it returns the same
  way, the route recall uses.

**Two versions built:**
- **Explicit streams** (`RT_STREAMS=1`): feedforward from layer 2/3 (the code with the sense
  column's activity); a ventral column (the word seen and heard, the hippocampus's return; its
  predicted next word sent down to the ear as the sound expected, `Brain::down`, a source of the
  plan; its activity the dorsal tufts' feedback); a dorsal auditory–motor column (layer 5: the
  meant sound) and a dorsal visual–motor column (layer 5: the mouse's acts); both layer 5s reach
  the basal ganglia and the pons.
- **Self-organizing** (`RT_SELFORG=1`): no streams wired. Biology suggests streams are largely
  shaped by input statistics on a coarse innate scaffold (rewired ferret auditory cortex comes
  to see; visual cortex reads Braille in the blind; a single network trained to predict can
  split into ventral- and dorsal-like pathways). Every association column grown by co-activity
  has layers 2/3 and 5; every column's layer 5 offers goals to every motor area; which column
  drives which is learned (the relay's reliability per source and context), and a column learns
  only where its input is active. In a 150-trial check the streams formed by themselves: the
  eye+ear column alone offered sounds (58% right), and eye+hand, ear+hand, hand+seen acts and the
  full column offered only mouse acts.

### Self-organizing, streams, the story's credit, and play: results
Seeds 0 and 2, 1,500 training trials, the teacher's own cursor, first test table.

| | read right | tell closed | silent | reach (test) | recall right |
|---|---|---|---|---|---|
| self-organizing | 0% / 0% | 0% / 0% | 100% / 100% | 0 / 0 | 8.3% / 6.6% |
| explicit streams | 1% / 5–12% | 0% / 6% | ~50% / 54–82% | 0 / 100% | 0.4% / 0.5% |
| self-organizing + touch + play (300) | 0–2% / 6–7% | 0% / 0% | ~80% / 29–100% | 0 / 100% | 3.0% / 9.2% |
| areas + the wait as its own credit episode | 0% / 0% | 0% / 0% | 100% / 100% | 0 / 0 | 0.3% / 0% |

None comes near the shared layers (read 40–50%, tell closed 20–40%). The rebuilt cortex is a
regression on the task so far.

- **Streams self-organize.** In every self-organizing run the eye+ear column alone offered sounds
  (right 61–62% in training); eye+hand, ear+hand, hand+seen acts and the full column offered only
  mouse acts (right 42–60%). Nothing wired that split: it follows where each column's input is
  active.
- **But the self-organizing network is silent at test.** The eye+ear column's goals were right in
  training, yet nothing is said. Something between its goal and the voice breaks; not diagnosed.
- **The story's credit, confirmed.** Ending the basal ganglia's episode with the wait (so the
  story's rewards cannot reach the wait's acts) removed the LEFT habit on the book (LEFT released
  there 54–57 times in training, against 1,228). But then acting earned only its effort cost and
  the rare opening, and the network stopped acting: at test it released nothing (proposals of
  LEFT, CLICK and RIGHT, none released), and it was silent.
- **Play opened a book in 297 of 300 trials, but did not teach clicking on a book:** chosen
  clicks landed on a book 57% then 47% (seed 0) and 50% then 44% (seed 2), with books on half
  the places. (A 150-trial check had suggested 47% then 65%: noise.)
- **Imagining the click lands on the story's opening:** 10 of 10 (seed 0) and 164 of 309 (seed
  2) imagined clicks while telling went to the row holding this story's opening. The association
  of opening and text is there; going on from it is what fails (recall right 3–9%).

**What was built for the mouse:**
- **Touch** (`RT_TOUCH=1`): moving, the hand feels the motion (a vibration); at rest, the texture of
  the book under the cursor (each place's its own); an empty place feels like nothing.
- **Play** (`RT_PLAY`): trials before any teaching, no teacher and no story, books on half the
  places; the network moves (babbling half the steps) and clicks; a click on a book opens it,
  which rewards the act (contingent sensory change is rewarding in itself to an infant), and the
  page's first words come into view one after another, so the columns learn that after a click
  on a book words begin. A click on an empty place does nothing and is sensed as nothing.

**Defaults now** (as asked): the self-organizing columns and the teacher's own cursor
(`RT_SELFORG=0` for the shared layers, `RT_TEACHER_CURSOR=0` for the teacher on the network's
cursor). Touch and play stay off by default.

### Touch as a fingertip; why the self-organizing network was silent
**Touch, redesigned.** The skin's receptors report different things: slowly adapting ones report
pressure and edges, fast ones vibration and texture. So touch is a fingertip's window over the
place under the cursor and its two neighbours:
- a **vibration mode** for what is under the cursor (its kind: the desk's book, a shelf's book,
  nothing), 16 bits;
- an **edge** on the side where a neighbour differs (8 bits each side): where a thing begins and
  ends, and whether the cursor is inside it or at its border (containment, extent);
- while moving, the **motion** itself.
The parts are sparse and add up. In play, books lie on half the places, so modes and edges tell
"over a book" from "over nothing".

**Defaults now:** the self-organizing columns, the teacher's own cursor, touch, and 300 trials of
play (`RT_SELFORG=0`, `RT_TEACHER_CURSOR=0`, `RT_TOUCH=0`, `RT_PLAY=0` turn them off).

**Why the self-organizing network was silent at test** (traced):
- **Telling with the book shut, nothing fired.** The hippocampus's return (32 bits every step)
  was added to a column's input only when a sense was active; with the book shut and nothing yet
  said, no sense is. Now the return reaches every column always (the entorhinal route back to
  the cortex).
- **Reading aloud, it said the next word** (saw "tom", said "went"). The association fuses
  seeing and hearing a word (they share 28 of 32 cells), which is what memory wants, but then
  "say what is seen" (reading) and "what comes next" (listening) look alike, and listening
  dominates. Now every area's layer 5 offers goals, the senses' columns too (as every cortical
  area's layer 5 projects to the striatum and the motor system), and the relay learns which to
  trust where. The eye's own column can learn "say what is seen" (the visual word form area's
  route to phonology).

In a 300-trial check after both fixes it speaks again with the book shut, and the senses offer
sounds (ear right 61%, eye 54%; the eye+ear association 56%). Measuring on 4 seeds.

### How the self-organizing columns connect, and speech play
**To the hippocampus.** It reads the association of eye and ear (a word, seen or heard) and, for
acts, of hand and seen acts; not the full area or any column's layer 2/3. Its return now goes to
every association column, which stopped the silence but blurs the self-organization (columns
whose senses are idle offer sounds because recall gives them word input). Closer to the biology:
the entorhinal cortex reads the top association areas and returns to them, so the index should
read whichever top columns grow, and return to those.

**To the cerebellum.** Its mossy fibres carry the raw sound heard (an old shortcut) and the pons,
which now carries every column's layer 5 (as the pons relays all of the cortex's layer 5); its
teaching signal is the step's target; its output is a source of the plan and lies on the tufts.
Missing: the hand and touch, which reach the cerebellum directly from the spinal cord in the brain.

**Speech play** (`RT_SPEECH_PLAY`, default 600 sounds): before any teaching, the network babbles
(random commands to the vocal tract) and hears itself, as an infant's canonical babbling. The ear's
area and column hear each sound while the voice's area holds the command (its efference copy), so
an auditory–motor association area grows between them: the dorsal speech stream, self-organized.
Half the time a caregiver answers with the same sound (contingent replies shape babbling; Goldstein
et al. 2003). Then a word heard alone evokes, through the ear–voice association, the command that
says it: the mirror for speech, a learned route from hearing to saying. After 600 sounds: **63 of 63
words**. Not yet used in the tasks (the inverse model still says the plan).

### The defaults with speech play, imitation and recall where its pointers point
4 seeds, 1,500 training trials, first test table: self-organizing columns, the senses' goals,
the teacher's own cursor, touch, mouse play (300), speech play (600), the imitation route, recall
reinstated in the eye–ear column only.

| seed | read same / other | tell closed | silent | recall right | own book opened at test |
|---|---|---|---|---|---|
| 0 | 9.3% / 6.4% | 8.0% | 13–26% | 27.1% | 0% |
| 1 | 4.8% / 1.2% | 3.1% | 10–17% | 23.6% | 100% |
| 2 | 12.4% / 9.9% | 11.1% | 8–17% | 31.9% | 0% |
| 3 | 18.5% / 30.9% | 26.5% | 13–18% | 28.5% | 0% |

- **It speaks on every seed** (the self-organizing default had been silent).
- **Recall is the best yet** (24–32% of telling steps), with recall reinstated only where its
  pointers point.
- **Imagining the click lands on this story's opening** 84% of the time (704 of 818).
- **Both speech routes work:** of about 22,000 planned words, 11,800 went through the imitation
  route (the ear–voice association from speech play: 63 of 63 words) and 10,900 through the
  voice's inverse model.
- **Still poor:** reading (1–31%, against 40–50% with the shared layers), and reaching the
  book at test (one seed in four). Play did not teach clicking on a book (chosen clicks on a book
  near chance).

### Deconstructing the hard-codes
An audit of the driver for places where it decides or senses what the network should learn
(the full list is kept with the commits). Removed so far, each now the default:
1. **Recall without the imagine door.** With areas the hippocampus always completes (no cue, no
   request, no reinstated context copied from a side map); acetylcholine sets the mode.
2. **Proposals undecoded.** The mouse is offered every distinct proposal as it is; the motor area
   makes of it what it can (the old decoder kept only patterns near a known motion and labelled
   them).
3. **Imagery by the forward model.** An imagined act is the proposal run through the hand's
   forward model without acting (how it would feel) and the mirror (how it would look), not an
   assembly built from a decoded label.
4. **The story's opening seen**, not felt as if by the network's own hand.
5. **The teacher's cursor looks like another's,** and the mirror's evoked feel of the teacher's
   act is resonance in the hand area, no longer the cortex's target as if its own hand had moved.
6. **Practice speaking** in a quarter of training tasks.
7. **Touch reports what is there** (a book or nothing; no desk-vs-shelf label).
8. **Seen acts from visual change** (the motion stream: the area sees onsets between glimpses,
   as the brain's motion areas do, not the act's code handed over); the mirror and the
   act–feel association happen inside every step, from co-activity, with no driver calls.
9. **Salience instead of a fixed play reward:** the dopamine after a click is the eye column's
   surprise at what then comes into view (the colliculus's short-latency route; Redgrave and
   Gurney), fading as the change comes to be expected.

Left: the find trials' place cells (the driver's shelf index), recall routed to the eye–ear
column by the driver's knowledge of where pointers point, the fixed effort cost, and the reading
reward computed by comparing the plan with the teacher's word.

### The deconstructed version, the ablation, and the hippocampus integrated
**The first deconstructed version (items 1–7) went silent** (55–93% silent, recall 0.1–0.4%).
Switching off each change in turn (seeds 2 and 3):

| | read | tell closed | silent | recall right |
|---|---|---|---|---|
| all items | 0–4% | 0–3% | 55–93% | 0.1–0.4% |
| the imagine route back (`RT_HC_AUTO=0`) | 4–18% | 5–17% | 9–34% | 8.3–8.6% |
| no practice (`RT_PRACTICE=0`) | 0–2% | 0–7% | 65–86% | 0–0.1% |

The always-completing hippocampus was the cause. Its readout said why: three quarters of its
completions were in encoding mode, because acetylcholine rose at any mismatch (a new story
mismatches at almost every word) and decayed only when something was heard, so in silence it
never fell.

**Integrated, not reverted.** What the imagine route did, done by the always-completing
hippocampus itself:
- **Acetylcholine follows attention to the outside** (Hasselmo): hearing someone raises it;
  it decays a quarter every step; imagining (attention turned inward) halves it.
- **Hold, follow or jump.** Each step recall offers what comes next. Nothing new represented
  (a pause): the offer holds. The cortex says or hears what was offered: the sequence moves on.
  It represents something else: the content is completed, **what before when** (among rows whose
  content matches about as well as the best, the one whose context is closest, then the most
  recent; not the row just written, not one offered in the last 4 steps), and that jump
  **reinstates the row's context** (retrieved context, as the temporal context model has it);
  following, the context blends a quarter.
- Imagining is the cortex representing the act; there is no separate door.

**The full defaults** (integrated hippocampus, multi-area index, the motion stream, salience,
and everything before), 4 seeds:

| seed | read same / other | tell same / other | tell closed | silent | recall right |
|---|---|---|---|---|---|
| 0 | 18.5% / 20.7% | 22.1% / 22.6% | 14.9% | 13–18% | 14.9% |
| 1 | 6.8% / 11.7% | 2.4% / 3.7% | 4.5% | 27–65% | 13.1% |
| 2 | 3.1% / 3.4% | 27.1% / 33.0% | 28.7% | 11–63% | 15.0% |
| 3 | 15.2% / 2.9% | 24.6% / 14.4% | 12.4% | 22–57% | 15.2% |

- **Telling with the book closed works on every seed** (4.5–28.7%), with recall right 13–15% on
  all four: the most even result yet, with no imagine door, no decoded labels, no driver-routed
  recall, no hand-labelled acts.
- **Still poor:** reaching the book at test (no seed opened it itself), so reading leans on
  hearing; reading 3–21%.

### Imagining, and where it happens
Imagery reuses the cortex that perceives or acts in that modality, run top-down: visual imagery
reactivates visual cortex down to V1 for fine detail (Kosslyn; Pearson), inner speech the
auditory and speech-motor areas, motor imagery premotor, supplementary motor and parietal cortex
(the plan runs, the output is inhibited). Laminar recordings put imagery and expectation in the
feedback layers (the tufts), not layer 4 (Muckli; Kok; Lawrence). Scenes are constructed by the
hippocampus and the default-mode network (amnesic patients cannot imagine new scenes; Hassabis
and Maguire; remembering and imagining the future share a network, Schacter and Addis); the
prefrontal cortex holds the goal; the basal ganglia and thalamus gate what is run internally;
reality monitoring (imagined or real) involves the prefrontal cortex.

Here imagining is narrow: one striatal channel picks a mouse proposal, run through the hand's
forward model and the mirror, as a cue to the hippocampus; nothing reaches the sensory columns'
tufts, and there is no inner speech or visual imagery. A general version: any column's goal can
be released to its motor area (do) or held from the body and run around the loop (imagine), the
basal ganglia choosing; the motor area's forward model predicts what it would produce, and that
reaches the matching sensory column on its tuft; recall arrives on the tufts too, as feedback
(not as if sensed); feedforward through layer 4 means real, tuft-only means imagined.

## Next
- **Imagining as one operation for every modality** (do or imagine, forward models onto the
  tufts, recall as feedback).
- **The mouse at test:** no seed opens the book itself; play does not teach clicking on a book.
- **Place without place cells:** the eye should see the scene while a shelf book is read, so the
  multi-area index carries "where".
- **Whether the story heard should cue telling** without the instruction.
