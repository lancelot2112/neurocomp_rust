# 108: A world and a mind, with every reward the mind's own

**Question.** Can the network learn when to listen, when to speak and when to reach, when nobody
tells it what phase it is in or what was right? Here every reward comes from its own innate
valuations of what it senses, and nothing outside hands it credit.

**Why.** In experiment 107 the driver did more than set up the task:
- it decided what a trial was and when the network listened or spoke;
- it gave a reward when a planned word matched the teacher's;
- it ended episodes and switched learning off at test.

Each of those was a decision the network should make or a signal it should not get. Training
plateaued: past 1,000 trials words planned right stayed at about 49%, and telling with the book
closed at about 5%. That points at the setup, not the amount of training.

## The split

**The world** (`src/program/world.rs`).
- **The body.** A vocal tract and a mouse carry out commands. A command that matches no motor
  pattern does nothing.
- **The senses.** Each step reports:
  - the eye: the screen, meaning the cursors, an open book's word, and what moved;
  - the ear: its own voice, someone else's, or silence;
  - touch: the hand's motion, a vibration over a book, an edge where a neighbour differs;
  - presence: someone there and attending, and whether they smile.
- **The people respond only to what they can observe.**
  - A caregiver, there half the time in speech play, answers a sound said aloud with the same
    sound, once the child pauses.
  - A teacher runs the lesson:
    - opens their book on screen and reads it aloud;
    - pauses, then asks "now read it" or "now tell it";
    - waits, showing how to open the book on their own copy;
    - then does the task or listens while the child does it;
    - listening, smiles at a right word said aloud and says the right word after a wrong one;
    - if the child chimes in while the teacher is talking, a right next word gets a smile.
- **No rewards.** A smile is a sight like any other.
- **An event log** records each step (what was said and by whom, what was wanted) for an observer
  outside both.

**The genome** (`Body`) gives the mind:
- the body's motor primitives: the vocal tract's syllables and the hand's moves, which brainstem
  pattern generators can produce spontaneously;
- the presence sense's innate tuning to a smile.

**The mind** (`src/program/mind.rs`, with the always-completing index in `src/program/index_hc.rs`)
runs one pipeline every step, `step(senses) -> acts`:
1. **Learn what came.** The columns learn from their last state what the effectors' senses
   received: the hand's touch if it acted, else the ear. The motor areas learn each act's result,
   with no babbling phase handed the world's answers.
2. **One hippocampal step over the whole cortical state.** The input is every sensing area's
   cells, sensed or imagined; nobody tells the hippocampus what was heard.
   - An input that changed is attention outward: acetylcholine rises and the state is stored.
   - The voice's efference copy marks a sound as its own: it is represented, not stored.
   - A new state that came from neither is attention turned inward: acetylcholine falls.
   - An unchanged state is nothing new, so the offer holds.
3. **Sense.** Every column, the motion stream, the mirror, the association columns, and the
   cerebellum. The relay's mix is now only the cortex's expectation (for surprise); nothing routes
   it to the voice.
4. **Value what came.** These innate valuations are the only dopamine:
   - *own effect:* after its own act, as surprising as the senses find what came;
   - *response:* someone present, and a sound from outside within two steps of its own sound
     said aloud;
   - *smile:* an innately recognised pattern in the presence sense;
   - *effort:* every act costs a little.
5. **Loops grow with the cortex** (Alexander, DeLong & Strick 1986).
   - **Column loops.** Every column with a layer 5, sensory or association, gets its own loop
     through the basal ganglia when it is created. The striatum sees each column's activity as
     its own group.
   - **A column's goal is learned.** Its layer 5 offers the fired cell whose output has been
     worth most when followed. That worth is learned from the dopamine that follows: the TD error
     reaches the goal cells followed in the last four steps.
   - **Taking a goal in.** The column's loop takes its goal in or not. An association column's
     loop can also gate its content into working memory (PBWM's stripes).
   - **Motor loops.** Each motor area's loop chooses among the goals taken in that it can plan,
     plus its own spontaneously active primitive (one per step). Its motor gate then releases the
     choice to the body or withholds it. Withheld, it is imagined: the forward model's result
     comes back top-down at the next step, to the cortex and to the hippocampus alike.

The mind never knows it is being tested and never stops learning. `examples/lessons.rs` is the
curriculum (speech play, mouse play, lessons) and the observer.

## Results

### The first version: silent in lessons
Before loops grew with the cortex, the relay's mixed plan was offered straight to the voice. The
mind (seeds 0 and 2) went through 600 steps of speech play, 300 mouse-play episodes, 1,000 lessons
and 200 test lessons.

| | seed 0 | seed 2 |
|---|---|---|
| speech play, said aloud (first half, second half, of 300 steps) | 123, 185 | 193, 258 |
| mouse play, clicks on a book | 11 of 35, then 0 of 1 | 379 of 820, then 400 of 724 |
| practice, silent | 99.9–100% | 99.7–100% |
| test, words right | 0.0% | 0.2–0.7% |
| test, silent | 100% | 71–79% |
| test, book opened while waiting | 0 | 0 |

- **Silent in the lessons.** Babbling held up in play, but practice was almost entirely silent
  and the test right rate was at chance.
- **Clicks were not tied to the book.** The rate of clicks landing on a book was about the share
  of places holding one.

That led to the redesign above: loops per column, goals of the columns' own, one hippocampal
step. Then the biology was checked for how cue → contingent reward is learned (next section).

### Conditioning probes
The question was whether the setup can map an external cue to a contingent reward at all, before
relying on it in lessons. `examples/conditioning.rs` runs a fresh mind in trials of 10 steps, with
a caregiver present throughout and a smile as the only outcome. Seed 0.
- **Pavlovian:** a tone (CS+, half the trials) is followed by a smile two steps later; another tone
  (CS−) by nothing.
- **Instrumental:** a light on for three steps (half the trials); a click while it is on, or the
  step after, gets a smile.
- **Delayed:** a light for one step; a click three to five steps later gets a smile.

**What was built from the biology.** Every part has a switch, so each can be ablated.
- **The hippocampus at the association level.** Its input and output are the association columns'
  codes (pairs, and the full area), as the entorhinal cortex's inputs are the top of the hierarchy
  (perirhinal, parahippocampal, prefrontal; Lavenex & Amaral 2000). Recall reaches the senses only
  through the columns' layer 4 back-projections, down the hierarchy (`MIND_HC=senses`: the old
  sensory-level index).
- **Topographic striatal input.** Each loop sees its own cortical territory (Alexander, DeLong &
  Strick; Haber's spirals). Before, every choice's 64-bit context was shared among about 20 groups,
  about 3 bits per column. The critic sees the limbic territory: the amygdala, recall, working
  memory and presence.
- **An amygdala.** A fast learner of cue value: Rescorla–Wagner on a time-tagged trace of onsets,
  with associability after Pearce–Hall and Mackintosh. It learns from outcomes only (a smile, a
  response), not from the surprise of the mind's own acts. That surprise is agency, not value
  (Redgrave & Gurney); it still reaches the critic as intrinsic drive. Its attended trace goes to
  the striatum.
- **Learned cholinergic pauses** (Aosaki et al. 1994; Morris et al. 2004): cues that came one or
  two steps before large dopamine errors learn to pause the interneurons, which opens actor
  plasticity.
- **A contingency detector** (Watson): a reply pays only as far as sounds from outside are likelier
  just after its own sound than otherwise. Before, the teacher's next word counted as a reply to
  anything the child said.
- **The motor gate sees the amygdala's value of the act's imagined outcome** (vicarious trial and
  error: Johnson & Redish 2007).

**Pavlovian results**

| version (300 trials) | value at the tone, CS+ / CS− (last 100 trials) | dopamine at the smile (first, last 100) | amygdala's prediction at the smile, CS+ / CS− (last 100) |
|---|---|---|---|
| critic sees everything; every valuation into the amygdala | +0.45 / +0.24 | +0.95, +0.72 | – |
| critic on the limbic territory, pauses balanced | +1.40 / +1.32 | +0.79, +0.85 | +0.00 / +0.00 |
| the same, own-effect and response valuations off | **+0.68 / −0.28** | **+0.25, −0.18** | **+0.46 / +0.13** |
| every valuation on, the amygdala learning from outcomes only | +0.40 / +0.41 | +0.70, +0.86 | +0.00 / +0.00 |
| the same, the amygdala summing cue values (Rescorla–Wagner) | +0.86 / +0.88 | +0.79, +0.84 | +0.05 / +0.02 |

- **It can work.** Without the own-effect and response valuations, the critic tells CS+ from CS−.
  Dopamine moves from the smile to the tone: at the tone +0.24 to +0.41 for CS+ and negative for
  CS−; at the smile it falls from +0.25 to −0.18. The amygdala learns CS+.
- **The mind's own activity masks it.** With those valuations on, the mind acts on about half the
  steps or more (0.9 acts per step in the Pavlovian probe), and every act's effect is a stream of
  rewards no cue predicts. The critic's value rises everywhere instead of at the tone. The
  amygdala's prediction stays near 0 even when it learns from outcomes only and sums cue values.
- **Attention to cues collapses.** Its attended trace fell to about 4–5 features per step, so the
  tone is barely seen.
- **A first version paused almost always** (on 2,405 of 3,000 steps), which made every dopamine
  error train every recent choice: wrong, and slow (130 ms per step). Learning pauses only from cues
  just before large errors, unlearned at the same rate, brought it to about 300 pauses.

**Instrumental results** (clicks in the window per trial, light against no light, per 100 trials)

| version | 0–100 | 100–200 | 200–300 | 300–400 |
|---|---|---|---|---|
| every valuation on | 0.08 / 0.10 | 0.62 / 0.56 | 0.40 / 0.21 | 0.02 / 0.00 |
| own-effect and response off | 0.04 / 0.18 | 0.62 / 0.56 | 0.72 / 0.84 | 0.18 / 0.18 |
| every valuation on, the amygdala summing | 0.06 / 0.14 | 0.86 / 0.86 | 0.26 / 0.49 | 0.00 / 0.00 |
| delayed (light for one step), every valuation on, summing | 0.18 / 0.12 | 0.04 / 0.12 | 0.11 / 0.16 | 0.00 / 0.00 |

- **No version learns to click for the light.** Clicking rises (smiles come, half the trials have
  the light), then collapses to nothing. Clicks without the light are not punished, only unpaid
  and costly, so the effort cost wins.
- **The critic learns the light when the mind is quiet,** but the hand's loop does not use it: at
  the end, value +0.67 at the light against −0.44 without, with clicks 0.18 against 0.18.

**What this says.**
- **Learning what a cue is worth works** in isolation. The parts that make it work are the critic
  on its own (limbic) territory, the amygdala and the time-tagged trace.
- **The mind's own effects drown it.** A reward for every act's surprise is a constant stream no
  cue predicts. Biologically the short-latency response to self-caused sensory change habituates
  per stimulus and is thought to train which acts have effects (agency), not what things are worth.
  Here it goes straight into the same critic as the smile. The next step is to separate the two:
  agency trains the motor loops' choice of acts with effects, value trains the critic, or the
  own-effect response habituates per act and stimulus.
- **Acting on a cue does not follow from knowing its value.** The hand loop sees the light only
  through the amygdala's attended trace, and the attention collapses. A loop also has no way to use
  "the critic now expects reward" except as a dopamine error after the fact. In the biology the
  amygdala's cue value invigorates and biases action directly (Pavlovian–instrumental transfer,
  through the central amygdala and the ventral striatum).

### The amygdala in bit cells, agency apart from value, cue value driving action
Three changes, from the probes above.
- **The amygdala in bit cells,** as the striatum is: populations of 16-bit mask cells that fire on
  12 (AND and popcount).
  - The age-tagged cue trace is hashed into a 4,096-bit context.
  - Appetitive and aversive value cells; the value is appetitive firing minus aversive, at 1/16
    each.
  - Learning follows the shared Rescorla–Wagner error: recruit cells on the trace for a positive
    error, and remove firing appetitive cells for a negative one (extinction).
  - Attention is a bit vector over features: set by surprise, by value or by novelty; a random
    1/32 clears each step.
  - Pause cells are masks over the cues one or two steps before a large dopamine error.
- **Agency apart from value.** Its own effect is now the forward model's error on its act. That
  fades per act and outcome as the model learns them, where before it was the senses' surprise
  after any act. It trains only the motor loops' last choices (`MaskStriatum::reinforce_last`); it
  no longer goes to the critic or the amygdala.
- **Pavlovian–instrumental transfer.** The amygdala's value of the cues now lowers the motor loops'
  release brake, up to two steps, or raises it for cues worth less than nothing.

**Results.** Seed 0, every valuation on.

| probe | 0–100 | 100–200 | 200–300 | 300–400 |
|---|---|---|---|---|
| Pavlovian: amygdala's prediction at the smile, CS+ / CS− | +0.20 / +0.00 | +0.64 / +0.01 | **+0.91 / +0.01** | |
| Pavlovian: dopamine at the smile | +0.92 | +0.86 | **+0.24** | |
| Pavlovian: critic's value at the tone, CS+ / CS− | −0.05 / +0.02 | +0.25 / +0.23 | +0.64 / +1.79 | |
| instrumental: clicks per trial in the window, light / none | 0.02 / 0.08 | 0.44 / 0.24 | 0.02 / 0.00 | 0.00 / 0.00 |
| delayed: clicks per trial in the window, light / none | 0.35 / 0.29 | 0.70 / 0.78 | 1.05 / 1.14 | 1.16 / 1.05 |
| delayed: smiles per light trial | 0.31 | 0.56 | 0.74 | 0.71 |

- **The amygdala now learns the tone with the mind fully active.** Before, its prediction fell to
  0. It discriminates sharply: +0.91 for CS+, +0.01 for CS−, with 55 appetitive cells and none
  aversive.
- **The smile becomes predicted.** Dopamine at the smile fell from +0.92 to +0.24.
- **The critic's value at the tone itself is not right yet.** CS− reads higher than CS+ in the last
  block. The critic sees the amygdala's attended features but learns over the whole limbic context,
  and at the tone step the tone is age 0, which predicts nothing on its own until a step later.
  Dopamine at the CS+ tone was +0.11 against −0.20 at CS− in the middle block, but −0.51 in the
  last. So the transfer of dopamine to the tone is not clean.
- **Instrumental: still no.** Clicking rises in the second block (0.44 with the light against 0.24
  without) and then stops altogether. The mind acted on only 0.11 of steps in this probe. Effort
  costs every click; the smile pays only light trials; the agency signal fades as the forward model
  learns that a click on nothing does nothing.
- **Delayed: clicking is learned, but not for the light.** Smiles rose from 0.31 to 0.71 per light
  trial, and clicks stayed equally frequent without the light (1.05–1.16 per trial). In this probe
  clicking at a steady rate pays; the light adds nothing the mind uses.
- **Pauses:** 167–1,127 steps of 3,000–4,000, learned from cues before large errors.

**What this says.**
- **Knowing what a cue is worth now works with the mind fully active,** in the amygdala's bit cells,
  once the agency signal is kept out of value.
- **Acting on a cue still does not.** Transfer lowers the brake whenever the light is worth
  something. But the light becomes worth something only through smiles that follow clicks, and
  clicking dies out in the instrumental probe before that happens. In the delayed probe, steady
  clicking is the simpler policy that pays.
- **The motor loop's view has the light only through the amygdala's attended trace.** That is a
  few features among the hand's own column and the association columns over it, so the loop's
  choice barely depends on the light.

**Next.**
1. **The critic should read the amygdala's prediction** (its value, as cells), not only its
   attended features. In the biology the basolateral amygdala drives the ventral striatum and the
   dopamine neurons at the cue.
2. **The motor loops' view should include the cue's value cells firing,** so "release when a valued
   cue is here" is a context the loop can learn, beyond the brake's general lowering.
3. **Avoid the click die-out:** a floor on spontaneous primitives, or effort that scales with
   fatigue rather than a flat cost per act. Then repeat with more seeds.

### No cost to act; value cells to the motor loops, then to the critic
Two steps, run one after the other (seed 0, every valuation on):
- **v8.** Acting costs nothing: the flat effort cost per act is gone, since it was killing exploration
  before anything could be learned. The motor loops' view also includes the amygdala's value cells
  firing on the cues now: "a valued cue is here" becomes a context the hand's loop can learn.
- **v9.** The critic's view also includes those value cells, as the basolateral amygdala drives
  the ventral striatum and, through it, the dopamine neurons at the cue (`MIND_AMY_CRITIC=0` turns
  it off).

| probe | measure | v8: 0–100 | 100–200 | 200–300 | 300–400 | v9: 0–100 | 100–200 | 200–300 | 300–400 |
|---|---|---|---|---|---|---|---|---|---|
| Pavlovian | critic's value at the tone, CS+ / CS− | +0.19 / +0.20 | +0.54 / +0.30 | **+0.83 / +0.30** | | +0.30 / +0.09 | +0.02 / +0.04 | +0.38 / +0.23 | |
| Pavlovian | dopamine at the tone, CS+ / CS− | −0.01 / +0.01 | +0.10 / +0.10 | **+0.49 / +0.01** | | +0.16 / −0.01 | +0.05 / +0.00 | +0.00 / +0.12 | |
| Pavlovian | dopamine at the smile | +0.89 | +0.86 | **+0.50** | | +0.56 | +0.81 | +0.60 | |
| Pavlovian | amygdala's prediction at the smile, CS+ / CS− | +0.14 / +0.00 | +0.32 / +0.02 | +0.95 / +0.00 | | +0.71 / +0.01 | +0.29 / +0.02 | +0.66 / +0.01 | |
| instrumental | clicks per trial in the window, light / none | 0.12 / 0.14 | 0.86 / 0.68 | 1.11 / 1.05 | 0.30 / 0.27 | 0.10 / 0.08 | 0.14 / 0.14 | 0.56 / 0.91 | 0.07 / 0.11 |
| instrumental | smiles per light trial | 0.08 | 0.62 | 0.79 | 0.20 | 0.10 | 0.14 | 0.49 | 0.05 |
| delayed | clicks per trial in the window, light / none | 0.02 / 0.18 | 0.08 / 0.16 | 0.42 / 0.26 | 0.68 / 0.84 | 0.16 / 0.18 | 1.04 / 0.80 | 0.49 / 0.58 | 0.45 / 0.66 |

- **v8 gives the first clean Pavlovian transfer with the mind fully active.**
  - Dopamine at the tone rose to +0.49 for CS+ against +0.01 for CS−.
  - Dopamine at the smile fell from +0.89 to +0.50.
  - The critic's value at the tone separated the right way, +0.83 against +0.30.
  - The amygdala's prediction was +0.95 against +0.00.
- **Without the cost, clicking no longer dies out early.** It rises to about one click per trial in
  the window, and smiles reach 0.79 per light trial in v8. But it is not tied to the light:
  clicks with and without the light stay about equal. The last block falls again (0.30 / 0.27).
- **v9 (value cells to the critic) did not help on this seed.** The Pavlovian transfer was weaker
  and less stable than in v8: dopamine at the CS+ tone ended at +0.00, the critic's value at
  +0.38 against +0.23. The amygdala's own prediction also wavered (+0.71, +0.29, +0.66), with more
  cells (239 appetitive against 68) and more onsets per step (28 against 20, as the mind acted on
  0.61 of steps). Clicking still did not follow the light: in the third block more without it
  (0.91) than with it (0.56).
- **This is one seed per version,** and a change in how much the mind acts changes everything
  downstream (onsets, attention, value cells). Differences of this size between v8 and v9 need
  more seeds before they mean anything.

**Where the instrumental failure sits.** The parts that learn a cue's value now work. What fails
is turning "the light is worth something" into "click when the light is on". These still hold:
- The smile pays only clicks during the light. Clicks without it get nothing, but cost nothing
  either, so the loop has little reason to hold back.
- The agency signal pays the first clicks whatever the light: their effect is new.
- The hand's choice is one go/no-go gate over a context in which the light is a few bits.

**Next.**
- More seeds for v8 and v9 (with `MIND_AMY_CRITIC=0`), to tell noise from effect.
- Then the instrumental contingency itself: a dip when a smile expected after a click does not
  come would teach "not without the light". That dip needs the critic to predict the smile after
  a click in light trials only, which is the conjunction it has not yet learned.

### Three seeds, four versions; the efference copy and the orbitofrontal split
- **The versions.** Seeds 0–2, every valuation on, 300 Pavlovian trials and 400 instrumental.
  - **v8:** no cost to act; the value cells reach the motor loops.
  - **v9:** v8, and the value cells reach the critic.
  - **v10:** v9, and the efference copy of the act just done reaches the amygdala's trace and the
    critic.
  - **v11:** the split. The amygdala sees cues only. An orbitofrontal/prelimbic learner (the same
    bit cells) sees cues and the efference copy, and its values reach the critic and the motor
    loops. The efference copy itself reaches only the motor loops, as motor cortex's collaterals
    reach the dorsal striatum.
- **Why the conjunction is needed.** The dip is the ordinary TD error. What was missing was an
  expectation that could fail: a click on nothing has no sensory effect, so without an efference
  copy no learner knew a click had happened.

**Pavlovian, last 100 trials**

| version | seed | dopamine at the tone, CS+ / CS− | dopamine at the smile | amygdala's prediction, CS+ / CS− |
|---|---|---|---|---|
| v8 | 0, 1, 2 | +0.15 / −0.11; −0.03 / −0.05; −0.16 / −0.07 | +0.91, +0.80, +0.90 | +0.93 / +0.01; +0.40 / +0.00; +0.01 / +0.00 |
| v9 | 0, 1, 2 | +0.12 / −0.01; −0.06 / −0.01; −0.03 / −0.03 | +0.54, +0.86, +0.87 | +0.31 / +0.01; +0.00 / +0.00; +0.00 / +0.00 |
| v10 | 0, 1, 2 | +0.06 / −0.05; +0.08 / −0.19; −0.19 / +0.17 | +0.95, **−0.09, −0.11** | +0.31 / +0.01; **+1.00 / +0.00; +1.00 / +0.00** |
| v11 | 0, 1, 2 | **+0.03 / −0.26; +0.07 / −0.16; +0.01 / −0.40** | +0.47, +0.67, +0.57 | **+0.91 / +0.02; +0.93 / +0.01; +0.83 / +0.01** |

- **v8's clean seed-0 transfer (dopamine +0.49 at CS+) did not replicate.** The same seed gave
  +0.15 in the sweep (different build, same switches), and seeds 1 and 2 none.
- **v9 is worse:** the amygdala learned nothing on two seeds of three.
- **v10 makes the smile fully predicted on two seeds** (dopamine at the smile about −0.1), but the
  critic's dopamine at the tone does not separate.
- **v11 is the most consistent.**
  - The amygdala learns CS+ on all three seeds (+0.83 to +0.93 against +0.01 to +0.02).
  - The critic now dips at CS− on all three (−0.16 to −0.40), with CS+ near 0, so the dopamine at
    the tone discriminates.
  - The smile is about half predicted (+0.47 to +0.67).
  - The orbitofrontal learner grows 36–350 cells and no aversive ones.

**Instrumental, clicks per trial in the window, light / none, by block of 100**

| version | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| v8 | 0.16/0.16, 0.52/0.72, 0.93/0.91, 0.23/0.48 | 0.09/0.09, 0.28/0.21, 0.69/0.53, 0.00/0.00 | 0.26/0.25, 0.59/0.78, 0.00/0.00, 0.00/0.00 |
| v9 | 0.08/0.12, 0.38/0.56, 0.68/0.84, 0.04/0.02 | 0.06/0.09, 0.08/0.04, 0.00/0.02, 0.00/0.00 | 0.07/0.12, 0.69/0.61, 0.89/0.80, 0.42/0.79 |
| v10 | 0.10/0.10, **0.94/0.66, 0.96/0.53**, 0.50/0.57 | 0.15/0.11, 0.04/0.06, 0.00/0.00, 0.00/0.00 | 0.21/0.14, 0.71/0.86, 0.86/0.98, 0.19/0.29 |
| v11 | 0.10/0.12, 0.06/0.14, 0.00/0.00, 0.00/0.00 | 0.13/0.32, 0.11/0.40, 0.29/0.16, 0.75/0.62 | 0.02/0.09, 0.16/0.12, 0.23/0.38, 0.56/0.79 |

- **No version learns to click for the light.** One run (v10, seed 0) clicked clearly more with
  the light for two blocks (0.96 against 0.53), then lost it.
- **The common failure is that clicking dies out,** in about half the runs, by block 3 or 4, even
  with no cost to act.

**What is missing.** Four things, from the biology.
1. **A reason to hold back.** With no cost, clicking always collects every smile there is: the
   best policy does not depend on the light. Animals discriminate because responding costs effort
   and time. The flat effort cost was removed because, without an exploration floor, it killed
   acting; the problem was the pair, not the cost.
2. **Exploration that rises when reward is scarce.** The agency signal fades once clicking on
   nothing is predictable. Curiosity noise fades with learning progress. So acting can stop, and
   then nothing more is learned. In the brain, tonic dopamine tracks the average reward rate (the
   opportunity cost of time; Niv et al. 2007). The locus coeruleus shifts to a tonic, exploratory
   mode when utility falls (Aston-Jones & Cohen 2005).
3. **Withholding as a learned choice.** A gate that withholds records no choice, so no dopamine
   ever credits "not clicking here". The indirect pathway learns no-go from dips in the brain
   (Frank). Here a dip after a click without the light now has an expectation to fall from (the
   orbitofrontal learner), but the actor has no record of the withholds to credit.
4. **The light is faint in the actor's view.** It is a few bits of a 64-bit context shared among
   6–8 groups, and only as an onset trace that fades over 4 steps. Nothing says "the light is on"
   while it stays on. A predictive cue gains attention gain in the brain, and the striatum gets
   sustained sensory context.

**Next.**
1. A slow average reward rate (tonic dopamine and noradrenaline) that sets both the exploration
   noise and a price per act (opportunity cost).
2. Withholds recorded as choices.
3. Sustained sensory state, and a learned gain per input group, in the striatum's context.
4. Then 3–5 seeds and more trials.

### What was missing, tried in order; coverage; affective touch; why acting dies out
**What was added, in order.** Each has a switch.
1. **A tonic level** (`MIND_TONIC`): fast and slow averages of the outcomes. Where the recent ones
   fall below the long-run ones, or nothing has paid lately, the striatum's choices get extra noise
   (exploration rises when utility falls; Aston-Jones & Cohen 2005; tonic dopamine, Niv et al.
   2007). Acting costs a little effort again (1/32 per act, `MIND_EFFORT`).
2. **Withholding as a choice** (`MIND_NOGO`): a candidate a loop held back is recorded, so the
   dopamine that follows can train it. The first version grew no-go on a burst ("holding back was
   right"). That is the reverse of D2 plasticity. **Corrected:** only the held-back candidate's
   no-go (D2) cells change; a dip potentiates them, a burst depresses them; the go cells were
   silent and do not change (Shen et al. 2008; Frank 2005).
3. **Sustained sensory state and salience gain** (`MIND_SALIENCE`): the motor loops see every
   sense's current code, not only onsets. A group the amygdala or the orbitofrontal learner values
   is repeated in the striatum's context, up to four times, so a predictive cue claims more of the
   64 bits.

**Why the light was not seen.** The eye area's layer 4 codes the whole frame with its 32
most-driven cells, and recruits only when fewer than 16 fire. The cursor is always in view and
already has its cells. So when the light came on, the eye's code kept 29.5–32 of its 32 cells:
the light had no cells of its own. Only the seen-acts area (onsets) coded it, for one step. The
tone, heard alone, always had a clean code; that is why the Pavlovian probe worked and the
instrumental one did not.
- **The fix: coverage in the senses' layer 4** (`MIND_COVERAGE`). Input left uncovered by the
  fired cells' synapses (a quarter or more) recruits cells on that part alone, and the winners grow
  with the input (k to 4k), as divisive normalization lets more input drive more cells. The light
  now adds about 30 cells: the eye's code goes from 32 to 61–65, keeping the cursor's 30–32.
- **Later:** a retinotopic eye, so things in different places have separate cells and local
  competition.

**Affective touch instead of a smile.** The smile was an innate detector reading a pattern in the
presence sense: a separate path.
- **It is now a sense of its own.** A slow, gentle stroke on the skin (the stimulus C-tactile
  afferents respond to) activates a run of neighbouring receptors on the skin's map.
- **It goes through its own area and column,** like any sense.
- **Its value is the receptor line's own activity:** a labelled line, as sweet taste's or pain's.
- **Only another person can give it.** This body's one effector moves a mouse and cannot reach its
  own skin. In the brain, touch one makes oneself is attenuated by the efference copy (one cannot
  tickle oneself, Blakemore et al. 1998), though some hedonic touch is not (genital sensation), so
  a body that can reach its own skin would need that attenuation.

**Results, instrumental** (600 trials; clicks per trial in the window, light / none, by block of
100)

| version | seed | 0–100 | 100–200 | 200–300 | 300–400 | 400–500 | 500–600 |
|---|---|---|---|---|---|---|---|
| tonic level | 0 | 0.10/0.18 | 0.22/0.24 | 0.53/0.40 | 0.61/0.89 | 0.76/0.64 | 0.12/0.17 |
| tonic level | 1 | 0.15/0.15 | 0.28/0.34 | 0.67/1.11 | 0.80/0.87 | 0.30/1.16 | 0.22/1.13 |
| tonic level | 2 | 0.30/0.35 | 0.84/0.61 | 0.45/0.38 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 |
| + withholding (first rule) | 0–2 | | | | | | dies out on all three by block 4–5 |
| + salience | 0 | 0.06/0.04 | 0.06/0.02 | 0.00/0.00 | … | | 0.00/0.00 |
| + salience | 1 | 0.15/0.06 | **0.70/0.00** | **0.91/0.04** | **0.96/0.02** | **0.88/0.00** | 0.41/0.07 |
| + salience | 2 | 0.07/0.12 | 0.24/0.24 | 0.86/0.57 | 0.76/0.76 | 0.48/0.26 | 0.00/0.00 |
| + coverage, affective touch | 0 | 0.20/0.09 | 0.41/0.00 | **0.68/0.00** | 0.06/0.00 | 0.00/0.00 | 0.10/0.12 |
| + coverage, affective touch | 1 | 0.15/0.24 | 0.25/0.13 | **0.79/0.00** | 0.66/0.11 | **0.94/0.00** | **0.93/0.03** |
| + coverage, affective touch | 2 | 0.07/0.13 | 0.38/0.15 | 0.78/0.51 | 0.88/0.79 | 0.91/0.81 | 0.92/0.96 |
| + coverage, affective touch | 3 | 0.13/0.16 | 0.10/0.04 | 0.00/0.00 | 0.02/0.00 | 0.00/0.00 | 0.15/0.00 |
| + coverage, affective touch | 4 | 0.23/0.33 | 0.29/0.69 | 0.56/0.64 | 0.35/0.46 | 0.21/0.38 | 0.83/0.28 |
| + the withhold rule corrected | 0 | 0.23/0.15 | 0.49/0.49 | 0.46/0.42 | 0.42/0.56 | 0.49/0.21 | **0.80/0.00** |
| + the withhold rule corrected | 1 | 0.04/0.15 | 0.06/0.11 | 0.02/0.00 | 0.02/0.00 | 0.00/0.00 | 0.00/0.00 |
| + the withhold rule corrected | 2 | 0.11/0.09 | 0.00/0.02 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 |
| + the withhold rule corrected | 3 | 0.12/0.12 | 0.08/0.06 | 0.17/0.02 | **0.80/0.00** | 0.54/0.33 | 0.23/0.27 |
| + the withhold rule corrected | 4 | 0.08/0.23 | **0.52/0.11** | **0.75/0.16** | **1.05/0.00** | 0.52/0.28 | 0.00/0.00 |

- **The mind now learns to click for the light.** Clicks with the light and almost none without
  (0.70–1.05 against 0.00–0.04), on one seed of three with salience, and on three or four of five
  seeds with coverage, for at least a block.
- **It does not keep it.** Only one run held it to the end (coverage, seed 1: 0.94 and 0.93
  against 0.00 and 0.03). The others lose clicking altogether, or lose the discrimination.

**Results, Pavlovian** (with coverage and affective touch; last 100 trials)
- **Seeds 0 and 2: clean.**
  - Dopamine at the tone: CS+ +0.45 and +0.22, against CS− −0.14 and −0.08.
  - Dopamine at the touch: fell to +0.14 and +0.07.
  - The amygdala's prediction: 1.00 against 0.00.
- **Seed 1 failed.** It acted on 94% of steps, and the amygdala saw only 1.7 onsets per step.

**Why acting dies out.** The hand loop's go and no-go cells and the dopamine, per block.
- **Under the first withhold rule, it was not plain depression.** No-go led go by 10–25% on a seed
  that learned and on one that died alike. On the seed that died, the hand's gate froze (137 go,
  46 no-go cells from block 2 on), because once the loop offered nothing the gate had no choices to
  learn from. The dopamine grew tenfold while the mind barely acted.
- **Under the corrected rule, it is depression in the indirect pathway: no-go runs away.**

  | seed | outcome | hand loop go / no-go cells at the end | dopamine per step (dips, bursts) at the worst |
  |---|---|---|---|
  | 1 | died | 92 / **2,048** (the cap) | −0.23, +0.24 |
  | 2 | died | 105 / 617 | −0.06, +0.06 |
  | 4 | learned, then died | 822 / **2,048** | −0.21, +0.22 |
  | 0 | learned at the end | 1,569 / 1,582 | −0.11, +0.12 |

  Two things combine.
  - **The critic makes large errors when nothing is happening.** Its value is large (0.6 to over
    1.0) and its state keeps changing (the hippocampus's offers, working memory, the attended trace,
    effort), so V(s′) − V(s) swings by 0.1–0.2 per step with no outcome at all. A quiet mind gets
    the largest dips.
  - **The withhold rule is asymmetric.** A dip recruits new no-go cells for every candidate held
    back on every loop, a few per step. A burst only moves the no-go cells that fired. Withholds
    are most of all choices. So each spurious dip adds no-go cells, and bursts cannot take them
    away at the same rate. The no-go pool fills to its cap (2,048) and the hand cannot act again.

**Next.**
1. **The critic's noise:** a quiet step should have no error.
   - The critic learns the value of states it keeps visiting with nothing happening (it learns only
     the latest state, TD(0); it should reach zero error there).
   - Or keep value off states that predict nothing (the limbic view's noise: recall and working
     memory changing).
2. **Symmetric no-go plasticity:**
   - A burst depresses as much as a dip potentiates.
   - New no-go cells come only where none fired (as go recruitment does).
   - Withholds are recorded only at the motor gates, not every column loop.
3. **Homeostasis:** the go and no-go pools of a loop kept in balance (synaptic scaling), so neither
   can run to its cap.

### A quiet critic, symmetric no-go, homeostasis
The three changes, added one at a time.
- **v18, a quiet critic.** The critic no longer sees recall and working memory, which change
  every step whatever happens (`MIND_CRITIC_ALL=1` restores them). It sees the amygdala's attended
  cues and value cells, the orbitofrontal learner's value cells, presence and affective touch.
- **v19, plus symmetric no-go at the motor gates** (`MIND_NOGO_SYM=0` off).
  - A dip strengthens the held-back candidate's no-go cells that fired, or recruits new ones only
    where none fired.
  - A burst removes as many of those that fired.
  - Withholds are recorded only at the motor gates, not at every column's loop.
- **v20, plus homeostasis** (`MIND_BALANCE=0` off): a channel's go and no-go pools are kept within
  twice each other plus 16 cells; the larger pool's stalest cells go (synaptic scaling).

Instrumental probe, 600 trials, seeds 0–4. v17 (the corrected withhold rule, before these) for
comparison.

**Clicks per trial in the window, light / none**

| version | seed | 0–100 | 100–200 | 200–300 | 300–400 | 400–500 | 500–600 |
|---|---|---|---|---|---|---|---|
| v18 | 0 | 0.02/0.12 | 0.18/0.24 | 0.74/0.79 | 1.04/0.77 | 1.00/1.27 | 1.02/0.98 |
| v18 | 1 | 0.31/0.37 | 0.57/0.69 | 0.86/1.08 | 0.40/0.77 | 0.92/1.08 | 0.82/0.96 |
| v18 | 2 | 0.04/0.07 | 0.15/0.21 | 0.22/0.29 | 0.72/0.74 | 0.43/0.69 | 0.87/0.82 |
| v18 | 3 | 0.12/0.17 | 0.24/0.18 | 0.87/0.80 | 0.45/0.80 | 0.71/0.98 | 1.12/1.18 |
| v18 | 4 | 0.09/0.08 | 0.11/0.14 | 0.40/0.16 | 0.72/0.24 | 1.07/0.24 | 1.09/1.22 |
| v19 | 0 | 0.08/0.04 | 0.26/0.40 | 0.78/0.90 | 0.60/0.69 | 0.80/1.12 | 1.09/0.83 |
| v19 | 1 | 0.31/0.37 | 0.51/0.55 | 0.86/1.02 | 1.02/1.19 | 1.04/1.00 | 1.27/0.83 |
| v19 | 2 | 0.04/0.07 | 0.21/0.13 | 0.68/0.83 | 0.66/0.90 | 0.86/0.86 | 1.03/0.88 |
| v19 | 3 | 0.26/0.19 | 0.50/0.65 | 0.64/0.70 | 0.40/0.47 | 0.80/0.87 | 1.02/1.16 |
| v19 | 4 | 0.22/0.18 | 0.42/0.53 | 0.88/0.86 | 0.93/1.05 | 0.87/1.20 | 1.07/0.89 |
| v20 | 0 | 0.08/0.04 | 0.04/0.20 | 0.25/0.66 | 0.22/0.53 | 0.40/0.50 | 0.85/0.65 |
| v20 | 1 | 0.25/0.32 | 0.12/0.04 | 0.22/0.18 | 0.54/0.37 | 0.61/0.59 | 0.87/0.95 |
| v20 | 2 | 0.16/0.04 | 0.18/0.10 | 0.31/0.16 | 0.84/0.76 | 0.57/0.66 | 0.80/0.77 |
| v20 | 3 | 0.20/0.20 | 0.11/0.30 | 0.36/0.09 | 0.74/0.31 | 0.53/0.60 | 0.65/0.96 |
| v20 | 4 | 0.20/0.14 | 0.70/0.40 | 1.00/0.96 | 0.51/0.89 | 1.00/0.94 | 0.93/0.93 |

**The last 100 trials:** the hand loop's go and no-go cells, the dopamine per step, and the acts
per step

| version | go / no-go cells (seeds 0–4) | dips per step | at a pause: dips / bursts | acts per step |
|---|---|---|---|---|
| v17 | 1,569/1,582; 92/2,048; 105/617; 581/1,214; 822/2,048 | −0.004 to −0.18 | 0–40 / 0–73 | 0.05–0.67 |
| v18 | 947/77; 965/100; 605/49; 808/141; 1,117/1,063 | −0.008 to −0.034 | 6–32 / 76–149 | 0.78–0.90 |
| v19 | 889/160; 1,200/124; 866/197; 1,511/114; 1,157/482 | −0.007 to −0.045 | 6–75 / 119–249 | 0.75–0.92 |
| v20 | 100/44; 424/205; 232/108; 382/185; 468/236 | −0.007 to −0.019 | 1–17 / 48–139 | 0.59–0.91 |

**What this says.**
- **Acting no longer dies out.** All 15 runs click to the end, at 0.6–1.3 clicks per trial in the
  window, where v17 lost it on three seeds of five.
- **The quiet critic is what fixed it.** Dips per step fell from as much as −0.18 to −0.01 to
  −0.03, and no-go no longer runs away. Symmetric no-go and homeostasis kept the pools smaller and
  in balance (v20: 100–468 go cells), but did not change the outcome.
- **The light is no longer used.** In every run, clicks with and without the light end about
  equal. The swing went from a no-go runaway to go dominance: bursts at a pause outnumber dips 5 to
  25 times, and nothing holds the hand back when the light is off.
  - **A click without the light gets no dip.** Nothing expected a touch then. The orbitofrontal
    learner's conjunctive cells extinguish "a click alone predicts a touch", so its prediction
    for a click without the light is zero, as it should be, and zero minus zero is no error.
  - **The only cost of a useless click is the effort,** 1/32. That is small beside the bursts the
    light trials bring.
  - **The two earlier successes were partly luck of balance.** v15 seed 1 and v17 seeds 0, 3 and 4
    held back without the light while no-go was large. Their holding back came from the runaway
    that later killed acting, not from the dip a useless click should bring.

**Next.**
1. **Opportunity cost (Niv et al. 2007).** An act costs the average reward rate, the slow average
   that is already kept: when acting usually pays, an act that brings nothing is a loss of what
   that time could have brought, so a useless click is a dip. The cost scales with how rich the
   world is, unlike a fixed effort.
2. **The act-alone expectation kept as a baseline.** In the orbitofrontal learner, the value of
   an act regardless of cues stays a prediction that fails without the light, instead of being
   extinguished entirely: a dip after a click without the light.
3. **Then the 5 seeds again,** and the Pavlovian probe to check it is unchanged.

### Opportunity cost and a per-act baseline
- **v21: opportunity cost** (`MIND_OPPORTUNITY=0` off). The outcomes in the two steps after each
  act are slowly averaged ("what an act usually brings"), and an act costs that, or the effort if
  more (Niv et al. 2007). An act that brings nothing is then a dip, in proportion to how much
  acting usually pays.
- **v22: plus a per-act baseline** (`MIND_ACT_BASELINE=0` off).
  - The orbitofrontal learner keeps baseline cells on the act's features alone, recruited and
    removed one at a time toward the outcomes. They track the act's average worth and are never
    extinguished.
  - Its conjunctive cells learn what the cues add.
  - A click without the light now has an expectation to fall short of.

**Instrumental, 600 trials: clicks per trial in the window, light / none**

| version | seed | 0–100 | 100–200 | 200–300 | 300–400 | 400–500 | 500–600 | what an act usually brings |
|---|---|---|---|---|---|---|---|---|
| v21 | 0 | 0.12/0.15 | 0.20/0.22 | **0.53/0.17** | 0.46/0.30 | 0.14/0.40 | 0.16/0.35 | +0.14 |
| v21 | 1 | 0.07/0.15 | 0.16/0.10 | 0.16/0.07 | 0.33/0.24 | 0.64/0.62 | 0.89/0.86 | +0.05 |
| v21 | 2 | 0.02/0.12 | 0.12/0.14 | 0.26/0.46 | 0.60/0.37 | 0.63/0.63 | 0.21/0.72 | +0.02 |
| v21 | 3 | 0.12/0.12 | 0.18/0.27 | 0.28/0.12 | **0.64/0.16** | 0.61/0.53 | 0.53/0.25 | +0.09 |
| v21 | 4 | 0.04/0.22 | 0.28/0.16 | 0.47/0.21 | 0.35/0.38 | 0.55/0.36 | 0.89/0.68 | +0.03 |
| v22 | 0 | 0.13/0.08 | 0.12/0.04 | **0.50/0.14** | 0.38/0.48 | 0.29/0.69 | 0.71/0.80 | +0.13 |
| v22 | 1 | 0.13/0.23 | 0.19/0.29 | 0.57/0.23 | **0.67/0.00** | **0.66/0.00** | **0.61/0.02** | +0.01 |
| v22 | 2 | 0.23/0.11 | **0.52/0.16** | 0.64/0.60 | 0.40/0.58 | 0.57/0.66 | 0.79/0.59 | +0.08 |
| v22 | 3 | 0.12/0.17 | **0.59/0.12** | 0.29/0.31 | 0.60/0.83 | 0.41/0.39 | 0.32/0.34 | +0.08 |
| v22 | 4 | 0.12/0.17 | **0.44/0.12** | 0.35/0.13 | 0.71/0.41 | 0.41/0.45 | 0.58/0.58 | +0.08 |

- **No run dies out:** 10 of 10 act to the end, at 0.42–0.80 acts per step. The hand loop's go and
  no-go pools stay balanced (go about 1.1–2 times no-go).
- **Discrimination now appears on most seeds, but mostly does not last.** One run held it from
  block 4 to the end: v22 seed 1, 0.67, 0.66, 0.61 clicks with the light against 0.00, 0.00, 0.02
  without. Six others show it for a block or two (0.44–0.64 against 0.12–0.17), then lose it.
- **The opportunity cost is small.** What an act usually brings settles at +0.01 to +0.14. Most
  acts bring nothing (the hand's acts include moves and clicks off the light), so the average is
  low, and so is the dip a useless click gets. The bursts of light trials (a touch is worth 1.0)
  still outweigh it.
- **Pavlovian, unchanged** (v22, seeds 0–2, last 100 trials):
  - The amygdala's prediction is +0.80 to +0.94 for CS+ against +0.00 to +0.04 for CS−.
  - Dopamine at the tone is higher for CS+ on all three: +0.10/+0.01, +0.22/+0.01, +0.00/−0.06.
  - Dopamine at the touch is +0.22 to +0.40.

**What this says.** Every part of the instrumental conjunction is now there:
- the light coded (coverage);
- the cue's value (amygdala) and the act's value with and without it (orbitofrontal: conjunction
  and baseline);
- the efference copy;
- dips for a useless act (opportunity cost, and the baseline's expectation);
- a quiet critic;
- balanced, symmetric no-go.

It learns the conjunction on most seeds for a while, and holds it on one. What it does not yet do
is keep it.
- **The average per act is diluted by every act of the hand,** including moves that are
  irrelevant here. It is one average for every act, where an animal's cost of an action is
  specific to the action (and to its context).
- **Discrimination is lost when clicking without the light drifts up again.** Nothing marks the
  light-off clicks as a mistake strongly enough to hold the line.

**Next, if this line continues.**
- Make the opportunity cost per act (each motor primitive's own average, from the orbitofrontal
  baseline itself rather than one average).
- Train longer (an animal's discriminated operant takes hundreds of rewarded trials more).
- Then return to the lessons with everything on, and compare with the first version's silence.

**Next (from the first probes).**
1. **Separate agency from value.** The own-effect signal trains the motor loops (which acts have
   effects) and habituates per act and outcome; the critic learns from outcomes.
2. **Pavlovian–instrumental transfer:** a cue's amygdala value raises the motor loops' readiness to
   release (the brake), as cue value invigorates responding.
3. **Keep cues attended** while their value is learned (the Mackintosh term), and check the delayed
   probe once the immediate one works.
4. **Then the lessons again,** comparing with the first version above.

### Per-act cost and credit, an attention gate, compression; what broke the tone
- **v23: per-act cost** (`MIND_PER_ACT=0` off). An act costs what the orbitofrontal baseline says
  *that* act usually brings, read from its own baseline cells, or the effort if more. The set of
  acts is not fixed: a new act recruits its own baseline cells the first time it is made, so the
  costs grow with the repertoire.
- **v24: plus per-act credit** (`MIND_ACT_CREDIT=0` off). When an act's two-step window closes,
  the loop and motor gate that chose it are reinforced at that choice by what came minus what
  the act was worth (`reinforce_at`). Each act gets its own credit, delayed to when its outcome is
  known, rather than sharing one dopamine error with whatever else happened that step.
- **v25: an attention gate** (`MIND_ATTEND_GATE`). The amygdala's and the orbitofrontal learner's
  value cells form and fire only on attended features (and one's own acts). A feature is attended
  when a surprise or a value follows it (Pearce–Hall and Mackintosh), and the attention fades by
  1/32 a step. A bug skipped the attention update when the gated context was empty, so nothing
  could ever win attention back. It was fixed by updating attention first, over the whole trace.
- **v26–27: compression**, three ways, each a switch:
  1. **Sleep compaction** (`MIND_COMPACT=0` off). At sleep, the cortical layers merge cells with
     the same output and basal masks, and prune cells idle for 20,000 steps or wrong at least 12
     times of their last 16. The amygdala's, orbitofrontal and striatal value cells are left
     alone: there the count of cells is the value.
  2. **Consolidation** (`MIND_CONSOLIDATE=0` off). During replay, the next row's recalled ear is
     the target for the previous step's goals, so the cortex learns the sequence. A row that the
     cortex completes from its first three quarters is marked consolidated and fades four times
     faster.
  3. **Predictive transmission** (`MIND_PREDICTIVE=1` on, off by default). A sense passes up only
     what the area did not predict.

**Instrumental, per-act cost and credit, 1500 trials** (clicks with light / none, every third
block, then the last block)

| version | seed | 200–300 | 500–600 | 800–900 | 1100–1200 | 1400–1500 |
|---|---|---|---|---|---|---|
| v23 | 0 | 0.47/0.24 | 0.73/0.61 | 0.87/1.07 | 0.84/0.89 | 1.09/1.00 |
| v23 | 4 | 0.84/0.74 | 0.92/1.02 | 1.02/0.97 | **1.08/0.02** | 0.00/0.00 |
| v24 | 0 | 0.80/0.57 | 0.98/1.14 | 0.90/0.93 | 0.90/1.07 | **0.68/0.10** |
| v24 | 1 | 0.69/0.38 | 0.95/1.07 | 0.80/0.82 | 0.80/0.74 | **0.65/0.06** |
| v24 | 3 | 0.84/0.61 | 0.63/0.93 | 0.86/0.93 | 0.87/0.94 | 0.83/0.91 |

- **The acting is high and stays high.** Most runs click about once per trial for hundreds of
  trials, with and without the light. Each act's own cost is near zero (what a click usually
  brings settles at +0.00 to +0.06), so a click without the light costs almost nothing.
- **Discrimination came late, if at all.** It appeared after more than 1,000 trials on three of ten
  runs: v24 seeds 0 and 1 in the last block (0.68/0.10, 0.65/0.06), and v23 seed 4. That run then
  stopped acting altogether, light or not: its no-go won, the die-out seen before the balance
  rule.
- **The Pavlovian probe broke at the same time** (two causes, found below: the attention gate,
  and a world that dropped the tone under the child's own voice). From v24 on, the amygdala predicted almost
  nothing for either tone (+0.00 to +0.02 at the touch's step on eight of nine runs; v25 seed 2
  reached +0.51), and the dopamine at the touch stayed near its
  naive +0.85, where v22 had +0.80 to +0.94 for CS+ against +0.00 to +0.04 for CS−.

**Compression, instrumental seed 0, at trial 500**

| | cortical cells | sensory predictions right | hippocampal rows | merged / pruned | rows consolidated |
|---|---|---|---|---|---|
| everything on | 31,277 | 75.5% | 236 | 114 / 34 | 855 |
| compaction and consolidation off | 31,128 | 69.9% | 781 | 0 / 0 | – |
| plus predictive transmission | 22,042 | 72.0% | 568 | 405 / 44 | – |

- **Consolidation does what it should for the hippocampus.** The store holds a third as many
  rows (236 against 781), because what the cortex has learned fades from it. The cortex predicts
  the senses better (75.5% against 69.9%), because it learned the replayed sequences.
- **Compaction barely changes the cortex's size.** It merged 114 cells and pruned 34 of about
  31,000. Duplicates are rare when cells are recruited only on uncovered input. The cortex grows
  about as fast as it would anyway.
- **Predictive transmission shrinks the cortex by 30%** (22,042 cells), but it hurts acting.
  Clicks stayed at 0.1–0.5 per trial, because what the cortex predicts, including the light it
  comes to expect, no longer reaches the loops as a fresh input. It stays off.

**What broke the tone: a bisect.** 200 Pavlovian trials, seed 0, one group of switches off at a
time. The amygdala's prediction at the touch's step, CS+ / CS−, in trials 100–200:

| switched off | CS+ / CS− | dopamine at the touch |
|---|---|---|
| nothing (v27) | +0.002 / +0.004 | +0.877 |
| per-act credit | +0.000 / +0.005 | +0.847 |
| opportunity cost, per-act cost and credit | +0.003 / +0.000 | +0.873 |
| own effect and response (a quiet mind) | +0.022 / +0.004 | +0.863 |
| per-act cost and credit | +0.295 / +0.001 | +0.819 |
| act baseline, per-act cost and credit | +0.259 / +0.011 | +0.859 |
| compaction, consolidation, per-act, credit, gate | +0.534 / +0.008 | +0.618 |
| **the attention gate alone** | **+0.995 / +0.007** | **+0.375** |

**The attention gate is the cause.** Without it, the tone is learned in 100 trials as well as it
ever was: CS+ predicted at +0.995 against +0.007, and the dopamine at the touch is cut to less than
half as the tone comes to predict it.

Why the gate hurts:
- A value cell is a 16-bit mask on the hashed context, and it fires when 12 of the bits are there.
  With the gate on, the context is only the attended features.
- Attention flickers. It is set for the whole trace on any large error, then fades bit by bit. So
  the attended set at the touch's step differs from trial to trial, even when the tone and the
  touch are the same.
- A cell recruited on one trial's context then misses most others. Recruiting and removing go on,
  but no cell keeps firing long enough to stand for the tone.
- The ungated context is the whole trace. It carries the same tone bits every time, so cells
  recruited on it keep firing.

In the brain, associability changes how fast a cue's value is learned, not whether its value can
be read out. The gate here did both. An attention signal that only scales learning (more cells
recruited for an attended cue, the same context to read) would match that, and is the way to
bring it back.

**v28: the gate off.** On seed 0 the tone is learned and held over 600 trials. The amygdala
predicts +0.94 to +1.00 for CS+ against +0.00 to +0.03 for CS−, and the dopamine at the touch
falls to +0.15–0.30. Seeds 1 and 2 learn it in the first 100 trials (+0.44, +0.47), then lose it.
Instrumental, 1000 trials, three seeds: clicking rises to about one per trial with and without the
light. The only discrimination is brief: seed 1 at 0.73/0.33 in trials 100–200, and seed 2 at
0.44/0.15 in trials 200–300.

**A second bisect, seed 1** (200 trials, CS+ prediction in trials 100–200):

| switched off | CS+ | acts per step |
|---|---|---|
| nothing | +0.04 | 0.71 |
| sleep | +0.06 | 0.62 |
| compaction | +0.08 | 0.67 |
| consolidation | +0.49 | 0.45 |
| per-act cost and credit | +0.74 | 0.31 |

Sleep and compaction are cleared. What tracks the loss is how much the child acts.

**v29: elemental value cells** (`MIND_ELEMENTAL=0` off). A value cell was a mask of 16 bits drawn
from the whole trace, about 100 bits of which the tone was a few. A cell then fired on the tone
only while about seven incidental onsets recurred too, and more acting made them vary. Now each
new amygdala cell samples one cue: one area's onsets at one age. This is how a lateral amygdala
cell's plasticity is specific to the input stream that drove it, and it makes the cells
Rescorla–Wagner's elements. Cells no longer drift toward whatever else was in the trace. The
orbitofrontal learner keeps its conjunctions of act and cues, since that is its job. Elemental
cells alone fixed seed 2 but not seeds 0 and 1.

**The real cause: the world masked the tone.** A diagnosis of the trace at the tone's step
(seed 0) showed it:
- In trials 0–100, the CS+ tone made about 18 ear onsets, 13 of them the same as last time.
- In trials 100–200 it made none.
- The world played the conditioning tone only when nothing else was heard. A voice replaced it,
  including the child's own.
- With per-act cost, the child talks on most steps, so it talked over the tone and never heard
  it. That is why the tone's loss tracked acting. The light was never masked: it is drawn over
  whatever else the eye sees.
- Now sounds at once mix: the tone is heard over a voice.

**v30: the tone mixes.** Pavlovian, 300 trials:

| seed | cells | CS+ / CS−, 0–100 | 100–200 | 200–300 | dopamine at the touch, 0–100 → 200–300 |
|---|---|---|---|---|---|
| 0 | elemental | +0.90 / +0.00 | +1.00 / +0.00 | +1.00 / +0.00 | +0.33 → +0.17 |
| 1 | elemental | +0.91 / +0.00 | +1.00 / +0.00 | +1.00 / +0.00 | +0.47 → +0.10 |
| 2 | elemental | +0.90 / +0.02 | +1.00 / +0.00 | +1.00 / +0.00 | +0.60 → +0.15 |
| 0 | conjunctive | +0.91 / +0.00 | +1.00 / +0.00 | +0.99 / +0.00 | +0.29 → +0.17 |
| 1 | conjunctive | +0.89 / +0.02 | +0.99 / +0.01 | +1.00 / +0.00 | +0.55 → +0.24 |
| 2 | conjunctive | +0.78 / +0.01 | +0.99 / +0.00 | +0.98 / +0.00 | +0.64 → +0.35 |

- **The tone is learned on every seed, with every switch on**, in under 100 trials, and held.
  The tone is now heard every time: 31–32 ear onsets, all the same as last time.
- **The dopamine at the touch is cut to a fifth or less** as the tone comes to predict it, and the
  dopamine at the CS+ tone rises above the CS−'s (+0.05 to +0.15 against −0.01 to −0.03).
- **Elemental cells learn faster and cancel the touch's dopamine further.** They stay on.

**What this says about the earlier per-act results.** The per-act bisect rows above measured
mostly how often the child talked over the tone, not the per-act machinery itself. The attention
gate's effect was separate and real: on seed 0, with the same acting, it alone made the
difference between +0.002 and +0.995.

**v30 instrumental, 1000 trials** (clicks with light / none, per 100 trials):

| seed | 0–100 | 200–300 | 400–500 | 600–700 | 800–900 | 900–1000 | what an act usually brings |
|---|---|---|---|---|---|---|---|
| 0 | 0.00/0.04 | 0.32/0.63 | 0.23/0.29 | 0.89/0.69 | 1.04/0.71 | 0.76/1.28 | +0.17 |
| 1 | 0.04/0.16 | **0.31/0.12** | 0.49/0.76 | 0.80/0.80 | 0.88/1.05 | 1.02/0.82 | +0.08 |
| 2 | 0.11/0.09 | 0.34/0.27 | 0.41/0.80 | 0.69/1.11 | 0.88/1.13 | 1.14/1.02 | +0.12 |

The tone fix does not touch this probe, and the light is not discriminated. Clicking rises to
about one per trial whether the light is on or not. The light makes clicks pay, and the
loops learn that clicking pays, but not that it pays only with the light.

The open problem is now the operant, not the cue:
- the light's value reaches the striatum (amygdala and orbitofrontal value cells, salience);
- a click off the light still costs only what a click usually brings (+0.08 to +0.17, against
  the 1.0 of a touch), so the dip it earns is small next to the bursts.

**Defaults now (v30):**
- The attention gate is off (`MIND_ATTEND_GATE=1` turns it on).
- Elemental amygdala cells are on.
- Per-act cost and credit, compaction and consolidation are on.
- Predictive transmission is off.

### The light: focused cells, a credit bug, and credit against the act's own expectation
**What was missing for the light** (diagnosis after v30):
- **The orbitofrontal learner could not stand for "click after the light".** Its cells sampled 16
  bits from a trace of about 100 items, of which the light and the click were a few.
- **The hand loop was blind to the light.** The striatum's context has 64 bits split equally among
  about a dozen input groups, so the light got 2–3. A cell of 16 bits that fires at 10 carried
  less than one.

**v31–32: focused cells** (`MIND_FOCUSED=0` off).
- **The striatum's context:** each input group gets a region of its own (a 64-bit word) with
  16 bits, and the candidate act gets the last region.
- **New striatal cells:** a new cell takes two regions, the candidate's and one other (two at
  random for the critic), and learns only within them. With v32, each region is a dendritic
  branch that must reach 5 of its 8 bits, so every input the cell takes must be there.
- **The orbitofrontal learner:** a new cell takes half its bits from one act's efference copy and
  half from one cue.
- **Neither version discriminated.** All three seeds clicked about once a trial with or without
  the light, or stopped acting.

**A striatum-only light probe** (`cargo test --release light_probe -- --ignored --nocapture`):
- one act, a light in view on half the trials, among constant and noisy inputs;
- a click with the light pays 1, and every click costs 1/8;
- each click is credited one step on with what came minus what it was expected to bring.

It showed that the learner itself could not do it, and it found the bug.
- **The bug.** `reinforce_at` dropped the Q16 scale: its update was |δ|/UNIT/4 where it should be
  |δ|·ONE/UNIT/4. A cell changed with odds of about 1 in 65,536. The per-act credit from v24 on
  never changed a cell. Nor did agency (`reinforce_last`, the own-effect signal to the motor
  loops) since it was added. Earlier results credited to either were the critic's broad dopamine
  alone.
- **The error.** Credited against the act's average worth (a fixed 1/8 here), cells blind to the
  light gain on every paid click and lose a little on every unpaid one, so clicking everywhere
  wins. Rescorla–Wagner's error is against the prediction for this act in this context. Then the
  light-blind cells settle at the average, and the cells that see the light take up the
  difference.

Clicks per trial in the window, light / none, first and last of ten blocks of 100 trials:

| cells | credit against | eligibility window | first | last |
|---|---|---|---|---|
| whole view | average worth | 0 | 0.80/0.69 | 0.70/0.72 |
| focused | average worth | 0 | 0.35/0.33 | 0.00/0.00 (stops) |
| whole view | its own q | 0 | 0.91/0.67 | 0.72/0.45 |
| **focused** | **its own q** | **0** | 1.13/0.46 | **0.66/0.00** |
| **focused** | **its own q** | **2** | 1.17/0.52 | **0.83/0.02** |
| focused | its own q | 4 | 1.54/1.06 | 1.36/0.11 |
| focused | its own q | 8 | 1.83/0.69 | 1.47/0.30 |

- **Discrimination needs both changes:** focused cells, so some cells can stand for the light with
  the click; and credit against the act's own expectation there.
- **A long eligibility window hurts.** Each burst also credits the clicks made before the light
  came on.

**v33 in the mind:**
- Fixes `reinforce_at`.
- Credits each act against its loop's and its gate's own q at the choice, a cell an eighth,
  clamped to 0–1 (`MIND_CREDIT_OWN=0`: against the act's average worth).
- Sets the striatum's eligibility window to 2 steps (`MIND_WINDOW`). In the brain the
  corticostriatal eligibility trace lasts about a second.

**Instrumental, 600 trials** (clicks per trial in the window, light / none):

| seed | 0–100 | 100–200 | 200–300 | 300–400 | 400–500 | 500–600 |
|---|---|---|---|---|---|---|
| 0 | 0.26/0.23 | 0.77/0.31 | 0.59/0.28 | 0.62/0.11 | 0.91/0.14 | **1.04/0.00** |
| 1 | 0.37/0.17 | 0.88/0.34 | 0.85/0.21 | 0.42/0.26 | 0.56/0.22 | 0.89/0.29 |
| 2 | 0.71/0.27 | 0.67/0.13 | 0.32/0.16 | **0.63/0.00** | **0.88/0.04** | **0.67/0.00** |
| 0, window 8 | 0.22/0.38 | 0.29/0.31 | 0.51/0.34 | 0.31/0.13 | 0.20/0.25 | 0.17/0.12 |

- **The light controls clicking on every seed for the first time.**
  - Clicks with the light are 2–4 times those without from the second block on.
  - Two seeds end with no clicks at all without the light.
  - None dies out.
- **What a click usually brings rises to +0.29 to +0.42** (+0.08 to +0.17 before), because the
  clicks now come when they pay.
- **The child acts less overall:** 0.27–0.37 acts per step, against 0.7–0.9 before.
- **With a window of 8, the discrimination is lost.**
- **The tone still holds** (seed 1, 300 trials): CS+ +0.89 → +0.99 against CS− +0.00 to +0.02, and
  the dopamine at the touch falls to +0.19.

**Defaults now (v33):**
- the attention gate off;
- elemental amygdala cells;
- focused striatal and orbitofrontal cells;
- per-act credit against the loops' own q;
- an eligibility window of 2;
- compaction and consolidation on;
- predictive transmission off.

**v33, the full conditioning set, three seeds.**

*Pavlovian, 600 trials* (the amygdala's prediction at the touch's step, CS+ / CS−; dopamine at
the tone, CS+ / CS−; dopamine at the touch):

| seed | 0–100 | 500–600 | dopamine at the tone, 500–600 | at the touch, 0–100 → 500–600 |
|---|---|---|---|---|
| 0 | +0.88 / +0.00 | +1.00 / +0.05 | +0.23 / +0.05 | +0.30 → +0.37 |
| 1 | +0.89 / +0.00 | +1.00 / +0.01 | +0.43 / −0.01 | +0.22 → +0.09 |
| 2 | +0.90 / +0.01 | +1.00 / +0.01 | +0.23 / −0.06 | +0.39 → +0.16 |

- The tone is learned within 100 trials and held to the end on every seed.
- The dopamine at the CS+ tone is higher than at the CS− in every block of every seed
  (+0.09 to +0.43 against −0.12 to +0.05): the burst has moved to the cue.

*Instrumental, 1000 trials* (clicks per trial in the window, light / none):

| seed | 0–100 | 200–300 | 400–500 | 500–600 | 600–700 | 700–800 | 800–900 | 900–1000 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.26/0.23 | 0.59/0.28 | 0.91/0.14 | **1.04/0.00** | **0.82/0.00** | **0.88/0.00** | **0.80/0.02** | **0.49/0.00** |
| 1 | 0.37/0.17 | 0.85/0.21 | 0.56/0.22 | 0.89/0.29 | 0.91/0.39 | **0.92/0.08** | **1.04/0.09** | 0.82/0.15 |
| 2 | 0.71/0.27 | 0.32/0.16 | 0.88/0.04 | **0.67/0.00** | **0.87/0.00** | **0.84/0.05** | **1.02/0.02** | **0.80/0.00** |

- **The discrimination holds over 1000 trials.**
  - Seeds 0 and 2 click almost never without the light from about trial 400 on.
  - Seed 1 gets there by trial 700, at 0.08–0.15 without the light.
- What a click usually brings is now +0.35 to +0.47.
- The child acts on 0.21–0.33 of steps.

*Delayed, 600 trials* (a light for one step, at step 2; a click 3–5 steps later pays):

| seed | 0–100 | 100–200 | 200–300 | 300–400 | 400–500 | 500–600 |
|---|---|---|---|---|---|---|
| 0 | 0.11/0.22 | 0.24/0.47 | 0.20/0.22 | 0.13/0.11 | 0.02/0.02 | 0.06/0.04 |
| 1 | 0.22/0.30 | 0.45/0.26 | 0.02/0.07 | 0.00/0.00 | 0.00/0.00 | 0.00/0.02 |
| 2 | 0.30/0.31 | 0.60/0.42 | 0.31/0.13 | 0.62/0.25 | 0.44/0.17 | 0.12/0.05 |

- **The delayed probe is not learned.** Clicking fades on every seed. Seed 2 discriminates for a
  while (0.62/0.25, 0.44/0.17) before fading too.
- **Why, as far as the parts go:**
  - When the click that pays is made, the light has been off for three to five steps.
  - The hand loop's view holds the eye's code now (no light) and the amygdala's attended trace.
    The trace keeps an onset four steps (ages 0–3), so a click five steps on has nothing of the
    light left in its view.
  - The working memory a loop can load holds goals, not cues.
  - So the striatum sees the light's after-effect only on some of the steps where a click pays,
    and the clicks it does credit look, in the view, like the clicks that do not pay.
- **What is missing is a cue held across the delay.** Prefrontal persistent activity, or the
  hippocampus's time cells, keep a cue that has gone. The trace here is short, by design. The
  next step would be to let a loop load a valued cue into working memory, as it now loads goals.

### Reversal, associability, the orbitofrontal cortex as an area, frustration
**Not checked here.** A design note proposed dividing the work: the amygdala sets how much to
learn from an outcome (Pearce–Hall associability, a salience tag for replay), and the
orbitofrontal cortex sets where the learning goes (a hidden task state Z, new states on outcome
surprise, contingent credit). Its figure of about 860 trials per reversal is not from this
repository; there was no reversal probe here.

**The serial reversal probe** (`COND=reversal`, `COND_PERIOD`, default 500).
- Every trial one of two lights is on for three steps (A on half the trials, B on the others).
- A click while the light is on, or the step after, gets a stroke if that light pays now: A for
  the first 500 trials, then B, then A, then B.
- Read: clicks per trial in the window, the paying light against the other, per block of 100.

**v34: associability** (`MIND_ALPHA=0` off).
- **In the amygdala:** each cue feature has α (0–255). It moves a quarter of the way toward how
  far the error after it exceeds the usual |error| (a running mean). A cue never seen before
  starts high.
- **In the striatum:**
  - A motor choice's learning gain follows the trace's α: ¼× when nothing has been surprising, up
    to about 2¼×. It applies to the per-act credit and to the dopamine error alike.
  - Where its new cells take their second input: a region is drawn in proportion to its cues' α.

**v35: the orbitofrontal cortex as a cortical area** (`MIND_OFC_COLUMN`, `src/program/ofc.rs`).
- **A column like the others.** Layer 4 takes four sources: the association areas' code (cues),
  the efference copy (the act just made), affective touch (the outcome as sensed), and the
  hippocampus's recall at the top of the hierarchy (context).
- **Layer 2/3** predicts the next input from this one, its own previous activity and, on its
  tuft, the state it holds. Its activity is the state Z.
- **An outcome layer** predicts the next outcome (Ô).
- **Holding:** a loop through the striatum (channel 300) chooses each step whether to hold Z on
  the tuft, as prefrontal working memory is gated (PBWM). It is learned from dopamine.
- **Z and the held state** reach the critic and every motor loop as two more regions.
- **Value on Z** (`MIND_OFC_VALUE_Z`): the orbitofrontal value cells read Z and the act, not the
  raw cue trace. A new cell pairs an act with a Z group. The per-act baseline is kept.
- **Splitting on outcome surprise** (`MIND_OFC_SPLIT`). An outcome mispredicted by at least ¾
  after an input expected at least half, with the cues' α at least a quarter, makes the area hold
  its state now: the one that carries the surprising outcome. The same cues then meet a new
  context, so they are learned afresh and the old cells are not retrained. This is
  noradrenaline's network reset (Bouret & Sara 2005).
- **The first outcome layer was wrong.** Its target was the stroke's layer 4 code, which changes
  with where on the skin each stroke lands, and it learned only when a touch came. It expected
  the outcome right 1 time in 15,857, and split 15,306 times in 2000 trials.
- **The fix:** its target is now the outcome's level on the labelled line (none, weak, middling,
  strong), so "nothing comes" is predicted too. It then expects the outcome right 97–99% of the
  time, and splits 8–171 times in 2000 trials.

**v36: frustration** (`MIND_FRUSTRATION=0` off; Amsel's frustration).
- **How it builds:** an act that brings less than expected adds the shortfall; more than expected
  takes it away. It fades by 1/16 a step.
- **Vigor:** it lowers the motor loops' brake by up to 2 (the extinction burst).
- **Variability:** it widens the motor noise by up to 6 (extinction-induced variability).
- **Its first expectation was the striatum's q, which was wrong.** A loop releases an act only at
  q of at least 2, so every act released expected at least ¼. Frustration sat at 1.1–1.6 the
  whole run, the noise and brake stayed at their limits, and the light probe lost its
  discrimination (0.25/0.27).
- **v36b measures it against the orbitofrontal value of the act in its context.** That should be
  about 0 for a click without the light once that is learned, and high for the old light just
  after a reversal.

**Reversal, 2000 trials** (paying light / the other, per block of 100; A pays in blocks 1–5 and
11–15):

| version | seed | 300–400 (A) | 500–600 (B) | 900–1000 (B) | 1400–1500 (A) | 1900–2000 (B) |
|---|---|---|---|---|---|---|
| v33 | 0 | 0.09/0.19 | 0.07/0.11 | 0.02/0.00 | 0.08/0.00 | 0.00/0.07 |
| v33 | 2 | 0.81/0.35 | 0.09/0.20 | 0.00/0.00 | 0.00/0.00 | 0.00/0.00 |
| + α | 0 | 0.80/0.55 | 0.35/0.60 | 0.02/0.20 | 0.12/0.06 | 0.02/0.00 |
| + α | 1 | 0.33/0.48 | 0.67/0.79 | 0.09/0.04 | 0.00/0.00 | 0.00/0.00 |
| + OFC area (no α) | 0 | 0.40/0.19 | 0.00/0.04 | 0.00/0.02 | 0.00/0.00 | 0.00/0.00 |
| + OFC area (no α) | 1 | 0.53/0.29 | 0.02/0.21 | 0.00/0.02 | 0.21/0.00 | 0.06/0.40 |
| + OFC area and α | 0 | 0.28/0.13 | 0.37/0.65 | 0.24/0.38 | 0.21/0.19 | 0.09/0.02 |
| + OFC area and α | 1 | 0.70/0.30 | 0.25/0.31 | 0.49/0.23 | 0.02/0.04 | **0.89/0.55** |
| + frustration (v36) | 0 | 0.32/0.48 | 0.17/0.22 | 0.45/0.47 | 0.77/0.50 | 0.69/0.36 |
| + frustration (v36) | 1 | 0.60/0.42 | 0.28/0.50 | 0.67/0.54 | 0.63/0.53 | 0.69/0.65 |

- **Without help, the first reversal kills the operant.**
  - The light that paid now earns dips, and the dips grow no-go cells on the click.
  - Nothing brings the click back for the other light, so clicking stops on most seeds and
    never returns.
- **α keeps the mind acting longer.** The OFC area with α is the only combination with a clear
  late reversal: seed 1, last phase, 0.77–0.96 for B against 0.37–0.55 for A.
- **The first frustration keeps the mind acting, but at random.** No run dies out, and acts rise
  to 0.65–0.72 a step. But the paying light and the other are close (0.69/0.65 at the end), and
  the light probe breaks.

**The light probe with the new parts** (seed 0, 600 trials, last block, light / none):
- OFC area without α: 0.70/0.16 (best block 1.12/0.38).
- α alone (v34, no OFC area): 0.09/0.07, clicking fading.
- With the first frustration: 0.25/0.27.

**v36b: frustration against the orbitofrontal value.** Frustration now stays small (0.03–0.23 on
average per block, where it was 1.1–1.6). The light probe, with every default on (the OFC area,
α, frustration), 600 trials, light / none:

| seed | frustration | 200–300 | 300–400 | 400–500 | 500–600 |
|---|---|---|---|---|---|
| 0 | on | 0.77/0.40 | 0.68/0.26 | 0.30/0.33 | 0.07/0.09 |
| 0 | off | 0.82/0.42 | 0.61/0.26 | 0.27/0.14 | 0.54/0.17 |
| 1 | on | 0.82/0.39 | 0.84/0.41 | 0.52/0.24 | 0.98/0.53 |

The light is discriminated, but less sharply than at v33 (1.04/0.00 on seed 0). On seed 0
clicking fades by the end with frustration on. The new parts cost some of what v33 had on this
probe.

*Reversal with the first frustration, α off* (v36, seed 1):
- It learns A (0.85/0.38).
- After the first reversal it keeps clicking on A, the light that no longer pays (0.30, 0.21
  against 0.02, 0.04 for B).
- It comes back to A when A pays again (0.51/0.29, 0.37/0.02).
- In the last phase it again clicks only on A (0.29, 0.20) and never on B.

That is perseveration: the old choice kept, the new one never tried, which is what
orbitofrontal lesions produce in animals. The variability frustration adds is not aimed at the
alternative.

**Lessons with everything as at v33** (seed 0; 600 steps of speech play, 300 mouse-play
episodes, 1000 lessons, 200 test lessons):

| | first version | v33 |
|---|---|---|
| speech play, said aloud (first, second half) | 123, 185 | 33, 37 |
| mouse play, clicks on a book | 11 of 35, then 0 of 1 | 33 of 91, then 39 of 70 |
| practice, silent (first, second half) | 99.9%, 100% | 80.1%, 94.5% |
| practice, words right | 0.0%, 0.0% | 0.3%, 0.1% |
| test, silent | 100% | 99.4–99.7% |
| test, words right | 0.0% | 0.0–0.1% |

- **The child talks in the first half of the lessons, then goes quiet.**
  - Aloud on 3337 of 16,813 story steps in the first half, and on 737 of 14,681 in the second.
  - At test it acts on 358 of 13,313 steps.
- **Why it goes quiet is not settled.**
  - Speaking paid, summed over training: dopamine from its own effect +4261, from responses
    +6275, from touches +314, against −1409 for effort.
  - But about 1.7% of what it said in the first half was right, near chance for 59 words.
  - So speaking was worth something on average, and was not tied to the moments it paid.
  - Per-act credit against the loops' own expectation, now working, may have taken the voice's
    go cells down wherever speaking brought less than the loop expected, which is most moments.
    That is to be checked.
- **The lessons need what the reversal and delayed probes need.** A state that says "asked to
  speak", and which word, held across the story. Without it, speaking is an act whose worth is
  the average over every moment, and that average is below its cost.
- The other seed (2) is still running.

**Reversal with frustration against the orbitofrontal value** (v36b, every default on, 2000
trials; paying light / the other):

| seed | 100–200 (A) | 400–500 (A) | 900–1000 (B) | 1000–1100 (A) | 1400–1500 (A) | 1900–2000 (B) |
|---|---|---|---|---|---|---|
| 0 | 0.54/0.33 | 0.23/0.35 | **0.61/0.45** | 0.10/0.44 | 0.02/0.02 | 0.00/0.00 |
| 1 | 0.83/0.35 | 0.52/0.27 | 0.05/0.07 | 0.04/0.04 | 0.14/0.14 | 0.04/0.04 |

- **Frustration now stays small:** 0.03–0.27 a block.
- **It does not bring reversal.**
  - Seed 0 shifts to B slowly over the second phase (0.61/0.45 at its end), keeps clicking on B
    when A pays again (0.44 on B, 0.10 on A), and stops.
  - Seed 1 learns A (0.83/0.35), and its clicking fades after the first reversal.
- **What frustration lacks is direction.** Its noise loosens every choice alike. An animal after
  a reversal tries the alternative because something represents it as an alternative: the
  other light's value, or the rule. Here nothing marks clicking on B as the thing to try.
- **The parts now in place for that:** the orbitofrontal state can split on the surprise, and
  the striatum can learn from it. What is missing is that the new state starts with the old
  values: a split should make the old light's learned value not apply, so the other light is
  tried as new.

**Next.**
1. After a split, the old state's value cells should not fire in the new state (they sit on Z,
   which changes), and the act's baseline should be relearned there. Check in the Bitfield
   Scope whether Z actually changes at a split, and whether the hand loop's cells that carried
   "click on A" stop firing.
2. Record a reversal trace around trial 500 for that check.

### Holding a state in the cortex itself: trace, recurrence, a held population
The orbitofrontal area is a general column, and it needed a bolt-on `held` copy, as prefrontal
working memory needed a separate store (`WorkingMemory`). That pointed at the column. Its layer
2/3 fires only on its basal input (this step's layer 4 code). Its recurrence (its own previous
activity) reaches only the tuft, where it primes and never drives. When the input stops, nothing
fires, so a column can hold nothing. Three versions followed, each replacing the store and the
copy.

**v37: a persistent trace** (`MIND_PERSIST=1`).
- Each column's layer 2/3 activity leaves a trace. Each bit clears at random with probability
  2^-τ a step, τ the area's own: none in the senses, 1 in the pairs, 3 in the full association
  area, 4 in the orbitofrontal area. Intrinsic timescales lengthen up the hierarchy (Murray et al.
  2014).
- The trace sits on the column's tuft and goes out with its activity. A loop's "take in"
  refreshes its column's trace.
- **Delayed, 600 trials, last block, light / none:**

  | seed | with the trace | without |
  |---|---|---|
  | 0 | 0.42/0.27 | 0.11/0.07 |
  | 1 | 0.04/0.06 | 0.06/0.04 |
  | 2 | 0.02/0.00 | 0.06/0.00 |

  Clicking fades either way.
- **The light probe weakens** (seed 0 best 1.08/0.36, last 0.48/0.48). The tone holds.
- **Why it cannot be enough:** the trace keeps whatever fired, noise included, and loses it at a
  fixed rate. It cannot clean up a state or keep only learned ones.

**v38: recurrent self-excitation** (an autoassociative map in each column).
- The map learns each input code as predicting itself, so a learned state, even half present,
  completes to the whole (tested: 18 of 32 bits complete to 32).
- The previous state, less a random sixteenth (noise) and completed, drives the code with the
  area's gain against layer 4's input, and the strongest 32 bits win. The gain is 0 in the senses,
  1 in the pairs, and 3 in the full association area and the orbitofrontal area.
- The loop's gate opens it for one step (the input becomes the state). Noise was first applied
  after the completion, which kept every state short of whole. It must come before, so the
  attractor restores it.
- A unit test held a learned state through 8 silent steps, held it against a new input with the
  gate closed, and switched on the gate.
- **In the mind it held one state, not the light.** A delayed-probe trace (40 trials) showed
  working memory keeping 79% of its bits from step to step, all the time. Five steps after the
  light, light trials' held states overlapped each other no more than they overlapped no-light
  trials' (59% against 65%; the orbitofrontal area 54% against 79%).
- At gain 3 the held state beat the input, so the light got in only when the gate happened to
  open at its step.
- Delayed did not discriminate (last block 0.00/0.26 to 0.35/0.53). The light probe was weak
  (best 0.75/0.57). Reversal seed 0 learned A (0.93/0.71) and stopped clicking after the switch.
  The tone held.

**v39: a held population beside the driven code.** A review of the cortex, given with the
request, put it this way:
- An area is mainly a driven contextual predictor.
- Driving input (layer 4) decides what it represents.
- Context and its own past arrive on the apical dendrites and select among driven
  interpretations; they do not create them.
- Local recurrence cleans up. Persistence, an attractor held without input, is a dial that turns
  up toward prefrontal cortex.
- Gain is set dynamically by the thalamus (mediodorsal amplification of prefrontal recurrence) and
  by acetylcholine: high acetylcholine favours feedforward encoding and suppresses recurrence, low
  acetylcholine releases completion (Hasselmo). Mismatch is what switches it.

So, in v39:
- **The driven code is layer 4's alone.** Recurrence no longer competes with input.
- **A held population sits beside it,** in the full association column and the orbitofrontal
  area only. The column's code is the two together, and in silence the held state alone.
- **Graded surprise loads it** (acetylcholine): the part of the input that layer 2/3 did not
  expect joins what is held, newest first, 32 bits in all. All-or-nothing surprise (load only
  when half the input is unexpected) loaded nothing: the light changes only part of the full
  area's input.
- **The gate loads the whole driven code** (the basal ganglia through the thalamus).
- **Otherwise the held state maintains itself** by completion against noise. What the map cannot
  complete fades.
- **Working memory** is the full association column's held population; the orbitofrontal hold is
  its own. The separate store and copy are not used (`MIND_RECUR=0` brings them back).
- A unit test: a scene, with A or B on some steps. An unexpected A is held while the expected
  scene drives the area, and through silence; the gate loads B.

**The held state now carries the light, weakly** (delayed trace, overlap between held states):

| | step 2, light / light | step 2, light / none | step 5, light / light | step 5, light / none |
|---|---|---|---|---|
| working memory | 0.79 | 0.66 | 0.81 | 0.75 |
| orbitofrontal hold | 0.80 | 0.59 | 0.73 | 0.65 |

**Probes, v39:**
- **Delayed, last block:** 0.15/0.23, 0.13/0.26, 0.39/0.39. Not learned.
- **Light probe:** seed 1 reaches 0.82/0.22. Seed 0 peaks at 0.92/0.49, then fades (0.28/0.11).
- **Reversal:**
  - Seed 0 keeps clicking through the first reversal and turns slowly to B (0.56/0.50, 0.52/0.34
    at the end of the phase), then fades in the third phase.
  - Seed 1 stops after the first reversal.
- **The tone holds** (+0.89 → +0.99, the touch's dopamine +0.44 → +0.31).

**What this says.** The cortex now holds what surprised it, with no store built for the purpose.
But the light is a small part of what is held (6–13 points of overlap), and the hand loop does
not yet use it.
- **The held state mixes the light with whatever else was new.** A mix is not a learned pattern,
  so completion cannot keep it whole, and it fades.
- **The hand loop sees the held state as one region among about twenty.** Its cells sample two
  regions at random, so few take the held one.

**Next** (from the review's own suggestion): a two-area test of the column alone.
- Only the upper area has a held population; the lower one is driven and clean.
- Does the upper one hold a context across a gap and a distractor while the lower one stays clean?
- Then raise the held population's weight in a loop's choice of regions by its associability.
  The light, surprising and followed by reward, should come to dominate what the hand loop
  samples.

**v40: the column alone, two areas** (`examples/two_area.rs`).
- **Set-up:**
  - A lower area: layer 4 over the input, with no held population.
  - An upper area: layer 4 over the lower area's code, with a held population.
  - A trial: a familiar scene for 3 steps, then a cue (A or B) joins it for one step, then a gap
    of 4 steps (the scene, or silence). A distractor C may join in the middle of the gap.
  - Read at the gap's end: is the upper area's held state closer to this trial's cue (as the
    upper area coded it) than to the other cue? And how much of the cue is left in the lower
    area's code?
- **Without layer 4 coverage the cue never reaches the upper area.** With 32 winners, the lower
  area's code for scene plus cue is the scene's own code. The upper area loaded on surprise 10
  times in 400 trials.
- In the mind, coverage was on only in the senses, not in the association areas. That is likely
  why the light was so faint in the full association column's held state.

**With coverage** (400 trials, by block of 50):

| gap | held closer to its own cue | own cue's bits held | other cue's bits held |
|---|---|---|---|
| silence, distractor | 84–90% | 20.5–22.2 | 0 |
| silence, no distractor | 72–74%, then 52% | 13.0–18.1 | 0 |
| scene, no distractor | 66% → 46% | 11.4 → 7.7 | 0 |
| scene, distractor | 50–58%, then 38% | 8.9–13.4 | 0 |

- **The lower area stays clean:** none of the cue's bits are left in its code at the gap's end.
- **The wrong cue is never held.** The misses are trials where nothing of the cue is held.
- **A familiar cue stops being held.** In the scene conditions the hold falls with training.
  As A and B become expected (each comes after the scene half the time), less of them is a
  surprise, and less loads.
- **Surprise is the right switch for what is new, not for what matters.** Holding a familiar,
  task-relevant cue is the gate's job (the basal ganglia, learned from reward), as the review
  said of prefrontal cortex.

**v41: in the mind.**
- Coverage now also covers the association areas' and the orbitofrontal area's layer 4.
- **A held state is tagged with the cues' associability when it loads** (the amygdala's α of the
  trace then), and the tag fades by 1/16 a step. The hand loop draws its new cells' second region
  in proportion to it, so a surprising, rewarded cue held in mind draws the loop's cells.

**Delayed trace, held states** (overlap between trials):

| | step 2, light / light | step 2, light / none | step 5, light / light | step 5, light / none | step 7, light / light | step 7, light / none |
|---|---|---|---|---|---|---|
| working memory | 0.54 | 0.32 | 0.46 | 0.36 | 0.33 | 0.40 |
| orbitofrontal hold | 0.33 | 0.34 | 0.32 | 0.36 | 0.42 | 0.34 |

- **With coverage the light is clearer in working memory** at its step (0.54 against 0.32), and
  some is left three steps on (0.46 against 0.36). It is gone by step 7.
- **The held states grew to 100–160 bits** (from about 32). Coverage lets a code grow up to four
  times k, and the novel part a load takes is not capped. A held state that large mixes the
  light with everything else that was new.

**Probes, v41:**
- **Delayed, last block:** 0.18/0.15, 0.44/0.35, 0.17/0.09. Not learned.
- **Light probe:** seed 1 at 0.64–0.82 against 0.36–0.48. Seed 0 peaks at 0.75/0.21, then fades
  to 0.05/0.04.
- **Reversal:** seed 0 learns A (0.79/0.33), and after the reversal clicks the old light
  (0.31/0.73). Neither seed reverses.
- **The tone holds** (+0.90 → +0.98; the touch's dopamine +0.53 → +0.34).

**What this says.**
- **The column now does in isolation what it should:**
  - it holds what was new, through silence and a distractor;
  - it never holds the wrong cue;
  - the area below stays driven and clean.
- **In the mind the held state is too big and too mixed.** And the mind's gate, the one thing
  that could hold a familiar, task-relevant light, is chosen by a loop that has no reason yet to
  open it at the light's step.

**Next:**
1. Cap a load at k bits, the most unexpected first (the held population is as sparse as the
   code).
2. Train the gate. It should learn to open when holding pays: the dopamine error at the click's
   outcome should reach the gate's choice at the cue step, three to five steps back. That needs
   either a longer eligibility for the gate's loop, or the held state itself being in the hand
   loop's view, so that the credit lands where the choice was made.

### Superposed held items, then superposition where readers recognize patterns
**v42: the two-area test, how to share the held vector.** Each fix in turn, with the share of
trials whose held state is closer to the right cue at the gap's end:

| version | scene gap | silent gap | scene, distractor | silent, distractor |
|---|---|---|---|---|
| a load capped at 32 bits | 100% | 62–72% | 0% | 68–82% |
| split capacity (a new item half the room) | 54–66% | 92% | 94–98% | 12–48% |
| superposed items, no lifetime | both cues held | both | both | both |
| **superposed items, 8-step lifetime** | **78–80%** | **74–86%** | **74–86%** | **88–96%** |

- **Capping a load at 32 bits:** the distractor, as unexpected as the cue, took the whole held
  state.
- **Splitting capacity** traded one failure for another.
- **Superposed items** (the suggestion): sparse codes barely overlap (32 of 2048 bits), so items
  can share one vector.
  - Each surprise adds an item of at most 32 bits beside what is held, 128 bits in all.
  - Completion restores every learned pattern still mostly present. Its cells fire on each item
    separately, so a union completes as a union.
  - The map learns each item as it is held, so what is held often becomes an attractor.
  - But nothing let go: completion restored the old items too, A and B piled up, and both cues
    looked held.
- **An adaptation lifetime fixed it.** A held bit is restored for only 8 steps after it loads,
  then fades. That lifetime is the area's real persistence.
- Final version: 18–22 of the right cue's bits held against under 1 of the other's. A distractor
  sits beside the cue instead of erasing it. The lower area stays clean.

The prefrontal gates (each association column's "take in", and the orbitofrontal hold) also
keep their eligibility for 8 steps now. A reward three to five steps after the cue can then
reach the gate's choice at the cue.

**v43: superposition where the readers recognize patterns.** Superposition helps where the reader
completes or matches whole patterns. It hurts where the reader samples random bits: the
amygdala's value cells did, until they were made elemental.
- **Layer 2/3 expects every next input its context supports** (`MIND_SUPERPOSE`): the union of
  the cells that burst (or of all that fired, if none burst), up to four. A surprise is then
  something not expected at all, where before it was anything other than the winner's guess.
- **The hippocampus reinstates the runner-up episode's continuation too,** when the cue fits two
  episodes about as well.
- **The striatum's focused context grows to 4096 bits (63 regions).** A motor loop's view can
  have more than 31 input groups, and past 31 two groups shared a region (a dendritic branch).
  This keeps them apart, the opposite of superposition, because the striatum's cells sample by
  region.

**Probes:**

| | v42, seed 0 | v42, seed 1 | v43, seed 0 | v43, seed 1 |
|---|---|---|---|---|
| delayed, best block | 0.62/0.39 | 0.72/0.65 (last) | 0.52/0.31 (first), then dies out | 0.47/0.29 |
| light, best blocks | 0.31/0.12 | 0.51/0.36, 0.24/0.09 | 0.53/0.16, 0.72/0.26, 0.51/0.16 | 0.57/0.14, 0.50/0.07 |
| reversal | **last phase 0.76–0.87 on B against 0.33–0.44** | does not reverse | B partly after the first (0.55/0.36), then fades | – |

- **Delayed on v43, seed 2:** one good block (0.72/0.47), then fades.
- **The tone holds:** CS+ +0.97 against +0.02 (v42), and +1.00 against +0.01 (v43), where the
  dopamine at the touch falls to +0.09.
- **v42 seed 0 is the first clear late reversal.** In the last phase B pays, and it clicks
  0.76–0.87 on B against 0.33–0.44 on A.
- **v43's light probe is the cleanest since v33:** 0.50/0.07 and 0.51/0.16 at the end.

**Why delayed still fails: the held vector is full.** The v42 delayed trace shows working memory
and the orbitofrontal hold at the 128-bit cap on every step. The light is no longer
distinguishable in them: working memory 0.27 light/light against 0.25 light/none, the
orbitofrontal hold 0.37 against 0.31.
- The two-area test was a clean world with one surprise per trial. The mind's world has small
  surprises at every step (the caregiver, its own acts, recall arriving), and each one loads an
  item.
- A bigger vector would not help. The load rate times the lifetime fills any cap. And the
  striatum reads each input group as 16 sampled bits, so a light that is 20 of 128 held bits gets
  2 or 3 of them, and fewer still from a larger vector.

**Next: selective loading, as a claustrum.** Acetylcholine release from the basal forebrain is
gated by salience (the central amygdala drives the nucleus basalis), not by surprise alone. The
claustrum, reciprocally connected with nearly all of the cortex, is linked to resisting
distractors (Atlan et al. 2018). So:
- Each column with a held population proposes its surprise item.
- A claustrum-like selector scores each by surprise × the associability of the cues behind it.
- It lets one item per step load, above a threshold that adapts to keep loads occasional. Others
  are suppressed.
- The gate still loads deliberately.
