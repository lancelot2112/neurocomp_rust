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

**Next (from the first probes).**
1. **Separate agency from value.** The own-effect signal trains the motor loops (which acts have
   effects) and habituates per act and outcome; the critic learns from outcomes.
2. **Pavlovian–instrumental transfer:** a cue's amygdala value raises the motor loops' readiness to
   release (the brake), as cue value invigorates responding.
3. **Keep cues attended** while their value is learned (the Mackintosh term), and check the delayed
   probe once the immediate one works.
4. **Then the lessons again,** comparing with the first version above.
