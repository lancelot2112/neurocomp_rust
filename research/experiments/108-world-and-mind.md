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

**Next.**
1. **Separate agency from value.** The own-effect signal trains the motor loops (which acts have
   effects) and habituates per act and outcome; the critic learns from outcomes.
2. **Pavlovian–instrumental transfer:** a cue's amygdala value raises the motor loops' readiness to
   release (the brake), as cue value invigorates responding.
3. **Keep cues attended** while their value is learned (the Mackintosh term), and check the delayed
   probe once the immediate one works.
4. **Then the lessons again,** comparing with the first version above.
