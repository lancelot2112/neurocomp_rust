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
1. **Learn what came.** The cortex learns from its last state what was sensed now: the next sound,
   or after a hand act what the hand felt. The motor areas learn each act's result, with no
   babbling phase handed the world's answers.
2. **Value what came.** These innate valuations are the only dopamine:
   - *own effect:* after its own act, as surprising as the sensory column finds the result;
   - *response:* someone present, and a sound from outside within two steps of its own sound
     said aloud;
   - *smile:* an innately recognised pattern in the presence sense;
   - *effort:* every act costs a little.
3. **Sense.** In order:
   - the hippocampus first: what the cortex represents of what came is stored if it came from
     outside, then completed;
   - then every column, the motion stream and the mirror, the association columns, the
     cerebellum, and the relay's mix into a plan.
4. **Choose.** One loop per effector, as the basal ganglia's loops run in parallel:
   - candidates are the cortex's goals that the effector can plan, plus one spontaneously active
     primitive of each effector per step;
   - the loop's gate takes one into the loop (or none), and the effector's motor gate releases it
     or withholds it;
   - withheld, it runs inwardly: imagined through the forward model, and completed by the
     hippocampus.

The mind never knows it is being tested and never stops learning. `examples/lessons.rs` is the
curriculum (speech play, mouse play, lessons) and the observer.

## Results

(running)
