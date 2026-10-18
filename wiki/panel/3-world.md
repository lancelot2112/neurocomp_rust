# Panel memo 3: the world, the body and the interface

## Where the world stands

The world (`src/world/mod.rs`) is four 512-bit senses and two 512-bit acts. Tokens, lights and moves are 16 random lines in one random word, and before `fix_body` they were drawn again in every lifetime. Lifetimes run a few hundred steps. Vocabularies hold two to eight tokens, and the hand has five moves. The UP and DOWN moves exist only for the screen, a grid of at most 32 cells.

After 30 island runs, here is what has been shown. Reflexes and held memory pass (recall to lag16). Learning from reward has not passed in any run: win-stay, the maze's turn and the repeat rung all sit near 0. Echo2 was passed by evolution once, in run 11. A hand-built genome reached echo2 at 300 against a goal of 150, but only after the world was changed: one-word codes, 8 lines, a glyph wire. Reading (glyph to sound, `copy`) stays near 0 even when built by hand.

Computer use and conversation need thousands of symbols, composition, horizons of thousands of steps, and associations learned within a lifetime between arbitrary codes. So the distance is not one ladder. It is several orders of magnitude on every axis at once.

## The central point for "evolve general agents"

Evolution cannot write a conversational agent's content into a genome. On these islands it finds roughly one circuit per tens to hundreds of generations (lesson 37). Language and software need megabytes of learned statistics. Experiment 03 needed 21,000 kernels to reach 50% next-byte accuracy on 20 KB of Alice.

So evolution has to find a learner, and lifetimes have to be long enough for the learner to absorb the world. Two things currently block that, and both are fundamental.

**1. A kernel cannot learn what it writes.** Its output is fixed at birth. Copy learning of outputs was tried and removed: echo2 reached at best 19, and longer lifetimes made it worse. The reading traces show the result. Every glyph-to-sound pairing is made once, by chance, and cannot die. Open vocabulary means arbitrary input codes mapped to arbitrary output codes, learned within a lifetime. Without a local output rule that works, or a births-plus-death scheme that does the same job, no world design will produce conversation. This is a primitive-level gap, and it should be fixed before more grammar options are added.

**2. Fitness does not reward getting better with experience.** In copycheck, echo at 800 tokens scored below echo at 200. A general agent's defining property is that its score rises with lifetime length. Nothing in `observe.rs` selects for that.

## The body/world split: right call, and push it further

Fixing the body (`ISL_BODY`) is correct, and it is how computers work. A keyboard, a font and a screen layout are anatomy. They do not change between users. Much of experiment 16 went into solving a problem the world created: voice codes and sounds drawn unrelated in random words. A topographic kernel cannot pair those (lesson 51).

The motorcheck numbers make the point. One wire scored respond 23, then 197 with place codes for motor, then 515 with stimuli placed too. Make the interface fixed and literal: one byte per keystroke out, raw bytes in. Then echo becomes a wire, as it already is when sound equals voice (461).

What must vary per lifetime is content: the words, what they mean, the file on screen, the task. If fixed codes make reading innate, that is fine. Learning should be forced by the content, not by a scrambled body.

## Interfaces and a ladder

**Byte terminal.** The eye channel is a 64-byte line of a character screen, which fits exactly in 512 bits. The act is one key byte per step, in a sliding window as CLAUDE.md prescribes. Keys typed appear on the screen, which keeps the world honest. Rungs:
- babble keys;
- type the character shown (a wire);
- retype a word shown k steps ago (recall);
- predict the next byte of scripted text, scored as bits-per-byte against an n-gram baseline;
- complete words from a vocabulary drawn per lifetime, growing 8, 64, 512.

Experiment 03 warns that tolerant matching fails on dense ASCII. The sparse front end has to be evolvable or learned. It is a test of the substrate, not a detail.

**Character grid with a cursor.** Generalize `Screen` from glyph cells to a 16x4 text grid with arrow keys. The task family is "make region B equal region A": insert, delete, move a word, pick a menu item by name. This needs no reward. The target is on screen, and the residue (A XOR B) is something the network can sense itself. Lesson 16 says prediction error is the signal, and echo marks were the one thing that unlocked reading (lesson 47). Goal-as-target is the honest, scalable replacement for the stroke, which in 30 runs evolution never learned to use.

**Scripted dialogue.** A teacher states per-lifetime facts ("ka is red"), then asks ("ka?"). This tests one-shot binding (lesson 22). A pretrained LLM may then serve as the teacher, or as the observer's judge, but only outside the network. It is noisy and costly, so it comes late.

**Imitation from recorded computer use.** Fold the recordings into the world as a demonstrator whose keystrokes arrive on a sense channel, then fade the demonstrator as `fade6` fades the teacher. The observer scores whether the network's own keys match what the demonstrator would have done. This fits the observer-only rule. Pixels come only after text works.

## Measuring success

Keep observer-only fitness. Ladders do not scale, though. One hand-designed rung per circuit is hand design in disguise, and the panel should say so. Replace them with procedurally generated task families, with lexicase over many draws, and score on held-out families no ladder contained. Three measures:
- **Held-out bits-per-byte** against n-gram and lookup baselines.
- **Task success on held-out edits** against scripted baselines, each with a blind control (lesson 36).
- **The learning slope:** the same genome scored at 1x, 4x and 16x lifetime length. This becomes a selection case.

## Feasibility, candidly

At current lifetimes and compute, about 3 hours per 600 generations for 8 islands of 32, a conversational or computer-operating agent is not reachable. The bottleneck is not the world. The substrate has not yet shown a within-lifetime learner of arbitrary maps, and evolution has not produced reward learning at all.

Three things are reachable within months, and would be real results:
- a byte-terminal agent that types, copies and edits short text under a fixed body;
- next-byte prediction competitive with n-grams;
- one-shot fact binding.

Conversation needs the learning slope to be positive and to keep rising over millions of steps.

## Next experiments

**1. A byte-terminal island with a fixed ASCII body.** Rungs: babble, echo, retype, next-byte. Score wires and a hand-written learner first (lesson 46). This shows whether a literal interface dissolves experiment 16's problems.

**2. Paired-associate binding with a hand-built check.** Facts drawn per lifetime, using the existing primitives: a held channel, plus births that copy a vector into their output. Then add one candidate output rule, a kernel's output moving toward lines that confirm in its destination, so wrong pairings can die. If no local rule passes by hand, stop and redesign the kernel before evolving more.

**3. The learning slope.** Rescore runs 26 to 30's best reading and prediction genomes at 4x and 16x lifetime length on a growing vocabulary. Then add "score rises with experience" as a lexicase case. If slopes stay flat or negative, evolution is finding reflexes, not learners, and no world ladder will change that.
