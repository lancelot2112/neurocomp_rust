# 17. The panel, and a learning rule that learns pairs

## The question

The stated goal (2026-10-17): **evolve general agents using bit math**, agents that can operate
a computer and converse, produced by evolution over bit-vector networks. No floats, no
gradient-trained network as the agent; a pretrained model may appear only as a teacher or as
part of the world. After runs 29 and 30, what limits the current path, and what should change?

## The panel

Four subagents, each reading the code, `CLAUDE.md`, the wiki and the run logs. They worked
read-only and wrote a memo from one angle each, then read the others' memos and answered them.
The memos are kept in [`../panel/`](../panel/):

| Angle | Round 1 | Round 2 |
|---|---|---|
| The primitives and the learning rules | [1-architecture](../panel/1-architecture.md) | [1-architecture-r2](../panel/1-architecture-r2.md) |
| The evolutionary search and the curriculum | [2-evolution](../panel/2-evolution.md) | [2-evolution-r2](../panel/2-evolution-r2.md) |
| The world, the body and the interface | [3-world](../panel/3-world.md) | [3-world-r2](../panel/3-world-r2.md) |
| Learning theory, neuroscience and scale | [4-theory](../panel/4-theory.md) | [4-theory-r2](../panel/4-theory-r2.md) |

### Where all four agreed

**The diagnosis.** The substrate can represent an agent: a kernel is a threshold unit with
binary weights, and AND-NOT and XOR make it universal. What it could not do is learn arbitrary
input-to-output maps within a lifetime, and evolution cannot supply what the runtime cannot do.
Four reasons:
1. **A kernel's output was fixed at birth.** A pairing made once, by chance, could never be
   corrected (the reader of experiment 16: eight glyph kernels, all writing token 3).
2. **Credit reached populations, not kernels.** A gate says when a whole population learns, not
   which kernel's act caused what followed.
3. **The learning rule was not in the genome.** Evolution could tune the runtime's rules (rates,
   gates) but not change them.
4. **Selection rewarded skills, not learning.** Ladders pay for what can be built in, so
   evolution builds it in. Runs 29 and 30 move fast on such rungs, then stall where a code must
   be learned within the lifetime. The variance measurement after run 20 found no difference
   between genomes on echo2 and cycle1 beyond noise.

**Feasibility.** Running a brain-sparse agent in real time would fit on one workstation, within
an order of magnitude or two (a rough estimate). Evolving its knowledge would not, by many
orders. Evolution contributes a few bits a generation; it can find a learner, and the learner
must do the rest from experience. The only reachable route is meta-learning: evolve the learner
on short, varied, cheap tasks, then check that it keeps learning in long lifetimes.

**Methodology.** Every run in [island-runs](../island-runs.md) is a single evolutionary seed,
each stacking one switch on the run before, so conclusions such as "size classes help" are
hints, not results. From now on a new mechanism is compared on 3 evolutionary seeds.

### What changed in round 2

- **The fixed body.** Memo 3 defended it (anatomy is fixed, as keyboards and fonts are); memo 4
  argued it lets evolution build in what should be learned. All four settled on: keep the body
  fixed; draw every mapping the agent must use (what glyphs mean, which act pays, facts,
  vocabulary) afresh each lifetime; score on held-out content; and add a learning-slope case.
  Memo 3 added an observer rule: a rung counts only on worlds with permuted mappings.
- **Memo 2 withdrew a red flag.** It had blamed the fixed body for run 30's zeros on feel-100;
  run 29 shows the same zeros without a fixed body. (Memo 4's round 2 still cites the flag; the
  withdrawal stands.)
- **Memos 1 and 2 changed the order.** They had wanted writable masks first. All four agreed on
  one mechanism at three levels, smallest first, each shown by hand before the next:
  1. a fixed three-factor output rule (eligibility, a modulator, an output move, and death for
     wrong pairings);
  2. the rule's conditions as a gene, seeded with that rule, so evolution varies a working rule;
  3. masks as writable channels, so a rule can be any circuit (the largest search space, last).
- **Memo 2 put the substrate ahead of the noise floor**, its own first demand, since no number of
  replicate seeds turns zero signal into a gradient.
- **The first gate:** a paired-associate capacity curve (memo 1's), scored at two lifetime
  lengths, with a blind control and a hand-written table as a ceiling. Next-byte prediction
  against an n-gram is the decisive later gate for language. A byte terminal and dialogue worlds
  come after that.

## What was done, and the numbers

1. **The capacity curve** (`examples/pairscheck`, `runs/checks/pairscheck-capacity.txt`). N
   glyph-to-token pairs (2, 8, 32, 128), glyphs drawn each lifetime, the teacher on six tokens in
   eight, scored by copy (goal 100) on 24 worlds at 400 and 3,200 trials, under run 30's body. A
   lookup table written in the harness scores 148 to 243 and gains with experience. Every network
   scores about 0 beyond N = 2: run 30's best reading, generalist and recall genomes, and the
   hand-made echo genome. **The gap was confirmed.**
2. **The three-factor output rule** (`EdgeSpec::elig`; [experiment 16](16-reward-learning-and-the-genome.md)).
   A kernel that fired k steps ago is checked while its gate is open, against what its
   destination holds then. Right, it counts up; wrong, an output line moves and it counts down;
   wrong at 0, it dies and its input is free for a new birth.
   - **With a right target** (each token heard like its voice code, so one wire echoes the
     teacher exactly), a wire plus a reader learns 128 pairs: copy 195 at N = 128 over 3,200
     trials, against the lookup's 237 (a lookup later found unfair: see 6); 143 to 147 at 400 trials; the same on 24 other worlds. The
     same reader without the rule scores 0. **The first network to learn arbitrary pairs in a
     lifetime.**
   - **With the hand-made genome's own listener as the target**, copy reaches only 26: the reader
     learns, but only the 3 or 4 tokens that listener can echo.
3. **Rule tables** (`EdgeSpec::rule`, 16 bits; seed `0x9405` is the runtime's own rules;
   `runs/checks/pairscheck-rules.txt`). The seed reproduces every number exactly.
   - **Death is what fixes wrong pairings:** without the output move the reader scores the same.
   - **Without death** it learns faster in short lifetimes (222 at N = 32, above the lookup's 213)
     and forgets in long ones (52 at N = 8 over 3,200 trials).
   - **Anti-Hebbian firing** breaks learning.
4. **Run 30** (a fixed body, 8-line one-word codes; stopped at 266): 24 rung moves, 7 past rungs
   run 29 never passed (recall to lag16; screen glow and torch; the maze feel-100 and feel;
   prediction cycle1-100 and cycle1; shapes feel, repeat and win-stay). Reading sat on echo2q all
   run.
5. **Run 31** (run 30 with `ISL_ELIG=1` and `ISL_PAIRS=1`, reading's pairs8 and pairs32 rungs
   after echo2q): started 9:09 PM ET. Reading passed echo2q at 24 and is on pairs8.

6. **The 3-seed pairs test** ([`runs/pairs-test-1/README.md`](../../runs/pairs-test-1/README.md)):
   reading alone on pairs8 > pairs32 > pairs128, 200 generations, 3 seeds, three arms (no rule,
   the rule, the rule as a table). No arm with the rule passed pairs8. The one pass, without the
   rule, was an echo wire that the island's held-out check scored copy 340 to 548 on pairs8 and
   pairs32. **Not a leak** (retracted): the genome that passed pairs8 (`runs/checks/evolved-reader/`;
   the case written up in [experiment 18](18-an-evolved-reader.md)) reads. Its eye kernels are born on each glyph copying the move being made, the echo of the
   teacher naming it, and write by XOR, so the code cancels the echo on taught steps and is said
   alone on untaught ones. Against a lookup table made fair (silent after a taught step) it reaches
   957 of 990 at N = 8 and 912 of 1014 at N = 32 over 1600 trials, and 678 of 1020 at N = 128; the
   hand-made reader with the three-factor rule about 190. The island lost it at pairs128 (200-trial
   lifetimes are too short for 128 pairs: 147 there), and the best genome at the end only echoes.
   The earlier "80-100% of the lookup table" for the rule was against an unfair table (a fifth of
   the fair one).
7. **Triggers** (`EdgeSpec::trigger`, `ISL_TRIGGERS=1`, one popcount and a compare each): a
   population's kernels count agreeing lines (at least, the default), their silent lines (at
   most), agreeing lines up to halfway between threshold and full (a band), or agreeing lines less
   those on outside the mask in its words (the whole tile, an XOR). Built and unit-tested; not yet
   checked on a task.

8. **The pairs test on 6 seeds** ([`runs/pairs-test-2/README.md`](../../runs/pairs-test-2/README.md),
   pairs8 > pairs16 > pairs32, goal 300): the reader found once in 18 runs (no rule, seed 2: pairs8
   at 10, pairs16 at 14); with the rule 0 of 12, without 1 of 6. Rare under every arm.

## What it means, and what is next

The panel's central gap, learning an arbitrary map within a lifetime, is closed by hand for the
first time, with three local bit operations: an age count, a gate, and death. Two things remain:
- **The target.** The rule learns whatever the destination holds when the check comes; reading
  needs a listener that echoes every token. Under the fixed body that is something evolution
  can build in.
- **Evolution.** Does evolution find the rule's timing in run 31, and improve on the seed table
  (`ISL_RULES=1`), compared on 3 evolutionary seeds?

Later, in the panel's order: a learning-slope selection case, permuted-mapping worlds as the
observer's rule, masks as writable channels, next-byte prediction against an n-gram, then a byte
terminal and dialogue.

## Motions: conveyors and travelling waves (`Token::Motion`, `ISL_MOTIONS=1`)

A delay a network can use, asked for after run 31's start: a channel that moves by itself each
step, before any population writes.
- **A conveyor** (`motion tag k 0`): the channel's words move one place along every 2^(k-1) steps
  (word i gets what word i-1 held, word 0 is emptied). Word j then holds what was written j moves
  ago: a delay line as long as the channel, up to 2,048 steps. Conveyors of different strides give
  delays on time scales a factor of two apart. A population covering the channel sees every delay
  at once.
- **A travelling wave** (`motion tag 0 1`): each line becomes the XOR of its two neighbours' (a
  frame ago), cellular rule 90 across the whole vector; a written pattern spreads out as a wave.

Both are a copy and a shift of words, done once a step (`neurocomp::moved`, unit-tested). Before
this, the only delays were a population's lag (1 to 8 frames), its span (the OR of up to 64
frames: that something happened, not what or when), held channels, and shifts inside one word. A
new token is a motion one time in about twenty under `ISL_MOTIONS`, for one of the genome's
channels; mutations flip its tag bits, step its stride (0 to 8) and toggle its wave. Not yet
tried on a task.

## Why the generalist babbles (run 30's best, generation 265)

The generalist is scored at the frontier: on each other island's current rung, the ones no network
has passed. Its held-out scores (generalist.csv) are almost all babble (about 1,000: say every token
at least once) and select (about 400); every frontier subtask is near 0 for every genome. Lexicase
has nothing to choose on but those two, so the generalist keeps a babbling path and drifts. In run
31, where reading starts on echo2q without a babble rung, its total fell from about 1,400 to about
300.

Its best network has 232 populations on 11 vectors, with 404 modulatory inputs (137 pauses, 119
gates, 105 inhibit inputs). 77 populations write back into their own channel, 70 of them on one
4,096-bit channel, mostly clear, and and nor. The lesion map (24 fresh worlds,
`runs/checks/lesions/run30-final.txt`) finds all 232 "matter", but with losses of both signs: taking
out one ear-to-channel population raises echo by 86% of its goal and lowers win-stay by 78% of its.
Every part perturbs every subtask (a mean share of 0.41 on its top subtask, 0.08 if spread evenly):
a tangle where any change helps one subtask and hurts another, not parts that each do a job. The
self-loops cost little: a clear loop empties its channel, and populations whose input is empty are
paused and do no work.

**Scored on passed rungs too** (`ISL_FRONTIER_PASSED=1`): besides each other island's current rung,
its last passed rung, in its own world, joins the cases as rows of its own, so the generalist is
selected on what the specialists already do. Run 32 has it.

## Are the loops memory? (recall, runs 30 and 31)

- **The generalist does not remember** (remember about 0 at lag16 on held-out worlds all run 30).
  Taking out its 77 self-loops one at a time raises remember more often than it lowers it (19
  against 2, at most 12% of the goal).
- **Run 30's recall island has lost its memory.** Its best genome at the end, and the one the rung
  log kept when lag8 was passed at 209, score remember 0 on 24 and 96 fresh worlds; all 32 of
  its genomes at the end: mean -1 at lag8, best 20, split-half r 0.32
  (`runs/checks/lesions/run30-recall-signal.txt`). The lag8 pass (held out 162/150) was made by
  the elite's mean, and the rung log kept only the first-ranked genome.
- **Run 31's recall population does remember, but not its first-ranked genome.** On 48 fresh worlds
  at lag5 its 32 genomes score mean 25, best 303 (split-half r 0.91, icc 0.34: real differences);
  one by one, genome 3 scores 303, genome 5 158, five more 56 to 93; genome 0, the one every
  "best genome" tool picks up, scores 0. The ranking the run keeps is not by remember.
- **Genome 3's memory path** (`runs/checks/lesions/run31-recall-g03*.txt`, 48 worlds, remember 381):
  an OR wire from C1 into the hand, gated and paused while C7 is empty (taking it out costs 3.7
  goals of remember), and two kernel populations from C7 into C1 (tonic 1 in 16, kernels that
  expect zeros: about 2 goals each). C7's own self-loops are clears, and taking them out raises
  remember a little. So the memory is not held in a self-loop; where the delay is carried (a
  lag, the clock, a held channel) needs a step-by-step trace.

**Where the delay is carried: a held channel that the light empties and that refills** (a trace
of one lifetime, `ISL_TRACE=recall`, world 60014, remember 861;
`runs/checks/lesions/run31-recall-g03-trace-60014.txt`). Genome 3 remembers in 15 of 16 worlds
(remember 174 to 861; one world 0). Per step of a trial, over 92 cue trials and 108 others:

| step | cue: C3 lines | cue: clicks | other: C3 lines | other: clicks |
|---|---|---|---|---|
| 0 (light A) | 47 | 17% | 44 | 18% |
| 1 | 0 | 15% | 45 | 16% |
| 2 | 6 | 14% | 45 | 13% |
| 3 | 11 | 98% | 46 | 9% |
| 5 | 20 | 90% | 47 | 8% |
| 7 | 27 | 70% | 49 | 5% |
| 9 | 33 | 53% | 50 | 3% |

The store is one 64-bit word, word 0 of C3 (the genome's one held channel; its other 15 words stay
empty; `ISL_TRACE_WORDS=3`, `runs/checks/lesions/run31-recall-g03-trace-60014-words.txt`). The step
after the light the whole word is cleared to 0. It is refilled by OR, not by an add (each later
pattern holds the earlier one's lines), in one or two chunks of 15 to 25 lines at a step that varies
from trial to trial: in one trial at step 6, in another not before the trial ends. The network
clicks exactly while the word is empty and stops the step it refills. The steady 4 to 5 lines a step
in the table is the average over trials of these jumps. So the memory is a latch the light resets
and a chance event sets again (the refill most likely from the kernels that fire spontaneously,
one step in 16), not a counter: it holds that a light came, for a random few steps, which is why it
climbs to lag5 and fades at lag8 and lag16.

From run 32 the rung log keeps the whole elite that passed (`ISL_RUNG_LOG`), best first.

**Integer triggers** (`trigger` 4 to 6, gene `lane`, under `ISL_TRIGGERS`): a kernel reads the lane
(8, 16, 32 or 64 bits) that holds its mask's first line as an unsigned number v, and compares it
with c, its threshold's share of its lines times the lane's largest value (a threshold stays a
fraction of the kernel's own mask): at least (v >= c), at most (v <= c), or near (within an eighth
of the lane's range). Met, its count is all its lines. With ADD and SUB in lanes a population can
count; with these another can act when a count reaches a value: a timer that counts, where run 31's
recall genome has only a latch with a random hold. Unit-tested; not yet tried on a task.

**A learned value and two lanes compared** (`trigger` 7 to 9). Trigger 7: a kernel keeps its own
number; it fires when its lane is within an eighth of the lane's range of it, and each time it
fires the number moves a quarter of the way toward what it saw (a newborn takes the first value it
sees): a cell tuned to a count. Triggers 8 and 9: the mask spans two lanes of a word, the one with
its first line and the one with its last; it fires when the first is greater (8) or less (9) than
the second: a count against a set point another population writes. Integer triggers (4 to 9) keep
their mask where it is (it names the lane): a firing or held-down kernel moves no mask line, and a
learned value learns by its number. Unit-tested; not yet tried on a task.
