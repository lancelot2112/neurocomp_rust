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
selected on what the specialists already do.

**The generalist on ladders of its own** (`ISL_FRONTIER_LADDERS=1`, which replaces the above in run
32): it holds a rung of its own on each other island's ladder, scored in that island's world, plus
one rung below it drawn at random (what it passed stays in selection), and moves up each ladder as
that island would (goal on its elite's mean for the rung's hold, then the held-out check). So it is
no longer scored on rungs it has not reached, only because a specialist reached them. Smoke test: a
random babbler as generalist passes reading's babble held out and moves to echo2q.

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

**Run 31's lag16 pass is real, by a slower latch** (generation 305, held out 158/150). Rescored at
lag16 on 48 fresh worlds, eight genomes (8 to 15 of the checkpoint at 310) score 185 to 223 (lag8:
211 to 287), the other 24 about 0. Genome 14 (`runs/checks/lesions/run31-recall-g14*.txt`) keeps word
0 of a channel (run 31 keeps every channel's lines unless something clears them): light A resets it
to the same 16-line pattern, and it refills by OR a few lines every step or two (16, 36, 41 by step
8, 52 by 14, 60 by 20), the network clicking while it is below full; on other trials it stays full.
The same kind of memory as genome 3's at lag5, a word's fill level fading back, with the refill
slowed so it holds about 20 steps. It works in some worlds and not at all in others (3 of 8 traced
worlds remember 439 to 628; in 5 it never clicks). Neither genome counts with numbers.

**Run 31's prediction passes are real** (cycle3 at 302, cycle4 at 313, then on order2; the first run
past cycle2). The checkpoint at 315, 32 genomes on 48 fresh worlds each (surprise, goal 200):
genomes 0 to 8 but 3 score 177 to 272 at cycle4 (genome 6 best, `runs/checks/lesions/run31-prediction-g06.txt`)
and 289 to 383 at cycle3; 9 and 10, 151 and 157; the rest 76 or less, 17 to 31 about 0. Here the
first-ranked genomes are the ones that work. Order2 (a cycle of six over three tokens, goal 150) is
not solved: best 57, icc 0.00 (no real difference between genomes). (Later, the audit below: these passes are real but not prediction; the old streams let a rare-token detector pass every cycle rung.)

**How genome 6 predicts** (`ISL_TRACE=prediction`, now a token-stream lifetime on a prediction rung;
world 60006, cycle 7 5 7 1, surprise 508; `runs/checks/lesions/run31-prediction-g06-*`). It
predicts in 4 of 6 traced worlds (222 to 508). A small held channel (256 bits, C2) carries the
mismatch: after an odd token it holds about 30 lines a step later, after a kept token under 2; the
network clicks three steps after an odd token 54% of the time, after a kept one 1%. The lesion map
(48 worlds, losses as shares of the goal): the C2-to-hand AND wire 1.06, the C2-to-hand OR wire 0.92,
the hand-to-C2 XOR wire 0.92 (the hand's last act folded into C2, a loop through the body), and a
the eye-to-C2 NOR #188 0.92; C2's own shift wires and a C2-to-eye write help when taken out. (The
genome has two populations with id 188, because a string crossover had copied one; the ear-to-voice
NAND under the same id loses 0.00. Crossover now renumbers repeated ids, and the lesion labels each
population from itself: `run31-prediction-g06-relabelled.txt`.)

**The word trace** (`ISL_TRACE_WORDS=2,11`, C2 and the act, `run31-prediction-g06-words.txt`)
shows a familiarity detector, not a prediction of the next token. The eye-to-C2 NOR fires on the
step after a token it knows: after 1, 5 or 7 (the cycle's tokens) 173 times in 180; after 2, 3, 4
or 6 (tokens the cycle never shows) 3 times in 15. It is silent for the first 9 steps of the
lifetime, then fires, so its mask most likely fits the frequent tokens during the lifetime. While
it fires, C2 sits at a fixed point: the hand's word is all ones, the hand-to-C2 XOR writes it, the
NOR clears it, and C2 stays empty (144 of 172 kept steps). When the NOR is silent C2 is not cleared,
fills (often all 64 lines), and the C2-to-hand wires and the hand-to-C2 XOR pass it around the loop
for two to four steps (ffff..., then a pattern, then its complement) until the hand's word is all
ones again. Clicks happen in that transient. In this lifetime the network clicks within three steps
after 17 of 28 odd tokens and after 12 of 108 kept ones. An odd token that is itself 1, 5 or 7 is
missed: the NOR fires on it. So the genome tells a rare token from a frequent one. It does not know
which token comes next. That is enough for cycle4, where odd tokens are drawn from all eight and so
are mostly ones the cycle never shows. It cannot be enough for order2, where an odd token is a
familiar token out of place, which is the rung run 31 has not solved.

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

## Auditing the rungs (run 31's ladders, run 32's settings)

Do the rungs test what their names say? Two checks: trivial networks scored on every rung, and a
reading of each probe's code.

**Cheater networks.** Ten genomes were scored on all 78 rungs of the 8 ladders, on 48 fresh worlds
each (`ISL_LESION=48`, each rung its own checkpoint). The ten: empty, senses wired to the act, ear to
voice and eye to hand, tonic kernels on the act or the hand (several rates), the self channel wired
to the act, and all ones on the act. Every goal probe is about 0 (within 30) except two rungs built
to be passed that way: reading's babble (a tonic voice 821 against 768) and echo2q (an ear-to-voice
wire 515 against 150; from echo2m8 on, the wire scores about 0). The table and the genomes are in
the session's audit folder, not in the repo.

**The code read** found rungs that trivial networks miss but simple timers pass:

- **distract** (delay 8, the other light flashed during the delay on half the trials): no-cue trials
  showed nothing at the cue's time, so clicking 8 steps after any light scores 380 to 540 (8 worlds),
  against a goal of 150.
- **vary** (each trial's delay drawn from 1 to 8): the window always sits a fixed time after the
  light, and clicks before it cost nothing, so two delay lines from the light score 670 to 810, one
  290 to 470. It did not test "held until the window, not timed".
- Fixed for run 33 (below): win-stay met by a clicker that ignores which light it sees, and the
  self channel's clock word timing a reversal or a block switch. Still open: the held-out
  set is the same 48 worlds at every recheck; under a fixed body the maze's turn rungs can be built
  in.

**The fix** (`ISL_RECALL_FAIR=1`, in run 32): after lag16 the ladder is two (the other light on
no-cue trials), then distract with two lights, its flash being the light the trial did not show
first, then vary with two lights, a tone as the window opens, and a click before the window
spending the trial. The world's test (8 worlds, policies that read the senses):

| rung | holds which light came | timer or latch on any light | delay lines from light A |
|---|---|---|---|
| distract (old) | 1024 | 380 to 540 (clicks 8 after any light) | |
| distract (fair) | 1024 | -37 (8 after any), 0 (latch), 60 (the latest light) | |
| vary (old) | about 1024 (clicks until 12 after) | | 670 to 810 (two), 290 to 470 (one) |
| vary (fair) | 1024 (held to the tone) | 0 (on the tone), 0 (any light held) | 290 to 470 (two: the earlier spends the trial) |

A delay line can still be right on the trials whose delay it matches, so a goal on fair vary must be
over about 500; it is the ladder's last rung, with no goal.

**Partial credit on remember** (`ISL_PARTIAL_REMEMBER=1`, with `ISL_PARTIAL`, in run 32): selection
uses remember~, where a cue trial clicked only in the two steps after its window counts half, against
the same on no-cue trials; the goal stays on remember. World test (lag8): on time 1024 on both, one
step late about 0 on remember and about 512 on remember~, clicking every third step about 0. A trial
spent by an early click (fair vary) scores nothing on either.

**The perception rungs** (code read plus simulation of the stream generator, 600-1500 lifetimes a
cell, surprise in parts per 1024; goal 200, order2 150):

| detector | cycle1 | cycle2 | cycle3 | cycle4 | order2 | cycle8 |
|---|---|---|---|---|---|---|
| sees the break mark (ceiling) | 781 | 778 | 776 | 775 | 777 | 780 |
| rare token (old streams) | 751 | 680 | 603 | 535 | 118 | 323 |
| model of the token before (old) | 700 | 706 | 697 | 695 | **503** | 687 |
| rare token (fair streams) | 749 | 0 | 0 | 0 | 0 | 1 |
| model of the token before (fair) | 716 | 693 | 717 | 716 | **-1** | 708 |
| token one period back compared (fair) | 781 | 760 | 717 | 635 | 438 | 497 |

A break was drawn from all the tokens, so mostly one the cycle never shows: every cycle rung was
passed by rarity, which is what run 31's genome 6 is. And order2's cycle (six of three tokens drawn
at random) was told by the token before alone. **Fair streams** (`ISL_FAIR_SEQUENCES=1`): a cycle
uses each of its tokens equally often (period 6 of 3 is A A B B C C, so each token has two
successors, set by the two before), a break is one of the cycle's tokens that may follow the token
shown before it, and breaks never come two in a row (a second break left a pair the cycle never has:
the model of the token before still scored 189 to 198 until that went). Cycle1 cannot be helped (one
token: any change is new). A detector that compares with the token one period back is prediction of
the cycle, and is meant to score.

Also found and fixed for run 32: **feel**'s goals were under what random moves after a stroke reach
(204 mean, best 273 in 48 worlds, `examples/feelcheck.rs`), and the fixed body makes the click
wireable (a click reflex 1024): now feel-300 then feel 500. **echo and copy** divided by tokens said,
so a genome saying one token a lifetime could pass by luck (an elite of eight on 48 worlds: 1 to 2 in
100 per check); `ISL_FAIR_SAID=1` divides by the tokens said or the tokens shown that count,
whichever is more. Found and not changed: under the fixed body echo2-8's sound-to-voice map can be
built in (the docs said learned in the lifetime); babble is met by any noisy voice; shapes' blur
rungs keep the two lights in different words in 7 worlds of 8, so a fixed location reflex passes
tell; the held-out set is reused at every check. Fine: copy and pairs (no leak: only untaught tokens
count), screen, the lowered-goal steps, the training and held-out worlds kept apart.

**Win-stay and the clock** (for run 33). Win-stay compared, after a click on a light, the click
rate on that light's next trial when the click was stroked against when it was not. A clicker that
ignores which light it sees and clicks while a stroke is recent met it: the stroked light pays
again, so its next trial comes while the stroke is still recent. `ISL_FAIR_WINSTAY=1` takes off the
same rates on the other light's next trial and halves the difference
(`win_stay_of`, test `fair_win_stay_cancels_a_clicker_driven_by_any_recent_stroke`, 24 worlds):

| policy | old | fair |
|---|---|---|
| recency clicker (any light within 16 steps of a stroke), no swaps | 314 | -308 |
| the same, swaps every 50 trials | 319 | -291 |
| learner (a light unless its last click on it went unstroked), no swaps | 1024 | 1017 |
| the same, swaps every 50 trials | 1024 | 742 |

The self channel's word 2 counted steps from 0, so a network could time the reversal at trial 100
(near step 1300, give or take about 20) or a selection block's end. `ISL_CLOCK_DRAWN=1` starts it each
lifetime at a number drawn from the lifetime's seed: traced on two worlds, the clock read 1 at step 2
in both with the switch off, `...bc1a` and `...8d8a` with it on, still one up a step.

## Phase codes against frame codes (`examples/phasecheck.rs`)

Can a fixed circuit of word operations tell a break in the token stream, and does it need phase?
Policies that read the eye's history as a network does (a token's phase: its glyph folded to a
byte, modelled as a byte drawn per token per lifetime), their clicks replayed into the real
generator's events and scored by surprise (graded in brackets), 48 worlds
(`runs/checks/phasecheck.txt`):

| stream | rung | rare | frames | window | comb | oracle |
|---|---|---|---|---|---|---|
| old | cycle4 | 521 | 7 | 2 | 656 | 783 |
| old | order2 | 93 | -28 | 0 | 493 | 782 |
| fair (run 32) | cycle4 | -2 | -31 | -24 | 636 | 778 |
| fair (run 32) | order2 | 2 | **766** | **749** | 459 | 788 |
| fair, varied orders | order2 | -1 | 0 | 2 | 664 | 930 |

- **frames** is a fixed rule on the last three frames by XOR (a break when "this token is the one
  before" answers as a step earlier); **window** the same rule read from byte lanes of one word
  that holds the last eight phases (shifted left 8 each step, the new phase ORed in): phase as
  position in a word. Both learn nothing, and both passed run 32's fair order2 (goal 150), because
  its cycle always had the shape A A B B C C. **Order cycles of varied shape**
  (`ISL_VARIED_ORDERS=1`): each token twice in a random order, all pairs different, no token twice
  in a row, and a break keeping the pairs with the tokens before and after it. Then the fixed rules
  score about 0, a model of the token before -1 and one of the two before 863 (the stream
  simulation, `freq7.py` in the session's scratchpad).
- **comb** is a phase accumulator: each token's phase added into a wrapping byte lane, a click when
  the sum of the last `period` phases changes. It is interference at the cycle's period, and it
  works on every cycle rung (459 to 758) because the rung's period is the same every lifetime: it
  predicts each token from the one a period back, which is real prediction of a cycle (a delay of
  the period, as a conveyor or the history ring gives). It also fires when a break leaves the
  window, which the graded score counts (over the oracle on cycle4).
- **Phase is not needed on these rungs**: a delay of one period (or the two tokens before) is
  enough, and the history ring or a conveyor gives it. Phase as position in a word (window) is the
  same computation in one word; phase accumulation (comb) is a delay line by another name. Phase
  codes would matter where the context is longer than the ring or the period varies.

**The rest of the audit, for run 33** (each behind its own switch; world tests in `observe.rs`):

| leak | fix | test |
|---|---|---|
| the same 48 held-out worlds at every check (a lucky genome passes after enough checks) | `ISL_FRESH_HELD=1`: new held-out worlds for each check | |
| maze turns wireable under the fixed body (cue on the left half, turn left) | `ISL_MAZE_MIRROR=1`: in half the lifetimes the cue's half is the side that does not pay | a wired turn 1024 as it was, -1024 mirrored (about 0 over lifetimes); a learner that flips after an unstroked choice 1024 and 1003 |
| shapes' blur rungs told apart by which word is lit | `ISL_SAME_WORD=1`: with blur or shift the two lights share a word | a detector of the eye's first half on tell under blur 2: 554 in their own words, 0 in one word |
| food from the observer's probe scores (the observer's judgement reached the program's energy) | `ISL_FOOD_WORLD=1`: an eighth of `food` for each stroke the world gives; no strokes, the ration only | |

Left as designed: under the fixed body echo2-8 test a built-in vocal map (the voice's sounds are
anatomy; the copy rungs, with glyphs drawn each lifetime, are learned), and babble asks only for a
varied voice, which noise gives. Food from strokes means reading and prediction lifetimes live on
the ration alone: watch their network sizes in run 33.

**Why nothing passes cued** (selection's first rung, select@600; `ISL_SELECT_PARTS=24` on run 32's
checkpoint at about 350): all eight of the elite babble in the reading blocks and never click,
respond 0, babble 981 to 1024, the voice in conditioning 32 to 47, clicks 0: select 457 to 464. To
reach 600 a genome must click at the lights too, and a click reflex fires at the glyphs in the
reading blocks as well: select = (1024 + 1024) / 2 - 1024 = about 0, far below babbling alone.
Only a reflex gated by the task cue scores (1024), two changes at once: a valley. **Partial credit
on select** (`ISL_PARTIAL_SELECT=1`, in run 33): selection uses select~, the costs at a quarter
(test `graded_select_climbs_from_babbling_to_clicking_before_the_task_gates_it`, 12 worlds):

| policy | select | select~ |
|---|---|---|
| babbler (reading blocks only, never clicks) | 512 | 512 |
| reflex (also clicks at anything the eye shows) | 3 | 769 |
| selector (clicks only in conditioning blocks) | 1024 | 1024 |

**Selection and prediction stuck at their first rungs (for run 34).** No run's selection island has
ever held cued's goal (select@600) on its training worlds: in run 33 at 100 the elite still babbled
and never clicked (respond 0, babble 981 to 1024, select 470 to 492), though select~ makes an
ungated click reflex worth 769. `ISL_SELECT_STEP=1` puts a step before cued, mixclick: the mixed
lifetime scored on responding in its conditioning blocks alone (goal 100), so clicking at the lights
pays before it has to be gated by the task. Prediction's first rung is not hard: run 32's
generalist passed cycle1-100 at generation 7 with one population (an eye-to-hand NOR of kernels that
expect zeros, 361 on 24 fresh worlds under run 33's settings), while the prediction island, its
networks grown to 64 looped populations, scored 13 and never passed it in 600 generations. Search,
not the test. `ISL_MIGRATE=1`: when the generalist passes a rung on an island's ladder the island has
not passed, its best two genomes are copied into the island's next population, to win or lose
there. Smoke test (the passer as the generalist, run 33's prediction elite as the island): two
migrations, the island past cycle1-100 at 2 and cycle1 at 3 (held out 622 and 379).

**The islands' best into the generalist** (`ISL_MIGRATE_UP=k`). Plasmid exchange,
the islands' old way of sharing circuits, has been off since run 22. Now every k generations each
island's best genome, and an island's best two whenever it passes a rung the generalist has not
passed on its ladder, join the generalist beside its own children, not in their place, up to
`ISL_GENERALIST_ROOM` (run 34: each island's best four every generation, room 64), scored with
them; breeding brings it back to its size, so a migrant that wins a parent's or an elite place
carries on. Lexicase keeps a specialist that wins some of the generalist's cases. Smoke test (run 32's one-population cycle1
passer as the prediction island, 8 genomes): prediction passed cycle1-100 at 1 and sent two up, the
generalist passed it at 2 and cycle1 at 3.

## Is there a gradient, does anything learn, and what would select learners? (run 33 at 235)

Three audits, with the goal stated as general-purpose learning agents.

**Gradient** (`examples/gradcheck.rs`, `GC_KIDS>0`; numbers in `runs/checks/gradcheck/`). From each
island's elite: 200 single mutants, 100 children bred as lexicase breeds them, 50 three-step walks,
all scored against the parent on the same 32 fresh worlds, gains re-scored on 32 new ones.

| island, rung | parent (sd over worlds) | mutants worse / neutral / better | score unchanged | landscape |
|---|---|---|---|---|
| recall, lag5 | 0 (0) | 4 / 94 / 2% | 55% | climbable: a bred child +174 confirmed, a walk +67 |
| switching, tell | 0-5 (0-22) | 0 / 98 / 2% | 44-47% | rare jumps (1 in 200, +105, marginal) |
| reading, pairs8 | -2 (46) | 3 / 97 / 0 | 86% | flat |
| shapes, blur1 | 25-78 (169-246) | 0 / 100 / 0 | 87% | flat, noisy |
| screen, torch | 21 (50) | 2 / 98 / 0 | 38% | flat (differences are noise) |
| prediction, cycle1-100 | 40 (149) | 8 / 92 / 0 | 71% | rugged: only worse |
| maze, feel-300 | 207 (167) | 22 / 78 / 0 | 63% | rugged: walks fall step by step |
| selection, cued | 505 graded (1.4) | 10 / 84 / 6% | 78% | plateau with cliffs (+3 gains, -504 falls) |

Where children do differ, the noise between worlds (sd 50 to 250 on maze, prediction, shapes, screen)
is larger than any one-step gain (30 or less); simulated on the measured data, lexicase on 24 lives
gives a zero-gain child with its own noise about as many children as a child 1 sd better: on noisy
rungs being different pays about as well as being better. **No parent scores higher in a longer
lifetime** (half against twice the trials, every difference within 1.3 se); the one rise is
reading's echo (430, 653, 748). The elites react; they do not learn their rungs.

**The world** (re-audit under run 34's switches, read only). Most rungs need no learning in the
lifetime: every prediction rung (the period is the same each lifetime, so a fixed delay
comparator passes), respond, feel, babble, echo, lag1 to lag16, the screen's navigation rungs, and
selection's rungs (glyphs are 8 lines in word 0, lights 16 lines in a random word, so "click when the
eye holds many lines, talk when word 0 holds few" passes cued, uncued and short). Rungs that do need
learning: the swaps (relearning), discriminate, win-stay, recall's two, distract and vary, symbol,
pick and crowd, pairs, read, and the mirrored maze (one bit). Fixed policies on rungs with a drawn
mapping now score +1024 or -1024 a lifetime, and with fresh held-out sets and a hold of 1 each check
is a new draw: a lottery pass every 25 to 70 generations. Also: pairs' silent first 50 tokens count
against copy (a one-shot learner's ceiling about 400); food from strokes may starve stroke-free
lifetimes; the grammar draws a kernel's motor primitive each lifetime, so a wired click may not be
expressible; touch feels moves that do nothing.

**The literature** (design review): open-ended worlds (Tierra, Avida, Polyworld, Geb, Lenia) gave
diversity and arms races, not learners; learning evolves when the world changes between lifetimes
but is predictable within one (Stephens 1991; Dunlap and Stephens 2009), and where it does not
change, learning is wired in (Hinton and Nowlan 1987): this project's lessons 41, 53 and 55. The
evidence for transferable in-lifetime learners is task distributions drawn per lifetime with
held-out tasks (meta-RL, Duan et al. 2016, Wang et al. 2018; XLand and AdA). POET and minimal
criterion coevolution (Wang et al. 2019; Brant and Stanley 2017) address missing gradients by
generating environments. Recommended: a varied-contingency world in the current harness before any
shared arena; the ladders kept as held-out measurement, scored by a learning slope (late minus early
in a lifetime) and a plasticity lesion (learning off: what learning bought).

**The learning battery** (`ISL_LEARN_BATTERY=n`, measure only; `runs/checks/learn-battery/`). Nine
rungs that need learning in the lifetime (switching discrim, winstay and swap1, recall two, reading
pairs8, maze turn and symbol, screen pick, shapes paying), each scored on n fresh worlds for each
island's best genomes and a network that does nothing: the probe over the lifetime, the slope (its
last quarter less its first), and the gap (learning on less the same lifetime with learning frozen,
`program::neurocomp::set_frozen`: no mask moves, births, copies, output moves, teachers or
exploration). Checked on a known learner, the evolved pairs reader: pairs8 373 on, -61 frozen, a gap
of **434** (se 10). (Births read a copy of the grow gene on the edge group; the first freeze missed
it and gave a gap of 0.) On pairs the slope is inflated for anything that copies, as the teacher is
silent for the first 50 tokens; there the gap is the measure.

Baseline, 24 worlds, each island's best two (19 genomes, 9 tasks): **no genome learns any battery
task**. In run 32's final checkpoint the largest gap is 29 (se 11); at run 33's 350, 37 (se 19, the
generalist on win-stay) and 18 (se 7, a recall genome on pairs8). With about 170 comparisons a run, a
few near 2 to 2.6 se are what chance gives; against the reader's 434 they are nothing. The elites'
battery scores are near 0, and where they are not (shapes on discrim, 83) frozen scores the same.
This is the floor every claim of a general learner has to beat.

**Step 2, in run 34**: what stayed the same from lifetime to lifetime drawn per lifetime. Twin worlds
(`ISL_TWINS=1`): seeds 2k and 2k+1 live one world, the second with its two lights' roles swapped and
the maze's cue mirrored, so a policy fixed to a light or a side scores the opposite on its twin.
The delay (`ISL_VARY_LAG=1`) and the period (`ISL_VARY_PERIOD=1`) drawn from 1 to the rung's for each
lifetime. Mixed lights as one-word codes (`ISL_MIXED_CODES=1`), as the glyphs are. Check: run 33's
shapes elite, a fixed reaction scoring 83 on discriminate (83 frozen), scores -30 frozen over twin
pairs (24 worlds): the reaction mostly cancels; not to exactly 0, since the network's state differs
when the codes' roles swap. What each code means was already drawn per lifetime (lights, glyphs,
cues); twins make a fixed answer to one of them pay nothing.

**A novelty island with species** (`ISL_NOVELTY=1`, in run 34). No goals: its genomes are selected on
how new their behaviour is (`observe::behaviour`, read from the log: click rates in conditioning and
in token streams, voice rates in each, clicks in the three steps after a stroke and after an odd
token, other moves, tokens said; every kind of lifetime is lived), the mean L1 distance to the 8
nearest in the island and in an archive (the two most novel each generation join it). A novel genome
whose mean on some probe, babble aside, reaches 200 and whose behaviour is at least 512 from every
species founds one: its founder is kept every generation and the island grows by four genomes, up to
`ISL_NOVELTY_MAX` (128). Its most novel join the generalist with `ISL_MIGRATE_UP`; the generalist
never climbs it. Smoke test, started from run 33's genomes (8 a generation): five species in four
generations (useful on respond 1000, 735, 524, 760 and remember 211), the island grown to 23. From
nothing every behaviour is 0, so the island starts on drift. Expect many species to be reflex
variants useful on respond; the 512 apart rule limits how many.

**More branches** (`ISL_BRANCHES_MANY=1`, for run 35). A population could carry one branch, a second
vector its fired kernels OR their outputs into. Now it may carry up to four more (`tags::Branch`),
each its own: a target (a channel or an output port, by tag), a write op, a rotation of what is
written, and a trigger, a gate on a channel or a port (it writes only while that vector had lines,
or none, a frame ago), or none. One firing reaches several places, each written its own way and in
its own context. A new population has one one time in four; a mutation adds (up to four), drops or
changes one (target bit, op, rotation, trigger). Checked: an eye-to-hand population with one more
branch into a private channel fills it with 54 to 80 lines a step, 0 without.

**Selection's longer ladder** (`ISL_SELECT_MORE=1`, runs 34 and 35). Run 33's selection island
passed cued (held out 641) and uncued at 565-568, the first selection passes of any run: graded
select did make it click (respond 0 to 350-500, clicks in reading about 0). Most likely it told the
blocks apart by the codes' look (glyphs 8 lines in word 0, lights 16 in a random word), which
`ISL_MIXED_CODES` removes in run 34. The ladder after uncued: short with a goal (select 600), then
**held** (the task cue felt only for a block's first 3 steps and the blocks' tasks drawn at random,
not alternating, so the task must be held through the block: selection joined to recall), then
**three** (a third task, still: glyphs shown, neither click nor voice; select becomes (respond +
babble + still) / 3 less the costs, still being 1024 less the rate of acting in still blocks).
Events carry the block's task for the observer only (`Event::task`); the world now draws three task
cues (the first two as before). Test `held_three_tasks_need_the_cue_held_through_the_block`, 12
worlds: a holder of the last cue felt 1022, a two-task selector (babbling in still blocks) 680, a
network acting only while the cue is felt 699 (inaction earns the still third, so three, the last
rung, has no goal).

## How run 33's selection passed cued (a deep dive, `runs/checks/selection-dive/`)

The elite that passed cued at 566 (14 populations; on 24 fresh worlds select 667, respond 506,
babble 1024, voice in conditioning 96, clicks in reading 1) does not read the task cue or hold a
task, and it is not the codes' look: with lights drawn like glyphs (`ISL_MIXED_CODES`) it scores 692,
on uncued 629, on short 620. Traced on uncued (`ISL_TRACE`, which now plays a mixed lifetime on a
selection rung, and logs each step's voice and the block's task): it **clicks when the eye was dark,
then lit, then lit with the same code** (106 of 120 such steps in conditioning blocks, 2 of 1458
others), a pattern a light makes coming out of darkness and a glyph stream, never dark, does not.
Its voice follows continuous seeing (45% of steps after four lit ones; about 10% after a dark one in
conditioning blocks, the small voice cost it pays). Lesions: two C2-to-voice XORs (0.48, 0.47), an
eye-to-hand XOR (0.16), an eye-to-hand clear (0.11), an eye-to-hand set wire and a hand-to-C2 clear
wire (0.06 each). Selection by the stimulus's timing: the rungs allow it because the two tasks show
the world differently.

The best at 860 (73 populations, select about 520, below the goal) found something else: a period-2
oscillator (#613, C9 to C9 NOR wire, a loss of 2.3 goals when removed) shares the steps between the
hand and the voice, both reflexes to a lit eye a step ago: it never clicks the step after C9 was high
(0 of 919), clicks on the other phase, and talks mostly on the high one; it clicks in reading blocks
too (about 330 to 460 per 1024 steps).

For run 34's ladder: held, two tasks, can still be passed by stimulus timing; three cannot, because
still blocks show the same glyph stream as reading blocks and only the cue tells babble from quiet.
A selection test proper wants every task's blocks to show the same stimuli, the cue alone saying
what to do.

**The same stimuli in every block** (lesson key `same=1`, on held and three in `ISL_SELECT_MORE`,
runs 34 and 35). Every block, whatever its task, is a conditioning block (lights in trials); clicks
pay only in respond blocks (`World::no_pay`); select's parts are taken by each step's task
(`Event::task`): responding in respond blocks, the voice's variety in babble blocks, the voice in
respond blocks and clicks in babble blocks as costs, still as before. Test
`with_the_same_stimuli_only_the_held_cue_tells_the_task` (12 worlds, the cue felt 3 steps, tasks
drawn): a holder of the last cue 1022, run 33's passing trick (click at lights, babble always) -257,
a network acting only while the cue is felt 515.

**Why maze and screen swing** (run 33 from 600 to 900): maze's goal score has a spread of 48 and a
generation-to-generation change of 44, about all of it the noise of 24 new training worlds a
generation (the gradient audit measured maze's spread between worlds at about 167, so 34 by chance
on a mean of 24); screen's spread, 54, is twice its step change, 29: a slower real drift, a lineage
gaining (to 197) and then lost, since nothing locks a gain short of a pass. Not the networks' size:
these islands' best have 1 to 8 populations.

**Long shifts** (`ISL_LONG_SHIFTS=1`, for run 35). Every write op acted within a 64-bit word, so a
pattern could travel a vector only through a motion channel (a conveyor, a rule-90 wave) or a chain
of channels. Two new ops, `lsl:k` and `lsr:k` (`KernelOp::Lsl`, `Lsr`, `kernel::class::long_shift`),
shift what is written by k bits as one long bit string, the bits leaving a word entering the next,
then OR it in: still one cheap operation (each word takes its own shifted plus its neighbour's
spill). A kernel's write may spill a word past its mask; a wire's wraps round its destination.
Check: a channel fed from the eye with an `lsl:16` self-loop wire carries each step's code up 16
lines a step, across the word boundary into the next word by the sixth step. A new op is one one
time in seven. (Run 33's genomes already held many motion channels, 48 rule-90 waves on screen and
74 conveyors on prediction, but removing them from the genomes that passed rungs changed nothing.)

## How run 33's screen passed torch (`runs/checks/screen-dive/`)

Torch: two cells, the target's glyph shown only while the cursor is on its cell, the cursor's mark
always shown; a click ends the trial. The passer (generation 934; 322 on 48 fresh worlds, goal
300) is six wires and populations and no channel of its own, and two of them carry it: eye-to-hand
wires writing `sub:3` (byte-lane subtract; lesion losses 0.49 and 0.53), each running only while the
eye had lines a frame ago and the voice had none. Traced (`ISL_TRACE` now plays a screen lifetime on
a screen rung): **it never moves the cursor**. It clicks when glyph and cursor are seen together
(100 clicks, every one on the target) and otherwise does nothing until the trial runs out. In two
cells the cursor starts on the target about half the time, so a click-when-seen reflex scores about
512, above the goal: torch, meant to ask for search, is passed without it.

How two wires make the reflex: subtracting the eye's picture from the hand in byte lanes turns each
seen line into a byte of high lines (0 - x wraps), and the hand reads a command as the move whose
word has the most lines, at least half a code's. The cursor's mark alone (8 lines) gives too few
for any move; glyph and cursor (23 lines) give enough to read as the click. A threshold detector
from wrapping arithmetic and the body's own decoder, with no kernel at all.

The fix (`ISL_CURSOR_AWAY=1`, runs 34 and 35): on every screen rung but the dark ones (glow), the
cursor never starts a trial on the target, so every hit takes a move. The passer, 48 fresh worlds:
322 without it, -341 with it.

**Partial credit on the screen** (`ISL_PARTIAL_SCREEN=1`, runs 34 and 35). With the cursor starting off
the target, searching earned nothing until a click followed: a sitter that clicks at what it sees
scores 0, a wanderer 0, only a searcher that moves and then clicks scores. Selection now uses
screen~: a trial in which the cursor reached the target without a hit counts half a hit (a hit is
worth cells - 1, its click less; half that). Test
`graded_screen_pays_search_before_the_click_on_what_it_finds` (two cells, torch, 12 worlds):

| policy | screen | screen~ |
|---|---|---|
| sitter (clicks at a glyph, never moves) | 0 | 0 |
| wanderer (moves at random, never clicks) | 0 | 512 |
| searcher (moves until it sees the glyph, then clicks) | 1024 | 1024 |

Events carry whether the cursor is on the target after the step's move, for the observer only
(`Event::on_target`).

## How run 33's screen passed two (`runs/checks/screen-dive/two-*`)

Two: two cells, the target's glyph always shown, the cursor's mark always shown. The screen elite
held out 265 at generation 1029 and 310 at 1030 (goal 300) and moved on to row. The rung log's
`passed two` block (generation 1031) does not hold that elite: all eight of its genomes, two to
seven populations each, score 0 on `two` on 48 fresh worlds, with run 33's binary and the current
one alike. The log wrote the elite a generation late, when the island was already scored on `row`,
where nothing scores and the thrift case puts the cheapest genomes first. The fix is in
`examples/islands.rs` (runs 34 and 35): the dump is the previous generation's elite, the one that
passed.

The passer's line lives on: the screen island's current genomes (generation 1055, on `row`)
score 172 to 334 on `two`. The best, 14 populations (`two-relative-g1055.txt`), held out 334 on
48 fresh worlds. Lesioned, its score rests on eye-to-hand wires: `add:5` (losses 0.15, 0.15 and
0.29 of the goal), `add:6` (0.47), `clear` (0.34) and `nand` (0.21). It is the torch passer's
design again, widened: byte-lane arithmetic from the eye into the hand, read by the hand's decoder
as a click only when enough lines are on. Traced (world 60003, 200 trials): **it never moves**.
It clicks when glyph and cursor are in one cell (23 lines in one eye word) and never when they are
apart (8 lines and 16), so 100 clicks, all on the target, in the half of the trials where the
cursor starts on it. With the cursor starting off the target (`ISL_CURSOR_AWAY=1`) it scores
-242. So two, like torch, was passed by the sitter reflex. The cursor-away and screen~ switches
already in runs 34 and 35 are what turn the screen into a search task.

| genome (48 fresh worlds) | two | two, cursor away |
|---|---|---|
| rung log's first `passed two` genome (2 populations) | 0 | -1024 |
| the other seven in that block | 0 | |
| the screen island's best at 1055 (14 populations) | 334 | -242 |

## Run 33's end (stopped at 1118 for run 34)

Run 33 ran open-ended from 6:44 AM ET and was stopped by hand at generation 1118 (4:16 PM ET).
31 rung moves, 9 of them the generalist's on its own ladders, all by generation 214:

| island | last rung passed (generation) | how, where we looked |
|---|---|---|
| switching | respond (11) | |
| recall | lag16 (469; lag8 at 445), on two | |
| reading | echo2q (33), on pairs8 | |
| prediction | cycle2 (519; cycle1 at 498), on cycle3 | fair streams: the first passes past cycle1 since they were made fair |
| maze | feel-300 (325) | |
| screen | two (1030), on row | torch and two by the sitter reflex: it never moves, and clicks when glyph and cursor share a cell |
| shapes | clean (22) | |
| selection | uncued (568), on short | cued and uncued by the onset-timing reflex (click at dark-lit-lit; babble while seeing) |
| generalist | its ladders 1,4,2,0,0,0,2,0,0 | |

Every move after 568 was screen's, and both were the sitter. The two hard islands we took apart
(selection, screen) passed by cheap reflexes the rungs did not rule out, and run 34 carries the
fixes for both (same-stimulus held and three rungs for selection; the cursor starting off the
target and screen~ for screen), and the voice one-word everywhere, the novelty island and the
islands' migrants into the generalist. The rung log's island dumps were a generation late through
run 33; runs 29 to 33's are being rescored and corrected (`runs/islandsNN/rungs-fixed.txt`).

## Does prediction need partial credit? (run 33's cycle3)

Prediction has had partial credit since run 24: selection uses surprise~, a click 4 to 6 steps
after a break counting half. Run 33's prediction island sat on cycle3 (a fair cycle of three
tokens, breaks 1 in 8) from generation 519 to its stop at 1118. Its final population, 24 fresh
worlds (`ISL_SIGNAL=24 ISL_SIGNAL_EACH=1`): surprise mean -4, best 25 (goal 200); the spread
between genomes is all noise (icc 0.00, split-half r -0.16). It had also lost what it passed: its
best on cycle2 was 18.

Strategies played on the stream itself (test
`held_frequent_breaks_do_not_help_a_partly_right_detector`, 24 worlds, 400 tokens):

| strategy | cycle3 | breaks 2 in 8, each held 2 steps |
|---|---|---|
| lag-3 comparator (click when a token differs from the one 3 back) | 718 | 489 |
| the same, right a third of the time | 273 | 279 |
| same token as the step before | 827 | 554 |
| clicking at random | -17 | 0 |

So surprise already pays a partly right detector in proportion: right a third of the time passes
cycle3's goal. Held and frequent breaks, meant as an easier cycle3, add nothing to it and cost the
full detector (its false alarms after each held break fall on steps counted as kept); that rung
was dropped. Cycle3's trouble is not the shape of the credit: no genome compares a token with the
one three steps back at all.

A leak found on the way: in a fair cycle of distinct tokens (period up to the token count) no
other token follows the one before, so a break is another of the cycle's tokens, and it may be the
token before itself; a cycle of distinct tokens never repeats, so a sameness detector scores 827
on cycle3, and every cycle2 break is a repeat. Run 33's population did not use it (sameness is
hard for kernels that count lines on). Not fixed yet.

Run 35 adds a step, cycle3-100 (`ISL_STEPS=...,cycle3=100`), which a lag-3 detector right about
one time in seven passes. Run 34 already draws the period per lifetime (`ISL_VARY_PERIOD=1`, 1 to
the rung's), so on cycle3 lag-1 and lag-2 comparators score in part of the lifetimes: that, not
held breaks, is the graded path to a lag-3 one.

**Shaped surprise (`ISL_PARTIAL_SURPRISE=1`, run 35).** Surprise gives nothing to a network that
clicks at all until its clicks follow breaks more than other steps: never, always and at random all
score 0, so a population that never clicks has no slope to climb. surprise+ (`observe::surprise_shaped`,
probe 31, for selection only; the goal stays on surprise) pays a quarter for the share of breaks
followed by a click (in 3 steps; 4 to 6 late counts half) and three quarters for that share less
the same after steps with no break. Test `shaped_surprise_pays_clicking_then_clicking_on_breaks`
(breaks 1 in 8, 12 streams):

| strategy | surprise | surprise+ |
|---|---|---|
| never clicks | 0 | 0 |
| clicks at random, 1 step in 4 | -6 | 161 |
| clicks always | 0 | 256 |
| a third of the breaks | 303 | 321 |
| every break, and 1 step in 4 at random | 344 | 421 |
| every break | 779 | 740 |

Clicking pays first; from clicking always, each false alarm dropped while the breaks are still
clicked raises the score. A click on every break tops out near 740, since a late window also catches
the next break's click.

## The novelty island's surprise species (run 34, `runs/checks/novelty-dive/`)

Run 34's novelty island founded species 4 at generation 53 for being useful on surprise (261, in
its plain world: period 3, breaks 1 in 8, the period drawn per lifetime with `ISL_VARY_PERIOD`).
Scored on prediction's cycle3, the island's 62 genomes at generation 85 (24 fresh worlds) are near
0 but one, 16 populations and 7 channels, which scores 301 (goal 200).

It is not a cycle3 predictor. With the period fixed (48 fresh worlds each):

| period | surprise |
|---|---|
| 1 (cycle1) | **669** |
| 2 (cycle2) | -10 |
| 3 (cycle3) | 7 |

Its cycle3 score is the third of lifetimes whose drawn period is 1 (on 48 other worlds, period
drawn, it scores 58: fewer of them). On cycle1 it is a strong detector of a token unlike the one
before, far past what the prediction island, still on cycle1-100 at generation 85, has reached.

Four populations carry it (each lesion takes the whole score): eye to C3 by `set` kernels, C3 into
itself by a NOR wire, and C3 and C2 into the hand by NAND wires. Traced (cycle1, world 60003, 21
breaks in 200 steps): it clicks 2 steps after a break on 17 of 21 breaks (inside the 3-step window)
and on 3 of 106 steps far from any break; otherwise it moves left, which costs nothing in a token
stream. C3's NOR self-wire makes a 2-step oscillator (its lines alternate about 120 and 360), and
the kernels' write on a changed token shifts its phase, so the click lands at a fixed phase after
the change: an onset detector, as run 33's selection reflex was, which works only while the token
otherwise stays the same.

Why the novelty island found it: its behaviour descriptor includes clicks just after odd tokens
(b-odd) and other moves (b-moves), so search for new behaviour explores exactly this, and the
species rule kept it as useful. Two lessons. The species label can mislead when the period is
drawn: the novelty island's surprise mixes periods, so a cycle1 detector looks like a cycle3 one.
And nothing carries it where it is useful: novelty migrants go only to the generalist, so the
prediction island, stuck below cycle1, never sees a genome that would pass cycle1 at once.

## Sending novelty's species to the islands, and a leak it found (`ISL_NOVEL_MIGRATE`, `ISL_VARY_UP`)

`ISL_NOVEL_MIGRATE=1` (run 35): a new species' founder is copied into every island whose current
rung is scored on the probe it was useful on. Smoke test, run 34's checkpoint at 120 with the
prediction, novelty and selection islands: at 123 a species useful on surprise (273) was sent into
prediction, which then passed cycle1-100 (125, held out 609), cycle1 (126, 428), cycle2 (127, 233)
and cycle3 (132, 222) in ten generations, after 120 on cycle1-100.

Too fast. Rescored with the period fixed (48 fresh worlds), the cycle2 and cycle3 passers are the
cycle1 detector: 438 and 293 with the period drawn per lifetime (`ISL_VARY_PERIOD`), -10 and 5
with it fixed. Drawing the period from 1 to the rung's lets a detector of a shorter cycle score in
the shorter lifetimes, enough to pass. Run 34's recall has the same leak through `ISL_VARY_LAG`:
its lag8 passer scores 260 with the delay drawn, 0 with it fixed at 8.

The fix, `ISL_VARY_UP=1` (run 35): the delay and the period are drawn from the rung's to twice it.
A shorter memory or detector scores in no lifetime; one wired to the rung's value alone, in about
half. The same passers drawn up: cycle3 passer 24 to 27, lag8 passer 0, while run 34's lag1 passer
(a real one-step memory) keeps 90 to 204 (goal 150).

So run 34's recall climb past the first rungs and any prediction pass under the drawn period are
not what they seem; its rung moves on those ladders should be read with this in mind.

## Run 34's end (stopped at 179 for run 35)

29 rung moves in 179 generations, 15 of them the generalist's on its own ladders, all but two by
generation 49 (run 33 had 16 by then): recall to two (82; the generalist on recall's ladder to two
at 128), switching to feel-300 (35), reading to pairs8 (49), shapes to blur1 (26), selection past
its step rung, mixclick (21); prediction, maze and screen passed nothing. Its rung log was the
first written in the generation that passed, and every recall passer rescored reproduces its
pass, but recall's passes from lag3 on were the drawn-delay leak (above). Generations slowed from
about 25 seconds to 98, as the novelty island grew: 39 species, from generation 148 on nearly all
founded for respond (200 to 897), some for remember (239 to 626) and one each for tell and
surprise. Run 35 (started 7:37 PM ET) adds more branches, long shifts, the cycle3-100 step, shaped
surprise, novelty's species sent to the islands and the delay and period drawn from the rung's up.

## Run 35's maze: feel and learn passed, by activity, not plasticity

Run 35's maze passed feel-300 (27), feel at its raised goal of 500 (47; the first island to) and
learn (57; repeat what was stroked, the cue's side mirrored in half the lifetimes, so a wired turn
scores about 0). Rescored on 48 fresh worlds, with learning on and frozen (`ISL_FROZEN=1`, new:
no mask moves, births, copies, output moves or exploration in any lifetime; measure only):

| passer | learning on | learning frozen | goal |
|---|---|---|---|
| feel | mean 548, best 596 | 521, best 577 | 500 |
| learn | mean 351, best 478 | 361, best 516 | 300 |

Both are real on fresh worlds and neither needs plasticity: feel is the click after a stroke, and
learn, though the side that pays changes between lifetimes, keeps its score frozen, so it holds
which move was stroked in its activity (a win-stay of memory, not of weights). Adaptation within a
lifetime, but not the learning the battery looks for.

## Is anything using the graded credit? (run 35 at 60)

Selection's lexicase cases use the graded probe in place of the plain one (`graded`), so prediction
is selected on surprise+ and screen on screen~; the log's tables show only a rung's own probes, so
they do not show it. The populations at generation 60, 24 fresh worlds (`ISL_SIGNAL_PROBE`, new: a
probe other than the rung's goal):

| island, rung | probe | mean | best | icc |
|---|---|---|---|---|
| prediction, cycle1-100 | surprise (the goal) | -6 | 38 | 0.03 |
| | surprise+ (selected on) | 118 | 214 | 0.27 |
| | b-click-seq (clicks in token streams) | 151 | 263 | |
| screen, torch | screen (the goal) | -50 | 0 | 0.23 |
| | screen~ (selected on) | 281 | 483 | 0.30 |

Both credits are in use and give selection a gradient the goals do not: prediction's networks now
click in the token stream (about 15% of steps), which surprise alone never paid, but their clicks
do not yet follow the breaks; screen's networks move the cursor onto the target in many trials
(screen~ 281) and do not yet click on it there (screen below 0: their clicks miss).

## Which triggers are used (run 35 at 60)

Run 35's 3,454 kernel populations: at-least 85%, at-most 9%, int-learned 2%, lanes< 2%, int>=
under 1%; band, tile, int<=, int~ and lanes> none. Only maze (at-most 36%) and switching (lanes<
20%) use another widely; a new population always started at at-least, so the others came from
mutation alone. Lesioned (each island's best, 24 fresh worlds), every population that matters uses
at-least or is a wire, but for switching's two lanes< populations (voice to act, NOR; losses 0.37
and 0.43 of its goal). `ISL_TRIGGER_DRAW=1` (run 36) draws a new population's trigger among all
ten, and its lane width, so the others are tried as often as the default.
