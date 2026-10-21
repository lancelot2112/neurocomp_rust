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

## Are branches used? A branch that carries recall's respond (run 35, `runs/checks/branch-dive/`)

Most of run 35's populations carry a branch (generation 95: 72% in switching, 82% in recall, 86%
in prediction, 58% in selection), and many of the populations that matter in the islands' best
genomes are branched. A lesion takes a population and its branch together, so recall's best
genome (generation 65, 28 populations, 17 branched) was scored with branches taken out instead,
48 fresh worlds:

| recall's best | respond (goal 100) | lag3 (goal 150) |
|---|---|---|
| as evolved | 483 | 106 |
| no branches at all | -207 | 117 |
| #51 without its branch | 24 | |
| each of the other 16 without its branch | 436 to 491 | |

One branch carries respond. Population #51 reads the voice (its own output port, a frame ago) with
five two-mask kernels, runs only while the eye had lines a frame ago (a pause on the eye), and
writes CLEAR into a channel of its own; its branch, on the hand, ORs the same kernels' outputs
there. So the click at a light is a collateral: a cell reading what the voice said, allowed to
run only while something is seen, whose second axon drives the hand. Without the branch the cell
still runs and the hand stays still. On lag3 the branches matter little (106 as evolved, 117
without). The dashboard's network diagrams do not draw branches yet.

## Run 35's end (stopped at 175 for run 36)

25 rung moves in 175 generations, none after 137. Maze passed feel-300 (27), feel at its raised
goal of 500 (47, the first island to) and learn (57; real on fresh worlds, the same with learning
frozen), screen glow-150 and glow (56, 57), recall lag1 to lag3 (17 to 137) with the delay drawn
from the rung's up, so its passes cannot be shorter memories, switching respond and tell (17, 106),
reading babble and echo2q (8, 107), shapes clean (104), selection mixclick (19). Prediction passed
nothing: surprise+ made its networks click (b-click-seq 151) but not on the breaks. Long shifts
were in the five most written ops of six islands by 100; branches carry real work (recall's
respond rests on one). The novelty island founded 64 species, nearly all useful on respond alone,
grew to its cap of 128, and generations slowed from 68 seconds (39) to 223 (175). Run 36 (started
12:26 AM ET) caps it at 32, founds species only for new combinations of useful probes or for a
probe none covers, draws new populations' triggers and keeps a species log.

## Run 36's recall: a long memory that needs learning (`runs/checks/recall36/`)

Run 36's recall passed lag4 at 262 and lag5 at 272, under delays drawn from the rung's up to twice
it. Three of the eight genomes the rung log kept for lag5 (77 populations each) score, on 48 fresh
worlds (the other five 0; goal 150):

| | learning on | learning frozen |
|---|---|---|
| lag5, delay fixed at 5 | 731 to 777 | |
| lag5, delay drawn 5 to 10 | 572 to 585 | 195 to 205 |
| lag8, delay fixed at 8 | 485 to 519 | |
| lag8, delay drawn 8 to 16 | 321 to 336 | 55 to 66 |

(the lag4 passers: 820 to 869 at 4, 538 to 560 drawn 4 to 8). A real memory, past its own rung,
and the first evolved network whose score rests on learning in the lifetime: frozen (no mask moves,
births, copies or output moves) it keeps about a third on its own rung and a fifth on the longer
delays. Every passer checked before (run 35's maze feel and learn included) scored the same frozen.

## Run 36's selection: four rungs in five generations, by acting on neither task (`runs/checks/selection36/`)

Run 36's selection passed cued (503), uncued (506), short (507) and held (508). The passers are
real on fresh worlds and need no learning (held, 48 worlds: 718 to 756 learning, 717 to 752
frozen; goal 600). Traced (held, world 60003), the passer does the same thing in both tasks:

| block | cue | steps | clicks | speaks |
|---|---|---|---|---|
| respond | shown | 81 | 4% | 19% |
| respond | gone | 436 | 8% | 22% |
| babble | shown | 303 | 8% | 22% |
| babble | gone | 1,709 | 7% | 23% |

select = (respond + babbled) / 2 less the voice rate in respond blocks and the click rate in babble
blocks. A network that clicks at every light (respond 1024: clicks on paying trials against
others) and speaks now and then everywhere (babble 1024: each token said once in all the babble
blocks) pays only its sparse rates, 22% and 8%: 1024 - 221 - 81 = 722. The rung tests had a timing
reflex speaking at every step (voice cost 1024, under 300); speaking sparsely is the leak. A fix
to try: score selection on the contrast between tasks, the voice in babble blocks less in respond
blocks and the clicks at paying lights in respond blocks less at the same lights in babble blocks,
so a network blind to the task scores 0.

Then recall passed lag8 (529) and lag16 (530, delays 16 to 32) in two generations. The lag16
passers score 1024 on 48 fresh worlds with the delay drawn or fixed at 16, and the same frozen.
Traced (world 60003): on a cue trial it clicks from the second step after the cue to the trial's
end, 26 steps running; on a no-cue trial never. A latch: set by the cue, reset by the trial's end,
blind to the delay. The remember probe counts a click in the window on cue trials against no-cue
trials, and on the lag rungs a click before the window costs nothing (only vary spends the trial
on an early click), so a latch scores 1024 at any delay. It holds one bit for the trial in its
activity, which is memory, but not recall timed to a delay; the lag5 learners (272) were another
lineage. A fix: an early click spends the trial on every recall rung.

## Recall as cues and questions (`ISL_RECALL_FLAG=1`, run 37)

On the lag rungs nothing in the world said when the answer window opened, only the trial's timing;
with the delay drawn per lifetime a network could learn it only from when strokes came, so
evolution found shorter memories (run 34) and a latch (run 36). The new world (`World::flag`) has
no trials: a cue (a light on the eye) now and then, and a question (a tone on the ear) now and
then. Click in the three steps after a question if a cue came since the last question; not after
two questions with no cue between; nowhere else. Before about half the questions a cue came once
or twice; quiet gaps of 1 to `fgap` steps between any two events, so how long the cue is held is
drawn, and the question says when to answer. A correct click is stroked.

`flag` = 1024 x (answered after a cue / such questions - answered after none / such questions -
stray clicks / questions). Test `flag_pays_a_bit_held_from_the_cue_to_the_question` (gaps to 8, 12
lifetimes): holder (set at the cue, answer and clear at the question) 1024, clicking at every
question 0, a latch clicking from the cue to the question -1024, clicking 3 steps after any cue
-249, at random -946, never 0. Run 36's recall population (the latch lineage) scores -480 on flag2
and -646 on flag8 (icc 0.93). The ladder: respond, flag2, flag4, flag8, flag16 (the gaps' limit),
fdistract (the other light in half the gaps), ftwo (which light is the cue drawn per lifetime,
learned from the strokes); goals 300.

## The attractant: unread changing inputs ask for a connection (`ISL_ATTRACT=1`, run 37)

In run 36 practically no population's learning was gated on the affect port (0% in every island
but prediction's 1%), so nothing in a network treats a stroke as a reward. A gate is aimed at
random; the affect port is one of six ports and ports one target in three. The attractant
(`program::neurocomp::set_attract`): each input port (eye, ear, touch, affect, the self channel)
that no population reads counts, in a `u8`, the steps its lines changed; when the count reaches a
population's own threshold (2^(its accident gene - 10), 4 to 256), the population with the lowest
threshold grows a trial connection from the port into its destination, a predicting (XOR) trial
as exploration grows, and the count starts again. A trial whose predictions come true KEEP times
over the generation is kept as an explored population reading that port (`take_notes`, kinds 8
to 12). Like a growth cone following a chemical, nothing in it says what the input means.

Test (run 36's settings, switching and recall, 12 generations from naked): 17 explored
populations reading ports kept (recall: ear 11, eye 2, self 2; switching: self 2), none without it.
It grows inputs, not gates: a gate's use cannot be judged locally, so reward gating is not built
by it.

## Continuous worlds for the other ladders (run 37)

Each behind its own switch, off by default (recall's is `ISL_RECALL_FLAG`, run 37):

- `ISL_WRONG_SIGNAL=1`: a wrong click is felt the step after, as the stroke follows a right one:
  every other line of the affect channel's first word, after a click on a light that does not pay,
  a click with no light to answer (in a trial's gaps or outside its window), a click after a
  question with no cue or away from any question, a turn to the maze's wrong side, a click off the
  screen's target. The network gets both outcomes, not a stroke or silence.
- `ISL_STRAY_COST=1`: switching's and shapes' respond, discriminate and reverse less the clicks
  made away from any light's window, per trial (`observe::stray_rate`). Test: a responder keeps
  1024 on respond, a constant clicker -1024.
- `ISL_SELECT_CONTRAST=1`: select and select~ as the contrast between tasks
  (`observe::select_contrast`): clicks in respond blocks against babble blocks and the voice in
  babble blocks against respond blocks, each (a - b)/(a + b), averaged (with still blocks, acting
  elsewhere against acting there, a third). Test (held, 12 lifetimes): a selector 1021 (old score
  1022); run 36's passer's strategy, clicking at every light and speaking now and then everywhere,
  11 (old score 569).
- `ISL_SCREEN_STREAM=1`: a click does not end the trial; a hit (or the time running out) moves the
  target to another free cell and the distractors with it, the walls and the cursor stay, and a
  miss goes on (screen and screen~ now at least -1024). Test (4 x 1, cursor off the target, 20
  targets): a searcher 20 hits, 699; a constant clicker no hit, 459 wrong signals, -1024.
- `ISL_MAZE_STREAM=1`: maze trials one after another with no gap. The lighter of the two maze
  plans: a continuous foraging world, with food appearing at places the cue tells, is not built.

The conditioning world (switching, shapes) was already a stream of lights at random gaps with the
light as the prompt; what it lacked was the wrong signal and the stray cost. Smoke test with all of
them, recall's and the attractant on (switching, recall, maze, screen, selection, 6 generations): no
errors.

## Run 36's end (stopped at 828 for run 37)

35 rung moves in 828 generations. The drawn triggers took hold at once: by 50 most islands read
mostly by lanes compared (reading 74% lanes>, prediction 91%, shapes 85% lanes<), learned integers
(recall, screen, the generalist) or int>= (selection), against 85% at-least in run 35. The novelty
island founded 9 species, each useful on a probe or combination no other had, and generations held
near 50 seconds (run 35: 223). Recall's lag5 passers needed learning; its lag8 and lag16 passes were
a latch; selection passed four rungs blind to the task; prediction passed cycle1-100 at 602;
switching reached feel. Run 37 (started 12:12 PM ET) answers the leaks: recall as cues and
questions, selection by contrast, stray clicks charged, wrong clicks felt, the screen and the maze
continuous, and unread inputs attracting trials.

**Misses cost (`ISL_MISS_COST=1`).** In run 37's first 120 generations recall's best network was
empty and it, like switching (until 112), sat below respond: with stray clicks charged, a network
that clicks at random scores below 0 and one that does nothing scores 0, and the thrift case
prefers it. With the miss cost a prompt that called for an act and got none is a wrong answer too:
respond, discriminate and reverse less the share of paying lights not clicked, remember and flag
less the cued trials or questions not answered, maze less the trials with no choice, the screen
less the trials with no hit, selection by contrast -1024 for never acting. Test
`misses_cost_so_doing_nothing_is_no_refuge`: a responder keeps 1024 on respond; never acting
scores -1024 on respond and on flag.

## Run 37's selection under the contrast score (`runs/checks/selection37/`)

Cued from 27 to 232 (population mean -16 to 323, mostly 30 to 200), then uncued (233) and short
(234) in one generation each (held out 643, 651, 601); on held from 235 the population sits at
about 0 (-15 to 15). The short passers, 48 fresh worlds: 638 to 723 on short, -16 to 19 on held. On
cued, uncued and short the tasks' stimuli differ (lights for respond, glyphs for babble), so
reacting to the stimulus passes; on held they are the same, and the contrast gives a task-blind
network 0, as built. But a trace (short, world 60003) shows the weakness: contrast 814 with the
old respond part at 30. It clicked a few times in respond blocks and never in babble blocks, and
(a - b)/(a + b) of click rates gives that full credit. A fix: contrast task performance, respond
measured on paying lights in respond blocks against babble blocks, the voice in babble blocks
against respond blocks, so a few clicks score little.

**The performance contrast (`ISL_SELECT_CONTRAST=2`, run 38).** `observe::select_performance`:
responding (paying lights clicked in their window, less clicks in the control window) in respond
blocks less in babble blocks, and the voice rate in babble blocks less in respond blocks, averaged;
with still blocks a third for acting less there than elsewhere. Test (held, 12 lifetimes): a
selector 1021, the task-blind strategy 4, a timid selector (one light in ten clicked, a token one
step in twenty) 165, where the rates' contrast gives it nearly full credit.

## Has quality gone up with the continuous worlds? (run 36 at 828 against run 37 at 258)

Rungs: run 37 had 18 moves by generation 258, run 36 20 and run 35 25 by the same point, on rungs
made harder to fake. The learning battery (`runs/checks/learn-battery/`; each run's own world,
24 fresh worlds, each island's best 4): genome-task pairs whose score clearly rests on learning
(on less frozen above 50 and above twice its standard error):

| | pairs | largest gap |
|---|---|---|
| run 36 (828) | 3 of 360 | 87 (se 21) |
| run 37 (258) | 19 of 320 | 276 (se 145) |

At this threshold about 7 of 320 would pass by chance, so run 37's 19 is a modest signal and run
36's 3 is none: learning starts to matter in run 37 (switching's discrim and swap1, shapes' paying,
maze's symbol, screen's pick), most in the recall, novelty and generalist genomes. The task scores
themselves stay low (the best full score on any battery task 136), and run 37's scores are not on
run 36's scale (stray clicks and misses cost differently). A plausible cause is the wrong signal: a
learner now feels both outcomes.

## Stacked rungs: each rung's world holds the rungs before it (`ISL_STACKED=1`, run 38)

Switching's tell is scored in the lights world and its feel rungs in a world of strokes with no
lights, and its repeat rung in a maze of no length; shapes' ladder the same. A tell-passer starts
feel at 0, and the climb chart drops there. Kept rungs give it partial credit in selection, but the
skills are learned side by side, not built on each other. Stacked: the feel rungs move into the
lights world itself, strokes at random in the gaps between trials (one step in eight), scored by
feel-gap (`observe::feel_in_gaps`: clicks in the three steps after a gap stroke against clicks at
other gap steps) beside respond and tell in the same lifetime; every later rung keeps the strokes
and feel-gap; the repeat rung goes (win-stay is its lights-world form). A click after a gap stroke
is an answer, not a stray (no stray cost, no wrong signal). Test
`feeling_in_the_lights_world_pays_a_stroke_follower_and_keeps_tell` (10 lifetimes): clicking at
light A and once after each stroke keeps tell at 1024 and scores 366 on feel-gap (each of the three
steps after a stroke counts, as in feel); light A alone 1024 and 0; clicking every step 0 and 0.
Run 37's switching population, on feel-300 in the old world, scores about 0 on the stacked feel
(-2 to 6): it keeps its tell and starts the stroke part from nothing. The maze is stacked too: its feel rungs in the junction-only maze its learn rung uses
(every step at the junction, where a click does nothing), strokes at random while it waits (one
step in eight, never on the step a choice is made nor a trial's first, where an earned stroke
lands; marked `Event::free_stroke`), scored by feel-maze (`observe::feel_in_maze`); every later
maze rung keeps them. Test `feeling_in_the_maze_pays_a_stroke_follower_and_keeps_the_turn` (10
lifetimes): turning to the cue's side and clicking after strokes 1004 on maze and 380 on
feel-maze; turning alone 1024 and 0. Run 37's maze population (feel-300 passed in the old world)
scores up to 96 on the stacked feel (mean 10): some of the old skill carries over.

## Run 37's end (stopped at 331 for run 38)

18 rung moves in 331 generations: reading to pairs8, switching to feel-300, shapes to blur1, maze
to feel, recall to flag2, selection to held, none after 235. Stray clicks charged with misses free
made doing nothing a refuge early (recall's best empty for 120 generations); selection passed the
rungs whose stimuli tell the task and stopped at held; recall's flag2 score rose steadily
(-336 to -116). Learning clearly mattered in 19 of 320 battery pairs at 258 (run 36: 3 of 360).
Run 38 (started 3:45 PM ET) adds the miss cost, selection by the tasks' performance contrasted and
stacked rungs on switching, shapes and the maze.

## Run 38's end (stopped at 112 for run 39)

9 rung moves in 112 generations: reading to pairs8 (4, 28), selection cued to held (38, 49, 56,
60), switching to tell and recall to flag2 (both 77), shapes to blur1 (102). The miss cost ended
the refuge of doing nothing: switching and recall passed respond at 77 (run 37: 112 and 147).
Selection under the performance contrast climbed cued to held in 22 generations and sits at about
0 on held (-26 at 112), as in run 37. But the full miss cost saturates: recall's best was empty at
-1024 on respond to generation 60, and on flag2 ran -1024 to -780; the screen sat at -1024 on
glow-150. Doing nothing scores -1024 and so does clicking at random (stray cost plus misses, cut at
-1024), so selection had nothing to tell apart there.

## The miss cost scaled and a soft floor (`ISL_MISS_SCALE`, `ISL_SOFT_FLOOR`, run 39)

`ISL_MISS_SCALE=<per 1024>` sets a miss's share of a wrong answer (`observe::missed`), 1024 as in
run 38; at 512 doing nothing scores -512. `ISL_SOFT_FLOOR=1` replaces the cut at -1024 with
`observe::floor`: scores from -512 up as they are, below it -512 - 512 x / (x + 512) for x below
-512, so -1024 becomes -768, -2048 -896, and no score reaches -1024; the order of bad scores is
kept. Test `misses_scaled_and_floored_softly_keep_bad_networks_apart`: a responder 1024 on respond,
never acting -512 (and -512 on flag), clicking every step -991. Run 39 (started 4:58 PM ET) is run
38 with both, at 512.

## Partial credit on cues and questions (`ISL_PARTIAL_FLAG=1`, flag~, run 40)

Under the halved miss cost run 39's recall passed respond at 85 and sat at -512 on flag2 (best
genome empty), the score for doing nothing: the soft floor ranked random clicking below it, but
nothing between doing nothing and holding the cue paid. flag~ (`observe::flag_graded`, probe 35),
used in selection while the goal stays on flag: a question answered in its three steps counts in
full, one answered in the three steps after that at half (not a stray); three quarters of the score
for answering cued questions more than plain ones, a quarter for answering cued ones at all; less
strays and the cued questions' missing credit at the miss cost. Test
`flag_graded_pays_answering_at_all_then_answering_the_cued` (12 lifetimes, gaps to 8, miss cost at
half): (flag, flag~) holder (1024, 1020), every question (0, 256), latch (-910, -885), late holder
(-774, 155), random (-815, -616), never (-512, -512). So clicking at the question tone pays before
the cue must be held, as surprise+ pays clicking after breaks before telling them apart. Run 40 is
run 39 with it.

## Partial credit on the other stuck rungs (`ISL_PARTIAL_MORE=1`, run 40)

A survey of the probes for ones with no graded selection score found four. (1) The stacked feel
rungs: `graded` mapped feel to feel~ (clicks 4 to 6 steps after a stroke at half, 7 to 9 at a
quarter), but stacking (run 38) moved those rungs to feel-gap and feel-maze, which had none, so the
feel rungs lost their partial credit; feel-gap~ and feel-maze~ (`observe::stroke_tiers`) restore
feel~'s tiers. (2) copy (reading's pairs8, at about -20 since run 39's 33): copy~ as echo~ is to
echo, lags 5 to 8 at half, per token shown. (3) tell (shapes' blur1): tell~, a click late in the
trial (step 6 or 7) at half. Test `graded_feel_copy_and_tell_pay_late_answers_in_part`: clicking 5
steps after a gap stroke scores -28 on feel-gap and 169 on feel-gap~; clicking light A at step 6,
0 on tell and 512 on tell~; saying the token shown 6 steps ago, 15 on copy and 453 on copy~ (on
time 861 and 874, random 10 and 6). (4) Selection's held: under the performance contrast
`select_graded` returns the contrast score, so select~'s partial credit was bypassed. Measured
(12 lifetimes, the cue felt 3 steps a block): a selector acting only while the cue is felt scores
31, one holding it 20 steps 97, never 0, a holder 1021: nothing between doing nothing and holding
through the block, which is why held sat at about 0 in runs 37 to 39. With `set_select_early`
select~ is the contrast averaged with the same over each block's first two trials: felt only 181,
held 20 steps 544, a holder 1008, blind -3. Run 40 has all four with flag~.

## Run 39's end (stopped at 204 for run 40)

10 rung moves in 204 generations, none after 100: reading to pairs8 (4, 33), selection cued to held
(24, 98, 99, 100), shapes to blur1 (51), recall to flag2 (85), switching to tell and feel-300 (99,
100; run 38 never passed tell). With misses at half and the soft floor, doing nothing scored -512
and random clicking below it, so the floor no longer hid the difference; but recall on flag2 (-539
at 204) and the screen (-512) kept best genomes that do nothing, and selection sat at about 0 on
held: no score paid anything between doing nothing and the full answer. Run 40 (started 7:29 PM ET)
adds flag~ and the other graded scores.

## Other agents as the world: predict or evade (`ISL_DUEL=1`, run 41)

See `wiki/other-agents.md`. A duel island whose world is other agents: each step a round, the
predictor winning when the two agents' moves (LEFT or not) match, the evader when they differ, the
roles flipping every `turn` rounds so each agent learns both. Partners: this generation's duel
elite, its hall of fame and the other islands' best. A fixed yardstick, `duel-bench`, four rules
no fixed policy beats, carries each rung's goal (duel64 to duel8, 100 to 400). Tests: a reader
scores 912 against an alternating copier (-912), a constant against a constant 0, a reader against
a constant 512; a network doing nothing -300 on the yardstick. Smoke test (3 generations,
population 8): the duel island's best at -297 on duel64.

## Run 40's end (stopped at 93 for run 41)

8 rung moves in 93 generations: reading to echo2q (4), selection cued to held (26, 32, 33, 53;
run 39: 24 to 100), switching respond, tell and feel-300 (56, 65; run 39: 99, 100), recall respond
(76; run 39: 85). With partial credit everywhere stuck, the early rungs came sooner; at 93 recall
on flag2 (-625), selection on held (-3), switching on feel-300 (0) and the screen (-512) had not
yet moved: too short to judge the graded scores there. Run 41 (started 8:17 PM ET) adds the duel
island. The first yardstick (four fixed rules) leaked: a wired copy-last and alternate scored about 213
and run 41's first start passed duel64 and duel32 by generation 6; it was stopped and restarted
with rules drawn per lifetime, each as likely as its opposite, on which every fixed policy measured
scores within 100 of 0 and a reader 918, and goals 200, 350, 500 and 650.

## Do we have learners? Run 41 at 142, and learning as a selection case (`ISL_LEARN_GAIN=1`, run 42)

The learning battery (now with recall's flag2 and the duel's duel64 against the yardstick's drawn
rules; `runs/checks/learn-battery/run41-g142.txt`, 12 worlds, the top 3 of each island): learning
on less frozen at least 50 and twice its error in 46 of 330 genome-task pairs (hurting in 6).
Switching's best on discrim and swap1: about -140 learning, -690 frozen. But no task scored above
0 learning (reading's pairs8 by prediction#1, 7, the one exception): frozen networks act wildly
(stray and wrong clicks under the stray and miss costs) and learning calms them. That is learning
in the lifetime, of a sort, not of the task. No duel genome learns (gaps 2 to 11).

What "learning" means here: what `program::neurocomp::set_frozen` turns off, every change to the
network's wiring in the lifetime: mask moves (Hebbian and anti-Hebbian), births, replication, the
three-factor output moves, the teacher, accidents, exploration and the shadow trials. Activity
(loops, histories, latches) runs frozen too, so memory held in activity is not counted.

`ISL_LEARN_GAIN=1`: probe 42, `gain`, a lexicase case on every rung but the novelty island's:
each lifetime scored again with the genome's learning frozen on that thread
(`set_frozen_here`), and each task probe's score learning, cut at what doing nothing scores
(`observe::idle_score`: -512 under misses at half on the probes misses are charged on, else 0),
less the same frozen, averaged. Cut at 0 nearly nothing in run 41 would gain; cut at doing
nothing, a network calmed by learning but no better than idle gains nothing, and one that learning
makes better than idle and than its frozen self does. Resumed from run 41's checkpoint at 141:
shapes' best gains 274 held out, switching's 18, the generalist's population 29; generations took
about 160 seconds beside run 41 (about doubling). Run 42 is run 41 with it.

## Run 41's end (stopped at 172 for run 42)

9 rung moves in 172 generations: reading to pairs8 (3, 108), selection cued to held (11 to 31),
recall to flag2 (86), switching to tell and feel-300 (89, 96). The duel island stayed on duel64:
its yardstick moved about 0 (13 at 172) and its best's held-out yardstick from 78 to 104 from
generation 130 on, within what fixed policies score; its genomes learn nothing in the battery.
Generations took 60 to 110 seconds with the duel. Run 42 (started 11:27 PM ET) adds the learning
gain as a selection case.

## Run 41's duel genome: the sticky hand (`runs/checks/duel41/`)

The duel island's best at 171, 185 populations: a lesion map (6 worlds) finds 62 that matter and
a core of wires carrying the act forward (two `sub:5` and an `lsl:32` wire from the act into
itself, a `set` wire from the senses into the hand); it repeats its own move 72% of the time. The
duel probe (`ISL_DUEL_PROBE=240`, new: by role and kind of drawn rule, learning and frozen) shows it
beating straight copying and reacting rules and losing less to their flipped versions, about 0
against patterns, better late in each turn but the same frozen: it adapts in activity, not by
learning. Saved in `wiki/mechanisms.md` as an evolved design.

## Run 42's learner, and development not counted (`ISL_LEARN_GAIN=2`, run 43)

The battery at 60 (8 worlds, each island's best, now with reading's babble and selection's cued):
learning clearly helps in 20 of 132 genome-task pairs; in 3 the frozen twin does exactly what doing
nothing does (development: it cannot act without growing first), in 7 it acts frozen and learning
makes it better than idle. Selection's best is the learner (`wiki/mechanisms.md`, the tuned eye):
eye-to-hand kernels whose Hebbian mask moves settle on this world's lights, and a hand-to-eye
efference copy; learning lifts it on cued (134 against 3), the maze's turn and symbol (21 and 12
against -258) and babble (896 against 640), and hurts it where telling lights apart matters. The
learning lesion (`ISL_LEARN_LESION`, new: one population's learning frozen at a time with
`set_frozen_pop_here`) puts what its learning buys in the eye-to-hand copies and the hand-to-eye
loop, which learn together. `ISL_LEARN_GAIN=2` (`set_learn_gain_strict`): a task whose frozen twin
scores exactly what doing nothing scores gains nothing, so development is not paid; run 43 queued
with it. A gain chart is on the dashboard.

## Switching's gain in run 42 is development (`runs/checks/switching42/`)

Switching's best held a gain of 580 to 760 from generation 150. The battery on its top 3 at 205
(12 worlds, `ISL_LEARN_WHO=switching`, with switching's respond and tell added): frozen, it scores
exactly what doing nothing scores on every light task (respond, discrim, swap1, shapes' paying:
-512; babble 0): it cannot act without growing in the lifetime. Learning, it responds to every light
(respond 1015, rising late, +41) and tells none apart (discrim 2, tell 0, win-stay 0). The learning
lesion finds no single population's learning carrying it (a self-loop on C1, #113, 31% of
respond's; nothing else over 10 points): the gain is births spread over its 57 populations, a
network that grows a responder each lifetime. That is what `ISL_LEARN_GAIN=2` (run 43) stops
paying for: a task whose frozen twin does exactly nothing gains nothing.

## Run 42's end (stopped at 210 for run 43)

9 rung moves in 210 generations: reading to pairs8 (5, 32), selection cued to held (16, 93 to 95),
switching to tell (66), recall to flag2 (70), the maze to feel (133). With the learning gain
selected, best genomes' gains rose to 580 to 760 on switching and the generalist and about 450 on
reading, but switching's is development (above); selection's gain fell to about 10 once on held.
Generations grew from about 20 seconds to 236 as networks grew and every lifetime was lived twice.
Run 43 (started 7:53 AM ET) stops paying for development.

## The act read back as what was done (`ISL_MUSCLE=1`, run 44)

The act vector is 1024 bits, the voice's 512 then the hand's, but every code lies in one 64-bit
word of its channel (the one-word voice's 8 lines in the voice's first word; each move's 16 lines in
one word of the hand), and the readout counts only a code's own lines. Since outputs are read
(run 29 on), what a population reading the act sees is what was written, every stray line with it.
The muscle's response (`evolve::set_muscle`, built in experiment 16 and never used in an island run)
replaces the act after each step with what the body did: the clean code of the move made and the
token said, every other line cleared. Then the act read back is the action taken, and a population
that wrote the act sees on its own lines whether it drove the body. `live_duel` now applies it to
both agents. Run 44 queued: run 43 with it.

## A typing world (`ISL_TYPING=1`, run 45)

Toward operating a computer: output letter by letter, as a keyboard, not a token a word. The
voice's word 0 keeps the one-word voice's tokens; its words 1 to 7 hold a keyboard of 35 keys (the
26 letters, space, Enter, Backspace, spares), each a code of 12 lines, five a word, no line shared,
drawn with the fixed body (the same every lifetime). A key is typed when at least half its lines
are on and they outnumber the other lines on in its word, the most covered winning, so a code with
3 lines missing and 3 stray still types its key and a flooded word types nothing. The world
(`World::typing`) shows the word on the eye as raw bytes (lower-case ASCII, input raw as always) in
the eye's first word and the line typed so far in its second; a letter or space key adds to the line
(at most 8), Backspace takes one off, Enter posts (on the first rungs the line posts itself when as
long as the word); a right post is stroked, a wrong one feels the wrong signal; a trial times out
after 3 steps a letter and 6. The muscle's response (run 44) shows the key typed on the act.
Probes: `type`, exact posts per trial; `type~` (selection, partial credit), a quarter for typing
anything and three quarters for letters right in place, a wrong letter -1/(letters - 1) so random
letters net 0, a missing or extra place -1. Test `typing_pays_the_word_typed_and_reads_keys_robustly`:
a typist reading the eye's bytes 1024 and 1024 on word3; random keys 0 and -101 (overfilling); doing
nothing and flooding the voice 0; on key4 a wrong key scores what nothing does and a right one 1024.
Ladder (`TYPING_LADDER`): key4 (one letter of four), key26, enter, word2, word3, word5, word8. Run 45
queued: run 44 with it.

Typing's partial credit revised at the user's call (doing nothing should be worse than typing, to
a point; too much and too little both cost): `type~` is three quarters letters (right +1, wrong
-1/(letters - 1), over the word's length) and a quarter length (+1 at the word's length, 2 less for
each word's length too many or too few, at least -1); a timed-out trial halves a score above 0 and
keeps one below. For "cat": "xyz" 226, "xyzx" and "x" between that and nothing, "xyzxyz" -286,
nothing -256; on one letter a wrong key 0 and nothing -256.

## Run 43's maze: learning on its own feel rung, and nowhere else (`runs/checks/maze43/`)

The maze's best's gain rose to 282 to 426 from generation 160 with its feel score (feel-maze, clicks
after strokes that come at random while it waits at the junction). The battery at 195 (12 worlds,
the top 3, with the maze's feel added): on feel-maze 235, 212 and 192 learning against 13, -14 and
85 frozen (gaps 222, 225, 106, each with an error near 100): learning that a network acting frozen
needs, not development. Elsewhere its learning only calms wild acting (respond -640 against -888,
below doing nothing) or hurts (the maze's turn -70 against 8). The learning lesion (64 populations)
finds no population whose learning carries it: freezing any one of nine populations' learning raises
feel-maze (to 247 to 419, a self-loop on the self channel, #301, most), so what helps is learning
spread over the network and several populations' learning works against it.

## Run 43's end (stopped at 223 for run 44)

10 rung moves, all by 113: reading to pairs8 (5, 113), selection cued to held (26 to 75), switching
to tell and recall to flag2 (32, 33; run 42: 66, 70), shapes to blur1 (72), the maze to feel (86).
With development not counted, best genomes' gains held at 400 to 500 on switching and the
generalist and 190 to 350 on reading; the maze's rose to 426 with its feel score and fell back. No
rung after 113. Run 44 (started 4:11 PM ET) is run 43 with the muscle's response and the typing
island together (the run that would have had the muscle's response alone was skipped).

## Run 44's keyboard: nothing typed in 50 generations, and credit for reaching toward a key

At generation 50 the typing island's top 4 scored -256 on type~ (what typing nothing scores),
learning and frozen alike: no genome typed a key. A key needs 6 of its 12 lines, in one of voice
words 1 to 7, outnumbering the other lines in its word; writing at random almost never does that,
so every genome scored alike and selection had no slope. Partial credit: the world records each
typing step's reach (`Event::key_reach`), the key the line needs next (the next letter; Enter when
the word is typed; Backspace after a wrong letter) and how near the voice came to it, its lines on
less the other lines on in its word, 0 to 12; a trial in which nothing was typed scores -256 plus
its mean reach over 12 times 256 (up to 0). Writing 5 of the right key's 12 lines each step scores
-178, 2 of them -224, nothing -256; a typed key scores as before (on one letter a wrong key 0, the
right one 1024). Run 44's genomes still scored -256 with it: they write nothing near the keyboard. Reach then
counted the other lines in the key's word at the code's share (12/52 each), so lines written at
random net about 0 and any write leaning to the code's lines counts.

## Unwritten outputs attract trials (`ISL_ATTRACT_OUT=1`), and keys among the body's codes

Why no genome typed: populations writing the act are born with kernels whose outputs are the
body's codes (the voice's tokens, the hand's moves: `tags::start_kernels`), and the keyboard's keys
were not among them, so nothing could write a key but by chance. Now, in a typing lifetime, the
keys are among those codes. Checked on run 44's checkpoint at 53 (key4, 8 worlds): genomes from
switching, the generalist, recall, the maze, selection and novelty, which never typed, type keys
(type~ up to 80; nothing scores -256); the typing island's own, which write nothing to the voice,
stay at -256. The output side's attractant, at the user's call (an unused muscle asks for a nerve):
each word of the act that holds a body code counts the steps nothing wrote into it; at a
population's own threshold (as the input attractant's, 2^(accident - 10)) that population grows a
trial reading its own source and writing that word, its newborn kernels writing the body's codes
there; the genome keeps it (`take_notes`, kind 13 voice, 14 hand) if the word then held a code the
body made (with the muscle's response, what was done). Test
`an_unwritten_word_of_the_act_attracts_a_trial_that_types_and_is_kept`: a population from the eye
into a channel, in a typing lifetime: the trial grows into the keyboard's first word, types 50
keys, is confirmed 185 times, and joins the genome as a population from the eye into the voice.
Run 44 restarted with both and typing's reaching credit (its first start's log in
`runs/checks/typing44/`).

## The body listed once (`world::SENSE_PORTS`, `ACT_PORTS`, `ORGANS`)

At the user's suggestion, after the keyboard had to be added by hand in several places (and was
missed in one: the codes what writes the act is born with): the body's ports (each sense channel
and act channel, its name and its genome tag) and its organs (voice, hand, keyboard: each one's
codes, when it is in use, and what it did on a step) are listed once in `world`; the genome's port
tags and names (`tags::INPUTS`, `OUTPUTS`, built from them at compile time), the attractants' ports
and kinds, the shadow predictors' targets, the kept trials' ports (`take_notes`), the codes
populations writing the act are born with (`World::body_codes`) and the muscle's response
(`World::act_done`) read them. The kinds of a lifetime's finds come from the lists
(`attract_in_kind`, `attract_out_kind`, `shadow_sense_kind`, ...), with a compile-time check that
a port too many does not collide. Behaviour unchanged: a fingerprint (`ISL_FINGERPRINT=3`, new:
every probe in every kind of lifetime, work, finds and wiring hashed for each island's first two
genomes) is identical before and after under run 44's settings on its checkpoint and run 33's on
its (`runs/checks/body-registry/`).

## The output attractant only where nothing is attached (`ISL_ATTRACT_OUT=2`)

Why generations slowed in run 44 (194 seconds at 50, against 109 in run 43 and 46 in run 41): the
frozen twin (run 42 on, about twice the lifetimes), the duel island (nine 800-round lifetimes a
world, the largest genomes, about 48 populations), networks growing through a run, and the output
attractant: on in every lifetime of every island, it grew trials into any word of the act left
unwritten a while, and by 50 genomes carried about 4.5 kept output trials each (a fifth of their
populations), most kept in worlds other than typing's. At the user's call the attractant should
answer only an output nothing is attached to: mode 2 counts a word of the act only if no
population's or branch's destination covers it (the whole act, its port, or that word) and no trial
is in it. Since what writes the voice is born with the keys' codes in typing lifetimes, anything
attached to the voice can already reach the keyboard. Test
`only_an_unattached_word_of_the_act_attracts_in_mode_two`: with nothing on the voice a trial grows
into the keyboard's first word; with a population writing the voice none grows into the voice
(only into hand words nothing covers). Mode 1 unchanged (run 44's fingerprint identical).

## Run 44's end (stopped at 83 for run 45), and run 45 made faster

12 rung moves in 83 generations of the second start: selection cued to held (11 to 32), switching,
recall and the maze past respond and feel (41), the maze to learn (43), reading to pairs8 (43),
switching to feel-300 (50), shapes to blur1 (78). The typing island typed from the restart: exact
posts on key4 22 to 58 per 1024 (best held out 46; goal 500). Generations grew to 448 seconds.
Run 45 (`ISL_ATTRACT_OUT=2`, `ISL_LEARN_GAIN_EVERY=4`, the duel island paused) timed on the same
checkpoint (run 44's at 25) with the same load: one generation 199 seconds against 563 for run
44's settings, about 2.8 times faster.
