# 12: Graded probes and lexicase selection

**Question.** Every single-score run stalled at "click somehow". Can small graded capacities,
selected by lexicase, give evolution steps to climb?

**Setup.** `observe::probes`, five capacities in parts per 1024 from a serial-reversal lifetime and
a delayed lifetime (200 trials each), each averaging 0 at chance and for a network that never
clicks:
- respond: clicks with a light up minus clicks without;
- tell: the light difference in a lifetime's second half, signed by the first half's;
- discriminate: more clicks on the paying light (the task);
- reverse: discrimination late in each reversal phase, averaged over phases;
- remember: more clicks after a light that is gone than after none.
Epsilon-lexicase over (world, probe) cases (`evolve_lexicase`). Tag genome, naked start, pop 32,
16 training worlds, 16 held out.

    PRB_SHOW=1 PRB_GENS=150 cargo run --release --example probes

**First version, withdrawn.** tell was |A - B| and reverse the worst phase. Noise alone pushed
tell to a population mean of 50, and a chance clicker always has one phase below 0 while a
silent genome scores 0, so lexicase selected silence: by generation 40 the top genome never
clicked. Both now average 0 at chance (tested with hand-written policies).

**Run with unbiased probes** (stopped at generation 80 to add replication and circuit copy):

| gen | best held out: respond tell disc reverse remember | population mean: respond tell disc reverse remember |
|---|---|---|
| 20 | 12 18 6 -5 20 | 1 0 0 1 0 |
| 40 | 1 16 -18 -21 -20 | 0 14 6 5 4 |
| 60 | 5 19 32 3 -6 | 5 14 4 5 6 |
| 80 | -10 14 -11 -14 -6 | 1 13 6 4 1 |

**What it means.**
- Clicking appeared by generation 20 with no silence penalty.
- The population means drifted a little above 0 (tell 13-15, discriminate 4-6) and stayed: small
  but consistent across rows, where chance is 0.
- The "best" genome (highest summed training means) is mostly a lucky specialist (remember 124
  on training, about 0 held out). Better to report the population and held-out elite.
- At generation 60 the best held a learned reflex evolved from nothing: a covering population
  growing repeat births (a kernel on what was seen before an act repeats the act), learning
  only while the act was on. Held-out discriminate 32, close to noise for one genome.

## Later runs (gaps between trials, ports, thrift per world)

- **Clock artefact.** With trials back to back, a blind policy clicking on steps 3-5 of every 10
  scores over 900 on respond; with gaps of 0-6 steps between trials it scores about 0. The rise
  in respond at generation 40 of the self-channel run was most likely the step counter, not
  sight. Runs since use gaps.
- **The self channel crowded out the senses.** Its tag sat one bit from the senses', taking half
  the tags that reached them (55 of 256 instead of 105). Tags now name internal channels only;
  senses and effectors are ports with a tag space of their own.
- **Noise.** One genome scored on 16 held-out worlds lands as far as +/-50 from 0 by luck: a
  genome reading no sense at all scored tell 50, discriminate 36 held out. Claims need many
  worlds, or one held-out set to pick and another to report.
- **First sensory skill (ports run, generations 90-110).** A covering population on all the
  senses into the hand, growing by repeat births, spread through the population: held-out tell
  202, 142, 110 in three successive rows (population tell on training 26, 71, 47), with
  discriminate and reverse still about 0. Work per step rose to about 300.
- **Births into act ports** (added after): a spontaneous eye -> hand population on 64 held-out
  worlds goes from respond 15 / tell 32 to respond 196 / tell 339 once births copy the move
  being made onto what is seen (`examples/port_cost.rs`). Discrimination still needs clicks to
  be tied to the stroke that follows.
