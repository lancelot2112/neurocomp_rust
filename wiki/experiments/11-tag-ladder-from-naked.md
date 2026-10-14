# 11: The tag genome grows from nothing (silence is death)

**Question.** From an empty string, can the tag grammar evolve a network that clicks on the
paying light, now that silence scores below any clicking, writes land in genome order, new
populations draw any op, and split, self-loop, sparse and hold are in the grammar?

**Setup.** Naked start (4 empty channels), pop 32, elite 8, worst-phase scoring, stages
instrumental:40, delayed:30, then reversal. 8 training lifetimes a genome on the first two
stages, 16 held out. Best network printed every 10 generations (`CUR_SHOW=1`).

    CUR_SHOW=1 CUR_GENOME=tags CUR_START=naked CUR_CHANNELS=4 CUR_SCORE=phase \
    CUR_STAGES=instrumental:40,delayed:30,reversal1:30,reversal:30 CUR_POP=32 CUR_ELITE=8 \
    CUR_LIVES=8 CUR_LIVES_REV=32 CUR_HELD=16 cargo run --release --example curriculum

| stage | gen | best (train) | mean | held: stage | held: reversal | held: worst phase |
|---|---|---|---|---|---|---|
| instrumental | 10 | -4096 | -4096 | -4096 | -4096 | -4096 |
| instrumental | 20 | 17 | -2705 | 5 | -4 | -337 |
| instrumental | 39 | 32 | -1103 | -259 | -264 | -590 |
| delayed | 10 | 36 | -116 | -267 | 2 | -326 |
| delayed | 29 | 50 | -478 | -240 | 6 | -307 |

Stopped after the delayed stage.

**What evolved.**
- Out of silence by generation 20 with a spontaneous (tonic 1/2) NOR population on the act:
  NOR into an empty act word turns most of it on, so some firing lands on the click. Segment
  copies then made six or seven of it.
- Readers of touch and ear appeared; nothing ever read the eye, so the lights could not be told
  apart.
- The new operators were used: split put channels between the clickers and the act, a NAND
  self-wire on a cleared channel gave a constant all-on word (a steady drive), held channels
  with shift self-loops appeared (empty, so inert).

**Meaning.** Silence-is-death works: every genome clicks within 20 generations. But with 8
training lifetimes, gains on training (up to 50) were luck: held out they were negative on the
stage and chance on reversal. Next: 24 training lifetimes on every stage.
