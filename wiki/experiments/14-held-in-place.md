# 14: Channels held in place (island run 4 against run 3)

**Question.** In the old runtime every channel is a ring of frames, cleared or copied every step,
and an edge reads the frame before. What changes if channels keep their lines from step to step
in one frame, so that memory comes free and the network must build any lag itself?

## Setup

The runtime change, switched on with `ISL_INPLACE=1` (`World::in_place`,
`RuntimeNetwork::from_graph_held`):
- every declared channel is one frame, held in place: it is never cleared or copied, and only
  clearing writes take lines away. The senses, the act and the self node keep the old timing;
- a read sees the channel as it is now, including writes made earlier in the same step, so
  genome order matters;
- there is no history: every lag, span, gate lag and repeat lag on a held channel reads it now.
  A network that needs the past builds it, e.g. with a `set` wire placed before the write it
  delays, or a `set` population that runs only while an enable channel is on (a latch). Both
  are tested in `tags.rs`;
- a new op, `set`: the destination word becomes what is written (an overwrite).

Two more changes went in at the same time:
- `ISL_READS=1`: the words a kernel population reads count as its work, so thrift sees a wide
  read;
- always on, and exact: whole-vector reads are borrowed instead of copied, and a covering birth
  scores only the tiles over words that have lines.

Run 4 is fresh from naked genomes. It uses all eight islands with the ladders as they stood at
the end of run 3 (recall starting at respond, shapes with win-stay before paying, screen
starting with a torch), stage moves checked on 48 held-out worlds, and 300 generations.

    ISL_INPLACE=1 ISL_READS=1 ISL_SCREEN=1 ISL_MAZE=1 ISL_PREDICTION=1 ISL_SHAPES=1 ISL_LADDERS=1 \
    ISL_CKPT=5 ISL_HELD=48 ISL_GENS=300 ISL_CHECKPOINT=runs/islands4/checkpoint.txt \
    ISL_LOG=runs/islands4/plasmids.txt cargo run --release --example islands

Run 3 (`runs/islands3`, experiment 13) is the comparison, but not a clean one. It started from
genomes that had already evolved for 100 generations without ladders, and its ladders changed
partway through; run 4 had every ladder from generation 0. A difference is the runtime and the
finished ladders together.

## Where each got to

| island | run 3 (400 generations) | run 4 (300 generations, from naked) |
|---|---|---|
| recall | never passed its first rung | respond at 224, lag1 at 237, lag2 at 240, lag3 at 293 (held out 180), then lag4 |
| reading | echo2 at 124; best echo 48 | echo2 at 26; echo 164 at 299 (met 2 of 3) |
| switching | swap1 (stage 4) | still on respond (59-106 near the end) |
| shapes | climbed to paying on training scores alone | stuck on clean: training up to 255, held out 72-85 |
| maze, screen, prediction | first rung, about 0 | first rung, about 0 |

Each island's best genome at the end, on 32 fresh worlds in the plain world. Prediction's
surprise and reading's echo are not shown, because the plain world has no stream to break
and uses 8 tokens where echo2 uses 2:

| island | run 3: respond / tell / remember | run 4: respond / tell / remember |
|---|---|---|
| switching | 37 / 188 / -13 | 117 / 268 / 1 |
| recall | -5 / -3 / -7 | 238 / 79 / 0 |
| shapes | 61 / 239 / 64 | 160 / 121 / -8 |
| maze | 1 / 7 / 1 | 112 / 237 / -8 |

Recall's best scores remember 0 at a delay of 3, although the island passed lag3 on held-out
worlds. Its elite holds two kinds of genome, side by side under lexicase:

| rank | respond | remember at delay 1 / 2 / 3 / 4 |
|---|---|---|
| 0 | 238 | 162 / 162 / 0 / 0 |
| 1 | 204 | 129 / 129 / 0 / 0 |
| 2 | -44 | 260 / 376 / 298 / 157 |
| 3 | -60 | 247 / 361 / 292 / 167 |

The elite's average passed the goal. No single genome both responds and remembers 3 steps yet.

## Size and speed

The best genome per island (populations, and channel bits):

| island | run 3 | run 4 |
|---|---|---|
| switching | 104, 41k | 61, 249k |
| recall | 83, 3,576k | 77, 7k |
| reading | 119, 37k | 217, 373k |
| generalist | 72, 77k | 140, 209k |
| prediction | 222, 65k | 81, 554k |
| maze | 240, 471k | 55, 0 |
| screen | 47, 86k | 5, 6k |
| shapes | 40, 29k | 81, 273k |

Run 4 did 300 generations in 51 minutes: about 10 s a generation on average, 15-22 s near the
end. Run 3's last 100 generations on the same machine took about 77 minutes (about 46 s each).

The old cost was hidden from thrift. On run 3's best recall genome, timing each part of a step
showed births took 120 of 131 thread-seconds: a covering birth scored all of its source's
tiles, about 8,000 on a 131,072-bit channel, every step. Copying reads and clearing channels
cost little. Scoring only the tiles over words that have lines took births to 10
thread-seconds, with identical results. Counting reads as work keeps recall's channels small:
3.6 million bits in run 3, 7,000 in run 4.

## Ops

Which write ops run 4's genomes use, at generation 265 (9,293 populations): `add` 38%, `or`
19%, `shl` 11%, `nor` 9%, `clear` 3%, and `set` 0.4%. Each island settles on one or two:
- reading: `add` 98%;
- switching: `add` 85%;
- screen: `sub` 90%;
- recall: `shl` 38%, `and` 33%;
- prediction: `shl` 78%;
- maze: `clear` 32%.

The two islands that need the past use `shl` most. A shifted write onto a held channel smears a
line forward over time, which is likely how they built lag without history; no circuit was
read to confirm it. One op across a whole island is also inheritance: a population's op
changed only as one of its 38 properties, about one mutation in 100, and copies carry it.

## Plasmids

| | run 3 (generations 300-399) | run 4 (300 generations) |
|---|---|---|
| scored on arrival | 64 | 223 |
| helped their recipient | 4 (6%) | 14 (6%) |
| median size | 20 populations | 4 |
| reached 8 or more of 16 genomes | 5 | 13 |

Two plasmids that helped reached a whole island's elite:
- #38, a one-population eye-to-voice reflex grown on maze. It raised the generalist's babble
  by 104 and copy by 23, and was whole in all 16 genomes at generation 160. It was gone by 200.
- #79, from recall: shapes' tell +86. It was kept only in part, then lost.

Plasmids that did nothing spread too: sealed ones (which read and write only their own
channel) rode a winning lineage into 14 of 16 genomes. One that hurt prediction on arrival
(surprise -78) was trimmed by mutation and kept in all 16 genomes, none whole. A verdict on
arrival judges the plasmid as it came, not the part that is kept. No plasmid made a lasting
difference to an island.

## What it means

- **Held memory makes memory tasks reachable.** Recall went from nothing to lag3, and reading's
  echo tripled. Those are the tasks that keep something for a few steps, and they got it without
  any history in the runtime.
- **The cost thrift saw was not the cost paid.** Births scanned whole channels every step, and
  reads were not counted. With both fixed, genomes stay lean, and late generations are 2 to 3
  times faster (not like for like: the genomes differ).
- **The held-out check works.** It kept shapes on its first rung while training scores were
  twice the held-out ones. Run 3 let shapes climb five rungs on such scores.
- **Maze, screen and prediction are stuck in both runtimes,** so their first rungs, not the
  runtime, are what holds them.
- **Plasmids spread by lineage as much as by use.** Helping on arrival and spreading come apart.
  Run 5 tests plasmids as bacteria keep them (small units, exclusion, a carrying cost) together
  with the op choice (an op mutation of its own, no redundant `rotr`).

Runs 5 to 7, which build on this runtime, are experiment 15.
