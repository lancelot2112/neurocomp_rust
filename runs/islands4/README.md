# Island run 4

The first fresh run on the new runtime: every island from naked genomes, generations 0 to 299,
with the ladders as they stood at the end of run 3 (recall starting at respond, shapes with
win-stay before paying, screen starting with a torch) and stage moves checked on 48 held-out
worlds.

What is new against run 3:
- `ISL_INPLACE=1`: every declared channel is held in place, one frame with no history. Only
  clearing writes take lines away, a write is seen by the populations after it in the same step,
  and a network that needs the past builds the delay itself.
- `ISL_READS=1`: the words a kernel population reads count as its work.
- The `set` op (overwrite), and faster births (only tiles over words that have lines are
  scored).

Files:
- `log.txt`: every generation's goal line with its UTC time and seconds, stage moves, held-out
  checks, plasmids (each classified on arrival, and its fate), and reports every 10 generations.
- `plasmids.txt`: every plasmid with what it was made of, alone/blind scores, its recipient with
  and without it, and its circuit.
- `checkpoint.txt`: every island at generation 299.

Stage moves: reading to echo2 at 26; recall to lag1 at 225, lag2 at 238, lag3 at 293, lag4 at
294 (each passed on held-out worlds). 299 generations took 51 minutes.

Resume or recheck (the run's runtime must be switched on):

    ISL_INPLACE=1 ISL_READS=1 ISL_SCREEN=1 ISL_MAZE=1 ISL_PREDICTION=1 ISL_SHAPES=1 ISL_LADDERS=1 \
    ISL_CKPT=5 ISL_HELD=48 ISL_GENS=400 ISL_RESUME=runs/islands4/checkpoint.txt \
    ISL_CHECKPOINT=runs/islands4/checkpoint.txt ISL_LOG=runs/islands4/plasmids.txt \
    cargo run --release --example islands >> runs/islands4/log.txt 2>&1

    RCK_INPLACE=1 RCK_TOP=4 RCK_FILE=runs/islands4/checkpoint.txt cargo run --release --example recheck
