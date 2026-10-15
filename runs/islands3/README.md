# Island run 3

The island run of experiment 13 (`wiki/experiments/13-islands-and-ladders.md`), saved to move
machines.

- `checkpoint-140.txt`: every island's population at generation 140, with its ladder stage.
- `log.txt`: the run's output up to the move: rows, goal progress, stage moves, exchanges, and
  each restart, marked with `----` and what changed.
- `plasmids.txt`: every plasmid moved, with the circuit it carried.

Resume (from the repository root, after `cargo build --release`):

    ISL_SCREEN=1 ISL_MAZE=1 ISL_PREDICTION=1 ISL_SHAPES=1 ISL_LADDERS=1 ISL_CKPT=5 ISL_HELD=48 \
    ISL_RESUME=runs/islands3/checkpoint-140.txt \
    ISL_CHECKPOINT=runs/islands3/checkpoint.txt ISL_LOG=runs/islands3/plasmids.txt \
    cargo run --release --example islands >> runs/islands3/log.txt 2>&1

Every other setting is the default (16 genomes an island, elite 4, 12 training worlds, 200
trials, gaps 0-6, exchange every 20, 2 plasmids, thrift on, 300 generations). Lifetimes are
deterministic by seed and spread over every core. The count of generations in a row on a goal
is not saved, so a resumed island needs its goal met 3 fresh generations.

Rechecking an island's elite on fresh worlds:

    RCK_TOP=4 RCK_FILE=runs/islands3/checkpoint.txt cargo run --release --example recheck
