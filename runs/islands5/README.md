# Island run 5 (stopped at generation 161)

Run 4's settings (fresh from naked, channels held in place, reads counted) plus:
- plasmids as bacteria keep them: at most 6 populations (`ISL_PLASMID_MAX=6`), exclusion
  (`ISL_EXCLUDE=1`), and a carrying cost of 16 a token (`ISL_CARRY=16`);
- the op choice (`ISL_OPS=1`): an op mutation of its own, and no `rotr` in the draws.

Planned for 300 generations. Two interruptions:
- After generation 70 it was moved to its own process from `checkpoint-70.txt`, with the same
  binary and settings, because a session job's time limit would have cut it off. The log has
  no marker there, because the file was locked; the second set of "resumed ..." lines marks it.
- It stopped after generation 161 at 06:29Z, when the session that launched it ended. Windows
  ends a session's processes, and Start-Process children are among them. Later runs are started
  with `Win32_Process.Create` (see `runs/run6.cmd`). Not resumed.

Stage moves: shapes to blur1 at 31; recall to lag1 at 58, lag2 at 122 and lag3 at 126 (run 4:
225, 238, 241); reading to echo2 at 116 (run 4: 26).

Generations slowed to 60-80 s at times: a few genomes cost 10-40 times their island's best
(shapes' population averaged 12,674 work a step at generation 60, its best 160). That is what
the energy budget of run 6 is for.

`checkpoint.txt` is generation 160 and `checkpoint-70.txt` the resume point. To resume:

    ISL_OPS=1 ISL_PLASMID_MAX=6 ISL_EXCLUDE=1 ISL_CARRY=16 ISL_INPLACE=1 ISL_READS=1 \
    ISL_SCREEN=1 ISL_MAZE=1 ISL_PREDICTION=1 ISL_SHAPES=1 ISL_LADDERS=1 ISL_CKPT=5 ISL_HELD=48 \
    ISL_GENS=300 ISL_RESUME=runs/islands5/checkpoint.txt ISL_CHECKPOINT=runs/islands5/checkpoint.txt \
    ISL_LOG=runs/islands5/plasmids.txt cargo run --release --example islands >> runs/islands5/log.txt
