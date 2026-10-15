# How to run and report the island experiment

Instructions for a Claude Code session driving `examples/islands` (read `CLAUDE.md`,
`wiki/experiments/13-islands-and-ladders.md` and `runs/islands3/README.md` first).

## Running

- Start the run detached (`setsid nohup ... &`) with output appended to `runs/islands3/log.txt`.
  It must survive the session and is never waited on with `sleep`.
- Watch it with a background monitor that tails the log and emits only what matters: stage moves
  (`moves to`), exchanges (`plasmids moved`), goal progress every 5 generations, restarts (`----`)
  and errors (`panicked`, `error`). Re-arm the monitor when it expires.
- To change the world, probes or ladders mid-run, never restart from scratch:
  1. build and test the change;
  2. wait for the next checkpoint (`ISL_CKPT=5`);
  3. stop the run;
  4. resume from the checkpoint with the new binary;
  5. append a `---- resumed at generation N: what changed ----` line to the log.
- Commit and push the code with each change (`cargo build --release` and `cargo test` first).
  Copy the newest checkpoint into `runs/islands3/` now and then.

## Reporting

- **Report every event as it arrives**, briefly: a stage move, an exchange, each 5-generation
  goal line. One or two sentences, or a small table when several islands move.
- **Lead with the number that matters and what it means.** For example: "switching moved to
  win-stay at generation 137: its population held tell above 100 for 3 generations."
- **Tables for comparisons**: island, stage, probe, population mean, goal, met count; or every
  island's best on every probe at an exchange.
- **Say what is noise.**
  - A held-out score on 12 worlds moves about ±25-50.
  - Before calling a skill real, recheck the island's elite on 64 fresh worlds
    (`examples/recheck`, `RCK_TOP=4`) and give the ± standard error.
- **Check what a genome reads.** Before believing a score, look at which ports the genome reads.
  A genome that reads no input port cannot have a real skill that needs sight.
- **Blind controls.**
  - Every new probe or world setting gets one (`examples/port_cost`): genomes that see nothing
    must score about 0.
  - When a control fails, find the bias, fix it and say so plainly; do not report the biased
    number.
- **Explain stalls.** When an island stalls, say why from what its genomes read and write and
  what the stage needs, and propose the smallest stepping stone. Do not tune genomes by hand.
- **Be plain.** No hype. Negative results are results. Write each lesson into the wiki
  (`wiki/experiments/13-...`, `wiki/lessons.md`).
