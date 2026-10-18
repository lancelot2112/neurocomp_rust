# The 3-seed pairs test (2026-10-17)

Reading alone (`ISL_ONLY=reading`) on a pairs ladder (`ISL_LADDER_reading`: pairs8, pairs32,
pairs128; glyph-to-token pairs drawn each lifetime, tokens heard like their voice codes, one's own
voice not heard, the teacher on six tokens in eight after the first 50; goal copy 100), run 31's
settings otherwise, 200 generations, evolutionary seeds 1-3, three arms:
- `none`: no three-factor rule;
- `elig`: `ISL_ELIG=1`, the runtime's rule;
- `rules`: `ISL_ELIG=1` and `ISL_RULES=1`, the rule a gene.

Copy on training (goal lines), mean over the run and best:

| arm | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| none | -17, 6 | **162, 353 (passed pairs8 at 10, pairs32 at 11)** | -12, 6 |
| elig | -16, 0 | -15, 4 | -20, 0 |
| rules | -12, 13 | -13, 8 | -10, 21 |

No arm with the rule passed pairs8 in 200 generations. The one pass (`none`, seed 2) is not
reading: its best genome is an XOR wire from the ear into the voice (an echo); on 24 fresh worlds
at pairs128 it scores echo 1026 and copy 0 (`ISL_LESION`). But that is the genome at the end; the
genome that passed pairs8 (the rung log, `runs/checks/evolved-reader/pairs8-passer.txt`) reads:
eye kernels born on each glyph copy the move being made (the echo of the teacher), writing by XOR,
so they cancel the echo on taught steps and speak alone on untaught ones. examples/pairscheck,
against a lookup table that is silent after taught steps: 957 of 990 at N = 8, 912 of 1014 at N =
32 (1600 trials), 678 of 1020 at N = 128. Not a leak (an earlier note here said so, retracted).
The island lost the reader at pairs128, where 200-trial lifetimes are too short (147).
