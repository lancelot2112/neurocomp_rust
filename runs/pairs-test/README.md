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
at pairs128 it scores echo 1026 and copy 0 (`ISL_LESION`), and in examples/pairscheck copy about
0 at every N. Yet its elite's held-out checks on pairs8 and pairs32 gave copy 340 to 548: in the
island's version of these rungs an echo earns copy. To be found and fixed before this test means
anything.
