# The pairs test, rerun on 6 seeds (2026-10-17)

As `runs/pairs-test/` (reading alone, run 31's settings, 200 generations, three arms: `none` no
three-factor rule, `elig` the rule, `rules` the rule as a gene), with the ladder pairs8 > pairs16 >
pairs32 (sizes a 200-trial lifetime can hold), goal copy 300, and seeds 1 to 6.

Copy on training (goal lines), mean and best over the run; rungs passed:

| arm | s1 | s2 | s3 | s4 | s5 | s6 |
|---|---|---|---|---|---|---|
| none | -17, 6 | **117, 216: pairs8 at 10, pairs16 at 14** | -12, 6 | -14, 21 | -17, 0 | -13, 14 |
| elig | -16, 0 | -15, 4 | -20, 0 | -4, 27 | -15, 7 | -13, 34 |
| rules | -12, 13 | -13, 8 | -10, 21 | -16, 23 | -16, 11 | -14, 5 |

One run in 18 found a reader (none, seed 2: the reader of wiki experiment 18 again, the same seed
as the first test), and it climbed to pairs32. Seeds 1 to 3 repeat the first test exactly until a
pass (runs are deterministic by seed). With the rule: 0 of 12; without: 1 of 6. That difference
is within chance at this count. What it does show: from a bare genome, in 200 generations of 32
genomes, finding the reader is rare under every arm.
