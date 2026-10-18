# The seeded pairs test (2026-10-17)

Every genome starts as the evolved reader (wiki experiment 18,
`runs/checks/evolved-reader/pairs8-passer.txt`): reading alone, 8 pairs, run 31's settings, 200
generations, three arms (none, elig, rules), seeds 1 to 3, on a ladder that takes the mirror away:
pm8 (each sound its voice code, one's own voice not heard) > pm6 > pm4 > pm2 > pm0 (6, 4, 2, 0
eighths of each sound's lines from its voice code, one's own voice heard), goal copy 300.

| arm | s1 | s2 | s3 |
|---|---|---|---|
| none | pm8 at 2; on pm6 (best 48) | pm8 at 2, pm6 at 46, pm4 at 47; on pm2 (best 26) | pm8 at 2; on pm6 (best 25) |
| elig | pm8 at 2; on pm6 (best 63) | pm8 at 2; on pm6 (best 53) | pm8 at 2, pm6 at 47, pm4 at 49; on pm2 (best 0) |
| rules | pm8 at 2; on pm6 (best 22) | pm8 at 2, pm6 at 68, pm4 at 69; on pm2 (best -2) | pm8 at 2; on pm6 (best 47) |

Every run kept the reader (pm8 at generation 2). Three of nine got past pm6 and pm4, one per arm,
and all three stopped at pm2. None reached pm0, sounds unrelated to voice codes, where a learned
listener is needed. pm6 changes two things at once: a quarter of each sound's
lines stop matching its voice code, and one's own voice is heard again. Hearing it, the echo wire
repeats the network's own voice (the voice loop of experiment 16), so pm6's step may be harder
for the second reason than the first.
