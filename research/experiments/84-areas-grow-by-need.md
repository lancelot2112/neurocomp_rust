# 84 · Higher areas grow by need

**Question.** The number of higher areas has been set by hand (`HIER_LEVELS`): one on most
tasks, three on the story-boundary task, which fails with one. Can the network start
with one area and add another only where the top area cannot account for what it reads,
as the roadmap proposed (a bud, promoted or pruned)?

**Code.** [`examples/episodic.rs`](../../examples/episodic.rs), `HIER_GROW=1` (with
`HIER_CHAIN=mix`, where each area votes in the thalamic mix as its own source):
- **A bud.** The top area keeps a candidate above it, the last of the chain, with a window
  `HIER_SPAN` (4) times longer. It reads and learns like any area (the column's residual,
  the words it failed to predict). Its vote is weighed by the mix, which keeps its
  reliability like any source's, but it is not counted: it runs in shadow and changes no
  answer.
- **Its evidence.** On each training word the mix's choice is made with and without the
  bud's vote: a **fix** where it was wrong without the bud and right with it, a **break**
  for the reverse. A story's answer counts `HIER_GROW_ANSWER` (16) times.
- **Promotion.** Every `HIER_GROW_EVERY` (250) training stories, a bud whose fixes beat its
  breaks by more than `HIER_GROW_Z` (2) standard deviations (a sign test, in integers:
  (fixes − breaks)² > z² · (fixes + breaks)), with at least `HIER_GROW_MIN` (32) fixes, becomes
  a full area that votes, and a new bud starts above it (up to `HIER_GROW_MAX` (4) areas).
- **Pruning.** A bud not promoted in `HIER_GROW_PATIENCE` (4) checks is replaced by a fresh
  one: what it saw was not structure within its reach.
- `GROW` report: areas at the end, and each promotion and pruning with its evidence.

The first rule (promote at fixes ≥ 2 × breaks) pruned the story-boundary bud at 1,534 fixes
against 925 breaks: clearly useful, but not twice as useful. The sign test asks only that
the help is real.

## Results

**Story boundary, from one area** (seeds 0 / 1 / 2, held-out):

| | Areas above the column | Held-out |
|---|---|---|
| one area, fixed | 1 | 12.0 / 17.4 / 8.4% |
| three areas, fixed (recorded) | 3 | 98.4 / 86.8 / 97.0% (94.1) |
| **one area, growing** | **3, every seed** | **87.8 / 85.0 / 95.8% (89.5)** |

Seed 0: area 3 (window 16 sentences) promoted at story 500 (fixes 1,352, breaks 802), area 4
(window 64) at story 750 (1,608 / 1,136); the bud above (window 256) pruned at stories 1,750
and 2,750 (540 / 633, then 549 / 783). Seeds 1 and 2 grow the same two areas by stories 1,000
and 500, and prune the third bud.

**The three-seed suite with growth on** (every entry keeps its own starting depth): every
entry within 1 point of its recorded mean (story boundary 94.1 → 94.1, slot memory 16.3 →
16.3, relation entries 63.5 → 63.5–63.7, belief 81.7 → 81.6). Saccades, role transfer and
sleep generalisation do not chain by the mix and are untouched.

Areas grown (seeds 0 / 1 / 2): story boundary (starting at 3) 3 / 3 / 3; full hippocampus 1 /
1 / 1; most memory entries 1–2; the relation and speech entries 3 / 3 / 2; index hippocampus
3 / 4 / 4; schema advantage 4 / 3 / 4.

## Findings
1. **The network finds the depth the task needs.** From one area, story boundary grows
   exactly the two areas a person had to add by hand, at windows of 16 and 64 sentences,
   and stops: the bud above is pruned on every seed. Accuracy goes from 8–17% to 85–96%
   (89.5 mean, against 94.1 with the depth set by hand).
2. **A shadow bud costs no accuracy.** It learns but cannot change an answer until promoted,
   so tasks that did not need it are unchanged.
3. **It grows more than it needs on other tasks.** Many entries grow to 2–4 areas with no
   gain in held-out accuracy: their buds fix more training words than they break, by a real
   but small margin, often on words that are not answers. Each area costs compute (its own
   prediction and learning each step).
4. **Pruning works where the surprise is out of reach**: the 256-sentence bud on story
   boundary breaks more than it fixes (633 against 540) and is pruned on every seed.

## Next
- Promote on what an area is worth, not only on whether it helps: charge each area its
  compute, or require the help to show on answers, so the relation entries stay at one area.
- An area that stops helping should be pruned too (demotion), not only buds.
- ~~Gate the top-down frame into L4 by the thalamus's record of the area's reliability~~
  (tried: [85](85-top-down-as-a-trusted-witness.md); it shuts out useful context).
