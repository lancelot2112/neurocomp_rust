# 25 · A chain of cortical areas: does each area reach further back in time?

**Question.** Experiment [24](24-cortical-hierarchy.md) put one higher area over the
column; its slow state spans the last 4 sentences. Higher cortical areas integrate over
longer windows (Hasson et al. 2008; Murray et al. 2014). If each area above spans 4× the
window of the one below, does every added area extend how far back the network can use
a fact?

**Code.**
- `HigherArea::input_with` and `span` in [`src/program/cortex.rs`](../../src/program/cortex.rs).
- In [`examples/episodic.rs`](../../examples/episodic.rs):
  - `HIER_LEVELS=n`: n higher areas; area k's window is `HIER_SPAN`^(k−1) sentences (4,
    16, 64).
  - `HIER_CHAIN=mix`: how the upper areas reach the column (see below).
  - `TASK=season` (`SEASON_LEN`).
  - Reports `CHAIN` and `SEASON`.

Run: `COPY_GROW=1 GROW_TRUST=0.5 STICKY=255 GENERALIZE=0.5 GENERALIZE_AFTER=1
TRUST_AT_TEST=0.5 GROW_GATE=3 SLEEP_EVERY=500 SURPRISE_GATE=1 MEMO=1 CANON=1 TASK=season
POLICIES=nomemory HIER=1 HIER_LEVELS=3 HIER_CHAIN=mix MIX=1 cargo run --release --example
episodic`.

## The chain
- **Areas 2, 3, 4** (area 1 is the column) are `HigherArea`s with windows of 4, 16 and 64
  sentences. Each holds the words the column found surprising over its window, plus the
  current sentence's bag.
- **All learn the column's residual** (the words it failed to predict), as in 24.
- **Top-down, two ways:**
  - *Frames* (default). Each area reads the prediction of the area above it as one more
    input frame. Top-down flows down the chain, and only area 2 reaches the column.
  - *Mix* (`HIER_CHAIN=mix`). No area feeds another. Each area's prediction is a source in
    the precision-weighted mix of 24 (`SourceMix`), weighted by its learned reliability
    per context and confidence.
- Every area runs at every word. An event-driven clock is not built yet.

## Task: season
> winter came . the cat slept . | it rained . the dog ran away . | ... | then it rained .
> john went to the **bedroom**

- **One episode per story:** a season is announced, then D filler stories (1–2 distractor
  sentences each, D from 0 to `SEASON_LEN` − 1), then the question.
- **The answer** is the name's place for that season (fixed mapping, distinct per season).
- **Nothing between the announcement and the question reveals the season,** so accuracy
  by D shows how far back the network can use it.

A first version used one continuous story stream: a season held for 32 stories, and
every story asked a question. It was solved without long reach. Every story's answer
reveals the season, so the areas inferred it from the last story (accuracy flat in D).
Episodic memory recalled the same name's last trip (97–99% once the season had held a
while). Hence one episode per story.

## Results (held-out answers; seeds 0 / 1 / 2)

**Short distances (D = 0–3).** Area 2's 4-sentence window covers about D ≤ 1:

| | All | D = 0 | D = 1 | D = 2–3 |
|---|---|---|---|---|
| Column alone | 0% | 0% | 0% | 0% |
| Area 2 | 63 / 55 / 68% | 64–84% | 68–88% | 42–54% |
| + area 3, as a frame into area 2 | 0 / 3 / 57% | 0–68% | 0–69% | 0–47% |
| + areas 3, 4, as frames | 45 / 22 / 40% | 26–56% | 31–61% | 16–35% |
| + area 3, mixed | 63 / 60 / 68% | 72–84% | 72–88% | 46–54% |
| + areas 3, 4, mixed | 63 / 62 / 68% | 72–84% | 74–88% | 48–54% |

**Long distances (D = 0–31).** Accuracy by D, mean of the three seeds:

| | All | 0 | 1 | 2–3 | 4–7 | 8–15 | 16+ |
|---|---|---|---|---|---|---|---|
| Column alone | 0% | 0 | 0 | 0 | 0 | 0 | 0 |
| Areas 2–4 as frames | 0 / 0 / 0% | 0 | 0 | 0 | 0 | 0 | 0 |
| Area 2 (with mix) | 3 / 11 / 8% | 29% | 24% | 17% | 5% | 5% | 5% |
| + area 3, mixed | 4 / 14 / 17% | 36% | 33% | 27% | 18% | 12% | 4% |
| **+ areas 3, 4, mixed** | 9 / 23 / 14% | **38%** | **38%** | **32%** | **23%** | **17%** | **9%** |

Cost at test (long distances): area 2 alone 31–40 µs/word; with area 3, 90–105; with
areas 3 and 4, 185–195. Kernels: area 2 1,400–4,100, area 3 600–2,500, area 4 900–2,000.

## Findings
1. **Each added area extends the reach.** With the mix, accuracy rises with every area at
   every distance beyond what the area below covers: at 4–7 filler stories 5 → 18 → 23%,
   at 8–15 5 → 12 → 17%. This is the expected ordering: the 16-sentence window covers
   up to about 7 filler stories, the 64-sentence one up to about 30.
2. **Upper areas must not feed the area below as a frame.** As an extra input frame of
   area 2, an upper area's noisier prediction became the key of area 2's deepest kernels.
   Area 2 got worse, and on two seeds the column stopped copying it altogether (0%). This
   is the frame-order problem of [24](24-cortical-hierarchy.md) again. As sources in the
   precision-weighted mix, a weak area is simply weighted down: no harm at short
   distances, and a gain at long ones.
3. **Absolute accuracy is low, and the reasons are clear:**
   - **The teaching signal is thin.** Every area learns only from the column's errors,
     and each episode gives one example, shared over 24 (name, season) pairs. With D up to
     3, about 1,500 of the 3,000 training episodes have the announcement inside area 2's
     window, and area 2 reaches 64–88% there. With D up to 31, only about 190 do, and the
     same area 2 reaches 24–29%. The upper areas face the same shortage at their
     distances.
   - **Bags have no order.** A 64-sentence window holds the season words of the last two
     or three episodes as well as the current one. An OR of words cannot say which came
     last, so the top area is often ambiguous.
   - **Growth samples a big bag.** The window holds every surprising word of up to 64
     sentences, and a new kernel samples its key from all of them. A kernel keyed on the
     season word is rare.
4. **Cost grows with every area,** because each runs at every word (about 6× for 3
   higher areas). An event-driven clock, where an area steps only when the area below
   reports a surprise or a window closes, is what makes a deep chain affordable.

## What would fix it
- **Recency in the state.** Let a newer surprising word suppress older ones of the same
  kind, or let the state decay. Then the latest season wins over earlier ones. This is
  what "the most recent context" means for a working-memory-like window.
- **More teaching signal for the upper areas:** self-supervised read-back. An area
  should learn not only from the column's errors but from whether it can regenerate
  what came in. See [output and self-supervision](../concepts/output-and-self-supervision.md),
  the next experiment.
- **An event-driven clock** for the upper areas.

## Biology
- **Temporal receptive windows** lengthen along the cortical hierarchy (Hasson et al. 2008;
  Lerner et al. 2011), and intrinsic timescales lengthen with it (Murray et al. 2014).
- **Mixing rather than chaining.** Higher areas project back to many lower areas and to
  the pulvinar, not only to the area directly below (Felleman & Van Essen 1991; Shipp
  2003). The mixed chain, in which every area's prediction reaches the final combination
  with its own weight, is closer to that than a strict relay down the chain.
