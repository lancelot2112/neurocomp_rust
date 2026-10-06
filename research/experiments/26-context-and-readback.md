# 26 · Context boundaries and read-back: keeping the right facts in reach

**Question.** In [25](25-area-chain.md) a chain of areas extended the network's reach,
but accuracy stayed low (9–23% on the long season task). Two causes were proposed:
- **No recency.** A long window holds the season words of several earlier stories.
- **Too little teaching signal** for the upper areas.

This experiment tests one fix for each:
- **Context boundaries:** is this still the same story?
- **Self-supervised read-back:** say back what you hold, and hear it.

**Code.**
- `HigherArea::clear` and `recent_words` in [`src/program/cortex.rs`](../../src/program/cortex.rs).
- In [`examples/episodic.rs`](../../examples/episodic.rs): `HIER_RESET`, `READBACK=top|all`,
  `READBACK_RARE`, and the `READBACK` report.

Run (best setting): `COPY_GROW=1 GROW_TRUST=0.5 STICKY=255 GENERALIZE=0.5
GENERALIZE_AFTER=1 TRUST_AT_TEST=0.5 GROW_GATE=3 SLEEP_EVERY=500 SURPRISE_GATE=1 MEMO=1
CANON=1 TASK=season POLICIES=nomemory HIER=1 HIER_LEVELS=3 HIER_CHAIN=mix MIX=1
HIER_RESET=1 cargo run --release --example episodic`.

## The two mechanisms
- **Context boundary (`HIER_RESET=1`).** At the start of a story every area forgets its
  window. A story boundary is a context boundary: the season of the last story is no
  longer the season. Here the boundary is given, as a reader knows when a new story
  starts.
- **Read-back (`READBACK=top` or `all`).** At each sentence end, the top area (or every
  area):
  1. is probed with the cue "." in its sentence frame plus its window;
  2. says the word it predicts;
  3. its target is the most recent *rare* word its window holds (seen in fewer than 2%
     of sentences so far), taken from its own input, so this is self-supervised;
  4. learns from a mismatch (training only);
  5. hears what it said: the word joins the next sentence's surprising words and so
     re-enters every area's window. This is rehearsal, as in the phonological loop.

## Results (season task, 0–31 filler stories; seeds 0 / 1 / 2)

| | All | 0 | 1 | 2–3 | 4–7 | 8–15 | 16+ | Test µs/word |
|---|---|---|---|---|---|---|---|---|
| 1 higher area | 3 / 11 / 8% | 12–41% | 16–29% | 8–26% | 0–9% | 1–7% | 2–9% | 57–70 |
| 3 higher areas (25) | 9 / 23 / 14% | 18–50% | 29–44% | 22–45% | 16–29% | 10–22% | 2–19% | 125–155 |
| 1 higher area + reset | 18 / 20 / 29% | 50–86% | 66–88% | 26–57% | 11–27% | 12–26% | 12–21% | 51–56 |
| 2 higher areas + reset | 36 / 54 / 56% | 55–86% | 77–88% | 84–96% | 77–95% | 35–86% | 16–24% | 79–84 |
| **3 higher areas + reset** | **86 / 96 / 98%** | 65–86% | 79–100% | 91–98% | 85–100% | 87–100% | **87–100%** | 120–130 |
| 3 + read-back (top) | 18 / 14 / 25% | 15–53% | 18–40% | 25–32% | 10–27% | 12–21% | 10–24% | 144–187 |
| 3 + read-back (all areas) | 18 / 24 / 26% | 12–35% | 32–43% | 18–32% | 22–30% | 18–27% | 18–26% | 184–301 |
| 3 + reset + read-back (top) | 75 / 87 / 90% | 52–66% | 71–85% | 81–88% | 77–96% | 76–90% | 77–94% | 182–192 |
| 1 + read-back (all) | 27 / 33 / 31% | 36–47% | 41–55% | 37–43% | 25–40% | 26–36% | 21–32% | 80–104 |
| 1 + reset + read-back (all) | 12 / 46 / 18% | 48–71% | 51–85% | 21–68% | 9–33% | 5–47% | 4–41% | 77–105 |

At test, the reading-back area said the target word at 80–97% of read-backs (40–61% when
every area reads back).

## Findings
1. **Context, not capacity, was the bottleneck.** With a boundary at each new story, the
   same three areas go from 9–23% to 86–98%. They answer 87–100% of questions with the
   announcement 16 or more filler stories back. In 25 the long windows held the season
   words of two or three earlier stories, and a bag of words cannot say which is current.
   Clearing them at the boundary leaves only the current story's season.
2. **Each area extends the reach exactly as its window predicts:**
   - **1 higher area** (4 sentences): good at 0–1 filler stories back.
   - **2 higher areas** (16): good up to 4–7 back, partly at 8–15.
   - **3 higher areas** (64): good at every distance tested.

   With context boundaries in place, depth buys reach.
3. **Read-back is learned well, but it does not add to a chain with boundaries.**
   - The areas learn to say back the fact they hold (80–97%).
   - **Without a reset,** rehearsal helps a single area a lot (3–11% → 27–33%, flat in
     distance): saying the season word again keeps it inside area 2's short window.
   - **With a reset and three areas,** the chain already holds the fact. Rehearsal then
     costs 7–11 points, because the rehearsed word is added to every sentence and
     changes the contexts the areas learn from.
   - **Read-back cannot replace the boundary.** Without one, it rehearses stale words of
     earlier stories as faithfully as current ones.
4. **Cost.** The reset is free. Read-back adds 20–50% (one more prediction per area per
   sentence).

## How to hold order, recency and context
Three separate things, from coarse to fine:
1. **Context: is this still the same story?** Event boundaries. People segment experience
   into events at points of high prediction error (event segmentation theory: Zacks et
   al. 2007). Cortical patterns shift sharply at those boundaries (Baldassano et al.
   2017), and memories are bound within events and separated across them (DuBrow &
   Davachi 2013). Here the boundary is given (`HIER_RESET`); it was the decisive fix.
   **Next:** detect it, as a sustained jump in the column's surprise, or as a learned
   "new story" signal, instead of being told.
2. **Recency within a context.** A drifting context: each area's state decays a little
   every sentence, so newer items are stronger (the temporal context model: Howard &
   Kahana 2002). In bits:
   - each sentence, each state bit survives with probability p (p = 0.5^(1/h) for a
     half-life of h sentences, matched to the area's window);
   - a newly surprising word enters with all its bits;
   - kernels match a fraction of their sampled bits, so an old word with half its bits
     gone no longer matches, and the newest of two competing words wins.

   Not built yet. It would replace fixed windows with forgetting curves.
3. **Order.** A bag keeps what, not when. Two complementary ways:
   - **Position binding:** rotate a word's code by its lag before adding it to the
     state, as the basal ganglia binds actions to context. "Winter, then summer" then
     differs from "summer, then winter".
   - **Leave order to the column:** the column predicts word by word and so holds local
     order already. The areas hold the gist (which facts are current), and read-back
     can replay them in order when needed (recitation).

## Biology
- **Event boundaries:** Zacks et al. 2007; Baldassano et al. 2017; DuBrow & Davachi 2013.
  Hippocampal activity rises at event boundaries, where the past event is consolidated
  and the next one begins (Ben-Yakov & Henson 2018).
- **Rehearsal:** inner speech keeps items in the phonological loop (Baddeley 1986). Here
  read-back rehearsal kept a fact within a single area's short window.
- **Temporal context:** a slowly drifting context representation gives recency and
  contiguity effects in memory (Howard & Kahana 2002).
