# 35 · A fading state, and what the entorhinal cortex would add

**Question.** The schema test of [34](34-schema-test.md) failed because the higher
area's state, a bag of every surprising word in its window, is no clean code for "the
current setting". Episodes bound to it recalled on filler words, not on the season. Does
a fading state, where recent words are strongest, give that clean context?

**Code.**
- `HigherArea::set_fade` in [`src/program/cortex.rs`](../../src/program/cortex.rs), with
  a unit test.
- In [`examples/episodic.rs`](../../examples/episodic.rs): `HIER_FADE`.

## The fading state
- **One context vector instead of a window.** At each sentence end, each of its bits
  survives with probability p = 0.5^(1 / half-life), and the sentence's surprising words
  then enter with all their bits.
- **The half-life is the area's span times `HIER_FADE`:** 4, 16 or 64 sentences at 1.
- **Recency is in the code:** a word's strength is how many of its bits remain. The unit
  test checks that a word 9 sentences old keeps under half its bits (half-life 4), a new
  one all of them, and that `clear` empties the state.
- **The drifting context** of the temporal context model (Howard & Kahana 2002), and of
  lateral entorhinal cortex, whose population activity drifts with time and so encodes
  when (Tsao et al. 2018).

## Results (seeds 0 / 1 / 2)

| Season task, 0–31 filler stories, no story boundaries | Window (25) | Fading, half-life = span | Fading, half-life = span / 2 |
|---|---|---|---|
| 1 higher area | 3 / 11 / 8% | 18 / 21 / 12% | 18 / 18 / 22% |
| 3 higher areas | 9 / 23 / 14% | 24 / 18 / 21% | 24 / 26 / 28% |
| *3 higher areas + story boundary (26), for reference* | *86 / 96 / 98%* | | |

| Schema test ([34](34-schema-test.md)), episodic memory, context-bound episodes | Trained names | New names, shown once | New names, shown 4 times |
|---|---|---|---|
| Window (34) | 55–68% | 5–16% | 8–18% |
| Fading state | 31–42% | 14–18% | 15–24% |
| No schema, fading | 5–18% | 14–17% | – |

## Findings
1. **Fading helps a little, and only a little.**
   - On the long season task it lifts both chains (to 12–28%), with accuracy flatter in
     distance than the window gave.
   - It is far from what a story boundary achieves (86–98%).
   - In the schema test, new names move to 14–24% (chance about 17%), and the trained
     names fall by about 25 points.
2. **Recency is not relevance.** The season is announced before the filler stories, so
   the filler words are always more recent than the season. A fading state favours them.
   A context code has to hold the setting because it is the setting, not because it
   came last.
3. **So the missing code is structural, not temporal.** What the episode should be bound
   to is "the setting slot is filled by winter", whatever came after.

## Entorhinal cortex: place, time, grid, schema and graph cells
The entorhinal cortex (EC) is the hippocampus's input and output. It supplies exactly
the kinds of code the schema test needs, in two streams:

| Code | Where | What it encodes | Here |
|---|---|---|---|
| **Place cells** | Hippocampus (CA1/CA3) | A conjunction: this place, in this context (O'Keefe & Dostrovsky 1971) | Episodes as conjunctions of words ([11](11-episodic-memory.md), [12](12-dentate-gyrus-ca3.md)) |
| **Time cells / temporal context** | Hippocampus; lateral EC drifts with time (Tsao et al. 2018) | When, as a slowly changing context | The fading state (this page) |
| **Grid cells** | Medial EC (Hafting et al. 2005) | A metric scaffold for space, reused for abstract spaces (Constantinescu et al. 2016) | None |
| **Object / landmark cells** | Lateral EC | What is where | The page index of landmarks ([29](29-saccades.md)) |
| **Structure ("graph", schema) codes** | Medial EC in the Tolman-Eichenbaum Machine (Whittington et al. 2020) | Position in a learned relational structure, factored apart from what fills it | Role cells ([31](31-role-cells-and-transfer.md)) are a first, coarse version |

The Tolman-Eichenbaum Machine (TEM) is the most relevant model.
- **Factored codes:** medial EC learns structural codes (where am I in this kind of
  structure), lateral EC carries sensory content (what is here). The hippocampus binds
  the two conjunctively.
- **Why that gives schemas:** the same structure code is reused in every environment of
  that kind, so in a new environment only the content has to be bound, in one shot.
- **Tse's rats are that case:** the layout (structure) is known, a new flavour–place pair
  is one new binding.
- **The transformer bridge:** Whittington et al. 2022 show that a transformer's position
  encoding plays the medial-EC role and attention the hippocampal binding.

**What that means here.** The schema test needs three pieces:
1. **A structure code for "the setting slot"** of a story: medial-EC-like, learned. The
   role cells of 31 already found slots such as "sentence start" and "noun after the",
   but too coarse to separate settings from names.
2. **Episodes bound as (structure code ⊗ content):** "setting = winter", "person = tom",
   "place = kitchen". The hippocampus then stores bindings, not bags.
3. **Recall cued by structure:** "what fills the setting slot now?" returns winter,
   however much filler came after. Then "tom, in winter → ?" recalls the one-shot
   episode. Copying from it is name-independent, so it would transfer.

## Next
- **Finer learned structure codes:** role cells keyed on the column's expectation *and*
  the previous role (a sequence of slots), so "setting" and "person" separate. That is
  closer to TEM's path-integrated structure.
- **Hippocampal binding of role ⊗ content** (bit-wise: rotate the content code by the
  role cell's index), stored per episode, recalled by role.
- **Then rerun the schema test.**

## Biology
- **Lateral EC time codes:** Tsao et al. 2018. **Medial EC grid codes:** Hafting et al.
  2005; for concepts, Constantinescu, O'Reilly & Behrens 2016.
- **TEM:** Whittington et al. 2020; the transformer link, Whittington, Warren & Behrens
  2022. **Cognitive maps for abstract knowledge:** Behrens et al. 2018.
