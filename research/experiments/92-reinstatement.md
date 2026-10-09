# 92 · The hippocampus as an index: reinstating the cortical state

**Question.** In index theory (Teyler & DiScenna 1986; Teyler & Rudy 2007), the hippocampus
stores a pointer to the cortical pattern of an experience, and recall *reinstates* that
pattern in cortex ([concept](../concepts/hippocampal-functions.md#an-index-to-the-cortex-not-a-store-of-vectors)).
Our hippocampus stored word bindings and returned words. In [90](90-holding-an-open-question.md)–[91](91-learned-working-memory-hold.md),
every error on the question task was the story's season, stated once at the start; a
learned hold that kept the season was a workaround. If recall instead brought back the
cortical state an event was stored in, would the season context return by itself?

**Code** ([`examples/episodic.rs`](../../examples/episodic.rs), `REINSTATE`):
- **Index:** each event is stored with the higher area's slow context at that moment (the
  entorhinal summary of the cortical state).
- **Reinstatement:** at every step the current sentence's content cues the hippocampus, and
  the states stored with the best events (up to 4) are added to the higher areas' context.
  Recall brings back the state, not words.
- **Which events:**
  - `REINSTATE=1`: soft context-dependent retrieval (`recall_soft`). This story's events get
    a bonus of 8; other stories' events answer only when this story has none.
  - `REINSTATE=here`: this story's events only.
- `STORE_TEST=1`: the hippocampus also encodes test stories, as it is never off. Before
  this, only the question task did.

No learned decision and no hand rule is involved.

## Results

**The question task** (`REINSTATE=1`, learned routing, five seeds, held-out):

| | Per seed | Mean | Season errors (seed 0, of 500) | Other-family errors |
|---|---|---|---|---|
| routing alone | 69.2 / 66.6 / 69.0 / 68.6 / 69.6 | 68.6 | 154 | 0 |
| reinstatement | 72.0 / 72.6 / 66.8 / 69.2 / 69.2 | 70.0 | 133 (100–152 across seeds) | 2–49 |

**The suite's engram entries** (`REINSTATE=1`, five seeds, held-out):

| Entry | Recorded | Reinstated |
|---|---|---|
| engram store | 67.4 | 33.4 |
| engram walk | 66.8 | 38.7 |
| engram walk only | 87.0 | 84.9 |
| inference replay | 59.4 | 39.4 |
| inference read | 45.6 | 8.8 |
| cooperate | 54.0 | 27.4 |
| relations / speak / speech motor | 62.8 / 63.1 / 62.9 | 33.0 / 34.3 / 33.6 |
| belief decides | 82.4 | 65.9 |

**This story only** (two seeds, held-out):

| Entry | Recorded | `here` | `here` + `STORE_TEST` | `STORE_TEST` alone |
|---|---|---|---|---|
| engram store | 67.4 | 61.8 | 69.5 | 69.6 |
| inference read | 45.6 | 44.7 | 43.6 | 41.3 |
| relations | 62.8 | 64.4 | 63.7 | 65.6 |

## Findings
1. **Reinstatement helps a little where the context is the story's own.** On the question
   task, stories are stored at test, and season errors fall. The mean rises by 1.4 (70.0
   vs 68.6), with no learned decision. A new error appears, the other family's place (up
   to 49 of 500 on one seed). The states of the known person's sentences are reinstated
   too.
2. **Reinstating other stories' contexts is destructive.** In the suite, test stories were
   never stored. Soft retrieval therefore fell back to training stories at every test
   step and reinstated their seasons and contexts. Answers fell 15–37 points.
3. **Restricted to this story, reinstatement is neutral on the suite.** With test stories
   stored, it matches test storage alone within noise. Without them, the cortex trains with
   reinstated context and is tested without it, and engram store loses 6 points.
4. **What is reinstated is a coarse state.** About 330–360 bits of context are added at
   every step, from up to four events. Brains gate this: reinstatement is strongest when
   the input is familiar and the hippocampus is in retrieval mode (low acetylcholine), and
   weak while encoding something new
   ([neuromodulation](../concepts/hippocampal-functions.md#acetylcholine-and-norepinephrine-modes-of-the-hippocampus)).
   Ours reinstates always, at full strength.

Nothing becomes a default. `STORE_TEST` changes what the suite measures (test stories feed
the hippocampus), so it is kept as an option.

## Next
- **A mode from the hippocampus's own novelty** (acetylcholine-like). Novel input means
  encode strongly and reinstate weakly; familiar input means retrieve and reinstate.
- **Reinstate the event's own state, not the window's.** Store the state the event was
  *part of*, including its own surprising words, and reinstate only the best event, so a
  known person's sentence does not bring its family along with the stranger's.
