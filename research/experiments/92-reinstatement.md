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

## Addendum: an acetylcholine-like mode from the hippocampus's novelty

**Build** (`ACH=1`; [neuromodulation](../concepts/hippocampal-functions.md#acetylcholine-and-norepinephrine-modes-of-the-hippocampus)).
Each step's recall gives a novelty signal. A tonic level follows it, moving a quarter of the
way each step.
- **High level (novel input), encoding mode:** recall's pull on the cortex is weakened. Only
  a share 1 − ACh of the reinstated state and of the entorhinal feedback passes.
- **Low level (familiar input), retrieval mode:** both pass.

Storage already follows novelty: a familiar event strengthens its row, a new one is
appended. Two novelty signals were tried:
- **Content:** CA1's mismatch, 1 − how much of the cue the best event covers.
- **Episode** (`row_here` on the engram store): content recalled from *another* story counts
  as full novelty ("seen, but not here"), and content from this story counts as CA1's
  mismatch. The hippocampus detects associative novelty of this kind, such as a known object
  in an unknown place.

**Results** (soft reinstatement, `REINSTATE=1`; held-out):

| | Suite, seeds 0–1 (engram store / inference read / relations) | Question task, five seeds | ACh at test |
|---|---|---|---|
| recorded / routing alone | 67.4 / 45.6 / 62.8 | 68.6 | |
| reinstatement | 35.0 / 10.1 / 31.5 | 70.0 | |
| + ACh, content novelty | 44.4 / 12.9 / 37.3 | 70.6 | 0.00–0.10 |
| + ACh, episode novelty | **66.4 / 46.8 / 62.6** | 68.7 | 0.93–1.00 |

1. **Content novelty barely moves.** A sentence's words ("lucy went to the") are always
   covered by some old event, so everything looks familiar.
2. **Episode novelty is a self-computed safety gate.** At test the suite's stories are new
   episodes, so the mode holds reinstatement back, and the damage of soft reinstatement is
   gone without a "this story only" rule. On two entries (inference read, relations) the
   level was not updated by their recall path and stayed at its starting value, so their
   reinstatement was held back by that default.
3. **On the question task it also removes the gain** (70.0 → 68.7, and the other-family errors
   with it). The signal comes from the sentence recall, whose best match is usually an old
   story's question, so the input looks novel here. Meanwhile the reinstatement draws on this
   story's events through its own, place-weighted retrieval. Signal and content come from
   different recalls.

**Next.** Take the mode's signal from the same retrieval that reinstates. When the
place-weighted retrieval finds this story's events, the context is familiar, so reinstate.
When it finds none, or only other stories', encode. One recall, one decision. The update
should also run on every recall path, not one.

