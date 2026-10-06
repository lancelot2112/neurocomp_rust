# 28 · Reading with actions: opening a book brings back its context

**Question.** In [26](26-context-and-readback.md)–[27](27-boundary-detection.md) the
network either was told where each story starts or inferred it from contradicted facts.
A real reader knows from their own actions: they open a book, turn a page, put it down.
The efference copy of an action marks the context change, and picking up a book you
have been reading brings its context back (context reinstatement). Can actions do the
same here, in a setting where contradiction detection must fail: returning to a book
read earlier?

**Code.**
- `HigherArea::save_context`, `restore_context` and `AreaContext` in
  [`src/program/cortex.rs`](../../src/program/cortex.rs).
- In [`examples/episodic.rs`](../../examples/episodic.rs): `TASK=books`,
  `BOOK_CTX=none|reset|reinstate|learned`, `BOOK_IDS`, and the `BOOKS` report.

Run (best setting): `COPY_GROW=1 GROW_TRUST=0.5 STICKY=255 GENERALIZE=0.5
GENERALIZE_AFTER=1 TRUST_AT_TEST=0.5 GROW_GATE=3 SLEEP_EVERY=500 SURPRISE_GATE=1 MEMO=1
CANON=1 TASK=books POLICIES=nomemory HIER=1 HIER_LEVELS=3 HIER_CHAIN=mix MIX=1
BOOK_CTX=reinstate BOOK_IDS=12 BOUNDARY=1 cargo run --release --example episodic`.

## Task: books
- **Three books at a time,** each with its own season, read in interleaved sessions.
- **A session:** `@open_x` (an action: book x is opened), the season announcement if
  this is the book's first session, 0–3 filler stories, "X went to the" → X's place in
  that book's season, then `@close`.
- **A book lasts 3–8 sessions,** then a new book with a new season replaces it.
- **Returning to a book,** its season was announced one or more sessions ago, often with
  other books (and their seasons) read in between.

**Actions:**
- They are tokens on the input stream, like an efference copy reaching cortex.
- With 3 actions (default), a new book reuses its slot's action.
- With `BOOK_IDS=12`, each new book gets the next of 12 actions, like a different
  physical book.

**What an `@open` does to the areas' windows:**
- **none:** nothing.
- **reset:** forget.
- **reinstate:** restore the windows saved at this book's last `@close` (reset if there
  are none). The store is a lookup keyed by the book's action: an idealised hippocampal
  recall by context, not a learned one.
- **learned:** a basal-ganglia choice of keep, reset or reinstate per `@open` action,
  rewarded by whether the session's answer comes out right.

## Results (held-out; seeds 0 / 1 / 2; three higher areas, mixed)
Chance is about 25%: the name narrows the place to one per season.

| `@open` does | 3 actions (slots reused) | 12 actions (one per book) |
|---|---|---|
| nothing | 33 / 25 / 25% | 29 / 29 / 27% |
| nothing, with contradiction detection (27) | 27 / 30 / 28% | – |
| reset | 34 / 31 / 28% | 30 / 27 / 26% |
| reinstate | 51 / 34 / 40% | 38 / 40 / 40% |
| **reinstate + contradiction detection** | 51 / 47 / 43% | **54 / 47 / 41%** |
| learned (keep / reset / reinstate) | 37 / 30 / 32% | – |
| learned + contradiction detection | 41 / 38 / 27% | 29 / 36 / 42% |
| reinstate, one higher area | 38 / 29 / 31% | – |

**By session** (reinstate + detection, 12 actions):

| Seed | First session of a book | Back to back | After 1–2 other sessions | After 3+ others |
|---|---|---|---|---|
| 0 | 45% | 53% | 55% | 59% |
| 1 | 42% | 53% | 46% | 41% |
| 2 | 41% | 41% | 45% | 43% |

**Reset** is good only in a book's first session (34–57%) and at chance after it, as
expected: it throws away the book's season.

The learned selector settled on mixtures (for example keep 33%, reset 35%, reinstate 32%)
and did no better than fixed reinstatement.

Cost: 200–500 µs/word at test (three higher areas and 12-action vocabulary; runs shared
the machine).

## Findings
1. **Actions bring back context.** Reinstating each book's windows at `@open` lifts
   accuracy from chance (25–33%) to 34–51%, and to 41–54% together with contradiction
   detection. Returning after three or more other sessions is about as good as reading back to
   back: the action recovers what interleaving would otherwise have destroyed.
2. **Actions and contradictions are complementary.** When a slot's action is reused for
   a new book, reinstatement restores the old book's season, and the new announcement
   then contradicts it. Detection cleans that up (seed 1: 34 → 47%).
3. **But the ceiling is far below the season task (90–99%),** even in a book's first
   session (41–45%), where the announcement is in the same session. Two limits, neither
   about actions:
   - **Crowded windows.** A book is read over several sessions, and every question
     leaves its name and place in the windows. A restored context holds the season plus
     five or six earlier question–answer pairs, and growth samples a new kernel's key
     from that whole bag. In the season task each window held one story with one
     question.
   - **Contradiction detection over-fires within a book.** It assumed one fact of each
     kind per story. Within a book, the second session's question name contradicts the
     first session's, so it "finds a boundary" there and forgets through it, season
     included. Its success in 27 rested on the season task's one-question-per-story
     shape.
4. **The learned choice did not beat a fixed one.** The reward (this session's answer)
   is near chance under every choice early in training, so the selector has little to
   learn from. It needs the areas to work first.

## What is hand-supplied (audit)
General mechanisms, the same on every task:
- the column (predictive kernels, surprise-driven growth);
- hippocampal memory;
- the basal ganglia;
- precision-weighted mixing;
- the chain of areas with longer windows, learning the column's residual.

Hand-set or task-tuned, to be replaced by learning:

| Piece | Where | What would make it general |
|---|---|---|
| Sentence as the window unit ("." is given) | areas | segment by the column's own predictability |
| "Rare" = seen in < 2% of sentences | read-back, boundaries | a learned novelty signal |
| "Same kind" = Jaccard ≥ 0.5 of neighbouring words | boundaries | learned categories (cf. [05](05-syntax.md)) |
| One fact of each kind per context | boundaries | per-kind capacity, learned from how often a kind repeats within a context |
| Decoding to word identities (`enc.decode`) | mixing, boundaries, read-back | the same operations on bit codes inside the network |
| Context store keyed by the action | reinstatement | hippocampal recall cued by the action's code |
| The reset given at story start | 26 | an action (this page) or detected (27) |

## Next
- **Reach for the book.** The actions here are imposed on the reader. The network should
  choose them:
  - a question names a book ("in book c, where did john go?");
  - the basal ganglia choose which book to open;
  - the reinstated context answers it.

  Choosing the action is then rewarded by the answer. This is active retrieval: acting
  to bring back the context one needs, as a person reaches for the right book.
- **Less crowded states,** so a restored context is usable. Facts should be held per
  kind, the newest of each kind kept (the recency of [26](26-context-and-readback.md)),
  instead of a bag of every surprising word.
- **Contradiction detection with learned capacity per kind.**

## Biology
- **Efference copy marks self-caused change:** a corollary discharge of saccades reaches
  frontal cortex through the mediodorsal thalamus (Sommer & Wurtz 2002), so the brain
  can tell "I moved" from "the world changed".
- **Context-dependent memory:** recall is best when the encoding context is reinstated
  (Godden & Baddeley 1975; Tulving & Thomson 1973). The temporal context model
  reinstates the context of a recalled item (Howard & Kahana 2002).
