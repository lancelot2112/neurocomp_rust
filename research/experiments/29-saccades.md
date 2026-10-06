# 29 · Active reading: look back instead of holding everything

**Question.** In [25](25-area-chain.md)–[28](28-reading-with-actions.md) the network had
to hold every fact it might need in its areas' windows. That took a chain of three
areas, context boundaries, and still failed when windows got crowded. A human reader
does not hold everything. The page stays in front of them, and when understanding fails
they look back: 10–15% of saccades in reading are regressions (Rayner 1998). Can the
network do the same: keep the page as external memory and learn where to look?

**Code.** [`examples/episodic.rs`](../../examples/episodic.rs): `SACCADE=learned|oracle`,
`SACCADE_COST`, and the `SACCADE` report.

Run: `COPY_GROW=1 GROW_TRUST=0.5 STICKY=255 GENERALIZE=0.5 GENERALIZE_AFTER=1
TRUST_AT_TEST=0.5 GROW_GATE=3 SLEEP_EVERY=500 SURPRISE_GATE=1 MEMO=1 CANON=1 TASK=season
POLICIES=nomemory HIER=1 HIER_LEVELS=1 MIX=1 SACCADE=learned cargo run --release
--example episodic`.

## Active reading
- **The page stays available:** the story's words, in order.
- **At every word** the basal ganglia choose a saccade, per context (previous word,
  current word; action codes bound to it by rotation, as for every selector):
  - **read on** (0);
  - **look back to the previous sentence** (1);
  - **look back to the top of the page** (2).
- **A regression re-reads the target sentence.**
  - It is perception only. The column predicts word by word, and surprise is computed as
    usual, but nothing learns.
  - The surprising words enter the areas' windows as a newly read sentence.
  - The eye then returns: it fixates the previous and the current word again, so the
    column's context is as before, and the next word is predicted with the refreshed
    windows.
- **Learning:** the choice is rewarded by whether the next prediction came out right,
  minus `SACCADE_COST` (default 0.1) for a regression. Exploration 10% in training;
  greedy at test.
- **Oracle** (for comparison): one look back to the page top, exactly at the answer.

The task is the season task of [25](25-area-chain.md): an announcement, 0–31 filler
stories, then the question.

## Results (held-out; seeds 0 / 1 / 2; one higher area, window 4 sentences)

| | All | 0 | 1 | 2–3 | 4–7 | 8–15 | 16+ | Regressions per story | Re-read words | Test µs/word |
|---|---|---|---|---|---|---|---|---|---|---|
| Reading straight through | 3 / 11 / 8% | 12–41% | 16–29% | 8–26% | 0–9% | 1–7% | 2–9% | – | – | 57–70 |
| + story boundary (26) | 18 / 20 / 29% | 50–86% | 66–88% | 26–57% | 11–27% | 12–26% | 12–21% | – | – | 51–56 |
| *3 higher areas + boundary (26)* | *86 / 96 / 98%* | 65–86% | 79–100% | 91–98% | 85–100% | 87–100% | 87–100% | – | – | 120–130 |
| Oracle saccade | 95 / 84 / 93% | 38–70% | 82–89% | 83–98% | 88–95% | 88–98% | 84–96% | 1.0 | 2.7% | 58–68 |
| **Learned saccades** | **99.6 / 88 / 97%** | 78–96% | 83–96% | 94–99% | 86–100% | 88–100% | 88–100% | 11–42 | 23–59% | 66–74 |
| Learned, cost 0.3 | 97 / 91 / 86% | 65–77% | 84–100% | 91–100% | 88–95% | 89–97% | 88–98% | 1.2–25 | 3–42% | 75–83 |
| Learned + story boundary | 99 / 44 / 99% | 68–100% | 49–100% | 48–99% | 42–99% | 37–99% | 45–100% | 11–36 | 25–50% | 61–66 |

The learned reader regressed right before every test answer (100%).

## Findings
1. **Looking back replaces holding.** With one higher area (a 4-sentence window), learned
   saccades answer 88–99.6% at every distance, against 3–11% reading straight through.
   They match the three-area chain with story boundaries (86–98%) at about half the cost
   (66–74 against 120–130 µs/word). The fact does not have to survive in a window for
   32 stories: the reader fetches it from the page when it is needed.
2. **The page also solves recency.** Without a story boundary, re-reading puts the
   current story's announcement into the short window as the newest sentence, and older
   seasons are long gone from it. The boundary detection of
   [27](27-boundary-detection.md) is not needed here; adding the given boundary changes
   little (and one seed did worse, 44%).
3. **Where to look is learned.** Nothing says that the answer needs the page top. The
   selector learns it from the reward of the next prediction: at "to the" it looks back,
   and it did so before every test answer.
4. **When to look is learned too generously.** The learned reader regresses 11–42 times
   per story, so 23–59% of the words it reads are re-reads. The oracle needs one regression
   per story (2.7% re-reads) for 84–95%.
   - Regressing where it does not help costs only a little reward (0.1), and the
     selector's 4-bit counters, stepped stochastically, do not separate a 1.0 from a 0.9
     reliably.
   - A cost of 0.3 brings one seed down to 1.2 regressions per story (3.3% re-reads) at
     91%. The others still regress 9–25 times.
   - Better: let the regression decision see the column's confidence (look back when
     unsure, as human regressions follow comprehension difficulty).
5. **Hand-supplied here** (see the audit in [28](28-reading-with-actions.md)):
   - The saccade targets are "the previous sentence" and "the page top". The season task
     puts its fact at the page top, so "page top" is a convenient landmark.
   - A general reader needs to find where a fact was: a spatial memory of where words of
     a kind appeared on the page (the hippocampus' "where" of the page). Then it can
     saccade to the location of the last season word, wherever it is.

## Learned targets (a page index) and confidence
The two hand-supplied parts above were the fixed targets and the missing "am I unsure?"
signal. Two options replace them:
- **Page index (`SACCADE=index`).** The page keeps an index of where each surprising
  word was read (its landmarks). A regression's candidates are "read on" plus one
  candidate per landmark word, at its latest position before the current sentence. The
  candidate's code is the word's own code, bound to the context by rotation, so the
  selector learns a value per (context, landmark word), as it learned per recalled item
  in [15](15-basal-ganglia-selector.md). Choosing a landmark re-reads the sentence
  holding it. Nothing marks the announcement; the reader must learn where the season was.
- **Confidence (`SACCADE_CONF=1`).** Before choosing, the column peeks at its own
  prediction of the next word (without top-down, no state change). That confidence (no
  prediction, < 0.5, < 0.8, ≥ 0.8) joins the selector's context, so it can learn to look
  back only when unsure.

| Seeds 0 / 1 / 2 | Accuracy | Regressions per story | Re-read words | Before the answer, re-read the first sentence |
|---|---|---|---|---|
| Fixed targets (above) | 99.6 / 88 / 97% | 11 / 42 / 26 | 23–59% | – |
| Fixed targets + confidence | 79 / 94 / 88% | 26 / 35 / 25 | 43–57% | 0 / 100 / 0% |
| **Page index** | **97 / 90 / 97%** | 15 / 20 / 30 | 30–49% | 74 / 100 / 100% |
| Page index + confidence | 95 / 84 / 85% | 29 / 28 / 24 | 45–52% | 100% |
| Page index, cost 0.3 | 90 / 76 / 94% | 18 / 10 / 10 | 25–38% | 98 / 77 / 100% |
| Page index + confidence, cost 0.3 | 90 / 91 / 96% | 19 / 4.4 / 17 | 11–40% | 100% |
| Page index, cost 0.5 | 84 / 90 / 96% | 14 / 16 / **1.2** | 3–35% | 74 / 100 / 100% |
| Page index + confidence, cost 0.5 | 82 / 79 / 92% | 15 / **1.5** / **1.05** | 3–41% | 77 / 77 / 100% |

1. **Targets are learned.** With the page index, the reader learns which landmark to
   look back to. Before almost every answer it re-reads the sentence holding the season
   (74–100%), and accuracy matches the fixed "page top" target (90–97%). The last
   hand-supplied landmark is gone.
2. **Confidence did not reduce regressions on its own.** With a 0.1 cost, adding the
   confidence bucket left 24–35 regressions per story and lowered accuracy on some
   seeds. The bucket splits every context four ways, so each value is learned from a
   quarter of the data.
3. **The efficient policy is learnable, but not reliably.**
   - With a higher regression cost, some seeds find about one regression per story
     (1.05–1.5, 3–4% re-reads) at 79–96%: the oracle's policy, learned. Others stay at
     10–19.
   - The selector's 4-bit counters, stepped stochastically, separate "read on, 1.0"
     from "look back, 0.9" only noisily. Whether a seed settles on looking back
     everywhere or only at the question is close to a coin flip.
   - Finer value resolution (8-bit counters, or the integer rates of
     [`KernelStats`](../../src/kernel/simple.rs)) is the likely fix, rather than more
     context.

## Biology
- **Regressions in reading** follow comprehension difficulty; about 30% of words are
  skipped and 10–15% of saccades go backwards (Rayner 1998). Models such as E-Z Reader
  make where to look next a decision driven by processing (Reichle et al. 1998).
- **Saccade choice** runs through the basal ganglia and superior colliculus, with frontal
  eye fields setting the target (Hikosaka et al. 2000). Here the basal ganglia choose
  the saccade from reward, as they choose recall and relays elsewhere in the network.
- **The world as external memory:** people re-fixate to fetch information just in time
  instead of holding it in working memory (Ballard, Hayhoe et al. 1995, 1997).
- **Efference copy:** the return saccade restores the column's context. The brain likewise
  uses the corollary discharge of each saccade to keep perception stable across eye
  movements (Sommer & Wurtz 2002).

## Next
- ~~Where facts are~~: done (page index, above).
- **Reliable "when":** finer value resolution in the saccade selector, so the
  one-regression policy is found on every seed (now some seeds only).
- **Skipping:** a fourth action, skip the next word when the column is confident (about
  30% of words in human reading).
- **Books again:** reach for the book (an action), then look back on its pages, instead of
  holding each book's context in windows ([28](28-reading-with-actions.md)).
