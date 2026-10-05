# 17 · Consolidation: hippocampal replay into a cortical semantic store

**Question.** The hippocampus ([11](11-episodic-memory.md)–[12](12-dentate-gyrus-ca3.md))
stores episodes one-shot but has limited capacity: old episodes are overwritten. Can
replay during "sleep" move facts into cortex, so they survive after the hippocampus has
forgotten them (complementary learning systems, [McClelland, McNaughton & O'Reilly 1995](../related-work.md#hippocampus-and-entorhinal-cortex))?

**Code.** `TASK=persist`, `Policy::Consolidate` (`POLICIES=consolidate`) in
[`examples/episodic.rs`](../../examples/episodic.rs); `EpisodicMemory::get` for replay.

## Task
- **Anchor names** (bill, fred, julie) each have one fixed place, stated only in the first
  300 of 3,000 training stories ("bill went to the kitchen .") and never again.
- **Normal names** (the six of the earlier tasks) move around in varied stories with
  questions.
- **Test:** half the questions ask about an anchor ("where is bill right now ?"). The anchor
  facts are thousands of episodes old; the hippocampus holds 200. Anchor questions are
  reported in the "held-out" column, normal ones in "seen pairs".

Two design bugs had to be fixed first:
1. With only three normal names, a name was as frequent as "where / is / right / now",
   so the recall cue included the question words and recalled an *earlier question* (and
   its answer): hippocampus-only was at chance even on normal questions. Anchors now
   have their own names, and all six normal names stay active.
2. The memory keeps storing at test ("storing is reading"), and a question's episode
   contains its answer ("… ? kitchen ."). The first anchor question then leaked its answer
   to every later one (hippocampus-only scored 99.4% on anchors). Test questions are no
   longer stored in this task.

## Mechanism
- **Sleep replay:** after each training story, the hippocampus replays 5 random stored
  episodes (`REPLAYS`).
- **Cortical semantic store:** a separate predictive `KernelClass` (one frame) learns
  cue → content from each replay: cue = the episode's rarest content (the name), target =
  its novel content minus the cue (the place, plus incidental words).
- **Waking recall:** the hippocampus first; if it recalls nothing, the cortex's prediction
  (`peek`) fills the memory frame the predictor reads.

## Results (varied-style stories, held-out = anchor questions; copy-credit predictor of [12](12-dentate-gyrus-ca3.md))

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| no memory: normal / anchor | 17.2 / 0% | 15.8 / 0% | 13.4 / 0% |
| hippocampus only (200 episodes): normal / anchor | 100 / 37.2% | 100 / 36.6% | 100 / 0% |
| **hippocampus + consolidation**: normal / anchor | **100 / 100%** | **100 / 100%** | **100 / 100%** |
| hippocampus, unlimited capacity: normal / anchor (upper bound) | 100 / 100% | 100 / 100% | 100 / 100% |

With consolidation the cortex supplied the memory frame at all 500 anchor answers on every
seed, from a semantic store of 286–331 kernels. (Hippocampus-only anchor scores of ~37% are
the guessing floor: with no recall the predictor falls back on one fixed place, right for
one of the three anchors.)

## Prioritised replay: questions decide what is consolidated
Settings that make the replay budget matter: anchors stated only in the first **30** stories
(`ANCHOR_STORIES=30`), sleep replays **1** episode after only **30%** of stories
(`REPLAYS=1 REPLAY_PROB=0.3`). During those 30 stories, half the training questions ask
about the anchor just stated (retrieval practice). Three modes (`REPLAY_MODE`):
- **random**: sleep replay picks episodes uniformly.
- **tagged**: when a question's hippocampal recall *contained the answer*, that episode is
  tagged; sleep replay picks episodes with weight 1 + 20 × tags. The tag stores the
  question's cue, and replaying a tagged episode trains the cortex under that cue.
- **awake**: that episode is replayed into the cortex *at once, at the question*, under the
  question's cue: a prefrontal-driven retrieval that forces a consolidation event.

| Anchor questions, small budget | Seed 0 | Seed 1 | Seed 2 | Cortical kernels |
|---|---|---|---|---|
| random replay | 33.6% | 64.2% | 66.2% | 159–181 |
| **tagged sleep replay** | **100%** | **100%** | **100%** | **90–103** |
| **awake replay at the question** | **100%** | **100%** | **100%** | 178–212 |

(Normal questions: 100% in every run.)

Two things had to be right, found by the runs that failed first:
1. **Credit = the recall contained the answer**, not "the predictor answered correctly".
   The anchor questions come in the first 30 stories, before the predictor has learned to
   read memory, so answer-based credit missed them (tagged 30–68%, awake 65–100%).
2. **The cortical key is the question's cue.** Keying a replay on the episode's rarest bits
   uses the bit counts at replay time; early on every count is tiny, so the key could be an
   adverb rather than the name, and the test-time cue (the name) did not match (awake on
   seed 2: 67% → 100%; tagged: ~30% → 100%).

**What this shows:** being asked about something, and having the hippocampus answer it,
is what should decide consolidation. The prefrontal query supplies both *which* episode
(the one recalled) and *under what key* (what was asked). Tagged sleep replay reaches the
same accuracy as awake replay with about half the cortical kernels, because sleep replays
repeat the same few tagged episodes instead of every question's recall.

## Findings
1. **Replay transfers facts from hippocampus to cortex.** Facts stated only early in
   training, long overwritten in the hippocampus, are answered perfectly from the cortical
   store; without replay they are lost. Consolidation reaches the upper bound of a
   hippocampus that never forgets (100% on every seed), with a cortical store of ~300
   kernels instead of every episode ever seen.
2. **Two stores, one reader:** the predictor does not know where its memory frame came
   from. Copy kernels trained on hippocampal recall work unchanged on cortical recall,
   because both produce the same kind of content (the place's word code).
3. **Caveats, and what's next:**
   - The cortical store learns one-shot (surprise-driven growth), so this shows *transfer*,
     not the slow, interleaved learning of CLS. Next: a slow cortical learner (several
     replays per fact) and interference between old and new facts.
   - Replay is random. **Prioritised replay** — tag episodes that a question recalled and
     that predicted the answer (the testing effect; [Mattar & Daw 2018](../related-work.md#hippocampus-and-entorhinal-cortex))
     and replay them more — should matter once the replay budget is small.
   - The anchors never change. Facts that are *updated* need the cortex to unlearn: the
     opposite problem.
