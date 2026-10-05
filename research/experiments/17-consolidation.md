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
| hippocampus, unlimited capacity: normal / anchor | UNLIMITED | | |

With consolidation the cortex supplied the memory frame at all 500 anchor answers on every
seed, from a semantic store of 286–331 kernels. (Hippocampus-only anchor scores of ~37% are
the guessing floor: with no recall the predictor falls back on one fixed place, right for
one of the three anchors.)

## Findings
1. **Replay transfers facts from hippocampus to cortex.** Facts stated only early in
   training, long overwritten in the hippocampus, are answered perfectly from the cortical
   store; without replay they are lost.
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
