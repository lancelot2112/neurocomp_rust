# 65 · A cue controller: the basal ganglia choose how the hippocampus is cued

**Question.** Recall is automatic: the hippocampus is cued at every word and thresholds
decide whether anything fires. The walk ([59](59-engram-walk.md)) is a hand-set edit of
the cue: when a rare word of the cue reached a row from another story, bridge through it.
Can a controller learn how to edit the cue instead?

**Code.**
- [`src/program/hippocampal_circuit.rs`](../../src/program/hippocampal_circuit.rs):
  `EpisodicCircuit::recall_as(cue, walk)`, the walk on or off for one recall (the engram
  store overrides it).
- [`examples/episodic.rs`](../../examples/episodic.rs): `CUE_CTL=1`. At every recall the
  basal ganglia choose one of four edits, per (the column's confidence band × the
  sentence's familiarity band):
  - **as is:** the automatic cue (this sentence's bindings + the story's context);
  - **walk:** the same, the bridge allowed;
  - **content only:** the story's context dropped;
  - **focus:** the sentence's least familiar binding + the story's context.
  - Reward: the recalled word is the next word read (1, else 0), minus `CUE_COST`
    (default 0.02) for the walk's second recall. Learns in training (not in replay);
    `CUE_TEST_LEARN=1` at test too.
  - The `CUE` report gives the choices, how often each gave the next word, and the
    learned policy table.

## Results (walk-only setup of 60: no semantic store, no rollout; hippocampus intact)

| Cue | Held out | Trained |
|---|---|---|
| no walk (60) | 21 / 17 / 25% | – |
| **walk always (hand rule, 60)** | **91 / 95 / 91%** | 89.8% |
| controller, learns in training | 20 / 91 / 61% | 81.1% |
| controller, learns at test too | 86 / 86 / 51% | 75.1% |
| controller, at test too, no walk cost | 83 / 87 / 65% | 74.3% |

At test (last row) each edit gave the next word about equally often (as is 72–74%, walk
72–79%, content only 69–72%, focus 56–70%), and the learned policy is close to random
across contexts.

## Findings
1. **The learned controller is worse than the hand rule** (78.5% at best against 92%),
   and costs trained names too (74–81% against 90%).
2. **Its reward does not see where cueing matters.** The walk pays at the question of a
   story with a new name, a few hundred steps per run; the next-word reward is dominated
   by the thousands of steps where every edit recalls the current story's own sentence.
   The four edits score within a few points of each other there, so the policy is noise.
3. **Its context lacks what the hand rule uses.** The rule bridges when a cue word is rare
   *and* the winning row is from another story. Confidence × familiarity bands do not
   carry that.
4. **What it would take:** reward at the answer (as learned stepping, 43, did), and a
   context that includes what the first recall returned (another story's row, a weak
   match). That is a controller that *reacts to the first recall*, which is what a
   prefrontal controller is thought to do: try, read the result, re-cue.

Not kept as a default.
