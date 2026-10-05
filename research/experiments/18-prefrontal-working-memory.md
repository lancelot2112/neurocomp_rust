# 18 · Prefrontal working memory: a gated slot that cues recall

**Question.** In the earlier tasks the recall cue was computed when the question arrived
(the rarest bits of the sentence). What if the question itself carries no useful cue
("where is the *person* ?") and the right cue has to be **held** from several sentences
earlier? Can a working-memory slot, loaded by a basal-ganglia gate, learn what to hold
(PBWM: [O'Reilly & Frank 2006](../related-work.md#credit-assignment))?

**Code.** [`src/program/prefrontal.rs`](../../src/program/prefrontal.rs) (`WorkingMemory`,
`PfcGate`); `BasalGanglia::reward_candidate` in
[`src/program/basal_ganglia.rs`](../../src/program/basal_ganglia.rs); `TASK=topic`,
`Policy::Pfc` in [`examples/episodic.rs`](../../examples/episodic.rs).
Run: `COPY_GROW=1 GROW_TRUST=0.5 STICKY=255 GENERALIZE=0.5 GENERALIZE_AFTER=1
TRUST_AT_TEST=0.5 TASK=topic POLICIES=pfc cargo run --release --example episodic`.

## Task
> then mary quickly went all the way over to the old kitchen . the cat slept . it rained .
> where is the person ? **kitchen**

One fact sentence (with optional fillers and modifiers), then 1–3 distractor sentences
("the cat slept .", "the dog ran away .", "it rained ."), then a question that names no one.
Held-out (name, place) pairs are never answered in training. 3,000 training and 1,000 test
stories; chance 1/6.

## Mechanism
- **Working memory:** one slot (`WorkingMemory`), holding a word code until overwritten.
- **Retrieval directed by PFC:** at each word, the episodic recall cue is the slot's
  content (instead of the rarest bits of the sentence); the recalled episode is the
  predictor's memory frame, as in [11](11-episodic-memory.md).
- **Gate (`PfcGate`):** at each word the basal ganglia choose *load* (put the word in the
  slot) or *keep*. The two action codes are bound to the word by rotation, so the
  bit-sliced go counters learn a value per (action, word). Keep is the default: it wins
  ties, so a load has to earn a higher value.
- **Dopamine:** at the training question, reward = 1 if the recall cued by working memory
  contained the answer.
- **Hand-set upper bound:** load every name.

## Results (held-out pairs, 3 seeds)

| Policy | Held-out | Answer in recall |
|---|---|---|
| no memory | 17 / 17 / 14% | 0% |
| hippocampus, cue = rarest bits at the question | 16 / 17 / 15% | 16–19% |
| working memory, hand-set gate (load names) | **100%** every seed | 100% |
| working memory, learned gate, reward over the eligibility trace | 16–21% | 16–17% |
| **working memory, learned gate, credit to the held item's load** | **100%** every seed | 100% |

## Findings
1. **Holding a cue is what makes the question answerable.** With the cue taken at the
   question, recall finds the right episode only by chance (16–19%): "where is the person"
   matches every story. Holding a word from the fact sentence in the slot makes the recall
   (and so the answer) perfect.
2. **The trace-based gate failed: credit spread over every choice.** With one reward
   shared by every load/keep decision in a 24-word eligibility trace (as in
   [15](15-basal-ganglia-selector.md)'s three-factor rule), the values came out muddled.
   Training rewards were 700/3000, and the values (load / keep) were mary 0.50 / 0.65,
   went 0.58 / 0.47 and "." 0.71 / 0.09. The gate loaded at ".", which ends every
   sentence, so each distractor wiped the slot. Three other fixes didn't help: keep as the
   default, a running-average reward baseline, and gain 1.0. All stayed at 16–22%.
3. **Crediting the gating event whose content is held works.** As in PBWM, the reward now
   goes only to the load that put the current item in the slot, and moves it toward the
   reward by its own error (a per-word bandit estimate). Keep needs no credit. All three
   seeds reach 100%, with 1,470–1,510 / 3,000 training rewards.
4. **The gate didn't learn "load names": it learned "load the fact sentence's own
   words".**

   | Word | Load | Keep |
   |---|---|---|
   | to | 0.98–1.00 | 0.49–0.52 |
   | went | 0.50–0.85 | 0.51–0.52 |
   | . | 0.00–0.03 | 0.51–0.52 |
   | cat | 0.05 | 0.51–0.53 |
   | where / person / ? | 0.16–0.28 | 0.50–0.54 |
   | names | 0.48–0.53 | 0.50–0.52 |

   - **What it loads at test:** "to", "over", "big", "old", and on two seeds "went" and
     some names. These are the words that occur only in the fact sentence.
   - **Why that's enough:** holding any of them recalls the most recent episode containing
     it, which is the fact sentence.
   - **Why names stay near 0.5:** a loaded name is usually overwritten by "to" before the
     question, so its load is rarely the one credited.
   - **The upshot:** the hand-set rule encoded our idea of the cue (the entity). Reward
     found a different cue that works just as well: whatever marks the fact sentence.
5. **The gate is still keyed on the word alone.** It works here because "to" never occurs
   in a distractor. If distractors also contained the fact sentence's words, the gate
   would need context, such as the previous word or the slot's content, as PBWM gates on
   input plus working-memory state. Not yet tested, and neither are more slots.

Later ([19](19-l5-shared-reward.md)): the held load is credited by advantage (reward − the
running-average reward). It still scores 100% on every seed with this task's reward, and
it also works with the cortex's L5 outcome as the reward.

## Biology
- **PBWM:** prefrontal stripes hold items, and the striatum's go / no-go pathways gate
  updates. Dopamine at the outcome trains the gating of the stripe whose content
  mattered. Credit goes to the gating action that produced the held content, not to every
  recent action ([O'Reilly & Frank 2006](../related-work.md#credit-assignment)).
- **Retrieval:** prefrontal-directed retrieval through nucleus reuniens to the hippocampus
  is the "cue from working memory" path used here.
