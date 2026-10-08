# 88 · Inner speech in place of the rollout

**Question.** The rollout ([44](44-closed-loop.md) and its predecessors) brings recalled facts
back at test by inserting words into the page ("lucy [is a jones] went to …"). The trigger
(a definite expectation the page contradicts), the ranking of sources (slot memory, semantic
store, higher area, column) and the insertion into the word slot are all written by hand.
Can it be replaced by the network speaking to itself? It would write its output, in the same
code as its input, to a buffer, then hear it as the next input, marked as its own. That is
the same act as answering aloud ([69](69-answering-by-speaking.md), [71](71-speech-routing.md)) turned
inward, like a phonological loop with corollary discharge.

**Code.** `PhonologicalLoop` in [`src/program/speech.rs`](../../src/program/speech.rs) (say
a vector, hear it once, check what was heard against the copy).
[`examples/episodic.rs`](../../examples/episodic.rs) options:
- `COMPLETE=speech` / `speech-test`: inner speech replaces the rollout. What is said is the
  integrated prediction (the mix's word), heard at the next step with the efference copy.
- `INNER_WHEN=definite`: speak only where the column's expectation holds one word, or while
  already speaking (the rollout's trigger). Without it, the network speaks at every surprise.
- `INNER_SAY=recall`: recall plans the utterance. The slot memory's word is said if it fits
  the expected kind; failing that, the semantic store's; failing that, the prediction. Since
  the last fix, what is said must also fit the column's expectation.
- `INNER_GATE=learned`: the basal ganglia choose speak or read on, rewarded at the answer.
- `INNER_SLOT=1`: speech is heard in its own slot of the column's row (beside the page word,
  as auditory input beside visual), not in the word slot. `INNER_SLOT=echo` adds
  subvocalisation: the page word is heard there too as it is read.
- `INNERDIAG` prints the steps after a new name.

## Results

Family consolidated, seeds 0 and 1 (rollout: 57.0 / 52.8, mean 54.9). Each variant was
screened after the bug fix noted below; I, G, B and K were rerun.

| Variant | Held-out | Spoken at test | What happened |
|---|---|---|---|
| rollout | 54.9 | about 1,500 | speaks almost only at held-out names, always supplies a surname |
| B: at every surprise, the mix's word | 44.2 | about 12,000 | the useful detail is buried in thousands of other utterances |
| K: at every surprise, recall planned | 40.7 | about 13,700 | same |
| C: as B, also while training | 28.6 | (38,000 in training) | training on contexts full of own words hurts |
| G: rollout's trigger, the mix's word | 42.9 | about 1,600 | the right places, but a wrong-family surname a third of the time |
| I: rollout's trigger, recall planned | 48.7 | about 1,600 | wrong family nearly gone; some seeds still lose the surname |
| H, J: heard slot (plain, echo) | 20.9, 33.3 | | the column never learned to follow speech in that slot ("sam is is is") |
| E: learned go/no-go | 10.0 | 142 | it learned silence: training never shows a new name, where speaking pays |

Five seeds, all rollout entries, `COMPLETE=speech-test INNER_WHEN=definite INNER_SAY=recall`
(the closed-loop and superposed entries vary rollout-only options, so under inner speech
they run identically to learned stepping):

| Entry | Rollout (recorded) | Inner speech | Per seed |
|---|---|---|---|
| learned stepping (also closed loop, superposed) | 54.5 (55.4, 57.2) | **62.2** | 57.4/67.2/66.4/62.8/57.0 |
| semantic store | 63.1 | 63.1 | 58.6/68.4/67.2/63.6/57.6 |
| inference replay | 59.4 | 59.4 | 62.2/31.4/66.6/70.8/66.0 |
| engram store | 67.4 | 66.1 | 69.8/52.8/71.2/71.0/65.8 |
| engram walk | 66.8 | 64.1 | 69.2/49.6/68.2/68.4/65.0 |
| index hippocampus | 66.0 | 63.0 | 66.8/49.2/68.2/66.0/65.0 |
| family stated | 38.0 | 35.3 | 33.8/27.8/50.0/29.8/35.0 |
| full hippocampus | 38.0 | 35.5 | 37.4/21.2/52.6/26.6/39.6 |
| hippocampus teaches cortex | 60.9 | 56.6 | 67.8/18.4/68.2/63.0/65.4 |
| family consolidated | 52.0 | 41.6 | 55.0/40.4/41.6/16.0/55.0 |

**A bug found on the way.** The old one-word completion path (`COMPLETE=test`) also fired
under `COMPLETE=speech*`, in training too, and broke the inner-speech chains ("tom [is]
went"). The figures above are after the fix.

## Findings
1. **Inner speech does what the rollout did, without the rollout's source ranking or its
   special insertion.** With the same trigger and recall planning the utterance, it matches
   the rollout within 3 points on six of ten entries. It is 8 points better on learned
   stepping, where it also replaces the three rollout refinements (learned stepping, the
   closed loop, superposition), all of which it beats.
2. **When matters most.** Speaking at every surprise (about 12,000 utterances per test)
   buries the few that matter (about 1,500, at new names), whichever word is said. The
   rollout's value was its selectivity.
3. **What matters next.** The integrated prediction alone says a wrong-family surname a third
   of the time. Recall must plan the content: the hippocampus's or semantic store's word,
   said by the cortex if it fits. Where recall offers nothing, the mix's word ("smith" for
   every new name in one seed) does worse than the rollout's fallback, the column's own
   word. Most of the remaining loss comes from two seeds (hippocampus teaches cortex seed 1
   at 18.4, family consolidated seed 3 at 16.0).
4. **A separate heard stream needs experience of being used.** In its own slot, speech
   reaches a column that never saw that slot carry anything useful, so it is ignored or
   repeated. Echoing the page word there (subvocalisation) does not teach the column that
   the stream can add something.
5. **The learned go/no-go cannot learn this yet.** Its reward exists only at test (new names
   never occur in training), so in training it learns silence. A plausible trigger needs a
   case it can practise: the next step's question act, where marking a gap pays off within
   the same story.

Nothing becomes a default. The rollout stays in the suite.

## Next
- **Questions as inner speech.** When the column is unsure of a new item and recall offers
  nothing, a tag ("?" + the item) goes into the loop. It holds the gap open, and the answer
  binds to it when it arrives later in the story. Its payoff arrives within the story, so a
  go/no-go can learn when to ask.
- The fallback word: the column's own word, gated by its expectation, against the mix's.
- The heard slot with training that uses it (facts said aloud that the page later relies
  on).
