# 70 · The efference copy and recitation

**Questions.**
1. When the network hears its own words, can a copy of what it said mark them as its
   own: predicted, not surprising, and a mismatch a real error?
2. Can it retell a story it has just read, each spoken word driving the next? The
   second step of the [output plan](../concepts/output-and-self-supervision.md), after
   answering by speaking ([69](69-answering-by-speaking.md)).

**Code.** [`examples/episodic.rs`](../../examples/episodic.rs),
[`src/program/engram.rs`](../../src/program/engram.rs),
[`src/program/hippocampal_circuit.rs`](../../src/program/hippocampal_circuit.rs).
- `EFFERENCE=1`: a word the network spoke is marked as its own when it comes back. If what
  is heard is what was said, it is fully predicted: no surprise, and it does not enter the
  higher area's window of surprising words. If it differs, it is a full surprise.
  `SELF_NOISE=p`: what is heard is another word with probability p (altered feedback).
- `RECITE=k`: after each test story, a copy is read with its first k words given (the
  cue); from there each word the network says is its next input. Scored by words right
  in their position, and by the story's content words (names, places, seasons) said
  anywhere.
- `RECITE_PLAN=1`: the hippocampus plans, the cortex speaks.
  - The engram store now keeps each episode's rows in order (`recent_episode`); at the
    first spoken step it recalls the episode just read and plays it forward from the event
    after the one the cue matches.
  - At each step the planned word is spoken if the column's expectation admits it (its
    kind fits here), else the cortex's own word.
  - Test stories are stored in the hippocampus (with this flag only), so there is an
    episode to retell.
- `EpisodicCircuit::recall_peek`: recall without strengthening what is recalled, for cues
  the network made itself.
- `RECITE_DRY=1` (control): speak, but hear the story's word.

## Results (family stated once; seeds 0 / 1 / 2; cooperation + replay + relation store; hippocampus intact)

| Retelling | Efference copy | Words right in place | Content words said | Test answers (held out) |
|---|---|---|---|---|
| none | – | – | – | 86 / 93 / 91% |
| cortex alone | off | 15.7 / 14.1 / 13.8% | 2.0 / 5.3 / 1.3% | 86 / 93 / 91% |
| cortex alone | on | 15.4 / 14.1 / 13.9% | 2.2 / 3.1 / 1.0% | 86 / 93 / 91% |
| **hippocampus plans, cortex speaks** | on | **96.1 / 97.5 / 93.0%** | **92.5 / 94.4 / 86.4%** | 76 / 73 / 74% |
| same, 10% altered feedback | on | 73.6 / 73.9 / 73.0% | 70.4 / 71.3 / 69.6% | 76 / 75 / 74% |

Efference: with the copy off, 4,065–5,961 of the network's own (unaltered) words were
surprising; with it on, none. Every altered word was caught (2,043 / 2,187 / 2,160).

Retellings (seed 0):
- cortex alone: "summer came . the cat slept . the dog ran away . the dog ran away . the
  dog ran away …";
- planned: "autumn came . next the cat slept . tom went to the hallway ." (exact).

## Findings
1. **The cortex alone cannot retell.** It falls into the most frequent sentence ("the dog
   ran away") and almost never says a name, place or season (1–5%). It has no
   representation of the story's order of events.
2. **The hippocampus supplies the order, the cortex the words.** Played forward from the
   episode just read, the plan is spoken at 93–98% of steps (the cortex's expectation
   admits the planned word), and retellings are near exact. This is sequence replay
   driving cortical production, as in recall of a narrative.
3. **The efference copy works as designed.** Own words stop being surprising (from 4,000–6,000
   per run to none), and altered feedback is always caught.
4. **Altered feedback derails the retelling** (96 → 74%): one wrong word heard pulls the
   cortex's expectation off the plan for the following steps, about 2.5 steps per
   altered word.
5. **Storing test stories interferes with later answers** (89.8 → 74–75%). The test
   stories' events ("lucy went to the bathroom", in another season) are recalled for later
   questions. More experience of the same kind, without a way to separate it, is
   interference; the brain separates episodes by context, which this store does only
   weakly (a place code per story).
6. **A bug, found and fixed on the way.** The first retellings dropped later test answers
   from 86 to 53%.
   - Cause: the role cells chain each word's slot on the previous word's slot, and that
     carry was never reset between stories. A real story ends with "."; a retelling that
     stopped mid-sentence shifted every slot of the next story, and its hippocampal cues
     missed.
   - Found by comparing fingerprints of the network's state at the first damaged answer
     with a control that hears the page (`RECITE_DRY`): only the hippocampal recall
     differed, because its cue did.
   - A retelling now restores the reading context (the areas' windows), the step carries
     (previous word, its slot, the column's expectation) and clears its unfinished
     sentence. The fast inhibitory loop keeps running and learns from what is heard next.
     The source mix does not record during a retelling (it would score itself against a
     page the network does not hear), and recall cued by its own words does not strengthen
     memories.

## Next
- **Brain-like routing of speech** ([71](71-speech-routing.md)): a motor area, a
  basal-ganglia go/no-go to speak, and the efference copy as a learned forward model.
- **Separate stored test episodes by context,** so experience to retell does not
  interfere with later recall.
