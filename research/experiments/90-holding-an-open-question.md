# 90 · Holding an open question, self-taught: a negative result and what blocks it

**Question.** [89](89-questions-as-inner-speech.md)'s question act was hand-written: an edit
rule rewrote sentences, and its gain turned out to be an artefact (the
[audit](../concepts/hand-written-rules.md) that followed lists such rules; rule 6 forbids new
ones). This experiment rebuilds the act from the network's own signals, with no rewriting
and no oracle. Can holding a novel item in working memory, and storing it with what is read
next, bind a stranger's fact so that it shapes the answer?

**Task.** As in 89: "winter came . lucy came . … the person is a smith . … mary is a jones
. … lucy went to the ___". The stranger's family is random in every story. Training uses
fresh strangers (`QUESTION_POOL=300`) and test uses the new names. The engram store's
settings are the engram-store entry's, without `SCHEMA_K`. Events are stored at test too.

**Code** ([`examples/episodic.rs`](../../examples/episodic.rs)): `QHOLD`, `QATTACH`,
`HELD_FIELD`.
- **Novelty:** the hippocampus's own novelty for each word's binding as it is bound (1 − the
  recall match), not a count.
- **Holding:** a binding more novel than the one held takes working memory. The reference
  is `QHOLD=novel` (take it at novelty ≥ 3/4). `QHOLD=learned`: the basal ganglia choose
  hold or ignore per novelty band, rewarded at the answer.
- **Cueing:** the held item is active, so it joins every recall cue (in a field of its own).
- **Binding:** at a sentence's end the held item may be stored with that sentence's event.
  `QATTACH=all` attaches to every later sentence. `QATTACH=learned`: the basal ganglia
  choose per sentence, keyed by its first two words (a symbolic shortcut, listed in the
  audit). Credit is tagged by recall: the attached event recalled for the answer gets the
  outcome.
- **Use:** recall returns to the column as entorhinal context (`HC_EC`), so nothing is said
  or rewritten.
- `QHOLD=oracle` holds the stranger: a labelled reference, used only to measure recall.
- `SELFDIAG` prints the cue and the recalled event at the answer.

## Results

Seeds 0 and 1, held-out (89's baseline without `HC_EC`: 68.2):

| Arm | Held-out | Item held at the answer | Attached events recalled for the answer |
|---|---|---|---|
| `HC_EC`, no holding | 53.8 / 30.2 | | |
| hold if novel, attach to all | 54.8 / 44.2 | 4 / 3 of 500 (the stranger) | 0 |
| hold if novel, learned attach | 55.4 / 27.4 | 3 / 3 | 0 |
| learned hold, learned attach | 41.6 / 49.0 | 500 / 500, never the stranger | 2 / 2 |
| *with story context (place bonus 2, current story's rows always candidates):* | | | |
| no holding | 23.4 / 22.4 | | |
| oracle hold (reference), learned attach | 25.2 / 23.2 | 500 (the stranger) | 83 / 0 |
| learned hold, learned attach | 25.2 / 21.8 | 500, never the stranger | 176 / 292 |

## Findings
1. **Recall answers the sentence, not the question.** At "lucy went to the", the
   hippocampus recalls the event most like that sentence:
   - without story context, an earlier test story's "lucy went to the bedroom" (all four cue
     words match), with another season's and family's place;
   - with the place bonus, this story's "lucy came .".

   Neither holds the fact, so a fact bound to lucy is never the one recalled. This blocked 89
   too. It also explains the weak, seed-dependent `HC_EC` baseline: the column has been
   reading old answers.
2. **Story context alone makes it worse (23%).** The suite's answers lean on recalled past
   stories with the same question, and the bonus replaces those with the current story's
   mentions.
3. **Novelty is the wrong signal for "open".** The new names recur through the test, and test
   events are stored, so a name is familiar after its first story. What should be held is an
   item unresolved *in this story*, not one never seen.
4. **The learned hold holds the wrong thing.** The first item taken blocks later ones (a
   newcomer must be more novel), so it holds an item from the opening sentences and never the
   stranger.
5. **Attaching to everything binds nothing.** With the held item on every later sentence,
   several events tie on it. Recall then picks the one that also names the item ("lucy came").

Nothing becomes a default. The hand-set act of 89 and its operators stay as labelled
references only.

## What it needs
- **A query, not a sentence, as the cue.** At the question, the held item alone (with this
  story's context) should cue recall, as a "who is lucy?" query, separate from the
  sentence-matching recall that predicts the next word. This is pattern completion from
  the item, which the engram walk already does for a bridge word.
- **"Open" as unresolved in this story.** An item with no fact bound to it yet in the
  current story, measured by recalling with the item and this story's place, not by
  lifetime novelty.
- **A takeover rule that lets a newer open item in**, such as decay of what is held, learned
  like the rest.
- **Context keys as vectors, not word ids**, so the attach choice generalises (the audit's
  symbolic shortcut).
