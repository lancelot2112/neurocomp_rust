# 50 · Hippocampal replay for rule extraction: neither prototypes nor events teach the rule yet

**Question.** CA3 settling is pattern completion. With a weak cue, or none, it pulls
toward the deepest attractor, and with overlapping memories that is a prototype, the
gist of many events. Experiment [49](49-sparse-binding-space.md) used tagged replay for
facts, and that worked. Here the question is whether the hippocampus's free-settling
replay can do the other job of sleep: supply the regularities, so the higher area's sleep
generalisation ([33](33-generalisation-during-sleep.md)) extracts the family rule from
it. That would replace the buffer of waking experience it reads now.

**Code.** In [`src/kernel/class.rs`](../../src/kernel/class.rs):
`KernelClass::set_record_waking`. In [`examples/episodic.rs`](../../examples/episodic.rs):
`REPLAY_GEN` (`=1` free settling, `=cued` cued by a random familiar binding),
`GEN_REPLAYS`, and the `GENDIAG` diagnostic.

## Design
- **Events keep their context in the readout,** so a replay brings back the story so
  far (e.g. the season) with the event.
- **At each sleep, 512 replays,** decoded by slot into content words and context words.
  Every content word becomes a target predicted from [the event's other words | its
  context]. These pairs go to the higher area's sleep generalisation.
- **The waking buffer is off** (`set_record_waking(false)`), so rules can only come from
  replay.
- **The test** is the schema task of [39](39-schema-advantage.md) (`FAMILY=1 SCHEMA_K=0`):
  new family members, answerable only through the family rule. It runs on the
  self-contained hippocampus of 49 with 16,384 CA3 cells.

## Results (seeds 0 / 1 / 2; held out = new family members, never seen)

| Sleep generalisation reads | Held-out right | Trained names | Rules formed |
|---|---|---|---|
| The area's own waking buffer (as in 33–49) | **56 / 59 / 52%** | 65% | 566–632 |
| Hippocampal free-settling replay | 29 / 23 / 30% | 61% | 231–368 |
| Hippocampal cued replay (512 per sleep) | 29 / 24 / 30% | 56% | 144–163 |
| Hippocampal cued replay (2,048 per sleep) | 29 / 19 / 30% | 55% | 135–200 |
| No sleep generalisation | 30 / 17 / 32% | 56% | – |
| No-schema control, cued replay | 20 / 10 / 12% | 16% | 132–184 |

What a free replay brings back: often a fairly specific event, e.g. content {john, smith,
went, to, the, hallway, .} with context {autumn, the cat slept, it rained, …}. Sometimes it
is a blend: content {then, it, rained, .} with three seasons in the context.

## Findings
1. **Replay-derived rules do not help yet.** Free or cued, the higher area forms 135–370
   rules from them, but new family members stay at the no-generalisation level
   (19–32%). Rules from the waking buffer reach 52–59%.
2. **Free settling is only partly a prototype.** In the sparse space with novelty-weighted
   codes, a free replay often lands on a specific, common event. With a weak anchor,
   settling favours what was repeated, which is a kind of generification. Here, though,
   the averaging is mild.
3. **The real mismatch is format and order.** The waking buffer holds the area's own
   inputs as it read them:
   - the sentence so far (only the words *before* the target);
   - a state of the recent *surprising* words.

   A replayed event is an unordered set of bindings, so the replayed pairs differ:
   - each word is predicted from all the others, including words that came after it;
   - the context holds every earlier word, not just the surprising ones.

   The rules formed fit the replay, not what the area sees while reading. Almost all
   candidates (4,000–6,600 per run) fail the replay test, and those kept rarely match
   reading.
4. **What this points to: the hippocampus must store and replay sequences.** Hippocampal
   replay in the brain is time-compressed *sequence* replay (theta sequences, sharp-wave
   ripple sequences). Played back through the cortex in order, it gives the cortex its
   own inputs back, in the format its predictions were learned in.
   - Here the hippocampus stores events as sets. Order is only implicit in the slots.
   - Teaching a sequence predictor from it needs a CA3 that also links each event (or
     word) to the next, heteroassociatively, so replay can run the sequence forward
     through the column and higher area.

## Next
- **Sequence memory in CA3:** heteroassociative links from each event's code to the next
  event's (and within an event, word to word). Replay then runs forward through the
  cortex and generates the area's inputs as reading would.
- **Then retire the waking buffer.** The cortex would keep no record of its own
  experience; the hippocampus would replay it.

## Biology
- **Sequence replay:** place-cell sequences replay in order (and in reverse) during
  sharp-wave ripples, time-compressed (Wilson & McNaughton 1994; Lee & Wilson 2002;
  Foster & Wilson 2006). Theta sequences compress paths during behaviour (Dragoi &
  Buzsáki 2006).
- **Gist and false memory from overlapping traces:** schema-consistent intrusions
  (Roediger & McDermott 1995). Overlapping replay as a route to schema extraction (Lewis
  & Durrant 2011).
- **CA3 heteroassociation:** recurrent collaterals can store sequences as well as
  patterns (Levy 1996; Lisman 1999).
