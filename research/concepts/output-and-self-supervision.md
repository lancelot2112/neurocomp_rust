# Output and self-supervision: speaking, and hearing yourself speak

So far the network only listens. Its only "output" is the column's L5 prediction of the
next word, scored against what actually comes. This page plans how it should produce
output of its own, and how hearing its own output can teach its higher areas, as hearing
our own voice calibrates our speech.

## Why output matters for learning
Experiment [25](../experiments/25-area-chain.md) showed the main limit of a deep chain of
areas: the upper areas learn only from the column's errors. That signal is rare, so they
learn slowly. A network that can say things back gets a teaching signal on every sentence
it reproduces, for every area that took part in producing it:
- **Reproduction tests the state.** If an area's state really encodes "winter came", the
  network can say "winter" when asked what season it is. If it cannot, that area's state
  is missing the fact, and the mismatch says so.
- **Hearing yourself closes the loop.** Our speech is calibrated by the ears. Altered
  auditory feedback makes speakers correct their pitch and vowels within a few hundred
  milliseconds (Houde & Jordan 1998), and the motor plan is checked against the
  predicted sound of its result (the DIVA model: Guenther 2006; Hickok & Poeppel 2007).

## What output looks like
**1. A speech area.**
- **What it says:** the same word codes the network reads, so what it says can come
  straight back in as input.
- **When it speaks:** a basal-ganglia go/no-go per step chooses "speak" or "listen", as
  the selectors already do for recall and relays.
- **What it says:** the word with the most evidence in the mix (`SourceMix`): the
  column's prediction, memory, and every area's top-down prediction. Speaking is
  committing to the current best interpretation.

**2. An efference copy.**
- **A self frame.** When the network speaks word w, a copy of the command tells L4 "this
  input is mine". This is one more frame (`self`), set while the network's own words come
  back in.
- **Less surprise.** The column predicts w, and the matching input is not surprising.
  This is sensory attenuation: we cannot tickle ourselves (Blakemore, Wolpert & Frith
  1998).
- **A mismatch is a real error.** If what comes back differs from what was meant (noise,
  or a deliberately altered channel), that difference is a strong surprise, and it is
  credited to the areas that produced the plan.

**3. Three ways to use it.**

| Mode | What happens | What it teaches |
|---|---|---|
| **Answering** | At a question, the network speaks its answer, or "unknown" below a confidence threshold | Questions no longer need the answer in the stream, and abstention becomes an output |
| **Recitation** | After a story, it is cued to retell it; each spoken word comes back as input and drives the next | Can the areas and memory drive the column with no outside input? (generation) |
| **Read-back** (self-supervised) | During training, after each sentence (or window), the areas regenerate it, and the heard result is compared with what was read | Every area gets an error on every sentence: what its state failed to carry |

## The read-back loop in detail
1. **Read.** The network reads a sentence as now. Each area's window collects its
   surprising words.
2. **Say it back.**
   - The external input is muted.
   - The column runs free: its input is its own last spoken word, plus memory and the
     areas' top-down frames, as usual.
   - The network speaks one word per step until it says "." or reaches a length limit.
3. **Hear it.** Each spoken word comes back through L4 with the `self` frame set.
4. **Compare.** At each step the spoken word is compared with the word read at that
   position.
   - **A match** confirms the winners that produced it: the column's kernel, and each
     area's prediction that voted for it.
   - **A mismatch** is a surprise, and the areas whose vote was wrong learn from it, as
     they now learn from the column's errors when reading. The read word is the target.
5. **Repeat at the slower scales.** Areas with longer windows read back at their own
   window boundary: area 3 every 16 sentences, area 4 every 64. They regenerate the facts
   their state holds (the surprising words), not the whole text. So the season word is
   asked for again many times while it is still in the window, which is the teaching
   signal experiment 25 lacked.

## Brain parallels
- **Efference copy / corollary discharge:** a copy of the motor command predicts its
  sensory result (Sperry 1950; von Holst & Mittelstaedt 1950).
- **Auditory feedback in speech:** the dorsal stream maps speech plans to their predicted
  sounds and corrects errors (Hickok & Poeppel 2007; Guenther 2006; Houde & Jordan
  1998).
- **Rehearsal:** the phonological loop keeps items alive by saying them inwardly
  (Baddeley 1986).
- **Generative training:** a recognition model and a generative model train each other
  (the wake–sleep algorithm: Hinton et al. 1995). Read-back is the waking, word-level
  version, and the network already has sleep for consolidation
  ([23](../experiments/23-compaction.md)).

## Status
Read-back is built in its rehearsal form ([26](../experiments/26-context-and-readback.md)).
- **Skill:** areas learn, self-supervised, to say back the most recent rare word they
  hold (80–97% at test), and what they say is heard again.
- **Effect:** it extends a single area's reach (3–11% → 27–33%), but it does not help a
  chain that already holds the fact once stories are separated by a context boundary.
- **Answering by speaking is built** ([69](../experiments/69-answering-by-speaking.md)):
  an `OutputBuffer` the cortex writes to, abstention from confidence (about 60% answered
  at 84–91% right with the hippocampus lesioned), and the spoken word heard in place of
  the page's.
- **The efference copy and recitation are built** ([70](../experiments/70-efference-copy-and-recitation.md)):
  own words are not surprising and a mismatch is caught; the hippocampus plans a
  retelling and the cortex speaks it (93–98% of words in place).
- **Still to build:** speech routed as in the brain (a motor area, a basal-ganglia
  go/no-go, a learned forward model), and read-back as training.

## The experiments, in order
1. **26 · Answering by speaking, with abstention.** The question stories end without the
   answer. The network speaks a word or "unknown". Score accuracy, coverage and
   calibration. Small: it reuses the mix and the confidence buckets.
2. **27 · Recitation.** Read a short story, then retell it from the cue. Score the share
   of the story reproduced in order, with and without memory and areas. This shows
   whether the top-down path can generate, not only bias.
3. **28 · Read-back as self-supervision for the area chain.** Rerun the season task of
   25, with each area learning also from its read-back error. Success: accuracy at
   distances 4–31 well above 25's 9–23%, with the same training stories.

Each step reuses what exists: word codes, the column's free-running prediction, the
basal-ganglia go/no-go, `SourceMix`, and the residual learning of the areas. The new
parts are the `self` frame and the read-back schedule.
