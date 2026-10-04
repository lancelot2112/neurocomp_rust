# 11 · Episodic autoassociative memory

**Question.** Can one-shot storage of episodes plus content-addressed recall do the
binding that thalamic routes do in [09](09-thalamic-attention.md)–[10](10-route-pool-inhibition.md),
without routes, without learning which positions matter, and on sentences of any shape?

**Code.** [`src/program/memory.rs`](../../src/program/memory.rs) (`EpisodicMemory`);
experiment [`examples/episodic.rs`](../../examples/episodic.rs) (commit `264dcb5`).
Run: `cargo run --release --example episodic` (~35 min; `TASK=short|long|varied`, `SEEDS=n`).

## The mechanism
- **Store, one shot.** Each sentence (up to ".") is stored once as the union of its word
  codes. There is no gradual learning in the memory.
- **Cue = the rarest content of the current sentence** (`rarest`: bits stored at most
  1.5× as often as the sentence's 10th-percentile bit). In "where is peter ?" that is
  *peter*, whatever the absolute frequencies are.
- **Recall:** the stored episode overlapping the cue most (≥ 70% of the cue's bits),
  the most recent one on ties. This is a hard-max version of attention over stored
  episodes ([Ramsauer et al. 2020](../related-work.md#binding-and-attention)).
- **Habituation:** per-bit counts of how many episodes contained each bit. The recalled
  episode keeps only bits in ≤ 40% of episodes (`novel`), minus the cue. Function words
  drop out and the place remains.
- The predictor reads `[current word | recalled novel content | previous word]`. The
  memory keeps storing at test time (storing is reading); the predictor's learning is off.

## Two things it needed
1. **Sparse codes.** Habituation is per bit. At 512 bits / 32 active, a word shares
   most of its bits with other words, so *peter*'s bits counted as frequent and the cue
   came out empty. At 2,048 bits a word still shares ~46% of its bits
   (1 − (1 − 32/2048)^39). At **8,192 bits** it is ~14%, and it works
   ([sparse codes and collisions](../concepts/sparse-codes-and-collisions.md)).
2. **A relative, not absolute, cue rule.** With a fixed 25% cutoff, names (27–28% of
   episodes) were dropped along with "where" and "is" (42–46%). Taking the rarest content
   relative to the sentence (10th-percentile bit × 1.5) keeps the name however the mix of
   sentences changes.

## Setup
Stories as in 09–10 (random filler between stories; 1–N facts about distinct people; a
question about a random one; one place per name held out of training; 3,000 training,
1,000 test stories; 5 seeds; chance 1/6), in three forms:
- **short:** "X went to the P ." / "where is X ?"
- **long:** "X went [all the way over] to the P ." / "where is X right now ?"
- **varied:** "X [quickly|slowly] went [all the way over] to the [big|old] P ." with every
  bracket optional, so the place sits at a variable distance from the name.

Baselines: no memory, and the **hand-set** thalamic relay routes (1,+4), (3,+4), (3,+8),
i.e. the best fixed offsets from 09–10.

## Results (held-out pairs)

| Stories | Facts | No memory | Fixed relay routes (hand-set) | **Episodic memory** |
|---|---|---|---|---|
| short | 1–2 | 16.7% | 100% | **95.0%** (98/98/100/79/100) |
| short | 1–3 | 16.4% | 100% | **99.2%** (98–100) |
| long | 1–2 | 17.3% | 97.7% | **96.8%** (90–100) |
| long | 1–3 | 17.1% | 99.8% | **98.2%** (92–100) |
| varied | 1–2 | 16.8% | 33.1% | **98.1%** (96–100) |
| varied | 1–3 | 16.6% | 34.4% | **98.5%** (97–100) |

Seen pairs score about the same as held-out everywhere (95–100%), so nothing is being
memorized pairwise.

## Findings
1. **One-shot episodic memory solves binding without routes.** It reaches 95–99% on
   held-out pairs in every condition, close to hand-set routes where those work, and
   learned nothing about positions to get there.
2. **It is position-free, which routes are not.** On the varied stories, where adverbs and
   adjectives shift the place by a variable number of words, the best fixed routes drop to
   33–34%. The memory stays at 98%. This is the first mechanism here that is robust to
   sentence shape.
3. **It only works with sparse codes and relative rarity.** Per-bit statistics need
   per-word bits (8,192 / 32), and the cue must be "the rarest thing in this sentence",
   not a fixed cutoff. Both connect to earlier findings: [sparse
   codes](../concepts/sparse-codes-and-collisions.md) for 05/06, and attention to the
   most informative input.
4. **Still hand-set:** sentence boundaries ("."), and the habituation and rarity constants.
   Episodes are bags of words, so sentences where word order carries meaning ("mary saw
   john in the kitchen") would be ambiguous. See
   [hippocampal functions](../concepts/hippocampal-functions.md) for what is missing (pattern
   separation, recurrent completion, loops, structure codes) and
   [12](12-dentate-gyrus-ca3.md) for the next step.
