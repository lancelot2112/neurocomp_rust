# Panel memo 2, round 2: evolution and the curriculum

## (a) The fixed body, and a correction

**First, a correction to my round-1 memo.** I said that switching and shapes score 0 on feel-100 in run 30 because of the fixed body. That is wrong. In run 29, which had no fixed body, both islands' goal lines read 0/100 (at most 8) at every 40-generation sample from 79 to 519. The 0 is the old feel stall. The body did not cause it, and I withdraw the red flag. The fixed body may even have helped feel: run 30's maze passed feel at 49, the first time it has passed in any run.

**Where I stand: memo 3 is right about anatomy, and memo 4 is right about the pressure.** They do not contradict each other.

A body redrawn every lifetime made evolution solve a problem no computer user faces, and it hid the real test. With the body fixed, the stalls that remain are clean ones: echo2, cued and win-stay now fail only because content changes from lifetime to lifetime. Keep the body fixed.

But a fixed body alone does pay off hard-wiring. Run 30's 17 fast moves are mostly evolution building in what used to be learned. So the fixed body has to come with a selection pressure that only learning can satisfy. That is (d).

## (b) How outputs come to be learned

I am changing my mind on the order. Writable masks, with the learning rule expressed as a circuit, are the most general change. They are also the largest search space, and evolution has to find a working rule in it before any gradient appears. At 32 genomes per island that is the echo2 problem again, one level up.

The smallest change that could work is memo 4's eligibility bit, combined with memo 3's local output rule. Concretely:
- **Eligibility:** a kernel's fired bit is kept in a short ring, and a learn move applies to the kernels that fired k steps earlier (k is a gene) when a modulator channel, also chosen by gene, is on.
- **Output rule:** the same gated move is also applied to the output mask, moving one line toward lines that turned out on in the destination.
- **Death:** a kernel whose confirmation count falls to zero is removed. This is births plus death.

These are all one-word ops, local, and gene-parameterised. Build them by hand first and check them by hand (lesson 46). Writable masks with a rule table come second, seeded with this rule, so that evolution varies a rule that already works instead of searching for one from nothing.

## (c) The noise floor or the substrate first

The substrate comes first, and I concede the point.

Replicating runs 26 to 30 costs about 12 to 15 hours of machine time and only tells us how reliable switches are on a substrate we all now agree is missing its learner. That money is better spent elsewhere.

What I keep is narrower. From now on, every comparison of a new mechanism gets 3 evolutionary seeds. The first of these, the output-learning check, uses short dedicated islands (the reading and selection islands, 200 generations each), which costs little. The old switch runs stay as single-seed hints and should be labelled as such in island-runs.md.

## (d) The yardstick

**The first gate is memo 1's paired-associate capacity curve:** N arbitrary pairs drawn each lifetime, with N = 2, 8, 32 and 128, scored on held-out recall. I choose it for three reasons:
- It isolates exactly the missing capacity, output learning.
- It runs in short lifetimes, so it is cheap enough to select on.
- It cannot be hard-wired under a fixed body, because the pairs change every lifetime.

Bits-per-byte scaling is the decisive later test, but it is too slow per lifetime to be a selection case at today's budget. The learning slope is better as a diagnostic.

**As a selection case,** each lifetime draws N from a schedule. The case score is recall at N, and lexicase takes one case per pooled pair of worlds; pooling addresses the single-world noise problem found by the signal measurement. Two further cases come from the same lifetime:
- a slope case: recall in the lifetime's second half minus its first;
- a ratchet: the largest N the elite held on held-out worlds becomes the next floor. This is learning progress, with no hand-written rung.

Thrift becomes a single case, not one per world.

## Revised top recommendation

Put credit-gated output learning into the primitives (an eligibility bit, a modulator channel, an output move and death). Then select for capacity on arbitrary associations drawn every lifetime, under a fixed body. Do not run more ladder-switch runs until that curve rises past N = 8.

**First experiment.** Build the paired-associate world. Score it with these genomes:
- today's best run-30 genomes;
- a blind control;
- a hand-built learner using the new rule.

Then run one 200-generation island on it, 3 seeds, with the rule's k and modulator as genes. The bar is that evolved recall at N = 8 beats the blind control and today's genomes on 48 held-out worlds. If the hand-built learner cannot pass N = 8, the kernel itself needs redesign before any further evolution.
