# Memo 4, round 2: theory response

## (a) The fixed body: I change my mind, with a condition

Memo 3 is right, and I overstated the case. My objection holds only when *content* is fixed. A fixed keyboard is anatomy. A fixed vocabulary is a lookup table written into the genome. So the version I accept has four parts:

- a fixed body;
- content drawn fresh each lifetime (which words exist, what they mean, which act pays);
- held-out content families that no training world contained;
- a learning-slope case in lexicase: the same genome scored early against late in a lifetime, or at 1x against 4x length.

Under those four, hard-wiring the body is harmless and saves evolution the job of solving a problem the world made. One caution: memo 2's red flag stands. Switching and shapes score exactly 0 on feel-100 in run 30, so a blind control has to come before any claim from that run.

## (b) Output learning: one mechanism at three levels

The proposals are not rivals. Each answers a different question:

- **Eligibility** says which kernel acted. That is my per-kernel fired bit at t-k.
- **A modulator** says when, and with what sign. That is a channel read by a learn gate.
- **A mask move** says what changes. That is memo 3's local output rule, or memo 1 and 2's writable masks.

A three-factor rule is AND(eligible, modulator) applied to a mask move, and that move can sit on the output side. Writable masks as channels are the general form: they let evolution write the rule itself as a circuit. They are also the largest search space and the costliest runtime change.

The smallest change that could work, in this order:

1. **A fixed local output rule in the runtime**, shown to work by hand. A kernel's output line drifts toward destination lines that confirm a step later and away from ones that do not, so a wrong pairing can die. Lesson 46 applies: if no hand-built rule learns paired associates, evolution will not find one either.
2. **Gate that rule with eligibility and a modulator channel.** This is one bit of history per kernel and one gate gene, so the evolved affect or target-residue channels finally have something to modulate.
3. **Only then expose masks as writable vectors**, so evolution can replace the seeded rule.

Skipping to step 3 asks evolution to invent learning from nothing. Twenty runs say it won't.

## (c) Noise floor versus substrate

Both are needed, and they answer different questions. The noise floor (4 seeds of run 26) decides whether the switches in runs 24 to 30 are real. Do it, on spare cores. But the stuck rungs have an icc of 0.00, and no number of replicate seeds turns zero signal into a gradient. So the noise floor should not hold back the substrate work. It is cheap insurance, not the critical path.

## (d) The first gate

I now put **memo 1's paired-associate capacity curve** first, ahead of my bits-per-byte test. It isolates the output-learning gap directly, needs only short lifetimes, and has a clean hand-written ceiling. Bits-per-byte mixes that gap with the separate problem of a sparse front end for dense ASCII (experiment 03). It stays the decisive gate *for language*, but it comes second. The learning slope is not a gate; it is the selection case that every later island carries.

## Compute, reconsidered

Each generation already runs about 768 lifetimes an island, about 4 x 10^6 lifetimes a run. So my 10^6 to 10^7 evaluations is roughly today's count, not 1,000 times it. The gap I estimated is lifetime length (about 10^5 times) multiplied by network size a step (10^3 to 10^6 times). Memo 2's call for 10 to 100 times the evaluations helps selection beat noise, but it does not close that gap.

The only route within reach is meta-learning. Evolve the learner on short, cheap, varied task draws, then run a few long lifetimes. That works only if what is learned in 300 steps generalizes to 10^6 steps, which the learning-slope case is there to test.

## Revised top recommendation

Put output learning into the substrate as a local three-factor rule (eligible kernel, modulator channel, output-mask move). Seed it by hand, then let evolution replace it once masks are writable. Select on learning within a lifetime: fixed body, content drawn each lifetime, a slope case.

**First experiment:** the paired-associate capacity curve, with N = 2, 8, 32 and 128 glyph-to-act pairs, codes drawn each lifetime under a fixed body. Compare three arms, each with a blind control and 5 seeds:

- today's runtime;
- plus the hand output rule;
- plus that rule gated by eligibility and a modulator.

Pass: held-out recall at N >= 32 on the hand arms. If none passes, redesign the kernel before any further island run.
