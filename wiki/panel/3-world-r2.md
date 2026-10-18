# Panel memo 3, round 2: the world and the body, revised

## (a) The fixed body: partly revised

I still hold that the body should be fixed. A keyboard, a font and a voice's sounds are anatomy, and redrawing them every lifetime set a problem no real agent faces. Much of experiment 16 went into solving it.

Memo 4 is right about the consequence, though. I said too lightly that "if reading becomes innate, that is fine". It is fine for an agent, but not for a search that should be selecting learners. Run 30's switching and shapes, at exactly 0 on feel-100, are the warning sign.

A fixed body keeps learning pressure only if the world draws fresh, each lifetime, every mapping the agent must use:
- which glyph names which sound or meaning;
- which act pays;
- the teacher's facts, the vocabulary, the screen layout and the task instruction.

I propose an observer rule: a rung counts only on worlds with permuted mappings, where a genome that memorised one mapping scores about 0. That is a blind control for innateness, beside lesson 36's control for blindness. Echo may stay innate, since it is body. Reading, binding and contingencies must not.

## (b) How outputs come to be learned

All three proposals supply the same missing piece: credit for what a kernel wrote, in a move that can undo a wrong pairing. Build them smallest first, each only after the one before passes a hand check:
1. **A fixed three-factor output rule, by hand.** Each kernel keeps an eligibility bit (it fired k steps ago, in its own ring). While a gene-named modulator channel is on, an eligible kernel moves one output line toward a confirmed line in its destination. A kernel that keeps failing dies, freeing its lines for a new birth. This is memo 4's eligibility bit with my births plus death, all u8 and word ops.
2. **The rule as a gene.** Memo 1's small table, seeded with rule 1, so evolution varies a working rule instead of searching from nothing.
3. **Masks as writable channels** (memos 1 and 2). The most expressive and the largest change, worth it only once step 2 shows evolution improving on its seed.

## (c) Noise floor or substrate first

They do not compete. The substrate check is built by hand and needs no evolution; the noise floor needs the workstation once run 30 ends. Run both, in parallel. Start no further one-option switch run until both are in: without the noise floor, comparisons between runs are not evidence, and without the substrate check no run can pass the rungs that matter.

## (d) The first gate

The paired-associate capacity curve (memo 1) comes first. It is the cheapest, it isolates output learning, and its mappings can be drawn per lifetime. My learning slope is the same test turned into selection pressure for the island after it. Memo 4's bits-per-byte scaling is the decisive later gate, and it means little until single associations can be learned and corrected.

I change my mind on order: the byte terminal comes after the gate. It tests the interface, which is not the bottleneck. What I keep from it: the paired-associate world shows its facts as raw bytes, so the yardstick and the later terminal share one interface.

## Revised top recommendation

Make lifetime learning of arbitrary maps something the substrate can demonstrably do, then the thing evolution selects:
- prove a three-factor output rule with births plus death, by hand;
- open the rule to the genome;
- evolve on permuted-mapping worlds, with the learning slope as a lexicase case;
- only then build the terminal and dialogue worlds, keeping the fixed body.

**First experiment: the paired-associate capacity curve.** Show N = 2, 8, 32 and 128 glyph-to-act pairs, drawn per lifetime, under the fixed body. Score held-out recall on 48 worlds with a blind control, for three learners: today's runtime, the hand-built three-factor rule, and a hand-written lookup as ceiling. If the hand rule cannot lift the curve past N of about 4, redesign the kernel before evolving anything more.
