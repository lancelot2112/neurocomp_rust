# Panel memo 1, round 2: architecture response

## (a) The fixed body

I side with memo 3, with memo 4's worry built in as a check. Fix the anatomy: voice codes, hand codes, and later a keyboard and a font. Experiment 16 spent most of its effort solving a scrambled-body problem that no real agent faces, and motorcheck shows how much a literal interface removes (one wire went from respond 23 to 515). Memo 4 is right that whatever stays fixed will be hard-wired, so the way to reconcile the two is to decide what is fixed. Glyph meanings, facts, task contingencies and vocabularies must keep being drawn fresh every lifetime, and selection must include cases that only a learner can pass. Run 30 still draws glyphs fresh, so reading still has to be learned.

Two conditions before I trust it:
- Memo 2's red flag needs a blind control. Switching and shapes score exactly 0 on feel-100 under the fixed body.
- A share of lifetimes should get a remapped body (keys permuted), so evolution cannot assume the body completely.

## (b) How outputs get learned

I am changing my order. Masks as writable channels with the rule as a circuit is the most general option and the biggest change. Evolving rules before any rule has worked by hand goes against lesson 46. The smallest change that could work is a stack of three:

1. **Death for wrong pairings (memo 3).** A kernel already keeps a u8 count of its confirmations (`confirm` in `src/kernel/simple.rs`). Let a kernel whose count falls to 0 after acting be removed, so its input goes uncovered and is reborn copying the move made then. The coincidence-confirmed reader failed exactly because "a fading kernel is replaced without the old one going" (319 kernels).
2. **An eligibility bit (memo 4).** A kernel's own fired bit, read from history at lag k, ANDed with a modulator channel the genome chooses. This gives credit per kernel instead of per population, which is what lesson 27's two-step window asks for. It is a few lines and keeps the no-globals rule.
3. **Masks as channels, only after 1 and 2 have shown on a hand-built check which rule works.** The working rule then becomes the seed that evolution varies (memo 2's experiment 3, memo 4's point 5).

Rotate-by-data, for indirection, stays on my list. It comes after these, because without output learning there is nothing for a pointer to point to.

## (c) Noise floor or substrate first

Both, because they do not compete. The noise floor needs no new code and is mostly machine time. The substrate checks are hand-built and cheap (partscheck-sized). I agree with memo 2 that the island-runs table is n=1 per switch, and that no more conclusions should be drawn from one-switch island runs until the floor across 4 evolutionary seeds is known. But a substrate gap that a hand-written learner cannot cross is not a matter of evolutionary noise. Echo2's icc of 0.00 says there is nothing to select. So: measure the floor on the machine, and settle the primitives at the desk in parallel.

## (d) The first gate

The paired-associate capacity curve, scored at two lifetime lengths, so memo 3's slope is measured inside it. Memo 4's byte-scaling test is the decisive later gate. As a first gate it mixes too many things: sparse front end, back-off and storage. And we already know the answer today (50.1% against 53.9%, below n-gram) without knowing why. The paired-associate curve isolates the one gap all four memos name, arbitrary input-to-output codes learned in a lifetime. It runs in minutes. If a hand-built learner cannot hold 32 pairs, byte prediction past n-gram is out of reach as well.

## Revised top recommendation

Make within-lifetime output learning work by hand first, as local bit rules: death for wrong pairings, then an eligibility bit with a modulator channel. Then put those rules' conditions into the genome as the seed that evolution varies, and select on fresh-content tasks with a learning-slope case. Until a hand-built learner passes, evolution has nothing to find.

## The first experiment

A paired-associate island check, under the fixed body:
- the world shows N glyph-to-act pairs with codes drawn per lifetime, N = 2, 8, 32, 128, at lifetimes of 1x and 8x;
- held-out recall, on 24 worlds and 3 seeds, with a blind control;
- three hand-built genomes compared: today's runtime, plus death on falling confirmation, plus the eligibility bit.

It passes if recall at N = 32 is well above chance and rises with lifetime length. If neither rule passes, the kernel itself needs redesign before any more island runs.
