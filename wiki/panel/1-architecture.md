# Panel memo 1: the primitives and the learning rules

## What the substrate can express

On expressiveness the substrate is fine. A kernel is a linear threshold unit with binary weights (`input_stats` and `agreement` in `src/kernel/simple.rs`). AND-NOT and XOR make it universal. Held channels are flip-flops, shifted writes are delay lines, and gates switch circuits on and off. Hand-made and evolved circuits already show recall to lag16 (run 21), shift-invariant shapes (run 26) and echo at twice its goal (partscheck, 8-line one-word codes: 300). So "can a bit-op network represent an agent" is not the open question. The open question is whether a lifetime can learn arbitrary mappings with this substrate, and whether evolution can find the circuits that do the learning. On both, the evidence says not yet, and for reasons built into the structure.

## The structural gaps

**1. A kernel cannot learn what it writes.** Its output is fixed at birth (lesson 51). It is either copied from the busiest destination word (`repeat`), a fresh random code (`new_code`), or whatever lay uncovered when it was born (the birth code in `src/program/neurocomp.rs`, around lines 1030-1060). After that only the input side learns. `learn_output` moves lines only for kernels that cancel. Three results trace back to this:
- The reader grew one kernel per glyph, all eight writing token 3.
- Across 80 timing settings, copy peaked at 16.
- Copy learning of outputs peaked at 19 on echo2 and was removed.

The experiment page puts it plainly: "No timing fixes a pairing made once, at birth, by chance." Conversation and operating a computer are both, at the core, learning very many arbitrary X-to-Y maps where Y is a code chosen by someone else. Today that can only happen by a lucky birth, never by correction.

**2. Topography forbids routing.** A kernel writes at the same relative place it reads. Glyph kernels could never reach their sound: no output overlapped its sound by more than 10 of 16 lines. The sentence check moved the problem from the way out to the way in. The fix that worked, `ISL_ONE_WORD` (every code in word 0), changed the world to fit the substrate. A screen, a text buffer or a dialogue needs content to move between positions and roles. Nothing in the substrate takes "where" as data.

**3. One 64-bit word holds three or four codes.** The crowding table (16-line codes unreadable once three are ORed together, 8 lines near 20%) together with lessons 4 and 22 (a binding must own its bits, and codes must be copied unchanged) limits working memory per kernel to a handful of items. Wide channels are cheap (lesson 35), but no kernel reads across them in a learned way.

**4. Credit is nearly absent.** The only signals are execution and learn gates (whole-population, a frame back) and confirmation, a cancelled XOR or coincidence. Win-stay has not passed since fresh worlds were introduced. The repeat rung sits near 0, and no feel rung passed until a goal was lowered. Confirmation by coincidence was "too rare and too lenient": wrong kernels replicated, 319 of them. A learn gate tells a whole population when to learn. It does not tell it which kernel caused the outcome. There is no eligibility trace at kernel level, even though the history ring could hold one.

**5. One-shot births bind whatever is present.** The reader was born while the network was babbling, so its pairs were chance. Lesson 15 also stands: one-shot conjunctions do not generalise. Inhibition cannot repair a wrong pairing, because the kernels involved never match at the same time (the `LQ=1` check).

**6. No indirection, so no variables.** XOR binding exists, but a reader has to unbind it, and rotated codes nobody unbinds stop copying (lesson 22). Rotation amounts are genes, not data. Without a value that says where or which, there is no pointer, no role/filler slot and no reuse of a learned map at a new position. Compositional structure has nothing to compose with.

## What this means for evolving general agents

Run 29 is the clearest sign. All 15 rung moves came by generation 74, then nothing for 460 generations. Every island was stuck on a rung that needs learning within a lifetime: feel, echo2, lag3, glow, cued, cycle1. Evolution is good at innate wiring: reflexes, place codes (motorcheck: tell 1024 once the codes are fixed), and now a fixed body. But the learning machinery itself (the move rule, the birth rule, confirmation) is hard-coded in the runtime, and evolution can only tune its parameters. A general agent is mostly learning machinery. If evolution is to produce one, the learning rules have to sit inside the search space, written in the same bit math.

The change I would argue for, explicitly challenging the "only primitives" list, is this: **make a kernel's masks readable and writable bit vectors, like any other channel.** Then:
- output learning is a population writing another population's output mask;
- three-factor credit is an AND of a modulator channel with a kernel's own fired bit at t-k from history;
- routing is a write whose target words come from a vector.

Everything stays a cheap word op with no floats, but plasticity becomes something circuits do, so evolution can find it. The second change: **a rotate or select whose amount is read from a word**, which is still one instruction. That gives pointers and position-free codes.

## Feasibility

To be candid: the best hand-made result is echo with eight tokens at 136, and reading (glyph to sound) has never worked. Conversation needs thousands of words, plus sequence structure and turn-taking. Operating a computer needs reading a screen, remembering goals across hundreds of steps (lesson 27: reward 25 or more steps away was not learnable), and mapping intentions to keystrokes. That is several orders of magnitude beyond today, with a budget of about 600 generations on nine islands. Without the two changes above, I do not believe evolution over this grammar gets there. The grammar cannot even state the missing learner compactly. With them, the next milestone is a lifetime that learns 100 arbitrary paired associates and reuses them. If that cannot be evolved within a run, more compute will not rescue the program.

## The next experiments

**1. A capacity curve for paired associates.** The world shows N arbitrary glyph-to-act pairs, with codes drawn each lifetime, and scores held-out recall, for N = 2, 8, 32 and 128. Include a hand-written bit-op learner as a ceiling and a blind control. Run it on today's substrate first. This becomes the yardstick every later change is judged by: today I expect it to collapse past N of about 4.

**2. Masks as channels, rule as gene.** Expose each population's output mask as a writable vector. Add a small rule-table gene: which of {fired, held down, gate open, confirmed} x {line on, line off} moves a mask bit, on the input or the output side. Seed it with today's Hebbian/anti-Hebbian entries. Evolve on the reading ladder and on experiment 1, and compare against run 30 on the same seeds. Success means evolution moves the rule away from its seed and copy passes on held-out worlds.

**3. Indirect addressing.** Add a rotate-by-data op, where the amount is the low bits of a word the kernel reads. Test two things: whether the sentence check becomes position-free, and whether a map learned at one place transfers to another (glyph wire not required). This is the smallest test of whether variables can arise at all.

A pretrained model belongs in the world only, as a teacher that speaks and reacts. Use it only after experiment 1 shows the learners can hold enough associations to profit from one.
