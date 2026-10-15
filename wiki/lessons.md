# Lessons

The durable lessons from the earlier branch (`ccr-8d772201-st7vqw`, research wiki under
`research/`, 108 experiments). Each one cites the experiments it rests on. Numbers are from
3 seeds unless noted, and a difference of a few points across 3 seeds is noise (lesson 30).

## Codes and density

1. **Sparsity is the main lever.** Set thresholds from the expected number of active bits in
   what the kernel reads, not as constants. The stock network's threshold of 16 on a 64-bit
   window saw about 4 active bits, so nothing ever fired (01).
2. **Keep density low where a code is read.** A kernel firing on 5 of its 8 bits matches
   unrelated input about 36% of the time at density 0.5, about 3% at 0.25, and about 1e-6 at
   0.03 (108).
   - Write one output bit per active cell, at scattered positions; never contiguous runs.
   - Dense outputs erased the signal downstream.
   - On this branch: keep each code inside one 64-bit word (topographic), so a one-word kernel
     sees it at density 1/4. Spreading codes over a whole vector forces whole-vector masks and
     32× the popcount work (experiment 01, update).
3. **Codes that are accumulated need more room than codes that are matched.**
   - Matching: 512 bits with 32 active is fine.
   - Statistics: aim for V·k/n ≲ 1 (V symbols, k active of n bits). Part-of-speech agreement
     went from 36% at 1024/32 to 75% at 32768/4 (05, 11).
4. **A binding must own its bits.** Packing about 12 role-filler bindings onto every bit, by
   rotation, made novelty unreadable per bit, so separation, tagging and replay all failed (48).
   Binding-owned sparse codes fixed all three (49).
5. **Break ties by hash, never by index or visit order.** Index tie-breaking collapsed every
   code into one attractor (49). Visit order biased winners (23).

## Kernels and their learning

6. **A cell can be just masks and popcount.** That holds when learning moves single bits,
   chosen by a random word uncorrelated with the input, and when the rate and threshold
   self-calibrate (104, 105). The fully bitwise cell beat strength synapses: +1.7 and +1.6
   over the suite, 20 of 25 entries up.
   - With fixed rates it lost 1.6.
   - With a deterministic index in place of the random word it lost 4.7, because the index
     lined up with the stories' structure.
7. **Keep mask density when learning.**
   - Hebbian moves a bit from an off line to an on line.
   - Anti-Hebbian moves one from an on line to an off line.
   - Without this, cells shrank and fired on anything ([kernel cells](kernel-cells-and-context.md)).
8. **The threshold is a fraction of the kernel's own mask** (for example 3/4), and matching
   tolerance is per frame, never global. A global 80% let a whole frame be wrong and still
   fire (03, 12).
9. **Coverage recruitment.** The part of the input no firing cell covers, if it meets one
   cell's popcount, gets new cells. Without it a new object had no code at all (the-math §1,
   108 layer 4).
   - Deal the input's lines among sibling cells born together, or they inhibit each other at
     birth.
10. **Grow on surprise, not on activity.** "Grow a conjunction where the prediction missed" is
    the only learning that made a working sequence learner.
    - Reading a book once: 56.9% next-character accuracy, against 59.0% for a back-off
      n-gram using 5× the storage (02, 03).
    - Hebbian firing alone drifted (01).
    - Don't grow in contexts already known to be unpredictable (23).
11. **Inhibition between overlapping cells, with anti-Hebbian in the inhibited, separates
    patterns**, but only when balanced and bounded (107, 108).
    - Go and no-go masks start from the same random mask and update push-pull.
    - Recruit only up to an asymptote.
    - Density stays fixed.
    - Unbalanced pools ran away, or one context erased another.
12. **Fast one-shot anti-Hebbian tags work.** Inhibiting a cell that just predicted right gives
    a story-long "already seen" set: elimination went from 16% to 100% (22).
    - Gate it to uncertain cells.
    - Keep it out of the slow learning path, which is off at test.
13. **Make failure learning slower than success learning.** Then a cell for something that
    happens sometimes survives, and one for something that never comes again fades, without a
    counter. Fading at 1/2 collapsed everything; at 1/16 it worked ([kernel cells](kernel-cells-and-context.md)).
14. **Avoid duplicate kernels.** Random sampling made thousands of copies. Canonical (min-hash)
    sampling plus hash-consing cut kernels 5–14× (23).
15. **One-shot conjunction cells do not generalise**, and nothing stops attractor loops.
    - On prose they scored 9.8% held out, against 8.4% for always saying ","; layer 2/3 fired
      on 0.9% of held-out words (106).
    - Partial-match firing and firing adaptation are the proposed fixes.

## Prediction, surprise and context

- **Raw bytes (this branch, experiment 03).** Two-mask kernels (a care mask and the values
  expected) at scales of 1 to 1024 bytes over a sliding window predict Alice's next byte at
  50.1%, against 53.9% for a frequency lookup of the longest context.
  - Exact matching only: chance agreement between ASCII bytes makes any tolerance match
    everything.
  - No anti-Hebbian across nested scales: it destroys back-off.
  - Every kernel that matched checks its own prediction (the third mask). Checking only the
    winner deadlocks.
- **Confirmations carry context (experiment 04).** A reader of "matched and came true" codes
  holds a reversal through a click between light and outcome (7–10/10 swaps), where the old
  reader failed (0–4) and a reader of matches alone fails (2–3). Small codes (matched, failed)
  destabilise an assembly reader, because kernels born after it settled show up as
  unexplained.

16. **Prediction error is the signal, and it must be bound to its context.** XOR of input and
    prediction carries what is new. The error alone loses which context it followed: two
    different surprises can give identical error bits. Pass on the surprising step together
    with what came before it, and gate predicted steps out by their own popcount
    ([kernel cells](kernel-cells-and-context.md)).
17. **A union of familiar parts looks familiar.** Context lives in conjunctions. Cells born on
    the whole input are dealt lines from every part, so each fires only on the conjunction.
    Reversal detection rose from 3–5/10 to 6–9/10 ([kernel cells](kernel-cells-and-context.md)).
18. **Context primes, input triggers.** In a two-compartment cell, apical context lowers the
    basal threshold. Context never fires a cell alone (99–105).
    - Let the context cell override only when it bursts (+0.7, 17 of 25 up).
    - Never make the context layer the only output (−10.4) or replace the one-shot layer
      (−15.0).
19. **The prior gates (AND); evidence superposes (OR).** OR-ing the network's own prediction
    back in counts the prior twice, and bits carry no strength (45).
    - Partial and blended codes are fine, because cells match at 80% (44).
20. **History needs boundaries and exact lags, not just length or decay.**
    - OR-bags of a window cannot tell the current fact from stale ones. Resets at a
      contradicted fact fixed it (26–28).
    - Recency decay did not (35).
    - Leaky traces leave random leftovers.
    - A chain of exact lag stages from the history ring worked
      ([kernel cells](kernel-cells-and-context.md)).
21. **Higher areas learn the lower area's residual**, and are mixed in, not chained as frames.
    Residual: 81–90% against 65–76% for learning every word (24). Chained as frames: 0% (25).

## Memory and binding

22. **Binding needs a one-shot store and codes copied unchanged.** Held-out pairs scored 0%
    from the predictor (06), 98% from an episodic store (11), and 100% by copying through a
    relay (09). Rotated or tagged codes that a reader does not unbind stop copying (66).
23. **Superposed Hebbian stores have low capacity on skewed codes.** Common bits add up into
    one attractor: 0% recall at 16k events, against 96–100% for one row per event (45, 48, 55).
    - If superposing, scale each input by its use: 1/n by a reciprocal table, or a homeostatic
      per-cell rate subtracted at read time (52, 53).
    - Shifts round single writes to zero.
24. **Novelty gates the write.** Novelty = popcount(present AND NOT recalled); scale the write by
    it. Each of the two mechanisms alone recovered 1 of 16 bits; together, 16 of 16 (46, 52).
25. **Replay is prioritised by an encoding tag and comes in the learner's own format and
    order.**
    - Cue-free replay never visited a one-shot event (48).
    - Unordered sets taught nothing (50).
    - Replay read as ordinary input worked (62).
    - Predict before learning on replay, or credit goes to a stale winner (94).
26. **Recall reaches the reader only in retrieval mode**, gated by familiarity from the same
    retrieval. Otherwise recall is noise the reader learns to ignore (92, 93, 107).

## Credit and control

27. **Credit is causal and local in time.**
    - Co-activation credit is worse than random (08).
    - Credit the act against its own expectation in that context, with an eligibility window
      of about 2 steps. A window of 8 lost the discrimination (108).
    - Payoffs 25 or more steps away are not learnable by reward (107).
28. **Learned gates fail on dense per-step reward**, and any per-use cost collapses them (43,
    65, 68, 83). Train and test must gate alike (83).
29. **The world must be honest.** A blocked move or a click on nothing must be sensed as
    nothing. Untagged demonstrations and masked stimuli created superstitions and false results
    (107, 108).

## Method

30. **Seed noise is large.** Changing 0.2% of training steps moved one seed by 22 points (85).
    - Use 5 or more seeds.
    - Compare variants only in the same build.
    - Keep runs deterministic, with a separate random stream per subsystem.
    - A changed draw pattern alone moved results 5 points (47).
31. **Integer and bit arithmetic loses nothing.** Floats moved to integers left 10 of 13 entries
    identical (47). Rates as (h+1)/(h+m+2) compare by cross-multiplying, and a threshold
    2^k/(2^k+1) is a shift (22).
32. **Hand-written rules hide what the network does.** The question act's 68 → 91% came from a
    hand rule (89, retracted). The right candidate matched at 925 of 1,000 answers and lost to a
    hand-set ranking (94). A hand rule is only allowed as a labelled reference.
33. **Measure against a dumb baseline, store each episode once, and keep test data out.** A text
    search's 67.6% held out had stored the test stories; corrected, 17% (57 → 58).
34. **Event-driven exact speedups pay, dense hardware does not.** Bit-position lists, an inverted
    index from bit to cells, canonical kernels and incremental match counts gave 85–190× (23).
    Build with hardware popcount (target x86-64-v3).
35. **Width is free until it is read (experiment 09).** A step costs the words an edge reads and
    writes, so a 131,072-bit channel read through a 64-word field costs the same as a 512-bit
    one. A kernel spread across a wide field pays for its mask's whole extent if its mask is
    dense; a sparse kernel (the words it touches) pays for its lines, about 6x cheaper at
    131,072 bits.
36. **Every probe needs a blind control, under every world it is scored in (experiments 12,
    13).** A genome that sees nothing must score about 0. Blind controls caught:
    - a clock-locked clicker scoring respond > 900;
    - near lags counted with more chances than far ones (a blind speaker's echo 85 with 2
      tokens, 10 with 8);
    - copy and win-stay scores from genomes that read no port.
    A stage's world can make a bias the plain world hides.
37. **A capacity is reachable only one circuit at a time (experiment 13).**
    - Copy needs two learned maps chained (glyph to sound, sound to voice) and scored 0 for 100
      generations on every island. Break such a capacity into stages scored alone, each one
      circuit beyond the last: babble, echo, then the teacher fading.
    - A ladder moves an island when the step is one circuit. Switching went from reading no port
      to passing respond in about 30 generations, then passed tell at once.
38. **Judge an elite on many fresh worlds, not the top genome on a few (experiment 13).**
    - One genome's mean on 12 held-out worlds moves by about ±25-50, enough to fake a skill
      twice in one run.
    - Lexicase's top genome changes almost every generation, and a capacity most of the
      population holds can be missing from it.
    - Report on 48 held-out worlds and recheck claims on 64 fresh ones (`examples/recheck`).
