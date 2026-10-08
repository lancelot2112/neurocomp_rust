# 86 · An evolvable genome: small changes, small effects?

**Question.** For the network to be evolved rather than designed, a small change to its
genome should usually make a small change to what it does, and new structure should be able
to appear without breaking what works. The genome's stack text ([genome](../concepts/genome.md))
is easy to read but positional: one inserted or deleted instruction moves every signal below
it. Does a gene-list form, with permanent ids and wiring by id, behave better under
mutation?

**Code.**
- [`src/program/genes.rs`](../../src/program/genes.rs): `GeneList`, the evolvable form.
  - Every gene has a permanent id; its inputs name genes by id and port.
  - Loops are explicit (read the step before), and the tick order comes from the wiring.
  - Each gene carries its own parameters, and numbers that follow from the wiring (a
    concat's slots, a predictor's frames) are not genes.
  - Each predictor has its own random stream, seeded by its id. With one shared stream, a
    gene nothing reads still shifted every other module's draws.
  - The stack text compiles to it. `mutate` applies one of five local operators:
    - `nudge`: a number × or ÷ 2^(1/4), a fraction ± 1/16, or a flag flipped;
    - `rewire`: one input to another gene's output;
    - `add`: a new gene nothing reads yet;
    - `duplicate`: a copy of a gene, which nothing reads yet;
    - `toggle`: a gene off or on.
  - `mutate_stack_text` mutates the stack text for comparison: `number` (the same nudge),
    `delete`, `insert` or `swap` a token.
- [`scripts/robustness.sh`](../../scripts/robustness.sh): the base genome and single
  mutants, each scored as mean held-out accuracy over three seeds, compared with its own
  form's base. `MUTATE=<op>:<k>` and `MUTATE_STACK=<op>:<k>` in the harness pick the k-th
  mutant.
- Unit test: four silent additions and four duplications leave the network's output
  bit-for-bit unchanged.

## Results (`hierarchy.gen` on the habit task, seeds 0 / 1 / 2; base 86.7% held-out in both forms)

| Mutation | Mutants | Within 5 points | Exactly 0 | Worse by 20+ | Range |
|---|---|---|---|---|---|
| Gene list: nudge | 10 | **10** | 1 | 0 | −2.8 to +4.3 |
| Gene list: add, duplicate | 6 | **6** | **6** | 0 | 0 |
| Gene list: rewire | 6 | 0 | 0 | 5 | −81 to −5.5 |
| Gene list: switch off | 4 | 0 | 0 | 4 | −87 to −50 |
| Stack: number | 10 | 8 | 7 | 0 | −7.3 to +0.7 |
| Stack: delete, insert, swap | 16 | 6 | 6 | **10** | to 0% (8 mutants) or 16.8% |

Two nudges and two rewirings drew the same mutation twice (the samples are small).

## Findings
1. **Parameters are smooth in both forms.** A nudged number moved the score by under 5 points
   in 18 of 20 mutants, and never by more than 7.3. Learning absorbs most of it: a changed
   sample size, threshold or window kept the circuit's function.
2. **Growth is neutral at birth only in the gene list.** Every added or duplicated gene left
   the score exactly unchanged (6 of 6): it computes and learns, but nothing reads it until
   a later mutation connects it. In the stack, 10 of 16 structural changes broke the network
   (0% or 16.8%): one token moved every signal below it.
3. **Changing an existing connection is still drastic in both:** rewiring a used input
   (5 of 6 worse by 35–81 points) or switching off a used gene (−50 to −87) replaces what the
   circuit computes. That is the same thing done abruptly, not a fault of the encoding.
4. **What would make those smooth too: graded wiring.** Give each connection a gain (the
   share of its bits that pass, as the routing shares of the work after 85 do). Rewiring
   then becomes adding a second source at gain 0 (neutral), followed by nudges of its gain;
   switching off becomes stepping a gain down. Every operator then has a neutral or small
   form, which is what lets evolution move through structure without falling off a cliff.
5. **The base genome scores the same in both forms** (86.7%), even though the gene list
   gives each predictor its own random stream: with canonical kernels, growth here draws
   almost nothing at random.

## Addendum: graded wiring

**Code.** Every input of a gene is a set of connections, each with a **gain**: the share of
its source's bits that pass (a fixed subset of bit positions per gain, as a synapse's
strength sets how much of a pathway gets through), several combined by a `Blend` kernel. A
connection at full gain is a plain wire (a compiled genome is unchanged); one at gain 0 is
not built (exactly neutral). New operators: `connect` (a source at gain 0), `strengthen` (a
source at 1/16, the first step after it), `weaken` (a connection loses 1/16), and `nudge`
now also moves gains. The gene list prints partial connections, e.g.
`[g8:0@prev + g4:0*3/16]`.

**Results** (as above; base 86.7%):

| Operator | Mutants | Within 5 points | Exactly 0 | Worse by 20+ |
|---|---|---|---|---|
| connect (gain 0) | 8 | 8 | **8** | 0 |
| weaken (16/16 → 15/16) | 8 | **8** (−2.0 to +0.3) | 1 | 0 |
| strengthen (0 → 1/16), first version | 8 | 3 | 2 | 4 (−51 to −87) |
| strengthen, after the two fixes below | 12 | **10** (−3.1 to +4.7) | 1 | 1 (−70) |
| rewire, abrupt (for comparison) | 4 | 0 | 0 | 3 (−41 to −81) |

The large drops of the first version had two causes, neither the gain itself:
- **Width.** A new source wider than the input it joined (the area's two-frame input into
  the column's one-frame top-down slot) reshaped the whole row. **Fix:** a blended input
  keeps its main connection's width; other sources are cut to it.
- **Teaching ports.** Extra bits on a predictor's teaching input become part of what it
  learns to predict. **Fix:** teaching ports take no new connections (the teacher's path is
  fixed, as a climbing fibre's is); they can still be weakened.

After the fixes, the two drops left (−10 and −70) both fed the surprise comparator's
compared input: extra bits there change what counts as surprising. Comparator inputs now
take no new connections either (this last fix is not separately remeasured).

**Findings.**
1. **Rewiring can now be gradual:** connect at 0 (neutral), strengthen by 1/16 steps
   (small: 10 of 12 within 5 points), weaken the old connection by 1/16 steps (small: 8 of
   8 within 2 points). The abrupt rewire cost 41–81 points in one step.
2. **Some wiring must be fixed for evolution to be smooth:** what a learner is taught by and
   what a comparator compares. That matches the brain, where teaching and error pathways
   (climbing fibres, the comparator inputs of CA1) are specific, not diffuse.
3. **Flags are still jumps:** `copy_growth` 0 → 1 cost 35 points in one nudge. A switch
   could become a probability (applied a growing share of the time).

## Addendum: moldable width

**Code.** An input's width (in word-sized frames) is a gene. Sources are composed to it frame
by frame by `Blend`: a wider source is folded (its frame j into frame j mod n, ORed:
converging pathways superpose, and a word keeps its code, so it can still be copied); a
narrower one is padded with empty frames. Unset, an input is as wide as its main connection
(compiled genomes unchanged). `reshape`: one input a frame wider or narrower.

**Results** (8 mutants, as above): widening the higher area's reading input (2 → 3 frames)
changed nothing (a trailing empty frame: kernels do not grow into it until something fills
it). Widening a pass-through gene's input (the bag, a delay, the concat's window slot,
1 → 2 frames) cost 50–87 points (7 mutants): those genes pass their width on, and a concat
pads every slot to its widest, so one wider signal moved every other signal's position in the
row the learner reads. `reshape` is now limited to a learner's reading input; a concat that
places each slot at its own width (so widening the last slot appends) would let it reach
further.

## Next
- Flags as probabilities.
- ~~Graded wiring~~ (addendum).
- Larger samples and more tasks; then a first search (mutate, score over several seeds,
  keep the better) from `hierarchy.gen`.
- Higher areas and grown areas as definitions placed by `sub:` (a gene that is a whole
  circuit, duplicated and diverging).
