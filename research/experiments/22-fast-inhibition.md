# 22 · A fast-learning inhibitory loop in L2/3

**Question.** Every inhibition in the cortex so far was fixed: the winning kernel blocks
others writing the same output, a hard winner-take-all whose strength is never learned.
Cortex also has interneuron loops whose effect changes within seconds (adaptation,
short-term depression, inhibition of return). Can a one-shot, decaying inhibitory loop
let the column use what *just* happened in this story, something its slowly learned
statistics cannot hold?

**Code.**
- `FastInhibition`, `KernelClass::set_fast_inhibition` and `fast_inhibit` in
  [`src/kernel/class.rs`](../../src/kernel/class.rs).
- `CorticalColumn::fast_inhibit` in [`src/program/cortex.rs`](../../src/program/cortex.rs).
- `TASK=elim`, `FAST_INHIBIT=hits|misses|both`, `FAST_TTL` and `FAST_RELSHIFT` in
  [`examples/episodic.rs`](../../examples/episodic.rs).

Run: `COPY_GROW=1 GROW_TRUST=0.5 STICKY=255 GENERALIZE=0.5 GENERALIZE_AFTER=1
TRUST_AT_TEST=0.5 TASK=elim POLICIES=nomemory FAST_INHIBIT=hits FAST_RELSHIFT=3 cargo run
--release --example episodic`.

## Task: elimination
> is it the kitchen ? no . is it the office ? no . is it the hallway ? no .
> is it the bedroom ? no . is it the garden ? no . is it the **bathroom**

Five of the six places are named in random order, each rejected, and the column must
predict the sixth. Nothing in the story states it, and every place is equally frequent in
training, so slowly learned statistics give chance (1/6). The answer is "the one I haven't
seen yet".

## Mechanism
- **Tags:** after each next word arrives, kernels get a one-shot inhibitory tag that
  lasts `FAST_TTL` steps (40, about one story).
  - `hits`, adaptation or inhibition of return: the kernels that predicted what
    happened.
  - `misses`, error-driven: the kernels whose prediction failed.
- **Effect:** a tagged kernel loses the winner competition to any untagged candidate.
  If nothing else matches, it still wins, so a context with a single continuation keeps
  working.
- **Test time:** the loop runs whether or not slow learning is on (`fast_inhibit`). It
  is a fast process, not training.
- **Gate:** only kernels whose hit rate is below 2^k/(2^k+1) are tagged, tested in
  integers as `(misses+1) << k > hits+1` (see [probability in bits](../concepts/probability-in-bits.md#done-the-predictors-own-statistics-in-integers-22)).
- **Storage:** tags are an expiry array, one step number per kernel. Tagging is one
  write and checking is one compare; nothing is counted down.

## Results (held-out, seeds 0 / 1 / 2)

| Fast inhibition | Elimination | Varied stories (episodic memory) |
|---|---|---|
| off | 16 / 17 / 14% | 100% |
| episodic memory, no inhibition | 17 / 19 / 21% | – |
| misses (inhibit failed guesses) | 16 / 17 / 14% | 99.8–100% |
| hits, no gate | **100 / 100 / 100%** | **78–84%** |
| hits, gate k = 0 (rate < 1/2) | 16% (seed 0) | 100% (seed 0) |
| **hits, gate k = 3 (rate < 8/9)** | **100 / 100 / 100%** | **100%** on every seed (1–2 and 1–3 facts) |

## Findings
1. **Inhibiting what just happened solves elimination; inhibiting failed guesses does
   not.** The answer is "the continuation not yet seen in this context". Adaptation
   provides exactly that: each named place's kernels are tagged, and the one untagged
   sibling wins. Error-driven tags only stop the column repeating its *own* wrong guesses,
   which are not the places the story named.
2. **The first version silently did nothing.** Tags were set inside slow learning
   (`feedback`), which is off at test, so at test the loop never fired (0–1 tagged kernels
   at the answer). A fast process must not depend on the slow one.
3. **Ungated adaptation has a real cost: it favours novelty everywhere.** On varied
   stories it fell to 78–84%. The diagnostic showed why: memory-copy kernels ("memory holds
   kitchen → kitchen") fire during the story, get tagged, and lose at the question to a
   habit kernel with hit rate 0.17.
4. **The gate's line falls where the brain would put it: inhibit only uncertain
   predictions.**
   - At "is it the", the winners' hit rates are 0.17–0.50.
   - Copy kernels sit at 1.00.
   - A cut at 8/9 tags the first and spares the second.
   - The cut at 1/2 fails, really (confirmed with the integer gate at k = 0): too many
     narrow kernels that are right as often as wrong stay untagged, and any one of them
     can win and repeat a named place.
   - The gate is a shift and a compare, with no divide.
5. **Elimination is easy for a transformer**
   ([comparison](../concepts/brain-transformer-comparison.md)): attention looks back at
   the named places, and a 28k-parameter model reaches 100% in 10 passes. Our column
   needed a new mechanism because its context is three frames, not the whole story.
   Fast inhibition is how a short-context system keeps a story-long "already seen" set.
6. **A cost the task exposed: 17,247 kernels matched the same context.** Every miss on
   an inherently unpredictable continuation grows another kernel. This led to the
   compaction work of [23](23-compaction.md).

## Integer only
This experiment also moved the predictor's statistics to integers. The kernel hit rates,
winner ranking, trust floors and this gate are all integer compares, on 8-bit counters
halved together. Results were unchanged, and per-kernel statistics shrank from 8 bytes to
2. Details are in [probability in bits](../concepts/probability-in-bits.md#done-the-predictors-own-statistics-in-integers-22).

## Biology
- **Adaptation and inhibition of return:** responses to a stimulus that just occurred
  are suppressed for seconds (stimulus-specific adaptation, partly through somatostatin
  interneurons and short-term synaptic depression). Attention is biased away from
  locations already visited (inhibition of return).
- **Uncertainty-dependent:** adaptation is strongest where the input is unpredictable,
  and an expected, reliable signal is not suppressed. The reliability gate is a crude
  version of that.
