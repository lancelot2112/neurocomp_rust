# 100 · Primed layer 5: the tuft primes, the input triggers

**Question.** In [99](99-two-compartment-layer5.md) a cell burst only when both of its
synapse sets passed a fixed threshold, an AND. In a pyramidal cell, apical input mostly
depolarizes below threshold: it *primes* the cell, and a primed cell fires on weaker input
from below, earlier than its neighbours, and bursts (Larkum 2013; the predictive state of
Hawkins' HTM). The context does not predict the next word; it says "the context is right for
me", and the input triggers. Does a graded, primed layer 5 do better, learning by Hebbian
and anti-Hebbian plasticity from random contexts? And does growing cells as needed beat a
fixed pool?

## Code
`PrimedLayer5` in [`src/program/layer5.rs`](../../src/program/layer5.rs) (two tests);
options `L5=primed`, `L5_CELLS` (default 8,192), `L5_GROW=1`:
- **Two popcounts.** The tuft's share of active connected synapses is the cell's
  *priming*: graded, carried over to the next words and decaying by 1/4 per step. Priming
  lowers the basal threshold from 80% of the basal synapses to 50% at full priming. A cell
  that fires with priming at or above the area's context threshold bursts. A burst beats a
  spike; then higher priming, then the burst trace. A burst overrides L2/3's prediction.
- **Matrix gain.** The context threshold is one value for the area, as diffuse matrix input
  is. It falls while nothing bursts and rises while bursts happen, between 0.3 and 0.9. Tufts
  stay selective; the matrix sets how selective the area is now.
- **Hebbian learning** on synapse permanences (connected at 128):
  - a confirmed winner strengthens its active basal synapses, and if it burst its active
    apical ones (+12), and weakens inactive ones (−6);
  - a contradicted burst depresses the winner's active apical synapses (−24): this context
    does not make it right;
  - **lateral inhibition:** the other primed cells that fired with a different output lose
    on their active apical synapses (−12, anti-Hebbian), so they separate from this pattern
    and can learn others.
- **Pool** (default): free cells are wired to random contexts, sampled from a random moment
  of experience (one free cell per step). Where no burst predicted the next input, the most
  primed free cell commits: it takes the input on its basal synapses and the next input as
  its output. A committed cell that keeps failing is freed.
- **Grown** (`L5_GROW=1`): no pool. Where nothing predicted the next input, a cell is
  created with its tuft sampled from the recent context (this step's and the previous
  step's) and its basal synapses from the input. Past the cap, the least recently useful
  cell is recycled. This is the limit of a large reserve of silent cells, where the one
  recruited is the one wired closest to the moment.

## Results
`LEARNING=three SLOW_P=0.25 SLEEP_P=1 REPLAY_PREDICT=1`, five seeds, held-out:

| Entry | Reference | Two-compartment AND ([99](99-two-compartment-layer5.md)) | Primed, pool | Primed, grown |
|---|---|---|---|---|
| family consolidated | 47.7 | 45.4 | 44.4 | 39.9 |
| hippocampus teaches cortex | 51.5 | 48.3 | 47.8 | 41.0 |
| index hippocampus | 34.1 | **45.9** | 38.6 | 42.3 |
| engram store | 45.4 | 47.9 | 43.0 | 46.4 |
| semantic store | 38.6 | 43.6 | 40.8 | 40.7 |
| mean change | | **+2.8** | −0.5 | −1.4 |

Per seed, hippocampus teaches cortex: pool 58.6 / 38.2 / 55.2 / 26.6 / 60.6, grown
35.2 / 56.2 / 42.8 / 35.0 / 35.6.

On seed 0 (five entries):

| | Pool | Grown |
|---|---|---|
| column alone right at test answers | 48–56% | 25–61% (four of five 51–61%) |
| mixed answer right | 50–70% | 46–63% |
| context threshold settles at | 0.30–0.39 | 0.51–0.63 |
| bursts / spikes won | about 50 / 50 | about 65 / 35 |
| cells | ~7,800 of 8,192 committed, ~190 freed | cap reached; ~17,500 grown, ~9,400 recycled |
| training cost | 5.6 ms per word | 6.2 ms per word |

The reference's column alone was right at 21%; the default network trains at about
0.2 ms per word.

## Findings
1. **Priming makes the column answer.** With either primed layer, the column alone is right
   at about half the test answers on most entries, against 21% in the reference and 29% with
   the AND. Graded priming with a lowered trigger is what lets the context-specific cell fire.
2. **The answers do not reach the mixed result.** On some seeds the mix rarely changes the
   column's answer (2–3% of answers), on others it overrules it badly. The mix's
   reliabilities are keyed by the old column's behaviour and by word pairs. The column
   improved; the integration did not follow. The held-out mean is no better than the
   reference, and the seeds spread widely (26.6–60.6 on one entry).
3. **The simpler AND layer of 99 remains the best on the held-out mean** (+2.8), though
   its column alone is weaker. The primed layers help the hippocampus-lesioned entry (index
   hippocampus) and lose on the two family entries.
4. **Growing on demand is not better than the pool:** it fills the cap and churns (about
   9,400 recycled), and it is not faster. The pool's random contexts work as well as cells
   wired to the moment.
5. **Cost:** both primed layers are about 25–30× slower than the default network per
   training word. The layer evaluates its cells twice per step, through an index that is not
   yet optimised.

Nothing becomes a default.

## Next
- **The integration:** the column's burst should enter the mix as its own per-moment
  evidence (a bursting column outweighs the word-keyed table), so a better column gives a
  better answer.
- **The primed L2/3** ([`L23=primed`](../../examples/episodic.rs)): the column's predictor
  itself as primed two-compartment cells, built and next to run.
- **Inhibitory interneurons instead of the gain rule:** SST cells inhibiting the tufts
  (the context threshold), VIP cells releasing them (driven by the matrix, top-down input,
  acetylcholine), PV cells for the lateral inhibition. Selectivity then varies by place and
  moment instead of being one value set by a rule.
- **Speed:** a single evaluation per step, connected synapses only in the index.
