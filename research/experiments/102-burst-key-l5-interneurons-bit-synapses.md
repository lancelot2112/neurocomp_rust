# 102 · A learned burst reliability, interneurons in layer 5, and bit synapses

**Question.** Three follow-ups to [101](101-primed-l23-vote-interneurons-speed.md), plus the
speed work started there:
1. the burst vote at face value hurt (−5.9). Can the mix *learn* how much to trust the column
   by its burst state, instead of by the last two words?
2. the SST/VIP interneurons did nothing in L2/3, where spikes predict too. Do they help in
   layer 5, where only bursts speak?
3. **bit synapses:** can learning run on bits alone, by adding, removing and moving them,
   with no strength counted per synapse? Each synapse is absent, *silent* (NMDA only: it
   counts only under a precondition), *active* (AMPA) or *sticky* (consolidated).

## Code
- **Learned burst reliability** (`BURST_KEY=1`): the mix keys the column's reliability by
  its burst state (burst or spike × priming band) instead of by the word pair. It is the same
  learned reliability (hits and misses) under a key without words.
- **Interneurons in layer 5** (`L5=primed INTERNEURONS=1`): SST and VIP set the context
  threshold, as in 101.
- **Bit synapses** (`L5_SYN=bits`, `PrimedLayer5::set_bit_synapses`;
  [`src/program/layer5.rs`](../../src/program/layer5.rs)):
  - three flags per synapse: active (AMPA), silent (NMDA), sticky;
  - **counting:** popcount(input & active) + (primed ? popcount(input & silent) : 0), against
    a fixed 80% of the basal synapses. Priming lowers the bar by letting silent synapses
    count (the NMDA precondition), which replaces the hand formula "80% falling to 50% with
    priming";
  - **a new cell's input synapses are silent:** it can fire only when its context primes it;
  - **a confirmed fire:** active silent synapses become active (1/2), active ones become
    sticky (1/8), unused non-sticky ones are pruned (1/16), and one silent synapse may grow
    onto an active input (1/4), into a pruned slot if there is one (the "move"). After a
    burst the same on the tuft, whose new synapses are active;
  - **a contradicted burst** prunes active non-sticky tuft synapses (1/4) and unsticks active
    sticky ones (1/8);
  - **lateral inhibition:** the losing primed cells prune their active non-sticky tuft
    synapses (1/8);
  - the vectorized backend has a third table, for silent synapses. Tests: bit-synapse cells
    learn the same input toward different outputs in two contexts, and the vectorized and
    event-driven counts agree exactly over 3,000 steps.
- **Speed:**
  - each cell keeps its connected-synapse counts current, instead of recounting them for
    every touched cell;
  - the build targets the native CPU (`.cargo/config.toml`), so `count_ones` uses the
    hardware popcount ([101](101-primed-l23-vote-interneurons-speed.md), finding 5).

## Results
`LEARNING=three SLOW_P=0.25 SLEEP_P=1 REPLAY_PREDICT=1 L5_VEC=1`, five seeds, held-out. The
reference is three systems + sleep + the replay fix:

| Entry | Reference | Primed L2/3 (101) | + learned burst key | + bit synapses | Primed L5 | + interneurons |
|---|---|---|---|---|---|---|
| family consolidated | 47.7 | 44.8 | 36.1 | 46.2 | 44.7 | 43.1 |
| hippocampus teaches cortex | 51.5 | 45.2 | 49.3 | 45.7 | 48.8 | **53.4** |
| index hippocampus | 34.1 | 41.8 | 40.3 | 36.1 | 38.5 | 38.6 |
| engram store | 45.4 | 47.1 | 44.6 | 44.0 | 46.2 | 45.6 |
| semantic store | 38.6 | 47.4 | 45.8 | 40.3 | 40.6 | 42.2 |
| mean change | | **+1.8** | −0.2 | −1.0 | +0.3 | **+1.1** |

**Bit synapses on seed 0:**
- about 12,000 cells committed and 50–80 freed (strength synapses: 100–170 freed);
- bursts about 75% of the column's predictions, spikes about 23%, the old kernels 3–5%;
- the column alone right at 29–40% of test answers (strength synapses: 40–65%).

**Interneurons in layer 5:** SST settles at 0.45 and VIP at 0.54 (seed 0, hippocampus
teaches cortex). There the column alone and the mix both reach 68.9%.

**Speed** (hippocampus teaches cortex, seed 0, primed L2/3, same held-out 54.4 each time):

| | Time |
|---|---|
| without a primed layer | 150 s |
| primed, event-driven (101) | 271 s |
| + vectorized, native instructions, counts kept per cell | **189 s** (1.26×) |
| bit synapses, vectorized | 214 s |

## Findings
1. **The learned burst reliability does not beat word-pair keys** (−0.2 vs +1.8). It gains
   on hippocampus teaches cortex (+4.1 over primed L2/3) and loses badly on family
   consolidated (36.1). Burst state alone cannot tell a reliable burst from an unreliable
   one. The word pairs also carry *where* in the sentence the column is answering.
2. **Interneurons help layer 5** (+0.8 over primed L5): hippocampus teaches cortex rises
   above the reference (53.4), with two seeds rescued (26.6 → 50.0, 60.6 → 64.0). Where only
   bursts reach the output, the inhibitory circuit's threshold matters, and a threshold driven
   by error (VIP) and activity (SST) works better than the gain rule.
3. **Bit synapses learn, but less well than strengths** (−1.0 vs +1.8). The column alone is
   weaker (29–40% vs 40–65%). With silent synapses, cells fire mostly when primed (75%
   bursts), and pruning is rare (50–80 cells freed), so the layer is more context-bound and
   less able to fire on input alone. Family consolidated is better than with strengths
   (46.2 vs 44.8). The rates (1/2, 1/8, 1/16, 1/4) are first guesses.
4. **The primed layer now costs 1.26× the network without it** (189 s vs 150 s), down from 1.8×.

Nothing becomes a default yet: the two positive variants (primed L2/3 +1.8, primed L5 with
interneurons +1.1) are under three learning systems and need the default suite first.

## Next
- Run the default suite (one learning system) with primed L2/3 and with primed L5 plus
  interneurons; promote what holds.
- Bit synapses: tune the transition rates (faster silent → active, so cells can also fire on
  input), and let the precondition be any depolarization (enough active input on the same
  dendrite), not only priming.
- Combine the two positive variants (primed L2/3 + primed L5 with interneurons).
- Why the family entries lose with most primed variants.
