# 06. Evolution with the edge and tag grammars, on serial reversal

**Setup.** Serial reversal in the world: the paying light swaps every 100 trials, 400 trials a
lifetime, new codes each lifetime.
- Evolution: population 32, keeping the best 8 each generation (mu+lambda), 40 generations.
- Each genome is scored on 16 lifetimes; the best is checked on 16 held-out lifetimes it was
  never selected on.
- Scores are discrimination, in parts per 1024.
- `examples/evolve_edges.rs` (edge genome), `examples/evolve_tags.rs` (tag grammar).

**Results.**

| Grammar, start | Held out at start | Best, training | Best, held out |
|---|---|---|---|
| edges, act path | 0 | 39 | −3 |
| edges, noise only | 0 | 27 | 10 |
| **edges, context circuit** | 0 | — | **97** (paying clicked 1043/3167, others 753/3233) |
| tags, act path | 32 | 69 | 34 |
| tags, bare (one tonic population; full op set) | 9 | 47 | −2 |

**Meaning.**
- With 16 lifetimes a genome and held-out checks, selection no longer reports noise as gains.
  Training rises a little and held out stays put, except in one run.
- The hand-written context circuit, as a starting point for edge evolution, is the only run with
  a held-out gain (0 to 97). The circuit's pieces (conjunctions, a held context, gated learning)
  give evolution something to tune that a bare start lacks.
- From a bare or act-only start, 40 generations of 32 genomes find no learning. The bare tag run
  did pick up lane arithmetic and a shift wire from the full op set, with no effect yet.
- The first run of each grammar crashed in generation 1 (a window past its source). Every string
  now builds a network that lives, and a genome that still breaks scores worst instead of ending
  the run.

**Next.**
- A curriculum: score instrumental first, reversal later, so a gradient exists before reversal
  is solvable.
- Larger populations and longer runs.
- Start tag evolution from the context circuit written as tokens, the start that worked with
  edges.
