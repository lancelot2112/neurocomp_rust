# 01. First evolution of graph programs in the world

**Question.** Can the base graph language, evolved in the world, find anything?

**Setup.** A genome is a `GraphProgram` plus each edge's kernel numbers:
- kernels per edge, synapses, threshold, output lines and op;
- a wiring seed.

Kernels into the act node write the body's motor primitives. Fitness is the observer's
discrimination in conditioning (click rate on paying trials minus the others, parts per 1024),
over 3–4 world seeds. The search keeps the best 8 of 48 each generation and fills up with
mutated copies (mu+lambda). `examples/evolve_world.rs`.

**Results** (held out = 5 world seeds evolution never saw).

| Task | World codes | Held-out discrimination | What it found |
|---|---|---|---|
| Instrumental | tied to the training seeds (first run) | 0 (682 on training) | memorised the codes of its training worlds |
| Instrumental | fixed across lifetimes | 362 | a wired-in reflex, clicking on most trials |
| Instrumental | new each lifetime | 204 (45 after the edge instructions were added) | "click when anything is seen" |
| Reversal | fixed | 208 | a bias toward clicking |
| Reversal | new each lifetime | −14, −36 | nothing |

**Meaning.**
- The world's physics (its codes) must be separated from its draws, and varied across
  lifetimes. Otherwise evolution memorises codes rather than evolving learning.
- The best programs stayed one or two instructions; only kernel numbers evolved. Learning that
  needs structure (reversal) was not found, before or after the edge instructions.
