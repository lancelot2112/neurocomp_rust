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

**Update: one-word kernels and topographic codes.** Kernels had been given masks spread over
the whole source vector, because the world's codes were scattered (16 of 512 lines, about 2 per
word). Now each code lives in one 64-bit word of its channel (density 1/4 there), and every
kernel reads one word and writes one word. Thresholds are quarters of the kernel's own mask.
- Instrumental, new codes each lifetime: held out **409**/1024 (121/312 paying clicked, 0/288
  others), up from 45–204.
- Reversal: −21, still nothing.
- The residue circuit (02) is unchanged: 6 kernels and zero residue on a cycle, 9 on the branch.
