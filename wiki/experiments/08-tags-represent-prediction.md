# 08. Can the tag grammar represent prediction and recursion, and keep them under mutation?

**Representation.**
- Recursion is a channel writing to itself (a sliding window is a test in `tags.rs`).
- Prediction is the residue circuit:
  - a channel R gets the eye copied in (OR);
  - a population reading the eye a frame back XORs its predictions into R;
  - its kernels are born where R is left on, and do not hold each other down.

`examples/representability.rs` writes the circuit in the tag grammar and in the edge genome and
feeds both a cycle of 5 codes for 300 steps. Both cancel it: early residue 96, late residue 0,
identical.

**Robustness.** 300 single random mutants of each circuit, each fed the same cycle; a mutant
"cancels" if its late residue is 0.

| Tag circuit with extra channels | Mutants of the circuit's own tokens: still cancel | Other mutants: still cancel |
|---|---|---|
| 0 | 162/192 (84%) | 78/108 (72%) |
| 8 | 45/55 (82%) | 238/245 (97%) |
| 32 | 12/14 | 284/286 |
| edge genome (all mutants) | 239/300 (80%) | |

**Meaning.**
- The tag grammar represents prediction and recursion, and keeps them as well as the edge genome
  does.
- Nearest-tag resolution absorbs most single tag flips: a flipped bit usually still lands on the
  same channel.
- Its weakness is crowding when channels are few. With no other channel, a newly inserted
  population resolves onto the circuit's channel and writes into it. With more channels, new
  tokens land elsewhere.
- The tag grammar's poor evolution from an empty string (experiment 07) is therefore more likely
  about growth (it starts with no channels, so everything crowds onto the senses and the act)
  than about representation.
