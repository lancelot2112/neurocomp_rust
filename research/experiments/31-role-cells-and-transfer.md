# 31 · Learned roles and transfer to new names

**Question.** [30](30-cortex-driven-saccades.md) showed that everything the network knows
is keyed on word identities: an unseen wording gave 0%. A schema would let a new filler
slot into known structure, as rats with a schema learn a new flavour–place pair in one
trial (Tse et al. 2007). The role codes for that must be *learned*, not computed by a
rule over word statistics. Two learned mechanisms are tested:
- **Role cells:** an association area that clusters the column's expectations by
  competitive Hebbian plasticity.
- **Synapse-level generalisation** in the higher area: pruning inputs that turned out not
  to matter.

**Code.**
- `RoleArea` and `HigherArea::input_lead` in [`src/program/cortex.rs`](../../src/program/cortex.rs),
  with a unit test.
- `KernelClass::peek_union` in [`src/kernel/class.rs`](../../src/kernel/class.rs).
- In [`examples/episodic.rs`](../../examples/episodic.rs): `ROLE=cells|raw`,
  `SEASON_RULE=season`, `NEW_NAMES`, and the `ROLES` report.

Run: `… TASK=season SEASON_LEN=4 SEASON_RULE=season NEW_NAMES=1 POLICIES=nomemory HIER=1
HIER_LEVELS=1 MIX=1 HIER_GENERALIZE=0.5 cargo run --release --example episodic` (other
settings as in 26).

## The transfer test
- **The season task with short distances** (0–3 filler stories, inside the higher area's
  reach).
- **A role-level rule** (`SEASON_RULE=season`): in a given season, everyone goes to the
  same place. With the default rule each name has its own place per season, so a new
  name's place would be unknowable.
- **New names** (`NEW_NAMES=1`): held-out test questions use names never seen in training
  (tom, lucy, sam). Known-name test questions are the comparison.

## Role cells (`RoleArea`)
- **Input:** the column's expectation for the next slot. That is the union of what its
  matching kernels predict, from a peek without top-down input or state change. It is a
  learned signal: it exists only because those kernels grew from experience.
- **Cells** (64 available): each has a prototype and a fixed output code.
  - The best-matching cell (cosine of bit sets) wins.
  - Its prototype gains the input's synapses with probability 0.1 and loses others with
    probability 0.025.
  - If no cell matches at least 0.5, an unused cell is recruited (adaptive resonance).
- **No labels, no rule over words:** what a cell stands for comes only from plasticity.
- **Wiring:** the winning cell's code (`ROLE=cells`), or the raw expectation
  (`ROLE=raw`), leads the higher area's input: `[role | slow state | sentence]`.

**What the cells came to stand for.** Seed 0 recruited 12 cells; the `ROLES` report
counted, at test, which words filled the slot each cell fired for:

| Cell | Words filling its slot |
|---|---|
| sentence start | the 1,942, it 940, **lucy 182, sam 162, tom 156** |
| noun after "the" | dog 1,161, cat 1,113, bedroom 260, kitchen 254, hallway 244, … |
| end of sentence | . 5,118 |
| fixed continuations | "away" (after "ran"), "ran", "slept", "rained": one cell each |

Seeds 1 and 2 learned the same structure (11–12 cells), plus a cell for verbs and
"came" after a subject ("came", "went", "rained").
- **The new names fall into the sentence-start cell** with "the" and "it", although they
  were never seen in training: their slot, not their identity, decides the cell.
- **Places share a cell with "dog" and "cat":** after "the", the column expects both.

## Results (seeds 0 / 1 / 2)

| Season-only rule | Known names | **New names** | Higher-area kernels |
|---|---|---|---|
| No role frame | 52 / 56 / 64% | 21 / 0 / 32% | 3,400–4,300 |
| Raw expectation frame | 68 / 50 / 54% | 48 / 5 / 22% | 5,100–6,100 |
| Role cells | 58 / 48 / 54% | 40 / 2 / 12% | 4,400–6,200 |
| **Generalisation in the higher area** | 65 / 66 / 71% | **61 / 66 / 72%** | **455–493** |
| Raw expectation + generalisation | 0 / 24 / 0% | 0 / 21 / 0% | 158–224 |
| Role cells + generalisation | 0 / 0 / 0% | 0 / 0 / 0% | 176–252 |

| Name-dependent rule (control) | Known names, test | Higher-area kernels |
|---|---|---|
| No generalisation | 66 / 57 / 73% | 4,000–4,300 |
| Generalisation in the higher area | 23 / 30 / 21% | 480–520 |

## Findings
1. **Role categories emerge from plasticity alone.** Clustering the column's
   expectations gives cells for "sentence start", "noun after the", "end of sentence" and
   fixed continuations. New names land in the right cell on first sight. These are
   learned roles in the sense of [variable binding](../concepts/variable-binding.md),
   though coarse: places and animals share one.
2. **But a role frame next to the identity frames does not give transfer.** The new name
   is surprising, so it also enters the higher area's slow state, and kernels key on
   bits sampled from that frame. A role code added alongside cannot stop the identity bits
   from being keyed (2–48% on new names: 40/2/12 with cells, 48/5/22 raw; noisy).
3. **Learned generalisation does.**
   - When a kernel's prediction is confirmed although some of its inputs were absent,
     it prunes those inputs (the near-miss rule of [12](12-dentate-gyrus-ca3.md)).
   - Under the season-only rule, names vary while the answer does not, so the name bits
     are pruned. New names then score as well as known ones (61–72% against 65–71%),
     with about 8× fewer kernels.
   - The network learned to ignore who filled the person slot. Nothing told it which
     inputs to drop.
4. **The same pruning over-generalises when identity matters.** Under the name-dependent
   rule it falls from 57–73% to 21–30%. It prunes after a single confirmed near-miss, so
   it cannot tell "this input never matters" from "it did not matter this once".
   (This is why it hurt the habit task in [24](24-cortical-hierarchy.md).)
5. **Role frames and generalisation together collapse (0%).** The pruned kernels end up
   keyed on the constant role code alone. Not pursued further.

## What a schema needs from here
The two halves are now visible:
- **General kernels** that ignore fillers: learned by pruning.
- **Specific kernels** that keep identity where it matters.

Pruning in place destroys the specific kernel. The next step is to keep both:
- **Generalise by spawning,** not by modifying. When a near-miss is confirmed, grow a
  more general copy without the unused inputs, and keep the original.
- **Let reliability arbitrate,** as everywhere else in L2/3. The general kernel wins where
  identity never matters, and is out-scored by the specific one where it does.

That is a schema with exceptions, learned from data. Retest on both rules: the target is
new names at known-name accuracy under the season rule, with no loss under the name rule.

## Biology
- **Category cells from competitive learning:** sparse codes from Hebbian competition
  with lateral inhibition (Földiák 1990); adaptive resonance (Carpenter & Grossberg 1987)
  for recruiting new categories only for new kinds of input.
- **Schemas speed learning:** Tse et al. 2007. **Generalisation and specifics side by side:**
  complementary learning systems keep general structure in cortex and specifics in the
  hippocampus (McClelland, McNaughton & O'Reilly 1995). Here both would live in L2/3,
  arbitrated by reliability.
