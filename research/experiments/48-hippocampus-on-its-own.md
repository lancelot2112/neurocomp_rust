# 48 · The hippocampus on its own: what works, and the binding space that blocks it

**Question.** After [46](46-full-hippocampus.md) the full circuit answered from memory,
but it still leaned on glue from the experiment harness:
- a second, hand-written list store (`EpisodicMemory`) supplied the familiarity
  statistics;
- episodes were whole stories, so a statement and an unrelated question shared slots
  (seed 2's swapped surnames);
- replay to the cortex came from the harness's sentence buffer, with novelty judged by
  word counts.

Can the circuit supply all of that itself?

**Code.**
- In [`src/program/hippocampal_circuit.rs`](../../src/program/hippocampal_circuit.rs):
  `Hippocampus::familiarity`, `store_event`, `HippocampusConfig::scale_all`.
- In [`src/program/hippocampus.rs`](../../src/program/hippocampus.rs):
  `DentateGyrus::separate_weighted`.
- In [`examples/episodic.rs`](../../examples/episodic.rs): `HIPPO_SELF`,
  `HIPPO_BIND_ALL`, `HIPPO_SCALE_ALL`, `HIPPO_CELLS`, `HIPPO_DECAY`,
  `SEMANTIC_HREPLAYS`, and the `SELFDIAG` diagnostic.
- The defaults are unchanged: every regression entry gives the same answers as before.

## What was built (`HIPPO_SELF=1`, with `HIPPO=full`)
1. **Familiarity from the circuit.** `familiarity(bits)` is the mean number of stored
   episodes that wrote those EC inputs. It reads the perforant path's own write counts,
   the ones its presynaptic scaling already uses. The list store is no longer written or
   read: familiarity bands, the gist of consolidation, and the semantic cue all use the
   circuit.
2. **Events instead of stories.** At each sentence end, one event is stored:
   - the sentence's slot bindings (content);
   - the story's earlier bindings, at a separate rotation (a lateral-EC-like context code).

   The recall cue has the same two parts. The context cues and disambiguates but does
   not answer a slot's readout.
3. **`store_event`: the dentate gyrus separates the content only.** The context is
   associated with the content's CA3 code through the perforant path and the readout.
   Without this, events from different stories with similar filler context got nearly the
   same CA3 code.
4. **Presynaptic scaling on every CA3 pathway** (`scale_all`): recurrent and Schaffer as
   well as perforant. Also an option for **novelty-weighted dentate gyrus input**: an EC
   input's drive on the granule cells shrinks with its write count.
5. **Replay from the circuit** (with `SEMANTIC`): at sleep, `replay()` starts CA3 from
   random cells, lets it settle, and reads the episode out. The episode is decoded through
   the slot cells (the cortex's structure code), and its least familiar word cues the
   semantic store with the rest.
6. **`HIPPO_BIND_ALL`:** bind every word of a sentence, not only the ones the column found
   surprising.

## Results (seeds 0 / 1 / 2; family stated once; no semantic store unless stated)

| | Held-out right | Family correct |
|---|---|---|
| Full circuit, story episodes (46) | 65 / 67 / 26% | 99 / 100 / 34% |
| On its own, events, 4,096 CA3 cells, every word bound | 42 / 50 / 39% | 67 / 69 / 61% |
| + presynaptic scaling on every pathway, 16,384 cells | 39 / 28 / 56% | 36 / 66 / 100% |
| On its own, events, surprising words bound, 4,096 cells | 43 / 22 / 39% | 68 / 32 / 61% |
| **On its own, events, surprising words bound, 16,384 cells** | **43 / 67 / 39%** | **68 / 100 / 61%** |
| Same, with scaling on every pathway | 42 / 50 / 39% | 69 / 69 / 61% |
| Same (16,384 cells), with semantic store fed by the circuit's replay | 43 / 41 / 44% | 74 / 67 / 68% |
| Same, hippocampus lesioned at test | 42 / 40 / 44% | 75 / 67 / 68% |

Cue-free replay: 7,200 replays per run, 5,600–6,500 decoded with content, **0 cued by a new
name** (63 on one seed in one setting). The novelty-weighted dentate gyrus input changed no
result.

## Findings
1. **The circuit can supply its own familiarity**, and the list store is not needed to
   run. What its familiarity can tell apart is the problem (finding 4).
2. **Events fix the case story episodes could not, and lose others.** Seed 2's statement
   is now recalled cleanly (family 100% in one setting, 61% in others, against 34%), but
   seed 0 drops from 99 to 68%.
   - On average events are no better than stories.
3. **Events need capacity.** Storing per sentence puts about 15,500 events into the
   circuit. In 4,096 CA3 cells (32 active per event) the recurrent weights saturate into
   one dominant attractor, and every cue recalls the same blend (about 2,000 bits). With
   16,384 cells recall becomes specific again: the `SELFDIAG` trace shows whole events
   such as "daniel is a smith".
   - Lowering the decay to match the store rate changed nothing.
4. **The binding space is too crowded for the hippocampus to read novelty from it.**
   - **The arithmetic:** a binding is a 32-bit word code rotated into 8,192 bits. About
     60 words in about 50 slots give some 3,000 bindings, so each EC bit belongs to about
     12 different bindings.
   - **Familiarity per bit** (what synapses can count) therefore mixes a novel binding
     ("sam" at the person slot) with common ones that share its bits.
   - **Consequences:**
     - weighting the dentate gyrus input by it changes nothing;
     - CA1's novelty is high for almost every event (mean 0.93–0.94);
     - with every word bound, events are dominated by "is", "a" and "." and get confused
       ("daniel is a smith" recalled for "sam is a smith").
   - The list memory got by only because it averaged over many bits and searched the
     whole list explicitly.
5. **Cue-free replay never visits a one-shot event.** Random CA3 starts settle into the
   strong attractors of events seen hundreds of times. The brain biases replay toward
   new or rewarded episodes by tagging them at encoding (Lisman & Grace 2005; Mattar &
   Daw 2018). Tagging needs a usable novelty signal, which finding 4 removes.
   - So the semantic store learns no new facts from the circuit's own replay. With the
     hippocampus lesioned, nothing changes.
6. **The cortex's surprise is still the best novelty filter.** Binding only the words the
   column found surprising beats binding every word: 68 / 100 / 61% against
   36 / 66 / 100% family correct at 16,384 cells.
   - In this system the column's prediction error is doing the perirhinal job: it marks
     what is new.

## What is still glued
- **Default runs still use the list store and story episodes.** `HIPPO_SELF` is an
  option, and not better yet.
- **The working route into the semantic store is the harness's sentence buffer**
  ([42](42-semantic-store.md)). The circuit's replay does not find new facts.
- **Answer-trace consolidation** ([41](41-consolidating-the-stated-family.md)) still uses
  traces the harness records.

## Next: a sparse binding space
The blocker is structural: content and structure are packed into the same 8,192 bits by
rotation. In the brain the hippocampus receives sparse, high-dimensional conjunctive
codes from entorhinal layer II, so a new item in a known role activates cells that
little else activates.
- **Bindings in their own sparse space:** each (slot, word) pair maps to its own sparse
  code (for example, the word's bits placed in a slot-specific field), held as an index
  list, not a dense vector. A binding's bits are then its own, and per-synapse counts
  measure its familiarity.
- **With that, three things become possible:**
  - novelty-weighted separation and encoding that single out the new binding;
  - tags on novel events, and replay started from them;
  - semantic consolidation from the hippocampus's own replay.
- **The cost:** the CA1 → EC readout must cover the larger space. It can stay lazy and
  sparse, rows allocated only for the bits in use.
- **A learned entorhinal layer** would produce such codes rather than have them assigned.
  That is the larger step after this one.

## Biology
- **Entorhinal layer II** projects sparse, high-dimensional codes to the dentate gyrus
  and CA3. The dentate gyrus's expansion (about 1:5 from EC in rats) exists to make
  similar inputs separable (Treves & Rolls 1992; Leutgeb et al. 2007). Here the input
  space is denser than the brain's.
- **Novelty tagging and prioritised replay:** Lisman & Grace 2005; Singer & Frank 2009;
  Mattar & Daw 2018.
- **Perirhinal familiarity** is computed over item representations in cortex (Brown &
  Aggleton 2001). Here the column's prediction error plays that role.
