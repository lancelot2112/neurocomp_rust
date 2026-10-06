# 49 · A sparse binding space: the hippocampus remembers, tags and teaches on its own

**Question.** [48](48-hippocampus-on-its-own.md) found the blocker. Bindings (a word in
a slot) were packed into 8,192 bits by rotation, about 12 bindings per bit. Synapses
could not tell a new binding from old ones, so novelty could not separate events, mark
them, or bring them back in replay. If each binding has inputs of its own, can the
hippocampus run on its own: remember one-shot facts, tag them, and teach the cortex by
its own replay?

**Code.**
- [`src/program/hippocampal_circuit.rs`](../../src/program/hippocampal_circuit.rs):
  `HippocampusConfig::{out_bits, hashed_fan_out}`, `store_split`, `ca1_code`, novelty
  tags (`take_tags`), `replay_from`.
- [`src/program/hippocampus.rs`](../../src/program/hippocampus.rs):
  `DentateGyrus::hashed`, hashed tie-breaking.
- [`examples/episodic.rs`](../../examples/episodic.rs): `SPARSE_BIND`, `REPLAY_TAGGED`,
  `HIPPO_FANOUT`, and the `TAGDIAG` and `SELFDIAG` diagnostics.
- The settings are the `hippocampus-teaches-cortex` entry of
  [`scripts/regress.tsv`](../../scripts/regress.tsv). Default runs are unchanged.

## The design
1. **Sparse binding space (EC layer II/III input).** A binding (word w in slot c) is w's
   32-bit code placed in slot c's own 8,192-bit field. The story's earlier bindings (the
   context) go in a second set of fields. That makes 128 fields, 1,048,576 input bits,
   held as index lists. A bit belongs to essentially one binding, so the perforant path's
   write counts measure each binding's familiarity.
2. **Compact output (EC layer V).** CA1 → subiculum → EC V still writes the 8,192-bit
   rotated code the cortex reads (the slot readout of [36](36-slot-binding-memory.md)).
   Input and output spaces differ, as superficial and deep EC do.
3. **Hashed projections.** The dentate gyrus and EC III → CA1 project each active input
   bit to 300 cells chosen by a fixed integer hash: the same statistics as a random
   fan-in, without a table over a million inputs.
   - Ties are broken by a hash of (input, cell). Breaking them by index gave every event
     the same low-index cells and collapsed all CA3 codes into one attractor. The
     table-based dentate gyrus has the same latent bias.
4. **Novelty-weighted codes.** With hashed projections, each input drives the dentate
   gyrus and CA1 less the more episodes have written it (1 / 2^⌊log2 n⌋, as on the
   perforant path). An event's new content, not its shared context, then sets its CA3
   and CA1 codes.
   - Without this on CA1, recall found the right CA3 event but read out a blend: every
     event's CA1 code was dominated by its common context.
5. **Every word is bound** (`HIPPO_BIND_ALL`). In 48 this hurt, because common words
   crowded the space. Here familiar words are weighted down. It also fixes a gap:
   "lucy is a jones" stored no family when the column happened to predict "jones".
6. **Novelty tags.** An event with at least 16 content inputs no event had written (a new
   binding) is tagged at storage. The tag holds its CA3 code and the new inputs
   themselves.
7. **Tagged replay** (`REPLAY_TAGGED`). At each sleep, every tagged event is reinstated
   from its CA3 code (no settling, which would drift into a common attractor) and read
   out, three times. The tag's new inputs, read as a word, cue the semantic store, and
   the replayed event's other words are the content. Then come 2,400 cue-free random
   replays.
   - Cueing by the least familiar decoded word failed: it picked "is" bound at a rare
     slot.

## Results (seeds 0 / 1 / 2; family stated once; consolidation, closed loop, evidence superposition)

| Memory | Hippocampus at test | Held-out right | Family correct |
|---|---|---|---|
| List memory + harness sentence buffer, semantic store (45) | intact | 69 / 69 / 60% | 99 / 100 / 99% |
| Same, lesioned | lesioned | 67 / 67 / 61% | 100% |
| Full circuit, story episodes, no semantic store (46) | intact | 65 / 67 / 26% | 99 / 100 / 34% |
| Sparse space, surprising words bound, no semantic store | intact | 44 / 41 / 40% | 69 / 68 / 66% |
| Sparse space, every word bound, tagged replay, 4,096 cells | intact | 65 / 39 / 60% | 100 / 35 / 100% |
| Same | **lesioned** | 65 / 38 / 55% | 100 / 35 / 91% |
| **Sparse space, every word bound, tagged replay, 16,384 cells** | intact | **65 / 59 / 59%** | **100 / 67 / 100%** |
| Same | **lesioned** | **64 / 59 / 53%** | **100 / 67 / 87%** |

Per run: 9–15 events tagged in the last sleeps, 6–9 replays cued by a new name, and the
semantic store supplying 540–1,530 rollout steps at test. Cost: 1.4–2.2 ms per word in
training (the event cache no longer helps, since every word changes the cue), and about
100 MB.

## Findings
1. **The hippocampus now runs its whole job itself.** It stores one-shot events, judges
   their novelty, tags the new ones, replays them in sleep, and so teaches the cortex. No
   list store, sentence buffer or word counts are involved.
   - With 16,384 cells, held-out answers are 59–65% intact and 53–64% with the
     hippocampus lesioned. The glued system reached 60–69% and 61–67%.
   - It is close, and every step is now the network's own.
2. **The consolidation is real.** With the hippocampus switched off at test, the family
   is still right 87–100% on two seeds. The semantic store learned it from the
   hippocampus's own tagged replay, a few replays per fact.
3. **Seed 1 still fails partly** (67%, 35% with 4,096 cells): one new member's family
   comes out wrong. Not yet diagnosed.
4. **Five properties were needed together:**
   - inputs a binding owns (the sparse space);
   - tie-breaking that does not bias every code;
   - novelty-weighted dentate gyrus and CA1 codes;
   - tags that remember what was new;
   - reinstating a tagged pattern without letting it drift.

   Missing any one sends recall or replay into the attractor of the common events, the
   failure mode of 45 and 48.
5. **The cortex's surprise is no longer needed as the novelty filter.** Binding every word
   works once familiarity is readable per binding. The hippocampus finds what is new by
   its own counts.

## What is still not the network's own
- **The binding space is assigned, not learned.** It is a word's code in a slot's own
  field. A learned entorhinal layer that develops sparse conjunctive codes is the next
  step.
- **Answer-trace consolidation** ([41](41-consolidating-the-stated-family.md)) still uses
  traces the harness records. It could use the same tagged replay.
- **The defaults still use the list store,** until the self-contained path matches it on
  every seed.

## Biology
- **Entorhinal layer II** gives the dentate gyrus and CA3 sparse, high-dimensional input,
  and its stellate cells form conjunctive codes. The dentate gyrus's expansion separates
  similar inputs (Leutgeb et al. 2007).
- **Novelty and synaptic tagging:** new events are tagged at encoding and captured or
  replayed later (Frey & Morris 1997; Redondo & Morris 2011). Novelty raises dopamine
  and acetylcholine at encoding (Lisman & Grace 2005; Hasselmo 2006).
- **Prioritised replay of new experience** during sleep (Singer & Frank 2009; Mattar &
  Daw 2018), and reactivation of specific ensembles rather than random ones (Wilson &
  McNaughton 1994).
- **Superficial vs deep EC:** input to the hippocampus from layers II/III, output to
  cortex from layer V (Witter et al. 2000). Here the input and output spaces differ in
  the same way.
