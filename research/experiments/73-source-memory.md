# 73 · Source memory: what was read versus what the network said

**Question.** If the network's own words (its retellings, [70](70-efference-copy-and-recitation.md))
are stored in the hippocampus with what it reads, does it confuse the two? And can a source
tag on each stored event prevent false memories, as reality monitoring does in people?

**Code.**
- [`src/program/engram.rs`](../../src/program/engram.rs):
  - every row carries a source: 0 the world (read), 1 the network itself (said, retold),
    2 a proposal ([74](74-proposals-and-premises.md));
  - an event merges only with a stored event of the same source;
  - recall can be limited to sources (`set_recall_sources`);
  - rarity counts (postings per id, rows per word) count only the sources recall may
    return;
  - the network's own words are never new facts (no consolidation tags).
- [`src/program/hippocampal_circuit.rs`](../../src/program/hippocampal_circuit.rs):
  `set_source`, `set_recall_sources`, `row_source`.
- Unit test `source_memory_separates_what_was_read_from_what_was_said`.
- [`examples/episodic.rs`](../../examples/episodic.rs): `SELF_STORE=1` stores the
  network's retellings (cortex alone: about 85% of their words wrong); `SOURCE_TAG=1`
  limits recall for reading and answering to world events; `SOURCE` report.

## Results (family stated once; seeds 0 / 1 / 2; cooperation + replay + relation store; hippocampus intact; retelling after each test story)

| Retellings stored | Recall about the world | Held out | Recalls at questions that returned the network's own words |
|---|---|---|---|
| no | – | 86 / 93 / 91% | – |
| yes | untagged | 62 / 58 / 63% | 350 / 471 / 402 of 1000 |
| yes | world only, rarity over all rows | 62 / 63 / 60% | 0 |
| **yes** | **world only, rarity over world rows** | **85 / 93 / 91%** | 0 |

## Findings
1. **Without source tags the network confabulates.** A third to a half of the recalls
   at questions return its own (mostly wrong) retellings, and new-name answers fall from
   about 90 to 61%: false memories built from its own words.
2. **The tag alone is not enough.** With recall limited to world events, answers still
   fell to 62%. The retellings mention the new names, so "lucy" was no longer rare in
   the store, and the walk, which bridges only through rare words, stopped firing. The
   network's own words had changed the store's *statistics*, not its contents.
3. **With rarity counted over the sources recall may use, storing one's own words costs
   nothing** (89.7%, the same as not storing them). Source memory has to apply to every
   use of the store, including what it counts.
4. **In the brain,** reality monitoring (telling imagined from perceived) involves the
   medial and anterior prefrontal cortex and fails in confabulation and some psychoses
   (Johnson & Raye 1981; Simons et al. 2017). Here the tag is given (the network knows
   when it spoke, from the efference copy); deciding it from the content is not modelled.
