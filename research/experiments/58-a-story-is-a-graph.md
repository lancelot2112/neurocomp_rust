# 58 · A story is a graph: one hop of a walk composes what lookup cannot

**Question.** A story maps a small world: who is who, who went where, when. If memory
were a graph of those relations, could a walk over it answer what text lookup cannot?
And does that need a parser, or does a generic graph of words, sentences and stories do?

**Code.** [`scripts/text_search_baseline.py`](../../scripts/text_search_baseline.py), on
story dumps of the experiment 49 settings with the family stated never (`SCHEMA_K=0`) or
once (`SCHEMA_K=1`) per new name.
- **Correction to [57](57-memory-vs-text.md):** the baseline had also stored each test
  story after answering it, so later questions found earlier test answers. Test stories
  are no longer stored, as the network stores nothing at test. `--keep-test` restores
  the old behaviour.

## The graph (no parsing)
**Nodes:** words, sentences, stories.

**Edges:** word ∈ sentence, sentence ∈ story.

**Methods**, from dumb to graph:
- **grep:** the most recent continuation of the question's longest suffix.
- **grep + context:** the continuation whose story shares the rarest words with this
  story.
- **3-gram + context:** the same, with every continuation of the question's last three
  words ("went to the") as a candidate.
- **Graph walk:** 3-gram + context, after one hop.
  - Each rare word of the question sentence (the name) walks to the sentences that
    contain it elsewhere (word → sentence → word).
  - Their words *replace* it in the query: "tom" → "tom is a smith" → {is, a, smith}.
  - Adding them instead of replacing fails: "tom" itself is so rare that the one story
    stating it wins, and that story's question is about someone else.

## Results (seed 0; 500 trained and 500 held-out questions; chance about 1/6)

| Method | Trained, `SCHEMA_K=1` | Held out, family stated once | Held out, family never stated |
|---|---|---|---|
| grep | 23% | 22% | 9% |
| grep + context | **92%** | 17% | 51% |
| 3-gram + context | 85% | 17% | 51% |
| **Graph walk (3-gram + context + 1 hop)** | 85% | **66%** | 51% |
| *The network, same setting* (49 / 55 / 56) | 63–89% | 61–64% | – |

With the family never stated, the season alone narrows the answer to a family's place in
that season: two candidates, so about 50%.

## Findings
1. **One hop of a generic graph walk does the composition.**
   - "tom is a smith" was stated once. The answer depends on the smiths' place in this
     season, learned from other stories.
   - The walk substitutes the name by what it was stated to be, then matches stories by
     season and family: 66%.
   - Text lookup gets 17%. The best network figures on the same setting are 61–64%.
2. **No parser and no relation labels were needed.** The graph is word ∈ sentence ∈
   story. The relation "is a" is implicit in the hop: a rare word stands for the words it
   occurred with.
3. **Substitution, not accumulation, makes it a walk.** The hop moves the query to new
   nodes. Adding the neighbours to the query keeps the rare word dominant and retrieves
   the wrong story.
4. **This is the hippocampal "big loop".** The output of one recall becomes the cue of
   the next. Experiment [13](13-big-loop.md) tried it with dense codes, and models of
   relational inference such as REMERGE (Kumaran & McClelland 2012) do the same.
   Explicit rows of an index or engram store make each step exact.

## What a graph-building memory needs
- **Nodes for things, rows for events, episodes for stories.** The engram store already
  has the last two: rows as hyperedges over ids, and the place code as episode. Things
  are the ids.
- **A walk operator:** recall → take the recalled row's other ids → replace the cue's id
  → recall again. Stop when the answer slot is filled or the walk returns.
- **Relations as edge types, learned:**
  - the slot pair of two ids in a row (subject slot → object slot) is a relation type the
    role cells already supply;
  - the same relation type across stories is the "metaphor": the same structure mapped
    onto different content.
- **Relations as operators where they compose:**
  - "next sentence" and "next story" are displacements, and their residue phases come
    for free;
  - kinship and "is a" do not commute and stay as a table of edges.

## Next
1. **`recall_walk` in the engram store,** used as the hippocampus's recall in the
   harness (the big loop), on `SCHEMA_K=1`.
2. **Storing each event once,** with the place code for the episode instead of copied
   context (57), so the walk's second step does the context matching.
3. **Edge types from role-cell slot pairs,** and a test of transfer: the same relation
   in a new story world.
