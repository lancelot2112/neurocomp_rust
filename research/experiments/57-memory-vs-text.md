# 57 · How big is a memory, compared with the text? And a dumb text search

> **Correction ([58](58-a-story-is-a-graph.md)):** the text search below also stored each
> test story after answering it, so later questions found earlier test answers. With
> test stories not stored, as for the network, grep + context gets 92% trained but only
> 17% held out (family stated once). The memory sizes are unaffected.

**Question.** Each hippocampus model stores the training stories in some form. How many
bytes does that take, against the text itself? And how well does the dumbest possible
memory do: keep the text and search it?

**Code.**
- `EpisodicCircuit::memory_bytes` for each circuit, and `EpisodicMemory::bytes`. The
  harness prints a `MEMORY` line.
- [`scripts/text_search_baseline.py`](../../scripts/text_search_baseline.py), run on the
  story dump (`DUMP_STORIES=dir`) of the experiment 49 settings.
- Seed 0. The dumped stories are drawn from the same generator as a run's stories, but
  are not the identical sequence.

## The text
3,000 training stories, about 74,600 words:

| Form | Bytes |
|---|---|
| Raw text | 319 KB |
| One byte per word id | 75 KB |
| gzip -9 | 17 KB |

## Results

| Memory | Bytes held | × raw text | Trained right | Held out right | Train µs/word |
|---|---|---|---|---|---|
| Counts circuit, 16,384 cells (49), lesioned | 183 MB | 575× | 69.8% | 64.4% | 2,300 |
| Index memory, bit keys (55), lesioned | 69 MB | 216× | 63.0% | 61.6% | 1,720 |
| Engram store, ids + context ids (56), intact | 15.5 MB | 49× | 89.0% | 60.6% | 390 |
| (the same rows as ids only, without stored outputs) | 1.9 MB | 6× | | | |
| List memory + sentence buffer (45), 3,000 dense episodes | 5.5 MB | 17× | 66.6% | 66.8% | 250 |
| **Text search: grep** (most recent match of the question's longest suffix) | 319 KB text (+3.2 MB n-gram index) | 1× | 25.6% | 26.2% | – |
| **Text search: grep + context** (the match whose story shares the rarest words with this one) | same | 1× | **92.8%** | **67.6%** | – |

## Findings
1. **Every memory built so far stores 6× to 575× more than the text, and the dumbest
   good search beats them all.**
   - Keeping the text and answering with the continuation of the question in the past
     story most like this one (shared words weighted by rarity) gets 92.8% of trained
     names and 67.6% held out.
   - The best numbers here were 89–95% trained (engram or index memory, intact) and
     about 61–67% held out.
2. **Why the text wins on size: it stores each episode once.**
   - Text search keeps a story once and computes the context at query time (which story
     is this occurrence in, and what else does that story contain).
   - The rows built here copy the context into every event: each engram row holds
     9.6 context ids for 4.5 content ids.
   - The circuits go further: dense weights over 16,384-cell populations, mostly empty.
3. **Why it wins on accuracy: the task rewards lookup.**
   - The held-out questions in this setting (`SCHEMA_K=1`) appear once in training, in a
     story with the same season. A good lookup is enough, and the text has it exactly.
   - Plain grep, the most recent match only, is at chance (26%). The story context is
     what does the work.
4. **What a network has to show over this baseline** is composition, not storage:
   - answering about something never seen in that combination (a new name whose family
     was stated once, `SCHEMA_K=0`);
   - generalising a rule, such as season × family → place;
   - doing it with a fixed, small store.

   The cortex's sleep generalisation and the semantic store are the parts that do this;
   the hippocampus models are, on this task, expensive text search.

## The design this points to
- **Store each episode once,** as a sequence of compact content rows (about 4.5 ids per
  event, about 4 bytes each, about 18 bytes per event: 0.3 MB for the training set).
- **Put the episode and time in the row's where-code,** not a copy of the context: a
  few phases in a residue code (periods such as 5, 7, 9 and 11, about 32 bits one-hot)
  and a slowly drifting time context.
- **Compute context at recall** from co-membership: rows in the same episode, nearby
  phases. That is the "grep + context" step, done by the where-index.
- **Make where-codes from learned relations,** not given ids. That is the next step.

## Biology
- **Indexing theory:** the hippocampus stores an index, not the content (Teyler &
  DiScenna 1986). The content stays in cortex.
- **Storage size in the brain is not tiny:** about 10^8 hippocampal neurons and 10^12
  synapses. The question here is what is worth storing, not whether it fits.
