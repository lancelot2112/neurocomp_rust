# 34 · The schema test: can a new fact be learned in one exposure? (not yet)

**Question.** Rats that have learned a schema (a layout of flavour–place pairs) learn a
new pair in one trial, and consolidate it within a day; rats without the schema cannot
(Tse et al. 2007). After [31](31-role-cells-and-transfer.md)–[33](33-generalisation-during-sleep.md)
the network holds general rules beside specific ones. Does that let it learn a new
name's specific fact from one exposure?

**Code.** In [`examples/episodic.rs`](../../examples/episodic.rs): `SCHEMA_K`,
`SCHEMA_PHASE`, `SEASON_RULE=random`, `MEM_CONTEXT`, `new_place`, `season_story_with`.

Run: `… TASK=season SEASON_LEN=4 POLICIES=nomemory HIER=1 HIER_LEVELS=1 MIX=1
HIER_DREAM=1 HIER_SLEEP_GEN=3 SCHEMA_K=1 cargo run --release --example episodic` (other
settings as in 31).

## Design (after Tse et al.)
- **Schema group:** trained on the name rule, where each of the six names has its own
  place per season. The network learns the structure: a season is announced, a name asks,
  the higher area supplies the place, the column copies it.
- **No-schema group** (`SEASON_RULE=random`): the same stories, with random places.
  There is no structure to learn, as with Tse's inconsistent layout.
- **New pairs** (`SCHEMA_K=k`): in the last 600 training stories, three new names (tom,
  lucy, sam) appear exactly k times per season, with their own fixed places.
  - Their mapping (`new_place`) has the opposite parity to every trained name's, so it
    cannot be copied from any of them.
  - k = 0 is the "never seen" baseline.
- **Test:** held-out questions about the new names (seasons at 0–3 filler stories, as in
  31), and questions about the trained names for comparison.

## Results (seeds 0 / 1 / 2; chance about 17%)

| | Trained names | **New names** |
|---|---|---|
| Schema, k = 0 (never seen) | 62 / 56 / 65% | 8 / 6 / 6% |
| Schema, k = 1 | 63 / 68 / 68% | **10 / 7 / 6%** |
| Schema, k = 2 | 46 / 73 / 63% | 5 / 6 / 9% |
| Schema, k = 4 | 60 / 62 / 62% | 12 / 18 / 8% |
| Schema, k = 1, no replay generalisation | 51 / 52 / 70% | 19 / 9 / 9% |
| No schema, k = 1 / 2 / 4 | 13–19% | 7–17% |
| **With episodic memory:** schema, k = 0 / 1 / 4 | 52–74% | 7–12% / 8–10% / 10–13% |
| No schema, k = 1 / 4 | 12–19% | 12–19% |
| **With context-bound episodes** (`MEM_CONTEXT`): schema, k = 0 / 1 / 4 | 55–68% | 2–11% / 5–16% / 8–18% |
| No schema, k = 1 / 4 | 10–19% | 11–18% |

With memory, the recall contained the answer at 47–58% of test answers (schema) and
29–34% (no schema), with or without context binding.

## Findings
1. **No one-exposure learning, with or without a schema.** After 1, 2 or 4 exposures per
   pair, new names stay at 5–18%, no better than never seen (6–8%) or the no-schema group.
   The schema group knows the trained names (56–74%) but does not assimilate new ones.
2. **The cortical route is one-shot but over-specific.** A single exposure does grow a
   kernel (growth is one-shot), but its key is sampled from that story's window, which
   holds the story's own filler words. At test the filler differs, and the kernel does
   not match. Generalisation from replay needs at least three confirmed near-misses, so
   it cannot fix a fact seen once. This is the complementary-learning-systems view:
   cortex learns slowly.
3. **The hippocampal route is not specific to the context.** Recall brings back episodes
   of the same name from any season. Binding the higher area's state into episodes and
   cues did not change that. The state is a bag dominated by incidental filler words, so
   recall matches on fillers rather than on the season.
4. **And the column does not copy from memory in this task.** On trained names, recall
   is right only when the season happens to match, so the column learns that memory is
   unreliable here. A schema would be exactly the learned procedure "copy the place from
   the episode of this name in this context", and it cannot be learned while recall
   ignores the context.

## What a schema needs (from these negatives)
- **A clean context code:** the current setting itself, not a bag of every surprising
  word. That is the recency and role work: a decaying state ([26](26-context-and-readback.md))
  or a role-level code for "the setting" ([31](31-role-cells-and-transfer.md)), so
  episodes are bound to (person, setting) and recall is specific to both.
- **Then the procedure is learnable on trained names:** copy the place from the
  context-matched episode. It is name-independent, so it would apply to a new name after
  one exposure. This is the hippocampal–prefrontal interplay Tse et al. found.
- **Consolidation:** replaying that one episode in the next sleeps would let the cortex
  take it over ([33](33-generalisation-during-sleep.md)), as in Tse's 48-hour
  consolidation.

## Biology
- **Tse et al. 2007:** with a schema, new paired associates are learned in one trial and
  become hippocampus-independent within 48 hours. Without it, neither happens.
- **Context binding:** episodic memory binds items to their spatial and temporal context
  (Howard & Kahana 2002; Eichenbaum's relational memory). The test shows why: without a
  clean context, one-shot memory cannot be retrieved for the right situation.
