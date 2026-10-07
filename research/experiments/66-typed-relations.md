# 66 · Typed relations: role- and frame-bound facts, navigable by relation code

**Question.** The semantic store ([42](42-semantic-store.md)) maps a word to a bag: "lucy"
→ {is, a, jones}. It has no roles, no direction and no relation type, and two facts about
one word merge. Can facts be stored as typed relations, learned from plain sentences, so
that a name and a relation give the filler ("tom" + father → "bob"), and relations chain
(tom's father's father)?

**Code.**
1. **In the harness's semantic store** ([`examples/episodic.rs`](../../examples/episodic.rs)):
   - `SEM_TYPED=roles`: each word of a fact is bound to its learned role (its code rotated
     by the role's offset, as in the hippocampus's slot bindings): "lucy" → {is@1, a@2,
     jones@3}. Word readers unbind every role; the higher area's context gets the
     role-bound content.
   - `SEM_TYPED=dir`: the cue is bound to its role too (subject ≠ object).
   - `SEM_FRAME=1`: the binding is keyed on the fact's frame too (the bindings at least
     half as familiar as its most familiar one), so relations of one word that fill the
     same role do not collide. The `FRAMES` report lists the frames learned.
2. **A general relation store:** [`src/program/relations.rs`](../../src/program/relations.rs),
   `RelationStore`.
   - **Relations are learned from word counts.** A fact's frame is the words above its
     largest frequency gap (at least `frame_ratio`, leaving at least two words below):
     "'s father is" recurs in every fact of the kind, the names only in facts about
     them. The words below, in reading order, are its fillers. A relation is a frame (a
     set of words, any order); its code is its index.
   - **Binding is a permutation.** For fillers i ≠ j of a fact of relation r: the key is
     filler i's code rotated by an offset hashed from (r, i, j), the value is filler j's
     code. "tom 's father is bob" stores tom ↦(father, 0→1) bob and bob ↦(father, 1→0) tom.
   - **The store is a predictive kernel class trained by replay** (`observe` while reading,
     `consolidate` in sleep), like the semantic store.
   - **Navigation:** `relation_for(words)` (the frame sharing most of the query's words),
     `ask(entity, r, from, to)`, `follow(entity, path)`, `about(entity)`.
   - Tests (family tree of 21 words, 22 parent facts and 3 distractors; seeds 0, 1, 2, 3,
     5, 9 all pass):
     - every child's father and mother, with no collision between the two;
     - the inverse (whose father is al → bob or cy);
     - chains: hal → father → father = cy, → father ×3 = al, → mother → father = bob;
       a missing link gives nothing.

## Results in the reading task (family stated once; seeds 0 / 1 / 2; hippocampus lesioned)

| Semantic store | Reader | Held out | Trained |
|---|---|---|---|
| bag (60) | rollout | 60 / 64 / 56% | 60.5% |
| **roles** | rollout | **68 / 68 / 55%** | 64.4% |
| roles + frame | rollout | 61 / 67 / 61% | 61.9% |
| roles + direction | rollout | 43 / 46 / 40% | 63.9% |
| bag (63) | cooperation + replay | 56 / 62 / 60% | 58.1% |
| roles | cooperation + replay | 43 / 11 / 26% | 63.3% |

## Findings
1. **Binding the content to roles helps the rollout:** new names 60 → 63.7%, trained
   names 60.5 → 64.4%. Unbinding by role gives the rollout cleaner words than the bag.
2. **Frames keep the gain and even out the seeds** (61 / 67 / 61%), but the harness's
   frame rule (half the most familiar binding) is crude: the FRAMES report shows names
   inside frames ("is@19 anna@0 a@20") and 44–55 frames for what are a handful of
   sentence kinds. The relation store's frequency-gap rule is the better parse; it is not
   wired into the reading task yet.
3. **Direction hurts here (43%).** The name is in a different role in the question
   ("where did lucy go") than in the fact ("lucy is a jones"), so the directed key is
   never hit. A directed relation needs a query that names the relation, as the relation
   store's does, not the role the word happens to have now.
4. **Role-bound bits are no use to a reader that does not unbind.** Given to the higher
   area as context, they collapse cooperation (59.5 → 27%). Rotated codes share no bits
   with the words the area learned, so nothing it knows transfers. Typed content must be
   read out (unbound) before it is used as context.
5. **The relation store does what the question asked in a world built for it:** a name
   and a relation give the filler, relations do not collide, and they chain. The relation
   code is learned (a frame from word statistics), the binding is a permutation, and the
   storage is the same kernel class as the rest of the cortex.

## Biology and related work
- Role–filler binding by permutation is the binding of vector-symbolic architectures
  (Plate's holographic reduced representations, Kanerva's hyperdimensional computing).
- Frames as the frequent, shared part of statements are schemas (Tse et al. 2007;
  [concepts](../concepts/)): what is constant across instances, against the fillers that
  vary.
- Following a relation chain is the "relational memory" role of the hippocampal–entorhinal
  system (Whittington et al., TEM 2020); here it is done by a cortical store after
  consolidation.

## Next
- **Wire the relation store into reading:** facts parsed by frequency gap at sleep; at a
  question, the cortex asks the relation named by the question's words for the sentence's
  new word, and the answer joins the higher area's context as plain words.
- **Relations of relations:** a path learned as a relation of its own ("grandfather" =
  father ∘ father) by replaying chains.
