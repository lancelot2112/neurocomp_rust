# 82 · Fact completion: the relation store's binding map as a general kernel class

**Question.** The relation store answered "tom, relation `_ is a _`, from slot 0 to slot 1"
from a store keyed by tom's code rotated by an offset hashed from the relation: a binding
scheme designed by hand. Can a general predictive kernel class, the same as the cortex's,
answer instead by completing the fact ("tom is a _") from the rest of it?

**Code.** [`src/program/relations.rs`](../../src/program/relations.rs), `completion`
(`REL_COMPLETE=1` in the harness):
- **A question is a fact with a blank.** The completion net's input has one frame per word,
  as the column's: the entity (at its position), then the fact's other words nearest the
  blank first (an empty frame for another slot), then the blank's position. It is trained
  by replay on the believed bindings (the Bayes module's verdicts), as the old store was.
- **Kernels start at the entity and grow word by word** into the context until the answer
  is settled ("ed 's father is _" needs "father"; "ed 's mother is _" shares "ed", "is").
- **Answers must read the whole question:** an answer comes only from a kernel whose context
  reaches every word of the question; a kernel keyed on the entity alone knows one of its
  relations, not necessarily this one. When the store has not been told, it says nothing.
- **Facts are remembered in their whole context:** at replay, if no kernel that reads the
  whole question predicts the fact, one is grown.
- [`src/kernel/class.rs`](../../src/kernel/class.rs), two general additions:
  `set_grow_on_ambiguity` (grow one frame deeper when the winner was right but another kernel
  at its depth disagreed: without it, "ed → bob" and "ed → bea", each right half the time,
  stayed tied) and `peek_deep` / `peek_union_deep` (look-ups among kernels reaching a depth).
- Relations (frames) still index the claims, the rules and the questions: what is replaced
  is the binding, not the frame finder.

## The path (relation-store unit tests in completion mode)
| Version | Tests passed (of 9) | What went wrong |
|---|---|---|
| frame finder from predictability alone | 0 | "father" and "mother" alternate in "X 's _ is Y" exactly as "smith" and "jones" do in "X is a _": predictability cannot tell a relation word from a value |
| every pair of positions a template | 1 | hundreds of junk templates and spurious rules |
| completion net over the counted frames, one context frame | 1 | general kernels on "'s father is" fired for everyone |
| entity frame first | 5 | two shallow kernels tied ("ed → bob", "ed → bea") |
| + ambiguity growth | 5 | 16-bit samples of the context missed the one differing word |
| one frame per word | 8 | the blank's mark shared the entity's frame: kernels sampled mostly the mark and matched every entity |
| mark in its own frame | 7 | answers from kernels keyed on the entity alone: "8's grandfather" answered with 8's father, so a wrong great-grandfather was inferred and kept |
| answers must read the whole question, facts kept in whole context | **9** | |

The full test suite passes in both modes (170).

## Results (three seeds)
| Entry | Rotated binding (recorded) | Completion |
|---|---|---|
| relations (67) | 67.4 / 63.5% | **69.1 / 66.4%** |
| speak (69) | 67.5 / 63.5% | **68.8 / 66.5%** |
| motor speech (71) | 67.5 / 63.5% | **68.9 / 66.4%** |
| belief-decides (78) | 63.7 / 81.7% | 63.4 / 80.6% (family questions 99.6–100%) |

Cost: training about 1.8 ms per word against 0.43, answering about 10.8 ms against 3.4; the
store holds about 15,600 kernels against about 80, because every fact is kept in its whole
context.

## Findings
1. **A general kernel class can do the binding.** With the column's layout (one frame per
   word, depth grown as needed), fact completion answers as well as the hand-designed
   rotation binding, a little better on the reading tasks, and says nothing where it was
   not told.
2. **Two general learning rules were missing from the kernel class:** growing deeper where
   same-depth kernels disagree, and answering only from kernels that read the whole
   question. Both are general (off by default elsewhere).
3. **Predictability alone cannot find frames:** a relation word and a value alternate in the
   same way. Relations still come from the counted frames; removing that index (claims,
   rules and topics keyed by the question with its blank) is the next part.
4. **It costs:** four times the training time and three times the answering time, from
   keeping every fact in whole context. Not yet the default.

## Next
- Cut the cost (prune whole-context kernels a shallower one already answers right).
- Key claims, rules and trust topics by the question with its blank, and drop the frames.
