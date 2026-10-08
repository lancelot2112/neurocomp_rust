# Epistemics: what the network believes, and why

Collected from [73](../experiments/73-source-memory.md)–[75](../experiments/75-bayes-module.md).
Nothing is fully trusted: neither the world, nor the network's own words, nor its
inferences.

## Three kinds of origin
| Origin | Example | Where tagged | Used for answering? |
|---|---|---|---|
| The world (read) | "lucy is a jones" on the page | hippocampus rows: source 0; relation-store claims: the narrator | yes, weighted by the narrator's trust |
| The network itself (said) | a retelling, a spoken answer | hippocampus rows: source 1 (known from the efference copy) | no: recall about the world excludes it |
| Its inferences (proposals) | "lucy went to the hallway", in winter | hippocampus rows: source 2 | only once validated |

## Rules
1. **Source memory:** every stored event knows where it came from, and every use of the
   store respects it, including its statistics (rarity). Otherwise one's own words still
   shift what is recalled ([73](../experiments/73-source-memory.md)).
2. **Proposals, not facts:** an inference counts once confirmed by the world, or supported
   by independent premises. Derivations from one premise are not independent
   ([74](../experiments/74-proposals-and-premises.md)).
3. **Trust is earned:** each source's trust is estimated from how its claims fare against
   the others' (the Bayes module, [75](../experiments/75-bayes-module.md)); conflicts are
   resolved by trust-weighted belief, and the losing claims are kept, not deleted.
4. **Open questions are curiosity:** proposals neither validated nor contradicted are what
   to look for next.

## Swapping the belief rule ([76](../experiments/76-belief-rules.md))
Belief is one rule in the Bayes module (`BELIEF=full|vote|graded|posterior`). Trusting
everything (`full`) let a lie heard first become a premise, and eight false inferences
became knowledge (64 → 50% on that seed); any graded rule rejected them. Proposals are now
validated by the module: the belief in their fact times their source's credibility.

## Belief deciding an answer ([78](../experiments/78-belief-decides-the-answer.md))
Asked a new name's family, where an honest narrator and the liar each stated it once,
graded and posterior belief answer 98–100% right; trusting everyone or counting votes
answers 29–68% (every conflict is a tie). Trust learned from other facts breaks the tie.
A source that contradicts itself among single-valued sources is now a conflict, not a
many-valued relation (that rule had stopped trust from being learned at all).

## Saying "unknown" ([79](../experiments/79-unknown-when-belief-is-split.md))
The answer is "unknown" unless the believed value is believed more than half. Only the
posterior uses this well: its "none of these" term leaves every value under one half when
two equally trusted sources disagree, and keeps a trusted source's value over one half
against a distrusted one. Graded belief always puts one side of a two-way conflict at or
over half (it answers the undecidable name, wrong); a vote puts every 1:1 conflict at one
half (it abstains on everything).

## Learning when to answer ([80](../experiments/80-learned-answer-or-unknown.md))
Answer or "unknown" is a basal-ganglia go/no-go, learned from practice quizzes (right: 1,
wrong: 0, "unknown": one half). Its context is how strongly and how decisively the value
is believed: the lead over the runner-up is what tells a conflict trust settles from one
it cannot. With it, graded belief abstains as well as the posterior; full and vote learn
that every conflict looks the same to them and abstain on all of them.

## Curiosity ([81](../experiments/81-curiosity.md))
`Curiosity` holds the open questions and learns, per state of uncertainty, how much a
search there gains (the rise in the answer's lead: the intrinsic reward, as dopamine
signals the value of information). The network can rank questions by it, or pick them by
its own basal-ganglia policy with no ranking rule. Asking a teacher (a source like any
other) settles what "unknown" left open; asking also teaches who errs, which moves trust
beyond the question asked. With practice the learned policy finds the undecided
question on its own; with a cost (energy for compute, refilled each sleep) it chooses when
to stop by itself. Trust is per topic: a narrator can be reliable about places and not
about families.

## What is still missing
- Proposals and the network's own claims as sources with earned trust (premise credibility
  already comes from belief: [76](../experiments/76-belief-rules.md)).
- Inference from every statement of a fact is built ([77](../experiments/77-infer-from-every-statement.md)):
  graded belief then validates only the true inferences.
- The believed fact carried into questions that need it as one step of several (a new
  name's place by its family, [78](../experiments/78-belief-decides-the-answer.md)).
- ~~"Unknown" as an answer when belief is split~~ ([79](../experiments/79-unknown-when-belief-is-split.md));
  next: the threshold learned as a go/no-go, and an "unknown" prompting a search for
  evidence.
- Deciding a source from content when the tag is lost (reality monitoring proper).
- Curiosity acting beyond asking: open questions steering replay and attention
  ([81](../experiments/81-curiosity.md) asks a teacher).
