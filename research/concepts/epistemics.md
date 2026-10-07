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

## What is still missing
- Proposals and the network's own claims as sources in the Bayes module, so that their
  trust, too, is earned (and premise credibility comes from belief, not row counts).
- A full posterior (log-odds) instead of a trust-weighted vote.
- Deciding a source from content when the tag is lost (reality monitoring proper).
- Curiosity acting: open proposals steering replay and attention.
