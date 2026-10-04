# 09 · Thalamus-like relay as attention

**Question.** Can a thalamus-like relay give the network the variable binding it
lacked in [06B](06-meaning.md) (0% on held-out name/place pairs), and can the network
learn which relay routes to use?

**Code.** [`src/program/thalamus.rs`](../../src/program/thalamus.rs) (`Thalamus`,
`RelayChannel`); experiment [`examples/thalamus.rs`](../../examples/thalamus.rs)
(commit `4e4411a`). Run: `cargo run --release --example thalamus` (~15 min);
`POLICIES=patient` runs only the learned variants.

## The mechanism
The relay keeps a short history (40 frames) of what a cortical node held. Each
**relay channel** is a routing rule `(q, v)`:
1. **query**: the frame `q` steps back from now;
2. **key match**: the most recent earlier frame overlapping the query by ≥ 80% of its
   bits (content addressing);
3. **value**: the frame `v` steps after that match, relayed into the channel's output frame.

This is hard attention with one fixed query/key/value pattern per channel, which is
what an induction head does ([Olsson et al. 2022](../related-work.md#binding-and-attention)).
In thalamic terms:
- channels are higher-order relay routes between cortical areas (pulvinar, mediodorsal);
- the small number of channels stands in for reticular-nucleus gating (Crick's
  searchlight);
- choosing channels by credit plays the role of task-dependent gating of thalamic relay
  ([related work](../related-work.md#thalamus-and-attention)).

For `mary went to the kitchen . where is mary ? → kitchen`, the right channel at `?`
is **(q = 1, v = +4)**: query *mary* one step back, find the earlier *mary*, relay
the word four steps later (*kitchen*).

## Setup
- Stories as in [06B](06-meaning.md): 6 names × 6 places, one place per name never
  paired with that name in training. 1-fact stories, and 2-fact stories where the
  second fact is a distractor.
- Predictor input: `[current word | relay channels | previous word]`. Unlike 06B, the
  predictor does **not** see the whole story, only one word back.
- 3,000 training stories, then 1,000 test stories with learning off (half
  held-out pairs). 5 seeds.
- Channel policies:
  - **none**;
  - **oracle**: (1,4) + (0,1);
  - **fixed random**;
  - **learned**: ablation credit, the lowest-credit channel re-pointed to a random
    (q ∈ 0..2, v ∈ 1..6) every 50 stories;
  - **learned + guided growth**;
  - **learned + grace period**.

## Results (answer accuracy, chance 1/6)

| Channels | 1 fact: seen | 1 fact: held-out | 2 facts: seen | 2 facts: held-out |
|---|---|---|---|---|
| none | 20.0% | **0%** | 20.4% | **0%** |
| **oracle (1,4)** | 100% | **100%** | 100% | **100%** |
| fixed random (4) | 19.1% | 0% | 18.9% | 0% |
| learned, 2 channels | 51.6% | 13.5% (runs 0/50/17/0/0) | 36.5% | 20.0% (100/0/0/0/0) |
| learned, 4 channels | 68.4% | 40.0% (0/100/100/0/0) | 30.7% | 17.2% (0/86/0/0/0) |
| learned + guided growth | 23.8% | 20.0% | 18.6% | 13.8% |
| learned + grace period | PATIENT_1_SEEN | PATIENT_1_HELD | PATIENT_2_SEEN | PATIENT_2_HELD |

## Findings
1. **Thalamic relay solves binding.** With the induction channel, held-out pairs go
   from 0% to **100%**, with or without a distractor. The predictor no longer stores
   (name, place) combinations. It learns "relay says *bedroom* → answer *bedroom*",
   which transfers to every name. This is the first held-out generalization in the
   project, and it comes from routing, not from more memory.
2. **Learning the route is the hard part.** With ablation credit, a run either finds
   (1,4) and reaches ~100%, or doesn't and stays near 0%. Bimodal, like the
   memory-unit task in [08](08-credit-assignment.md).
3. **Credit-guided growth hurt here** (opposite of 08). Restricting new kernels to
   the current word plus one channel also stopped them using the previous word, which
   ordinary next-word prediction needs. That degraded the predictor and with it the
   credit signal. Guidance has to be applied where it matters (e.g. only on surprising
   ticks), not everywhere.
4. Distractors make learning harder (17% vs 40% held-out with 4 learned channels)
   but don't affect the oracle at all. The relay picks the right fact by content.

## Next
- Make the route search smarter than uniform random re-pointing: propose channels
  from where the surprise was (e.g. the query lag at which a matching word occurred).
- Soft relays: let several matches contribute (summed codes), closer to softmax attention.
- Use the relay on real text ([03](03-book-scale-char-prediction.md),
  [05](05-syntax.md)): induction channels should help wherever text repeats itself.
