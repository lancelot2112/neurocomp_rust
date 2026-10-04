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
  paired with that name in training (training stories never contain a held-out pair,
  even as a distractor).
- **Each story has 1 to N facts about distinct people (N = 2 or 3) and asks about a
  random one**, so the answer's position varies. See the shortcut below for why.
- Predictor input: `[current word | 4 relay channels | previous word]`. Unlike 06B, the
  predictor does **not** see the whole story, only one word back.
- 3,000 training stories, then 1,000 test stories with learning off (half
  held-out pairs). 5 seeds.
- Channel policies:
  - **none**;
  - **oracle**: (1,4) + (0,1);
  - **fixed random**;
  - **learned**: ablation credit; the lowest-credit channel is re-pointed to a random
    (q ∈ 0..2, v ∈ 1..6) every 50 stories;
  - **learned + hindsight proposals**: on every wrong prediction the thalamus asks
    which of all 18 routes *would have* relayed the word that actually came
    (`Thalamus::routes_that_would_relay`) and gives them a vote; the re-pointed channel
    takes the most voted unused route instead of a random one.

## Results (answer accuracy on held-out pairs, chance 1/6)

| Channels | 1–2 facts: seen | 1–2 facts: held-out | 1–3 facts: seen | 1–3 facts: held-out |
|---|---|---|---|---|
| none | 18.9% | **0%** | 19.2% | **0%** |
| **oracle (1,4)** | 100% | **100%** | 100% | **100%** |
| fixed random | 19.5% | 0% | 20.2% | 0% |
| learned (ablation credit, random re-pointing) | 52.2% | 40.0% (100/0/0/0/100) | 51.8% | 41.2% (100/0/100/0/6) |
| **learned + hindsight proposals** | **84.0%** | **60.0%** (100/100/0/100/0) | **83.8%** | **60.0%** (100/100/0/0/100) |

Every successful run holds (1, +4) among its final channels.

### A shortcut we caught (and why the setup changed)
The first version used fixed-length stories that always asked about the first fact,
and the story's final "." was never fed to the relay. Route **(0, +6)**, "the word
six after the previous `?`", then hit the answer **by position alone**. A trace showed a
100% run whose channels did not include (1,4). Earlier numbers from that version
(commit `4e4411a`: learned 40% / 17%; grace period 11% / 15%; guided growth 20% / 14%;
proposals 61% / 68%) are confounded by this and should not be compared with the table
above. The oracle and no-relay rows were unaffected (100% / 0% either way).

## Findings
1. **Thalamic relay solves binding.** With the induction route, held-out pairs go
   from 0% to **100%**, with up to two distractor facts. The predictor no longer
   stores (name, place) combinations. It learns "relay says *bedroom* → answer
   *bedroom*", which transfers to every name. This is the first held-out
   generalization in the project, and it comes from routing, not from more memory.
2. **Hindsight proposals make route learning work more often** (40% → 60% held-out,
   seen pairs 52% → 84%). Asking "which route would have carried what I just failed
   to predict?" points straight at the useful route instead of waiting for random
   re-pointing to hit it.
3. **Still all-or-nothing.** Each run either keeps (1,4) and scores 100%, or doesn't
   and scores ~0%. See the failure analysis below.
4. Older variants (credit-guided growth, a grace period for new channels) were worse
   than plain learning in the confounded version. They have not been re-run on the
   fixed task.

FAILURE_ANALYSIS

## Next
- ~~Propose routes from surprise~~: done (hindsight proposals).
- Soft relays: let several matches contribute (summed codes), closer to softmax attention.
- Use the relay on real text ([03](03-book-scale-char-prediction.md),
  [05](05-syntax.md)): induction channels should help wherever text repeats itself.
