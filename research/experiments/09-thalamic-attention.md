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

### Why runs fail (final channels of every hindsight-proposal run)

| Facts | Run | Held-out | Seen | Final channels |
|---|---|---|---|---|
| 1–2 | 0 | 100% | 100% | (2,2) (0,5) (2,1) **(1,4)** |
| 1–2 | 1 | 100% | 100% | (1,5) **(1,4)** (1,6) (2,1) |
| 1–2 | 2 | **0%** | 100% | (2,6) (1,5) (0,5) **(1,4)** |
| 1–2 | 3 | 100% | 100% | (2,2) (0,2) **(1,4)** (2,1) |
| 1–2 | 4 | **0%** | 20% | (2,2) (1,2) (0,5) (2,1) |
| 1–3 | 0 | 100% | 100% | (2,2) (1,2) **(1,4)** (0,1) |
| 1–3 | 1 | 100% | 100% | (1,5) **(1,4)** (1,6) (2,1) |
| 1–3 | 2 | **0%** | 100% | (2,6) (1,5) (0,5) **(1,4)** |
| 1–3 | 3 | **0%** | 19% | (2,2) (0,2) (1,2) (0,1) |
| 1–3 | 4 | 100% | 100% | (2,3) (0,3) **(1,4)** (2,1) |

Two distinct failure modes:
1. **Route found but used non-generally** (run 2). (1,4) is present and seen pairs are
   perfect, but held-out pairs fail. The answer kernels combine the relayed place with
   bits that also identify the name (most likely the previous word, which is the name
   just before `?`). So the predictor still memorizes pairs, now *with* the relay as
   one ingredient. This is the overspecific-kernel problem from
   [08](08-credit-assignment.md) again: synapse-level credit is the missing piece.
2. **Route never found** (1–2 run 4, 1–3 run 3). No (1,4) at the end; seen pairs at chance.
   Votes go to routes that are right by coincidence often enough to keep them, or the
   route was found and then lost.

Each failure mode has its own fix: (1) needs kernels that drop the inputs they don't
need (e.g. the `generalize` rule, or proposals that also tell growth *which frame*
carried the answer); (2) needs proposals weighted by how *consistently* a route
carries the surprising word, not just how often.

## Follow-up: learning not to depend on the name (synapse-level credit)
Failure mode 1 above, answer kernels that also require the name, can be learned away
without a task-specific rule. The training data itself shows the name is irrelevant: the
same relayed place appears with different names, and the answer is always the relayed
place. The generic rule `GrowthConfig::generalize` says: a kernel that nearly matched
and would have been right drops its silent connections. With `generalize_after = N`
(commit `c25e9de`), a connection is dropped only after N such confirmations, and its
count resets whenever it is active while the kernel is right, like HTM's gradual
permanence. Run: `GENERALIZE=0.5 GENERALIZE_AFTER=3 POLICIES=proposed_only ...`.

| Hindsight proposals + | 1–2 facts: seen | 1–2 facts: held-out | 1–3 facts: seen | 1–3 facts: held-out |
|---|---|---|---|---|
| no generalization | 84.0% | 60.0% (100/100/0/100/0) | 83.8% | 60.0% (100/100/0/0/100) |
| generalize, one-shot (N = 1) | 40.8% | 42.5% (16/66/15/16/99) | 26.3% | 26.9% (18/13/68/16/19) |
| **generalize, N = 3** | **83.5%** | **82.2%** (81/100/66/82/82) | 68.0% | **67.5%** (100/89/49/17/83) |
| generalize, N = 5 | 49.8% | 48.2% (18/100/16/7/100) | 72.5% | 73.0% (100/18/65/100/83) |

### Findings
1. **The network learns not to depend on the name.** With any generalization,
   held-out ≈ seen in every run. Memorizing pairs disappears. Nothing about names, places
   or routes was put in by hand: the rule only says "drop inputs that keep being
   irrelevant when you're right".
2. **One-shot pruning over-generalizes.** With 6 places, a different relayed place gives the
   right answer by coincidence 1 time in 6. After one such event, kernels throw away the
   relay bits too and collapse to "`?` → some place" (most runs at chance).
3. **Requiring repeated evidence fixes most of it.** N = 3 lifts held-out from 60% to
   **82%** (1–2 facts). All five runs hold (1,4) and score ≥ 66%, so no run fails
   outright any more. With up to 3 facts it is 67.5%, with one run still at chance.
4. N trades off over-generalization (too small) against too-slow generalization within
   3,000 stories (too large: N = 5 is bimodal again on 1–2 facts). N = 3 is the best
   setting tried; the right value probably depends on how often coincidences happen
   (here, 1 in 6).

## Next
- ~~Propose routes from surprise~~: done (hindsight proposals).
- ~~Route-aware growth~~: not needed; gradual synapse-level credit learned the same
  thing from the data (follow-up above).
- Consistency-weighted votes: score routes by hits / (hits + misses) when they relay
  something on surprise ticks, so routes that are right by coincidence lose.
- Soft relays: let several matches contribute (summed codes), closer to softmax attention.
- Use the relay on real text ([03](03-book-scale-char-prediction.md),
  [05](05-syntax.md)): induction channels should help wherever text repeats itself.
