# 81 · Curiosity: an "unknown" that prompts a search

**Question.** The network now says "unknown" where its belief is split
([79](79-unknown-when-belief-is-split.md), [80](80-learned-answer-or-unknown.md)), but
nothing follows. A curious agent would look for the evidence that settles the question.
Can a curiosity module pick which open questions are worth a search, and can the network
learn that choice itself, rewarded only by the information a search brings?

**Code.**
- [`src/program/curiosity.rs`](../../src/program/curiosity.rs), `Curiosity<K>` (a base
  module, generic over the key):
  - open questions, each with a context (its state) and its uncertainty;
  - per context, the information searches there actually gained (the rise in the answer's
    lead over the runner-up): the intrinsic reward;
  - `pick`: a fixed ranking, the expected gain of each question, its own uncertainty
    (one minus its lead) as one search's worth of prior, corrected by the gains realised
    in its context;
  - `pick_learned`: **the network's own policy.** The open questions are candidates to a
    basal-ganglia selector, each coded by its state (its belief band's bits joined with its
    lead band's, so what is learned about one state carries to its neighbours); the
    selector releases a question, with exploration, and the information gained is its
    reward. No ranking rule, no prior.
  - Unit tests: searches go where they have paid off; the learned policy, after practice,
    asks in the paying state 40+ times in 50.
- [`examples/episodic.rs`](../../examples/episodic.rs): `ASK=n` (with `NARRATORS`): at each
  sleep, once the facts are weighed, the network may ask a teacher n questions ("who is
  sam?"). The teacher answers truthfully but is a source of its own, trusted only as far as
  its claims hold up; its answers are weighed in a second pass, and the gain each brought is
  measured. `ASK_POLICY=curious` (the fixed ranking), `random`, or `learned`. `CURIOSITY`
  report: questions by kind, the new names asked and when, and per context the gain and
  the policy's learned value.

## Results (as 80, posterior belief, learned answer-or-unknown; 3 questions per sleep, 6 sleeps; seeds 0 / 1 / 2)

18 questions per run, among 27–30 open questions (practice names, trained names, the new
names from the last sleep on).

| Policy | New names asked | Sam answered (right / unknown) | Tom and lucy right |
|---|---|---|---|
| none | – | 0 / 100% every seed | 90–100% |
| random | none, any seed | 0 / 100% every seed | 97–100% |
| fixed ranking | seed 2: sam, tom, lucy | seed 1: 96% right (not asked: see below); seed 2: **100% right** | seed 0: 72–100%; seeds 1 and 2: **0% (all "unknown")** |
| learned | seed 0: tom, lucy; seed 2: sam, tom | seed 2: **100% right**; seeds 0, 1: unknown | **91–100% every seed** |

What the learned policy valued after 18 questions (seed 2): the least decided states
(belief 3/8, lead 0–4/16: gains 0.37–0.50) at 0.50; settled states (belief 5/8, lead 7/16:
gain 0.28) at 0.45. Random asking spent 5–8 of its 18 questions on trained names whose
answers were already settled (gain 0.00–0.11).

## Findings
1. **A search settles what "unknown" left open.** Where sam was asked (fixed ranking or
   learned, seed 2), his family was answered right on every question (100%), where every
   other run said "unknown".
2. **Random asking never reached the new names:** with 27–30 open questions and 3 a sleep,
   it spent its budget on practice and settled trained names.
3. **The learned policy points the right way but has had too little experience:** 18
   questions move its values from 0.50 only slightly (settled states 0.40–0.45 against
   0.45–0.50 for undecided ones), so its choices are still mostly exploration. It asked
   about a new name on two seeds of three, and sam on one.
4. **Asking changes more than the question asked.** The teacher's answers about practice
   names also move the narrators' trust (they reveal which narrator erred). That once
   settled sam without asking about him (fixed ranking, seed 1). But it also moved tom and
   lucy into states the answer-or-unknown go/no-go had never practised (belief 6/8, lead
   7/16, after being asked about), and an unpractised state defaults to "unknown": the
   fixed ranking lost tom and lucy on two seeds that way. The learned policy kept them
   (91–100%).
5. **Order matters: ask after weighing.** The first version chose its questions before the
   sleep's new facts were weighed, so the new names' conflicts were never known in time to
   be asked about. It now asks after consolidation and weighs the answers in a second pass.
6. **A confound removed on the way:** when the same honest narrator always erred in the
   practice conflicts, correcting practice names taught distrust of him, which settled sam
   indirectly. The erring narrator now alternates.

## Next
- More experience for the learned policy: let it ask during practice as well (each quiz
  round a chance to ask), so its values separate before the new names arrive.
- The go/no-go should generalise to unpractised states (codes that share bits across
  neighbouring bands, as the asking policy's do) instead of defaulting to "unknown".
- Other searches than asking: replay priority and attention while reading for open
  questions; and a cost per question, so the budget itself is learned.
