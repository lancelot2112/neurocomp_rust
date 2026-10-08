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

## Addendum: asking during practice, learning its own policy, and paying for it

**Code (since the first runs).**
- `ASK_ROUNDS=r`: asking happens in r rounds per sleep, each followed by a share of the
  practice quiz, so the learned policy sees each round's gains before choosing the next
  round's questions.
- `ASK_POLICY=cost`, `Curiosity::decide`: **the network balances cost itself.** Each
  question uses energy, the compute its answer took to weigh (relation-store replays,
  `COST_UNIT` replays to a full reserve), from a reserve refilled by `POWER` (0.5) each
  sleep. At each step the basal-ganglia selector chooses a question or *stop*; every
  choice is coded with the reserve's band and how far into the sleep it is. Stop is worth
  one half; asking one half plus half the information gained minus the energy spent,
  weighed up to twice as heavily as the reserve empties. Nothing says when to stop. Unit
  test: at a cost of 0.25 it learns to ask where a search gains 0.8 and to stop rather than
  ask where it gains nothing.
- Two fixes the runs forced (both in [`bayes.rs`](../../src/program/bayes.rs) and the
  harness):
  - **Trust per source per topic** (each relation a topic; default). Once the frame fix
    let "mary smith went to the kitchen" parse with the family as a filler, every story
    brought true family claims, and a liar who lies only in "X is a Y" statements earned
    0.60 overall, close to the honest narrators: tom's and lucy's conflicts looked as
    undecided as sam's, and the network said "unknown" to all of them. Per topic, its trust
    on families is 0.48 again (honest 0.59 and 0.63). Unit test: a source honest about one
    topic and lying about another loses trust on the second only, and a 1:1 conflict there
    is settled more decisively than with one trust per source.
  - **The go/no-go keeps one code per state.** Codes shared across belief and lead bands
    (so unpractised states would borrow from neighbours) leaked "answer" from tom's state
    into sam's, which shares its belief band: sam answered wrong on two seeds of three.
    Backed out (`GONOGO_CODES=shared` keeps it as an option).

**Results** (posterior belief, learned answer-or-unknown, trust per topic; 2 questions a
round, 4 rounds a sleep; seeds 0 / 1 / 2). Family questions right (sam: right or
"unknown"):

| Policy | Questions asked | Sam asked | Sam | Tom and lucy |
|---|---|---|---|---|
| none | 0 | – | "unknown", every seed | 91–100% |
| random | 48 | 1 seed of 3 | right on that seed (89%), else "unknown" | lucy "unknown" on seed 1 |
| fixed ranking | 48 | **every seed** | **96–100% right** | 94–100% |
| learned policy | 48 | **every seed** | **99–100% right** | seed 1 lucy, seed 2 both "unknown" |
| cost, cheap (0.06 a question) | 33–40 (stops 4–8) | every seed | 93–100% right | seed 2 both "unknown" |
| cost, 0.25 a question | 12–14 (stops 13–17) | never | "unknown" | 72–100% |
| cost, 0.5 a question | 6–7 (stops 17–18) | never | "unknown" | seed 1 both "unknown" |

**Findings.**
1. **With practice the learned policy finds the undecided question.** It asked about sam on
   every seed and settled him (99–100% right), as the fixed ranking does; random asking
   reached him on one seed in three. Its learned values separate (48 questions): asking in
   undecided states 0.47–0.50, in settled ones 0.14–0.21.
2. **The network balances cost by itself:** cheap questions, it asks 33–40 times and stops
   4–8 times; four times dearer, 12–14 questions and 13–17 stops; eight times, 6–7. Energy
   runs out (0.00 left) where questions are dear.
3. **It does not yet save for later.** At 0.25 a question it spends its reserve in the early
   sleeps on practice and trained names and never reaches sam, who appears only at the last
   sleep; it then says "unknown" about him, which is right for what it knows. Stop is worth
   a flat one half and the reserve is capped, so nothing rewards keeping energy for a
   question that has not arrived yet.
4. **An asked name can land in a state the go/no-go never practised** (high belief and lead
   after the teacher's answer) and default to "unknown": tom and lucy on some seeds under
   the learned policy and cheap cost. Generalising by shared codes leaked instead (above);
   the fix is still open.
5. **Trust has topics.** A source's reliability is not one number: the same narrator can be
   right about places and wrong about families, and the module now learns that.

## Next
- ~~More experience for the learned policy~~ (asking during practice: addendum).
- A go/no-go that generalises to unpractised states without leaking between neighbours
  (shared codes leaked: addendum).
- Saving energy for later: a value for the reserve itself, or a sense of what is still
  to come, so the cost policy does not spend everything early.
- Other searches than asking: replay priority and attention while reading for open
  questions; and a cost per question, so the budget itself is learned.
