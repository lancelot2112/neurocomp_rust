# 75 · The Bayes module: source trust, and conflicts resolved by it

**Question.** Neither the world nor the network itself should be fully trusted. With
facts tagged by who claimed them, can the network learn how trustworthy each source is,
and resolve conflicting facts by it?

**Code.**
- [`src/program/bayes.rs`](../../src/program/bayes.rs), `Bayes`:
  - every piece of knowledge is a claim: a value for a key, by a source;
  - a key one source gave several values ("al's children") is many-valued, not a conflict;
  - trust and belief estimated together (truth discovery, after TruthFinder): a value's
    belief is the summed trust of its sources; a source's trust is the mean share of
    belief its claims win over keys with at least two claims (one win and one loss as a
    prior); eight rounds, integers only;
  - `claim`, `resolve`, `believed`, `trust`, `claims`, `contested`; `use_trust = false` is
    a plain vote.
  - Not yet a full posterior: belief is a trust-weighted vote; a Bayesian posterior would
    add each source's log-odds.
  - Unit tests: a reliable source breaks a tie a vote cannot; one source with several
    values is not a conflict.
- [`src/program/relations.rs`](../../src/program/relations.rs): facts are observed with
  their source (`observe_from`); only the believed value of a conflict is consolidated;
  a verdict that changes with trust is replayed. Unit test
  `trust_resolves_conflicts_between_sources`: three honest narrators each state half of
  40 facts, a fourth states all and lies in 3 of 4; the liar's trust falls below every
  honest one's, at least 90% of checkable facts resolve true, including 1:1 ties (seven
  seeds).
- [`examples/episodic.rs`](../../examples/episodic.rs): `NARRATORS=k` (each training story
  has a known narrator), `LIAR=p` (the last narrator swaps the family in "X is a Y" with
  probability p), `TRUST=vote`, `TRUST` report.

## Results (cooperation + replay + relation store; hippocampus lesioned; new names stated 3×; three narrators)

| Narrators | Conflicts resolved by | Trust (narrators 1, 2, 3) | Held out |
|---|---|---|---|
| honest | trust | 0.94–0.95 each | 61 / 60 / 51% |
| one liar (p = .75; 159–203 lies) | **trust** | seed 0: .67 .67 **.61**; seed 1: .73 .73 **.27**; seed 2: .79 .83 **.43** | 57 / 60 / 51% |
| one liar | vote | .50 each | 57 / 60 / 51% |

Claims about the new names (seed 1, trust): tom: smith by 1 (0.73) against jones by the
liar (0.27); lucy: jones by 1 and 2 (0.84) against smith by the liar (0.16).
With a vote, tom's conflict is a 1:1 tie (0.50 each).

## Findings
1. **The liar's trust is learned from the conflicts alone** on two seeds of three (0.27
   and 0.43 against 0.73–0.83), with no label of who lies. On seed 0 the lies happened to
   fall where few other claims could contradict them (0.61 against 0.67).
2. **Where trust is learned, conflicts resolve to the truth,** including the 1:1 ties a
   vote cannot break (tom: smith 0.73 against jones 0.27).
3. **Answers do not change** (trust and vote: identical figures), and the lies barely hurt
   (honest 57.1%, liar 56.0% mean). In this task the believed family fact is not what
   decides the answer: the rollout and the higher area answer from the stories' patterns,
   and a minority false claim rarely wins anyway. The module works; this task does not
   lean on it.
4. **One mechanism for all sources.** Narrators, the network's own words and its proposals
   can all be sources in the module, each trusted as far as its claims have held up. That
   is the common form of "thoughts count once validated" and "a narrator is believed as
   far as it has been reliable".

## Next
- A task where the believed fact decides the answer (a question about a family member
  with conflicting statements).
- The posterior form: log-odds per source.
- Proposals and self-generated claims as sources in the module ([74](74-proposals-and-premises.md)).
