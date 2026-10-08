# 89 · Questions as inner speech: holding a gap open and binding its answer

> **Correction (after the first write-up).** The hand-set ask opened its question on the
> story's first unfamiliar word, which was the *season* ("autumn came ."), not the stranger.
> The restatements were "autumn is a jones", not "lucy is a jones", so the 68 → 91% below
> is not binding to the stranger. With the ask aimed at the stranger, restating does
> nothing (68.8%). See [the correction section](#correction-what-the-gain-was) for the
> runs; Findings 1–2 below are withdrawn.

**Question.** [88](88-inner-speech.md) used inner speech to insert detail where recall has
it. The other use is to mark what is *not* known: a question held open until the story
answers it, so that the answer binds to the right thing when it arrives. Does holding a gap
open and saying the answer with it bind a fact that would otherwise pass by? And can the
basal ganglia learn when to ask and what to say, given that the payoff arrives within the
story, so training can show it?

**Task** (`QUESTION=1`, with `FAMILY FAMILY_STATED`). A stranger arrives. A fact about them
comes later, without their name. A known person of the other family is stated too, so the
story holds both surnames and only the binding tells which is the stranger's:

> winter came . kim came . the dog ran away . the person is a smith . it rained . mary is a
> jones . the cat slept . kim went to the *[place by the smith rule]* .

The stranger's family is random in every story, so only this story can tell it. Strangers:
- in training: practice names (half the stories, `QUESTION_P`), or with `QUESTION_POOL=300`
  made-up names met about five times each;
- in test stories that are not held out (the "seen" column): practice names;
- in held-out test stories: new names.

The hippocampus (the engram store, `HIPPO_SELF` with the engram-store entry's settings,
without `SCHEMA_K`) stores events at test as well as in training, since the stranger's fact
is only in the story being read.

**Code** ([`examples/episodic.rs`](../../examples/episodic.rs)): `question_story`,
`QUESTION_ACT`, `QUESTION_CONTROL`, `QUESTION_POOL`. Two acts, at a sentence's end:
- **Ask:** the sentence's least familiar word becomes an open question, held in the loop for
  the rest of the story.
- **Restate:** at a later sentence's end, while a question is open, the network says that
  sentence again with the open item in place of its first k words: "the person is a
  smith ." → "kim is a smith ." The restatement is read as inner steps (marked as its own;
  the cortex does not learn from it), and the hippocampus stores it as an event.

The acts can be set two ways:
- `QUESTION_ACT=1` is a hand-set reference: ask below familiarity band 4; restate a sentence
  that starts with "the", up to its "is".
- `QUESTION_ACT=learned`: the basal ganglia choose both, rewarded at the answer less
  STEP_COST per act:
  - ask or not, per the word's familiarity band and the column's confidence band;
  - k = 0, 1 or 2, per the sentence's first two words.

`QUESTION_CONTROL=1` restates with another stranger's name instead of the open item: the
same repetition of the fact, bound to the wrong person.

## Results

Five seeds, `QUESTION_POOL=300` (held-out = new names; seen = practice names at test):

| Act | Held-out | Per seed | Seen |
|---|---|---|---|
| none | 68.2 | 69.2/65.8/67.6/67.8/70.4 | 24.6 |
| **ask and restate (hand-set)** | **90.8** | 91.4/93.6/93.4/87.2/88.4 | **60.9** |
| restate naming someone else (control) | 29.1 | 28.8/31.6/31.8/24.4/28.8 | 23.2 |

The learned acts, seeds 0 and 1, held-out (what it restated at test):

| Training strangers | Held-out | Restatements at test |
|---|---|---|
| practice names | 54.6 / 45.8 | the stranger's fact always, but with k = 1 ("kim person is a smith"); known people's facts too ("sandra is a jones" → "kim is a jones") |
| fresh pool | 66.2 / 57.8 | the stranger's fact with k = 2 (right) every time, plus distractors and known people's facts |
| fresh pool, reward baseline | 47.8 / 78.2 | erratic: one seed stopped restating the stranger's fact |

## Findings
1. *(Withdrawn, see the correction.)* **Holding a gap open and saying its answer with it binds the fact.** The hand-set act
   lifts held-out answers from 68 to 91% at five seeds. The control shows the gain is the
   binding, not the repetition. The same restatement under someone else's name drops
   answers to 29%: the fact is then bound to the wrong person, and the network uses it.
2. *(Withdrawn, see the correction.)* **It helps most where memory is crowded.** Practice names met in many earlier stories with
   other families go from 25 to 61%: the restated event stands out against old episodes
   that disagree.
3. **The learned act gets part of the way.** With fresh strangers in training, which is the
   case where the act pays off, it learns the right restatement of the stranger's fact.
   With practice names it could not: restating did not help them in training, so there was
   nothing to learn from.
4. **It cannot yet learn what not to say.** It also restates known people's facts under the
   stranger's name, the very wrong binding the control shows is costly. One reward per
   story, shared among about six acts, does not say which act was wrong. The reward
   baseline made this worse.

Nothing becomes a default; the suite is unchanged.

## Correction: what the gain was

A credit rule built next (each restatement's stored event tagged with its act, the act
rewarded if its event is recalled for the answer) found **no** restated event recalled at
any answer. A diagnostic (`QDIAG`) showed the open question was the season word. Two fixes:
- a less familiar word takes over the open question (the most novel item holds it);
- the hand-set reference asks about the stranger itself, an oracle used only to measure the
  act's worth, never as the mechanism.

The subject of the same restatement was then varied (`QUESTION_SUBJECT=season`,
`QUESTION_CONTROL`), at five seeds with `QUESTION_POOL=300`:

| The restatement's subject | Held-out | Per seed | Familiar names |
|---|---|---|---|
| no restatement | 68.2 | 69.2/65.8/67.6/67.8/70.4 | 24.6 |
| the stranger ("lucy is a smith") | **68.8** | 70.6/68.2/67.8/67.2/70.4 | 22.9 |
| the season ("winter came is a smith") | **87.5** | 93.8/90.4/93.6/71.2/88.4 | 61.7 |
| another name | 37.9 | 27.0/30.2/29.2/32.2/70.8 | 23.4 |

1. **Binding the fact to the stranger does not reach the answer.** Its event is stored, but
   at "lucy went to the" the hippocampus recalls it at 2 of 285 answers. The rollout never
   brings "is a smith" back after the name (0 completions in held-out stories). Nothing
   carries the bound fact into the question.
2. **The gain came from placing the season next to the stranger's family.** The answer
   follows a (season, family) rule, and the season is stated only at the story's start. A
   restatement that names the season beside the right family gives the cortex both near the
   question. This is recency in the cortex, not hippocampal binding. It is also an accident
   of the task's rule, not a general mechanism.
3. **Naming another person hurts (38%).** That name's own memories are then recalled with a
   family that conflicts with the stranger's.
4. The learned acts above were trained with the season-subject behaviour, so their figures
   say nothing about binding either.

## Next
- **Make the bound fact reachable.** At the question, the open item ("lucy") should cue
  recall of its restated event and say the fact back ("lucy [is a smith] went to the"):
  recall-planned inner speech ([88](88-inner-speech.md)) cued by the open question. Only
  once that works can a credit rule tell good restatements from bad.
- **Credit closer to the act.** A restatement could be judged when it happens, not at the
  answer. If it contradicts a binding the story already holds ("kim is a jones" right after
  "john is a jones" put jones on john), the hippocampus's mismatch (CA1) is an immediate
  error signal. The answer would then only credit the ask.
- **Asking with a word.** The question held as a token in the loop ("? kim"), heard in the
  heard slot, so the column can learn from it as context.
- The same act for other gaps: an unexplained event ("the dog ran away" → why?), or an
  unknown place.
