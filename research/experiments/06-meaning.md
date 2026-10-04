# 06 · Meaning: topical similarity and fact binding

**Question.** Does the network capture what words mean, and can it use text to keep
track of facts?

**Code.** [`examples/meaning.rs`](../../examples/meaning.rs). Run:
`cargo run --release --example meaning` (~2 min).

## (A) Word meaning from topical context
Same mechanism as [05b](05-syntax.md), but the context is the *content words* within
±5 tokens (the 60 most frequent words are skipped as topic-free), over the whole Brown
corpus. Test set: 92 words in 12 hand-made semantic groups (numbers, colors, family,
time, weekdays, body, politics, money, religion, music/art, science, food). Metric:
is a word's nearest neighbor among the test words in its own group?

| Setting | Accuracy |
|---|---|
| chance | 7.4% |
| mask 512 bits, 1 strengthen move per occurrence | 25.0% |
| mask 128, 8 moves | 50–52% |
| mask 256, 16 moves | 57.6–58.7% |
| mask 512, 32 moves | 59.8% |
| count vectors + cosine (reference) | 70.7% |

Neighborhoods (best setting, among all 3,000 words):
`mother → wife, friends, father, woman` · `monday → saturday, sunday, afternoon, p.m.` ·
`red → black, white` · `coffee → drink, hot, dinner` ·
`research → available, development, information, program` ·
`eyes → saw, looked, face, hand`. *money* and *music* stay noisy.

**Finding.** One strengthen move per occurrence is too slow for words seen a few
hundred times: their masks barely leave the random start. Several moves per occurrence
(a larger learning rate) is the fix. Settings vary ±3 points between runs.

## (B) Fact binding (synthetic, bAbI-style)
Stories like `mary went to the kitchen . [john went to the garden .] where is mary ? kitchen`.
6 names × 6 places, with one place per name **never paired with that name in
training**. The predictive network reads 3,000 stories with the whole story in its
context window, then is tested on 1,000 more.

| Facts per story | Seen name/place pairs | Held-out pairs |
|---|---|---|
| 1 | 100% | **0%** |
| 2 (one distractor) | 92–95% | **0%** |

On held-out pairs it always answers with *some* place, just never the right one. It
answers with a place it has seen with that name.

**Finding.** The network memorizes *combinations*. It has no way to bind a role filler
("mary") to a value ("kitchen") and retrieve it by content later. That operation is
exactly what attention provides. This is the clearest architectural gap found so far:
[variable binding](../concepts/variable-binding.md). bAbI task 1
([Weston et al. 2015](../related-work.md#binding-and-attention)) is the standard
version of this test.
