# 04. A reader of the match vector holding context through a reversal

> **Withdrawn harness.** The reader here was the earlier branch's holder rewritten as plain code
> in the harness (assemblies as line sets, u8 drive counters, a switch rule), not kernels. It
> broke this branch's rules, so `examples/reversal.rs` was deleted. The finding about what the
> match vector carries (confirmations hold context, matches alone do not) stands as a property
> of the codes. The reader is being rebuilt in the graph grammar (experiment 05).

**Question.** Does the match vector, with nested codes (small = matched, big = matched and
prediction came true), carry enough for a reader to hold a context through a reversal?
Especially in the case where the old reader failed: a click between light and outcome.

**Setup.** `examples/reversal.rs`. Light A or B at random, then (`REV_CLICK=1`) a click, then
paid or not. The paying light pays half the time and swaps every 500 trials; 8 phases, 10 seeds.
- **Layer 1:** kernels read the last two steps (two lag slots) and predict the next step, one
  kernel per continuation, with no mutual inhibition. Every kernel that matched writes into M.
- **Codes in M (`REV_CODES`):** `nested` writes the small code and, if right, the big code;
  `match` writes the small code only (no confirmation); `confirm` writes the big code only, when
  right.
- **Layer 2:** the reader, the earlier branch's holder as a harness rule. Context assemblies
  expect lines of M. A rival that explains a step the active one does not gains drive (leak 1
  in 16), and the reader switches at drive = bitlen(explained).

**Results** (swaps 2–7; detected and back are out of 10 per swap).

| Task, codes | Detected | Back | False alarms per swap |
|---|---|---|---|
| light → outcome, nested | 5–10 | 4–7 | 6–242 (thrash in A phases) |
| light → outcome, match only | 5–10 | 3–6 | 4–105 |
| light → outcome, confirm only | 8–10 | 5–10 | 10–66 |
| light → click → outcome, nested | 5–10 | 2–7 | 86–175 |
| light → click → outcome, match only (control) | 2–3 | 0–2 | 32–41 |
| **light → click → outcome, confirm only** | **7–10** | **5–7** | **8–32** |
| (old branch reader on the click sequence) | 0–4 | | |

`REV_RES=8` (a whole code unexplained, instead of half) did not reduce false alarms.

**Meaning.**
- **The confirmed level carries the context.** "Light X, then paid, as predicted" is only
  confirmed in the phase where X pays. A reader of confirmations holds the context through the
  click (7–10/10), where the old reader failed. Without confirmation (match only) it does not
  (2–3/10).
- **The small codes destabilise this reader.** When layer 1 grows a kernel after the reader's
  assembly has settled (for example "B is paid", born in phase B), that kernel's small code
  appears in later phase-A steps. Only the phase-B assembly expects it, so B looks like a better
  explanation of a step that happens in both phases, and the reader thrashes between the two.
  A reader should read the confirmed level (its threshold on all 8 lines) for context. The small
  level, surprise bound to context, is for something else, such as triggering learning.
- **False alarms remain** (8–32 per swap). The reader's assemblies are a union of expected lines,
  learned only while young. A settled assembly cannot absorb codes of kernels born later, which
  is the same stale-knowledge problem in a milder form.
- The reader is a harness rule (the old holder). The next step is to express it as kernels and
  let evolution find or replace it.
