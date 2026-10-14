# 03. Raw bytes, a sliding window and two-mask kernels at several scales

**Question.** Can kernels at several scales over a sliding window of raw bytes predict the next
byte, with no codes designed for them?

**Setup.** `examples/bytes.rs`, the first 20,000 bytes of Alice.
- **Window:** `W` is 8192 bits, the last 1024 bytes, newest in byte 0. It shifts by a byte each
  step.
- **Kernels:** two masks, a care mask over the last s bytes (s = 1, 2, 4, ... 1024) and the
  pattern expected there. They fire at q/4 of the cared lines agreeing. The best match goes
  first; overlapping kernels are held down. The winner writes its byte into the prediction byte.
- **Births** (a harness rule): on a wrong or missing prediction, a kernel is born on the current
  window at twice the scale of the kernel that predicted (one byte if none did), writing the byte
  that came.

**Results** (last 2,000 bytes).

| Variant | Right | Kernels |
|---|---|---|
| commonest byte | 17.2% | |
| longest exact context, most frequent next (lookup) | 53.9% | |
| exact kernels (4/4) | **45.0%** | 11,822 |
| exact, held-down kernels learn anti-Hebbian | 34.2% | 13,636 |
| tolerant (3/4) | 9.2% | 17,946 |
| tolerant, with held-down learning | 10.3% | 17,916 |

**Meaning.**
- Exact two-mask kernels with back-off by scale work on raw bytes with no designed codes. The
  gap to the lookup is mostly that a kernel keeps the byte that followed at its birth, while the
  lookup counts what usually follows.
- **Anti-Hebbian must not apply across nested scales.** A longer context that wins holds down
  the shorter kernels inside it. Pushing them off their pattern destroys the back-off they
  provide. Separation is for kernels at the same scale (alternatives).
- **Tolerance is meaningless on raw bytes counted plainly.** ASCII bytes agree on most bits by
  chance, so 3/4 agreement matches almost anything, and the widest kernel wins on raw count.
  Tolerant matching on dense input needs agreement measured against chance, or codes from a
  first layer that are sparse.
- Cost: about 10 s for 20,000 bytes with no index (every kernel checked every step).

## Options tested before committing to a design

**The third mask (confirmation).** Each kernel has three outputs:
- its match;
- its prediction (the byte);
- a confirmation mask, written into a confirmation vector when its prediction came true.

Confirmed, it counts up a `u8`; disconfirmed, down. A wrong winner also gets a sibling at its
own scale for the byte that came. Among equally good matches, the most confirmed wins.
`BYTES_CONFIRM=1`.

**Best match by share.** A kernel's strength is the share of its lines that agree, then its
width, then its confirmations (it had been the plain count of agreeing lines).

| Variant (last 2,000 bytes of 20,000) | Right | Kernels |
|---|---|---|
| exact, no confirmation | 45.0% | 11,822 |
| exact, confirmation checked by the winner only | 45.0% | 23,572 |
| **exact, confirmation checked by every matched kernel** | **50.1%** | 21,726 |
| tolerant 3/4, ranked by share | 15.9% | 17,603 |
| tolerant 3/4, ranked by share, confirmation | 11.2–16.8% | 28–29k |
| longest-context lookup (reference) | 53.9% | |

**Meaning.**
- **Confirmation must be local to every kernel that matched**, held down or not. When only the
  winner is checked, a new sibling never wins, so it is never confirmed and never wins: a
  deadlock (identical accuracy, twice the kernels). With every matched kernel checking its own
  prediction, the most often right sibling rises, and accuracy comes within 4 points of the
  frequency lookup.
- The third mask is a sense of being right, local to each kernel. Here it only breaks ties; it
  could also gate learning (strengthen only the confirmed), or be read by another layer.
- **Tolerance on raw bytes stays broken, even ranked by share.** Chance agreement between ASCII
  bytes is too high for any fraction under 1 to separate them. Tolerant matching needs a sparse
  first layer.

## A sparsifying first layer

**Setup** (`BYTES_SPARSE=1`):
- **Layer 1:** exact 8-bit two-mask kernels on the newest byte, born as each new byte value
  appears. Each writes its own random code of `BYTES_K` lines in a 64-bit slot. These are learned
  codes in the sense that a new kernel is born with a random output, not designed per byte.
- **Sliding window S:** the codes slide through it, one word a byte, 128 bytes deep.
- **Layer 2:** sparse kernels count lines that are on, at 1–128 slots, firing at q/4.

Bugs and confounds found while testing, in order:
1. **The prediction byte was read as the OR of every kernel that fired** (harness bug). In
   tolerant mode, kernels matching mostly older slots overlap the winner by less than half, are
   not held down, and OR their bytes in. Fixed: the winner writes the slot (as CLAUDE.md says).
2. **Hebbian drift.** A kernel firing at 3/4 moves a synapse off a silent line each time, so its
   context wanders. `BYTES_NOHEBB=1` turns that off.

| Variant (last 2,000 of 20,000 bytes, confirmation on) | Right |
|---|---|
| raw window, exact | 50.1% |
| sparse layer, exact (K = 4, 8 or 16 lines a byte) | 50.1% |
| sparse, 3/4, OR-decoded slot (bug) | 37.2% |
| sparse, 3/4, winner writes the slot | 46.8% |
| sparse, 3/4, winner writes, no Hebbian on fire | **49.2%** |
| sparse, 2/4, OR-decoded slot (bug) | 3.0% |
| sparse, 2/4, winner writes the slot | 35.4% |
| sparse, 2/4, winner writes, no Hebbian on fire | 45.8% |

Not the cause (each tested, no change): code density (K = 4, 8, 16); birth scale taken from the
widest exact match instead of the winner; sibling suppression by a fuzzy match; confirming only
exact matches.

**Meaning.**
- The sparse first layer makes tolerant matching possible. On raw bytes 3/4 matching gave 11–16%;
  on sparse codes it gives 49.2%.
- It does not yet help. Fuzzy kernels win about 10 times in 20,000 steps and are rarely right.
  Exact contexts at some scale almost always exist in text this size. Tolerance should matter on
  novel text (held out), not yet tested.
- **Decode from the winner only, never from the OR of what fired.** That is now a rule.
- **Hebbian on firing hurts tolerant kernels that predict.** A kernel firing on a partial match
  pulls its mask toward the partial context. Gating Hebbian by confirmation (strengthen only when
  the prediction came true) is the natural next test.
