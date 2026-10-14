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
