# 01 · Stock network on a reading task

**Question.** Before changing anything: can the network as originally written learn
next-character prediction?

**Code.** [`examples/read_text.rs`](../../examples/read_text.rs) at commit `28818d7`
(networks A–C). Run: `cargo run --release --example read_text`.

## Setup
- Corpus: a 263-character toy text (23 distinct characters) repeated 40 times; score
  the last pass plus a 73-character held-out sentence of the same words in a new order.
- Encoding: each character is a fixed random sparse code, 32 of 512 bits
  ([sparse codes](../concepts/sparse-codes-and-collisions.md)).
- The original network has no output that names a character, so we read its hidden
  state with an external **probe** (a vote table `counts[bit][next_char]`, built online).
  The probe is not part of the network.
- Networks: **A** stock `RuntimeNetwork` (empty program, default kernel); **B** 256
  random `SimpleKernel`s input → hidden; **C** B plus a recurrent hidden → hidden
  loop (`GraphProgram::stay()`).

## Results (first run, before any fixes)

| Model | Next-char accuracy |
|---|---|
| most frequent char | 24.4% |
| 1-char n-gram | 61.5% |
| 3-char n-gram | 85.1% |
| 6-char n-gram | 92.0% |
| A: stock | 24.4% (nothing ever fires) |
| B: 256 kernels, threshold 5 | 58–61% |
| C: recurrent (as built) | 58–61% |
| C: recurrent, read-back forced to 1, threshold 3 | 63–67% |

## Findings
1. **No memory of earlier characters.** The hidden layer took 22–23 distinct states,
   about one per character: it re-encoded the current letter and nothing else.
2. **Most kernels idle.** Only 50–140 of 256 kernels ever fired; thresholds were a cliff
   (3 noisy, 5 works, 8 silent).
3. **The stock default can never fire on sparse input**: threshold 16 on a 64-bit
   window that sees ~4 active bits.
4. **Two bugs** turned up ([bugs page](../bugs-and-fixes.md)): the `Stay()` self-loop
   read the frame currently being written, and inhibition was set by output word but
   checked by input word.
5. Firing reinforces firing. Nothing rewards a kernel for predicting what comes next,
   so there is no learning signal tied to the reading task.

## What we changed because of this
Prediction became the learning signal, and kernels are grown where predictions fail.
See [02](02-predictive-growth.md) and
[surprise-driven growth](../concepts/surprise-driven-growth.md).
