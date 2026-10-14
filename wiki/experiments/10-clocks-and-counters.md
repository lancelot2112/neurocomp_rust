# 10: Clocks and counters from the existing ops

**Question.** Tonic populations fire at random (chance 2^-k a step). Do we need a new primitive
for periodic firing ("tick every x steps") or counting ("tick every x times this input is on")?

**Setup** (tag grammar, `examples/clock.rs`):
- **Clock:** channel K, one self-wire with op NOR reading K k frames back. K is cleared each
  step, so K(t) = NOT K(t-k), a square wave of period 2k on the wire's landing word.
- **Counter:** channel C, a self-wire (OR, one frame back) that carries C forward, and a
  population on the senses that ADDs its output mask into C in 8-bit lanes when it fires.
  Writes land set-then-add, so each firing adds the same constant to the carried value.
- **Tonic,** for comparison.

    cargo run --release --example clock     # CLK_LAG=2

**Result** (k = 2, input on two steps of every three):
- K is on for 2 steps and off for 2, exactly, from the start.
- C's lane goes up by 16 each step after the input was on. Line 4 flips on every firing, line 5
  every 2, line 7 every 8, and the lane wraps after 16: a divide-by-2^n of the input count.
- Tonic fires at random, with no period.

**Meaning.**
- Clocks and counters need no new primitive. A clock is one NOR self-wire. A counter is a carry
  self-wire plus an ADD population. A reader of line j of a lane ticks every 2^n firings.
- A clock's period is 2 × its lag, and lags go up to 8 (period 16). Longer periods come from a
  counter fed by a clock.
- Phase reset composes too: an AND-NOT write from an input into K clears it, and the wave
  restarts.
- Both are dense (the NOR fills its whole landing word), unlike the sparse codes elsewhere.
- Not tested: whether evolution finds them. A self-wire needs the `from` and `to` tags to resolve
  to the same channel, and the op must be NOR or ADD.
