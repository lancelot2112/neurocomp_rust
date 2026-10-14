# 09: Channels of thousands to 131,072 bits

**Question.** These are meant to be high-dimensional systems. Can a channel hold 10^4 to 10^5
bits, and what does a step cost then?

**Changes.**
- The tag grammar's channel width, field and spread go up to 2^11 words (131,072 bits)
  (`tags::MAX_WIDTH`). Offsets `at` and `place` are 16-bit; a mutation moves one by one word or
  by a whole field. The edge genome's widest vector is 2048 words (`MAX_WORDS`); windows and
  landing spans can grow that wide.
- A step costs what an edge reads and writes, not the vector's width:
  - an edge reads only its window's words of the frames it spans (no copy of whole frames);
  - wires and noise touch only the words they land on (before, AND/NAND/NOR wires also changed
    words outside their landing span);
  - learning no longer copies the destination; births find uncovered lines word by word and walk
    set bits with `trailing_zeros`.
- The only work left at full width is clearing each vector's new frame (a memset).

**Setup.** The context circuit (tag grammar, experiment 06) with channel Y set to 2^w words.
Reversal, 60 trials, 4 lifetimes per row. `fixed`: the conjunction population Z reads a 64-word
field of Y as in the seed. `whole`: Z's field is all of Y.

    cargo run --release --example wide

| w | bits | fixed ms/life | whole ms/life | whole steps/s |
|---|---|---|---|---|
| 3 | 512 | 5.3 | 5.3 | 113k |
| 6 | 4,096 | 8.9 | 7.0 | 85k |
| 8 | 16,384 | 7.0 | 20.5 | 29k |
| 10 | 65,536 | 7.1 | 40.7 | 15k |
| 11 | 131,072 | 7.1 | 69.4 | 8.6k |

**Meaning.**
- With a fixed field the cost is flat from 512 to 131,072 bits: width is free until something
  reads it.
- With a whole-channel field the cost grows with the field. A profile puts 62% of it in kernel
  matching (`agreement` and the match list). A kernel born across a wide field has a dense mask
  spanning from its first to its last line, so a 16-line kernel over 2048 words scans 2048
  words. A sparse mask (a list of the words it touches) would cost its lines instead. Not done
  yet.
- Discrimination does not change with width here (the circuit's score is near zero on this
  short run at every width); this is a cost check, not a learning result.
