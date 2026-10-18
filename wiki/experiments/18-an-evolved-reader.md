# 18. An evolved reader: what it learns, and what is built in

A case from the first 3-seed pairs test ([`runs/pairs-test/README.md`](../../runs/pairs-test/README.md),
seed 2, the arm without the three-factor rule). In 10 generations, from a bare genome, evolution
found a network that learns arbitrary glyph-to-token pairs within each lifetime, at 90-97% of a fair
lookup table for 8 and 32 pairs. It does this without the output-learning rule built for the
purpose ([experiment 17](17-the-panel-and-the-learning-rule.md)), and better than that rule did by
hand. It was first mistaken for a scoring leak: the genome at the end of that run only echoes, and
the reader was found only in the rung log. The genome is
[`runs/checks/evolved-reader/pairs8-passer.txt`](../../runs/checks/evolved-reader/pairs8-passer.txt).

## The task

Reading lifetimes of 200 tokens, one shown a step (with gaps of 0 to 6 steps). Each token has a
glyph on the eye, drawn afresh every lifetime, so no genome can know which glyph means which token.
After the first 50 tokens the teacher says six tokens in eight. Each token is heard like the
voice's code for it (the mirror lesson: `mirror=8 quiet=1`), and one's own voice is not heard.
Scored by copy: the token whose glyph was shown 1 to 4 steps before, when the teacher was silent,
minus the same at 9 to 12 steps, per token said. A token said on a taught step counts against it.

## The genome

Two populations. Six declared channels, which nothing uses.

| # | Reads | Writes | What it is |
|---|---|---|---|
| 1 | the ear port | the voice | a wire, op `add` in 16-bit lanes; into an empty voice word this is a copy |
| 4 | all the senses (tag 17 resolves to the senses port) | the voice | 8 starting kernels and births (`grow=1`), tiles of 128 bits, sparse, op `xor`, a heat input from a port (tag 156) |

**Lesion map** (24 fresh worlds, the island's own settings; `ISL_LESION`): with both, copy 734 and
echo 121.
- **Without #1:** copy falls by 734.
- **Without #4:** copy falls by 792 and echo rises by 834.

So the network echoes almost nothing while it reads.

## How it works

1. **The echo (#1).** When the teacher names a token, its sound reaches the ear. Here the sound is
   the voice's code, so the wire puts the right voice code into the voice a step later.
2. **The pairing (#4's births).** A kernel is born where the senses have lines no firing kernel
   reads, which happens when a glyph is new. A birth into a port of the act copies the move being
   made, the busiest word of the voice now. At that step that word is the echo of the teacher naming
   the glyph. So the new kernel sits on the glyph and writes its token's voice code: a one-shot
   pairing, made at first sight.
3. **Silence on taught steps (#4's XOR).** When the glyph is shown again with the teacher speaking,
   the echo writes the code and the kernel XORs the same code in. They cancel, and the network says
   nothing. When the teacher is silent, only the kernel writes, and the network says the token. Copy
   counts exactly those steps and counts taught ones against it, so cancelling the echo is worth as
   much as reading.

## Learning or hard-coding

**Learned within the lifetime:** every glyph-to-token pairing.
- The glyphs are drawn afresh each lifetime, and the blind control scores about 0.
- Score rises with experience: at N = 32, 445 over 200 trials and 912 over 1,600.
- Capacity grows to 128 pairs: 678 over 1,600 trials.

**Built in by evolution (the genome, the same every lifetime):**
- the echo wire;
- the birth policy (copy the move being made onto what is new);
- the XOR that silences taught steps;
- reading all the senses at a 128-bit scale.

**General, not tied to codes.** Nothing in it uses a particular code: it copies a sound, copies a
move and cancels equal codes. So it does not depend on the fixed body. Run under `ISL_BODY` off,
with voice codes drawn each lifetime, it scores identically, cell for cell. With codes of 16 lines
in random words instead of 8 lines in one (`ONE_WORD=0`), it scores within 1%.

**Built in by the world.** It depends on the echo being right, which here comes free because the
mirror lesson makes each sound equal its voice code:

| copy, N pairs, 200 / 1,600 trials | 8 | 32 | 128 |
|---|---|---|---|
| fair lookup table | 966 / 990 | 1016 / 1014 | 1018 / 1020 |
| the reader, mirror lesson (sound = voice code) | 788 / 957 | 445 / 912 | 147 / 678 |
| the body drawn each lifetime | 788 / 957 | 445 / 912 | 147 / 678 |
| codes 16 lines in random words | 788 / 957 | 442 / 911 | 145 / 675 |
| half of each sound's lines from its voice code (`MIRROR=4`) | 788 / 957 | 386 / 765 | 92 / 431 |
| sounds unrelated to voice codes (`MIRROR=0`) | 0 / 0 | -29 / 0 | -2 / 0 |

(`runs/checks/evolved-reader/ablation-*.txt`, 24 worlds each.) With unrelated sounds the wire
copies a sound into the voice that is not the token's code. The echo says the wrong token, so
every birth pairs the glyph with the wrong code.

## What it shows

- **One-shot binding is enough when the target is right at birth.** A pairing that is right the
  first time needs no correction, so the network never needs to learn what it writes. Output
  learning (the three-factor rule) only matters when the first pairing can be wrong.
- **The rule may get in the way.** In the 3-seed tests the reader evolved in one seed of three (and
  again in the rerun), always in the arm without the three-factor rule. Over 6 seeds (`runs/pairs-test-2/`) it was found
  once in 18 runs: 1 of 6 without the rule, 0 of 12 with it, a difference within chance at this
  count; finding it from a bare genome is rare under every arm. A guess: the rule's death removes kernels whose check fails while the target is not
  there, killing pairs that births had made right.
- **The world did half the work.** Reading here is "say what the teacher says, remembered by
  glyph". The hard part of reading, hearing a sound and finding the motor code that makes it, the
  mirror lesson gives away. Without it the same genome scores 0. The real reading rungs (echo2 and
  after, sounds unrelated to voice codes) need a learned listener first, and then this pairing
  trick on top.
- **It cannot generalise.** It is a lookup built by births: a glyph never seen with the teacher is
  never read, and similar glyphs are not read alike. The cap on kernels a population may hold (512)
  bounds how many pairs it can keep.

## Seeded with it ([`runs/pairs-seeded/README.md`](../../runs/pairs-seeded/README.md))

Every genome started as this reader, on a ladder taking the mirror away (pm8, then 6, 4, 2 and 0
eighths of each sound's lines from its voice code, one's own voice heard from pm6 on), 3 arms × 3
seeds, 200 generations. All nine kept it (pm8 at generation 2). Three, one per arm, passed pm6 and
pm4 (generations 46 to 69) and stopped at pm2; none reached pm0. pm6 also turns one's own voice
back on, and the echo wire then repeats the network's own words (the voice loop of experiment 16),
so the step confounds a weaker echo with a voice loop.

## Next

- Rerun the pairs ladders without the mirror (sounds unrelated to voice codes), so the reader must
  sit on top of a learned listener.
- Test the guess about the rule: the same reader genome with `elig` set, with and without death in
  its rule table.
- Score pairs that change partway through a lifetime (remapping), which a one-shot reader cannot
  follow and output learning should.
