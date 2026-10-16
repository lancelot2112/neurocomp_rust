# Island run 16 (600 generations)

Run 15's settings plus ports for gates, pauses and writes (`ISL_PORTGATE=1`): a population's
gate (it learns only while X had lines on, or none) and pause (it runs only while ...) may name an
input port (a sense, or the self channel) or an output port as well as a channel, and a
population writing outward may write a sense instead. Started 02:25Z, finished 05:47Z (3 h 22 min).

Stage moves (each passed on held-out worlds):
- switching: tell at 25, feel at 26 (run 15: 59);
- recall: lag1 at 33 (the earliest of any run), lag2 at 46, lag3 at 77, lag4 (its whole ladder)
  at 353;
- shapes: blur1 at 38, then stuck until 147; blur2 to feel between 147 and 157;
- **prediction: cycle1 passed at 332**, the second run to pass it (run 9 at 415);
- reading: echo2 at 23, never passed.

The feel rung (goal 300): switching averaged 67 over 574 generations (best 99; run 15: 11),
shapes 46 over 443 (73 over the last 100, best 96; run 15: 0), the maze 36 over 600 (69 over the
last 100). Screen's glow averaged 62 (best 163; about 0 in most runs). None passed.

How the port features spread in the elites:
- **port gates first:** 100% of screen's elite by 50, and 53-80% on six islands by 250;
- **port pauses next,** with no help (run 18 gives it): recall 30% at 100, prediction 22%,
  selection 49% at 300 and 100% at 550, screen 50% at 500;
- **writing a sense last:** 5-23% of populations on six islands from 300 (switching, the
  generalist, shapes).

A circuit that reaches the hand from the stroke moved between islands: #248 (switching to shapes,
280) was in all of shapes' elite as shapes' feel rose from 0, and the score kept rising after it
died out.

Plasmids: of 523 scored, 19 helped (4%), 111 hurt, 22 were mixed and 371 neutral. #24 (shapes to
the generalist, 40) and #473 (shapes to reading, 540) each fixed whole on their island for a
time.

`checkpoint.txt` is generation 599; `plasmids.txt` records every plasmid.
