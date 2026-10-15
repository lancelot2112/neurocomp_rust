# Island run 12 (stopped at generation 250)

Run 11's settings plus two-mask kernels (`ISL_ZEROS=1`): a population may expect, so each of its
kernels matches ones and zeros alike on its lines. Kernels are born on half on and half off
lines, expecting what those lines held, and starting kernels expect every line off. The aim was
to let a network sense a click that went unstroked, for the repeat rung.

Stopped after generation 250 (57 minutes), with nothing new on the rungs it was meant to help,
to move to run 13.

Stage moves (each passed on held-out worlds):
- shapes: clean to repeat between 19 and 57 (run 11: 230 to 250; run 9: shift64 by 84).
- reading: echo2 at 13, never passed (run 11 passed it at 337).
- recall: lag1 at 64 (the earliest in any run), lag2 at 169, lag3 at 191.
- switching: tell at 107, repeat at 110.

On the repeat rung, switching averaged -4 (142 generations, best 103) and shapes 0 (195
generations, best 194). Run 11's switching averaged 48. Prediction's cycle1 averaged 14 (run 11:
78), screen's glow 11, the maze's learn 0.

Populations expecting zeros were kept on every island. At 250 they were 60% of the maze's
kernel populations, 50% of screen's and 33% of recall's. On switching they reached 42% at 100,
then fell to 2-14%. Generations took about 15 s, against 8-16 s in run 11, since a kernel
expecting zeros keeps its population running on empty input.

Plasmids: of 191 scored, 24 helped (13%), 29 hurt, 11 were mixed and 127 neutral. #81 (maze to
switching) gave repeat +271 on arrival. #70 (switching to the generalist) spread to 30 of 32
networks, then died out.

`checkpoint.txt` is generation 250; `plasmids.txt` records the plasmids up to 240.
