# Island run 17 (stopped at generation 250)

Run 16's settings plus shadow explorers (`ISL_SHADOW=1`) and recall's long ladder
(`ISL_RECALL_LONG=1`). A stuck or unlucky population grows a shadow that predicts, a step ahead,
one target after another from what it reads, writing nothing. A kernel is right when the residue
on its lines is under its threshold (`hit_quarters`). The best target is kept as a predictor into a
channel of its own, tagged explored and marked.

Stopped after generation 250 (1 h 11 min), behind run 16 on the feel rung, to move to run 18.

Stage moves (each passed on held-out worlds):
- shapes: clean to feel between 75 and 96 (run 16: 157);
- switching: tell at 90, feel at 135 (run 16: 26);
- recall: lag1 at 169, lag2 at 207, lag3 at 209 (run 16: lag3 at 77);
- reading: echo2 at 12, never passed.

The feel rung: switching 0 (116 generations), shapes 0 (155), the maze 3 (251); run 16 had 67, 46
and 36. Screen's glow averaged 43, selection's cued 233, prediction's cycle1 28.

**Kept predictors bloated islands one after another,** and each did worse while it carried them:
- recall: 264 predictors in its elite at 100 (147 of its own hand) and on its first rung. It shed
  them (84 at 150, 0 at 200) and climbed lag1 to lag3 between 169 and 209.
- switching: 114 at 50, 0 by 100.
- selection: 0 at 150, 228 at 200, its cued score falling from 350 to 182.
- the generalist: 331 at 50, 605 at 250 (99 populations, 391 marked).

A genome keeps one predictor a generation, predictors of steady targets (the hand, the self
channel) score well, and their marks keep them from deletion. The cost (more work, more
populations) is small for each one, so selection removes them slowly.

Plasmids: of 215 scored, 17 helped (8%), 32 hurt, 7 were mixed and 159 neutral. #130 (screen to
reading, 160) gave echo +672 on arrival, the largest echo gain of any plasmid, and was gone
within 20 generations. #202 (switching to reading, 240) gave +320.

`checkpoint.txt` is generation 250; `plasmids.txt` records the plasmids up to 240.
