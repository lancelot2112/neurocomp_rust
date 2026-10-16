# Island run 14 (stopped at generation 250)

Run 13's settings plus a ninth island, task selection (`ISL_SELECTION=1`): one lifetime of
blocks alternating a conditioning block (click at a light) and a reading block (tokens shown),
scored by `select`; its ladder is cued (the block's task felt on the touch channel), uncued,
then short blocks.

Stopped after generation 250 (1 h 19 min), with the reward rungs and echo2 flat and selection
falling, to move to run 15 (the feel rung).

Stage moves (each passed on held-out worlds), the fastest climbs of any run:
- shapes: clean to repeat between 19 and 36;
- switching: tell at 37, repeat at 59;
- recall: lag1 at 66, lag2 at 70, lag3 at 74, **lag4 at 139: its whole ladder**, the second run to
  climb it (run 9 at 186);
- reading: echo2 at 14, never passed.

Means over the run: selection's cued rung 296 (goal 600; best 475; above 400 from 50 to 140, then
down to 146 by 250, as its best network lost its loops); switching on repeat -7 (193 generations,
best 172), shapes 4 (216, best 113); prediction's cycle1 27; reading's echo2 -7.

Plasmids: of 215 scored, 19 helped (9%), 28 hurt, 12 were mixed and 156 neutral. Two
(#114 shapes to the generalist, #132 selection to the generalist) were each in 27-28 of the
generalist's 32 networks at 180, both whole, then fell to 6 by 200. #202 (switching to reading)
gave echo +253 on arrival.

`checkpoint.txt` is generation 250; `plasmids.txt` records the plasmids up to 240.
