# Island run 38

Run 37's settings with selection scored as the tasks' performance contrasted
(`ISL_SELECT_CONTRAST=2`), misses costing (`ISL_MISS_COST=1`: a prompt with no act a wrong answer)
and stacked rungs on switching, shapes and the maze (`ISL_STACKED=1`). Started 3:45 PM ET
2026-10-19, stopped by hand at generation 112 (4:57 PM ET) for run 39.

9 rung moves. The miss cost ended doing nothing as a refuge: switching and recall passed respond at
77 (run 37: 112 and 147). Selection climbed cued to held in 22 generations and sits at about 0 on
held. But the full cost saturated: recall on flag2 (-1024 to -780) and the screen (-1024) sat at
the floor, where doing nothing and clicking at random score alike. Run 39 halves the miss cost and
adds a soft floor. Two novelty species (select, tell). See wiki experiment 17.
