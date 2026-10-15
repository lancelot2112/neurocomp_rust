# Island run 9

Run 8's settings with only the win-stay probe fixed: a lifetime with no stroked click to compare
against an unstroked one scores 0 (a clicker that ignored strokes scored +-1024 by which light
it kept to). 600 generations in about 1 h 45 min (`runs/run9.cmd`).

Stage moves (each passed on held-out worlds): shapes clean to shift64 by 84, win-stay at 101;
reading to echo2 at 20; recall lag1 at 97, lag2 at 107, lag3 at 114, lag4 at 186 (run 8: 210-420);
switching tell at 226, win-stay at 229; prediction to cycle2 at 415, the first pass of cycle1 in
any run.

Mean goal score over generations 100-599: switching win-stay 3 (best 75), shapes win-stay 2
(best 50), screen glow 111 (best 204; runs 7-8: 45-48), maze turn -1, reading echo2 -14,
prediction cycle1 79 then cycle2 61.

With the probe fixed a genome that ignores strokes scores exactly 0 on win-stay, so its mean of
2-3 says no genome learned from reward. Run 10 adds a repeat rung below it.

Plasmids: 458 moved, 2 still carried at the end.

`checkpoint.txt` is generation 599. Recheck with
`RCK_INPLACE=1 RCK_READS=1 RCK_RATION=256 RCK_FOOD=32 RCK_FILE=runs/islands9/checkpoint.txt`.
