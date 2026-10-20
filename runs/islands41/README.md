# Island run 41

Run 40's settings with a duel island (`ISL_DUEL=1`): predict or evade against other agents, the
roles flipping every few rounds so each agent learns both; partners the duel elite, its hall of
fame and the other islands' best; goals on a yardstick of rules drawn per lifetime. First started
8:17 PM ET 2026-10-19 with a yardstick of four fixed rules that leaked (`runs/checks/duel-leak/`);
restarted about 8:25 PM ET; stopped by hand at generation 172 (11:26 PM ET) for run 42.

9 rung moves, none for the duel island: its yardstick stayed within what fixed policies score
(best held out 78 to 104 from 130 on, goal 200). The learning battery at 142
(`runs/checks/learn-battery/run41-g142.txt`): learning clearly helps in 46 of 330 genome-task
pairs, mostly by calming networks that act wildly frozen; no task above 0. Run 42 makes what
learning buys a selection case. See wiki experiment 17 and `wiki/other-agents.md`.
