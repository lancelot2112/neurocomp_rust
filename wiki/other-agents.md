# Concept: other agents as the world (open-ended evolution)

Status: first step built (2026-10-19): the duel island, `ISL_DUEL=1`. Queued for run 41.

## Why

Every island so far climbs a ladder we wrote: a fixed distribution of worlds per rung. Evolution
solves a rung by wiring a reaction, and learning pays only where we vary the world between
lifetimes (lesson 56). Each stall has been met with another hand-made fix (steps, stray and miss
costs, partial credit). We are the only source of new complexity, so the system cannot outgrow us.

A partner drawn from an evolving population is a world that

- differs from lifetime to lifetime (another partner, whose habits no genome can wire in ahead,
  because the population keeps changing),
- holds still within a lifetime (one partner keeps its habits, so they can be learned),
- grows harder by itself (as agents get better at reading each other, each is a harder partner).

That is the condition lesson 56 says selects for learning, with no one designing it. A partner is
only bits on the ear and the eye, so nothing breaks the rules: no oracle, no task rule inside the
network.

## The first game: predict or evade, both roles (`World::duel`)

Two agents, each in its own world of the same physics (the same body codes), linked by the harness
(`evolve::live_duel`). Each step is a round; each agent's hand move is read as LEFT or not.

- The **predictor** wins a round when the two moves match, the **evader** when they differ.
- The **roles flip** every `turn` rounds (`Lesson::duel_turn`), each agent starting in the role
  its world draws, so every agent plays both and must learn both: reading a partner's pattern, and
  being unreadable to the partner's reading.
- Senses: the eye shows the partner's last move as the agent feels its own; the ear a tone for the
  role of the round to come. The winner is stroked the step after; the loser feels the wrong signal
  (`ISL_WRONG_SIGNAL`).
- Score (`observe::duel`): rounds won less rounds lost, per round, in each role, the two averaged.
  Zero-sum between the two agents: with a copy of itself, about 0.

Partners (`evolve_islands`, each generation, `set_duel_partners`): the duel island's elite as it
was, its hall of fame (its best every 10 generations, the last 16) and every other island's best
(they do not know the game; they are the world's easy and odd partners). Each lifetime seed meets
one partner, the same for every genome in the generation, so genomes are compared fairly.

The yardstick (`duel-bench`, `evolve::duel_bench`): the mean duel score against four rules drawn
per lifetime (`evolve::scripted`), each playing both roles. As evader, half the draws a pattern of
period 2 to 5 with its moves drawn, half a reaction to the predictor (its last move or the one
before, or their opposites); as predictor, the other's last move, the one before or its majority
of three, or their opposites. Each rule is as likely as its opposite. The first yardstick, four
fixed rules, leaked: a wired reaction (copy the partner's last move as predictor, alternate as
evader) scored about 213, and run 41's first start passed duel64 and duel32 by generation 6. On the
drawn rules every fixed policy measured scores within 100 of 0 (never 16, always -17, alternate 14,
copy lag 2 to 12 46 to 84, copy last and alternate -53), a reader of the partner 918
(`no_fixed_policy_beats_the_duel_yardstick`, 96 rules).

The ladder (`DUEL_LADDER`): duel64, duel32, duel16 and duel8, the turn shortening and the
yardstick's goal rising (200, 350, 500, 650), then an open rung. Selection uses the duel against
this generation's partners and the yardstick; each goal is on the yardstick, on held-out worlds.

Tests: `the_duel_pays_reading_the_partner_in_both_roles` (a reader 912 against an alternating
copier, which scores -912; constant against constant 0; a reader against a constant 512; half the
rounds in each role), `the_duel_yardstick_scores_doing_nothing_about_zero`.

## What can go wrong, and what is in place

- **Cycling** (rock, paper, scissors forever) and **forgetting**: the hall of fame keeps old
  partners in the pool.
- **Disengagement** (one side wins outright, the gradient vanishes): the other islands' best and
  the hall of fame give partners of every strength. Not yet: pairing by skill, POET's minimal
  criterion.
- **No fixed measure** (scores against moving partners cannot be compared across generations): the
  yardstick is fixed; the learning battery and the other ladders are the held-out measures of
  whether general learning rises.
- **Cost**: five more lifetimes per seed on the duel island (one partner, four rules).

## Next games

- **Speaker and listener** (cooperative): one agent sees which light pays, the other hears only the
  first's voice and acts; both are paid when the act is right; roles flip. Each partner's code
  differs, so it must be learned: the way toward conversation. Convention lock-in is the risk: draw
  partners from other islands and the hall of fame, or scramble the channel per lifetime.
- Two agents on one screen, competing for targets.
