# Open questions

1. **Composable integer and rotate kernels.** No experiment tested `ADD`/`SUB` or `ROTATE` as
   general write ops that the network composes. Rotation was used only for binding, and add or
   subtract only inside bit-sliced counters. What do evolved networks do with them?
2. **Generalisation.** Exact-bit masks key on identity: a new wording of a question scored 0%
   (30), and one-shot conjunctions fired on 0.9% of held-out prose (106). Partial-match firing,
   factored role-and-content codes, or something evolved?
3. **Competition without a hand order.** How should cells compete so the right candidate wins,
   without collapsing on some seeds (94, 97)? Local overlap inhibition is this branch's answer;
   does it scale?
4. **Rarity without global counts.** Superposed stores need each input scaled by its use (see
   [mechanisms](mechanisms.md)). Which local form works?
5. **Attractor loops and fading.** Nothing stops a recurrent loop; firing adaptation is proposed
   (106).
6. **Context holding.** Reversal detection reaches 6–9/10 with about 10 false alarms per swap
   ([kernel cells](kernel-cells-and-context.md)). Can the next layer also see that a step was
   fully predicted, which a higher area may need?
7. **Consolidation.** How does a slow learner absorb what a one-shot store holds without the
   crutch effect, where the store answers and removes the surprise that would teach the rest?
8. **Instrumental control.** Acting on a cue failed in every version of the integrated mind,
   and worked only in isolated circuits with short eligibility (108).
9. **A fallback when nothing matches.** An n-gram always backs off to a shorter context; a
   kernel network may predict nothing.
10. **Density under fan-in.** When many channels are ORed into one vector, what keeps its density
    low: k-winners caps, gains, or evolved sub-masks?
11. **Can evolution find and improve the learning rule?** By hand, the three-factor output rule
   learns 128 arbitrary pairs in a lifetime when its target is right
   ([experiment 17](experiments/17-the-panel-and-the-learning-rule.md)). Does evolution find its
   timing (run 31), improve on the seed table (`ISL_RULES`), and build the right target (a
   listener that echoes every token), on 3 evolutionary seeds?
12. **Does a lifetime learner keep learning?** Select on the learning slope (late minus early in a
   lifetime), and check that a learner found on 400-step lifetimes still gains over 10^5 steps
   and on raw bytes against an n-gram.
