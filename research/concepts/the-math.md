# The mind's arithmetic: how an experience propagates and informs action

An explicit account of what `src/program/mind.rs` computes each step, as of 2026-10-12 (the
configuration after [108](../experiments/108-world-and-mind.md)'s per-act credit). Everything is
bit vectors, popcounts and integer arithmetic; no floating point anywhere.

## Notation

| symbol | meaning |
|---|---|
| `x[n]` | a bit vector `n` bits wide |
| `\|x\|` | popcount: the number of 1 bits in `x` |
| `x & y`, `x \| y`, `x ∖ y` | AND, OR, AND-NOT (`x & ¬y`) |
| `x ⊇ₖ m` | at least `k` of the bits of mask `m` are set in `x`: `\|x & m\| ≥ k` |
| `{i}` | the vector with bit `i` set; `⋃ᵢ` is OR over `i` |
| `h(·)` | a fixed hash (multiply–xorshift) to an integer |
| `topₙ(s)` | the `n` indices with the largest scores `s` |
| `sample(x, m)` | `m` bits drawn at random from the set bits of `x` |
| `cell c = (S_c[w], out_c)` | a cell: a sparse synapse mask `S_c` (16 set bits) over an input `w` wide, and an output pattern |
| `fires(c, x) := x ⊇θ S_c` | a cell fires when θ of its synapses are present |
| `P(x) := { c ∈ P : fires(c, x) }` | the cells of a pool `P` that fire on `x` |
| `≡` | is defined as |
| reals like `0.25` | Q16 fixed point: `1.0 ≡ 65536` |

Sizes: `B = 2048` (a frame), `k = 32` (bits in a code). Sensing areas `a ∈ {eye, ear, hand,
seen-acts, presence, affect}`; motor areas `m ∈ {voice, hand}`.

## 0. The world (outside the mind)

- **Senses.** Each step the world returns `s_t = (eye[B], ear[B], touch[B], pres[B], aff[B])`.
  Sounds, glyphs, cursors, moves and the light are fixed random codes with `|·| = 32`, ORed
  together when several are present. A stroke on the skin is a run of 32 neighbouring bits:
  `aff = ⋃_{j<32} {u + j}`.
- **Acts.** The mind returns `a_t = (cmd_voice[B] or ∅, cmd_hand[B] or ∅)`. A command articulates
  word `w` if `w = argmax_w |cmd & M_w|` and `|cmd & M_w| ≥ |M_w|/2` (`M_w` is word `w`'s motor
  code); otherwise it does nothing.

## 1. Layer 4: a sense's code (`Pool`)

Area `a` turns its raw input `x[B]` (the seen-acts area gets the eye's onsets,
`x = eye_t ∖ eye_{t−1}`) into a code:
```
n_c   = |x & S_c|                               (S_c: 16 synapses)
F     = top_K { c : n_c ≥ θ }                    θ = 10
K     = k, or with coverage  clamp(|x|, k, 4k)
e_a[B] = ⋃_{c ∈ F} {c}
```
**Learning.**
- **Winners.** Each winner, with probability 1/4, moves one synapse from an absent input bit to
  a present one.
- **Recruitment.** If `|F| < k/2`, cells are recruited up to `k`: `S_new = sample(x, 16)`.
- **Coverage.** Let `U = x ∖ ⋃_{c∈F} S_c` (input no winner covers). If `|U| ≥ |x|/4` and
  `|U| ≥ 10`, recruit `min(|U|, k)` cells with `S_new = sample(U, 16)`.

So a new thing in view adds cells: the cursor and light give `|e_eye| ≈ 64`. Without coverage
the light was invisible (`|e_eye| = 32`, all the cursor's).

## 2. Association columns grow

For each pair of areas `(a, b)`, count co-active steps: `n_ab += [|e_a| > 0 ∧ |e_b| > 0]`.
- **Pairs.** At `n_ab = 8` a pairwise column is created (at most 16 pairs). Its layer 4 is a
  pool over `[e_a | e_b]` (2B wide) with θ = 8, so it can complete from either half.
- **The full area.** Once there are 2 pairs, a full association column is created over the pair
  codes. Its layer 4 runs over `[e_p₁ | e_p₂ | …]`.

Every column, sensory or association, has the same three layers.

## 3. A column's layers 2/3 and 5 (`BitCells`)

A row is `r = [input[B] | ctx[B] | extra[B]]`. A cell has basal synapses on the input side and
apical synapses on the context (the tuft), each either active or silent.
```
p_c   = |apical_c & ctx| / |apical_c|                      (priming: the tuft's agreement)
burst = p_c ≥ θ_p                                          (θ_p adapts, ≈ 0.8)
fires(c) ≡ b_c + [burst]·s_c ≥ ⌈|basal_c| · thr_c⌉        b_c: active basal synapses present,
                                                            s_c: silent ones present
winner  = argmax_{c fired} (burst, p_c, trace_c)
```
- **Layer 2/3** predicts the column's next input: `ê_a = out(winner)`. Its surprise is
  `σ_a = 1 − |ê_a & e_a| / |e_a|`.
- **Layer 5** offers a goal: `g = out(c*)`, where `c* = argmax_{c fired} (W_c, burst)` (`W_c` is
  the cell's learned worth, section 8).

**Learning toward a target `T`** (the next input, or the next sense the effectors produced):
- If `|out(winner) & T| ≥ |T|/2`, the cells that predicted it strengthen.
- Otherwise a cell is grown: silent basal synapses on the input, apical ones on the context,
  `out = T`.

**A column's inputs and context.**
- A sensory column's input is `e_a ∪ td_a`; its context is the association columns' layer 2/3
  activity, OR the top-down `td_a`.
- An association column's input is its layer 4 code over the areas' codes, OR the hippocampus's
  offer for that column.
- Every layer 5's tuft is working memory OR the cerebellum's prediction.

**The cortex's expectation (the thalamic relay's mix).**
```
score[i] = Σ_{sources s with i ∈ g_s} ( w_s(ctx) / 16 + 1 )
expect   = top_k(score)
```
Each source's weight `w_s` is its learned reliability in this context. It is used only for the
surprise below; nothing routes it to the voice.

## 4. The hippocampus (`IndexHc`)

- **State.** `z = pack(assoc layer-4 codes)`, 17 slots of B bits, one per association column at a
  fixed slot. Context `C` = 32 integers in [0, 2¹⁶). Acetylcholine `A ∈ [0, 1]`. Rows are
  `(keys = C ∪ (z + base), points to z, successor link)`.
- **Mode.** Each step `A ← A − A/4`. If `z` is unchanged, nothing is new and the offer holds.
  Otherwise:
  - *outward* (an input onset that is not one's own sound): `A ← A + (1 − A)/2`, and a row is
    stored;
  - *own* (efference copy): `z` is represented, nothing is stored;
  - *inward* (new but neither): `A ← A/2`.
- **Drift.** The 4 cells `c` of `z` with the smallest `h(c)` each replace one context slot:
  `C[h(c) mod 32] ← h(c, C[h(c) mod 32]) mod 2¹⁶`. A learned boundary cell (from the cortex's
  surprise `1 − |expect & ear| / k`) replaces 16 slots.
- **Completion.**
  1. If the expected row's content keys hold at least half of `z`, follow: offer its successor.
  2. Else take the candidate rows whose content score is at least 3/4 of the best (not stored
     this step, not offered in the last 4) and jump to `argmax |keys ∩ C|`. A jump reinstates
     that row's context; following blends a quarter of the slots.
- **Output.** The offer reaches the cortex only if `A < 1/2` (retrieval). Slot `j` goes into
  association column `j` as extra input. From there it cascades down: each source's expected code
  is `top_k` of the votes of the code's cells' synapses on that source (layer 4 back-projection),
  pair → areas, full → pairs → areas.

## 5. Innate valuations (the only reward)

| term | value |
|---|---|
| `r₀` own effect | `err_fwd / 4`, the motor area's forward-model error on the act: how far the result was from the efference copy's prediction. It fades as the act becomes predictable |
| `r₁` response | `(p_after − p_else)/2`, if someone is present and a sound from outside begins within 2 steps after its own sound. `p_after` and `p_else` are running averages (1/32 per step) of a sound from outside beginning after its own sound, and otherwise |
| `r₂` affective touch | `min(1, \|aff\| / k)`, the receptor line's own activity |
| `r₃` cost | `−Σ_{acts} max(ε, W_act)`, with `ε = 1/32` and `W_act` the act's usual worth (section 6) |

`r₀` trains only the motor loops' last choices (agency). `r₁ + r₂ + r₃` go to the critic.
`r₁ + r₂` (outcomes) teach the amygdala and the orbitofrontal learner.

## 6. The amygdala and the orbitofrontal learner (bit cells)

**Onsets and the trace.**
- `o_t = { slot·B + i : e_t[i] ∧ ¬e_{t−1}[i] }`. The orbitofrontal learner also gets the
  efference copy's features: `(6 + m)·B + i` for each bit `i` of `cmd_m`.
- The trace `T_t` holds `(f, age)` for the onsets of the last 4 steps.
- Context: `χ[4096] = ⋃_{(f,age) ∈ T} { h(f, age) mod 4096 }`.

**Value.**
```
cells: masks m[4096], |m| = 16, fire if χ ⊇₁₂ m
V(χ)  = ( |A⁺(χ)| − |A⁻(χ)| + |Base(χ_act)| ) / 16          (appetitive, aversive, per-act baseline)
```
`χ_act` is built from the act features only. The prediction for this step uses the trace's ages
1–3.

**Learning** (outcome `r = r₁ + r₂`, `err = r − V`):
- `err ≥ 1/16`: the firing appetitive cells move one bit toward `χ`; up to `n = min(4, ⌊16·err⌋)`
  firing aversive cells are removed, and appetitive cells are recruited for the rest
  (`m = sample(χ, 16)`).
- `err ≤ −1/16`: up to `n` firing appetitive cells are removed (extinction). If `r < 0`, aversive
  cells are recruited.
- Baseline (orbitofrontal only): with `e_b = r − |Base(χ_act)|/16`, one cell is recruited if
  `e_b ≥ 1/16` and one removed if `e_b ≤ −1/16`. It is an average per act, never extinguished.
  `W_act = |Base(χ_act)| / 16`: each new act gets its own cells.

**Attention.** A bit vector `att[6B]` over features.
- `att[f] ← 1` when `f` is first seen, or when `f` is in the trace and `|err| ≥ 1/8` or
  `|V| ≥ 1/8`.
- Each step every bit clears with probability 1/32.
- The striatum gets `{ f·4 + age : (f, age) ∈ T, att[f] }` (at most 64).

**Pause cells.** Masks over untagged features. One is recruited on the cues of 1–2 steps before
a dopamine error `|δ| ≥ 1/4` if none fired; a firing one is removed with probability 1/4 when no
large error followed. If one fires now, the interneurons pause.

## 7. The striatum (`MaskStriatum`, d = 2048, 64 context bits)

**Context.** A loop's view is a list of input groups `G₁ … G_n`: a column's layer 2/3 activity,
the senses' sustained codes, the amygdala's attended trace and value cells, the orbitofrontal
learner's value cells, the efference copy, working memory. Each group sits at its own offset, and
valued groups are repeated (salience gain: ×(1 + min(3, ⌊4|V|⌋))).
```
x[2048] = singles ∪ pairs
singles = ⋃_{groups g} the ⌊32/n⌋ members of g with smallest h(g, ·)
pairs   = 32 bits h(pick_g, pick_g′) for picks from two groups      (conjunctions)
x_cand  = 32 state bits ∪ 32 bits h(state pick, candidate pick)
```
**Choice.** Pools `P⁺` (go) and `P⁻` (no-go) of masks with 16 bits over 2048, firing at ≥ 10:
```
q_i  = |P⁺(x_i)| − |P⁻(x_i)|
i*   = argmax_i ( q_i + U[−J, J] )        J = 3 + curiosity (0–2) + tonic (0–3)
release iff q_{i*} ≥ β                    β = 2 − clamp(⌊4·V_amy(cues)⌋, −2, 2)   (transfer)
```
**Critic.**
```
V(x)  = ( |C⁺(x)| − |C⁻(x)| ) / 8
δ_t   = clip₋₃^{+3}( r_t + V_t − V_t/8 − V_{t−1} )        γ = 7/8, in steps of 1/8
```
The critic's view is the limbic groups: the amygdala's attended cues and value cells, the
orbitofrontal value cells, presence, affect.

**Learning.** Let `n = ⌊|δ| / 0.5⌋` plus a stochastic remainder.
- **Critic.** The last state's context: for `δ > 0`, the firing `C⁺` cells tighten one bit toward
  `x`, `n` `C⁺` cells are recruited, and `n` firing `C⁻` cells move away; the reverse for `δ < 0`.
- **Actors.** Only at a cholinergic pause (`|δ| ≥ 1/4`, or a pause cell fires). Each choice in the
  last 8 steps, with weight `(4/5)^age`: for `δ > 0`, the go cells that fired tighten (and become
  sticky), `n` go cells are recruited, and `n` firing no-go cells move away; for `δ < 0`, the
  reverse.
- **Withholds** (motor gates only). For `δ < 0`, the firing no-go cells tighten, or `n` are
  recruited if none fired. For `δ > 0`, `n` firing no-go cells are removed.
- **Balance.** `|P| ≤ 2|P′| + 16`; the stalest cells go.
- **Agency and per-act credit,** outside δ:
  - `r₀` goes to the motor channels' last choices: `δ := r₀`.
  - When an act's 2-step outcome window closes, `δ := outcome − W_act` (its advantage) goes to that
    act's loop and gate choices of 2 steps back.

## 8. Loops: from goals to acts

For each column `j` with a layer 5 (one loop per column, created with it):
1. The goal `g_j` is offered on `j`'s loop channel, with `x = view(j)`: `taken_j` if released.
2. An association column's second channel may load working memory: `wm ← code_j`.

For each motor area `m`:
1. **Candidates:** the taken goals that `m` can plan (a command from the imitation route for the
   voice, or the inverse model), sent to the effector whose forward model reproduces the goal
   best, plus one spontaneous primitive of `m` (the genome's pattern generators).
2. **Choose** on `m`'s loop channel, then on `m`'s gate: release, giving `a_t[m] = cmd`, or
   withhold.
3. **Withheld** acts are imagined: `td ← encode(forward_m(cmd))`, top-down at the next step, to
   the cortex and the hippocampus alike.

**The worth of goal cells.** For the goal cells followed in the last 4 steps:
`W_c ← W_c + (δ·2^{−age} − W_c/8) / 8`. Layer 5 then offers its most valued fired cell.

## 9. One experience, step by step

The instrumental probe: the light is on, the mind clicks, a stroke follows.

| step | what happens |
|---|---|
| t | The light comes on. The eye's layer 4: cursor cells and ~32 new light cells (coverage); `e_eye` grows from 32 to 64 bits. The seen-acts area codes the onset. The onsets enter the amygdala's and the orbitofrontal learner's traces at age 0. The eye column's layer 2/3 is surprised (`σ_eye` high). If the light was attended or valued, its trace bits enter the striatum's context. Its value cells V_amy (if learned) lower the motor brake β. |
| t | The hand loop: candidates are the hand's taken goals plus a spontaneous primitive (say a click). `q = \|P⁺(x)\| − \|P⁻(x)\|` with noise; released if `q ≥ β`. `cmd_hand` goes to the world. |
| t+1 | The world carries out the click (no visible effect). The mind senses the result: the hand's forward-model error gives `r₀` (agency, to the hand's choices of step t). The efference copy enters the orbitofrontal learner's trace. The cost `r₃ = −max(ε, W_click)`. A 2-step outcome window opens. |
| t+2 | The stroke arrives: `aff` holds 32 bits, so `r₂ = 1`. The critic: `δ = r + V_{t+2} − V_{t+2}/8 − V_{t+1}`. If the click+light conjunction already predicted it, δ is small; if not, a burst. The amygdala learns light → touch (ages 2–3); the orbitofrontal learner learns light + click → touch (conjunctive cells), and the click's baseline moves one cell. At a pause, the actors of the last 8 steps learn from δ, the click's choice most (weight 0.8 at age 2). The window closes: `δ := 1 − W_click` to the hand's loop and gate choices of step t. The hippocampus stores the new state (an outward onset), with context `C`. |
| later, light off | A click gets no stroke: `r₂ = 0`, `r₃ = −max(ε, W_click)`. The window closes with `δ := 0 − W_click < 0`, to the hand's choices: their no-go side learns "not here". The critic's dip, if the orbitofrontal baseline predicted a touch, does the same at a pause. |

## 10. What grows, and how growth informs action

| what | grows when | bounded by |
|---|---|---|
| layer 4 cells | an input is not yet covered | `B` cells per area (stalest replaced) |
| layer 2/3 and 5 cells | a prediction missed a target | 8,192 per layer |
| association columns | two areas are co-active 8 steps | 16 pairs + 1 full |
| loops | a column is created: its loop channel(s) start empty | one per column |
| striatal cells | dopamine at a pause (actors), every δ (critic) | 2,048 per pool; balance `\|P\| ≤ 2\|P′\| + 16` |
| hippocampal rows | each outward onset | forgetting of unused rows |
| amygdala and orbitofrontal cells | an outcome error of ≥ 1/16 | 2,048 per pool |
| per-act baselines | an act is done and its outcome differs from its worth | one cell per step |

How action comes to depend on experience:
1. **The senses learn to code what appears** (layer 4, coverage).
2. **The columns learn what follows what,** and layer 5 what the effectors' senses received next:
   the goals.
3. **The amygdala and the orbitofrontal learner learn which cues, and which acts with which cues,
   come before outcomes.** Their value cells feed the critic (dopamine at the cue and at the act)
   and the motor loops' views (the context a loop can learn "release here" in), and lower the
   brake (transfer).
4. **The striatum's go and no-go cells learn which candidates to take and release,** in which
   context, from δ (state value), from per-act advantage, and from agency. Its pools grow with the
   contexts it meets.
5. **New columns bring new loops,** so the space of things that can be chosen grows with the
   cortex; layer 5's goal cells that were followed and paid gain worth, and are offered first.
