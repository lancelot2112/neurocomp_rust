# 106 · Three systems separated: the cerebellar circuit, recurrent layer 2/3, a thalamic relay, and a control loop for inner speech and saccades

**Question.** After [105](105-bitwise-default-sticky-and-l5-output.md) the cortex's layer 2/3
was still a one-shot conjunction memoriser, the cerebellum was that same kernel class
relabelled, and the thalamic vote was weighted by tables keyed on word ids. This page covers
the steps that separate the three learning systems and replace those parts with circuits:
- stickiness by tagging and capture;
- the cerebellar circuit, its pontine input, its output to layer 1, its olive, and the red
  nucleus;
- the three-system default;
- layer 4 and recurrent layer 2/3;
- the bitwise thalamic gate and the thalamic relay;
- a learned control loop for inner speech and saccades;
- a prose driver.

Several suites are still running; their rows say so.

## 1. Stickiness by tagging and capture (default)
`BitCells::set_tag_capture`: a confirmed fire tags the synapses it confirmed (one more mask).
A synapse becomes sticky only if it is confirmed again while still tagged, active in two
consecutive confirmed fires. A contradicted fire clears the tags of the active synapses. This
replaces the random consolidation gate (Frey & Morris 1997).

All 25 suite entries, five seeds, against the bitwise default of 105: **+0.8, 16 of 25 up**
(role transfer +7.0, family stated +6.2, inference replay +4.5, engram store +3.9; index
hippocampus −4.9, slot memory −3.1). It became the default.

**A relaying layer 5** (`L5_OUT=relay`: a silent layer 5 passes layer 2/3's prediction on)
lost −8.9 (3 of 25 up). Its single spikes are context-free habits, and letting them override
layer 2/3 costs where the answer depends on the story.

## 2. The cerebellar circuit
The cerebellum was the cortex's kernel class (`make_l23`) relabelled. Now it has its own
circuit ([`src/program/cerebellar_circuit.rs`](../../src/program/cerebellar_circuit.rs)):
- **Mossy fibres** carry a copy of the cortex's input. With `CB_MOSSY=pons` they carry the
  current word and the pons's recoding of what layer 5 fired instead.
- **Granule cells:** a fixed random expansion. Four dendrites per cell; a cell fires on at least
  two coincident inputs, and Golgi inhibition raises the threshold when too many fire.
  `CB_CROSS`: the four dendrites go to different sources (frames).
- **Purkinje cells:** one per output bit. Parallel-fibre synapses start potent and learn by
  climbing-fibre depression, a bit cleared per active granule cell. Restoration is slower.
- **Deep nuclei:** a relative readout. The bits whose Purkinje cells are most depressed fire,
  as many as a target usually has, so granule cells depressed for every output cancel out.
- **Counting** is bitwise: the active granule cells' depressed-synapse bitsets go through
  bit-sliced counters. The granule side is event-driven, because a dense granule bitset would
  cost far more at this sparsity.

Seed 0, hippocampus teaches cortex, three systems (the kernel cerebellum: right at 62.1% of
answers, held-out 55.2):

| Circuit | Cerebellum right | Held-out |
|---|---|---|
| absolute readout | 25–33% | 28–37 |
| relative readout, 32k granule cells | 40.7% | 48.2 |
| two-input floor, cross-source, 131k | 32.6% | 39.6 |
| two-input floor, cross-source, 262k | **53.2%** | 38.2 |
| two-input floor, same-source, 131k | 34.4% | 49.2 |

More granule cells make the cerebellum more accurate (the Marr–Albus expansion). Held-out
does not follow yet: the vote does not use a better source.

**Wiring** (seed 0, same entry):

| | Cerebellum right | Held-out |
|---|---|---|
| kernel cerebellum, output into layer 5's tuft (`CB_L1`) | 58.4% | **57.2** (+2.0) |
| kernel cerebellum, mossy fibres from the pons | 27.1% | 54.4 |
| circuit, pons | 23.8% | 25.0 |
| circuit, pons, layer 1 output | 23.8% | 27.8 |

The deep nuclei → motor thalamus → layer 1 route helps. Mossy fibres from layer 5 alone starve
the cerebellum: layer 5 fires a handful of cells per step, while real pontine input converges
from layer 5 of many areas.

**The olive** (seed 0; default: family consolidated 27.4, hippocampus teaches cortex 38.4):

| | Family consolidated | Hippocampus teaches cortex |
|---|---|---|
| recovery of synapses active without a climbing fibre (`CB_LTP=nocf`) | 30.0 | **45.8** |
| a sparse, synchronous olive (`CB_OLIVE=sparse`) | 36.6 | 39.8 |
| both | **38.8** | 39.2 |

All six cells are above the default. Five seeds on five entries are running.

**Defaults now:** the circuit with relative readout, cross-source granule cells (262,144),
the two-input floor, and output to layer 5's tuft. `CB=kernels` restores the old cerebellum.

## 3. Three learning systems by default
`LEARNING` now defaults to three (a slow cortex, the cerebellar circuit, the hippocampus), with
sleep-opened plasticity and replay that predicts before learning. `LEARNING=one` restores the
single fast cortex. All 25 entries, five seeds, against the one-system default: **−10.9, 6 of
25 up**. The losses are on the memory and consolidation entries (inference replay −32.9,
engram store −26.7, semantic store −25.4, hippocampus teaches cortex −24.9). The slow learners
take far less from 3,000 stories than the one-shot layer 2/3 did. The regression table now
records this default; the one-system figures are kept with the run.

`TRAIN_STORIES` sets the number of training stories. Runs at 10,000 and 30,000 are queued.

## 4. Layer 4 and recurrent layer 2/3
In the cortex, layer 2/3 is driven by layer 4's recoding of the thalamic input and, mostly, by
other layer 2/3 cells (recurrent and horizontal connections). Its context is its own ongoing
activity, and its output goes to layer 5 and to other areas, not to the thalamus.
- **Layer 4** ([`src/program/layer4.rs`](../../src/program/layer4.rs), `L4=1`): 2,048 cells
  with 16 synapses each on the word code. The 32 most driven fire (k-winners-take-all). A winner
  moves one synapse from an input bit that was off to one that was on (competitive Hebbian).
- **Recurrent layer 2/3** (`L23_REC=1`, with `L23=primed`): the cells layer 2/3 fired at the
  previous step, projected into one frame, replace the pasted previous-word frame.

Seed 0 (held-out):

| | Family consolidated | Hippocampus teaches cortex |
|---|---|---|
| three-system default | 27.4 | 38.4 |
| + primed layer 2/3 | 40.4 | 32.2 |
| + layer 4 | 32.4 | 34.2 |
| **+ layer 4 + recurrence** | **41.6** | **49.2** |

Recurrence is what makes the plausible layer 2/3 work. The full suite is running.

## 5. The thalamus
- **A bitwise thalamic gate** ([`src/program/thalamic_gate.rs`](../../src/program/thalamic_gate.rs),
  `THAL=bits`): each source's reliability is the potent share of binary synapses on the context
  pattern (current and previous input), learned stochastically from hits and misses. It replaces
  the word-pair tables. Seed 0 in the one-system network: 48.2 / 59.8 / 96.2 against
  48.6 / 60.0 / 98.0 with the tables. Its suite on the three-system default is queued.
- **The thalamic relay** (`THAL=relay`): a source's weight is driver strength × the learned
  context gate × agreement with the cerebellum:
  - a burst passes fully;
  - a single spike passes at one half, and consecutive spikes depress the synapse further
    (½, ¼, ⅛), as depressing driver synapses do;
  - memory, the higher area and the cerebellum pass fully.

  Layer 5's spikes reach the vote this way, as weak drivers. `L5_GROW=l23` makes layer 5 grow a
  cell whenever layer 2/3 grows a kernel. Both suites are queued.

## 6. The red nucleus, inner speech and a control loop
- **The red nucleus** (`RN=1`): on steps the network produces itself, the cortex's intended word
  reaches the olive, so the cerebellum also learns a forward model of what the network says.
  In the suite the network speaks only at test, so the path never fired (0 steps).
- **Inner speech during training** (`INNER_TRAIN=1`) lets it fire, with the test-time rollout
  kept.
- **A learned control loop** (`INNER_GATE=pfc`) replaces the rules on when and what:
  - *context:* an anterior-cingulate-like conflict signal (how many words the sources proposed
    in the vote), the vote's confidence, hippocampal novelty and the prefrontal state;
  - *choice:* the basal ganglia pick silence, the prediction, or recall;
  - *reward:* dopamine at every step, from whether speaking turned the next word from wrong to
    right;
  - two hand filters are dropped: "must fit the expected kind", and "must not be the next
    word", which peeked at the page.

Fair comparison, seed 0, held-out (inner speech in training only, the rollout kept at test):

| Entry | Default | No completion | Rule speech | + red nucleus | Learned loop | Loop + red nucleus |
|---|---|---|---|---|---|---|
| family consolidated | 27.4 | 28.4 | 26.8 | 31.8 | 26.0 | 25.4 |
| closed loop | 47.8 | 28.4 | 46.2 | 40.8 | 33.0 | 29.0 |
| learned stepping | 40.4 | 28.4 | 45.6 | 39.6 | 38.4 | 31.4 |
| hippocampus teaches cortex | 38.4 | 22.0 | 39.8 | 45.6 | **57.2** | 37.0 |
| engram store | 42.2 | 24.6 | 38.2 | 36.8 | 48.6 | **59.6** |
| semantic store | 45.8 | 28.0 | 48.2 | 45.6 | 40.2 | 35.4 |

The red nucleus fired on 159–1,714 steps with rule speech and on 5,060–7,798 with the loop.

An earlier version of this comparison replaced the rollout with inner speech, so it measured
the rollout's absence: without any completion, five of six entries lose 13–19 points.

- **Saccades on the same loop** (`SACCADE_CTX=loop`): the same activity context and basal
  ganglia choose read on / look back a sentence / look back to the page top, rewarded by the
  next prediction minus a regression's cost. This replaces the word-id context. On the saccades
  entry, the loop scored 99.8 / 22.6 / 100.0 over seeds 0–2, against 94.0 / 92.8 / 97.4 for the
  word-id version. All seeds over-regress (53–103 look-backs per story), and seed 1 locked onto
  the previous sentence.

## 7. A prose driver
[`examples/prose.rs`](../../examples/prose.rs) wires the plausible loop from library parts:
- layer 4;
- recurrent layer 2/3;
- layer 5;
- the cerebellar circuit fed by the pons;
- a thalamic relay over bit patterns (weighted bits summed, the 32 strongest kept: no word is
  decoded inside the network);
- speech through the learned motor path (babbling, an inverse model, the vocal tract as the
  world).

A first try on 4,500 words of Alice: held-out next word right 7.4%, the same as always saying
",". Writing from "alice was" gave a memorised fragment ("the fire, and washing her face and
she's such a nice soft"), then silence: a silent step left the input unchanged. Now a silent
step hears its own plan (an efference copy).

Full runs (seed 0; vocabulary of the 3,000 commonest words; the last tenth held out, learning
off):

| Text | Words read | Reading, last stretch | Held-out | Always "," | Layer 2/3 proposals (right) | Layer 5 | Cerebellum |
|---|---|---|---|---|---|---|---|
| Alice | 28,054 | 9.8% | **9.8%** | 8.4% | 784 of 3,117 (9.8%) | 133 (12.0%) | 3,117 (11.4%) |
| Tiny Shakespeare | 225,130 | 10.0% | **6.3%** | 8.3% | 223 of 25,014 (0.9%) | 0 | 25,014 (6.3%) |

Written text quotes training sentences, then loops on one:
- *Alice* ("alice was"): "alice was sitting next to see if he would deny it usually … how to
  see if he would deny it usually … and went down on one knee. here, the miserable hatter
  dropped his teacup and went down on one knee. here, the miserable hatter dropped his teacup
  and …".
- *Shakespeare* ("the king"): "the king so bold to me, in the eldest sister. nor is your firm
  resolve unknown to me, in the eldest sister. nor is your firm resolve unknown to me, …".

Two problems show:
- **The cortex does not generalise.** Its cells are conjunctions of a context seen once. On new
  text they rarely fire (Shakespeare: layer 2/3 on 0.9% of held-out words, layer 5 never), so
  only the cerebellum votes.
- **Nothing stops a loop.** No cell tires, so a recurrent attractor repeats the same sentence.
  Firing adaptation (cells that fired recently are harder to fire) is the plausible fix.

## Findings
1. **Separating the systems costs ~11 points for now,** because the slow learners need more
   data. **Recurrent layer 2/3 is the first part to win back much of it** (seed 0: +14 and +11).
2. **The cerebellar circuit learns better with a bigger expansion and a relative readout,** and
   biological changes to its olive (recovery without a teacher, a sparse synchronous olive)
   help on both entries tried. Its output helps most when it primes the cortex through layer 1.
3. **The learned control loop gives the largest single gains** (hippocampus teaches cortex
   +18.8, engram store +17.4 with the red nucleus) but is unstable across entries. The same
   holds for saccades on the loop (two seeds near perfect, one collapsed). The control loop's
   context is too coarse, and its costs need calibrating.
4. **Tag and capture is the one change here measured on the full suite** and it gains (+0.8).
   It is the default.

## Pending (running or queued)
- Recurrent layer 2/3 + layer 4 on the full suite.
- The olive changes at five seeds.
- The bitwise gate and the relay (with and without layer 5 growth) on the three-system default.
- More training data (10,000 and 30,000 stories).
- Saccade loop fixes (a richer context, a self-calibrating cost).
- The prose driver: adaptation against loops; cells that generalise (partial-match firing).
