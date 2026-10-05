# 20 · Layer 6 corticothalamic gating: cortex learns which relays to let through

**Question.** In [16](16-thalamic-gate-memory-channel.md) the basal ganglia released one
thalamic channel per word (a relay route or memory recall), learned from a reward. Cortex
also controls the thalamus directly. Layer 6 projects back to the relay nuclei it receives
from, and to the inhibitory reticular nucleus, enhancing some relays and suppressing
others. Can a gate driven by the column itself, learned without any reward, decide which
channels reach cortex?

**Code.**
- `CorticothalamicGate` in [`src/program/thalamus.rs`](../../src/program/thalamus.rs).
- `CorticalColumn::outcome_via` (L5 attribution, [19](19-l5-shared-reward.md)).
- `Policy::L6Gate { gated }` in [`examples/episodic.rs`](../../examples/episodic.rs).

Run: `COPY_GROW=1 GROW_TRUST=0.5 STICKY=255 GENERALIZE=0.5 GENERALIZE_AFTER=1
TRUST_AT_TEST=0.5 TASK=varied POLICIES=l6 cargo run --release --example episodic`.

## Mechanism
- **Channels:** the three fixed relay routes of [16](16-thalamic-gate-memory-channel.md),
  (1,4), (3,4) and (3,8), plus memory recall. Unlike the basal-ganglia gate, **each channel
  has its own L4 frame**, and a closed channel's frame is empty. Any number can be open
  at once: gain control, not winner-take-one.
- **Gain:** a facilitation value per (channel, cortical context). The context is the
  current input, the L6 state. The value is the channel's sparse code bound to the context
  by rotation, read through 4-plane bit-sliced counters, as in the basal ganglia. A channel
  is open when its facilitation is ≥ 0.5.
  - Every channel starts at 0.5, so all relays pass until cortex learns to suppress them.
  - During training, a closed channel opens anyway with probability 0.05, so it can
    recover.
- **Learning is Hebbian; there is no reward.** After each prediction, every open channel
  that relayed something is strengthened if the prediction **read its frame and came
  true** (`outcome_via` ≥ 0.5), and weakened otherwise. Each bit steps with probability 0.3.
- **Baseline:** the same frames with every channel always open.

## Results (varied stories, held-out pairs, seeds 0 / 1 / 2)

| | Held-out, 1–2 / 1–3 facts | Channels passed per word at test (of 4) | Open at test answers | Wall time per run |
|---|---|---|---|---|
| all channels open | 100 / 100% on every seed | 3.49–3.51 | every channel 100% | 1,053–1,236 s |
| **L6 gate** | **100 / 100% on every seed** | **0.04–0.19** | **memory 100%, every route 0%** | **445–506 s** |
| basal-ganglia gate, one channel ([16](16-thalamic-gate-memory-channel.md)) | 100 / 100% on every seed | 1 (by construction) | memory 100% | – |

## Findings
1. **Cortex learns to shut almost all of the thalamus without losing anything.**
   - The gate passes 0.04–0.19 channels per word, against 3.5 when everything is open:
     about 95–99% of relay traffic is suppressed.
   - Memory opens at every answer, and the routes never do.
   - Accuracy is unchanged (100%), and runs take 2.4× less wall time, since L2/3 matches
     against empty frames almost everywhere.
2. **It needs no reward.** The basal-ganglia gate of [16](16-thalamic-gate-memory-channel.md)
   needed a dopamine signal per choice. Here the column's own attribution (which frame the
   winning kernel read, and whether its prediction came true) is enough. A channel that
   cortex never uses in a context closes there. This is the division of labour in the
   [map](../concepts/architecture-map.md):
   - the basal ganglia **select** under reward;
   - L6 **filters** by use;
   - both act on the same relays.
3. **Gating is per context, and it is what makes it cheap.** Memory is open where it is
   used: at the answer, and in the 1–3 fact stories a little more (0.19 per word on seed
   0). It stays closed after ordinary words, where the predictor answers from the current
   and previous word alone.
4. **Limit: the routes are useless on this task.** The routes alone score 36–41%
   ([16](16-thalamic-gate-memory-channel.md)), and the predictor never needs them once
   memory is open, so "close every route" is the right answer here. A task where different
   channels carry the answer in different contexts would test whether L6 opens *several*
   channels where they are needed. That is the next test.

## Biology
- **Layer 6 corticothalamic cells:** they send feedback to the thalamic nucleus that
  drives their column, and collaterals to the thalamic reticular nucleus (TRN). The direct
  path is excitatory and focal; the TRN path inhibits the surround. The net effect is to
  boost the relays that cortex is attending to and suppress the rest, i.e. gain control
  over many channels at once.
- **Learning:** corticothalamic synapses are plastic and show facilitation. A Hebbian
  rule keyed on whether the relayed input was used is a simple stand-in.
- **Basal ganglia versus L6:** the basal ganglia act through the inhibitory output
  nuclei (GPi / SNr) onto the thalamus, while L6 acts through direct feedback and the TRN.
  Here both are learned gates on the same relays, one from reward and one from use. They
  have not yet been run together.
