# Panel memo 4: learning theory, neuroscience and scale

## Where the project stands

The substrate is sound and cheap. The results so far show evolution building small reflex and memory circuits, not learners. In run 29 every rung was passed by generation 74 and none in the 460 generations after. The elites run 2 to 27 populations and do about 2 to 22 units of work a step. The rungs that stay stuck across thirty runs (win-stay, the repeat rung, the maze's turn, echo2, cycle1) are the ones that need learning from experience within a lifetime. The rungs that pass (respond, lag1 to lag16, clean, shift) can be wired in by the genome. The best hand-arranged learner on raw bytes (experiment 03) gets 50.1% next-byte accuracy, while a longest-context frequency lookup gets 53.9%. So the substrate's lifetime learning, even when set up by hand, is below an n-gram. Language models win precisely by keeping improving past n-gram statistics as data grows.

That gap, not compute, is the main obstacle to evolving general agents with bit math.

## What the brain has that this lacks

- **Credit assignment across time.** Animals use three-factor plasticity: a synapse marks itself eligible when its pre- and post-synaptic cells co-fire, and a broadcast neuromodulator (dopamine, acetylcholine) arriving within seconds turns that mark into a lasting change. Here a population's learn gate is checked in the same step. There is no per-kernel eligibility trace, so a stroke two steps after an act cannot credit that act. Lesson 27 says an eligibility window of about 2 steps is what works. The run statistics show the structure is already there (48% of populations learn under a gate, 993 of them fed from affect) but it is never used.
- **A fast binding store.** The hippocampus binds an episode in one shot, and the cortex learns slowly from replay of it. Lessons 22 to 26 found the same thing: superposed Hebbian stores collapse, and a one-row-per-event store reaches 98%. The current runtime has no such store as something evolution can easily grow.
- **Prediction error driving learning everywhere.** Lesson 10 says to grow on surprise. In the runtime, though, prediction-driven learning exists only in the populations that evolution happens to make predictors.
- **Scale and experience.** A brain has about 10^14 synapses and 10^9 seconds of experience, and it learns from teachers who shape its input. Here lifetimes are 200 to 400 steps. No learner of any kind gets language from 300 steps.
- **A genome that encodes rules, not wiring.** About 750 MB of DNA specifies learning machinery and a developmental program. The synapses are filled in by experience. This project's genome still lists populations one by one.

## Compute, honestly

The harness runs 9 x 32 x 24 = 6,912 lifetimes of about 300 steps, roughly 2 million network-steps, in 25 s. That is about 80,000 network-steps a second. At about 10 work units a step, the network itself does about 10^6 kernel matches a second. The 16-thread workstation's peak for AND plus popcount is around 10^12 to 10^13 bit-ops a second, so the networks use about a millionth of the machine. The rest goes to the world, the observer and evolution.

For comparison, a 7B transformer needs about 10^10 FLOPs a token, and a brain does about 10^14 to 10^15 synaptic events a second. Counting one binary synapse event as about one bit-op, one real-time agent of brain-like sparsity fits within one to two orders of magnitude of this machine. Running the agent is not the barrier.

Evolving it is. Suppose a lifetime long enough to show conversation is 10^7 to 10^8 steps, at 10^6 to 10^9 bit-ops a step. Suppose also 10^3 to 10^4 lifetimes a generation, to beat the ±25-50 noise lesson 38 measured, and 10^3 or more generations. That comes to 10^20 to 10^24 bit-ops: years to millennia on this workstation, before counting the world. Evolution here contributes only a few bits of selected information a generation. It cannot write an agent's knowledge. It can only find a learner, and that learner must then do the rest from experience.

## What must change for evolution to find a general agent

1. **Select for learning, not for skills.** The ladders reward circuits that evolution can hard-wire, so evolution hard-wires them. Run 30's fixed body moves further the wrong way: when codes are the same in every lifetime, they can be built into the genome. Fitness should be the slope of a learning curve on tasks drawn fresh each lifetime: arbitrary associations, arbitrary reward contingencies, an arbitrary small grammar. Each needs a blind control and a hand-written learner as a reference (lessons 36 and 46). That gives meta-learning pressure, which is the only thing that grows with scale.
2. **Add an eligibility bit as a primitive, and let any channel act as a modulator.** Give each kernel a small history of its own firing, an eligibility ring like `BitVecHistory`, and let a gated learn move apply to kernels that fired k steps earlier. This is a local three-factor rule. It does not break "no rewards inside the network" (the world still gives none, and the affect port is a sensation, like sweetness), and it does not break "no globals" (the modulator is a channel that genes choose to read). I would reinterpret those two rules rather than relax them: a broadcast channel is allowed when reading it is a gene's choice.
3. **An indirect, developmental genome.** Circuit copy and plasmids are the first steps. To reach 10^9 or more lines, the grammar needs repeated motifs with a parameter, like HyperNEAT, so that one gene makes a cortex-like sheet. Without that, the size of the genome limits the size of the network.
4. **Long lifetimes with a teacher in the world.** A pretrained language model is allowed as the world: a conversation partner and a scripted computer terminal, with an LLM judge as the observer outside the network. It supplies the social teaching that brains get. It must never feed scores into the network.
5. **Evolution with a hand-written seed.** CLAUDE.md already allows hand-written genomes as starting points. Seed populations with a working three-factor learner and a one-shot store, written as kernels, and let evolution improve them. Waiting for evolution to discover reward learning from a naked genome has failed for twenty runs.

## Where bit math could genuinely win

A binary synapse event costs one AND plus a popcount share, against a 16-bit multiply-add. That is plausibly 10 to 100 times better in energy, and better again on in-memory or FPGA hardware. Sparse event-driven updates, with zero cost for paused circuits, follow how brains save energy. Kanerva's sparse distributed memory approximates transformer attention (Bricken and Pehlevan), so a bit-vector associative store is a credible route to attention-like retrieval. Masks are readable, which makes the networks easy to interpret. None of this has been shown at a scale that matters yet.

## Is the path plausible?

As things stand, no. Evolution over bit-math networks producing a conversing, computer-using agent has no precedent at any scale. Evolved plasticity rules (Najarro and Risi; Jordan et al.; Confavreux et al.) work only on small control tasks, and meta-learned RL rules needed meta-gradients and huge compute. It becomes plausible only if a bit-math learner exists whose held-out prediction keeps improving with experience, past an n-gram. That is the fundamental thing to establish first. If no such learner exists, evolution cannot find one, and more island runs will not change that.

## The next three experiments

1. **The data-scaling test (decisive).** Use the experiment 03 byte harness with lifetimes of 10^3, 10^4, 10^5 and 10^6 bytes. Hand-seed the learner (surprise-driven births plus a one-shot store), then evolve only its learning genes. Measure held-out bits per byte against an n-gram with the same storage. If the curve flattens at n-gram level, no evolved bit-math agent will converse.
2. **An eligibility trace with a modulator channel**, on win-stay, repeat and the maze's turn. Compare a hand-seeded and an evolved learner against the current runtime, with blind controls, 48 held-out worlds and several seeds.
3. **A meta-learning island.** Every lifetime draws a new random association or contingency task, and fitness is late-lifetime score minus early-lifetime score. Run it beside a skill island to see whether evolution, under that pressure, finds learning machinery it does not find now.
