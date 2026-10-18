# Panel memo 2: the evolutionary search and the curriculum

## What the runs actually show

Thirty island runs share one pattern. Evolution passes a rung quickly when the answer can be built into the genome. It stalls, often for good, when the answer has to be learned within a lifetime.

Run 29 passed all 15 of its rungs by generation 74 and none in the 460 generations after. Run 30 adds a fixed body, so the voice codes, the hand codes and the feel of moves are now anatomy. It made 17 moves by generation 63, and the maze passed feel and learn for the first time in any run. The rungs that hold across runs are reading's echo2 (about 0 for 490 generations in run 25), win-stay and repeat, cycle1, and lag3/lag4. Each of them needs a code that is redrawn every lifetime, or credit from reward, to be acquired in that lifetime.

The signal measurement after run 20 is the most important number in the wiki. On echo2 and cycle1 the share of variance between genomes (icc) is 0.00. The split-half correlation is -0.46 on echo2 and 0.14 on cycle1. Among 32 genomes there is nothing for selection to climb.

So the stalls are not mainly about tuning. Evolution here is a good finder of reflexes and wiring, and it has not yet found a lifetime learner. A general agent is almost entirely a lifetime learner.

## The search itself is small and under-replicated

Each run uses 9 islands of 32 genomes, an elite of 8, 24 training worlds of 200 trials, 48 held-out worlds and 600 generations. That is about 19,000 genome evaluations per island per run, with no exchange between islands since run 22. NEAT-era work used similar budgets, but on tasks with dense gradients. Lexicase selection is known to need large populations (hundreds to thousands) to keep its specialists.

At n=32, epsilon is the median absolute deviation over the population. On the maze it often comes out as 0 with most genomes tied (signal section), so a case either selects nothing or selects at random. Converging by 74 and then stalling for 460 generations is the textbook sign of lost diversity. Plasmids have helped on arrival 4-13% of the time and have never made a lasting difference.

Thrift is stronger than it looks. `lexicase_breed` adds one thrift case per world. A rung with one goal probe therefore has 24 probe cases and 24 thrift cases, and half of all case orderings start by filtering for cheapness. That explains elites of 7 to 68 populations. Removing it is no fix either: run 23 bloated to 684 populations and passed 4 rungs against 8. Neither extreme is what is wanted. What is wanted is to keep large and small networks apart, and size classes (run 26: 21 rungs, the best result) did exactly that.

Size classes are a crude form of quality-diversity. The obvious next step is a real MAP-Elites archive over behaviour descriptors such as size, work, ports used and loop share, with thrift as one descriptor rather than half the cases.

The methodology also has a hole. Lesson 30 says seed noise is large and asks for 5 or more seeds. Yet every run compared in island-runs.md is a single evolutionary seed, and the 11-to-21 spread in rungs passed has no error bar. Runs 24 to 30 each stack one switch on the one before, so the conclusions ("size classes help", "branches give a fast start") are n=1 with path dependence. The human is the outer loop of the search, one hand-designed switch per run. That is the opposite of "improvements come from evolution".

## The curriculum

The ladders are hand-written, and the step rungs show their limit. feel-100, glow-150 and the mirrored echo rungs were all climbable, and none carried over: echo2m4 was passed, then echo2 sat at 0. A smaller step helps only when the remaining gap is quantitative. When the gap is a new kind of thing (learn the code in the lifetime), splitting the goal number does not create a gradient.

Run 30 has its own red flag. Switching and shapes, both on feel-100, score exactly 0 on training and on held-out worlds for 50 or more generations. The fixed body seems to have removed what feel measured. A blind-control check is needed before any reading of run 30.

## Can this kind of evolution reach general agents?

Not as built. The fundamental obstacle is that the learning rule is not in the genome. Hebbian and anti-Hebbian moves of one bit, births and inhibition are fixed in the runtime; the genome only places them and gates them. Evolution is asked to find a general learner while being barred from changing how learning works.

The literature on evolved plasticity points the same way: Soltoggio and Risi, Najarro and Risi's evolved Hebbian rules, and Kirsch and Schmidhuber's VSML and general-purpose in-context learners. A general learner emerges only when two things hold together:
- **The learning machinery is expressible in the evolved substrate.** VSML: the same cells do learning and acting.
- **The task distribution is wide enough** that memorising stops paying. Kirsch et al. found a sharp transition only past thousands of distinct tasks.

Here each rung is one task family with a few tokens.

The existing lifetime learner sets the ceiling. On raw Alice text it predicts 50.1% of next bytes against 53.9% for a frequency lookup. Conversation needs millions of tokens of lifetime exposure feeding a learner that beats n-grams by a wide margin.

My candid estimate: at this compute, a conversing or computer-using agent will not come from this search, by orders of magnitude. The plausible route is for evolution to discover the learner (the rules, the gates, the growth program). Lifetime learning on a long, rich stream then does the rest, with an LLM inside the world as teacher and as a source of endless graded tasks (allowed as world, not agent).

## What must change

1. **Make plasticity bit-math the genome can evolve.** Masks should be vectors that kernels may write with the existing ops: AND-NOT to drop a line, OR to add one, XOR to move one. A learning rule then becomes a circuit, and evolution can find rules other than one-bit Hebbian moves. This stays inside the primitives.
2. **Use open-ended task generation instead of fixed ladders.** Draw tasks from a parameterised family (number of tokens, lag, contingency, mapping redrawn each lifetime), choose them by learning progress, and keep a minimal-criterion archive in the manner of POET. Fitness should reward the gain from early to late in a lifetime, which measures learning directly, rather than end scores.
3. **Use quality-diversity and far more evaluations.** Populations of hundreds per niche, a MAP-Elites archive, and evaluations spread over many machines. This is a budget 10-100 times today's.

## The next experiments

1. **Find the noise floor.** Run the run-26 configuration (and run 29's) for 300 generations on 4 evolutionary seeds each. If rungs passed varies by plus or minus 5, most of the switches from runs 24 to 30 are unproven. This costs less than one further switch run.
2. **Scale on the stalled rungs.** Run only echo2 and cycle1 islands at populations of 32, 128 and 512 with total evaluations held equal, and track the icc. If no size gives a gradient, the grammar cannot express the needed learner, and adding compute will not fix it.
3. **Evolve a learning rule.** Add mask-writing kernels, and start from a seed with built-in learning turned off. Ask first whether evolution rediscovers anything Hebbian on lag and feel. Then ask whether it beats the built-in rule on echo2 when every lifetime redraws the sound codes across many token sets.

A yes on experiment 3 is the first real evidence that this line can evolve learners rather than reflexes. A no means the substrate needs rethinking before anything is scaled.
