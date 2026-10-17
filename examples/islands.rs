//! Islands: populations pressed on different capacities, trading circuits.
//!
//! Each island is the tag genome from a naked start, evolving by epsilon-lexicase on its own
//! probes (`observe::probes`):
//! - switching: respond, tell, discriminate, reverse, win-stay (serial reversal);
//! - recall: respond, remember (a light gone a few steps ago);
//! - reading: copy (say the token shown on the eye a few steps before) and echo (say the token
//!   the teacher said a few steps before: the stepping stone, sound to voice with no glyph);
//! - generalist: all ten (surprise on a cycle of 3 broken one time in 8);
//! - prediction (`ISL_PREDICTION=1`): surprise, clicking after a token that broke the stream's
//!   cycle;
//! - maze (`ISL_MAZE=1`): a T-maze, turning at the junction to the side of the food the cue shows;
//! - screen (`ISL_SCREEN=1`): move a cursor over a grid of cells to the target glyph and click,
//!   later round walls (a 2-D maze);
//! - shapes (`ISL_SHAPES=1`): respond, tell and discriminate with the lights distorted into
//!   shapes, a fresh exemplar every trial.
//! Every `ISL_EVERY` generations each island takes `ISL_PLASMIDS` plasmids (one population of a
//! donor with the circuit of any channel it touches, made private; or the population alone if it
//! only reads and writes ports) from the best genomes of other islands into its new children, so a circuit grown under one pressure is tried under another. Each plasmid is
//! recorded (from, to, the circuit) in `ISL_LOG`; at each exchange every island's best genome is
//! scored on every probe on held-out worlds (does a circuit carry over?).
//!
//! Each plasmid is numbered (`#n`): its populations' ids carry the number, so its carriers can be
//! counted every generation until none is left (mutation and one-point crossover keep ids; a
//! copied segment or a circuit taken within an island by crossover gets new ones, so spread inside
//! an island is undercounted). At the exchange each is measured (populations, channels, kernels,
//! wires, what it reads and writes, what could hold state) and classified: its role from its
//! wiring (reflex, sensor, driver, babbler, relay, sealed), what it scores alone against its blind
//! control (`ISL_PLASMID_WORLDS` held-out worlds, 16), and the recipient's stage probes with it and
//! without it (helps, hurts, mixed: gains on some probes and losses on others, or neutral: a
//! change of `NOTICE`).
//!
//! As bacteria keep plasmids: `ISL_PLASMID_MAX` (0: any size) cuts a circuit of more populations
//! to the chosen one and its nearest neighbours in it (a small mobile unit, not a busy channel's
//! tangle); `ISL_EXCLUDE=1` sends nothing to a recipient that already carries every population
//! of it (one plasmid of a kind); `ISL_CARRY` (0) charges that much work per step for each token
//! a genome carries, so what does nothing costs something.
//!
//! `ISL_OPS=1`: the op choice. A population's op is drawn again by a mutation of its own (one in
//! sixteen), not only as one of its many properties, and ops are drawn without `rotr` (the same as
//! `rotl` by 64 - n), so `clear`, `xor` and the rest are tried far more often.
//!
//! Learning from reward has a rung of its own (`repeat`), below switching's and shapes'
//! win-stay and maze's turn: a maze with no stem, the food on one side all lifetime and no cue
//! (ladder keys `fixed` and `nocue`), so which turn pays is learned from the stroke alone; the
//! `repeat` probe scores a learner high and a turner that ignores strokes 0. Win-stay, with the
//! probe fixed, averaged about 0 in run 9: no genome learned from reward.
//!
//! `ISL_EPI=1`: epigenetic marks. Each population's lifetimes count its predictions that came
//! true; its mark becomes half the old one plus 16 for each doubling, children inherit it, and
//! mutation duplicates a marked population more often and deletes it less often.
//! `ISL_EXPLORE=1` (with `ISL_EPI`): a population whose kernels all go 64 steps without firing
//! grows, once a lifetime, a trial connection (an XOR population reading its destination and
//! writing back into its source, or the destination); one whose predictions come true 16 times
//! or more over a generation's lifetimes is added to the genome, tagged `explored`.
//!
//! `ISL_ZEROS=1`: two-mask kernels in the grammar. A population may expect (a property drawn for
//! one new population in four, and flipped by mutation): each kernel then expects a value on each
//! of its lines, ones and zeros alike, so it can match the absence of input (a click that went
//! unstroked). Born on half on lines and half off, expecting what they held; starting kernels
//! expect every line off. Without it, a kernel counts only lines that are on.
//!
//! `ISL_ACCIDENT=1` (with `ISL_EXPLORE`): freak accidents. A new population draws its own rate
//! (`accident`, one every 2^12 to 2^18 steps; mutation moves it a step) and grows its trial
//! connection by accident at that rate even when it is not stuck, still once a lifetime and kept
//! the same way.
//!
//! `ISL_PORTGATE=1`: ports for gates, pauses and writes. A population's gate (it learns only while
//! X had lines on, or none) and its pause (it runs only while X ...) may name an input port (a
//! sense, or the self channel) or an output port (the act, voice or hand), as well as a declared
//! channel; a population writing outward may write a sense instead (what it writes is sensed
//! with the world's input a step later; the world never reads it). So one population can learn
//! only after a stroke, with no channel copying the affect first.
//!
//! `ISL_SHADOW=1` (with `ISL_EXPLORE`): shadow explorers. A stuck (or, with accidents, an
//! unlucky) population grows a shadow instead of a trial XOR loop: it reads what the population
//! reads and predicts, a step ahead, one vector after another (a sense, the self channel, the
//! voice or hand, a channel), writing nothing into the network. Each step it checks: a kernel
//! whose residue (prediction XOR what came on, on its lines) is under a quarter of its lines
//! (`hit_quarters`, a gene) is reinforced (Hebbian on what it fired on), one that missed is
//! weakened (anti-Hebbian) and moves its output toward what came on, and unpredicted lines recruit
//! kernels. A target it cannot predict within 128 steps is dropped for the next. Over a
//! generation, the best-predicted target (net 64 true) is kept: a channel of its own and a
//! predictor into it, still learning against the target, tagged explored: an internal copy of
//! what that vector will be, for other populations to read or gate on.
//!
//! `ISL_PAUSES=1`: pauses drawn as gates are. A new population is paused (runs only while a
//! channel, or with `ISL_PORTGATE` a port, had lines on or none) one time in four, and a pause made
//! by mutation draws what it names in the same step, so a port pause is one step away, not two.
//!
//! `ISL_RECALL_LONG=1`: recall's long ladder. After lag4: lag5, lag8 and lag16 (a delayed trial
//! lengthened to fit its window), distract (the other light flashes once during the delay on half
//! the trials and does not pay), vary (each trial's delay drawn from 1 to 8: the memory held until
//! the window opens, not timed), two (the other light shows on no-cue trials: which light came).
//!
//! `ISL_SHADOW_HINT=1` (with `ISL_SHADOW`): shadows as indicators. A shadow starts only by
//! accident (each population's own rate), not when its population is stuck; what it predicts
//! well over a generation is not built but noted on the population as a hint (shared information
//! between its input and that target), which its gate or pause takes, half the time, when they
//! mutate. Nothing is added to the network.
//!
//! `ISL_HINT_USE=1` (with `ISL_SHADOW_HINT`): edges built on hints. A mutation of its own, one
//! in sixteen while any population carries a hint, adds a new population of any kind on a hinted
//! path: from the hinted population's input to the hinted vector, or from the hinted vector to its
//! destination. A hint makes an edge likelier there; the edge's kind is drawn as usual. Where one
//! way feeds back (closes a loop, or runs from a higher level to a lower) and the other does not,
//! it goes the way that feeds back three times in four. An output port is read too (the act a step
//! ago), so a hint at the hand can also go from the hand to the hinted population's destination.
//!
//! `ISL_READ_OUT=1`: any population may read an output port (the act, voice or hand a step ago,
//! an efference copy): a new one reading from outside does one time in four, and a property
//! mutation switches between an input and an output port.
//!
//! `ISL_HINT_BAND=1` (with `ISL_SHADOW_HINT`): a shadow scores a target by its kernels that are
//! partly right (more than `hint_low` quarters of their lines came on, a gene, and fewer than
//! `hit_quarters`), so hints mark what the source is correlated with, not what it determines.
//!
//! `ISL_SELF_WRITE=1`: the self channel may be written like a sense (on top of what the runtime
//! writes there each step).
//!
//! `ISL_LOOPS=1`: a mutation of its own (one in sixteen) closes a loop: a population's path is
//! walked forward 1 to 4 populations, ports included, and a new population of any kind is added
//! from where it ends back to where it starts.
//!
//! `ISL_CARRY_ENERGY=1` (with `ISL_RATION`): the carrying cost is energy. Each step a network
//! pays `ISL_CARRY` a token of its genome from its energy store, the store its work comes out of,
//! so carrying and computing compete for what it earns; the cost is no longer added to its work.
//!
//! `ISL_SIGNAL=n` (with `ISL_RESUME`): measure only. Each island's resumed genomes are scored on n
//! fresh worlds on their rung's goal probe and on the rung below, and the spread between genomes
//! is set against the noise across worlds (icc, split-half r, epsilon). Nothing evolves.
//!
//! `ISL_ELITE_PASS=1`: a rung's goal is checked on the elite's mean on the training worlds (its
//! best `ISL_ELITE` genomes), not the whole population's, before the elite's held-out check. The
//! goals line still shows the population's mean.
//!
//! `ISL_PARTIAL=1`: partial credit. In selection, echo, surprise and feel are replaced by graded
//! versions (echo~, surprise~, feel~: a late or near answer counts in part); the rung's goal is
//! still checked on the probe itself.
//!
//! `ISL_INHIBIT=1`: inhibit inputs. A population may carry one (a channel or a port): each step
//! the lines it had on a frame ago are taken before any of the population's kernels fires, so a
//! kernel whose mask lies more than half on them is held down. A new population has one one time
//! in four.
//!
//! `ISL_OVERLAP=1`: the overlap rule as a gene. A population's `inhibit_quarters` (1 to 4,
//! starting at 2) sets how much of a kernel's mask taken lines must cover to hold it down (more
//! than that many quarters; 4: never), for its own kernels and its inhibit input alike.
//!
//! `ISL_KEEP_RUNGS=1`: passed rungs kept. Each generation an island above its first rung is also
//! scored on one rung it passed (drawn at random, in that rung's world), and those scores join its
//! selection cases; the rung's goal is still judged on the current rung alone.
//!
//! `ISL_SIZE_CLASSES=1`: size classes. Each island's genomes are sorted by size into classes a
//! factor of four apart (1-3 tokens, 4-15, 16-63, 64-255, 256+); each occupied class keeps its own
//! lexicase choice among the elite, and half of all children have a parent chosen within a class
//! drawn at random, so small and large networks both stay in the population. The op lines add how
//! many genomes are in each class.
//!
//! `ISL_THRIFT_RELAX=<scale>` (with thrift): thrift relaxed while stuck. Each selection takes the
//! thrift cases with chance scale / (scale + g), g the generations the island has been on its
//! current rung (64: half after 64 generations, a fifth after 256), and full again after a pass,
//! so small networks pass the easy rungs and networks may grow on a rung that holds them up. The
//! count of generations on a rung starts again at 0 when a run is resumed from a checkpoint.
//!
//! `ISL_RUNG_LOG=<file>`: each time an island passes a rung, its best genome is appended to the
//! file (headed `island <name> generation <g> stage <s> passed <rung>`), so the network that passed
//! each rung is kept. Only a record: it changes nothing in the run.
//!
//! `ISL_HEAT=1`: heat inputs. A population may carry one (a channel or a port): each step every
//! one of its kernels' thresholds moves by its own threshold times how full that vector was a
//! frame ago, down when warm, up when cool. A new population has one one time in four.
//!
//! `ISL_MIRROR=1`: easier rungs before reading's echo2, the same lesson (two tokens, every one
//! said by the teacher) with each token's sound made like the voice's code for it, in eighths of
//! its lines (`World::mirror_sounds`): echo2q (heard exactly as said, and one's own voice not
//! heard: one wire from the ear to the voice scores 503), echo2m8 (heard exactly as said, one's
//! own voice heard again: a copy now locks onto one token, scoring 0, unless what the voice said
//! is taken away: the ear at lag 2 ORed into the voice and the voice at lag 3 cleared from it
//! scores 431), echo2m4 (half the lines shared, the least a copy can say: 427 with the same two
//! wires), then echo2 (nothing shared: the mapping is learned in the lifetime). Goals as echo2's.
//! `examples/mirrorcheck.rs` scores the wires in `runs/echo-wires/`.
//!
//! `ISL_FEEL_GOAL=<score>` (300): the feel rung's goal. No network can reach 300 on average: in
//! feel nothing says which hand code is the click, and moving at random after a stroke scores
//! about 204 (`examples/feelcheck.rs`); 160 is passed by that.
//!
//! `ISL_WIRE_COST=<k>`: the wiring cost. Each step a genome is charged k for each bit of Hamming
//! distance between the tags of the two ends of each of its populations (a channel's tag, or a
//! port's own: ports and channels share one tag space for this), as the carrying cost is charged.
//! Near ends are cheap and far ones dear, so circuits that work together gather in one part of
//! tag space, near the ports they use: parts. A loop on one channel is free.
//!
//! `ISL_COMPETE=1`: populations compete for their destination. At each destination the
//! populations writing it run strongest hold first (with `ISL_PRUNE` their grip, without it their
//! best kernel's share the step before; ties in the genome's order), and
//! each word an earlier population's kernels wrote this step is claimed: a kernel more than half
//! (its overlap rule) of whose output words are claimed is held down and learns anti-Hebbian, as
//! kernels of one population are where their inputs overlap. A move or a token is a code in one
//! word, so one population writes each: the winners write the act, not the OR of every reflex.
//! Wires and noise do not compete.
//!
//! `ISL_PRUNE=1` (with `ISL_COMPETE`): a population lets go of its destination when it keeps
//! losing. Its hold (the gene `grip`, 16 to start, halved or doubled by mutation) goes up one each
//! step it writes there and down one each step it only loses; at 0 it runs no more that lifetime.
//! Every lifetime starts wired broadly and keeps the winners, as a newborn's muscles end with one
//! input each; the genome is not changed.
//!
//! `ISL_BRANCH=1`: branches. A population's fired kernels may also write their outputs into a
//! second vector (a channel or an output port), at the same place, by OR, as an axon's
//! collaterals: a copy of what it did for others to read, a line to another population's inhibit
//! input, two targets driven by one decision. A new population has one one time in four; with
//! competition a branch writes only words no other population claimed, and claims them. The op
//! lines say how many of the best's populations branch.
//!
//! The op lines (with `ISL_COMPETE`) also give the contest: kernels held down by another
//! population's claim, and populations that let go, per 1000 steps of the island's probing.
//!
//! `ISL_LESION=n` (with `ISL_RESUME`): measure only. Each island's best genome is scored on n fresh
//! worlds on each of its subtasks (the generalist: every other island's rung), then with each of
//! its populations taken out in turn: the loss on each subtask as a share of its rung's goal, and
//! how much of each population's loss falls on one subtask (near 1: a part for that subtask; near
//! 1/subtasks: tangled).
//!
//! `ISL_STEPS=<rung>=<goal>+<goal>,...`: steps up to a rung. Before each rung of that name (on
//! every ladder that has one) copies of it are put in, the same world and probes, with these
//! goals in turn: `feel=100+200` makes feel-100 > feel-200 > feel. For rungs no run passes.
//!
//! `ISL_FEEL=1`: a feel rung before each repeat rung (switching's, shapes' and the maze's learn):
//! strokes come at random whatever the program does, and the `feel` probe scores clicking in the
//! 3 steps after one. One circuit, from the affect port to the hand. The repeat rung needs two
//! (affect copied into a channel, and a repeating population gated on it: a gate names a
//! channel, not a port), and run 10's genomes chose on almost every trial but repeated a turn
//! as often after a miss as after a stroke.
//!
//! `ISL_SELECTION=1`: a ninth island, task selection: one lifetime with one network, blocks that
//! alternate between responding (click at a light) and babbling (vary the voice), scored by
//! `select` (both tasks done, minus doing the other's thing); its ladder starts with each block's
//! task felt as a cue, then none, then short blocks.
//!
//! `ISL_FRONTIER=1`: the generalist is scored at the frontier, on each other island's current
//! stage (its probes, in its world; each probe from the first island that has it), not on all
//! twelve probes in the plain world. In the plain world most probes are the top of another
//! island's ladder and score about 0 for every genome (run 8's best: babble 845, tell 293, the
//! other nine 0-13), so lexicase found a gradient only in babble and tell.
//!
//! `ISL_FRESH=1`: the training worlds are new every generation (the same for every genome in
//! it). With fixed training worlds a population can fit them: run 6's maze elite scored 476-658
//! on its 12 training worlds and -69 to -177 on fresh ones, having wired the hand's codes those
//! worlds drew instead of learning them each lifetime.
//!
//! `ISL_RATION` (0: none): energy. Each lifetime step a network gets this much and pays its work;
//! it eats what it scores: every 50 steps its last block is scored by its task's probes, and each
//! 1024 points are `ISL_FOOD` (32) rations. With its store below 0 it rests (does not run or act)
//! until the ration refills it. Compute beyond the ration must be earned by scoring, so cost and
//! score have to make sense together, and a runaway genome no longer sets a generation's time.
//!
//! Every `ISL_REPORT` generations, per island: its population's means on its own probes (training
//! worlds), its best genome's probes on held-out worlds, its work per step and which ports its
//! genomes read and write.
//!
//! `ISL_POP` (16 a island), `ISL_ELITE` (4), `ISL_GENS` (300), `ISL_LIVES` (12 training worlds),
//! `ISL_HELD` (12), `ISL_TRIALS` (200), `ISL_GAP` (6), `ISL_EVERY` (20), `ISL_PLASMIDS` (2),
//! `ISL_REPORT` (10), `ISL_LOG` (plasmids.txt), `ISL_SEED` (7), `ISL_THRIFT` (1), `ISL_READS` (0; 1: the words a kernel
//! population reads count as its work, so thrift sees a wide read), `ISL_INPLACE` (0; 1: every declared channel held in
//! place, one frame, never cleared or copied: only clearing writes take lines away, and a write
//! is seen by the populations after it in the same step; no history: every lag reads it as it is
//! now, so a delay is built by the network, e.g. a `set` wire placed before the write it delays,
//! or a latch: a `set` population that runs only while an enable channel is on).
//!
//! Curricula (`ISL_LADDERS=1`): an island can climb a ladder of stages, each its own probes, its
//! own world (`Lesson`: tokens and how often the teacher speaks) and a goal; it moves to the next
//! stage once its population's training mean meets the goal for `hold` generations in a row
//! (`evolve_islands`). Switching climbs respond, tell, win-stay, discriminate (no swaps), then one
//! swap, then three; recall clicks at a light, then after one gone 1, 2, 3 (the plain task), then 4 steps
//! before the window. Reading's: babble, then echo with 2, 4 and 8 tokens
//! with the teacher on every token, then the teacher fading (6, 4, 2, 1 in 8) so the glyph must
//! carry more of the work. `ISL_LADDER_<island>` replaces an island's ladder: stages separated by
//! `;`, each `name key=value ...` with keys `probes` (probe names joined by `+`), `tokens`,
//! `teach` (eighths), `period` and `noise` (surprise's stream), `blur` and `shift` (shapes), `maze`, `gone` and `symbol` (the T-maze), `sw`, `sh`, `lit`, `cell`, `walls`, `torch` and `dark` (the screen), `swaps` (of the paying light in a reversal lifetime), `delay` (steps from the
//! delayed light to the response window), `goal` (`probe@score`, joined by `+`) and `hold` (`ISL_HOLD`, 3: generations in a row the goal is met on the training worlds before the held-out check). The cross-island table
//! at each exchange is always in the plain world, on every probe.
//!
//! Checkpoints (also every `ISL_CKPT` generations, if set): at every exchange each island's whole population is written to `ISL_CHECKPOINT`
//! (checkpoint.txt, as text); `ISL_RESUME=<file>` starts every island named in it from its saved
//! population and generation (islands not in it start naked), so a change to the world or the
//! probes can be injected into a running experiment: stop it, change, resume.

use std::cell::{Cell, RefCell};
use std::io::Write;

use neurocomp::evolve::tags::{Pop, TagGenome, Token};
use neurocomp::evolve::{evolve_islands, probe_all_in, probe_means, Island, Lesson, Physics, Stage, ALL_PROBES};
use neurocomp::world::observe::{ProbeRow, PROBES};
use rand::rngs::StdRng;
use rand::Rng;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn row(m: &ProbeRow, only: &[usize]) -> String {
    (0..m.len()).map(|p| if only.contains(&p) { format!("{:>7}", m[p]) } else { format!("{:>7}", ".") }).collect::<Vec<_>>().join("")
}

/// A moved population's id: the top bit, its plasmid's number and its place in the plasmid.
const TAGGED: u32 = 1 << 31;

fn plasmid_of(id: u32) -> Option<u32> {
    (id & TAGGED != 0).then_some((id & !TAGGED) >> 8)
}

/// A probe score change this large counts (a single genome on a few worlds is noisy).
const NOTICE: i32 = 40;

/// A plasmid moved between islands, followed until no genome of its island carries it.
struct Moved {
    n: u32,
    /// the donor island (none: moved before a resume)
    from: Option<usize>,
    to: usize,
    at: usize,
    ids: Vec<u32>,
    peak: usize,
    alive: bool,
    kind: String,
}

/// Whether a genome carries any of a plasmid's populations, and all of them.
fn carries(g: &TagGenome, ids: &[u32]) -> (bool, bool) {
    let has: Vec<bool> = ids.iter().map(|id| g.tokens.iter().any(|t| matches!(t, Token::Pop(p) if p.id == *id))).collect();
    (has.iter().any(|&h| h), has.iter().all(|&h| h))
}

/// What a plasmid is made of: its size, what it reads and writes (ports, or "host": channels it
/// names but does not declare, which resolve among the recipient's), and what could hold state;
/// and its role from that wiring.
fn measure(tokens: &[Token]) -> (String, &'static str) {
    let g = TagGenome { tokens: tokens.to_vec(), next_id: 0 };
    let declared: Vec<u8> = tokens.iter().filter_map(|t| if let Token::Channel { tag, .. } = t { Some(*tag) } else { None }).collect();
    let pops: Vec<&Pop> = tokens.iter().filter_map(|t| if let Token::Pop(p) = t { Some(p) } else { None }).collect();
    let host = |t: u8| !declared.contains(&t);
    let (mut reads, mut writes) = g.ports();
    if pops.iter().any(|p| (!p.from_port && host(p.from)) || p.gate.is_some_and(|(t, _)| host(t)) || p.pause.is_some_and(|(t, _)| host(t))) {
        reads.push("host".into());
    }
    if pops.iter().any(|p| !p.to_port && host(p.to)) {
        writes.push("host".into());
    }
    let mut memory = Vec::new();
    let holds = tokens.iter().filter(|t| matches!(t, Token::Channel { hold: true, .. })).count();
    if holds > 0 { memory.push("holds") }
    if pops.iter().any(|p| !p.from_port && !p.to_port && p.from == p.to) { memory.push("loop") }
    if pops.iter().any(|p| p.lag > 1 || p.span > 1) { memory.push("history") }
    if pops.iter().any(|p| p.repeat) { memory.push("repeat") }
    if pops.iter().any(|p| p.grow) { memory.push("grows") }
    let gated = pops.iter().any(|p| p.gate.is_some() || p.pause.is_some());
    let kernels: usize = pops.iter().filter(|p| !p.wire).map(|p| p.count as usize).sum();
    let wires = pops.iter().filter(|p| p.wire).count();
    let sense = reads.iter().any(|r| ["S", "eye", "ear", "touch", "affect"].contains(&r.as_str()));
    let act = writes.iter().any(|w| ["A", "voice", "hand"].contains(&w.as_str()));
    let (host_in, host_out) = (reads.iter().any(|r| r == "host"), writes.iter().any(|w| w == "host"));
    let role = match (sense, act) {
        (true, true) => "reflex (senses to act)",
        (true, false) if host_out => "sensor (senses into the host)",
        (true, false) => "sealed sensor (reaches nothing)",
        (false, true) if host_in => "driver (host to act)",
        (false, true) => "babbler (acts on no sense)",
        (false, false) if host_in && host_out => "relay (host to host)",
        _ => "sealed (reaches nothing)",
    };
    let list = |v: &[String]| if v.is_empty() { "nothing".to_string() } else { v.join("+") };
    let text = format!(
        "{} populations, {} channels{}, {kernels} kernels, {wires} wires; reads {}; writes {}; memory {}{}",
        pops.len(), declared.len(), if holds > 0 { format!(" ({holds} hold)") } else { String::new() },
        list(&reads), list(&writes), if memory.is_empty() { "none".to_string() } else { memory.join("+") }, if gated { "; gated" } else { "" },
    );
    (text, role)
}

/// The ops some genomes write with, most used first (share of their populations), and how
/// recursive the first is (`TagGenome::loops`): populations on a loop, self-loops among them, and
/// the largest loop's channels.
fn ops_and_loops(genomes: &[TagGenome]) -> String {
    let mut count: Vec<(String, usize)> = Vec::new();
    let mut total = 0usize;
    for g in genomes {
        for t in &g.tokens {
            if let Token::Pop(p) = t {
                let name = p.op.text().split(':').next().unwrap_or("").to_string();
                match count.iter_mut().find(|c| c.0 == name) {
                    Some(c) => c.1 += 1,
                    None => count.push((name, 1)),
                }
                total += 1;
            }
        }
    }
    count.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let ops: Vec<String> = count.iter().take(5).map(|(n, k)| format!("{n} {}%", k * 100 / total.max(1))).collect();
    let (on, selfs, largest, pops) = genomes.first().map_or((0, 0, 0, 0), |g| g.loops());
    // (explored: populations kept from exploration; marked: populations with a mark, and the mean)
    let all: Vec<&Pop> = genomes.iter().flat_map(|g| g.tokens.iter().filter_map(|t| if let Token::Pop(p) = t { Some(p) } else { None })).collect();
    let explored = all.iter().filter(|p| p.explored).count();
    let marked: Vec<u32> = all.iter().filter(|p| p.mark > 0).map(|p| p.mark as u32).collect();
    let marks = if marked.is_empty() { String::new() } else { format!("; marked {} (mean {})", marked.len(), marked.iter().sum::<u32>() / marked.len() as u32) };
    // (with ISL_ZEROS: the share of kernel populations that expect zeros as well as ones)
    let kernels = all.iter().filter(|p| !p.wire).count();
    let expect = if neurocomp::evolve::tags::zeros() { format!("; expecting {}%", all.iter().filter(|p| p.expect && !p.wire).count() * 100 / kernels.max(1)) } else { String::new() };
    // (with ISL_ACCIDENT: the share of kernel populations prone to accidents, and their mean
    // exponent: one accident every 2^k steps)
    let prone: Vec<u32> = all.iter().filter(|p| !p.wire && p.accident > 0).map(|p| p.accident as u32).collect();
    let accident = if neurocomp::evolve::tags::accident() { format!("; accident-prone {}% (1 in 2^{})", prone.len() * 100 / kernels.max(1), prone.iter().sum::<u32>() / (prone.len() as u32).max(1)) } else { String::new() };
    // (with ISL_PORTGATE: populations gated or paused on a port, and writing a sense)
    let ported = if neurocomp::evolve::tags::port_gates() {
        let gp = all.iter().filter(|p| p.gate.is_some() && p.gate_at > 0).count();
        let pp = all.iter().filter(|p| p.pause.is_some() && p.pause_at > 0).count();
        let wi = all.iter().filter(|p| p.to_port && p.to_input).count();
        let ro = all.iter().filter(|p| p.from_port && p.from_output).count();
        let mut reading = if neurocomp::evolve::tags::hint_use() || neurocomp::evolve::tags::read_outputs() { format!(", reading an output {}%", ro * 100 / all.len().max(1)) } else { String::new() };
        // (with ISL_INHIBIT: populations with an inhibit input)
        if neurocomp::evolve::tags::inhibit_inputs() {
            let held = all.iter().filter(|p| !p.wire && p.inhibitor.is_some()).count();
            reading += &format!(", inhibited {}%", held * 100 / all.len().max(1));
        }
        // (with ISL_OVERLAP: the overlap rule, in quarters, among the kernel populations)
        if neurocomp::evolve::tags::overlap_genes() {
            let q: Vec<usize> = (1..=4).map(|k| all.iter().filter(|p| !p.wire && p.inhibit_quarters.clamp(1, 4) == k).count()).collect();
            reading += &format!(", overlap quarters 1:{} 2:{} 3:{} 4:{}", q[0], q[1], q[2], q[3]);
        }
        // (with ISL_HEAT: populations with a heat input, warm and cool)
        if neurocomp::evolve::tags::heat_inputs() {
            let warm = all.iter().filter(|p| !p.wire && p.heat.is_some_and(|h| h.1)).count();
            let cool = all.iter().filter(|p| !p.wire && p.heat.is_some_and(|h| !h.1)).count();
            reading += &format!(", heated {}% (warm {}, cool {})", (warm + cool) * 100 / all.len().max(1), warm, cool);
        }
        format!("; port-gated {}%, port-paused {}%, writing a sense {}%{reading}", gp * 100 / all.len().max(1), pp * 100 / all.len().max(1), wi * 100 / all.len().max(1))
    } else { String::new() };
    // (with ISL_SHADOW: predictors kept from shadow explorers, and what they predict)
    let predictors = if neurocomp::program::neurocomp::shadow() {
        let mut what: Vec<(String, usize)> = Vec::new();
        for p in all.iter().filter(|p| p.teacher.is_some()) {
            let (t, at) = p.teacher.unwrap();
            let name = match at { 1 => ["S", "eye", "ear", "touch", "affect", "M"][[0x00u8, 0x07, 0x38, 0xC1, 0xE0, 0x80].iter().enumerate().min_by_key(|p| (*p.1 ^ t).count_ones()).unwrap().0].to_string(), 2 => ["A", "voice", "hand"][[0xFFu8, 0x1F, 0xF8].iter().enumerate().min_by_key(|p| (*p.1 ^ t).count_ones()).unwrap().0].to_string(), _ => "channel".to_string() };
            match what.iter_mut().find(|w| w.0 == name) { Some(w) => w.1 += 1, None => what.push((name, 1)) }
        }
        what.sort_by(|a, b| b.1.cmp(&a.1));
        format!("; predictors {}{}", what.iter().map(|w| w.1).sum::<usize>(), if what.is_empty() { String::new() } else { format!(" ({})", what.iter().map(|w| format!("{} {}", w.0, w.1)).collect::<Vec<_>>().join(", ")) })
    } else { String::new() };
    // (with ISL_SHADOW_HINT: populations carrying a hint, and how many gate or pause on it)
    let hints = if neurocomp::evolve::tags::shadow_hints() {
        let h = all.iter().filter(|p| p.hint.is_some()).count();
        let used = all.iter().filter(|p| p.hint.is_some_and(|x| p.gate.is_some_and(|g| g.0 == x.0 && p.gate_at == x.1) || p.pause.is_some_and(|g| g.0 == x.0 && p.pause_at == x.1))).count();
        format!("; hinted {}% (gated or paused on the hint {})", h * 100 / all.len().max(1), used)
    } else { String::new() };
    format!("ops {}; loops: {on} of {pops} wired populations on a loop ({selfs} self-loops), the largest {largest} channels; explored {explored}{marks}{expect}{accident}{ported}{predictors}{hints}", if ops.is_empty() { "none".into() } else { ops.join(", ") })
}

/// The probes where `a` scores at least `NOTICE` above `b`, named.
fn above(a: &ProbeRow, b: &ProbeRow, only: &[usize]) -> Vec<String> {
    only.iter().filter(|&&p| a[p] - b[p] >= NOTICE).map(|&p| PROBES[p].to_string()).collect()
}

/// `name a/b` for each probe in `only` where either is non-zero.
fn pairs(a: &ProbeRow, b: &ProbeRow, only: &[usize]) -> String {
    let v: Vec<String> = only.iter().filter(|&&p| a[p] != 0 || b[p] != 0).map(|&p| format!("{} {}/{}", PROBES[p], a[p], b[p])).collect();
    if v.is_empty() { "all 0".into() } else { v.join(" ") }
}

/// A probe by (the start of) its name.
fn probe_named(n: &str) -> usize {
    PROBES.iter().position(|p| p.starts_with(n)).unwrap_or_else(|| panic!("no probe named {n}"))
}

/// A ladder from its text (see the header).
fn ladder(text: &str) -> Vec<Stage> {
    text.split(';').filter(|x| !x.trim().is_empty()).map(|st| {
        let mut it = st.split_whitespace();
        let name = it.next().unwrap_or("stage").to_string();
        let mut stage = Stage { name, probes: vec![], lesson: Lesson::default(), goal: vec![], hold: env("ISL_HOLD", 3) };
        for kv in it {
            let (k, v) = kv.split_once('=').expect("key=value");
            match k {
                "probes" => stage.probes = v.split('+').map(probe_named).collect(),
                "tokens" => stage.lesson.tokens = v.parse().expect("tokens"),
                "teach" => stage.lesson.teach = v.parse().expect("teach"),
                "swaps" => stage.lesson.swaps = v.parse().expect("swaps"),
                "delay" => stage.lesson.delay = v.parse().expect("delay"),
                "vary" => stage.lesson.vary_delay = v.parse().expect("vary"),
                "distract" => stage.lesson.distract = v == "1",
                "two" => stage.lesson.two_lights = v == "1",
                "period" => stage.lesson.period = v.parse().expect("period"),
                "noise" => stage.lesson.noise = v.parse().expect("noise"),
                "blur" => stage.lesson.blur = v.parse().expect("blur"),
                "shift" => stage.lesson.shift = v.parse().expect("shift"),
                "maze" => stage.lesson.maze_len = v.parse().expect("maze"),
                "gone" => stage.lesson.cue_gone = v == "1",
                "symbol" => stage.lesson.symbol = v == "1",
                "fixed" => stage.lesson.maze_fixed = v == "1",
                "nocue" => stage.lesson.maze_nocue = v == "1",
                "mirror" => stage.lesson.mirror = v.parse().expect("mirror"),
                "quiet" => stage.lesson.quiet = v == "1",
                "motor" => stage.lesson.motor_place = v == "1",
                "babble" => stage.lesson.babble = v.parse().expect("babble"),
                "senses" => stage.lesson.sense_place = v == "1",
                "cue" => stage.lesson.mixed_cue = v == "1",
                "block" => stage.lesson.mixed_block = v.parse().expect("block"),
                "sw" => stage.lesson.screen_w = v.parse().expect("sw"),
                "sh" => stage.lesson.screen_h = v.parse().expect("sh"),
                "lit" => stage.lesson.screen_lit = v.parse().expect("lit"),
                "cell" => stage.lesson.screen_cell = v.parse().expect("cell"),
                "walls" => stage.lesson.screen_walls = v.parse().expect("walls"),
                "torch" => stage.lesson.screen_torch = v == "1",
                "dark" => stage.lesson.screen_dark = v == "1",
                "hold" => stage.hold = v.parse().expect("hold"),
                "goal" => stage.goal = v.split('+').map(|g| {
                    let (p, x) = g.split_once('@').expect("probe@score");
                    (probe_named(p), x.parse().expect("score"))
                }).collect(),
                _ => panic!("unknown key {k}"),
            }
        }
        stage
    }).collect()
}

/// Reading's ladder: a varied voice, then echoing (sound to voice, no glyph) with more and more
/// tokens, then the teacher fading so the glyph must carry more of the work.
const READING_LADDER: &str = "babble probes=babble tokens=2 teach=8 goal=babble@768; \
    echo2 probes=babble+echo tokens=2 teach=8 goal=echo@150; \
    echo4 probes=babble+echo tokens=4 teach=8 goal=echo@150; \
    echo8 probes=babble+echo tokens=8 teach=8 goal=echo@150; \
    fade6 probes=echo+copy tokens=8 teach=6 goal=copy@100; \
    fade4 probes=echo+copy tokens=8 teach=4 goal=copy@100; \
    fade2 probes=echo+copy tokens=8 teach=2 goal=copy@100; \
    read probes=echo+copy tokens=8 teach=1";

/// Switching's ladder: see a light, tell the two apart, repeat what was stroked, prefer the
/// paying light, then follow one swap, then three.
const SWITCHING_LADDER: &str = "respond probes=respond swaps=0 goal=respond@100; \
    tell probes=respond+tell swaps=0 goal=tell@100; \
    repeat probes=respond+tell+repeat swaps=0 maze=0 fixed=1 nocue=1 goal=repeat@300; \
    winstay probes=respond+tell+win-stay swaps=0 goal=win-stay@80; \
    discrim probes=respond+tell+win-stay+discriminate swaps=0 goal=discriminate@100; \
    swap1 probes=respond+tell+win-stay+discriminate+reverse swaps=1 goal=reverse@80; \
    swap3 probes=respond+tell+win-stay+discriminate+reverse swaps=3";

/// Recall's ladder: click at a light (respond, as switching starts: a population that never
/// clicks at a light scores 0 on every lag, and one that does passes lag1 already), then after a
/// light gone 1 step before the window, then 2, then 3 (the plain task), then 4.
const RECALL_LADDER: &str = "respond probes=respond swaps=0 goal=respond@100; \
    lag1 probes=respond+remember delay=1 goal=remember@150; \
    lag2 probes=respond+remember delay=2 goal=remember@150; \
    lag3 probes=respond+remember delay=3 goal=remember@150; \
    lag4 probes=respond+remember delay=4";

/// Recall's long ladder (`ISL_RECALL_LONG=1`): lag4 and on to longer delays (the trial lengthened
/// to fit), a distracting flash of the other light during the delay, a delay drawn per trial (the
/// memory held until the window, not timed), and two lights (which one came, not that one did).
const RECALL_LONG_LADDER: &str = "respond probes=respond swaps=0 goal=respond@100; \
    lag1 probes=respond+remember delay=1 goal=remember@150; \
    lag2 probes=respond+remember delay=2 goal=remember@150; \
    lag3 probes=respond+remember delay=3 goal=remember@150; \
    lag4 probes=respond+remember delay=4 goal=remember@150; \
    lag5 probes=respond+remember delay=5 goal=remember@150; \
    lag8 probes=respond+remember delay=8 goal=remember@150; \
    lag16 probes=respond+remember delay=16 goal=remember@150; \
    distract probes=respond+remember delay=8 distract=1 goal=remember@150; \
    vary probes=respond+remember vary=8 goal=remember@150; \
    two probes=respond+remember delay=8 two=1";

/// Prediction's ladder: a stream of tokens in a cycle, one in eight replaced by another; the
/// program is scored on clicking after a token that broke the cycle (surprise), with no code to
/// learn for any token. A break in a cycle of k is a token unlike the one k steps back, so each
/// stage asks for a longer reach into the past; the last ones need the two tokens before (a cycle
/// of 6 over 3 tokens repeats tokens) and a longer cycle.
const PREDICTION_LADDER: &str = "cycle1 probes=surprise period=1 noise=1 goal=surprise@200; \
    cycle2 probes=surprise period=2 noise=1 goal=surprise@200; \
    cycle3 probes=surprise period=3 noise=1 goal=surprise@200; \
    cycle4 probes=surprise period=4 noise=1 goal=surprise@200; \
    order2 probes=surprise period=6 tokens=3 noise=1 goal=surprise@150; \
    cycle8 probes=surprise period=8 noise=1";

/// Shapes' ladder: the two lights become shapes, a fresh exemplar every trial (each line moved
/// within its word with chance `blur` in 8, the whole pattern rotated by up to `shift` lines,
/// which carries it into other words and tiles), and the program must still tell them apart;
/// then repeat what was stroked (win-stay, as switching's ladder: shapes' best told the shapes
/// apart, tell 191 held out, but clicked the stroked one less, win-stay -112, discriminate
/// -108), then prefer the paying one, then follow one swap of it.
const SHAPES_LADDER: &str = "clean probes=respond+tell swaps=0 goal=tell@150; \
    blur1 probes=respond+tell swaps=0 blur=1 goal=tell@150; \
    blur2 probes=respond+tell swaps=0 blur=2 goal=tell@150; \
    shift8 probes=respond+tell swaps=0 blur=1 shift=8 goal=tell@120; \
    shift64 probes=respond+tell swaps=0 blur=2 shift=64 goal=tell@100; \
    repeat probes=respond+tell+repeat swaps=0 blur=2 shift=16 maze=0 fixed=1 nocue=1 goal=repeat@300; \
    winstay probes=respond+tell+win-stay swaps=0 blur=2 shift=16 goal=win-stay@80; \
    paying probes=respond+tell+win-stay+discriminate swaps=0 blur=2 shift=16 goal=discriminate@100; \
    swap1 probes=respond+tell+discriminate+reverse swaps=1 blur=2 shift=16";

/// Maze's ladder: a T-maze, stroked at the food's arm. Turn to the side the cue is on with no stem,
/// then walk a stem of 2 first, then with the cue only at the stem's foot (remember the side), a
/// longer stem, then a cue that is a light's code (which side it means is learned from the
/// stroke), then that code only at the foot. The hand's codes are drawn each lifetime, so even
/// the first stage is learned within a lifetime, not wired in.
const MAZE_LADDER: &str = "learn probes=repeat+maze maze=0 fixed=1 nocue=1 goal=repeat@300; \
    turn probes=maze maze=0 goal=maze@200; \
    walk probes=maze maze=2 goal=maze@200; \
    gone probes=maze maze=2 gone=1 goal=maze@150; \
    long probes=maze maze=4 gone=1 goal=maze@150; \
    symbol probes=maze maze=0 symbol=1 goal=maze@150; \
    symbolgone probes=maze maze=2 gone=1 symbol=1";

/// Screen's ladder: move a cursor to the glyph on the screen and click it (stroked on the target).
/// Two cells side by side in the dark with a torch (no cursor mark, and the glyph shows only
/// while the cursor is on it: the eye is empty unless it is on the target, so clicking at what
/// is seen is a respond; runs 4-6 scored 0-24 on the torch alone, where the cursor's mark is
/// always seen and the glyph must be told from it), then the torch with the cursor shown (on the
/// plain two cells, telling the cursor on the glyph from beside it, the population scored 0-54
/// of 300 for 200 generations), then two cells plain,
/// then four in a row, then a 2 x 2 grid (up and down too), then 4 x 2;
/// then a 2-D maze: a 4 x 4 grid of 32-bit cells, then with 3 walls, then 6 (a move into a wall
/// does nothing; the way round must be found), then 8 x 4 of 16-bit cells with 10 walls; then a
/// distractor beside the target (which glyph is the target is learned from the stroke), then
/// three distractors on the full screen.
const SCREEN_LADDER: &str = "glow probes=screen sw=2 sh=1 lit=1 torch=1 dark=1 goal=screen@300; \
    torch probes=screen sw=2 sh=1 lit=1 torch=1 goal=screen@300; \
    two probes=screen sw=2 sh=1 lit=1 goal=screen@300; \
    row probes=screen sw=4 sh=1 lit=1 goal=screen@250; \
    square probes=screen sw=2 sh=2 lit=1 goal=screen@250; \
    full probes=screen sw=4 sh=2 lit=1 goal=screen@200; \
    grid probes=screen sw=4 sh=4 cell=32 lit=1 goal=screen@200; \
    walls probes=screen sw=4 sh=4 cell=32 walls=3 lit=1 goal=screen@150; \
    maze probes=screen sw=4 sh=4 cell=32 walls=6 lit=1 goal=screen@120; \
    bigmaze probes=screen sw=8 sh=4 cell=16 walls=10 lit=1 goal=screen@100; \
    pick probes=screen sw=2 sh=2 lit=2 goal=screen@150; \
    crowd probes=screen sw=4 sh=2 lit=4";

/// Task selection's ladder: one lifetime, one network, blocks alternating responding (click at a
/// light, the hand) and babbling (vary the voice, the tokens shown), scored by `select` (the two
/// tasks' mean minus doing the other's thing). First with each block's task felt as a cue on the
/// touch channel, then with none (the task read from the scene), then short blocks.
const SELECTION_LADDER: &str = "cued probes=select tokens=2 cue=1 goal=select@600; \
    uncued probes=select tokens=2 cue=0 goal=select@600; \
    short probes=select tokens=2 cue=0 block=10";

/// The world the generalist and the cross-island table use: the plain world, with a cycle of 3
/// broken one time in 8 for surprise (the plain stream has no pattern to break).
fn plain() -> Lesson {
    Lesson { period: 3, noise: 1, ..Lesson::default() }
}

fn main() {
    neurocomp::program::neurocomp::set_in_place(env("ISL_INPLACE", 0u8) == 1);
    neurocomp::program::neurocomp::set_count_reads(env("ISL_READS", 0u8) == 1);
    neurocomp::evolve::set_carry(env("ISL_CARRY", 0u64));
    neurocomp::evolve::set_carry_energy(env("ISL_CARRY_ENERGY", 0u8) == 1);
    neurocomp::kernel::class::set_op_choice(env("ISL_OPS", 0u8) == 1);
    neurocomp::evolve::set_epigenetics(env("ISL_EPI", 0u8) == 1);
    neurocomp::program::neurocomp::set_explore(env("ISL_EXPLORE", 0u8) == 1);
    neurocomp::evolve::tags::set_zeros(env("ISL_ZEROS", 0u8) == 1);
    neurocomp::evolve::tags::set_accident(env("ISL_ACCIDENT", 0u8) == 1);
    neurocomp::evolve::tags::set_port_gates(env("ISL_PORTGATE", 0u8) == 1);
    neurocomp::program::neurocomp::set_shadow(env("ISL_SHADOW", 0u8) == 1);
    neurocomp::evolve::tags::set_pause_draws(env("ISL_PAUSES", 0u8) == 1);
    neurocomp::evolve::tags::set_shadow_hints(env("ISL_SHADOW_HINT", 0u8) == 1);
    neurocomp::evolve::tags::set_hint_use(env("ISL_HINT_USE", 0u8) == 1);
    neurocomp::evolve::tags::set_read_outputs(env("ISL_READ_OUT", 0u8) == 1);
    neurocomp::evolve::tags::set_hint_band(env("ISL_HINT_BAND", 0u8) == 1);
    neurocomp::evolve::tags::set_self_writes(env("ISL_SELF_WRITE", 0u8) == 1);
    neurocomp::evolve::tags::set_close_loops(env("ISL_LOOPS", 0u8) == 1);
    neurocomp::evolve::tags::set_heat_inputs(env("ISL_HEAT", 0u8) == 1);
    neurocomp::evolve::tags::set_inhibit_inputs(env("ISL_INHIBIT", 0u8) == 1);
    neurocomp::evolve::tags::set_overlap_genes(env("ISL_OVERLAP", 0u8) == 1);
    neurocomp::evolve::set_elite_pass(env("ISL_ELITE_PASS", 0u8) == 1);
    neurocomp::evolve::set_partial(env("ISL_PARTIAL", 0u8) == 1);
    neurocomp::evolve::set_keep_rungs(env("ISL_KEEP_RUNGS", 0u8) == 1);
    neurocomp::evolve::set_size_classes(env("ISL_SIZE_CLASSES", 0u8) == 1);
    neurocomp::evolve::set_thrift_relax(env("ISL_THRIFT_RELAX", 0usize));
    neurocomp::evolve::set_wire_cost(env("ISL_WIRE_COST", 0u64));
    neurocomp::program::neurocomp::set_write_compete(env("ISL_COMPETE", 0u8) == 1);
    neurocomp::evolve::tags::set_branches(env("ISL_BRANCH", 0u8) == 1);
    neurocomp::evolve::tags::set_out_copy(env("ISL_OUT_COPY", 0u8) == 1);
    neurocomp::evolve::set_muscle(env("ISL_MUSCLE", 0u8) == 1);
    neurocomp::program::neurocomp::set_prune(env("ISL_COMPETE", 0u8) == 1 && env("ISL_PRUNE", 0u8) == 1);
    // (energy: a ration a step, and food in rations for each 1024 points scored; before any
    // lesson is made)
    let ration: u64 = env("ISL_RATION", 0);
    neurocomp::evolve::set_energy(ration, ration * env("ISL_FOOD", 32u64));
    let pop: usize = env("ISL_POP", 16);
    let elite: usize = env("ISL_ELITE", 4);
    let gens: usize = env("ISL_GENS", 300);
    let lives: u64 = env("ISL_LIVES", 12);
    let held_n: u64 = env("ISL_HELD", 12);
    let trials: usize = env("ISL_TRIALS", 200);
    let gap: usize = env("ISL_GAP", 6);
    let every: usize = env("ISL_EVERY", 20);
    let plasmids: usize = env("ISL_PLASMIDS", 2);
    let report_every: usize = env("ISL_REPORT", 10);
    let seed: u64 = env("ISL_SEED", 7);
    let thrift = env("ISL_THRIFT", 1u8) == 1;
    let log_path = std::env::var("ISL_LOG").unwrap_or_else(|_| "plasmids.txt".into());
    let resume = std::env::var("ISL_RESUME").ok();
    let mut log = if resume.is_some() {
        std::fs::OpenOptions::new().create(true).append(true).open(&log_path).expect("plasmid log")
    } else {
        std::fs::File::create(&log_path).expect("plasmid log")
    };
    let ck_path = std::env::var("ISL_CHECKPOINT").unwrap_or_else(|_| "checkpoint.txt".into());
    let ckpt: usize = env("ISL_CKPT", 0);
    let train: Vec<u64> = (0..lives).collect();
    let held: Vec<u64> = (10_000..10_000 + held_n).collect();
    let mut islands = vec![
        Island::plain("switching", vec![0, 1, 2, 3, 5]),
        Island::plain("recall", vec![0, 4]),
        Island::plain("reading", vec![6, 7]),
        Island { name: "generalist".into(), frontier: env("ISL_FRONTIER", 0u8) == 1, stages: vec![Stage { name: "all".into(), probes: ALL_PROBES.to_vec(), lesson: plain(), goal: vec![], hold: 0 }] },
    ];
    // (`ISL_PREDICTION=1`: a fifth island, prediction, on the surprise probe; it starts naked, or
    // from a checkpoint that has it)
    if env("ISL_PREDICTION", 0u8) == 1 {
        islands.push(Island { name: "prediction".into(), frontier: false, stages: vec![Stage { name: "all".into(), probes: vec![9], lesson: plain(), goal: vec![], hold: 0 }] });
    }
    // (`ISL_MAZE=1`: a T-maze; starts naked or from a checkpoint)
    if env("ISL_MAZE", 0u8) == 1 {
        islands.push(Island { name: "maze".into(), frontier: false, stages: vec![Stage { name: "all".into(), probes: vec![10], lesson: plain(), goal: vec![], hold: 0 }] });
    }
    // (`ISL_SCREEN=1`: a screen with a cursor; starts naked or from a checkpoint)
    if env("ISL_SCREEN", 0u8) == 1 {
        islands.push(Island { name: "screen".into(), frontier: false, stages: vec![Stage { name: "all".into(), probes: vec![11], lesson: plain(), goal: vec![], hold: 0 }] });
    }
    // (`ISL_SHAPES=1`: shapes, telling two distorted lights apart; starts naked or from a checkpoint)
    if env("ISL_SHAPES", 0u8) == 1 {
        let lesson = Lesson { swaps: 0, blur: 2, shift: 16, ..Lesson::default() };
        islands.push(Island { name: "shapes".into(), frontier: false, stages: vec![Stage { name: "all".into(), probes: vec![0, 1, 2], lesson, goal: vec![], hold: 0 }] });
    }
    // (`ISL_SELECTION=1`: task selection, one lifetime of blocks alternating two tasks; starts
    // naked or from a checkpoint)
    if env("ISL_SELECTION", 0u8) == 1 {
        let lesson = Lesson { tokens: 2, mixed_cue: true, ..Lesson::default() };
        islands.push(Island { name: "selection".into(), frontier: false, stages: vec![Stage { name: "all".into(), probes: vec![13], lesson, goal: vec![], hold: 0 }] });
    }
    // (`ISL_LADDERS=1`: the default ladders; `ISL_LADDER_<island>`: one's own)
    let ladders = env("ISL_LADDERS", 0u8) == 1;
    for isl in islands.iter_mut() {
        if let Ok(text) = std::env::var(format!("ISL_LADDER_{}", isl.name)) {
            isl.stages = ladder(&text);
        } else if ladders {
            match isl.name.as_str() {
                "reading" => isl.stages = ladder(READING_LADDER),
                "switching" => isl.stages = ladder(SWITCHING_LADDER),
                "recall" => isl.stages = ladder(if env("ISL_RECALL_LONG", 0u8) == 1 { RECALL_LONG_LADDER } else { RECALL_LADDER }),
                "prediction" => isl.stages = ladder(PREDICTION_LADDER),
                "shapes" => isl.stages = ladder(SHAPES_LADDER),
                "maze" => isl.stages = ladder(MAZE_LADDER),
                "screen" => isl.stages = ladder(SCREEN_LADDER),
                "selection" => isl.stages = ladder(SELECTION_LADDER),
                _ => {}
            }
        }
    }
    // (`ISL_FEEL=1`: a feel rung before each repeat rung, switching's, shapes' and the maze's:
    // strokes at random, click after one; the rung it is copied from, with feel for repeat)
    if env("ISL_FEEL", 0u8) == 1 {
        let (feel, repeat, maze) = (probe_named("feel"), probe_named("repeat"), probe_named("maze"));
        for isl in islands.iter_mut() {
            if let Some(k) = isl.stages.iter().position(|s| s.name == "repeat" || s.name == "learn") {
                let mut st = isl.stages[k].clone();
                st.name = "feel".into();
                st.probes = st.probes.iter().copied().filter(|&p| p != repeat && p != maze).chain([feel]).collect();
                st.goal = vec![(feel, env("ISL_FEEL_GOAL", 300))];
                isl.stages.insert(k, st);
            }
        }
    }
    // (`ISL_MIRROR=1`: reading's easier rungs before echo2: the same lesson with each token heard
    // as it is said, then less and less like it, so echoing starts as copying the ear into the
    // voice and moves toward learning the sound of each token)
    if env("ISL_MIRROR", 0u8) == 1 {
        for isl in islands.iter_mut() {
            if let Some(k) = isl.stages.iter().position(|s| s.name == "echo2") {
                let base = isl.stages[k].clone();
                for (j, (name, m, quiet)) in [("echo2q", 8usize, true), ("echo2m8", 8, false), ("echo2m4", 4, false)].into_iter().enumerate() {
                    let mut st = base.clone();
                    st.name = name.into();
                    st.lesson.mirror = m;
                    st.lesson.quiet = quiet;
                    isl.stages.insert(k + j, st);
                }
            }
        }
    }
    // (`ISL_STEPS=rung=goal+goal,...`: steps up to a rung: copies of it put before it, the same
    // world and probes, with lower goals, so a rung no run passes is climbed in parts)
    if let Ok(spec) = std::env::var("ISL_STEPS") {
        for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (name, goals) = part.split_once('=').expect("ISL_STEPS: rung=goal+goal");
            let goals: Vec<i32> = goals.split('+').map(|g| g.parse().expect("ISL_STEPS: a goal")).collect();
            for isl in islands.iter_mut() {
                if let Some(k) = isl.stages.iter().position(|s| s.name == name) {
                    let base = isl.stages[k].clone();
                    for (j, &g) in goals.iter().enumerate() {
                        let mut st = base.clone();
                        st.name = format!("{name}-{g}");
                        st.goal = st.goal.iter().map(|&(p, _)| (p, g)).collect();
                        isl.stages.insert(k + j, st);
                    }
                }
            }
        }
    }
    let tokens = (0..4).map(|i| Token::Channel { tag: (0x11u8).wrapping_mul(i as u8 + 1) ^ 0x5A, width: 3, hold: false }).collect();
    let naked = TagGenome { tokens, next_id: 1 };
    // starting populations: naked, or a checkpoint's (by island name)
    let mut start: Vec<Vec<TagGenome>> = vec![vec![naked.clone()]; islands.len()];
    let mut first = 0usize;
    let mut stages = vec![0usize; islands.len()];
    if let Some(path) = &resume {
        let text = std::fs::read_to_string(path).expect("checkpoint");
        for part in text.split("island ").skip(1) {
            let (head, body) = part.split_once('\n').unwrap_or((part, ""));
            let mut h = head.split_whitespace();
            let name = h.next().unwrap_or("");
            let g: usize = h.nth(1).and_then(|x| x.parse().ok()).unwrap_or(0);
            // (and its stage, if it was climbing a ladder: `stage n`)
            let st: usize = h.nth(1).and_then(|x| x.parse().ok()).unwrap_or(0);
            if let Some(i) = islands.iter().position(|isl| isl.name == name) {
                start[i] = TagGenome::from_text(body).expect("checkpoint genomes");
                first = first.max(g + 1);
                stages[i] = st.min(islands[i].stages.len() - 1);
                println!("resumed {name} from {path}: {} genomes, generation {g}, stage {} ({})", start[i].len(), stages[i], islands[i].stages[stages[i]].name);
            }
        }
    }
    println!("islands: {} of {pop}, elite {elite}, {lives} worlds, {held_n} held out, {trials} trials, gaps 0-{gap}, exchange every {every} ({plasmids} plasmids an island), thrift {thrift}, reads counted {}, channels held in place {}, carrying cost {} a token, plasmids of at most {} populations (0: any), exclusion {}, op choice {}, ration {} a step (0: no energy), food {} for 1024 points scored, fresh training worlds {}, epigenetic marks {}, exploration {}{}",
        islands.iter().map(|i| i.name.as_str()).collect::<Vec<_>>().join(", "), neurocomp::program::neurocomp::count_reads(), neurocomp::program::neurocomp::in_place(), neurocomp::evolve::carry(), env("ISL_PLASMID_MAX", 0usize), env("ISL_EXCLUDE", 0u8) == 1, neurocomp::kernel::class::op_choice(), neurocomp::evolve::energy().0, neurocomp::evolve::energy().1, env("ISL_FRESH", 0u8) == 1, neurocomp::evolve::epigenetics(), neurocomp::program::neurocomp::explore(), format!("{}{}{}", if neurocomp::evolve::tags::zeros() { ", expecting zeros true" } else { "" }, if neurocomp::evolve::tags::accident() { ", accidents true" } else { "" }, if neurocomp::evolve::tags::port_gates() { ", port gates true" } else { "" }) + if neurocomp::program::neurocomp::shadow() { ", shadow explorers true" } else { "" } + if neurocomp::evolve::tags::pause_draws() { ", pauses drawn true" } else { "" } + if neurocomp::evolve::tags::shadow_hints() { ", shadow hints true" } else { "" } + if neurocomp::evolve::tags::hint_use() { ", edges on hints true" } else { "" } + if neurocomp::evolve::tags::read_outputs() { ", outputs read true" } else { "" } + if neurocomp::evolve::tags::hint_band() { ", hint bands true" } else { "" } + if neurocomp::evolve::tags::self_writes() { ", self writes true" } else { "" } + if neurocomp::evolve::tags::close_loops() { ", loops closed true" } else { "" } + if neurocomp::evolve::tags::heat_inputs() { ", heat inputs true" } else { "" } + if neurocomp::evolve::carry_energy() { ", carrying paid as energy true" } else { "" } + if neurocomp::evolve::elite_pass() { ", goals judged on the elite true" } else { "" } + if neurocomp::evolve::partial() { ", partial credit true" } else { "" } + if neurocomp::evolve::keep_rungs() { ", passed rungs kept true" } else { "" } + if neurocomp::evolve::tags::inhibit_inputs() { ", inhibit inputs true" } else { "" } + if neurocomp::evolve::tags::overlap_genes() { ", overlap genes true" } else { "" } + if neurocomp::evolve::size_classes() { ", size classes true" } else { "" } + &if neurocomp::evolve::thrift_relax() > 0 { format!(", thrift relaxed while stuck {}", neurocomp::evolve::thrift_relax()) } else { String::new() } + &if neurocomp::evolve::wire_cost() > 0 { format!(", wiring cost {} a bit of tag distance", neurocomp::evolve::wire_cost()) } else { String::new() } + if neurocomp::program::neurocomp::write_compete() { ", populations compete for their destination true" } else { "" } + if neurocomp::program::neurocomp::prune() { ", pruning true" } else { "" } + if neurocomp::evolve::tags::branches() { ", branches true" } else { "" } + if neurocomp::evolve::tags::out_copy() { ", copy learning of outputs true" } else { "" } + if neurocomp::evolve::muscle() { ", the muscle's response true" } else { "" });
    let names: String = PROBES.iter().map(|p| format!("{:>7}", &p[..p.len().min(6)])).collect();
    for isl in islands.iter().filter(|i| i.stages.len() > 1) {
        println!("{} climbs: {}", isl.name, isl.stages.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(" > "));
    }
    // (`ISL_SIGNAL=n`, with `ISL_RESUME`: measure only, nothing evolves. Each island's genomes as
    // resumed are scored on n fresh worlds on their rung's goal probe, and on the rung below: how
    // much of the spread between genomes stands above the noise across worlds, which is what
    // selection on that rung has to go on.)
    let signal: u64 = env("ISL_SIGNAL", 0);
    if signal > 0 {
        let worlds: Vec<u64> = (50_000..50_000 + signal).collect();
        println!("signal on {signal} fresh worlds (island, rung, goal probe, genomes): mean, best genome; sd between genome means; sd across worlds in a genome; noise in one mean; icc (share of one world's spread that is between genomes); split-half r (genome means on odd worlds against even); epsilon (median over worlds of the spread across genomes)");
        let sd = |v: &[f64]| { let m = v.iter().sum::<f64>() / v.len().max(1) as f64; (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len().max(2) - 1) as f64).sqrt() };
        let median = |mut v: Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
        for (i, isl) in islands.iter().enumerate() {
            let mut ks = vec![stages[i]];
            if stages[i] > 0 { ks.push(stages[i] - 1) }
            for k in ks {
                let st = &isl.stages[k];
                let Some(&(p, goal)) = st.goal.first() else { continue };
                let scored = probe_all_in(start[i].clone(), trials, Physics::Varied, &worlds, gap, &st.probes, st.lesson.clone());
                let v: Vec<Vec<f64>> = scored.iter().map(|s| s.probes.iter().map(|r| r[p] as f64).collect()).collect();
                let means: Vec<f64> = v.iter().map(|w| w.iter().sum::<f64>() / w.len() as f64).collect();
                let within = v.iter().map(|w| sd(w).powi(2)).sum::<f64>() / v.len() as f64;
                let between = sd(&means).powi(2);
                let sigma_b = (between - within / signal as f64).max(0.0);
                let icc = sigma_b / (sigma_b + within).max(1e-9);
                let half = |odd: usize| -> Vec<f64> { v.iter().map(|w| { let h: Vec<f64> = w.iter().skip(odd).step_by(2).copied().collect(); h.iter().sum::<f64>() / h.len().max(1) as f64 }).collect() };
                let (a, b) = (half(0), half(1));
                let (ma, mb) = (a.iter().sum::<f64>() / a.len() as f64, b.iter().sum::<f64>() / b.len() as f64);
                let cov = a.iter().zip(&b).map(|(x, y)| (x - ma) * (y - mb)).sum::<f64>();
                let r = cov / ((a.iter().map(|x| (x - ma).powi(2)).sum::<f64>() * b.iter().map(|y| (y - mb).powi(2)).sum::<f64>()).sqrt()).max(1e-9);
                let eps = median((0..signal as usize).map(|w| { let col: Vec<f64> = v.iter().map(|g| g[w]).collect(); let m = median(col.clone()); median(col.iter().map(|x| (x - m).abs()).collect()) }).collect());
                let grand = means.iter().sum::<f64>() / means.len() as f64;
                let best = means.iter().cloned().fold(f64::MIN, f64::max);
                println!("  {:<10} {:<8} {} (goal {goal}), {} genomes: mean {grand:.0}, best {best:.0}; between {:.1}; within {:.1}; noise {:.1}; icc {icc:.2}; split-half r {r:.2}; epsilon {eps:.1}{}",
                    isl.name, st.name, PROBES[p], v.len(), sd(&means), within.sqrt(), (within / signal as f64).sqrt(), if k == stages[i] { "" } else { "  (the rung below)" });
            }
        }
        return;
    }
    // (`ISL_LESION=n`, with `ISL_RESUME`: measure only, nothing evolves. Each island's best genome
    // as resumed is scored on n fresh worlds on each of its subtasks, then again with each of its
    // populations taken out in turn, on the same worlds: what each subtask loses without it.)
    let lesion: u64 = env("ISL_LESION", 0);
    if lesion > 0 {
        let worlds: Vec<u64> = (60_000..60_000 + lesion).collect();
        println!("lesion map on {lesion} fresh worlds: each island's best genome scored on its subtasks, then with each population taken out; a loss is the subtask's mean less the lesioned mean, as a share of that rung's goal");
        for (i, isl) in islands.iter().enumerate() {
            let Some(best) = start[i].first() else { continue };
            // the subtasks: an island's rung's probes in its world; at the frontier, each other
            // island's rung's probes in its world (each probe once), with that rung's goal
            let mut groups: Vec<(String, Vec<usize>, Lesson, i32)> = Vec::new();
            let mut seen: Vec<usize> = Vec::new();
            for (j, other) in islands.iter().enumerate() {
                if other.frontier || (!isl.frontier && j != i) || (isl.frontier && j == i) {
                    continue;
                }
                let st = &other.stages[stages[j]];
                let new: Vec<usize> = st.probes.iter().copied().filter(|p| !seen.contains(p)).collect();
                if new.is_empty() {
                    continue;
                }
                seen.extend(&new);
                let goal = st.goal.first().map_or(300, |g| g.1).max(50);
                groups.push((format!("{} {}", other.name, st.name), new, st.lesson.clone(), goal));
            }
            let score = |g: &TagGenome| -> Vec<f64> {
                let mut out = Vec::new();
                for (_, probes, lesson, _) in &groups {
                    let s = probe_all_in(vec![g.clone()], trials, Physics::Varied, &worlds, gap, probes, lesson.clone());
                    for &p in probes {
                        out.push(s[0].probes.iter().map(|r| r[p] as f64).sum::<f64>() / worlds.len() as f64);
                    }
                }
                out
            };
            let names: Vec<String> = groups.iter().flat_map(|(n, probes, _, _)| probes.iter().map(move |&p| format!("{n}:{}", &PROBES[p][..PROBES[p].len().min(6)]))).collect();
            let goals: Vec<f64> = groups.iter().flat_map(|(_, probes, _, goal)| probes.iter().map(move |_| *goal as f64)).collect();
            let base = score(best);
            let (_, edges, _) = best.wiring();
            let label = |id: u32| edges.iter().find(|e| e.0 == id).map_or(String::new(), |e| format!("{}->{} {}{}", e.1, e.2, e.3, if e.4 { " wire" } else { "" }));
            println!("  {} (best genome, {} populations); subtasks and their means: {}", isl.name, best.populations(), names.iter().zip(&base).map(|(n, b)| format!("{n} {b:.0}")).collect::<Vec<_>>().join(", "));
            let mut shares = Vec::new();
            let mut tops = vec![0usize; names.len()];
            let mut mattering = 0usize;
            for (k, t) in best.tokens.iter().enumerate() {
                let Token::Pop(p) = t else { continue };
                let mut g = best.clone();
                g.tokens.remove(k);
                let les = score(&g);
                let loss: Vec<f64> = base.iter().zip(&les).zip(&goals).map(|((b, l), goal)| (b - l) / goal).collect();
                let total: f64 = loss.iter().map(|x| x.abs()).sum();
                let (top, &most) = loss.iter().enumerate().max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap()).unwrap_or((0, &0.0));
                let matters = total >= 0.05;
                if matters {
                    mattering += 1;
                    shares.push(most.abs() / total);
                    tops[top] += 1;
                }
                println!("    #{:<6} {:<34} {}{}", p.id, label(p.id), loss.iter().map(|x| format!("{:>6.2}", x)).collect::<Vec<_>>().join(""), if matters { format!("   top {} ({:.0}% of its loss)", names[top], 100.0 * most.abs() / total) } else { "   (matters little)".into() });
            }
            let mean_share = if shares.is_empty() { 0.0 } else { shares.iter().sum::<f64>() / shares.len() as f64 };
            println!("  {}: {mattering} of {} populations matter (a loss of 5% of a goal or more in all); their mean share on their top subtask {:.2} (spread evenly over {} subtasks it would be {:.2}); populations by top subtask: {}", isl.name, best.populations(), mean_share, names.len(), 1.0 / names.len().max(1) as f64, names.iter().zip(&tops).filter(|(_, c)| **c > 0).map(|(n, c)| format!("{n} {c}")).collect::<Vec<_>>().join(", "));
        }
        return;
    }
    let mut was = stages.clone();
    // plasmids: those already in the populations (moved before a resume), then each one moved
    let plasmid_worlds: Vec<u64> = held.iter().copied().take(env("ISL_PLASMID_WORLDS", 16usize)).collect();
    let plasmid_max: usize = env("ISL_PLASMID_MAX", 0);
    let exclude = env("ISL_EXCLUDE", 0u8) == 1;
    let mut followed: Vec<Moved> = Vec::new();
    for (i, popn) in start.iter().enumerate() {
        let mut found: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
        for g in popn {
            for t in &g.tokens {
                if let Token::Pop(p) = t {
                    if let Some(n) = plasmid_of(p.id) {
                        let ids = found.entry(n).or_default();
                        if !ids.contains(&p.id) { ids.push(p.id) }
                    }
                }
            }
        }
        for (n, ids) in found {
            followed.push(Moved { n, from: None, to: i, at: first.saturating_sub(1), ids, peak: 0, alive: true, kind: "moved before the resume".into() });
        }
    }
    let number = Cell::new(followed.iter().map(|m| m.n + 1).max().unwrap_or(0));
    // (each transfer's plasmid number, recipient, child, tokens and the plasmids it came from)
    let pending: RefCell<Vec<(u32, TagGenome, TagGenome, Vec<Token>, Vec<u32>)>> = RefCell::new(Vec::new());
    // each generation's line carries the UTC time and the seconds since the last one
    let clock = || { let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) % 86400; format!("{:02}:{:02}:{:02}Z", t / 3600, t / 60 % 60, t % 60) };
    let mut last = std::time::Instant::now();
    println!("started {}", clock());
    // a plasmid: one circuit of the donor (a random declared channel and what touches it)
    let transfer = |recipient: &TagGenome, donor: &TagGenome, rng: &mut StdRng| -> (TagGenome, String) {
        let n = number.get();
        number.set(n + 1);
        if donor.populations() == 0 {
            pending.borrow_mut().push((n, recipient.clone(), recipient.clone(), vec![], vec![]));
            return (recipient.clone(), "(nothing: the donor has no population)".into());
        }
        let mut child = recipient.clone();
        let k = rng.gen_range(0..donor.populations());
        let moved = child.take_population_plasmid_upto(donor, k, plasmid_max, rng);
        // (the same population but for its id and channel tags)
        let bare = |p: &Pop| Pop { id: 0, from: 0, to: 0, gate: p.gate.map(|(_, on)| (0, on)), pause: p.pause.map(|(_, on)| (0, on)), mark: 0, explored: false, ..p.clone() };
        // exclusion: a recipient that already carries every population of it takes nothing (two
        // copies of one circuit would pile up, as a cell holds one plasmid of a kind)
        if exclude && moved.iter().all(|t| match t { Token::Pop(q) => recipient.tokens.iter().any(|r| matches!(r, Token::Pop(p) if bare(p) == bare(q))), _ => true }) {
            pending.borrow_mut().push((n, recipient.clone(), recipient.clone(), vec![], vec![]));
            return (recipient.clone(), "(excluded: the recipient already carries it)".into());
        }
        // the plasmid's populations get numbered ids; one that was itself moved before names the
        // plasmid it came from
        let at = child.tokens.len() - moved.len();
        let mut parents = Vec::new();
        for (j, t) in child.tokens[at..].iter_mut().enumerate() {
            if let Token::Pop(q) = t {
                for d in &donor.tokens {
                    if let Token::Pop(p) = d {
                        if let Some(m) = plasmid_of(p.id) {
                            if bare(p) == bare(q) && !parents.contains(&m) { parents.push(m) }
                        }
                    }
                }
                q.id = TAGGED | (n << 8) | (j.min(255) as u32);
            }
        }
        let tokens = child.tokens[at..].to_vec();
        let text = TagGenome { tokens: tokens.clone(), next_id: 0 }.describe();
        pending.borrow_mut().push((n, recipient.clone(), child.clone(), tokens, parents));
        (child, text)
    };
    evolve_islands(&islands, start, stages, first, pop, trials, Physics::Varied, &train, env("ISL_FRESH", 0u8) == 1, &held, gap, elite, gens, seed, thrift, every, plasmids, transfer, |g, r| {
        // the plasmids followed: how many of their island's genomes carry them (this generation's
        // new ones are in children, scored from the next)
        for m in followed.iter_mut().filter(|m| m.alive && m.at < g) {
            let c = r.ranked[m.to].iter().filter(|x| carries(x, &m.ids).0).count();
            m.peak = m.peak.max(c);
            if c == 0 {
                m.alive = false;
                println!("{g:>4}  plasmid #{} gone from {} after {} generations (at most {} of {pop} carried it)", m.n, islands[m.to].name, g - m.at, m.peak);
                let _ = writeln!(log, "generation {g}: #{} gone from {} after {} generations (peak {} of {pop})", m.n, islands[m.to].name, g - m.at, m.peak);
            }
        }
        // the plasmids moved this generation: measured, scored alone against their blind
        // controls, and their recipients scored with and without them
        let fresh: Vec<_> = pending.borrow_mut().drain(..).collect();
        assert_eq!(fresh.len(), r.moved.len(), "one pending plasmid a move");
        let mut moved_lines = Vec::new();
        if !fresh.is_empty() {
            let alone: Vec<TagGenome> = fresh.iter().flat_map(|f| { let a = TagGenome { tokens: f.3.clone(), next_id: 0 }; [a.blind(), a] }).collect();
            let scored = probe_all_in(alone, trials, Physics::Varied, &plasmid_worlds, gap, &ALL_PROBES, plain());
            for (k, ((n, recipient, child, tokens, parents), (from, to, what))) in fresh.into_iter().zip(&r.moved).enumerate() {
                let (from, to) = (*from, *to);
                let route = format!("#{n} {}->{}", islands[from].name, islands[to].name);
                if tokens.is_empty() {
                    moved_lines.push(format!("{route}: {}", what.trim_matches(|c| c == '(' || c == ')')));
                    let _ = writeln!(log, "generation {g}: {route}\n{what}");
                    continue;
                }
                let (made, role) = measure(&tokens);
                let blind = probe_means(&scored[2 * k].probes);
                let solo = probe_means(&scored[2 * k + 1].probes);
                let sees = above(&solo, &blind, &ALL_PROBES);
                let artefact: Vec<String> = ALL_PROBES.iter().filter(|&&p| blind[p] >= NOTICE && solo[p] - blind[p] < NOTICE).map(|&p| PROBES[p].to_string()).collect();
                let st = &islands[to].stages[r.stage[to]];
                let host = probe_all_in(vec![recipient, child], trials, Physics::Varied, &plasmid_worlds, gap, &st.probes, st.lesson);
                let (without, with) = (probe_means(&host[0].probes), probe_means(&host[1].probes));
                let judged: Vec<usize> = if st.goal.is_empty() { st.probes.clone() } else { st.goal.iter().map(|&(p, _)| p).collect() };
                // (an island scored on many probes can keep a plasmid that gains on one and loses
                // on others: lexicase takes each probe as its own case, so that is "mixed")
                let gains = judged.iter().any(|&p| with[p] - without[p] >= NOTICE);
                let losses = judged.iter().any(|&p| with[p] - without[p] <= -NOTICE);
                let verdict = match (gains, losses) { (true, true) => "mixed", (false, true) => "hurts", (true, false) => "helps", _ => "neutral" };
                let deltas: Vec<String> = judged.iter().filter(|&&p| with[p] != without[p]).map(|&p| format!("{} {:+}", PROBES[p], with[p] - without[p])).collect();
                let kind = format!(
                    "{role}; {}{}; {verdict} in {} ({})",
                    if sees.is_empty() { "senses for nothing alone".to_string() } else { format!("senses for {}", sees.join("+")) },
                    if artefact.is_empty() { String::new() } else { format!("; scores {} blind", artefact.join("+")) },
                    islands[to].name, if deltas.is_empty() { "no change".to_string() } else { deltas.join(", ") },
                );
                let pops = tokens.iter().filter(|t| matches!(t, Token::Pop(_))).count();
                moved_lines.push(format!("{route} ({pops} populations): {kind}"));
                let from_text = if parents.is_empty() { String::new() } else { format!(" (from {})", parents.iter().map(|m| format!("#{m}")).collect::<Vec<_>>().join(", ")) };
                let _ = writeln!(log, "generation {g}: {route}{from_text}\n  made of: {made}\n  alone/blind: {}\n  in {} on {}, with/without: {}\n  kind: {kind}\n{what}",
                    pairs(&solo, &blind, &ALL_PROBES), islands[to].name, st.name, pairs(&with, &without, &st.probes));
                let ids = tokens.iter().filter_map(|t| if let Token::Pop(p) = t { Some(p.id) } else { None }).collect();
                followed.push(Moved { n, from: Some(from), to, at: g, ids, peak: 0, alive: true, kind });
            }
        }
        // a checkpoint at every exchange, every `ISL_CKPT` generations, and at the end: each
        // island's population, best first
        if (every > 0 && g % every == 0) || (ckpt > 0 && g % ckpt == 0) || g + 1 == gens {
            let mut ck = String::new();
            for (i, isl) in islands.iter().enumerate() {
                ck += &format!("island {} generation {g} stage {}\n", isl.name, r.stage[i]);
                for x in &r.ranked[i] {
                    ck += &x.to_text();
                }
            }
            let _ = std::fs::write(&ck_path, ck);
        }
        // every generation, one line: each climbing island's goal probes against their goals
        let goals: Vec<String> = islands.iter().enumerate().filter(|(_, isl)| isl.stages.len() > 1).filter_map(|(i, isl)| {
            let st = &isl.stages[r.stage[i]];
            if st.goal.is_empty() {
                return None;
            }
            let g: Vec<String> = st.goal.iter().map(|&(p, x)| format!("{} {}/{x}", PROBES[p], r.means[i][p])).collect();
            Some(format!("{} {} {} (met {}/{})", isl.name, st.name, g.join(" "), r.met[i], st.hold.max(1)))
        }).collect();
        let took = last.elapsed().as_secs_f64();
        last = std::time::Instant::now();
        println!("{g:>4}  {} {took:.0}s{}", clock(), if goals.is_empty() { String::new() } else { format!("  goals: {}", goals.join("; ")) });
        // an island that held its goal on the training worlds, checked on the held-out ones
        for (i, isl) in islands.iter().enumerate() {
            if let Some(m) = &r.rechecked[i] {
                let st = &isl.stages[r.stage[i]];
                let pass = st.goal.iter().all(|&(p, x)| m[p] >= x);
                let g2: Vec<String> = st.goal.iter().map(|&(p, x)| format!("{} {}/{x}", PROBES[p], m[p])).collect();
                println!("{g:>4}  {} held its goal on training; its elite held out: {} ({})", isl.name, g2.join(" "), if pass { "moves on" } else { "stays" });
            }
        }
        // an island that met its stage's goal: it is scored on the next stage from this generation
        for (i, isl) in islands.iter().enumerate() {
            if r.stage[i] != was[i] {
                println!("{g:>4}  {} moves to stage {} ({}): goal met", isl.name, r.stage[i], isl.stages[r.stage[i]].name);
                // (`ISL_RUNG_LOG`) the best genome that passed, kept as it was when it passed
                if let Ok(path) = std::env::var("ISL_RUNG_LOG") {
                    if let (Ok(mut f), Some(best)) = (std::fs::OpenOptions::new().create(true).append(true).open(&path), r.ranked[i].first()) {
                        use std::io::Write;
                        let _ = write!(f, "island {} generation {g} stage {} passed {}\n{}", isl.name, r.stage[i], isl.stages[was[i]].name, best.to_text());
                    }
                }
                was[i] = r.stage[i];
            }
        }
        if !r.moved.is_empty() {
            println!("{g:>4}  plasmids moved: {}", r.moved.iter().map(|(f, t, w)| format!("{}->{} ({} populations)", islands[*f].name, islands[*t].name, w.lines().filter(|l| l.contains("->")).count())).collect::<Vec<_>>().join(", "));
            for l in &moved_lines {
                println!("      {l}");
            }
            // the earlier plasmids still carried, and by how many (in the elite: the kept)
            let alive: Vec<String> = followed.iter().filter(|m| m.alive && m.at < g).map(|m| {
                let c = r.ranked[m.to].iter().filter(|x| carries(x, &m.ids).0).count();
                let e = r.ranked[m.to].iter().take(elite).filter(|x| carries(x, &m.ids).0).count();
                let whole = r.ranked[m.to].iter().filter(|x| carries(x, &m.ids).1).count();
                format!("#{} {}->{} (generation {}) {c}/{pop}, {e} in the elite, {whole} whole", m.n, m.from.map_or("?", |f| islands[f].name.as_str()), islands[m.to].name, m.at)
            }).collect();
            let gone = followed.iter().filter(|m| !m.alive).count();
            println!("      plasmids still carried: {} ({gone} gone){}", alive.len(), if alive.is_empty() { String::new() } else { format!(": {}", alive.join("; ")) });
            // every island's best on every probe, held out: does a circuit carry over?
            println!("      every island's best on every probe, held out:{}", "");
            println!("      {:<11}{names}", "");
            for (i, isl) in islands.iter().enumerate() {
                let h = probe_all_in(vec![r.ranked[i][0].clone()], trials, Physics::Varied, &held, gap, &ALL_PROBES, plain());
                println!("      {:<11}{}", isl.name, row(&probe_means(&h[0].probes), &ALL_PROBES));
            }
        }
        if g % report_every == 0 || g + 1 == gens {
            println!("{g:>4}  {:<11}{names}   work   size  ports", "");
            for (i, isl) in islands.iter().enumerate() {
                let best = &r.ranked[i][0];
                let st = &isl.stages[r.stage[i]];
                // what it is scored on, and its best held out on the same: its stage, or (at the
                // frontier) each other island's current stage in that stage's world, each probe
                // from the first island that has it
                let (shown, held_row) = if isl.frontier {
                    let mut shown: Vec<usize> = Vec::new();
                    let mut m = [0i32; neurocomp::world::observe::NPROBES];
                    for (j, other) in islands.iter().enumerate() {
                        if j == i || other.frontier {
                            continue;
                        }
                        let os = &other.stages[r.stage[j]];
                        let new: Vec<usize> = os.probes.iter().copied().filter(|p| !shown.contains(p)).collect();
                        if new.is_empty() {
                            continue;
                        }
                        let h = probe_all_in(vec![best.clone()], trials, Physics::Varied, &held, gap, &new, os.lesson);
                        let hm = probe_means(&h[0].probes);
                        for &p in &new {
                            m[p] = hm[p];
                        }
                        shown.extend(new);
                    }
                    (shown, m)
                } else {
                    let h = probe_all_in(vec![best.clone()], trials, Physics::Varied, &held, gap, &st.probes, st.lesson);
                    (st.probes.clone(), probe_means(&h[0].probes))
                };
                let (rd, wr) = best.ports();
                let at = if isl.stages.len() > 1 { format!("  [stage {} {}, {} generations]", r.stage[i], st.name, r.held[i]) } else if isl.frontier { "  [at the frontier]".to_string() } else { String::new() };
                println!("      {:<11}{}  {:>5}  {:>5}  reads {} writes {}  (population, training){at}", isl.name, row(&r.means[i], &shown), r.work[i], best.tokens.len(), rd.join("+"), wr.join("+"));
                println!("      {:<11}{}  (its best, held out)", "", row(&held_row, &shown));
                // (`ISL_SIZE_CLASSES`) how many of the population are in each size class
                let sizes = if neurocomp::evolve::size_classes() {
                    let mut c = [0usize; 5];
                    for g in &r.ranked[i] {
                        c[neurocomp::evolve::size_class(g.tokens.len())] += 1;
                    }
                    format!("; sizes 1-3:{} 4-15:{} 16-63:{} 64-255:{} 256+:{}", c[0], c[1], c[2], c[3], c[4])
                } else {
                    String::new()
                } + &if neurocomp::evolve::wire_cost() > 0 { format!("; wiring {} bits", best.wire_length()) } else { String::new() } + &match r.contest.get(i) {
                    Some(&(steps, lost, let_go)) if neurocomp::program::neurocomp::write_compete() && steps > 0 => format!("; contest: {} kernels held down by another population and {:.2} populations let go per 1000 steps", lost * 1000 / steps, let_go as f64 * 1000.0 / steps as f64),
                    _ => String::new(),
                } + &if neurocomp::evolve::tags::branches() { format!("; branches {} of {}", best.pops_branched(), best.populations()) } else { String::new() };
                println!("      {:<11}{}{sizes}  (the elite's ops; its best's loops)", "", ops_and_loops(&r.ranked[i][..elite.min(r.ranked[i].len())]));
            }
        }
    });
    // the plasmids' fates: those still carried at the end, with what they are
    let all = followed.iter().filter(|m| m.from.is_some()).count();
    let kept: Vec<&Moved> = followed.iter().filter(|m| m.alive && m.from.is_some()).collect();
    println!("plasmids: {all} moved, {} still carried at the end", kept.len());
    for m in kept {
        println!("      #{} {}->{} (generation {}, peak {}/{pop}): {}", m.n, m.from.map_or("?", |f| islands[f].name.as_str()), islands[m.to].name, m.at, m.peak, m.kind);
    }
    println!("plasmids recorded in {log_path}");
}
