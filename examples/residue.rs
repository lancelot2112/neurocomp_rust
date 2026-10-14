//! Is prediction-and-residue expressible in the graph language? A hand-written program:
//!
//! ```text
//! EdgeWire               the input is copied (OR) into the residue node R
//! ReadInput EdgeLag      a second edge from the input, reading the frame before (the context)
//! EdgeXor EdgeGrow       ... its kernels write by XOR (cancel) and are born where R is left on
//! ```
//!
//! So R = input(t) XOR prediction(from input(t-1)). Fed a sequence of random sparse codes, R's
//! popcount is the residue: what came unpredicted plus what was predicted and did not come.
//!
//! `RES_SHARE=1` adds `EdgeShare` (the predictor's kernels do not hold each other down).
//!
//! `RES_SYMBOLS` (distinct codes), `RES_BRANCH` (after the last symbol, the sequence restarts at
//! one of two places: an unpredictable step), `RES_STEPS`, `RES_SEEDS`.

use neurocomp::bitvec::bithistory::AdvanceMode;
use neurocomp::bitvec::BitVector;
use neurocomp::kernel::{KernelClass, KernelGroup};
use neurocomp::program::graph::{GraphInstruction as I, GraphProgram, ProgramDefaults};
use neurocomp::program::neurocomp::RuntimeNetwork;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let symbols: usize = env("RES_SYMBOLS", 6);
    let branch = std::env::var("RES_BRANCH").is_ok();
    let steps: usize = env("RES_STEPS", 600);
    let seeds: u64 = env("RES_SEEDS", 5);
    // (`RES_SHARE=1`: the predictor edge's kernels do not hold each other down)
    let mut code = vec![I::EdgeWire(), I::ReadInput(), I::EdgeLag(), I::EdgeXor(), I::EdgeGrow()];
    if std::env::var("RES_SHARE").is_ok() {
        code.push(I::EdgeShare());
    }
    let prog = GraphProgram { code };
    let defaults = ProgramDefaults { default_bits: 512, ..ProgramDefaults::default() };
    let g = prog.build_graph(&defaults);
    println!("residue: {symbols} symbols, {}, {steps} steps, {seeds} seeds", if branch { "branching restart" } else { "a fixed cycle" });
    println!("{:>6}  {:>10}  {:>8}", "steps", "residue", "kernels");
    let mut rows: Vec<(u64, u64, usize)> = vec![(0, 0, 0); steps / 50];
    for seed in 0..seeds {
        let mut rng = StdRng::seed_from_u64(seed);
        let all: Vec<usize> = (0..512).collect();
        let codes: Vec<BitVector> = (0..symbols).map(|_| BitVector::from_bits(&all.choose_multiple(&mut rng, 16).copied().collect::<Vec<_>>(), 512)).collect();
        let mut net = RuntimeNetwork::from_graph_with(&g, 512, 512, seed, |_| KernelGroup::from_classes(vec![KernelClass::with_kernels(vec![])]));
        let mut at = 0usize;
        for t in 0..steps {
            net.tick(AdvanceMode::Clear, 0);
            net.write_input(&codes[at], 0);
            // R is node 1 (the program's first node)
            let r = net.read_node(1).count_ones() as u64;
            let row = &mut rows[(t / 50).min(steps / 50 - 1)];
            row.0 += r;
            row.1 += 1;
            row.2 = row.2.max(net.edges.iter().map(|e| e.group_len()).sum());
            at += 1;
            if at == symbols {
                at = if branch && rng.gen_bool(0.5) { 1 } else { 0 };
            }
        }
    }
    for (i, r) in rows.iter().enumerate() {
        println!("{:>6}  {:>10.1}  {:>8}", (i + 1) * 50, r.0 as f64 / r.1 as f64, r.2);
    }
}
