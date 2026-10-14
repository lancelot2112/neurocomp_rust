//! Can each grammar represent prediction (and its recursion), and does it keep it under mutation?
//!
//! The residue circuit (experiment 02), written in the tag grammar and in the edge genome: a
//! residue channel R gets the eye channel copied in (OR), and a population reading the eye a frame
//! back XORs its predictions into R, its kernels born where R is left on and not holding each
//! other down. Fed a repeating cycle of codes, R should go to nothing.
//!
//! Then robustness: `REP_MUTANTS` single random mutants of each circuit, each fed the same cycle.
//! Per grammar: how many still cancel (late residue 0), how many changed nothing, how many broke.
//!
//! `REP_SYMBOLS`, `REP_STEPS`, `REP_MUTANTS`, `REP_DISTRACT`.

use neurocomp::bitvec::bithistory::AdvanceMode;
use neurocomp::bitvec::BitVector;
use neurocomp::evolve::genes::{EdgeGene, GeneGenome, INPUT};
use neurocomp::evolve::tags::{Pop, TagGenome, Token, SENSES};
use neurocomp::evolve::Evolvable;
use neurocomp::kernel::KernelOp;
use neurocomp::program::graph::EdgeKind;
use neurocomp::world::{self, clustered_code, World};
use rand::rngs::StdRng;
use rand::SeedableRng;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

const R_TAG: u8 = 0x3C;

fn tag_circuit() -> TagGenome {
    let r = R_TAG;
    let copy = Pop { field: 3, ..Pop::wire(1, SENSES, r) };
    let predict = Pop { lag: 2, field: 3, op: KernelOp::Xor, grow: true, share: true, ..Pop::new(2, SENSES, r) };
    TagGenome { tokens: vec![Token::Channel { tag: r, width: 3, hold: false }, Token::Pop(copy), Token::Pop(predict)], next_id: 3 }
}

fn edge_circuit() -> GeneGenome {
    let r = 2;
    let mut copy = EdgeGene::new(100, INPUT, r, EdgeKind::Wire, KernelOp::Or);
    copy.spec.window = Some((0, 8));
    let mut predict = EdgeGene::new(101, INPUT, r, EdgeKind::Kernels, KernelOp::Xor);
    predict.spec.window = Some((0, 8));
    predict.spec.input_read_back = 2;
    predict.spec.grow = true;
    predict.spec.inhibit = false;
    predict.spec.wide = true;
    predict.spec.out_words = 8;
    GeneGenome { edges: vec![copy, predict], next_id: 102 }
}

/// Early and late residue (popcount of R summed over the first 10 and the last 50 steps), with R
/// the node at `r_index`; None if R does not exist.
fn residue<G: Evolvable>(g: &G, r_index: Option<usize>, symbols: usize, steps: usize) -> Option<(u64, u64)> {
    let r = r_index?;
    let w = World::new(1, 4);
    let mut net = g.build(&w, 1);
    if r >= net.nodes.len() {
        return None;
    }
    let mut rng = StdRng::seed_from_u64(2);
    let codes: Vec<BitVector> = (0..symbols).map(|_| clustered_code(world::CH_BITS, 16, &mut rng)).collect();
    let (mut early, mut late) = (0u64, 0u64);
    for t in 0..steps {
        net.tick(AdvanceMode::Clear, 0);
        net.write_input(&codes[t % symbols], 0);
        let p = net.read_node(r).count_ones() as u64;
        if t < 10 { early += p } else if t >= steps - 50 { late += p }
    }
    Some((early, late))
}

fn main() {
    let symbols: usize = env("REP_SYMBOLS", 5);
    let steps: usize = env("REP_STEPS", 300);
    let mutants: usize = env("REP_MUTANTS", 300);
    // where R is in each grammar's compiled graph
    let tag_r = |g: &TagGenome| -> Option<usize> {
        // R is the channel the grammar resolves R's tag to: the nearest declared tag (the senses
        // and act are not R)
        let mut best: Option<(u32, usize)> = None;
        let mut i = 0;
        for t in &g.tokens {
            if let Token::Channel { tag, .. } = t {
                i += 1;
                let d = (*tag ^ R_TAG).count_ones();
                if best.map_or(true, |(bd, _)| d < bd) {
                    best = Some((d, i));
                }
            }
        }
        best.map(|b| b.1)
    };
    let edge_r = |g: &GeneGenome| -> Option<usize> { g.vectors().into_iter().filter(|&v| v != 1).position(|v| v == 2) };
    let (mut t0, e0) = (tag_circuit(), edge_circuit());
    // (`REP_DISTRACT=n`: n more channels with random tags around the tag circuit, for a flipped
    // tag bit to land on instead)
    {
        use rand::Rng;
        let n: usize = env("REP_DISTRACT", 0);
        let mut r = StdRng::seed_from_u64(77);
        for _ in 0..n {
            let mut tag: u8 = r.r#gen();
            while tag == R_TAG || tag == SENSES || tag == 0xFF { tag = r.r#gen(); }
            t0.tokens.push(Token::Channel { tag, width: 3, hold: false });
        }
    }
    println!("representability: the residue circuit on a cycle of {symbols} codes, {steps} steps");
    println!("{:<6} {:>14} {:>14}", "grammar", "early residue", "late residue");
    let (te, tl) = residue(&t0, tag_r(&t0), symbols, steps).unwrap();
    let (ee, el) = residue(&e0, edge_r(&e0), symbols, steps).unwrap();
    println!("{:<6} {te:>14} {tl:>14}", "tags");
    println!("{:<6} {ee:>14} {el:>14}", "edges");
    println!();
    println!("robustness: {mutants} single random mutants of each circuit");
    println!("{:<6} {:>9} {:>11} {:>8} {:>10}", "grammar", "cancels", "unchanged", "broken", "R gone");
    let mut rng = StdRng::seed_from_u64(5);
    let tally = |res: Vec<(Option<(u64, u64)>, bool)>| {
        let (mut ok, mut same, mut broke, mut gone) = (0, 0, 0, 0);
        for (r, unchanged) in res {
            match r {
                None => gone += 1,
                Some((_, 0)) => { ok += 1; if unchanged { same += 1 } }
                Some(_) => broke += 1,
            }
        }
        (ok, same, broke, gone)
    };
    let tag_res: Vec<_> = (0..mutants).map(|_| { let m = t0.mutate(&mut rng); let u = m == t0; (residue(&m, tag_r(&m), symbols, steps), u) }).collect();
    let edge_res: Vec<_> = (0..mutants).map(|_| { let m = e0.mutate(&mut rng); let u = m == e0; (residue(&m, edge_r(&m), symbols, steps), u) }).collect();
    // the tag mutants split: those that changed the circuit's own tokens (R's channel or its two
    // populations) and those that changed something else (a new token, a distractor)
    let own = |m: &TagGenome| {
        let pops: Vec<&Pop> = m.tokens.iter().filter_map(|t| if let Token::Pop(p) = t { Some(p) } else { None }).collect();
        let orig: Vec<&Pop> = t0.tokens.iter().filter_map(|t| if let Token::Pop(p) = t { Some(p) } else { None }).collect();
        let r_same = m.tokens.iter().any(|t| *t == Token::Channel { tag: R_TAG, width: 3, hold: false });
        !(r_same && orig.iter().all(|o| pops.iter().any(|p| p == o)))
    };
    let mut rng2 = StdRng::seed_from_u64(5);
    let (mut touched, mut touched_ok, mut other, mut other_ok) = (0, 0, 0, 0);
    for _ in 0..mutants {
        let m = t0.mutate(&mut rng2);
        let ok = matches!(residue(&m, tag_r(&m), symbols, steps), Some((_, 0)));
        if own(&m) { touched += 1; touched_ok += ok as u32 } else { other += 1; other_ok += ok as u32 }
    }
    let (a, b, c, d) = tally(tag_res);
    println!("{:<6} {a:>9} {b:>11} {c:>8} {d:>10}", "tags");
    println!("  tags: mutants that changed the circuit's own tokens {touched} (still cancel {touched_ok}); that changed something else {other} (still cancel {other_ok})");
    let (a, b, c, d) = tally(edge_res);
    println!("{:<6} {a:>9} {b:>11} {c:>8} {d:>10}", "edges");
}
