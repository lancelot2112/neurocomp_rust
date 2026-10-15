//! Re-scores each island's best genome in a checkpoint on many fresh worlds (all probes), to tell
//! a held-out score from noise. `RCK_FILE` (checkpoint.txt), `RCK_WORLDS` (64), `RCK_TRIALS`
//! (200), `RCK_GAP` (6), `RCK_FROM` (50,000: the first world seed), `RCK_TOP` (1: how many of
//! each island's ranked genomes, from the top), `RCK_DELAY` (3: steps from the delayed light to
//! the response window, for remember), `RCK_BLIND` (0; 1: each genome's blind control too),
//! `RCK_SWAPS` (3), `RCK_BLUR` (0) and `RCK_SHIFT` (0) (the reversal lifetime: shapes' stages),
//! `RCK_MAZE` (2: the maze's stem), `RCK_ISLAND` (all: only the island of that name), `RCK_READS` and `RCK_INPLACE` (0; 1: reads counted as work, channels held in place, as
//! `ISL_READS` and `ISL_INPLACE`: a checkpoint from such a run is rechecked so), `RCK_RATION` and
//! `RCK_FOOD` (as `ISL_RATION`, `ISL_FOOD`).

use neurocomp::evolve::tags::TagGenome;
use neurocomp::evolve::{probe_all_in, probe_means, Lesson, Physics, ALL_PROBES};
use neurocomp::world::observe::PROBES;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let path = std::env::var("RCK_FILE").unwrap_or_else(|_| "checkpoint.txt".into());
    let n: u64 = env("RCK_WORLDS", 64);
    let trials: usize = env("RCK_TRIALS", 200);
    let gap: usize = env("RCK_GAP", 6);
    let from: u64 = env("RCK_FROM", 50_000);
    neurocomp::program::neurocomp::set_in_place(env("RCK_INPLACE", 0u8) == 1);
    neurocomp::program::neurocomp::set_count_reads(env("RCK_READS", 0u8) == 1);
    let ration: u64 = env("RCK_RATION", 0);
    neurocomp::evolve::set_energy(ration, ration * env("RCK_FOOD", 32u64));
    let top: usize = env("RCK_TOP", 1);
    let lesson = Lesson { delay: env("RCK_DELAY", 3), swaps: env("RCK_SWAPS", 3), blur: env("RCK_BLUR", 0), shift: env("RCK_SHIFT", 0), maze_len: env("RCK_MAZE", 2), ..Lesson::default() };
    let only = std::env::var("RCK_ISLAND").ok();
    let blind = env("RCK_BLIND", 0u8) == 1;
    let worlds: Vec<u64> = (from..from + n).collect();
    let text = std::fs::read_to_string(&path).expect("checkpoint");
    let names: String = PROBES.iter().map(|p| format!("{:>7}", &p[..p.len().min(6)])).collect();
    println!("{:<11}{names}   reads / writes  ({n} worlds)", "");
    for part in text.split("island ").skip(1) {
        let (head, body) = part.split_once('\n').unwrap_or((part, ""));
        if only.as_deref().is_some_and(|o| head.split_whitespace().next() != Some(o)) { continue }
        let gs = TagGenome::from_text(body).expect("genomes");
        let picked: Vec<(String, TagGenome)> = gs.iter().take(top).enumerate().flat_map(|(k, g)| {
            let mut v = vec![(format!("rank {k}"), g.clone())];
            if blind { v.push((format!("rank {k} blind"), g.blind())) }
            v
        }).collect();
        for (k, best) in &picked {
        let (rd, wr) = best.ports();
        let h = probe_all_in(vec![best.clone()], trials, Physics::Varied, &worlds, gap, &ALL_PROBES, lesson);
        let m = probe_means(&h[0].probes);
        let row: String = m.iter().map(|x| format!("{x:>7}")).collect();
        // per-world spread of each probe: the standard error of the mean
        let se: String = (0..m.len()).map(|p| {
            let xs: Vec<f64> = h[0].probes.iter().map(|r| r[p] as f64).collect();
            let mean = xs.iter().sum::<f64>() / xs.len() as f64;
            let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (xs.len() as f64 - 1.0).max(1.0);
            format!("{:>7}", format!("±{:.0}", (var / xs.len() as f64).sqrt()))
        }).collect();
        println!("{:<11}{row}   {} / {}  ({k}, work {})", head.split_whitespace().next().unwrap_or(""), rd.join("+"), wr.join("+"), h[0].work);
        println!("{:<11}{se}   (standard error)", "");
        let (on, selfs, largest, pops) = best.loops();
        let mut ops: Vec<(String, usize)> = Vec::new();
        for t in &best.tokens {
            if let neurocomp::evolve::tags::Token::Pop(p) = t {
                let name = p.op.text().split(':').next().unwrap_or("").to_string();
                match ops.iter_mut().find(|c| c.0 == name) { Some(c) => c.1 += 1, None => ops.push((name, 1)) }
            }
        }
        ops.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let all: usize = ops.iter().map(|c| c.1).sum();
        let ops: Vec<String> = ops.iter().take(5).map(|(n, k)| format!("{n} {}%", k * 100 / all.max(1))).collect();
        println!("{:<11}ops {} (of {all} populations); loops: {on} of {pops} wired populations on a loop ({selfs} self-loops), the largest {largest} channels", "", ops.join(", "));
        }
    }
}
