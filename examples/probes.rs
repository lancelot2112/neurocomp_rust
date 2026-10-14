//! Evolving on graded probes instead of one task score.
//!
//! The tag genome from a naked start (an empty string and `PRB_CHANNELS` empty channels) evolves
//! by epsilon-lexicase selection on five probes (`observe::probes`), each a small capacity on a
//! smooth scale in parts per 1024:
//! - respond: clicks while a light is up minus clicks with none (a blind babbler scores 0);
//! - tell: clicks differently to the two lights, either way round;
//! - discriminate: more clicks on the paying light (the task);
//! - reverse: the worst phase of a serial reversal (it follows every swap);
//! - remember: more clicks after a light that is gone than after none.
//! Each (world seed, probe) pair is a lexicase case, so a genome best at any one capacity in a
//! few worlds can be a parent. The probes are read by the observer from the event log; the
//! network never sees them.
//!
//! Every `PRB_EVERY` generations: of the `PRB_SHORTLIST` genomes with the highest training scores,
//! the one best on a held-out set (its probe means there), reported on a second held-out set; the
//! population's means on its training worlds; how many genomes read and write each port; and the
//! genome size. `PRB_SHOW=1` prints the network.
//!
//! `PRB_POP` (32), `PRB_ELITE` (8), `PRB_GENS` (100), `PRB_LIVES` (16 training worlds),
//! `PRB_HELD` (16), `PRB_TRIALS` (200 a lifetime), `PRB_CHANNELS` (4), `PRB_EVERY` (10),
//! `PRB_SEED` (7), `PRB_SHOW` (0), `PRB_THRIFT` (0; 1: one case per world prefers less work per
//! step), `PRB_GAP` (0; n: 0 to n empty steps between trials).

use neurocomp::evolve::tags::{TagGenome, Token};
use neurocomp::evolve::{evolve_lexicase, probe_all, probe_means, Physics};
use neurocomp::world::observe::PROBES;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn row(m: &[i32; 5]) -> String {
    m.iter().map(|x| format!("{x:>7}")).collect::<Vec<_>>().join("")
}

fn main() {
    let pop: usize = env("PRB_POP", 32);
    let elite: usize = env("PRB_ELITE", 8);
    let gens: usize = env("PRB_GENS", 100);
    let lives: u64 = env("PRB_LIVES", 16);
    let held_n: u64 = env("PRB_HELD", 16);
    let trials: usize = env("PRB_TRIALS", 200);
    let channels: usize = env("PRB_CHANNELS", 4);
    let every: usize = env("PRB_EVERY", 10);
    let seed: u64 = env("PRB_SEED", 7);
    let show = env("PRB_SHOW", 0u8) == 1;
    // (`PRB_THRIFT=1`: one more lexicase case prefers less work per step)
    let thrift = env("PRB_THRIFT", 0u8) == 1;
    // (`PRB_GAP=n`: 0 to n empty steps between trials, so a light's time cannot be counted)
    let gap: usize = env("PRB_GAP", 0);
    let train: Vec<u64> = (0..lives).collect();
    // held-out worlds: one set picks the genome shown (among the top of the population), another
    // reports it, so the number shown is not the best of several draws
    let held: Vec<u64> = (10_000..10_000 + held_n).collect();
    let report_set: Vec<u64> = (30_000..30_000 + held_n).collect();
    let shortlist: usize = env("PRB_SHORTLIST", 8);
    let tokens = (0..channels).map(|i| Token::Channel { tag: (0x11u8).wrapping_mul(i as u8 + 1) ^ 0x5A, width: 3, hold: false }).collect();
    let start = TagGenome { tokens, next_id: 1 };
    println!("probes: tag genome from a naked start ({channels} empty channels); pop {pop}, elite {elite}, {lives} worlds, {held_n} held out, {trials} trials a lifetime, gaps 0-{gap}, thrift {thrift}");
    let names: String = PROBES.iter().map(|p| format!("{:>7}", &p[..p.len().min(6)])).collect();
    println!("{:>4}  {:>5}  {:<35}  {:<35}  {:<35}  {:>13}", "gen", "size", "shown: held-out pick", "shown: second held-out set", "population mean (training)", "work: shown pop");
    println!("{:>4}  {:>5}  {names}  {names}  {names}", "", "");
    let last = evolve_lexicase(vec![start; pop], trials, Physics::Varied, &train, gap, elite, gens, seed, thrift, |g, r| {
        if g % every == 0 || g + 1 == gens {
            // the top of the population by training score, judged on held-out worlds
            let top: Vec<TagGenome> = r.ranked.iter().take(shortlist).cloned().collect();
            let judged = probe_all(top, trials, Physics::Varied, &held, gap);
            let pick = judged.iter().max_by_key(|j| probe_means(&j.probes).iter().map(|&x| x as i64).sum::<i64>()).unwrap();
            let shown = probe_all(vec![pick.genome.clone()], trials, Physics::Varied, &report_set, gap);
            let hm = probe_means(&shown[0].probes);
            // how many genomes read each input port and write each output port
            let mut reads: std::collections::BTreeMap<String, usize> = Default::default();
            let mut writes: std::collections::BTreeMap<String, usize> = Default::default();
            for x in &r.ranked {
                let (a, b) = x.ports();
                for p in a { *reads.entry(p).or_default() += 1 }
                for p in b { *writes.entry(p).or_default() += 1 }
            }
            let fmt = |m: &std::collections::BTreeMap<String, usize>| m.iter().map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join(", ");
            println!("{g:>4}  {:>5}  {}  {}  {}  {:>6} {:>6}", pick.genome.tokens.len(), row(&probe_means(&pick.probes)), row(&hm), row(&r.pop_means), shown[0].work, r.pop_work);
            println!("      genomes reading: {}; writing: {}", fmt(&reads), fmt(&writes));
            if show {
                print!("{}", pick.genome.describe());
            }
        }
    });
    let h = probe_all(vec![last[0].clone()], trials, Physics::Varied, &held, gap);
    println!("final best, held out: {}  work {}", row(&probe_means(&h[0].probes)), h[0].work);
    print!("{}", last[0].describe());
}
