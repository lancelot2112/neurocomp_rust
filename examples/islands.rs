//! Islands: populations pressed on different capacities, trading circuits.
//!
//! Each island is the tag genome from a naked start, evolving by epsilon-lexicase on its own
//! probes (`observe::probes`):
//! - switching: respond, tell, discriminate, reverse, win-stay (serial reversal);
//! - recall: respond, remember (a light gone a few steps ago);
//! - reading: copy (say the token shown on the eye a few steps before);
//! - generalist: all seven.
//! Every `ISL_EVERY` generations each island takes `ISL_PLASMIDS` plasmids (one population of a
//! donor with the circuit of any channel it touches, made private; or the population alone if it
//! only reads and writes ports) from the best genomes of other islands into its new children, so a circuit grown under one pressure is tried under another. Each plasmid is
//! recorded (from, to, the circuit) in `ISL_LOG`; at each exchange every island's best genome is
//! scored on every probe on held-out worlds (does a circuit carry over?).
//!
//! Every `ISL_REPORT` generations, per island: its population's means on its own probes (training
//! worlds), its best genome's probes on held-out worlds, its work per step and which ports its
//! genomes read and write.
//!
//! `ISL_POP` (16 a island), `ISL_ELITE` (4), `ISL_GENS` (300), `ISL_LIVES` (12 training worlds),
//! `ISL_HELD` (12), `ISL_TRIALS` (200), `ISL_GAP` (6), `ISL_EVERY` (20), `ISL_PLASMIDS` (2),
//! `ISL_REPORT` (10), `ISL_LOG` (plasmids.txt), `ISL_SEED` (7), `ISL_THRIFT` (1).
//!
//! Checkpoints: at every exchange each island's whole population is written to `ISL_CHECKPOINT`
//! (checkpoint.txt, as text); `ISL_RESUME=<file>` starts every island named in it from its saved
//! population and generation (islands not in it start naked), so a change to the world or the
//! probes can be injected into a running experiment: stop it, change, resume.

use std::io::Write;

use neurocomp::evolve::tags::{TagGenome, Token};
use neurocomp::evolve::{evolve_islands, probe_all, probe_means, Island, Physics, ALL_PROBES};
use neurocomp::world::observe::{ProbeRow, PROBES};
use rand::rngs::StdRng;
use rand::Rng;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn row(m: &ProbeRow, only: &[usize]) -> String {
    (0..m.len()).map(|p| if only.contains(&p) { format!("{:>7}", m[p]) } else { format!("{:>7}", ".") }).collect::<Vec<_>>().join("")
}

fn main() {
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
    let train: Vec<u64> = (0..lives).collect();
    let held: Vec<u64> = (10_000..10_000 + held_n).collect();
    let islands = vec![
        Island { name: "switching".into(), probes: vec![0, 1, 2, 3, 5] },
        Island { name: "recall".into(), probes: vec![0, 4] },
        Island { name: "reading".into(), probes: vec![6] },
        Island { name: "generalist".into(), probes: ALL_PROBES.to_vec() },
    ];
    let tokens = (0..4).map(|i| Token::Channel { tag: (0x11u8).wrapping_mul(i as u8 + 1) ^ 0x5A, width: 3, hold: false }).collect();
    let naked = TagGenome { tokens, next_id: 1 };
    // starting populations: naked, or a checkpoint's (by island name)
    let mut start: Vec<Vec<TagGenome>> = vec![vec![naked.clone()]; islands.len()];
    let mut first = 0usize;
    if let Some(path) = &resume {
        let text = std::fs::read_to_string(path).expect("checkpoint");
        for part in text.split("island ").skip(1) {
            let (head, body) = part.split_once('\n').unwrap_or((part, ""));
            let mut h = head.split_whitespace();
            let name = h.next().unwrap_or("");
            let g: usize = h.nth(1).and_then(|x| x.parse().ok()).unwrap_or(0);
            if let Some(i) = islands.iter().position(|isl| isl.name == name) {
                start[i] = TagGenome::from_text(body).expect("checkpoint genomes");
                first = first.max(g + 1);
                println!("resumed {name} from {path}: {} genomes, generation {g}", start[i].len());
            }
        }
    }
    println!("islands: {} of {pop}, elite {elite}, {lives} worlds, {held_n} held out, {trials} trials, gaps 0-{gap}, exchange every {every} ({plasmids} plasmids an island), thrift {thrift}",
        islands.iter().map(|i| i.name.as_str()).collect::<Vec<_>>().join(", "));
    let names: String = PROBES.iter().map(|p| format!("{:>7}", &p[..p.len().min(6)])).collect();
    // a plasmid: one circuit of the donor (a random declared channel and what touches it)
    let transfer = |recipient: &TagGenome, donor: &TagGenome, rng: &mut StdRng| -> (TagGenome, String) {
        if donor.populations() == 0 {
            return (recipient.clone(), "(nothing: the donor has no population)".into());
        }
        let mut child = recipient.clone();
        let k = rng.gen_range(0..donor.populations());
        let moved = child.take_population_plasmid(donor, k, rng);
        let text = TagGenome { tokens: moved, next_id: 0 }.describe();
        (child, text)
    };
    evolve_islands(&islands, start, first, pop, trials, Physics::Varied, &train, gap, elite, gens, seed, thrift, every, plasmids, transfer, |g, r| {
        for (from, to, what) in &r.moved {
            let _ = writeln!(log, "generation {g}: {} -> {}\n{what}", islands[*from].name, islands[*to].name);
        }
        // a checkpoint at every exchange (and at the end): each island's population, best first
        if (every > 0 && g % every == 0) || g + 1 == gens {
            let mut ck = String::new();
            for (i, isl) in islands.iter().enumerate() {
                ck += &format!("island {} generation {g}\n", isl.name);
                for x in &r.ranked[i] {
                    ck += &x.to_text();
                }
            }
            let _ = std::fs::write(&ck_path, ck);
        }
        if !r.moved.is_empty() {
            println!("{g:>4}  plasmids moved: {}", r.moved.iter().map(|(f, t, w)| format!("{}->{} ({} populations)", islands[*f].name, islands[*t].name, w.lines().filter(|l| l.contains("->")).count())).collect::<Vec<_>>().join(", "));
            // every island's best on every probe, held out: does a circuit carry over?
            println!("      every island's best on every probe, held out:{}", "");
            println!("      {:<11}{names}", "");
            for (i, isl) in islands.iter().enumerate() {
                let h = probe_all(vec![r.ranked[i][0].clone()], trials, Physics::Varied, &held, gap, &ALL_PROBES);
                println!("      {:<11}{}", isl.name, row(&probe_means(&h[0].probes), &ALL_PROBES));
            }
        }
        if g % report_every == 0 || g + 1 == gens {
            println!("{g:>4}  {:<11}{names}   work   size  ports", "");
            for (i, isl) in islands.iter().enumerate() {
                let best = &r.ranked[i][0];
                let h = probe_all(vec![best.clone()], trials, Physics::Varied, &held, gap, &isl.probes);
                let (rd, wr) = best.ports();
                println!("      {:<11}{}  {:>5}  {:>5}  reads {} writes {}  (population, training)", isl.name, row(&r.means[i], &isl.probes), r.work[i], best.tokens.len(), rd.join("+"), wr.join("+"));
                println!("      {:<11}{}  (its best, held out)", "", row(&probe_means(&h[0].probes), &isl.probes));
            }
        }
    });
    println!("plasmids recorded in {log_path}");
}
