//! Does copy learning of outputs (`Pop::out_copy`) let a network learn to echo when the sound of a
//! token and the voice's code for it share nothing (reading's echo2: two tokens, every one taught,
//! codes drawn afresh each lifetime)? A hand-made genome of two populations: a babbler (kernels
//! from the self channel into the voice, firing spontaneously one step in four, so the network
//! says tokens and hears itself) and a listener (kernels from the ear into the voice, with copy
//! learning at a frame offset, growing kernels on what it hears, each kernel moving its output toward what the voice held when what
//! it heard came in). Scored on 24 worlds, by offset and rate, against the listener without copy
//! learning; and the listener alone (no babbling). It takes the template population line from a
//! genome file: `cargo run --release --example copycheck -- runs/echo-wires/wire.txt` (`TEACH=n`:
//! the teacher says n tokens in 8, 8 by default; `MUSCLE=1`: the act shows what the body did;
//! `BABBLE=n`: the teacher silent for the first n eighths of each lifetime).
use neurocomp::evolve::tags::TagGenome;
use neurocomp::evolve::{probe_all_in, Lesson, Physics};
use neurocomp::world::observe::PROBES;

fn main() {
    // (`SENTENCE=1`) sentences in every lifetime (`evolve::set_sentences`)
    neurocomp::evolve::set_sentences(std::env::var("SENTENCE").is_ok_and(|v| v == "1"));
    neurocomp::evolve::tags::set_out_copy(true);
    let text = std::fs::read_to_string(std::env::args().nth(1).expect("a genome file with one wire")).unwrap();
    let line = text.lines().find(|l| l.starts_with("pop ")).expect("a pop line").to_string();
    let echo = PROBES.iter().position(|p| *p == "echo").unwrap();
    let worlds: Vec<u64> = (80_000..80_024).collect();
    let kernels = |id: u32, from: u8, tonic: u8, copy: u8, rate: u8| {
        let mut l = line.replace(" wire=1 ", " wire=0 ").replace(" from=56 ", &format!(" from={from} ")).replace(" count=2 ", " count=16 ").replace(" tonic=0 ", &format!(" tonic={tonic} ")).replace(" out_learn=4 ", &format!(" out_learn={rate} ")).replace(" gate=142:1 ", " gate=- ").replace(" grow=0 ", if tonic == 0 { " grow=1 " } else { " grow=0 " });
        l = l.replacen("pop id=871 ", &format!("pop id={id} "), 1);
        if copy > 0 {
            l.push_str(&format!(" out_copy={copy}"));
        }
        l
    };
    let teach: u32 = std::env::var("TEACH").ok().and_then(|x| x.parse().ok()).unwrap_or(8);
    // (`MUSCLE=1`) the act shows what the body did (`evolve::set_muscle`)
    neurocomp::evolve::set_muscle(std::env::var("MUSCLE").is_ok_and(|v| v == "1"));
    // (`BABBLE=n`) the teacher silent for the first n eighths of each lifetime
    let babble_first: usize = std::env::var("BABBLE").ok().and_then(|x| x.parse().ok()).unwrap_or(0);
    for trials in [200usize, 800] {
        println!("lifetimes of {trials} tokens, the teacher saying {teach} in 8 after the first {babble_first} eighths, the muscle's response {}:", neurocomp::evolve::muscle());
        for (copy, rate) in [(0u8, 0u8), (1, 0), (2, 0), (3, 0), (1, 2), (2, 2), (1, 4), (2, 4)] {
            let mut row = Vec::new();
            for babble in [true, false] {
                let mut text = String::from("genome 3\n");
                if babble {
                    text += &kernels(1, 0x80, 2, 0, 4);
                    text.push('\n');
                }
                text += &kernels(2, 0x38, 0, copy, rate);
                text.push('\n');
                let g = TagGenome::from_text(&text).unwrap().remove(0);
                let lesson = Lesson { tokens: 2, teach, babble: babble_first * trials / 8, ..Lesson::default() };
                let s = probe_all_in(vec![g], trials, Physics::Varied, &worlds, 6, &[echo], lesson);
                row.push(s[0].probes.iter().map(|r| r[echo] as i64).sum::<i64>() / worlds.len() as i64);
            }
            println!("  copy offset {copy} rate 2^-{rate}: echo with babbling {}, alone {}", row[0], row[1]);
        }
    }
}
