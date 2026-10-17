//! What a single wire can do with the body's codes drawn afresh each lifetime, as place codes
//! (`Lesson::motor_place`: each move and token owns a word, the same every lifetime), and with the
//! lights and the stroke at fixed places too (`Lesson::sense_place`). A wire from a sense port to
//! the hand or the voice, landing at each word of it in turn, scored on 24 worlds on the probes a
//! reflex should pass: respond (click at a light), tell (click at the paying light), feel (click
//! after a stroke), and echo (with each token heard as it is said, one's own voice unheard).
//! `cargo run --release --example mirrorcheck` is its reading counterpart; this one takes the
//! template population line from a genome file: `cargo run --release --example motorcheck --
//! runs/echo-wires/wire.txt`.
use neurocomp::evolve::tags::TagGenome;
use neurocomp::evolve::{probe_all_in, Lesson, Physics};
use neurocomp::world::observe::PROBES;

fn main() {
    // (`SENTENCE=1`) sentences in every lifetime (`evolve::set_sentences`)
    neurocomp::evolve::set_sentences(std::env::var("SENTENCE").is_ok_and(|v| v == "1"));
    let text = std::fs::read_to_string(std::env::args().nth(1).expect("a genome file with one wire")).unwrap();
    let line = text.lines().find(|l| l.starts_with("pop ")).expect("a pop line").to_string();
    let probe = |name: &str| PROBES.iter().position(|p| *p == name).unwrap();
    let worlds: Vec<u64> = (70_000..70_024).collect();
    // (from tag, to tag, probe, lesson tokens/teach for reading)
    let cases: [(&str, u8, u8, &str); 4] = [("eye to hand", 0x07, 0xF8, "respond"), ("eye to hand", 0x07, 0xF8, "tell"), ("affect to hand", 0xE0, 0xF8, "feel"), ("ear to voice", 0x38, 0x1F, "echo")];
    for (motor, senses) in [(false, false), (true, false), (true, true)] {
        println!("motor place codes {motor}, lights and stroke placed {senses}:");
        for (name, from, to, p) in cases {
            let mut scores = Vec::new();
            for d in 0..8 {
                let l = line.replace(" from=56 ", &format!(" from={from} ")).replace(" to=31 ", &format!(" to={to} ")).replace(" cover=1 ", " cover=0 ").replace(" at=61583 ", " at=0 ").replace(" field=0 ", " field=3 ").replace(" place=0 ", &format!(" place={d} "));
                let g = TagGenome::from_text(&format!("genome 2\n{l}\n")).unwrap().remove(0);
                let reading = p == "echo";
                let lesson = Lesson { motor_place: motor, sense_place: senses, tokens: if reading { 2 } else { 8 }, teach: if reading { 8 } else { 4 }, mirror: if reading { 8 } else { 0 }, quiet: reading, ..Lesson::default() };
                let s = probe_all_in(vec![g], 200, Physics::Varied, &worlds, 6, &[probe(p)], lesson);
                scores.push(s[0].probes.iter().map(|r| r[probe(p)] as i64).sum::<i64>() / worlds.len() as i64);
            }
            let best = scores.iter().enumerate().max_by_key(|x| x.1).unwrap();
            println!("  {name:<15} {p:<8} by landing word 0-7: {:?}; best {} at word {}", scores, best.1, best.0);
        }
    }
}
