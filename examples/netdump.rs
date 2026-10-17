//! Each island's best network from a checkpoint, as rows of a graph (CSV on stdout): its bit
//! vectors, its populations with their ops, and their modulatory inputs (gates, pauses, heat and
//! inhibit inputs). `cargo run --release --example netdump -- <checkpoint> <run>`.
use neurocomp::evolve::tags::TagGenome;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("a checkpoint");
    let run = args.get(2).cloned().unwrap_or_default();
    let text = std::fs::read_to_string(path).expect("checkpoint");
    println!("run,island,generation,kind,id,from,to,label,bits");
    for part in text.split("island ").skip(1) {
        let (head, body) = part.split_once('\n').unwrap_or((part, ""));
        let mut h = head.split_whitespace();
        let island = h.next().unwrap_or("").to_string();
        let generation = h.nth(1).unwrap_or("").to_string();
        let Ok(genomes) = TagGenome::from_text(body) else { continue };
        let Some(best) = genomes.first() else { continue };
        let (nodes, edges, mods) = best.wiring();
        for (name, bits, hold) in nodes {
            println!("{run},{island},{generation},vector,,{name},,{},{bits}", if hold { "holds" } else { "" });
        }
        for (id, from, to, op, wire) in edges {
            println!("{run},{island},{generation},edge,{id},{from},{to},{}{},", op, if wire { " wire" } else { "" });
        }
        for (id, kind, from, on) in mods {
            println!("{run},{island},{generation},mod,{id},{from},{id},{}{},", kind, if on { "" } else { " (while off)" });
        }
    }
}
