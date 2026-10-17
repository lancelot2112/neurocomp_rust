//! Each island's best network from checkpoints or rung logs, as rows of a graph (CSV on stdout):
//! its bit vectors, its populations with their ops, and their modulatory inputs (gates, pauses,
//! heat and inhibit inputs). Each block of a file (`island <name> generation <g> stage <s> ...`)
//! is one snapshot; a checkpoint's first genome in a block is its island's best, as is a rung log's
//! (`ISL_RUNG_LOG`). `cargo run --release --example netdump -- <file> <run> [<file> <run> ...]`.
use neurocomp::evolve::tags::TagGenome;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    println!("run,island,generation,snapshot,kind,id,from,to,label,bits");
    for pair in args.chunks(2) {
        let (path, run) = (&pair[0], pair.get(1).cloned().unwrap_or_default());
        let Ok(text) = std::fs::read_to_string(path) else { continue };
        let rungs = path.contains("rungs");
        for part in text.split("island ").skip(1) {
            let (head, body) = part.split_once('\n').unwrap_or((part, ""));
            let w: Vec<&str> = head.split_whitespace().collect();
            let island = w.first().copied().unwrap_or("").to_string();
            let generation = w.get(2).copied().unwrap_or("").to_string();
            let snapshot = if rungs { format!("passed {} at {}", w.get(6).copied().unwrap_or("a rung"), generation) } else { format!("final (generation {generation})") };
            let Ok(genomes) = TagGenome::from_text(body) else { continue };
            let Some(best) = genomes.first() else { continue };
            let (nodes, edges, mods) = best.wiring();
            let pre = format!("{run},{island},{generation},{snapshot}");
            for (name, bits, hold) in nodes {
                println!("{pre},vector,,{name},,{},{bits}", if hold { "holds" } else { "" });
            }
            for (id, from, to, op, wire) in edges {
                println!("{pre},edge,{id},{from},{to},{}{},", op, if wire { " wire" } else { "" });
            }
            for (id, kind, from, on) in mods {
                println!("{pre},mod,{id},{from},{id},{}{},", kind, if on { "" } else { " (while off)" });
            }
        }
    }
}
