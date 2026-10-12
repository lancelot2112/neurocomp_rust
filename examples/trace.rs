//! Record a mind's run, step by step, for the system viewer (`viewer/`): every area's active
//! bits, working memory, the orbitofrontal state and what it holds, the amygdala's onsets and
//! value cells, the hand loop's striatal context with the masks of its go and no-go cells that
//! fired, and the dopamine error, value, frustration and events.
//!
//! COND (instr; pavlov, delayed, reversal), SEED (0), TRACE_WARM (trials run before recording,
//! 300), TRACE_TRIALS (trials recorded, 40), TRACE_OUT (trace.json); the mind's switches as for
//! the conditioning probe.

use neurocomp::program::mind::{Config, Mind};
use neurocomp::program::world::{Acts, Conditioning, World, CLICK};
use std::fmt::Write as _;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn list(v: &[usize]) -> String {
    let mut s = String::from("[");
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        let _ = write!(s, "{x}");
    }
    s.push(']');
    s
}

fn main() {
    let seed: u64 = env("SEED", 0);
    let warm: usize = env("TRACE_WARM", 300);
    let rec: usize = env("TRACE_TRIALS", 40);
    let out: String = env("TRACE_OUT", "trace.json".to_string());
    let kind = match std::env::var("COND").unwrap_or_else(|_| "instr".into()).as_str() {
        "pavlov" => Conditioning::Pavlov,
        "delayed" => Conditioning::Delayed,
        "reversal" => Conditioning::Reversal,
        _ => Conditioning::Instrumental,
    };
    let mut world = World::new(seed);
    let mut mind = Mind::with_config(world.body(), seed, Config::from_env());
    world.conditioning(kind, warm + rec);
    let mut acts = Acts::default();
    let mut trial = 0usize;
    let mut steps: Vec<String> = Vec::new();
    let f = |x: i32| x as f64 / 65536.0;
    while !world.done() {
        let senses = world.step(&acts);
        acts = mind.step(&senses);
        let ev = world.events.last().unwrap().clone();
        let Some((t, cue)) = ev.cond else { continue };
        if t == 0 && world.events.len() > 1 {
            trial += 1;
            if trial % 50 == 0 {
                mind.sleep(3);
            }
        }
        if trial < warm {
            continue;
        }
        let r = &mind.report;
        let mut s = String::new();
        let lit = match kind {
            Conditioning::Instrumental | Conditioning::Reversal => (2..=4).contains(&t),
            Conditioning::Delayed => t == 2,
            Conditioning::Pavlov => t == 2,
        };
        let _ = write!(
            s,
            "{{\"trial\":{},\"t\":{},\"cue\":{},\"lit\":{},\"click\":{},\"stroke\":{},\"spoke\":{},\"delta\":{:.4},\"value\":{:.4},\"frus\":{:.4},\"amy\":{:.4}",
            trial - warm,
            t,
            cue,
            lit && (cue || kind == Conditioning::Reversal),
            ev.moved == Some(CLICK),
            ev.stroked,
            r.spoke,
            f(r.delta),
            f(r.value),
            f(r.frustration),
            f(r.amy_predicted)
        );
        s.push_str(",\"p\":{");
        for (i, (name, bits)) in mind.snapshot().iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            let _ = write!(s, "\"{name}\":{}", list(bits));
        }
        s.push('}');
        let (view, (go, nogo)) = mind.hand_view();
        let _ = write!(s, ",\"go\":{go},\"nogo\":{nogo}");
        if let Some((ctx, gm, nm)) = view {
            let _ = write!(s, ",\"ctx\":{}", list(&ctx));
            let _ = write!(s, ",\"gm\":[{}]", gm.iter().map(|m| list(m)).collect::<Vec<_>>().join(","));
            let _ = write!(s, ",\"nm\":[{}]", nm.iter().map(|m| list(m)).collect::<Vec<_>>().join(","));
        }
        s.push('}');
        steps.push(s);
    }
    let meta = format!("{{\"kind\":\"{kind:?}\",\"seed\":{seed},\"warm\":{warm},\"trials\":{rec},\"bits\":2048}}");
    let json = format!("{{\"meta\":{meta},\"steps\":[\n{}\n]}}\n", steps.join(",\n"));
    std::fs::write(&out, json).expect("write the trace");
    eprintln!("{} steps written to {out}", steps.len());
}
