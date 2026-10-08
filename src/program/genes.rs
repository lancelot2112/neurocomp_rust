//! The genome as a list of genes: the form evolution works on.
//!
//! The stack text (`genomes/*.gen`) is easy to read and write, but a poor thing to mutate:
//! insert or delete one instruction and every signal below it on the stack moves, so one
//! change rewires everything downstream; a `set:` changes every predictor placed after it;
//! a loop pairs with whichever `feedback` is open. Here the same network is a list of
//! genes, each local:
//! - every gene has a permanent **id**; its inputs name other genes by id and port, not by
//!   position, so adding or removing a gene leaves the others' wiring alone;
//! - a loop is **explicit** (`prev`: the source's value from the step before), and the tick
//!   order comes from the wiring (a topological sort), not from the order of the list;
//! - each gene carries **its own parameters** (a predictor its whole learning rule), so
//!   changing one changes one module;
//! - numbers that follow from the wiring are not genes: a `concat`'s slots are its
//!   inputs, a predictor's frames its input's width;
//! - each predictor draws from **its own random stream**, seeded by its id, so a new gene
//!   does not shift every other module's draws;
//! - wiring is **graded**: an input is a set of connections, each with a gain (the share of
//!   its source's bits that pass). A new connection enters at gain 0 (neutral) and grows by
//!   steps of 1/16; one at full gain is a plain wire, so a compiled genome is unchanged.
//!
//! `GeneList::from_genome` compiles the stack text (behaviour unchanged but for the random
//! streams); `mutate` applies one local change; `build` makes the network.

use std::collections::BTreeMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::program::modules::{Gene, Genes, Genome, Kind, NetOp, Network, Predictor, Schedule, Src as NetSrc};

/// Where an input reads from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The network's input port (0: the word, 1: the sentence clock).
    In(usize),
    /// Gene `id`'s output port.
    Gene(u32, usize),
}

/// One connection into an input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conn {
    pub src: Source,
    /// Read the source's value from the step before (a loop).
    pub prev: bool,
    /// The share of the source's bits that pass (`Q16`; 65536 = all, 0 = none).
    pub gain: u32,
}

/// An input port: the connections into it (none: unconnected), and its width in frames
/// (word-sized), if set: sources are folded or padded to it (see `Blend`). Unset, the input
/// is as wide as its main connection.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Input {
    pub conns: Vec<Conn>,
    pub frames: Option<u32>,
}

const FULL: u32 = 65536;
const STEP: u32 = 4096; // 1/16

impl Input {
    fn wire(src: Option<Source>, prev: bool) -> Input {
        Input { conns: src.map(|s| Conn { src: s, prev, gain: FULL }).into_iter().collect(), frames: None }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeneNode {
    pub id: u32,
    pub kind: Kind,
    /// Named numbers: a predictor's `bits`, `sample` and learning rule; a window's `span` and
    /// `keep`; a surprise's `threshold`.
    pub params: BTreeMap<String, i64>,
    pub inputs: Vec<Input>,
    pub enabled: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeneList {
    pub inputs: usize,
    pub genes: Vec<GeneNode>,
    pub outputs: Vec<Input>,
    /// The update loop: `sleep_every`, `reset_at_story`, `reset_at_sentence`.
    pub schedule: BTreeMap<String, i64>,
    next_id: u32,
}

/// How a number mutates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scale {
    /// Not mutable (an interface size, such as the word's width).
    Fixed,
    /// A count or size: multiplied or divided by 2^(1/4).
    Log,
    /// A fraction over 65,536: moved by 1/16.
    Fraction,
    /// On or off: flipped.
    Flag,
}

fn scale_of(key: &str) -> Scale {
    match key {
        "bits" => Scale::Fixed,
        "sample" | "span" | "generalize_after" | "growth_gate" | "growth_gate_min" | "sticky" | "replay" | "max_kernels" | "sleep_every" => Scale::Log,
        "generalize" | "growth_prob" | "grow_trust" | "trust_floor" | "match_fraction" | "surprise_fraction" | "threshold" => Scale::Fraction,
        _ => Scale::Flag,
    }
}

const SCHEDULE_KEYS: [&str; 3] = ["sleep_every", "reset_at_story", "reset_at_sentence"];

fn arity(kind: Kind, n: usize) -> usize {
    match kind {
        Kind::Predict | Kind::Gate | Kind::And | Kind::Or | Kind::Xor | Kind::Clear | Kind::Bag | Kind::Window => 2,
        Kind::Surprise => 3,
        Kind::Delay => 1,
        Kind::Concat => n,
    }
}

fn kind_name(k: Kind) -> &'static str {
    Kind::ALL.iter().find(|x| x.1 == k).map_or("?", |x| x.0)
}

impl GeneList {
    /// Compile the last definition of a stack genome (`make:`, `set:`, numbers, signals,
    /// loops). `place` of a fixed `Prim` and `sub:` are not supported yet.
    pub fn from_genome(g: &Genome) -> Result<GeneList, String> {
        let def = g.defs.last().ok_or("an empty genome")?;
        #[derive(Clone, Copy)]
        enum Sig {
            Real(Source),
            Pending(usize),
        }
        let mut stack: Vec<Option<Sig>> = Vec::new();
        let mut nums: Vec<i64> = Vec::new();
        let mut reg: BTreeMap<String, i64> = BTreeMap::new();
        let mut schedule: BTreeMap<String, i64> = BTreeMap::new();
        let mut bound: Vec<Option<Option<Sig>>> = Vec::new();
        let mut open: Vec<usize> = Vec::new();
        let mut genes: Vec<GeneNode> = Vec::new();
        let mut pending: Vec<(usize, usize, usize)> = Vec::new(); // (gene index, port, feedback)
        let mut outs: Vec<Option<Sig>> = Vec::new();
        for op in &def.code {
            match op {
                NetOp::In(i) => {
                    if *i < def.inputs {
                        stack.push(Some(Sig::Real(Source::In(*i))));
                    }
                }
                NetOp::Num(n) => nums.push(*n),
                NetOp::Set(gene) => {
                    let name = Gene::ALL.iter().find(|x| x.1 == *gene).map(|x| x.0).unwrap_or("?").to_string();
                    let v = nums.pop().unwrap_or(0);
                    if SCHEDULE_KEYS.contains(&name.as_str()) {
                        schedule.insert(name, v);
                    } else {
                        reg.insert(name, v);
                    }
                }
                NetOp::Make(kind) => {
                    let mut pop = |n: usize| -> Vec<i64> {
                        let mut v: Vec<i64> = (0..n).map(|_| nums.pop().unwrap_or(0)).collect();
                        v.reverse();
                        v
                    };
                    let mut params = BTreeMap::new();
                    let mut n_in = 0;
                    match kind {
                        Kind::Predict => {
                            let a = pop(3);
                            params = reg.clone();
                            params.insert("bits".into(), a[0]);
                            params.insert("sample".into(), a[2]);
                        }
                        Kind::Concat => n_in = pop(1)[0].max(1) as usize,
                        Kind::Window => {
                            let a = pop(2);
                            params.insert("span".into(), a[0]);
                            params.insert("keep".into(), a[1]);
                        }
                        Kind::Surprise => {
                            params.insert("threshold".into(), pop(1)[0]);
                        }
                        _ => {}
                    }
                    let n = arity(*kind, n_in);
                    let mut args: Vec<Option<Sig>> = (0..n).map(|_| stack.pop().flatten()).collect();
                    args.reverse();
                    let id = genes.len() as u32 + 1;
                    let gi = genes.len();
                    let inputs = args
                        .iter()
                        .enumerate()
                        .map(|(port, a)| match a {
                            Some(Sig::Real(s)) => Input::wire(Some(*s), false),
                            Some(Sig::Pending(f)) => {
                                pending.push((gi, port, *f));
                                Input::default()
                            }
                            None => Input::default(),
                        })
                        .collect();
                    genes.push(GeneNode { id, kind: *kind, params, inputs, enabled: true });
                    let n_out = if *kind == Kind::Predict { 3 } else { 1 };
                    for p in 0..n_out {
                        stack.push(Some(Sig::Real(Source::Gene(id, p))));
                    }
                }
                NetOp::Dup => {
                    if let Some(&t) = stack.last() {
                        stack.push(t);
                    }
                }
                NetOp::Swap => {
                    let n = stack.len();
                    if n >= 2 {
                        stack.swap(n - 1, n - 2);
                    }
                }
                NetOp::Drop => {
                    stack.pop();
                }
                NetOp::Over => {
                    let n = stack.len();
                    if n >= 2 {
                        stack.push(stack[n - 2]);
                    }
                }
                NetOp::Pick(k) => {
                    let n = stack.len();
                    if *k < n {
                        stack.push(stack[n - 1 - k]);
                    }
                }
                NetOp::Feedback => {
                    bound.push(None);
                    open.push(bound.len() - 1);
                    stack.push(Some(Sig::Pending(bound.len() - 1)));
                }
                NetOp::Close => {
                    let s = stack.pop().flatten();
                    if let Some(f) = open.pop() {
                        bound[f] = Some(s);
                    }
                }
                NetOp::Out => outs.push(stack.pop().flatten()),
                NetOp::Zero => stack.push(None),
                NetOp::Nop => {}
                NetOp::Place(_) | NetOp::Sub(_) => return Err("the gene list does not take place: or sub: yet".into()),
            }
        }
        let resolve = |mut s: Option<Sig>| -> Option<Source> {
            for _ in 0..=bound.len() {
                match s {
                    Some(Sig::Real(src)) => return Some(src),
                    Some(Sig::Pending(f)) => s = bound[f].flatten(),
                    None => return None,
                }
            }
            None
        };
        for (gi, port, f) in pending {
            genes[gi].inputs[port] = Input::wire(resolve(Some(Sig::Pending(f))), true);
        }
        // a source at or after its reader in the stack's order was read from the step before
        let pos: BTreeMap<u32, usize> = genes.iter().enumerate().map(|(i, g)| (g.id, i)).collect();
        for i in 0..genes.len() {
            for inp in genes[i].inputs.iter_mut() {
                for c in inp.conns.iter_mut() {
                    if let Source::Gene(j, _) = c.src {
                        c.prev = pos[&j] >= i;
                    }
                }
            }
        }
        let outputs = outs.into_iter().map(|o| Input::wire(resolve(o), false)).collect();
        let next_id = genes.len() as u32 + 1;
        Ok(GeneList { inputs: def.inputs, genes, outputs, schedule, next_id })
    }

    fn index_of(&self, id: u32) -> Option<usize> {
        self.genes.iter().position(|g| g.id == id)
    }

    /// The tick order: same-step sources before their readers (a topological sort over
    /// connections with gain), ties in list order. A same-step cycle (possible after a
    /// mutation) is read from the step before. Returns the order and, per gene, input and
    /// connection, whether it reads the step before.
    fn order(&self) -> (Vec<usize>, Vec<Vec<Vec<bool>>>) {
        let n = self.genes.len();
        let mut prev: Vec<Vec<Vec<bool>>> = self.genes.iter().map(|g| g.inputs.iter().map(|i| i.conns.iter().map(|c| c.prev).collect()).collect()).collect();
        let live = |i: usize, k: usize, c: usize| self.genes[i].inputs[k].conns[c].gain > 0;
        let mut placed = vec![false; n];
        let mut order = Vec::with_capacity(n);
        while order.len() < n {
            let waits = |i: usize, prev: &Vec<Vec<Vec<bool>>>, placed: &Vec<bool>| -> bool {
                self.genes[i].inputs.iter().enumerate().any(|(k, inp)| {
                    inp.conns.iter().enumerate().any(|(c, conn)| match conn.src {
                        Source::Gene(j, _) if !prev[i][k][c] && live(i, k, c) => self.index_of(j).map_or(false, |x| !placed[x] && self.genes[x].enabled),
                        _ => false,
                    })
                })
            };
            match (0..n).find(|&i| !placed[i] && !waits(i, &prev, &placed)) {
                Some(i) => {
                    placed[i] = true;
                    order.push(i);
                }
                None => {
                    let i = (0..n).find(|&i| !placed[i]).unwrap();
                    for k in 0..self.genes[i].inputs.len() {
                        for c in 0..self.genes[i].inputs[k].conns.len() {
                            if let Source::Gene(j, _) = self.genes[i].inputs[k].conns[c].src {
                                if self.index_of(j).map_or(false, |x| !placed[x]) {
                                    prev[i][k][c] = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        (order, prev)
    }

    /// Build the network (each predictor's random stream seeded by `seed` and its id) and
    /// the update loop.
    pub fn build(&self, seed: u64) -> (Network, Schedule) {
        let mut net = Network::new("genes", self.inputs);
        let (order, prev) = self.order();
        let word_bits = self.genes.iter().filter_map(|g| g.params.get("bits")).copied().max().unwrap_or(64).max(64) as usize;
        let enabled = |j: u32| self.index_of(j).map_or(false, |x| self.genes[x].enabled);
        let mut width: BTreeMap<(u32, usize), usize> = BTreeMap::new();
        let mut child_of: BTreeMap<u32, usize> = BTreeMap::new();
        // wires to genes not placed yet, made at the end (read from the step before)
        let mut late: Vec<(usize, usize, u32, usize)> = Vec::new();
        for &gi in &order {
            let g = &self.genes[gi];
            let conn_width = |c: &Conn| match c.src {
                Source::In(i) => if i == 0 { word_bits } else { 64 },
                Source::Gene(j, p) => width.get(&(j, p)).copied().unwrap_or(word_bits),
            };
            // an input's width is its main (strongest, then first) connection's
            let main = |inp: &Input| -> Option<Conn> { inp.conns.iter().filter(|c| c.gain > 0).fold(None, |m: Option<&Conn>, c| if m.map_or(true, |m| c.gain > m.gain) { Some(c) } else { m }).copied() };
            let natural: Vec<usize> = (0..g.inputs.len().max(1)).map(|k| g.inputs.get(k).and_then(main).map(|c| conn_width(&c)).unwrap_or(word_bits)).collect();
            let in_widths: Vec<usize> = (0..natural.len()).map(|k| match g.inputs.get(k).and_then(|i| i.frames) {
                Some(n) => n.max(1) as usize * word_bits,
                None => natural[k],
            }).collect();
            let w_in = |k: usize| -> usize { in_widths.get(k).copied().unwrap_or(word_bits) };
            let w0 = w_in(0);
            match g.kind {
                Kind::Predict => {
                    let bits = g.params.get("bits").copied().unwrap_or(word_bits as i64).max(64) as usize;
                    width.insert((g.id, 0), bits);
                    width.insert((g.id, 1), bits);
                    width.insert((g.id, 2), 64);
                }
                Kind::Concat => {
                    let w = (0..g.inputs.len()).map(w_in).max().unwrap_or(word_bits);
                    width.insert((g.id, 0), w * g.inputs.len());
                }
                _ => {
                    width.insert((g.id, 0), w0);
                }
            }
            let module: Box<dyn crate::program::modules::Module> = match g.kind {
                Kind::Predict => {
                    let mut genes = Genes::default();
                    for (k, &v) in &g.params {
                        if let Some(&(_, gene)) = Gene::ALL.iter().find(|x| x.0 == k) {
                            genes.set(gene, v);
                        }
                    }
                    let bits = g.params.get("bits").copied().unwrap_or(word_bits as i64).max(64) as usize;
                    let frames = w0.div_ceil(bits).max(1);
                    let sample = g.params.get("sample").copied().unwrap_or(16).max(1) as usize;
                    let mut p = Predictor::new(bits, genes.spec.class(bits, frames, sample));
                    p.confidence_port = true;
                    p.no_sleep = genes.spec.no_sleep;
                    p.own_rng = Some(StdRng::seed_from_u64(seed ^ (g.id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
                    Box::new(p)
                }
                k => {
                    let mut nums: Vec<i64> = match k {
                        Kind::Concat => vec![g.inputs.len() as i64],
                        Kind::Window => vec![g.params.get("span").copied().unwrap_or(4), g.params.get("keep").copied().unwrap_or(0)],
                        Kind::Surprise => vec![g.params.get("threshold").copied().unwrap_or(32768)],
                        _ => vec![],
                    };
                    Genes::default().make(k, &mut nums)
                }
            };
            if !g.enabled {
                child_of.insert(g.id, net.place(Box::<crate::program::modules::Delay>::default(), &[]));
                continue;
            }
            // one connection's source: the input port, an earlier gene (through a delay when
            // it is read from the step before), or a later gene (wired at the end)
            let mut srcs: Vec<Option<NetSrc>> = Vec::new();
            let mut late_here: Vec<(usize, u32, usize)> = Vec::new(); // (port, gene, out port)
            for (k, inp) in g.inputs.iter().enumerate() {
                let live: Vec<(usize, &Conn)> = inp.conns.iter().enumerate().filter(|(_, c)| c.gain > 0 && match c.src {
                    Source::Gene(j, _) => enabled(j),
                    Source::In(_) => true,
                }).collect();
                let mut resolve = |net: &mut Network, c: usize, conn: &Conn| -> Result<NetSrc, (u32, usize)> {
                    match conn.src {
                        Source::In(i) => Ok(NetSrc::In(i)),
                        Source::Gene(j, p) => match child_of.get(&j) {
                            Some(&cj) if prev[gi][k][c] => Ok(NetSrc::Child(net.place(Box::<crate::program::modules::Delay>::default(), &[Some(NetSrc::Child(cj, p))]), 0)),
                            Some(&cj) => Ok(NetSrc::Child(cj, p)),
                            None => Err((j, p)),
                        },
                    }
                };
                // a single connection at full gain, at its own width, is a plain wire
                let reshaped = inp.frames.map_or(false, |n| n as usize * word_bits != natural[k]);
                let s = match live.as_slice() {
                    [] => None,
                    [(c, conn)] if conn.gain >= FULL && !reshaped => match resolve(&mut net, *c, conn) {
                        Ok(s) => Some(s),
                        Err((j, p)) => {
                            late_here.push((k, j, p));
                            None
                        }
                    },
                    many => {
                        let gains: Vec<u32> = many.iter().map(|(_, c)| c.gain.min(FULL)).collect();
                        let mut ins: Vec<Option<NetSrc>> = Vec::new();
                        let mut later: Vec<(usize, u32, usize)> = Vec::new();
                        for (port, (c, conn)) in many.iter().enumerate() {
                            match resolve(&mut net, *c, conn) {
                                Ok(s) => ins.push(Some(s)),
                                Err((j, p)) => {
                                    ins.push(None);
                                    later.push((port, j, p));
                                }
                            }
                        }
                        let fw = word_bits.div_ceil(64);
                        let b = net.place(Box::new(crate::program::modules::Blend::new(gains, w_in(k).div_ceil(word_bits), fw)), &ins);
                        for (port, j, p) in later {
                            late.push((b, port, j, p));
                        }
                        Some(NetSrc::Child(b, 0))
                    }
                };
                srcs.push(s);
            }
            let c = net.place(module, &srcs);
            for (k, j, p) in late_here {
                late.push((c, k, j, p));
            }
            child_of.insert(g.id, c);
        }
        for (c, k, j, p) in late {
            if let Some(&cj) = child_of.get(&j) {
                net.connect(c, k, NetSrc::Child(cj, p));
            }
        }
        for o in &self.outputs {
            let s = o.conns.first().and_then(|c| match c.src {
                Source::In(i) => Some(NetSrc::In(i)),
                Source::Gene(j, p) => child_of.get(&j).filter(|_| enabled(j)).map(|&c| NetSrc::Child(c, p)),
            });
            net.export(s);
        }
        let mut genes = Genes::default();
        for (k, &v) in &self.schedule {
            if let Some(&(_, gene)) = Gene::ALL.iter().find(|x| x.0 == k) {
                genes.set(gene, v);
            }
        }
        (net, genes.schedule)
    }

    /// One line per gene; a connection below full gain shows it in sixteenths.
    pub fn to_text(&self) -> String {
        let mut s = format!("inputs {}\n", self.inputs);
        let conn = |c: &Conn| {
            let src = match c.src {
                Source::In(p) => format!("in:{p}"),
                Source::Gene(j, p) => format!("g{j}:{p}"),
            };
            format!("{src}{}{}", if c.prev { "@prev" } else { "" }, if c.gain < FULL { format!("*{}/16", c.gain / STEP) } else { String::new() })
        };
        let input = |i: &Input| {
            let body = match i.conns.len() {
                0 => "zero".to_string(),
                1 => conn(&i.conns[0]),
                _ => format!("[{}]", i.conns.iter().map(conn).collect::<Vec<_>>().join(" + ")),
            };
            match i.frames {
                Some(n) => format!("{body}|{n} frames"),
                None => body,
            }
        };
        for g in &self.genes {
            let ins: Vec<String> = g.inputs.iter().map(input).collect();
            let ps: Vec<String> = g.params.iter().map(|(k, v)| format!("{k}={v}")).collect();
            s += &format!("{}g{} = {}({}) {}\n", if g.enabled { "" } else { "# off: " }, g.id, kind_name(g.kind), ins.join(", "), ps.join(" "));
        }
        for o in &self.outputs {
            s += &format!("out {}\n", input(o));
        }
        let sch: Vec<String> = self.schedule.iter().map(|(k, v)| format!("{k}={v}")).collect();
        s += &format!("schedule {}\n", sch.join(" "));
        s
    }

    /// Whether gene `a`'s output reaches gene `b` within one step (same-step connections).
    fn reaches(&self, a: u32, b: u32) -> bool {
        let mut seen = vec![a];
        let mut i = 0;
        while i < seen.len() {
            let x = seen[i];
            i += 1;
            if x == b {
                return true;
            }
            for g in &self.genes {
                let reads = g.inputs.iter().any(|inp| inp.conns.iter().any(|c| !c.prev && matches!(c.src, Source::Gene(y, _) if y == x)));
                if reads && !seen.contains(&g.id) {
                    seen.push(g.id);
                }
            }
        }
        false
    }

    /// Bit-signal outputs a connection may read: the word and every gene's non-scalar
    /// outputs.
    fn bit_sources(&self) -> Vec<Source> {
        let mut v = vec![Source::In(0)];
        for g in &self.genes {
            let n = if g.kind == Kind::Predict { 2 } else { 1 };
            for p in 0..n {
                v.push(Source::Gene(g.id, p));
            }
        }
        v
    }

    /// An input's natural width in frames: its main connection's (a predictor's output and
    /// the word are one frame; a concat's is the sum of its inputs').
    fn natural_frames(&self, gi: usize, k: usize, word_bits: u32) -> u32 {
        let main = self.genes[gi].inputs[k].conns.iter().filter(|c| c.gain > 0).max_by_key(|c| c.gain).copied();
        let mut seen = 0;
        let mut frames_of = |src: Source| -> u32 {
            fn go(l: &GeneList, src: Source, depth: &mut u32) -> u32 {
                *depth += 1;
                if *depth > 64 {
                    return 1;
                }
                match src {
                    Source::In(_) => 1,
                    Source::Gene(j, _) => match l.index_of(j).map(|x| &l.genes[x]) {
                        Some(g) if g.kind == Kind::Concat => g.inputs.len() as u32 * g.inputs.iter().map(|i| i.frames.unwrap_or_else(|| i.conns.iter().max_by_key(|c| c.gain).map_or(1, |c| go(l, c.src, depth)))).max().unwrap_or(1),
                        Some(g) if g.kind == Kind::Predict => 1,
                        Some(g) => g.inputs.first().map_or(1, |i| i.frames.unwrap_or_else(|| i.conns.iter().max_by_key(|c| c.gain).map_or(1, |c| go(l, c.src, depth)))),
                        None => 1,
                    },
                }
            }
            go(self, src, &mut seen)
        };
        let _ = word_bits;
        main.map_or(1, |c| frames_of(c.src))
    }

    /// The input ports a new connection may join: bit signals, not a clock or a
    /// confidence, not a predictor's teaching port (the teacher's path is fixed, as a
    /// climbing fibre's is: extra bits there become part of what is learned), and not a
    /// comparator's (what is compared with a prediction decides what counts as surprising).
    fn open_ports(&self) -> Vec<(usize, usize)> {
        self.bit_ports().into_iter().filter(|&(i, k)| !(self.genes[i].kind == Kind::Predict && k == 1) && self.genes[i].kind != Kind::Surprise).collect()
    }

    /// The bit-signal input ports (not a clock or a confidence).
    fn bit_ports(&self) -> Vec<(usize, usize)> {
        self.genes
            .iter()
            .enumerate()
            .flat_map(|(i, g)| {
                let ports: Vec<usize> = match g.kind {
                    Kind::Bag | Kind::Window => vec![0],
                    Kind::Surprise => vec![0, 1],
                    _ => (0..g.inputs.len()).collect(),
                };
                ports.into_iter().map(move |k| (i, k))
            })
            .collect()
    }

    /// A new connection into `(gene, input)` from a random source not already there.
    fn new_conn(&mut self, gi: usize, k: usize, gain: u32, rng: &mut StdRng) -> Option<String> {
        let id = self.genes[gi].id;
        let have: Vec<Source> = self.genes[gi].inputs[k].conns.iter().map(|c| c.src).collect();
        let cands: Vec<Source> = self.bit_sources().into_iter().filter(|s| !have.contains(s) && !matches!(s, Source::Gene(j, _) if *j == id)).collect();
        if cands.is_empty() {
            return None;
        }
        let src = cands[rng.gen_range(0..cands.len())];
        let prev = match src {
            Source::Gene(j, _) => self.reaches(id, j),
            Source::In(_) => false,
        };
        self.genes[gi].inputs[k].conns.push(Conn { src, prev, gain });
        Some(format!("g{id} input {k} <- {src:?}{} at {}/16", if prev { " (step before)" } else { "" }, gain / STEP))
    }

    /// Apply one local mutation. Operators:
    /// - `nudge`: one number (× or ÷ 2^(1/4), a fraction ± 1/16, a flag flipped) or one
    ///   connection's gain (± 1/16);
    /// - `connect`: a new connection at gain 0 (neutral); `strengthen`: a new one at 1/16
    ///   (the step after);
    /// - `weaken`: a connection with gain loses 1/16;
    /// - `add`: a gene nothing reads yet; `duplicate`: a silent copy of a gene;
    /// - `rewire` (one input replaced by another source at full gain) and `toggle` (a gene
    ///   off or on): the abrupt forms, kept for comparison.
    ///
    /// Returns a description.
    pub fn mutate(&mut self, op: &str, rng: &mut StdRng) -> String {
        match op {
            "nudge" => {
                // (gene or schedule, key) for numbers; (gene, input, conn) for gains
                let mut nums: Vec<(Option<usize>, String)> = Vec::new();
                for (i, g) in self.genes.iter().enumerate() {
                    for k in g.params.keys() {
                        if scale_of(k) != Scale::Fixed {
                            nums.push((Some(i), k.clone()));
                        }
                    }
                }
                for k in self.schedule.keys() {
                    nums.push((None, k.clone()));
                }
                let gains: Vec<(usize, usize, usize)> = self
                    .bit_ports()
                    .into_iter()
                    .flat_map(|(i, k)| (0..self.genes[i].inputs[k].conns.len()).map(move |c| (i, k, c)))
                    .collect();
                let total = nums.len() + gains.len();
                if total == 0 {
                    return "nudge: nothing to nudge".into();
                }
                let r = rng.gen_range(0..total);
                let up = rng.gen_bool(0.5);
                if r >= nums.len() {
                    let (i, k, c) = gains[r - nums.len()];
                    let id = self.genes[i].id;
                    let conn = &mut self.genes[i].inputs[k].conns[c];
                    let v = conn.gain;
                    conn.gain = if up { (v + STEP).min(FULL) } else { v.saturating_sub(STEP) };
                    return format!("nudge g{id} input {k} gain from {:?}: {}/16 -> {}/16", conn.src, v / STEP, conn.gain / STEP);
                }
                let (gi, key) = nums[r].clone();
                let map = match gi {
                    Some(i) => &mut self.genes[i].params,
                    None => &mut self.schedule,
                };
                let v = map[&key];
                let nv = match scale_of(&key) {
                    Scale::Log => {
                        let x = if up { (v * 1189 + 500) / 1000 } else { (v * 1000 + 594) / 1189 };
                        if x == v { if up { v + 1 } else { (v - 1).max(0) } } else { x.max(0) }
                    }
                    Scale::Fraction => (v + if up { 4096 } else { -4096 }).clamp(0, 65536),
                    Scale::Flag => (v == 0) as i64,
                    Scale::Fixed => v,
                };
                map.insert(key.clone(), nv);
                format!("nudge {} {key}: {v} -> {nv}", gi.map_or("schedule".to_string(), |i| format!("g{}", self.genes[i].id)))
            }
            "connect" | "strengthen" => {
                let ports = self.open_ports();
                if ports.is_empty() {
                    return format!("{op}: no input");
                }
                let (gi, k) = ports[rng.gen_range(0..ports.len())];
                let gain = if op == "connect" { 0 } else { STEP };
                self.new_conn(gi, k, gain, rng).map_or(format!("{op}: no new source"), |d| format!("{op} {d}"))
            }
            "reshape" => {
                // one input one frame wider or narrower (its sources folded or padded to it)
                let ports = self.open_ports();
                if ports.is_empty() {
                    return "reshape: no input".into();
                }
                let (gi, k) = ports[rng.gen_range(0..ports.len())];
                let id = self.genes[gi].id;
                let word_bits = self.genes.iter().filter_map(|g| g.params.get("bits")).copied().max().unwrap_or(64).max(64) as u32;
                let now = self.genes[gi].inputs[k].frames.unwrap_or_else(|| self.natural_frames(gi, k, word_bits));
                let up = now <= 1 || rng.gen_bool(0.5);
                let n = if up { (now + 1).min(8) } else { now - 1 };
                self.genes[gi].inputs[k].frames = Some(n);
                format!("reshape g{id} input {k}: {now} -> {n} frames")
            }
            "weaken" => {
                let used: Vec<(usize, usize, usize)> = self
                    .bit_ports()
                    .into_iter()
                    .flat_map(|(i, k)| (0..self.genes[i].inputs[k].conns.len()).map(move |c| (i, k, c)))
                    .filter(|&(i, k, c)| self.genes[i].inputs[k].conns[c].gain > 0)
                    .collect();
                if used.is_empty() {
                    return "weaken: no connection".into();
                }
                let (i, k, c) = used[rng.gen_range(0..used.len())];
                let id = self.genes[i].id;
                let conn = &mut self.genes[i].inputs[k].conns[c];
                let v = conn.gain;
                conn.gain = v.saturating_sub(STEP);
                format!("weaken g{id} input {k} from {:?}: {}/16 -> {}/16", conn.src, v / STEP, conn.gain / STEP)
            }
            "rewire" => {
                let ports = self.bit_ports();
                if ports.is_empty() {
                    return "rewire: no input".into();
                }
                let (gi, k) = ports[rng.gen_range(0..ports.len())];
                let old = self.genes[gi].inputs[k].clone();
                self.genes[gi].inputs[k].conns.clear();
                let d = self.new_conn(gi, k, FULL, rng);
                if d.is_none() {
                    self.genes[gi].inputs[k] = old;
                }
                format!("rewire (abrupt) {}", d.unwrap_or_default())
            }
            "add" | "duplicate" => {
                let new = if op == "duplicate" && !self.genes.is_empty() {
                    let mut g = self.genes[rng.gen_range(0..self.genes.len())].clone();
                    g.id = self.next_id;
                    g
                } else {
                    let kinds = [Kind::Bag, Kind::Window, Kind::Surprise, Kind::Delay, Kind::Predict, Kind::Or];
                    let kind = kinds[rng.gen_range(0..kinds.len())];
                    let srcs = self.bit_sources();
                    let mut pick = || Input::wire(Some(srcs[rng.gen_range(0..srcs.len())]), false);
                    let clock = Input::wire(Some(Source::In(1)), false);
                    let mut params = BTreeMap::new();
                    let inputs = match kind {
                        Kind::Bag => vec![pick(), clock],
                        Kind::Window => {
                            params.insert("span".into(), 4);
                            params.insert("keep".into(), 1);
                            vec![pick(), clock]
                        }
                        Kind::Surprise => {
                            params.insert("threshold".into(), 32768);
                            vec![pick(), pick(), Input::default()]
                        }
                        Kind::Predict => {
                            if let Some(p) = self.genes.iter().find(|g| g.kind == Kind::Predict) {
                                params = p.params.clone();
                            }
                            vec![pick(), Input::wire(Some(Source::In(0)), false)]
                        }
                        Kind::Delay => vec![pick()],
                        _ => vec![pick(), pick()],
                    };
                    GeneNode { id: self.next_id, kind, params, inputs, enabled: true }
                };
                self.next_id += 1;
                let d = format!("{op} g{} = {}", new.id, kind_name(new.kind));
                self.genes.push(new);
                d + " (nothing reads it yet)"
            }
            "toggle" => {
                if self.genes.is_empty() {
                    return "toggle: no gene".into();
                }
                let i = rng.gen_range(0..self.genes.len());
                self.genes[i].enabled = !self.genes[i].enabled;
                format!("toggle (abrupt) g{} {}", self.genes[i].id, if self.genes[i].enabled { "on" } else { "off" })
            }
            _ => format!("unknown operator {op}"),
        }
    }
}

/// The stack text's counterpart: one mutation of its tokens. `number` nudges a number
/// (as `nudge` does); `delete`, `insert` (a random word of the language) and `swap` (two
/// neighbours) are the stack's structural changes. Returns the new text and a description.
pub fn mutate_stack_text(text: &str, op: &str, rng: &mut StdRng) -> (String, String) {
    let mut toks: Vec<String> = text.lines().flat_map(|l| l.split('#').next().unwrap_or("").split_whitespace().map(String::from).collect::<Vec<_>>()).collect();
    // only the body of the last definition: not `def`, its name and inputs, or `end`
    let body: Vec<usize> = (0..toks.len()).filter(|&i| !(toks[i] == "def" || toks[i] == "end" || (i >= 1 && toks[i - 1] == "def") || (i >= 2 && toks[i - 2] == "def"))).collect();
    if body.is_empty() {
        return (text.into(), "no body".into());
    }
    let at = body[rng.gen_range(0..body.len())];
    let desc = match op {
        "number" => {
            let nums: Vec<usize> = body.iter().copied().filter(|&i| toks[i].parse::<i64>().is_ok()).collect();
            if nums.is_empty() {
                return (text.into(), "no number".into());
            }
            let i = nums[rng.gen_range(0..nums.len())];
            let v: i64 = toks[i].parse().unwrap();
            let up = rng.gen_bool(0.5);
            let x = if up { (v * 1189 + 500) / 1000 } else { (v * 1000 + 594) / 1189 };
            let nv = if x == v { if up { v + 1 } else { (v - 1).max(0) } } else { x.max(0) };
            toks[i] = nv.to_string();
            format!("number at {i}: {v} -> {nv}")
        }
        "delete" => {
            let t = toks.remove(at);
            format!("delete `{t}` at {at}")
        }
        "insert" => {
            let words = ["dup", "swap", "drop", "over", "zero", "in:0", "in:1", "make:delay", "make:bag", "make:or", "out", "1", "2"];
            let w = words[rng.gen_range(0..words.len())];
            toks.insert(at, w.to_string());
            format!("insert `{w}` at {at}")
        }
        "swap" => {
            let j = if at + 1 < toks.len() && body.contains(&(at + 1)) { at + 1 } else { at.saturating_sub(1) };
            toks.swap(at, j);
            format!("swap `{}` and `{}`", toks[j], toks[at])
        }
        _ => format!("unknown operator {op}"),
    };
    // one token per line keeps `#` comments out of the way
    (toks.join("\n"), desc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hierarchy_compiles_to_genes_and_back_to_the_same_circuit() {
        let g = Genome::parse(include_str!("../../genomes/hierarchy.gen")).unwrap();
        let list = GeneList::from_genome(&g).unwrap();
        let text = list.to_text();
        // two predictors, a window over surprises, and loops from the column back to surprise
        assert_eq!(list.genes.iter().filter(|x| x.kind == Kind::Predict).count(), 2);
        assert!(text.contains("window(") && text.contains("@prev"), "{text}");
        let (net, schedule) = list.build(1);
        assert_eq!(schedule.sleep_every, 500);
        assert_eq!(crate::program::modules::Module::n_outputs(&net), 1);
    }

    #[test]
    fn silent_additions_leave_the_output_unchanged() {
        use crate::bitvec::BitVector;
        use crate::program::modules::{Ctx, Module};
        let g = Genome::parse(include_str!("../../genomes/hierarchy.gen")).unwrap();
        let base = GeneList::from_genome(&g).unwrap();
        let mut rng = StdRng::seed_from_u64(3);
        let mut grown = base.clone();
        for _ in 0..4 {
            grown.mutate("add", &mut rng);
            grown.mutate("duplicate", &mut rng);
            grown.mutate("connect", &mut rng);
        }
        let words: Vec<BitVector> = (0..6).map(|i| BitVector::from_bits(&(0..32).map(|j| (i * 97 + j * 13) % 8192).collect::<Vec<_>>(), 8192)).collect();
        let run = |list: &GeneList| -> Vec<BitVector> {
            let (mut net, _) = list.build(7);
            let mut r = StdRng::seed_from_u64(9);
            let mut outs = Vec::new();
            for step in 0..60 {
                let w = &words[step % 6];
                let clock = BitVector::EMPTY;
                net.tick(&[w, &clock], &mut Ctx { rng: &mut r, learn: true });
                outs.push(net.output(0).clone());
            }
            outs
        };
        let (a, b) = (run(&base), run(&grown));
        assert!(a.iter().zip(&b).all(|(x, y)| x.as_words() == y.as_words()), "genes nothing reads, and connections at gain 0, change nothing");
    }

    #[test]
    fn a_wider_source_folds_frame_by_frame_and_a_narrower_one_is_padded() {
        use crate::bitvec::BitVector;
        use crate::program::modules::{Blend, Ctx, Module};
        // three one-word frames: bits 0, 64 + 1, 128 + 2
        let x = BitVector::from_bits(&[0, 65, 130], 192);
        let mut r = StdRng::seed_from_u64(1);
        let mut fold = |frames: usize| {
            let mut b = Blend::new(vec![65536], frames, 1);
            b.tick(&[&x], &mut Ctx { rng: &mut r, learn: false });
            b.output(0).as_words().to_vec()
        };
        assert_eq!(fold(1), vec![0b111], "all three frames superposed in one");
        assert_eq!(fold(2), vec![0b101, 0b10], "frames 0 and 2 in the first, 1 in the second");
        assert_eq!(fold(4), vec![1, 2, 4, 0], "padded with an empty frame");
    }

    #[test]
    fn reshaped_inputs_build_and_run() {
        use crate::bitvec::BitVector;
        use crate::program::modules::{Ctx, Module};
        let g = Genome::parse(include_str!("../../genomes/hierarchy.gen")).unwrap();
        let mut list = GeneList::from_genome(&g).unwrap();
        let mut rng = StdRng::seed_from_u64(5);
        for _ in 0..6 {
            list.mutate("reshape", &mut rng);
        }
        assert!(list.to_text().contains("frames"), "{}", list.to_text());
        let (mut net, _) = list.build(1);
        let w = BitVector::from_bits(&(0..32).collect::<Vec<_>>(), 8192);
        let mut r = StdRng::seed_from_u64(2);
        for _ in 0..20 {
            net.tick(&[&w, &BitVector::EMPTY], &mut Ctx { rng: &mut r, learn: true });
        }
        assert_eq!(net.output(0).bit_len(), 8192);
    }
}
