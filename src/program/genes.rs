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
//!   does not shift every other module's draws.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Input {
    pub src: Option<Source>,
    /// Read the source's value from the step before (a loop).
    pub prev: bool,
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
                            Some(Sig::Real(s)) => Input { src: Some(*s), prev: false },
                            Some(Sig::Pending(f)) => {
                                pending.push((gi, port, *f));
                                Input { src: None, prev: true }
                            }
                            None => Input { src: None, prev: false },
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
            genes[gi].inputs[port].src = resolve(Some(Sig::Pending(f)));
        }
        // a source at or after its reader in the stack's order was read from the step before
        let pos: BTreeMap<u32, usize> = genes.iter().enumerate().map(|(i, g)| (g.id, i)).collect();
        for i in 0..genes.len() {
            for inp in genes[i].inputs.iter_mut() {
                if let Some(Source::Gene(j, _)) = inp.src {
                    inp.prev = pos[&j] >= i;
                }
            }
        }
        let outputs = outs.into_iter().map(|o| Input { src: resolve(o), prev: false }).collect();
        let next_id = genes.len() as u32 + 1;
        Ok(GeneList { inputs: def.inputs, genes, outputs, schedule, next_id })
    }

    fn index_of(&self, id: u32) -> Option<usize> {
        self.genes.iter().position(|g| g.id == id)
    }

    /// The tick order: same-step inputs before their readers (a topological sort), ties in
    /// list order. A same-step cycle (possible after rewiring) is read from the step before.
    fn order(&self) -> (Vec<usize>, Vec<Vec<bool>>) {
        let n = self.genes.len();
        let mut prev: Vec<Vec<bool>> = self.genes.iter().map(|g| g.inputs.iter().map(|i| i.prev).collect()).collect();
        let mut placed = vec![false; n];
        let mut order = Vec::with_capacity(n);
        while order.len() < n {
            let ready = (0..n).find(|&i| {
                !placed[i]
                    && self.genes[i].inputs.iter().zip(&prev[i]).all(|(inp, &p)| match inp.src {
                        Some(Source::Gene(j, _)) if !p => self.index_of(j).map_or(true, |k| placed[k] || !self.genes[k].enabled),
                        _ => true,
                    })
            });
            match ready {
                Some(i) => {
                    placed[i] = true;
                    order.push(i);
                }
                None => {
                    // a cycle: the first unplaced gene reads its unplaced inputs from the step before
                    let i = (0..n).find(|&i| !placed[i]).unwrap();
                    for (k, inp) in self.genes[i].inputs.iter().enumerate() {
                        if let Some(Source::Gene(j, _)) = inp.src {
                            if self.index_of(j).map_or(false, |x| !placed[x]) {
                                prev[i][k] = true;
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
        // static widths (bits) of every output, in tick order
        let mut width: BTreeMap<(u32, usize), usize> = BTreeMap::new();
        let in_width = |i: usize| if i == 0 { word_bits } else { 64 };
        let mut child_of: BTreeMap<u32, usize> = BTreeMap::new();
        for &gi in &order {
            let g = &self.genes[gi];
            let w_in = |k: usize| -> usize {
                match g.inputs.get(k).and_then(|x| x.src) {
                    Some(Source::In(i)) => in_width(i),
                    Some(Source::Gene(j, p)) => width.get(&(j, p)).copied().unwrap_or(word_bits),
                    None => word_bits,
                }
            };
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
            // the module
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
            // its wires; a loop from an earlier gene goes through a delay placed just before it
            let mut srcs: Vec<Option<NetSrc>> = Vec::new();
            for (k, inp) in g.inputs.iter().enumerate() {
                let s = match inp.src {
                    None => None,
                    Some(Source::In(i)) => Some(NetSrc::In(i)),
                    Some(Source::Gene(j, p)) => match (self.index_of(j), child_of.get(&j)) {
                        (Some(x), _) if !self.genes[x].enabled => None,
                        (None, _) => None,
                        (Some(_), Some(&c)) if prev[gi][k] => {
                            let d = net.place(Box::<crate::program::modules::Delay>::default(), &[Some(NetSrc::Child(c, p))]);
                            Some(NetSrc::Child(d, 0))
                        }
                        (Some(_), Some(&c)) => Some(NetSrc::Child(c, p)),
                        // a loop from a gene not placed yet: read later, so the step before
                        (Some(_), None) => None,
                    },
                };
                srcs.push(s);
            }
            let c = if g.enabled { net.place(module, &srcs) } else { net.place(Box::<crate::program::modules::Delay>::default(), &[]) };
            child_of.insert(g.id, c);
        }
        // loops to genes placed after their reader: wire now (the network reads them from
        // the step before)
        for &gi in &order {
            let g = &self.genes[gi];
            if !g.enabled {
                continue;
            }
            let c = child_of[&g.id];
            for (k, inp) in g.inputs.iter().enumerate() {
                if let Some(Source::Gene(j, p)) = inp.src {
                    if let (Some(&cj), Some(x)) = (child_of.get(&j), self.index_of(j)) {
                        if cj > c && self.genes[x].enabled {
                            net.connect(c, k, NetSrc::Child(cj, p));
                        }
                    }
                }
            }
        }
        for o in &self.outputs {
            net.export(match o.src {
                Some(Source::In(i)) => Some(NetSrc::In(i)),
                Some(Source::Gene(j, p)) => child_of.get(&j).filter(|_| self.index_of(j).map_or(false, |x| self.genes[x].enabled)).map(|&c| NetSrc::Child(c, p)),
                None => None,
            });
        }
        let mut genes = Genes::default();
        for (k, &v) in &self.schedule {
            if let Some(&(_, gene)) = Gene::ALL.iter().find(|x| x.0 == k) {
                genes.set(gene, v);
            }
        }
        (net, genes.schedule)
    }

    /// One line per gene.
    pub fn to_text(&self) -> String {
        let mut s = format!("inputs {}\n", self.inputs);
        let src = |i: &Input| match i.src {
            Some(Source::In(p)) => format!("in:{p}"),
            Some(Source::Gene(j, p)) => format!("g{j}:{p}{}", if i.prev { "@prev" } else { "" }),
            None => "zero".into(),
        };
        for g in &self.genes {
            let ins: Vec<String> = g.inputs.iter().map(src).collect();
            let ps: Vec<String> = g.params.iter().map(|(k, v)| format!("{k}={v}")).collect();
            s += &format!("{}g{} = {}({}) {}\n", if g.enabled { "" } else { "# off: " }, g.id, kind_name(g.kind), ins.join(", "), ps.join(" "));
        }
        for o in &self.outputs {
            s += &format!("out {}\n", src(o));
        }
        let sch: Vec<String> = self.schedule.iter().map(|(k, v)| format!("{k}={v}")).collect();
        s += &format!("schedule {}\n", sch.join(" "));
        s
    }

    /// Whether gene `a`'s output reaches gene `b` within one step (same-step wires only).
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
                if g.inputs.iter().any(|inp| !inp.prev && inp.src == Some(Source::Gene(x, 0)) || !inp.prev && matches!(inp.src, Some(Source::Gene(y, _)) if y == x)) && !seen.contains(&g.id) {
                    seen.push(g.id);
                }
            }
        }
        false
    }

    /// Bit-signal outputs a rewired input may read: the word and every gene's non-scalar
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

    /// Apply one local mutation. Operators: `nudge` one number; `rewire` one input; `add` a
    /// gene that nothing reads yet (silent); `duplicate` a gene (the copy is silent);
    /// `toggle` a gene on or off. Returns a description.
    pub fn mutate(&mut self, op: &str, rng: &mut StdRng) -> String {
        match op {
            "nudge" => {
                let mut slots: Vec<(Option<usize>, String)> = Vec::new();
                for (i, g) in self.genes.iter().enumerate() {
                    for k in g.params.keys() {
                        if scale_of(k) != Scale::Fixed {
                            slots.push((Some(i), k.clone()));
                        }
                    }
                }
                for k in self.schedule.keys() {
                    slots.push((None, k.clone()));
                }
                if slots.is_empty() {
                    return "nudge: nothing to nudge".into();
                }
                let (gi, key) = slots[rng.gen_range(0..slots.len())].clone();
                let map = match gi {
                    Some(i) => &mut self.genes[i].params,
                    None => &mut self.schedule,
                };
                let v = map[&key];
                let up = rng.gen_bool(0.5);
                let nv = match scale_of(&key) {
                    Scale::Log => {
                        // × or ÷ 2^(1/4), by at least 1
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
            "rewire" => {
                let slots: Vec<(usize, usize)> = self
                    .genes
                    .iter()
                    .enumerate()
                    .flat_map(|(i, g)| {
                        let bit_ports: Vec<usize> = match g.kind {
                            Kind::Bag | Kind::Window => vec![0],
                            Kind::Surprise => vec![0, 1],
                            _ => (0..g.inputs.len()).collect(),
                        };
                        bit_ports.into_iter().map(move |k| (i, k))
                    })
                    .collect();
                if slots.is_empty() {
                    return "rewire: no input".into();
                }
                let (gi, k) = slots[rng.gen_range(0..slots.len())];
                let id = self.genes[gi].id;
                let cands: Vec<Source> = self.bit_sources().into_iter().filter(|s| *s != Source::Gene(id, 0) && *s != Source::Gene(id, 1)).collect();
                let s = cands[rng.gen_range(0..cands.len())];
                let prev = match s {
                    Source::Gene(j, _) => self.reaches(id, j),
                    Source::In(_) => false,
                };
                let old = self.genes[gi].inputs[k];
                self.genes[gi].inputs[k] = Input { src: Some(s), prev };
                format!("rewire g{id} input {k}: {:?} -> {:?}{}", old.src, s, if prev { " (from the step before)" } else { "" })
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
                    let mut pick = || Input { src: Some(srcs[rng.gen_range(0..srcs.len())]), prev: false };
                    let mut params = BTreeMap::new();
                    let inputs = match kind {
                        Kind::Bag => vec![pick(), Input { src: Some(Source::In(1)), prev: false }],
                        Kind::Window => {
                            params.insert("span".into(), 4);
                            params.insert("keep".into(), 1);
                            vec![pick(), Input { src: Some(Source::In(1)), prev: false }]
                        }
                        Kind::Surprise => {
                            params.insert("threshold".into(), 32768);
                            vec![pick(), pick(), Input { src: None, prev: false }]
                        }
                        Kind::Predict => {
                            if let Some(p) = self.genes.iter().find(|g| g.kind == Kind::Predict) {
                                params = p.params.clone();
                            }
                            vec![pick(), Input { src: Some(Source::In(0)), prev: false }]
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
                format!("toggle g{} {}", self.genes[i].id, if self.genes[i].enabled { "on" } else { "off" })
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
        assert!(a.iter().zip(&b).all(|(x, y)| x.as_words() == y.as_words()), "genes nothing reads change nothing");
    }
}
