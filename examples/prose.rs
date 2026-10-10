//! Reading and writing prose with the plausible loop.
//!
//! Reads a text (PROSE_FILE, default data/alice.txt) word by word with learning on, measures
//! next-word prediction on a held-out tail, then speaks: it says its plan through a learned
//! motor path, hears the word, and reads on from it.
//!
//! - **Layer 4** recodes the heard word (learned k-winners-take-all).
//! - **Recurrent layer 2/3** (bitwise primed cells, self-calibrating, tag and capture): basal
//!   input is layer 4's code and its own previous activity (the context); its tuft (layer 1)
//!   gets the cerebellum's output through the motor thalamus.
//! - **Layer 5** (bitwise primed cells): basal input is layer 4's code and layer 2/3's
//!   prediction; its tuft gets the cerebellum's output. Bursts are strong drivers of the
//!   thalamus, single spikes weak ones.
//! - **The cerebellar circuit:** mossy fibres carry the heard word and the pons's recoding of
//!   what layer 5 fired; granule expansion, Purkinje depression by the climbing fibre,
//!   relative readout.
//! - **The thalamic relay** weighs each source (layer 2/3, layer 5, the cerebellum) by its
//!   driver (burst, spike, tonic), its learned reliability in this context, and agreement with
//!   the cerebellum, sums the weighted patterns per bit and keeps the strongest 32 bits
//!   (k-winners-take-all): the plan. No word is decoded inside the network.
//! - **Speaking** (after babbling): the motor area's learned inverse model turns the plan into
//!   a command; the vocal tract (the world) says the word whose articulation it matches; the
//!   word is heard and read on.
//!
//! Options: PROSE_FILE, PROSE_VOCAB (most frequent words kept, default 3000), PROSE_WORDS
//! (words read, default all), PROSE_GEN (words to write, default 200), PROSE_PROMPT,
//! PROSE_TEMP (sampling noise on the relay's weights, 0–100, default 20), SEED.

mod common;

use common::Encoder;
use neurocomp::bitvec::BitVector;
use neurocomp::fixed::{Q16, ONE};
use neurocomp::program::{BitCells, CerebellarCircuit, Driver, Layer4, MotorArea, ThalamicRelay, VocalTract};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const BITS: usize = 2048;
const ACTIVE: usize = 32;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        let c = c.to_ascii_lowercase();
        if c.is_ascii_alphabetic() || (c == '\'' && !cur.is_empty()) {
            cur.push(c);
        } else {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            if ".,;:!?".contains(c) {
                out.push(c.to_string());
            }
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn frames(fs: &[&BitVector]) -> BitVector {
    let mut w = Vec::new();
    for f in fs {
        w.extend_from_slice(f.as_words());
    }
    BitVector::from_words(w)
}

fn overlap(a: &BitVector, b: &BitVector) -> u32 {
    a.as_words().iter().zip(b.as_words()).map(|(x, y)| (x & y).count_ones()).sum()
}

/// A fixed projection of cells into one frame (8 bits per cell): the recurrent frame and the pons.
fn project(cells: &[usize]) -> BitVector {
    let mut v = BitVector::new(BITS, Some(0));
    for &c in cells {
        for i in 0..8u64 {
            let mut h = (c as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            h ^= h >> 31;
            v.bit_set((h % BITS as u64) as usize);
        }
    }
    v
}

struct Net {
    l4: Layer4,
    l23: BitCells,
    l5: BitCells,
    cb: CerebellarCircuit,
    relay: ThalamicRelay,
    rec23: BitVector,
    pons: BitVector,
}

/// One step's activity: what each source proposed, and the plan.
struct Step {
    l23_row: BitVector,
    l5_row: BitVector,
    mossy: BitVector,
    /// (source, driver, pattern)
    props: Vec<(u8, Driver, BitVector)>,
    plan: BitVector,
}

impl Net {
    fn new(seed: u64) -> Self {
        let mk = |cells: usize| {
            let mut l = BitCells::new(BITS / 64, 16, cells);
            l.set_meta(true);
            l.set_tag_capture(true);
            l.set_interneurons(true);
            l
        };
        let mut cb = CerebellarCircuit::new(131_072, BITS, ONE / 256, seed.wrapping_add(77));
        cb.set_rates(2, 0);
        cb.set_cross_frames(true);
        Self {
            l4: Layer4::new(BITS, BITS, 16, ACTIVE, seed.wrapping_add(99)),
            l23: mk(16384),
            l5: mk(8192),
            cb,
            relay: ThalamicRelay::new(),
            rec23: BitVector::new(BITS, Some(0)),
            pons: BitVector::new(BITS, Some(0)),
        }
    }

    /// Hear `word` (its sensory code) and form the plan for the next word.
    fn step(&mut self, word: &BitVector, learn: bool, temp: u32, rng: &mut StdRng) -> Step {
        let l4 = if learn { self.l4.encode_learn(word, rng) } else { self.l4.encode(word) };
        let mossy = frames(&[word, &self.pons]);
        let cb_out = self.cb.predict(&mossy).clone();
        let mut props: Vec<(u8, Driver, BitVector)> = Vec::new();
        if cb_out.count_ones() > 0 {
            props.push((2, Driver::Tonic(0), cb_out.clone()));
        }
        // layer 2/3: [layer 4 | cerebellum (tuft) | its own previous activity]
        let l23_row = frames(&[&l4, &cb_out, &self.rec23]);
        let p23 = self.l23.predict(&l23_row, BITS);
        self.rec23 = project(&self.l23.last_fired());
        let o23 = p23.as_ref().map(|x| x.0.clone()).unwrap_or_else(|| BitVector::new(BITS, Some(0)));
        if let Some((o, _, burst)) = p23 {
            props.push((0, if burst { Driver::Burst } else { Driver::Tonic(1) }, o));
        }
        // layer 5: [layer 4 | cerebellum (tuft) | layer 2/3's prediction]
        let l5_row = frames(&[&l4, &cb_out, &o23]);
        let p5 = self.l5.predict(&l5_row, BITS);
        self.pons = project(&self.l5.last_fired());
        self.relay.step(matches!(p5, Some((_, _, false))));
        if let Some((o, _, burst)) = p5 {
            props.push((1, if burst { Driver::Burst } else { Driver::Spike }, o));
        }
        // the thalamic relay: weighted patterns summed per bit, the strongest bits kept
        let mut score = vec![0u32; BITS];
        for (src, drv, pat) in &props {
            let agrees = *src != 2 && overlap(pat, &cb_out) >= 16;
            let mut w = self.relay.weight(*src, *drv, word, agrees) as u64;
            if temp > 0 {
                w = w * (100 + rng.gen_range(0..=temp) as u64) / 100;
            }
            for (wi, &x) in pat.as_words().iter().enumerate() {
                let mut x = x;
                while x != 0 {
                    score[wi * 64 + x.trailing_zeros() as usize] += (w >> 4) as u32 + 1;
                    x &= x - 1;
                }
            }
        }
        let mut order: Vec<usize> = (0..BITS).filter(|&b| score[b] > 0).collect();
        order.sort_unstable_by(|a, b| score[*b].cmp(&score[*a]).then(a.cmp(b)));
        let mut plan = BitVector::new(BITS, Some(0));
        for &b in order.iter().take(ACTIVE) {
            plan.bit_set(b);
        }
        Step { l23_row, l5_row, mossy, props, plan }
    }

    fn learn(&mut self, st: &Step, word: &BitVector, target: &BitVector, rng: &mut StdRng) {
        self.l23.learn(&st.l23_row, target, rng);
        self.l5.learn(&st.l5_row, target, rng);
        self.cb.learn(&st.mossy, target, rng);
        for (src, drv, pat) in &st.props {
            self.relay.record(*src, *drv, word, overlap(pat, target) >= 16, rng);
        }
    }
}

fn main() {
    let seed: u64 = env("SEED", 0);
    let mut rng = StdRng::seed_from_u64(seed);
    let path = std::env::var("PROSE_FILE").unwrap_or_else(|_| "data/alice.txt".into());
    let text = std::fs::read_to_string(&path).expect("PROSE_FILE");
    let toks = tokens(&text);
    // vocabulary: the most frequent words, the rest "<unk>"
    let vmax: usize = env("PROSE_VOCAB", 3000);
    let mut freq: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for t in &toks {
        *freq.entry(t.as_str()).or_insert(0) += 1;
    }
    let mut by: Vec<(&str, usize)> = freq.into_iter().collect();
    by.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let mut vocab: Vec<String> = vec!["<unk>".into()];
    vocab.extend(by.iter().take(vmax).map(|x| x.0.to_string()));
    let index: std::collections::HashMap<&str, usize> = vocab.iter().enumerate().map(|(i, w)| (w.as_str(), i)).collect();
    let ids: Vec<usize> = toks.iter().map(|t| *index.get(t.as_str()).unwrap_or(&0)).collect();
    let n_words: usize = env("PROSE_WORDS", ids.len()).min(ids.len());
    let ids = &ids[..n_words];
    let split = n_words * 9 / 10;
    let enc = Encoder::new(vocab.len(), BITS, ACTIVE, &mut rng);
    println!("{path}: {} words, vocabulary {} (+<unk>), reading {} and testing on {}", toks.len(), vocab.len() - 1, split, n_words - split);

    let mut net = Net::new(seed);
    let t0 = std::time::Instant::now();
    // reading, with learning
    let (mut right, mut seen) = (0usize, 0usize);
    for t in 0..split.saturating_sub(1) {
        let word = &enc.codes[ids[t]];
        let target = &enc.codes[ids[t + 1]];
        let st = net.step(word, true, 0, &mut rng);
        right += (overlap(&st.plan, target) >= 24) as usize;
        seen += 1;
        net.learn(&st, word, target, &mut rng);
        if (t + 1) % 5000 == 0 {
            println!("  read {:>6} words: next word right {:.1}% over the last stretch ({:.0} s)", t + 1, 100.0 * right as f64 / seen as f64, t0.elapsed().as_secs_f64());
            right = 0;
            seen = 0;
        }
    }
    // held-out: learning off
    let mut by_src = [(0usize, 0usize); 3];
    let (mut right, mut seen) = (0usize, 0usize);
    for t in split..n_words - 1 {
        let word = &enc.codes[ids[t]];
        let target = &enc.codes[ids[t + 1]];
        let st = net.step(word, false, 0, &mut rng);
        right += (overlap(&st.plan, target) >= 24) as usize;
        seen += 1;
        for (src, _, pat) in &st.props {
            by_src[*src as usize].0 += 1;
            by_src[*src as usize].1 += (overlap(pat, target) >= 24) as usize;
        }
    }
    let pct = |a: usize, b: usize| 100.0 * a as f64 / b.max(1) as f64;
    println!(
        "held-out next word right {:.1}% ({} words); layer 2/3 {:.1}% of {} proposals, layer 5 {:.1}% of {}, cerebellum {:.1}% of {}",
        pct(right, seen),
        seen,
        pct(by_src[0].1, by_src[0].0),
        by_src[0].0,
        pct(by_src[1].1, by_src[1].0),
        by_src[1].0,
        pct(by_src[2].1, by_src[2].0),
        by_src[2].0
    );
    // a baseline: always the most frequent word
    let common_id = 1;
    let base = (split..n_words - 1).filter(|&t| ids[t + 1] == common_id).count();
    println!("(always saying {:?} would be right {:.1}%)", vocab[common_id], pct(base, n_words - 1 - split));

    // babble, then speak
    let tract = VocalTract::new(vocab.len(), BITS, ACTIVE, &mut rng);
    let mut motor = MotorArea::new(BITS);
    motor.babble(&tract, &enc.codes, 3, &mut rng);
    let temp: u32 = env("PROSE_TEMP", 20);
    let prompt = std::env::var("PROSE_PROMPT").unwrap_or_else(|_| "alice was".into());
    let mut said: Vec<String> = Vec::new();
    // what the network hears next: the word it said, or, when nothing came out, its own plan
    // (an efference copy, as in inner speech), so a silent step still moves it on
    let mut heard = enc.codes[0].clone();
    for w in tokens(&prompt) {
        heard = enc.codes[*index.get(w.as_str()).unwrap_or(&0)].clone();
        net.step(&heard, false, temp, &mut rng);
        said.push(w);
    }
    let mut silent = 0;
    for _ in 0..env("PROSE_GEN", 200usize) {
        let st = net.step(&heard, false, temp, &mut rng);
        let word = motor.plan(&st.plan).and_then(|m| tract.articulate(&m));
        match word {
            Some(w) => {
                said.push(vocab[w].clone());
                heard = enc.codes[w].clone();
            }
            None => {
                silent += 1;
                said.push("…".into());
                heard = st.plan.clone();
            }
        }
    }
    let mut out = String::new();
    for w in &said {
        if !".,;:!?".contains(w.as_str()) && !out.is_empty() {
            out.push(' ');
        }
        out.push_str(w);
    }
    println!("\nwritten ({} words, {} silent steps):\n{}", said.len(), silent, out);
    let _: Q16 = 0;
}
