//! Reading aloud and retelling, learned from a teacher's examples, chosen by a spoken
//! instruction.
//!
//! Each trial: the teacher tells a short story (S) aloud. Then a book is opened, at the same
//! story or at another one (P), or stays closed, and the teacher says "now read it" or "now
//! tell it". The teacher then does it, word by word: reading says the book's words (P),
//! telling says the heard story (S) whatever the book shows. The network hears and sees
//! everything the child would and learns, at each word, to plan the word the teacher says.
//! Nothing says which source to use: when the book shows another story, the eye and the
//! memory propose different words, and only the instruction tells which one is wanted.
//!
//! At test (new stories, cortex learning off), the network does it itself: it plans, says the
//! word through the learned motor path, and hears itself.
//!
//! Everything goes through the network; no word reaches speech except through the cortex:
//! - **Eye and ear** are separate senses with their own codes: the visual word at the fixation
//!   (the book's word t, or nothing when it is closed) and the heard word (the last word said).
//!   Each has its own layer 4 (learned k-winners-take-all).
//! - **The hippocampus** stores the story as it is heard: each word (content) with the word
//!   before it and the story's context (context). Cued by the last word said and the context,
//!   it recalls the next word, and the recall returns to the cortex as an input (EC → cortex).
//! - **The prefrontal cortex** holds what a basal-ganglia gate loads from what is heard
//!   (`PfcGate`, after PBWM). Its credit is whether the words spoken while it held that item
//!   were right, so it learns to hold the instruction word. What it holds reaches the cells'
//!   tuft (layer 1) as context and the thalamic relay's context.
//! - **Recurrent layer 2/3** (bitwise primed cells): basal input is the eye's layer 4, the
//!   ear's layer 4, the hippocampal recall and its own previous activity; its tuft gets the
//!   prefrontal content and the cerebellum's output. A cell whose tuft matches bursts and
//!   wins over one that only spikes.
//! - **Layer 5**, **the cerebellar circuit** (mossy fibres: the ear and the pons) and **the
//!   thalamic relay** as in the prose driver; the plan is the relay's strongest 32 bits.
//! - **Speaking:** the motor area's inverse model (learned by babbling) turns the plan into a
//!   command; the vocal tract says a word; the network hears it.
//!
//! Options: RT_TRAIN (trials, default 3000), RT_TEST (default 300), RT_GATE=fixed (diagnosis:
//! the instruction word is always loaded, no gate), RT_TELL=apart (the teacher never retells
//! with the book open at the same story), RT_TRACE, SEED.

mod common;

use common::Encoder;
use neurocomp::bitvec::BitVector;
use neurocomp::fixed::ONE;
use neurocomp::program::{
    BitCells, CerebellarCircuit, Driver, Gate, Hippocampus, HippocampusConfig, Layer4, MotorArea, PfcGate, ThalamicRelay, VocalTract,
    WorkingMemory,
};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng};

const BITS: usize = 2048;
const ACTIVE: usize = 32;
/// The hippocampus's context space (after the BITS content bits).
const CTX_SPACE: usize = 65_536;

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
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

fn ones(v: &BitVector) -> Vec<usize> {
    let mut out = Vec::new();
    for (wi, &x) in v.as_words().iter().enumerate() {
        let mut x = x;
        while x != 0 {
            out.push(wi * 64 + x.trailing_zeros() as usize);
            x &= x - 1;
        }
    }
    out
}

fn zero() -> BitVector {
    BitVector::new(BITS, Some(0))
}

/// A fixed projection of cells into one frame (8 bits per cell): the recurrent frame and the pons.
fn project(cells: &[usize]) -> BitVector {
    let mut v = zero();
    for &c in cells {
        for i in 0..8u64 {
            let mut h = (c as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
            h ^= h >> 31;
            v.bit_set((h % BITS as u64) as usize);
        }
    }
    v
}

// ---- the stories ----

const NAMES: [(&str, &str); 6] = [("mary", "she"), ("sue", "she"), ("anna", "she"), ("john", "he"), ("tom", "he"), ("bob", "he")];
const PLACES: [&str; 8] = ["park", "farm", "river", "shop", "school", "garden", "forest", "beach"];
const ANIMALS: [&str; 8] = ["cat", "dog", "bird", "fox", "horse", "duck", "frog", "owl"];
const ADJS: [&str; 6] = ["happy", "hungry", "small", "big", "sleepy", "wet"];
const FOODS: [&str; 6] = ["apple", "bread", "cake", "fish", "seed", "carrot"];
const THINGS: [&str; 6] = ["ball", "hat", "box", "stick", "key", "shell"];
const OTHER: [&str; 16] =
    ["went", "to", "the", "a", ".", "saw", "was", "gave", "found", "played", "with", "then", "home", "now", "read", "tell"];
const IT: &str = "it";

fn vocabulary() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = Vec::new();
    for (n, p) in NAMES {
        v.push(n);
        if !v.contains(&p) {
            v.push(p);
        }
    }
    for list in [&PLACES[..], &ANIMALS, &ADJS, &FOODS, &THINGS, &OTHER, &[IT]] {
        v.extend_from_slice(list);
    }
    v
}

fn story(rng: &mut StdRng) -> Vec<&'static str> {
    let (n, p) = NAMES[rng.gen_range(0..NAMES.len())];
    let place = PLACES[rng.gen_range(0..PLACES.len())];
    let animal = ANIMALS[rng.gen_range(0..ANIMALS.len())];
    let adj = ADJS[rng.gen_range(0..ADJS.len())];
    let food = FOODS[rng.gen_range(0..FOODS.len())];
    let thing = THINGS[rng.gen_range(0..THINGS.len())];
    let mut s = vec![n, "went", "to", "the", place, ".", p, "saw", "a", animal, "."];
    let mut middle: Vec<Vec<&str>> = vec![
        vec!["the", animal, "was", adj, "."],
        vec![n, "gave", "the", animal, "a", food, "."],
        vec![p, "found", "a", thing, "."],
        vec![p, "played", "with", "the", animal, "."],
    ];
    middle.shuffle(rng);
    for m in middle.into_iter().take(rng.gen_range(1..=3)) {
        s.extend(m);
    }
    s.extend(["then", n, "went", "home", "."]);
    s
}

// ---- the network ----

struct Net {
    l4e: Layer4,
    l4a: Layer4,
    l23: BitCells,
    l5: BitCells,
    cb: CerebellarCircuit,
    relay: ThalamicRelay,
    rec23: BitVector,
    pons: BitVector,
    wm: WorkingMemory,
    gate: PfcGate,
    hc: Hippocampus,
}

struct Step {
    l23_row: BitVector,
    l5_row: BitVector,
    mossy: BitVector,
    ctx: BitVector,
    props: Vec<(u8, Driver, BitVector)>,
    plan: BitVector,
}

impl Net {
    fn new(seed: u64) -> Self {
        let mk = |cells: usize, tail: usize| {
            let mut l = BitCells::new(BITS / 64, 16, cells);
            l.set_meta(true);
            l.set_tag_capture(true);
            l.set_interneurons(true);
            l.set_basal_tail(tail);
            l.set_clustered(std::env::var("RT_CLUSTER").map_or(true, |v| v != "0"));
            l
        };
        let mut cb = CerebellarCircuit::new(131_072, BITS, ONE / 256, seed.wrapping_add(77));
        cb.set_rates(2, 0);
        cb.set_cross_frames(true);
        Self {
            l4e: Layer4::new(BITS, BITS, 16, ACTIVE, seed.wrapping_add(98)),
            l4a: Layer4::new(BITS, BITS, 16, ACTIVE, seed.wrapping_add(99)),
            // [eye L4 | PFC (tuft) | cerebellum (tuft) | ear L4 | recall | own previous activity]
            l23: mk(16384, 3),
            // [eye L4 | PFC (tuft) | cerebellum (tuft) | ear L4 | layer 2/3's prediction]
            l5: mk(8192, 2),
            cb,
            relay: ThalamicRelay::new(),
            rec23: zero(),
            pons: zero(),
            wm: WorkingMemory::new(BITS, 1),
            gate: PfcGate::new(BITS, 1, ONE * 9 / 10, seed.wrapping_add(11)),
            hc: {
                // RT_HC_DECAY (per-store decay of every weight, default 0.999), RT_HC_CENTER=1
                // (homeostatic centering of CA3's drive), RT_HC_SCALE=1 (presynaptic scaling
                // on every pathway)
                // input: the content (BITS) and the sparse context space; output: the content
                let mut cfg = HippocampusConfig::new(BITS + CTX_SPACE, seed.wrapping_add(300));
                cfg.out_bits = BITS;
                cfg.hashed_fan_out = Some(env("RT_HC_FANOUT", 300usize));
                cfg.decay = env("RT_HC_DECAY", 0.999f32);
                cfg.center = std::env::var("RT_HC_CENTER").is_ok();
                cfg.scale_all = std::env::var("RT_HC_SCALE").is_ok();
                Hippocampus::new(cfg)
            },
        }
    }

    /// The hippocampus's cue and stored context: a sparse conjunctive code (64 of CTX_SPACE
    /// bits, after the content half) set by the story's context and the last two sounds heard
    /// (silence before anyone has spoken), through a fixed random projection, as entorhinal
    /// conjunctive cells would give. A conjunction is rarely repeated, so its bits keep their
    /// drive under the hippocampus's presynaptic scaling (single words and a reused context,
    /// written thousands of times, are scaled to nothing).
    fn hc_context(prev: Option<&BitVector>, prev2: Option<&BitVector>, silence: &BitVector, story: u64) -> Vec<usize> {
        let mut h: u64 = story.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for (k, v) in [prev.unwrap_or(silence), prev2.unwrap_or(silence)].into_iter().enumerate() {
            for w in v.as_words() {
                h = (h ^ w.rotate_left(17 * k as u32 + 5)).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
                h ^= h >> 29;
            }
        }
        let mut c: Vec<usize> = (0..2 * ACTIVE as u64)
            .map(|i| {
                let mut x = h ^ i.wrapping_mul(0xD6E8_FEB8_6659_FD93);
                x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                x ^= x >> 31;
                BITS + (x % CTX_SPACE as u64) as usize
            })
            .collect();
        c.sort_unstable();
        c.dedup();
        c
    }

    /// What the hippocampus recalls for this cue: the content half, as a frame.
    fn recall(&self, cue: &[usize]) -> BitVector {
        let r = self.hc.recall(cue);
        let mut v = zero();
        if r.strength > 0 {
            for &b in r.ec.iter().filter(|&&b| b < BITS) {
                v.bit_set(b);
            }
        }
        v
    }

    /// See `eye` (or nothing), hear `ear` (or nothing), with `recall` from the hippocampus;
    /// form the plan for the word to say.
    fn step(&mut self, eye: &BitVector, ear: &BitVector, recall: &BitVector, learn: bool, rng: &mut StdRng) -> Step {
        let e4 = if eye.count_ones() == 0 { zero() } else if learn { self.l4e.encode_learn(eye, rng) } else { self.l4e.encode(eye) };
        let a4 = if ear.count_ones() == 0 { zero() } else if learn { self.l4a.encode_learn(ear, rng) } else { self.l4a.encode(ear) };
        let pfc = self.wm.content();
        let mossy = frames(&[ear, &self.pons]);
        let cb_out = self.cb.predict(&mossy).clone();
        let mut props: Vec<(u8, Driver, BitVector)> = Vec::new();
        if cb_out.count_ones() > 0 {
            props.push((2, Driver::Tonic(0), cb_out.clone()));
        }
        let l23_row = frames(&[&e4, &pfc, &cb_out, &a4, recall, &self.rec23]);
        let p23 = self.l23.predict(&l23_row, BITS);
        self.rec23 = project(&self.l23.last_fired());
        let o23 = p23.as_ref().map(|x| x.0.clone()).unwrap_or_else(zero);
        if let Some((o, _, burst)) = p23 {
            props.push((0, if burst { Driver::Burst } else { Driver::Tonic(1) }, o));
        }
        let l5_row = frames(&[&e4, &pfc, &cb_out, &a4, &o23]);
        let p5 = self.l5.predict(&l5_row, BITS);
        self.pons = project(&self.l5.last_fired());
        self.relay.step(matches!(p5, Some((_, _, false))));
        if let Some((o, _, burst)) = p5 {
            props.push((1, if burst { Driver::Burst } else { Driver::Spike }, o));
        }
        // the relay's context: what the prefrontal cortex holds and what was heard
        let ctx = frames(&[&pfc, ear]);
        let mut score = vec![0u32; BITS];
        for (src, drv, pat) in &props {
            let agrees = *src != 2 && overlap(pat, &cb_out) >= 16;
            let w = self.relay.weight(*src, *drv, &ctx, agrees) as u64;
            for b in ones(pat) {
                score[b] += (w >> 4) as u32 + 1;
            }
        }
        let mut order: Vec<usize> = (0..BITS).filter(|&b| score[b] > 0).collect();
        order.sort_unstable_by(|a, b| score[*b].cmp(&score[*a]).then(a.cmp(b)));
        let mut plan = zero();
        for &b in order.iter().take(ACTIVE) {
            plan.bit_set(b);
        }
        Step { l23_row, l5_row, mossy, ctx, props, plan }
    }

    fn learn(&mut self, st: &Step, target: &BitVector, rng: &mut StdRng) {
        self.l23.learn(&st.l23_row, target, rng);
        self.l5.learn(&st.l5_row, target, rng);
        self.cb.learn(&st.mossy, target, rng);
        for (src, drv, pat) in &st.props {
            self.relay.record(*src, *drv, &st.ctx, overlap(pat, target) >= 16, rng);
        }
    }

    /// The prefrontal gate sees a heard word (`key`, its code `ear`): load it or keep.
    fn hear_gate(&mut self, key: usize, ear: &BitVector, explore: bool, rng: &mut StdRng) {
        let g = if explore { self.gate.decide(key, Some(rng)) } else { self.gate.decide::<StdRng>(key, None) };
        if g == Gate::Load {
            self.wm.load(0, ear);
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Book {
    Same,
    Other,
    Closed,
}

#[derive(Default, Clone, Copy)]
struct Tally {
    words: usize,
    right: usize,
    said_story: usize,
    said_book: usize,
    silent: usize,
}

fn main() {
    let seed: u64 = env("SEED", 0);
    let mut rng = StdRng::seed_from_u64(seed);
    let vocab = vocabulary();
    let id = |w: &str| vocab.iter().position(|v| *v == w).unwrap();
    let ear = Encoder::new(vocab.len(), BITS, ACTIVE, &mut rng);
    let eye = Encoder::new(vocab.len(), BITS, ACTIVE, &mut rng);
    // the sound of silence: what the ear hears before anyone speaks
    let silence = Encoder::new(1, BITS, ACTIVE, &mut rng).codes[0].clone();
    let tract = VocalTract::new(vocab.len(), BITS, ACTIVE, &mut rng);
    let mut motor = MotorArea::new(BITS);
    motor.babble(&tract, &ear.codes, 3, &mut rng);
    let mut net = Net::new(seed);
    let fixed_gate = std::env::var("RT_GATE").map_or(false, |v| v == "fixed");
    let tell_apart = std::env::var("RT_TELL").map_or(false, |v| v == "apart");
    let (n_train, n_test): (usize, usize) = (env("RT_TRAIN", 3000), env("RT_TEST", 300));
    println!("vocabulary {}; {} training trials, {} test trials; gate {}", vocab.len(), n_train, n_test, if fixed_gate { "fixed (diagnosis)" } else { "learned" });
    let t0 = std::time::Instant::now();
    // results per (instruction read/tell, book), at test; and with the prefrontal content removed
    let mut res = [[Tally::default(); 3]; 2];
    let mut res_noinstr = [[Tally::default(); 3]; 2];
    let mut train_right = (0usize, 0usize);
    // the hippocampal recall's own accuracy while telling: (recalled the wanted word, steps)
    let mut hc_right = (0usize, 0usize);
    let mut hc_bits = 0usize;
    // the last 500 training trials (the teacher's words heard): recall right, and the plan
    // right per (instruction, book)
    let mut hc_train = (0usize, 0usize);
    let mut trace: usize = env("RT_TRACE", 0);
    let mut by_cond = [[(0usize, 0usize); 3]; 2];
    let books = [Book::Same, Book::Other, Book::Closed];
    for trial in 0..n_train + 2 * n_test {
        let testing = trial >= n_train;
        let no_instr = trial >= n_train + n_test;
        let learn = !testing;
        let s = story(&mut rng);
        // the story's context (a lateral-EC-like pattern as large as the two-word cue), new for
        // each story told
        let story_ctx: u64 = rng.next_u64();
        // 1. the teacher tells the story; the network predicts each next word and stores it
        let mut prev: Option<BitVector> = None;
        let mut prev2: Option<BitVector> = None;
        for t in 0..s.len() {
            let w = id(s[t]);
            let heard = &ear.codes[w];
            net.hear_gate(w, heard, learn && !fixed_gate, &mut rng);
            let cue = Net::hc_context(prev.as_ref(), prev2.as_ref(), &silence, story_ctx);
            let rec = net.recall(&cue);
            let st = net.step(&zero(), heard, &rec, learn, &mut rng);
            if learn && t + 1 < s.len() {
                net.learn(&st, &ear.codes[id(s[t + 1])], &mut rng);
            }
            net.hc.store_split(&ones(heard), &cue, &ones(heard));
            prev2 = prev.replace(heard.clone());
        }
        if trace > 0 && testing {
            // probe: cue the hippocampus with the true preceding words, right after hearing
            let (mut p1, mut p2): (Option<BitVector>, Option<BitVector>) = (None, None);
            let mut line = String::new();
            for t in 0..s.len().min(8) {
                let r = net.hc.recall(&Net::hc_context(p1.as_ref(), p2.as_ref(), &silence, story_ctx));
                let mut v = zero();
                for &b in r.ec.iter().filter(|&&b| b < BITS) {
                    v.bit_set(b);
                }
                line += &format!(" {}:{}/{}(s{})", s[t], overlap(&v, &ear.codes[id(s[t])]), v.count_ones(), r.strength);
                p2 = p1.replace(ear.codes[id(s[t])].clone());
            }
            println!("  probe right after hearing:{line}");
        }
        // 2. the book and the instruction
        let read = rng.gen_bool(0.5);
        // RT_TELL=apart: in training the teacher retells with the book closed or open at
        // another story, never at the same one (the test keeps all three)
        let book = if read {
            [Book::Same, Book::Other][rng.gen_range(0..2)]
        } else if tell_apart && !testing {
            [Book::Other, Book::Closed][rng.gen_range(0..2)]
        } else {
            books[rng.gen_range(0..3)]
        };
        let p = match book {
            Book::Same => s.clone(),
            Book::Other => story(&mut rng),
            Book::Closed => Vec::new(),
        };
        for w in ["now", if read { "read" } else { "tell" }, IT] {
            let wi = id(w);
            if fixed_gate {
                if w == "read" || w == "tell" {
                    net.wm.load(0, &ear.codes[wi]);
                }
            } else {
                net.hear_gate(wi, &ear.codes[wi], learn, &mut rng);
            }
            net.step(&zero(), &ear.codes[wi], &zero(), false, &mut rng);
        }
        if no_instr {
            net.wm.clear();
        }
        // 3. the task: the teacher does it (training), or the network does (test)
        let target: &Vec<&str> = if read { &p } else { &s };
        let mut prev: Option<BitVector> = None;
        let mut prev2: Option<BitVector> = None;
        let mut tally = Tally::default();
        for t in 0..target.len() {
            let seen = p.get(t).map(|w| eye.codes[id(w)].clone()).unwrap_or_else(zero);
            let cue = Net::hc_context(prev.as_ref(), prev2.as_ref(), &silence, story_ctx);
            let rec = net.recall(&cue);
            let heard = prev.clone().unwrap_or_else(zero);
            let st = net.step(&seen, &heard, &rec, learn, &mut rng);
            let want = id(target[t]);
            if !read {
                hc_right.0 += (overlap(&rec, &ear.codes[want]) >= 16) as usize;
                hc_right.1 += 1;
                hc_bits += rec.count_ones() as usize;
                if learn && trial + 500 >= n_train {
                    hc_train.0 += (overlap(&rec, &ear.codes[want]) >= 16) as usize;
                    hc_train.1 += 1;
                }
            }
            if learn && trial + 500 >= n_train {
                let bi = books.iter().position(|b| *b == book).unwrap();
                let c = &mut by_cond[!read as usize][bi];
                c.0 += (overlap(&st.plan, &ear.codes[want]) >= 24) as usize;
                c.1 += 1;
            }
            if learn {
                let right = overlap(&st.plan, &ear.codes[want]) >= 24;
                train_right.0 += right as usize;
                train_right.1 += 1;
                net.learn(&st, &ear.codes[want], &mut rng);
                if !fixed_gate {
                    net.gate.reward(if right { ONE as i32 } else { 0 }, &mut rng);
                }
                // the teacher's word is heard
                prev2 = prev.replace(ear.codes[want].clone());
                net.hear_gate(want, &ear.codes[want], !fixed_gate, &mut rng);
            } else {
                let said = motor.plan(&st.plan).and_then(|m| tract.articulate(&m));
                if trace > 0 && book == Book::Closed && !read && t < 6 {
                    trace -= (t == 5) as usize;
                    let near = ear.decode(&st.plan).map_or("-", |w| vocab[w]);
                    println!(
                        "  trace t={t} want {:<7} plan bits {:>2} (nearest {:<7} overlap {:>2}) sources {:?} recall overlap {:>2} said {:?}",
                        target[t],
                        st.plan.count_ones(),
                        near,
                        overlap(&st.plan, &ear.codes[want]),
                        st.props.iter().map(|x| x.0).collect::<Vec<_>>(),
                        overlap(&rec, &ear.codes[want]),
                        said.map(|w| vocab[w])
                    );
                }
                tally.words += 1;
                match said {
                    Some(w) => {
                        tally.right += (w == want) as usize;
                        tally.said_story += (w == id(s[t.min(s.len() - 1)]) && t < s.len()) as usize;
                        tally.said_book += p.get(t).map_or(false, |b| w == id(b)) as usize;
                        prev2 = prev.replace(ear.codes[w].clone());
                        if !fixed_gate {
                            net.hear_gate(w, &ear.codes[w], false, &mut rng);
                        }
                    }
                    None => {
                        tally.silent += 1;
                        prev2 = prev.replace(st.plan.clone());
                    }
                }
            }
        }
        if testing {
            let bi = books.iter().position(|b| *b == book).unwrap();
            let slot = if no_instr { &mut res_noinstr[!read as usize][bi] } else { &mut res[!read as usize][bi] };
            slot.words += tally.words;
            slot.right += tally.right;
            slot.said_story += tally.said_story;
            slot.said_book += tally.said_book;
            slot.silent += tally.silent;
        }
        if learn && (trial + 1) % 500 == 0 {
            println!(
                "  trial {:>5}: words planned right {:.1}% over the last 500 trials ({:.0} s)",
                trial + 1,
                100.0 * train_right.0 as f64 / train_right.1.max(1) as f64,
                t0.elapsed().as_secs_f64()
            );
            train_right = (0, 0);
        }
    }
    let pct = |a: usize, b: usize| 100.0 * a as f64 / b.max(1) as f64;
    for (title, r) in [("with the instruction held", &res), ("prefrontal content removed before the task", &res_noinstr)] {
        println!("\ntest, {title} (new stories, words said aloud):");
        println!("  {:<6} {:<12} {:>6} {:>8} {:>13} {:>12} {:>7}", "asked", "book", "words", "right", "said story's", "said book's", "silent");
        for (ii, instr) in ["read", "tell"].iter().enumerate() {
            for (bi, b) in ["same story", "other story", "closed"].iter().enumerate() {
                let x = r[ii][bi];
                if x.words == 0 {
                    continue;
                }
                println!(
                    "  {:<6} {:<12} {:>6} {:>7.1}% {:>12.1}% {:>11.1}% {:>6.1}%",
                    instr,
                    b,
                    x.words,
                    pct(x.right, x.words),
                    pct(x.said_story, x.words),
                    pct(x.said_book, x.words),
                    pct(x.silent, x.words)
                );
            }
        }
    }
    println!("\nhippocampal recall held the wanted word on {:.1}% of telling steps (overlap at least 16 of 32 bits); {:.0} bits recalled on average", pct(hc_right.0, hc_right.1), hc_bits as f64 / hc_right.1.max(1) as f64);
    println!("last 500 training trials, the teacher's words heard: recall right {:.1}% of telling steps; plan right: read same {:.1}%, read other {:.1}%, tell same {:.1}%, tell other {:.1}%, tell closed {:.1}%",
        pct(hc_train.0, hc_train.1),
        pct(by_cond[0][0].0, by_cond[0][0].1), pct(by_cond[0][1].0, by_cond[0][1].1),
        pct(by_cond[1][0].0, by_cond[1][0].1), pct(by_cond[1][1].0, by_cond[1][1].1), pct(by_cond[1][2].0, by_cond[1][2].1));
    let (rd, tl) = (net.gate.value(Gate::Load, id("read")), net.gate.value(Gate::Load, id("tell")));
    let (nw, it, the) = (net.gate.value(Gate::Load, id("now")), net.gate.value(Gate::Load, id(IT)), net.gate.value(Gate::Load, id("the")));
    let q = |v: u32| v as f64 / ONE as f64;
    println!(
        "\nprefrontal gate, value of loading: read {:.2}, tell {:.2}, now {:.2}, it {:.2}, the {:.2}",
        q(rd as u32),
        q(tl as u32),
        q(nw as u32),
        q(it as u32),
        q(the as u32)
    );
}
