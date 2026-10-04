//! Thalamus-like relay as attention: can the network bind facts it was never
//! trained on?
//!
//! Run:  cargo run --release --example thalamus
//!
//! Task (as in experiment 06B): stories such as
//!   "mary went to the kitchen . [john went to the garden .] where is mary ? kitchen"
//! with one place per name held out of training. Stories have a random number of
//! facts and ask about a random one, so the answer's position varies. Without attention the
//! predictor memorizes (name, place) pairs: 100% on seen pairs, 0% on held-out.
//!
//! Here a `Thalamus` watches the word stream. Each relay channel finds the most
//! recent earlier occurrence of the word `q` steps back and relays the word `v`
//! steps after it (an induction-head-style hard attention). The predictor reads
//! [current word | relay channel 0 | relay channel 1 | previous word].
//! Channels: none, oracle (q=1, v=4), fixed random, or learned by ablation
//! credit (re-predict without a channel's frame; every review, re-point the
//! channel with the least credit to a random (q, v)). "LearnedGuided" adds
//! credit-guided growth: a new kernel samples the current word plus ONE relay
//! channel (odds credit + 1), so it doesn't also require the name or other
//! channels' incidental content. "LearnedPatient": a newly re-pointed channel is
//! protected for 3 reviews, giving the predictor time to start using it.
//! "LearnedProposed": hindsight route proposals. Whenever the prediction is
//! wrong, the thalamus checks which of all 18 (q, v) routes would have relayed
//! the word that actually came (`Thalamus::routes_that_would_relay`) and gives
//! each a vote. At review the least credited channel is re-pointed to the most
//! voted route not already in use, instead of a random one.

mod common;

use std::collections::HashMap;

use common::Encoder;
use neurocomp::bitvec::BitVector;
use neurocomp::kernel::{GrowthConfig, KernelClass, SimpleKernel};
use neurocomp::program::{RelayChannel, Thalamus};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

const BITS: usize = 512;
const CHANNELS: usize = 4;
const NAMES: &[&str] = &["mary", "john", "sandra", "daniel", "anna", "peter"];
const PLACES: &[&str] = &["kitchen", "garden", "office", "hallway", "bathroom", "bedroom"];
const TRAIN: usize = 3000;
const TEST: usize = 1000;
const REVIEW_EVERY: usize = 50; // stories between channel re-pointing (learned policy)

#[derive(Clone, Copy, PartialEq, Debug)]
enum Policy {
    NoThalamus,
    Oracle,
    FixedRandom,
    Learned,
    LearnedGuided,
    LearnedPatient,
    LearnedProposed,
}

struct Story {
    words: Vec<&'static str>,
    answer_at: usize,
    held_out: bool,
}

/// One place per name is never paired with it in training (a diagonal).
fn allowed(n: usize, p: usize) -> bool {
    p != n % PLACES.len()
}

/// A story with 1..=`max_facts` facts about distinct people, then a question about
/// a random one of them. Varying length and question target rules out positional
/// shortcuts ("the answer is always k words after the last `?`"): only routing by
/// content (the asked name) can find the answer.
fn story(rng: &mut StdRng, max_facts: usize, want_held_out: bool) -> Story {
    loop {
        let facts = rng.gen_range(1..=max_facts);
        let mut names: Vec<usize> = (0..NAMES.len()).collect();
        names.shuffle(rng);
        let mut words = Vec::new();
        let mut loc = Vec::new();
        let mut ok = true;
        for &n in &names[..facts] {
            let p = rng.gen_range(0..PLACES.len());
            loc.push((n, p));
            words.extend([NAMES[n], "went", "to", "the", PLACES[p], "."]);
        }
        let (q, a) = loc[rng.gen_range(0..facts)];
        // training stories may not contain any held-out pair, even as a distractor
        ok &= want_held_out || loc.iter().all(|&(n, p)| allowed(n, p));
        if !ok || allowed(q, a) == want_held_out {
            continue;
        }
        words.extend(["where", "is", NAMES[q], "?"]);
        let answer_at = words.len();
        words.extend([PLACES[a], "."]);
        return Story { words, answer_at, held_out: want_held_out };
    }
}

fn random_channel(rng: &mut StdRng) -> RelayChannel {
    RelayChannel { query_lag: rng.gen_range(0..=2), value_offset: rng.gen_range(1..=6) }
}

struct Outcome {
    seen: f64,
    held_out: f64,
    channels: Vec<RelayChannel>,
}

fn run(policy: Policy, facts: usize, seed: u64) -> Outcome {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vocab: Vec<&str> = vec!["went", "to", "the", ".", "where", "is", "?"];
    vocab.extend(NAMES);
    vocab.extend(PLACES);
    let index: HashMap<&str, usize> = vocab.iter().enumerate().map(|(i, w)| (*w, i)).collect();
    let enc = Encoder::new(vocab.len(), BITS, 32, &mut rng);

    let channels = match policy {
        Policy::NoThalamus => vec![],
        Policy::Oracle => vec![RelayChannel { query_lag: 1, value_offset: 4 }, RelayChannel { query_lag: 0, value_offset: 1 }],
        Policy::FixedRandom | Policy::Learned | Policy::LearnedGuided | Policy::LearnedPatient | Policy::LearnedProposed => (0..CHANNELS).map(|_| random_channel(&mut rng)).collect(),
    };
    let n_ch = channels.len();
    let mut th = Thalamus::new(BITS, 40, channels);
    let mut credit = vec![0f64; n_ch];
    let mut age = vec![0usize; n_ch]; // reviews since a channel was re-pointed
    let mut votes: HashMap<(usize, usize), f64> = HashMap::new(); // hindsight route votes
    let all_routes: Vec<RelayChannel> = (0..=2)
        .flat_map(|q| (1..=6).map(move |v| RelayChannel { query_lag: q, value_offset: v }))
        .collect();
    let mut class: KernelClass<SimpleKernel> = KernelClass::predictive(GrowthConfig {
        max_kernels: 100_000,
        frame_words: BITS / 64,
        max_frames: n_ch + 2,
        sample_bits: 16,
        match_fraction: 0.8,
        surprise_fraction: 0.5,
        generalize: None,
    });

    let mut prev: Option<usize> = None;
    let (mut res_seen, mut res_held) = ((0usize, 0usize), (0usize, 0usize));
    for s_i in 0..TRAIN + TEST {
        let testing = s_i >= TRAIN;
        let s = story(&mut rng, facts, testing && s_i % 2 == 1);
        let ids: Vec<usize> = s.words.iter().map(|w| index[w]).collect();
        for t in 0..ids.len() - 1 {
            let code = &enc.codes[ids[t]];
            th.observe(code);
            // [current | relay 0 | relay 1 | previous]
            let mut words = code.as_words().to_vec();
            words.extend_from_slice(th.relay().as_words());
            words.extend_from_slice(prev.map_or(&[0u64; BITS / 64][..], |p| enc.codes[p].as_words()));
            let input = BitVector::from_words(words);

            let mut out = BitVector::new(BITS, Some(0));
            class.process_predictive(&input, &mut out);
            let next = ids[t + 1];
            let right = enc.decode(&out) == Some(next);
            if testing && t + 1 == s.answer_at && std::env::var("TRACE").is_ok() && s_i < TRAIN + 3 {
                let carriers: Vec<String> = (0..n_ch)
                    .filter(|&c| th.relay_channel(th.channels[c]).map_or(false, |v| v.as_words() == enc.codes[next].as_words()))
                    .map(|c| format!("{:?}", th.channels[c]))
                    .collect();
                eprintln!("story {:?} -> answer carried by {:?}", s.words, carriers);
            }
            if testing && t + 1 == s.answer_at {
                let r = if s.held_out { &mut res_held } else { &mut res_seen };
                r.0 += right as usize;
                r.1 += 1;
            }
            if !testing {
                let learned = matches!(policy, Policy::Learned | Policy::LearnedGuided | Policy::LearnedPatient | Policy::LearnedProposed);
                if policy == Policy::LearnedProposed && !right {
                    for r in th.routes_that_would_relay(&enc.codes[next], &all_routes) {
                        *votes.entry((r.query_lag, r.value_offset)).or_default() += 1.0;
                    }
                }
                if policy == Policy::LearnedGuided {
                    // frame 0 (current word) + one relaying channel picked by credit;
                    // the previous word only when no channel relays anything
                    let live: Vec<usize> =
                        (0..n_ch).filter(|&c| input.as_words()[(c + 1) * BITS / 64..(c + 2) * BITS / 64].iter().any(|&w| w != 0)).collect();
                    let mut mask = BitVector::new((n_ch + 2) * BITS, Some(0));
                    for b in 0..BITS {
                        mask.bit_set(b);
                    }
                    let extra = if live.is_empty() {
                        n_ch + 1
                    } else {
                        let w: Vec<f64> = live.iter().map(|&c| credit[c].max(0.0) + 1.0).collect();
                        let mut r = rng.gen_range(0.0..w.iter().sum::<f64>());
                        let mut pick = live[0];
                        for (&c, &wc) in live.iter().zip(&w) {
                            if r < wc {
                                pick = c;
                                break;
                            }
                            r -= wc;
                        }
                        pick + 1
                    };
                    for b in extra * BITS..(extra + 1) * BITS {
                        mask.bit_set(b);
                    }
                    class.set_growth_mask(Some(mask));
                }
                if learned {
                    // ablation credit: would the answer change without this channel?
                    for c in 0..n_ch {
                        let mut ablated = input.clone();
                        for b in (c + 1) * BITS..(c + 2) * BITS {
                            ablated.bit_clear(b);
                        }
                        let right_without = class.peek(&ablated).and_then(|o| enc.decode(o)) == Some(next);
                        credit[c] += (right && !right_without) as u8 as f64 - (!right && right_without) as u8 as f64;
                    }
                }
                class.feedback(&input, &enc.codes[next], &mut rng);
            }
            prev = Some(ids[t]);
        }
        let last = *ids.last().unwrap();
        th.observe(&enc.codes[last]);
        prev = Some(last);

        if matches!(policy, Policy::Learned | Policy::LearnedGuided | Policy::LearnedPatient | Policy::LearnedProposed)
            && !testing
            && s_i % REVIEW_EVERY == REVIEW_EVERY - 1
        {
            let grace = if policy == Policy::LearnedPatient { 3 } else { 0 };
            let worst = (0..n_ch).filter(|&c| age[c] >= grace).min_by(|&a, &b| credit[a].partial_cmp(&credit[b]).unwrap());
            if let Some(worst) = worst {
                let proposal = if policy == Policy::LearnedProposed {
                    votes
                        .iter()
                        .filter(|((q, v), _)| !th.channels.iter().any(|c| c.query_lag == *q && c.value_offset == *v))
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap().then(b.0.cmp(a.0)))
                        .map(|(&(q, v), _)| RelayChannel { query_lag: q, value_offset: v })
                } else {
                    None
                };
                th.channels[worst] = proposal.unwrap_or_else(|| random_channel(&mut rng));
                credit[worst] = 0.0;
                age[worst] = 0;
            }
            for a in age.iter_mut() {
                *a += 1;
            }
            for v in votes.values_mut() {
                *v *= 0.5;
            }
            for c in credit.iter_mut() {
                *c *= 0.5;
            }
        }
    }
    let pct = |r: (usize, usize)| 100.0 * r.0 as f64 / r.1.max(1) as f64;
    Outcome { seen: pct(res_seen), held_out: pct(res_held), channels: th.channels.clone() }
}

fn main() {
    println!("answer accuracy on {TEST} test stories after {TRAIN} training stories (learning off at test); chance 1/6");
    println!("channel (q, v): relay the word v steps after the last earlier occurrence of the word q steps back");
    println!();
    for facts in [2usize, 3] {
        println!("1-{facts} facts per story, question about a random one (others are distractors)");
        let policies: Vec<Policy> = match std::env::var("POLICIES").as_deref() {
            Ok("patient") => vec![Policy::Learned, Policy::LearnedPatient],
            Ok("proposed") => vec![Policy::Learned, Policy::LearnedProposed],
            Ok("proposed_only") => vec![Policy::LearnedProposed],
            Ok("main") => vec![Policy::NoThalamus, Policy::Oracle, Policy::FixedRandom, Policy::Learned, Policy::LearnedProposed],
            _ => vec![
                Policy::NoThalamus,
                Policy::Oracle,
                Policy::FixedRandom,
                Policy::Learned,
                Policy::LearnedGuided,
                Policy::LearnedPatient,
                Policy::LearnedProposed,
            ],
        };
        for policy in policies {
            let runs: Vec<Outcome> = (0..5).map(|seed| run(policy, facts, seed)).collect();
            let mean = |f: fn(&Outcome) -> f64| runs.iter().map(f).sum::<f64>() / runs.len() as f64;
            if std::env::var("ALL_CHANNELS").is_ok() {
                for (i, o) in runs.iter().enumerate() {
                    eprintln!(
                        "    {policy:?} run {i}: held-out {:.0}% seen {:.0}% channels {}",
                        o.held_out,
                        o.seen,
                        o.channels.iter().map(|c| format!("({},{})", c.query_lag, c.value_offset)).collect::<Vec<_>>().join(" ")
                    );
                }
            }
            println!(
                "  {:<12} seen pairs {:5.1}%   held-out pairs {:5.1}%   held-out runs [{}]   e.g. channels {}",
                format!("{policy:?}"),
                mean(|o| o.seen),
                mean(|o| o.held_out),
                runs.iter().map(|o| format!("{:.0}", o.held_out)).collect::<Vec<_>>().join(" "),
                runs[0].channels.iter().map(|c| format!("({},{})", c.query_lag, c.value_offset)).collect::<Vec<_>>().join(" ")
            );
        }
        println!();
    }
}
