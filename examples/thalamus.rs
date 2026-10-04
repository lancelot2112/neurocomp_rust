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
//! "LearnedOpen": open-ended route discovery (`Thalamus::discover_routes`).
//! Instead of checking a fixed menu, the thalamus searches its history for
//! earlier occurrences of the surprising word and proposes every (q, v) for which
//! a word in the current context (up to 8 back) matched a word shortly (up to 12)
//! before it. Candidates are ranked by consistency (`RouteScores`: of the
//! surprises where a route relayed something, how often was it the right word),
//! not by raw vote counts.
//!
//! TASK=long uses stories that need routes outside the old 18-route menu
//! (q <= 2, v <= 6): facts are "X went to the P ." (place 4 after the name) or
//! "X went all the way over to the P ." (8 after), and questions are
//! "where is X right now ?" (name 3 before "?"). Correct routes: (3,4) and (3,8).
//!
//! JITTER=1 puts 0-4 random filler words ("then", "later", "so", "next", "after")
//! before each story, so no fixed offset from the previous story's words lands on
//! the answer (open discovery otherwise finds such cross-story shortcuts, e.g.
//! "10 words after the previous where").

mod common;

use std::collections::HashMap;

use common::Encoder;
use neurocomp::bitvec::BitVector;
use neurocomp::kernel::{GrowthConfig, KernelClass, SimpleKernel};
use neurocomp::program::{RelayChannel, RouteScores, Thalamus};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

const BITS: usize = 512;
/// Relay channels for learned policies (CHANNELS env var overrides).
fn channels() -> usize {
    std::env::var("CHANNELS").ok().and_then(|v| v.parse().ok()).unwrap_or(4)
}
const NAMES: &[&str] = &["mary", "john", "sandra", "daniel", "anna", "peter"];
const PLACES: &[&str] = &["kitchen", "garden", "office", "hallway", "bathroom", "bedroom"];
const FILLERS: &[&str] = &["then", "later", "so", "next", "after"];
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
    LearnedOpen,
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
fn story(rng: &mut StdRng, max_facts: usize, long: bool, want_held_out: bool) -> Story {
    let jitter = std::env::var("JITTER").is_ok();
    loop {
        let facts = rng.gen_range(1..=max_facts);
        let mut names: Vec<usize> = (0..NAMES.len()).collect();
        names.shuffle(rng);
        let mut words = Vec::new();
        if jitter {
            for _ in 0..rng.gen_range(0..=4) {
                words.push(*FILLERS.choose(rng).unwrap());
            }
        }
        let mut loc = Vec::new();
        let mut ok = true;
        for &n in &names[..facts] {
            let p = rng.gen_range(0..PLACES.len());
            loc.push((n, p));
            if long && rng.gen_bool(0.5) {
                words.extend([NAMES[n], "went", "all", "the", "way", "over", "to", "the", PLACES[p], "."]);
            } else {
                words.extend([NAMES[n], "went", "to", "the", PLACES[p], "."]);
            }
        }
        let (q, a) = loc[rng.gen_range(0..facts)];
        // training stories may not contain any held-out pair, even as a distractor
        ok &= want_held_out || loc.iter().all(|&(n, p)| allowed(n, p));
        if !ok || allowed(q, a) == want_held_out {
            continue;
        }
        if long {
            words.extend(["where", "is", NAMES[q], "right", "now", "?"]);
        } else {
            words.extend(["where", "is", NAMES[q], "?"]);
        }
        let answer_at = words.len();
        words.extend([PLACES[a], "."]);
        return Story { words, answer_at, held_out: want_held_out };
    }
}

/// A random route from the old menu (q <= 2, v <= 6).
fn random_channel(rng: &mut StdRng) -> RelayChannel {
    RelayChannel { query_lag: rng.gen_range(0..=2), value_offset: rng.gen_range(1..=6) }
}

const OPEN_MAX_Q: usize = 8;
const OPEN_MAX_V: usize = 12;

/// A random route from the open-ended range used by discovery.
fn random_open_channel(rng: &mut StdRng) -> RelayChannel {
    RelayChannel { query_lag: rng.gen_range(0..=OPEN_MAX_Q), value_offset: rng.gen_range(1..=OPEN_MAX_V) }
}

struct Outcome {
    seen: f64,
    held_out: f64,
    channels: Vec<RelayChannel>,
}

fn run(policy: Policy, facts: usize, long: bool, seed: u64) -> Outcome {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vocab: Vec<&str> = vec!["went", "to", "the", ".", "where", "is", "?", "all", "way", "over", "right", "now"];
    vocab.extend(FILLERS);
    vocab.extend(NAMES);
    vocab.extend(PLACES);
    let index: HashMap<&str, usize> = vocab.iter().enumerate().map(|(i, w)| (*w, i)).collect();
    let enc = Encoder::new(vocab.len(), BITS, 32, &mut rng);

    let channels = match policy {
        Policy::NoThalamus => vec![],
        Policy::Oracle if long => vec![RelayChannel { query_lag: 3, value_offset: 4 }, RelayChannel { query_lag: 3, value_offset: 8 }],
        Policy::Oracle => vec![RelayChannel { query_lag: 1, value_offset: 4 }, RelayChannel { query_lag: 0, value_offset: 1 }],
        Policy::LearnedOpen => (0..channels()).map(|_| random_open_channel(&mut rng)).collect(),
        Policy::FixedRandom | Policy::Learned | Policy::LearnedGuided | Policy::LearnedPatient | Policy::LearnedProposed => (0..channels()).map(|_| random_channel(&mut rng)).collect(),
    };
    let n_ch = channels.len();
    let mut th = Thalamus::new(BITS, 40, channels);
    let mut credit = vec![0f64; n_ch];
    let mut age = vec![0usize; n_ch]; // reviews since a channel was re-pointed
    let mut votes: HashMap<(usize, usize), f64> = HashMap::new(); // hindsight route votes
    let mut route_scores = RouteScores::default(); // open discovery: consistency per route
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
        // GENERALIZE=f turns on synapse-level credit (drop silent inputs of near-matching
        // kernels that would have been right); off by default.
        generalize: std::env::var("GENERALIZE").ok().and_then(|v| v.parse().ok()),
        generalize_after: std::env::var("GENERALIZE_AFTER").ok().and_then(|v| v.parse().ok()).unwrap_or(1),
    });

    let mut prev: Option<usize> = None;
    let (mut res_seen, mut res_held) = ((0usize, 0usize), (0usize, 0usize));
    for s_i in 0..TRAIN + TEST {
        let testing = s_i >= TRAIN;
        let s = story(&mut rng, facts, long, testing && s_i % 2 == 1);
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
                let learned = matches!(
                    policy,
                    Policy::Learned | Policy::LearnedGuided | Policy::LearnedPatient | Policy::LearnedProposed | Policy::LearnedOpen
                );
                if !right && policy == Policy::LearnedProposed {
                    for r in th.routes_that_would_relay(&enc.codes[next], &all_routes) {
                        *votes.entry((r.query_lag, r.value_offset)).or_default() += 1.0;
                    }
                }
                if !right && policy == Policy::LearnedOpen {
                    route_scores.observe_surprise(&th, &enc.codes[next], OPEN_MAX_Q, OPEN_MAX_V);
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

        if matches!(
            policy,
            Policy::Learned | Policy::LearnedGuided | Policy::LearnedPatient | Policy::LearnedProposed | Policy::LearnedOpen
        ) && !testing
            && s_i % REVIEW_EVERY == REVIEW_EVERY - 1
        {
            let grace = if policy == Policy::LearnedPatient { 3 } else { 0 };
            let worst = (0..n_ch).filter(|&c| age[c] >= grace).min_by(|&a, &b| credit[a].partial_cmp(&credit[b]).unwrap());
            if let Some(worst) = worst {
                let proposal = if policy == Policy::LearnedOpen {
                    route_scores.best_unused(&th.channels, 2.0)
                } else if policy == Policy::LearnedProposed {
                    votes
                        .iter()
                        .filter(|((q, v), _)| !th.channels.iter().any(|c| c.query_lag == *q && c.value_offset == *v))
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap().then(b.0.cmp(a.0)))
                        .map(|(&(q, v), _)| RelayChannel { query_lag: q, value_offset: v })
                } else {
                    None
                };
                // Open discovery: only swap a route for one that is more consistent
                // (avoids churning out a good route the predictor hasn't used yet).
                let keep = policy == Policy::LearnedOpen
                    && proposal.map_or(true, |p| route_scores.precision(p) <= route_scores.precision(th.channels[worst]));
                if !keep {
                    th.channels[worst] = proposal.unwrap_or_else(|| {
                        if policy == Policy::LearnedOpen { random_open_channel(&mut rng) } else { random_channel(&mut rng) }
                    });
                    credit[worst] = 0.0;
                    age[worst] = 0;
                }
            }
            for a in age.iter_mut() {
                *a += 1;
            }
            for v in votes.values_mut() {
                *v *= 0.5;
            }
            route_scores.decay(0.9);
            for c in credit.iter_mut() {
                *c *= 0.5;
            }
        }
    }
    let pct = |r: (usize, usize)| 100.0 * r.0 as f64 / r.1.max(1) as f64;
    Outcome { seen: pct(res_seen), held_out: pct(res_held), channels: th.channels.clone() }
}

fn main() {
    if let Ok(g) = std::env::var("GENERALIZE") {
        let after = std::env::var("GENERALIZE_AFTER").unwrap_or_else(|_| "1".into());
        println!("synapse-level credit on: generalize near misses matching >= {g} of their connections, after {after} confirmation(s)");
    }
    let long = std::env::var("TASK").as_deref() == Ok("long");
    if std::env::var("JITTER").is_ok() {
        println!("JITTER: 0-4 random filler words before each story");
    }
    if long {
        println!("TASK=long: facts \"X went [all the way over] to the P .\", questions \"where is X right now ?\" (routes (3,4) and (3,8))");
    }
    println!("answer accuracy on {TEST} test stories after {TRAIN} training stories (learning off at test); chance 1/6");
    println!("channel (q, v): relay the word v steps after the last earlier occurrence of the word q steps back");
    println!();
    for facts in [2usize, 3] {
        println!("1-{facts} facts per story, question about a random one (others are distractors)");
        let policies: Vec<Policy> = match std::env::var("POLICIES").as_deref() {
            Ok("patient") => vec![Policy::Learned, Policy::LearnedPatient],
            Ok("proposed") => vec![Policy::Learned, Policy::LearnedProposed],
            Ok("proposed_only") => vec![Policy::LearnedProposed],
            Ok("oracle") => vec![Policy::Oracle],
            Ok("open_only") => vec![Policy::LearnedOpen],
            Ok("open") => vec![Policy::NoThalamus, Policy::Oracle, Policy::LearnedProposed, Policy::LearnedOpen],
            Ok("open_cmp") => vec![Policy::Oracle, Policy::LearnedProposed, Policy::LearnedOpen],
            Ok("main") => vec![Policy::NoThalamus, Policy::Oracle, Policy::FixedRandom, Policy::Learned, Policy::LearnedProposed],
            _ => vec![
                Policy::NoThalamus,
                Policy::Oracle,
                Policy::FixedRandom,
                Policy::Learned,
                Policy::LearnedGuided,
                Policy::LearnedPatient,
                Policy::LearnedProposed,
                Policy::LearnedOpen,
            ],
        };
        for policy in policies {
            let runs: Vec<Outcome> = (0..5).map(|seed| run(policy, facts, long, seed)).collect();
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
