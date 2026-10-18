//! Phase codes against frame codes on the prediction rungs: can a fixed circuit of word operations
//! tell a break in the token stream, and does it need phase? Each lifetime's stream comes from the
//! real generator (`live_lesson` on a prediction rung's lesson, a network that does nothing; fair
//! streams with `FAIR=1`, the default), and each policy's clicks are replayed into its events and
//! scored by the surprise probe (`observe::surprise`, and the graded one), on `WORLDS` worlds (48).
//!
//! A policy sees what the eye saw a step ago and earlier (the history ring), as a network does. Two
//! glyphs are the same token when their words XOR to nothing; a token's phase is its glyph folded
//! to one byte (XOR of its eight bytes: four shifts and XORs), modelled here as a byte drawn per
//! token per lifetime. The policies:
//! - **blind**: clicks every third step (the floor);
//! - **rare**: clicks after a token seen less than one step in seven so far (the old shortcut);
//! - **frames**: the order-2 rule from the eye's last three frames, by XOR: a break when "this token
//!   is the one before" has the same answer as a step earlier (cycles that say each token twice in
//!   a row, A A B B C C, alternate the answer). A fixed circuit, nothing learned;
//! - **window**: the same rule read from one word holding the last eight phases as byte lanes (each
//!   step shifted left 8 and the new phase ORed into lane 0): phase as position in a word;
//! - **comb**: a phase accumulator (`ADD` in a byte lane, wrapping), clicking when the sum of the
//!   last `period` phases changes from one step to the next: interference at the cycle's period,
//!   true of any cycle, but it also fires when a break leaves the window;
//! - **oracle**: clicks after every break (the ceiling).
//!
//! `cargo run --release --example phasecheck`
use neurocomp::evolve::tags::TagGenome;
use neurocomp::evolve::{live_lesson, Lesson, Physics, Task};
use neurocomp::world::{observe, Event, CLICK};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn var(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|x| x.parse().ok()).unwrap_or(d)
}

/// The events with clicks where `click(t)` says (a click at step t answers what was shown before t).
fn replay(events: &[Event], click: &[bool]) -> Vec<Event> {
    let mut out = events.to_vec();
    for (e, &c) in out.iter_mut().zip(click) {
        e.moved = if c { Some(CLICK) } else { None };
    }
    out
}

fn main() {
    neurocomp::evolve::set_body(var("BODY", 1) as u64);
    neurocomp::evolve::set_one_word(var("ONE_WORD", 8));
    neurocomp::evolve::set_fair_sequences(var("FAIR", 1) == 1);
    neurocomp::evolve::set_varied_orders(var("VARIED", 0) == 1);
    let worlds = var("WORLDS", 48) as u64;
    let empty = TagGenome { tokens: vec![], next_id: 1 };
    println!("surprise (graded) on {worlds} worlds, mean; fair streams {}, varied orders {}", neurocomp::evolve::fair_sequences(), neurocomp::evolve::varied_orders());
    println!("{:<8} {:>13} {:>13} {:>13} {:>13} {:>13} {:>13}", "rung", "blind", "rare", "frames", "window", "comb", "oracle");
    for (name, period, tokens) in [("cycle2", 2usize, 8usize), ("cycle4", 4, 8), ("order2", 6, 3), ("cycle8", 8, 8)] {
        let lesson = Lesson { period, tokens, noise: 1, ..Lesson::default() };
        let mut sums = vec![(0i64, 0i64); 6];
        for wseed in 0..worlds {
            let seed = 70_000 + wseed;
            let (ev, _) = live_lesson(&empty, Task::Sequence(200), Physics::Varied, seed ^ 0x9ED, 0, lesson.clone());

            let n = ev.len();
            // what the eye showed at each step (None between tokens)
            let shown: Vec<Option<usize>> = ev.iter().map(|e| e.shown).collect();
            let mut r = StdRng::seed_from_u64(seed ^ 0xF01D);
            let phase: Vec<u8> = (0..tokens).map(|_| r.gen_range(0..=255u8)).collect();
            let at = |t: usize, back: usize| -> Option<usize> { if t >= back { shown[t - back] } else { None } };
            let mut policies: Vec<Vec<bool>> = vec![vec![false; n]; 6];
            // rare: counts so far
            let mut count = vec![0usize; tokens];
            let mut seen = 0usize;
            // window: one word of eight byte lanes; comb: the accumulator and its past values
            let mut window = 0u64;
            let mut acc: Vec<u8> = vec![0];
            for t in 1..n {
                // what this step can read: the eye a step ago (and before), as a lag-1 edge does
                let now = at(t, 1);
                policies[0][t] = t % 3 == 0;
                if let Some(x) = now {
                    policies[1][t] = seen >= 8 && count[x] * 7 < seen;
                    count[x] += 1;
                    seen += 1;
                    // frames: the same-as-before answer now and a step earlier, by XOR of frames
                    if let (Some(a), Some(b), Some(c)) = (at(t, 1), at(t, 2), at(t, 3)) {
                        policies[2][t] = (a == b) == (b == c);
                    }
                    window = (window << 8) | phase[x] as u64;
                    let (l0, l1, l2) = (window & 0xFF, (window >> 8) & 0xFF, (window >> 16) & 0xFF);
                    if t >= 4 {
                        policies[3][t] = (l0 == l1) == (l1 == l2);
                    }
                    let last = *acc.last().unwrap();
                    acc.push(last.wrapping_add(phase[x]));
                    let k = acc.len() - 1;
                    if k > period + 1 {
                        let s_now = acc[k].wrapping_sub(acc[k - period]);
                        let s_before = acc[k - 1].wrapping_sub(acc[k - 1 - period]);
                        policies[4][t] = s_now != s_before;
                    }
                }
                policies[5][t] = ev[t - 1].odd;
            }
            
            for (k, p) in policies.iter().enumerate() {
                let rv = replay(&ev, p);
                sums[k].0 += observe::surprise(&rv) as i64;
                sums[k].1 += observe::surprise_graded(&rv) as i64;
            }
        }
        let cell = |k: usize| format!("{} ({})", sums[k].0 / worlds as i64, sums[k].1 / worlds as i64);
        println!("{:<8} {:>13} {:>13} {:>13} {:>13} {:>13} {:>13}", name, cell(0), cell(1), cell(2), cell(3), cell(4), cell(5));
    }
}
