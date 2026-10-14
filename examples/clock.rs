//! Can the tag grammar build clocks and counters from its ops, with no new primitive?
//!
//! - **Clock**: a channel K with one self-wire, op NOR, reading K `k` frames back. K starts
//!   empty, so K(t) = NOT K(t-k): a square wave of period 2k (on the wire's landing word).
//! - **Counter**: a channel C with a self-wire (OR, one frame back) that carries C forward, and a
//!   population reading the senses that ADDs its output into C in 8-bit lanes when it fires.
//!   Each firing adds the same constant, so line j of a lane flips every 2^(j-b) firings (b the
//!   lowest line of the constant): "a tick every 2^n times this input was on".
//! - **Tonic** for comparison: a population firing spontaneously with chance 2^-k a step.
//!
//! The senses show an input (one word all on) on a fixed pattern of steps. Prints each circuit's
//! lines on per step. `CLK_LAG` (k, default 2), `CLK_STEPS` (default 24).

use neurocomp::bitvec::bithistory::AdvanceMode;
use neurocomp::bitvec::BitVector;
use neurocomp::evolve::tags::{Pop, TagGenome, Token, SENSES};
use neurocomp::evolve::Evolvable;
use neurocomp::kernel::KernelOp;
use neurocomp::world::{self, World};

fn env<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

const K: u8 = 0x3C;
const C: u8 = 0xC3;
const T: u8 = 0x5A;

fn main() {
    let lag: u8 = env("CLK_LAG", 2);
    let steps: usize = env("CLK_STEPS", 24);
    let clock = Pop { lag, op: KernelOp::Nor, ..Pop::wire(1, K, K) };
    let carry = Pop::wire(2, C, C);
    let count = Pop { op: KernelOp::Add(3), count: 1, fan_in: 8, fan_out: 1, ..Pop::new(3, SENSES, C) };
    let tonic = Pop { count: 1, tonic: 2, fan_out: 4, ..Pop::new(4, SENSES, T) };
    let g = TagGenome {
        tokens: vec![
            Token::Channel { tag: K, width: 0, hold: false },
            Token::Channel { tag: C, width: 0, hold: false },
            Token::Channel { tag: T, width: 0, hold: false },
            Token::Pop(clock),
            Token::Pop(carry),
            Token::Pop(count),
            Token::Pop(tonic),
        ],
        next_id: 5,
    };
    let w = World::new(1, 4);
    let mut net = g.build(&w, 1);
    // nodes: senses 0, then the channels in order
    let (k, c, t) = (1, 2, 3);
    let on = BitVector::from_words({
        let mut v = vec![0u64; world::SENSE_BITS / 64];
        v[0] = u64::MAX;
        v
    });
    let off = BitVector::new(world::SENSE_BITS, Some(0));
    println!("{:>4} {:>6} {:>6} {:>18} {:>6}", "step", "input", "K", "C lane 0 (bits)", "tonic");
    for s in 0..steps {
        let input = s % 3 != 2; // on two steps of every three
        net.tick(AdvanceMode::Clear, 0);
        net.write_input(if input { &on } else { &off }, 0);
        let lane = net.read_node(c).as_words()[0] & 0xFF;
        println!(
            "{:>4} {:>6} {:>6} {:>10} {:08b} {:>6}",
            s,
            input as u8,
            net.read_node(k).count_ones(),
            lane,
            lane,
            net.read_node(t).count_ones()
        );
    }
}
