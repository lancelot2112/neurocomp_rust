//! Pattern separation check: four inputs, two pairs sharing half their lines. The separator
//! develops on three of them; then the fourth arrives (with the others) and is learned only as
//! its novelty reopens plasticity. How much each code overlaps the others, and whether it is
//! stable.
use neurocomp::bitvec::BitVector;
use neurocomp::program::assembly::Assembly;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() {
    let mut rng = StdRng::seed_from_u64(1);
    let mut sparse = |k: usize, rng: &mut StdRng| {
        let mut v = BitVector::new(1024, Some(0));
        while (v.count_ones() as usize) < k {
            v.bit_set(rng.gen_range(0..1024));
        }
        v
    };
    let cue = [sparse(16, &mut rng), sparse(16, &mut rng)];
    let codes: Vec<BitVector> = (0..4)
        .map(|i| {
            let mut v = sparse(16, &mut rng);
            for (w, c) in v.as_words_mut().iter_mut().zip(cue[i / 2].as_words()) {
                *w |= c;
            }
            v
        })
        .collect();
    let ov = |a: &BitVector, b: &BitVector| a.as_words().iter().zip(b.as_words()).map(|(x, y)| (x & y).count_ones()).sum::<u32>();
    let mut sep = Assembly::coding(1024, 16, 1024, 32, 7);
    for _ in 0..2000 {
        let i = [0usize, 1, 2][rng.gen_range(0..3usize)];
        sep.code(&codes[i], true);
    }
    let show = |sep: &mut Assembly, label: &str| {
        let c: Vec<BitVector> = codes.iter().map(|x| sep.code(x, false)).collect();
        println!("{label}: A paid/A unpaid {}, B unpaid/B paid {}, A paid/B paid {}", ov(&c[0], &c[1]), ov(&c[2], &c[3]), ov(&c[0], &c[3]));
    };
    show(&mut sep, "after development (B paid never seen)");
    for round in [10usize, 50, 200, 1000] {
        for _ in 0..round {
            let i = rng.gen_range(0..4usize);
            sep.code(&codes[i], true);
        }
        show(&mut sep, &format!("after {round} more of all four"));
    }
}
