//! Pattern separation check: four inputs, two pairs sharing half their lines, presented at
//! random; how much their codes overlap before and after learning.
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
    let mut prev: Option<Vec<BitVector>> = None;
    for round in [0usize, 50, 200, 1000, 4000, 4000] {
        let mut done = 0;
        while done < round {
            let i = rng.gen_range(0..4);
            sep.code(&codes[i], true);
            done += 1;
        }
        let c: Vec<BitVector> = codes.iter().map(|x| sep.code(x, false)).collect();
        let kept: Vec<u32> = prev.as_ref().map_or(vec![], |p| p.iter().zip(&c).map(|(a, b)| ov(a, b)).collect());
        println!("after {round:>5} more: within pair {} {}, across pairs {} {}, same code as before {:?}", ov(&c[0], &c[1]), ov(&c[2], &c[3]), ov(&c[0], &c[2]), ov(&c[1], &c[3]), kept);
        prev = Some(c);
    }
}
