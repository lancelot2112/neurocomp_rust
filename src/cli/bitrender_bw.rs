use std::io::{self, Write};

pub fn render_bits_bw<I, W>(mut bits: I, cell_bits: usize, cols: usize, out: &mut W) -> io::Result<()>
where
    I: Iterator<Item = bool>,
    W: Write,
{
    let mut col = 0;
    let mut acc = 0usize;
    let mut ones = 0usize;

    while let Some(bit) = bits.next() {
        acc += 1;
        if bit { ones += 1; }
        if acc == cell_bits {
            write!(out, "{}", shade_symbol(ones, cell_bits))?;
            acc = 0;
            ones = 0;
            col += 1;
            if col == cols { writeln!(out)?; col = 0; }
        }
    }
    if acc > 0 {
        write!(out, "{}", shade_symbol(ones, acc))?;
    }
    if col != 0 { writeln!(out)?; }
    Ok(())
}

fn shade_symbol(ones: usize, total: usize) -> char {
    let density = ones * 100 / total;
    match density {
        0 => '░',         // empty
        1..=40 => '▒',    // mid
        _ => '▓',         // dense
    }
}

pub fn render_words_bw<I, W>(iter: I, cols: usize, out: &mut W) -> io::Result<()>
where
    I: Iterator<Item = u64>,
    W: Write,
{
    render_bits_bw(iter.flat_map(|w| (0..64).rev().map(move |b| (w >> b) & 1 == 1)), 1, cols, out)
}