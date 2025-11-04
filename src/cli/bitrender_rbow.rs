use std::io::{self, Write};

pub fn render_bits_rainbow<I, W>(
    mut bits: I,
    cell_bits: usize,
    cols: usize,
    out: &mut W,
) -> io::Result<()>
where
    I: Iterator<Item = bool>,
    W: Write,
{
    assert!(cell_bits > 0, "cell_bits must be > 0");
    let mut col = 0;
    let mut acc = 0usize;
    let mut ones = 0usize;

    while let Some(bit) = bits.next() {
        acc += 1;
        if bit {
            ones += 1;
        }

        if acc == cell_bits {
            write_cell(out, ones, acc)?;
            acc = 0;
            ones = 0;
            col += 1;
            if col == cols {
                writeln!(out)?;
                col = 0;
            }
        }
    }

    if acc > 0 {
        write_cell(out, ones, acc)?;
        col += 1;
    }

    if col != 0 {
        writeln!(out)?;
    }

    Ok(())
}

pub fn render_words_rainbow<I, W>(iter: I, cols: usize, out: &mut W) -> io::Result<()>
where
    I: Iterator<Item = u64>,
    W: Write,
{
    let bits = iter.flat_map(|word| (0..64).rev().map(move |b| (word >> b) & 1 == 1));
    render_bits_rainbow(bits, 1, cols, out)
}

fn write_cell<W: Write>(out: &mut W, ones: usize, total: usize) -> io::Result<()> {
    let density = ones as f64 / total as f64;
    let (r, g, b) = rainbow_color(density);
    write!(out, "\x1b[48;2;{};{};{}m \x1b[0m", r, g, b)
}

/// Map density in [0,1] onto a rainbow via HSV (hue sweep 0.0 → 0.8).
fn rainbow_color(density: f64) -> (u8, u8, u8) {
    let d = density.clamp(0.0, 1.0);
    let hue = (1.0 - d) * 0.8; // start at red, go through yellow→green→cyan→blue
    hsv_to_rgb(hue, 1.0, 1.0)
}

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let h = (h.fract() * 6.0).clamp(0.0, 6.0);
    let i = h.floor() as i32;
    let f = h - i as f64;

    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    let (r, g, b) = match i {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };

    (
        (r * 255.0 + 0.5) as u8,
        (g * 255.0 + 0.5) as u8,
        (b * 255.0 + 0.5) as u8,
    )
}