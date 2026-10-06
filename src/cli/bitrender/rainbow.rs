use std::io::{self, Write};

use crate::bitvec::bitdensity::BitDensity;

pub fn render_density_rainbow<I, W>(densities: I, cols: usize, out: &mut W) -> io::Result<()>
where
    I: Iterator<Item = BitDensity>,
    W: Write,
{
    let mut col = 0;
    for density in densities {
        write_cell_rainbow(out, density.density)?;
        col += 1;
        if col == cols {
            writeln!(out)?;
            col = 0;
        }
    }
    if col != 0 {
        writeln!(out)?;
    }
    Ok(())
}

pub fn render_bits_rainbow<I, W>(bits: I, cols: usize, out: &mut W) -> io::Result<()>
where
    I: Iterator<Item = bool>,
    W: Write,
{
    let mut col = 0;
    for bit in bits {
        let density = if bit { 255 } else { 0 };
        write_cell_rainbow(out, density)?;
        col += 1;
        if col == cols {
            writeln!(out)?;
            col = 0;
        }
    }
    if col != 0 {
        writeln!(out)?;
    }
    Ok(())
}

fn write_cell_rainbow<W: Write>(out: &mut W, density: u8) -> io::Result<()> {
    let d = density as f64 / 255.0; // float: report (display colour)
    let (r, g, b) = rainbow_color(d);
    write!(out, "\x1b[48;2;{};{};{}m \x1b[0m", r, g, b)
}

/// Map density in [0,1] onto a rainbow via HSV (hue sweep 0.0 → 0.8).
fn rainbow_color(density: f64) -> (u8, u8, u8) { // float: report (display colour)
    let d = density.clamp(0.0, 1.0);
    let hue = (1.0 - d) * 0.8; // start at red, go through yellow→green→cyan→blue
    hsv_to_rgb(hue, 1.0, 1.0)
}

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) { // float: report (display colour)
    let h = (h.fract() * 6.0).clamp(0.0, 6.0);
    let i = h.floor() as i32;
    let f = h - i as f64; // float: report (display colour)

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