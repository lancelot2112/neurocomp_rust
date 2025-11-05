use std::io::{self, Write};

use crate::bitvec::bitdensity::BitDensity;

pub fn render_density_bw<I, W>(densities: I, cols: usize, out: &mut W) -> io::Result<()>
where
    I: Iterator<Item = BitDensity>,
    W: Write,
{
    let mut col = 0;
    for density in densities {
        write_cell_ansi(out, density.density)?;
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

fn write_cell_ansi<W: Write>(out: &mut W, density: u8) -> io::Result<()> {
    // Map density 0-255 to grayscale 0-255
    write!(out, "\x1b[48;2;{0};{0};{0}m \x1b[0m", density)
}

pub fn render_density_bw_unicode<I, W>(densities: I, cols: usize, out: &mut W) -> io::Result<()>
where
    I: Iterator<Item = BitDensity>,
    W: Write,
{
    let mut col = 0;
    for density in densities {
        write!(out, "{}", shade_symbol(density.density))?;
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

fn shade_symbol(density: u8) -> char {
    match density {
        0..=63 => '░',      // 0-25%
        64..=127 => '▒',    // 25-50%
        128..=191 => '▓',   // 50-75%
        _ => '█',           // 75-100%
    }
}

pub fn render_bits_bw<I, W>(bits: I, cols: usize, out: &mut W) -> io::Result<()>
where
    I: Iterator<Item = bool>,
    W: Write,
{
    let mut col = 0;
    for bit in bits {
        let ch = if bit { '█' } else { ' ' };
        write!(out, "{}", ch)?;
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

pub fn render_bits_bw_ansi<I, W>(bits: I, cols: usize, out: &mut W) -> io::Result<()>
where
    I: Iterator<Item = bool>,
    W: Write,
{
    let mut col = 0;
    for bit in bits {
        let shade = if bit { 255 } else { 0 };
        write!(out, "\x1b[48;2;{0};{0};{0}m \x1b[0m", shade)?;
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