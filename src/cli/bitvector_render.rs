use std::io::{self, Write};
use crate::bitvec::BitVector;
use crate::common::bytes::{ByteRender, ByteVector};

// existing helper kept but changed to accept a writer
fn rgb_to_xterm256(r: u8, g: u8, b: u8) -> u8 {
    // map 0..255 -> 0..5
    let r6 = (r as u16 * 5 / 255) as u8;
    let g6 = (g as u16 * 5 / 255) as u8;
    let b6 = (b as u16 * 5 / 255) as u8;
    16 + 36 * r6 + 6 * g6 + b6
}

// changed: accept a writer so callers can choose stdout or any other writer
fn print_rgb_cells<W: Write>(
    out: &mut W,
    bytes: &[u8],
    cells_per_row: usize,
    use_truecolor: bool,
) -> io::Result<()> {
    let mut i = 0;
    while i + 2 < bytes.len() {
        for _col in 0..cells_per_row {
            if i + 2 >= bytes.len() {
                break;
            }
            let r = bytes[i];
            let g = bytes[i + 1];
            let b = bytes[i + 2];
            if use_truecolor {
                write!(out, "\x1b[48;2;{};{};{}m \x1b[0m", r, g, b)?;
            } else {
                let idx = rgb_to_xterm256(r, g, b);
                write!(out, "\x1b[48;5;{}m \x1b[0m", idx)?;
            }
            i += 3;
        }
        writeln!(out)?;
    }
    // if trailing bytes (<3) print remaining as hex / fallback
    if i < bytes.len() {
        write!(out, "  ")?;
        for b in &bytes[i..] {
            write!(out, "{:02x} ", b)?;
        }
        writeln!(out)?;
    }
    Ok(())
}

impl ByteRender for BitVector {
    fn write_raw(&self, width: usize) -> io::Result<()> {
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        self.write_raw_to(&mut handle, width)
    }
    fn write_raw_to<W: Write>(&self, out : &mut W, width: usize ) -> io::Result<()> {
                let total = self.len();
        let width = if width == 0 { 64 } else { width };
        let rows = (total + width - 1) / width;

        let row_prefix = format!("{:04}: ", 0);
        let prefix_width = row_prefix.len();
        let groups = (width + 7) / 8;

        // Map a linear position (pos) to the displayed index such that the full-vector
        // MSB is on the left: displayed = total - 1 - pos
        let map_display_idx = |pos: usize| -> Option<usize> {
            if pos < total {
                Some(total - 1 - pos)
            } else {
                None
            }
        };

        for r in 0..rows {
            let row_start = r * width;

            // header for this row: label each 8-bit group by the displayed index of its leftmost bit
            write!(out, "{}", " ".repeat(prefix_width))?;
            for g in 0..groups {
                let pos = row_start + g * 8;
                let label = match map_display_idx(pos) {
                    Some(idx) => idx,
                    None => 0,
                };
                write!(out, "{:<9}", label)?;
            }
            writeln!(out)?;

            // data row: for each group print 8 bits left->right where left is more significant
            //write!(out, "{:04}: ", row_start)?;
            for g in 0..groups {
                for b in 0..8 {
                    let pos = row_start + g * 8 + b;
                    if pos >= total { break; }
                    let displayed_idx = map_display_idx(pos).unwrap();
                    let ch = if self.get(displayed_idx) { '▓' } else { '░' };
                    write!(out, "{}", ch)?;
                }
                if g + 1 < groups { write!(out, " ")?; }
            }
            writeln!(out)?;
        }
        Ok(())
    }

    fn write_rgb(&self, cells_per_row: usize, use_truecolor: bool) -> io::Result<()> {
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        self.write_rgb_to(&mut handle, cells_per_row, use_truecolor)
    }
    fn write_rgb_to<W: Write>(&self, out: &mut W, cells_per_row: usize, use_truecolor: bool) -> io::Result<()> {
        let bytes = self.as_bytes();
        print_rgb_cells(out, &bytes, cells_per_row, use_truecolor)
    }
}

// changed: now returns io::Result so caller can handle errors and compose writers
fn print_raw_bitvector(bv: &BitVector, term_width: usize) -> io::Result<()> {
    let mut out = io::stdout();
    bv.write_raw_to(&mut out, term_width)
}

fn print_rgb_bitvector(bv: &BitVector, use_truecolor: bool) -> io::Result<()> {
    let mut out = io::stdout();
    bv.write_rgb_to(&mut out, 32, use_truecolor)
}