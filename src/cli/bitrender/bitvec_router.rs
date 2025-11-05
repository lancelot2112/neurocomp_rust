use std::io::{self, stdout, Write};
use either::Either;

use crate::bitvec::BitVector;
use crate::cli::bitrender::blackwhite::{render_density_bw, render_density_bw_unicode};
use crate::cli::bitrender::braille::{render_words_braille, render_words_braille_boxed};
use crate::cli::bitrender::hexdump::{render_words_hexdump, render_words_hex};
use crate::cli::bitrender::rainbow::render_density_rainbow;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    /// Black/white grayscale ANSI (256 shades)
    Grayscale,
    /// Black/white Unicode blocks (4 shades)
    GrayscaleUnicode,
    /// Rainbow spectrum ANSI
    Rainbow,
    /// Braille characters (8 bits per char)
    Braille,
    /// Braille with box borders
    BrailleBoxed,
    /// Hex dump (like hex editor)
    HexDump,
    /// Simple hex (no ASCII preview)
    Hex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitOrder {
    /// Most significant bit first (high index to low)
    MostSignificantFirst,
    /// Least significant bit first (low index to high)
    LeastSignificantFirst,
}

pub struct RenderConfig {
    pub mode: RenderMode,
    pub cols: usize,
    pub cell_bits: usize, // for density-based modes
    pub bit_order: BitOrder,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            mode: RenderMode::Grayscale,
            cols: 64,
            cell_bits: 8,
            bit_order: BitOrder::LeastSignificantFirst,
        }
    }
}

pub fn render_bitvector(bv: &BitVector, config: &RenderConfig) -> io::Result<()> {
    let mut out = stdout();
    render_bitvector_to(bv, config, &mut out)
}

pub fn render_bitvector_to<W: Write>(
    bv: &BitVector,
    config: &RenderConfig,
    out: &mut W,
) -> io::Result<()> {
    match config.mode {
        RenderMode::Grayscale => {
            let densities = match config.bit_order {
                BitOrder::LeastSignificantFirst => Either::Left(bv.as_lsdensity(config.cell_bits)),
                BitOrder::MostSignificantFirst => Either::Right(bv.as_msdensity(config.cell_bits)),
            };
            render_density_bw(densities, config.cols, out)
        }
        RenderMode::GrayscaleUnicode => {
            let densities = match config.bit_order {
                BitOrder::LeastSignificantFirst => Either::Left(bv.as_lsdensity(config.cell_bits)),
                BitOrder::MostSignificantFirst => Either::Right(bv.as_msdensity(config.cell_bits)),
            };
            render_density_bw_unicode(densities, config.cols, out)
        }
        RenderMode::Rainbow => {
            let densities = match config.bit_order {
                BitOrder::LeastSignificantFirst => Either::Left(bv.as_lsdensity(config.cell_bits)),
                BitOrder::MostSignificantFirst => Either::Right(bv.as_msdensity(config.cell_bits)),
            };
            render_density_rainbow(densities, config.cols, out)
        }
        RenderMode::Braille => {
            let words = match config.bit_order {
                BitOrder::LeastSignificantFirst => Either::Left(bv.as_lswords()),
                BitOrder::MostSignificantFirst => Either::Right(bv.as_mswords()),
            };
            render_words_braille(words, config.cols, out)
        }
        RenderMode::BrailleBoxed => {
            let words = match config.bit_order {
                BitOrder::LeastSignificantFirst => Either::Left(bv.as_lswords()),
                BitOrder::MostSignificantFirst => Either::Right(bv.as_mswords()),
            };
            render_words_braille_boxed(words, config.cols, out)
        }
        RenderMode::HexDump => {
            let words_per_row = config.cols / 16;
            let words = match config.bit_order {
                BitOrder::LeastSignificantFirst => Either::Left(bv.as_lswords()),
                BitOrder::MostSignificantFirst => Either::Right(bv.as_mswords()),
            };
            render_words_hexdump(words, words_per_row.max(1), out)
        }
        RenderMode::Hex => {
            let words_per_row = config.cols / 16;
            let words = match config.bit_order {
                BitOrder::LeastSignificantFirst => Either::Left(bv.as_lswords()),
                BitOrder::MostSignificantFirst => Either::Right(bv.as_mswords()),
            };
            render_words_hex(words, words_per_row.max(1), out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_all_modes() {
        let bv = BitVector::from_words(vec![0xDEADBEEF_CAFEBABEu64]);
        
        for mode in [
            RenderMode::Grayscale,
            RenderMode::GrayscaleUnicode,
            RenderMode::Rainbow,
            RenderMode::Braille,
            RenderMode::BrailleBoxed,
            RenderMode::HexDump,
            RenderMode::Hex,
        ] {
            for bit_order in [BitOrder::LeastSignificantFirst, BitOrder::MostSignificantFirst] {
                let config = RenderConfig {
                    mode,
                    cols: 64,
                    cell_bits: 8,
                    bit_order,
                };
                let mut out = Vec::new();
                render_bitvector_to(&bv, &config, &mut out).unwrap();
                assert!(!out.is_empty(), "Mode {:?} with {:?} produced no output", mode, bit_order);
            }
        }
    }

     #[test]
     #[ignore="Visual test"]
    fn test_visual_comparison() {
         use rand::Rng; // add this import

        // Generate random bitvector with 8 words (512 bits)
        let mut rng = rand::thread_rng();
        let words: Vec<u64> = (0..8).map(|_| rng.r#gen::<u64>()).collect();
        let bv = BitVector::from_words(words);
        
        println!("\n=== RANDOM BITVECTOR VISUAL TEST ===\n");
        
        let modes = [
            RenderMode::Hex,
            RenderMode::HexDump,
            RenderMode::Braille,
            RenderMode::BrailleBoxed,
            RenderMode::GrayscaleUnicode,
            RenderMode::Grayscale,
            RenderMode::Rainbow,
        ];
        
        for &bit_order in &[BitOrder::MostSignificantFirst, BitOrder::LeastSignificantFirst] {
            println!("\n### {:?} ###\n", bit_order);
            
            for &mode in &modes {
                let config = RenderConfig {
                    mode,
                    cols: 64,
                    cell_bits: 8,
                    bit_order,
                };
                
                println!("--- {:?} ---", mode);
                render_bitvector(&bv, &config).unwrap();
                println!();
            }
        }
    }
}