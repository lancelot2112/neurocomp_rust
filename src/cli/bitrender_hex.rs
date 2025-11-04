use std::io::{self, Write};

/// Render words as a hex dump with offset, hex bytes, and ASCII preview
pub fn render_words_hexdump<I, W>(words: I, words_per_row: usize, out: &mut W) -> io::Result<()>
where
    I: Iterator<Item = u64>,
    W: Write,
{
    let mut offset = 0usize;
    let mut row_words = Vec::with_capacity(words_per_row);
    
    for word in words {
        row_words.push(word);
        
        if row_words.len() == words_per_row {
            render_hex_row(offset, &row_words, out)?;
            offset += words_per_row * 8; // 8 bytes per word
            row_words.clear();
        }
    }
    
    if !row_words.is_empty() {
        render_hex_row(offset, &row_words, out)?;
    }
    
    Ok(())
}

fn render_hex_row<W: Write>(offset: usize, words: &[u64], out: &mut W) -> io::Result<()> {
    // Offset column (8 hex digits)
    write!(out, "{:08x}  ", offset)?;
    
    // Hex bytes (8 bytes grouped together per word, space between words)
    let mut all_bytes = Vec::new();
    for (i, &word) in words.iter().enumerate() {
        let bytes = word.to_le_bytes();
        for byte in bytes {
            write!(out, "{:02x}", byte)?;
            all_bytes.push(byte);
        }
        if i < words.len() - 1 {
            write!(out, " ")?; // space between 8-byte groups
        }
    }
    
    // ASCII column
    write!(out, "  |")?;
    for &byte in &all_bytes {
        let ch = if (0x20..=0x7e).contains(&byte) {
            (byte as u8) as char
        } else {
            '.'
        };
        write!(out, "{}", ch)?;
    }
    writeln!(out, "|")?;
    
    Ok(())
}

/// Simple hex renderer without ASCII preview
pub fn render_words_hex<I, W>(words: I, words_per_row: usize, out: &mut W) -> io::Result<()>
where
    I: Iterator<Item = u64>,
    W: Write,
{
    let mut col = 0;
    for word in words {
        write!(out, "{:016x}", word)?;
        col += 1;
        if col >= words_per_row {
            writeln!(out)?;
            col = 0;
        } else {
            write!(out, " ")?;
        }
    }
    if col != 0 {
        writeln!(out)?;
    }
    Ok(())
}

pub fn render_bits_hex<I, W>(bits: I, nibbles_per_row: usize, out: &mut W) -> io::Result<()>
where
    I: Iterator<Item = bool>,
    W: Write,
{
    let mut col = 0;
    let mut buffer = 0u8;
    let mut filled = 0;
    
    for bit in bits {
        buffer = (buffer << 1) | (bit as u8);
        filled += 1;
        
        if filled == 4 {
            write!(out, "{:x}", buffer)?;
            buffer = 0;
            filled = 0;
            col += 1;
            
            if col >= nibbles_per_row {
                writeln!(out)?;
                col = 0;
            }
        }
    }
    
    if filled > 0 {
        buffer <<= 4 - filled;
        write!(out, "{:x}", buffer)?;
    }
    
    if col != 0 {
        writeln!(out)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_words_hexdump() {
        let mut out = Vec::new();
        render_words_hexdump(
            [0xDEADBEEF_CAFEBABEu64, 0x0123456789ABCDEFu64].into_iter(),
            2,
            &mut out,
        )
        .unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("00000000"));
        assert!(s.contains("bebafecaefbeadde efcdab8967452301"));
        assert!(s.contains("|"));
    }

    #[test]
    fn test_render_words_hex() {
        let mut out = Vec::new();
        render_words_hex([0xDEADBEEFu64].into_iter(), 4, &mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert_eq!(s.trim(), "00000000deadbeef");
        assert!(!s.contains("|"));
    }
}