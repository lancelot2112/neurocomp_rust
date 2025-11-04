use std::io::{self, Write};

pub fn render_words_braille<I, W>(words: I, bytes_per_row: usize, out: &mut W) -> io::Result<()>
where
    I: Iterator<Item = u64>,
    W: Write,
{
    let mut col = 0;
    for word in words {
        let bytes = word.to_le_bytes();
        for byte in bytes {
            write!(out, "{}│", byte_to_braille(byte))?;
            col += 1;
            if col >= bytes_per_row {
                writeln!(out)?;
                col = 0;
            }
        }
    }
    if col != 0 {
        writeln!(out)?;
    }
    Ok(())
}

pub fn render_words_braille_boxed<I, W>(words: I, bytes_per_row: usize, out: &mut W) -> io::Result<()>
where
    I: Iterator<Item = u64>,
    W: Write,
{
    let mut col = 0;
    for word in words {
        let bytes = word.to_le_bytes();
        for byte in bytes {
            write!(out, "{}┃", byte_to_braille(byte))?;
            col += 1;
            if col >= bytes_per_row {
                writeln!(out)?;
                col = 0;
            }
        }
    }
    if col != 0 {
        writeln!(out)?;
    }
    Ok(())
}

pub fn render_bits_braille<I, W>(bits: I, bytes_per_row: usize, out: &mut W) -> io::Result<()>
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
        
        if filled == 8 {
            write!(out, "{}│", byte_to_braille(buffer))?;
            buffer = 0;
            filled = 0;
            col += 1;
            
            if col >= bytes_per_row {
                writeln!(out)?;
                col = 0;
            }
        }
    }
    
    if filled > 0 {
        buffer <<= 8 - filled;
        write!(out, "{}│", byte_to_braille(buffer))?;
    }
    
    if col != 0 {
        writeln!(out)?;
    }
    Ok(())
}

#[inline]
fn byte_to_braille(byte: u8) -> char {
    // Remap bits to match desired layout:
    // Input:  bit 7 6 5 4 3 2 1 0
    // Wanted: │ 0 4 │
    //         │ 1 5 │
    //         │ 2 6 │
    //         │ 3 7 │
    // Unicode braille standard:
    //         │ 0 3 │
    //         │ 1 4 │
    //         │ 2 5 │
    //         │ 6 7 │
    
    let remapped = 
        ((byte & 0x01) << 0) |  // bit 0 → dot 0
        ((byte & 0x02) << 0) |  // bit 1 → dot 1
        ((byte & 0x04) << 0) |  // bit 2 → dot 2
        ((byte & 0x08) << 3) |  // bit 3 → dot 6
        ((byte & 0x10) >> 1) |  // bit 4 → dot 3
        ((byte & 0x20) >> 1) |  // bit 5 → dot 4
        ((byte & 0x40) >> 1) |  // bit 6 → dot 5
        ((byte & 0x80) << 0);   // bit 7 → dot 7
    
    char::from_u32(0x2800 + remapped as u32).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_to_braille() {
        assert_eq!(byte_to_braille(0x00), '⠀');
        assert_eq!(byte_to_braille(0xFF), '⣿');
        assert_eq!(byte_to_braille(0x01), '⠁');
        assert_eq!(byte_to_braille(0x80), '⢀');
        assert_eq!(byte_to_braille(0xAA), '⣒');
    }

    #[test]
    fn test_render_words_braille_boxed() {
        let mut out = Vec::new();
        render_words_braille_boxed([0x00u64, 0xFFu64].into_iter(), 16, &mut out).unwrap();
        let s = String::from_utf8(out).unwrap();
        assert!(s.contains("⠀┃")); // zero is visible
        assert!(s.contains("⣿┃")); // 0xFF
    }
}