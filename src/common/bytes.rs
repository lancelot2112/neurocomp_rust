use std::io;
use std::io::Write;

pub trait ByteVector {
    fn as_bytes(&self) -> &[u8];
}
/// Rendering extension trait for BitVector.
pub trait ByteRender : where Self: ByteVector {
    fn write_raw_to<W: Write>(&self, out: &mut W, width: usize) -> io::Result<()>;

    fn write_raw(&self, width: usize) -> io::Result<()> {
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        self.write_raw_to(&mut handle, width)
    }

    fn write_rgb_to<W: Write>(&self, out: &mut W, cells_per_row: usize, use_truecolor: bool) -> io::Result<()>;

    fn write_rgb(&self, cells_per_row: usize, use_truecolor: bool) -> io::Result<()> {
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        self.write_rgb_to(&mut handle, cells_per_row, use_truecolor)
    }
}