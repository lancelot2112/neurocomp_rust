pub mod bitvector;
pub use bitvector::*;

pub mod bitnot;          // new module with Not and in-place invert
pub mod bitand;
pub mod bitor;
pub mod bitxor;
pub mod bitshift;
pub mod bitrot;
pub mod bits;
pub mod bitcount;
pub mod bitmask;
pub mod bititer;
pub mod bitslice;
pub mod bitdensity;
pub mod bithistory;
pub mod bitgen;

/*
Implements new methods in bitvector so nothing to re-export here
pub use bitnot::*;
pub use bitand::*;
pub use bitor::*;
pub use bitxor::*;
pub use bitshift::*;
pub use bitrot::*;
pub use bits::*;
pub use bitcount::*;
pub use bitmask::*;
*/
pub use bititer::*;
pub use bitslice::*;
pub use bitdensity::*;
pub use bithistory::*;
pub use bitgen::*;
