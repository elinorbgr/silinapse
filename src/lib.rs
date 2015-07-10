//! Silinapse -- A library of silicon synapses

extern crate num;

use num::Float;

pub mod feedforward;
pub mod util;

/// A trait representing anything that can process an input to generate an output.
///
/// This computation is not supposed to alter the internal state of the object.
pub trait Compute<F: Float>{
    /// Process input into output
    fn compute(&self, input: &[F]) -> Vec<F>;
}
