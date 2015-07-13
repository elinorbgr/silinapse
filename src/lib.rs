//! Silinapse -- A library of silicon synapses
//!
//! Accross the library, if the input slices do not contain enough
//! values for the unit they are given to, the missing values will
//! be inferred to be `0.0`. Samewise, superfluous values are ignored.
//!
//! The whole library is parametred over a type `F`, which can be any `Float` type
//! (currently `f32` or `f64`, but maybe others in the future).

#![warn(missing_docs)]

extern crate num;

use num::Float;

pub mod activations;
pub mod feedforward;
pub mod training;
pub mod util;

/// A trait representing anything that can process an input to generate an output.
///
/// This computation is not supposed to alter the internal state of the object.
pub trait Compute<F: Float>{
    /// Process input into output
    fn compute(&self, input: &[F]) -> Vec<F>;
    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
}

/// This trait describes a training method. It does not hold a lot of constraints
/// by itself, but networks implementing the same training method should be
/// trainable in the same way.
pub trait Method {}

/// A trait for networks that can be trained using a certain method of
/// unsupervised training.
pub trait UnsupervisedTrain<F: Float, M: Method> {
    fn unsupervised_train(&mut self, rule: &M, input: &[F]);
}

/// A trait for networks that can be trained using a certain method of
/// supervised training.
pub trait SupervisedTrain<F: Float, M: Method> {
    fn supervised_train(&mut self, rule: &M, input: &[F], target: &[F]);
}

/// A trait for networks that can be trained using a certain method in a
/// back-propagation way: the training returns a values vector that is
/// to be used as a target value for the previous layer.
pub trait BackpropTrain<F: Float, M: Method> {
    fn backprop_train(&mut self, rule: &M, input: &[F], target: &[F]) -> Vec<F>;
}