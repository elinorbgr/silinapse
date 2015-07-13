//! A set of classic activation functions.

use num::{Float, one, zero};

/// Identity function, do not change its input.
///
/// Very bad for training but can be useful for debugging, or in some
/// special cases.
pub fn identity<F: Float>(x: F) -> F { x }

/// Sigmoid function. A classic smooth learning function.
///
/// Its values are `0.0` at `-inf`, `0.5` at `0` and `1.0` at `+inf`
pub fn sigmoid<F: Float>(x: F) -> F {
    one::<F>() / ( one::<F>() + x.exp() )
}

/// Step function. Cannot be used for learning, but can be used
/// to normalize data.
///
/// It outputs `1.0` if input was positive, and `-0.0` if input was negative.
pub fn step<F: Float>(x: F) -> F {
    if x.is_sign_positive() { one() } else { zero() }
}