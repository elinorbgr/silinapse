//! A set of classic activation functions.

use num::{Float, one, zero};

/// Wraps two functions or closures as an activation function that can be
/// used by a network.
pub struct ActivationFunction<F, V, D>
    where F: Float,
          V: Fn(F) -> F,
          D: Fn(F) -> F
{
    _marker: ::std::marker::PhantomData<F>,
    /// Mathematical definition of the activation function, to be evaluated
    /// at any point.
    pub value: V,
    /// Mathematical derivative of the activation function, to be evaluated
    /// at any point.
    pub derivative: D
}

impl<F, V, D> ActivationFunction<F, V, D>
    where F: Float,
          V: Fn(F) -> F,
          D: Fn(F) -> F
{
    /// Create an `ActivationFunction` out of two functions or closures.
    pub fn new(value: V, derivative: D) -> ActivationFunction<F, V, D> {
        ActivationFunction {
            _marker: ::std::marker::PhantomData,
            value: value,
            derivative: derivative
        }
    }
}

/// Identity function, do not change its input.
///
/// Very bad for training but can be useful for debugging, or in some
/// special cases.
pub fn identity<F: Float>() -> ActivationFunction<F, fn(F) -> F, fn(F) -> F> {
    ActivationFunction::new(identity_val, identity_der)
}

fn identity_val<F: Float>(x: F) -> F { x }
fn identity_der<F: Float>(_x: F) -> F { one() }

/// Sigmoid function. A classic smooth learning function.
///
/// Its values are `0.0` at `-inf`, `0.5` at `0` and `1.0` at `+inf`
pub fn sigmoid<F: Float>() -> ActivationFunction<F, fn(F) -> F, fn(F) -> F> {
    ActivationFunction::new(sigmoid_val, sigmoid_der)
}

fn sigmoid_val<F: Float>(x: F) -> F { one::<F>() / ( one::<F>() + (-x).exp() ) }
fn sigmoid_der<F: Float>(x: F) -> F { x.exp() / ( one::<F>() + x.exp() ).powi(2) }

/// Step function. Cannot be used for learning, but can be used
/// to normalize data.
///
/// It outputs `1.0` if input was positive, and `-0.0` if input was negative.
pub fn step<F: Float>() -> ActivationFunction<F, fn(F) -> F, fn(F) -> F> {
    ActivationFunction::new(step_val, step_der)
}

fn step_val<F: Float>(x: F) -> F { if x.is_sign_positive() { one() } else { zero() } }
fn step_der<F: Float>(_x: F) -> F { zero() }

/// Gaussian function. Reaches its maximum `1.0` at `0.0`, and smoothly converges
/// towards `0.0` on both infinities.
pub fn gaussian<F: Float>() -> ActivationFunction<F, fn(F) -> F, fn(F) -> F> {
    ActivationFunction::new(gauss_val, gauss_der)
}

fn gauss_val<F: Float>(x: F) -> F { (-x.powi(2)).exp() }
// such a terrible way to make a two: v~~~~~~~~~~~~~~~~~~~v
fn gauss_der<F: Float>(x: F) -> F { -(one::<F>()+one::<F>())*x*(-x.powi(2)).exp() }