//! A list of available training methods
//!
//! These types describe the parameters of each learning that can be
//! tune by the user.

use num::Float;

use Method;

/// The gradient descend approach, consisting on finding a minimum of the
/// error by going down its gradient.
pub struct GradientDescent<F: Float> {
    /// The learning rate associated with this gradient descent rule.
    ///
    /// A very small value will make the training slow, but a too big one
    /// will make it unstable.
    pub rate: F
}

impl<F: Float> Method for GradientDescent<F> {}

/// The perceptron rule, a classic learning rule for one-layered
/// feedforward networks.
pub struct PerceptronRule<F: Float> {
    /// The learning rate associated with this perceptron rule.
    ///
    /// A very small value will make the training slow, but a too big one
    /// will make it unstable.
    pub rate: F
}

impl<F: Float> Method for PerceptronRule<F> {}