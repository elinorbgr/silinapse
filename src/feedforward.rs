//! Constructions related to feed-forward networks

use std::cmp::min;

use num::{Float, one, zero};

use Compute;

/// A feedforward layer
pub struct FeedforwardLayer<F: Float, A: Fn(F) -> F> {
    inputs: usize,
    coeffs: Vec<F>,
    biases: Vec<F>,
    activation: A
}

impl<F: Float, A: Fn(F) -> F> FeedforwardLayer<F, A> {
    /// Creates a new linear feedforward layer with all its weights set
    /// to 1 and its biases set to 0
    pub fn new(inputs: usize, outputs: usize, activation: A) -> FeedforwardLayer<F, A> {
        FeedforwardLayer {
            inputs: inputs,
            coeffs: vec![one(); inputs*outputs],
            biases: vec![zero(); outputs],
            activation: activation
        }
    }
}

impl<F: Float, A: Fn(F) -> F> Compute<F> for FeedforwardLayer<F, A> {
    fn compute(&self, input: &[F]) -> Vec<F> {
        let n = self.biases.len();
        let mut out = self.biases.clone();
        for i in 0..min(self.inputs, input.len()) {
            for j in 0..n {
                out[j] = out[j] + self.coeffs[i*n + j] * input[i]
            }
        }
        
        for o in &mut out {
            *o = (self.activation)(*o);
        }

        out
    }
}