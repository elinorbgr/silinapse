//! A set of utility method to combine networks.

use std::marker::PhantomData;

use num::Float;

use Compute;

/*
 * Chaining
 */

/// An adapter tha chains two networks, linking the first's ouput to
/// the second's input.
pub struct Chain<F, M, N> where M: Compute<F>, N: Compute<F> {
    _marker: PhantomData<F>,
    first: M,
    second: N
}

impl<F, M, N> Chain<F, M, N>
    where F: Float, M: Compute<F>, N: Compute<F>
{
    /// Chains the two given adapters
    pub fn new(first: M, second: N) -> Chain<F, M, N> {
        Chain { _marker: PhantomData, first: first, second: second }
    }
}

impl<F, M, N> Compute<F> for Chain<F, M, N>
    where F: Float, M:Compute<F>, N: Compute<F>
{
    fn compute(&self, input: &[F]) -> Vec<F> {
        self.second.compute(&self.first.compute(input))
    }
}

/*
 * Parallelizing
 */

/// An adapter that feeds the same input to two networks, and concatenate
/// their outputs into its output.
pub struct Parallel<F, M, N> {
    _marker: PhantomData<F>,
    first: M,
    second: N
}

impl<F, M, N> Parallel<F, M, N>
    where F: Float, M: Compute<F>, N: Compute<F>
{
    /// Chains the two given adapters
    pub fn new(first: M, second: N) -> Parallel<F, M, N> {
        Parallel { _marker: PhantomData, first: first, second: second }
    }
}

impl<F, M, N> Compute<F> for Parallel<F, M, N>
    where F: Float, M:Compute<F>, N: Compute<F>
{
    fn compute(&self, input: &[F]) -> Vec<F> {
        let mut v = self.first.compute(&input);
        v.extend(self.second.compute(&input));
        v
    }
}