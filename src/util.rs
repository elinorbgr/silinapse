//! A set of utility method to combine networks.

use std::marker::PhantomData;

use num::Float;

use Compute;
use {Method, UnsupervisedTrain, SupervisedTrain, BackpropTrain};

/*
 * Chaining
 */

/// An adapter tha chains two networks, linking the first's ouput to
/// the second's input.
pub struct Chain<F, A, B> where A: Compute<F>, B: Compute<F> {
    _marker: PhantomData<F>,
    first: A,
    second: B
}

impl<F, A, B> Chain<F, A, B>
    where F: Float, A: Compute<F>, B: Compute<F>
{
    /// Chains the two given adapters
    pub fn new(first: A, second: B) -> Chain<F, A, B> {
        Chain { _marker: PhantomData, first: first, second: second }
    }
}

impl<F, A, B> Compute<F> for Chain<F, A, B>
    where F: Float, A:Compute<F>, B: Compute<F>
{
    fn compute(&self, input: &[F]) -> Vec<F> {
        self.second.compute(&self.first.compute(input))
    }

    fn input_size(&self) -> usize {
        self.first.input_size()
    }

    fn output_size(&self) -> usize {
        self.second.output_size()
    }
}

/// The backpropagation training on a chain is computed this way:
///
/// - first compute the output of the first layer
/// - use the output of this layer to train the second with the target
/// - use the output of the training of the second layer as a target
///   to train the first
impl<F, A, B, M> BackpropTrain<F, M> for Chain<F, A, B>
    where F: Float,
          A: BackpropTrain<F, M> + Compute<F>,
          B: BackpropTrain<F, M> + Compute<F>,
          M: Method
{
    fn backprop_train(&mut self, rule: &M, input: &[F], target: &[F]) -> Vec<F> {
        let mid_input = self.first.compute(input);
        let mid_target = self.second.backprop_train(rule, &mid_input, target);
        self.first.backprop_train(rule, input, &mid_target)
    }
}

/// The supervised training on a chain is computed the same way as the
/// backprop training, simply discarding its output.
impl<F, A, B, M> SupervisedTrain<F, M> for Chain<F, A, B>
    where F: Float,
          A: BackpropTrain<F, M> + Compute<F>,
          B: BackpropTrain<F, M> + Compute<F>,
          M: Method
{
    fn supervised_train(&mut self, rule: &M, input: &[F], target: &[F]) {
        self.backprop_train(rule, input, target);
    }
}

/*
 * Parallelizing
 */

/// An adapter that feeds the same input to two networks, and concatenate
/// their outputs into its output.
pub struct Parallel<F, A, B> {
    _marker: PhantomData<F>,
    first: A,
    second: B
}

impl<F, A, B> Parallel<F, A, B>
    where F: Float, A: Compute<F>, B: Compute<F>
{
    /// Chains the two given adapters
    pub fn new(first: A, second: B) -> Parallel<F, A, B> {
        Parallel { _marker: PhantomData, first: first, second: second }
    }
}

impl<F, A, B> Compute<F> for Parallel<F, A, B>
    where F: Float, A:Compute<F>, B: Compute<F>
{
    fn compute(&self, input: &[F]) -> Vec<F> {
        let mut v = self.first.compute(&input);
        v.extend(self.second.compute(&input));
        v
    }

    fn input_size(&self) -> usize {
        ::std::cmp::max(self.first.input_size(), self.second.input_size())
    }

    fn output_size(&self) -> usize {
        self.first.output_size() + self.second.output_size()
    }
}

impl<F, A, B, M> UnsupervisedTrain<F, M> for Parallel<F, A, B>
    where F: Float,
          A: UnsupervisedTrain<F, M> + Compute<F>,
          B: UnsupervisedTrain<F, M> + Compute<F>,
          M: Method
{
    fn unsupervised_train(&mut self, rule: &M, input: &[F]) {
        self.first.unsupervised_train(rule, input);
        self.second.unsupervised_train(rule, input);
    }
}

impl<F, A, B, M> SupervisedTrain<F, M> for Parallel<F, A, B>
    where F: Float,
          A: SupervisedTrain<F, M> + Compute<F>,
          B: SupervisedTrain<F, M> + Compute<F>,
          M: Method
{
    fn supervised_train(&mut self, rule: &M, input: &[F], target: &[F]) {
        let n = self.first.output_size();
        if target.len() < n {
            self.first.supervised_train(rule, input, target);
            self.second.supervised_train(rule, input, &[]);
        } else {
            self.first.supervised_train(rule, input, &target[..n]);
            self.second.supervised_train(rule, input, &target[n..]);
        }
    }
}

/*
 * Fixed output
 */

/// A network that returns a fixed output, whatever the input is.
pub struct FixedOutput<F: Float> {
    output: Vec<F>
}

impl<F: Float> FixedOutput<F> {
    /// Creates a new fixed output network that will always return the
    /// values provided as `output`.
    pub fn new(output: &[F]) -> FixedOutput<F> {
        FixedOutput { output: output.to_owned() }
    }
}

impl<F: Float> Compute<F> for FixedOutput<F> {
    fn compute(&self, _input: &[F]) -> Vec<F> {
        self.output.clone()
    }

    fn input_size(&self) -> usize {
        0
    }

    fn output_size(&self) -> usize {
        self.output.len()
    }
}