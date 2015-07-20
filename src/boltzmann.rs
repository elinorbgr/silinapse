use num::{Float, one, zero};

use rand::{Rand, random, thread_rng};
use rand::distributions::{IndependentSample, Range};

use SymmetricMatrix;

pub struct BoltzmannMachine<F: Float> {
    values: Vec<F>,
    biases: Vec<F>,
    coeffs: SymmetricMatrix<F>
}

impl<F: Float> BoltzmannMachine<F> {
    pub fn new(coeffs: SymmetricMatrix<F>) -> BoltzmannMachine<F> {
        let n = coeffs.size();
        BoltzmannMachine {
            values: vec![one(); n],
            biases: vec![zero(); n],
            coeffs: coeffs
        }
    }

    pub fn with_biases(coeffs: SymmetricMatrix<F>, biases: Vec<F>) -> BoltzmannMachine<F> {
        let n = coeffs.size();
        assert!(biases.len() == n, "The biases count must be equal to the nodes count.");
        BoltzmannMachine {
            values: vec![one(); n],
            biases: biases,
            coeffs: coeffs
        }
    }


    pub fn values(&self) -> &[F] {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut [F] {
        &mut self.values
    }
}

impl<F: Float + Rand> BoltzmannMachine<F> {
    pub fn tick_all_sequential(&mut self, temperature: F) {
        let n = self.values.len();
        for i in 0..n {
            let mut val = self.biases[i];
            for j in 0..n {
                if i!=j {
                    val = val + self.values[j] * self.coeffs[(i,j)];
                }
            }
            val = -val / temperature;
            if random::<F>() < (one::<F>() + val.exp()).recip() {
                val = one::<F>();
            } else {
                val = zero::<F>();
            };
            self.values[i] = val;
        }
    }

    pub fn tick_one_random(&mut self, temperature: F, avoid: &[usize]) {
        let n = self.biases.len();
        let limits = Range::<usize>::new(0, n);
        let mut rng = thread_rng();
        let mut idx = limits.ind_sample(&mut rng);
        while avoid.contains(&idx) {
            idx = limits.ind_sample(&mut rng);
        }
        let mut val = self.biases[idx];
        for j in 0..n {
            if idx!=j {
                val = val + self.values[j] * self.coeffs[(idx,j)];
            }
        }
        val = -val / temperature;
        if random::<F>() < (one::<F>() + val.exp()).recip() {
            val = one::<F>();
        } else {
            val = zero::<F>();
        };
        self.values[idx] = val;
    }
}
