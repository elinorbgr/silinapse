use std::ops::{Index, IndexMut};

use num::{Float, zero};

pub struct SymmetricMatrix<F: Float> {
    size: usize,
    values: Vec<F>
}

impl<F: Float> SymmetricMatrix<F> {
    pub fn zeros(n: usize) -> SymmetricMatrix<F> {
        let coeffs = n*(n+1)/2;
        SymmetricMatrix {
            size: n,
            values: vec![zero(); coeffs]
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

fn order_tuple(t: (usize, usize)) -> (usize, usize) {
    if t.0 < t.1 { t } else { (t.1, t.0) }
}

impl<F: Float> Index<(usize, usize)> for SymmetricMatrix<F> {
    type Output = F;
    fn index(&self, index: (usize, usize)) -> &F {
        let (i, j) = order_tuple(index);
        let k = j*(j+1)/2 + i;
        &self.values[k]
    }
}

impl<F: Float> IndexMut<(usize, usize)> for SymmetricMatrix<F> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut F {
        let (i, j) = order_tuple(index);
        let k = j*(j+1)/2 + i;
        &mut self.values[k]
    }
}

#[cfg(test)]
mod tests {
    use super::SymmetricMatrix;

    #[test]
    fn symmetry() {
        let mut matrix = SymmetricMatrix::<f32>::zeros(8);
        // lower diagonal
        for i in 0..8 {
            for j in 0..(i+1) {
                matrix[(i,j)] = (i as f32) * 100.0 + (j as f32);
            }
        }
        // check values of upper diagonal
        for i in 0..8 {
            for j in i..8 {
                assert_eq!(matrix[(i,j)], (j as f32) * 100.0 + (i as f32));
            }
        }
    }
}