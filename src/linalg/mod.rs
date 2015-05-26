//! Basic linear algebra utilities.
//!
//! This module introduces `Matrix` and `Vector` types, used for basic linear
//! algebra calculus in the library.
//!
//! Mathematical operations on them can be done using the classic `+ - * /` operators.
//!
//! # Examples
//!
//! ```
//! use silinapse::linalg::{Matrix, Vector};
//!
//! // a vector of size 5 filled with 1.0
//! let x = Vector::<f32>::ones(5);
//! // a vector of size 6 filled with 0.5
//! let y = Vector::<f32>::ones(5) * 0.5;
//! let sum = &x + &y;
//! let scalar_product = &x * &y;
//!
//! // a (2, 3) matrix filled with ones
//! let m = Matrix::<f32>::ones(2,3);
//! // a 3d vector filled with ones
//! let a = Vector::<f32>::ones(3);
//! // apply this matrix on the vector from the left to make a 2d vector
//! let b = &m * &a;
//! // apply the matrix on the new vector from the right to make a new 3d vector
//! let c = &b * &m;
//! // c = a * 6
//! assert_eq!(c, &a * 6.0)
//! ```

use num::traits::{One, Zero};
use rand::{Rand, random};
use std::convert::AsRef;
use std::ops::{Add, Sub, Mul, Div, Index, IndexMut};

//
// Vector
//

/// A vector.
///
/// Can be used for various computation as long as the sizes of the vector and/or
/// matrices are compatibles. The library will panic if an incompatible operation
/// is attempted.
#[derive(Clone,PartialEq,Eq,Debug)]
pub struct Vector<T> {
    v: Vec<T>
}

impl<T: Zero> Vector<T> {
    /// Create a new zero-filled vector.
    pub fn zeros(size: usize) -> Vector<T> {
        let mut v = Vec::with_capacity(size);
        for _ in 0..size { v.push(Zero::zero()); }
        Vector { v: v }
    }
}

impl<T: One> Vector<T> {
    /// Create a new one-filles vector.
    pub fn ones(size: usize) -> Vector<T> {
        let mut v = Vec::with_capacity(size);
        for _ in 0..size { v.push(One::one()); }
        Vector { v: v }
    }
}

impl<T: Clone> Vector<T> {
    /// Creates a vector rom the values of given slice by cloning them.
    pub fn from_slice<S>(s: S) -> Vector<T>
        where S: AsRef<[T]>
    {
        Vector {
            v: s.as_ref().iter().map(|x| x.clone()).collect()
        }
    }

    /// Maps given closure on the values of this vector to cerate a new one
    /// of the same size.
    pub fn map<F>(&self, mut f: F) -> Vector<T>
        where F: FnMut(T) -> T
    {
        Vector {
            v: self.v.iter().map(|x| f(x.clone())).collect()
        }
    }
}

impl<T: Rand> Vector<T> {
    /// Create a new vector filled with random values using the default
    /// random generation of T.
    pub fn random(size: usize) -> Vector<T> {
        let mut v = Vec::with_capacity(size);
        for _ in 0..size { v.push(random()); }
        Vector { v: v }
    }
}

impl<T> Vector<T> {
    /// The size (or dimension) of this vector.
    pub fn dim(&self) -> usize {
        self.v.len()
    }

    /// Converts this vector into a single row matrix.
    pub fn into_row_matrix(self) -> Matrix<T> {
        Matrix {
            w: self.v.len(),
            v: self.v
        }
    }

    /// Converts this vector into a single column matrix.
    pub fn into_column_matrix(self) -> Matrix<T> {
        Matrix {
            w: 1,
            v: self.v
        }
    }
}

impl<T> Index<usize> for Vector<T> {
    type Output = T;

    fn index<'a>(&'a self, i: usize) -> &'a T {
        &self.v[i]
    }
}

impl<T> IndexMut<usize> for Vector<T> {
    fn index_mut<'a>(&'a mut self, i: usize) -> &'a mut T {
        &mut self.v[i]
    }
}

impl<'a, 'b, T: Add<Output=T> + Clone> Add<&'b Vector<T>> for Vector<T> {
    type Output = Vector<T>;

    fn add(self, rhs: &'b Vector<T>) -> Vector<T> {
        assert!(rhs.dim() == self.dim(), "Trying to sum two vectors of different sizes !");
        Vector {
            v: self.v.into_iter().zip(rhs.v.iter()).map(|(x,y)| x + y.clone()).collect()
        }
    }
}

impl<'a, 'b, T: Add<Output=T> + Clone> Add<&'b Vector<T>> for &'a Vector<T> {
    type Output = Vector<T>;

    fn add(self, rhs: &'b Vector<T>) -> Vector<T> {
        assert!(rhs.dim() == self.dim(), "Trying to sum two vectors of different sizes !");
        Vector {
            v: self.v.iter().zip(rhs.v.iter()).map(|(x,y)| x.clone() + y.clone()).collect()
        }
    }
}

impl<'a, 'b, T: Sub<Output=T> + Clone> Sub<&'b Vector<T>> for Vector<T> {
    type Output = Vector<T>;

    fn sub(self, rhs: &'b Vector<T>) -> Vector<T> {
        assert!(rhs.dim() == self.dim(), "Trying to substract two vectors of different sizes !");
        Vector {
            v: self.v.into_iter().zip(rhs.v.iter()).map(|(x,y)| x- y.clone()).collect()
        }
    }
}

impl<'a, 'b, T: Sub<Output=T> + Clone> Sub<&'b Vector<T>> for &'a Vector<T> {
    type Output = Vector<T>;

    fn sub(self, rhs: &'b Vector<T>) -> Vector<T> {
        assert!(rhs.dim() == self.dim(), "Trying to substract two vectors of different sizes !");
        Vector {
            v: self.v.iter().zip(rhs.v.iter()).map(|(x,y)| x.clone() - y.clone()).collect()
        }
    }
}

impl<'a, 'b, T: Add<Output=T> + Mul<Output=T> + Clone + Zero> Mul<&'b Vector<T>> for &'a Vector<T> {
    type Output = T;

    fn mul(self, rhs: &'b Vector<T>) -> T {
        assert!(rhs.dim() == self.dim(), "Trying to take the scalar product of two vectors of different sizes !");
        self.v.iter().zip(rhs.v.iter()).fold(Zero::zero(), |s, (x, y)| s + x.clone() * y.clone())
    }
}

impl<T: Mul<Output=T> + Clone> Mul<T> for Vector<T> {
    type Output = Vector<T>;

    fn mul(self, rhs: T) -> Vector<T> {
        Vector {
            v: self.v.into_iter().map(|x| x * rhs.clone()).collect()
        }
    }
}

impl<'a, T: Mul<Output=T> + Clone> Mul<T> for &'a Vector<T> {
    type Output = Vector<T>;

    fn mul(self, rhs: T) -> Vector<T> {
        Vector {
            v: self.v.iter().map(|x| x.clone() * rhs.clone()).collect()
        }
    }
}

impl<T: Div<Output=T> + Clone> Div<T> for Vector<T> {
    type Output = Vector<T>;

    fn div(self, rhs: T) -> Vector<T> {
        Vector {
            v: self.v.into_iter().map(|x| x.clone() / rhs.clone()).collect()
        }
    }
}

impl<'a, T: Div<Output=T> + Clone> Div<T> for &'a Vector<T> {
    type Output = Vector<T>;

    fn div(self, rhs: T) -> Vector<T> {
        Vector {
            v: self.v.iter().map(|x| x.clone() / rhs.clone()).collect()
        }
    }
}

//
// Matrix
//

/// A matrix.
///
/// Can be used for various computation as long as the sizes of the vector and/or
/// matrices are compatibles. The library will panic if an incompatible operation
/// is attempted.
#[derive(Clone,PartialEq,Eq,Debug)]
pub struct Matrix<T> {
    w: usize,
    v: Vec<T>
}

impl<T> Matrix<T> {
    #[inline]
    fn coords_to_index(&self, c: (usize, usize)) -> usize {
        c.0 * self.w + c.1
    }

    /// The dimensions of this matrix: `(height, width)`
    pub fn dims(&self) -> (usize, usize) {
        (self.v.len()/self.w, self.w)
    }
}

impl<T: Zero> Matrix<T> {
    /// Create a new zero-filled matrix.
    pub fn zeros(height: usize, width: usize) -> Matrix<T> {
        let mut v = Vec::with_capacity(height * width);
        for _ in 0..(height*width) { v.push(Zero::zero()); }
        Matrix {
            w: width,
            v: v
        }
    }
}

impl<T: One> Matrix<T> {
    /// Create a new one-filled matrix.
    pub fn ones(height: usize, width: usize) -> Matrix<T> {
        let mut v = Vec::with_capacity(height * width);
        for _ in 0..(height*width) { v.push(One::one()); }
        Matrix {
            w: width,
            v: v
        }
    }
}

impl<T: Clone> Matrix<T> {
    /// Create a new matrix from given slice of slices.
    ///
    /// # Example
    ///
    /// ```
    /// use silinapse::linalg::Matrix;
    ///
    /// // Create a (2, 3) matrix:
    /// let m = Matrix::from_slices([[1.0f32, 2.0, 3.0],
    ///                              [4.0   , 5.0, 6.0]]);
    /// ```
    pub fn from_slices<S, SS>(ss: SS) -> Matrix<T>
        where S: AsRef<[T]>,
              SS: AsRef<[S]>
    {
        let slices = ss.as_ref();
        let h = slices.len();
        assert!(h > 0, "Attempted to create a mtrix of size 0.");
        let w = slices[0].as_ref().len();
        assert!(w > 0, "Attempted to create a mtrix of size 0.");
        let mut v = Vec::with_capacity(w * slices.len());
        for slice in slices {
            assert!(slice.as_ref().len() == w, "All rows of a matrix must have the same length.");
            v.extend(slice.as_ref().iter().map(|x| x.clone()));
        }
        Matrix {
            w: w,
            v: v
        }
    }
}

impl<T: Rand> Matrix<T> {
    /// Create a new random matrix using the default random generation of T.
    pub fn random(height: usize, width: usize) -> Matrix<T> {
        let mut v = Vec::with_capacity(height * width);
        for _ in 0..(height*width) { v.push(random()); }
        Matrix {
            w: width,
            v: v
        }
    }
}

impl<T> Index<(usize,usize)> for Matrix<T> {
    type Output = T;

    fn index<'a>(&'a self, c: (usize, usize)) -> &'a T {
        &self.v[self.coords_to_index(c)]
    }
}

impl<T> IndexMut<(usize,usize)> for Matrix<T> {
    fn index_mut<'a>(&'a mut self, c: (usize, usize)) -> &'a mut T {
        let c = self.coords_to_index(c);
        &mut self.v[c]
    }
}

impl<'a, 'b, T: Add<Output=T> + Clone> Add<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, rhs: &'b Matrix<T>) -> Matrix<T> {
        assert!(rhs.dims() == self.dims(), "Trying to sum two matrices of different sizes !");
        let v = self.v.iter().zip(rhs.v.iter()).map(|(x,y)| x.clone() + y.clone()).collect::<Vec<_>>();
        Matrix {
            w: self.w,
            v: v
        }
    }
}

impl<'a, 'b, T: Sub<Output=T> + Clone> Sub<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, rhs: &'b Matrix<T>) -> Matrix<T> {
        assert!(rhs.dims() == self.dims(), "Trying to substract two vectors of different sizes !");
        let v = self.v.iter().zip(rhs.v.iter()).map(|(x,y)| x.clone() - y.clone()).collect::<Vec<_>>();
        Matrix {
            w: self.w,
            v: v
        }
    }
}

impl<'a, T: Mul<Output=T> + Clone> Mul<T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: T) -> Matrix<T> {
        let v = self.v.iter().map(|x| x.clone() * rhs.clone()).collect::<Vec<_>>();
        Matrix {
            w: self.w,
            v: v
        }
    }
}

impl<'a, T: Div<Output=T> + Clone> Div<T> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn div(self, rhs: T) -> Matrix<T> {
        let v = self.v.iter().map(|x| x.clone() / rhs.clone()).collect::<Vec<_>>();
        Matrix {
            w: self.w,
            v: v
        }
    }
}

impl<'a, 'b, T: Add<Output=T> + Mul<Output=T> + Clone + Zero> Mul<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: &'b Matrix<T>) -> Matrix<T> {
        assert!(self.dims().1 == rhs.dims().0, "Attempted to take the product of matrices with incompatible sizes.");
        let mut m = Matrix::<T>::zeros(self.dims().0, rhs.dims().1);
        for i in 0..self.dims().0 {
            for j in 0..rhs.dims().1 {
                for k in 0..self.dims().1 {
                    m[(i,j)] = m[(i,j)].clone() + self[(i,k)].clone() * rhs[(k,j)].clone();
                }
            }
        }
        m
    }
}

impl<'a,'b, T: Add<Output=T> + Mul<Output=T> + Clone + Zero> Mul<&'b Vector<T>> for &'a Matrix<T> {
    type Output = Vector<T>;

    fn mul(self, rhs: &'b Vector<T>) -> Vector<T> {
        assert!(self.dims().1 == rhs.dim(), "Attempted to apply a matrix to a vector of incompatible size.");
        let mut v = Vector::<T>::zeros(self.dims().0);
        for i in 0..self.dims().0 {
            for j in 0..rhs.dim() {
                v[i] = v[i].clone() + self[(i,j)].clone() * rhs[j].clone();
            }
        }
        v
    }
}

impl<'a, 'b, T: Add<Output=T> + Mul<Output=T> + Clone + Zero> Mul<&'b Matrix<T>> for &'a Vector<T> {
    type Output = Vector<T>;

    fn mul(self, rhs: &'b Matrix<T>) -> Vector<T> {
        assert!(rhs.dims().0 == self.dim(), "Attempted to apply a matrix to a vector of incompatible size.");
        let mut v = Vector::<T>::zeros(rhs.dims().1);
        for i in 0..rhs.dims().0 {
            for j in 0..rhs.dims().1 {
                v[j] = v[j].clone() + rhs[(i,j)].clone() * self[i].clone();
            }
        }
        v
    }
}

#[cfg(test)]
mod test {

    use super::Vector;
    use super::Matrix;

    #[test]
    fn basic_vector_operations() {
        let a = Vector::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let b = Vector::from_slice([0.5f32, 1.0, 1.5, 2.0, 2.5]);
        assert_eq!(&b * 2.0, a);
        assert_eq!(&a / 2.0, b);
        assert_eq!(&a - &b, b);
        assert_eq!(&b + &b, a);
    }

    #[test]
    fn vector_product() {
        let a = Vector::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let b = Vector::from_slice([0.5f32, 1.0, 1.5, 2.0, 2.5]);
        assert_eq!(&a * &b, 27.5f32);
    }

    #[test]
    #[should_panic]
    fn wrong_vector_sizes_sum() {
        let a = Vector::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let b = Vector::from_slice([0.5f32, 1.0]);
        let _c = &a + &b;
    }

    #[test]
    fn basic_matrix_operations() {
        let a = Matrix::from_slices([[1.0f32, 2.0],
                                     [2.0f32, 3.0]]);
        let b = Matrix::from_slices([[0.5f32, 1.0],
                                     [1.0f32, 1.5]]);
        assert_eq!(&b * 2.0, a);
        assert_eq!(&a / 2.0, b);
        assert_eq!(&a - &b, b);
        assert_eq!(&b + &b, a);
    }

    #[test]
    fn matrix_vector_operations() {
        let a = Matrix::from_slices([[1.0f32, 2.0, 3.0],
                                     [3.0f32, 4.0, 5.0]]);
        let x = Vector::from_slice([0.5f32, 1.0, 1.5]);
        let y = Vector::from_slice([0.5f32, 1.0]);
        assert_eq!(&a * &x, Vector::from_slice([7f32, 13.0]));
        assert_eq!(&y * &a, Vector::from_slice([3.5f32, 5.0, 6.5]));
    }

    #[test]
    #[should_panic]
    fn wrong_matrix_sizes_sum() {
        let a = Matrix::from_slices([[1.0f32, 2.0],
                                     [2.0f32, 3.0]]);
        let b = Matrix::from_slices([[0.5f32, 1.0, 0.0],
                                     [1.0f32, 1.5, 0.0]]);
        let _c = &a + &b;
    }

}