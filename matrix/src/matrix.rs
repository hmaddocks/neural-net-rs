use ndarray::Array2;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Sub};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Matrix {
    pub data: Array2<f64>,
}

impl Matrix {
    /// Augments an input matrix with bias terms (adds a row of 1.0s).
    ///
    /// # Arguments
    /// * `input` - Input matrix to augment
    ///
    /// # Returns
    /// A new matrix with an additional row of 1.0s for bias terms
    pub fn augment_with_bias(&self) -> Self {
        let mut new_data = Array2::zeros((self.rows() + 1, self.cols()));
        new_data
            .slice_mut(ndarray::s![..self.rows(), ..])
            .assign(&self.data);
        new_data.row_mut(self.rows()).fill(1.0);
        Matrix { data: new_data }
    }

    pub fn elementwise_multiply(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            (self.rows(), self.cols()),
            (other.rows(), other.cols()),
            "Matrix dimensions must match for elementwise multiplication"
        );
        Matrix {
            data: &self.data * &other.data,
        }
    }

    pub fn random(rows: usize, cols: usize) -> Self {
        let scale = 1.0 / (rows as f64).sqrt(); // Xavier/Glorot initialization
        let dist = Uniform::new(-scale, scale);
        Matrix {
            data: Array2::random((rows, cols), dist),
        }
    }

    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "Invalid matrix dimensions");
        Matrix {
            data: Array2::from_shape_vec((rows, cols), data)
                .expect("Failed to create matrix from data"),
        }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            data: Array2::zeros((rows, cols)),
        }
    }

    pub fn dot_multiply(&self, other: &Matrix) -> Self {
        assert_eq!(
            self.cols(),
            other.rows(),
            "Invalid dimensions for matrix multiplication"
        );
        Matrix {
            data: self.data.dot(&other.data),
        }
    }

    pub fn transpose(&self) -> Self {
        Matrix {
            data: self.data.t().to_owned(),
        }
    }

    pub fn map<F>(&self, func: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        Matrix {
            data: self.data.mapv(func),
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[[row, col]]
    }

    pub fn rows(&self) -> usize {
        self.data.nrows()
    }

    pub fn cols(&self) -> usize {
        self.data.ncols()
    }
}

impl Add for &Matrix {
    type Output = Matrix;

    fn add(self, other: &Matrix) -> Self::Output {
        assert_eq!(
            (self.rows(), self.cols()),
            (other.rows(), other.cols()),
            "Cannot add matrices with different dimensions"
        );
        Matrix {
            data: &self.data + &other.data,
        }
    }
}

impl Sub for &Matrix {
    type Output = Matrix;

    fn sub(self, other: &Matrix) -> Self::Output {
        assert_eq!(
            (self.rows(), self.cols()),
            (other.rows(), other.cols()),
            "Cannot subtract matrices with different dimensions"
        );
        Matrix {
            data: &self.data - &other.data,
        }
    }
}

impl Default for Matrix {
    fn default() -> Self {
        Self::zeros(0, 0)
    }
}

impl From<Vec<f64>> for Matrix {
    /// Converts a vector into a Matrix, treating it as a column vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix::matrix::Matrix;
    ///
    /// let vec = vec![1.0, 2.0, 3.0];
    /// let matrix: Matrix = vec.into();
    /// assert_eq!(matrix.rows(), 3);
    /// assert_eq!(matrix.cols(), 1);
    /// ```
    fn from(vec: Vec<f64>) -> Self {
        let rows = vec.len();
        Matrix::new(rows, 1, vec)
    }
}

/// A trait for converting a vector into a Matrix with specified dimensions
pub trait IntoMatrix {
    /// Converts self into a Matrix with specified dimensions
    fn into_matrix(self, rows: usize, cols: usize) -> Matrix;
}

impl IntoMatrix for Vec<f64> {
    /// Converts a vector into a Matrix with specified dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use matrix::matrix::{Matrix, IntoMatrix};
    ///
    /// let vec = vec![1.0, 2.0, 3.0, 4.0];
    /// let matrix = vec.into_matrix(2, 2);
    /// assert_eq!(matrix.rows(), 2);
    /// assert_eq!(matrix.cols(), 2);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if rows * cols != vec.len()
    fn into_matrix(self, rows: usize, cols: usize) -> Matrix {
        Matrix::new(rows, cols, self)
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in self.data.rows() {
            writeln!(
                f,
                "{}",
                row.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join("\t")
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix;

    #[test]
    fn test_random_matrix() {
        let rows = 3;
        let cols = 4;
        let matrix = Matrix::random(rows, cols);

        assert_eq!(matrix.rows(), rows);
        assert_eq!(matrix.cols(), cols);
        assert_eq!(matrix.data.len(), rows * cols);

        for &num in matrix.data.iter() {
            assert!(num >= -1.0 && num < 1.0);
        }
    }

    #[test]
    fn test_add() {
        let matrix1 = matrix![
            1.0, 2.0;
            3.0, 4.0
        ];

        let matrix2 = matrix![
            5.0, 6.0;
            7.0, 8.0
        ];

        let result = &matrix1 + &matrix2;

        let expected = matrix![
            6.0, 8.0;
            10.0, 12.0
        ];

        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic(expected = "Cannot add matrices with different dimensions")]
    fn test_add_mismatched_dimensions() {
        let matrix1 = matrix![
            1.0, 2.0;
            3.0, 4.0
        ];

        let matrix2 = matrix![
            5.0, 6.0;
            7.0, 8.0;
            9.0, 10.0
        ];

        let _ = &matrix1 + &matrix2;
    }

    #[test]
    fn test_subtract_same_dimensions() {
        let matrix1 = matrix![
            1.0, 2.0;
            3.0, 4.0
        ];

        let matrix2 = matrix![
            5.0, 6.0;
            7.0, 8.0
        ];

        let result = &matrix1 - &matrix2;

        let expected = matrix![
            -4.0, -4.0;
            -4.0, -4.0
        ];

        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic(expected = "Cannot subtract matrices with different dimensions")]
    fn test_subtract_different_dimensions() {
        let matrix1 = matrix![
            1.0, 2.0;
            3.0, 4.0
        ];

        let matrix2 = matrix![
            5.0, 6.0, 7.0;
            8.0, 9.0, 10.0
        ];

        let _ = &matrix1 - &matrix2;
    }

    #[test]
    fn test_dot_multiply() {
        let a = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0
        ];
        let b = matrix![
            7.0, 8.0;
            9.0, 10.0;
            11.0, 12.0
        ];

        let result = a.dot_multiply(&b);

        let expected_result = matrix![
            58.0, 64.0;
            139.0, 154.0
        ];

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_matrix_addition() {
        let a = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0
        ];

        let b = matrix![
            5.0, 6.0, 7.0;
            8.0, 9.0, 10.0;
            11.0, 12.0, 13.0
        ];

        let expected_result = matrix![
            6.0, 8.0, 10.0;
            12.0, 14.0, 16.0;
            18.0, 20.0, 22.0
        ];

        let result = &a + &b;

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_transpose_2x2() {
        let matrix = matrix![
            1.0, 2.0;
            3.0, 4.0
        ];
        let transposed = matrix.transpose();

        let expected = matrix![
            1.0, 3.0;
            2.0, 4.0
        ];
        assert_eq!(transposed, expected);
    }

    #[test]
    fn test_transpose_3x3() {
        let matrix = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0
        ];
        let transposed = matrix.transpose();

        let expected = matrix![
            1.0, 4.0, 7.0;
            2.0, 5.0, 8.0;
            3.0, 6.0, 9.0
        ];
        assert_eq!(transposed, expected);
    }

    #[test]
    fn test_transpose_4x3() {
        let matrix = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0;
            10.0, 11.0, 12.0
        ];
        let transposed = matrix.transpose();

        let expected = matrix![
            1.0, 4.0, 7.0, 10.0;
            2.0, 5.0, 8.0, 11.0;
            3.0, 6.0, 9.0, 12.0
        ];
        assert_eq!(transposed, expected);
    }

    #[test]
    fn test_map_add_one() {
        let matrix = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

        let transformed = matrix.map(|x| x + 1.0);

        let expected = Matrix::new(2, 2, vec![2.0, 3.0, 4.0, 5.0]);

        assert_eq!(transformed, expected);
    }

    #[test]
    fn test_map_square() {
        let matrix = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

        let transformed = matrix.map(|x| x * x);

        let expected = Matrix::new(2, 2, vec![1.0, 4.0, 9.0, 16.0]);

        assert_eq!(transformed, expected);
    }

    #[test]
    fn test_map_with_closure() {
        let m = matrix![
            1.0, 2.0;
            3.0, 4.0
        ];

        // Test with simple closure
        let result = m.map(|x| x * 2.0);
        assert_eq!(
            result,
            matrix![
                2.0, 4.0;
                6.0, 8.0
            ]
        );
    }

    #[test]
    fn test_map_with_capturing_closure() {
        let m = matrix![
            1.0, 2.0;
            3.0, 4.0
        ];

        let multiplier = 3.0;
        let result = m.map(|x| x * multiplier);
        assert_eq!(
            result,
            matrix![
                3.0, 6.0;
                9.0, 12.0
            ]
        );
    }

    #[test]
    fn test_map_with_function() {
        let m = matrix![
            -1.0, -2.0;
            3.0, -4.0
        ];

        fn abs(x: f64) -> f64 {
            x.abs()
        }

        let result = m.map(abs);
        assert_eq!(
            result,
            matrix![
                1.0, 2.0;
                3.0, 4.0
            ]
        );
    }

    #[test]
    fn test_map_chaining() {
        let m = matrix![
            1.0, 2.0;
            3.0, 4.0
        ];

        let result = m.map(|x| x * 2.0).map(|x| x + 1.0);
        assert_eq!(
            result,
            matrix![
                3.0, 5.0;
                7.0, 9.0
            ]
        );
    }

    #[test]
    fn test_from_vec() {
        let vec = vec![1.0, 2.0, 3.0];
        let matrix: Matrix = vec.into();

        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 1);
        assert_eq!(
            matrix.data.into_iter().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn test_into_matrix() {
        let vec = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = vec.into_matrix(2, 2);

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 2);
        assert_eq!(
            matrix.data.into_iter().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    #[should_panic(expected = "Invalid matrix dimensions")]
    fn test_into_matrix_invalid_dimensions() {
        let vec = vec![1.0, 2.0, 3.0];
        let _matrix = vec.into_matrix(2, 2); // Should panic
    }

    #[test]
    fn test_augment_with_bias() {
        let input = matrix![
            1.0, 2.0;
            3.0, 4.0
        ];

        let augmented = input.augment_with_bias();

        assert_eq!(augmented.rows(), 3); // Original rows + 1
        assert_eq!(augmented.cols(), 2); // Same number of columns

        // Check original data is preserved
        assert_eq!(augmented.get(0, 0), 1.0);
        assert_eq!(augmented.get(0, 1), 2.0);
        assert_eq!(augmented.get(1, 0), 3.0);
        assert_eq!(augmented.get(1, 1), 4.0);

        // Check bias row
        assert_eq!(augmented.get(2, 0), 1.0);
        assert_eq!(augmented.get(2, 1), 1.0);
    }

    #[test]
    fn test_augment_with_bias_empty() {
        let input = Matrix::zeros(0, 3);
        let augmented = input.augment_with_bias();

        assert_eq!(augmented.rows(), 1); // Just the bias row
        assert_eq!(augmented.cols(), 3);
        assert_eq!(
            augmented.data.into_iter().collect::<Vec<_>>(),
            vec![1.0, 1.0, 1.0]
        );
    }
}
