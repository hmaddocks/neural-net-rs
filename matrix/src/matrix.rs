use ndarray::{Array2, ArrayView2, Axis, azip, concatenate, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Mul, Sub};

/// A 2D matrix implementation optimized for neural network operations.
///
/// This implementation uses ndarray internally for efficient matrix operations
/// and provides a high-level interface for common neural network computations.
///
/// # Features
/// - Efficient matrix operations using ndarray
/// - Neural network specific operations (bias augmentation, Xavier initialization)
/// - Broadcasting support for element-wise operations
/// - Serialization support via serde
///
/// # Example
/// ```
/// use matrix::matrix::Matrix;
///
/// // Create a 2x3 matrix
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let matrix = Matrix::new(2, 3, data);
///
/// // Perform matrix operations
/// let transposed = matrix.transpose();
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Matrix {
    pub data: Array2<f64>,
}

impl Matrix {
    /// Creates a new matrix with the specified dimensions and data.
    ///
    /// # Arguments
    /// * `rows` - Number of rows in the matrix
    /// * `cols` - Number of columns in the matrix
    /// * `data` - Vector containing the matrix elements in row-major order
    ///
    /// # Panics
    /// Panics if data.len() != rows * cols
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "Invalid matrix dimensions");
        Matrix {
            data: Array2::from_shape_vec((rows, cols), data)
                .expect("Failed to create matrix from data"),
        }
    }

    /// Creates a new matrix filled with zeros.
    ///
    /// # Arguments
    /// * `rows` - Number of rows in the matrix
    /// * `cols` - Number of columns in the matrix
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            data: Array2::zeros((rows, cols)),
        }
    }

    /// Creates a new matrix with random weights using Xavier/Glorot initialization.
    ///
    /// Xavier initialization helps maintain the variance of activations and gradients
    /// across layers, preventing vanishing or exploding gradients.
    ///
    /// The weights are initialized from a uniform distribution U(-scale, scale) where:
    /// scale = sqrt(6/(n_inputs + n_outputs))
    ///
    /// # Arguments
    /// * `rows` - Number of rows in the matrix (outputs)
    /// * `cols` - Number of columns in the matrix (inputs)
    pub fn random(rows: usize, cols: usize) -> Self {
        // Improved Xavier/Glorot initialization for better convergence
        let scale = (6.0 / (rows + cols) as f64).sqrt();
        let dist = Uniform::new(-scale, scale);
        Matrix {
            data: Array2::random((rows, cols), dist),
        }
    }

    /// Augments an input matrix with bias terms (adds a row of 1.0s).
    ///
    /// # Arguments
    /// * `input` - Input matrix to augment
    ///
    /// # Returns
    /// A new matrix with an additional row of 1.0s for bias terms
    pub fn augment_with_bias(&self) -> Self {
        let bias_row = Array2::ones((1, self.cols()));
        Matrix {
            data: concatenate![Axis(0), self.data, bias_row],
        }
    }

    /// Performs element-wise multiplication of two matrices.
    ///
    /// # Arguments
    /// * `other` - The matrix to multiply element-wise with
    ///
    /// # Panics
    /// Panics if the dimensions of the matrices don't match
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

    /// Performs matrix multiplication (dot product).
    ///
    /// # Arguments
    /// * `other` - The matrix to multiply with
    ///
    /// # Panics
    /// Panics if the number of columns in self does not match
    /// the number of rows in other.
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

    /// Returns a transposed copy of the matrix.
    ///
    /// # Returns
    /// A new matrix where rows and columns are swapped
    pub fn transpose(&self) -> Self {
        Matrix {
            data: self.data.t().to_owned(),
        }
    }

    /// Applies a function to each element of the matrix.
    ///
    /// # Arguments
    /// * `func` - Function that takes a f64 and returns a f64
    ///
    /// # Type Parameters
    /// * `F` - Type of the function to apply
    ///
    /// # Examples
    /// ```
    /// use matrix::matrix::Matrix;
    /// let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    /// let result = m.map(|x| x * 2.0);
    /// ```
    pub fn map<F>(&self, func: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        Matrix {
            data: self.data.mapv(func),
        }
    }

    /// Get a view of the matrix
    pub fn view(&self) -> ArrayView2<f64> {
        self.data.view()
    }

    /// Apply a function element-wise with broadcasting
    pub fn broadcast_apply<F>(&self, other: &Matrix, f: F) -> Matrix
    where
        F: Fn(f64, f64) -> f64,
    {
        Matrix {
            data: azip!(&self.data, &other.data).map_collect(|&a, &b| f(a, b)),
        }
    }

    /// Sum along specified axis
    pub fn sum_axis(&self, axis: Axis) -> Matrix {
        Matrix {
            data: self.data.sum_axis(axis).insert_axis(axis),
        }
    }

    /// Mean along specified axis
    pub fn mean_axis(&self, axis: Axis) -> Matrix {
        Matrix {
            data: self.data.mean_axis(axis).unwrap().insert_axis(axis),
        }
    }

    /// Concatenate matrices along specified axis
    pub fn concatenate(matrices: &[&Matrix], axis: Axis) -> Matrix {
        let views: Vec<_> = matrices.iter().map(|m| m.data.view()).collect();
        Matrix {
            data: concatenate(axis, &views[..]).unwrap(),
        }
    }

    /// Slice the matrix
    pub fn slice(&self, rows: std::ops::Range<usize>, cols: std::ops::Range<usize>) -> Matrix {
        Matrix {
            data: self.data.slice(ndarray::s![rows, cols]).to_owned(),
        }
    }

    /// Create a matrix from a slice
    pub fn from_slice(
        slice: &[f64],
        rows: usize,
        cols: usize,
    ) -> Result<Self, ndarray::ShapeError> {
        Ok(Matrix {
            data: Array2::from_shape_vec((rows, cols), slice.to_vec())?,
        })
    }

    /// Returns the value at the specified position in the matrix.
    ///
    /// # Arguments
    /// * `row` - Row index (0-based)
    /// * `col` - Column index (0-based)
    ///
    /// # Panics
    /// Panics if the indices are out of bounds
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[[row, col]]
    }

    /// Returns the number of rows in the matrix.
    pub fn rows(&self) -> usize {
        self.data.nrows()
    }

    /// Returns the number of columns in the matrix.
    pub fn cols(&self) -> usize {
        self.data.ncols()
    }

    /// Get a specific column from the matrix
    ///
    /// # Arguments
    /// * `col` - The index of the column to get
    ///
    /// # Returns
    /// A new Matrix containing the specified column
    pub fn col(&self, col: usize) -> Matrix {
        Matrix {
            data: self
                .data
                .slice(s![.., col])
                .to_owned()
                .into_shape_with_order((self.rows(), 1))
                .unwrap(),
        }
    }
}

/// Implementation of the Add trait for Matrix references.
///
/// # Arguments
/// * `other` - The matrix to add to this one
///
/// # Returns
/// A new matrix containing the sum of the two matrices
///
/// # Panics
/// Panics if the dimensions of the matrices don't match
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

/// Implementation of the Sub trait for Matrix references.
///
/// # Arguments
/// * `other` - The matrix to subtract from this one
///
/// # Returns
/// A new matrix containing the difference of the two matrices
///
/// # Panics
/// Panics if the dimensions of the matrices don't match
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

/// Implementation of the Mul trait for f64 and Matrix references.
///
/// # Arguments
/// * `other` - The scalar to multiply this matrix with
///
/// # Returns
/// A new matrix containing the product of the scalar and the matrix
impl Mul<f64> for &Matrix {
    type Output = Matrix;

    fn mul(self, other: f64) -> Self::Output {
        Matrix {
            data: &self.data * other,
        }
    }
}

impl Mul<Matrix> for f64 {
    type Output = Matrix;

    fn mul(self, matrix: Matrix) -> Self::Output {
        &matrix * self
    }
}

/// Implementation of the Default trait for Matrix.
///
/// Creates a 0x0 matrix filled with zeros.
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

/// Implementation of the Display trait for Matrix.
///
/// Formats the matrix for display with tab-separated values and newlines between rows.
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

    #[test]
    fn test_view() {
        let m = matrix![
            1.0, 2.0;
            3.0, 4.0
        ];
        let view = m.view();
        assert_eq!(view[[0, 0]], 1.0);
        assert_eq!(view[[0, 1]], 2.0);
        assert_eq!(view[[1, 0]], 3.0);
        assert_eq!(view[[1, 1]], 4.0);
    }

    #[test]
    fn test_broadcast_apply() {
        let m1 = matrix![
            1.0, 2.0;
            3.0, 4.0
        ];
        let m2 = matrix![
            2.0, 3.0;
            4.0, 5.0
        ];

        // Test addition
        let sum = m1.broadcast_apply(&m2, |a, b| a + b);
        assert_eq!(
            sum,
            matrix![
                3.0, 5.0;
                7.0, 9.0
            ]
        );

        // Test multiplication
        let product = m1.broadcast_apply(&m2, |a, b| a * b);
        assert_eq!(
            product,
            matrix![
                2.0, 6.0;
                12.0, 20.0
            ]
        );
    }

    #[test]
    fn test_sum_axis() {
        let m = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0
        ];

        // Sum along rows (Axis(1))
        let row_sums = m.sum_axis(Axis(1));
        assert_eq!(
            row_sums,
            matrix![
                6.0;
                15.0
            ]
        );

        // Sum along columns (Axis(0))
        let col_sums = m.sum_axis(Axis(0));
        assert_eq!(col_sums, matrix![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_mean_axis() {
        let m = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0
        ];

        // Mean along rows (Axis(1))
        let row_means = m.mean_axis(Axis(1));
        assert_eq!(
            row_means,
            matrix![
                2.0;
                5.0
            ]
        );

        // Mean along columns (Axis(0))
        let col_means = m.mean_axis(Axis(0));
        assert_eq!(col_means, matrix![2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_concatenate() {
        let m1 = matrix![
            1.0, 2.0;
            3.0, 4.0
        ];
        let m2 = matrix![
            5.0, 6.0;
            7.0, 8.0
        ];

        // Concatenate along rows (Axis(0))
        let vertical = Matrix::concatenate(&[&m1, &m2], Axis(0));
        assert_eq!(
            vertical,
            matrix![
                1.0, 2.0;
                3.0, 4.0;
                5.0, 6.0;
                7.0, 8.0
            ]
        );

        // Concatenate along columns (Axis(1))
        let horizontal = Matrix::concatenate(&[&m1, &m2], Axis(1));
        assert_eq!(
            horizontal,
            matrix![
                1.0, 2.0, 5.0, 6.0;
                3.0, 4.0, 7.0, 8.0
            ]
        );
    }

    #[test]
    fn test_slice() {
        let m = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0
        ];

        // Slice middle row and columns
        let middle = m.slice(1..2, 1..2);
        assert_eq!(middle, matrix![5.0]);

        // Slice multiple rows and columns
        let subset = m.slice(0..2, 1..3);
        assert_eq!(
            subset,
            matrix![
                2.0, 3.0;
                5.0, 6.0
            ]
        );
    }

    #[test]
    fn test_from_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0];

        // Create 2x2 matrix
        let m = Matrix::from_slice(&data, 2, 2).unwrap();
        assert_eq!(
            m,
            matrix![
                1.0, 2.0;
                3.0, 4.0
            ]
        );

        // Test error case with wrong dimensions
        assert!(Matrix::from_slice(&data, 3, 3).is_err());
    }
}
