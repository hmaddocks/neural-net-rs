use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;

// Adjust threshold based on matrix size for better performance
const PARALLEL_THRESHOLD: usize = 500; // Lower threshold for more parallel operations
const CHUNK_SIZE: usize = 32; // Cache-friendly chunk size for matrix operations

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Matrix {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) data: Vec<f64>,
}

impl Matrix {
    #[must_use]
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Data length must match rows * cols"
        );
        Self { rows, cols, data }
    }

    #[inline(always)]
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline(always)]
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline(always)]
    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    #[must_use]
    pub fn random(rows: usize, cols: usize) -> Self {
        let data = (0..rows * cols)
            .into_par_iter()
            .map_init(rand::thread_rng, |rng, _| rng.gen_range(-1.0..1.0))
            .collect();

        Self { rows, cols, data }
    }

    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; cols * rows],
        }
    }

    #[must_use]
    pub fn elementwise_multiply(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "Matrix rows must match");
        assert_eq!(self.cols, other.cols, "Matrix columns must match");

        let data = if self.data.len() >= PARALLEL_THRESHOLD {
            self.data
                .par_iter()
                .zip(other.data.par_iter())
                .map(|(&a, &b)| a * b)
                .collect()
        } else {
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a * b)
                .collect()
        };

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    #[must_use]
    pub fn dot_multiply(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, other.rows,
            "Invalid matrix dimensions for multiplication"
        );

        let mut result = vec![0.0; self.rows * other.cols];
        let other_t = other.transpose(); // Transpose for better cache locality

        if self.rows * other.cols >= PARALLEL_THRESHOLD {
            result
                .par_chunks_mut(other.cols)
                .enumerate()
                .for_each(|(i, row)| {
                    for j in (0..other.cols).step_by(CHUNK_SIZE) {
                        let chunk_end = (j + CHUNK_SIZE).min(other.cols);
                        let row_offset = i * self.cols;

                        // Process in cache-friendly chunks
                        for jj in j..chunk_end {
                            unsafe {
                                let mut sum = 0.0;
                                let other_col_offset = jj * other.rows;

                                // Use pointer arithmetic for faster access
                                let self_ptr = self.data.as_ptr().add(row_offset);
                                let other_ptr = other_t.data.as_ptr().add(other_col_offset);

                                for k in 0..self.cols {
                                    sum += *self_ptr.add(k) * *other_ptr.add(k);
                                }
                                // SAFETY: Indices are guaranteed to be in bounds by the matrix dimensions
                                *row.get_unchecked_mut(jj) = sum;
                            }
                        }
                    }
                });
        } else {
            // Sequential version for small matrices
            for i in 0..self.rows {
                let row_offset = i * self.cols;
                for j in 0..other.cols {
                    unsafe {
                        let mut sum = 0.0;
                        let other_col_offset = j * other.rows;

                        let self_ptr = self.data.as_ptr().add(row_offset);
                        let other_ptr = other_t.data.as_ptr().add(other_col_offset);

                        for k in 0..self.cols {
                            sum += *self_ptr.add(k) * *other_ptr.add(k);
                        }
                        // SAFETY: Indices are guaranteed to be in bounds by the matrix dimensions
                        *result.get_unchecked_mut(i * other.cols + j) = sum;
                    }
                }
            }
        }

        Matrix {
            rows: self.rows,
            cols: other.cols,
            data: result,
        }
    }

    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut data = vec![0.0; self.rows * self.cols];

        if self.data.len() >= PARALLEL_THRESHOLD {
            data.par_chunks_mut(self.rows)
                .enumerate()
                .for_each(|(j, chunk)| {
                    for i in 0..self.rows {
                        // SAFETY: Indices are guaranteed to be in bounds by the matrix dimensions
                        chunk[i] = unsafe { *self.data.get_unchecked(i * self.cols + j) };
                    }
                });
        } else {
            for i in 0..self.rows {
                for j in 0..self.cols {
                    data[j * self.rows + i] =
                        // SAFETY: Indices are guaranteed to be in bounds by the matrix dimensions
                        unsafe { *self.data.get_unchecked(i * self.cols + j) };
                }
            }
        }

        Matrix {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }

    #[must_use]
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f64) -> f64 + Sync + Send,
    {
        let data = if self.data.len() >= PARALLEL_THRESHOLD {
            self.data.par_iter().map(|&x| f(x)).collect()
        } else {
            self.data.iter().map(|&x| f(x)).collect()
        };

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    #[must_use]
    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "Matrix rows must match");
        assert_eq!(self.cols, other.cols, "Matrix columns must match");

        let data = if self.data.len() >= PARALLEL_THRESHOLD {
            self.data
                .par_iter()
                .zip(other.data.par_iter())
                .map(|(&a, &b)| a + b)
                .collect()
        } else {
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a + b)
                .collect()
        };

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }

    #[must_use]
    pub fn subtract(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "Matrix rows must match");
        assert_eq!(self.cols, other.cols, "Matrix columns must match");

        let data = if self.data.len() >= PARALLEL_THRESHOLD {
            self.data
                .par_iter()
                .zip(other.data.par_iter())
                .map(|(&a, &b)| a - b)
                .collect()
        } else {
            self.data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a - b)
                .collect()
        };

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

impl Default for Matrix {
    fn default() -> Self {
        Self::zeros(0, 0)
    }
}

impl From<Vec<f64>> for Matrix {
    fn from(vec: Vec<f64>) -> Self {
        let rows = vec.len();
        Matrix {
            rows,
            cols: 1,
            data: vec,
        }
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.data == other.data
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{:8.4}", unsafe {
                    // SAFETY: Indices are guaranteed to be in bounds by the matrix dimensions
                    *self.data.get_unchecked(i * self.cols + j)
                })?;
            }
            writeln!(f)?;
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

        assert_eq!(matrix.rows, rows);
        assert_eq!(matrix.cols, cols);
        assert_eq!(matrix.data.len(), rows * cols);

        for &num in &matrix.data {
            assert!(num >= -1.0 && num < 1.0);
        }
    }

    #[test]
    fn test_elementwise_multiply() {
        // Create two matrices for testing
        let matrix1 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let matrix2 = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        // Perform element-wise multiplication
        let result = matrix1.elementwise_multiply(&matrix2);

        // Define the expected result
        let expected_result = Matrix::new(2, 2, vec![5.0, 12.0, 21.0, 32.0]);

        // Check if the actual result matches the expected result
        assert_eq!(result, expected_result);
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

        let result = matrix1.subtract(&matrix2);

        let expected = matrix![
            -4.0, -4.0;
            -4.0, -4.0
        ];

        assert_eq!(result, expected);
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
    #[should_panic(expected = "Matrix columns must match")]
    fn test_subtract_different_dimensions() {
        let matrix1 = matrix![
            1.0, 2.0;
            3.0, 4.0
        ];

        let matrix2 = matrix![
            5.0, 6.0, 7.0;
            8.0, 9.0, 10.0
        ];

        let _ = matrix1.subtract(&matrix2);
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

        let result = a.add(&b);

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
        let matrix = Matrix {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };

        let transformed = matrix.map(|x| x + 1.0);

        let expected = Matrix {
            rows: 2,
            cols: 2,
            data: vec![2.0, 3.0, 4.0, 5.0],
        };

        assert_eq!(transformed, expected);
    }

    #[test]
    fn test_map_square() {
        let matrix = Matrix {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };

        let transformed = matrix.map(|x| x * x);

        let expected = Matrix {
            rows: 2,
            cols: 2,
            data: vec![1.0, 4.0, 9.0, 16.0],
        };

        assert_eq!(transformed, expected);
    }
}
