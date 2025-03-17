use rand::Rng;
use std::fmt;
use std::ops::{Add, Sub};

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn elementwise_multiply(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "Matrix dimensions must match for elementwise multiplication"
        );

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a * b)
                .collect(),
        }
    }

    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::rng();
        let scale = 1.0 / (rows as f64).sqrt(); // Xavier/Glorot initialization

        Matrix {
            rows,
            cols,
            data: (0..rows * cols)
                .map(|_| rng.random_range(-scale..scale))
                .collect(),
        }
    }

    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "Invalid matrix dimensions");
        Matrix { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; cols * rows],
        }
    }

    pub fn dot_multiply(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, other.rows,
            "Invalid dimensions for matrix multiplication"
        );

        let mut result = vec![0.0; self.rows * other.cols];

        for i in 0..self.rows {
            for j in 0..other.cols {
                result[i * other.cols + j] = (0..self.cols)
                    .map(|k| self.data[i * self.cols + k] * other.data[k * other.cols + j])
                    .sum();
            }
        }

        Matrix {
            rows: self.rows,
            cols: other.cols,
            data: result,
        }
    }

    pub fn transpose(&self) -> Self {
        let mut result = vec![0.0; self.cols * self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }

        Matrix {
            rows: self.cols,
            cols: self.rows,
            data: result,
        }
    }

    pub fn map<F>(&self, func: F) -> Self
    where
        F: Fn(f64) -> f64,
    {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&x| func(x)).collect(),
        }
    }

    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row < self.rows && col < self.cols {
            Some(self.data[row * self.cols + col])
        } else {
            None
        }
    }

    // For backward compatibility with tests
    pub fn subtract(&self, other: &Matrix) -> Matrix {
        self - other
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        self + other
    }
}

impl Add for &Matrix {
    type Output = Matrix;

    fn add(self, other: &Matrix) -> Matrix {
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "Matrix dimensions must match for addition"
        );

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a + b)
                .collect(),
        }
    }
}

impl Sub for &Matrix {
    type Output = Matrix;

    fn sub(self, other: &Matrix) -> Matrix {
        assert_eq!(
            (self.rows, self.cols),
            (other.rows, other.cols),
            "Cannot subtract matrices with different dimensions"
        );

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a - b)
                .collect(),
        }
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

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..self.rows {
            let row_slice = &self.data[row * self.cols..(row + 1) * self.cols];
            writeln!(
                f,
                "{}",
                row_slice
                    .iter()
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
}
