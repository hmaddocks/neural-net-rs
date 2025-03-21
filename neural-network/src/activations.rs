use matrix::matrix::Matrix;
use serde::{Deserialize, Serialize};
use std::f64::consts::E;

/// Type of activation function
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ActivationType {
    Sigmoid,
    Softmax,
}

/// Represents an activation function used in neural networks.
///
/// An activation function consists of both the function itself and its derivative,
/// along with optional vector-based implementations for improved performance.
///
/// # Examples
///
/// ```
/// use neural_network::activations::{Activation, SIGMOID};
///
/// let result = (SIGMOID.function.unwrap())(0.5);
/// let derivative = (SIGMOID.derivative.unwrap())(result);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Activation {
    /// The scalar activation function
    pub function: Option<fn(f64) -> f64>,
    /// The derivative of the scalar activation function
    pub derivative: Option<fn(f64) -> f64>,
    /// Optional vectorized implementation of the activation function
    pub vector_function: Option<fn(&Matrix) -> Matrix>,
    /// Optional vectorized implementation of the derivative
    pub vector_derivative: Option<fn(&Matrix) -> Matrix>,
    /// The type of activation function, used for serialization
    pub activation_type: ActivationType,
}

impl Serialize for Activation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.activation_type.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Activation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let activation_type = ActivationType::deserialize(deserializer)?;

        Ok(match activation_type {
            ActivationType::Sigmoid => SIGMOID,
            ActivationType::Softmax => SOFTMAX,
        })
    }
}

impl Activation {
    /// Applies the activation function to a matrix.
    ///
    /// Uses the vector implementation if available, otherwise falls back to element-wise application.
    ///
    /// # Arguments
    ///
    /// * `input` - The input matrix to apply the activation function to
    pub fn apply_vector(&self, input: &Matrix) -> Matrix {
        match self.vector_function {
            Some(f) => f(input),
            None => match self.function {
                Some(f) => input.map(f),
                None => panic!("No vector or scalar function implementation available"),
            },
        }
    }

    /// Applies the derivative of the activation function to a matrix.
    ///
    /// Uses the vector implementation if available, otherwise falls back to element-wise application.
    ///
    /// # Arguments
    ///
    /// * `input` - The input matrix to apply the derivative to
    pub fn derivative_vector(&self, input: &Matrix) -> Matrix {
        match self.vector_derivative {
            Some(f) => f(input),
            None => match self.derivative {
                Some(f) => input.map(f),
                None => panic!("No vector or scalar derivative implementation available"),
            },
        }
    }
}

/// Standard sigmoid activation function.
///
/// Implements the logistic function: f(x) = 1 / (1 + e^(-x))
/// with derivative: f'(x) = f(x) * (1 - f(x))
pub const SIGMOID: Activation = Activation {
    function: Some(|x| 1.0 / (1.0 + E.powf(-x))),
    derivative: Some(|x| x * (1.0 - x)),
    vector_function: Some(|m| m.map(|x| 1.0 / (1.0 + E.powf(-x)))),
    vector_derivative: Some(|m| m.map(|x| x * (1.0 - x))),
    activation_type: ActivationType::Sigmoid,
};

/// Standard softmax activation function.
///
/// Implements the softmax function: f(x) = e^x / sum(e^x)
/// with derivative: f'(x) = f(x) * (1 - f(x))
pub const SOFTMAX: Activation = Activation {
    function: None,
    derivative: None,
    vector_function: Some(|m| {
        let mut exp_values = Vec::with_capacity(m.rows * m.cols);
        exp_values.extend(m.data.iter().map(|x| E.powf(*x)));
        let mut exp_matrix = Matrix::new(m.rows, m.cols, exp_values);

        // For column vectors, treat the entire vector as one group
        if m.cols == 1 {
            let sum: f64 = exp_matrix.data.iter().sum();
            for i in 0..m.rows {
                exp_matrix.data[i] /= sum;
            }
            return exp_matrix;
        }

        // For matrices, apply softmax to each row independently
        for i in 0..m.rows {
            let row_start = i * m.cols;
            let row_end = (i + 1) * m.cols;
            let row_sum: f64 = exp_matrix.data[row_start..row_end].iter().sum();

            for j in 0..m.cols {
                exp_matrix.data[row_start + j] /= row_sum;
            }
        }
        exp_matrix
    }),
    vector_derivative: Some(|m| {
        // For softmax derivative, we need the Jacobian matrix
        // ∂softmax(x[i])/∂x[j] = softmax(x[i]) * (δ[i,j] - softmax(x[j]))
        let softmax_output = SOFTMAX.apply_vector(m);

        // For column vectors, create a diagonal matrix with derivatives
        if m.cols == 1 {
            let mut result = Matrix::zeros(m.rows, m.rows);
            for i in 0..m.rows {
                let si = softmax_output.data[i];
                result.data[i * m.rows + i] = si * (1.0 - si);
                for j in 0..m.rows {
                    if i != j {
                        result.data[i * m.rows + j] = -si * softmax_output.data[j];
                    }
                }
            }
            return result;
        }

        // For matrices, compute derivatives for each row independently
        let mut result = Matrix::zeros(m.rows * m.cols, m.cols);
        for i in 0..m.rows {
            let row_start = i * m.cols;
            for j in 0..m.cols {
                let si = softmax_output.data[row_start + j];
                let result_row = (i * m.cols + j) * m.cols;
                for k in 0..m.cols {
                    let sk = softmax_output.data[row_start + k];
                    if k == j {
                        result.data[result_row + k] = si * (1.0 - sk);
                    } else {
                        result.data[result_row + k] = -si * sk;
                    }
                }
            }
        }
        result
    }),
    activation_type: ActivationType::Softmax,
};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use matrix::matrix::IntoMatrix;
    use serde_json;

    #[test]
    fn test_sigmoid_activation() {
        let x = 0.0;
        let result = (SIGMOID.function.unwrap())(x);
        assert_relative_eq!(result, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_derivative() {
        let x = 0.5;
        let result = (SIGMOID.derivative.unwrap())(x);
        assert_relative_eq!(result, 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_vector_function() {
        let input = vec![0.0, 1.0, -1.0, 2.0].into_matrix(2, 2);

        // Test vector function
        let result = SIGMOID.apply_vector(&input);
        assert_relative_eq!(result.get(0, 0), 0.5, epsilon = 1e-10);
        assert_relative_eq!(
            result.get(0, 1),
            1.0 / (1.0 + E.powf(-1.0)),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_sigmoid_vector_derivative() {
        let output = vec![0.5, 0.7, 0.3, 0.8].into_matrix(2, 2);
        let derivative = SIGMOID.derivative_vector(&output);
        assert_relative_eq!(derivative.get(0, 0), 0.25, epsilon = 1e-10);
        assert_relative_eq!(derivative.get(0, 1), 0.7 * (1.0 - 0.7), epsilon = 1e-10);
    }

    #[test]
    fn test_activation_fallback() {
        let input = vec![0.0, 1.0].into_matrix(2, 1);

        // SIGMOID should use element-wise fallback
        let result = SIGMOID.apply_vector(&input);
        assert_relative_eq!(result.get(0, 0), 0.5, epsilon = 1e-10);
        assert_relative_eq!(
            result.get(1, 0),
            1.0 / (1.0 + E.powf(-1.0)),
            epsilon = 1e-10
        );
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_matrix_bounds_check() {
        let input = vec![0.0, 1.0].into_matrix(2, 1);
        let _ = input.get(0, 1); // Should panic - accessing column 1 in a 2x1 matrix
    }

    #[test]
    fn test_softmax_scalar_functions_not_used() {
        // Verify that scalar functions are None
        assert!(SOFTMAX.function.is_none());
        assert!(SOFTMAX.derivative.is_none());
    }

    #[test]
    fn test_softmax_activation() {
        // Test with simple input [0.0, 1.0]
        let input = vec![0.0, 1.0].into_matrix(2, 1);
        let result = SOFTMAX.apply_vector(&input);

        // For input [0.0, 1.0]:
        // e^0 = 1.0
        // e^1 ≈ 2.718
        // sum ≈ 3.718
        // Therefore:
        // softmax(0) ≈ 1.0/3.718 ≈ 0.269
        // softmax(1) ≈ 2.718/3.718 ≈ 0.731
        assert_relative_eq!(result.get(0, 0), 0.269, epsilon = 1e-3);
        assert_relative_eq!(result.get(1, 0), 0.731, epsilon = 1e-3);

        // Verify probabilities sum to 1
        let sum: f64 = result.data.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_softmax_derivative() {
        // Test with input [0.0, 1.0]
        let input = vec![0.0, 1.0].into_matrix(2, 1);
        let derivative = SOFTMAX.derivative_vector(&input);

        // For input [0.0, 1.0]:
        // softmax(0) ≈ 0.269
        // softmax(1) ≈ 0.731
        // Therefore:
        // ∂softmax(0)/∂x[0] = 0.269 * (1 - 0.269) ≈ 0.197
        // ∂softmax(0)/∂x[1] = 0.269 * (-0.731) ≈ -0.197
        // ∂softmax(1)/∂x[0] = 0.731 * (-0.269) ≈ -0.197
        // ∂softmax(1)/∂x[1] = 0.731 * (1 - 0.731) ≈ 0.197
        assert_relative_eq!(derivative.get(0, 0), 0.197, epsilon = 1e-3);
        assert_relative_eq!(derivative.get(0, 1), -0.197, epsilon = 1e-3);
        assert_relative_eq!(derivative.get(1, 0), -0.197, epsilon = 1e-3);
        assert_relative_eq!(derivative.get(1, 1), 0.197, epsilon = 1e-3);
    }

    #[test]
    fn test_softmax_2d_matrix() {
        // Test with 2x2 matrix
        let input = vec![0.0, 1.0, -1.0, 2.0].into_matrix(2, 2);
        let result = SOFTMAX.apply_vector(&input);

        // For each row:
        // Row 1: [0.0, 1.0] -> [0.269, 0.731]
        // Row 2: [-1.0, 2.0] -> [0.047, 0.953]
        assert_relative_eq!(result.get(0, 0), 0.269, epsilon = 1e-3);
        assert_relative_eq!(result.get(0, 1), 0.731, epsilon = 1e-3);
        assert_relative_eq!(result.get(1, 0), 0.047, epsilon = 1e-3);
        assert_relative_eq!(result.get(1, 1), 0.953, epsilon = 1e-3);

        // Verify each row sums to 1
        let row1_sum = result.get(0, 0) + result.get(0, 1);
        let row2_sum = result.get(1, 0) + result.get(1, 1);
        assert_relative_eq!(row1_sum, 1.0, epsilon = 1e-10);
        assert_relative_eq!(row2_sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_softmax_2d_matrix_derivative() {
        // Test with 2x2 matrix
        let input = vec![0.0, 1.0, -1.0, 2.0].into_matrix(2, 2);
        let derivative = SOFTMAX.derivative_vector(&input);

        // For each row, the derivative matrix should be:
        // Row 1: [0.197, -0.197; -0.197, 0.197]
        // Row 2: [0.045, -0.045; -0.045, 0.045]
        assert_relative_eq!(derivative.get(0, 0), 0.197, epsilon = 1e-3);
        assert_relative_eq!(derivative.get(0, 1), -0.197, epsilon = 1e-3);
        assert_relative_eq!(derivative.get(1, 0), -0.197, epsilon = 1e-3);
        assert_relative_eq!(derivative.get(1, 1), 0.197, epsilon = 1e-3);
        assert_relative_eq!(derivative.get(2, 0), 0.045, epsilon = 1e-3);
        assert_relative_eq!(derivative.get(2, 1), -0.045, epsilon = 1e-3);
        assert_relative_eq!(derivative.get(3, 0), -0.045, epsilon = 1e-3);
        assert_relative_eq!(derivative.get(3, 1), 0.045, epsilon = 1e-3);
    }

    #[test]
    fn test_activation_serialization() {
        // Test serialization
        let json = serde_json::to_string(&SIGMOID).unwrap();
        assert_eq!(json, "\"Sigmoid\"");
    }

    #[test]
    fn test_activation_deserialization() {
        // Test deserialization
        let activation: Activation = serde_json::from_str("\"Sigmoid\"").unwrap();

        assert!(activation.vector_function.is_some());
        assert!(activation.vector_derivative.is_some());
    }
}
