use crate::matrix::Matrix;
use serde::{Deserialize, Serialize};
use std::f64::consts::E;

/// Type of activation function
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ActivationType {
    Sigmoid,
    Softmax,
}

impl ActivationType {
    /// Creates a new activation function instance based on the type
    pub fn create_activation(&self) -> Box<dyn ActivationFunction> {
        match self {
            ActivationType::Sigmoid => Box::new(Sigmoid),
            ActivationType::Softmax => Box::new(Softmax),
        }
    }
}

/// Trait defining the interface for activation functions.
///
/// An activation function must implement both scalar and vector operations,
/// along with their derivatives. The vector operations are optional but
/// recommended for performance.
///
/// # Examples
///
/// ```
/// use neural_network::activations::{ActivationFunction, ActivationType};
///
/// let sigmoid = ActivationType::Sigmoid.create_activation();
/// let result = sigmoid.apply(0.5);
/// let derivative = sigmoid.derivative(result);
/// ```
pub trait ActivationFunction: Send + Sync {
    /// Applies the activation function to a scalar value
    fn apply(&self, x: f64) -> f64;

    /// Applies the derivative of the activation function to a scalar value
    fn derivative(&self, x: f64) -> f64;

    /// Applies the activation function to a matrix
    fn apply_vector(&self, input: &Matrix) -> Matrix;

    /// Applies the derivative of the activation function to a matrix
    fn apply_derivative_vector(&self, input: &Matrix) -> Matrix;

    /// Returns the type of activation function
    fn activation_type(&self) -> ActivationType;
}

/// Standard sigmoid activation function.
///
/// Implements the logistic function: f(x) = 1 / (1 + e^(-x))
/// with derivative: f'(x) = f(x) * (1 - f(x))
#[derive(Debug, Clone, Copy)]
pub struct Sigmoid;

impl ActivationFunction for Sigmoid {
    fn apply(&self, x: f64) -> f64 {
        1.0 / (1.0 + E.powf(-x))
    }

    fn derivative(&self, x: f64) -> f64 {
        x * (1.0 - x)
    }

    fn apply_vector(&self, input: &Matrix) -> Matrix {
        input.map(|x| 1.0 / (1.0 + E.powf(-x)))
    }

    fn apply_derivative_vector(&self, input: &Matrix) -> Matrix {
        input.map(|x| x * (1.0 - x))
    }

    fn activation_type(&self) -> ActivationType {
        ActivationType::Sigmoid
    }
}

/// Standard softmax activation function.
///
/// Implements the softmax function: f(x) = e^x / sum(e^x)
/// with derivative: f'(x) = f(x) * (1 - f(x))
#[derive(Debug, Clone, Copy)]
pub struct Softmax;

impl ActivationFunction for Softmax {
    fn apply(&self, _x: f64) -> f64 {
        panic!("Softmax cannot be applied to scalar values")
    }

    fn derivative(&self, _x: f64) -> f64 {
        panic!("Softmax derivative cannot be applied to scalar values")
    }

    fn apply_vector(&self, input: &Matrix) -> Matrix {
        // For column vectors, treat the entire vector as one group
        if input.cols() == 1 {
            // Find maximum value for numerical stability
            let max_val = input.data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            // Calculate exp(x - max) for each element and sum
            let exp_matrix = input.map(|x| E.powf(x - max_val));
            let sum = exp_matrix.data.sum();

            // Normalize
            exp_matrix.map(|x| x / sum)
        } else {
            // For matrices, apply softmax to each column independently
            let exp_matrix = input.map(|x| E.powf(x));
            let sums = exp_matrix.sum_axis(ndarray::Axis(0));

            // Normalize each column by its sum
            exp_matrix.broadcast_apply(&sums, |x, sum| x / sum)
        }
    }

    fn apply_derivative_vector(&self, input: &Matrix) -> Matrix {
        if input.cols() == 1 {
            let softmax_output = self.apply_vector(input);
            let rows = input.rows();
            let probs = softmax_output
                .data
                .as_slice()
                .expect("Failed to get softmax output data");

            let mut result = Matrix::zeros(rows, rows);

            for i in 0..rows {
                for j in 0..rows {
                    result.data[[i, j]] = if i == j {
                        probs[i] * (1.0 - probs[i])
                    } else {
                        -probs[i] * probs[j]
                    };
                }
            }

            result
        } else {
            panic!("Softmax derivative not implemented for matrices with multiple columns")
        }
    }

    fn activation_type(&self) -> ActivationType {
        ActivationType::Softmax
    }
}

/// Global instances of activation functions
pub const SIGMOID: Sigmoid = Sigmoid;
pub const SOFTMAX: Softmax = Softmax;

/// Wrapper type for serializing activation functions
#[derive(Serialize, Deserialize)]
pub struct ActivationWrapper {
    activation_type: ActivationType,
}

/// Trait for serializing activation functions
pub trait ActivationFunctionSerialize: ActivationFunction {
    /// Serializes an activation function to a JSON string
    fn to_json(&self) -> String {
        serde_json::to_string(&ActivationWrapper {
            activation_type: self.activation_type(),
        })
        .expect("Failed to serialize activation function")
    }

    /// Deserializes an activation function from a JSON string
    fn from_json(json: &str) -> Result<Box<dyn ActivationFunction>, serde_json::Error> {
        let wrapper: ActivationWrapper =
            serde_json::from_str(json).expect("Failed to deserialize activation function");
        match wrapper.activation_type {
            ActivationType::Sigmoid => Ok(Box::new(SIGMOID)),
            ActivationType::Softmax => Ok(Box::new(SOFTMAX)),
        }
    }
}

// Implement the serialization trait for all activation functions
impl ActivationFunctionSerialize for Sigmoid {}
impl ActivationFunctionSerialize for Softmax {}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use matrix::matrix::IntoMatrix;

    #[test]
    fn test_sigmoid_activation() {
        let x = 0.0;
        let result = SIGMOID.apply(x);
        assert_relative_eq!(result, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_derivative() {
        let x = 0.5;
        let result = SIGMOID.derivative(x);
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
        let derivative = SIGMOID.apply_derivative_vector(&output);
        assert_relative_eq!(derivative.get(0, 0), 0.25, epsilon = 1e-10);
        assert_relative_eq!(derivative.get(0, 1), 0.7 * (1.0 - 0.7), epsilon = 1e-10);
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
        let derivative = SOFTMAX.apply_derivative_vector(&input);

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
    #[should_panic(expected = "Softmax cannot be applied to scalar values")]
    fn test_softmax_scalar_activation() {
        SOFTMAX.apply(0.5);
    }

    #[test]
    #[should_panic(expected = "Softmax derivative cannot be applied to scalar values")]
    fn test_softmax_scalar_derivative() {
        SOFTMAX.derivative(0.5);
    }

    #[test]
    fn test_activation_serialization() {
        // Test serialization
        let json = SIGMOID.to_json();
        assert_eq!(json, r#"{"activation_type":"Sigmoid"}"#);
    }

    #[test]
    fn test_activation_deserialization() {
        // Test deserialization
        let json = r#"{"activation_type":"Sigmoid"}"#;
        let activation = <Sigmoid as ActivationFunctionSerialize>::from_json(json);
        assert_eq!(
            activation.unwrap().activation_type(),
            ActivationType::Sigmoid
        );
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Test with large numbers that could cause overflow
        let input = vec![1000.0, 1000.1].into_matrix(2, 1);
        let result = SOFTMAX.apply_vector(&input);

        // Should still sum to 1.0
        let sum: f64 = result.data.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_softmax_zero_input() {
        // Test with all zeros
        let input = vec![0.0, 0.0].into_matrix(2, 1);
        let result = SOFTMAX.apply_vector(&input);

        // Should give equal probabilities
        assert_relative_eq!(result.get(0, 0), 0.5, epsilon = 1e-10);
        assert_relative_eq!(result.get(1, 0), 0.5, epsilon = 1e-10);
    }
}
