use matrix::matrix::Matrix;
use serde::{Deserialize, Serialize};
use std::f64::consts::E;

/// Type of activation function
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationType {
    Sigmoid,
    SigmoidVector,
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
/// let result = (SIGMOID.function)(0.5);
/// let derivative = (SIGMOID.derivative)(result);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Activation {
    /// The scalar activation function
    pub function: fn(f64) -> f64,
    /// The derivative of the scalar activation function
    pub derivative: fn(f64) -> f64,
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
            ActivationType::SigmoidVector => SIGMOID_VECTOR,
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
            None => input.map(self.function),
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
            None => input.map(self.derivative),
        }
    }
}

/// Standard sigmoid activation function.
///
/// Implements the logistic function: f(x) = 1 / (1 + e^(-x))
/// with derivative: f'(x) = f(x) * (1 - f(x))
pub const SIGMOID: Activation = Activation {
    function: |x| 1.0 / (1.0 + E.powf(-x)),
    derivative: |x| x * (1.0 - x),
    vector_function: None,
    vector_derivative: None,
    activation_type: ActivationType::Sigmoid,
};

/// Vectorized version of the sigmoid activation function.
///
/// Provides optimized implementations for matrix operations.
pub const SIGMOID_VECTOR: Activation = Activation {
    function: |x| 1.0 / (1.0 + E.powf(-x)),
    derivative: |x| x * (1.0 - x),
    vector_function: Some(|m| m.map(|x| 1.0 / (1.0 + E.powf(-x)))),
    vector_derivative: Some(|m| m.map(|x| x * (1.0 - x))),
    activation_type: ActivationType::SigmoidVector,
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
        let result = (SIGMOID.function)(x);
        assert_relative_eq!(result, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_derivative() {
        let x = 0.5;
        let result = (SIGMOID.derivative)(x);
        assert_relative_eq!(result, 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_sigmoid_vector_operations() {
        let input = vec![0.0, 1.0, -1.0, 2.0].into_matrix(2, 2);

        // Test vector function
        let result = SIGMOID_VECTOR.apply_vector(&input);
        assert_relative_eq!(result.get(0, 0), 0.5, epsilon = 1e-10);
        assert_relative_eq!(
            result.get(0, 1),
            1.0 / (1.0 + E.powf(-1.0)),
            epsilon = 1e-10
        );

        // Test vector derivative
        let output = vec![0.5, 0.7, 0.3, 0.8].into_matrix(2, 2);
        let derivative = SIGMOID_VECTOR.derivative_vector(&output);
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
    fn test_activation_serialization() {
        // Test serialization
        let json = serde_json::to_string(&SIGMOID).unwrap();
        assert_eq!(json, "\"Sigmoid\"");

        let json = serde_json::to_string(&SIGMOID_VECTOR).unwrap();
        assert_eq!(json, "\"SigmoidVector\"");

        // Test deserialization
        let activation: Activation = serde_json::from_str("\"Sigmoid\"").unwrap();
        assert!(activation.vector_function.is_none());
        assert_relative_eq!((activation.function)(0.0), 0.5, epsilon = 1e-10);

        let activation: Activation = serde_json::from_str("\"SigmoidVector\"").unwrap();
        assert!(activation.vector_function.is_some());
        assert_relative_eq!((activation.function)(0.0), 0.5, epsilon = 1e-10);
    }
}
