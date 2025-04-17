use crate::matrix::Matrix;
use serde::{Deserialize, Serialize};

/// Type of regularization function
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum RegularizationType {
    L1,
    L2,
}

impl RegularizationType {
    /// Creates a new regularization function instance based on the type
    pub fn create_regularization(&self) -> Box<dyn RegularizationFunction> {
        match self {
            RegularizationType::L1 => Box::new(L1),
            RegularizationType::L2 => Box::new(L2),
        }
    }
}

/// Trait defining the interface for regularization functions.
///
/// A regularization function must implement both the regularization term calculation
/// and the gradient calculation for backpropagation.
pub trait RegularizationFunction: Send + Sync {
    /// Calculates the regularization term for a set of weights
    fn calculate_term(&self, weights: &[Matrix], rate: f64) -> f64;

    /// Calculates the regularization gradient for a weight matrix
    fn calculate_gradient(&self, weights: &Matrix, rate: f64) -> Matrix;

    /// Returns the type of regularization function
    fn regularization_type(&self) -> RegularizationType;
}

/// L1 regularization (Lasso)
///
/// Implements L1 regularization: f(w) = rate * sum(|w|)
/// with gradient: f'(w) = rate * sign(w)
#[derive(Debug, Clone, Copy)]
pub struct L1;

impl RegularizationFunction for L1 {
    fn calculate_term(&self, weights: &[Matrix], rate: f64) -> f64 {
        weights
            .iter()
            .map(|w| w.data.iter().map(|&x| x.abs()).sum::<f64>())
            .sum::<f64>()
            * rate
    }

    fn calculate_gradient(&self, weights: &Matrix, rate: f64) -> Matrix {
        weights.map(|x| rate * x.signum())
    }

    fn regularization_type(&self) -> RegularizationType {
        RegularizationType::L1
    }
}

/// L2 regularization (Ridge)
///
/// Implements L2 regularization: f(w) = (rate/2) * sum(w^2)
/// with gradient: f'(w) = rate * w
#[derive(Debug, Clone, Copy)]
pub struct L2;

impl RegularizationFunction for L2 {
    fn calculate_term(&self, weights: &[Matrix], rate: f64) -> f64 {
        weights
            .iter()
            .map(|w| w.data.iter().map(|&x| x * x).sum::<f64>())
            .sum::<f64>()
            * (rate / 2.0)
    }

    fn calculate_gradient(&self, weights: &Matrix, rate: f64) -> Matrix {
        weights * rate
    }

    fn regularization_type(&self) -> RegularizationType {
        RegularizationType::L2
    }
}

/// Global instances of regularization functions
pub const L1_REG: L1 = L1;
pub const L2_REG: L2 = L2;

/// Wrapper type for serializing regularization functions
#[derive(Serialize, Deserialize)]
pub struct RegularizationWrapper {
    regularization_type: RegularizationType,
}

/// Trait for serializing regularization functions
pub trait RegularizationFunctionSerialize: RegularizationFunction {
    /// Serializes a regularization function to a JSON string
    fn to_json(&self) -> String {
        serde_json::to_string(&RegularizationWrapper {
            regularization_type: self.regularization_type(),
        })
        .expect("Failed to serialize regularization function")
    }

    /// Deserializes a regularization function from a JSON string
    fn from_json(json: &str) -> Result<Box<dyn RegularizationFunction>, serde_json::Error> {
        let wrapper: RegularizationWrapper = serde_json::from_str(json)?;
        match wrapper.regularization_type {
            RegularizationType::L1 => Ok(Box::new(L1_REG)),
            RegularizationType::L2 => Ok(Box::new(L2_REG)),
        }
    }
}

// Implement the serialization trait for all regularization functions
impl RegularizationFunctionSerialize for L1 {}
impl RegularizationFunctionSerialize for L2 {}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use matrix::matrix::IntoMatrix;

    #[test]
    fn test_l1_regularization_term() {
        let weights = vec![
            vec![1.0, -2.0, 3.0, -4.0].into_matrix(2, 2),
            vec![5.0, -6.0, 7.0, -8.0, 9.0, -10.0].into_matrix(2, 3),
        ];

        let rate = 0.01;
        let l1_term = L1_REG.calculate_term(&weights, rate);

        // Calculate expected value manually:
        // Sum of absolute values of first matrix: 1 + 2 + 3 + 4 = 10
        // Sum of absolute values of second matrix: 5 + 6 + 7 + 8 + 9 + 10 = 45
        // Total sum: 10 + 45 = 55
        // L1 term: 55 * 0.01 = 0.55

        assert_relative_eq!(l1_term, 0.55, epsilon = 1e-10);
    }

    #[test]
    fn test_l1_regularization_gradient() {
        let weights = vec![1.0, -2.0, 3.0, -4.0].into_matrix(2, 2);
        let rate = 0.01;

        let gradient = L1_REG.calculate_gradient(&weights, rate);

        // Expected gradient: rate * sign(w)
        assert_relative_eq!(gradient.get(0, 0), 0.01, epsilon = 1e-10);
        assert_relative_eq!(gradient.get(0, 1), -0.01, epsilon = 1e-10);
        assert_relative_eq!(gradient.get(1, 0), 0.01, epsilon = 1e-10);
        assert_relative_eq!(gradient.get(1, 1), -0.01, epsilon = 1e-10);
    }

    #[test]
    fn test_l2_regularization_term() {
        let weights = vec![
            vec![1.0, 2.0, 3.0, 4.0].into_matrix(2, 2),
            vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0].into_matrix(2, 3),
        ];

        let rate = 0.01;
        let l2_term = L2_REG.calculate_term(&weights, rate);

        // Calculate expected value manually:
        // Sum of squares of first matrix: 1² + 2² + 3² + 4² = 1 + 4 + 9 + 16 = 30
        // Sum of squares of second matrix: 5² + 6² + 7² + 8² + 9² + 10² = 25 + 36 + 49 + 64 + 81 + 100 = 355
        // Total sum: 30 + 355 = 385
        // L2 term: 385 * (0.01 / 2) = 1.925

        assert_relative_eq!(l2_term, 1.925, epsilon = 1e-10);
    }

    #[test]
    fn test_l2_regularization_gradient() {
        let weights = vec![1.0, 2.0, 3.0, 4.0].into_matrix(2, 2);
        let rate = 0.01;

        let gradient = L2_REG.calculate_gradient(&weights, rate);

        // Expected gradient: rate * w
        assert_relative_eq!(gradient.get(0, 0), 0.01, epsilon = 1e-10);
        assert_relative_eq!(gradient.get(0, 1), 0.02, epsilon = 1e-10);
        assert_relative_eq!(gradient.get(1, 0), 0.03, epsilon = 1e-10);
        assert_relative_eq!(gradient.get(1, 1), 0.04, epsilon = 1e-10);
    }

    #[test]
    fn test_regularization_serialization() {
        // Test serialization
        let json = L1_REG.to_json();
        assert_eq!(json, r#"{"regularization_type":"L1"}"#);
    }

    #[test]
    fn test_regularization_deserialization() {
        // Test deserialization
        let json = r#"{"regularization_type":"L1"}"#;
        let reg = <L1 as RegularizationFunctionSerialize>::from_json(json);
        assert_eq!(reg.unwrap().regularization_type(), RegularizationType::L1);
    }
}
