//! Regularization techniques for neural networks to prevent overfitting.
//!
//! This module provides implementations of common regularization methods:
//! - L1 regularization (Lasso): Adds a penalty term proportional to the absolute value of weights
//! - L2 regularization (Ridge): Adds a penalty term proportional to the square of weights
//!
//! Regularization helps prevent overfitting by discouraging complex models with large weights.

use matrix::Matrix;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Types of regularization available for neural network training.
///
/// Regularization helps prevent overfitting by adding penalty terms to the loss function
/// based on the network's weights.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RegularizationType {
    /// L1 regularization (Lasso) adds a penalty term proportional to the absolute values of weights.
    /// This promotes sparsity by driving some weights exactly to zero, effectively performing
    /// feature selection.
    L1,
    /// L2 regularization (Ridge) adds a penalty term proportional to the squared values of weights.
    /// This prevents any single weight from becoming too large and generally results in small,
    /// diffuse weight values across the network.
    L2,
}

impl RegularizationType {
    pub fn create_regularization(&self) -> Regularization {
        match self {
            RegularizationType::L1 => Regularization::L1(L1::new(1.0)),
            RegularizationType::L2 => Regularization::L2(L2::new(1.0)),
        }
    }
}

/// Concrete regularization implementations with their associated parameters.
///
/// This enum contains the actual regularization implementations that can be applied
/// during network training.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Regularization {
    /// L1 regularization implementation with associated rate parameter
    L1(L1),
    /// L2 regularization implementation with associated rate parameter
    L2(L2),
}

impl Regularization {
    /// Calculates the regularization term to be added to the loss function.
    ///
    /// # Arguments
    /// * `weights` - Slice of weight matrices from the neural network
    /// * `rate` - Regularization rate (lambda) controlling the strength of regularization
    ///
    /// # Returns
    /// The regularization term to be added to the loss function
    pub fn calculate_term(&self, weights: &[Matrix], rate: f64) -> f64 {
        match self {
            Regularization::L1(l1) => l1.calculate_term(weights, rate),
            Regularization::L2(l2) => l2.calculate_term(weights, rate),
        }
    }

    /// Calculates the gradient of the regularization term with respect to weights.
    ///
    /// This gradient is added to the weight updates during backpropagation.
    ///
    /// # Arguments
    /// * `weights` - Weight matrix for which to calculate the gradient
    /// * `rate` - Regularization rate (lambda) controlling the strength of regularization
    ///
    /// # Returns
    /// Matrix containing the gradient of the regularization term
    pub fn calculate_gradient(&self, weights: &Matrix, rate: f64) -> Matrix {
        match self {
            Regularization::L1(l1) => l1.calculate_gradient(weights, rate),
            Regularization::L2(l2) => l2.calculate_gradient(weights, rate),
        }
    }
}

/// L1 (Lasso) regularization implementation.
///
/// L1 regularization adds a penalty term proportional to the absolute values of weights:
/// L1 penalty = λ * Σ|w_i|
///
/// Key characteristics:
/// - Promotes sparsity by driving some weights exactly to zero
/// - Performs implicit feature selection
/// - More robust to outliers than L2
/// - Useful when you suspect many features are irrelevant
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct L1 {
    /// Base regularization rate
    rate: f64,
}

impl L1 {
    /// Creates a new L1 regularization instance with the specified rate.
    ///
    /// # Arguments
    /// * `rate` - Base regularization rate
    ///
    /// # Returns
    /// A new L1 regularization instance
    pub fn new(rate: f64) -> Self {
        L1 { rate }
    }

    /// Calculates the L1 regularization term for the loss function.
    ///
    /// The L1 term is the sum of absolute values of all weights multiplied by the rate:
    /// L1 term = rate * Σ|w_i|
    ///
    /// # Arguments
    /// * `weights` - Slice of weight matrices
    /// * `rate` - Regularization rate (lambda)
    ///
    /// # Returns
    /// The L1 regularization term
    fn calculate_term(&self, weights: &[Matrix], rate: f64) -> f64 {
        weights
            .iter()
            .map(|w| w.0.iter().map(|&x| x.abs()).sum::<f64>())
            .sum::<f64>()
            * rate
    }

    /// Calculates the gradient of the L1 regularization term.
    ///
    /// The gradient of the L1 term is the sign of each weight multiplied by the rate:
    /// ∂L1/∂w_i = rate * sign(w_i)
    ///
    /// # Arguments
    /// * `weights` - Weight matrix
    /// * `rate` - Regularization rate (lambda)
    ///
    /// # Returns
    /// Matrix containing the gradient of the L1 term
    fn calculate_gradient(&self, weights: &Matrix, rate: f64) -> Matrix {
        weights.map(|x| rate * x.signum())
    }
}

/// L2 (Ridge) regularization implementation.
///
/// L2 regularization adds a penalty term proportional to the squared values of weights:
/// L2 penalty = (λ/2) * Σ(w_i²)
///
/// Key characteristics:
/// - Prevents any single weight from becoming too large
/// - Results in small, diffuse weight values across the network
/// - Handles correlated features well
/// - Generally preferred for most neural networks when feature selection is not needed
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct L2 {
    /// Base regularization rate
    rate: f64,
}

impl L2 {
    /// Creates a new L2 regularization instance with the specified rate.
    ///
    /// # Arguments
    /// * `rate` - Base regularization rate
    ///
    /// # Returns
    /// A new L2 regularization instance
    pub fn new(rate: f64) -> Self {
        L2 { rate }
    }

    /// Calculates the L2 regularization term for the loss function.
    ///
    /// The L2 term is the sum of squared values of all weights multiplied by the rate/2:
    /// L2 term = (rate/2) * Σ(w_i²)
    ///
    /// # Arguments
    /// * `weights` - Slice of weight matrices
    /// * `rate` - Regularization rate (lambda)
    ///
    /// # Returns
    /// The L2 regularization term
    fn calculate_term(&self, weights: &[Matrix], rate: f64) -> f64 {
        weights
            .iter()
            .map(|w| w.0.iter().map(|&x| x * x).sum::<f64>())
            .sum::<f64>()
            * (rate / 2.0)
    }

    /// Calculates the gradient of the L2 regularization term.
    ///
    /// The gradient of the L2 term is each weight multiplied by the rate:
    /// ∂L2/∂w_i = rate * w_i
    ///
    /// # Arguments
    /// * `weights` - Weight matrix
    /// * `rate` - Regularization rate (lambda)
    ///
    /// # Returns
    /// Matrix containing the gradient of the L2 term
    fn calculate_gradient(&self, weights: &Matrix, rate: f64) -> Matrix {
        weights * rate
    }
}

impl fmt::Display for Regularization {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Regularization::L1(l1) => write!(f, "L1({})", l1.rate),
            Regularization::L2(l2) => write!(f, "L2({})", l2.rate),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use matrix::IntoMatrix;

    #[test]
    fn test_l1_regularization_term() {
        let weights = vec![
            vec![1.0, -2.0, 3.0, -4.0].into_matrix(2, 2),
            vec![5.0, -6.0, 7.0, -8.0, 9.0, -10.0].into_matrix(2, 3),
        ];

        let rate = 0.01;
        let l1 = L1::new(rate);
        let l1_term = l1.calculate_term(&weights, rate);

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

        let l1 = L1::new(rate);
        let gradient = l1.calculate_gradient(&weights, rate);

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
        let l2 = L2::new(rate);
        let l2_term = l2.calculate_term(&weights, rate);

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

        let l2 = L2::new(rate);
        let gradient = l2.calculate_gradient(&weights, rate);

        // Expected gradient: rate * w
        assert_relative_eq!(gradient.get(0, 0), 0.01, epsilon = 1e-10);
        assert_relative_eq!(gradient.get(0, 1), 0.02, epsilon = 1e-10);
        assert_relative_eq!(gradient.get(1, 0), 0.03, epsilon = 1e-10);
        assert_relative_eq!(gradient.get(1, 1), 0.04, epsilon = 1e-10);
    }

    // Removed serialization tests as they were using removed functionality
}
