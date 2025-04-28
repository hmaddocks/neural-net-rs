use matrix::Matrix;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RegularizationType {
    L1,
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Regularization {
    L1(L1),
    L2(L2),
}

impl Regularization {
    pub fn calculate_term(&self, weights: &[Matrix], rate: f64) -> f64 {
        match self {
            Regularization::L1(l1) => l1.calculate_term(weights, rate),
            Regularization::L2(l2) => l2.calculate_term(weights, rate),
        }
    }

    pub fn calculate_gradient(&self, weights: &Matrix, rate: f64) -> Matrix {
        match self {
            Regularization::L1(l1) => l1.calculate_gradient(weights, rate),
            Regularization::L2(l2) => l2.calculate_gradient(weights, rate),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct L1 {
    rate: f64,
}

impl L1 {
    pub fn new(rate: f64) -> Self {
        L1 { rate }
    }

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
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct L2 {
    rate: f64,
}

impl L2 {
    pub fn new(rate: f64) -> Self {
        L2 { rate }
    }

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
