//! Neural network primitives and operations
//!
//! This module provides fundamental neural network building blocks:
//! - Activation functions (softmax)
//! - Normalization layers (RMSNorm)

use crate::value::{ValueArena, ValueId};

const RMSNORM_EPS: f64 = 1e-5;

/// Softmax activation with numerical stability
///
/// Converts a vector of logits into a probability distribution that sums to 1.
/// Uses the log-sum-exp trick for numerical stability by subtracting the max value.
///
/// # Arguments
/// * `arena` - The ValueArena for creating new values
/// * `logits` - Input logits
///
/// # Returns
/// Probability distribution that sums to 1
///
/// # Example
/// ```ignore
/// let logits = vec![arena.create(1.0), arena.create(2.0), arena.create(3.0)];
/// let probs = softmax(&mut arena, &logits);
/// // probs will sum to approximately 1.0
/// ```
pub fn softmax(arena: &mut ValueArena, logits: &[ValueId]) -> Vec<ValueId> {
    // Find max for numerical stability
    let max_val = logits
        .iter()
        .map(|&id| arena.data(id))
        .fold(f64::NEG_INFINITY, f64::max);

    // Compute exp(x - max)
    let exps: Vec<ValueId> = logits
        .iter()
        .map(|&logit| {
            let shifted = arena.add_scalar(logit, -max_val);
            arena.exp(shifted)
        })
        .collect();

    // Compute sum of exps
    let total = exps
        .iter()
        .fold(arena.create(0.0), |acc, &exp_val| arena.add(acc, exp_val));

    // Divide each exp by total
    exps.into_iter()
        .map(|exp_val| arena.div(exp_val, total))
        .collect()
}

/// RMS (Root Mean Square) normalization
///
/// Normalizes a vector by dividing by the root mean square of its elements.
/// This is a simpler alternative to LayerNorm that doesn't use learnable parameters
/// or centering (mean subtraction).
///
/// Formula: `y_i = x_i / sqrt(mean(x^2) + eps)`
///
/// # Arguments
/// * `arena` - The ValueArena for creating new values
/// * `x` - Input vector
///
/// # Returns
/// Normalized vector with the same shape as input
///
/// # Example
/// ```ignore
/// let x = vec![arena.create(1.0), arena.create(2.0), arena.create(3.0)];
/// let normalized = rmsnorm(&mut arena, &x);
/// ```
pub fn rmsnorm(arena: &mut ValueArena, x: &[ValueId]) -> Vec<ValueId> {
    // Compute mean square: sum(x_i^2) / n
    let mut ms = arena.create(0.0);
    for &xi in x {
        let sq = arena.mul(xi, xi);
        ms = arena.add(ms, sq);
    }
    let n = arena.create(x.len() as f64);
    ms = arena.div(ms, n);

    // Compute scale: (ms + eps)^(-0.5)
    let eps = arena.create(RMSNORM_EPS);
    let ms_eps = arena.add(ms, eps);
    let scale = arena.pow(ms_eps, -0.5);

    x.iter().map(|&xi| arena.mul(xi, scale)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let mut arena = ValueArena::new();
        let logits = vec![arena.create(1.0), arena.create(2.0), arena.create(3.0)];

        let probs = softmax(&mut arena, &logits);

        // Check sum to 1
        let sum: f64 = probs.iter().map(|&p| arena.data(p)).sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Check all positive
        for &p in &probs {
            assert!(arena.data(p) > 0.0);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let mut arena = ValueArena::new();
        // Large values that would overflow without max subtraction
        let logits = vec![
            arena.create(1000.0),
            arena.create(1001.0),
            arena.create(1002.0),
        ];

        let probs = softmax(&mut arena, &logits);

        // Should still sum to 1 and be valid probabilities
        let sum: f64 = probs.iter().map(|&p| arena.data(p)).sum();
        assert!((sum - 1.0).abs() < 1e-10);
        for &p in &probs {
            let val = arena.data(p);
            assert!(val > 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_rmsnorm() {
        let mut arena = ValueArena::new();
        let x = vec![arena.create(1.0), arena.create(2.0), arena.create(3.0)];

        let y = rmsnorm(&mut arena, &x);

        // Verify mean square
        let ms = (1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0) / 3.0;
        let scale = (ms + 1e-5_f64).powf(-0.5);

        assert!((arena.data(y[0]) - 1.0 * scale).abs() < 1e-6);
        assert!((arena.data(y[1]) - 2.0 * scale).abs() < 1e-6);
        assert!((arena.data(y[2]) - 3.0 * scale).abs() < 1e-6);
    }

    #[test]
    fn test_rmsnorm_zero_vector() {
        let mut arena = ValueArena::new();
        let x = vec![arena.create(0.0), arena.create(0.0), arena.create(0.0)];

        let y = rmsnorm(&mut arena, &x);

        // Should not panic or produce NaN due to epsilon
        for &yi in &y {
            let val = arena.data(yi);
            assert!(val.is_finite());
        }
    }
}
