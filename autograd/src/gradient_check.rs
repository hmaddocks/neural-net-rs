//! Finite-difference gradient checks for validating autograd backward rules.
//!
//! These utilities compare reverse-mode gradients against central-difference
//! approximations. They are the regression oracle for every op in Phase 1.

//! Finite-difference gradient checks for validating autograd backward rules.
//!
//! These utilities compare reverse-mode gradients against central-difference
//! approximations. They are the regression oracle for every op in Phase 1.
//!
//! # Example
//!
//! ```
//! use autograd::{Graph, central_difference, gradients_match, DEFAULT_EPSILON, DEFAULT_TOLERANCE};
//! use ndarray::array;
//!
//! let input = array![[2.0, 3.0]];
//! let mut graph = Graph::new();
//! let x = graph.leaf(input.clone());
//! let y = graph.relu(x);
//! graph.backward(y);
//! let autograd = graph.grad(x).clone();
//!
//! let numeric = central_difference(
//!     |value| value.mapv(|v| v.max(0.0)).sum(),
//!     &input,
//!     DEFAULT_EPSILON,
//! );
//! assert!(gradients_match(&autograd, &numeric, DEFAULT_TOLERANCE));
//! ```

use ndarray::Array2;

/// Default perturbation for central-difference gradient checks.
pub const DEFAULT_EPSILON: f64 = 1e-6;

/// Default absolute tolerance when comparing autograd and numerical gradients.
pub const DEFAULT_TOLERANCE: f64 = 1e-4;

/// Computes a central-difference gradient for a scalar loss `f(input)`.
///
/// Perturbs each element of `input` by `±epsilon` and evaluates `loss` to estimate
/// `∂loss/∂input`. Slower than autograd but useful as an independent correctness check.
///
/// # Example
///
/// ```
/// use autograd::central_difference;
/// use ndarray::array;
///
/// let input = array![[1.0, 2.0]];
/// let grad = central_difference(|x| x.mapv(|v| v * v).sum(), &input, 1e-7);
/// assert!((grad[(0, 0)] - 2.0).abs() < 1e-5);
/// ```
pub fn central_difference<F>(mut loss: F, input: &Array2<f64>, epsilon: f64) -> Array2<f64>
where
    F: FnMut(&Array2<f64>) -> f64,
{
    let mut gradient = Array2::zeros(input.raw_dim());
    for row in 0..input.nrows() {
        for col in 0..input.ncols() {
            let mut plus = input.clone();
            let mut minus = input.clone();
            plus[(row, col)] += epsilon;
            minus[(row, col)] -= epsilon;
            gradient[(row, col)] = (loss(&plus) - loss(&minus)) / (2.0 * epsilon);
        }
    }
    gradient
}

/// Returns whether every element of `autograd` and `numerical` agree within `tolerance`.
///
/// Shapes must match. Comparison is elementwise absolute difference, not relative error.
pub fn gradients_match(autograd: &Array2<f64>, numerical: &Array2<f64>, tolerance: f64) -> bool {
    if autograd.dim() != numerical.dim() {
        return false;
    }

    autograd
        .iter()
        .zip(numerical.iter())
        .all(|(left, right)| (left - right).abs() <= tolerance)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array2, array};

    use super::{DEFAULT_EPSILON, DEFAULT_TOLERANCE, central_difference, gradients_match};
    use crate::graph::Graph;

    fn loss_sum(output: &Array2<f64>) -> f64 {
        output.sum()
    }

    fn autograd_grad<F>(input: Array2<f64>, build: F) -> Array2<f64>
    where
        F: FnOnce(&mut Graph, TensorId) -> TensorId,
    {
        let mut graph = Graph::new();
        let leaf = graph.leaf(input);
        let output = build(&mut graph, leaf);
        graph.backward(output);
        graph.grad(leaf).clone()
    }

    fn autograd_grad_binary<F>(
        left: Array2<f64>,
        right: Array2<f64>,
        build: F,
    ) -> (Array2<f64>, Array2<f64>)
    where
        F: FnOnce(&mut Graph, TensorId, TensorId) -> TensorId,
    {
        let mut graph = Graph::new();
        let left_id = graph.leaf(left);
        let right_id = graph.leaf(right);
        let output = build(&mut graph, left_id, right_id);
        graph.backward(output);
        (graph.grad(left_id).clone(), graph.grad(right_id).clone())
    }

    use crate::graph::TensorId;

    #[test]
    fn matmul_matches_central_difference() {
        let left = array![[0.3, -0.2], [0.5, 0.1]];
        let right = array![[1.0, 0.4], [-0.3, 0.8]];

        let (autograd_left, autograd_right) =
            autograd_grad_binary(left.clone(), right.clone(), |graph, a, b| {
                graph.matmul(a, b)
            });

        let numeric_left =
            central_difference(|value| loss_sum(&value.dot(&right)), &left, DEFAULT_EPSILON);
        let numeric_right =
            central_difference(|value| loss_sum(&left.dot(value)), &right, DEFAULT_EPSILON);

        assert!(gradients_match(
            &autograd_left,
            &numeric_left,
            DEFAULT_TOLERANCE
        ));
        assert!(gradients_match(
            &autograd_right,
            &numeric_right,
            DEFAULT_TOLERANCE
        ));
    }

    #[test]
    fn add_with_broadcast_matches_central_difference() {
        let left = array![[1.0, -2.0]];
        let right = array![[0.5], [1.5]];

        let (autograd_left, autograd_right) =
            autograd_grad_binary(left.clone(), right.clone(), |graph, a, b| graph.add(a, b));

        let numeric_left = central_difference(
            |value| loss_sum(&crate::broadcast::broadcast_add(value, &right).expect("broadcast")),
            &left,
            DEFAULT_EPSILON,
        );
        let numeric_right = central_difference(
            |value| loss_sum(&crate::broadcast::broadcast_add(&left, value).expect("broadcast")),
            &right,
            DEFAULT_EPSILON,
        );

        assert!(gradients_match(
            &autograd_left,
            &numeric_left,
            DEFAULT_TOLERANCE
        ));
        assert!(gradients_match(
            &autograd_right,
            &numeric_right,
            DEFAULT_TOLERANCE
        ));
    }

    #[test]
    fn mul_with_broadcast_matches_central_difference() {
        let left = array![[2.0], [3.0]];
        let right = array![[4.0, -1.0]];

        let (autograd_left, autograd_right) =
            autograd_grad_binary(left.clone(), right.clone(), |graph, a, b| graph.mul(a, b));

        let numeric_left = central_difference(
            |value| loss_sum(&crate::broadcast::broadcast_mul(value, &right).expect("broadcast")),
            &left,
            DEFAULT_EPSILON,
        );
        let numeric_right = central_difference(
            |value| loss_sum(&crate::broadcast::broadcast_mul(&left, value).expect("broadcast")),
            &right,
            DEFAULT_EPSILON,
        );

        assert!(gradients_match(
            &autograd_left,
            &numeric_left,
            DEFAULT_TOLERANCE
        ));
        assert!(gradients_match(
            &autograd_right,
            &numeric_right,
            DEFAULT_TOLERANCE
        ));
    }

    #[test]
    fn relu_matches_central_difference() {
        let input = array![[-1.0, 0.5], [2.0, -0.25]];
        let autograd = autograd_grad(input.clone(), |graph, x| graph.relu(x));
        let numeric = central_difference(
            |value| value.mapv(|element| element.max(0.0)).sum(),
            &input,
            DEFAULT_EPSILON,
        );

        assert!(gradients_match(&autograd, &numeric, DEFAULT_TOLERANCE));
    }

    #[test]
    fn sigmoid_matches_central_difference() {
        let input = array![[-1.0, 0.5], [2.0, -0.25]];
        let autograd = autograd_grad(input.clone(), |graph, x| graph.sigmoid(x));
        let numeric = central_difference(
            |value| value.mapv(|element| 1.0 / (1.0 + (-element).exp())).sum(),
            &input,
            DEFAULT_EPSILON,
        );

        assert!(gradients_match(&autograd, &numeric, DEFAULT_TOLERANCE));
    }

    #[test]
    fn log_matches_central_difference() {
        let input = array![[1.0, 2.0], [3.0, 4.5]];
        let autograd = autograd_grad(input.clone(), |graph, x| graph.log(x));
        let numeric =
            central_difference(|value| value.mapv(f64::ln).sum(), &input, DEFAULT_EPSILON);

        assert!(gradients_match(&autograd, &numeric, DEFAULT_TOLERANCE));
    }

    #[test]
    fn pow_matches_central_difference() {
        let input = array![[1.5, -2.0], [0.5, 3.0]];
        let autograd = autograd_grad(input.clone(), |graph, x| graph.pow(x, 3.0));
        let numeric = central_difference(
            |value| value.mapv(|element| element.powf(3.0)).sum(),
            &input,
            DEFAULT_EPSILON,
        );

        assert!(gradients_match(&autograd, &numeric, DEFAULT_TOLERANCE));
    }

    #[test]
    fn softmax_matches_central_difference() {
        let input = array![[1.0, 2.0, 0.5], [-1.0, 0.0, 2.0]];
        let autograd = autograd_grad(input.clone(), |graph, x| graph.softmax(x));
        let numeric = central_difference(
            |value| {
                let mut graph = Graph::new();
                let leaf = graph.leaf(value.clone());
                let output = graph.softmax(leaf);
                graph.data(output).sum()
            },
            &input,
            DEFAULT_EPSILON,
        );

        assert!(gradients_match(&autograd, &numeric, DEFAULT_TOLERANCE));
    }

    #[test]
    fn rmsnorm_matches_central_difference() {
        let input = array![[1.0, 2.0, -1.0], [0.5, -0.5, 1.5]];
        let autograd = autograd_grad(input.clone(), |graph, x| graph.rmsnorm(x));
        let numeric = central_difference(
            |value| {
                let mut graph = Graph::new();
                let leaf = graph.leaf(value.clone());
                let output = graph.rmsnorm(leaf);
                graph.data(output).sum()
            },
            &input,
            DEFAULT_EPSILON,
        );

        assert!(gradients_match(&autograd, &numeric, DEFAULT_TOLERANCE));
    }

    #[test]
    fn central_difference_is_exact_for_quadratic() {
        let input = array![[2.0, 3.0]];
        let numeric = central_difference(|value| value.mapv(|x| x * x).sum(), &input, 1e-7);

        assert_relative_eq!(numeric[(0, 0)], 4.0, epsilon = 1e-5);
        assert_relative_eq!(numeric[(0, 1)], 6.0, epsilon = 1e-5);
    }
}
