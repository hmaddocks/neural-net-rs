//! Autograd-based forward pass for [`Layer`] and [`Network`].
//!
//! neural-net-rs stores activations in column-batch layout `(features, batch)` with bias
//! embedded as the last column of each weight matrix. The autograd engine uses row-batch
//! layout `(batch, features)` with explicit bias vectors. This module converts between the
//! two conventions so forward results match the hand-written path.

use crate::activations::Activation;
use crate::layer::Layer;
use autograd::Graph;
use matrix::Matrix;
use ndarray::{Array2, s};

/// Converts column-batch layout `(features, batch)` to row-batch `(batch, features)`.
pub fn column_batch_to_row_batch(input: &Matrix) -> Array2<f64> {
    input.0.t().to_owned()
}

/// Converts row-batch layout `(batch, features)` to column-batch `(features, batch)`.
pub fn row_batch_to_column_batch(output: &Array2<f64>) -> Matrix {
    Matrix(output.t().to_owned())
}

/// Splits an augmented weight matrix `(out, in + 1)` into autograd `(in, out)` weights
/// and a `(1, out)` bias row.
pub fn split_augmented_weight(weight: &Matrix) -> (Array2<f64>, Array2<f64>) {
    let out_dim = weight.rows();
    let in_dim = weight.cols().saturating_sub(1);
    assert_eq!(
        weight.cols(),
        in_dim + 1,
        "weight matrix must include a bias column"
    );

    let w = weight.0.slice(s![.., ..in_dim]).t().to_owned();
    let b = Array2::from_shape_fn((1, out_dim), |(_, col)| weight.get(col, in_dim));
    (w, b)
}

/// Applies an activation through the autograd graph.
fn apply_activation(
    graph: &mut Graph,
    activation: Activation,
    linear: autograd::TensorId,
) -> autograd::TensorId {
    match activation {
        Activation::Sigmoid => graph.sigmoid(linear),
        Activation::ReLU => graph.relu(linear),
        Activation::Softmax => graph.softmax(linear),
    }
}

impl Layer {
    /// Runs the layer forward pass through autograd and returns column-batch output.
    ///
    /// `input_with_bias` uses neural-net-rs layout `(features + 1, batch)` with a final
    /// row of ones. `weight` is `(out, in + 1)` including the bias column. The returned
    /// matrix matches [`Layer::process_forward`] for the same inputs.
    pub fn process_forward_autograd(
        &self,
        graph: &mut Graph,
        weight: &Matrix,
        input_with_bias: &Matrix,
    ) -> Matrix {
        let in_dim = weight.cols() - 1;
        assert_eq!(
            input_with_bias.rows(),
            in_dim + 1,
            "input must include a bias row"
        );

        let features = Matrix(input_with_bias.0.slice(s![..in_dim, ..]).to_owned());
        let input_row = column_batch_to_row_batch(&features);
        let input_id = graph.leaf(input_row);

        let (w, b) = split_augmented_weight(weight);
        let w_id = graph.leaf(w);
        let b_id = graph.leaf(b);

        let mm = graph.matmul(input_id, w_id);
        let linear = graph.add(mm, b_id);
        let output_id = match self.activation {
            Some(activation) => apply_activation(graph, activation, linear),
            None => linear,
        };

        row_batch_to_column_batch(graph.data(output_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use matrix::IntoMatrix;

    #[test]
    fn split_augmented_weight_separates_bias_column() {
        let weight = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6].into_matrix(2, 3);
        let (w, b) = split_augmented_weight(&weight);

        assert_eq!(w.shape(), &[2, 2]);
        assert_eq!(b.shape(), &[1, 2]);
        assert_relative_eq!(w[(0, 0)], 0.1);
        assert_relative_eq!(w[(1, 0)], 0.2);
        assert_relative_eq!(w[(0, 1)], 0.4);
        assert_relative_eq!(w[(1, 1)], 0.5);
        assert_relative_eq!(b[(0, 0)], 0.3);
        assert_relative_eq!(b[(0, 1)], 0.6);
    }

    #[test]
    fn column_and_row_batch_round_trip() {
        let input = vec![0.5, 0.6, 0.7, 0.8].into_matrix(2, 2);
        let row = column_batch_to_row_batch(&input);
        let back = row_batch_to_column_batch(&row);

        assert_eq!(back.rows(), input.rows());
        assert_eq!(back.cols(), input.cols());
        for row_idx in 0..input.rows() {
            for col_idx in 0..input.cols() {
                assert_relative_eq!(
                    back.get(row_idx, col_idx),
                    input.get(row_idx, col_idx),
                    epsilon = 1e-12
                );
            }
        }
    }

    #[test]
    fn process_forward_autograd_matches_manual_sigmoid() {
        let layer = Layer::new(2, Some(Activation::Sigmoid));
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6].into_matrix(2, 3);
        let inputs = vec![0.5, 0.6, 1.0].into_matrix(3, 1);

        let manual = layer.process_forward(&weights, &inputs);

        let mut graph = Graph::new();
        let autograd = layer.process_forward_autograd(&mut graph, &weights, &inputs);

        assert_eq!(manual.rows(), autograd.rows());
        assert_eq!(manual.cols(), autograd.cols());
        for row_idx in 0..manual.rows() {
            for col_idx in 0..manual.cols() {
                assert_relative_eq!(
                    manual.get(row_idx, col_idx),
                    autograd.get(row_idx, col_idx),
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn process_forward_autograd_matches_manual_relu() {
        let layer = Layer::new(2, Some(Activation::ReLU));
        let weights = vec![-0.5, 0.2, 0.3, 0.4, -0.1, 0.6].into_matrix(2, 3);
        let inputs = vec![0.5, 0.6, -0.2, 0.3, 1.0, 1.0].into_matrix(3, 2);

        let manual = layer.process_forward(&weights, &inputs);

        let mut graph = Graph::new();
        let autograd = layer.process_forward_autograd(&mut graph, &weights, &inputs);

        for row_idx in 0..manual.rows() {
            for col_idx in 0..manual.cols() {
                assert_relative_eq!(
                    manual.get(row_idx, col_idx),
                    autograd.get(row_idx, col_idx),
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn process_forward_autograd_matches_manual_softmax() {
        let layer = Layer::new(3, Some(Activation::Softmax));
        let weights = Matrix::new(
            3,
            4,
            vec![1.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.2, 0.0, 0.0, 1.0, 0.3],
        );
        let inputs = Matrix::new(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0]);

        let manual = layer.process_forward(&weights, &inputs);

        let mut graph = Graph::new();
        let autograd = layer.process_forward_autograd(&mut graph, &weights, &inputs);

        for col_idx in 0..manual.cols() {
            let manual_sum: f64 = (0..manual.rows()).map(|row| manual.get(row, col_idx)).sum();
            let autograd_sum: f64 = (0..autograd.rows())
                .map(|row| autograd.get(row, col_idx))
                .sum();
            assert_relative_eq!(manual_sum, 1.0, epsilon = 1e-10);
            assert_relative_eq!(autograd_sum, 1.0, epsilon = 1e-10);
        }

        for row_idx in 0..manual.rows() {
            for col_idx in 0..manual.cols() {
                assert_relative_eq!(
                    manual.get(row_idx, col_idx),
                    autograd.get(row_idx, col_idx),
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn process_forward_autograd_matches_manual_linear() {
        let layer = Layer::new(2, None);
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6].into_matrix(2, 3);
        let inputs = vec![0.5, 0.6, 1.0].into_matrix(3, 1);

        let manual = layer.process_forward(&weights, &inputs);

        let mut graph = Graph::new();
        let autograd = layer.process_forward_autograd(&mut graph, &weights, &inputs);

        assert_relative_eq!(manual.get(0, 0), autograd.get(0, 0), epsilon = 1e-10);
        assert_relative_eq!(manual.get(1, 0), autograd.get(1, 0), epsilon = 1e-10);
    }
}
