//! Autograd-based forward and backward passes for [`Layer`] and [`Network`].
//!
//! neural-net-rs stores activations in column-batch layout `(features, batch)` with bias
//! embedded as the last column of each weight matrix. The autograd engine uses row-batch
//! layout `(batch, features)` with explicit bias vectors. This module converts between the
//! two conventions so forward and backward results match the hand-written path.

use crate::activations::Activation;
use crate::layer::Layer;
use autograd::{Graph, TensorId};
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

/// Weight and bias parameter handles for one layer in the autograd graph.
#[derive(Debug, Clone, Copy)]
pub struct LayerParams {
    /// `(in, out)` weight matrix.
    pub w: TensorId,
    /// `(1, out)` bias row.
    pub b: TensorId,
}

/// Registers current augmented weights as parameter leaves in the graph.
pub fn layer_params_from_weight(graph: &mut Graph, weight: &Matrix) -> LayerParams {
    let (w, b) = split_augmented_weight(weight);
    LayerParams {
        w: graph.leaf(w),
        b: graph.leaf(b),
    }
}

/// Merges autograd `(in, out)` weight and `(1, out)` bias gradients into neural-net-rs
/// augmented layout `(out, in + 1)`.
pub fn merge_augmented_weight_grad(
    w_grad: &Array2<f64>,
    b_grad: &Array2<f64>,
    out_dim: usize,
    in_dim: usize,
) -> Matrix {
    Matrix(Array2::from_shape_fn(
        (out_dim, in_dim + 1),
        |(row, col)| {
            if col < in_dim {
                w_grad[(col, row)]
            } else {
                b_grad[(0, row)]
            }
        },
    ))
}

/// Applies an activation through the autograd graph.
fn apply_activation(graph: &mut Graph, activation: Activation, linear: TensorId) -> TensorId {
    match activation {
        Activation::Sigmoid => graph.sigmoid(linear),
        Activation::ReLU => graph.relu(linear),
        Activation::Softmax => graph.softmax(linear),
    }
}

impl Layer {
    /// Runs the layer forward pass from row-batch input using existing parameter nodes.
    ///
    /// Returns `(linear, output)` where `linear` is the pre-activation value and `output`
    /// applies the configured activation when present.
    pub fn forward_autograd_from_row(
        &self,
        graph: &mut Graph,
        params: &LayerParams,
        input_row: TensorId,
    ) -> (TensorId, TensorId) {
        let mm = graph.matmul(input_row, params.w);
        let linear = graph.add(mm, params.b);
        let output = match self.activation {
            Some(activation) => apply_activation(graph, activation, linear),
            None => linear,
        };
        (linear, output)
    }

    /// Runs the layer forward pass from row-batch input, returning the activated output id.
    pub fn forward_autograd_output_from_row(
        &self,
        graph: &mut Graph,
        params: &LayerParams,
        input_row: TensorId,
    ) -> TensorId {
        self.forward_autograd_from_row(graph, params, input_row).1
    }

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

        let params = layer_params_from_weight(graph, weight);
        let (_, output_id) = self.forward_autograd_from_row(graph, &params, input_id);

        row_batch_to_column_batch(graph.data(output_id))
    }
}

/// Seeds reverse-mode autodiff from the pre-activation output of the final layer.
///
/// neural-net-rs applies `(targets - outputs)` directly to activated outputs without
/// multiplying by the final activation derivative. Seeding the linear node preserves
/// that convention while still backpropagating activation derivatives on hidden layers.
pub fn backward_with_output_delta(
    graph: &mut Graph,
    linear_output: TensorId,
    outputs: &Matrix,
    targets: &Matrix,
) {
    let delta = targets - outputs;
    let seed = column_batch_to_row_batch(&delta);
    graph.backward_with_seed(linear_output, seed);
}

/// Extracts weight gradients for all layers from the graph after backward.
pub fn extract_weight_gradients(
    graph: &Graph,
    params: &[LayerParams],
    weights: &[Matrix],
) -> Vec<Matrix> {
    params
        .iter()
        .zip(weights)
        .map(|(param, weight)| {
            let in_dim = weight.cols() - 1;
            let out_dim = weight.rows();
            merge_augmented_weight_grad(graph.grad(param.w), graph.grad(param.b), out_dim, in_dim)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use matrix::IntoMatrix;
    use ndarray::array;

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

    #[test]
    fn autograd_backward_matches_manual_sigmoid_two_layer() {
        use ndarray::{Axis, array};

        let inputs = Matrix(array![[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]]);
        let targets = Matrix(array![[0.0, 1.0, 1.0, 0.0]]);
        let w1 = Matrix(array![
            [0.1, 0.2, 0.3],
            [0.4, -0.2, 0.5],
            [-0.1, 0.3, 0.2],
            [0.6, -0.4, 0.1]
        ]);
        let w2 = Matrix(array![[0.2, -0.1, 0.4, 0.3, 0.5]]);

        let hidden_layer = Layer::new(4, Some(Activation::Sigmoid));
        let output_layer = Layer::new(1, None);
        let weights = [w1.clone(), w2.clone()];

        let input_aug = {
            let mut with_bias = inputs.0.clone();
            let bias = Array2::ones((1, inputs.cols()));
            with_bias.append(Axis(0), bias.view()).expect("append bias");
            Matrix(with_bias)
        };
        let hidden_out = hidden_layer.process_forward(&w1, &input_aug);
        let hidden_aug = {
            let mut with_bias = hidden_out.0.clone();
            let bias = Array2::ones((1, hidden_out.cols()));
            with_bias.append(Axis(0), bias.view()).expect("append bias");
            Matrix(with_bias)
        };
        let outputs = output_layer.process_forward(&w2, &hidden_aug);

        let mut graph = Graph::new();
        let params: Vec<_> = weights
            .iter()
            .map(|weight| layer_params_from_weight(&mut graph, weight))
            .collect();

        let input_row = graph.leaf(column_batch_to_row_batch(&inputs));
        let hidden = hidden_layer.forward_autograd_output_from_row(&mut graph, &params[0], input_row);
        let (linear_output, _output) =
            output_layer.forward_autograd_from_row(&mut graph, &params[1], hidden);
        backward_with_output_delta(&mut graph, linear_output, &outputs, &targets);
        let autograd = extract_weight_gradients(&graph, &params, &weights);

        let grad_output = Matrix((&outputs.0 - &targets.0) * 2.0);

        let mut deltas = vec![grad_output.clone()];
        let weight_no_bias = w2.0.slice(s![.., ..w2.cols() - 1]);
        let propagated = weight_no_bias.t().dot(&deltas[0].0);
        let derivative = hidden_out.0.mapv(|value| value * (1.0 - value));
        deltas.push(Matrix(propagated * derivative.clone()));

        let mse_w2 = deltas[0].0.dot(&hidden_aug.0.t());
        let _mse_w1 = deltas[1].0.dot(&input_aug.0.t());
        let network_delta_out = &targets.0 - &outputs.0;
        let network_w2 = network_delta_out.dot(&hidden_aug.0.t());
        let network_delta_hidden = weight_no_bias.t().dot(&network_delta_out);
        let network_w1 = (network_delta_hidden * derivative).dot(&input_aug.0.t());

        assert_relative_eq!(autograd[1].get(0, 0), network_w2[(0, 0)], epsilon = 1e-9);
        assert_relative_eq!(autograd[0].get(0, 0), network_w1[(0, 0)], epsilon = 1e-9);
        assert_relative_eq!(autograd[0].get(3, 2), network_w1[(3, 2)], epsilon = 1e-9);
        assert_relative_eq!(autograd[1].get(0, 0), mse_w2[(0, 0)] * -0.5, epsilon = 1e-9);
    }

    #[test]
    fn merge_augmented_weight_grad_matches_manual_layout() {
        let w_grad =
            Array2::from_shape_fn((2, 2), |(row, col)| (row + 1) as f64 + col as f64 * 0.1);
        let b_grad = array![[0.3, 0.6]];
        let merged = merge_augmented_weight_grad(&w_grad, &b_grad, 2, 2);

        assert_relative_eq!(merged.get(0, 0), 1.0);
        assert_relative_eq!(merged.get(1, 0), 1.1);
        assert_relative_eq!(merged.get(0, 1), 2.0);
        assert_relative_eq!(merged.get(1, 1), 2.1);
        assert_relative_eq!(merged.get(0, 2), 0.3);
        assert_relative_eq!(merged.get(1, 2), 0.6);
    }
}
