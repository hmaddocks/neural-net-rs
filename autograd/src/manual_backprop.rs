//! Hand-written backpropagation matching the formulas in `neural_network`.
//!
//! The recurrence here mirrors `Network::accumulate_gradients` and
//! `Layer::compute_hidden_delta`, adapted from neural-net-rs's column-batch layout with
//! augmented bias weights to autograd's row-batch layout with explicit bias tensors.

#![allow(dead_code)] // reference implementations exercised by this module's tests

use ndarray::{Array2, Axis};

/// Parameter gradients for the XOR ReLU MLP (`2 → H → 1`) under `sum((y - t)^2)` loss.
pub fn relu_xor_parameter_grads(
    inputs: &Array2<f64>,
    targets: &Array2<f64>,
    w1: &Array2<f64>,
    b1: &Array2<f64>,
    w2: &Array2<f64>,
    b2: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let hidden_pre = inputs.dot(w1) + b1;
    let relu_mask = hidden_pre.mapv(|value| if value > 0.0 { 1.0 } else { 0.0 });
    let hidden = hidden_pre.mapv(|value| value.max(0.0));
    let predictions = hidden.dot(w2) + b2;

    let grad_predictions = (predictions - targets) * 2.0;
    let grad_w2 = hidden.t().dot(&grad_predictions);
    let grad_b2 = grad_predictions.sum_axis(Axis(0)).insert_axis(Axis(0));
    let grad_hidden = grad_predictions.dot(&w2.t());
    let grad_hidden_pre = grad_hidden * relu_mask;
    let grad_w1 = inputs.t().dot(&grad_hidden_pre);
    let grad_b1 = grad_hidden_pre.sum_axis(Axis(0)).insert_axis(Axis(0));

    (grad_w1, grad_b1, grad_w2, grad_b2)
}

/// Weight gradients for a sigmoid MLP in neural-net-rs column layout under MSE loss.
///
/// `grad_output` is `2 * (outputs - targets)` with the same column layout as `outputs`.
/// Each `weights[i]` has shape `(next_layer, current_layer + 1)` including the bias column.
pub fn sigmoid_network_weight_grads(
    grad_output: &Array2<f64>,
    weights: &[Array2<f64>],
    layer_outputs: &[Array2<f64>],
) -> Vec<Array2<f64>> {
    let mut deltas = Vec::with_capacity(weights.len());
    deltas.push(grad_output.clone());

    let mut previous_delta = deltas[0].clone();
    for layer in (0..weights.len() - 1).rev() {
        let next_weight = &weights[layer + 1];
        let current_output = &layer_outputs[layer + 1];
        let weight_no_bias = next_weight.slice(ndarray::s![.., ..next_weight.ncols() - 1]);
        let propagated = weight_no_bias.t().dot(&previous_delta);
        let derivative = current_output.mapv(|value| value * (1.0 - value));
        previous_delta = propagated * derivative;
        deltas.push(previous_delta.clone());
    }

    (0..weights.len())
        .map(|layer| {
            let delta = &deltas[weights.len() - 1 - layer];
            let mut input_with_bias = layer_outputs[layer].clone();
            let bias_row = Array2::ones((1, input_with_bias.ncols()));
            input_with_bias
                .append(Axis(0), bias_row.view())
                .expect("bias row should append");
            delta.dot(&input_with_bias.t())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, Axis, array};

    use super::{relu_xor_parameter_grads, sigmoid_network_weight_grads};
    use crate::gradient_check::{DEFAULT_TOLERANCE, gradients_match};
    use crate::graph::Graph;
    use crate::xor::{INPUTS, TARGETS, XorMlp};

    #[test]
    fn relu_xor_autograd_matches_manual_backprop() {
        let mut model = XorMlp::from_fixed_weights();
        let inputs = Array2::from_shape_fn((4, 2), |(row, col)| INPUTS[row][col]);
        let targets = Array2::from_shape_fn((4, 1), |(row, _)| TARGETS[row]);
        let neg_targets = Array2::from_shape_fn((4, 1), |(row, _)| -TARGETS[row]);

        let autograd = model.parameter_gradients(&inputs, &neg_targets);
        let manual = relu_xor_parameter_grads(
            &inputs,
            &targets,
            model.weights().0,
            model.weights().1,
            model.weights().2,
            model.weights().3,
        );

        assert!(gradients_match(&autograd.0, &manual.0, DEFAULT_TOLERANCE));
        assert!(gradients_match(&autograd.1, &manual.1, DEFAULT_TOLERANCE));
        assert!(gradients_match(&autograd.2, &manual.2, DEFAULT_TOLERANCE));
        assert!(gradients_match(&autograd.3, &manual.3, DEFAULT_TOLERANCE));
    }

    #[test]
    fn sigmoid_column_layout_matches_neural_network_formulas() {
        let inputs = array![[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]];
        let targets = array![[0.0, 1.0, 1.0, 0.0]];
        let w1 = array![
            [0.1, 0.2, 0.3],
            [0.4, -0.2, 0.5],
            [-0.1, 0.3, 0.2],
            [0.6, -0.4, 0.1]
        ];
        let w2 = array![[0.2, -0.1, 0.4, 0.3, 0.5]];

        let input_aug = {
            let mut with_bias = inputs.clone();
            let bias = Array2::ones((1, inputs.ncols()));
            with_bias.append(Axis(0), bias.view()).expect("append bias");
            with_bias
        };
        let hidden_pre = w1.dot(&input_aug);
        let hidden = hidden_pre.mapv(|value: f64| 1.0 / (1.0 + (-value).exp()));
        let hidden_aug = {
            let mut with_bias = hidden.clone();
            let bias = Array2::ones((1, hidden.ncols()));
            with_bias.append(Axis(0), bias.view()).expect("append bias");
            with_bias
        };
        let outputs = w2.dot(&hidden_aug);

        let layer_outputs = vec![inputs.clone(), hidden, outputs.clone()];
        let grad_output = 2.0 * (&outputs - &targets);
        let manual = sigmoid_network_weight_grads(&grad_output, &[w1, w2], &layer_outputs);

        let mut graph = Graph::new();
        let x = graph.leaf(array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
        let w1_ag = graph.leaf(array![[0.1, 0.4, -0.1, 0.6], [0.2, -0.2, 0.3, -0.4]]);
        let b1 = graph.leaf(array![[0.3, 0.5, 0.2, 0.1]]);
        let w2 = graph.leaf(array![[0.2], [-0.1], [0.4], [0.3]]);
        let b2 = graph.leaf(array![[0.5]]);

        let mm1 = graph.matmul(x, w1_ag);
        let linear1 = graph.add(mm1, b1);
        let hidden = graph.sigmoid(linear1);
        let mm2 = graph.matmul(hidden, w2);
        let predictions = graph.add(mm2, b2);
        let neg_targets = graph.leaf(array![[0.0], [-1.0], [-1.0], [0.0]]);
        let diff = graph.add(predictions, neg_targets);
        let squared = graph.mul(diff, diff);
        let loss = graph.sum(squared);
        graph.backward(loss);

        let autograd_w1 = graph.grad(w1_ag).clone();
        let autograd_b1 = graph.grad(b1).clone();
        let autograd_w2 = graph.grad(w2).clone();
        let autograd_b2 = graph.grad(b2).clone();

        let nn_w1_from_autograd = Array2::from_shape_fn((4, 3), |(row, col)| {
            if col < 2 {
                autograd_w1[(col, row)]
            } else {
                autograd_b1[(0, row)]
            }
        });
        let nn_w2_from_autograd = Array2::from_shape_fn((1, 5), |(row, col)| {
            if col < 4 {
                autograd_w2[(col, row)]
            } else {
                autograd_b2[(0, row)]
            }
        });

        assert!(gradients_match(
            &manual[0],
            &nn_w1_from_autograd,
            DEFAULT_TOLERANCE
        ));
        assert!(gradients_match(
            &manual[1],
            &nn_w2_from_autograd,
            DEFAULT_TOLERANCE
        ));
    }
}
