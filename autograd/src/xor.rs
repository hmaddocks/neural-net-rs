//! Tiny XOR MLP trained end-to-end on the autograd engine.

use crate::graph::{Graph, TensorId};
use ndarray::Array2;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

const INPUTS: [[f64; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const TARGETS: [f64; 4] = [0.0, 1.0, 1.0, 0.0];
const PARAM_COUNT: usize = 4;

/// A two-layer ReLU network for the XOR problem: 2 → 8 → 1.
pub struct XorMlp {
    graph: Graph,
    w1: TensorId,
    b1: TensorId,
    w2: TensorId,
    b2: TensorId,
}

impl XorMlp {
    /// Creates a new XOR network with Xavier-initialized weights.
    pub fn new(seed: u64) -> Self {
        let mut graph = Graph::new();
        let mut rng = StdRng::seed_from_u64(seed);

        let w1 = graph.leaf(random_matrix(2, 8, &mut rng));
        let b1 = graph.leaf(Array2::zeros((1, 8)));
        let w2 = graph.leaf(random_matrix(8, 1, &mut rng));
        let b2 = graph.leaf(Array2::zeros((1, 1)));

        Self {
            graph,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Runs full-batch training and returns the final mean squared error.
    pub fn train(&mut self, epochs: usize, learning_rate: f64) -> f64 {
        let inputs = xor_inputs();
        let neg_targets = xor_neg_targets();
        let sample_count = inputs.nrows() as f64;
        let effective_lr = learning_rate / (2.0 * sample_count);

        let mut final_loss = f64::MAX;
        for _ in 0..epochs {
            self.graph.zero_grad();
            let predictions = self.forward(&inputs);
            let neg_targets_id = self.graph.leaf(neg_targets.clone());
            let diff = self.graph.add(predictions, neg_targets_id);
            let squared = self.graph.mul(diff, diff);
            let loss = self.graph.sum(squared);

            self.graph.backward(loss);
            sgd_step(&mut self.graph, self.w1, effective_lr);
            sgd_step(&mut self.graph, self.b1, effective_lr);
            sgd_step(&mut self.graph, self.w2, effective_lr);
            sgd_step(&mut self.graph, self.b2, effective_lr);

            final_loss = self.graph.data(loss)[(0, 0)] / sample_count;
            self.graph.clear_computation_graph(PARAM_COUNT);
        }

        final_loss
    }

    /// Returns predictions for all four XOR inputs as a `(4, 1)` matrix.
    pub fn predict_all(&mut self) -> Array2<f64> {
        let inputs = xor_inputs();
        let output = self.forward(&inputs);
        let predictions = self.graph.data(output).clone();
        self.graph.clear_computation_graph(PARAM_COUNT);
        predictions
    }

    fn forward(&mut self, inputs: &Array2<f64>) -> TensorId {
        let input = self.graph.leaf(inputs.clone());
        let linear1 = self.graph.matmul(input, self.w1);
        let biased1 = self.graph.add(linear1, self.b1);
        let hidden = self.graph.relu(biased1);
        let linear2 = self.graph.matmul(hidden, self.w2);
        self.graph.add(linear2, self.b2)
    }
}

fn xor_inputs() -> Array2<f64> {
    Array2::from_shape_fn((4, 2), |(row, col)| INPUTS[row][col])
}

fn xor_neg_targets() -> Array2<f64> {
    Array2::from_shape_fn((4, 1), |(row, _)| -TARGETS[row])
}

fn random_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Array2<f64> {
    let scale = (6.0 / (rows + cols) as f64).sqrt();
    Array2::from_shape_fn((rows, cols), |_| rng.random_range(-scale..scale))
}

fn sgd_step(graph: &mut Graph, param: TensorId, learning_rate: f64) {
    let updated = graph.data(param) - &(graph.grad(param) * learning_rate);
    graph.set_data(param, updated);
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::{TARGETS, XorMlp};

    #[test]
    fn xor_mlp_converges_with_autograd() {
        let mut model = XorMlp::new(42);
        let final_loss = model.train(8_000, 1.0);

        assert!(
            final_loss < 0.05,
            "expected mean squared error below 0.05, got {final_loss}"
        );

        let predictions = model.predict_all();
        for row in 0..4 {
            assert_relative_eq!(predictions[(row, 0)], TARGETS[row], epsilon = 0.25);
        }
    }

    #[test]
    fn xor_mlp_learns_correct_ordering() {
        let mut model = XorMlp::new(7);
        let _ = model.train(4_000, 0.5);
        let predictions = model.predict_all();

        let output_00 = predictions[(0, 0)];
        let output_01 = predictions[(1, 0)];
        let output_10 = predictions[(2, 0)];
        let output_11 = predictions[(3, 0)];

        assert!(output_01 > output_00);
        assert!(output_10 > output_00);
        assert!(output_01 >= output_11);
        assert!(output_10 >= output_11);
    }
}
