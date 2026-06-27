use crate::broadcast::{broadcast_add, broadcast_mul, sum_broadcast_grad};
use crate::graph::{Graph, TensorId};
use ndarray::Array2;

const RMSNORM_EPS: f64 = 1e-5;

impl Graph {
    /// Matrix multiplication: `(m, k) @ (k, n) -> (m, n)`.
    pub fn matmul(&mut self, left: TensorId, right: TensorId) -> TensorId {
        let left_data = self.data(left);
        let right_data = self.data(right);
        assert_eq!(
            left_data.ncols(),
            right_data.nrows(),
            "matmul requires left cols == right rows"
        );

        let output = left_data.dot(right_data);
        self.op(
            output,
            vec![left, right],
            Some(Box::new(move |graph, id| {
                let grad_out = graph.grad(id).clone();
                let left_value = graph.data(left).clone();
                let right_value = graph.data(right).clone();
                graph.add_grad(left, &grad_out.dot(&right_value.t()));
                graph.add_grad(right, &left_value.t().dot(&grad_out));
            })),
        )
    }

    /// Elementwise addition with broadcasting.
    pub fn add(&mut self, left: TensorId, right: TensorId) -> TensorId {
        let left_data = self.data(left);
        let right_data = self.data(right);
        let output = broadcast_add(left_data, right_data)
            .unwrap_or_else(|| panic!("incompatible broadcast shapes for add"));

        self.op(
            output,
            vec![left, right],
            Some(Box::new(move |graph, id| {
                let grad_out = graph.grad(id).clone();
                graph.add_grad(left, &sum_broadcast_grad(&grad_out, graph.data(left).dim()));
                graph.add_grad(
                    right,
                    &sum_broadcast_grad(&grad_out, graph.data(right).dim()),
                );
            })),
        )
    }

    /// Elementwise multiplication with broadcasting.
    pub fn mul(&mut self, left: TensorId, right: TensorId) -> TensorId {
        let left_data = self.data(left);
        let right_data = self.data(right);
        let output = broadcast_mul(left_data, right_data)
            .unwrap_or_else(|| panic!("incompatible broadcast shapes for mul"));

        self.op(
            output,
            vec![left, right],
            Some(Box::new(move |graph, id| {
                let grad_out = graph.grad(id).clone();
                let left_value = graph.data(left).clone();
                let right_value = graph.data(right).clone();
                let grad_left = broadcast_mul(&grad_out, &right_value)
                    .expect("mul backward left shape mismatch");
                let grad_right = broadcast_mul(&grad_out, &left_value)
                    .expect("mul backward right shape mismatch");
                graph.add_grad(left, &sum_broadcast_grad(&grad_left, left_value.dim()));
                graph.add_grad(right, &sum_broadcast_grad(&grad_right, right_value.dim()));
            })),
        )
    }

    /// ReLU activation applied elementwise.
    pub fn relu(&mut self, input: TensorId) -> TensorId {
        let input_data = self.data(input).clone();
        let output = input_data.mapv(|value| value.max(0.0));

        self.op(
            output,
            vec![input],
            Some(Box::new(move |graph, id| {
                let grad_out = graph.grad(id).clone();
                let mask = graph
                    .data(input)
                    .mapv(|value| if value > 0.0 { 1.0 } else { 0.0 });
                graph.add_grad(input, &(&grad_out * &mask));
            })),
        )
    }

    /// Natural logarithm applied elementwise.
    pub fn log(&mut self, input: TensorId) -> TensorId {
        let input_data = self.data(input).clone();
        let output = input_data.mapv(f64::ln);

        self.op(
            output,
            vec![input],
            Some(Box::new(move |graph, id| {
                let grad_out = graph.grad(id).clone();
                let input_value = graph.data(input).clone();
                graph.add_grad(input, &(&grad_out / &input_value));
            })),
        )
    }

    /// Power function `input^exponent` applied elementwise.
    pub fn pow(&mut self, input: TensorId, exponent: f64) -> TensorId {
        let input_data = self.data(input).clone();
        let output = input_data.mapv(|value| value.powf(exponent));

        self.op(
            output,
            vec![input],
            Some(Box::new(move |graph, id| {
                let grad_out = graph.grad(id).clone();
                let input_value = graph.data(input).clone();
                let local = input_value.mapv(|value| exponent * value.powf(exponent - 1.0));
                graph.add_grad(input, &(&grad_out * &local));
            })),
        )
    }

    /// Softmax applied independently to each row (last axis).
    pub fn softmax(&mut self, input: TensorId) -> TensorId {
        let input_data = self.data(input).clone();
        let output = softmax_forward(&input_data);

        self.op(
            output,
            vec![input],
            Some(Box::new(move |graph, id| {
                let grad_out = graph.grad(id).clone();
                let softmax_out = graph.data(id).clone();
                graph.add_grad(input, &softmax_backward(&grad_out, &softmax_out));
            })),
        )
    }

    /// RMS normalization applied independently to each row.
    pub fn rmsnorm(&mut self, input: TensorId) -> TensorId {
        let input_data = self.data(input).clone();
        let output = rmsnorm_forward(&input_data);

        self.op(
            output,
            vec![input],
            Some(Box::new(move |graph, id| {
                let grad_out = graph.grad(id).clone();
                let input_value = graph.data(input).clone();
                let output_value = graph.data(id).clone();
                graph.add_grad(
                    input,
                    &rmsnorm_backward(&grad_out, &input_value, &output_value),
                );
            })),
        )
    }
}

fn softmax_forward(input: &Array2<f64>) -> Array2<f64> {
    let mut output = input.clone();
    for mut row in output.rows_mut() {
        let max = row.fold(f64::NEG_INFINITY, |acc, &value| acc.max(value));
        row.mapv_inplace(|value| (value - max).exp());
        let sum = row.sum();
        row.mapv_inplace(|value| value / sum);
    }
    output
}

fn softmax_backward(grad_output: &Array2<f64>, softmax_output: &Array2<f64>) -> Array2<f64> {
    let mut grad_input = Array2::zeros(grad_output.raw_dim());
    for row in 0..grad_output.nrows() {
        let dot = grad_output
            .row(row)
            .iter()
            .zip(softmax_output.row(row).iter())
            .map(|(grad, prob)| grad * prob)
            .sum::<f64>();
        for col in 0..grad_output.ncols() {
            let prob = softmax_output[(row, col)];
            let grad = grad_output[(row, col)];
            grad_input[(row, col)] = prob * (grad - dot);
        }
    }
    grad_input
}

fn rmsnorm_forward(input: &Array2<f64>) -> Array2<f64> {
    let mut output = Array2::zeros(input.raw_dim());
    for row in 0..input.nrows() {
        let values = input.row(row);
        let mean_square =
            values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64;
        let scale = (mean_square + RMSNORM_EPS).powf(-0.5);
        for col in 0..input.ncols() {
            output[(row, col)] = input[(row, col)] * scale;
        }
    }
    output
}

fn rmsnorm_backward(
    grad_output: &Array2<f64>,
    input: &Array2<f64>,
    _output: &Array2<f64>,
) -> Array2<f64> {
    let mut grad_input = Array2::zeros(input.raw_dim());
    for row in 0..input.nrows() {
        let width = input.ncols() as f64;
        let mean_square = input
            .row(row)
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            / width;
        let scale = (mean_square + RMSNORM_EPS).powf(-0.5);
        let dot = grad_output
            .row(row)
            .iter()
            .zip(input.row(row).iter())
            .map(|(grad, value)| grad * value)
            .sum::<f64>();
        let scale_cubed = scale.powi(3);
        for col in 0..input.ncols() {
            let value = input[(row, col)];
            let grad = grad_output[(row, col)];
            grad_input[(row, col)] = scale * grad - (scale_cubed / width) * dot * value;
        }
    }
    grad_input
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::array;

    use super::*;
    use crate::graph::Graph;

    fn backward_from_output(graph: &mut Graph, output: TensorId) {
        graph.backward(output);
    }

    #[test]
    fn matmul_forward_and_backward() {
        let mut graph = Graph::new();
        let left = graph.leaf(array![[1.0, 2.0], [3.0, 4.0]]);
        let right = graph.leaf(array![[5.0, 6.0], [7.0, 8.0]]);
        let product = graph.matmul(left, right);

        assert_relative_eq!(graph.data(product)[(0, 0)], 19.0);
        backward_from_output(&mut graph, product);
        assert_relative_eq!(graph.grad(left)[(0, 0)], 11.0);
        assert_relative_eq!(graph.grad(right)[(0, 0)], 4.0);
    }

    #[test]
    fn add_supports_broadcasting() {
        let mut graph = Graph::new();
        let bias = graph.leaf(array![[1.0, 2.0]]);
        let values = graph.leaf(array![[10.0], [20.0]]);
        let sum = graph.add(values, bias);

        assert_relative_eq!(graph.data(sum)[(1, 1)], 22.0);
        backward_from_output(&mut graph, sum);
        assert_relative_eq!(graph.grad(bias)[(0, 0)], 2.0);
        assert_relative_eq!(graph.grad(bias)[(0, 1)], 2.0);
        assert_relative_eq!(graph.grad(values)[(0, 0)], 2.0);
        assert_relative_eq!(graph.grad(values)[(1, 0)], 2.0);
    }

    #[test]
    fn mul_supports_broadcasting() {
        let mut graph = Graph::new();
        let scale = graph.leaf(array![[2.0]]);
        let values = graph.leaf(array![[3.0], [4.0]]);
        let product = graph.mul(values, scale);

        assert_relative_eq!(graph.data(product)[(1, 0)], 8.0);
        backward_from_output(&mut graph, product);
        assert_relative_eq!(graph.grad(values)[(0, 0)], 2.0);
        assert_relative_eq!(graph.grad(scale)[(0, 0)], 7.0);
    }

    #[test]
    fn relu_zeros_negative_inputs() {
        let mut graph = Graph::new();
        let input = graph.leaf(array![[-1.0, 2.0]]);
        let output = graph.relu(input);

        assert_relative_eq!(graph.data(output)[(0, 0)], 0.0);
        assert_relative_eq!(graph.data(output)[(0, 1)], 2.0);
        backward_from_output(&mut graph, output);
        assert_relative_eq!(graph.grad(input)[(0, 0)], 0.0);
        assert_relative_eq!(graph.grad(input)[(0, 1)], 1.0);
    }

    #[test]
    fn log_and_pow_backward() {
        let mut graph = Graph::new();
        let input = graph.leaf(array![[4.0]]);
        let logged = graph.log(input);
        let squared = graph.pow(input, 2.0);

        backward_from_output(&mut graph, logged);
        assert_relative_eq!(graph.grad(input)[(0, 0)], 0.25);
        graph.zero_grad();

        backward_from_output(&mut graph, squared);
        assert_relative_eq!(graph.grad(input)[(0, 0)], 8.0);
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let mut graph = Graph::new();
        let input = graph.leaf(array![[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]);
        let output = graph.softmax(input);

        assert_relative_eq!(graph.data(output).row(0).sum(), 1.0);
        assert_relative_eq!(graph.data(output).row(1).sum(), 1.0);
        backward_from_output(&mut graph, output);
        assert_relative_eq!(graph.grad(input).sum(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn rmsnorm_scales_each_row() {
        let mut graph = Graph::new();
        let input = graph.leaf(array![[1.0, 2.0, 3.0]]);
        let output = graph.rmsnorm(input);

        let mean_square = (1.0 + 4.0 + 9.0) / 3.0;
        let scale = (mean_square + RMSNORM_EPS).powf(-0.5);
        assert_relative_eq!(graph.data(output)[(0, 1)], 2.0 * scale);
        backward_from_output(&mut graph, output);
        assert!(graph.grad(input)[(0, 0)].is_finite());
    }
}
