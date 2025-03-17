use crate::activations::Activation;
use matrix::matrix::Matrix;

#[derive(Builder)]
pub struct Network {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    data: Vec<Matrix>,
    activation: Activation,
    learning_rate: f64,
    momentum: f64,
    prev_weight_updates: Vec<Matrix>,
}

impl Network {
    pub fn new(layers: Vec<usize>, activation: Activation, learning_rate: f64) -> Self {
        let layer_pairs: Vec<_> = layers.windows(2).collect();

        let weights = layer_pairs
            .iter()
            .map(|pair| {
                let (input_size, output_size) = (pair[0], pair[1]);
                Matrix::random(output_size, input_size + 1) // Add one for bias
            })
            .collect();

        let prev_weight_updates = layer_pairs
            .iter()
            .map(|pair| {
                let (input_size, output_size) = (pair[0], pair[1]);
                Matrix::zeros(output_size, input_size + 1)
            })
            .collect();

        Network {
            layers,
            weights,
            data: Vec::new(),
            activation,
            learning_rate,
            momentum: 0.9,
            prev_weight_updates,
        }
    }

    fn augment_with_bias(input: Matrix) -> Matrix {
        let mut augmented = Vec::with_capacity(input.data.len() + input.cols);
        augmented.extend_from_slice(&input.data);
        augmented.extend(std::iter::repeat(1.0).take(input.cols));
        Matrix::new(input.rows + 1, input.cols, augmented)
    }

    fn process_layer(weight: &Matrix, input: &Matrix, activation: &Activation) -> Matrix {
        let output = weight.dot_multiply(input);
        if let Some(vector_fn) = activation.vector_function {
            vector_fn(&output)
        } else {
            output.map(activation.function)
        }
    }

    pub fn feed_forward(&mut self, inputs: Matrix) -> Matrix {
        assert!(
            self.layers[0] == inputs.data.len(),
            "Invalid number of inputs. Expected {}, got {}",
            self.layers[0],
            inputs.data.len()
        );

        // Store original input
        self.data = vec![inputs.clone()];

        // Process through layers functionally
        let result = self.weights.iter().fold(inputs, |current, weight| {
            let with_bias = Self::augment_with_bias(current);
            let output = Self::process_layer(weight, &with_bias, &self.activation);
            self.data.push(output.clone());
            output
        });

        result
    }

    pub fn back_propogate(&mut self, outputs: Matrix, targets: Matrix) {
        let mut errors = targets.subtract(&outputs);
        let mut gradients = if let Some(vector_derivative) = self.activation.vector_derivative {
            vector_derivative(&outputs)
        } else {
            outputs.map(self.activation.derivative)
        };

        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients
                .elementwise_multiply(&errors)
                .map(|x| x * self.learning_rate);

            let layer_input = Self::augment_with_bias(self.data[i].clone());
            let weight_updates = gradients.dot_multiply(&layer_input.transpose());

            // Apply momentum using functional update
            self.weights[i] = self.weights[i]
                .add(&weight_updates.add(&self.prev_weight_updates[i].map(|x| x * self.momentum)));

            self.prev_weight_updates[i] = weight_updates;

            if i > 0 {
                // Propagate error through weights (excluding bias weights)
                let weight_no_bias = Matrix::new(
                    self.weights[i].rows,
                    self.weights[i].cols - 1,
                    self.weights[i].data[..self.weights[i].rows * (self.weights[i].cols - 1)]
                        .to_vec(),
                );
                errors = weight_no_bias.transpose().dot_multiply(&errors);
                gradients = if let Some(vector_derivative) = self.activation.vector_derivative {
                    vector_derivative(&self.data[i])
                } else {
                    self.data[i].map(self.activation.derivative)
                };
            }
        }
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u32) {
        for epoch in 1..=epochs {
            if epochs < 100 || epoch % (epochs / 100) == 0 {
                println!("Epoch {} of {}", epoch, epochs);
            }

            inputs.iter().zip(&targets).for_each(|(input, target)| {
                let outputs = self.feed_forward(Matrix::from(input.clone()));
                self.back_propogate(outputs, Matrix::from(target.clone()));
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::SIGMOID_VECTOR;

    #[test]
    fn test_network_creation() {
        let layers = vec![3, 4, 2];
        let network = Network::new(layers.clone(), SIGMOID_VECTOR, 0.5);

        assert_eq!(network.layers, layers);
        assert_eq!(network.weights.len(), 2);
        assert_eq!(network.weights[0].rows, 4);
        assert_eq!(network.weights[0].cols, 4); // 3 inputs + 1 bias
        assert_eq!(network.weights[1].rows, 2);
        assert_eq!(network.weights[1].cols, 5); // 4 inputs + 1 bias
        assert_eq!(network.learning_rate, 0.5);
    }

    #[test]
    fn test_feed_forward() {
        let layers = vec![2, 3, 1];
        let mut network = Network::new(layers, SIGMOID_VECTOR, 0.5);

        // First layer: 3 outputs, 3 inputs (2 inputs + 1 bias)
        network.weights[0] = Matrix::new(
            3,
            3,
            vec![
                0.1, 0.2, 0.3, // Weights for first neuron (2 inputs + bias)
                0.4, 0.5, 0.6, // Weights for second neuron (2 inputs + bias)
                0.7, 0.8, 0.9, // Weights for third neuron (2 inputs + bias)
            ],
        );

        // Second layer: 1 output, 4 inputs (3 from previous layer + 1 bias)
        network.weights[1] = Matrix::new(
            1,
            4,
            vec![0.1, 0.2, 0.3, 0.4], // Weights for output neuron (3 inputs + bias)
        );

        let input = Matrix::new(2, 1, vec![0.5, 0.8]);
        let output = network.feed_forward(input);

        assert_eq!(output.rows, 1);
        assert_eq!(output.cols, 1);
        // Output should be deterministic given fixed weights
        assert!(output.data[0] > 0.0 && output.data[0] < 1.0);
    }

    #[test]
    #[should_panic(expected = "Invalid number of inputs")]
    fn test_feed_forward_invalid_inputs() {
        let layers = vec![2, 3, 1];
        let mut network = Network::new(layers, SIGMOID_VECTOR, 0.5);

        // Wrong number of inputs (3 instead of 2)
        let input = Matrix::new(3, 1, vec![0.5, 0.8, 0.3]);
        network.feed_forward(input);
    }

    #[test]
    fn test_training() {
        let layers = vec![2, 4, 1]; // Reduced size for faster testing
        let mut network = Network::new(layers, SIGMOID_VECTOR, 0.5);

        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];

        let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

        // Train for fewer epochs in test
        network.train(inputs.clone(), targets.clone(), 1000);

        // Test that network can learn XOR pattern
        let input1 = Matrix::new(2, 1, vec![0.0, 0.0]);
        let input2 = Matrix::new(2, 1, vec![1.0, 1.0]);
        let output1 = network.feed_forward(input1);
        let output2 = network.feed_forward(input2);

        // Allow for some variance in the outputs
        assert!(output1.data[0] < 0.4); // Should be closer to 0
        assert!(output2.data[0] < 0.4); // Should be closer to 0

        // Test the other cases
        let input3 = Matrix::new(2, 1, vec![0.0, 1.0]);
        let input4 = Matrix::new(2, 1, vec![1.0, 0.0]);
        let output3 = network.feed_forward(input3);
        let output4 = network.feed_forward(input4);

        assert!(output3.data[0] > 0.6); // Should be closer to 1
        assert!(output4.data[0] > 0.6); // Should be closer to 1
    }
}
