use matrix::matrix::Matrix;

use crate::activations::Activation;

#[derive(Builder)]
pub struct Network {
    layers: Vec<usize>, // amount of neurons in each layer, [72,16,10]
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    data: Vec<Matrix>,
    activation: Activation,
    learning_rate: f64,
    momentum: f64,
    prev_weight_updates: Vec<Matrix>,
    prev_bias_updates: Vec<Matrix>,
}

impl Network {
    pub fn new(layers: Vec<usize>, activation: Activation, learning_rate: f64) -> Self {
        let mut weights = vec![];
        let mut biases = vec![];
        let mut prev_weight_updates = vec![];
        let mut prev_bias_updates = vec![];

        for i in 0..layers.len() - 1 {
            weights.push(Matrix::random(layers[i + 1], layers[i]));
            biases.push(Matrix::random(layers[i + 1], 1));
            prev_weight_updates.push(Matrix::zeros(layers[i + 1], layers[i]));
            prev_bias_updates.push(Matrix::zeros(layers[i + 1], 1));
        }

        Network {
            layers,
            weights,
            biases,
            data: vec![],
            activation,
            learning_rate,
            momentum: 0.9,
            prev_weight_updates,
            prev_bias_updates,
        }
    }

    pub fn feed_forward(&mut self, inputs: Matrix) -> Matrix {
        assert!(
            self.layers[0] == inputs.data.len(),
            "Invalid Number of Inputs"
        );

        let mut current = inputs;

        self.data = vec![current.clone()];

        for i in 0..self.layers.len() - 1 {
            current = self.weights[i]
                .dot_multiply(&current)
                .add(&self.biases[i])
                .map(self.activation.function);

            self.data.push(current.clone());
        }

        current
    }

    pub fn back_propogate(&mut self, outputs: Matrix, targets: Matrix) {
        let mut errors = targets.subtract(&outputs);

        let mut gradients = outputs.clone().map(self.activation.derivative);

        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients
                .elementwise_multiply(&errors)
                .map(|x| x * self.learning_rate);

            let weight_updates = gradients.dot_multiply(&self.data[i].transpose());
            let bias_updates = gradients.clone();

            // Apply momentum
            self.weights[i] = self.weights[i].add(&weight_updates.add(&self.prev_weight_updates[i].map(|x| x * self.momentum)));
            self.biases[i] = self.biases[i].add(&bias_updates.add(&self.prev_bias_updates[i].map(|x| x * self.momentum)));

            // Store updates for next iteration
            self.prev_weight_updates[i] = weight_updates;
            self.prev_bias_updates[i] = bias_updates;

            errors = self.weights[i].transpose().dot_multiply(&errors);
            gradients = self.data[i].map(self.activation.derivative);
        }
    }

    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u32) {
        for i in 1..=epochs {
            if epochs < 100 || i % (epochs / 100) == 0 {
                println!("Epoch {} of {}", i, epochs);
            }
            for j in 0..inputs.len() {
                let outputs = self.feed_forward(Matrix::from(inputs[j].clone()));
                self.back_propogate(outputs, Matrix::from(targets[j].clone()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::SIGMOID;

    #[test]
    fn test_network_creation() {
        let layers = vec![3, 4, 2];
        let network = Network::new(layers.clone(), SIGMOID, 0.5);

        assert_eq!(network.layers, layers);
        assert_eq!(network.weights.len(), 2);
        assert_eq!(network.biases.len(), 2);
        assert_eq!(network.weights[0].rows, 4);
        assert_eq!(network.weights[0].cols, 3);
        assert_eq!(network.weights[1].rows, 2);
        assert_eq!(network.weights[1].cols, 4);
        assert_eq!(network.learning_rate, 0.5);
    }

    #[test]
    fn test_feed_forward() {
        let layers = vec![2, 3, 1];
        let mut network = Network::new(layers, SIGMOID, 0.5);

        // Set deterministic weights and biases for testing
        network.weights[0] = Matrix::new(
            3,
            2,
            vec![
                0.1, 0.2, // First row
                0.3, 0.4, // Second row
                0.5, 0.6, // Third row
            ],
        );
        network.weights[1] = Matrix::new(1, 3, vec![0.7, 0.8, 0.9]);
        network.biases[0] = Matrix::new(3, 1, vec![0.1, 0.2, 0.3]);
        network.biases[1] = Matrix::new(1, 1, vec![0.4]);

        let input = Matrix::new(2, 1, vec![0.5, 0.8]);
        let output = network.feed_forward(input);

        assert_eq!(output.rows, 1);
        assert_eq!(output.cols, 1);
        // Output should be deterministic given fixed weights and biases
        assert!(output.data[0] > 0.0 && output.data[0] < 1.0);
    }

    #[test]
    #[should_panic(expected = "Invalid Number of Inputs")]
    fn test_feed_forward_invalid_inputs() {
        let layers = vec![2, 3, 1];
        let mut network = Network::new(layers, SIGMOID, 0.5);

        // Wrong number of inputs (3 instead of 2)
        let input = Matrix::new(3, 1, vec![0.5, 0.8, 0.3]);
        network.feed_forward(input);
    }

    #[test]
    fn test_training() {
        let layers = vec![2, 8, 8, 1]; // Two hidden layers with 8 neurons each
        let mut network = Network::new(layers, SIGMOID, 0.5); // Increased learning rate

        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];

        let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

        // Train for more epochs
        network.train(inputs.clone(), targets.clone(), 5000);

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
