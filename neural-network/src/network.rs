/// Neural network implementation using a feed-forward architecture with backpropagation.
///
/// This module provides a flexible neural network implementation that supports:
/// - Configurable layer sizes
/// - Custom activation functions with vector operations support
/// - Momentum-based learning
/// - Batch training capabilities
/// - Model saving and loading
///
/// # Example
/// ```
/// use neural_network::{network::Network, activations::ActivationType, network_config::NetworkConfig};
/// use tempfile::tempdir;
///
/// let dir = tempdir().unwrap();
/// let model_path = dir.path().join("model.json");
///
/// // Create network configuration
/// let mut config = NetworkConfig::default();
/// config.layers = vec![2, 3, 1];
/// config.activations = vec![ActivationType::Sigmoid, ActivationType::Sigmoid];
/// config.learning_rate = 0.1;
/// config.momentum = Some(0.8);
///
/// // Create and save network
/// let mut network = Network::new(&config);
/// network.save(model_path.to_str().unwrap()).expect("Failed to save model");
/// ```
use crate::activations::{ActivationFunction, ActivationType};
use crate::network_config::NetworkConfig;
use matrix::matrix::Matrix;
use serde::{Deserialize, Serialize};
use std::io;

/// A feed-forward neural network with configurable layers and activation functions.
#[derive(Serialize, Deserialize)]
pub struct Network {
    /// Sizes of each layer in the network, including input and output layers
    layers: Vec<usize>,
    /// Weight matrices between layers, including bias weights
    weights: Vec<Matrix>,
    /// Cached layer outputs for backpropagation
    #[serde(skip)]
    data: Vec<Matrix>,
    /// Activation functions for each layer
    #[serde(skip)]
    activations: Vec<Box<dyn ActivationFunction + Send + Sync>>,
    /// Types of activation functions for serialization
    activation_types: Vec<ActivationType>,
    /// Learning rate for weight updates
    learning_rate: f64,
    /// Momentum coefficient for weight updates
    momentum: f64,
    /// Previous weight updates for momentum calculation
    #[serde(skip)]
    prev_weight_updates: Vec<Matrix>,
}

impl Network {
    /// Creates a new neural network with specified configuration.
    ///
    /// # Arguments
    /// * `network_config` - Configuration struct containing:
    ///   - `layers`: Vector of layer sizes, including input and output layers
    ///   - `activations`: Vector of activation types for each layer (must be one less than number of layers)
    ///   - `learning_rate`: Learning rate for weight updates during training
    ///   - `momentum`: Optional momentum coefficient for weight updates (defaults to 0.9)
    ///
    /// # Returns
    /// A new `Network` instance with randomly initialized weights
    ///
    /// # Panics
    /// Panics if the number of activation functions doesn't match the number of layers minus one
    pub fn new(network_config: &NetworkConfig) -> Self {
        assert!(
            network_config.activations.len() == network_config.layers.len() - 1,
            "Number of activation functions ({}) must be one less than number of layers ({})",
            network_config.activations.len(),
            network_config.layers.len()
        );

        let layer_pairs: Vec<_> = network_config.layers.windows(2).collect();

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

        let mut data = Vec::with_capacity(network_config.layers.len());
        data.resize(network_config.layers.len(), Matrix::default());

        Network {
            layers: network_config.layers.clone(),
            weights,
            data,
            activations: network_config.get_activations(),
            activation_types: network_config.activations.clone(),
            learning_rate: network_config.learning_rate,
            momentum: network_config.momentum.unwrap_or(0.9),
            prev_weight_updates,
        }
    }

    /// Trains the network on a dataset for a specified number of epochs.
    ///
    /// # Arguments
    /// * `inputs` - Vector of input vectors
    /// * `targets` - Vector of target output vectors
    /// * `epochs` - Number of training epochs
    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u32) {
        for epoch in 1..=epochs {
            let epoch_start = std::time::Instant::now();
            let mut total_error = 0.0;
            let mut correct_predictions = 0;
            let total_samples = inputs.len();

            inputs.iter().zip(&targets).for_each(|(input, target)| {
                let outputs = self.feed_forward(Matrix::from(input.clone()));
                let error = &Matrix::from(target.clone()) - &outputs;
                total_error += error.data.iter().map(|x| x * x).sum::<f64>();

                // Calculate accuracy - handle single output case differently
                if outputs.data.len() == 1 {
                    // Binary classification - compare with threshold
                    let predicted = outputs.data[0] >= 0.5;
                    let actual = target[0] >= 0.5;
                    if predicted == actual {
                        correct_predictions += 1;
                    }
                } else {
                    // Multi-class classification
                    let predicted = outputs
                        .data
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap();
                    let actual = target
                        .iter()
                        .position(|&val| (val - 1.0).abs() < f64::EPSILON)
                        .expect(
                            "Target vector should contain exactly one 1.0 value (one-hot encoding)",
                        );
                    if predicted == actual {
                        correct_predictions += 1;
                    }
                }

                self.back_propagate(outputs, Matrix::from(target.clone()));
            });

            let avg_error = total_error / total_samples as f64;
            let accuracy = (correct_predictions as f64 / total_samples as f64) * 100.0;
            let epoch_duration = epoch_start.elapsed();
            println!(
                "Epoch {} ({:.2?}): Average Error = {:.6}, Accuracy = {:.2}%",
                epoch, epoch_duration, avg_error, accuracy
            );

            // Early stopping if error is very small
            if avg_error < 1e-6 {
                println!("Reached target error. Stopping training.");
                break;
            }

            // Check for NaN or inf values in weights
            if self
                .weights
                .iter()
                .any(|w| w.data.iter().any(|&x| x.is_nan() || x.is_infinite()))
            {
                println!(
                    "Warning: Detected NaN or infinite values in weights at epoch {}!",
                    epoch
                );
                break;
            }
        }
    }

    /// Performs forward propagation through the network.
    ///
    /// # Arguments
    /// * `inputs` - Input matrix to process
    ///
    /// # Returns
    /// The network's output matrix
    ///
    /// # Panics
    /// Panics if the number of inputs doesn't match the first layer size
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
        let result = self
            .weights
            .iter()
            .enumerate()
            .fold(inputs, |current, (i, weight)| {
                let with_bias = current.augment_with_bias();
                let output = process_layer(weight, &with_bias, &*self.activations[i]);
                self.data.push(output.clone());
                output
            });

        result
    }

    /// Performs prediction without storing intermediate outputs.
    /// This is more efficient for inference-only use cases.
    ///
    /// # Arguments
    /// * `inputs` - Input matrix to process
    ///
    /// # Returns
    /// The network's output matrix
    ///
    /// # Panics
    /// Panics if the number of inputs doesn't match the first layer size
    pub fn predict(&self, inputs: Matrix) -> Matrix {
        assert!(
            self.layers[0] == inputs.data.len(),
            "Invalid number of inputs. Expected {}, got {}",
            self.layers[0],
            inputs.data.len()
        );

        // Process through layers functionally without storing intermediates
        self.weights
            .iter()
            .enumerate()
            .fold(inputs, |current, (i, weight)| {
                let with_bias = current.augment_with_bias();
                process_layer(weight, &with_bias, &*self.activations[i])
            })
    }

    /// Calculates gradients for a layer based on its activation function.
    ///
    /// # Arguments
    /// * `activation` - The activation function for the layer
    /// * `outputs` - The layer's outputs
    /// * `errors` - The error values
    ///
    /// # Returns
    /// The calculated gradients
    fn calculate_gradients(
        activation: &Box<dyn ActivationFunction + Send + Sync>,
        outputs: &Matrix,
        errors: &Matrix,
    ) -> Matrix {
        if matches!(activation.activation_type(), ActivationType::Softmax) {
            // For softmax with cross-entropy loss, the gradient simplifies to (output - target)
            errors.clone()
        } else {
            // For other activation functions, use their derivatives
            activation
                .derivative_vector(outputs)
                .elementwise_multiply(errors)
        }
    }

    /// Performs backpropagation to update network weights.
    ///
    /// # Arguments
    /// * `outputs` - Current network outputs
    /// * `targets` - Target outputs for training
    pub fn back_propagate(&mut self, outputs: Matrix, targets: Matrix) {
        let mut errors = &targets - &outputs;
        let mut gradients = if self.activations.is_empty() {
            // If no activation functions, just use the error directly
            errors.clone()
        } else {
            // For each layer, use the appropriate gradient calculation
            let last_layer_idx = self.activations.len() - 1;
            Self::calculate_gradients(&self.activations[last_layer_idx], &outputs, &errors)
        };

        for i in (0..self.layers.len() - 1).rev() {
            gradients = gradients.map(|x| x * self.learning_rate);

            let layer_input = self.data[i].clone().augment_with_bias();
            let weight_updates = gradients.dot_multiply(&layer_input.transpose());

            // Apply momentum
            let momentum_term = self.prev_weight_updates[i].map(|x| x * self.momentum);
            let updates = &weight_updates + &momentum_term;

            // Update weights
            self.weights[i] = &self.weights[i] + &updates;
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
                gradients =
                    Self::calculate_gradients(&self.activations[i - 1], &self.data[i], &errors);
            }
        }
    }

    /// Saves the trained network to a JSON file.
    ///
    /// # Arguments
    /// * `path` - Path to save the model file
    ///
    /// # Returns
    /// Result indicating success or containing an IO error
    ///
    /// # Example
    /// ```
    /// use neural_network::{network::Network, activations::ActivationType, network_config::NetworkConfig};
    /// use tempfile::tempdir;
    ///
    /// let dir = tempdir().unwrap();
    /// let model_path = dir.path().join("model.json");
    ///
    /// // Create network configuration
    /// let mut config = NetworkConfig::default();
    /// config.layers = vec![2, 3, 1];
    /// config.activations = vec![ActivationType::Sigmoid, ActivationType::Sigmoid];
    /// config.learning_rate = 0.1;
    /// config.momentum = Some(0.8);
    ///
    /// // Create and save network
    /// let mut network = Network::new(&config);
    /// network.save(model_path.to_str().unwrap()).expect("Failed to save model");
    /// ```
    pub fn save(&self, path: &str) -> io::Result<()> {
        let json = match serde_json::to_string_pretty(self) {
            Ok(json) => json,
            Err(e) => return Err(e.into()),
        };
        std::fs::write(path, json)
    }

    /// Loads a trained network from a JSON file.
    ///
    /// # Arguments
    /// * `path` - Path to the model file
    ///
    /// # Returns
    /// Result containing the loaded Network or an IO error
    ///
    /// # Example
    /// ```no_run
    /// # use neural_network::{network::Network, activations::ActivationType};
    /// # use tempfile::tempdir;
    /// # let dir = tempdir().unwrap();
    /// # let model_path = dir.path().join("model.json");
    /// let network = Network::load(model_path.to_str().unwrap()).expect("Failed to load model");
    /// ```
    pub fn load(path: &str) -> io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let mut network: Network = serde_json::from_str(&json)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // Initialize non-serialized fields
        network.data = Vec::new();
        network.prev_weight_updates = network
            .weights
            .iter()
            .map(|w| Matrix::zeros(w.rows, w.cols))
            .collect();

        // Create activation functions from types
        network.activations = network
            .activation_types
            .iter()
            .map(|t| t.create_activation())
            .collect();

        // Verify that we have the correct number of activation functions
        if network.activations.len() != network.layers.len() - 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Invalid number of activation functions. Expected {}, got {}",
                    network.layers.len() - 1,
                    network.activations.len()
                ),
            ));
        }

        Ok(network)
    }
}

/// Processes a single layer of the network.
///
/// # Arguments
/// * `weight` - Weight matrix for the layer
/// * `input` - Input matrix including bias terms
/// * `activation` - Activation function to apply
///
/// # Returns
/// The processed output matrix after applying weights and activation
fn process_layer(
    weight: &Matrix,
    input: &Matrix,
    activation: &(dyn ActivationFunction + Send + Sync),
) -> Matrix {
    let output = weight.dot_multiply(input);
    activation.apply_vector(&output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::{ActivationFunction, ActivationType, Sigmoid, Softmax};
    use crate::network_config::NetworkConfig;
    use approx::assert_relative_eq;
    use matrix::matrix::IntoMatrix;
    use tempfile::tempdir;

    #[test]
    fn test_network_creation() {
        let mut config = NetworkConfig::default();
        config.layers = vec![3, 4, 2];
        config.activations = vec![ActivationType::Sigmoid, ActivationType::Sigmoid];
        config.learning_rate = 0.5;
        config.momentum = Some(0.7);
        let network = Network::new(&config);

        assert_eq!(network.layers, config.layers);
        assert_eq!(network.weights.len(), 2);
        assert_eq!(network.weights[0].rows, 4);
        assert_eq!(network.weights[0].cols, 4); // 3 inputs + 1 bias
        assert_eq!(network.weights[1].rows, 2);
        assert_eq!(network.weights[1].cols, 5); // 4 inputs + 1 bias
        assert_eq!(network.learning_rate, 0.5);
        assert_eq!(network.momentum, 0.7);
    }

    #[test]
    fn test_feed_forward() {
        let mut config = NetworkConfig::default();
        config.layers = vec![2, 3, 1];
        config.activations = vec![ActivationType::Sigmoid, ActivationType::Sigmoid];
        config.learning_rate = 0.5;
        config.momentum = Some(0.8);
        let mut network = Network::new(&config);

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
        let mut config = NetworkConfig::default();
        config.layers = vec![2, 3, 1];
        config.activations = vec![ActivationType::Sigmoid, ActivationType::Sigmoid];
        config.learning_rate = 0.5;
        config.momentum = Some(0.8);
        let mut network = Network::new(&config);

        // Wrong number of inputs (3 instead of 2)
        let input = Matrix::new(3, 1, vec![0.5, 0.8, 0.3]);
        network.feed_forward(input);
    }

    #[test]
    fn test_predict() {
        let mut config = NetworkConfig::default();
        config.layers = vec![2, 3, 1];
        config.activations = vec![ActivationType::Sigmoid, ActivationType::Sigmoid];
        config.learning_rate = 0.5;
        config.momentum = Some(0.8);
        let mut network = Network::new(&config);

        // First layer: 3 outputs, 3 inputs (2 inputs + 1 bias)
        network.weights[0] = Matrix::new(3, 3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);

        // Second layer: 1 output, 4 inputs (3 inputs + 1 bias)
        network.weights[1] = Matrix::new(1, 4, vec![0.1, 0.2, 0.3, 0.4]);

        let input = Matrix::new(2, 1, vec![0.5, 0.8]);

        // Compare feed_forward and predict outputs
        let output_ff = network.feed_forward(input.clone());
        let output_predict = network.predict(input);

        assert_eq!(output_ff.data, output_predict.data);
        assert_eq!(output_ff.rows, output_predict.rows);
        assert_eq!(output_ff.cols, output_predict.cols);
    }

    #[test]
    #[should_panic(expected = "Invalid number of inputs")]
    fn test_predict_invalid_inputs() {
        let mut config = NetworkConfig::default();
        config.layers = vec![2, 3, 1];
        config.activations = vec![ActivationType::Sigmoid, ActivationType::Sigmoid];
        config.learning_rate = 0.5;
        config.momentum = Some(0.8);
        let network = Network::new(&config);

        // Wrong number of inputs (3 instead of 2)
        let input = Matrix::new(3, 1, vec![0.5, 0.8, 0.3]);
        network.predict(input);
    }

    #[test]
    fn test_training() {
        let mut config = NetworkConfig::default();
        config.layers = vec![2, 4, 1]; // Reduced size for faster testing
        config.activations = vec![ActivationType::Sigmoid, ActivationType::Sigmoid];
        config.learning_rate = 0.5;
        config.momentum = Some(0.8);
        let mut network = Network::new(&config);

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

    #[test]
    fn test_network_serialization() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("test_model.json");

        // Create and train a simple network
        let mut config = NetworkConfig::default();
        config.layers = vec![2, 3, 1];
        config.activations = vec![ActivationType::Sigmoid, ActivationType::Sigmoid];
        config.learning_rate = 0.1;
        config.momentum = Some(0.8);
        let mut network = Network::new(&config);
        let inputs = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let targets = vec![vec![1.0], vec![0.0]];
        network.train(inputs, targets, 10);

        // Save the network
        network.save(model_path.to_str().unwrap()).unwrap();

        // Load the network
        let mut loaded_network = Network::load(model_path.to_str().unwrap()).unwrap();

        // Verify the loaded network has the same structure
        assert_eq!(network.layers, loaded_network.layers);
        assert_eq!(network.weights.len(), loaded_network.weights.len());
        assert_eq!(network.learning_rate, loaded_network.learning_rate);
        assert_eq!(network.momentum, loaded_network.momentum);

        // Test that both networks produce the same output
        let test_input = vec![0.5, 0.5];
        let original_output = network.feed_forward(Matrix::from(test_input.clone()));
        let loaded_output = loaded_network.feed_forward(Matrix::from(test_input));

        for i in 0..original_output.data.len() {
            assert_relative_eq!(
                original_output.data[i],
                loaded_output.data[i],
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_different_activations_per_layer() {
        let mut config = NetworkConfig::default();
        config.layers = vec![2, 3, 2];
        config.activations = vec![ActivationType::Sigmoid, ActivationType::Softmax];
        config.learning_rate = 0.5;
        config.momentum = Some(0.8);
        let mut network = Network::new(&config);

        // Test forward propagation
        let input = vec![0.5, 0.3].into_matrix(2, 1);
        let output = network.feed_forward(input);

        // Verify output dimensions
        assert_eq!(output.rows, 2);
        assert_eq!(output.cols, 1);

        // Verify output is valid (sums to 1 due to softmax)
        let sum: f64 = output.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Test backpropagation with matching dimensions
        let target = vec![1.0, 0.0].into_matrix(2, 1);
        network.back_propagate(output, target);

        // Verify weights were updated
        assert!(!network.weights[0].data.iter().all(|&x| x == 0.0));
        assert!(!network.weights[1].data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_multiple_hidden_layers_with_softmax() {
        let mut config = NetworkConfig::default();
        // Create a network with 4 layers: input -> hidden1 -> hidden2 -> output
        config.layers = vec![2, 4, 3, 2];
        // Use different activation functions for each layer transition
        config.activations = vec![
            ActivationType::Sigmoid, // input -> hidden1
            ActivationType::Softmax, // hidden1 -> hidden2
            ActivationType::Sigmoid, // hidden2 -> output
        ];
        config.learning_rate = 0.1;
        config.momentum = Some(0.9);
        let mut network = Network::new(&config);

        // Test forward propagation
        let input = vec![0.5, 0.3].into_matrix(2, 1);
        let output = network.feed_forward(input);

        // Verify output dimensions
        assert_eq!(output.rows, 2);
        assert_eq!(output.cols, 1);

        // Verify output is valid (between 0 and 1 due to sigmoid)
        for &x in output.data.iter() {
            assert!(x >= 0.0 && x <= 1.0);
        }

        // Test backpropagation with matching dimensions
        let target = vec![1.0, 0.0].into_matrix(2, 1);
        network.back_propagate(output, target);

        // Verify all weight matrices were updated
        for (i, weights) in network.weights.iter().enumerate() {
            assert!(
                !weights.data.iter().all(|&x| x == 0.0),
                "Weights for layer {} were not updated",
                i
            );
        }

        // Verify intermediate layer outputs
        assert_eq!(network.data.len(), 4); // input + 3 layer outputs
        assert_eq!(network.data[1].rows, 4); // hidden1 layer size
        assert_eq!(network.data[2].rows, 3); // hidden2 layer size
        assert_eq!(network.data[3].rows, 2); // output layer size

        // Verify hidden2 layer outputs sum to 1 (softmax property)
        let hidden2_sum: f64 = network.data[2].data.iter().sum();
        assert!(
            (hidden2_sum - 1.0).abs() < 1e-10,
            "Hidden2 layer outputs should sum to 1 (softmax), got {}",
            hidden2_sum
        );
    }

    #[test]
    fn test_calculate_gradients() {
        // Test with Sigmoid activation
        let sigmoid = Box::new(Sigmoid) as Box<dyn ActivationFunction + Send + Sync>;
        let outputs = vec![0.7, 0.3].into_matrix(2, 1);
        let errors = vec![0.1, -0.1].into_matrix(2, 1);

        let gradients = Network::calculate_gradients(&sigmoid, &outputs, &errors);

        // For sigmoid, gradient = error * output * (1 - output)
        assert_eq!(gradients.rows, 2);
        assert_eq!(gradients.cols, 1);
        assert_relative_eq!(gradients.data[0], 0.1 * 0.7 * 0.3, epsilon = 1e-10);
        assert_relative_eq!(gradients.data[1], -0.1 * 0.3 * 0.7, epsilon = 1e-10);

        // Test with Softmax activation
        let softmax = Box::new(Softmax) as Box<dyn ActivationFunction + Send + Sync>;
        let outputs = vec![0.7, 0.3].into_matrix(2, 1);
        let errors = vec![0.1, -0.1].into_matrix(2, 1);

        let gradients = Network::calculate_gradients(&softmax, &outputs, &errors);

        // For softmax with cross-entropy loss, gradient = error directly
        assert_eq!(gradients.rows, 2);
        assert_eq!(gradients.cols, 1);
        assert_relative_eq!(gradients.data[0], 0.1, epsilon = 1e-10);
        assert_relative_eq!(gradients.data[1], -0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_gradients_edge_cases() {
        // Test with sigmoid(0) = 0.5 outputs
        let sigmoid = Box::new(Sigmoid) as Box<dyn ActivationFunction + Send + Sync>;
        let outputs = vec![0.5, 0.5].into_matrix(2, 1); // sigmoid(0) = 0.5
        let errors = vec![0.1, -0.1].into_matrix(2, 1);

        let gradients = Network::calculate_gradients(&sigmoid, &outputs, &errors);

        // For sigmoid at x=0, output is 0.5 and derivative is 0.25
        assert_relative_eq!(gradients.data[0], 0.1 * 0.5 * 0.5, epsilon = 1e-10);
        assert_relative_eq!(gradients.data[1], -0.1 * 0.5 * 0.5, epsilon = 1e-10);

        // Test with one output (edge case for softmax)
        let softmax = Box::new(Softmax) as Box<dyn ActivationFunction + Send + Sync>;
        let outputs = vec![1.0].into_matrix(1, 1);
        let errors = vec![0.1].into_matrix(1, 1);

        let gradients = Network::calculate_gradients(&softmax, &outputs, &errors);

        // For softmax with single output, gradient = error directly
        assert_eq!(gradients.rows, 1);
        assert_eq!(gradients.cols, 1);
        assert_relative_eq!(gradients.data[0], 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_gradients_large_values() {
        // Test with large values to ensure numerical stability
        let sigmoid = Box::new(Sigmoid) as Box<dyn ActivationFunction + Send + Sync>;
        let outputs = vec![0.999, 0.001].into_matrix(2, 1);
        let errors = vec![1.0, -1.0].into_matrix(2, 1);

        let gradients = Network::calculate_gradients(&sigmoid, &outputs, &errors);

        // Should handle large values without numerical instability
        assert!(gradients.data[0].is_finite());
        assert!(gradients.data[1].is_finite());
        assert!(gradients.data[0] > 0.0);
        assert!(gradients.data[1] < 0.0);
    }
}
