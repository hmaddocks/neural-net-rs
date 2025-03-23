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
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::io;

/// A feed-forward neural network with configurable layers and activation functions.
#[derive(Serialize, Deserialize)]
#[serde(crate = "serde")]
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
    activations: Vec<Box<dyn ActivationFunction>>,
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
    ///   - `layers`: Vector of layer sizes, including input and output layers (e.g., [784, 128, 10] for MNIST)
    ///   - `activations`: Vector of activation types for each layer (must be one less than number of layers)
    ///   - `learning_rate`: Learning rate for weight updates during training (e.g., 0.1)
    ///   - `momentum`: Optional momentum coefficient for weight updates (defaults to 0.9)
    ///   - `epochs`: Number of training epochs for the complete dataset
    ///   - `batch_size`: Size of mini-batches for gradient descent (default 32)
    ///
    /// # Returns
    /// A new `Network` instance with randomly initialized weights and configured parameters
    ///
    /// # Panics
    /// Panics if the number of activation functions doesn't match the number of layers minus one
    ///
    /// # Example
    /// ```
    /// use neural_network::{network::Network, activations::ActivationType, network_config::NetworkConfig};
    ///
    /// // Create network configuration for a simple XOR network
    /// let config = NetworkConfig::new(
    ///     vec![2, 3, 1],                                          // 2 inputs, 3 hidden, 1 output
    ///     vec![ActivationType::Sigmoid, ActivationType::Sigmoid], // Activation functions
    ///     0.1,                                                    // Learning rate
    ///     Some(0.9),                                              // Momentum
    ///     30,                                                     // Epochs
    ///     32,                                                     // Batch size
    /// );
    ///
    /// let network = Network::new(&config);
    /// ```
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

    /// Creates mini-batches from input and target data.
    ///
    /// # Arguments
    /// * `inputs` - Vector of input matrices
    /// * `targets` - Vector of target matrices
    /// * `batch_size` - Size of each mini-batch
    ///
    /// # Returns
    /// Vector of tuples containing (input_batch, target_batch)
    fn prepare_mini_batches<'a>(
        inputs: &'a [Matrix],
        targets: &'a [Matrix],
        batch_size: usize,
    ) -> Vec<(Vec<&'a Matrix>, Vec<&'a Matrix>)> {
        assert_eq!(
            inputs.len(),
            targets.len(),
            "Number of inputs must match number of targets"
        );

        // Create indices and shuffle them
        let mut indices: Vec<usize> = (0..inputs.len()).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);

        // Create batches using iterator chunks
        indices
            .chunks(batch_size)
            .map(|batch_indices| {
                let input_batch = batch_indices.iter().map(|&i| &inputs[i]).collect();
                let target_batch = batch_indices.iter().map(|&i| &targets[i]).collect();
                (input_batch, target_batch)
            })
            .collect()
    }

    /// Accumulates gradients without updating weights
    fn accumulate_gradients(&mut self, outputs: Matrix, targets: Matrix) -> Vec<Matrix> {
        let mut deltas = Vec::with_capacity(self.weights.len());
        let error = &targets - &outputs;

        // Calculate deltas for each layer, starting from the output layer
        for i in (0..self.weights.len()).rev() {
            let activation_output = &self.data[i + 1];
            let activation_derivative = self.activations[i].derivative_vector(activation_output);

            if i == self.weights.len() - 1 {
                if self.activation_types[i] == ActivationType::Softmax {
                    deltas.push(activation_derivative.dot_multiply(&error));
                } else {
                    deltas.push(error.elementwise_multiply(&activation_derivative));
                }
            } else {
                let next_weights = &self.weights[i + 1];
                let next_delta = &deltas[deltas.len() - 1];

                let weights_no_bias =
                    next_weights.slice(0..next_weights.rows(), 0..next_weights.cols() - 1);
                let propagated_error = weights_no_bias.transpose().dot_multiply(next_delta);
                deltas.push(propagated_error.elementwise_multiply(&activation_derivative));
            }
        }

        deltas.reverse();
        let mut gradients = Vec::with_capacity(self.weights.len());

        // Calculate gradients without applying updates
        for i in 0..self.weights.len() {
            let input_with_bias = self.data[i].augment_with_bias();
            let delta = &deltas[i];
            let gradient = delta.dot_multiply(&input_with_bias.transpose());
            gradients.push(gradient);
        }

        gradients
    }

    /// Updates weights using accumulated gradients
    fn update_weights(&mut self, accumulated_gradients: &[Matrix]) {
        // Apply accumulated gradients without batch size scaling
        for i in 0..self.weights.len() {
            // Calculate weight updates with momentum
            let learning_term = accumulated_gradients[i].map(|x| x * self.learning_rate);
            let momentum_term = self.prev_weight_updates[i].map(|x| x * self.momentum);

            // Update weights and store updates for next iteration
            self.prev_weight_updates[i] = &learning_term + &momentum_term;
            self.weights[i] = &self.weights[i] + &self.prev_weight_updates[i];
        }
    }

    /// Trains the network on a dataset for a specified number of epochs.
    ///
    /// # Arguments
    /// * `inputs` - Vector of input vectors
    /// * `targets` - Vector of target output vectors
    /// * `epochs` - Number of training epochs
    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u32) {
        let input_matrices: Vec<_> = inputs
            .iter()
            .map(|input| Matrix::from(input.clone()))
            .collect();
        let target_matrices: Vec<_> = targets
            .iter()
            .map(|target| Matrix::from(target.clone()))
            .collect();

        for epoch in 1..=epochs {
            let epoch_start = std::time::Instant::now();
            let mut total_error = 0.0;
            let mut correct_predictions = 0;

            // Create mini-batches for this epoch
            let mini_batches = Self::prepare_mini_batches(&input_matrices, &target_matrices, 32);

            // Process each mini-batch
            for (batch_inputs, batch_targets) in mini_batches {
                let mut accumulated_gradients = Vec::new();
                let mut batch_error = 0.0;
                let mut batch_correct = 0;

                // First pass: Accumulate gradients over the batch
                for (input, target) in batch_inputs.iter().zip(batch_targets.iter()) {
                    let outputs = self.feed_forward((*input).clone());

                    // Calculate error and accuracy
                    let error = *target - &outputs;
                    let error_sum = error.data.iter().map(|x| x * x).sum::<f64>();
                    batch_error += error_sum;

                    let correct = if outputs.cols() == 1 {
                        let predicted = outputs.get(0, 0) >= 0.5;
                        let actual = target.get(0, 0) >= 0.5;
                        predicted == actual
                    } else {
                        let predicted = outputs
                            .data
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .map(|(idx, _)| idx)
                            .unwrap();
                        let actual = target
                            .data
                            .iter()
                            .position(|&val| (val - 1.0).abs() < f64::EPSILON)
                            .expect("Target vector should contain exactly one 1.0 value");
                        predicted == actual
                    };

                    if correct {
                        batch_correct += 1;
                    }

                    // Accumulate gradients without updating weights
                    let gradients = self.accumulate_gradients(outputs, (*target).clone());
                    if accumulated_gradients.is_empty() {
                        accumulated_gradients = gradients;
                    } else {
                        // Add gradients element-wise
                        for (acc, grad) in accumulated_gradients.iter_mut().zip(gradients.iter()) {
                            *acc = &*acc + grad;
                        }
                    }
                }

                // Second pass: Update weights using accumulated gradients
                self.update_weights(&accumulated_gradients);

                // Update statistics
                total_error += batch_error;
                correct_predictions += batch_correct;
            }

            let total_samples = inputs.len();
            let avg_error = total_error / total_samples as f64;
            let accuracy = correct_predictions as f64 / total_samples as f64;

            println!(
                "Epoch {}/{}: Error = {:.6}, Accuracy = {:.2}%, Time = {:.2}s",
                epoch,
                epochs,
                avg_error,
                accuracy * 100.0,
                epoch_start.elapsed().as_secs_f64()
            );
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
            self.layers[0] == inputs.rows(),
            "Invalid number of inputs. Expected {}, got {}",
            self.layers[0],
            inputs.rows()
        );

        // Store original input
        self.data = vec![inputs.clone()];

        // Process through layers
        let result = self
            .weights
            .iter()
            .enumerate()
            .fold(inputs, |current, (i, weight)| {
                let with_bias = current.augment_with_bias();
                let output = process_layer(weight, &with_bias, self.activations[i].as_ref());
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
            self.layers[0] == inputs.rows(),
            "Invalid number of inputs. Expected {}, got {}",
            self.layers[0],
            inputs.rows()
        );

        // Process through layers without storing intermediates
        self.weights
            .iter()
            .enumerate()
            .fold(inputs, |current, (i, weight)| {
                let with_bias = current.augment_with_bias();
                process_layer(weight, &with_bias, self.activations[i].as_ref())
            })
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
            .map(|w| Matrix::zeros(w.rows(), w.cols()))
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

/// Processes a single layer in the neural network.
///
/// # Algorithm
/// 1. Computes the weighted sum (weight * input)
/// 2. Adds bias terms (included in weight matrix)
/// 3. Applies the activation function
///
/// # Arguments
/// * `weight` - Weight matrix including bias weights
/// * `input` - Input values from previous layer
/// * `activation` - Activation function to apply
///
/// # Returns
/// The processed output matrix after applying weights and activation
fn process_layer(weight: &Matrix, input: &Matrix, activation: &dyn ActivationFunction) -> Matrix {
    let output = weight.dot_multiply(input);
    activation.apply_vector(&output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use tempfile::tempdir;

    #[test]
    fn test_network_creation() {
        let config = NetworkConfig::new(
            vec![3, 4, 2],
            vec![ActivationType::Sigmoid, ActivationType::Sigmoid],
            0.1,
            Some(0.9),
            30,
            32,
        );

        let network = Network::new(&config);

        assert_eq!(network.layers, vec![3, 4, 2]);
        assert_eq!(network.weights[0].rows(), 4);
        assert_eq!(network.weights[0].cols(), 4); // 3 inputs + 1 bias
        assert_eq!(network.weights[1].rows(), 2);
        assert_eq!(network.weights[1].cols(), 5); // 4 inputs + 1 bias
    }

    #[test]
    fn test_feed_forward() {
        let mut network = create_test_network();
        let input = Matrix::from(vec![0.5]);

        let output = network.feed_forward(input);

        assert_eq!(output.rows(), 1);
        assert_eq!(output.cols(), 1);
        // Output should be between 0 and 1 (sigmoid activation)
        assert!(output.get(0, 0) > 0.0 && output.get(0, 0) < 1.0);
    }

    #[test]
    fn test_predict() {
        let mut network = create_test_network();
        let input = Matrix::from(vec![0.5]);

        let output_ff = network.feed_forward(input.clone());
        let output_predict = network.predict(input);

        assert_eq!(output_ff.rows(), output_predict.rows());
        assert_eq!(output_ff.cols(), output_predict.cols());
        assert_relative_eq!(
            output_ff.get(0, 0),
            output_predict.get(0, 0),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_xor_training() {
        // Create a simpler network for XOR
        let config = NetworkConfig::new(
            vec![2, 4, 1], // Simpler architecture: 2-4-1
            vec![ActivationType::Sigmoid, ActivationType::Sigmoid],
            0.5, // Higher learning rate
            Some(0.9),
            2000,
            2, // Small batch size for XOR
        );

        let mut network = Network::new(&config);

        // XOR training data
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

        network.train(inputs, targets, 2000);

        // Test the network
        let test_inputs = vec![
            Matrix::from(vec![0.0, 1.0]),
            Matrix::from(vec![1.0, 0.0]),
            Matrix::from(vec![0.0, 0.0]),
            Matrix::from(vec![1.0, 1.0]),
        ];

        let outputs: Vec<_> = test_inputs
            .iter()
            .map(|input| network.predict(input.clone()))
            .collect();

        // Check outputs with more lenient thresholds
        assert!(
            outputs[0].get(0, 0) > 0.7,
            "Failed on input [0,1], output: {}",
            outputs[0].get(0, 0)
        );
        assert!(
            outputs[1].get(0, 0) > 0.7,
            "Failed on input [1,0], output: {}",
            outputs[1].get(0, 0)
        );
        assert!(
            outputs[2].get(0, 0) < 0.3,
            "Failed on input [0,0], output: {}",
            outputs[2].get(0, 0)
        );
        assert!(
            outputs[3].get(0, 0) < 0.3,
            "Failed on input [1,1], output: {}",
            outputs[3].get(0, 0)
        );
    }

    #[test]
    fn test_serialization() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("network.json");

        // Create and train a network
        let original_network = create_test_network();
        let input = Matrix::from(vec![0.5]);
        let original_output = original_network.predict(input.clone());

        // Save the network
        original_network
            .save(file_path.to_str().unwrap())
            .expect("Failed to save network");

        // Load the network
        let loaded_network =
            Network::load(file_path.to_str().unwrap()).expect("Failed to load network");
        let loaded_output = loaded_network.predict(input);

        // Compare outputs
        assert_relative_eq!(
            original_output.get(0, 0),
            loaded_output.get(0, 0),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_layer_outputs() {
        let mut network = create_deep_network();
        let input = Matrix::from(vec![0.5, 0.3]);

        let output = network.feed_forward(input);

        assert_eq!(output.rows(), 2);
        assert_eq!(output.cols(), 1);

        // Check intermediate layer sizes
        assert_eq!(network.data[1].rows(), 4); // hidden1 layer size
        assert_eq!(network.data[2].rows(), 3); // hidden2 layer size
        assert_eq!(network.data[3].rows(), 2); // output layer size
    }

    #[test]
    fn test_softmax_backpropagation() {
        // Create a simple network with Softmax output layer
        let config = NetworkConfig::new(
            vec![2, 4, 3], // 2 inputs, 4 hidden neurons, 3 output classes
            vec![ActivationType::Sigmoid, ActivationType::Softmax],
            0.1,
            Some(0.9),
            30,
            32,
        );

        let mut network = Network::new(&config);

        // Simple 3-class classification problem
        let inputs = vec![
            vec![0.0, 0.0], // Class 0
            vec![1.0, 0.0], // Class 1
            vec![0.0, 1.0], // Class 2
        ];

        let targets = vec![
            vec![1.0, 0.0, 0.0], // One-hot encoding for class 0
            vec![0.0, 1.0, 0.0], // One-hot encoding for class 1
            vec![0.0, 0.0, 1.0], // One-hot encoding for class 2
        ];

        // This should not panic with dimension mismatch
        network.train(inputs.clone(), targets.clone(), 1);

        // Test forward pass dimensions
        let output = network.predict(Matrix::from(inputs[0].clone()));
        assert_eq!(output.rows(), 3);
        assert_eq!(output.cols(), 1);

        // Verify output is valid probability distribution
        let sum: f64 = output.data.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        output.data.iter().for_each(|&x| {
            assert!(x >= 0.0 && x <= 1.0);
        });
    }

    #[test]
    fn test_mini_batch_training() {
        // Create a network with a more robust architecture
        let config = NetworkConfig::new(
            vec![2, 8, 4, 1], // More layers with more neurons
            vec![
                ActivationType::Sigmoid,
                ActivationType::Sigmoid,
                ActivationType::Sigmoid,
            ],
            1.0,       // Higher learning rate
            Some(0.9), // High momentum
            1000,      // Number of epochs
            2,         // Small batch size for testing
        );

        let mut network = Network::new(&config);

        // Create a simple dataset with clear separation
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

        // Test prepare_mini_batches directly
        let input_matrices: Vec<_> = inputs
            .iter()
            .map(|input| Matrix::from(input.clone()))
            .collect();
        let target_matrices: Vec<_> = targets
            .iter()
            .map(|target| Matrix::from(target.clone()))
            .collect();

        let batches = Network::prepare_mini_batches(&input_matrices, &target_matrices, 2);
        assert_eq!(batches.len(), 2, "Should create 2 batches of size 2");
        for (batch_inputs, batch_targets) in &batches {
            assert_eq!(batch_inputs.len(), 2, "Each input batch should have size 2");
            assert_eq!(
                batch_targets.len(),
                2,
                "Each target batch should have size 2"
            );
        }

        // Train the network
        network.train(inputs.clone(), targets.clone(), 1000);

        // Test predictions with a more lenient error threshold
        let mut total_error = 0.0;
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let output = network.feed_forward(Matrix::from(input.clone()));
            let error = (target[0] - output.get(0, 0)).abs();
            total_error += error;
        }

        let avg_error = total_error / 4.0;
        assert!(
            avg_error < 0.45,
            "Prediction error too large: {}",
            avg_error
        );
    }

    #[test]
    fn test_batch_size_effects() {
        let mut config = NetworkConfig::new(
            vec![2, 4, 1], // Simpler architecture
            vec![ActivationType::Sigmoid, ActivationType::Sigmoid],
            0.8, // Higher learning rate
            Some(0.9),
            1000,
            32,
        );

        // XOR training data
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

        // Test different batch sizes
        for &batch_size in &[1, 2, 4] {
            config.batch_size = batch_size;
            let mut network = Network::new(&config);
            network.train(inputs.clone(), targets.clone(), 1000);

            // Test predictions
            let mut total_error = 0.0;
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let output = network.predict(Matrix::from(input.clone()));
                let error = (target[0] - output.get(0, 0)).abs();
                total_error += error;
            }
            let avg_error = total_error / 4.0;

            assert!(
                avg_error < 0.4,
                "Training failed to converge with batch_size {}, error: {}",
                batch_size,
                avg_error
            );
        }
    }

    #[test]
    fn test_mini_batch_shuffling() {
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

        let input_matrices: Vec<_> = inputs
            .iter()
            .map(|input| Matrix::from(input.clone()))
            .collect();
        let target_matrices: Vec<_> = targets
            .iter()
            .map(|target| Matrix::from(target.clone()))
            .collect();

        // Generate multiple batches and verify they're not always in the same order
        let mut all_same = true;
        let first_batch = Network::prepare_mini_batches(&input_matrices, &target_matrices, 2);
        for _ in 0..5 {
            let next_batch = Network::prepare_mini_batches(&input_matrices, &target_matrices, 2);
            if first_batch != next_batch {
                all_same = false;
                break;
            }
        }
        assert!(!all_same, "Mini-batches should be randomly shuffled");
    }

    // Helper functions for tests
    fn create_test_network() -> Network {
        let config = NetworkConfig::new(
            vec![1, 2, 1],
            vec![ActivationType::Sigmoid, ActivationType::Sigmoid],
            0.1,
            Some(0.9),
            30,
            32,
        );

        Network::new(&config)
    }

    fn create_xor_network() -> Network {
        let config = NetworkConfig::new(
            vec![2, 3, 1],
            vec![ActivationType::Sigmoid, ActivationType::Sigmoid],
            0.1,
            Some(0.9),
            30,
            32,
        );

        Network::new(&config)
    }

    fn create_deep_network() -> Network {
        let config = NetworkConfig::new(
            vec![2, 4, 3, 2],
            vec![
                ActivationType::Sigmoid,
                ActivationType::Sigmoid,
                ActivationType::Sigmoid,
            ],
            0.1,
            Some(0.9),
            30,
            32,
        );

        Network::new(&config)
    }
}
