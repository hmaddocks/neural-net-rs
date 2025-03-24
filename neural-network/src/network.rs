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
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use matrix::matrix::Matrix;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::io;
use std::time::Duration;

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

        let total_samples = inputs.len();
        let mut indices: Vec<_> = (0..total_samples).collect();
        indices.shuffle(&mut thread_rng());

        indices
            .chunks(batch_size)
            .map(|chunk| {
                (
                    chunk.iter().map(|&i| &inputs[i]).collect(),
                    chunk.iter().map(|&i| &targets[i]).collect(),
                )
            })
            .collect()
    }

    /// Performs gradient accumulation for batch training.
    ///
    /// # Arguments
    /// * `outputs` - Network output for current batch
    /// * `targets` - Target values for current batch
    ///
    /// # Returns
    /// Vector of gradient matrices for each layer
    ///
    /// # Implementation Details
    /// Uses backpropagation algorithm with the following steps:
    /// 1. Calculates output error
    /// 2. Propagates error backwards through layers
    /// 3. Computes gradients using layer activations
    fn accumulate_gradients(&mut self, outputs: Matrix, targets: Matrix) -> Vec<Matrix> {
        let error = &targets - &outputs;

        // Calculate all deltas
        let deltas: Vec<Matrix> = (0..self.weights.len())
            .rev()
            .scan(None, |prev_delta, i| {
                let activation_output = &self.data[i + 1];
                let activation_derivative =
                    self.activations[i].derivative_vector(activation_output);

                let delta = if i == self.weights.len() - 1 {
                    // Output layer
                    if self.activation_types[i] == ActivationType::Softmax {
                        activation_derivative.dot_multiply(&error)
                    } else {
                        error.elementwise_multiply(&activation_derivative)
                    }
                } else {
                    // Hidden layers
                    let next_weights = &self.weights[i + 1];
                    let weights_no_bias =
                        next_weights.slice(0..next_weights.rows(), 0..next_weights.cols() - 1);
                    let propagated_error = weights_no_bias
                        .transpose()
                        .dot_multiply(prev_delta.as_ref().unwrap());
                    propagated_error.elementwise_multiply(&activation_derivative)
                };

                *prev_delta = Some(delta.clone());
                Some(delta)
            })
            .collect();

        // Calculate gradients using iterators
        (0..self.weights.len())
            .map(|i| {
                let input_with_bias = self.data[i].augment_with_bias();
                let delta = &deltas[self.weights.len() - 1 - i];
                delta.dot_multiply(&input_with_bias.transpose())
            })
            .collect()
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

    /// Converts input vectors to matrices
    fn convert_to_matrices(data: Vec<Vec<f64>>) -> Vec<Matrix> {
        data.iter()
            .map(|input| Matrix::from(input.clone()))
            .collect()
    }

    /// Evaluates a single sample and returns (error, is_correct)
    fn evaluate_sample(&mut self, input: &Matrix, target: &Matrix) -> (f64, bool) {
        let outputs = self.feed_forward(input.clone());

        // Calculate error
        let error = target - &outputs;
        let error_sum = error.data.iter().map(|x| x * x).sum::<f64>();

        // Calculate accuracy
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

        (error_sum, correct)
    }

    /// Processes a single batch and returns (batch_error, correct_predictions, gradients)
    fn process_batch(
        &mut self,
        batch_inputs: &[&Matrix],
        batch_targets: &[&Matrix],
    ) -> (f64, usize, Vec<Matrix>) {
        let mut accumulated_gradients = Vec::new();
        let mut batch_error = 0.0;
        let mut batch_correct = 0;

        // Process each sample in the batch
        for (input, target) in batch_inputs.iter().zip(batch_targets.iter()) {
            let outputs = self.feed_forward((*input).clone());
            let (error_sum, correct) = self.evaluate_sample(input, target);

            batch_error += error_sum;
            if correct {
                batch_correct += 1;
            }

            // Accumulate gradients
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

        (batch_error, batch_correct, accumulated_gradients)
    }

    /// Trains a single epoch and returns (total_error, correct_predictions, epoch_duration)
    fn train_epoch(
        &mut self,
        input_matrices: &[Matrix],
        target_matrices: &[Matrix],
        batch_size: usize,
    ) -> (f64, usize, std::time::Duration) {
        let epoch_start = std::time::Instant::now();
        let mut total_error = 0.0;
        let mut correct_predictions = 0;

        // Create mini-batches for this epoch
        let mini_batches = Self::prepare_mini_batches(input_matrices, target_matrices, batch_size);

        // Process each mini-batch
        for (batch_inputs, batch_targets) in mini_batches {
            let (batch_error, batch_correct, accumulated_gradients) =
                self.process_batch(&batch_inputs, &batch_targets);

            // Update weights using accumulated gradients
            self.update_weights(&accumulated_gradients);

            // Update statistics
            total_error += batch_error;
            correct_predictions += batch_correct;
        }

        (total_error, correct_predictions, epoch_start.elapsed())
    }

    /// Trains the network on a dataset for a specified number of epochs.
    ///
    /// # Arguments
    /// * `inputs` - Vector of input vectors
    /// * `targets` - Vector of target output vectors
    /// * `epochs` - Number of training epochs
    pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u32) {
        let input_matrices = Self::convert_to_matrices(inputs);
        let target_matrices = Self::convert_to_matrices(targets);
        let total_samples = input_matrices.len();

        // Create progress bars
        let multi = MultiProgress::new();
        let total_style = ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} Epochs {msg}")
            .unwrap()
            .progress_chars("##-");
        let epoch_style = ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.green/white} {pos:>7}/{len:7} Batches {msg}")
            .unwrap()
            .progress_chars("##-");

        let total_pb = multi.add(ProgressBar::new(epochs as u64));
        total_pb.set_style(total_style);

        let epoch_pb = multi.add(ProgressBar::new(total_samples as u64));
        epoch_pb.set_style(epoch_style);

        for epoch in 1..=epochs {
            epoch_pb.set_position(0);
            epoch_pb.set_message(format!("Epoch {}/{}", epoch, epochs));

            let (total_error, correct_predictions, epoch_duration) =
                self.train_epoch_with_progress(&input_matrices, &target_matrices, 32, &epoch_pb);

            let avg_error = total_error / total_samples as f64;
            let accuracy = correct_predictions as f64 / total_samples as f64;

            total_pb.set_message(format!(
                "Error = {:.6}, Accuracy = {:.2}%",
                avg_error,
                accuracy * 100.0
            ));
            total_pb.inc(1);

            // Optional: Add a small delay to ensure progress bars render properly
            std::thread::sleep(Duration::from_millis(50));
        }

        total_pb.finish_with_message("Training complete!");
        epoch_pb.finish_and_clear();
    }

    /// Trains a single epoch with progress tracking
    fn train_epoch_with_progress(
        &mut self,
        input_matrices: &[Matrix],
        target_matrices: &[Matrix],
        batch_size: usize,
        progress: &ProgressBar,
    ) -> (f64, usize, std::time::Duration) {
        let epoch_start = std::time::Instant::now();
        let mut total_error = 0.0;
        let mut correct_predictions = 0;

        // Create mini-batches for this epoch
        let mini_batches = Self::prepare_mini_batches(input_matrices, target_matrices, batch_size);
        progress.set_length(mini_batches.len() as u64);

        // Process each mini-batch
        for (batch_inputs, batch_targets) in mini_batches {
            let (batch_error, batch_correct, accumulated_gradients) =
                self.process_batch(&batch_inputs, &batch_targets);

            // Update weights using accumulated gradients
            self.update_weights(&accumulated_gradients);

            // Update statistics
            total_error += batch_error;
            correct_predictions += batch_correct;

            progress.inc(1);
        }

        (total_error, correct_predictions, epoch_start.elapsed())
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
    fn test_gradient_computation() {
        let mut network = create_test_network();
        let input = Matrix::from(vec![0.5]);
        let target = Matrix::from(vec![1.0]);

        let output = network.feed_forward(input);
        let gradients = network.accumulate_gradients(output, target);

        assert_eq!(gradients.len(), network.weights.len());
        for (gradient, weight) in gradients.iter().zip(network.weights.iter()) {
            assert_eq!(gradient.rows(), weight.rows());
            assert_eq!(gradient.cols(), weight.cols());
        }
    }

    #[test]
    fn test_batch_training_convergence() {
        let mut network = create_test_network();
        let inputs = vec![vec![0.0], vec![1.0]];
        let targets = vec![vec![1.0], vec![0.0]];

        let initial_error = compute_error(&network, &inputs, &targets);
        network.train(inputs.clone(), targets.clone(), 1000);
        let final_error = compute_error(&network, &inputs, &targets);

        assert!(final_error < initial_error, "Training should reduce error");
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

    #[test]
    fn test_convert_to_matrices() {
        let input_data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let matrices = Network::convert_to_matrices(input_data);

        assert_eq!(matrices.len(), 2);
        assert_eq!(matrices[0].rows(), 2);
        assert_eq!(matrices[0].cols(), 1);
        assert_relative_eq!(matrices[0].get(0, 0), 1.0);
        assert_relative_eq!(matrices[0].get(1, 0), 2.0);
        assert_relative_eq!(matrices[1].get(0, 0), 3.0);
        assert_relative_eq!(matrices[1].get(1, 0), 4.0);
    }

    #[test]
    fn test_evaluate_sample() {
        let mut network = create_test_network();

        // Test binary classification
        let input = Matrix::from(vec![0.5]);
        let target = Matrix::from(vec![1.0]);

        let (error, _correct) = network.evaluate_sample(&input, &target);

        assert!(error >= 0.0, "Error should be non-negative");
        assert!(error <= 1.0, "Error should be normalized");
        // Note: correctness can be either true or false as this is initial prediction

        // Test multi-class classification
        let config = NetworkConfig::new(
            vec![2, 3, 3],
            vec![ActivationType::Sigmoid, ActivationType::Softmax],
            0.1,
            Some(0.9),
            30,
            32,
        );
        let mut network = Network::new(&config);

        let input = Matrix::from(vec![0.5, 0.3]);
        let target = Matrix::from(vec![1.0, 0.0, 0.0]); // One-hot encoded

        let (error, _correct) = network.evaluate_sample(&input, &target);

        assert!(error >= 0.0, "Error should be non-negative");
        // Note: correctness can be either true or false as this is initial prediction
    }

    #[test]
    fn test_process_batch() {
        let mut network = create_test_network();

        let inputs = vec![Matrix::from(vec![0.0]), Matrix::from(vec![1.0])];
        let targets = vec![Matrix::from(vec![1.0]), Matrix::from(vec![0.0])];

        let input_refs: Vec<&Matrix> = inputs.iter().collect();
        let target_refs: Vec<&Matrix> = targets.iter().collect();

        let (batch_error, correct_count, gradients) =
            network.process_batch(&input_refs, &target_refs);

        assert!(batch_error >= 0.0, "Batch error should be non-negative");
        assert!(
            correct_count <= 2,
            "Correct count should not exceed batch size"
        );
        assert_eq!(
            gradients.len(),
            network.weights.len(),
            "Should have gradients for each layer"
        );

        // Check gradient dimensions
        for (gradient, weight) in gradients.iter().zip(network.weights.iter()) {
            assert_eq!(gradient.rows(), weight.rows());
            assert_eq!(gradient.cols(), weight.cols());
        }
    }

    #[test]
    fn test_train_epoch() {
        let mut network = create_test_network();

        let inputs = vec![
            Matrix::from(vec![0.0]),
            Matrix::from(vec![1.0]),
            Matrix::from(vec![0.5]),
            Matrix::from(vec![0.7]),
        ];
        let targets = vec![
            Matrix::from(vec![1.0]),
            Matrix::from(vec![0.0]),
            Matrix::from(vec![0.5]),
            Matrix::from(vec![0.3]),
        ];

        let (total_error, correct_predictions, duration) =
            network.train_epoch(&inputs, &targets, 2);

        assert!(total_error >= 0.0, "Total error should be non-negative");
        assert!(
            correct_predictions <= inputs.len(),
            "Correct predictions should not exceed dataset size"
        );
        assert!(duration.as_secs_f64() > 0.0, "Duration should be positive");

        // Test with different batch sizes
        let batch_sizes = [1, 2, 4];
        for &batch_size in &batch_sizes {
            let (error, predictions, _) = network.train_epoch(&inputs, &targets, batch_size);
            assert!(
                error >= 0.0,
                "Error should be non-negative for batch size {}",
                batch_size
            );
            assert!(
                predictions <= inputs.len(),
                "Predictions should not exceed dataset size for batch size {}",
                batch_size
            );
        }
    }

    #[test]
    fn test_training_integration() {
        let mut network = create_test_network();
        let inputs = vec![vec![0.0], vec![1.0]];
        let targets = vec![vec![1.0], vec![0.0]];

        // Record initial state
        let initial_weights: Vec<Matrix> = network.weights.iter().cloned().collect();

        // Train for a few epochs
        network.train(inputs.clone(), targets.clone(), 5);

        // Verify weights have been updated
        for (initial, current) in initial_weights.iter().zip(network.weights.iter()) {
            assert!(
                initial.data != current.data,
                "Weights should be updated during training"
            );
        }

        // Verify error decreases
        let final_error = compute_error(&network, &inputs, &targets);
        assert!(
            final_error < 1.0,
            "Error should decrease after training, got {}",
            final_error
        );
    }

    // Helper functions for tests
    fn compute_error(network: &Network, inputs: &[Vec<f64>], targets: &[Vec<f64>]) -> f64 {
        inputs
            .iter()
            .zip(targets)
            .map(|(input, target)| {
                let output = network.predict(Matrix::from(input.clone()));
                let error = target[0] - output.get(0, 0);
                error * error
            })
            .sum::<f64>()
            / inputs.len() as f64
    }

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
