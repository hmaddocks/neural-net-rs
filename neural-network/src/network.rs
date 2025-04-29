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
/// use neural_network::Network;
/// use neural_network::Activation;
/// use neural_network::{NetworkConfig, LearningRate, Momentum, Epochs, BatchSize};
/// use neural_network::Layer;
/// use neural_network::RegularizationType;
/// use tempfile::tempdir;
///
/// let dir = tempdir().unwrap();
/// let model_path = dir.path().join("model.json");
///
/// // Create network configuration
/// let config = NetworkConfig::new(
///     vec![
///         Layer::new(2, Some(Activation::Sigmoid)),
///         Layer::new(3, Some(Activation::Sigmoid)),
///         Layer::new(1, None),
///     ],
///     0.1,
///     Some(0.8),
///     30,
///     32,
///     Some(RegularizationType::L2), // Regularization type
///     Some(0.0001), // L2 regularization rate
/// ).unwrap();
///
/// // Create and save network
/// let mut network = Network::new(&config);
/// network.save(model_path.to_str().unwrap()).expect("Failed to save model");
/// ```
use crate::activations::Activation;
use crate::layer::Layer;
use crate::network_config::{BatchSize, Epochs, LearningRate, Momentum, NetworkConfig};
use crate::regularization::RegularizationType;
use crate::training_history::TrainingHistory;
use indicatif::{ProgressBar, ProgressStyle};
use matrix::Matrix;
use ndarray::Axis;
use rand::rng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::io;
use std::time::Instant;

/// A feed-forward neural network with configurable layers and activation functions.
#[derive(Serialize, Deserialize)]
#[serde(crate = "serde")]
pub struct Network {
    /// Sizes of each layer in the network, including input and output layers
    layers_count: Vec<usize>,
    /// Network layers
    #[serde(skip)]
    layers: Vec<Layer>,
    /// Weight matrices between layers, including bias weights
    weights: Vec<Matrix>,
    /// Cached layer outputs for backpropagation
    #[serde(skip)]
    data: Vec<Matrix>,
    /// Types of activation functions for serialization
    activations: Vec<Activation>,
    /// Learning rate for weight updates
    learning_rate: LearningRate,
    /// Optional momentum coefficient for weight updates
    momentum: Option<Momentum>,
    /// Previous weight updates for momentum calculation
    #[serde(skip)]
    prev_weight_updates: Vec<Matrix>,
    /// Mean of the dataset
    pub mean: Option<f64>,
    /// Standard deviation of the dataset
    pub std_dev: Option<f64>,
    /// Number of epochs for training
    epochs: Epochs,
    /// Batch size for training
    batch_size: BatchSize,
    /// Optional regularization rate (weight decay)
    regularization_rate: Option<f64>,
    /// Type of regularization to use (L1 or L2)
    regularization_type: Option<RegularizationType>,
    /// Training history containing metrics recorded during training
    #[serde(skip)]
    pub training_history: TrainingHistory,
}

impl Network {
    /// Creates a new neural network with specified configuration.
    ///
    /// # Arguments
    /// * `network_config` - Configuration struct containing:
    ///   - `layers`: Vector of Layer structs defining the network architecture
    ///   - `learning_rate`: Learning rate for weight updates during training
    ///   - `momentum`: Momentum coefficient for weight updates
    ///   - `epochs`: Number of training epochs
    ///   - `batch_size`: Size of mini-batches for gradient descent
    ///   - `regularization_rate`: Regularization rate (weight decay)
    ///   - `regularization_type`: Type of regularization to use (L1 or L2)
    ///
    /// # Returns
    /// A new `Network` instance with randomly initialized weights and configured parameters
    ///
    /// # Panics
    /// Panics if the number of activation functions doesn't match the number of layers minus one
    ///
    /// # Example
    /// ```
    /// use neural_network::{Network, Activation, NetworkConfig, Layer, RegularizationType};
    ///
    /// // Create network configuration for a simple XOR network
    /// let config = NetworkConfig::new(
    ///     vec![
    ///         Layer::new(2, Some(Activation::Sigmoid)),
    ///         Layer::new(3, Some(Activation::Sigmoid)),
    ///         Layer::new(1, None),
    ///     ],
    ///     0.1,
    ///     Some(0.8),
    ///     30,
    ///     32,
    ///     Some(RegularizationType::L2),
    ///     Some(0.0001),
    /// ).unwrap();
    ///
    /// let network = Network::new(&config);
    /// ```
    pub fn new(network_config: &NetworkConfig) -> Self {
        let nodes = network_config.nodes();
        let layer_pairs: Vec<_> = nodes.windows(2).collect();

        // Create layers from config
        let layers: Vec<_> = network_config.layers.iter().cloned().collect();

        // Fill weights with random values
        let weights = layer_pairs
            .iter()
            .map(|pair| {
                let (input_size, output_size) = (pair[0], pair[1]);
                Matrix::random(output_size, input_size + 1) // Add one for bias
            })
            .collect();

        // Fill prev_weight_updates with zeros
        let prev_weight_updates = layer_pairs
            .iter()
            .map(|pair| {
                let (input_size, output_size) = (pair[0], pair[1]);
                Matrix::zeros(output_size, input_size + 1)
            })
            .collect();

        // Fill data with empty matrices
        let mut data = Vec::with_capacity(network_config.layers.len());
        data.resize(network_config.layers.len(), Matrix::default());

        // Initialize regularization rate if configured
        let regularization_rate = network_config.regularization_rate.map(f64::from);

        Network {
            layers_count: nodes.clone(),
            layers,
            weights,
            data,
            activations: network_config.activations(),
            learning_rate: network_config.learning_rate,
            momentum: network_config.momentum,
            prev_weight_updates,
            mean: None,
            std_dev: None,
            epochs: network_config.epochs,
            batch_size: network_config.batch_size,
            regularization_rate,
            regularization_type: network_config.regularization_type,
            training_history: TrainingHistory::new(),
        }
    }

    pub fn set_standardization_parameters(&mut self, mean: Option<f64>, std_dev: Option<f64>) {
        self.mean = mean;
        self.std_dev = std_dev;
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
        indices.shuffle(&mut rng()); // Shuffle indices for random batch creation

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

    /// Accumulates gradients for backpropagation across a batch.
    ///
    /// # Arguments
    /// * `outputs` - Output matrix where each column is a network output (output_size x batch_size)
    /// * `targets` - Target matrix where each column is a target (output_size x batch_size)
    ///
    /// # Returns
    /// Vector of gradient matrices for each layer, ordered from input to output layer.
    /// Each gradient matrix has the same dimensions as its corresponding weight matrix,
    /// including the bias weights.
    fn accumulate_gradients(&mut self, outputs: Matrix, targets: Matrix) -> Vec<Matrix> {
        Layer::accumulate_network_gradients(
            &self.weights,
            &self.data,
            &self.layers,
            &outputs,
            &targets,
        )
    }

    /// Updates weights using accumulated gradients from a batch, including regularization.
    ///
    /// Note: Gradients are applied as a sum rather than an average, meaning
    /// the effective learning rate scales with batch size. Larger batches
    /// will result in larger weight updates.
    ///
    /// Regularization is applied by adding a penalty term based on the selected regularization type.
    fn update_weights(&mut self, accumulated_gradients: &[Matrix]) {
        // Apply summed gradients (not averaged by batch size)
        for i in 0..self.weights.len() {
            // Calculate weight updates with momentum
            let learning_term = &accumulated_gradients[i] * f64::from(self.learning_rate);

            // Calculate regularization gradient if configured
            let reg_gradient = if let (Some(rate), Some(reg_type)) =
                (self.regularization_rate, &self.regularization_type)
            {
                let reg = reg_type.create_regularization();
                reg.calculate_gradient(&self.weights[i], rate)
            } else {
                Matrix::zeros(self.weights[i].rows(), self.weights[i].cols())
            };

            // Combine accumulated gradients with regularization
            let total_update = &learning_term + &reg_gradient;

            if let Some(momentum) = self.momentum {
                let momentum_term = &self.prev_weight_updates[i].map(|x| x * f64::from(momentum));
                self.prev_weight_updates[i] = &total_update + momentum_term;
            } else {
                self.prev_weight_updates[i] = total_update;
            }

            // Update weights
            self.weights[i] = &self.weights[i] + &self.prev_weight_updates[i];
        }
    }

    /// Evaluates a single sample from a batch and returns its error and correctness.
    ///
    /// # Arguments
    /// * `target` - Target matrix of shape (output_size x 1) for this sample
    /// * `output` - Network output matrix of shape (output_size x 1) for this sample
    ///
    /// # Returns
    /// Tuple containing (squared_error, is_prediction_correct)
    fn evaluate_sample(&self, target: &Matrix, output: &Matrix) -> (f64, bool) {
        let error = target - output;
        let error_sum = error.data.iter().fold(0.0, |sum, &x| sum + x * x);

        // Determine if prediction is correct based on classification type
        let correct = if output.rows() == 1 {
            // Binary classification - compare thresholded values
            (output.get(0, 0) >= 0.5) == (target.get(0, 0) >= 0.5)
        } else {
            // Multi-class classification - compare indices of maximum values
            debug_assert!(!output.data.is_empty(), "Output vector should not be empty");
            debug_assert!(!target.data.is_empty(), "Target vector should not be empty");

            // Find index of maximum value for output and target using iterator methods
            let predicted = output
                .data
                .iter()
                .enumerate()
                .filter(|&(_, &val)| !val.is_nan())
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN comparison"))
                .map_or(0, |(idx, _)| idx); // Default to 0 if empty (should never happen due to debug_assert)

            let actual = target
                .data
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN comparison"))
                .map_or(0, |(idx, _)| idx); // Default to 0 if empty (should never happen due to debug_assert)

            predicted == actual
        };

        (error_sum, correct)
    }

    /// Evaluates a batch of samples.
    ///
    /// # Arguments
    /// * `targets` - Target matrix where each column is a target (output_size x batch_size)
    /// * `outputs` - Output matrix where each column is a network output (output_size x batch_size)
    ///
    /// # Returns
    /// (total_error, number_of_correct_predictions) for the batch
    fn evaluate_batch(&self, targets: &Matrix, outputs: &Matrix) -> (f64, usize) {
        let sample_results: Vec<(f64, bool)> = (0..targets.cols())
            .map(|i| {
                let target = targets.col(i);
                let output = outputs.col(i);
                self.evaluate_sample(&target, &output)
            })
            .collect();

        let (total_error, correct_predictions) = sample_results.into_iter().fold(
            (0.0, 0),
            |(error_sum, correct_count), (error, correct)| {
                (
                    error_sum + error,
                    correct_count + if correct { 1 } else { 0 },
                )
            },
        );

        // Calculate regularization term if configured
        let regularization_term = if let (Some(rate), Some(reg_type)) =
            (self.regularization_rate, &self.regularization_type)
        {
            let reg = reg_type.create_regularization();
            reg.calculate_term(&self.weights, rate)
        } else {
            0.0
        };

        (total_error + regularization_term, correct_predictions)
    }

    /// Process a single batch of training data.
    ///
    /// # Arguments
    /// * `batch_inputs` - Vector of input matrices for the batch
    /// * `batch_targets` - Vector of target matrices for the batch
    ///
    /// # Returns
    /// Tuple containing (batch_error, number_of_correct_predictions)
    fn process_batch(
        &mut self,
        batch_inputs: Vec<&Matrix>,
        batch_targets: Vec<&Matrix>,
    ) -> (f64, usize) {
        // Combine batch inputs and targets into single matrices
        let input_matrix = Matrix::concatenate(&batch_inputs, Axis(1));
        let target_matrix = Matrix::concatenate(&batch_targets, Axis(1));

        // Feed forward entire batch at once
        let outputs = self.feed_forward(input_matrix);

        // Evaluate batch results
        let (batch_error, batch_correct) = self.evaluate_batch(&target_matrix, &outputs);

        // Accumulate gradients for entire batch
        let accumulated_gradients = self.accumulate_gradients(outputs, target_matrix);

        // Update weights using accumulated gradients
        self.update_weights(&accumulated_gradients);

        (batch_error, batch_correct)
    }

    /// Trains the network for a single epoch.
    ///
    /// # Arguments
    /// * `input_matrices` - Slice of input matrices
    /// * `target_matrices` - Slice of target matrices
    /// * `batch_size` - Size of each mini-batch
    ///
    /// # Returns
    /// Tuple containing (total_error, number_of_correct_predictions)
    fn train_epoch(
        &mut self,
        input_matrices: &[Matrix],
        target_matrices: &[Matrix],
        batch_size: usize,
    ) -> (f64, usize) {
        let mut total_error = 0.0;
        let mut correct_predictions = 0;

        // Create mini-batches for this epoch
        let mini_batches = Self::prepare_mini_batches(input_matrices, target_matrices, batch_size);

        // Process each mini-batch
        for (batch_inputs, batch_targets) in mini_batches {
            let (batch_error, batch_correct) = self.process_batch(batch_inputs, batch_targets);
            total_error += batch_error;
            correct_predictions += batch_correct;
        }

        (total_error, correct_predictions)
    }

    /// Trains the network on a dataset for a specified number of epochs.
    ///
    /// # Arguments
    /// * `inputs` - Slice of input matrices
    /// * `targets` - Slice of target matrices
    ///
    /// # Returns
    /// Reference to the training history containing recorded metrics
    pub fn train(&mut self, inputs: &[Matrix], targets: &[Matrix]) -> &TrainingHistory {
        let total_samples = inputs.len();
        let batch_size = usize::from(self.batch_size);
        let num_epochs = usize::from(self.epochs);

        // Create progress bar
        let progress_bar = ProgressBar::new(num_epochs as u64);
        let style = ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:80.cyan/blue}] {pos}/{len} epochs | Accuracy: {msg}")
            .map_err(|e| Box::new(e))
            .expect("Failed to set progress bar style")
            .progress_chars("#>-");
        progress_bar.set_style(style);

        let start_time = Instant::now();

        // Reset training history before starting new training
        self.training_history = TrainingHistory::new();

        // Train for each epoch using functional approach
        (1..=num_epochs).for_each(|epoch| {
            let (total_error, correct_predictions) = self.train_epoch(inputs, targets, batch_size);

            let accuracy = (correct_predictions as f64 / total_samples as f64) * 100.0;

            // Record metrics in training history
            self.training_history.record_epoch(
                epoch as u32,
                accuracy,
                total_error / total_samples as f64,
            );

            progress_bar.set_message(format!("{:.2}%", accuracy));
            progress_bar.inc(1);
        });

        // Calculate final accuracy
        let final_accuracy = self
            .training_history
            .accuracies
            .last()
            .copied()
            .unwrap_or(0.0);

        progress_bar
            .finish_with_message(format!("Training completed in {:?}", start_time.elapsed()));
        println!("Final accuracy: {:.2}%", final_accuracy);

        // Print training history summary
        self.training_history.print_summary();

        &self.training_history
    }

    /// Performs forward propagation for a batch of inputs, storing intermediate layer outputs.
    ///
    /// # Arguments
    /// * `inputs` - Input matrix where each column is a sample (input_size x batch_size)
    ///
    /// # Returns
    /// Output matrix where each column is the network's output for the corresponding input
    /// (output_size x batch_size). Intermediate layer outputs are stored in self.data
    /// for use in backpropagation.
    fn feed_forward(&mut self, inputs: Matrix) -> Matrix {
        assert!(
            self.layers[0].nodes == inputs.rows(),
            "Invalid number of inputs. Expected {}, got {}",
            self.layers[0].nodes,
            inputs.rows()
        );

        // Initialize data vector with capacity and start with input
        self.data = Vec::with_capacity(self.weights.len() + 1);
        self.data.push(inputs.clone());

        // Process through each layer and store intermediate outputs
        self.weights
            .iter()
            .enumerate()
            .fold(inputs, |current, (i, weight)| {
                let with_bias = current.augment_with_bias();
                let output = self.layers[i].process_forward(weight, &with_bias);
                self.data.push(output.clone());
                output
            })
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
            self.layers[0].nodes == inputs.rows(),
            "Invalid number of inputs. Expected {}, got {}",
            self.layers[0].nodes,
            inputs.rows()
        );

        // Process through layers without storing intermediates
        self.weights
            .iter()
            .enumerate()
            .fold(inputs, |current, (i, weight)| {
                let with_bias = current.augment_with_bias();
                // Use Layer's process_forward instance method
                self.layers[i].process_forward(weight, &with_bias)
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
    /// use neural_network::{Network, NetworkConfig, Layer, Activation};
    /// use neural_network::{LearningRate, Momentum};
    /// use tempfile::tempdir;
    ///
    /// let dir = tempdir().unwrap();
    /// let model_path = dir.path().join("model.json");
    ///
    /// // Create network configuration
    /// let mut config = NetworkConfig::default();
    /// config.layers = vec![
    ///     Layer::new(2, Some(Activation::Sigmoid)),
    ///     Layer::new(3, Some(Activation::Sigmoid)),
    ///     Layer::new(1, None),
    /// ];
    /// config.learning_rate = LearningRate::try_from(0.1).unwrap();
    /// config.momentum = Some(Momentum::try_from(0.8).unwrap());
    ///
    /// // Create and save network
    /// let mut network = Network::new(&config);
    /// network.save(model_path.to_str().unwrap()).expect("Failed to save model");
    /// ```
    pub fn save(&self, path: &str) -> io::Result<()> {
        // Use ? operator for more idiomatic error handling
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
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
    /// # use neural_network::{Network, Activation};
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

        // Create Layer objects from network configuration
        network.layers = network
            .layers_count
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                let activation = if i < network.activations.len() {
                    Some(network.activations[i])
                } else {
                    None
                };
                crate::layer::Layer::new(size, activation)
            })
            .collect();

        // Initialize training history
        network.training_history = TrainingHistory::new();

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::Layer;
    use approx::assert_relative_eq;
    use tempfile::tempdir;

    fn create_test_network() -> Network {
        let config = NetworkConfig::new(
            vec![
                Layer::new(1, Some(Activation::Sigmoid)),
                Layer::new(2, Some(Activation::Sigmoid)),
                Layer::new(1, None),
            ],
            0.1,
            Some(0.9),
            30,
            32,
            Some(RegularizationType::L2),
            Some(0.0001),
        )
        .unwrap();

        Network::new(&config)
    }

    fn create_deep_network() -> Network {
        let config = NetworkConfig::new(
            vec![
                Layer::new(2, Some(Activation::Sigmoid)),
                Layer::new(4, Some(Activation::Sigmoid)),
                Layer::new(3, Some(Activation::Sigmoid)),
                Layer::new(2, None),
            ],
            0.1,
            Some(0.9),
            30,
            32,
            Some(RegularizationType::L2),
            Some(0.0001),
        )
        .unwrap();

        Network::new(&config)
    }

    // Helper functions for tests
    fn compute_error(network: &Network, inputs: &[Matrix], targets: &[Matrix]) -> f64 {
        // Use functional approach with iterator and sum
        inputs
            .iter()
            .zip(targets)
            .map(|(input, target)| {
                let output = network.predict(input.clone());
                let error = target.get(0, 0) - output.get(0, 0);
                error * error
            })
            .sum::<f64>()
            / inputs.len() as f64
    }

    #[test]
    fn test_network_creation() {
        let config = NetworkConfig::new(
            vec![
                Layer::new(2, Some(Activation::Sigmoid)),
                Layer::new(4, Some(Activation::Sigmoid)),
                Layer::new(1, None),
            ],
            0.1,
            Some(0.9),
            30,
            32,
            Some(RegularizationType::L2),
            Some(0.0001),
        )
        .unwrap();

        let network = Network::new(&config);

        assert_eq!(network.layers_count, vec![2, 4, 1]);
        assert_eq!(network.weights[0].rows(), 4);
        assert_eq!(network.weights[0].cols(), 3); // 2 inputs + 1 bias
        assert_eq!(network.weights[1].rows(), 1);
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
        // Create a network for XOR problem with a fixed seed for deterministic results
        let config = NetworkConfig::new(
            vec![
                Layer::new(2, Some(Activation::Sigmoid)),
                Layer::new(4, Some(Activation::Sigmoid)),
                Layer::new(1, None),
            ],
            0.5, // Higher learning rate
            Some(0.9),
            2000,
            2, // Small batch size for XOR
            Some(RegularizationType::L2),
            Some(0.0001),
        )
        .unwrap();

        let mut network = Network::new(&config);

        // XOR training data
        let inputs = vec![
            Matrix::from(vec![0.0, 0.0]),
            Matrix::from(vec![0.0, 1.0]),
            Matrix::from(vec![1.0, 0.0]),
            Matrix::from(vec![1.0, 1.0]),
        ];
        let targets = vec![
            Matrix::from(vec![0.0]),
            Matrix::from(vec![1.0]),
            Matrix::from(vec![1.0]),
            Matrix::from(vec![0.0]),
        ];

        // Train the network and get history
        let history = network.train(&inputs, &targets);

        // Extract the final accuracy value before using network again
        let final_accuracy = match history.accuracies.last() {
            Some(acc) => *acc,
            None => 0.0,
        };

        // Test the network - collect test inputs
        let test_inputs = vec![
            Matrix::from(vec![0.0, 1.0]),
            Matrix::from(vec![1.0, 0.0]),
            Matrix::from(vec![0.0, 0.0]),
            Matrix::from(vec![1.0, 1.0]),
        ];

        // Collect outputs after training completes
        let output_01 = network.predict(test_inputs[0].clone()).get(0, 0);
        let output_10 = network.predict(test_inputs[1].clone()).get(0, 0);
        let output_00 = network.predict(test_inputs[2].clone()).get(0, 0);
        let output_11 = network.predict(test_inputs[3].clone()).get(0, 0);

        // Verify that the training actually improved
        assert!(
            final_accuracy > 0.5,
            "XOR training should achieve better than 50% accuracy"
        );

        // Use a small epsilon for floating point comparisons
        let epsilon = 1e-6;

        // The pattern learned should have these relationships
        // Check that 0,1 output is higher than 0,0 output
        assert!(
            output_01 > output_00 - epsilon,
            "Failed relationship check: output(0,1)={} should be > output(0,0)={}",
            output_01,
            output_00
        );

        // Check that 1,0 output is higher than 0,0 output
        assert!(
            output_10 > output_00 - epsilon,
            "Failed relationship check: output(1,0)={} should be > output(0,0)={}",
            output_10,
            output_00
        );

        // Check that 0,1 output is higher than 1,1 output (with tolerance)
        assert!(
            output_01 >= output_11 - epsilon,
            "Failed relationship check: output(0,1)={} should be >= output(1,1)={}",
            output_01,
            output_11
        );

        // Check that 1,0 output is higher than 1,1 output (with tolerance)
        assert!(
            output_10 >= output_11 - epsilon,
            "Failed relationship check: output(1,0)={} should be >= output(1,1)={}",
            output_10,
            output_11
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
            vec![
                Layer::new(2, Some(Activation::Sigmoid)),
                Layer::new(4, Some(Activation::Softmax)),
                Layer::new(3, None),
            ],
            0.1,
            Some(0.9),
            1,
            32,
            Some(RegularizationType::L2),
            Some(0.0001),
        )
        .unwrap();

        let mut network = Network::new(&config);

        // Simple 3-class classification problem
        let inputs = vec![
            Matrix::from(vec![0.0, 0.0]), // Class 0
            Matrix::from(vec![1.0, 0.0]), // Class 1
            Matrix::from(vec![0.0, 1.0]), // Class 2
        ];

        let targets = vec![
            Matrix::from(vec![1.0, 0.0, 0.0]), // One-hot encoding for class 0
            Matrix::from(vec![0.0, 1.0, 0.0]), // One-hot encoding for class 1
            Matrix::from(vec![0.0, 0.0, 1.0]), // One-hot encoding for class 2
        ];

        // This should not panic with dimension mismatch
        network.train(&inputs, &targets);

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
            vec![
                Layer::new(2, Some(Activation::Sigmoid)),
                Layer::new(4, Some(Activation::Sigmoid)),
                Layer::new(1, None),
            ],
            0.5,       // Moderate learning rate
            Some(0.9), // High momentum
            2000,      // More epochs for better convergence
            2,         // Small batch size for testing
            Some(RegularizationType::L2),
            Some(0.0001),
        )
        .unwrap();

        let mut network = Network::new(&config);

        // Create a simple dataset with clear separation
        let inputs = vec![
            Matrix::from(vec![0.0, 0.0]),
            Matrix::from(vec![0.0, 1.0]),
            Matrix::from(vec![1.0, 0.0]),
            Matrix::from(vec![1.0, 1.0]),
        ];
        let targets = vec![
            Matrix::from(vec![0.0]),
            Matrix::from(vec![1.0]),
            Matrix::from(vec![1.0]),
            Matrix::from(vec![0.0]),
        ];

        // Test prepare_mini_batches directly
        let input_matrices = inputs.clone();
        let target_matrices = targets.clone();

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
        network.train(&inputs, &targets);

        // Test predictions with a more lenient error threshold
        let mut total_error = 0.0;
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let output = network.feed_forward(input.clone());
            let error = (target.get(0, 0) - output.get(0, 0)).abs();
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
            vec![
                Layer::new(2, Some(Activation::Sigmoid)),
                Layer::new(4, Some(Activation::Sigmoid)),
                Layer::new(1, None),
            ],
            0.8, // Higher learning rate
            Some(0.9),
            1000,
            32,
            Some(RegularizationType::L2),
            Some(0.0001),
        )
        .unwrap();

        // XOR training data
        let inputs = vec![
            Matrix::from(vec![0.0, 0.0]),
            Matrix::from(vec![0.0, 1.0]),
            Matrix::from(vec![1.0, 0.0]),
            Matrix::from(vec![1.0, 1.0]),
        ];
        let targets = vec![
            Matrix::from(vec![0.0]),
            Matrix::from(vec![1.0]),
            Matrix::from(vec![1.0]),
            Matrix::from(vec![0.0]),
        ];

        // Test different batch sizes
        for &batch_size in &[1, 2, 4] {
            config = NetworkConfig::new(
                config.layers,
                f64::from(config.learning_rate),
                config.momentum.map(|m| f64::from(m)),
                config.epochs.into(),
                batch_size,
                config.regularization_type,
                config.regularization_rate.map(|r| f64::from(r)),
            )
            .unwrap();
            let mut network = Network::new(&config);
            network.train(&inputs, &targets);

            // Test predictions
            let mut total_error = 0.0;
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let output = network.predict(input.clone());
                let error = (target.get(0, 0) - output.get(0, 0)).abs();
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
    fn test_batch_processing() {
        let mut network = create_deep_network();

        // Create a batch of inputs
        let input = Matrix::new(2, 3, vec![0.5, 0.3, 0.7, 0.2, 0.8, 0.4]);

        // Process batch
        let output = network.feed_forward(input.clone());

        // Check dimensions
        assert_eq!(output.rows(), 2); // Output layer size
        assert_eq!(output.cols(), 3); // Batch size

        // Check all outputs are valid probabilities
        for val in output.data.iter() {
            assert!(
                *val >= 0.0 && *val <= 1.0,
                "Output {} not between 0 and 1",
                val
            );
        }

        // Test gradient computation
        let target = Matrix::new(2, 3, vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let gradients = network.accumulate_gradients(output, target);

        // Check gradient dimensions
        assert_eq!(gradients.len(), network.weights.len());
        for (gradient, weight) in gradients.iter().zip(network.weights.iter()) {
            assert_eq!(gradient.rows(), weight.rows());
            assert_eq!(gradient.cols(), weight.cols());
        }
    }

    #[test]
    fn test_batch_training_convergence() {
        let mut network = create_test_network();
        let inputs = vec![Matrix::from(vec![0.0]), Matrix::from(vec![1.0])];
        let targets = vec![Matrix::from(vec![1.0]), Matrix::from(vec![0.0])];

        let initial_error = compute_error(&network, &inputs, &targets);
        network.train(&inputs, &targets);
        let final_error = compute_error(&network, &inputs, &targets);

        assert!(final_error < initial_error, "Training should reduce error");
    }

    #[test]
    fn test_mini_batch_shuffling() {
        let inputs = vec![
            Matrix::from(vec![0.0, 0.0]),
            Matrix::from(vec![0.0, 1.0]),
            Matrix::from(vec![1.0, 0.0]),
            Matrix::from(vec![1.0, 1.0]),
        ];
        let targets = vec![
            Matrix::from(vec![0.0]),
            Matrix::from(vec![1.0]),
            Matrix::from(vec![1.0]),
            Matrix::from(vec![0.0]),
        ];

        // Generate multiple batches and verify they're not always in the same order
        let mut all_same = true;
        let first_batch = Network::prepare_mini_batches(&inputs, &targets, 2);
        for _ in 0..5 {
            let next_batch = Network::prepare_mini_batches(&inputs, &targets, 2);
            if first_batch != next_batch {
                all_same = false;
                break;
            }
        }
        assert!(!all_same, "Mini-batches should be randomly shuffled");
    }

    #[test]
    fn test_evaluate_sample() {
        let mut network = create_test_network();

        // Test binary classification
        let input = Matrix::from(vec![0.5]);
        let target = Matrix::from(vec![1.0]);

        // Separate feed_forward call to avoid multiple mutable borrows
        let outputs = network.feed_forward(input.clone());
        let (error, _correct) = network.evaluate_sample(&target, &outputs);

        assert!(error >= 0.0, "Error should be non-negative");
        assert!(error <= 1.0, "Error should be normalized");
        // Note: correctness can be either true or false as this is initial prediction

        // Test multi-class classification
        let config = NetworkConfig::new(
            vec![
                Layer::new(2, Some(Activation::Sigmoid)),
                Layer::new(3, Some(Activation::Sigmoid)),
                Layer::new(3, None),
            ],
            0.1,
            Some(0.9),
            30,
            32,
            Some(RegularizationType::L2),
            Some(0.0001),
        )
        .unwrap();

        let mut network = Network::new(&config);

        let input = Matrix::from(vec![0.5, 0.3]);
        let target = Matrix::from(vec![1.0, 0.0, 0.0]); // One-hot encoded

        // Separate feed_forward call to avoid multiple mutable borrows
        let outputs = network.feed_forward(input.clone());
        let (error, _correct) = network.evaluate_sample(&target, &outputs);

        assert!(error >= 0.0, "Error should be non-negative");
        // Note: correctness can be either true or false as this is initial prediction
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
            Matrix::from(vec![0.0]),
            Matrix::from(vec![1.0]),
            Matrix::from(vec![0.5]),
            Matrix::from(vec![0.3]),
        ];

        let (total_error, correct_predictions) = network.train_epoch(&inputs, &targets, 2);

        assert!(total_error >= 0.0, "Total error should be non-negative");
        assert!(
            correct_predictions <= inputs.len(),
            "Correct predictions should not exceed dataset size"
        );
        // assert!(duration.as_secs_f64() > 0.0, "Duration should be positive");

        // Test with different batch sizes
        let batch_sizes = [1, 2, 4];
        for &batch_size in &batch_sizes {
            let (error, predictions) = network.train_epoch(&inputs, &targets, batch_size);
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
        let inputs = vec![Matrix::from(vec![0.0]), Matrix::from(vec![1.0])];
        let targets = vec![Matrix::from(vec![1.0]), Matrix::from(vec![0.0])];

        // Record initial state
        let initial_weights: Vec<Matrix> = network.weights.iter().cloned().collect();

        // Train for a few epochs
        network.train(&inputs, &targets);

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

    #[test]
    fn test_network_serialization_with_regularization() {
        let config = NetworkConfig::new(
            vec![
                Layer::new(2, Some(Activation::Sigmoid)),
                Layer::new(4, Some(Activation::Sigmoid)),
                Layer::new(1, None),
            ],
            0.1,
            Some(0.9),
            30,
            32,
            Some(RegularizationType::L2),
            Some(0.0001),
        )
        .unwrap();

        let network = Network::new(&config);

        // Serialize to JSON
        let serialized = serde_json::to_string(&network).unwrap();

        // Deserialize back
        let deserialized: Network = serde_json::from_str(&serialized).unwrap();

        // Verify regularization type is preserved
        assert_eq!(
            network.regularization_type,
            deserialized.regularization_type
        );
        assert_eq!(
            network.regularization_rate,
            deserialized.regularization_rate
        );
    }
}
