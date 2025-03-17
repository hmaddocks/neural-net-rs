use crate::activations::Activation;
use anyhow::{Result, anyhow};
use matrix::matrix::Matrix;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// A neural network implementation with configurable layers and activation function.
///
/// This implementation provides a flexible neural network that can be configured with
/// any number of layers and neurons, as well as different activation functions.
/// The network supports training through backpropagation and can be serialized to/from JSON.
///
/// # Examples
///
/// ```
/// use neural_network::{Network, SIGMOID};
/// use matrix::matrix::Matrix;
///
/// // Create a network with 2 inputs, 3 hidden neurons, and 1 output
/// let mut network = Network::new(vec![2, 3, 1], vec![SIGMOID, SIGMOID], 0.1);
///
/// // Train the network
/// let inputs = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
/// let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
/// network.train(inputs, targets, 10000).unwrap();
///
/// // Use the trained network
/// let result = network.feed_forward(&Matrix::from(vec![1.0, 0.0])).unwrap();
/// println!("Result: {:?}", result.data());
/// ```
#[derive(Serialize, Deserialize, Clone, Builder)]
pub struct Network {
    /// The number of neurons in each layer, including input and output layers
    layers: Vec<usize>,
    /// Weight matrices between each pair of adjacent layers
    weights: Vec<Matrix>,
    /// Bias matrices for each layer (except the input layer)
    biases: Vec<Matrix>,
    /// Cached activations for each layer during forward propagation
    #[serde(skip)]
    data: Vec<Matrix>,
    /// The activation functions used by each layer (except input layer)
    activations: Vec<Activation>,
    /// The learning rate used during training
    learning_rate: f64,
}

impl Network {
    /// Creates a new neural network with the specified layer sizes, activation functions, and learning rate.
    ///
    /// # Parameters
    ///
    /// * `layers` - A vector of layer sizes, where the first element is the number of inputs,
    ///              the last element is the number of outputs, and any elements in between
    ///              represent hidden layers.
    /// * `activations` - A vector of activation functions to use for each layer (except input layer).
    ///                   Must have length equal to layers.len() - 1.
    /// * `learning_rate` - The learning rate to use during training (typically a small value like 0.1).
    ///
    /// # Examples
    ///
    /// ```
    /// use neural_network::{Network, SIGMOID, SOFTMAX};
    ///
    /// // Create a network with sigmoid for hidden layer and softmax for output
    /// let network = Network::new(vec![2, 2, 1], vec![SIGMOID, SOFTMAX], 0.1);
    /// ```
    pub fn new(layers: Vec<usize>, activations: Vec<Activation>, learning_rate: f64) -> Self {
        assert_eq!(
            activations.len(),
            layers.len() - 1,
            "Number of activation functions must match number of layers minus 1"
        );

        let (weights, biases): (Vec<Matrix>, Vec<Matrix>) = layers
            .windows(2)
            .map(|window| {
                (
                    Matrix::random(window[1], window[0]),
                    Matrix::random(window[1], 1),
                )
            })
            .unzip();

        // Pre-allocate data vector with capacity
        let mut data = Vec::with_capacity(layers.len());
        data.resize(layers.len(), Matrix::default());

        Network {
            layers,
            weights,
            biases,
            data,
            activations,
            learning_rate,
        }
    }

    /// Performs forward propagation through the network to compute an output.
    ///
    /// This method takes an input matrix and passes it through each layer of the network,
    /// applying weights, biases, and the activation function at each step.
    ///
    /// # Parameters
    ///
    /// * `inputs` - A matrix containing the input values. The number of elements must match
    ///              the size of the input layer.
    ///
    /// # Returns
    ///
    /// * `Result<Matrix>` - The output matrix if successful, or an error if the input size is invalid.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of inputs doesn't match the size of the input layer.
    ///
    /// # Examples
    ///
    /// ```
    /// use neural_network::{Network, SIGMOID, SOFTMAX};
    /// use matrix::matrix::Matrix;
    ///
    /// let mut network = Network::new(vec![2, 3, 1], vec![SIGMOID, SIGMOID], 0.1);
    /// let input = Matrix::from(vec![0.5, 0.8]);
    /// let output = network.feed_forward(&input).unwrap();
    /// ```
    pub fn feed_forward(&mut self, inputs: &Matrix) -> Result<Matrix> {
        if self.layers[0] != inputs.data().len() {
            return Err(anyhow!(
                "Invalid number of inputs: expected {}, got {}",
                self.layers[0],
                inputs.data().len()
            ));
        }

        // Store input in data vector
        self.data[0] = inputs.clone();
        let mut current = inputs.clone();

        // Process each layer
        for i in 0..self.layers.len() - 1 {
            current = {
                let weighted = self.weights[i].dot_multiply(&current);
                let biased = weighted.add(&self.biases[i]);
                let activated = Matrix::from(self.activations[i].apply_vector(biased.data()));
                self.data[i + 1] = activated.clone();
                activated
            };
        }

        Ok(current)
    }

    /// Performs backpropagation to update weights and biases based on the error.
    ///
    /// This method implements the backpropagation algorithm, which calculates the gradient
    /// of the loss function with respect to the weights and biases, and updates them
    /// to minimize the error.
    ///
    /// # Parameters
    ///
    /// * `outputs` - The output matrix from the forward pass.
    /// * `targets` - The expected output matrix (ground truth).
    ///
    /// # Note
    ///
    /// This method should be called after `feed_forward()` to update the network based on
    /// the error between the actual output and the expected output.
    ///
    /// # Examples
    ///
    /// ```
    /// use neural_network::{Network, SIGMOID, SOFTMAX};
    /// use matrix::matrix::Matrix;
    ///
    /// let mut network = Network::new(vec![2, 3, 1], vec![SIGMOID, SIGMOID], 0.1);
    /// let input = Matrix::from(vec![0.5, 0.8]);
    /// let target = Matrix::from(vec![1.0]);
    ///
    /// let output = network.feed_forward(&input).unwrap();
    /// network.back_propogate(output, &target);
    /// ```
    pub fn back_propogate(&mut self, outputs: Matrix, targets: &Matrix) {
        let mut errors = targets.subtract(&outputs);

        // For each layer, starting from the output layer
        for i in (0..self.layers.len() - 1).rev() {
            // Calculate gradients for each node in the current layer
            let mut all_gradients = vec![0.0; self.layers[i + 1]];

            // For each node in the current layer
            for j in 0..self.layers[i + 1] {
                // Get the derivative vector for this node
                let node_derivatives =
                    self.activations[i].derivative_vector(self.data[i + 1].data(), j);

                // Update the gradient for this node using the error
                all_gradients[j] = node_derivatives[j] * errors.data()[j] * self.learning_rate;
            }

            let gradients = Matrix::from(all_gradients);

            // Update weights and biases
            let weight_deltas = gradients.dot_multiply(&self.data[i].transpose());
            self.weights[i] = self.weights[i].add(&weight_deltas);
            self.biases[i] = self.biases[i].add(&gradients);

            // Prepare for next layer
            if i > 0 {
                errors = self.weights[i].transpose().dot_multiply(&errors);
            }
        }
    }

    /// Trains the network using the provided inputs and targets for a specified number of epochs.
    ///
    /// This method performs supervised learning by repeatedly passing inputs through the network,
    /// comparing the outputs to the expected targets, and adjusting the weights and biases
    /// through backpropagation.
    ///
    /// # Parameters
    ///
    /// * `inputs` - A vector of input vectors, where each inner vector represents one training example.
    /// * `targets` - A vector of target vectors, where each inner vector represents the expected output
    ///               for the corresponding input.
    /// * `epochs` - The number of complete passes through the training data.
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Ok if training completes successfully, or an error if inputs and targets don't match.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of inputs doesn't match the number of targets.
    ///
    /// # Examples
    ///
    /// ```
    /// use neural_network::{Network, SIGMOID};
    ///
    /// // Create a network for XOR problem
    /// let mut network = Network::new(vec![2, 2, 1], vec![SIGMOID, SIGMOID], 0.1);
    ///
    /// // XOR training data
    /// let inputs = vec![
    ///     vec![0.0, 0.0],
    ///     vec![0.0, 1.0],
    ///     vec![1.0, 0.0],
    ///     vec![1.0, 1.0]
    /// ];
    /// let targets = vec![
    ///     vec![0.0],
    ///     vec![1.0],
    ///     vec![1.0],
    ///     vec![0.0]
    /// ];
    ///
    /// // Train for 10,000 epochs
    /// network.train(inputs, targets, 10000).unwrap();
    /// ```
    pub fn train(
        &mut self,
        inputs: Vec<Vec<f64>>,
        targets: Vec<Vec<f64>>,
        epochs: u32,
    ) -> Result<()> {
        if inputs.len() != targets.len() {
            return Err(anyhow!("Number of inputs must match number of targets"));
        }

        // Pre-compute matrices for inputs and targets
        let input_matrices: Vec<Matrix> = inputs.into_iter().map(Matrix::from).collect();
        let target_matrices: Vec<Matrix> = targets.into_iter().map(Matrix::from).collect();

        // Training loop
        for epoch in 1..=epochs {
            if epochs < 100 || epoch % (epochs / 100) == 0 {
                println!("Epoch {} of {}", epoch, epochs);
            }

            // Process each training example
            for (input, target) in input_matrices.iter().zip(target_matrices.iter()) {
                let outputs = self.feed_forward(input)?;
                self.back_propogate(outputs, target);
            }
        }

        Ok(())
    }

    /// Saves the network to a file in JSON format.
    ///
    /// This method serializes the network structure, weights, and biases to a JSON file,
    /// allowing it to be loaded later.
    ///
    /// # Parameters
    ///
    /// * `path` - The path where the network should be saved.
    ///
    /// # Returns
    ///
    /// * `Result<()>` - Ok if the save is successful, or an error if serialization or file writing fails.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails or if the file cannot be written.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use neural_network::{Network, SIGMOID};
    ///
    /// let network = Network::new(vec![2, 3, 1], vec![SIGMOID, SIGMOID], 0.1);
    /// network.save("my_network.json").unwrap();
    /// ```
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Loads a network from a JSON file.
    ///
    /// This method deserializes a previously saved network from a JSON file,
    /// restoring its structure, weights, and biases.
    ///
    /// # Parameters
    ///
    /// * `path` - The path to the saved network file.
    ///
    /// # Returns
    ///
    /// * `Result<Self>` - The loaded network if successful, or an error if deserialization or file reading fails.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if deserialization fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use neural_network::{Network, SIGMOID};
    ///
    /// let network = Network::load("my_network.json").unwrap();
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let mut network: Network = serde_json::from_str(&json)?;
        // Reinitialize data vector
        network.data = vec![Matrix::default(); network.layers.len()];
        Ok(network)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::SIGMOID;
    use approx::assert_relative_eq;
    use tempfile::NamedTempFile;

    /// Helper function to create a simple network for testing
    fn create_test_network() -> Network {
        Network::new(vec![2, 3, 1], vec![SIGMOID, SIGMOID], 0.1)
    }

    /// Helper function to create XOR training data
    fn create_xor_data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
        (inputs, targets)
    }

    #[test]
    fn test_network_creation() {
        let network = create_test_network();

        // Check network structure
        assert_eq!(network.layers, vec![2, 3, 1]);
        assert_eq!(network.weights.len(), 2);
        assert_eq!(network.biases.len(), 2);
        assert_eq!(network.data.len(), 3);

        // Check dimensions of weight matrices
        assert_eq!(network.weights[0].rows(), 3);
        assert_eq!(network.weights[0].cols(), 2);
        assert_eq!(network.weights[1].rows(), 1);
        assert_eq!(network.weights[1].cols(), 3);

        // Check dimensions of bias matrices
        assert_eq!(network.biases[0].rows(), 3);
        assert_eq!(network.biases[0].cols(), 1);
        assert_eq!(network.biases[1].rows(), 1);
        assert_eq!(network.biases[1].cols(), 1);
    }

    #[test]
    fn test_feed_forward() -> Result<()> {
        let mut network = create_test_network();

        // Test with valid input
        let input = Matrix::from(vec![0.5, 0.5]);
        let output = network.feed_forward(&input)?;

        // Check output dimensions
        assert_eq!(output.rows(), 1);
        assert_eq!(output.cols(), 1);

        // Test with invalid input
        let invalid_input = Matrix::from(vec![0.5, 0.5, 0.5]);
        let result = network.feed_forward(&invalid_input);
        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_back_propagation() -> Result<()> {
        let mut network = create_test_network();

        // Store initial weights and biases
        let initial_weights = network.weights.clone();
        let initial_biases = network.biases.clone();

        // Perform forward and backward pass
        let input = Matrix::from(vec![0.5, 0.5]);
        let target = Matrix::from(vec![1.0]);

        let output = network.feed_forward(&input)?;
        network.back_propogate(output, &target);

        // Verify weights and biases have changed
        for i in 0..network.weights.len() {
            assert!(network.weights[i].data() != initial_weights[i].data());
            assert!(network.biases[i].data() != initial_biases[i].data());
        }

        Ok(())
    }

    #[test]
    fn test_train_input_validation() {
        let mut network = create_test_network();

        // Test with mismatched inputs and targets
        let inputs = vec![vec![0.0, 0.0], vec![0.0, 1.0]];
        let targets = vec![vec![0.0]]; // Only one target for two inputs

        let result = network.train(inputs, targets, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_train_xor() -> Result<()> {
        let mut network = Network::new(vec![2, 4, 1], vec![SIGMOID, SIGMOID], 0.5);
        let (inputs, targets) = create_xor_data();

        // Train for a small number of epochs for testing
        network.train(inputs.clone(), targets.clone(), 1000)?;

        // Test predictions
        let test_inputs = inputs
            .iter()
            .map(|input| Matrix::from(input.clone()))
            .collect::<Vec<_>>();
        let test_targets = targets
            .iter()
            .map(|target| Matrix::from(target.clone()))
            .collect::<Vec<_>>();

        // Verify that predictions are closer to targets after training
        for (input, target) in test_inputs.iter().zip(test_targets.iter()) {
            let output = network.feed_forward(input)?;
            let error = (output.data()[0] - target.data()[0]).abs();

            // Allow some error margin since we're not training to convergence
            assert!(error < 0.4, "Error too large: {}", error);
        }

        Ok(())
    }

    #[test]
    fn test_save_and_load() -> Result<()> {
        let network = create_test_network();

        // Create a temporary file
        let temp_file = NamedTempFile::new()?;
        let temp_path = temp_file.path().to_path_buf();

        // Save network
        network.save(&temp_path)?;

        // Verify file contains data
        let file_content = fs::read_to_string(&temp_path)?;
        assert!(!file_content.is_empty());
        assert!(file_content.contains("layers"));
        assert!(file_content.contains("weights"));
        assert!(file_content.contains("biases"));

        // Load network
        let loaded_network = Network::load(&temp_path)?;

        // Verify structure is preserved
        assert_eq!(loaded_network.layers, network.layers);
        assert_eq!(loaded_network.weights.len(), network.weights.len());
        assert_eq!(loaded_network.biases.len(), network.biases.len());

        // Verify weights and biases are preserved with approximate equality
        for i in 0..network.weights.len() {
            let original_data = network.weights[i].data();
            let loaded_data = loaded_network.weights[i].data();

            assert_eq!(original_data.len(), loaded_data.len());
            for j in 0..original_data.len() {
                assert_relative_eq!(original_data[j], loaded_data[j], epsilon = 1e-10);
            }

            let original_biases = network.biases[i].data();
            let loaded_biases = loaded_network.biases[i].data();

            assert_eq!(original_biases.len(), loaded_biases.len());
            for j in 0..original_biases.len() {
                assert_relative_eq!(original_biases[j], loaded_biases[j], epsilon = 1e-10);
            }
        }

        Ok(())
    }

    #[test]
    fn test_deterministic_output() -> Result<()> {
        let mut network = create_test_network();
        let input = Matrix::from(vec![0.5, 0.5]);

        // Run feed_forward multiple times and check for consistent output
        let first_output = network.feed_forward(&input)?;

        for _ in 0..5 {
            let output = network.feed_forward(&input)?;
            assert_eq!(output.data(), first_output.data());
        }

        Ok(())
    }

    #[test]
    fn test_softmax_output() -> Result<()> {
        // Create a network with softmax output
        let mut network = Network::new(vec![2, 3], vec![crate::activations::SOFTMAX], 0.1);

        // Test input
        let input = Matrix::from(vec![0.5, -0.2]);
        let output = network.feed_forward(&input)?;

        // Verify output sums to 1.0 (softmax property)
        let sum: f64 = output.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Softmax output should sum to 1.0");

        // Verify all outputs are between 0 and 1
        for &val in output.data() {
            assert!(
                val >= 0.0 && val <= 1.0,
                "Softmax outputs should be probabilities"
            );
        }

        Ok(())
    }

    #[test]
    fn test_multi_class_training() -> Result<()> {
        // Create a network for 3-class classification
        let mut network = Network::new(
            vec![2, 4, 3], // 2 inputs, 4 hidden, 3 outputs
            vec![SIGMOID, crate::activations::SOFTMAX],
            0.1,
        );

        // Simple training data: three points in 2D space
        let test_inputs = vec![
            vec![0.0, 0.0], // Class 0
            vec![1.0, 0.0], // Class 1
            vec![0.0, 1.0], // Class 2
        ];

        let targets = vec![
            vec![1.0, 0.0, 0.0], // One-hot encoding for class 0
            vec![0.0, 1.0, 0.0], // One-hot encoding for class 1
            vec![0.0, 0.0, 1.0], // One-hot encoding for class 2
        ];

        // Train for a few epochs
        network.train(test_inputs.clone(), targets, 1000)?;

        // Test predictions
        for (i, input) in test_inputs.iter().enumerate() {
            let output = network.feed_forward(&Matrix::from(input.clone()))?;

            // Verify output is valid probability distribution
            let sum: f64 = output.data().iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "Output should sum to 1.0");

            // Verify highest probability matches target class
            let pred_class = output
                .data()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            assert_eq!(pred_class, i, "Should predict correct class after training");
        }

        Ok(())
    }
}
