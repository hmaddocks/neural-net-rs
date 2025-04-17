//! Defines the structure of a layer within the neural network.
//!
//! Each layer consists of a number of nodes and an optional activation function.
use crate::activations::{ActivationFunction, ActivationType};
use crate::matrix::Matrix;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents a single layer in the neural network.
///
/// Contains the number of nodes (neurons) in the layer and the
/// type of activation function to be applied to the output of this layer.
#[derive(Serialize, Deserialize)]
pub struct Layer {
    /// The number of nodes (neurons) in this layer.
    pub nodes: usize,
    /// The activation function applied to the output of this layer.
    /// If `None`, no activation function is applied (e.g., for the output layer).
    pub activation: Option<ActivationType>,
    /// The concrete activation function (not serialized)
    #[serde(skip)]
    activation_fn: Option<Box<dyn ActivationFunction>>,
}

// Manual Debug implementation since activation_fn doesn't implement Debug
impl std::fmt::Debug for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Layer")
            .field("nodes", &self.nodes)
            .field("activation", &self.activation)
            .finish()
    }
}

// Manual Clone implementation since Box<dyn ActivationFunction> doesn't implement Clone
impl Clone for Layer {
    fn clone(&self) -> Self {
        // Create a new Layer with cloned fields
        // activation_fn will be recreated from activation
        Self {
            nodes: self.nodes,
            activation: self.activation.clone(),
            activation_fn: self.activation.map(|a_type| a_type.create_activation()),
        }
    }
}

// Manual PartialEq implementation since Box<dyn ActivationFunction> doesn't implement PartialEq
impl PartialEq for Layer {
    fn eq(&self, other: &Self) -> bool {
        // Compare just the nodes and activation type, not the function instance
        self.nodes == other.nodes && self.activation == other.activation
    }
}

impl Layer {
    /// Creates a new [`Layer`].
    pub fn new(nodes: usize, activation: Option<ActivationType>) -> Self {
        let activation_fn = activation.map(|a_type| a_type.create_activation());
        Self {
            nodes,
            activation,
            activation_fn,
        }
    }

    /// Initialize the activation function from the activation type
    /// This is used after deserializing a Layer
    pub fn initialize_activation(&mut self) {
        self.activation_fn = self.activation.map(|a_type| a_type.create_activation());
    }

    /// Gets a reference to the activation function if available
    pub fn get_activation(&self) -> Option<&dyn ActivationFunction> {
        self.activation_fn.as_deref()
    }

    /// Process a layer in feed-forward operation
    ///
    /// # Arguments
    /// * `weight` - Weight matrix for the connections to this layer
    /// * `input` - Input matrix with bias already augmented
    ///
    /// # Returns
    /// The processed output matrix after applying weights and activation
    pub fn process_forward(&self, weight: &Matrix, input: &Matrix) -> Matrix {
        let weighted_input = weight.dot_multiply(input);

        match &self.activation_fn {
            Some(activation) => activation.apply_vector(&weighted_input),
            None => weighted_input, // No activation for output layer
        }
    }

    /// Computes the delta for a hidden layer during backpropagation
    ///
    /// # Arguments
    /// * `next_layer_weights` - Weight matrix of the next layer
    /// * `next_layer_delta` - Delta values from the next layer
    /// * `current_output` - Output values of the current layer
    ///
    /// # Returns
    /// The computed delta matrix for the current layer
    pub fn compute_hidden_delta(
        &self,
        next_layer_weights: &Matrix,
        next_layer_delta: &Matrix,
        current_output: &Matrix,
    ) -> Matrix {
        // Remove bias weights for backpropagation
        let weight_no_bias = next_layer_weights.slice(
            0..next_layer_weights.rows(),
            0..next_layer_weights.cols() - 1,
        );

        // Propagate error backward
        let propagated_error = weight_no_bias.transpose().dot_multiply(next_layer_delta);

        // Apply activation derivative if present
        if let Some(activation) = &self.activation_fn {
            // Calculate activation derivative for current layer
            let activation_derivative = activation.apply_derivative_vector(current_output);
            propagated_error.elementwise_multiply(&activation_derivative)
        } else {
            // For layers with no activation function, just return the propagated error
            propagated_error
        }
    }

    /// Computes gradients for a layer during backpropagation
    ///
    /// # Arguments
    /// * `delta` - Delta values for the current layer
    /// * `previous_output` - Output from the previous layer (with bias term)
    ///
    /// # Returns
    /// Gradient matrix for the current layer's weights
    pub fn compute_gradients(delta: &Matrix, previous_output: &Matrix) -> Matrix {
        delta.dot_multiply(&previous_output.transpose())
    }

    /// Computes the error for the output layer.
    ///
    /// # Arguments
    /// * `outputs` - Output matrix from the network
    /// * `targets` - Target matrix
    ///
    /// # Returns
    /// The error matrix for the output layer
    pub fn compute_output_error(outputs: &Matrix, targets: &Matrix) -> Matrix {
        targets - outputs
    }

    /// Accumulates gradients for backpropagation across all layers.
    ///
    /// # Arguments
    /// * `weights` - Network weight matrices
    /// * `layer_outputs` - Cached outputs from each layer
    /// * `network_layers` - Vector of network layers
    /// * `outputs` - Final output matrix
    /// * `targets` - Target matrix
    ///
    /// # Returns
    /// Vector of gradient matrices for each layer, ordered from input to output layer
    pub fn accumulate_network_gradients(
        weights: &[Matrix],
        layer_outputs: &[Matrix],
        network_layers: &[Layer],
        outputs: &Matrix,
        targets: &Matrix,
    ) -> Vec<Matrix> {
        // Calculate initial error
        let error = Self::compute_output_error(outputs, targets);

        // Calculate deltas for all layers
        let mut deltas = Vec::with_capacity(weights.len());
        deltas.push(error.clone());

        // Calculate deltas for hidden layers using functional approach
        let mut prev_delta = error;
        deltas.extend((0..weights.len() - 1).rev().map(|i| {
            let weight = &weights[i + 1];
            let current_output = &layer_outputs[i + 1];
            let layer = &network_layers[i];

            // Compute delta for hidden layer
            let delta = layer.compute_hidden_delta(weight, &prev_delta, current_output);

            prev_delta = delta.clone();
            delta
        }));

        // Calculate gradients
        (0..weights.len())
            .map(|i| {
                let input_with_bias = layer_outputs[i].augment_with_bias();
                let delta = &deltas[weights.len() - 1 - i];
                Self::compute_gradients(delta, &input_with_bias)
            })
            .collect()
    }

    /// Evaluates a prediction from the output layer and determines if it's correct.
    ///
    /// # Arguments
    /// * `target` - Target matrix of shape (output_size x 1) for this sample
    /// * `output` - Network output matrix of shape (output_size x 1) for this sample
    ///
    /// # Returns
    /// Tuple containing (squared_error, is_prediction_correct)
    pub fn evaluate_prediction(target: &Matrix, output: &Matrix) -> (f64, bool) {
        // Calculate squared error sum using functional approach
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

    /// Calculates the L2 regularization term for a set of weight matrices.
    ///
    /// # Arguments
    /// * `weights` - Vector of weight matrices
    /// * `regularization_rate` - The L2 regularization rate
    ///
    /// # Returns
    /// The L2 regularization term to be added to the error
    pub fn calculate_l2_regularization(weights: &[Matrix], regularization_rate: f64) -> f64 {
        weights
            .iter()
            .map(|w| w.data.iter().map(|&x| x * x).sum::<f64>())
            .sum::<f64>()
            * (regularization_rate / 2.0)
    }

    /// Applies L2 regularization to weight updates during backpropagation.
    ///
    /// # Arguments
    /// * `weights` - Current weight matrix
    /// * `regularization_rate` - The L2 regularization rate
    ///
    /// # Returns
    /// The L2 regularization gradient for the weights
    pub fn apply_l2_regularization(weights: &Matrix, regularization_rate: Option<f64>) -> Matrix {
        regularization_rate
            .map(|rate| weights * rate)
            .unwrap_or_else(|| Matrix::zeros(weights.rows(), weights.cols()))
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{{ nodes: {}, activation: {:?} }}",
            self.nodes, self.activation
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use matrix::matrix::IntoMatrix;

    #[test]
    fn test_layer_new_with_activation() {
        let layer = Layer::new(10, Some(ActivationType::Sigmoid));
        assert_eq!(layer.nodes, 10);
        assert_eq!(layer.activation, Some(ActivationType::Sigmoid));
        assert!(layer.activation_fn.is_some());
    }

    #[test]
    fn test_layer_new_without_activation() {
        let layer = Layer::new(5, None);
        assert_eq!(layer.nodes, 5);
        assert_eq!(layer.activation, None);
        assert!(layer.activation_fn.is_none());
    }

    #[test]
    fn test_process_forward() {
        // Create a simple layer with sigmoid activation
        let layer = Layer::new(2, Some(ActivationType::Sigmoid));

        // Create sample weights and inputs
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6].into_matrix(2, 3); // 2 output nodes, 2 input nodes + 1 bias
        let inputs = vec![0.5, 0.6, 1.0].into_matrix(3, 1); // 2 inputs + 1 bias term

        // Process the layer
        let output = layer.process_forward(&weights, &inputs);

        // Verify output dimensions
        assert_eq!(output.rows(), 2);
        assert_eq!(output.cols(), 1);

        // Manual calculation for verification:
        // First node: (0.1 * 0.5) + (0.2 * 0.6) + (0.3 * 1.0) = 0.05 + 0.12 + 0.3 = 0.47
        // Sigmoid(0.47) ≈ 0.6154
        // Second node: (0.4 * 0.5) + (0.5 * 0.6) + (0.6 * 1.0) = 0.2 + 0.3 + 0.6 = 1.1
        // Sigmoid(1.1) ≈ 0.7503

        assert_relative_eq!(output.get(0, 0), 0.6154, epsilon = 1e-4);
        assert_relative_eq!(output.get(1, 0), 0.7503, epsilon = 1e-4);
    }

    #[test]
    fn test_process_forward_without_activation() {
        // Create a layer without activation function (like an output layer)
        let layer = Layer::new(2, None);

        // Create sample weights and inputs
        let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6].into_matrix(2, 3);
        let inputs = vec![0.5, 0.6, 1.0].into_matrix(3, 1);

        // Process the layer
        let output = layer.process_forward(&weights, &inputs);

        // Without activation, output should just be the weighted sum
        // First node: (0.1 * 0.5) + (0.2 * 0.6) + (0.3 * 1.0) = 0.05 + 0.12 + 0.3 = 0.47
        // Second node: (0.4 * 0.5) + (0.5 * 0.6) + (0.6 * 1.0) = 0.2 + 0.3 + 0.6 = 1.1
        assert_relative_eq!(output.get(0, 0), 0.47, epsilon = 1e-10);
        assert_relative_eq!(output.get(1, 0), 1.1, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_hidden_delta() {
        // Create layer with sigmoid activation
        let layer = Layer::new(2, Some(ActivationType::Sigmoid));

        // Create test data
        let next_weights = vec![0.1, 0.2, 0.3].into_matrix(1, 3); // 1 output node, 2 input nodes + 1 bias
        let next_delta = vec![0.5].into_matrix(1, 1);
        let current_output = vec![0.6, 0.7].into_matrix(2, 1);

        // Compute delta
        let delta = layer.compute_hidden_delta(&next_weights, &next_delta, &current_output);

        // Verify output dimensions
        assert_eq!(delta.rows(), 2);
        assert_eq!(delta.cols(), 1);

        // Verify the calculations
        // Weight_no_bias = [0.1, 0.2]
        // Propagated error = [0.1, 0.2]ᵀ * [0.5] = [0.05, 0.1]
        // Derivative = [0.6 * (1-0.6), 0.7 * (1-0.7)] = [0.24, 0.21]
        // Delta = [0.05, 0.1] .* [0.24, 0.21] = [0.012, 0.021]

        assert_relative_eq!(delta.get(0, 0), 0.012, epsilon = 1e-4);
        assert_relative_eq!(delta.get(1, 0), 0.021, epsilon = 1e-4);
    }

    #[test]
    fn test_compute_hidden_delta_without_activation() {
        // Create layer without activation
        let layer = Layer::new(2, None);

        // Create test data
        let next_weights = vec![0.1, 0.2, 0.3].into_matrix(1, 3);
        let next_delta = vec![0.5].into_matrix(1, 1);
        let current_output = vec![0.6, 0.7].into_matrix(2, 1);

        // Compute delta
        let delta = layer.compute_hidden_delta(&next_weights, &next_delta, &current_output);

        // Verify dimensions
        assert_eq!(delta.rows(), 2);
        assert_eq!(delta.cols(), 1);

        // Without activation, delta is just the propagated error
        // Weight_no_bias = [0.1, 0.2]
        // delta = [0.1, 0.2]ᵀ * [0.5] = [0.05, 0.1]
        assert_relative_eq!(delta.get(0, 0), 0.05, epsilon = 1e-10);
        assert_relative_eq!(delta.get(1, 0), 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_gradients() {
        // Create test data
        let delta = vec![0.1, 0.2].into_matrix(2, 1);
        let prev_output = vec![0.5, 0.6, 1.0].into_matrix(3, 1); // Includes bias

        // Compute gradients
        let gradients = Layer::compute_gradients(&delta, &prev_output);

        // Verify dimensions
        assert_eq!(gradients.rows(), 2);
        assert_eq!(gradients.cols(), 3);

        // Verify calculations (same as before)
        assert_relative_eq!(gradients.get(0, 0), 0.05, epsilon = 1e-10);
        assert_relative_eq!(gradients.get(0, 1), 0.06, epsilon = 1e-10);
        assert_relative_eq!(gradients.get(0, 2), 0.1, epsilon = 1e-10);
        assert_relative_eq!(gradients.get(1, 0), 0.1, epsilon = 1e-10);
        assert_relative_eq!(gradients.get(1, 1), 0.12, epsilon = 1e-10);
        assert_relative_eq!(gradients.get(1, 2), 0.2, epsilon = 1e-10);
    }

    #[test]
    fn test_initialize_activation() {
        // Create layer with activation but without activation function
        let mut layer = Layer {
            nodes: 5,
            activation: Some(ActivationType::Sigmoid),
            activation_fn: None,
        };

        // Initialize activation function
        layer.initialize_activation();

        // Check that activation function is now present
        assert!(layer.activation_fn.is_some());
    }

    #[test]
    fn test_evaluate_prediction() {
        // Test binary classification
        let target_binary = vec![1.0].into_matrix(1, 1);
        let output_binary_correct = vec![0.8].into_matrix(1, 1);
        let output_binary_incorrect = vec![0.3].into_matrix(1, 1);

        let (error_correct, is_correct) =
            Layer::evaluate_prediction(&target_binary, &output_binary_correct);
        assert!(
            is_correct,
            "Binary prediction with 0.8 should be considered correct when target is 1.0"
        );
        assert_relative_eq!(error_correct, 0.04, epsilon = 1e-10); // (1-0.8)² = 0.04

        let (error_incorrect, is_incorrect) =
            Layer::evaluate_prediction(&target_binary, &output_binary_incorrect);
        assert!(
            !is_incorrect,
            "Binary prediction with 0.3 should be considered incorrect when target is 1.0"
        );
        assert_relative_eq!(error_incorrect, 0.49, epsilon = 1e-10); // (1-0.3)² = 0.49

        // Test multi-class classification
        let target_multi = vec![0.0, 1.0, 0.0].into_matrix(3, 1);
        let output_multi_correct = vec![0.1, 0.7, 0.2].into_matrix(3, 1);
        let output_multi_incorrect = vec![0.1, 0.2, 0.7].into_matrix(3, 1);

        let (_, multi_correct) = Layer::evaluate_prediction(&target_multi, &output_multi_correct);
        assert!(
            multi_correct,
            "Multi-class prediction should be correct when highest value matches target"
        );

        let (_, multi_incorrect) =
            Layer::evaluate_prediction(&target_multi, &output_multi_incorrect);
        assert!(
            !multi_incorrect,
            "Multi-class prediction should be incorrect when highest value does not match target"
        );
    }

    #[test]
    fn test_calculate_l2_regularization() {
        // Create sample weights
        let weights = vec![
            vec![0.1, 0.2, 0.3, 0.4].into_matrix(2, 2),
            vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0].into_matrix(2, 3),
        ];

        let reg_rate = 0.01;
        let l2_term = Layer::calculate_l2_regularization(&weights, reg_rate);

        // Calculate expected value manually:
        // Sum of squares of first matrix: 0.1² + 0.2² + 0.3² + 0.4² = 0.01 + 0.04 + 0.09 + 0.16 = 0.3
        // Sum of squares of second matrix: 0.5² + 0.6² + 0.7² + 0.8² + 0.9² + 1.0² = 0.25 + 0.36 + 0.49 + 0.64 + 0.81 + 1.0 = 3.55
        // Total sum: 0.3 + 3.55 = 3.85
        // L2 term: 3.85 * (0.01 / 2) = 3.85 * 0.005 = 0.01925

        assert_relative_eq!(l2_term, 0.01925, epsilon = 1e-10);
    }

    #[test]
    fn test_apply_l2_regularization() {
        // Create sample weights
        let weights = vec![0.1, 0.2, 0.3, 0.4].into_matrix(2, 2);

        // Test with regularization rate
        let reg_rate = 0.01;
        let l2_gradient = Layer::apply_l2_regularization(&weights, Some(reg_rate));

        // Check dimensions
        assert_eq!(l2_gradient.rows(), weights.rows());
        assert_eq!(l2_gradient.cols(), weights.cols());

        // Check values: each weight should be multiplied by reg_rate
        assert_relative_eq!(l2_gradient.get(0, 0), 0.1 * 0.01, epsilon = 1e-10);
        assert_relative_eq!(l2_gradient.get(0, 1), 0.2 * 0.01, epsilon = 1e-10);
        assert_relative_eq!(l2_gradient.get(1, 0), 0.3 * 0.01, epsilon = 1e-10);
        assert_relative_eq!(l2_gradient.get(1, 1), 0.4 * 0.01, epsilon = 1e-10);

        // Test with no regularization
        let zero_gradient = Layer::apply_l2_regularization(&weights, None);

        // Check all values are zero
        for i in 0..zero_gradient.rows() {
            for j in 0..zero_gradient.cols() {
                assert_relative_eq!(zero_gradient.get(i, j), 0.0, epsilon = 1e-10);
            }
        }
    }
}
