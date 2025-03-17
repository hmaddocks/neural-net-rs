use neural_network::{Activation, SIGMOID, SOFTMAX};

/// Configuration parameters for neural network training.
#[derive(Debug)]
pub struct TrainingConfig {
    /// Size of each training batch
    pub batch_size: usize,
    /// Number of training epochs
    pub epochs: u32,
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Number of nodes in each hidden layer
    pub hidden_layers: Vec<usize>,
    /// Activation functions for each layer (except input layer)
    pub activation_functions: Vec<Activation>,
    /// Number of epochs to wait for improvement before early stopping
    pub early_stopping_patience: u32,
    /// Minimum improvement in accuracy required to reset patience counter
    pub early_stopping_min_delta: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        // Use sigmoid for hidden layers and softmax for output layer
        let mut activations = vec![SIGMOID; 2]; // For the two hidden layers
        activations.push(SOFTMAX); // For the output layer

        Self {
            batch_size: 100,
            epochs: 30,
            learning_rate: 0.001,
            hidden_layers: vec![128, 64],
            activation_functions: activations,
            early_stopping_patience: 5,
            early_stopping_min_delta: 0.001,
        }
    }
}
