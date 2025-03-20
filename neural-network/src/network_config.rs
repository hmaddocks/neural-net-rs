use crate::activations::{Activation, ActivationType, SIGMOID, SOFTMAX};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Configuration for a neural network.
///
/// This struct holds all the parameters needed to define and train a neural network,
/// including layer sizes, activation functions, and training hyperparameters.
///
/// # Example
///
/// ```
/// use neural_network::network_config::NetworkConfig;
///
/// let config = NetworkConfig::default();
/// assert_eq!(config.layers, vec![784, 128, 10]); // MNIST-like architecture
/// ```
#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Sizes of each layer in the network, including input and output layers.
    /// For example, `[784, 128, 10]` represents a network with:
    /// - 784 input neurons
    /// - 128 hidden neurons
    /// - 10 output neurons
    pub layers: Vec<usize>,

    /// Activation functions for each layer transition.
    /// The length should be one less than the number of layers.
    /// Each activation function is applied to the output of its corresponding layer.
    pub activations: Vec<Activation>,

    /// Learning rate for gradient descent.
    /// Controls how much the weights are adjusted during training.
    pub learning_rate: f64,

    /// Optional momentum coefficient for gradient descent.
    /// When specified, helps accelerate training and avoid local minima.
    pub momentum: Option<f64>,

    /// Number of training epochs.
    /// One epoch represents one complete pass through the training dataset.
    pub epochs: usize,
}

impl NetworkConfig {
    /// Loads a network configuration from a JSON file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the JSON configuration file
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing either the loaded `NetworkConfig` or an error
    /// if the file cannot be read or parsed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use neural_network::network_config::NetworkConfig;
    /// use std::path::Path;
    ///
    /// let config = NetworkConfig::load(Path::new("config.json")).unwrap();
    /// ```
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let config_str = fs::read_to_string(path)?;
        let config: NetworkConfig = serde_json::from_str(&config_str)?;
        Ok(config)
    }

    /// Returns a vector of activation functions based on the configuration.
    ///
    /// This method converts the stored activation types into their corresponding
    /// function implementations.
    ///
    /// # Returns
    ///
    /// A vector of `Activation` instances containing the actual activation functions.
    pub fn get_activations(&self) -> Vec<Activation> {
        self.activations
            .iter()
            .map(|activation| match activation.activation_type {
                ActivationType::Sigmoid => SIGMOID,
                ActivationType::Softmax => SOFTMAX,
            })
            .collect()
    }
}

/// Default implementation provides a common configuration suitable for MNIST-like datasets.
///
/// The default configuration uses:
/// - Input layer: 784 neurons (28x28 pixels)
/// - Hidden layer: 128 neurons
/// - Output layer: 10 neurons (digits 0-9)
/// - Sigmoid activation for both layer transitions
/// - Learning rate: 0.1
/// - Momentum: 0.9
/// - 30 training epochs
impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            layers: vec![784, 128, 10], // Common MNIST-like default architecture
            activations: vec![SIGMOID, SIGMOID], // Sigmoid for hidden and output layers
            learning_rate: 0.1,
            momentum: Some(0.9),
            epochs: 30,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_load_config() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("test_config.json");

        let config_json = r#"{
            "layers": [784, 200, 10],
            "activations": ["Sigmoid", "Sigmoid"],
            "learning_rate": 0.01,
            "momentum": 0.5,
            "epochs": 30
        }"#;

        let mut file = File::create(&config_path).unwrap();
        file.write_all(config_json.as_bytes()).unwrap();

        let config = NetworkConfig::load(&config_path).unwrap();
        assert_eq!(config.layers, vec![784, 200, 10]);
        assert_eq!(config.activations, vec![SIGMOID, SIGMOID]);
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.momentum, Some(0.5));
        assert_eq!(config.epochs, 30);
    }

    #[test]
    fn test_default_config() {
        let config = NetworkConfig::default();
        assert_eq!(config.layers, vec![784, 128, 10]);
        assert_eq!(config.activations, vec![SIGMOID, SIGMOID]);
        assert_eq!(config.learning_rate, 0.1);
        assert_eq!(config.momentum, Some(0.9));
        assert_eq!(config.epochs, 30);
    }
}
