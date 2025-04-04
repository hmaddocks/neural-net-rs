use crate::activations::{ActivationFunction, ActivationType, Sigmoid, Softmax};
use crate::layer::Layer;
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
/// use neural_network::{network_config::NetworkConfig, layer::Layer, activations::ActivationType};
///
/// let config = NetworkConfig::default();
/// assert_eq!(config.layers, vec![Layer { nodes: 784, activation: Some(ActivationType::Sigmoid) }, Layer { nodes: 128, activation: Some(ActivationType::Sigmoid) }, Layer { nodes: 10, activation: None }]); // MNIST-like architecture
/// ```
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NetworkConfig {
    /// A collection of [`Layer`]s in the network, including input and output layers.
    /// For example, `vec![Layer { nodes: 784, activation: Some(ActivationType::Sigmoid) }, Layer { nodes: 128, activation: Some(ActivationType::Sigmoid) }, Layer { nodes: 10, activation: None }]` represents a network with:
    /// - 784 input neurons (e.g., 28x28 pixels) and sigmoid activation
    /// - 128 hidden neurons and sigmoid activation
    /// - 10 output neurons and no activation
    pub layers: Vec<Layer>,

    /// Learning rate for gradient descent.
    /// Controls how much the weights are adjusted during training.
    pub learning_rate: f64,

    /// Optional momentum coefficient for gradient descent.
    /// When specified, helps accelerate training and avoid local minima.
    pub momentum: Option<f64>,

    /// Number of training epochs.
    /// One epoch represents one complete pass through the training dataset.
    pub epochs: usize,

    /// Size of mini-batches for gradient descent.
    /// Larger batches provide more stable gradients but slower training.
    /// Default is 32.
    pub batch_size: usize,
}

impl NetworkConfig {
    /// Creates a new NetworkConfig with default values except for the layer sizes.
    ///
    /// # Arguments
    ///
    /// * `layers` - A vector of [`Layer`]s
    /// * `learning_rate` - Learning rate for gradient descent
    /// * `momentum` - Optional momentum coefficient
    /// * `epochs` - Number of training epochs
    /// * `batch_size` - Size of mini-batches for gradient descent
    ///
    /// # Returns
    ///
    /// A new NetworkConfig instance
    pub fn new(
        layers: Vec<Layer>,
        learning_rate: f64,
        momentum: Option<f64>,
        epochs: usize,
        batch_size: usize,
    ) -> Self {
        Self {
            layers,
            learning_rate,
            momentum,
            epochs,
            batch_size,
        }
    }

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

    pub fn nodes(&self) -> Vec<usize> {
        self.layers.iter().map(|layer| layer.nodes).collect()
    }

    pub fn activations_types(&self) -> Vec<ActivationType> {
        self.layers
            .iter()
            .filter_map(|layer| layer.activation)
            .collect()
    }

    pub fn activations(&self) -> Vec<Box<dyn ActivationFunction>> {
        self.activations_types()
            .iter()
            .map(|activation_type| match activation_type {
                ActivationType::Sigmoid => Box::new(Sigmoid) as Box<dyn ActivationFunction>,
                ActivationType::Softmax => Box::new(Softmax) as Box<dyn ActivationFunction>,
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
/// - Batch size: 32
impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            layers: vec![
                Layer {
                    nodes: 784,
                    activation: Some(ActivationType::Sigmoid),
                },
                Layer {
                    nodes: 128,
                    activation: Some(ActivationType::Sigmoid),
                },
                Layer {
                    nodes: 10,
                    activation: None,
                },
            ], // Common MNIST-like default architecture
            learning_rate: 0.1,
            momentum: Some(0.9),
            epochs: 30,
            batch_size: 32,
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
            "layers": [{"nodes": 784, "activation": "Sigmoid"}, {"nodes": 200, "activation": "Sigmoid"}, {"nodes": 10}],
            "learning_rate": 0.01,
            "momentum": 0.5,
            "epochs": 30,
            "batch_size": 32
        }"#;

        let mut file = File::create(&config_path).unwrap();
        file.write_all(config_json.as_bytes()).unwrap();

        let config = NetworkConfig::load(&config_path).unwrap();
        assert_eq!(
            config.layers,
            vec![
                Layer {
                    nodes: 784,
                    activation: Some(ActivationType::Sigmoid)
                },
                Layer {
                    nodes: 200,
                    activation: Some(ActivationType::Sigmoid)
                },
                Layer {
                    nodes: 10,
                    activation: None
                }
            ]
        );
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.momentum, Some(0.5));
        assert_eq!(config.epochs, 30);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_default_config() {
        let config = NetworkConfig::default();
        assert_eq!(
            config.layers,
            vec![
                Layer {
                    nodes: 784,
                    activation: Some(ActivationType::Sigmoid)
                },
                Layer {
                    nodes: 128,
                    activation: Some(ActivationType::Sigmoid)
                },
                Layer {
                    nodes: 10,
                    activation: None
                }
            ]
        );
        assert_eq!(config.learning_rate, 0.1);
        assert_eq!(config.momentum, Some(0.9));
        assert_eq!(config.epochs, 30);
    }
}
