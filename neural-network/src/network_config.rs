use crate::activations::{ActivationFunction, ActivationType, Sigmoid, Softmax};
use crate::layer::Layer;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LearningRate(f64);

impl LearningRate {
    pub fn new(value: f64) -> Option<Self> {
        if value > 0.0 {
            Some(Self(value))
        } else {
            None
        }
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Momentum(f64);

impl Momentum {
    pub fn new(value: f64) -> Option<Self> {
        if (0.0..1.0).contains(&value) {
            Some(Self(value))
        } else {
            None
        }
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Epochs(usize);

impl Epochs {
    pub fn new(value: usize) -> Option<Self> {
        if value > 0 {
            Some(Self(value))
        } else {
            None
        }
    }

    pub fn value(&self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchSize(usize);

impl BatchSize {
    pub fn new(value: usize) -> Option<Self> {
        if value > 0 {
            Some(Self(value))
        } else {
            None
        }
    }

    pub fn value(&self) -> usize {
        self.0
    }
}

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
    pub learning_rate: LearningRate,

    /// Optional momentum coefficient for gradient descent.
    /// When specified, helps accelerate training and avoid local minima.
    pub momentum: Option<Momentum>,

    /// Number of training epochs.
    /// One epoch represents one complete pass through the training dataset.
    pub epochs: Epochs,

    /// Size of mini-batches for gradient descent.
    /// Larger batches provide more stable gradients but slower training.
    /// Default is 32.
    pub batch_size: BatchSize,
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
    ) -> Option<Self> {
        Some(Self {
            layers,
            learning_rate: LearningRate::new(learning_rate)?,
            momentum: momentum.and_then(Momentum::new),
            epochs: Epochs::new(epochs)?,
            batch_size: BatchSize::new(batch_size)?,
        })
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
            learning_rate: LearningRate::new(0.1).unwrap(),
            momentum: Some(Momentum::new(0.9).unwrap()),
            epochs: Epochs::new(30).unwrap(),
            batch_size: BatchSize::new(32).unwrap(),
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
    fn test_learning_rate_validation() {
        // Valid cases
        assert!(LearningRate::new(0.1).is_some());
        assert!(LearningRate::new(1.0).is_some());
        assert!(LearningRate::new(0.001).is_some());

        // Invalid cases
        assert!(LearningRate::new(0.0).is_none());
        assert!(LearningRate::new(-0.1).is_none());
    }

    #[test]
    fn test_momentum_validation() {
        // Valid cases
        assert!(Momentum::new(0.0).is_some());
        assert!(Momentum::new(0.5).is_some());
        assert!(Momentum::new(0.9).is_some());

        // Invalid cases
        assert!(Momentum::new(-0.1).is_none());
        assert!(Momentum::new(1.0).is_none());
        assert!(Momentum::new(1.1).is_none());
    }

    #[test]
    fn test_epochs_validation() {
        // Valid cases
        assert!(Epochs::new(1).is_some());
        assert!(Epochs::new(30).is_some());
        assert!(Epochs::new(usize::MAX).is_some());

        // Invalid cases
        assert!(Epochs::new(0).is_none());
    }

    #[test]
    fn test_batch_size_validation() {
        // Valid cases
        assert!(BatchSize::new(1).is_some());
        assert!(BatchSize::new(32).is_some());
        assert!(BatchSize::new(usize::MAX).is_some());

        // Invalid cases
        assert!(BatchSize::new(0).is_none());
    }

    #[test]
    fn test_newtype_value_access() {
        let learning_rate = LearningRate::new(0.1).unwrap();
        assert_eq!(learning_rate.value(), 0.1);

        let momentum = Momentum::new(0.9).unwrap();
        assert_eq!(momentum.value(), 0.9);

        let epochs = Epochs::new(30).unwrap();
        assert_eq!(epochs.value(), 30);

        let batch_size = BatchSize::new(32).unwrap();
        assert_eq!(batch_size.value(), 32);
    }

    #[test]
    fn test_newtype_serde() {
        let learning_rate = LearningRate::new(0.1).unwrap();
        let serialized = serde_json::to_string(&learning_rate).unwrap();
        let deserialized: LearningRate = serde_json::from_str(&serialized).unwrap();
        assert_eq!(learning_rate, deserialized);

        let momentum = Momentum::new(0.9).unwrap();
        let serialized = serde_json::to_string(&momentum).unwrap();
        let deserialized: Momentum = serde_json::from_str(&serialized).unwrap();
        assert_eq!(momentum, deserialized);

        let epochs = Epochs::new(30).unwrap();
        let serialized = serde_json::to_string(&epochs).unwrap();
        let deserialized: Epochs = serde_json::from_str(&serialized).unwrap();
        assert_eq!(epochs, deserialized);

        let batch_size = BatchSize::new(32).unwrap();
        let serialized = serde_json::to_string(&batch_size).unwrap();
        let deserialized: BatchSize = serde_json::from_str(&serialized).unwrap();
        assert_eq!(batch_size, deserialized);
    }

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
        assert_eq!(config.learning_rate, LearningRate::new(0.01).unwrap());
        assert_eq!(config.momentum, Some(Momentum::new(0.5).unwrap()));
        assert_eq!(config.epochs, Epochs::new(30).unwrap());
        assert_eq!(config.batch_size, BatchSize::new(32).unwrap());
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
        assert_eq!(config.learning_rate, LearningRate::new(0.1).unwrap());
        assert_eq!(config.momentum, Some(Momentum::new(0.9).unwrap()));
        assert_eq!(config.epochs, Epochs::new(30).unwrap());
        assert_eq!(config.batch_size, BatchSize::new(32).unwrap());
    }
}
