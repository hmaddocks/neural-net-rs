use crate::activations::{ActivationFunction, ActivationType, Sigmoid, Softmax};
use crate::layer::Layer;
use crate::matrix::Matrix;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::ops::{Div, Mul};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LearningRate(f64);

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RegularizationRate(f64);

impl TryFrom<f64> for RegularizationRate {
    type Error = &'static str;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if value >= 0.0 {
            Ok(Self(value))
        } else {
            Err("Regularization rate must be non-negative")
        }
    }
}

impl From<RegularizationRate> for f64 {
    fn from(rate: RegularizationRate) -> Self {
        rate.0
    }
}

// impl Mul<RegularizationRate> for f64 {
//     type Output = f64;

//     fn mul(self, rate: RegularizationRate) -> Self::Output {
//         self * rate.0
//     }
// }

impl Mul<RegularizationRate> for &Matrix {
    type Output = Matrix;

    fn mul(self, rate: RegularizationRate) -> Self::Output {
        self * rate.0
    }
}

impl Div<f64> for RegularizationRate {
    type Output = f64;

    fn div(self, value: f64) -> Self::Output {
        self.0 / value
    }
}

impl TryFrom<f64> for LearningRate {
    type Error = &'static str;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if value > 0.0 {
            Ok(Self(value))
        } else {
            Err("Learning rate must be greater than 0.0")
        }
    }
}

impl From<LearningRate> for f64 {
    fn from(lr: LearningRate) -> Self {
        lr.0
    }
}

impl Mul<LearningRate> for f64 {
    type Output = f64;

    fn mul(self, lr: LearningRate) -> Self::Output {
        self * lr.0
    }
}

impl Mul<LearningRate> for Matrix {
    type Output = Matrix;

    fn mul(self, lr: LearningRate) -> Self::Output {
        &self * lr.0
    }
}

impl Mul<LearningRate> for &Matrix {
    type Output = Matrix;

    fn mul(self, lr: LearningRate) -> Self::Output {
        self * lr.0
    }
}

impl Mul<Matrix> for LearningRate {
    type Output = Matrix;

    fn mul(self, matrix: Matrix) -> Self::Output {
        &matrix * self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Momentum(f64);

impl TryFrom<f64> for Momentum {
    type Error = &'static str;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if (0.0..1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err("Momentum must be between 0.0 and 1.0")
        }
    }
}

impl From<Momentum> for f64 {
    fn from(momentum: Momentum) -> Self {
        momentum.0
    }
}

impl Mul<Momentum> for f64 {
    type Output = f64;

    fn mul(self, momentum: Momentum) -> Self::Output {
        self * momentum.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Epochs(usize);

impl TryFrom<usize> for Epochs {
    type Error = &'static str;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        if value > 0 {
            Ok(Self(value))
        } else {
            Err("Number of epochs must be greater than 0")
        }
    }
}

impl From<Epochs> for usize {
    fn from(epochs: Epochs) -> Self {
        epochs.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchSize(usize);

impl TryFrom<usize> for BatchSize {
    type Error = &'static str;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        if value > 0 {
            Ok(Self(value))
        } else {
            Err("Batch size must be greater than 0")
        }
    }
}

impl From<BatchSize> for usize {
    fn from(batch_size: BatchSize) -> Self {
        batch_size.0
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
#[serde(default)]
pub struct NetworkConfig {
    /// A collection of [`Layer`]s in the network, including input and output layers.
    /// For example,
    /// vec![Layer { nodes: 784, activation: Some(ActivationType::Sigmoid) },
    ///      Layer { nodes: 128, activation: Some(ActivationType::Sigmoid) },
    ///      Layer { nodes: 10, activation: None }]
    /// represents a network with:
    /// - 784 input neurons (e.g., 28x28 pixels) with sigmoid activation
    /// - 128 hidden neurons with sigmoid activation
    /// - 10 output neurons with no activation
    pub layers: Vec<Layer>,

    /// Learning rate for gradient descent.
    /// Controls how much the weights are adjusted during training.
    pub learning_rate: LearningRate,

    /// Momentum coefficient for gradient descent.
    /// When specified, helps accelerate training and avoid local minima.
    pub momentum: Momentum,

    /// Number of training epochs.
    /// One epoch represents one complete pass through the training dataset.
    pub epochs: Epochs,

    /// Size of mini-batches for gradient descent.
    /// Larger batches provide more stable gradients but slower training.
    /// Default is 32.
    pub batch_size: BatchSize,

    /// L2 regularization rate (weight decay)
    pub regularization_rate: RegularizationRate,
}

impl NetworkConfig {
    /// Creates a new NetworkConfig with default values except for the layer sizes.
    ///
    /// # Arguments
    ///
    /// * `layers` - A vector of [`Layer`]s
    /// * `learning_rate` - Learning rate for gradient descent
    /// * `momentum` - Momentum coefficient for gradient descent
    /// * `epochs` - Number of training epochs
    /// * `batch_size` - Size of mini-batches for gradient descent
    /// * `regularization_rate` - L2 regularization rate (weight decay)
    ///
    /// # Returns
    ///
    /// A new NetworkConfig instance
    pub fn new(
        layers: Vec<Layer>,
        learning_rate: f64,
        momentum: f64,
        epochs: usize,
        batch_size: usize,
        regularization_rate: f64,
    ) -> Option<Self> {
        Some(Self {
            layers: layers,
            learning_rate: LearningRate::try_from(learning_rate).ok()?,
            momentum: Momentum::try_from(momentum).ok()?,
            epochs: Epochs::try_from(epochs).ok()?,
            batch_size: BatchSize::try_from(batch_size).ok()?,
            regularization_rate: RegularizationRate::try_from(regularization_rate).ok()?,
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

    /// Returns a vector containing the number of nodes in each layer of the network.
    ///
    /// This method iterates through all layers and collects their node counts into a vector,
    /// preserving the layer order from input to output.
    pub fn nodes(&self) -> Vec<usize> {
        self.layers.iter().map(|layer| layer.nodes).collect()
    }

    /// Returns a vector of activation types for all layers that have an activation function.
    ///
    /// Layers without an activation function are filtered out. The resulting vector
    /// contains activation types in order from input to output layers.
    pub fn activations_types(&self) -> Vec<ActivationType> {
        self.layers
            .iter()
            .filter_map(|layer| layer.activation)
            .collect()
    }

    /// Creates a vector of boxed activation functions based on the network's configuration.
    ///
    /// This method converts each ActivationType into its corresponding activation function
    /// implementation. The resulting vector contains trait objects that can be used
    /// for forward and backward propagation in the network.
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
/// - Learning rate: 0.01
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
            learning_rate: LearningRate::try_from(0.01).unwrap(),
            momentum: Momentum::try_from(0.5).unwrap(),
            epochs: Epochs::try_from(30).unwrap(),
            batch_size: BatchSize::try_from(32).unwrap(),
            regularization_rate: RegularizationRate::try_from(0.0001).unwrap(), // Small default L2 regularization
        }
    }
}

impl fmt::Display for NetworkConfig {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Network Configuration:")?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "  Layer {}: {}", i, layer)?;
        }
        writeln!(
            f,
            "  Learning Rate:       {:.4}",
            f64::from(self.learning_rate)
        )?;
        writeln!(f, "  Momentum:            {:.4}", f64::from(self.momentum))?;
        writeln!(f, "  Epochs:              {}", usize::from(self.epochs))?;
        writeln!(f, "  Batch Size:          {}", usize::from(self.batch_size))?;
        write!(
            f,
            "  Regularization Rate: {:.4}",
            f64::from(self.regularization_rate)
        )?;
        Ok(())
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
        assert!(LearningRate::try_from(0.1).is_ok());

        // Invalid cases
        assert!(LearningRate::try_from(0.0).is_err());
        assert!(LearningRate::try_from(-0.1).is_err());
    }

    #[test]
    fn test_momentum_validation() {
        // Valid cases
        assert!(Momentum::try_from(0.0).is_ok());
        assert!(Momentum::try_from(0.5).is_ok());
        assert!(Momentum::try_from(0.9).is_ok());

        // Invalid cases
        assert!(Momentum::try_from(-0.1).is_err());
        assert!(Momentum::try_from(1.0).is_err());
        assert!(Momentum::try_from(1.1).is_err());
    }

    #[test]
    fn test_epochs_conversion() {
        // Valid cases using TryFrom
        assert!(Epochs::try_from(1).is_ok());
        assert!(Epochs::try_from(30).is_ok());
        assert!(Epochs::try_from(usize::MAX).is_ok());

        // Invalid cases using TryFrom
        assert!(Epochs::try_from(0).is_err());
        assert_eq!(
            Epochs::try_from(0).unwrap_err(),
            "Number of epochs must be greater than 0"
        );

        // Test From<Epochs> for usize
        let epochs = Epochs::try_from(42).unwrap();
        assert_eq!(usize::from(epochs), 42);
    }

    #[test]
    fn test_batch_size_validation() {
        // Valid cases
        assert!(BatchSize::try_from(1).is_ok());
        assert!(BatchSize::try_from(32).is_ok());
        assert!(BatchSize::try_from(usize::MAX).is_ok());

        // Invalid cases
        assert!(BatchSize::try_from(0).is_err());
    }

    #[test]
    fn test_newtype_value_access() {
        let learning_rate = LearningRate::try_from(0.1).unwrap();
        assert_eq!(f64::from(learning_rate), 0.1);

        let momentum = Momentum::try_from(0.9).unwrap();
        assert_eq!(f64::from(momentum), 0.9);

        let epochs = Epochs::try_from(30).unwrap();
        assert_eq!(usize::from(epochs), 30);

        let batch_size = BatchSize::try_from(32).unwrap();
        assert_eq!(usize::from(batch_size), 32);
    }

    #[test]
    fn test_newtype_serde() {
        let learning_rate = LearningRate::try_from(0.1).unwrap();
        let serialized = serde_json::to_string(&learning_rate).unwrap();
        let deserialized: LearningRate = serde_json::from_str(&serialized).unwrap();
        assert_eq!(learning_rate, deserialized);

        let momentum = Momentum::try_from(0.9).unwrap();
        let serialized = serde_json::to_string(&momentum).unwrap();
        let deserialized: Momentum = serde_json::from_str(&serialized).unwrap();
        assert_eq!(momentum, deserialized);

        let epochs = Epochs::try_from(30).unwrap();
        let serialized = serde_json::to_string(&epochs).unwrap();
        let deserialized: Epochs = serde_json::from_str(&serialized).unwrap();
        assert_eq!(epochs, deserialized);

        let batch_size = BatchSize::try_from(32).unwrap();
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
        assert_eq!(config.learning_rate, LearningRate::try_from(0.01).unwrap());
        assert_eq!(config.momentum, Momentum::try_from(0.5).unwrap());
        assert_eq!(config.epochs, Epochs::try_from(30).unwrap());
        assert_eq!(config.batch_size, BatchSize::try_from(32).unwrap());
    }

    #[test]
    fn test_load_bad_config() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("test_config.json");

        let config_json = r#"{
            "layers": [{"nodes": 784, "activation": "Sigmoid"}, {"nodes": 200, "activation": "Sigmoid"}, {"nodes": 10}],
            "learning_rate": 0.01,
            "momentum": 0.5,
            "epochs": -30,
            "batch_size": 32
        }"#;

        let mut file = File::create(&config_path).unwrap();
        file.write_all(config_json.as_bytes()).unwrap();

        let error = NetworkConfig::load(&config_path).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("invalid value: integer `-30`, expected usize")
        );
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
        assert_eq!(config.learning_rate, LearningRate::try_from(0.01).unwrap());
        assert_eq!(config.momentum, Momentum::try_from(0.5).unwrap());
        assert_eq!(config.epochs, Epochs::try_from(30).unwrap());
        assert_eq!(config.batch_size, BatchSize::try_from(32).unwrap());
    }
}
