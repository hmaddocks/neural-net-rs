// Modules
mod activations;
mod layer;
mod network;
mod network_config;
mod regularization;
mod training_history;

pub use activations::{ActivationFunction, ActivationType};
pub use layer::Layer;
pub use matrix::Matrix;
pub use network::Network;
pub use network_config::{BatchSize, Epochs, LearningRate, Momentum, NetworkConfig};
pub use regularization::RegularizationType;
pub use training_history::TrainingHistory;
