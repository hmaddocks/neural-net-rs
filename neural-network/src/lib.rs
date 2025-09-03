extern crate approx;
extern crate indicatif;
extern crate matrix;
extern crate ndarray;
extern crate rand;
extern crate serde;
extern crate serde_json;
extern crate tempfile;
// Modules
mod activations;
mod layer;
mod network;
mod network_config;
mod regularization;
mod training_history;

pub use activations::Activation;
pub use layer::Layer;
pub use network::Network;
pub use network_config::{BatchSize, Epochs, LearningRate, Momentum, NetworkConfig};
pub use regularization::RegularizationType;
pub use training_history::TrainingHistory;
