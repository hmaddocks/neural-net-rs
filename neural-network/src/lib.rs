// Modules
pub mod activations;
pub mod layer;
pub mod matrix {
    pub use matrix::matrix::Matrix;
}
pub mod network;
pub mod network_config;
pub mod training_history;

pub use crate::activations::ActivationType;
pub use crate::layer::Layer;
pub use crate::matrix::Matrix;
pub use crate::network::Network;
pub use crate::network_config::NetworkConfig;
pub use crate::training_history::TrainingHistory;
