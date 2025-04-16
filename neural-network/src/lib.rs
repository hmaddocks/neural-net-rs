extern crate derive_builder;

pub mod activations;
pub mod layer;
pub mod network;
pub mod network_config;
pub mod training_history;
pub mod matrix {
    pub use matrix::matrix::Matrix;
}
