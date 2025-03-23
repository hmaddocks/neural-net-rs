extern crate derive_builder;

pub mod activations;
pub mod network;
pub mod network_config;
pub mod matrix {
    pub use matrix::matrix::Matrix;
}
