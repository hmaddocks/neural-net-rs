#[macro_use]
extern crate derive_builder;

pub mod activations;
pub mod network;

pub use activations::{Activation, ActivationType, SIGMOID, SOFTMAX};
pub use network::Network;

pub mod prelude {
    pub use crate::Network;
    pub use crate::SIGMOID;
}
