#[macro_use]
extern crate derive_builder;

mod activations;
mod network;

pub use activations::SIGMOID;
pub use network::Network;

pub mod prelude {
    pub use crate::Network;
    pub use crate::SIGMOID;
}
