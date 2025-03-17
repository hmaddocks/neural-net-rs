#[macro_use]
extern crate derive_builder;

pub mod activations;
pub mod network;

pub mod matrix {
    pub use matrix::matrix::Matrix;
}
