//! MiniGPT: A minimal GPT implementation in pure Rust
//!
//! A minimal GPT implementation in pure Rust.
//! This project started by using Andrej Karpathy's Python implementation as a reference.
//! It demonstrates that everything beyond the core algorithm is essentially optimization for efficiency.

pub mod data;
pub mod inference;
pub mod matrix;
pub mod model;
pub mod neural_network;
pub mod optimizer;
pub mod persistence;
pub mod tokenizer;
pub mod value;

// Re-export commonly used types
pub use matrix::{LcgRng, Matrix, Rng};
pub use model::{GPTConfig, StateDict};
pub use neural_network::{rmsnorm, softmax};
pub use optimizer::AdamOptimizer;
pub use tokenizer::Tokenizer;
pub use value::{Value, ValueArena};
