//! Defines the structure of a layer within the neural network.
//!
//! Each layer consists of a number of nodes and an optional activation function.
use crate::activations::ActivationType;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents a single layer in the neural network.
///
/// Contains the number of nodes (neurons) in the layer and the
/// type of activation function to be applied to the output of this layer.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Layer {
    /// The number of nodes (neurons) in this layer.
    pub nodes: usize,
    /// The activation function applied to the output of this layer.
    /// If `None`, no activation function is applied (e.g., for the input layer).
    pub activation: Option<ActivationType>,
}

impl Layer {
    /// Creates a new [`Layer`].
    pub fn new(nodes: usize, activation: Option<ActivationType>) -> Self {
        Self { nodes, activation }
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{{ nodes: {}, activation: {:?} }}",
            self.nodes, self.activation
        )
    }
}
