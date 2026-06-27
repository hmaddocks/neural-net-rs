//! Tensor-level reverse-mode automatic differentiation.
//!
//! This crate provides an arena-based autograd engine backed by [`ndarray`]. Tensors are
//! stored as [`Array2<f64>`] nodes in a computation graph; ops record edges and
//! [`Graph::backward`] runs reverse-mode autodiff. It is the shared numerical core for both
//! the MLP and GPT models in this workspace.
//!
//! # Core types
//!
//! | Type | Role |
//! |------|------|
//! | [`Graph`] | Arena that owns all nodes; you build and differentiate here |
//! | [`TensorId`] | Opaque handle to a node (parameter, input, or intermediate) |
//! | [`Tensor`] | Standalone rank-2 wrapper for construction and I/O outside the graph |
//! | [`Node`] | One graph entry: forward value, gradient, children, backward rule |
//!
//! # Layout conventions
//!
//! All graph tensors are **2-D** (`Array2<f64>`). Typical shapes:
//!
//! - **Batch × features** for activations: `(batch_size, features)`
//! - **In × out** for weight matrices: `(input_dim, output_dim)`
//! - **1 × features** for bias vectors (broadcast over batch rows)
//!
//! [`Graph::matmul`] uses standard matrix multiply: `(m, k) @ (k, n) → (m, n)`.
//! [`Graph::add`] and [`Graph::mul`] support NumPy-style 2-D broadcasting (row/column
//! vectors expand to match a matrix).
//!
//! Row-wise ops ([`Graph::softmax`], [`Graph::rmsnorm`]) normalize each **row**
//! independently, matching a `(batch, features)` layout.
//!
//! # Quick start
//!
//! Build a small expression, differentiate it, and read parameter gradients:
//!
//! ```
//! use autograd::Graph;
//! use ndarray::array;
//!
//! let mut graph = Graph::new();
//!
//! // Parameters and input (leaf nodes keep their ids across training steps).
//! let w = graph.leaf(array![[0.5], [0.5]]); // (2, 1)
//! let x = graph.leaf(array![[1.0, 2.0]]);  // (1, 2) — one sample, two features
//!
//! // Forward: y = sum((x @ w)^2)
//! let mm = graph.matmul(x, w);
//! let y = graph.mul(mm, mm);
//! let loss = graph.sum(y);
//!
//! // Reverse mode: seed loss with 1, propagate to leaves.
//! graph.backward(loss);
//!
//! assert!(graph.grad(w)[(0, 0)].abs() > 0.0);
//! ```
//!
//! # Available operations
//!
//! [`Graph`] implements the following differentiable ops (see `ops.rs`):
//!
//! | Method | Description |
//! |--------|-------------|
//! | [`Graph::matmul`] | Matrix multiply |
//! | [`Graph::add`] | Elementwise add with broadcasting |
//! | [`Graph::mul`] | Elementwise multiply with broadcasting |
//! | [`Graph::relu`] | ReLU activation |
//! | [`Graph::sigmoid`] | Sigmoid activation |
//! | [`Graph::log`] | Natural log |
//! | [`Graph::pow`] | Elementwise power |
//! | [`Graph::softmax`] | Softmax per row |
//! | [`Graph::rmsnorm`] | RMS normalization per row |
//! | [`Graph::sum`] | Sum all elements to a `(1, 1)` scalar |
//!
//! # Training loop pattern
//!
//! 1. **Create parameters** as leaf nodes once; keep their [`TensorId`]s.
//! 2. Each step: [`Graph::zero_grad`], build the forward graph from inputs, compute loss,
//!    call [`Graph::backward`].
//! 3. Read gradients with [`Graph::grad`], update with [`Graph::set_data`].
//! 4. Call [`Graph::clear_computation_graph`] with the number of parameter leaves to drop
//!    intermediates and avoid unbounded arena growth.
//!
//! [`XorMlp`] demonstrates this end-to-end on the XOR problem (2 → 8 → 1 ReLU, MSE loss).
//!
//! # Gradient validation
//!
//! The [`central_difference`] and [`gradients_match`] helpers compare autograd gradients
//! against finite-difference approximations. Every op has a corresponding unit test in
//! `gradient_check.rs`.
//!
//! [`Array2<f64>`]: ndarray::Array2

#![deny(missing_docs)]

extern crate ndarray;

mod broadcast;
mod gradient_check;
mod graph;
mod manual_backprop;
mod ops;
mod tensor;
mod xor;

pub use gradient_check::{DEFAULT_EPSILON, DEFAULT_TOLERANCE, central_difference, gradients_match};
pub use graph::{Graph, Node, TensorId};
pub use tensor::Tensor;
pub use xor::XorMlp;

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn ndarray_blas_matmul_is_available() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let product = a.dot(&b);

        let top_left = product.get((0, 0)).copied();
        let bottom_right = product.get((1, 1)).copied();

        assert_eq!(top_left, Some(19.0));
        assert_eq!(bottom_right, Some(50.0));
        assert_relative_eq!(product.sum(), 134.0);
    }
}
