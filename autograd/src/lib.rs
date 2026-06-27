//! Tensor-level reverse-mode automatic differentiation.
//!
//! This crate provides an arena-based autograd engine backed by [`ndarray`]. Tensors are
//! stored as `Array2<f64>` nodes in a computation graph; ops record edges and [`Graph::backward`]
//! runs reverse-mode autodiff. It is the shared numerical core for both the MLP and GPT
//! models in this workspace.

#![deny(missing_docs)]

extern crate ndarray;

mod broadcast;
mod graph;
mod ops;
mod tensor;

pub use graph::{Graph, Node, TensorId};
pub use tensor::Tensor;

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
