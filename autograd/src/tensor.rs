use ndarray::{Array2, ArrayView2, ShapeError};
use serde::{Deserialize, Serialize};
use std::fmt;

/// A dense rank-2 tensor used as the numeric payload in the autograd graph.
///
/// Tensors wrap [`Array2<f64>`] and provide shape-safe construction and access.
/// Forward values and gradients for autograd live in a [`Graph`](crate::Graph) arena,
/// referenced by [`TensorId`](crate::TensorId).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tensor(Array2<f64>);

impl Tensor {
    /// Creates a tensor with the given shape, filling it from row-major `data`.
    ///
    /// # Errors
    ///
    /// Returns [`ShapeError`] when `data.len()` does not equal `rows * cols`.
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, ShapeError> {
        Array2::from_shape_vec((rows, cols), data).map(Self)
    }

    /// Creates a tensor filled with zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self(Array2::zeros((rows, cols)))
    }

    /// Wraps an existing `Array2` without copying.
    pub fn from_array(array: Array2<f64>) -> Self {
        Self(array)
    }

    /// Creates a tensor from a flat slice in row-major order.
    ///
    /// # Errors
    ///
    /// Returns [`ShapeError`] when `slice.len()` does not equal `rows * cols`.
    pub fn from_slice(slice: &[f64], rows: usize, cols: usize) -> Result<Self, ShapeError> {
        Self::new(rows, cols, slice.to_vec())
    }

    /// Borrows the underlying array.
    pub fn array(&self) -> &Array2<f64> {
        &self.0
    }

    /// Mutably borrows the underlying array.
    pub fn array_mut(&mut self) -> &mut Array2<f64> {
        &mut self.0
    }

    /// Returns a view of the underlying array.
    pub fn view(&self) -> ArrayView2<'_, f64> {
        self.0.view()
    }

    /// Returns the element at `(row, col)`, or `None` when out of bounds.
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        self.0.get((row, col)).copied()
    }

    /// Returns the number of rows.
    pub fn rows(&self) -> usize {
        self.0.nrows()
    }

    /// Returns the number of columns.
    pub fn cols(&self) -> usize {
        self.0.ncols()
    }

    /// Returns `(rows, cols)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    /// Returns the total number of elements.
    pub fn num_elements(&self) -> usize {
        self.0.len()
    }

    /// Consumes the tensor and returns the inner array.
    pub fn into_array(self) -> Array2<f64> {
        self.0
    }
}

impl From<Array2<f64>> for Tensor {
    fn from(array: Array2<f64>) -> Self {
        Self::from_array(array)
    }
}

impl From<Tensor> for Array2<f64> {
    fn from(tensor: Tensor) -> Self {
        tensor.into_array()
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::array;

    use super::Tensor;

    #[test]
    fn new_from_valid_data() {
        let result = Tensor::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(result.is_ok());
        let tensor = match result {
            Ok(value) => value,
            Err(_) => Tensor::zeros(0, 0),
        };

        assert_eq!(tensor.shape(), (2, 2));
        assert_eq!(tensor.get(0, 0), Some(1.0));
        assert_eq!(tensor.get(1, 1), Some(4.0));
    }

    #[test]
    fn new_rejects_invalid_length() {
        let result = Tensor::new(2, 2, vec![1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }

    #[test]
    fn zeros_has_expected_shape_and_values() {
        let tensor = Tensor::zeros(3, 2);

        assert_eq!(tensor.shape(), (3, 2));
        assert_relative_eq!(tensor.array().sum(), 0.0);
    }

    #[test]
    fn from_array_and_into_array_round_trip() {
        let array = array![[1.0, 2.0], [3.0, 4.0]];
        let tensor = Tensor::from_array(array.clone());
        let recovered = tensor.into_array();

        assert_eq!(recovered, array);
    }

    #[test]
    fn from_slice_matches_new() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let from_slice_result = Tensor::from_slice(&data, 2, 2);
        let from_vec_result = Tensor::new(2, 2, data.to_vec());
        assert!(from_slice_result.is_ok());
        assert!(from_vec_result.is_ok());
        let from_slice = match from_slice_result {
            Ok(value) => value,
            Err(_) => Tensor::zeros(0, 0),
        };
        let from_vec = match from_vec_result {
            Ok(value) => value,
            Err(_) => Tensor::zeros(0, 0),
        };

        assert_eq!(from_slice, from_vec);
    }

    #[test]
    fn get_returns_none_for_out_of_bounds_indices() {
        let tensor = Tensor::zeros(2, 2);

        assert_eq!(tensor.get(2, 0), None);
        assert_eq!(tensor.get(0, 2), None);
    }

    #[test]
    fn view_matches_underlying_array() {
        let tensor = Tensor::from_array(array![[5.0, 6.0]]);
        let viewed = tensor.view();

        assert_eq!(viewed.nrows(), 1);
        assert_eq!(viewed.ncols(), 2);
        assert_relative_eq!(viewed.sum(), 11.0);
    }
}
