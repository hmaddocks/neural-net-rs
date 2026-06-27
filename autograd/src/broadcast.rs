use ndarray::Array2;

/// Returns the broadcast output shape for two 2-D tensors, if compatible.
pub fn broadcast_shape(left: (usize, usize), right: (usize, usize)) -> Option<(usize, usize)> {
    let rows = match (left.0, right.0) {
        (a, b) if a == b => a,
        (1, b) => b,
        (a, 1) => a,
        _ => return None,
    };
    let cols = match (left.1, right.1) {
        (a, b) if a == b => a,
        (1, b) => b,
        (a, 1) => a,
        _ => return None,
    };
    Some((rows, cols))
}

fn input_index(
    input_rows: usize,
    input_cols: usize,
    out_row: usize,
    out_col: usize,
) -> (usize, usize) {
    let row = if input_rows == 1 { 0 } else { out_row };
    let col = if input_cols == 1 { 0 } else { out_col };
    (row, col)
}

/// Elementwise addition with NumPy-style 2-D broadcasting.
pub fn broadcast_add(left: &Array2<f64>, right: &Array2<f64>) -> Option<Array2<f64>> {
    let (rows, cols) = broadcast_shape(left.dim(), right.dim())?;
    Some(Array2::from_shape_fn((rows, cols), |(row, col)| {
        let (left_row, left_col) = input_index(left.nrows(), left.ncols(), row, col);
        let (right_row, right_col) = input_index(right.nrows(), right.ncols(), row, col);
        left[(left_row, left_col)] + right[(right_row, right_col)]
    }))
}

/// Elementwise multiplication with NumPy-style 2-D broadcasting.
pub fn broadcast_mul(left: &Array2<f64>, right: &Array2<f64>) -> Option<Array2<f64>> {
    let (rows, cols) = broadcast_shape(left.dim(), right.dim())?;
    Some(Array2::from_shape_fn((rows, cols), |(row, col)| {
        let (left_row, left_col) = input_index(left.nrows(), left.ncols(), row, col);
        let (right_row, right_col) = input_index(right.nrows(), right.ncols(), row, col);
        left[(left_row, left_col)] * right[(right_row, right_col)]
    }))
}

/// Sums an upstream gradient into the shape of a broadcast input.
pub fn sum_broadcast_grad(grad: &Array2<f64>, input_shape: (usize, usize)) -> Array2<f64> {
    if grad.dim() == input_shape {
        return grad.clone();
    }

    let mut reduced = Array2::zeros(input_shape);
    for row in 0..grad.nrows() {
        for col in 0..grad.ncols() {
            let (input_row, input_col) = input_index(input_shape.0, input_shape.1, row, col);
            reduced[(input_row, input_col)] += grad[(row, col)];
        }
    }
    reduced
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::array;

    use super::{broadcast_add, sum_broadcast_grad};

    #[test]
    fn broadcast_add_expands_row_vector() {
        let left = array![[1.0, 2.0, 3.0]];
        let right = array![[10.0], [20.0]];
        let sum = broadcast_add(&left, &right).expect("broadcast add should succeed");

        assert_eq!(sum.dim(), (2, 3));
        assert_relative_eq!(sum[(0, 0)], 11.0);
        assert_relative_eq!(sum[(1, 2)], 23.0);
    }

    #[test]
    fn sum_broadcast_grad_reduces_broadcast_dimensions() {
        let grad = array![[1.0, 2.0], [3.0, 4.0]];
        let reduced = sum_broadcast_grad(&grad, (1, 2));

        assert_relative_eq!(reduced[(0, 0)], 4.0);
        assert_relative_eq!(reduced[(0, 1)], 6.0);
    }
}
