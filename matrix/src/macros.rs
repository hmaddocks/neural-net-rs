#[macro_export]
macro_rules! matrix {
    ( $( $($val:expr),+ );* $(;)? ) => {
        {
            let mut data = Vec::<f64>::new();
            let mut rows = 0;
            let mut cols = 0;
            $(
                let row_data = vec![$($val),+];
                data.extend(row_data);
                rows += 1;
                let row_len = vec![$($val),+].len();
                if cols == 0 {
                    cols = row_len;
                } else if cols != row_len {
                    panic!("Inconsistent number of elements in the matrix rows");
                }
            )*

            Matrix {
                rows,
                cols,
                data,
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    #[test]
    fn test_matrix_macro() {
        let m = matrix![
            1.0, 2.0, 3.0;
            4.0, 5.0, 6.0;
            7.0, 8.0, 9.0
        ];

        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 3);
        assert_eq!(m.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_matrix_macro_single_row() {
        let m = matrix![1.0, 2.0, 3.0];
        assert_eq!(m.rows, 1);
        assert_eq!(m.cols, 3);
        assert_eq!(m.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_matrix_macro_single_column() {
        let m = matrix![
            1.0;
            2.0;
            3.0
        ];
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 1);
        assert_eq!(m.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_matrix_macro_single_element() {
        let m = matrix![42.0];
        assert_eq!(m.rows, 1);
        assert_eq!(m.cols, 1);
        assert_eq!(m.data, vec![42.0]);
    }

    #[test]
    fn test_matrix_macro_non_square() {
        let m = matrix![
            1.0, 2.0;
            3.0, 4.0;
            5.0, 6.0
        ];
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 2);
        assert_eq!(m.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_matrix_macro_with_expressions() {
        let x = 1.0;
        let y = 2.0;
        let m = matrix![
            x + y, x * y;
            y - x, x / y
        ];
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 2);
        assert_eq!(m.data, vec![3.0, 2.0, 1.0, 0.5]);
    }
}
