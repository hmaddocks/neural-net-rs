use crate::MnistError;
use matrix::matrix::Matrix;

#[derive(Debug)]
pub struct StandardizationParams {
    mean: f64,
    std_dev: f64,
}

impl StandardizationParams {
    /// Returns a new StandardizationParams with the given mean and standard deviation.
    ///
    /// # Arguments
    /// * `mean` - The mean of the data
    /// * `std_dev` - The standard deviation of the data
    ///
    /// # Returns
    /// * `StandardizationParams` - A new StandardizationParams with the given mean and standard deviation
    pub fn new(mean: f64, std_dev: f64) -> StandardizationParams {
        StandardizationParams { mean, std_dev }
    }

    /// Builds a new StandardizationParams from a set of matrices.
    ///
    /// # Arguments
    /// * `mnist_data` - A slice of matrices containing the data to be standardized
    ///
    /// # Returns
    /// * `StandardizationParams` - A new StandardizationParams with the computed mean and standard deviation
    pub fn build(mnist_data: &[Matrix]) -> Result<StandardizationParams, &'static str> {
        let total_pixels: Vec<f64> = mnist_data
            .iter()
            .map(|matrix| {
                matrix
                    .data
                    .as_slice()
                    .ok_or("Failed to get data slice from matrix")
            })
            .collect::<Result<Vec<&[f64]>, _>>()? // Collect results first
            .into_iter()
            .flat_map(|slice| slice.iter().copied())
            .collect();

        if total_pixels.is_empty() {
            return Err("No data available for standardization");
        }

        let mean = total_pixels.iter().sum::<f64>() / total_pixels.len() as f64;
        let variance = total_pixels
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / total_pixels.len() as f64;
        let std_dev = variance.sqrt();

        Ok(StandardizationParams { mean, std_dev })
    }

    /// Returns the mean of the data.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Returns the standard deviation of the data.
    pub fn std_dev(&self) -> f64 {
        self.std_dev
    }
}

pub struct StandardizedMnistData {
    params: StandardizationParams,
}

impl StandardizedMnistData {
    /// Returns a new StandardizedMnistData with the given MnistData and StandardizationParams.
    ///
    /// # Arguments
    /// * `params` - The StandardizationParams to be used for standardization
    ///
    /// # Returns
    /// * `StandardizedMnistData` - A new StandardizedMnistData with the given StandardizationParams
    pub fn new(params: StandardizationParams) -> StandardizedMnistData {
        StandardizedMnistData { params }
    }

    /// Standardizes a set of matrices.
    ///
    /// # Arguments
    /// * `mnist_data` - A slice of matrices containing the data to be standardized
    ///
    /// # Returns
    /// * `Vec<Matrix>` - A vector of standardized matrices
    pub fn standardize(&self, mnist_data: &[Matrix]) -> Result<Vec<Matrix>, MnistError> {
        mnist_data
            .iter()
            .map(|matrix| self.standardize_matrix(matrix))
            .collect()
    }

    /// Standardizes a single matrix.
    ///
    /// # Arguments
    /// * `matrix` - A matrix containing the data to be standardized
    ///
    /// # Returns
    /// * `Matrix` - A standardized matrix
    ///
    /// # Errors
    /// * `MnistError::Io` - If the matrix data cannot be accessed
    pub fn standardize_matrix(&self, matrix: &Matrix) -> Result<Matrix, MnistError> {
        let standardized_data: Vec<f64> = matrix
            .data
            .as_slice()
            .ok_or(MnistError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to access matrix data",
            )))?
            .iter()
            .map(|&x| (x - self.params.mean) / self.params.std_dev)
            .collect();

        Ok(Matrix::new(matrix.rows(), matrix.cols(), standardized_data))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_standardization_params() {
        let data = vec![
            Matrix::new(2, 1, vec![1.0, 2.0]),
            Matrix::new(2, 1, vec![3.0, 4.0]),
        ];
        let params = StandardizationParams::build(&data);

        let params = params.expect("Failed to build standardization params");
        assert_relative_eq!(params.mean, 2.5, epsilon = 1e-10);
        assert_relative_eq!(params.std_dev, 1.118033988749895, epsilon = 1e-10);
    }

    #[test]
    fn test_standardize_matrix() {
        let data = vec![
            Matrix::new(2, 1, vec![1.0, 2.0]),
            Matrix::new(2, 1, vec![3.0, 4.0]),
        ];
        let params = StandardizationParams::build(&data);
        let params = params.expect("Failed to build standardization params");
        let standardized_data = StandardizedMnistData::new(params)
            .standardize(&data)
            .unwrap();

        assert_relative_eq!(
            standardized_data[0].get(0, 0),
            -1.3416407864998738,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            standardized_data[0].get(1, 0),
            -0.4472135954999579,
            epsilon = 1e-10
        );
    }
}
