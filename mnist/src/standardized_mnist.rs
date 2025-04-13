use crate::MnistError;
use indicatif::{ProgressBar, ProgressStyle};
use matrix::matrix::Matrix;
use ndarray::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mean(f64);

impl From<Mean> for f64 {
    fn from(mean: Mean) -> Self {
        mean.0
    }
}

impl From<f64> for Mean {
    fn from(value: f64) -> Self {
        Mean(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StdDev(f64);

impl TryFrom<f64> for StdDev {
    type Error = &'static str;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if value <= 0.0 {
            Err("Standard deviation must be positive")
        } else {
            Ok(StdDev(value))
        }
    }
}

impl From<StdDev> for f64 {
    fn from(std_dev: StdDev) -> Self {
        std_dev.0
    }
}

#[derive(Debug)]
pub struct StandardizationParams {
    mean: Mean,
    std_dev: StdDev,
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
        StandardizationParams {
            mean: Mean::from(mean),
            std_dev: StdDev::try_from(std_dev).unwrap(),
        }
    }

    /// Builds a new StandardizationParams from a set of matrices.
    ///
    /// # Arguments
    /// * `mnist_data` - A slice of matrices containing the data to be standardized
    ///
    /// # Returns
    /// * `StandardizationParams` - A new StandardizationParams with the computed mean and standard deviation
    pub fn build(mnist_data: &[Matrix]) -> Result<StandardizationParams, MnistError> {
        if mnist_data.is_empty() {
            return Err(MnistError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "No data available for standardization",
            )));
        }

        let progress = ProgressBar::new(mnist_data.len() as u64);
        let style = ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:80.cyan/blue}] {pos:>7}/{len:7} {msg}",
            )
            .map_err(|e| e)?
            .progress_chars("##-");
        progress.set_style(style);
        progress.set_message("Processing matrices for standardization...");

        // Concatenate all matrices into a single ndarray
        let total_pixels: Array1<f64> = mnist_data
            .iter()
            .inspect(|_| progress.inc(1))
            .filter_map(|matrix| matrix.data.as_slice())
            .flat_map(|slice| slice.iter().copied())
            .collect();

        progress.finish();

        if total_pixels.is_empty() {
            return Err(MnistError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Failed to get data from matrices",
            )));
        }

        // Calculate mean and standard deviation using ndarray-stats
        let mean = total_pixels.mean().unwrap_or(0.0);
        let std_dev = total_pixels.std(0.0);

        Ok(StandardizationParams {
            mean: Mean::from(mean),
            std_dev: StdDev::try_from(std_dev).unwrap(),
        })
    }

    /// Returns the mean of the data.
    pub fn mean(&self) -> f64 {
        f64::from(self.mean)
    }

    /// Returns the standard deviation of the data.
    pub fn std_dev(&self) -> f64 {
        f64::from(self.std_dev)
    }
}

pub struct StandardizedMnistData<'a> {
    params: &'a StandardizationParams,
}

impl<'a> StandardizedMnistData<'a> {
    /// Returns a new StandardizedMnistData with the given MnistData and StandardizationParams.
    ///
    /// # Arguments
    /// * `params` - The StandardizationParams to be used for standardization
    ///
    /// # Returns
    /// * `StandardizedMnistData` - A new StandardizedMnistData with the given StandardizationParams
    pub fn new(params: &'a StandardizationParams) -> StandardizedMnistData<'a> {
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
        let progress = ProgressBar::new(mnist_data.len() as u64);

        let style = ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:80.cyan/blue}] {pos:>7}/{len:7} {msg}",
            )
            .map_err(|e| e)?
            .progress_chars("##-");
        progress.set_style(style);
        progress.set_message("Standardizing matrices...");
        let mut standardized = Vec::with_capacity(mnist_data.len());

        for matrix in mnist_data {
            standardized.push(self.standardize_matrix(matrix)?);
            progress.inc(1);
        }

        progress.finish();
        Ok(standardized)
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
            .map(|&x| (x - f64::from(self.params.mean)) / f64::from(self.params.std_dev))
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
        assert_relative_eq!(params.mean.0, 2.5, epsilon = 1e-10);
        assert_relative_eq!(params.std_dev.0, 1.118033988749895, epsilon = 1e-10);
    }

    #[test]
    fn test_standardize_matrix() {
        let data = vec![
            Matrix::new(2, 1, vec![1.0, 2.0]),
            Matrix::new(2, 1, vec![3.0, 4.0]),
        ];
        let params = StandardizationParams::build(&data);
        let params = params.expect("Failed to build standardization params");
        let standardized_data = StandardizedMnistData::new(&params)
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
