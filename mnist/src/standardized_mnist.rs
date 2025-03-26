use crate::mnist::{MnistData, MnistError};
use matrix::matrix::Matrix;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct StandardizationParams {
    pub mean: f64,
    pub std_dev: f64,
}

impl StandardizationParams {
    pub fn new(data: &[Matrix]) -> Self {
        let total_pixels: Vec<f64> = data
            .iter()
            .flat_map(|matrix| matrix.data.as_slice().unwrap())
            .copied()
            .collect();

        let mean = total_pixels.iter().sum::<f64>() / total_pixels.len() as f64;
        let variance = total_pixels
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / total_pixels.len() as f64;
        let std_dev = variance.sqrt();

        Self { mean, std_dev }
    }

    pub fn standardize(&self, matrix: &Matrix) -> Matrix {
        let standardized_data: Vec<f64> = matrix
            .data
            .as_slice()
            .unwrap()
            .iter()
            .map(|&x| (x - self.mean) / self.std_dev)
            .collect();

        Matrix::new(matrix.rows(), matrix.cols(), standardized_data)
    }
}

#[derive(Debug)]
pub struct StandardizedMnistData {
    pub data: MnistData,
    pub params: StandardizationParams,
}

impl StandardizedMnistData {
    pub fn new(mnist_data: MnistData) -> Result<Self, MnistError> {
        let params = StandardizationParams::new(mnist_data.images());

        let standardized_images: Vec<Matrix> = mnist_data
            .images()
            .iter()
            .map(|img| params.standardize(img))
            .collect();

        Ok(Self {
            data: MnistData::new(standardized_images, mnist_data.labels().to_vec())?,
            params,
        })
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
        let params = StandardizationParams::new(&data);

        assert_relative_eq!(params.mean, 2.5, epsilon = 1e-10);
        assert_relative_eq!(params.std_dev, 1.118033988749895, epsilon = 1e-10);
    }

    #[test]
    fn test_standardize_matrix() {
        let data = vec![
            Matrix::new(2, 1, vec![1.0, 2.0]),
            Matrix::new(2, 1, vec![3.0, 4.0]),
        ];
        let params = StandardizationParams::new(&data);

        let standardized = params.standardize(&data[0]);
        assert_relative_eq!(standardized.get(0, 0), -1.3416407864998738, epsilon = 1e-10);
        assert_relative_eq!(standardized.get(1, 0), -0.4472135954999579, epsilon = 1e-10);
    }
}
