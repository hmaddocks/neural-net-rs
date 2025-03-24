//! Module for standardizing MNIST dataset.
//!
//! This module provides functionality to standardize MNIST data by
//! subtracting the mean and dividing by the standard deviation.

use crate::mnist::{MnistData, MnistError};
use matrix::matrix::Matrix;

/// Represents standardized MNIST data
#[derive(Debug)]
pub struct StandardizedMnistData {
    data: MnistData,
    mean: f64,
    std_dev: f64,
}

impl StandardizedMnistData {
    /// Creates a new StandardizedMnistData by standardizing the given MnistData.
    ///
    /// Standardization formula: (x - mean) / std_dev
    ///
    /// # Arguments
    /// * `data` - The MNIST data to standardize
    ///
    /// # Returns
    /// * `Ok(StandardizedMnistData)` containing the standardized data
    /// * `Err(MnistError)` if standardization fails
    pub fn new(data: MnistData) -> Result<Self, MnistError> {
        let (standardized_data, mean, std_dev) = standardize_mnist_data(data)?;
        Ok(Self {
            data: standardized_data,
            mean,
            std_dev,
        })
    }

    /// Returns a reference to the standardized MNIST data
    pub fn data(&self) -> &MnistData {
        &self.data
    }

    /// Returns the mean value used for standardization
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Returns the standard deviation value used for standardization
    pub fn std_dev(&self) -> f64 {
        self.std_dev
    }
}

/// Calculates the mean of all pixel values across all images
fn calculate_mean(images: &[Matrix]) -> f64 {
    if images.is_empty() {
        return 0.0;
    }

    let total_pixels: usize = images.iter().map(|img| img.rows() * img.cols()).sum();
    let sum: f64 = images
        .iter()
        .flat_map(|img| img.data.iter())
        .sum();

    sum / total_pixels as f64
}

/// Calculates the standard deviation of all pixel values across all images
fn calculate_std_dev(images: &[Matrix], mean: f64) -> f64 {
    if images.is_empty() {
        return 1.0;
    }

    let total_pixels: usize = images.iter().map(|img| img.rows() * img.cols()).sum();
    let sum_squared_diff: f64 = images
        .iter()
        .flat_map(|img| img.data.iter().map(|&pixel| (pixel - mean).powi(2)))
        .sum();

    (sum_squared_diff / total_pixels as f64).sqrt()
}

/// Standardizes MNIST data by subtracting the mean and dividing by the standard deviation
///
/// # Arguments
/// * `mnist_data` - The MNIST data to standardize
///
/// # Returns
/// * `Ok((MnistData, mean, std_dev))` containing the standardized data and standardization parameters
/// * `Err(MnistError)` if standardization fails
fn standardize_mnist_data(mnist_data: MnistData) -> Result<(MnistData, f64, f64), MnistError> {
    let images = mnist_data.images();
    let labels = mnist_data.labels();

    let mean = calculate_mean(images);
    let std_dev = calculate_std_dev(images, mean);

    // Avoid division by zero
    let std_dev = if std_dev == 0.0 { 1.0 } else { std_dev };

    let standardized_images: Vec<Matrix> = images
        .iter()
        .map(|img| {
            let standardized_data: Vec<f64> = img
                .data
                .iter()
                .map(|&pixel| (pixel - mean) / std_dev)
                .collect();
            
            Matrix::new(img.rows(), img.cols(), standardized_data)
        })
        .collect();

    // Create a new MnistData with standardized images and original labels
    let standardized_mnist_data = MnistData::new(standardized_images, labels.to_vec())?;

    Ok((standardized_mnist_data, mean, std_dev))
}

/// Loads and standardizes the MNIST training dataset
pub fn load_standardized_training_data() -> Result<StandardizedMnistData, MnistError> {
    let training_data = crate::mnist::load_training_data()?;
    StandardizedMnistData::new(training_data)
}

/// Loads and standardizes the MNIST test dataset
pub fn load_standardized_test_data() -> Result<StandardizedMnistData, MnistError> {
    let test_data = crate::mnist::load_test_data()?;
    StandardizedMnistData::new(test_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mnist::MnistData;
    use matrix::matrix::Matrix;

    #[test]
    fn test_calculate_mean() {
        // Create a simple set of test images
        let images = vec![
            Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
            Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]),
        ];
        
        let mean = calculate_mean(&images);
        assert_eq!(mean, 4.5); // (1+2+3+4+5+6+7+8)/8 = 36/8 = 4.5
    }

    #[test]
    fn test_calculate_std_dev() {
        // Create a simple set of test images
        let images = vec![
            Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
            Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]),
        ];
        
        let mean = 4.5;
        let std_dev = calculate_std_dev(&images, mean);
        
        // Variance: ((1-4.5)^2 + (2-4.5)^2 + ... + (8-4.5)^2) / 8
        // = (3.5^2 + 2.5^2 + 1.5^2 + 0.5^2 + 0.5^2 + 1.5^2 + 2.5^2 + 3.5^2) / 8
        // = (12.25 + 6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25 + 12.25) / 8
        // = 42 / 8 = 5.25
        // Std Dev: sqrt(5.25) ≈ 2.29
        assert!((std_dev - 2.29).abs() < 0.01);
    }

    #[test]
    fn test_standardize_mnist_data() {
        // Create test data
        let images = vec![
            Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
            Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]),
        ];
        
        let labels = vec![
            Matrix::new(2, 1, vec![1.0, 0.0]),
            Matrix::new(2, 1, vec![0.0, 1.0]),
        ];
        
        let mnist_data = MnistData::new(images, labels).unwrap();
        let (standardized_data, mean, std_dev) = standardize_mnist_data(mnist_data).unwrap();
        
        assert_eq!(mean, 4.5);
        assert!((std_dev - 2.29).abs() < 0.01);
        
        // Check that the standardized values are correct
        let standardized_images = standardized_data.images();
        
        // First image, first pixel: (1.0 - 4.5) / 2.29 ≈ -1.53
        assert!((standardized_images[0].get(0, 0) - (-1.53)).abs() < 0.01);
        
        // Second image, last pixel: (8.0 - 4.5) / 2.29 ≈ 1.53
        assert!((standardized_images[1].get(1, 1) - 1.53).abs() < 0.01);
    }
}
