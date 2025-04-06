//! MNIST dataset loader module for neural network training and testing.
//!
//! This module provides functionality to load and handle the MNIST dataset of handwritten digits.
//! It includes utilities for reading both images and labels from the IDX file format used by MNIST.
//! The data is normalized and converted into matrix format suitable for neural network training.

use indicatif::style::TemplateError;
use indicatif::{ProgressBar, ProgressStyle};
use matrix::matrix::Matrix;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const IMAGE_MAGIC_NUMBER: u32 = 2051;
pub const LABEL_MAGIC_NUMBER: u32 = 2049;
pub const INPUT_NODES: usize = 784;
pub const OUTPUT_NODES: usize = 10;

/// Errors that can occur while handling MNIST data
#[derive(Debug, Error)]
pub enum MnistError {
    /// Wrapper for standard I/O errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Error for invalid magic numbers in MNIST files
    #[error("Invalid magic number for {kind} file: expected {expected}, got {actual}")]
    InvalidMagicNumber {
        kind: &'static str,
        expected: u32,
        actual: u32,
    },
    /// Error for mismatches between images and labels
    #[error("Data mismatch: {0}")]
    DataMismatch(String),
    /// Error for invalid image dimensions
    #[error(
        "Invalid image dimensions: expected {expected} pixels, got {actual} pixels ({rows}x{cols})"
    )]
    InvalidDimensions {
        expected: usize,
        actual: usize,
        rows: usize,
        cols: usize,
    },
    /// Error for invalid progress bar styles
    #[error("Failed to set progress bar style: {0}")]
    ProgressStyleError(#[from] TemplateError),
    /// Error for empty label matrix
    #[error("Label matrix is empty")]
    EmptyLabelMatrix,
    /// Error for label matrix with incorrect dimensions
    #[error("Label matrix has incorrect dimensions: expected {expected_rows}x{expected_cols}, got {actual_rows}x{actual_cols}")]
    InvalidLabelDimensions {
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },
    /// Error when the expected digit cannot be found in the label matrix
    #[error("Could not find the digit in the label matrix")]
    DigitNotFound,
}

/// Container for MNIST dataset pairs (images and their corresponding labels)
#[derive(Debug)]
pub struct MnistData {
    images: Vec<Matrix>,
    labels: Vec<Matrix>,
}

impl MnistData {
    /// Creates a new MnistData instance from vectors of image and label matrices.
    ///
    /// # Arguments
    /// * `images` - Vector of matrices representing the images
    /// * `labels` - Vector of matrices representing one-hot encoded labels
    ///
    /// # Returns
    /// * `Ok(MnistData)` if the number of images matches the number of labels
    /// * `Err(MnistError::DataMismatch)` if there's a mismatch between images and labels
    ///
    /// # Example
    /// ```
    /// use mnist::mnist::MnistData;
    /// use matrix::matrix::Matrix;
    ///
    /// let images = vec![Matrix::new(784, 1, vec![0.5; 784])];
    /// let labels = vec![Matrix::new(10, 1, vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])];
    /// let mnist_data = MnistData::new(images, labels).unwrap();
    /// ```
    pub fn new(images: Vec<Matrix>, labels: Vec<Matrix>) -> Result<Self, MnistError> {
        if images.len() != labels.len() {
            return Err(MnistError::DataMismatch(format!(
                "Number of images ({}) does not match number of labels ({})",
                images.len(),
                labels.len()
            )));
        }
        Ok(Self { images, labels })
    }

    /// Loads MNIST images and labels from the specified file paths.
    ///
    /// # Arguments
    /// * `images_path` - Path to the images file
    /// * `labels_path` - Path to the labels file
    ///
    /// # Returns
    /// * `Ok(MnistData)` containing paired images and labels
    /// * `Err(MnistError)` if loading fails
    fn load_mnist_data(
        images_path: PathBuf,
        labels_path: PathBuf,
    ) -> Result<MnistData, MnistError> {
        let multi_progress = indicatif::MultiProgress::new();
        let style = ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:80.cyan/blue}] {pos:>7}/{len:7} {msg}",
            )
            .map_err(|e| e)?
            .progress_chars("##-");

        let images_progress = multi_progress.add(ProgressBar::new(0));
        let labels_progress = multi_progress.add(ProgressBar::new(0));
        images_progress.set_style(style.clone());
        labels_progress.set_style(style);

        let images = Self::read_mnist_images(images_path, &images_progress)?;
        let labels = Self::read_mnist_labels(labels_path, &labels_progress)?;

        Ok(Self { images, labels })
    }

    pub fn len(&self) -> usize {
        self.images.len()
    }

    pub fn images(&self) -> &[Matrix] {
        &self.images
    }

    pub fn labels(&self) -> &[Matrix] {
        &self.labels
    }

    /// Reads a 32-bit unsigned integer in big-endian format from a file
    fn read_u32(file: &mut File) -> std::io::Result<u32> {
        let mut buffer = [0; 4];
        match file.read_exact(&mut buffer) {
            Ok(_) => Ok(u32::from_be_bytes(buffer)),
            Err(e) => Err(e),
        }
    }

    /// Reads MNIST image data from an IDX file format.
    ///
    /// # Arguments
    /// * `path` - Path to the MNIST image file
    /// * `progress` - Progress bar for tracking loading progress
    ///
    /// # Returns
    /// * `Ok(Vec<Matrix>)` containing normalized image matrices (pixel values scaled to 0.0-1.0)
    /// * `Err(MnistError)` if file reading fails or format is invalid
    ///
    /// # Format
    /// The IDX file format consists of:
    /// * 32-bit magic number (2051)
    /// * 32-bit number of images
    /// * 32-bit number of rows
    /// * 32-bit number of columns
    /// * Pixels in row-major order (1 byte per pixel)
    fn read_mnist_images(
        path: impl AsRef<Path>,
        progress: &ProgressBar,
    ) -> Result<Vec<Matrix>, MnistError> {
        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(e) => return Err(e.into()),
        };

        let magic_number = Self::read_u32(&mut file)?;
        if magic_number != IMAGE_MAGIC_NUMBER {
            return Err(MnistError::InvalidMagicNumber {
                kind: "images",
                expected: IMAGE_MAGIC_NUMBER,
                actual: magic_number,
            });
        }

        let num_images = Self::read_u32(&mut file)? as usize;
        let num_rows = Self::read_u32(&mut file)? as usize;
        let num_cols = Self::read_u32(&mut file)? as usize;
        let pixels_per_image = num_rows * num_cols;

        if pixels_per_image != INPUT_NODES {
            return Err(MnistError::InvalidDimensions {
                expected: INPUT_NODES,
                actual: pixels_per_image,
                rows: num_rows,
                cols: num_cols,
            });
        }

        progress.set_length(num_images as u64);
        progress.set_message("Loading images...");

        let mut images = Vec::with_capacity(num_images);
        let mut buffer = vec![0u8; pixels_per_image];

        for _ in 0..num_images {
            file.read_exact(&mut buffer)?;
            let data = Vec::from_iter(buffer.iter().map(|&pixel| f64::from(pixel) / 255.0));
            images.push(Matrix::new(pixels_per_image, 1, data));
            progress.inc(1);
        }

        progress.finish_with_message("Images loaded successfully");
        Ok(images)
    }

    /// Reads MNIST label data from an IDX file format.
    ///
    /// # Arguments
    /// * `path` - Path to the MNIST label file
    /// * `progress` - Progress bar for tracking loading progress
    ///
    /// # Returns
    /// * `Ok(Vec<Matrix>)` containing one-hot encoded label matrices
    /// * `Err(MnistError)` if file reading fails or format is invalid
    ///
    /// # Format
    /// The IDX file format consists of:
    /// * 32-bit magic number (2049)
    /// * 32-bit number of labels
    /// * Labels (1 byte per label)
    fn read_mnist_labels(
        path: impl AsRef<Path>,
        progress: &ProgressBar,
    ) -> Result<Vec<Matrix>, MnistError> {
        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(e) => return Err(e.into()),
        };

        let magic_number = Self::read_u32(&mut file)?;
        if magic_number != LABEL_MAGIC_NUMBER {
            return Err(MnistError::InvalidMagicNumber {
                kind: "labels",
                expected: LABEL_MAGIC_NUMBER,
                actual: magic_number,
            });
        }

        let num_labels = Self::read_u32(&mut file)? as usize;
        progress.set_length(num_labels as u64);
        progress.set_message("Loading labels...");

        let mut labels = Vec::with_capacity(num_labels);
        let mut buffer = [0u8; 1];

        for _ in 0..num_labels {
            file.read_exact(&mut buffer)?;
            let mut data = vec![0.0; OUTPUT_NODES];
            data[buffer[0] as usize] = 1.0;

            labels.push(Matrix::new(OUTPUT_NODES, 1, data));
            progress.inc(1);
        }

        progress.finish_with_message("Labels loaded successfully");
        Ok(labels)
    }
}

/// Returns the path to the MNIST data directory
fn get_mnist_dir() -> Result<PathBuf, MnistError> {
    let current_dir = match std::env::current_dir() {
        Ok(dir) => dir,
        Err(e) => {
            eprintln!("Failed to get current directory: {}", e);
            return Err(e.into());
        }
    };
    Ok(current_dir.join("mnist").join("data"))
}

/// Loads the standard MNIST training dataset
pub fn load_training_data() -> Result<MnistData, MnistError> {
    let mnist_dir = get_mnist_dir()?;
    MnistData::load_mnist_data(
        mnist_dir.join("train-images-idx3-ubyte"),
        mnist_dir.join("train-labels-idx1-ubyte"),
    )
}

/// Loads the standard MNIST test dataset
pub fn load_test_data() -> Result<MnistData, MnistError> {
    let mnist_dir = get_mnist_dir()?;
    MnistData::load_mnist_data(
        mnist_dir.join("t10k-images-idx3-ubyte"),
        mnist_dir.join("t10k-labels-idx1-ubyte"),
    )
}

/// Gets the actual digit from a one-hot encoded label matrix
///
/// # Arguments
/// * `label` - One-hot encoded label matrix
///
/// # Returns
/// * `usize` - Actual digit
///
/// # Example
/// ```
/// let label = matrix::Matrix::new(10, 1, vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1]);
/// assert!(mnist::get_actual_digit(&label).is_ok());
/// assert_eq!(mnist::get_actual_digit(&label).unwrap(), 8);
/// ```
pub fn get_actual_digit(label: &Matrix) -> Result<usize, MnistError> {
    // Check if the matrix has zero rows or columns, indicating it's effectively empty
    if label.rows() == 0 || label.cols() == 0 {
        return Err(MnistError::EmptyLabelMatrix);
    }
    if label.rows() != OUTPUT_NODES || label.cols() != 1 {
        return Err(MnistError::InvalidLabelDimensions {
            expected_rows: OUTPUT_NODES,
            expected_cols: 1,
            actual_rows: label.rows(),
            actual_cols: label.cols(),
        });
    }

    let mut max_val = label.get(0, 0);
    let mut max_idx = 0;

    // Find the index (digit) with the highest probability
    for i in 1..label.rows() {
        let val = label.get(i, 0);
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    // Basic sanity check: If the max value is very small, something might be wrong
    // or it could just be an uncertain prediction. We'll assume the index is valid
    // if we found *some* max value. A check for exactly 1.0 isn't robust if the
    // input isn't perfectly one-hot.
    if max_val >= 0.0 {
        // Check if any value is significantly high (e.g., > 0.5)
        // This is a heuristic check. In a true one-hot vector, one value would be 1.0.
        // In a probability distribution from softmax, one value is typically highest.
        // If all values are small, it might indicate an issue, but we still return the max index.
        Ok(max_idx)
    } else {
        // This case should theoretically not happen if label contains non-negative probabilities
        Err(MnistError::DigitNotFound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_get_actual_digit() {
        // Test case 1: Valid label for digit 8
        let label1_result = get_actual_digit(&Matrix::new(
            10,
            1,
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, // One-hot for 8
            ],
        ));
        assert!(label1_result.is_ok());
        assert_eq!(label1_result.unwrap(), 8);

        // Test case 2: Valid label for digit 0 (slightly noisy probabilities)
        let label2_result = get_actual_digit(&Matrix::new(
            10,
            1,
            vec![
                0.9, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, // Highest prob for 0
            ],
        ));
        assert!(label2_result.is_ok());
        assert_eq!(label2_result.unwrap(), 0);

        // Test case 3: Empty matrix
        let label3 = Matrix::new(0, 0, vec![]); // Use constructor for 0x0 matrix
        assert!(matches!(
            get_actual_digit(&label3),
            Err(MnistError::EmptyLabelMatrix)
        ));

        // Test case 4: Incorrect dimensions (wrong rows)
        let label4 = Matrix::new(9, 1, vec![0.1; 9]);
        assert!(matches!(
            get_actual_digit(&label4),
            Err(MnistError::InvalidLabelDimensions { .. }) // Check the variant type
        ));

        // Test case 5: Incorrect dimensions (wrong cols)
        let label5 = Matrix::new(10, 2, vec![0.1; 20]);
        assert!(matches!(
            get_actual_digit(&label5),
            Err(MnistError::InvalidLabelDimensions { .. }) // Check the variant type
        ));

        // Test case 6: All zeros (should still find index 0 as max, technically)
        // Depending on interpretation, this could be an error or return 0.
        // Current implementation finds the first max, which is index 0.
        let label6 = Matrix::new(10, 1, vec![0.0; 10]);
        let label6_result = get_actual_digit(&label6);
        assert!(label6_result.is_ok());
        assert_eq!(label6_result.unwrap(), 0);

        // Test case 7: Negative values (should theoretically not happen, but test edge case)
        // If max_val is negative, the current logic might hit the DigitNotFound error,
        // although finding the max index should still work mathematically. Let's test.
        let label7 = Matrix::new(10, 1, vec![-0.5; 10]);
        // Max value is -0.5, max index is 0. The check `max_val >= 0.0` fails.
        assert!(matches!(
            get_actual_digit(&label7),
            Err(MnistError::DigitNotFound)
        ));
    }

    // Helper function to create a dummy MNIST file for testing
    fn create_test_mnist_file(
        path: &Path,
        magic_number: u32,
        count: u32,
        data: &[u8],
    ) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        file.write_all(&magic_number.to_be_bytes())?;
        file.write_all(&count.to_be_bytes())?;

        if magic_number == IMAGE_MAGIC_NUMBER {
            // For images, write dimensions (28x28)
            file.write_all(&28u32.to_be_bytes())?;
            file.write_all(&28u32.to_be_bytes())?;
        }

        file.write_all(data)?;
        Ok(())
    }

    #[test]
    fn test_mnist_data_new_valid() {
        let images = vec![Matrix::new(784, 1, vec![0.5; 784])];
        let labels = vec![Matrix::new(
            10,
            1,
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )];
        let mnist_data = MnistData::new(images, labels).unwrap();
        assert_eq!(mnist_data.len(), 1);
    }

    #[test]
    fn test_mnist_data_new_mismatch() {
        let images = vec![Matrix::new(784, 1, vec![0.5; 784])];
        let labels = vec![
            Matrix::new(
                10,
                1,
                vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            Matrix::new(
                10,
                1,
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
        ];
        assert!(MnistData::new(images, labels).is_err());
    }

    #[test]
    fn test_read_mnist_images_valid() {
        let dir = tempdir().unwrap();
        let image_path = dir.path().join("test_images");

        // Create test image file with 2 images
        let image_data: Vec<u8> = (0..784 * 2).map(|i| (i % 256) as u8).collect();
        create_test_mnist_file(&image_path, IMAGE_MAGIC_NUMBER, 2, &image_data).unwrap();

        let progress_bar = ProgressBar::new(0);
        let images = MnistData::read_mnist_images(image_path, &progress_bar).unwrap();

        assert_eq!(images.len(), 2);
        assert_eq!(images[0].rows(), 784);
        assert_eq!(images[0].cols(), 1);
        assert_eq!(images[1].rows(), 784);
        assert_eq!(images[1].cols(), 1);
    }

    #[test]
    fn test_read_mnist_images_invalid_magic() {
        let dir = tempdir().unwrap();
        let image_path = dir.path().join("test_images");

        // Create test image file with wrong magic number
        let image_data: Vec<u8> = (0..784 * 2).map(|i| (i % 256) as u8).collect();
        create_test_mnist_file(&image_path, 1234, 2, &image_data).unwrap();

        let progress_bar = ProgressBar::new(0);
        assert!(MnistData::read_mnist_images(image_path, &progress_bar).is_err());
    }

    #[test]
    fn test_read_mnist_labels_valid() {
        let dir = tempdir().unwrap();
        let label_path = dir.path().join("test_labels");

        // Create test label file with 2 labels
        let label_data = vec![0u8, 1u8];
        create_test_mnist_file(&label_path, LABEL_MAGIC_NUMBER, 2, &label_data).unwrap();

        let progress_bar = ProgressBar::new(0);
        let labels = MnistData::read_mnist_labels(label_path, &progress_bar).unwrap();

        assert_eq!(labels.len(), 2);
        assert_eq!(labels[0].rows(), 10);
        assert_eq!(labels[0].cols(), 1);
        assert_eq!(labels[1].rows(), 10);
        assert_eq!(labels[1].cols(), 1);

        // Check one-hot encoding
        assert_eq!(labels[0].get(0, 0), 1.0); // First label should be 0
        assert_eq!(labels[1].get(1, 0), 1.0); // Second label should be 1
    }
}
