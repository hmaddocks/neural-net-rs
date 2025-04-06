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
/// assert_eq!(mnist::get_actual_digit(&label), Ok(8));
/// ```
pub fn get_actual_digit(label: &Matrix) -> Result<usize, &'static str> {
    if label.data.is_empty() {
        return Err("Empty matrix provided");
    }
    label
        .data
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.partial_cmp(b)
                .ok_or("Cannot compare values (possible NaN)")
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .ok_or("Failed to find maximum value")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_get_actual_digit() {
        // Test case 1: One-hot encoded vector for digit 0
        let digit_0 = Matrix::new(
            10,
            1,
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_eq!(get_actual_digit(&digit_0), Ok(0));

        // Test case 2: One-hot encoded vector for digit 5
        let digit_5 = Matrix::new(
            10,
            1,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_eq!(get_actual_digit(&digit_5), Ok(5));

        // Test case 4: Non-binary values (softmax output)
        let softmax_output = Matrix::new(
            10,
            1,
            vec![0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.5, 0.05, 0.05],
        );
        assert_eq!(get_actual_digit(&softmax_output), Ok(7));

        // Test case 5: Very close values
        let close_values = Matrix::new(
            10,
            1,
            vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1001, 0.1],
        );
        assert_eq!(get_actual_digit(&close_values), Ok(8));
    }

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
