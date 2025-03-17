//! MNIST dataset loader module for neural network training and testing.
//!
//! This module provides functionality to load and handle the MNIST dataset of handwritten digits.
//! It includes utilities for reading both images and labels from the IDX file format used by MNIST.
//! The data is normalized and converted into matrix format suitable for neural network training.

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

    pub fn len(&self) -> usize {
        self.images.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    pub fn images(&self) -> &[Matrix] {
        &self.images
    }

    pub fn labels(&self) -> &[Matrix] {
        &self.labels
    }
}

/// Creates a progress bar with a consistent style
pub(crate) fn create_progress_style(template: &str) -> ProgressStyle {
    ProgressStyle::default_bar()
        .template(template)
        .unwrap()
        .progress_chars("##-")
}

/// Reads a 32-bit unsigned integer in big-endian format from a file
fn read_u32(file: &mut File) -> std::io::Result<u32> {
    let mut buffer = [0; 4];
    file.read_exact(&mut buffer)?;
    Ok(u32::from_be_bytes(buffer))
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
pub fn read_mnist_images(
    path: impl AsRef<Path>,
    progress: &ProgressBar,
) -> Result<Vec<Matrix>, MnistError> {
    let mut file = File::open(path)?;

    let magic_number = read_u32(&mut file)?;
    if magic_number != IMAGE_MAGIC_NUMBER {
        return Err(MnistError::InvalidMagicNumber {
            kind: "images",
            expected: IMAGE_MAGIC_NUMBER,
            actual: magic_number,
        });
    }

    let num_images = read_u32(&mut file)? as usize;
    let num_rows = read_u32(&mut file)? as usize;
    let num_cols = read_u32(&mut file)? as usize;
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
pub fn read_mnist_labels(
    path: impl AsRef<Path>,
    progress: &ProgressBar,
) -> Result<Vec<Matrix>, MnistError> {
    let mut file = File::open(path)?;

    let magic_number = read_u32(&mut file)?;
    if magic_number != LABEL_MAGIC_NUMBER {
        return Err(MnistError::InvalidMagicNumber {
            kind: "labels",
            expected: LABEL_MAGIC_NUMBER,
            actual: magic_number,
        });
    }

    let num_labels = read_u32(&mut file)? as usize;
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

/// Returns the path to the MNIST data directory
pub fn get_mnist_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").expect("HOME environment variable not set"))
        .join("Documents")
        .join("NMIST")
}

/// Loads the standard MNIST training dataset
pub fn load_training_data() -> Result<MnistData, MnistError> {
    let mnist_dir = get_mnist_dir();
    load_mnist_data(
        mnist_dir.join("train-images-idx3-ubyte"),
        mnist_dir.join("train-labels-idx1-ubyte"),
    )
}

/// Loads the standard MNIST test dataset
pub fn load_test_data() -> Result<MnistData, MnistError> {
    let mnist_dir = get_mnist_dir();
    load_mnist_data(
        mnist_dir.join("t10k-images-idx3-ubyte"),
        mnist_dir.join("t10k-labels-idx1-ubyte"),
    )
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
pub fn load_mnist_data(
    images_path: PathBuf,
    labels_path: PathBuf,
) -> Result<MnistData, MnistError> {
    let multi_progress = indicatif::MultiProgress::new();
    let style = create_progress_style(
        "{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
    );

    let images_progress = multi_progress.add(ProgressBar::new(0));
    let labels_progress = multi_progress.add(ProgressBar::new(0));
    images_progress.set_style(style.clone());
    labels_progress.set_style(style);

    let images = read_mnist_images(images_path, &images_progress)?;
    let labels = read_mnist_labels(labels_path, &labels_progress)?;

    MnistData::new(images, labels)
}

/// Gets the actual digit from a one-hot encoded label matrix
pub fn get_actual_digit(label: &Matrix) -> usize {
    label
        .data()
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_fs::prelude::*;
    use std::io::Write;

    fn create_test_mnist_file(
        path: &Path,
        magic_number: u32,
        count: u32,
        data: &[u8],
    ) -> std::io::Result<()> {
        let mut file = File::create(path)?;

        // Write header
        file.write_all(&magic_number.to_be_bytes())?;
        file.write_all(&count.to_be_bytes())?;

        if magic_number == IMAGE_MAGIC_NUMBER {
            // Add image dimensions (28x28)
            file.write_all(&28u32.to_be_bytes())?;
            file.write_all(&28u32.to_be_bytes())?;
        }

        // Write data
        file.write_all(data)?;
        Ok(())
    }

    #[test]
    fn test_mnist_data_new_valid() {
        let images = vec![Matrix::zeros(784, 1), Matrix::zeros(784, 1)];
        let labels = vec![Matrix::zeros(10, 1), Matrix::zeros(10, 1)];

        let result = MnistData::new(images, labels);
        assert!(result.is_ok());

        let data = result.unwrap();
        assert_eq!(data.len(), 2);
        assert!(!data.is_empty());
    }

    #[test]
    fn test_mnist_data_new_mismatch() {
        let images = vec![Matrix::zeros(784, 1)];
        let labels = vec![Matrix::zeros(10, 1), Matrix::zeros(10, 1)];

        let result = MnistData::new(images, labels);
        assert!(result.is_err());

        match result {
            Err(MnistError::DataMismatch(msg)) => {
                assert!(msg.contains("does not match"));
            }
            _ => panic!("Expected DataMismatch error"),
        }
    }

    #[test]
    fn test_read_mnist_images_valid() -> Result<(), Box<dyn std::error::Error>> {
        let temp = assert_fs::TempDir::new()?;
        let file_path = temp.child("test-images");

        // Create test image data: 2 images of 28x28 pixels
        let image_data = vec![0u8; 784 * 2]; // Two blank images
        create_test_mnist_file(
            file_path.path(),
            IMAGE_MAGIC_NUMBER,
            2, // 2 images
            &image_data,
        )?;

        let progress = ProgressBar::new(2);
        let result = read_mnist_images(file_path.path(), &progress);

        assert!(result.is_ok());
        let images = result.unwrap();
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].rows(), 784);
        assert_eq!(images[0].cols(), 1);

        Ok(())
    }

    #[test]
    fn test_read_mnist_images_invalid_magic() -> Result<(), Box<dyn std::error::Error>> {
        let temp = assert_fs::TempDir::new()?;
        let file_path = temp.child("test-images");

        // Create test file with wrong magic number
        create_test_mnist_file(
            file_path.path(),
            0x12345678, // Wrong magic number
            1,
            &vec![0u8; 784],
        )?;

        let progress = ProgressBar::new(2);
        let result = read_mnist_images(file_path.path(), &progress);

        assert!(result.is_err());
        match result {
            Err(MnistError::InvalidMagicNumber {
                kind,
                expected,
                actual,
            }) => {
                assert_eq!(kind, "images");
                assert_eq!(expected, IMAGE_MAGIC_NUMBER);
                assert_eq!(actual, 0x12345678);
            }
            _ => panic!("Expected InvalidMagicNumber error"),
        }

        Ok(())
    }

    #[test]
    fn test_read_mnist_labels_valid() -> Result<(), Box<dyn std::error::Error>> {
        let temp = assert_fs::TempDir::new()?;
        let file_path = temp.child("test-labels");

        // Create test label data: 2 labels (0 and 1)
        let label_data = vec![0u8, 1u8];
        create_test_mnist_file(
            file_path.path(),
            LABEL_MAGIC_NUMBER,
            2, // 2 labels
            &label_data,
        )?;

        let progress = ProgressBar::new(2);
        let result = read_mnist_labels(file_path.path(), &progress);

        assert!(result.is_ok());
        let labels = result.unwrap();
        assert_eq!(labels.len(), 2);

        // Check one-hot encoding
        assert_eq!(labels[0].data()[0], 1.0); // First label should be 0
        assert_eq!(labels[1].data()[1], 1.0); // Second label should be 1

        Ok(())
    }
}
