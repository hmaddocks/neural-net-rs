use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use matrix::matrix::Matrix;
use indicatif::{ProgressBar, ProgressStyle};
use thiserror::Error;

pub const IMAGE_MAGIC_NUMBER: u32 = 2051;
pub const LABEL_MAGIC_NUMBER: u32 = 2049;
pub const INPUT_NODES: usize = 784;  // 28x28 pixels
pub const OUTPUT_NODES: usize = 10;  // digits 0-9

#[derive(Debug, Error)]
pub enum MnistError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid magic number for {kind} file: expected {expected}, got {actual}")]
    InvalidMagicNumber {
        kind: &'static str,
        expected: u32,
        actual: u32,
    },
    #[error("Data mismatch: {0}")]
    DataMismatch(String),
}

#[derive(Debug)]
pub struct MnistData {
    images: Vec<Matrix>,
    labels: Vec<Matrix>,
}

impl MnistData {
    pub fn new(images: Vec<Matrix>, labels: Vec<Matrix>) -> Result<Self, MnistError> {
        if images.len() != labels.len() {
            return Err(MnistError::DataMismatch(
                format!("Number of images ({}) does not match number of labels ({})",
                    images.len(), labels.len())
            ));
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

fn read_u32(file: &mut File) -> std::io::Result<u32> {
    let mut buffer = [0; 4];
    file.read_exact(&mut buffer)?;
    Ok(u32::from_be_bytes(buffer))
}

fn create_progress_style(template: &str) -> ProgressStyle {
    ProgressStyle::with_template(template)
        .unwrap()
        .progress_chars("##-")
}

pub fn read_mnist_images(path: impl AsRef<Path>, progress: &ProgressBar) -> Result<Vec<Matrix>, MnistError> {
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

    progress.set_length(num_images as u64);
    progress.set_message("Loading images...");

    let mut images = Vec::with_capacity(num_images);
    let mut buffer = vec![0u8; pixels_per_image];

    for _ in 0..num_images {
        file.read_exact(&mut buffer)?;
        let data: Vec<f64> = buffer.iter()
            .map(|&pixel| f64::from(pixel) / 255.0)
            .collect();

        images.push(Matrix::new(pixels_per_image, 1, data));
        progress.inc(1);
    }

    progress.finish_with_message("Images loaded successfully");
    Ok(images)
}

pub fn read_mnist_labels(path: impl AsRef<Path>, progress: &ProgressBar) -> Result<Vec<Matrix>, MnistError> {
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

pub fn load_mnist_data() -> Result<MnistData, MnistError> {
    let multi_progress = indicatif::MultiProgress::new();
    let style = create_progress_style(
        "{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}"
    );

    let images_progress = multi_progress.add(ProgressBar::new(0));
    let labels_progress = multi_progress.add(ProgressBar::new(0));
    images_progress.set_style(style.clone());
    labels_progress.set_style(style);

    let home = std::env::var("HOME").expect("HOME environment variable not set");
    let images_path = PathBuf::from(&home).join("Documents/NMIST/train-images-idx3-ubyte");
    let labels_path = PathBuf::from(&home).join("Documents/NMIST/train-labels-idx1-ubyte");

    let images = read_mnist_images(images_path, &images_progress)?;
    let labels = read_mnist_labels(labels_path, &labels_progress)?;

    MnistData::new(images, labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_fs::prelude::*;
    use std::io::Write;

    fn create_test_progress_bar() -> ProgressBar {
        ProgressBar::hidden()
    }

    fn create_test_mnist_file(path: &Path, magic_number: u32, count: u32, data: &[u8]) -> std::io::Result<()> {
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
        let image_data = vec![0u8; 784 * 2];  // Two blank images
        create_test_mnist_file(
            file_path.path(),
            IMAGE_MAGIC_NUMBER,
            2,  // 2 images
            &image_data
        )?;

        let progress = create_test_progress_bar();
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
            0x12345678,  // Wrong magic number
            1,
            &vec![0u8; 784]
        )?;

        let progress = create_test_progress_bar();
        let result = read_mnist_images(file_path.path(), &progress);
        
        assert!(result.is_err());
        match result {
            Err(MnistError::InvalidMagicNumber { kind, expected, actual }) => {
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
            2,  // 2 labels
            &label_data
        )?;

        let progress = create_test_progress_bar();
        let result = read_mnist_labels(file_path.path(), &progress);
        
        assert!(result.is_ok());
        let labels = result.unwrap();
        assert_eq!(labels.len(), 2);
        
        // Check one-hot encoding
        assert_eq!(labels[0].data()[0], 1.0);  // First label should be 0
        assert_eq!(labels[1].data()[1], 1.0);  // Second label should be 1
        
        Ok(())
    }
}
