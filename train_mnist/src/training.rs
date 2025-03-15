//! Training module for the neural network implementation.
//!
//! This module provides the training infrastructure for the neural network, including:
//! - Configuration management via `TrainingConfig`
//! - Training loop implementation with early stopping
//! - Progress visualization using progress bars
//! - Model persistence through save/load functionality

use crate::mnist::{MnistData, MnistError, INPUT_NODES, OUTPUT_NODES};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use matrix::matrix::Matrix;
use neural_network::activations::SIGMOID;
use neural_network::network::Network;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::path::Path;

/// Configuration parameters for neural network training.
#[derive(Debug)]
pub struct TrainingConfig {
    /// Size of each training batch
    pub batch_size: usize,
    /// Number of training epochs
    pub epochs: u32,
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Number of nodes in each hidden layer
    pub hidden_layers: Vec<usize>,
    /// Number of epochs to wait for improvement before early stopping
    pub early_stopping_patience: u32,
    /// Minimum improvement in accuracy required to reset patience counter
    pub early_stopping_min_delta: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            epochs: 30,
            learning_rate: 0.1,
            hidden_layers: vec![128, 64],
            early_stopping_patience: 5,
            early_stopping_min_delta: 0.001,
        }
    }
}

/// Trainer manages the neural network training process.
///
/// The trainer handles:
/// - Network initialization
/// - Training loop execution
/// - Early stopping
/// - Progress visualization
/// - Model persistence
pub struct Trainer {
    network: Network,
    config: TrainingConfig,
}

impl Trainer {
    /// Creates a new trainer with the specified configuration.
    ///
    /// # Arguments
    /// * `config` - Training configuration parameters
    pub fn new(config: TrainingConfig) -> Self {
        let mut layer_sizes = vec![INPUT_NODES];
        layer_sizes.extend(&config.hidden_layers);
        layer_sizes.push(OUTPUT_NODES);

        let network = Network::new(layer_sizes, SIGMOID, config.learning_rate);

        Self { network, config }
    }

    /// Trains the neural network using the provided MNIST data.
    ///
    /// This function:
    /// - Initializes progress bars for visualization
    /// - Shuffles training data for each epoch
    /// - Processes data in batches
    /// - Updates network weights using backpropagation
    /// - Implements early stopping based on accuracy plateaus
    ///
    /// # Arguments
    /// * `data` - MNIST training data
    ///
    /// # Returns
    /// * `Result<(), MnistError>` - Ok if training completes successfully
    pub fn train(&mut self, data: &MnistData) -> Result<(), MnistError> {
        let multi_progress = MultiProgress::new();
        let epoch_style = create_progress_style(
            "{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} Epoch {msg}",
        );
        let batch_style = create_progress_style(
            "{spinner:.yellow} [{elapsed_precise}] {bar:40.yellow/blue} {pos:>7}/{len:7} Batch {msg}"
        );

        let epoch_progress = multi_progress.add(ProgressBar::new(self.config.epochs as u64));
        let batch_progress = multi_progress.add(ProgressBar::new(0));
        epoch_progress.set_style(epoch_style);
        batch_progress.set_style(batch_style);

        println!(
            "\nStarting training with batch size {}",
            self.config.batch_size
        );
        let mut indices: Vec<usize> = (0..data.len()).collect();
        let mut rng = thread_rng();

        let mut best_accuracy = 0.0;
        let mut patience_counter = 0;

        for epoch in 1..=self.config.epochs {
            indices.shuffle(&mut rng);
            let (mut correct, mut total) = (0, 0);

            batch_progress.set_length((indices.len() / self.config.batch_size) as u64);
            batch_progress.set_position(0);
            batch_progress.set_message(format!("in Epoch {}", epoch));

            for batch_indices in indices.chunks(self.config.batch_size) {
                for &idx in batch_indices {
                    let output = self
                        .network
                        .feed_forward(&data.images()[idx])
                        .map_err(|e| MnistError::DataMismatch(e.to_string()))?;
                    self.network
                        .back_propogate(output.clone(), &data.labels()[idx]);

                    if self.get_prediction(&output) == self.get_prediction(&data.labels()[idx]) {
                        correct += 1;
                    }
                    total += 1;
                }
                batch_progress.inc(1);
            }

            let accuracy = (correct as f64 / total as f64) * 100.0;
            epoch_progress.set_message(format!("- Accuracy: {:.2}%", accuracy));
            epoch_progress.inc(1);

            // Early stopping check
            if accuracy > best_accuracy + self.config.early_stopping_min_delta {
                best_accuracy = accuracy;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.early_stopping_patience {
                    epoch_progress.finish_with_message(format!(
                        "Early stopping at epoch {} with best accuracy: {:.2}%",
                        epoch, best_accuracy
                    ));
                    batch_progress.finish_and_clear();
                    return Ok(());
                }
            }
        }

        epoch_progress.finish_with_message("Training completed!");
        batch_progress.finish_and_clear();
        Ok(())
    }

    /// Gets the predicted class index from a network output matrix.
    ///
    /// # Arguments
    /// * `matrix` - Output matrix from the neural network
    ///
    /// # Returns
    /// * `usize` - Index of the highest probability class
    pub fn get_prediction(&self, matrix: &Matrix) -> usize {
        matrix
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap()
    }

    /// Saves the trained network to a file in JSON format.
    ///
    /// # Arguments
    /// * `path` - Path where the network should be saved
    ///
    /// # Returns
    /// * `Result<(), MnistError>` - Ok if save succeeds
    pub fn save_network<P: AsRef<Path>>(&self, path: P) -> Result<(), MnistError> {
        self.network
            .save(path)
            .map_err(|e| MnistError::DataMismatch(e.to_string()))
    }

    /// Loads a trained network from a file.
    ///
    /// # Arguments
    /// * `path` - Path to the saved network file
    /// * `config` - Configuration for the trainer
    ///
    /// # Returns
    /// * `Result<Self, MnistError>` - Ok(Trainer) if load succeeds
    pub fn load_network<P: AsRef<Path>>(
        path: P,
        config: TrainingConfig,
    ) -> Result<Self, MnistError> {
        let network = Network::load(path).map_err(|e| MnistError::DataMismatch(e.to_string()))?;
        Ok(Self { network, config })
    }
}

/// Creates a progress bar style with the specified template.
///
/// # Arguments
/// * `template` - Template string for the progress bar
///
/// # Returns
/// * `ProgressStyle` - Configured progress bar style
fn create_progress_style(template: &str) -> ProgressStyle {
    ProgressStyle::with_template(template)
        .unwrap()
        .progress_chars("##-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.batch_size, 100);
        assert_eq!(config.epochs, 30);
        assert_eq!(config.learning_rate, 0.1);
        assert_eq!(config.hidden_layers, vec![128, 64]);
        assert_eq!(config.early_stopping_patience, 5);
        assert_eq!(config.early_stopping_min_delta, 0.001);
    }

    #[test]
    fn test_trainer_initialization() {
        let config = TrainingConfig::default();
        let mut trainer = Trainer::new(config);

        // Test the trainer by feeding forward a sample input
        let input = Matrix::zeros(INPUT_NODES, 1);
        let result = trainer.network.feed_forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.rows(), OUTPUT_NODES);
        assert_eq!(output.cols(), 1);
    }

    #[test]
    fn test_get_prediction() {
        let config = TrainingConfig::default();
        let trainer = Trainer::new(config);

        // Create a test output matrix with a clear maximum
        let mut data = vec![0.1; 10];
        data[3] = 0.9; // Make class 3 the predicted class
        let output = Matrix::new(10, 1, data);

        assert_eq!(trainer.get_prediction(&output), 3);
    }

    #[test]
    fn test_trainer_single_batch() -> Result<(), MnistError> {
        let config = TrainingConfig {
            batch_size: 2,
            epochs: 1,
            learning_rate: 0.1,
            hidden_layers: vec![4], // Smaller network for testing
            early_stopping_patience: 5,
            early_stopping_min_delta: 0.001,
        };
        let mut trainer = Trainer::new(config);

        // Create minimal test data
        let images = vec![Matrix::zeros(INPUT_NODES, 1), Matrix::zeros(INPUT_NODES, 1)];
        let mut label_data1 = vec![0.0; OUTPUT_NODES];
        let mut label_data2 = vec![0.0; OUTPUT_NODES];
        label_data1[0] = 1.0; // First image is class 0
        label_data2[1] = 1.0; // Second image is class 1
        let labels = vec![
            Matrix::new(OUTPUT_NODES, 1, label_data1),
            Matrix::new(OUTPUT_NODES, 1, label_data2),
        ];

        let data = MnistData::new(images, labels)?;
        trainer.train(&data)?;

        Ok(())
    }
}
