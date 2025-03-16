//! Training module for the neural network implementation.
//!
//! This module provides the training infrastructure for the neural network, including:
//! - Configuration management via `TrainingConfig`
//! - Training loop implementation with early stopping
//! - Progress visualization using progress bars
//! - Model persistence through save/load functionality

use anyhow::Result;
use crate::mnist::{MnistData, MnistError, INPUT_NODES, OUTPUT_NODES};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use matrix::matrix::Matrix;
use neural_network::activations::SIGMOID;
use neural_network::network::Network;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::thread_rng;
use rayon::prelude::*;
use std::path::Path;

/// Configuration parameters for neural network training.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Size of each training batch
    pub batch_size: usize,
    /// Number of training epochs
    pub epochs: u32,
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Momentum coefficient for weight updates
    pub momentum: f64,
    /// Number of nodes in each hidden layer
    pub hidden_layers: Vec<usize>,
    /// Number of epochs to wait for improvement before early stopping
    pub early_stopping_patience: u32,
    /// Minimum improvement in accuracy required to reset patience counter
    pub early_stopping_min_delta: f64,
    /// Number of threads to use for parallel training
    pub num_threads: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            epochs: 30,
            learning_rate: 0.001,
            momentum: 0.9,
            hidden_layers: vec![128, 64],
            early_stopping_patience: 5,
            early_stopping_min_delta: 0.001,
            num_threads: num_cpus::get(),
        }
    }
}

/// Training history containing metrics recorded during training
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    /// Accuracy values for each epoch
    pub accuracies: Vec<f64>,
    /// Loss values for each epoch
    pub losses: Vec<f64>,
    /// Best accuracy achieved during training
    pub best_accuracy: f64,
    /// Epoch where best accuracy was achieved
    pub best_epoch: u32,
}

impl TrainingHistory {
    fn new() -> Self {
        Self {
            accuracies: Vec::new(),
            losses: Vec::new(),
            best_accuracy: 0.0,
            best_epoch: 0,
        }
    }

    fn record_epoch(&mut self, epoch: u32, accuracy: f64, loss: f64) {
        self.accuracies.push(accuracy);
        self.losses.push(loss);

        if accuracy > self.best_accuracy {
            self.best_accuracy = accuracy;
            self.best_epoch = epoch;
        }
    }

    /// Prints a summary of the training history
    pub fn print_summary(&self) {
        println!("\nTraining History Summary:");
        println!("------------------------");
        println!(
            "Best accuracy: {:.2}% (epoch {})",
            self.best_accuracy, self.best_epoch
        );
        println!(
            "Final accuracy: {:.2}%",
            self.accuracies.last().unwrap_or(&0.0)
        );
        println!("Final loss: {:.4}", self.losses.last().unwrap_or(&0.0));

        // Print accuracy progression at 25% intervals
        let len = self.accuracies.len();
        if len >= 4 {
            println!("\nAccuracy progression:");
            for i in 0..=3 {
                let idx = i * (len - 1) / 3;
                println!(
                    "Epoch {}: {:.2}% (loss: {:.4})",
                    idx + 1,
                    self.accuracies[idx],
                    self.losses[idx]
                );
            }
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
    history: TrainingHistory,
    thread_networks: Vec<Network>,
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

        let main_network = Network::new(layer_sizes.clone(), SIGMOID, config.learning_rate);

        // Create thread-specific networks
        let thread_networks = (0..config.num_threads)
            .map(|_| Network::new(layer_sizes.clone(), SIGMOID, config.learning_rate))
            .collect();

        let history = TrainingHistory::new();

        Self {
            network: main_network,
            config,
            history,
            thread_networks,
        }
    }

    /// Returns the training history containing accuracy and loss metrics
    pub fn history(&self) -> &TrainingHistory {
        &self.history
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
            "\nStarting training with batch size {} using {} threads",
            self.config.batch_size, self.config.num_threads
        );
        let mut indices: Vec<usize> = (0..data.len()).collect();
        let mut rng = rand::thread_rng();

        let mut best_accuracy = 0.0;
        let mut patience_counter = 0;

        // Clone progress bar for thread updates
        let thread_progress = batch_progress.clone();

        for epoch in 1..=self.config.epochs {
            indices.shuffle(&mut rng);
            let (mut correct, mut total) = (0, 0);
            let mut epoch_loss = 0.0;

            // Split data into chunks for parallel processing
            let chunk_size = indices.len() / self.config.num_threads;
            let chunks: Vec<_> = indices.chunks(chunk_size).collect();

            // Calculate total number of batches across all threads
            let total_batches = chunks.iter()
                .map(|chunk| (chunk.len() + self.config.batch_size - 1) / self.config.batch_size)
                .sum::<usize>();
            batch_progress.set_length(total_batches as u64);
            batch_progress.set_position(0);
            batch_progress.set_message(format!("in Epoch {}", epoch));

            // Process chunks in parallel
            let results: Vec<_> = chunks
                .into_par_iter()
                .enumerate()
                .map(|(thread_idx, chunk)| {
                    let mut thread_network = self.thread_networks[thread_idx].clone();
                    let mut thread_correct = 0;
                    let mut thread_total = 0;
                    let mut thread_loss = 0.0;

                    for batch_indices in chunk.chunks(self.config.batch_size) {
                        for &idx in batch_indices {
                            let output = thread_network
                                .feed_forward(&data.images()[idx])
                                .map_err(|e| MnistError::DataMismatch(e.to_string()))?;

                            let loss = calculate_loss(&output, &data.labels()[idx]);
                            thread_loss += loss;

                            thread_network.back_propogate(output.clone(), &data.labels()[idx]);

                            if self.get_prediction(&output)
                                == self.get_prediction(&data.labels()[idx])
                            {
                                thread_correct += 1;
                            }
                            thread_total += 1;
                        }
                        // Update progress from thread
                        thread_progress.inc(1);
                    }

                    Ok::<_, MnistError>((thread_network, thread_correct, thread_total, thread_loss))
                })
                .collect::<Result<Vec<_>, _>>()?;

            // Synchronize networks and accumulate metrics
            for (thread_idx, (thread_network, thread_correct, thread_total, thread_loss)) in
                results.into_iter().enumerate()
            {
                self.thread_networks[thread_idx] = thread_network;
                correct += thread_correct;
                total += thread_total;
                epoch_loss += thread_loss;
            }

            // Synchronize all networks by averaging weights and velocities
            for thread_network in &mut self.thread_networks {
                thread_network.sync_with(&self.network)?;
            }

            // Update main network from first thread network (they should all be the same now)
            self.network = self.thread_networks[0].clone();

            let accuracy = (correct as f64 / total as f64) * 100.0;
            let avg_loss = epoch_loss / total as f64;

            self.history.record_epoch(epoch, accuracy, avg_loss);

            epoch_progress.set_message(format!(
                "- Accuracy: {:.2}%, Loss: {:.4}",
                accuracy, avg_loss
            ));
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

        // Print training history summary
        self.history.print_summary();

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
        Ok(Self {
            network,
            config,
            history: TrainingHistory::new(),
            thread_networks: vec![],
        })
    }
}

/// Helper function to create a consistent progress bar style
fn create_progress_style(template: &str) -> ProgressStyle {
    ProgressStyle::with_template(template)
        .unwrap()
        .progress_chars("=> ")
}

/// Calculate mean squared error loss between predicted and target values
fn calculate_loss(predicted: &Matrix, target: &Matrix) -> f64 {
    let mut sum_squared_error = 0.0;
    for (p, t) in predicted.data().iter().zip(target.data().iter()) {
        sum_squared_error += (p - t).powi(2);
    }
    sum_squared_error / (2.0 * predicted.data().len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.batch_size, 100);
        assert_eq!(config.epochs, 30);
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.momentum, 0.9);
        assert_eq!(config.hidden_layers, vec![128, 64]);
        assert_eq!(config.early_stopping_patience, 5);
        assert_eq!(config.early_stopping_min_delta, 0.001);
        assert_eq!(config.num_threads, num_cpus::get());
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
            momentum: 0.9,
            hidden_layers: vec![4], // Smaller network for testing
            early_stopping_patience: 5,
            early_stopping_min_delta: 0.001,
            num_threads: 1,
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

    #[test]
    fn test_training_history() {
        // Create trainer with default config
        let config = TrainingConfig::default();
        let trainer = Trainer::new(config);

        // Initial history should be empty
        assert!(trainer.history().accuracies.is_empty());
        assert!(trainer.history().losses.is_empty());
        assert_eq!(trainer.history().best_accuracy, 0.0);
        assert_eq!(trainer.history().best_epoch, 0);
    }

    #[test]
    fn test_history_recording() {
        // Create a test history
        let mut history = TrainingHistory::new();

        // Record some test epochs
        history.record_epoch(1, 85.5, 0.25);
        history.record_epoch(2, 90.0, 0.15);
        history.record_epoch(3, 88.0, 0.18);

        // Check accuracy recording
        assert_eq!(history.accuracies, vec![85.5, 90.0, 88.0]);
        assert_eq!(history.losses, vec![0.25, 0.15, 0.18]);

        // Best accuracy should be 90.0 from epoch 2
        assert_eq!(history.best_accuracy, 90.0);
        assert_eq!(history.best_epoch, 2);
    }

    #[test]
    fn test_loss_calculation() {
        // Create test matrices
        let predicted = Matrix::new(2, 1, vec![0.8, 0.2]);
        let target = Matrix::new(2, 1, vec![1.0, 0.0]);

        // Calculate loss
        let loss = calculate_loss(&predicted, &target);

        // Expected MSE: ((0.8 - 1.0)^2 + (0.2 - 0.0)^2) / (2 * 2)
        let expected_loss = ((0.8_f64 - 1.0).powi(2) + (0.2_f64 - 0.0).powi(2)) / 4.0;
        assert!((loss - expected_loss).abs() < 1e-10);
    }
}
