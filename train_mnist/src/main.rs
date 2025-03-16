mod mnist;
mod training;

use mnist::MnistError;
use std::path::Path;
use training::{Trainer, TrainingConfig};

fn main() -> Result<(), MnistError> {
    println!("Loading MNIST dataset...");
    let data = mnist::load_mnist_data()?;
    println!("\nSuccessfully loaded {} training examples", data.len());

    let config = TrainingConfig {
        batch_size: 100,
        epochs: 30,
        learning_rate: 0.001,
        momentum: 0.9,
        hidden_layers: vec![128, 64],
        early_stopping_patience: 5,
        early_stopping_min_delta: 0.001,
        num_threads: num_cpus::get(),
    };
    let mut trainer = Trainer::new(config);

    println!("\nInitializing neural network...");
    trainer.train(&data)?;
    println!("\nTraining completed successfully!");

    // Save the trained network
    let save_path = Path::new("trained_network.json");
    trainer.save_network(save_path)?;
    println!("\nNetwork saved to {}", save_path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use matrix::matrix::Matrix;
    use mnist::{INPUT_NODES, OUTPUT_NODES};
    use tempfile::tempdir;

    #[test]
    fn test_end_to_end() -> Result<(), MnistError> {
        // Create test data
        let mut images = Vec::new();
        let mut labels = Vec::new();
        for i in 0..10 {
            let mut image = vec![0.0; INPUT_NODES];
            image[i] = 1.0;
            images.push(Matrix::from(image));

            let mut label = vec![0.0; OUTPUT_NODES];
            label[i] = 1.0;
            labels.push(Matrix::from(label));
        }

        // Create a temporary directory for test data
        let temp_dir = tempdir()?;
        let temp_path = temp_dir.path();

        // Train for a single epoch
        let config = TrainingConfig {
            batch_size: 100,
            epochs: 1,
            learning_rate: 0.001,
            momentum: 0.9,
            hidden_layers: vec![32],
            early_stopping_patience: 5,
            early_stopping_min_delta: 0.001,
            num_threads: 1,
        };

        // Save test network
        let mut trainer = Trainer::new(config.clone());
        let save_path = temp_path.join("test_network.json");
        trainer.save_network(&save_path)?;

        // Load test network and verify it works
        let _loaded_trainer = Trainer::load_network(&save_path, config)?;

        Ok(())
    }
}
