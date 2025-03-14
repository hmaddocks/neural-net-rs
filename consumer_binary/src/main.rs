mod mnist;
mod training;

use mnist::MnistError;
use training::{Trainer, TrainingConfig};

fn main() -> Result<(), MnistError> {
    println!("Loading MNIST dataset...");
    let data = mnist::load_mnist_data()?;
    println!("\nSuccessfully loaded {} training examples", data.len());

    let config = TrainingConfig::default();
    let mut trainer = Trainer::new(config);

    println!("\nInitializing neural network...");
    trainer.train(&data)?;
    println!("\nTraining completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnist::{INPUT_NODES, OUTPUT_NODES};
    use matrix::matrix::Matrix;

    #[test]
    fn test_end_to_end() -> Result<(), MnistError> {
        // Create a minimal dataset
        let images = vec![Matrix::zeros(INPUT_NODES, 1)];
        let mut labels = vec![0.0; OUTPUT_NODES];
        labels[0] = 1.0;
        let labels = vec![Matrix::new(OUTPUT_NODES, 1, labels)];
        
        let data = mnist::MnistData::new(images, labels)?;
        
        // Train for a single epoch
        let config = TrainingConfig {
            batch_size: 1,
            epochs: 1,
            learning_rate: 0.1,
            hidden_layers: vec![4],
        };
        
        let mut trainer = Trainer::new(config);
        trainer.train(&data)?;
        
        Ok(())
    }
}
