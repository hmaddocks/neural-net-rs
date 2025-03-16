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
    pub fn new() -> Self {
        Self {
            accuracies: Vec::new(),
            losses: Vec::new(),
            best_accuracy: 0.0,
            best_epoch: 0,
        }
    }

    pub fn record_epoch(&mut self, epoch: u32, accuracy: f64, loss: f64) {
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
