use indicatif::{ProgressBar, ProgressStyle};
use matrix::matrix::Matrix;
use mnist::mnist::{get_actual_digit, load_test_data};
use neural_network::network::Network;
use std::fs::File;
use std::io::Read;

/// Calculates metrics for a digit from the confusion matrix
fn calculate_metrics(confusion_matrix: &[[usize; 10]; 10], digit: usize) -> (f64, f64, f64, f64) {
    let true_positives = confusion_matrix[digit][digit];
    let total_predictions: usize = confusion_matrix[digit].iter().sum();
    let total_actuals: usize = (0..10).map(|i| confusion_matrix[i][digit]).sum();

    let accuracy = (true_positives as f64) / (total_predictions as f64);
    let precision = if total_actuals == 0 {
        0.0
    } else {
        true_positives as f64 / total_actuals as f64
    };
    let recall = if total_predictions == 0 {
        0.0
    } else {
        true_positives as f64 / total_predictions as f64
    };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    (accuracy, precision, recall, f1)
}

/// Prints detailed metrics for the neural network's performance, including per-digit statistics
/// and overall accuracy.
///
/// # Metrics Explained
/// * Accuracy - Percentage of correct predictions for a specific digit
/// * Precision - Of all cases predicted as digit X, what percentage were actually X
/// * Recall - Of all actual cases of digit X, what percentage were correctly identified
/// * F1 Score - Harmonic mean of precision and recall (2 * precision * recall)/(precision + recall)
///
/// The overall accuracy represents the total correct predictions across all digits.
fn print_metrics(confusion_matrix: [[usize; 10]; 10], total: usize) {
    let total_correct: usize = (0..10).map(|i| confusion_matrix[i][i]).sum();
    let overall_accuracy = (total_correct as f64) / (total as f64) * 100.0;

    println!("\nPer-digit Metrics:");
    println!("Digit  | Accuracy | Precision | Recall  | F1 Score");
    println!("-------|----------|-----------|---------|----------");

    for digit in 0..10 {
        let (accuracy, precision, recall, f1) = calculate_metrics(&confusion_matrix, digit);
        println!(
            "   {}   |  {:.1}%   |   {:.1}%   |  {:.1}%  |   {:.1}%",
            digit,
            accuracy * 100.0,
            precision * 100.0,
            recall * 100.0,
            f1 * 100.0
        );
    }

    println!("\nOverall Accuracy: {:.2}%", overall_accuracy);
}

/// Prints a confusion matrix showing the neural network's prediction performance.
///
/// # Matrix Interpretation
/// * Rows (Actual) - The true digit labels from the dataset
/// * Columns (Predicted) - What the network predicted
/// * Each cell [i][j] - Number of times digit i was predicted as digit j
/// * Diagonal elements [i][i] - Correct predictions for digit i
/// * Off-diagonal elements - Various types of mistakes:
///   - Row examination shows how a particular digit was misclassified
///   - Column examination shows what digits were mistaken for a particular prediction
///
/// Example: If cell [7][1] = 5, this means the digit 7 was incorrectly classified as 1 five times
fn print_confusion_matrix(confusion_matrix: [[usize; 10]; 10]) {
    // Print confusion matrix
    println!("\nConfusion Matrix:");
    println!("           Predicted");
    println!("Actual     0    1    2    3    4    5    6    7    8    9");
    println!("      +--------------------------------------------------");
    for (i, row) in confusion_matrix.iter().enumerate() {
        print!("  {}   |", i);
        for &count in row.iter() {
            print!(" {:4}", count);
        }
        println!();
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading test data...");
    let test_data = load_test_data()?;

    println!("Loading trained network...");
    let mut file = File::open("trained_network.json")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let mut network: Network = serde_json::from_str(&contents)?;

    println!("\nTesting network predictions...");
    let mut confusion_matrix = [[0usize; 10]; 10];
    let total = test_data.len();

    let progress_bar = ProgressBar::new(total as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:80.cyan/blue}] {pos}/{len} ({percent}%)")
            .unwrap()
            .progress_chars("#>-")
    );

    test_data
        .images()
        .iter()
        .zip(test_data.labels().iter())
        .for_each(|(image, label)| {
            let output = network.feed_forward(Matrix::new(784, 1, image.data.clone()));
            let predicted = get_actual_digit(&output);
            let actual = get_actual_digit(label);

            confusion_matrix[actual][predicted] += 1;
            progress_bar.inc(1);
        });

    progress_bar.finish_with_message("Testing complete");

    print_confusion_matrix(confusion_matrix);

    print_metrics(confusion_matrix, total);

    Ok(())
}
