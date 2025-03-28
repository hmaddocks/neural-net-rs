use indicatif::{ProgressBar, ProgressStyle};
use mnist::mnist::{get_actual_digit, load_test_data};
use mnist::standardized_mnist::{StandardizationParams, StandardizedMnistData};
use neural_network::network::Network;

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
/// The matrix is a 10x10 grid where:
/// - Rows represent the actual digit (0-9)
/// - Columns represent what the model predicted (0-9)
/// - Each cell [i][j] contains the count of how many times:
///   * The actual digit was i
///   * The model predicted j
///
/// Example:
/// ```text
/// Actual 5: |  12   1   0  39   4  808   8   3  11   6
///           |   ↑   ↑   ↑   ↑   ↑   ↑    ↑   ↑   ↑   ↑
/// Predicted |   0   1   2   3   4   5    6   7   8   9
/// ```
/// This row shows that when the actual digit was 5:
/// - 808 times it was correctly predicted as 5
/// - 39 times it was incorrectly predicted as 3
/// - 12 times it was incorrectly predicted as 0
/// - etc.
///
/// The diagonal (where row index equals column index) shows correct predictions.
/// Everything off the diagonal represents mistakes.
fn print_confusion_matrix(confusion_matrix: [[usize; 10]; 10]) {
    // Print confusion matrix
    println!("\nConfusion Matrix:");
    println!("      Predicted →");
    println!("Actual     0    1    2    3    4    5    6    7    8    9");
    println!("  ↓   +--------------------------------------------------");
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
    let test_data = match load_test_data() {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to load test data: {}", e);
            return Err(e.into());
        }
    };

    println!("Standardizing test data...");
    let standardized_params = StandardizationParams::build(&test_data.images());
    let standardised_test_data =
        StandardizedMnistData::new(standardized_params).standardize(&test_data.images());

    println!("Loading trained network...");
    let model_path = match std::env::current_dir() {
        Ok(path) => path.join("models").join("trained_network.json"),
        Err(e) => {
            eprintln!("Failed to get current directory: {}", e);
            return Err(e.into());
        }
    };

    let network = Network::load(model_path.to_str().unwrap())?;

    println!("\nTesting network predictions...");
    let mut confusion_matrix = [[0usize; 10]; 10];
    let total = standardised_test_data.len();

    let progress_bar = ProgressBar::new(total as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:80.cyan/blue}] {pos}/{len} ({percent}%)")
            .unwrap()
            .progress_chars("#>-")
    );

    standardised_test_data
        .iter()
        .zip(test_data.labels().iter())
        .for_each(|(image, label)| {
            let output = network.predict(image.clone());
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
