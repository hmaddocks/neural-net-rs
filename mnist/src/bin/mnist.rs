use anyhow::{anyhow, Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use mnist::mnist::{get_actual_digit, load_test_data, load_training_data};
use mnist::{StandardizationParams, StandardizedMnistData};
use neural_network::network::Network;
use neural_network::network_config::NetworkConfig;
use serde_json;
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};

type ConfusionMatrix = [[usize; 10]; 10];

/// Metrics for a single digit classification
struct DigitMetrics {
    accuracy: f64,
    precision: f64,
    recall: f64,
    f1: f64,
}

/// Helper function to create a consistent progress bar style
fn create_progress_bar(total: u64) -> Result<ProgressBar> {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:80.cyan/blue}] {pos}/{len} ({percent}%)")
            .context("Failed to set progress bar template")?
            .progress_chars("#>-")
    );
    Ok(pb)
}

/// Calculates metrics for a digit from the confusion matrix
fn calculate_metrics(confusion_matrix: &ConfusionMatrix, digit: usize) -> DigitMetrics {
    let true_positives = confusion_matrix[digit][digit];
    let total_predictions: usize = confusion_matrix[digit].iter().sum();
    let total_actuals: usize = (0..10).map(|i| confusion_matrix[i][digit]).sum();

    let accuracy = (true_positives as f64) / (total_predictions.max(1) as f64);
    let precision = true_positives as f64 / total_actuals.max(1) as f64;
    let recall = true_positives as f64 / total_predictions.max(1) as f64;
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    DigitMetrics {
        accuracy,
        precision,
        recall,
        f1,
    }
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
fn print_metrics(confusion_matrix: ConfusionMatrix, total: usize) {
    let total_correct: usize = (0..10).map(|i| confusion_matrix[i][i]).sum();
    let overall_accuracy = (total_correct as f64) / (total as f64) * 100.0;

    println!("\nPer-digit Metrics:");
    println!("Digit  | Accuracy | Precision | Recall  | F1 Score");
    println!("-------|----------|-----------|---------|----------");

    for digit in 0..10 {
        let metrics = calculate_metrics(&confusion_matrix, digit);
        println!(
            "   {}   |  {:.1}%   |   {:.1}%   |  {:.1}%  |   {:.1}%",
            digit,
            metrics.accuracy * 100.0,
            metrics.precision * 100.0,
            metrics.recall * 100.0,
            metrics.f1 * 100.0
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
fn print_confusion_matrix(confusion_matrix: ConfusionMatrix) {
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

fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;

    let mut parts = Vec::new();
    if hours > 0 {
        parts.push(format!("{}h", hours));
    }
    if minutes > 0 {
        parts.push(format!("{}m", minutes));
    }
    if seconds > 0 || parts.is_empty() {
        parts.push(format!("{}s", seconds));
    }

    parts.join(" ")
}

fn test() -> Result<()> {
    println!("Loading test data...");
    let test_data = load_test_data().context("Failed to load test data")?;

    println!("Loading trained network...");
    let model_path =
        std::env::current_dir().map_err(|e| anyhow!("Failed to get current directory: {}", e))?;
    let model_path = model_path.join("models").join("trained_network.json");
    let model_path = model_path
        .to_str()
        .ok_or(anyhow!("Failed to convert model path to string"))?;
    let network =
        Network::load(model_path).map_err(|e| anyhow!("Failed to load trained network: {}", e))?;

    let standardized_params = if let (Some(mean), Some(std_dev)) = (network.mean, network.std_dev) {
        StandardizationParams::new(mean, std_dev)
    } else {
        StandardizationParams::build(&test_data.images())
            .map_err(|e| anyhow!("Failed to build standardization parameters: {}", e))?
    };

    let standardised_test_data = StandardizedMnistData::new(standardized_params)
        .standardize(&test_data.images())
        .context("Failed to standardize test data")?;

    println!("\nTesting network predictions...");
    let mut confusion_matrix = [[0usize; 10]; 10];
    let total = standardised_test_data.len();

    let progress_bar = create_progress_bar(total as u64)?;

    standardised_test_data
        .iter()
        .zip(test_data.labels().iter())
        .try_for_each(|(image, label)| -> Result<()> {
            let output = network.predict(image.clone());
            let predicted = get_actual_digit(&output)
                .map_err(|e| anyhow!("Failed to get predicted digit: {}", e))?;
            let actual = get_actual_digit(label)
                .map_err(|e| anyhow!("Failed to get actual digit: {}", e))?;
            confusion_matrix[actual][predicted] += 1;
            progress_bar.inc(1);
            Ok(())
        })?;

    progress_bar.finish_with_message("Testing complete");

    print_confusion_matrix(confusion_matrix);

    print_metrics(confusion_matrix, total);

    Ok(())
}

fn train() -> Result<()> {
    println!("Loading MNIST training data...");
    let mnist_data = load_training_data().context("Failed to load training data")?;

    let standardized_params = StandardizationParams::build(&mnist_data.images())
        .map_err(|e| anyhow!("Failed to build standardization parameters: {}", e))?;
    println!(
        "Standardizing MNIST data, mean: {:.4}, std_dev: {:.4}...",
        standardized_params.mean(),
        standardized_params.std_dev()
    );
    let standardized_data =
        StandardizedMnistData::new(standardized_params).standardize(&mnist_data.images())?;

    println!("Loading network configuration...");
    let config_path = std::env::current_dir()
        .context("Failed to get current directory")?
        .join("config.json");
    let network_config = NetworkConfig::load(&config_path)
        .map_err(|e| anyhow!("Failed to load network configuration: {}", e))?;

    println!("Creating network...");
    // Create network from configuration
    let mut network = Network::new(&network_config);

    println!("{network_config}");

    println!("Training network...");
    let start_time = Instant::now();
    network.train(&standardized_data, &mnist_data.labels());

    let total_duration = start_time.elapsed();
    println!(
        "Total training time: {} ({:.2?})",
        format_duration(total_duration),
        total_duration
    );

    println!("Saving trained network...");
    let network_json = serde_json::to_string(&network).context("Failed to serialize network")?;
    let model_path = std::env::current_dir()
        .context("Failed to get current directory")?
        .join("models")
        .join("trained_network.json");

    let mut file = File::create(&model_path).context("Failed to create model file")?;
    file.write_all(network_json.as_bytes())
        .context("Failed to write model file")?;
    println!("Network trained and saved to {}", model_path.display());
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Command::Train => train().context("Failed to train network")?,
        Command::Test => test().context("Failed to test network")?,
    }

    Ok(())
}

#[derive(clap::Parser)]
#[command(name = "mnist", about = "MNIST dataset processing", long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
#[command(about = "MNIST neural network operations")]
enum Command {
    /// Train a new neural network on the MNIST dataset
    Train,
    /// Test a trained neural network on the MNIST test set
    Test,
}
