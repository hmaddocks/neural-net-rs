use anyhow::{anyhow, Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use mnist::{get_actual_digit, load_test_data, load_training_data};
use mnist::{StandardizationParams, StandardizedMnistData};
use ndarray::Axis;
use neural_network::{Matrix, Network, NetworkConfig, TrainingHistory};
use plotters::prelude::*;
use serde_json;
use std::{
    fs::File,
    io::Write,
    time::{Duration, Instant},
};

/// A confusion matrix for tracking model predictions vs actual values
#[derive(Debug, Default)]
struct ConfusionMatrix {
    matrix: [[usize; 10]; 10],
}

impl ConfusionMatrix {
    /// Creates a new empty confusion matrix
    pub fn new() -> Self {
        Self {
            matrix: [[0; 10]; 10],
        }
    }

    /// Records a prediction in the confusion matrix
    pub fn record(&mut self, actual: usize, predicted: usize) {
        self.matrix[actual][predicted] += 1;
    }

    /// Gets the value at a specific position
    pub fn get(&self, actual: usize, predicted: usize) -> usize {
        self.matrix[actual][predicted]
    }
}

impl std::fmt::Display for ConfusionMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // write confusion matrix
        writeln!(f, "\nConfusion Matrix:")?;
        writeln!(f, "      Predicted →")?;
        writeln!(
            f,
            "Actual     0    1    2    3    4    5    6    7    8    9"
        )?;
        writeln!(
            f,
            "  ↓   +--------------------------------------------------"
        )?;
        for i in 0..10 {
            write!(f, "  {}   |", i)?;
            for j in 0..10 {
                write!(f, " {:4}", self.get(i, j))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

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
    let true_positives = confusion_matrix.get(digit, digit);
    let total_predictions: usize = (0..10).map(|i| confusion_matrix.get(digit, i)).sum();
    let total_actuals: usize = (0..10).map(|i| confusion_matrix.get(i, digit)).sum();

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
fn print_metrics(confusion_matrix: &ConfusionMatrix, total: usize) {
    let total_correct: usize = (0..10).map(|i| confusion_matrix.get(i, i)).sum();
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

    let standardised_test_data = StandardizedMnistData::new(&standardized_params)
        .standardize(&test_data.images())
        .context("Failed to standardize test data")?;

    // Combine test data into matrices for batch processing
    let test_matrix =
        Matrix::concatenate(&standardised_test_data.iter().collect::<Vec<_>>(), Axis(1));

    // Get predictions for all test data at once
    let output_matrix = network.predict(test_matrix);

    // Combine labels into matrix
    let label_matrix = Matrix::concatenate(&test_data.labels().iter().collect::<Vec<_>>(), Axis(1));

    println!("\nTesting network predictions...");
    let total = standardised_test_data.len();
    let progress_bar = create_progress_bar(total as u64)?;
    let mut confusion_matrix = ConfusionMatrix::new();

    // Process predictions and update confusion matrix
    (0..output_matrix.cols()).try_for_each(|i| -> Result<()> {
        let output = output_matrix.col(i);
        let label = label_matrix.col(i);
        let predicted = get_actual_digit(&output)
            .map_err(|e| anyhow!("Failed to get predicted digit: {}", e))?;
        let actual =
            get_actual_digit(&label).map_err(|e| anyhow!("Failed to get actual digit: {}", e))?;
        confusion_matrix.record(actual, predicted);
        progress_bar.inc(1);
        Ok(())
    })?;

    progress_bar.finish_with_message("Testing complete");

    println!("{}", confusion_matrix);

    print_metrics(&confusion_matrix, total);

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
        StandardizedMnistData::new(&standardized_params).standardize(&mnist_data.images())?;

    println!("Loading network configuration...");
    let config_path = std::env::current_dir()
        .context("Failed to get current directory")?
        .join("config.json");
    let network_config = NetworkConfig::load(&config_path)
        .map_err(|e| anyhow!("Failed to load network configuration: {}", e))?;

    println!("Creating network...");
    // Create network from configuration
    let mut network = Network::new(&network_config);

    let mean = standardized_params.mean();
    let std_dev = standardized_params.std_dev();
    network.set_standardization_parameters(Some(mean), Some(std_dev));

    println!("{network_config}");

    println!("Training network...");
    let start_time = Instant::now();
    let training_history = network.train(&standardized_data, &mnist_data.labels());

    let total_duration = start_time.elapsed();
    println!(
        "Total training time: {} ({:.2?})",
        format_duration(total_duration),
        total_duration
    );

    // Save training history to file
    println!("Saving training history...");
    save_training_history(training_history)?;

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

/// Saves the training history to a JSON file
fn save_training_history(history: &TrainingHistory) -> Result<()> {
    let history_json =
        serde_json::to_string_pretty(history).context("Failed to serialize training history")?;

    let history_path = std::env::current_dir()
        .context("Failed to get current directory")?
        .join("models")
        .join("training_history.json");

    // Create models directory if it doesn't exist
    if let Some(parent) = history_path.parent() {
        std::fs::create_dir_all(parent).context("Failed to create models directory")?;
    }

    let mut file = File::create(&history_path).context("Failed to create training history file")?;
    file.write_all(history_json.as_bytes())
        .context("Failed to write training history file")?;

    println!("Training history saved to {}", history_path.display());
    Ok(())
}

/// Creates an SVG graph of the training history accuracies and losses
fn create_training_history_graph() -> Result<()> {
    println!("Loading training history...");

    // Load training history from JSON file
    let history_path = std::env::current_dir()
        .map_err(|e| anyhow!("Failed to get current directory: {}", e))?
        .join("models")
        .join("training_history.json");

    if !history_path.exists() {
        return Err(anyhow!(
            "Training history file not found at {}",
            history_path.display()
        ));
    }

    let history_json =
        std::fs::read_to_string(&history_path).context("Failed to read training history file")?;

    let history: TrainingHistory =
        serde_json::from_str(&history_json).context("Failed to parse training history JSON")?;

    // Create output directory if it doesn't exist
    let output_dir = std::env::current_dir()
        .map_err(|e| anyhow!("Failed to get current directory: {}", e))?
        .join("graphs");

    if !output_dir.exists() {
        std::fs::create_dir(&output_dir).context("Failed to create graphs directory")?;
    }

    // Create SVG file for the graph
    let output_path = output_dir.join("training_history.svg");

    // Set up the plot
    let root = SVGBackend::new(&output_path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE).context("Failed to fill drawing area")?;

    // Determine the ranges for the chart
    let epoch_range = 0.0..(history.accuracies.len() as f64);
    let loss_max = history.losses.iter().fold(0.0, |a: f64, &b| a.max(b));

    // Define formatter functions
    let accuracy_formatter = |y: &f64| -> String { format!("{:.1}%", *y) };
    let x_formatter = |x: &f64| -> String { format!("{}", (*x).round()) };

    // Create a single chart for both metrics
    let mut chart = ChartBuilder::on(&root)
        .caption("Training Metrics", ("sans-serif", 30).into_font())
        .margin(10)
        .margin_bottom(60) // Add extra space at the bottom for the legend
        .x_label_area_size(40)
        .y_label_area_size(60)
        .right_y_label_area_size(60) // Add space for second scale on right
        .build_cartesian_2d(epoch_range.clone(), 0f64..100f64)
        .context("Failed to build chart")?
        .set_secondary_coord(epoch_range.clone(), 0f64..loss_max); // Add secondary coordinate system for loss

    // Configure the mesh for accuracy (left y-axis)
    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .disable_mesh()
        .y_label_formatter(&accuracy_formatter)
        .x_label_formatter(&x_formatter)
        .x_desc("Epoch")
        .y_desc("Accuracy (%)")
        .draw()
        .context("Failed to draw mesh")?;

    // Configure the secondary mesh for loss (right y-axis)
    chart
        .configure_secondary_axes()
        .y_labels(10)
        .y_label_formatter(&|y: &f64| format!("{:.4}", y))
        .y_desc("Loss")
        .draw()
        .context("Failed to draw secondary axes")?;

    // Create the accuracy line series
    let accuracy_points: Vec<(f64, f64)> = history
        .accuracies
        .iter()
        .enumerate()
        .map(|(i, &acc)| (i as f64, acc))
        .collect();

    // Create the loss line series - no need to scale since we have dual axes
    let loss_points: Vec<(f64, f64)> = history
        .losses
        .iter()
        .enumerate()
        .map(|(i, &loss)| (i as f64, loss))
        .collect();

    // Draw the accuracy line
    chart
        .draw_series(LineSeries::new(accuracy_points, &BLUE))
        .context("Failed to draw accuracy line")?
        .label("Accuracy")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Draw the loss line using secondary coordinate system
    chart
        .draw_secondary_series(LineSeries::new(loss_points, &RED))
        .context("Failed to draw loss line")?
        .label("Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Draw the legend
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .margin(10)
        .draw()
        .context("Failed to draw legend")?;

    println!("Training history graph saved to {}", output_path.display());
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Command::Train => train().context("Failed to train network")?,
        Command::Test => test().context("Failed to test network")?,
        Command::Graph => {
            create_training_history_graph().context("Failed to create training history graph")?
        }
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
    /// Create an SVG graph of training history accuracies and losses
    Graph,
}
