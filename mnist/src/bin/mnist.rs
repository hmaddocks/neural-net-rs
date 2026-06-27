use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand, ValueEnum};
use mnist::{
    ConfusionMatrix, MnistArtifacts, TrainSettings, load_test_data, load_training_data, test_mnist,
    train_mnist,
};
use neural_network::BackpropEngine;
use std::time::{Duration, Instant};

/// Metrics for a single digit classification
struct DigitMetrics {
    accuracy: f64,
    precision: f64,
    recall: f64,
    f1: f64,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliBackpropEngine {
    Manual,
    Autograd,
}

impl From<CliBackpropEngine> for BackpropEngine {
    fn from(engine: CliBackpropEngine) -> Self {
        match engine {
            CliBackpropEngine::Manual => BackpropEngine::Manual,
            CliBackpropEngine::Autograd => BackpropEngine::Autograd,
        }
    }
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

fn print_metrics(confusion_matrix: &ConfusionMatrix, total: usize) {
    let total_correct: usize = (0..10).map(|i| confusion_matrix.get(i, i)).sum();
    let overall_accuracy = (total_correct as f64) / (total as f64) * 100.0;

    println!("\nPer-digit Metrics:");
    println!("Digit  | Accuracy | Precision | Recall  | F1 Score");
    println!("-------|----------|-----------|---------|----------");

    for digit in 0..10 {
        let metrics = calculate_metrics(confusion_matrix, digit);
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

fn workspace_root() -> Result<std::path::PathBuf> {
    std::env::current_dir().context("Failed to get current directory")
}

fn artifacts() -> Result<MnistArtifacts> {
    Ok(MnistArtifacts::from_workspace_root(workspace_root()?))
}

fn run_test() -> Result<()> {
    println!("Loading test data...");
    let test_data = load_test_data().context("Failed to load test data")?;

    let artifacts = artifacts()?;
    println!(
        "Loading trained network from {}...",
        artifacts.model_path.display()
    );

    println!("\nTesting network predictions...");
    let confusion_matrix = test_mnist(&artifacts.model_path, &test_data)
        .map_err(|error| anyhow!("Failed to evaluate confusion matrix: {error}"))?;

    println!("{confusion_matrix}");
    print_metrics(&confusion_matrix, test_data.len());

    Ok(())
}

fn run_train(backprop_engine: BackpropEngine) -> Result<()> {
    println!("Loading MNIST training data...");
    let training_data = load_training_data().context("Failed to load training data")?;

    let artifacts = artifacts()?;
    println!("Training with {:?} backprop engine...", backprop_engine);
    println!(
        "Loading network configuration from {}...",
        artifacts.config_path.display()
    );

    let start_time = Instant::now();
    train_mnist(
        &training_data,
        &artifacts,
        TrainSettings {
            backprop_engine,
            ..TrainSettings::default()
        },
    )
    .map_err(|error| anyhow!("Failed to train network: {error}"))?;

    let total_duration = start_time.elapsed();
    println!(
        "Total training time: {} ({:.2?})",
        format_duration(total_duration),
        total_duration
    );
    println!(
        "Network trained and saved to {}",
        artifacts.model_path.display()
    );
    println!(
        "Training history saved to {}",
        artifacts.history_path.display()
    );

    Ok(())
}

fn run_graph() -> Result<()> {
    let artifacts = artifacts()?;
    println!(
        "Loading training history from {}...",
        artifacts.history_path.display()
    );

    mnist::render_training_graph(&artifacts.history_path, &artifacts.graph_path)
        .map_err(|error| anyhow!("Failed to create training history graph: {error}"))?;

    println!(
        "Training history graph saved to {}",
        artifacts.graph_path.display()
    );
    Ok(())
}

#[derive(Parser)]
#[command(name = "mnist", about = "MNIST dataset processing", long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
#[command(about = "MNIST neural network operations")]
enum Command {
    /// Train a new neural network on the MNIST dataset
    Train {
        #[arg(long, value_enum, default_value_t = CliBackpropEngine::Manual)]
        backprop_engine: CliBackpropEngine,
    },
    /// Test a trained neural network on the MNIST test set
    Test,
    /// Create an SVG graph of training history accuracies and losses
    Graph,
}

fn main() -> Result<()> {
    let args = Args::parse();

    match args.command {
        Command::Train { backprop_engine } => {
            run_train(backprop_engine.into()).context("Failed to train network")?
        }
        Command::Test => run_test().context("Failed to test network")?,
        Command::Graph => run_graph().context("Failed to create training history graph")?,
    }

    Ok(())
}
