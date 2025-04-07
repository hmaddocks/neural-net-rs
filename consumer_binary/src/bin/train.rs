use anyhow::{anyhow, Result};
use mnist::mnist::load_training_data;
use mnist::{StandardizationParams, StandardizedMnistData};
use neural_network::network::Network;
use neural_network::network_config::NetworkConfig;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

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

fn main() -> Result<()> {
    println!("Loading MNIST training data...");
    let mnist_data = match load_training_data() {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to load training data: {}", e);
            return Err(e.into());
        }
    };

    println!("Standardizing MNIST data...");
    let standardized_params = StandardizationParams::build(&mnist_data.images())
        .map_err(|e| anyhow!("Failed to build standardization parameters: {}", e))?;
    let standardized_data =
        StandardizedMnistData::new(standardized_params).standardize(&mnist_data.images())?;

    println!("Loading network configuration...");
    // Get the path to config.json in the consumer_binary root
    let config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config.json");
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
    let network_json = match serde_json::to_string(&network) {
        Ok(json) => json,
        Err(e) => {
            eprintln!("Failed to serialize network: {}", e);
            return Err(e.into());
        }
    };
    let model_path = match std::env::current_dir() {
        Ok(path) => path.join("models").join("trained_network.json"),
        Err(e) => {
            eprintln!("Failed to get current directory: {}", e);
            return Err(e.into());
        }
    };

    let mut file = match File::create(&model_path) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Failed to create model file: {}", e);
            return Err(e.into());
        }
    };
    match file.write_all(network_json.as_bytes()) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("Failed to write model file: {}", e);
            return Err(e.into());
        }
    };

    println!("Network trained and saved to {}", model_path.display());
    Ok(())
}
