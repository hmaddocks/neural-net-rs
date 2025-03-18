use mnist::mnist::{MnistData, load_training_data};
use neural_network::activations::SIGMOID;
use neural_network::network::Network;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading MNIST training data...");
    let mnist_data: MnistData = match load_training_data() {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to load training data: {}", e);
            return Err(e.into());
        }
    };

    println!("Creating network...");
    // Initialize network with lower learning rate and momentum
    let mut network = Network::new(vec![784, 200, 10], SIGMOID, 0.01, Some(0.5));

    println!("Training network...");
    let inputs: Vec<Vec<f64>> = mnist_data.images().iter().map(|m| m.data.clone()).collect();

    let targets: Vec<Vec<f64>> = mnist_data.labels().iter().map(|m| m.data.clone()).collect();

    // network.train(inputs, targets, 10000);
    network.train(inputs, targets, 30);

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
