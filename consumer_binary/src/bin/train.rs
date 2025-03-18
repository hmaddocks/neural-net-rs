use mnist::mnist::{MnistData, load_training_data};
use neural_network::activations::SIGMOID;
use neural_network::network::Network;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading MNIST training data...");
    let mnist_data: MnistData = load_training_data()?;

    println!("Creating network...");
    let mut network = Network::new(vec![784, 128, 10], SIGMOID, 0.1);

    println!("Training network...");
    let inputs: Vec<Vec<f64>> = mnist_data.images().iter().map(|m| m.data.clone()).collect();

    let targets: Vec<Vec<f64>> = mnist_data.labels().iter().map(|m| m.data.clone()).collect();

    network.train(inputs, targets, 10000);

    println!("Saving trained network...");
    let network_json = serde_json::to_string(&network)?;
    let mut file = File::create("trained_network.json")?;
    file.write_all(network_json.as_bytes())?;

    println!("Network trained and saved to trained_network.json");
    Ok(())
}
