use neural_network::activations::SIGMOID;
use neural_network::network::Network;
use std::fs::File;
use std::io::Write;

fn main() {
    // Create training data
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0]];

    // Create and train network
    let mut network = Network::new(vec![2, 3, 1], SIGMOID, 0.5);
    network.train(inputs, targets, 100000);

    // Save the trained network to a file
    let network_json = serde_json::to_string(&network).expect("Failed to serialize network");
    let mut file = File::create("trained_network.json").expect("Failed to create file");
    file.write_all(network_json.as_bytes()).expect("Failed to write to file");
    println!("Network trained and saved to trained_network.json");
}
