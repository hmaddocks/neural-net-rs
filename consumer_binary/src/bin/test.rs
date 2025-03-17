use neural_network::matrix::Matrix;
use neural_network::network::Network;
use std::fs::File;
use std::io::Read;

fn main() {
    // Load the trained network from file
    let mut file = File::open("trained_network.json").expect("Failed to open network file");
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("Failed to read file");
    let mut network: Network =
        serde_json::from_str(&contents).expect("Failed to deserialize network");

    // Test the network with all possible inputs
    let test_inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    println!("Testing network predictions:");
    for input in test_inputs {
        // Clone input since we need it both for network input and printing
        let result = network.feed_forward(Matrix::from(input.clone()));
        println!("Input: {:?}, Output: {:?}", input, result);
    }
}
