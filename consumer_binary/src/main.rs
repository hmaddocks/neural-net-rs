use neural_network::activations::SIGMOID;
use neural_network::matrix::Matrix;
use neural_network::network::Network;
use std::env;
use anyhow::Result;

fn main() -> Result<()> {
    env::set_var("RUST_BACKTRACE", "1");
    
    // Training data for XOR function
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]]; // XOR truth table

    // Create and train network
    let mut network = Network::new(vec![2, 4, 1], SIGMOID, 0.1);
    network.train(inputs, targets, 100000)?;

    // Test the network
    let test_inputs = [
        Matrix::from(vec![0.0, 0.0]),
        Matrix::from(vec![0.0, 1.0]),
        Matrix::from(vec![1.0, 0.0]),
        Matrix::from(vec![1.0, 1.0]),
    ];

    println!("\nTesting XOR function:");
    for input in &test_inputs {
        let output = network.feed_forward(input)?;
        println!("Input: {:?} -> Output: {:.3}", input.data(), output.data()[0]);
    }

    Ok(())
}
