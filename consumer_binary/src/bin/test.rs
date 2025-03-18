use mnist::mnist::{load_test_data, get_actual_digit};
use neural_network::network::Network;
use std::fs::File;
use std::io::Read;
use matrix::matrix::Matrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading test data...");
    let test_data = load_test_data()?;
    
    println!("Loading trained network...");
    let mut file = File::open("trained_network.json")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let mut network: Network = serde_json::from_str(&contents)?;

    println!("\nTesting network predictions...");
    let mut correct = 0;
    let total = test_data.len();
    
    for (i, (image, label)) in test_data.images().iter().zip(test_data.labels().iter()).enumerate() {
        let output = network.feed_forward(Matrix::new(784, 1, image.data.clone()));
        let predicted = get_actual_digit(&output);
        let actual = get_actual_digit(label);
        
        if predicted == actual {
            correct += 1;
        }
        
        if i % 100 == 0 {
            println!("Progress: {}/{} images tested", i, total);
        }
    }
    
    let accuracy = (correct as f64 / total as f64) * 100.0;
    println!("\nResults:");
    println!("Total test images: {}", total);
    println!("Correct predictions: {}", correct);
    println!("Accuracy: {:.2}%", accuracy);
    
    Ok(())
}
