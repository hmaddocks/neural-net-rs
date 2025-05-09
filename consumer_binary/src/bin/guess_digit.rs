use anyhow::{anyhow, Result};
use image::DynamicImage;
use matrix::matrix::Matrix;
use mnist::mnist::get_actual_digit;
use mnist::{StandardizationParams, StandardizedMnistData};
use neural_network::network::Network;

fn process_image(img: DynamicImage) -> Result<Matrix, anyhow::Error> {
    // Resize to 28x28 if needed
    let img = img.resize_exact(28, 28, image::imageops::FilterType::Lanczos3);

    // Convert to grayscale
    let img = img.to_luma8();

    // Convert to normalized f64 values (0.0-1.0)
    let data: Vec<f64> = img.pixels().map(|p| f64::from(p.0[0]) / 255.0).collect();

    // Create matrix in the format expected by the network (784x1)
    Ok(Matrix::new(784, 1, data))
}

fn main() -> Result<()> {
    // Get command line argument
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <path-to-image>", args[0]);
        std::process::exit(1);
    }

    let image_path = &args[1];

    // Load the image
    println!("Loading image from {}...", image_path);
    let img = image::open(image_path)?;

    // Process the image
    println!("Processing image...");
    let input_matrix = process_image(img)?;

    // Load the trained network
    println!("Loading trained network...");
    let model_path = std::env::current_dir()?
        .join("models")
        .join("trained_network.json");
    let network: Network = Network::load(
        model_path
            .to_str()
            .ok_or(anyhow!("Failed to get model path"))?,
    )?;

    let mean = network
        .mean
        .ok_or(anyhow!("Network missing mean parameter"))?;
    let std_dev = network
        .std_dev
        .ok_or(anyhow!("Network missing standard deviation parameter"))?;
    let standardized_params = StandardizationParams::new(mean, std_dev);
    let standardized_input =
        StandardizedMnistData::new(standardized_params).standardize_matrix(&input_matrix)?;

    // Make prediction
    println!("Making prediction...");
    let output = network.predict(standardized_input);
    let predicted_digit = get_actual_digit(&output)?;

    println!("\nPredicted digit: {}", predicted_digit);
    Ok(())
}
