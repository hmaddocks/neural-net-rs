use indicatif::{ProgressBar, ProgressStyle};
use neural_network::network::Network;
use rayon::prelude::*;
use std::error::Error;
use std::path::Path;
use std::sync::{Arc, Mutex};
use train_mnist::{
    mnist,
    training::{Trainer, TrainingConfig},
};

fn main() -> Result<(), Box<dyn Error>> {
    // Load network and create trainer
    println!("Loading trained network...");
    let network = Network::load(Path::new("trained_network.json"))?;
    let trainer = Trainer::new(TrainingConfig::default());

    // Load test data
    println!("Loading MNIST test data...");
    let test_data = mnist::load_mnist_test_data()?;
    println!(
        "Successfully loaded {} test examples",
        test_data.images().len()
    );

    // Initialize progress bar with custom style
    let progress_bar = Arc::new(Mutex::new(
        ProgressBar::new(test_data.images().len() as u64),
    ));
    progress_bar.lock().unwrap().set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
            )
            .unwrap()
            .progress_chars("=>-"),
    );
    progress_bar.lock().unwrap().set_message("Testing model...");

    // Initialize per-digit accuracy tracking
    let correct_per_digit = Arc::new(Mutex::new(vec![0u32; 10]));
    let total_per_digit = Arc::new(Mutex::new(vec![0u32; 10]));
    let correct_predictions = Arc::new(Mutex::new(0u32));

    // Create test data chunks for parallel processing
    let chunk_size = test_data.images().len() / rayon::current_num_threads();
    let image_chunks: Vec<_> = test_data.images().chunks(chunk_size).collect();
    let label_chunks: Vec<_> = test_data.labels().chunks(chunk_size).collect();
    let chunks = image_chunks.into_iter().zip(label_chunks.into_iter());

    // Process chunks in parallel
    chunks.par_bridge().for_each(|(images, labels)| {
        let mut local_network = network.clone();
        let pb = Arc::clone(&progress_bar);
        let correct = Arc::clone(&correct_predictions);
        let correct_digits = Arc::clone(&correct_per_digit);
        let total_digits = Arc::clone(&total_per_digit);

        // Process each image in the chunk
        for (image, label) in images.iter().zip(labels.iter()) {
            let output = local_network.feed_forward(image).unwrap();
            let predicted = trainer.get_prediction(&output);
            let actual = mnist::get_actual_digit(label);

            // Update statistics
            {
                let mut total = total_digits.lock().unwrap();
                total[actual] += 1;

                if predicted == actual {
                    let mut correct_count = correct.lock().unwrap();
                    *correct_count += 1;

                    let mut correct_by_digit = correct_digits.lock().unwrap();
                    correct_by_digit[actual] += 1;
                }
            }
            pb.lock().unwrap().inc(1);
        }
    });

    // Calculate and display results
    let pb = progress_bar.lock().unwrap();
    pb.finish_with_message("Testing completed!");

    let total_correct = *correct_predictions.lock().unwrap();
    let total_accuracy = (total_correct as f64) / (test_data.images().len() as f64) * 100.0;

    println!("\nTest Results:");
    println!("Total test examples: {}", test_data.images().len());
    println!("Correct predictions: {}", total_correct);
    println!("Overall accuracy: {:.2}%", total_accuracy);

    println!("\nPer-digit accuracy:");
    println!("Digit | Correct | Total | Accuracy");
    println!("------|---------|--------|----------");

    let correct_by_digit = correct_per_digit.lock().unwrap();
    let total_by_digit = total_per_digit.lock().unwrap();

    for digit in 0..10 {
        let accuracy = (correct_by_digit[digit] as f64) / (total_by_digit[digit] as f64) * 100.0;
        println!(
            "   {digit}  |   {correct:^5} |  {total:^4}  | {accuracy:>6.2}%",
            digit = digit,
            correct = correct_by_digit[digit],
            total = total_by_digit[digit],
            accuracy = accuracy
        );
    }

    Ok(())
}
