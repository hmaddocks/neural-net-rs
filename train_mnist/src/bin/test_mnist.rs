use indicatif::{ProgressBar, ProgressStyle};
use neural_network::network::Network;
use rayon::prelude::*;
use std::error::Error;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use train_mnist::{
    mnist,
    training::{Trainer, TrainingConfig},
};

/// A confusion matrix for MNIST digit classification.
///
/// The matrix is a 10x10 grid where:
/// - Rows represent the actual digit (0-9)
/// - Columns represent what the model predicted (0-9)
/// - Each cell [i][j] contains the count of how many times:
///   * The actual digit was i
///   * The model predicted j
///
/// Example:
/// ```text
/// Actual 5: |  12   1   0  39   4  808   8   3  11   6
///           |   ↑   ↑   ↑   ↑   ↑   ↑    ↑   ↑   ↑   ↑
/// Predicted |   0   1   2   3   4   5    6   7   8   9
/// ```
/// This row shows that when the actual digit was 5:
/// - 808 times it was correctly predicted as 5
/// - 39 times it was incorrectly predicted as 3
/// - 12 times it was incorrectly predicted as 0
/// - etc.
///
/// The diagonal (where row index equals column index) shows correct predictions.
/// Everything off the diagonal represents mistakes.
type ConfusionMatrix = [[u32; 10]; 10];

fn main() -> Result<(), Box<dyn Error>> {
    let start_time = Instant::now();

    // Load network and create trainer
    println!("Loading trained network...");
    let network = Network::load(Path::new("trained_network.json"))?;
    let trainer = Trainer::new(TrainingConfig::default());
    let load_time = start_time.elapsed();
    println!("Network loaded in {:.2?}", load_time);

    // Load test data
    let data_start = Instant::now();
    println!("Loading MNIST test data...");
    let test_data = mnist::load_mnist_test_data()?;
    println!(
        "Successfully loaded {} test examples in {:.2?}",
        test_data.images().len(),
        data_start.elapsed()
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

    // Initialize tracking variables
    let correct_per_digit = Arc::new(Mutex::new(vec![0u32; 10]));
    let total_per_digit = Arc::new(Mutex::new(vec![0u32; 10]));
    let correct_predictions = Arc::new(Mutex::new(0u32));
    let confusion_matrix: Arc<Mutex<ConfusionMatrix>> = Arc::new(Mutex::new([[0; 10]; 10]));
    let confidence_sums = Arc::new(Mutex::new(vec![0f64; 10]));
    let confidence_counts = Arc::new(Mutex::new(vec![0u32; 10]));

    // Create test data chunks for parallel processing
    let test_start = Instant::now();
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
        let conf_matrix = Arc::clone(&confusion_matrix);
        let conf_sums = Arc::clone(&confidence_sums);
        let conf_counts = Arc::clone(&confidence_counts);

        // Process each image in the chunk
        for (image, label) in images.iter().zip(labels.iter()) {
            let output = local_network.feed_forward(image).unwrap();
            let predicted = trainer.get_prediction(&output);
            let actual = mnist::get_actual_digit(label);

            // Get confidence score (max output value)
            let confidence = *output
                .data()
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();

            // Update statistics
            {
                let mut total = total_digits.lock().unwrap();
                total[actual] += 1;

                let mut matrix = conf_matrix.lock().unwrap();
                matrix[actual][predicted] += 1;

                let mut conf_sum = conf_sums.lock().unwrap();
                let mut conf_count = conf_counts.lock().unwrap();
                conf_sum[predicted] += confidence;
                conf_count[predicted] += 1;

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
    let test_time = test_start.elapsed();
    let pb = progress_bar.lock().unwrap();
    pb.finish_with_message("Testing completed!");

    let total_correct = *correct_predictions.lock().unwrap();
    let total_accuracy = (total_correct as f64) / (test_data.images().len() as f64) * 100.0;

    println!("\nTiming Metrics:");
    println!("Network load time: {:.2?}", load_time);
    println!("Data load time: {:.2?}", data_start.elapsed());
    println!("Testing time: {:.2?}", test_time);
    println!("Total time: {:.2?}", start_time.elapsed());

    println!("\nTest Results:");
    println!("Total test examples: {}", test_data.images().len());
    println!("Correct predictions: {}", total_correct);
    println!("Overall accuracy: {:.2}%", total_accuracy);

    // Display per-digit performance metrics:
    // - Digit: The actual digit (0-9) being analyzed
    // - Correct: Number of times the model correctly identified this digit
    // - Total: Total number of test examples for this digit
    // - Accuracy: Percentage of correct predictions (Correct/Total * 100)
    // - Precision: When the model predicts a digit, how often is it correct?
    // - Recall: Out of all actual instances of a digit, how many were found?
    // - F1 Score: Harmonic mean of precision and recall (balances both metrics)
    // - Avg Confidence: Average confidence score when predicting this digit,
    //                  showing how "sure" the model is about its predictions
    println!("\nPer-digit Performance:");
    println!("Digit | Correct | Total | Accuracy | Precision | Recall | F1 Score");
    println!("------|---------|--------|----------|-----------|--------|----------");

    let correct_by_digit = correct_per_digit.lock().unwrap();
    let total_by_digit = total_per_digit.lock().unwrap();

    // Get all metrics first
    let metrics: Vec<_> = {
        let matrix = confusion_matrix.lock().unwrap();
        (0..10)
            .map(|digit| {
                let accuracy =
                    (correct_by_digit[digit] as f64) / (total_by_digit[digit] as f64) * 100.0;

                // Calculate precision: TP / (TP + FP)
                let true_positives = matrix[digit][digit] as f64;
                let false_positives = (0..10).fold(0.0, |sum, i| {
                    if i != digit {
                        sum + matrix[i][digit] as f64
                    } else {
                        sum
                    }
                });
                let precision = true_positives / (true_positives + false_positives);

                // Calculate recall: TP / (TP + FN)
                let false_negatives = (0..10).fold(0.0, |sum, j| {
                    if j != digit {
                        sum + matrix[digit][j] as f64
                    } else {
                        sum
                    }
                });
                let recall = true_positives / (true_positives + false_negatives);

                // Calculate F1 score: 2 * (precision * recall) / (precision + recall)
                let f1_score = if precision + recall > 0.0 {
                    2.0 * (precision * recall) / (precision + recall)
                } else {
                    0.0
                };

                (accuracy, precision, recall, f1_score)
            })
            .collect()
    };

    // Display metrics
    for (digit, &(accuracy, precision, recall, f1_score)) in metrics.iter().enumerate() {
        println!(
            "   {digit}  |   {correct:^5} |  {total:^4}  | {accuracy:>6.2}% |  {precision:>6.2}% | {recall:>5.2}% |  {f1:>6.2}%",
            digit = digit,
            correct = correct_by_digit[digit],
            total = total_by_digit[digit],
            accuracy = accuracy,
            precision = precision * 100.0,
            recall = recall * 100.0,
            f1 = f1_score * 100.0
        );
    }

    // Add explanatory comment for the metrics
    println!("\nMetric Explanations:");
    println!("- Precision: When the model predicts a digit, how often is it correct?");
    println!("- Recall: Out of all actual instances of a digit, how many were found?");
    println!("- F1 Score: Harmonic mean of precision and recall (balances both metrics)");
    println!("  * Higher values are better (max 100%)");
    println!("  * Low precision = Many false positives (predicts digit when it's not)");
    println!("  * Low recall = Many false negatives (misses digit when it is present)");

    // Display confusion matrix
    println!("\nConfusion Matrix:");
    println!("Actual → | Predicted →");
    println!("         | 0    1    2    3    4    5    6    7    8    9   ");
    println!("---------|--------------------------------------------");

    let matrix = confusion_matrix.lock().unwrap();
    for i in 0..10 {
        print!("    {i}    |", i = i);
        for j in 0..10 {
            print!(" {:4}", matrix[i][j]);
        }
        println!();
    }

    Ok(())
}
