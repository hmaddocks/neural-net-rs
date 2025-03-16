use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use mnist;
use mnist::mnist::get_actual_digit;
use neural_network::network::Network;
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use training::training::{Trainer, TrainingConfig};

/// Metrics for a single digit (0-9), tracking correct predictions and total occurrences.
#[derive(Debug, Default)]
struct DigitMetrics {
    correct: AtomicU32,
    total: AtomicU32,
}

/// Collection of metrics for evaluating MNIST model performance.
///
/// This struct tracks:
/// - Per-digit performance metrics (correct/total predictions)
/// - Confusion matrix for detailed error analysis
/// - Confidence scores for each prediction
#[derive(Debug)]
struct TestMetrics {
    /// Per-digit metrics tracking correct predictions and total occurrences
    per_digit: Vec<DigitMetrics>,
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
    confusion_matrix: Arc<Mutex<[[u32; 10]; 10]>>,
    /// Sum of confidence scores for each predicted digit
    confidence_sums: Arc<Mutex<[f64; 10]>>,
    /// Count of predictions made for each digit
    confidence_counts: Arc<Mutex<[u32; 10]>>,
}

impl TestMetrics {
    /// Creates a new TestMetrics instance with zeroed counters.
    fn new() -> Self {
        Self {
            per_digit: (0..10).map(|_| DigitMetrics::default()).collect(),
            confusion_matrix: Arc::new(Mutex::new([[0; 10]; 10])),
            confidence_sums: Arc::new(Mutex::new([0.0; 10])),
            confidence_counts: Arc::new(Mutex::new([0; 10])),
        }
    }

    /// Updates metrics with a new prediction result.
    ///
    /// # Arguments
    /// * `actual` - The true digit (0-9)
    /// * `predicted` - The model's predicted digit (0-9)
    /// * `confidence` - The model's confidence score for this prediction
    fn update(&self, actual: usize, predicted: usize, confidence: f64) {
        self.per_digit[actual].total.fetch_add(1, Ordering::Relaxed);
        if predicted == actual {
            self.per_digit[actual]
                .correct
                .fetch_add(1, Ordering::Relaxed);
        }

        let mut matrix = self.confusion_matrix.lock().unwrap();
        matrix[actual][predicted] += 1;

        let mut sums = self.confidence_sums.lock().unwrap();
        let mut counts = self.confidence_counts.lock().unwrap();
        sums[predicted] += confidence;
        counts[predicted] += 1;
    }

    /// Calculates performance metrics for each digit.
    ///
    /// Returns a vector of tuples (accuracy, precision, recall, f1_score) for each digit.
    /// - Accuracy: Percentage of correct predictions for this digit
    /// - Precision: When the model predicts this digit, how often is it correct?
    /// - Recall: Out of all actual instances of this digit, how many were found?
    /// - F1 Score: Harmonic mean of precision and recall
    fn calculate_performance(&self) -> Vec<(f64, f64, f64, f64)> {
        let matrix = self.confusion_matrix.lock().unwrap();
        (0..10)
            .map(|digit| {
                let correct = self.per_digit[digit].correct.load(Ordering::Relaxed) as f64;
                let total = self.per_digit[digit].total.load(Ordering::Relaxed) as f64;
                let accuracy = (correct / total) * 100.0;

                let true_positives = matrix[digit][digit] as f64;
                let false_positives = (0..10)
                    .filter(|&i| i != digit)
                    .map(|i| matrix[i][digit] as f64)
                    .sum::<f64>();
                let false_negatives = (0..10)
                    .filter(|&j| j != digit)
                    .map(|j| matrix[digit][j] as f64)
                    .sum::<f64>();

                let precision = true_positives / (true_positives + false_positives);
                let recall = true_positives / (true_positives + false_negatives);
                let f1_score = match precision + recall {
                    sum if sum > 0.0 => 2.0 * (precision * recall) / sum,
                    _ => 0.0,
                };

                (accuracy, precision, recall, f1_score)
            })
            .collect()
    }

    /// Displays detailed performance metrics for the model.
    ///
    /// Shows the following metrics for each digit (0-9):
    /// - Correct: Number of times the model correctly identified this digit
    /// - Total: Total number of test examples for this digit
    /// - Accuracy: Percentage of correct predictions (Correct/Total * 100)
    /// - Precision: When the model predicts a digit, how often is it correct?
    /// - Recall: Out of all actual instances of a digit, how many were found?
    /// - F1 Score: Harmonic mean of precision and recall (balances both metrics)
    fn display_results(&self, test_data_len: usize) {
        let metrics = self.calculate_performance();
        let total_correct: u32 = self
            .per_digit
            .iter()
            .map(|m| m.correct.load(Ordering::Relaxed))
            .sum();
        let total_accuracy = (total_correct as f64) / (test_data_len as f64) * 100.0;

        println!("\nTest Results:");
        println!("Total test examples: {}", test_data_len);
        println!("Correct predictions: {}", total_correct);
        println!("Overall accuracy: {:.2}%", total_accuracy);

        println!("\nPer-digit Performance:");
        println!("Digit | Correct | Total | Accuracy | Precision | Recall | F1 Score");
        println!("------|---------|--------|----------|-----------|--------|----------");

        for (digit, &(accuracy, precision, recall, f1_score)) in metrics.iter().enumerate() {
            let correct = self.per_digit[digit].correct.load(Ordering::Relaxed);
            let total = self.per_digit[digit].total.load(Ordering::Relaxed);
            println!(
                "   {digit}  |   {correct:^5} |  {total:^4}  | {accuracy:>6.2}% |  {precision:>6.2}% | {recall:>5.2}% |  {f1:>6.2}%",
                digit = digit,
                correct = correct,
                total = total,
                accuracy = accuracy,
                precision = precision * 100.0,
                recall = recall * 100.0,
                f1 = f1_score * 100.0
            );
        }

        self.display_confusion_matrix();
    }

    /// Displays the confusion matrix showing the model's prediction patterns.
    ///
    /// The matrix shows how often each actual digit (rows) was predicted as each possible digit (columns).
    /// The diagonal represents correct predictions, while off-diagonal elements show misclassifications.
    fn display_confusion_matrix(&self) {
        println!("\nConfusion Matrix:");
        println!("Actual → | Predicted →");
        println!("         | 0    1    2    3    4    5    6    7    8    9   ");
        println!("---------|--------------------------------------------------");

        let matrix = self.confusion_matrix.lock().unwrap();
        for i in 0..10 {
            print!("    {i}    |");
            for j in 0..10 {
                print!(" {:4}", matrix[i][j]);
            }
            println!();
        }
    }
}

fn main() -> Result<()> {
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
    let test_data = mnist::mnist::load_test_data()?;
    println!(
        "Successfully loaded {} test examples in {:.2?}",
        test_data.images().len(),
        data_start.elapsed()
    );

    // Initialize progress bar
    let progress_bar = ProgressBar::new(test_data.images().len() as u64);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
            )?
            .progress_chars("=>-"),
    );
    progress_bar.set_message("Testing model...");
    let progress_bar = Arc::new(progress_bar);

    // Initialize metrics tracking
    let metrics = Arc::new(TestMetrics::new());
    let test_start = Instant::now();

    // Process data in parallel chunks
    let chunk_size = test_data.images().len() / rayon::current_num_threads();
    test_data
        .images()
        .chunks(chunk_size)
        .zip(test_data.labels().chunks(chunk_size))
        .par_bridge()
        .for_each(|(images, labels)| {
            let mut local_network = network.clone();
            let pb = Arc::clone(&progress_bar);
            let metrics = Arc::clone(&metrics);

            for (image, label) in images.iter().zip(labels.iter()) {
                let output = local_network.feed_forward(image).unwrap();
                let predicted = trainer.get_prediction(&output);
                let actual = get_actual_digit(label);

                let confidence = output
                    .data()
                    .iter()
                    .fold(f64::NEG_INFINITY, |max, &x| max.max(x));

                metrics.update(actual, predicted, confidence);
                pb.inc(1);
            }
        });

    // Display results
    progress_bar.finish_with_message("Testing completed!");

    println!("\nTiming Metrics:");
    println!("Network load time: {:.2?}", load_time);
    println!("Data load time: {:.2?}", data_start.elapsed());
    println!("Testing time: {:.2?}", test_start.elapsed());
    println!("Total time: {:.2?}", start_time.elapsed());

    metrics.display_results(test_data.images().len());

    println!("\nMetric Explanations:");
    println!("- Precision: When the model predicts a digit, how often is it correct?");
    println!("- Recall: Out of all actual instances of a digit, how many were found?");
    println!("- F1 Score: Harmonic mean of precision and recall (balances both metrics)");
    println!("  * Higher values are better (max 100%)");
    println!("  * Low precision = Many false positives (predicts digit when it's not)");
    println!("  * Low recall = Many false negatives (misses digit when it is present)");

    Ok(())
}
