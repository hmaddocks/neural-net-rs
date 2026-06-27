//! MNIST test-set evaluation helpers.

use crate::mnist::load_test_data_from_dir;
use crate::{MnistData, StandardizationParams, StandardizedMnistData, get_actual_digit};
use matrix::Matrix;
use ndarray::Axis;
use neural_network::Network;
use std::fmt;
use std::path::PathBuf;

/// Pre-migration MNIST test accuracy baseline for the manual backprop path.
pub const MNIST_ACCURACY_BASELINE: f64 = 97.35;

/// Confusion matrix captured from `models/trained_network.json` on the MNIST test set.
///
/// Regenerate after retraining the baseline checkpoint:
/// ```text
/// cargo run --bin mnist --release -- test
/// ```
pub const MNIST_CONFUSION_MATRIX_BASELINE: [[usize; 10]; 10] = [
    [972, 1, 0, 1, 0, 2, 1, 2, 1, 0],
    [0, 1127, 2, 1, 0, 0, 1, 1, 3, 0],
    [3, 1, 1001, 6, 2, 0, 2, 13, 4, 0],
    [1, 0, 10, 975, 0, 13, 0, 6, 3, 2],
    [0, 0, 2, 0, 963, 0, 5, 1, 2, 9],
    [4, 0, 0, 8, 1, 867, 6, 1, 4, 1],
    [5, 3, 1, 0, 1, 2, 946, 0, 0, 0],
    [2, 4, 11, 1, 0, 0, 0, 1005, 2, 3],
    [5, 0, 3, 5, 5, 3, 3, 2, 944, 4],
    [4, 4, 0, 6, 11, 7, 1, 7, 7, 962],
];

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// Tracks model predictions versus actual digit labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ConfusionMatrix {
    matrix: [[usize; 10]; 10],
}

impl ConfusionMatrix {
    pub const fn new() -> Self {
        Self {
            matrix: [[0; 10]; 10],
        }
    }

    pub const fn from_rows(rows: [[usize; 10]; 10]) -> Self {
        Self { matrix: rows }
    }

    pub const fn record(&mut self, actual: usize, predicted: usize) {
        self.matrix[actual][predicted] += 1;
    }

    pub const fn get(&self, actual: usize, predicted: usize) -> usize {
        self.matrix[actual][predicted]
    }

    pub const fn rows(&self) -> &[[usize; 10]; 10] {
        &self.matrix
    }

    pub fn overall_accuracy(&self, total: usize) -> f64 {
        let correct = (0..10).map(|digit| self.get(digit, digit)).sum::<usize>();
        correct as f64 / total as f64 * 100.0
    }

    pub fn mismatches_against(
        &self,
        baseline: &[[usize; 10]; 10],
    ) -> Vec<(usize, usize, usize, usize)> {
        (0..10)
            .flat_map(|actual| {
                (0..10).filter_map(move |predicted| {
                    let expected = baseline[actual][predicted];
                    let actual_count = self.get(actual, predicted);
                    (actual_count != expected).then_some((
                        actual,
                        predicted,
                        expected,
                        actual_count,
                    ))
                })
            })
            .collect()
    }
}

impl fmt::Display for ConfusionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\nConfusion Matrix:")?;
        writeln!(f, "      Predicted →")?;
        writeln!(
            f,
            "Actual     0    1    2    3    4    5    6    7    8    9"
        )?;
        writeln!(
            f,
            "  ↓   +--------------------------------------------------"
        )?;
        for row in 0..10 {
            write!(f, "  {row}   |")?;
            for col in 0..10 {
                write!(f, " {:4}", self.get(row, col))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/// Path to the committed MNIST test images file, if present in the workspace.
pub fn mnist_test_images_path() -> PathBuf {
    workspace_root().join("mnist/data/t10k-images-idx3-ubyte")
}

/// Path to the committed baseline trained model, if present in the workspace.
pub fn trained_model_path() -> PathBuf {
    workspace_root().join("models/trained_network.json")
}

/// Returns true when both the baseline model and MNIST test set are on disk.
pub fn oracle_fixtures_available() -> bool {
    mnist_test_images_path().is_file()
        && workspace_root()
            .join("mnist/data/t10k-labels-idx1-ubyte")
            .is_file()
        && trained_model_path().is_file()
}

/// Loads the MNIST test set from the workspace data directory.
pub fn load_workspace_test_data() -> Result<MnistData, crate::MnistError> {
    load_test_data_from_dir(workspace_root().join("mnist/data"))
}

fn standardized_test_inputs(
    network: &Network,
    test_data: &MnistData,
) -> Result<Vec<Matrix>, crate::MnistError> {
    let standardized_params =
        if let (Some(mean), Some(std_dev)) = network.standardization_parameters() {
            StandardizationParams::new(mean, std_dev)
        } else {
            StandardizationParams::build(test_data.images())?
        };

    StandardizedMnistData::new(&standardized_params).standardize(test_data.images())
}

/// Builds a confusion matrix for a trained network on the MNIST test set.
pub fn evaluate_confusion_matrix(
    network: &Network,
    test_data: &MnistData,
) -> Result<ConfusionMatrix, crate::MnistError> {
    let standardized = standardized_test_inputs(network, test_data)?;
    let input_refs: Vec<&Matrix> = standardized.iter().collect();
    let test_matrix = Matrix::concatenate(&input_refs, Axis(1));
    let outputs = network.predict(test_matrix);

    let mut confusion_matrix = ConfusionMatrix::new();
    for index in 0..outputs.cols() {
        let predicted = get_actual_digit(&outputs.col(index))?;
        let actual = get_actual_digit(&test_data.labels()[index])?;
        confusion_matrix.record(actual, predicted);
    }

    Ok(confusion_matrix)
}

/// Computes overall test-set accuracy for a trained network.
pub fn evaluate_test_accuracy(
    network: &Network,
    test_data: &MnistData,
) -> Result<f64, crate::MnistError> {
    let confusion_matrix = evaluate_confusion_matrix(network, test_data)?;
    Ok(confusion_matrix.overall_accuracy(test_data.len()))
}

/// Loads the baseline model and evaluates it on the workspace MNIST test set.
pub fn evaluate_baseline_model_accuracy() -> Result<f64, Box<dyn std::error::Error>> {
    let model_path = trained_model_path();
    let path = model_path
        .to_str()
        .ok_or("trained model path is not valid UTF-8")?;
    let network = Network::load(path)?;
    let test_data = load_workspace_test_data()?;
    Ok(evaluate_test_accuracy(&network, &test_data)?)
}

/// Loads the baseline model and returns its confusion matrix on the test set.
pub fn evaluate_baseline_confusion_matrix() -> Result<ConfusionMatrix, Box<dyn std::error::Error>> {
    let model_path = trained_model_path();
    let path = model_path
        .to_str()
        .ok_or("trained model path is not valid UTF-8")?;
    let network = Network::load(path)?;
    let test_data = load_workspace_test_data()?;
    Ok(evaluate_confusion_matrix(&network, &test_data)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mnist_accuracy_meets_baseline_oracle() {
        if !oracle_fixtures_available() {
            eprintln!("Skipping MNIST accuracy oracle: fixtures not available");
            return;
        }

        let accuracy = evaluate_baseline_model_accuracy().expect("evaluate baseline model");
        assert!(
            accuracy >= MNIST_ACCURACY_BASELINE,
            "MNIST test accuracy {:.2}% is below baseline {:.2}%",
            accuracy,
            MNIST_ACCURACY_BASELINE
        );
    }

    #[test]
    fn test_confusion_matrix_matches_baseline_oracle() {
        if !oracle_fixtures_available() {
            eprintln!("Skipping MNIST confusion matrix oracle: fixtures not available");
            return;
        }

        let confusion_matrix =
            evaluate_baseline_confusion_matrix().expect("evaluate baseline confusion matrix");
        let mismatches = confusion_matrix.mismatches_against(&MNIST_CONFUSION_MATRIX_BASELINE);

        assert!(
            mismatches.is_empty(),
            "confusion matrix differs from baseline at {} cell(s): {:?}",
            mismatches.len(),
            mismatches
        );
    }
}
