use mnist::{ConfusionMatrix, MnistData, evaluate_confusion_matrix, oracle_fixtures_available};
use neural_network::Network;
use serde::Serialize;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ConfusionMatrixResponse {
    pub matrix: [[usize; 10]; 10],
    pub accuracy: f64,
    pub total: usize,
}

impl ConfusionMatrixResponse {
    pub fn from_evaluation(
        network: &Network,
        test_data: &MnistData,
    ) -> Result<Self, mnist::MnistError> {
        let confusion = evaluate_confusion_matrix(network, test_data)?;
        Ok(Self::from_matrix(&confusion, test_data.len()))
    }

    pub fn from_matrix(confusion: &ConfusionMatrix, total: usize) -> Self {
        Self {
            matrix: *confusion.rows(),
            accuracy: confusion.overall_accuracy(total),
            total,
        }
    }
}

pub fn load_confusion_matrix(network: &Network) -> Option<ConfusionMatrixResponse> {
    if !oracle_fixtures_available() {
        println!(
            "MNIST test set not found — confusion matrix omitted.\n\
             Place files under mnist/data/ or run from workspace root."
        );
        return None;
    }

    println!("Evaluating confusion matrix on MNIST test set...");
    let test_data = mnist::load_workspace_test_data().ok()?;
    let response = ConfusionMatrixResponse::from_evaluation(network, &test_data).ok()?;
    println!("Test accuracy: {:.2}%", response.accuracy);
    Some(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnist::MNIST_CONFUSION_MATRIX_BASELINE;

    #[test]
    fn builds_response_from_baseline_rows() {
        let confusion = ConfusionMatrix::from_rows(MNIST_CONFUSION_MATRIX_BASELINE);
        let response = ConfusionMatrixResponse::from_matrix(&confusion, 10_000);

        assert_eq!(response.matrix[0][0], 972);
        assert!(response.accuracy > 96.0);
        assert_eq!(response.total, 10_000);
    }
}
