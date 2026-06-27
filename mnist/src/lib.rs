mod evaluation;
mod mnist;
mod pipeline;
mod standardized_mnist;

#[cfg(test)]
mod test_sync;

pub use crate::evaluation::{
    ConfusionMatrix, MNIST_ACCURACY_BASELINE, MNIST_CONFUSION_MATRIX_BASELINE,
    evaluate_baseline_confusion_matrix, evaluate_baseline_model_accuracy,
    evaluate_confusion_matrix, evaluate_test_accuracy, load_workspace_test_data,
    mnist_test_images_path, oracle_fixtures_available, trained_model_path,
};
pub use crate::mnist::*;
pub use crate::pipeline::{
    MnistArtifacts, TrainSettings, render_training_graph, test_mnist, test_mnist_accuracy,
    train_mnist,
};
pub use crate::standardized_mnist::{StandardizationParams, StandardizedMnistData};
