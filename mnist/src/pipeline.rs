//! End-to-end MNIST train, test, and graph pipeline shared by the CLI and tests.

use crate::{
    ConfusionMatrix, MnistData, StandardizationParams, StandardizedMnistData,
    evaluate_confusion_matrix, evaluate_test_accuracy,
};
use neural_network::{BackpropEngine, Epochs, Network, NetworkConfig, TrainingHistory};
use plotters::prelude::*;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Paths used by the MNIST CLI and integration tests.
#[derive(Debug, Clone)]
pub struct MnistArtifacts {
    pub config_path: PathBuf,
    pub model_path: PathBuf,
    pub history_path: PathBuf,
    pub graph_path: PathBuf,
}

impl MnistArtifacts {
    pub fn from_workspace_root(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref();
        Self {
            config_path: root.join("config.json"),
            model_path: root.join("models/trained_network.json"),
            history_path: root.join("models/training_history.json"),
            graph_path: root.join("graphs/training_history.svg"),
        }
    }

    pub fn ensure_parent_dirs(&self) -> std::io::Result<()> {
        for path in [&self.model_path, &self.history_path, &self.graph_path] {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
        }
        Ok(())
    }
}

/// Training options for the MNIST pipeline.
#[derive(Debug, Clone, Copy)]
pub struct TrainSettings {
    pub backprop_engine: BackpropEngine,
    pub max_training_samples: Option<usize>,
    pub epochs_override: Option<usize>,
}

impl Default for TrainSettings {
    fn default() -> Self {
        Self {
            backprop_engine: BackpropEngine::Manual,
            max_training_samples: None,
            epochs_override: None,
        }
    }
}

fn subset_training_data(data: &MnistData, limit: usize) -> MnistData {
    let limit = limit.min(data.len());
    MnistData::new(
        data.images()[..limit].to_vec(),
        data.labels()[..limit].to_vec(),
    )
    .expect("training subset preserves image/label alignment")
}

/// Trains a network on MNIST, saving the model and training history.
pub fn train_mnist(
    training_data: &MnistData,
    artifacts: &MnistArtifacts,
    settings: TrainSettings,
) -> Result<(), Box<dyn std::error::Error>> {
    let training_data = settings
        .max_training_samples
        .map(|limit| subset_training_data(training_data, limit))
        .unwrap_or_else(|| {
            MnistData::new(
                training_data.images().to_vec(),
                training_data.labels().to_vec(),
            )
            .expect("clone training data")
        });

    let standardized_params = StandardizationParams::build(training_data.images())?;
    let standardized_data =
        StandardizedMnistData::new(&standardized_params).standardize(training_data.images())?;

    let mut network_config = NetworkConfig::load(&artifacts.config_path)?;
    if let Some(epochs) = settings.epochs_override {
        network_config.epochs = Epochs::try_from(epochs).map_err(|_| "invalid epoch override")?;
    }

    let mut network = Network::new(&network_config);
    network.set_backprop_engine(settings.backprop_engine);
    network.set_standardization_parameters(
        Some(standardized_params.mean()),
        Some(standardized_params.std_dev()),
    );

    let history = network.train(&standardized_data, training_data.labels());
    artifacts.ensure_parent_dirs()?;
    save_training_history(history, &artifacts.history_path)?;
    network.save(&artifacts.model_path)?;

    Ok(())
}

/// Evaluates a saved model on the MNIST test set.
pub fn test_mnist(
    model_path: impl AsRef<Path>,
    test_data: &MnistData,
) -> Result<ConfusionMatrix, Box<dyn std::error::Error>> {
    let path = model_path
        .as_ref()
        .to_str()
        .ok_or("model path is not valid UTF-8")?;
    let network = Network::load(path)?;
    Ok(evaluate_confusion_matrix(&network, test_data)?)
}

/// Returns overall test-set accuracy for a saved model.
pub fn test_mnist_accuracy(
    model_path: impl AsRef<Path>,
    test_data: &MnistData,
) -> Result<f64, Box<dyn std::error::Error>> {
    let path = model_path
        .as_ref()
        .to_str()
        .ok_or("model path is not valid UTF-8")?;
    let network = Network::load(path)?;
    Ok(evaluate_test_accuracy(&network, test_data)?)
}

/// Renders training history to an SVG graph.
pub fn render_training_graph(
    history_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let history_path = history_path.as_ref();
    let output_path = output_path.as_ref();

    if !history_path.is_file() {
        return Err(format!(
            "training history file not found at {}",
            history_path.display()
        )
        .into());
    }

    let history_json = fs::read_to_string(history_path)?;
    let history: TrainingHistory = serde_json::from_str(&history_json)?;

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let root = SVGBackend::new(output_path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let epoch_range = 0.0..(history.accuracies.len() as f64);
    let loss_max = history.losses.iter().copied().fold(0.0, f64::max);
    let accuracy_formatter = |y: &f64| format!("{y:.1}%");
    let x_formatter = |x: &f64| format!("{}", x.round());

    let mut chart = ChartBuilder::on(&root)
        .caption("Training Metrics", ("sans-serif", 30).into_font())
        .margin(10)
        .margin_bottom(60)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .right_y_label_area_size(60)
        .build_cartesian_2d(epoch_range.clone(), 0.0..100.0)?
        .set_secondary_coord(epoch_range, 0.0..loss_max);

    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .disable_mesh()
        .y_label_formatter(&accuracy_formatter)
        .x_label_formatter(&x_formatter)
        .x_desc("Epoch")
        .y_desc("Accuracy (%)")
        .draw()?;

    chart
        .configure_secondary_axes()
        .y_labels(10)
        .y_label_formatter(&|y: &f64| format!("{y:.4}"))
        .y_desc("Loss")
        .draw()?;

    let accuracy_points = history
        .accuracies
        .iter()
        .enumerate()
        .map(|(index, accuracy)| (index as f64, *accuracy))
        .collect::<Vec<_>>();
    let loss_points = history
        .losses
        .iter()
        .enumerate()
        .map(|(index, loss)| (index as f64, *loss))
        .collect::<Vec<_>>();

    chart
        .draw_series(LineSeries::new(accuracy_points, &BLUE))?
        .label("Accuracy")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart
        .draw_secondary_series(LineSeries::new(loss_points, &RED))?
        .label("Loss")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .margin(10)
        .draw()?;

    Ok(())
}

fn save_training_history(
    history: &TrainingHistory,
    history_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = history_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let history_json = serde_json::to_string_pretty(history)?;
    let mut file = File::create(history_path)?;
    file.write_all(history_json.as_bytes())?;
    Ok(())
}

#[cfg(test)]
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

#[cfg(test)]
fn workspace_training_data() -> Result<MnistData, Box<dyn std::error::Error>> {
    use crate::mnist::load_training_data_from_dir;
    Ok(load_training_data_from_dir(
        workspace_root().join("mnist/data"),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{load_workspace_test_data, oracle_fixtures_available};
    use neural_network::BackpropEngine;
    use tempfile::tempdir;

    #[test]
    fn test_autograd_mnist_pipeline_end_to_end() {
        let _lock = crate::test_sync::integration_lock();
        if !oracle_fixtures_available() {
            eprintln!("Skipping MNIST e2e pipeline: fixtures not available");
            return;
        }

        let temp = tempdir().expect("tempdir");
        let artifacts = MnistArtifacts {
            config_path: workspace_root().join("config.json"),
            model_path: temp.path().join("trained_network.json"),
            history_path: temp.path().join("training_history.json"),
            graph_path: temp.path().join("training_history.svg"),
        };

        let training_data = workspace_training_data().expect("load training data");
        let settings = TrainSettings {
            backprop_engine: BackpropEngine::Autograd,
            max_training_samples: Some(256),
            epochs_override: Some(2),
        };

        train_mnist(&training_data, &artifacts, settings).expect("train with autograd core");

        let test_data = load_workspace_test_data().expect("load test data");
        let accuracy =
            test_mnist_accuracy(&artifacts.model_path, &test_data).expect("evaluate trained model");
        assert!(
            accuracy > 50.0,
            "autograd smoke training should exceed chance, got {accuracy:.2}%"
        );

        render_training_graph(&artifacts.history_path, &artifacts.graph_path)
            .expect("render training graph");

        let svg = fs::read_to_string(&artifacts.graph_path).expect("read graph svg");
        assert!(svg.contains("<svg"), "graph output should be SVG");
        assert!(artifacts.history_path.is_file());
        assert!(artifacts.model_path.is_file());
    }

    #[test]
    fn test_render_training_graph_from_workspace_history() {
        let _lock = crate::test_sync::integration_lock();
        if !oracle_fixtures_available() {
            eprintln!("Skipping graph smoke test: fixtures not available");
            return;
        }

        let artifacts = MnistArtifacts::from_workspace_root(workspace_root());
        if !artifacts.history_path.is_file() {
            eprintln!("Skipping graph smoke test: training history not available");
            return;
        }

        let temp = tempdir().expect("tempdir");
        let output = temp.path().join("training_history.svg");
        render_training_graph(&artifacts.history_path, &output).expect("render graph");

        let svg = fs::read_to_string(output).expect("read graph svg");
        assert!(svg.contains("<svg"));
    }
}
