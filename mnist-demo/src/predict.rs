use anyhow::{Context, anyhow};
use matrix::Matrix;
use mnist::get_actual_digit;
use mnist::{StandardizationParams, StandardizedMnistData};
use neural_network::Network;
use serde::{Deserialize, Serialize};

pub const INPUT_PIXELS: usize = 784;

#[derive(Debug, Deserialize)]
pub struct PredictRequest {
    pub pixels: Vec<f64>,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct PredictResponse {
    pub probabilities: Vec<f64>,
    pub prediction: usize,
}

pub fn predict_from_pixels(
    network: &Network,
    params: &StandardizationParams,
    pixels: &[f64],
) -> anyhow::Result<PredictResponse> {
    if pixels.len() != INPUT_PIXELS {
        anyhow::bail!("expected {INPUT_PIXELS} pixel values, got {}", pixels.len());
    }

    let input = Matrix::new(INPUT_PIXELS, 1, pixels.to_vec());
    let standardized = StandardizedMnistData::new(params)
        .standardize_matrix(&input)
        .map_err(|error| anyhow!("failed to standardize input: {error}"))?;
    let output = network.predict(standardized);

    let probabilities = (0..output.rows())
        .map(|index| output.get(index, 0))
        .collect::<Vec<_>>();
    let prediction = get_actual_digit(&output).context("failed to read prediction")?;

    Ok(PredictResponse {
        probabilities,
        prediction,
    })
}

pub fn standardization_params_from_network(
    network: &Network,
) -> anyhow::Result<StandardizationParams> {
    let (mean, std_dev) = network.standardization_parameters();
    Ok(StandardizationParams::new(
        mean.ok_or_else(|| anyhow!("network missing mean parameter"))?,
        std_dev.ok_or_else(|| anyhow!("network missing std_dev parameter"))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnist::MnistArtifacts;

    #[test]
    fn rejects_wrong_pixel_count() {
        let artifacts = MnistArtifacts::from_workspace_root(".");
        if !artifacts.model_path.exists() {
            return;
        }

        let network = Network::load(artifacts.model_path.to_str().unwrap()).unwrap();
        let params = standardization_params_from_network(&network).unwrap();
        let result = predict_from_pixels(&network, &params, &[0.0; 100]);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("expected 784 pixel values")
        );
    }

    #[test]
    fn probabilities_sum_to_one() {
        let artifacts = MnistArtifacts::from_workspace_root(".");
        if !artifacts.model_path.exists() {
            eprintln!(
                "skipping: no trained model at {}",
                artifacts.model_path.display()
            );
            return;
        }

        let network = Network::load(artifacts.model_path.to_str().unwrap()).unwrap();
        let params = standardization_params_from_network(&network).unwrap();
        let pixels = (0..INPUT_PIXELS)
            .map(|index| (index % 28) as f64 / 27.0)
            .collect::<Vec<_>>();

        let response = predict_from_pixels(&network, &params, &pixels).unwrap();

        assert_eq!(response.probabilities.len(), 10);
        let sum: f64 = response.probabilities.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "expected softmax sum ≈ 1.0, got {sum}"
        );
        assert!(response.prediction < 10);
    }
}
