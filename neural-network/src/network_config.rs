use crate::activations::{Activation, ActivationType, SIGMOID, SOFTMAX};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub layers: Vec<usize>,
    pub activations: Vec<Activation>,
    pub learning_rate: f64,
    pub momentum: Option<f64>,
    pub epochs: usize,
}

impl NetworkConfig {
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let config_str = fs::read_to_string(path)?;
        let config: NetworkConfig = serde_json::from_str(&config_str)?;
        Ok(config)
    }

    pub fn get_activations(&self) -> Vec<Activation> {
        self.activations
            .iter()
            .map(|activation| match activation.activation_type {
                ActivationType::Sigmoid => SIGMOID,
                ActivationType::Softmax => SOFTMAX,
            })
            .collect()
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            layers: vec![784, 128, 10], // Common MNIST-like default architecture
            activations: vec![SIGMOID, SIGMOID], // Sigmoid for hidden and output layers
            learning_rate: 0.1,
            momentum: Some(0.9),
            epochs: 30,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_load_config() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("test_config.json");

        let config_json = r#"{
            "layers": [784, 200, 10],
            "activation_functions": ["sigmoid", "sigmoid"],
            "learning_rate": 0.01,
            "momentum": 0.5,
            "epochs": 30
        }"#;

        let mut file = File::create(&config_path).unwrap();
        file.write_all(config_json.as_bytes()).unwrap();

        let config = NetworkConfig::load(&config_path).unwrap();
        assert_eq!(config.layers, vec![784, 200, 10]);
        assert_eq!(config.activations, vec![SIGMOID, SIGMOID]);
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.momentum, Some(0.5));
        assert_eq!(config.epochs, 30);
    }

    #[test]
    fn test_default_config() {
        let config = NetworkConfig::default();
        assert_eq!(config.layers, vec![784, 128, 10]);
        assert_eq!(config.activations, vec![SIGMOID, SIGMOID]);
        assert_eq!(config.learning_rate, 0.1);
        assert_eq!(config.momentum, Some(0.9));
        assert_eq!(config.epochs, 30);
    }
}
