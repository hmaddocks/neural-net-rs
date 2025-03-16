use crate::activations::Activation;
use anyhow::{anyhow, Result};
use matrix::matrix::Matrix;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// A neural network implementation with configurable layers and activation function
#[derive(Serialize, Deserialize, Clone, Builder)]
pub struct Network {
    layers: Vec<usize>,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    #[serde(skip)]
    data: Vec<Matrix>,
    #[serde(skip)]
    weight_velocities: Vec<Matrix>,
    #[serde(skip)]
    bias_velocities: Vec<Matrix>,
    activation: Activation,
    learning_rate: f64,
    momentum: f64,
}

impl Network {
    /// Creates a new neural network with the specified layer sizes, activation function, and learning rate
    pub fn new(layers: Vec<usize>, activation: Activation, learning_rate: f64) -> Self {
        let (weights, biases): (Vec<Matrix>, Vec<Matrix>) = layers
            .windows(2)
            .map(|window| {
                (
                    Matrix::random(window[1], window[0]),
                    Matrix::random(window[1], 1),
                )
            })
            .unzip();

        // Initialize velocities with zeros
        let weight_velocities: Vec<Matrix> = layers
            .windows(2)
            .map(|window| Matrix::zeros(window[1], window[0]))
            .collect();

        let bias_velocities: Vec<Matrix> = layers
            .windows(2)
            .map(|window| Matrix::zeros(window[1], 1))
            .collect();

        // Pre-allocate data vector with capacity
        let mut data = Vec::with_capacity(layers.len());
        data.resize(layers.len(), Matrix::default());

        Network {
            layers,
            weights,
            biases,
            data,
            weight_velocities,
            bias_velocities,
            activation,
            learning_rate,
            momentum: 0.9, // Default momentum value
        }
    }

    /// Performs forward propagation through the network
    pub fn feed_forward(&mut self, inputs: &Matrix) -> Result<Matrix> {
        if self.layers[0] != inputs.data().len() {
            return Err(anyhow!(
                "Invalid number of inputs: expected {}, got {}",
                self.layers[0],
                inputs.data().len()
            ));
        }

        // Store input in data vector
        self.data[0] = inputs.clone();
        let mut current = inputs.clone();

        // Process each layer
        for i in 0..self.layers.len() - 1 {
            current = {
                let weighted = self.weights[i].dot_multiply(&current);
                let biased = weighted.add(&self.biases[i]);
                let activated = biased.map(self.activation.function);
                self.data[i + 1] = activated.clone();
                activated
            };
        }

        Ok(current)
    }

    /// Performs backpropagation to update weights and biases
    pub fn back_propogate(&mut self, outputs: Matrix, targets: &Matrix) {
        let mut errors = targets.subtract(&outputs);
        let mut gradients = outputs.map(self.activation.derivative);

        for i in (0..self.layers.len() - 1).rev() {
            // Calculate gradients
            gradients = gradients
                .elementwise_multiply(&errors)
                .map(|x| x * self.learning_rate);

            // Calculate weight updates with momentum
            let weight_deltas = gradients.dot_multiply(&self.data[i].transpose());
            self.weight_velocities[i] = self.weight_velocities[i]
                .map(|x| x * self.momentum)
                .add(&weight_deltas);
            self.weights[i] = self.weights[i].add(&self.weight_velocities[i]);

            // Calculate bias updates with momentum
            self.bias_velocities[i] = self.bias_velocities[i]
                .map(|x| x * self.momentum)
                .add(&gradients);
            self.biases[i] = self.biases[i].add(&self.bias_velocities[i]);

            // Prepare for next layer
            errors = self.weights[i].transpose().dot_multiply(&errors);
            gradients = self.data[i].map(self.activation.derivative);
        }
    }

    /// Trains the network using the provided inputs and targets
    pub fn train(
        &mut self,
        inputs: Vec<Vec<f64>>,
        targets: Vec<Vec<f64>>,
        epochs: u32,
    ) -> Result<()> {
        if inputs.len() != targets.len() {
            return Err(anyhow!("Number of inputs must match number of targets"));
        }

        // Pre-compute matrices for inputs and targets
        let input_matrices: Vec<Matrix> = inputs.into_par_iter().map(Matrix::from).collect();
        let target_matrices: Vec<Matrix> = targets.into_par_iter().map(Matrix::from).collect();

        // Training loop
        for epoch in 1..=epochs {
            if epochs < 100 || epoch % (epochs / 100) == 0 {
                println!("Epoch {} of {}", epoch, epochs);
            }

            // Process each training example
            for (input, target) in input_matrices.iter().zip(target_matrices.iter()) {
                let outputs = self.feed_forward(input)?;
                self.back_propogate(outputs, target);
            }
        }

        Ok(())
    }

    /// Saves the network to a file in JSON format
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Loads a network from a JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let json = fs::read_to_string(path)?;
        let mut network: Network = serde_json::from_str(&json)?;

        // Reinitialize data and velocity vectors
        network.data = vec![Matrix::default(); network.layers.len()];
        network.weight_velocities = network
            .layers
            .windows(2)
            .map(|window| Matrix::zeros(window[1], window[0]))
            .collect();
        network.bias_velocities = network
            .layers
            .windows(2)
            .map(|window| Matrix::zeros(window[1], 1))
            .collect();

        Ok(network)
    }

    /// Synchronizes weights, biases and their velocities with another network
    pub fn sync_with(&mut self, other: &Network) -> Result<()> {
        if self.layers != other.layers {
            return Err(anyhow!("Cannot sync networks with different architectures"));
        }

        // Average weights and velocities
        for i in 0..self.weights.len() {
            self.weights[i] = self.weights[i].add(&other.weights[i]).map(|x| x * 0.5);
            self.biases[i] = self.biases[i].add(&other.biases[i]).map(|x| x * 0.5);
            self.weight_velocities[i] = self.weight_velocities[i]
                .add(&other.weight_velocities[i])
                .map(|x| x * 0.5);
            self.bias_velocities[i] = self.bias_velocities[i]
                .add(&other.bias_velocities[i])
                .map(|x| x * 0.5);
        }

        Ok(())
    }
}
