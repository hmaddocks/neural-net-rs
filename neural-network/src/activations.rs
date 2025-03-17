use serde::{Deserialize, Serialize};
use std::f64::consts::E;

#[derive(Clone, Copy, Debug)]
pub struct Activation {
    // Single element activation
    pub function: fn(f64) -> f64,
    // Single element derivative
    pub derivative: fn(f64) -> f64,
    // Vector activation (None if not supported)
    pub vector_function: Option<fn(&[f64]) -> Vec<f64>>,
    // Vector derivative (None if not supported)
    pub vector_derivative: Option<fn(&[f64], usize) -> Vec<f64>>,
    activation_type: ActivationType,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum ActivationType {
    Sigmoid,
    Softmax,
}

const SIGMOID_IMPL: Activation = Activation {
    function: |x| 1.0 / (1.0 + E.powf(-x)),
    derivative: |x| x * (1.0 - x),
    vector_function: None,
    vector_derivative: None,
    activation_type: ActivationType::Sigmoid,
};

const SOFTMAX_IMPL: Activation = Activation {
    // For single element use, just compute exp(x)
    function: |x| E.powf(x),
    // Basic derivative for single element
    derivative: |x| x * (1.0 - x),
    // Full softmax implementation for vectors
    vector_function: Some(|x: &[f64]| {
        let max_x = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f64> = x.iter().map(|&x_i| E.powf(x_i - max_x)).collect();
        let sum: f64 = exps.iter().sum();
        exps.into_iter().map(|x_i| x_i / sum).collect()
    }),
    // Softmax derivative with respect to the j-th input
    // Takes the output of softmax (y) and the index j
    vector_derivative: Some(|y: &[f64], j: usize| {
        y.iter()
            .enumerate()
            .map(|(i, &y_i)| {
                if i == j {
                    y_i * (1.0 - y_i)
                } else {
                    -y_i * y[j]
                }
            })
            .collect()
    }),
    activation_type: ActivationType::Softmax,
};

impl Activation {
    pub fn new(activation_type: ActivationType) -> Self {
        match activation_type {
            ActivationType::Sigmoid => SIGMOID_IMPL,
            ActivationType::Softmax => SOFTMAX_IMPL,
        }
    }

    // Apply activation to a vector of inputs
    pub fn apply_vector(&self, x: &[f64]) -> Vec<f64> {
        match self.vector_function {
            Some(f) => f(x),
            None => x.iter().map(|&x_i| (self.function)(x_i)).collect(),
        }
    }

    // Compute derivative with respect to the j-th input
    pub fn derivative_vector(&self, y: &[f64], j: usize) -> Vec<f64> {
        match self.vector_derivative {
            Some(f) => f(y, j),
            None => y
                .iter()
                .enumerate()
                .map(
                    |(i, &y_i)| {
                        if i == j { (self.derivative)(y_i) } else { 0.0 }
                    },
                )
                .collect(),
        }
    }
}

impl Serialize for Activation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.activation_type.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Activation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let activation_type = ActivationType::deserialize(deserializer)?;
        Ok(Activation::new(activation_type))
    }
}

pub const SIGMOID: Activation = SIGMOID_IMPL;
pub const SOFTMAX: Activation = SOFTMAX_IMPL;
