use serde::{Deserialize, Serialize};
use std::f64::consts::E;

#[derive(Clone, Copy, Debug)]
pub struct Activation {
    pub function: fn(f64) -> f64,
    pub derivative: fn(f64) -> f64,
    activation_type: ActivationType,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
enum ActivationType {
    Sigmoid,
    // Add other activation types as needed
}

impl Activation {
    pub fn new(activation_type: ActivationType) -> Self {
        match activation_type {
            ActivationType::Sigmoid => Self {
                function: |x| 1.0 / (1.0 + E.powf(-x)),
                derivative: |x| x * (1.0 - x),
                activation_type,
            },
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

pub const SIGMOID: Activation = Activation {
    function: |x| 1.0 / (1.0 + E.powf(-x)),
    derivative: |x| x * (1.0 - x),
    activation_type: ActivationType::Sigmoid,
};
