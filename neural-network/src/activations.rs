use matrix::matrix::Matrix;
use std::f64::consts::E;

#[derive(Clone, Copy, Debug)]
pub struct Activation {
    pub function: fn(f64) -> f64,
    pub derivative: fn(f64) -> f64,
    pub vector_function: Option<fn(&Matrix) -> Matrix>,
    pub vector_derivative: Option<fn(&Matrix) -> Matrix>,
}

pub const SIGMOID: Activation = Activation {
    function: |x| 1.0 / (1.0 + E.powf(-x)),
    derivative: |x| x * (1.0 - x),
    vector_function: None,
    vector_derivative: None,
};

pub const SIGMOID_VECTOR: Activation = Activation {
    function: |x| 1.0 / (1.0 + E.powf(-x)),
    derivative: |x| x * (1.0 - x),
    vector_function: Some(|m| m.map(|x| 1.0 / (1.0 + E.powf(-x)))),
    vector_derivative: Some(|m| m.map(|x| x * (1.0 - x))),
};
