//! Adam optimizer implementation with learning rate scheduling
//!
//! This module implements the Adam optimizer (Adaptive Moment Estimation),
//! which is widely used for training neural networks.

use crate::value::{ValueArena, ValueId};

/// Adam optimizer with first and second moment estimation
pub struct AdamOptimizer {
    /// Learning rate
    learning_rate: f64,
    /// Exponential decay rate for first moment estimates
    beta1: f64,
    /// Exponential decay rate for second moment estimates
    beta2: f64,
    /// Small constant for numerical stability
    eps: f64,
    /// First moment buffer (momentum)
    m: Vec<f64>,
    /// Second moment buffer (RMSProp)
    v: Vec<f64>,
    /// Current training step (for bias correction)
    step: usize,
}

impl AdamOptimizer {
    /// Create a new Adam optimizer
    ///
    /// # Arguments
    /// * `learning_rate` - Initial learning rate
    /// * `num_params` - Number of parameters to optimize
    /// * `beta1` - Exponential decay rate for first moment (default: 0.85)
    /// * `beta2` - Exponential decay rate for second moment (default: 0.99)
    /// * `eps` - Small constant for numerical stability (default: 1e-8)
    pub fn new(learning_rate: f64, num_params: usize, beta1: f64, beta2: f64, eps: f64) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            eps,
            m: vec![0.0; num_params],
            v: vec![0.0; num_params],
            step: 0,
        }
    }

    /// Create a new Adam optimizer with default hyperparameters
    pub fn new_default(learning_rate: f64, num_params: usize) -> Self {
        Self::new(learning_rate, num_params, 0.85, 0.99, 1e-8)
    }

    /// Perform a single optimization step
    ///
    /// # Arguments
    /// * `arena` - The ValueArena containing the parameters
    /// * `params` - List of parameter IDs to update
    /// * `lr_schedule` - Optional learning rate schedule function
    pub fn step<F>(&mut self, arena: &mut ValueArena, params: &[ValueId], lr_schedule: Option<F>)
    where
        F: Fn(usize) -> f64,
    {
        self.step += 1;

        // Apply learning rate schedule if provided
        let lr = if let Some(schedule) = lr_schedule {
            schedule(self.step)
        } else {
            self.learning_rate
        };

        // Update each parameter
        for (i, &p) in params.iter().enumerate() {
            let grad = arena.grad(p);

            // Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad;

            // Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad * grad;

            // Compute bias-corrected first moment estimate
            let m_hat = self.m[i] / (1.0 - self.beta1.powi(self.step as i32));

            // Compute bias-corrected second raw moment estimate
            let v_hat = self.v[i] / (1.0 - self.beta2.powi(self.step as i32));

            // Update parameter
            let data = arena.data(p);
            let new_data = data - lr * m_hat / (v_hat.sqrt() + self.eps);
            arena.set_data(p, new_data);

            // Zero gradient
            arena.set_grad(p, 0.0);
        }
    }

    /// Get the current step number
    pub fn current_step(&self) -> usize {
        self.step
    }

    /// Reset the optimizer state
    pub fn reset(&mut self) {
        self.step = 0;
        for i in 0..self.m.len() {
            self.m[i] = 0.0;
            self.v[i] = 0.0;
        }
    }
}

/// Linear learning rate decay schedule
///
/// # Arguments
/// * `initial_lr` - Initial learning rate
/// * `num_steps` - Total number of training steps
///
/// # Returns
/// A closure that computes the learning rate for a given step
pub fn linear_decay(initial_lr: f64, num_steps: usize) -> impl Fn(usize) -> f64 {
    move |step: usize| {
        if step >= num_steps {
            0.0
        } else {
            initial_lr * (1.0 - step as f64 / num_steps as f64)
        }
    }
}

/// Cosine annealing learning rate schedule
///
/// # Arguments
/// * `initial_lr` - Initial learning rate
/// * `num_steps` - Total number of training steps
///
/// # Returns
/// A closure that computes the learning rate for a given step
pub fn cosine_annealing(initial_lr: f64, num_steps: usize) -> impl Fn(usize) -> f64 {
    move |step: usize| {
        if step >= num_steps {
            0.0
        } else {
            let progress = step as f64 / num_steps as f64;
            0.5 * initial_lr * (1.0 + (std::f64::consts::PI * progress).cos())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let opt = AdamOptimizer::new_default(0.01, 100);
        assert_eq!(opt.current_step(), 0);
        assert_eq!(opt.m.len(), 100);
        assert_eq!(opt.v.len(), 100);
    }

    #[test]
    fn test_linear_decay() {
        let schedule = linear_decay(0.01, 100);

        assert!((schedule(0) - 0.01).abs() < 1e-10);
        assert!((schedule(50) - 0.005).abs() < 1e-10);
        assert!((schedule(100) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_annealing() {
        let schedule = cosine_annealing(0.01, 100);

        assert!((schedule(0) - 0.01).abs() < 1e-10);
        assert!(schedule(50) < schedule(0));
        assert!(schedule(50) > schedule(100));
    }

    #[test]
    fn test_optimizer_step() {
        let mut arena = ValueArena::new();
        let param = arena.create(1.0);
        arena.set_grad(param, 0.1);

        let mut opt = AdamOptimizer::new_default(0.01, 1);
        opt.step(&mut arena, &[param], None::<fn(usize) -> f64>);

        // Parameter should have been updated
        assert_ne!(arena.data(param), 1.0);
        // Gradient should be zeroed
        assert_eq!(arena.grad(param), 0.0);
        // Step counter should increment
        assert_eq!(opt.current_step(), 1);
    }

    #[test]
    fn test_optimizer_reset() {
        let mut arena = ValueArena::new();
        let param = arena.create(1.0);
        arena.set_grad(param, 0.1);

        let mut opt = AdamOptimizer::new_default(0.01, 1);
        opt.step(&mut arena, &[param], None::<fn(usize) -> f64>);

        assert_eq!(opt.current_step(), 1);

        opt.reset();
        assert_eq!(opt.current_step(), 0);
        assert_eq!(opt.m[0], 0.0);
        assert_eq!(opt.v[0], 0.0);
    }
}
