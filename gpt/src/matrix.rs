//! Matrix operations and utilities for the GPT model
//!
//! This module provides:
//! - Matrix type and initialization
//! - Linear algebra operations (matrix-vector multiplication)
//! - Random number generation for parameter initialization

use crate::value::{ValueArena, ValueId};

/// Type alias for a matrix of Value IDs
pub type Matrix = Vec<Vec<ValueId>>;

/// Simple random number generator trait
pub trait Rng {
    fn gauss(&mut self, mean: f64, std: f64) -> f64;
}

/// Linear Congruential Generator for reproducible random numbers
pub struct LcgRng {
    state: u64,
}

impl LcgRng {
    /// Create a new LCG with the given seed
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    fn next(&mut self) -> u64 {
        const A: u64 = 1664525;
        const C: u64 = 1013904223;
        self.state = self.state.wrapping_mul(A).wrapping_add(C);
        self.state
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next() as f64) / (u64::MAX as f64)
    }
}

impl Rng for LcgRng {
    /// Generate a sample from a Gaussian distribution using Box-Muller transform
    fn gauss(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std * z0
    }
}

/// Create a matrix with Gaussian random initialization
pub fn create_matrix<R: Rng>(
    arena: &mut ValueArena,
    nout: usize,
    nin: usize,
    std: f64,
    rng: &mut R,
) -> Matrix {
    let mut matrix = Vec::with_capacity(nout);
    for _ in 0..nout {
        let mut row = Vec::with_capacity(nin);
        for _ in 0..nin {
            let val = arena.create(rng.gauss(0.0, std));
            row.push(val);
        }
        matrix.push(row);
    }
    matrix
}

/// Matrix-vector multiplication: y = W @ x
///
/// # Arguments
/// * `arena` - The ValueArena for creating new values
/// * `w` - Weight matrix [nout, nin]
/// * `x` - Input vector [nin]
///
/// # Returns
/// Output vector [nout]
pub fn linear(arena: &mut ValueArena, w: &Matrix, x: &[ValueId]) -> Vec<ValueId> {
    w.iter()
        .map(|row| {
            row.iter()
                .zip(x)
                .fold(arena.create(0.0), |sum, (&w_val, &x_val)| {
                    let prod = arena.mul(w_val, x_val);
                    arena.add(sum, prod)
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear() {
        let mut arena = ValueArena::new();

        // Create a 2x3 weight matrix
        let w = vec![
            vec![arena.create(1.0), arena.create(2.0), arena.create(3.0)],
            vec![arena.create(4.0), arena.create(5.0), arena.create(6.0)],
        ];

        // Input vector [1, 2, 3]
        let x = vec![arena.create(1.0), arena.create(2.0), arena.create(3.0)];

        let y = linear(&mut arena, &w, &x);

        // Expected: [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
        assert_eq!(y.len(), 2);
        assert!((arena.data(y[0]) - 14.0).abs() < 1e-10);
        assert!((arena.data(y[1]) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_rng_reproducible() {
        let mut rng1 = LcgRng::new(42);
        let mut rng2 = LcgRng::new(42);

        for _ in 0..10 {
            assert_eq!(rng1.gauss(0.0, 1.0), rng2.gauss(0.0, 1.0));
        }
    }
}
