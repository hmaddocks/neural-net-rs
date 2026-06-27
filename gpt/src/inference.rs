//! Inference utilities for text generation
//!
//! This module provides functionality for generating text using a trained GPT model,
//! including temperature-controlled sampling for controlling creativity.

use crate::model::{GPTConfig, StateDict, gpt};
use crate::neural_network::softmax;
use crate::value::{ValueArena, ValueId};
use anyhow::{Context, Result};
use rand::SeedableRng;
use rand::distr::{Distribution, weighted::WeightedIndex};
use rand::rngs::SmallRng;

/// Apply temperature scaling to logits
///
/// # Arguments
/// * `arena` - The ValueArena for creating new values
/// * `logits` - Input logits
/// * `temperature` - Temperature parameter (lower = more deterministic, higher = more random)
///
/// # Returns
/// Temperature-scaled logits
pub fn apply_temperature(
    arena: &mut ValueArena,
    logits: &[ValueId],
    temperature: f64,
) -> Vec<ValueId> {
    assert!(
        temperature > 0.0 && temperature.is_finite(),
        "Temperature must be positive and finite, got: {}",
        temperature
    );

    // Fast path: temperature = 1.0 means no scaling needed
    if (temperature - 1.0).abs() < f64::EPSILON {
        return logits.to_vec();
    }

    logits
        .iter()
        .map(|&logit| arena.div_scalar(logit, temperature))
        .collect()
}

type KVCache = Vec<Vec<Vec<ValueId>>>;

/// Generate a single sample from the model
///
/// # Arguments
/// * `arena` - The ValueArena for creating new values
/// * `state_dict` - Model parameters
/// * `config` - Model configuration
/// * `bos_token` - Beginning of sequence token ID
/// * `temperature` - Sampling temperature (0, 1] (lower = more deterministic)
/// * `rng` - Random number generator for sampling
///
/// # Returns
/// A vector of token IDs representing the generated sequence (excluding BOS)
///
/// # Errors
/// Returns an error if the probability distribution is invalid (e.g., all zeros or contains NaN/infinity)
pub fn generate_sample(
    arena: &mut ValueArena,
    state_dict: &StateDict,
    config: &GPTConfig,
    bos_token: usize,
    temperature: f64,
    rng: &mut SmallRng,
) -> Result<Vec<usize>> {
    let mut keys: KVCache = vec![Vec::with_capacity(config.block_size); config.n_layer];
    let mut values: KVCache = vec![Vec::with_capacity(config.block_size); config.n_layer];
    let mut token_id = bos_token;
    let mut sample = Vec::new();

    for pos_id in 0..config.block_size {
        // Forward pass through the model
        let logits = gpt(
            arena,
            state_dict,
            config,
            token_id,
            pos_id,
            &mut keys,
            &mut values,
        );

        // Apply temperature scaling
        let scaled_logits = apply_temperature(arena, &logits, temperature);

        // Convert to probabilities with softmax
        let probs = softmax(arena, &scaled_logits);

        // Extract probability values
        let prob_values: Vec<f64> = probs.iter().map(|&p| arena.data(p)).collect();

        // Sample next token using WeightedIndex distribution
        let dist = WeightedIndex::new(&prob_values).with_context(|| {
            format!(
                "Failed to create probability distribution at position {}",
                pos_id
            )
        })?;
        token_id = dist.sample(rng);

        // Stop if we hit BOS token
        if token_id == bos_token {
            break;
        }

        sample.push(token_id);

        // Clear the arena to prevent memory buildup during generation
        // We don't need gradients during inference
        arena.zero_grad();
    }

    Ok(sample)
}

/// Generate multiple samples from the model
///
/// # Arguments
/// * `arena` - The ValueArena for creating new values
/// * `state_dict` - Model parameters
/// * `config` - Model configuration
/// * `bos_token` - Beginning of sequence token ID
/// * `temperature` - Sampling temperature (> 0.0)
///   - Values < 1.0: more deterministic (sharper distribution)
///   - Value = 1.0: unchanged distribution
///   - Values > 1.0: more random (flatter distribution)
///   - Typical range: 0.1 to 2.0
/// * `num_samples` - Number of samples to generate
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// A vector of token sequences
pub fn generate_samples(
    arena: &mut ValueArena,
    state_dict: &StateDict,
    config: &GPTConfig,
    bos_token: usize,
    temperature: f64,
    num_samples: usize,
    seed: u64,
) -> Result<Vec<Vec<usize>>> {
    let mut rng = SmallRng::seed_from_u64(seed);

    (0..num_samples)
        .map(|_| generate_sample(arena, state_dict, config, bos_token, temperature, &mut rng))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_categorical() {
        let mut rng = SmallRng::seed_from_u64(42);
        let weights = vec![0.1, 0.2, 0.7];

        // Sample many times and check distribution roughly matches
        let mut counts = vec![0; 3];
        let dist = WeightedIndex::new(&weights)
            .expect("Failed to create WeightedIndex with valid weights");

        for _ in 0..1000 {
            let idx = dist.sample(&mut rng);
            counts[idx] += 1;
        }

        // The third category should have most samples (roughly 700/1000)
        assert!(counts[2] > counts[1]);
        assert!(counts[1] > counts[0]);
    }

    #[test]
    fn test_sample_categorical_uniform() {
        let mut rng = SmallRng::seed_from_u64(42);
        let weights = vec![1.0, 1.0, 1.0];

        let mut counts = vec![0; 3];
        let dist = WeightedIndex::new(&weights)
            .expect("Failed to create WeightedIndex with uniform weights");

        for _ in 0..1000 {
            let idx = dist.sample(&mut rng);
            counts[idx] += 1;
        }

        // All should be roughly equal (within 20%)
        for count in counts {
            let ratio = count as f64 / 333.0;
            assert!(ratio > 0.8 && ratio < 1.2);
        }
    }

    #[test]
    fn test_apply_temperature() {
        let mut arena = ValueArena::new();
        let logits = vec![arena.create(1.0), arena.create(2.0), arena.create(3.0)];

        let scaled = apply_temperature(&mut arena, &logits, 0.5);

        assert!((arena.data(scaled[0]) - 2.0).abs() < 1e-10);
        assert!((arena.data(scaled[1]) - 4.0).abs() < 1e-10);
        assert!((arena.data(scaled[2]) - 6.0).abs() < 1e-10);
    }
}
