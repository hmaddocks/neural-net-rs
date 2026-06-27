//! GPT model architecture implementation
//!
//! This module implements a simplified GPT-2 variant with the following modifications:
//! - LayerNorm → RMSNorm: Simpler normalization
//! - No biases: Reduces parameter count
//! - GeLU → ReLU: Simpler activation function

use crate::matrix::{Matrix, Rng, create_matrix, linear};
use crate::neural_network::{rmsnorm, softmax};
use crate::value::{ValueArena, ValueId};

const INIT_STD: f64 = 0.08;
const MLP_EXPANSION: usize = 4; // for 4 * n_embd

/// Configuration for the GPT model
#[derive(Debug, Clone)]
pub struct GPTConfig {
    /// Number of transformer layers
    pub n_layer: usize,
    /// Embedding dimension
    pub n_embd: usize,
    /// Maximum context length
    pub block_size: usize,
    /// Number of attention heads
    pub n_head: usize,
    /// Vocabulary size
    pub vocab_size: usize,
}

impl GPTConfig {
    /// Create a new GPT configuration
    pub fn new(
        n_layer: usize,
        n_embd: usize,
        block_size: usize,
        n_head: usize,
        vocab_size: usize,
    ) -> Self {
        assert_eq!(n_embd % n_head, 0, "n_embd must be divisible by n_head");
        Self {
            n_layer,
            n_embd,
            block_size,
            n_head,
            vocab_size,
        }
    }

    /// Get the dimension of each attention head
    pub const fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
}

/// State dictionary containing all model parameters
#[derive(Debug)]
pub struct StateDict {
    /// Token embeddings: [vocab_size, n_embd]
    pub wte: Matrix,
    /// Position embeddings: [block_size, n_embd]
    pub wpe: Matrix,
    /// Language modeling head: [vocab_size, n_embd]
    pub lm_head: Matrix,
    /// Per-layer parameters
    pub layers: Vec<LayerParams>,
}

/// Parameters for a single transformer layer
#[derive(Debug)]
pub struct LayerParams {
    /// Query projection: [n_embd, n_embd]
    pub attn_wq: Matrix,
    /// Key projection: [n_embd, n_embd]
    pub attn_wk: Matrix,
    /// Value projection: [n_embd, n_embd]
    pub attn_wv: Matrix,
    /// Output projection: [n_embd, n_embd]
    pub attn_wo: Matrix,
    /// MLP first layer: [MLP_EXPANSION*n_embd, n_embd]
    pub mlp_fc1: Matrix,
    /// MLP second layer: [n_embd, MLP_EXPANSION*n_embd]
    pub mlp_fc2: Matrix,
}

impl StateDict {
    /// Initialize model parameters with Gaussian random values
    ///
    /// # Arguments
    /// * `arena` - The ValueArena to create parameters in
    /// * `config` - Model configuration
    /// * `rng` - Random number generator
    pub fn init<R: Rng>(arena: &mut ValueArena, config: &GPTConfig, rng: &mut R) -> Self {
        let std = INIT_STD;

        // Initialize embeddings
        let wte = create_matrix(arena, config.vocab_size, config.n_embd, std, rng);
        let wpe = create_matrix(arena, config.block_size, config.n_embd, std, rng);
        let lm_head = create_matrix(arena, config.vocab_size, config.n_embd, std, rng);

        // Initialize layers
        let mut layers = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            layers.push(LayerParams {
                attn_wq: create_matrix(arena, config.n_embd, config.n_embd, std, rng),
                attn_wk: create_matrix(arena, config.n_embd, config.n_embd, std, rng),
                attn_wv: create_matrix(arena, config.n_embd, config.n_embd, std, rng),
                attn_wo: create_matrix(arena, config.n_embd, config.n_embd, std, rng),
                mlp_fc1: create_matrix(
                    arena,
                    MLP_EXPANSION * config.n_embd,
                    config.n_embd,
                    std,
                    rng,
                ),
                mlp_fc2: create_matrix(
                    arena,
                    config.n_embd,
                    MLP_EXPANSION * config.n_embd,
                    std,
                    rng,
                ),
            });
        }

        Self {
            wte,
            wpe,
            lm_head,
            layers,
        }
    }

    /// Collect all parameters into a flat vector
    pub fn parameters(&self) -> Vec<ValueId> {
        [&self.wte, &self.wpe, &self.lm_head]
            .into_iter()
            .chain(self.layers.iter().flat_map(|layer| {
                [
                    &layer.attn_wq,
                    &layer.attn_wk,
                    &layer.attn_wv,
                    &layer.attn_wo,
                    &layer.mlp_fc1,
                    &layer.mlp_fc2,
                ]
            }))
            .flat_map(|matrix| matrix.iter().flatten())
            .copied()
            .collect()
    }
}

/// GPT forward pass for a single token
///
/// # Arguments
/// * `arena` - The ValueArena for creating new values
/// * `state_dict` - Model parameters
/// * `config` - Model configuration
/// * `token_id` - Input token ID
/// * `pos_id` - Position ID in the sequence
/// * `keys` - KV cache for keys (modified in-place)
/// * `values` - KV cache for values (modified in-place)
///
/// # Returns
/// Logits over vocabulary [vocab_size]
pub fn gpt(
    arena: &mut ValueArena,
    state_dict: &StateDict,
    config: &GPTConfig,
    token_id: usize,
    pos_id: usize,
    keys: &mut [Vec<Vec<ValueId>>],
    values: &mut [Vec<Vec<ValueId>>],
) -> Vec<ValueId> {
    // Token embedding + position embedding
    let tok_emb = &state_dict.wte[token_id];
    let pos_emb = &state_dict.wpe[pos_id];

    let mut x: Vec<ValueId> = tok_emb
        .iter()
        .zip(pos_emb)
        .map(|(&t, &p)| arena.add(t, p))
        .collect();

    x = rmsnorm(arena, &x);

    // Process each transformer layer
    for li in 0..config.n_layer {
        let layer = &state_dict.layers[li];

        // 1) Multi-head Attention block
        // Compute normalized version separately to avoid cloning x
        let x_normalized = rmsnorm(arena, &x);

        // Compute Q, K, V projections on normalized input
        let q = linear(arena, &layer.attn_wq, &x_normalized);
        let k = linear(arena, &layer.attn_wk, &x_normalized);
        let v = linear(arena, &layer.attn_wv, &x_normalized);

        // Add to KV cache (move semantics, no clone needed)
        keys[li].push(k);
        values[li].push(v);

        // Multi-head attention
        let mut x_attn = Vec::with_capacity(config.n_embd);
        let head_dim = config.head_dim();

        for h in 0..config.n_head {
            let hs = h * head_dim;

            // Extract head from Q
            let q_h = &q[hs..hs + head_dim];

            // Extract heads from all K, V in cache
            let k_h: Vec<Vec<ValueId>> = keys[li]
                .iter()
                .map(|k_t| k_t[hs..hs + head_dim].to_vec())
                .collect();

            let v_h: Vec<Vec<ValueId>> = values[li]
                .iter()
                .map(|v_t| v_t[hs..hs + head_dim].to_vec())
                .collect();

            // Compute attention scores
            let scale = arena.create((head_dim as f64).sqrt());
            let mut attn_logits = Vec::with_capacity(k_h.len());

            for k_t in &k_h {
                // Dot product: sum(q_h[j] * k_t[j])
                let mut dot = arena.create(0.0);
                for j in 0..head_dim {
                    let prod = arena.mul(q_h[j], k_t[j]);
                    dot = arena.add(dot, prod);
                }
                // Scale by sqrt(head_dim)
                let scaled = arena.div(dot, scale);
                attn_logits.push(scaled);
            }

            // Softmax to get attention weights
            let attn_weights = softmax(arena, &attn_logits);

            // Weighted sum of values
            for j in 0..head_dim {
                let mut sum = arena.create(0.0);
                for t in 0..v_h.len() {
                    let prod = arena.mul(attn_weights[t], v_h[t][j]);
                    sum = arena.add(sum, prod);
                }
                x_attn.push(sum);
            }
        }

        // Output projection
        let x_attn_proj = linear(arena, &layer.attn_wo, &x_attn);

        // Residual connection: add projected attention output to original x
        x = x
            .iter()
            .zip(x_attn_proj)
            .map(|(&residual, attn)| arena.add(residual, attn))
            .collect();

        // 2) MLP block
        // Compute normalized version separately to avoid cloning x
        let x_normalized = rmsnorm(arena, &x);
        let mut x_mlp = linear(arena, &layer.mlp_fc1, &x_normalized);

        // ReLU activation
        x_mlp = x_mlp.iter().map(|&xi| arena.relu(xi)).collect();

        let x_mlp_proj = linear(arena, &layer.mlp_fc2, &x_mlp);

        // Residual connection: add projected MLP output to original x
        x = x
            .iter()
            .zip(x_mlp_proj)
            .map(|(&residual, mlp)| arena.add(residual, mlp))
            .collect();
    }

    // Final projection to vocabulary
    linear(arena, &state_dict.lm_head, &x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::LcgRng;

    #[test]
    fn test_forward_pass_deterministic() {
        // Test that forward pass is deterministic and correct after clone optimization
        let mut arena1 = ValueArena::new();
        let mut arena2 = ValueArena::new();

        // Create a minimal config for testing
        let config = GPTConfig::new(2, 8, 16, 2, 10);

        // Initialize with same seed for reproducibility
        let mut rng1 = LcgRng::new(42);
        let mut rng2 = LcgRng::new(42);

        let state_dict1 = StateDict::init(&mut arena1, &config, &mut rng1);
        let state_dict2 = StateDict::init(&mut arena2, &config, &mut rng2);

        // Initialize KV caches
        let mut keys1 = vec![vec![]; config.n_layer];
        let mut values1 = vec![vec![]; config.n_layer];
        let mut keys2 = vec![vec![]; config.n_layer];
        let mut values2 = vec![vec![]; config.n_layer];

        // Run forward pass on same input
        let token_id = 0;
        let pos_id = 0;

        let logits1 = gpt(
            &mut arena1,
            &state_dict1,
            &config,
            token_id,
            pos_id,
            &mut keys1,
            &mut values1,
        );

        let logits2 = gpt(
            &mut arena2,
            &state_dict2,
            &config,
            token_id,
            pos_id,
            &mut keys2,
            &mut values2,
        );

        // Check that outputs are identical
        assert_eq!(logits1.len(), logits2.len());
        for (i, (&l1, l2)) in logits1.iter().zip(logits2).enumerate() {
            let v1 = arena1.data(l1);
            let v2 = arena2.data(l2);
            assert!(
                (v1 - v2).abs() < 1e-10,
                "Logit {} differs: {} vs {}",
                i,
                v1,
                v2
            );
        }

        // Verify KV cache has correct shape
        assert_eq!(keys1.len(), config.n_layer);
        assert_eq!(values1.len(), config.n_layer);
        for li in 0..config.n_layer {
            assert_eq!(keys1[li].len(), 1); // One timestep
            assert_eq!(values1[li].len(), 1);
            assert_eq!(keys1[li][0].len(), config.n_embd);
            assert_eq!(values1[li][0].len(), config.n_embd);
        }
    }

    #[test]
    fn test_forward_pass_multi_step() {
        // Test that forward pass works correctly across multiple steps
        let mut arena = ValueArena::new();
        let config = GPTConfig::new(1, 4, 8, 1, 5);
        let mut rng = LcgRng::new(123);
        let state_dict = StateDict::init(&mut arena, &config, &mut rng);

        let mut keys = vec![vec![]; config.n_layer];
        let mut values = vec![vec![]; config.n_layer];

        // Process 3 tokens
        for step in 0..3 {
            let logits = gpt(
                &mut arena,
                &state_dict,
                &config,
                step % config.vocab_size,
                step,
                &mut keys,
                &mut values,
            );

            // Verify output shape
            assert_eq!(logits.len(), config.vocab_size);

            // Verify KV cache grows
            for li in 0..config.n_layer {
                assert_eq!(keys[li].len(), step + 1);
                assert_eq!(values[li].len(), step + 1);
            }
        }
    }
}
