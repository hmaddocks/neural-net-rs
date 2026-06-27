//! Engine-agnostic weight persistence for the GPT model.
//!
//! Weights are stored as named, row-major `f64` arrays in a JSON file,
//! mirroring the `trained_network.json` convention from `neural-network`.
//! The format is independent of the autograd core so it survives the
//! Phase 3 engine migration with only a thin glue rewrite.
//!
//! # File layout
//! ```json
//! {
//!   "config":  { "n_layer": 1, "n_embd": 16, … },
//!   "vocab":   ["a", "b", …],
//!   "weights": {
//!     "wte":    { "rows": 28, "cols": 16, "data": […] },
//!     "layers": [ { "attn_wq": { … }, … } ]
//!   }
//! }
//! ```

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::matrix::Matrix;
use crate::model::{GPTConfig, LayerParams, StateDict};
use crate::tokenizer::Tokenizer;
use crate::value::ValueArena;

// ── Config ──────────────────────────────────────────────────────────────────

/// Serializable copy of [`GPTConfig`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub block_size: usize,
    pub n_head: usize,
    pub vocab_size: usize,
}

impl From<&GPTConfig> for CheckpointConfig {
    fn from(c: &GPTConfig) -> Self {
        Self {
            n_layer: c.n_layer,
            n_embd: c.n_embd,
            block_size: c.block_size,
            n_head: c.n_head,
            vocab_size: c.vocab_size,
        }
    }
}

// ── Weight matrices ──────────────────────────────────────────────────────────

/// A 2-D weight matrix stored as a flat row-major `f64` array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl WeightMatrix {
    /// Snapshot an arena-backed matrix into plain `f64` values.
    fn from_arena_matrix(arena: &ValueArena, matrix: &Matrix) -> Self {
        let rows = matrix.len();
        let cols = matrix.first().map(|r| r.len()).unwrap_or(0);
        let data = matrix
            .iter()
            .flat_map(|row| row.iter().map(|&id| arena.data(id)))
            .collect();
        Self { rows, cols, data }
    }

    /// Restore an arena-backed matrix from stored `f64` values.
    ///
    /// Each `f64` becomes a fresh leaf `Value` node with no gradient history,
    /// which is correct for both inference and continued training.
    fn into_arena_matrix(self, arena: &mut ValueArena) -> Result<Matrix> {
        if self.cols == 0 {
            bail!("weight matrix has zero columns");
        }
        let expected = self.rows * self.cols;
        if self.data.len() != expected {
            bail!(
                "weight matrix size mismatch: expected {} elements ({}×{}), got {}",
                expected,
                self.rows,
                self.cols,
                self.data.len()
            );
        }
        Ok(self
            .data
            .chunks(self.cols)
            .map(|row| row.iter().map(|&v| arena.create(v)).collect())
            .collect())
    }
}

// ── Per-layer weights ────────────────────────────────────────────────────────

/// Serializable weights for one transformer layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerWeights {
    pub attn_wq: WeightMatrix,
    pub attn_wk: WeightMatrix,
    pub attn_wv: WeightMatrix,
    pub attn_wo: WeightMatrix,
    pub mlp_fc1: WeightMatrix,
    pub mlp_fc2: WeightMatrix,
}

/// All serializable model weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Weights {
    pub wte: WeightMatrix,
    pub wpe: WeightMatrix,
    pub lm_head: WeightMatrix,
    pub layers: Vec<LayerWeights>,
}

// ── Checkpoint ───────────────────────────────────────────────────────────────

/// A complete, self-contained model checkpoint.
///
/// Stores the hyperparameters, tokenizer vocabulary, and all weight matrices
/// so that a `generate` invocation needs only this file — no dataset required.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Model hyperparameters (hard-coded until Phase 4).
    pub config: CheckpointConfig,
    /// Sorted character vocabulary, excluding the BOS token.
    pub vocab: Vec<char>,
    /// All weight matrices as plain `f64`.
    pub weights: Weights,
}

impl Checkpoint {
    /// Build a checkpoint from a live, trained model.
    pub fn new(
        arena: &ValueArena,
        config: &GPTConfig,
        tokenizer: &Tokenizer,
        state_dict: &StateDict,
    ) -> Self {
        // Collect sorted chars (everything up to, but not including, the BOS token).
        let vocab: Vec<char> = (0..tokenizer.bos_token())
            .filter_map(|i| tokenizer.id_to_char(i))
            .collect();

        let weights = Weights {
            wte: WeightMatrix::from_arena_matrix(arena, &state_dict.wte),
            wpe: WeightMatrix::from_arena_matrix(arena, &state_dict.wpe),
            lm_head: WeightMatrix::from_arena_matrix(arena, &state_dict.lm_head),
            layers: state_dict
                .layers
                .iter()
                .map(|l| LayerWeights {
                    attn_wq: WeightMatrix::from_arena_matrix(arena, &l.attn_wq),
                    attn_wk: WeightMatrix::from_arena_matrix(arena, &l.attn_wk),
                    attn_wv: WeightMatrix::from_arena_matrix(arena, &l.attn_wv),
                    attn_wo: WeightMatrix::from_arena_matrix(arena, &l.attn_wo),
                    mlp_fc1: WeightMatrix::from_arena_matrix(arena, &l.mlp_fc1),
                    mlp_fc2: WeightMatrix::from_arena_matrix(arena, &l.mlp_fc2),
                })
                .collect(),
        };

        Self {
            config: CheckpointConfig::from(config),
            vocab,
            weights,
        }
    }

    /// Serialize and write to a pretty-printed JSON file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let file = std::fs::File::create(path)
            .with_context(|| format!("failed to create '{}'", path.display()))?;
        serde_json::to_writer_pretty(file, self)
            .with_context(|| format!("failed to serialize checkpoint to '{}'", path.display()))?;
        Ok(())
    }

    /// Read and deserialize a JSON checkpoint file.
    pub fn load(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read '{}'", path.display()))?;
        serde_json::from_str(&json)
            .with_context(|| format!("failed to parse checkpoint from '{}'", path.display()))
    }

    /// Decompose the checkpoint into usable components.
    ///
    /// Returns `(GPTConfig, Tokenizer, ValueArena, StateDict)`.
    ///
    /// The returned arena contains only fresh leaf parameter nodes with no
    /// gradient history — correct for both inference and continued training.
    /// Parameters are placed in the arena in the same order as
    /// [`StateDict::init`] so that [`StateDict::parameters`] returns the
    /// correct IDs.
    pub fn into_components(self) -> Result<(GPTConfig, Tokenizer, ValueArena, StateDict)> {
        let config = GPTConfig::new(
            self.config.n_layer,
            self.config.n_embd,
            self.config.block_size,
            self.config.n_head,
            self.config.vocab_size,
        );
        let tokenizer = Tokenizer::from_chars(self.vocab);
        let mut arena = ValueArena::new();
        let w = self.weights;

        // Restore matrices in the same order as StateDict::init so that
        // ValueId indices are consistent with parameters().
        let wte = w.wte.into_arena_matrix(&mut arena).context("wte")?;
        let wpe = w.wpe.into_arena_matrix(&mut arena).context("wpe")?;
        let lm_head = w.lm_head.into_arena_matrix(&mut arena).context("lm_head")?;

        let layers = w
            .layers
            .into_iter()
            .enumerate()
            .map(|(i, l)| {
                Ok(LayerParams {
                    attn_wq: l
                        .attn_wq
                        .into_arena_matrix(&mut arena)
                        .with_context(|| format!("layer {i} attn_wq"))?,
                    attn_wk: l
                        .attn_wk
                        .into_arena_matrix(&mut arena)
                        .with_context(|| format!("layer {i} attn_wk"))?,
                    attn_wv: l
                        .attn_wv
                        .into_arena_matrix(&mut arena)
                        .with_context(|| format!("layer {i} attn_wv"))?,
                    attn_wo: l
                        .attn_wo
                        .into_arena_matrix(&mut arena)
                        .with_context(|| format!("layer {i} attn_wo"))?,
                    mlp_fc1: l
                        .mlp_fc1
                        .into_arena_matrix(&mut arena)
                        .with_context(|| format!("layer {i} mlp_fc1"))?,
                    mlp_fc2: l
                        .mlp_fc2
                        .into_arena_matrix(&mut arena)
                        .with_context(|| format!("layer {i} mlp_fc2"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let state_dict = StateDict {
            wte,
            wpe,
            lm_head,
            layers,
        };
        Ok((config, tokenizer, arena, state_dict))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::generate_sample;
    use crate::matrix::LcgRng;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn tiny_setup() -> (GPTConfig, ValueArena, StateDict, Tokenizer) {
        // vocab: a b c d  → vocab_size = 5 (4 chars + BOS)
        let chars = vec!['a', 'b', 'c', 'd'];
        let tokenizer = Tokenizer::from_chars(chars);
        let config = GPTConfig::new(1, 8, 8, 2, tokenizer.vocab_size());
        let mut arena = ValueArena::new();
        let mut rng = LcgRng::new(99);
        let state_dict = StateDict::init(&mut arena, &config, &mut rng);
        (config, arena, state_dict, tokenizer)
    }

    /// Every weight value must survive a JSON round-trip to within 1 ULP.
    ///
    /// The `serde_json` in this workspace serialises `f64` without `ryu`, so
    /// the decimal ↔ binary conversion may differ by at most 1 ULP (~1e-15
    /// relative).  The inference oracle below is the binding correctness test.
    #[test]
    fn test_weight_round_trip() {
        let (config, arena, state_dict, tokenizer) = tiny_setup();

        let checkpoint = Checkpoint::new(&arena, &config, &tokenizer, &state_dict);
        let json = serde_json::to_string(&checkpoint).expect("serialize");
        let loaded: Checkpoint = serde_json::from_str(&json).expect("deserialize");
        let (_, _, loaded_arena, loaded_sd) = loaded.into_components().expect("into_components");

        let orig: Vec<f64> = state_dict
            .parameters()
            .iter()
            .map(|&id| arena.data(id))
            .collect();
        let restored: Vec<f64> = loaded_sd
            .parameters()
            .iter()
            .map(|&id| loaded_arena.data(id))
            .collect();

        assert_eq!(orig.len(), restored.len(), "parameter count changed");
        for (i, (a, b)) in orig.iter().zip(&restored).enumerate() {
            // Allow at most 1 ULP of error introduced by the decimal round-trip.
            let diff = (a - b).abs();
            let ulp_bound = a.abs() * f64::EPSILON * 8.0;
            assert!(
                diff <= ulp_bound,
                "parameter {i} changed by more than 1 ULP after round-trip: {a} vs {b} (diff={diff})"
            );
        }
    }

    /// The tokenizer vocabulary must survive a round-trip unchanged.
    #[test]
    fn test_tokenizer_round_trip() {
        let (config, arena, state_dict, original) = tiny_setup();
        let checkpoint = Checkpoint::new(&arena, &config, &original, &state_dict);
        let json = serde_json::to_string(&checkpoint).expect("serialize");
        let loaded: Checkpoint = serde_json::from_str(&json).expect("deserialize");
        let (_, restored, _, _) = loaded.into_components().expect("into_components");

        assert_eq!(original.vocab_size(), restored.vocab_size());
        assert_eq!(original.bos_token(), restored.bos_token());
        assert_eq!(original.encode("abcd"), restored.encode("abcd"));
    }

    /// Oracle: save → load → generate must produce identical token sequences.
    #[test]
    fn test_inference_oracle() {
        let (config, mut arena, state_dict, tokenizer) = tiny_setup();
        let seed = 1042u64;
        let temperature = 0.8;
        let bos = tokenizer.bos_token();

        // Generate from the original model.
        let orig_sample = generate_sample(
            &mut arena,
            &state_dict,
            &config,
            bos,
            temperature,
            &mut SmallRng::seed_from_u64(seed),
        )
        .expect("generate original");

        // Round-trip through JSON.
        let checkpoint = Checkpoint::new(&arena, &config, &tokenizer, &state_dict);
        let json = serde_json::to_string(&checkpoint).expect("serialize");
        let loaded: Checkpoint = serde_json::from_str(&json).expect("deserialize");
        let (loaded_config, loaded_tok, mut loaded_arena, loaded_sd) =
            loaded.into_components().expect("into_components");

        // Generate from the loaded model with the same seed.
        let loaded_sample = generate_sample(
            &mut loaded_arena,
            &loaded_sd,
            &loaded_config,
            loaded_tok.bos_token(),
            temperature,
            &mut SmallRng::seed_from_u64(seed),
        )
        .expect("generate loaded");

        assert_eq!(
            orig_sample, loaded_sample,
            "inference oracle: samples differ after save/load"
        );
    }
}
