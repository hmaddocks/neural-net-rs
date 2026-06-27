//! Data loading utilities for downloading and processing datasets
//!
//! This module provides functionality to download datasets from URLs and
//! process them into a usable format for training.

use anyhow::{Context, Result};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::fs;
use std::io::{BufRead, BufReader};

/// Load documents from a file, downloading it if necessary
///
/// # Arguments
/// * `file_path` - Path to the local file
///
/// # Returns
/// A `Result` containing a vector of non-empty document strings
pub fn load_docs(file_path: &str) -> Result<Vec<String>> {
    // Read file and parse into documents (lines)
    let file =
        fs::File::open(file_path).with_context(|| format!("Failed to open file: {}", file_path))?;
    let reader = BufReader::new(file);

    let docs: Vec<String> = reader
        .lines()
        .collect::<std::io::Result<Vec<_>>>()
        .context("Failed to read lines from file")?
        .into_iter()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect();

    Ok(docs)
}

/// Shuffle a vector using the Fisher-Yates algorithm via rand crate
///
/// # Arguments
/// * `items` - A mutable slice to shuffle in-place
/// * `seed` - Random seed for reproducibility
pub fn shuffle<T>(items: &mut [T], seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    items.shuffle(&mut rng);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shuffle_deterministic() {
        let mut vec1 = vec![1, 2, 3, 4, 5];
        let mut vec2 = vec![1, 2, 3, 4, 5];

        shuffle(&mut vec1, 42);
        shuffle(&mut vec2, 42);

        assert_eq!(vec1, vec2);
    }

    #[test]
    fn test_shuffle_different_seeds() {
        let mut vec1 = vec![1, 2, 3, 4, 5];
        let mut vec2 = vec![1, 2, 3, 4, 5];

        shuffle(&mut vec1, 42);
        shuffle(&mut vec2, 123);

        // With high probability, different seeds give different results
        // (could theoretically fail, but very unlikely)
        assert_ne!(vec1, vec2);
    }

    #[test]
    fn test_shuffle_preserves_elements() {
        let mut vec = vec![1, 2, 3, 4, 5];
        shuffle(&mut vec, 42);

        let mut sorted = vec.clone();
        sorted.sort();

        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_shuffle_empty() {
        let mut vec: Vec<i32> = vec![];
        shuffle(&mut vec, 42);
        assert_eq!(vec.len(), 0);
    }

    #[test]
    fn test_shuffle_single() {
        let mut vec = vec![42];
        shuffle(&mut vec, 42);
        assert_eq!(vec, vec![42]);
    }
}
