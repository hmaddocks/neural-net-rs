use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use gpt::data::{load_docs, shuffle};
use gpt::inference::generate_sample;
use gpt::matrix::LcgRng;
use gpt::model::{GPTConfig, StateDict, gpt as gpt_forward};
use gpt::neural_network::softmax;
use gpt::optimizer::{AdamOptimizer, linear_decay};
use gpt::persistence::Checkpoint;
use gpt::tokenizer::Tokenizer;
use gpt::value::ValueArena;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::io::{self, Write};
use std::path::Path;

// ── CLI definition ───────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "gpt", about = "MiniGPT: A minimal GPT implementation in Rust")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train the model and save weights to a checkpoint file.
    Train {
        /// Path to the training data (one name per line).
        #[arg(long, default_value = "data/names.txt")]
        data: String,
        /// Destination path for the weight checkpoint.
        #[arg(long, default_value = "models/gpt_weights.json")]
        output: String,
    },
    /// Load a saved checkpoint and generate samples without retraining.
    Generate {
        /// Path to a checkpoint produced by `gpt train`.
        #[arg(long, default_value = "models/gpt_weights.json")]
        weights: String,
        /// Number of samples to generate.
        #[arg(long, default_value_t = 20)]
        num_samples: usize,
        /// Sampling temperature (> 0; lower = more deterministic).
        #[arg(long, default_value_t = 0.5)]
        temperature: f64,
        /// Random seed for reproducible inference.
        #[arg(long, default_value_t = 1042)]
        seed: u64,
    },
}

// ── Entry point ──────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train { data, output } => train(&data, Path::new(&output)),
        Commands::Generate {
            weights,
            num_samples,
            temperature,
            seed,
        } => generate(Path::new(&weights), num_samples, temperature, seed),
    }
}

// ── Train ────────────────────────────────────────────────────────────────────

/// Train the model and save a weight checkpoint.
///
/// Hyperparameters stay hard-coded here until Phase 4 introduces `config.json`.
fn train(data_path: &str, output: &Path) -> Result<()> {
    println!("MiniGPT: A minimal GPT implementation in Rust");
    println!("==============================================\n");

    // Fixed seed — reproducibility is the goal, not configurability (Phase 4).
    let seed: u64 = 42;

    // Load & shuffle dataset
    println!("Loading dataset from '{data_path}'…");
    let mut docs = load_docs(data_path).context("Failed to load dataset")?;
    shuffle(&mut docs, seed);
    println!("num docs: {}", docs.len());

    // Build tokenizer from corpus
    let tokenizer = Tokenizer::from_docs(docs.iter().map(|s| s.as_str()));
    println!("vocab size: {}", tokenizer.vocab_size());

    // Model configuration (hard-coded until Phase 4)
    let config = GPTConfig::new(
        1,  // n_layer
        16, // n_embd
        16, // block_size
        4,  // n_head
        tokenizer.vocab_size(),
    );

    // Initialize parameters
    println!("Initializing model…");
    let mut arena = ValueArena::new();
    let mut rng = LcgRng::new(seed);
    let state_dict = StateDict::init(&mut arena, &config, &mut rng);
    let params = state_dict.parameters();
    println!("num params: {}", params.len());

    // Optimizer
    let learning_rate = 0.01;
    let num_steps = 1000;
    let mut optimizer = AdamOptimizer::new(learning_rate, params.len(), 0.85, 0.99, 1e-8);

    // Training loop
    println!("\nTraining for {num_steps} steps…");
    for step in 0..num_steps {
        let doc = &docs[step % docs.len()];
        let tokens = tokenizer.encode_with_bos(doc);
        let n = tokens.len().min(config.block_size + 1) - 1;

        let mut keys: Vec<Vec<Vec<_>>> = vec![Vec::new(); config.n_layer];
        let mut values: Vec<Vec<Vec<_>>> = vec![Vec::new(); config.n_layer];
        let mut losses = Vec::new();

        for pos_id in 0..n {
            let token_id = tokens[pos_id];
            let target_id = tokens[pos_id + 1];

            let logits = gpt_forward(
                &mut arena,
                &state_dict,
                &config,
                token_id,
                pos_id,
                &mut keys,
                &mut values,
            );

            let probs = softmax(&mut arena, &logits);
            let loss_t = arena.log(probs[target_id]);
            let neg_loss = arena.neg(loss_t);
            losses.push(neg_loss);
        }

        // Mean cross-entropy loss
        let mut loss = arena.create(0.0);
        for &l in &losses {
            loss = arena.add(loss, l);
        }
        let n_val = arena.create(n as f64);
        loss = arena.div(loss, n_val);

        arena.backward(loss);
        let loss_value = arena.data(loss);

        let schedule = linear_decay(learning_rate, num_steps);
        optimizer.step(&mut arena, &params, Some(schedule));
        arena.clear_computation_graph(params.len());

        if (step + 1) % 10 == 0 || step == 0 {
            print!(
                "\rstep {:4} / {num_steps} | loss {loss_value:.4} ",
                step + 1
            );
            io::stdout().flush()?;
        }
    }
    println!("\n");

    // Save checkpoint
    println!("Saving checkpoint to '{}'…", output.display());
    let checkpoint = Checkpoint::new(&arena, &config, &tokenizer, &state_dict);
    checkpoint.save(output)?;
    println!(
        "Done. Run `gpt generate --weights {}` to sample.",
        output.display()
    );

    Ok(())
}

// ── Generate ─────────────────────────────────────────────────────────────────

/// Load a saved checkpoint and generate text samples without retraining.
fn generate(weights_path: &Path, num_samples: usize, temperature: f64, seed: u64) -> Result<()> {
    println!("Loading checkpoint from '{}'…", weights_path.display());
    let checkpoint = Checkpoint::load(weights_path)?;
    let (config, tokenizer, mut arena, state_dict) = checkpoint.into_components()?;

    println!("vocab size:  {}", tokenizer.vocab_size());
    println!(
        "model:       {} layer(s), {} embedding dims, {} heads",
        config.n_layer, config.n_embd, config.n_head
    );

    println!("\n--- inference (hallucinated names) ---");
    let mut rng = SmallRng::seed_from_u64(seed);

    for sample_idx in 0..num_samples {
        let sample = generate_sample(
            &mut arena,
            &state_dict,
            &config,
            tokenizer.bos_token(),
            temperature,
            &mut rng,
        )
        .with_context(|| format!("Failed to generate sample {}", sample_idx + 1))?;

        let text = tokenizer.decode(&sample);
        println!("sample {:2}: {text}", sample_idx + 1);

        // Clear gradients between samples (no backward needed during inference)
        arena.zero_grad();
    }

    Ok(())
}
