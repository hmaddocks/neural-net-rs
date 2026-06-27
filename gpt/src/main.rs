use anyhow::{Context, Result};
use gpt::data::{load_docs, shuffle};
use gpt::inference::generate_sample;
use gpt::matrix::LcgRng;
use gpt::model::{GPTConfig, StateDict, gpt as gpt_forward};
use gpt::neural_network::softmax;
use gpt::optimizer::{AdamOptimizer, linear_decay};
use gpt::tokenizer::Tokenizer;
use gpt::value::ValueArena;
use rand::SeedableRng;
use rand::rngs::SmallRng;

fn main() -> Result<()> {
    println!("MiniGPT: A minimal GPT implementation in Rust");
    println!("==============================================\n");

    // Random seed for reproducibility
    let seed: u64 = 42;

    // Load dataset
    println!("Loading dataset...");
    let mut docs = load_docs("data/names.txt").context("Failed to load dataset")?;
    shuffle(&mut docs, seed);
    println!("num docs: {}", docs.len());

    // Create tokenizer
    let tokenizer = Tokenizer::from_docs(docs.iter().map(|s| s.as_str()));
    println!("vocab size: {}", tokenizer.vocab_size());

    // Initialize model configuration
    let config = GPTConfig::new(
        1,  // n_layer
        16, // n_embd
        16, // block_size
        4,  // n_head
        tokenizer.vocab_size(),
    );

    // Initialize model parameters
    println!("Initializing model...");
    let mut arena = ValueArena::new();
    let mut rng = LcgRng::new(seed);
    let state_dict = StateDict::init(&mut arena, &config, &mut rng);
    let params = state_dict.parameters();
    println!("num params: {}", params.len());

    // Initialize optimizer
    let learning_rate = 0.01;
    let num_steps = 1000;
    let mut optimizer = AdamOptimizer::new(learning_rate, params.len(), 0.85, 0.99, 1e-8);

    // Training loop
    println!("\nTraining for {} steps...", num_steps);
    for step in 0..num_steps {
        // Get document for this step
        let doc = &docs[step % docs.len()];
        let tokens = tokenizer.encode_with_bos(doc);
        let n = tokens.len().min(config.block_size + 1) - 1;

        // Forward pass
        let mut keys: Vec<Vec<Vec<_>>> = vec![Vec::new(); config.n_layer];
        let mut values: Vec<Vec<Vec<_>>> = vec![Vec::new(); config.n_layer];
        let mut losses = Vec::new();

        for pos_id in 0..n {
            let token_id = tokens[pos_id];
            let target_id = tokens[pos_id + 1];

            // Forward through model
            let logits = gpt_forward(
                &mut arena,
                &state_dict,
                &config,
                token_id,
                pos_id,
                &mut keys,
                &mut values,
            );

            // Compute cross-entropy loss
            let probs = softmax(&mut arena, &logits);
            let loss_t = arena.log(probs[target_id]);
            let neg_loss = arena.neg(loss_t);
            losses.push(neg_loss);
        }

        // Average loss
        let mut loss = arena.create(0.0);
        for &l in &losses {
            loss = arena.add(loss, l);
        }
        let n_val = arena.create(n as f64);
        loss = arena.div(loss, n_val);

        // Backward pass
        arena.backward(loss);

        // Get loss value before clearing computation graph
        let loss_value = arena.data(loss);

        // Optimizer step with learning rate decay
        let schedule = linear_decay(learning_rate, num_steps);
        optimizer.step(&mut arena, &params, Some(schedule));

        // Clear computation graph to prevent unbounded memory growth
        // This keeps only the model parameters and discards intermediate computations
        arena.clear_computation_graph(params.len());

        // Print progress
        if (step + 1) % 10 == 0 || step == 0 {
            print!(
                "\rstep {:4} / {:4} | loss {:.4} ",
                step + 1,
                num_steps,
                loss_value,
            );
            use std::io::{self, Write};
            io::stdout().flush()?;
        }
    }
    println!("\n");

    // Inference: generate new samples
    println!("--- inference (new, hallucinated names) ---");
    let temperature = 0.5;
    let mut inference_rng = SmallRng::seed_from_u64(seed + 1000);

    for sample_idx in 0..20 {
        let sample = generate_sample(
            &mut arena,
            &state_dict,
            &config,
            tokenizer.bos_token(),
            temperature,
            &mut inference_rng,
        )
        .with_context(|| format!("Failed to generate sample {}", sample_idx + 1))?;

        let text = tokenizer.decode(&sample);
        println!("sample {:2}: {}", sample_idx + 1, text);

        // Clear arena after each sample to prevent memory buildup
        arena.zero_grad();
    }

    Ok(())
}
