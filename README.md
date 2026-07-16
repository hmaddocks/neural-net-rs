# neural-net-rs

## MNIST

This code implements a neural network to train and test on the MNIST dataset of
handwritten digits. The network architecture and hyperparameters are fully
configurable via `config.json`: you can specify any number of layers, the node
count and activation function per layer (Sigmoid, ReLU, or Softmax), learning
rate, number of epochs, mini-batch size, optional momentum, and optional L1/L2
regularization. Here are the some recent training statistics:

```text
"Confusion Matrix:
           Predicted
Actual     0    1    2    3    4    5    6    7    8    9
      +--------------------------------------------------
  0   |  972    0    0    2    0    2    1    1    2    0
  1   |    0 1126    1    2    0    1    2    1    2    0
  2   |    7    1 1003    1    3    1    1    8    7    0
  3   |    0    0    5  984    0    7    0    7    5    2
  4   |    1    0    4    0  953    1    2    1    2   18
  5   |    4    1    0   10    1  863    4    0    7    2
  6   |    6    3    0    1    2    4  936    2    4    0
  7   |    2    9   13    2    0    0    0  991    0   11
  8   |    5    1    2    2    8    5    3    3  940    5
  9   |    7    5    1    7   11    1    2    6    2  967

Per-digit Metrics:
Digit  | Accuracy | Precision | Recall  | F1 Score
-------|----------|-----------|---------|----------
   0   |  99.2%   |   96.8%   |  99.2%  |   98.0%
   1   |  99.2%   |   98.3%   |  99.2%  |   98.7%
   2   |  97.2%   |   97.5%   |  97.2%  |   97.3%
   3   |  97.4%   |   97.3%   |  97.4%  |   97.4%
   4   |  97.0%   |   97.4%   |  97.0%  |   97.2%
   5   |  96.7%   |   97.5%   |  96.7%  |   97.1%
   6   |  97.7%   |   98.4%   |  97.7%  |   98.1%
   7   |  96.4%   |   97.2%   |  96.4%  |   96.8%
   8   |  96.5%   |   96.8%   |  96.5%  |   96.7%
   9   |  95.8%   |   96.2%   |  95.8%  |   96.0%

Overall Accuracy: 97.35%

```

I don't know if this is good or bad :) Training took about 20 minutes on the
60,000 images

---

neural-net-rs is a Rust-based neural network framework designed for educational
purposes. This project aims to provide a simple yet informative implementation
of neural networks in the Rust programming language.

[![Neural Net Rust](https://img.youtube.com/vi/DKbz9pNXVdE/0.jpg)](https://www.youtube.com/watch?v=DKbz9pNXVdE)

## Features

- **Educational Focus:** neural-net-rs is created with the primary goal of
  helping users understand the fundamentals of neural networks in Rust.
- **Simplicity:** The framework prioritizes simplicity to facilitate a smooth
  learning experience for beginners in deep learning.
- **Flexibility:** While keeping things simple, neural-net-rs is designed to be
  flexible, allowing users to experiment with different neural network architectures.

This code was originally forked from
[neural-net-rs](https://github.com/codemoonsxyz/neural-net-rs). It has been
changed and extended so much that it no longer bears any resemblance to the
original code. I would like to thank the original author for their work and for
inspiring me to extend this project.


## Getting Started

### Prerequisites

[Install Rust](https://www.rust-lang.org/learn/get-started)

### Installation

```bash
git clone https://github.com/your-username/neural-net-rs.git
cd neural-net-rs
cargo build
```

### Usage

#### MNIST (feed-forward MLP)

The `mnist` binary trains a configurable MLP on the MNIST handwritten-digit dataset.
Network architecture and hyperparameters are read from `config.json` in the
working directory.

```bash
# Train the network and save the model to models/trained_network.json
# (default: manual backprop — the original hand-written path)
cargo run --bin mnist --release -- train

# Train on the shared autograd core (parity / experimental; slower than manual)
cargo run --bin mnist --release -- train --backprop-engine autograd

# Test the saved model and print accuracy metrics (inference only; same command
# whether the model was trained with manual or autograd backprop)
cargo run --bin mnist --release -- test

# Render a training-history graph to graphs/training_history.svg
cargo run --bin mnist -- graph
```

#### MNIST live demo (local presentation)

The `mnist-demo` binary serves a browser UI for drawing digits and running live
inference against the trained MLP. Run from the workspace root so it can find
`models/trained_network.json`.

```bash
# Train first if you do not have a checkpoint yet
cargo run --bin mnist --release -- train

# Start the demo (opens http://127.0.0.1:8765 in your browser on macOS)
cargo run --bin mnist-demo --release -- --open
```

Draw on the canvas — probabilities update live while you draw (75 ms debounce).
Press Ctrl+C to stop the server when finished.

#### GPT (character-level language model)

The `gpt` binary trains a minimal GPT transformer on a names dataset and generates
new samples. Training and inference are separate subcommands — you train once,
save a checkpoint, then generate from it as many times as you like without retraining.
Run from the workspace root so it can find `data/names.txt`.

```bash
# Train for 1 000 steps and save a checkpoint (default: models/gpt_weights.json)
cargo run --bin gpt --release -- train

# Generate 20 names from the saved checkpoint
cargo run --bin gpt --release -- generate

# Generate with custom options
cargo run --bin gpt --release -- generate --num-samples 50 --temperature 0.8 --seed 42

# Use a custom data file or checkpoint path
cargo run --bin gpt --release -- train   --data data/names.txt --output models/my_run.json
cargo run --bin gpt --release -- generate --weights models/my_run.json
```

The model is a single-layer transformer (n\_embd=16, block\_size=16, 4 attention
heads) trained with Adam + linear LR decay. Expected training time: ~15 s on an
M1 Mac. The checkpoint stores the full model config, tokenizer vocabulary, and
all weight matrices as a plain JSON file — no dataset needed at generation time.
