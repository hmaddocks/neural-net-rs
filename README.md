# neural-net-rs

This code was originally based on [neural-net-rs](https://github.com/your-username/neural-net-rs). It has been changed and extended so much that it no longer bears any resemblance to the original code. I would like to thank the original author for their work and for inspiring me to extend this project.

## MNIST

This code (currently) implements a neural network to train and test on the MNIST dataset of handwritten digits. The network has four layers: an input layer, two hidden layers, and an output layer. The activation function is the original sigmoid function and a learning rate of 0.001. Here are the most recent training statistics:

```text
Test Results:
Total test examples: 10000
Correct predictions: 9592
Overall accuracy: 95.92%

Per-digit Performance:
Digit | Correct | Total  | Accuracy | Precision | Recall | F1 Score
------|---------|--------|----------|-----------|--------|----------
   0  |    970  |  980   |  98.98%  |   96.52%  | 98.98% |   97.73%
   1  |   1118  |  1135  |  98.50%  |   98.16%  | 98.50% |   98.33%
   2  |    980  |  1032  |  94.96%  |   95.89%  | 94.96% |   95.42%
   3  |    962  |  1010  |  95.25%  |   95.15%  | 95.25% |   95.20%
   4  |    949  |  982   |  96.64%  |   94.71%  | 96.64% |   95.67%
   5  |    842  |  892   |  94.39%  |   95.68%  | 94.39% |   95.03%
   6  |    927  |  958   |  96.76%  |   95.86%  | 96.76% |   96.31%
   7  |    981  |  1028  |  95.43%  |   96.37%  | 95.43% |   95.89%
   8  |    924  |  974   |  94.87%  |   94.96%  | 94.87% |   94.92%
   9  |    939  |  1009  |  93.06%  |   95.52%  | 93.06% |   94.28%

Confusion Matrix:
Actual → | Predicted →
         | 0       1    2    3    4    5    6    7    8    9
---------|--------------------------------------------------
    0    |  970    0    0    1    1    2    2    1    2    1
    1    |    0 1118    2    3    0    1    4    0    6    1
    2    |    7    1  980   11    4    0    8    9   11    1
    3    |    1    0   11  962    1   14    1    6    8    6
    4    |    2    0    4    0  949    0   10    0    2   15
    5    |    6    1    1   13    2  842    8    2   12    5
    6    |    8    3    3    1    4    5  927    2    5    0
    7    |    1    8   17    0    8    1    0  981    1   11
    8    |    4    2    3   11    5    7    6    8  924    4
    9    |    6    6    1    9   28    8    1    9    2  939

Metric Explanations:
- Precision: When the model predicts a digit, how often is it correct?
- Recall: Out of all actual instances of a digit, how many were found?
- F1 Score: Harmonic mean of precision and recall (balances both metrics)
  * Higher values are better (max 100%)
  * Low precision = Many false positives (predicts digit when it's not)
  * Low recall = Many false negatives (misses digit when it is present)

```

I don't know if this is good or bad :) Training took about 15 minutes on the 60,000 images

---

neural-net-rs is a Rust-based neural network framework designed for educational purposes. This project aims to provide a simple yet informative implementation of neural networks in the Rust programming language.

[![Neural Net Rust](https://img.youtube.com/vi/DKbz9pNXVdE/0.jpg)](https://www.youtube.com/watch?v=DKbz9pNXVdE)

## Features

- **Educational Focus:** neural-net-rs is created with the primary goal of helping users understand the fundamentals of neural networks in Rust.
- **Simplicity:** The framework prioritizes simplicity to facilitate a smooth learning experience for beginners in deep learning.
- **Flexibility:** While keeping things simple, neural-net-rs is designed to be flexible, allowing users to experiment with different neural network architectures.

## Getting Started

### Prerequisites

[Install Rust](https://www.rust-lang.org/learn/get-started)

### Installation

```bash
git clone https://github.com/hmaddocks/neural-net-rs
cd neural-net-rs
cargo build
