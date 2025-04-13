# neural-net-rs

This code was originally forked from [neural-net-rs](https://github.com/codemoonsxyz/neural-net-rs). It has been changed and extended so much that it no longer bears any resemblance to the original code. I would like to thank the original author for their work and for inspiring me to extend this project.

I've merged everything into the `main` branch now so you can ignore this...

~~There are two interesting branches~~

* `main` This is my "stream of consciousness" code that poured out of me one weekend. It became overly complicated because I added multithreading half way through then tried to back it out. Threading isn't the best way to optimise neural networks. Not immediately anyway.
* `mnist` This was a new start where I concentrate on getting the code correct first. I'm pretty happy where this code is and I'll use this branch as the basis for future development.

## MNIST

This code (currently) implements a neural network to train and test on the MNIST dataset of handwritten digits. The network has three layers: an input layer, a hidden layer, and an output layer. The activation function is the original sigmoid function and a learning rate of 0.01. Here are the most recent training statistics:

```text
"layers": [784, 200, 10],
"activation": "Sigmoid",
"learning_rate": 0.01,
"momentum": 0.5

Confusion Matrix:
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

I don't know if this is good or bad :) Training took about 20 minutes on the 60,000 images

---

neural-net-rs is a Rust-based neural network framework designed for educational purposes. This project aims to provide a simple yet informative implementation of neural networks in the Rust programming language.

[![Neural Net Rust](https://img.youtube.com/vi/DKbz9pNXVdE/0.jpg)](https://www.youtube.com/watch?v=DKbz9pNXVdE)

## Features

* **Educational Focus:** neural-net-rs is created with the primary goal of helping users understand the fundamentals of neural networks in Rust.
* **Simplicity:** The framework prioritizes simplicity to facilitate a smooth learning experience for beginners in deep learning.
* **Flexibility:** While keeping things simple, neural-net-rs is designed to be flexible, allowing users to experiment with different neural network architectures.

## Getting Started

### Prerequisites

[Install Rust](https://www.rust-lang.org/learn/get-started)

### Installation

```bash
git clone https://github.com/your-username/neural-net-rs.git
cd neural-net-rs
cargo build
```
