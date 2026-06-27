//! Criterion benchmarks for the MLP manual backprop path (pre-migration baseline).
//!
//! Save baseline before autograd migration:
//! ```text
//! cargo bench -p neural-network -- --save-baseline manual_pre_migration
//! ```
//!
//! Compare after migration:
//! ```text
//! cargo bench -p neural-network -- --baseline manual_pre_migration
//! ```

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use matrix::Matrix;
use ndarray::Axis;
use neural_network::{
    Activation, BackpropEngine, Layer, Network, NetworkConfig, RegularizationType,
};
use std::path::Path;

const MNIST_INPUTS: usize = 784;
const MNIST_OUTPUTS: usize = 10;
const MNIST_BATCH: usize = 32;
const MNIST_EPOCH_SAMPLES: usize = 512;

fn mnist_mlp_config() -> NetworkConfig {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../config.json");
    NetworkConfig::load(&path).expect("load workspace config.json")
}

fn synthetic_mnist_samples(count: usize) -> (Vec<Matrix>, Vec<Matrix>) {
    let inputs = (0..count)
        .map(|sample| {
            let data = (0..MNIST_INPUTS)
                .map(|pixel| ((sample * MNIST_INPUTS + pixel) % 256) as f64 / 255.0)
                .collect();
            Matrix::new(MNIST_INPUTS, 1, data)
        })
        .collect();

    let targets = (0..count)
        .map(|sample| {
            let mut data = vec![0.0; MNIST_OUTPUTS];
            data[sample % MNIST_OUTPUTS] = 1.0;
            Matrix::new(MNIST_OUTPUTS, 1, data)
        })
        .collect();

    (inputs, targets)
}

fn batched_input(batch_size: usize) -> Matrix {
    let (inputs, _) = synthetic_mnist_samples(batch_size);
    let refs: Vec<&Matrix> = inputs.iter().collect();
    Matrix::concatenate(&refs, Axis(1))
}

fn batched_targets(batch_size: usize) -> Matrix {
    let (_, targets) = synthetic_mnist_samples(batch_size);
    let refs: Vec<&Matrix> = targets.iter().collect();
    Matrix::concatenate(&refs, Axis(1))
}

fn mnist_mlp_manual_forward(c: &mut Criterion) {
    let config = mnist_mlp_config();
    let mut group = c.benchmark_group("mnist_mlp_manual");
    group.throughput(Throughput::Elements(MNIST_BATCH as u64));

    let input = batched_input(MNIST_BATCH);
    let mut network = Network::new(&config);
    assert_eq!(network.backprop_engine(), BackpropEngine::Manual);

    group.bench_function("feed_forward_batch32", |b| {
        b.iter(|| {
            black_box(network.feed_forward(black_box(input.clone())));
        });
    });

    group.finish();
}

fn mnist_mlp_manual_backward(c: &mut Criterion) {
    let config = mnist_mlp_config();
    let mut group = c.benchmark_group("mnist_mlp_manual");
    group.throughput(Throughput::Elements(MNIST_BATCH as u64));

    let input = batched_input(MNIST_BATCH);
    let targets = batched_targets(MNIST_BATCH);
    let mut network = Network::new(&config);

    group.bench_function("feed_forward_and_backward_batch32", |b| {
        b.iter(|| {
            let outputs = network.feed_forward(black_box(input.clone()));
            black_box(network.accumulate_gradients(&outputs, &targets));
        });
    });

    group.finish();
}

fn mnist_mlp_manual_train_epoch(c: &mut Criterion) {
    let mut config = mnist_mlp_config();
    config.epochs = neural_network::Epochs::try_from(1).expect("one epoch");
    config.batch_size = neural_network::BatchSize::try_from(MNIST_BATCH).expect("batch size");

    let (inputs, targets) = synthetic_mnist_samples(MNIST_EPOCH_SAMPLES);
    let mut group = c.benchmark_group("mnist_mlp_manual");
    group.throughput(Throughput::Elements(MNIST_EPOCH_SAMPLES as u64));

    group.bench_function("train_one_epoch_512_samples", |b| {
        b.iter(|| {
            let mut network = Network::new(black_box(&config));
            assert_eq!(network.backprop_engine(), BackpropEngine::Manual);
            black_box(network.train(black_box(&inputs), black_box(&targets)));
        });
    });

    group.finish();
}

fn train_xor_network(c: &mut Criterion) {
    let inputs = vec![
        Matrix::from(vec![0.0, 0.0]),
        Matrix::from(vec![0.0, 1.0]),
        Matrix::from(vec![1.0, 0.0]),
        Matrix::from(vec![1.0, 1.0]),
    ];
    let targets = vec![
        Matrix::from(vec![0.0]),
        Matrix::from(vec![1.0]),
        Matrix::from(vec![1.0]),
        Matrix::from(vec![0.0]),
    ];

    let config = NetworkConfig::new(
        vec![
            Layer::new(2, Some(Activation::Sigmoid)),
            Layer::new(4, Some(Activation::Softmax)),
            Layer::new(1, None),
        ],
        0.5,
        Some(0.9),
        100,
        2,
        Some(RegularizationType::L2),
        Some(0.0001),
    )
    .expect("xor config");

    c.bench_function("xor_train_100_epochs", |b| {
        b.iter(|| {
            let mut network = Network::new(black_box(&config));
            black_box(network.train(black_box(&inputs), black_box(&targets)));
        });
    });
}

criterion_group!(
    benches,
    mnist_mlp_manual_forward,
    mnist_mlp_manual_backward,
    mnist_mlp_manual_train_epoch,
    train_xor_network,
);
criterion_main!(benches);
