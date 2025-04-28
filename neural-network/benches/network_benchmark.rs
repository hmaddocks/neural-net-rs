use criterion::{Criterion, black_box, criterion_group, criterion_main};
use matrix::Matrix;
use neural_network::{ActivationType, Layer, Network, NetworkConfig, RegularizationType};

fn train_xor_network(c: &mut Criterion) {
    // XOR training data
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

    // Test different network configurations
    let configs = vec![
        // Small network
        NetworkConfig::new(
            vec![
                Layer::new(2, Some(ActivationType::Sigmoid)),
                Layer::new(4, Some(ActivationType::Softmax)),
                Layer::new(1, None),
            ],
            0.5,
            Some(0.9),
            100,
            2,
            Some(RegularizationType::L2),
            Some(0.0001),
        )
        .unwrap(),
    ];

    // Benchmark each configuration
    for (i, config) in configs.iter().enumerate() {
        let name = format!("train_xor_{}_layer_network", i + 2);
        c.bench_function(&name, |b| {
            b.iter(|| {
                let mut network = Network::new(black_box(config));
                network.train(black_box(&inputs), black_box(&targets));
            })
        });
    }
}

criterion_group!(benches, train_xor_network);
criterion_main!(benches);
