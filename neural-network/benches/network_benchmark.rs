use criterion::{black_box, criterion_group, criterion_main, Criterion};
use matrix::matrix::Matrix;
use neural_network::{
    activations::ActivationType, layer::Layer, network::Network, network_config::NetworkConfig,
};

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
                Layer {
                    nodes: 2,
                    activation: Some(ActivationType::Sigmoid),
                },
                Layer {
                    nodes: 4,
                    activation: Some(ActivationType::Softmax),
                },
                Layer {
                    nodes: 1,
                    activation: None,
                },
            ],
            0.5,
            0.9,
            100,
            2,
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
