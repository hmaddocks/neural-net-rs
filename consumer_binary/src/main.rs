use std::fs::File;
use std::io::{self, Read};
use std::path::PathBuf;
use matrix::matrix::Matrix;
use neural_network::activations::SIGMOID;
use neural_network::network::Network;
use rand::seq::SliceRandom;
use rand::thread_rng;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};

const IMAGE_MAGIC_NUMBER: u32 = 2051;
const LABEL_MAGIC_NUMBER: u32 = 2049;
const BATCH_SIZE: usize = 100;
const EPOCHS: u32 = 30;
const LEARNING_RATE: f64 = 0.1;

struct MnistData {
    images: Vec<Matrix>,
    labels: Vec<Matrix>,
}

fn read_u32(file: &mut File) -> io::Result<u32> {
    let mut buffer = [0; 4];
    file.read_exact(&mut buffer)?;
    Ok(u32::from_be_bytes(buffer))
}

fn read_mnist_images(path: PathBuf, progress: &ProgressBar) -> io::Result<Vec<Matrix>> {
    let mut file = File::open(&path)?;
    
    // Read header
    let magic_number = read_u32(&mut file)?;
    if magic_number != IMAGE_MAGIC_NUMBER {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid magic number for images file",
        ));
    }

    let num_images = read_u32(&mut file)? as usize;
    let num_rows = read_u32(&mut file)? as usize;
    let num_cols = read_u32(&mut file)? as usize;
    let pixels_per_image = num_rows * num_cols;

    progress.set_length(num_images as u64);
    progress.set_message("Loading images...");

    // Read image data
    let mut images = Vec::with_capacity(num_images);
    let mut buffer = vec![0u8; pixels_per_image];

    for _ in 0..num_images {
        file.read_exact(&mut buffer)?;
        
        // Convert u8 pixels to f64 and normalize to range [0, 1]
        let data: Vec<f64> = buffer
            .iter()
            .map(|&pixel| f64::from(pixel) / 255.0)
            .collect();

        images.push(Matrix::new(pixels_per_image, 1, data));
        progress.inc(1);
    }

    progress.finish_with_message("Images loaded successfully");
    Ok(images)
}

fn read_mnist_labels(path: PathBuf, progress: &ProgressBar) -> io::Result<Vec<Matrix>> {
    let mut file = File::open(&path)?;
    
    // Read header
    let magic_number = read_u32(&mut file)?;
    if magic_number != LABEL_MAGIC_NUMBER {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid magic number for labels file",
        ));
    }

    let num_labels = read_u32(&mut file)? as usize;
    progress.set_length(num_labels as u64);
    progress.set_message("Loading labels...");

    let mut labels = Vec::with_capacity(num_labels);
    let mut buffer = [0u8; 1];

    for _ in 0..num_labels {
        file.read_exact(&mut buffer)?;
        
        // Create one-hot encoded vector for the label
        let mut data = vec![0.0; 10];
        data[buffer[0] as usize] = 1.0;
        
        labels.push(Matrix::new(10, 1, data));
        progress.inc(1);
    }

    progress.finish_with_message("Labels loaded successfully");
    Ok(labels)
}

fn load_mnist_data() -> io::Result<MnistData> {
    let multi_progress = MultiProgress::new();
    let sty = ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
        .unwrap()
        .progress_chars("##-");

    let images_progress = multi_progress.add(ProgressBar::new(0));
    let labels_progress = multi_progress.add(ProgressBar::new(0));
    images_progress.set_style(sty.clone());
    labels_progress.set_style(sty);

    let home = std::env::var("HOME").expect("HOME environment variable not set");
    let images_path = PathBuf::from(&home).join("Documents/NMIST/train-images-idx3-ubyte");
    let labels_path = PathBuf::from(&home).join("Documents/NMIST/train-labels-idx1-ubyte");

    let images = read_mnist_images(images_path, &images_progress)?;
    let labels = read_mnist_labels(labels_path, &labels_progress)?;

    if images.len() != labels.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Number of images does not match number of labels",
        ));
    }

    Ok(MnistData { images, labels })
}

fn train_network(data: &MnistData) -> io::Result<Network> {
    println!("\nInitializing neural network...");
    let mut network = Network::new(
        vec![784, 128, 64, 10],  // Input layer: 784 (28x28), Hidden layers: 128, 64, Output layer: 10 (digits 0-9)
        SIGMOID,
        LEARNING_RATE,
    );

    let multi_progress = MultiProgress::new();
    let epoch_style = ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} Epoch {msg}"
    ).unwrap().progress_chars("##-");
    
    let batch_style = ProgressStyle::with_template(
        "{spinner:.yellow} [{elapsed_precise}] {bar:40.yellow/blue} {pos:>7}/{len:7} Batch {msg}"
    ).unwrap().progress_chars("##-");

    let epoch_progress = multi_progress.add(ProgressBar::new(EPOCHS as u64));
    let batch_progress = multi_progress.add(ProgressBar::new(0));
    epoch_progress.set_style(epoch_style);
    batch_progress.set_style(batch_style);

    println!("\nStarting training with batch size {}", BATCH_SIZE);
    let mut indices: Vec<usize> = (0..data.images.len()).collect();
    let mut rng = thread_rng();

    for epoch in 1..=EPOCHS {
        // Shuffle indices for random batch selection
        indices.shuffle(&mut rng);
        let mut correct = 0;
        let mut total = 0;

        batch_progress.set_length((indices.len() / BATCH_SIZE) as u64);
        batch_progress.set_position(0);
        batch_progress.set_message(format!("in Epoch {}", epoch));

        // Process in batches
        for batch_indices in indices.chunks(BATCH_SIZE) {
            for &idx in batch_indices {
                let output = network.feed_forward(&data.images[idx])
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                network.back_propogate(output.clone(), &data.labels[idx]);

                // Calculate accuracy
                let predicted = output.data().iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();
                let actual = data.labels[idx].data().iter()
                    .enumerate()
                    .find(|(_, &val)| val == 1.0)
                    .map(|(idx, _)| idx)
                    .unwrap();
                if predicted == actual {
                    correct += 1;
                }
                total += 1;
            }
            batch_progress.inc(1);
        }

        let accuracy = (correct as f64 / total as f64) * 100.0;
        epoch_progress.set_message(format!("- Accuracy: {:.2}%", accuracy));
        epoch_progress.inc(1);
    }

    epoch_progress.finish_with_message("Training completed!");
    batch_progress.finish_and_clear();
    Ok(network)
}

fn main() -> io::Result<()> {
    println!("Loading MNIST dataset...");
    let data = load_mnist_data()?;
    println!("\nSuccessfully loaded {} training examples", data.images.len());

    match train_network(&data) {
        Ok(_) => {
            println!("\nTraining completed successfully!");
            Ok(())
        }
        Err(e) => {
            eprintln!("\nError during training: {}", e);
            Err(e)
        }
    }
}
