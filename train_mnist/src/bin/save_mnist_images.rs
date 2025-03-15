use image::{ImageBuffer, Luma};
use indicatif::ProgressBar;
use std::path::PathBuf;
use train_mnist::mnist::{read_mnist_images, read_mnist_labels};

fn save_image(image_data: &[f64], index: usize, prefix: &str) -> Result<(), image::ImageError> {
    let img = ImageBuffer::from_fn(28, 28, |x, y| {
        let pixel = image_data[y as usize * 28 + x as usize];
        Luma([(pixel * 255.0) as u8])
    });
    img.save(format!("{}_image_{}.png", prefix, index))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var("HOME")?;
    let mnist_dir = PathBuf::from(home).join("Documents").join("NMIST");

    // Process training data
    println!("Processing training data...");
    let progress = ProgressBar::new(0);
    let train_images = read_mnist_images(mnist_dir.join("train-images-idx3-ubyte"), &progress)?;
    let train_labels = read_mnist_labels(mnist_dir.join("train-labels-idx1-ubyte"), &progress)?;

    // Save first 5 training images
    for i in 0..5 {
        save_image(train_images[i].data(), i, "train")?;
        println!(
            "Training image {} label: {}",
            i,
            train_labels[i]
                .data()
                .iter()
                .position(|&x| x > 0.9)
                .unwrap_or(0)
        );
    }

    // Process test data
    println!("\nProcessing test data...");
    let test_images = read_mnist_images(mnist_dir.join("t10k-images-idx3-ubyte"), &progress)?;
    let test_labels = read_mnist_labels(mnist_dir.join("t10k-labels-idx1-ubyte"), &progress)?;

    // Save first 5 test images
    for i in 0..5 {
        save_image(test_images[i].data(), i, "test")?;
        println!(
            "Test image {} label: {}",
            i,
            test_labels[i]
                .data()
                .iter()
                .position(|&x| x > 0.9)
                .unwrap_or(0)
        );
    }

    println!("\nImages have been saved as PNG files in the current directory.");
    Ok(())
}
