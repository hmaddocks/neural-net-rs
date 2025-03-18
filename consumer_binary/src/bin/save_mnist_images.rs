use image::{ImageBuffer, Luma};
use indicatif::{ProgressBar, ProgressStyle};
use mnist::mnist::{read_mnist_images, read_mnist_labels};
use std::path::PathBuf;

fn save_image(
    image_data: &[f64],
    index: usize,
    prefix: &str,
    label: usize,
) -> Result<(), image::ImageError> {
    let img = ImageBuffer::from_fn(28, 28, |x, y| {
        let pixel = image_data[y as usize * 28 + x as usize];
        Luma([(pixel * 255.0) as u8])
    });
    img.save(format!("{}_{}_image_{}.png", prefix, label, index))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var("HOME")?;
    let mnist_dir = PathBuf::from(home).join("Documents").join("NMIST");

    // Process training data
    println!("Processing training data...");
    let progress_bar = ProgressBar::new(5);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:80.cyan/blue}] {pos}/{len} ({percent}%)")
            .unwrap()
            .progress_chars("#>-")
    );
    let train_images = read_mnist_images(mnist_dir.join("train-images-idx3-ubyte"), &progress_bar)?;
    let train_labels = read_mnist_labels(mnist_dir.join("train-labels-idx1-ubyte"), &progress_bar)?;

    // Save first 5 training images
    for i in 0..5 {
        let label = train_labels[i]
            .data
            .iter()
            .position(|&x| x > 0.9)
            .unwrap_or(0);
        save_image(&train_images[i].data, i, "train", label)?;
        println!("Training image {} label: {}", i, label);
    }

    // Process test data
    println!("\nProcessing test data...");
    let test_images = read_mnist_images(mnist_dir.join("t10k-images-idx3-ubyte"), &progress_bar)?;
    let test_labels = read_mnist_labels(mnist_dir.join("t10k-labels-idx1-ubyte"), &progress_bar)?;

    // Save first 5 test images
    for i in 0..5 {
        let label = test_labels[i]
            .data
            .iter()
            .position(|&x| x > 0.9)
            .unwrap_or(0);
        save_image(&test_images[i].data, i, "test", label)?;
        println!("Test image {} label: {}", i, label);
    }

    println!("\nImages have been saved as PNG files in the current directory.");
    Ok(())
}
