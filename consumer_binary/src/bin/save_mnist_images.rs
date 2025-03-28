use image::{ImageBuffer, Luma};
use indicatif::{ProgressBar, ProgressStyle};
use mnist::mnist::{load_test_data, load_training_data};

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
    // Process training data
    println!("Processing training data...");
    let progress_bar = ProgressBar::new(5);
    progress_bar.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:80.cyan/blue}] {pos}/{len} ({percent}%)")
            .unwrap()
            .progress_chars("#>-")
    );
    let training_data = match load_training_data() {
        Ok(images) => images,
        Err(e) => {
            eprintln!("Failed to read training images: {}", e);
            return Err(e.into());
        }
    };
    let train_images = training_data.images();
    let train_labels = training_data.labels();

    // Save first 5 training images
    for i in 0..5 {
        let label = train_labels[i]
            .data
            .iter()
            .position(|&x| x > 0.9)
            .unwrap_or(0);

        match save_image(train_images[i].data.as_slice().unwrap(), i, "train", label) {
            Ok(()) => println!("Training image {} label: {}", i, label),
            Err(e) => {
                eprintln!("Failed to save training image {}: {}", i, e);
                return Err(e.into());
            }
        }
    }

    // Process test data
    println!("\nProcessing test data...");
    let test_data = match load_test_data() {
        Ok(images) => images,
        Err(e) => {
            eprintln!("Failed to read test images: {}", e);
            return Err(e.into());
        }
    };
    let test_images = test_data.images();
    let test_labels = test_data.labels();

    // Save first 5 test images
    for i in 0..5 {
        let label = test_labels[i]
            .data
            .iter()
            .position(|&x| x > 0.9)
            .unwrap_or(0);
        match save_image(test_images[i].data.as_slice().unwrap(), i, "test", label) {
            Ok(()) => println!("Test image {} label: {}", i, label),
            Err(e) => {
                eprintln!("Failed to save test image {}: {}", i, e);
                return Err(e.into());
            }
        }
    }

    println!("\nImages have been saved as PNG files in the current directory.");
    Ok(())
}
