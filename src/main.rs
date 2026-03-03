mod base;
mod extract_dataset;

use std::{thread, sync::Arc, time::{Duration, Instant}};
use base::*;
use extract_dataset::*;

fn main() -> Result<(), std::io::Error>{
    let train_images_path = "MNIST/train-images.idx3-ubyte".to_string();
    let train_labels_path = "MNIST/train-labels.idx1-ubyte".to_string();
    let test_images_path = "MNIST/t10k-images.idx3-ubyte".to_string();
    let test_labels_path = "MNIST/t10k-labels.idx1-ubyte".to_string();
    
    let train_inputs = mnist_input(train_images_path)?;
    let train_outputs = mnist_output(train_labels_path)?;
    let test_inputs = mnist_input(test_images_path)?;
    let test_outputs = mnist_output(test_labels_path)?;
    
    let train_data = Arc::new(DataSet::new(train_inputs, train_outputs)?);
    let test_data = Arc::new(DataSet::new(test_inputs, test_outputs)?);

    let mut network1 = Network::new(3, vec![784, 5, 10])?;
    // let mut network2 = network1.clone();

    let start1 = Instant::now();
    network1 = network1.multithread_back_propagation(train_data.clone(), 10)?;
    let duration1 = start1.elapsed();
    /*
    let start2 = Instant::now();
    network2 = network2.back_propagation(train_data.as_ref(), 10)?;
    let duration2 = start2.elapsed();
    */

    println!(
        "Время и точность в многопоточном алгоритме: {:?}, {}",
        duration1, network1.test(&test_data)?
    );

    /*
    println!(
        "Время и точность в последовательном алгоритме: {:?}, {}",
        duration2, network2.test(&test_data)?
    );
    */

    Ok(())
}
