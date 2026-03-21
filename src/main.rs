mod base;
mod adaptive;
mod classic;
mod extract_dataset;

use std::{sync::Arc, time::Instant};
use base::*;
use adaptive::*;
use classic::*;
use extract_dataset::*;

fn main() -> Result<(), std::io::Error>{
    // don't touch
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
    // don't touch
    
    let mut network = AdaptiveNetwork::new(vec![784, 5, 10])?;
    let start = Instant::now();
    test(&mut network, &test_data)?;
    network = multithread_back_propagation(network, train_data.clone(), 10)?;
    test(&mut network, &test_data)?;
    println!("{:?}", start.elapsed());

    Ok(())
}
