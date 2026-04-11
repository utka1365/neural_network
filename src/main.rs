mod only_std;
mod with_ndarray;

use std::{sync::Arc, time::Instant};
use only_std::base::*;
use only_std::adaptive::*;
use only_std::classic::*;
use only_std::hybrid::*;
use only_std::extract_dataset::*;
use with_ndarray::hybrid;

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

    let mut network = HybridNetwork::new(vec![784, 800, 10])?;
    let mut i = 1;
    let start = Instant::now();

    loop {
        println!("{i} эпоха:");
        network = mini_batch_back_propagation(network, train_data.as_ref(), 1, 128)?;
        test(&mut network, &test_data)?;
        i += 1;
        println!("{}", start.elapsed().as_secs());
    }

    Ok(())
}