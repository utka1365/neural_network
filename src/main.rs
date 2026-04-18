mod only_std;
mod with_ndarray;

use std::{sync::Arc, time::Instant};
use only_std::extract_dataset::*;
use crate::with_ndarray::hybrid::Activation::{ReLU, Sigmoid};

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
    
    // let train_data = Arc::new(only_std::base::DataSet::new(train_inputs, train_outputs)?);
    // let test_data = Arc::new(only_std::base::DataSet::new(test_inputs, test_outputs)?);
    let train_data = Arc::new(with_ndarray::hybrid::DataSet::new(train_inputs, train_outputs)?);
    let test_data = Arc::new(with_ndarray::hybrid::DataSet::new(test_inputs, test_outputs)?);
    // don't touch

    /*let mut network = hybrid::HybridNetwork::new(
        vec![784, 800, 10]
    )?;*/

    let mut network = with_ndarray::hybrid::Network::new(vec![784, 128, 128, 128, 10], vec![Sigmoid, Sigmoid, Sigmoid, Sigmoid])?;
    let mut i = 1;
    let start = Instant::now();

    loop {
        println!("{i} эпоха:");
        network.train_adam(train_data.as_ref(), 1, 10);
        network.test(test_data.as_ref());
        /*network = base::mini_batch_back_propagation(network, train_data.as_ref(), 1, 128)?;
        base::test(&mut network, test_data.as_ref())?;*/
        i += 1;
        println!("{}", start.elapsed().as_secs());
    }

    Ok(())
}