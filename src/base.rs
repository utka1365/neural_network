use std::{
    f64::consts::E,
    io::{Error, ErrorKind},
    sync::{Arc, Mutex},
    thread
};
pub const LEARNING_RATE: f64 = 0.002;
pub const RELU_KOEFF: f64 = 0.01;

// this trait is needed for same work with different types of the networks
pub trait Trainee{
    fn step_forward(&mut self, input: &Vec<f64>);
    fn step_backward(&mut self, output: &Vec<f64>);
    fn get_layers(&self) -> &Vec<Vec<f64>>;
    fn get_mut_layers(&mut self) -> &mut Vec<Vec<f64>>;
    fn borrow_weights(self) -> Vec<Vec<Vec<f64>>>;
    fn get_mut_weights(&mut self) -> &mut Vec<Vec<Vec<f64>>>;
}

pub struct DataSet{
    count: i32,
    inputs: Vec<Vec<f64>>,
    outputs: Vec<Vec<f64>>
}

impl DataSet{
    pub fn new(inputs: Vec<Vec<f64>>, outputs: Vec<Vec<f64>>) -> Result<Self, Error>{
        if inputs.len() != outputs.len(){
            return Err(
                Error::new(
                    ErrorKind::Other,
                    "Count of input vectors must be equal to a count of output vectors".to_string())
            );
        }

        if inputs.len() == 0{
            return Err(
                Error::new(ErrorKind::Other,
                           "Dataset can't be empty".to_string()
                )
            )
        }

        for i in 0..inputs.len(){
            if inputs[i].len() != inputs[0].len() || outputs[i].len() != outputs[0].len(){
                return Err(
                    Error::new(
                        ErrorKind::Other,
                        "All vectors must be the same length".to_string()
                    )
                );
            }
        }

        Ok(Self{
            count: inputs.len() as i32,
            inputs,
            outputs
        })
    }
}

pub fn validate_dataset(network: &impl Trainee, data: &DataSet) -> bool {
    // dataset inputs and outputs must have correct length
    if network.get_layers()[0].len() != data.inputs[0].len() ||
        network.get_layers()[network.get_layers().len() - 1].len() != data.outputs[0].len(){
        return false;
    }

    true
}

fn mean_squared_error(network: &impl Trainee, output: &Vec<f64>) -> f64{
    let mut accuracy = 0.0;
    let layers = network.get_layers();

    for output_neuron in 1..layers[layers.len()-1].len(){
        accuracy += (layers[layers.len()-1][output_neuron] - output[output_neuron-1]).powi(2);
    }

    // mean deviation on output layer
    accuracy / output.len() as f64
}

pub fn multithread_back_propagation<T: Clone + Trainee + Send + 'static>(
    mut network: T, data: Arc<DataSet>, epoch_count: i32
) -> Result<T, Error> {
    if !validate_dataset(&network, data.as_ref()){
        return Err(
            Error::new(
                ErrorKind::Other,
                "dataset is invalid".to_string()
            )
        );
    }

    let thread_count: i32 = thread::available_parallelism()?.get() as i32;
    let mut threads = Vec::new();
    let new_weights: Arc<Mutex<Vec<Vec<Vec<Vec<f64>>>>>> =
        Arc::new(Mutex::new(vec![Vec::new(); thread_count as usize]));

    for i in 0..thread_count {
        let data = Arc::clone(&data);
        let mut copy = network.clone();
        let new_weights = Arc::clone(&new_weights);
        let thread = thread::spawn(move || {
            for _ in 0..epoch_count {
                for j in 0..(data.count / thread_count) {
                    //println!("{:?}", copy.get_layers()[copy.get_layers().len()-1]);
                    copy.step_forward(
                        &data.inputs[(j + i * data.count / thread_count) as usize]
                    );
                    copy.step_backward(
                        &data.outputs[(j + i * data.count / thread_count) as usize]
                    );
                    //println!("{:?}", copy.get_layers()[copy.get_layers().len()-1]);
                }
            }

            let mut new_weights = new_weights.lock().unwrap();
            new_weights[i as usize] = copy.borrow_weights()
        });
        threads.push(thread);
    }

    for thread in threads {
        thread.join().unwrap();
    }

    let new_weights = new_weights.lock().unwrap();
    let prev_weights = network.get_mut_weights();

    for layer in 0..prev_weights.len() {
        for neuron in 0..prev_weights[layer].len() {
            for weight in 0..prev_weights[layer][neuron].len() {
                let mut sum = 0.0;

                for i in 0..thread_count {
                    sum += new_weights[i as usize][layer][neuron][weight];
                }

                prev_weights[layer][neuron][weight] = sum / thread_count as f64;
            }
        }
    }

    Ok(network)
}

// this function tests network on a test set and returns the accuracy
pub fn test(network: &mut impl Trainee, data: &DataSet) -> Result<(), Error>{
    if !validate_dataset(network, &data){
        return Err(
            Error::new(ErrorKind::Other, "dataset is invalid".to_string()
            )
        );
    }
    let mut accuracy = 0.0;
    let cnt_layers = network.get_layers().len();

    for vector in 0..data.count as usize{
        network.step_forward(&data.inputs[vector]);
        let layers = network.get_mut_layers();
        println!("{:?}", layers[layers.len()-1]);
        println!("{:?}", data.outputs[vector]);
        for output_neuron in 0..layers[cnt_layers-1].len(){
            // (Yi - Di) ^ 2
            accuracy +=
                (layers[cnt_layers-1][output_neuron] - data.outputs[vector][output_neuron])
                    .powi(2);
        }
    }
    //println!("{:?}", network.get_mut_weights());
    accuracy /= 100000.0;
    println!("{accuracy}");

    Ok(())
}

// the sigmoid function: 1 / (1 + e^(-x)),
// where e = 2,71828... - the Euler number,
// x - function argument
pub fn sigmoid(x: f64) -> f64{
    //println!("{x}");
    1.0 / (1.0 + E.powf(-x))
}

pub fn sigmoid_diff(x: f64) -> f64{
    x * (1.0 - x)
}

// ReLu function:
// x < 0 => x * k
// x >= 0 => x
pub fn ReLU(x: f64) -> f64{
    if x < 0.0{
        RELU_KOEFF * x
    } else {
        x
    }
}

pub fn ReLU_diff(x: f64) -> f64{
    if x < 0.0{
        RELU_KOEFF
    } else {
        1.0
    }
}