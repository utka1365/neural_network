use std::{
    f64::consts::E,
    io::{Error, ErrorKind},
    sync::{Arc, Mutex},
    thread
};
use std::ptr::with_exposed_provenance;
use rand::prelude::SliceRandom;
use rand::random_range;

pub const RELU_KOEFF: f64 = 0.01;

// this trait is needed for same work with different types of the networks
pub trait Trainee{
    fn step_forward(&mut self, input: &Vec<f64>);
    fn step_backward(&mut self, output: &Vec<f64>);
    fn mini_batch_step_backward(&mut self, gradients: &mut Vec<Vec<Vec<f64>>>, output: &Vec<f64>);
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

pub fn back_propagation<T: Trainee>(
    mut network: T, data: &DataSet, epoch_count: i32,
) -> Result<T, Error>{
    if !validate_dataset(&network, data){
        return Err(
            Error::new(
                ErrorKind::Other,
                "dataset is invalid".to_string()
            )
        );
    }

    for _ in 0..epoch_count{
        for vector in 0..data.count as usize / 1 {
            network.step_forward(&data.inputs[vector]);
            network.step_backward(&data.outputs[vector]);
        }
    }

    Ok(network)
}

pub fn mini_batch_back_propagation<T: Trainee>(
    mut network: T, data: &DataSet, epoch_count: i32, batch_size: i32
) -> Result<T, Error> {
    if !validate_dataset(&network, data){
        return Err(
            Error::new(
                ErrorKind::Other,
                "dataset is invalid".to_string()
            )
        );
    } else if batch_size <= 0 {
        return Err(
            Error::new(
                ErrorKind::Other,
                "batch size must be greater than 0".to_string()
            )
        );
    }

    let mut rng = rand::rng();
    let mut indices = (0..data.count as usize).collect::<Vec<usize>>();

    for _ in 0..epoch_count{
        indices.shuffle(&mut rng);

        for i in 0..(data.count / batch_size) as usize{
            let mut gradients: Vec<Vec<Vec<f64>>> = Vec::new();
            let weights = network.get_mut_weights();

            for layer in 0..weights.len() {
                let mut curr_neuron = Vec::new();
                for neuron in 0..weights[layer].len() {
                    curr_neuron.push(vec![0.0; weights[layer][neuron].len()])
                }
                gradients.push(curr_neuron);
            }

            for point in 0..batch_size as usize{
                network.step_forward(&data.inputs[i*batch_size as usize + point]);
                network.mini_batch_step_backward(
                    &mut gradients, &data.outputs[i*batch_size as usize + point]
                );
            }

            let weights = network.get_mut_weights();
            for layer in 0..weights.len() {
                for neuron in 0..weights[layer].len() {
                    for weight in 0..weights[layer][neuron].len() {
                        weights[layer][neuron][weight] +=
                            gradients[layer][neuron][weight] / batch_size as f64;
                    }
                }
            }
        }
    }

    Ok(network)
}

pub fn multithread_mini_batch_back_propagation<T: Clone + Trainee + Send + 'static>(
    mut network: T, data: Arc<DataSet>, epoch_count: i32, batch_size: i32
) -> Result<T, Error>{
    if !validate_dataset(&network, data.as_ref()){
        return Err(
            Error::new(
                ErrorKind::Other,
                "dataset is invalid".to_string()
            )
        );
    } else if batch_size <= 0 {
        return Err(
            Error::new(
                ErrorKind::Other,
                "batch size must be greater than 0".to_string()
            )
        );
    }

    let weights = network.get_mut_weights();
    let mut gradients: Vec<Vec<Vec<f64>>> = Vec::new();

    for layer in 0..weights.len() {
        let mut curr_layer = Vec::new();
        for neuron in 0..weights[layer].len() {
            curr_layer
                .push(vec![0.0; weights[layer][neuron].len()])
        }
        gradients.push(curr_layer);

    }

    let gradients: Arc<Mutex<Vec<Vec<Vec<f64>>>>> = Arc::new(Mutex::new(gradients));
    let network = Arc::new(Mutex::new(network));
    let mut rng = rand::rng();
    let mut indices = (0..data.count as usize).collect::<Vec<usize>>();
    let thread_count: i32 = thread::available_parallelism()?.get() as i32;
    let batches_count = data.count / batch_size / thread_count;

    for _ in 0..epoch_count {
        indices.shuffle(&mut rng);
        let mut batches: Vec<Vec<Vec<usize>>> = Vec::new();

        for thread in 0..thread_count {
            let mut thread_batches: Vec<Vec<usize>> = Vec::new();

            for batch in 0..batches_count {
                let mut batch_indices: Vec<usize> = Vec::new();

                for point in 0..batch_size {
                    batch_indices
                        .push(indices[(thread*batch*batch_size + point) as usize]);
                }

                thread_batches.push(batch_indices);
            }

            batches.push(thread_batches);
        }

        let batches = Arc::new(batches);

        for batch in 0..batches_count as usize {
            let mut threads = Vec::new();

            for thread in 0..thread_count as usize {
                let network = Arc::clone(&network);
                let gradients = Arc::clone(&gradients);
                let data = Arc::clone(&data);
                let batches = Arc::clone(&batches);
                let mut copy = network.as_ref().lock().unwrap().clone();
                let thread = thread::spawn(move || {
                    let mut local_gradients: Vec<Vec<Vec<f64>>> = Vec::new();
                    let weights = copy.get_mut_weights();

                    for layer in 0..weights.len() {
                        let mut curr_layer = Vec::new();
                        for neuron in 0..weights[layer].len() {
                            curr_layer
                                .push(vec![0.0; weights[layer][neuron].len()])
                        }
                        local_gradients.push(curr_layer);
                    }

                    for point in 0..batch_size as usize {
                        copy.step_forward(&data.inputs[batches[thread][batch][point]]);
                        copy.mini_batch_step_backward(
                            &mut local_gradients, &data.outputs[batches[thread][batch][point]],
                        );
                    }

                    let mut gradients = gradients.as_ref().lock().unwrap();

                    for layer in 0..gradients.len() {
                        for neuron in 0..gradients[layer].len() {
                            for weight in 0..gradients[layer][neuron].len() {
                                gradients[layer][neuron][weight] +=
                                    local_gradients[layer][neuron][weight];
                                local_gradients[layer][neuron][weight] = 0.0;
                            }
                        }
                    }
                });
                threads.push(thread);
            }


            for thread in threads {
                thread.join().unwrap();
            }

            let mut gradients = gradients.as_ref().lock().unwrap();
            let mut network = network.as_ref().lock().unwrap();
            let weights = network.get_mut_weights();

            for layer in 0..gradients.len() {
                for neuron in 0..gradients[layer].len() {
                    for weight in 0..gradients[layer][neuron].len() {
                        weights[layer][neuron][weight] +=
                            gradients[layer][neuron][weight] / (batch_size * thread_count) as f64;
                        gradients[layer][neuron][weight] = 0.0;
                    }
                }
            }
        }
    }


    Ok(
        Arc::try_unwrap(network)
        .map_err(|_| "Error unwrapping Arc(network)")
        .unwrap()
        .into_inner()
        .unwrap()
    )
}

pub fn multithread_back_propagation<T: Clone + Trainee + Send + 'static>(
    mut network: T, data: Arc<DataSet>, epoch_count: i32,
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
                    copy.step_forward(
                        &data.inputs[(j + i * data.count / thread_count) as usize]
                    );
                    copy.step_backward(
                        &data.outputs[(j + i * data.count / thread_count) as usize]
                    );
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
pub fn test(network: &mut impl Trainee, data: &DataSet) -> Result<f64, Error>{
    if !validate_dataset(network, data){
        return Err(
            Error::new(ErrorKind::Other, "dataset is invalid".to_string()
            )
        );
    }

    let mut accuracy = 0.0;
    let mut cnt = 0;
    let cnt_layers = network.get_layers().len();
    let thread_count = thread::available_parallelism()?.get() as i32;

    for vector in 0..data.count as usize{
        network.step_forward(&data.inputs[vector]);
        let layers = network.get_layers();
        let mut max = 0;
        let mut label = 0;
        println!("{:?}", layers[cnt_layers-1]);
        println!("{:?}", data.outputs[vector]);
        for output_neuron in 0..layers[cnt_layers-1].len(){
            if data.outputs[vector][output_neuron] == 1.0{
                label = output_neuron;
            }
            // (Yi - Di) ^ 2
            accuracy +=
                (layers[cnt_layers-1][output_neuron] - data.outputs[vector][output_neuron])
                    .powi(2);

            if layers[cnt_layers-1][output_neuron] > layers[cnt_layers-1][max]{
                max = output_neuron;
            }
        }

        if max == label {
            cnt += 1;
        }
    }
    accuracy /= 100000.0;

    println!("Среднеквадратичная ошибка: {accuracy}");
    println!("Точность: {}", cnt as f64 / 10000.0);

    Ok(accuracy)
}

pub fn multithread_test<T: Clone + Trainee + Send + 'static>(network: &T, data: Arc<DataSet>) -> Result<f64, Error>{
    if !validate_dataset(network, data.as_ref()){
        return Err(
            Error::new(ErrorKind::Other, "dataset is invalid".to_string()
            )
        );
    }

    let thread_count = thread::available_parallelism()?.get() as i32;
    let mut threads = Vec::new();
    let accuracy = Arc::new(Mutex::new(0.0));
    let cnt = Arc::new(Mutex::new(0));
    let cnt_layers = network.get_layers().len();

    for i in 0..thread_count {
        let data = Arc::clone(&data);
        let accuracy = Arc::clone(&accuracy);
        let cnt = Arc::clone(&cnt);
        let mut copy = network.clone();
        let mut local_accuracy = 0.0;
        let mut local_cnt = 0;
        let thread = thread::spawn(move || {
            for vector in 0..data.count / thread_count {
                copy.step_forward(&data.inputs[(i*thread_count+vector) as usize]);
                let layers = copy.get_layers();
                let mut max = 0;
                let mut label = 0;

                for output_neuron in 0..layers[cnt_layers-1].len(){
                    if data.outputs[vector as usize][output_neuron] == 1.0{
                        label = output_neuron;
                    }

                    local_accuracy += (layers[cnt_layers-1][output_neuron] -
                        data.outputs[(i*thread_count+vector) as usize][output_neuron]).powi(2);

                    if layers[cnt_layers-1][output_neuron] > layers[cnt_layers-1][max]{
                        max = output_neuron;
                    }
                }

                if max == label {
                    local_cnt += 1;
                }
            }

            let mut accuracy = accuracy.as_ref().lock().unwrap();
            let mut cnt = cnt.as_ref().lock().unwrap();
            *accuracy += local_accuracy;
            *cnt += local_cnt
        });
        threads.push(thread);
    }

    for thread in threads {
        thread.join().unwrap();
    }

    println!("{}", cnt.lock().unwrap());
    println!("{}", *accuracy.lock().unwrap() / 600000.0);

    Ok(*accuracy.lock().unwrap())
}

// the sigmoid function: 1 / (1 + e^(-x)),
// where e = 2,71828... - the Euler number,
// x - function argument
pub fn sigmoid(x: f64) -> f64{
    if x >= 10.0{ 0.999 }
    else if x <= -10.0 { 0.001 }
    else { 1.0 / (1.0 + E.powf(-x)) }
}

// y = sigmoid(x)
pub fn sigmoid_diff(y: f64) -> f64{
    y * (1.0 - y)
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