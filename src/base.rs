use std::{
    f64::consts::E,
    io::{Error, ErrorKind},
    sync::{Arc, Mutex},
    thread
};
use rand::random_range;

const LEARNING_RATE: f64 = 0.001;
const THREAD_COUNT: i32 = 10;
const RELU_KOEFF: f64 = 0.001;

#[derive(Clone)]
pub struct Network{
    // array of layers
    // every layer consists of array of neurons
    // every neuron is floating point number, which is sum of inputs for this neuron
    layers: Vec<Vec<f64>>,

    // also array of layers
    // every layer consists of array of neurons
    // but every neuron is array of weights of input edges
    weights: Vec<Vec<Vec<f64>>>
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

impl Network{
    // create new network
    // cnt_layers - count of layers in network
    // cnt_neurons - array, which consists counts of neurons in each layer
    pub fn new(cnt_layers: i32, cnt_neurons: Vec<i32>) -> Result<Self, Error>{
        // neural network must have input and output layers
        if cnt_layers < 2{
            return Err(
                Error::new(
                    ErrorKind::Other, 
                    "Neural network must have at least two layers".to_string()
                )
            )
        }

        let mut layers: Vec<Vec<f64>> = Vec::new();
        for i in 0..cnt_layers as usize{
            // push new layer of neurons
            layers.push(vec![0.0; cnt_neurons[i] as usize]);
            // append a bias into layer
            layers[i].insert(0, 1.0);
        }

        let mut weights: Vec<Vec<Vec<f64>>> = Vec::new();

        // neurons of first layer don't have input edges
        for i in 0..(cnt_layers - 1) as usize{
            weights.push(Vec::new());

            for _ in 0..cnt_neurons[i+1] as usize{
                // fill weights for every neuron with random numbers
                let mut curr_neuron: Vec<f64> = Vec::new();
                for _ in 0..cnt_neurons[i] as usize{
                    curr_neuron.push(random_range(0.0..2.0));
                }

                weights[i].push(curr_neuron);
            }
        }

        Ok(Self{
            layers,
            weights
        })
    }

    fn validate_dataset(&self, data: &DataSet) -> bool{
        // dataset inputs and outputs must have correct length
        if self.layers[0].len() - 1 != data.inputs[0].len() ||
            self.layers[self.layers.len() - 1].len() - 1 != data.outputs[0].len(){
            return false;
        }

        true
    }

    fn step_forward(&mut self, input: &Vec<f64>){
        // initialize input layer of network
        for i in 1..self.layers[0].len() {
            self.layers[0][i] = input[i-1];
        }

        // for every neuron
        for layer in 1..self.layers.len(){
            for neuron in 1..self.layers[layer].len(){
                let mut sum = 0.0;

                // calculate sum of inputs for current neuron
                for prev_neuron in 1..self.layers[layer-1].len(){
                    sum += self.layers[layer-1][prev_neuron] *
                        self.weights[layer-1][neuron-1][prev_neuron-1];
                }
                if layer == 1 && neuron == 1{
                    //println!("{}", sum);
                }
                // calculate sigmoid function
                self.layers[layer][neuron] = ReLu(sum);
            }
        }
    }

    fn step_backward(&mut self, output: &Vec<f64>){
        let cnt_layers = self.layers.len();

        // backpropagation for the output layer.
        // from second neuron, because first neuron is bias.
        for neuron in 1..self.layers[cnt_layers-1].len(){
            let activate_value = self.layers[cnt_layers-1][neuron];
            let diff = ReLu_diff(activate_value);
            let error =
                self.layers[cnt_layers-1][neuron] - output[neuron-1];

            // changing the bias
            self.layers[cnt_layers-2][0] += LEARNING_RATE * error * diff;

            // changing the weights of other edges
            for edge in 0..self.weights[cnt_layers-2][neuron-1].len(){
                self.weights[cnt_layers-2][neuron-1][edge] -= LEARNING_RATE *
                    error * activate_value * diff * self.layers[cnt_layers-2][edge+1];
            }
        }

        // backpropagation for other layers
        for layer in (1..(cnt_layers - 1)).rev(){
            for neuron in 1..self.layers[layer].len(){
                let activate_value = self.layers[layer][neuron];
                let diff = ReLu_diff(activate_value);

                let mut error = 0.0;
                for next_neuron in 1..self.layers[layer+1].len(){
                    let next_neuron_value = self.layers[layer+1][next_neuron];
                    error += next_neuron_value * ReLu_diff(next_neuron_value) *
                        self.weights[layer][next_neuron-1][neuron-1];
                }

                self.layers[layer-1][0] -= LEARNING_RATE * error * activate_value * diff;

                for edge in 0..self.weights[layer-1][neuron-1].len(){
                    self.weights[layer-1][neuron-1][edge] -= LEARNING_RATE *
                        error * diff * activate_value * self.layers[layer-1][edge+1];
                }
            }
        }
    }

    fn mean_square_error(&self, output: &Vec<f64>) -> f64{
        let mut accuracy = 0.0;

        for output_neuron in 1..self.layers[self.layers.len()-1].len(){
            accuracy +=
                (
                    self.layers[self.layers.len()-1][output_neuron] -
                        output[output_neuron-1]
                ).powi(2);
        }

        // mean deviation on output layer
        accuracy / output.len() as f64
    }

    pub fn back_propagation(mut self, data: &DataSet, epoch_count: i32) -> Result<Self, Error>{
        if !self.validate_dataset(data){
            return Err(
                Error::new(
                    ErrorKind::Other, 
                    "dataset is invalid".to_string()
                )
            );
        }

        let mut accuracy = 0.0;

        for _ in 0..epoch_count{
            
            // for every point in dataset
            for vector in 0..data.count as usize{
                self.step_forward(&data.inputs[vector]);

                // calculate mean square error for this vector
                accuracy += self.mean_square_error(&data.outputs[vector]);

                self.step_backward(&data.outputs[vector]);
            }
        }

        println!("{}", accuracy / 60000.0);

        Ok(self)
    }

    pub fn multithread_back_propagation(
        mut self, data: Arc<DataSet>, epoch_count: i32
    ) -> Result<Self, Error>{
        if !self.validate_dataset(data.as_ref()){
            return Err(
                Error::new(
                    ErrorKind::Other,
                    "dataset is invalid".to_string()
                )
            );
        }

        for _ in 0..epoch_count{
            println!("{:?}", self.weights[0][0]);
            let mut threads = Vec::new();
            let new_weights: Arc<Mutex<Vec<Vec<Vec<Vec<f64>>>>>> =
                Arc::new(Mutex::new(vec![Vec::new(); THREAD_COUNT as usize]));

            // constant count of threads(need to change)
            for i in 0..THREAD_COUNT {
                let data = Arc::clone(&data);
                let mut network = self.clone();
                let new_weights = Arc::clone(&new_weights);
                let thread = thread::spawn(move || {
                    for j in 0..(data.count / THREAD_COUNT) {
                        network.step_forward(
                            &data.inputs[(j + i * data.count / THREAD_COUNT) as usize]
                        );
                        network.step_backward(
                            &data.outputs[(j + i * data.count / THREAD_COUNT) as usize]
                        );
                    }

                    let mut new_weights = new_weights.lock().unwrap();
                    new_weights[i as usize] = network.get_weights()
                });

                threads.push(thread);
            }

            for thread in threads {
                thread.join().unwrap();
            }

            let new_weights = new_weights.lock().unwrap();

            for layer in 0..self.weights.len() {
                for neuron in 0..self.weights[layer].len() {
                    for weight in 0..self.weights[layer][neuron].len() {
                        let mut sum = 0.0;

                        for i in 0..THREAD_COUNT {
                            sum += new_weights[i as usize][layer][neuron][weight];
                        }

                        self.weights[layer][neuron][weight] = sum / THREAD_COUNT as f64;
                    }
                }
            }
        }

        Ok(self)
    }

    pub fn adaptive_back_propagation(&mut self, data: &DataSet) -> Result<f64, Error>{
        Ok(0.0)
    }

    // this function tests network on a test set and returns the accuracy
    pub fn test(&mut self, data: &DataSet) -> Result<f64, Error>{
        if !self.validate_dataset(&data){
            return Err(
                Error::new(ErrorKind::Other, "dataset is invalid".to_string()
                )
            );
        }
        
        let mut cnt_right_predictions = 0;
        let cnt_layers = self.layers.len();
        
        for vector in 0..data.count as usize{
            self.step_forward(&data.inputs[vector]);

            // Is prediction right?
            let mut max: usize = 1;
            for output_neuron in 1..self.layers[cnt_layers-1].len(){
                if self.layers[cnt_layers-1][output_neuron] > self.layers[cnt_layers-1][max]{
                    max = output_neuron;
                }
                /*// (Yi - Di) ^ 2
                accuracy +=
                    (
                        self.layers[cnt_layers-1][output_neuron] -
                            data.outputs[vector][output_neuron-1]
                    ).powi(2);*/
            }

            if data.outputs[vector][max-1] == 1.0{
                cnt_right_predictions += 1;
            }
        }

        //println!("{}", cnt_right_predictions);
        Ok(cnt_right_predictions as f64 / 60000.0)
    }

    // this function returns the array with weights of all edges
    fn get_weights(self) -> Vec<Vec<Vec<f64>>>{
        self.weights
    }
}

// the sigmoid function: 1 / (1 + e^(-x)),
// where e = 2,71828... - the Euler number,
// x - function argument
pub fn sigmoid(x: f64) -> f64{
    1.0 / (1.0 + E.powf(-x))
}

pub fn ReLu(x: f64) -> f64{
    if x < 0.0{
        x * RELU_KOEFF
    } else {
        x
    }
}

pub fn ReLu_diff(x: f64) -> f64{
    if x < 0.0{
        RELU_KOEFF
    } else {
        1.0
    }
}