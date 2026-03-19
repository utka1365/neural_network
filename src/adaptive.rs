use std::{
    io::{Error, ErrorKind},
};
use rand::random_range;
use crate::base::*;

#[derive(Clone)]
pub struct AdaptiveNetwork{
    layers: Vec<Vec<f64>>,
    weights: Vec<Vec<Vec<f64>>>,
    // deltas of every neuron
    deltas: Vec<Vec<f64>>,
    // inputs of every neuron
    sums: Vec<Vec<f64>>
}

impl AdaptiveNetwork{
    pub fn new(cnt_neurons: Vec<i32>) -> Result<Self, Error>{
        let cnt_layers = cnt_neurons.len();

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
        let mut weights: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut deltas: Vec<Vec<f64>> = Vec::new();
        let mut sums: Vec<Vec<f64>> = Vec::new();

        for i in 0..cnt_layers {
            // push new layer of neurons
            layers.push(vec![0.0; cnt_neurons[i] as usize]);
            deltas.push(vec![0.0; cnt_neurons[i] as usize]);
            sums.push(vec![0.0; cnt_neurons[i] as usize]);
        }

        // cnt_layers - 1, because neurons of first layer don't have input edges
        for i in 0..cnt_layers - 1 {
            weights.push(Vec::new());

            for _ in 0..cnt_neurons[i+1] as usize{
                // fill weights for every neuron with random numbers
                let mut curr_neuron: Vec<f64> = Vec::new();
                // append a bias into layer
                curr_neuron.push(0.0);
                for _ in 0..cnt_neurons[i] as usize{
                    curr_neuron.push(random_range(0.0..0.1));
                }

                weights[i].push(curr_neuron);
            }
        }

        Ok(Self{
            layers,
            weights,
            deltas,
            sums
        })
    }

    // debug
    fn print_values(&self){
        println!("{:?}", self.layers[self.layers.len()-1]);
    }
}

impl Trainee for AdaptiveNetwork{
    fn step_forward(&mut self, input: &Vec<f64>){
        // initialize input layer of network
        for i in 0..self.layers[0].len() {
            self.layers[0][i] = input[i];
        }

        // for every neuron
        for layer in 1..self.layers.len(){
            for neuron in 0..self.layers[layer].len(){
                let mut sum = self.weights[layer-1][neuron][0];

                // calculate sum of inputs for current neuron
                for prev_neuron in 0..self.layers[layer-1].len(){
                    sum += self.layers[layer-1][prev_neuron] *
                        self.weights[layer-1][neuron][prev_neuron+1];
                }
                // calculate sigmoid function
                self.layers[layer][neuron] = ReLU(sum);
            }
        }
    }

    fn step_backward(&mut self, output: &Vec<f64>){
        let cnt_layers = self.layers.len();

        // backpropagation for the output layer
        for neuron in 0..self.layers[cnt_layers-1].len(){
            let activate_value = self.layers[cnt_layers-1][neuron];
            let diff = ReLU_diff(activate_value);
            let error = (self.layers[cnt_layers-1][neuron] - output[neuron]) * diff;
            self.layers[cnt_layers-1][neuron] = error;
            // changing the bias
            self.weights[cnt_layers-2][neuron][0] -= LEARNING_RATE * error;

            // changing the weights of other edges
            for edge in 1..self.weights[cnt_layers-2][neuron].len(){
                self.weights[cnt_layers-2][neuron][edge] -= LEARNING_RATE *
                    error * self.layers[cnt_layers-2][edge-1];
            }
        }

        // backpropagation for other layers
        for layer in (1..(cnt_layers - 1)).rev(){
            for neuron in 0..self.layers[layer].len(){
                let activate_value = self.layers[layer][neuron];
                let diff = ReLU_diff(activate_value);
                let mut error = 0.0;

                for next_neuron in 0..self.layers[layer+1].len(){
                    error += self.layers[layer+1][next_neuron] *
                        self.weights[layer][next_neuron][neuron+1];
                }

                error *= diff;
                self.layers[layer][neuron] = error;
                self.weights[layer-1][neuron][0] -= LEARNING_RATE * error;

                for edge in 1..self.weights[layer-1][neuron].len(){
                    self.weights[layer-1][neuron][edge] -= LEARNING_RATE *
                        error * self.layers[layer-1][edge-1];
                }
            }
        }
    }

    fn get_layers(&self) -> &Vec<Vec<f64>> {
        &self.layers
    }

    fn get_mut_layers(&mut self) -> &mut Vec<Vec<f64>>{
        &mut self.layers
    }

    fn borrow_weights(self) -> Vec<Vec<Vec<f64>>>{
        self.weights
    }

    fn get_mut_weights(&mut self) -> &mut Vec<Vec<Vec<f64>>>{
        &mut self.weights
    }
}