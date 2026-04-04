use std::{
    io::{Error, ErrorKind},
};
use rand::random_range;
use crate::base::*;

const LN_VALUE: f64 = 6.90675;
const KOEFF: f64 = 1.0;
const LAMBDA: f64 = 0.5;
pub const LEARNING_RATE: f64 = 1.0;

#[derive(Clone)]
pub struct HybridNetwork{
    layers: Vec<Vec<f64>>,
    weights: Vec<Vec<Vec<f64>>>,
    // deltas of every neuron
    deltas: Vec<Vec<f64>>,
    // inputs of every neuron
    sums: Vec<Vec<f64>>
}

impl HybridNetwork{
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
        for i in 0..cnt_layers-1 {
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
    pub fn print_values(&self){
        println!("{:?}", self.layers[self.layers.len()-1]);
    }
}

impl Trainee for HybridNetwork{
    fn step_forward(&mut self, input: &Vec<f64>){
        // initialize input layer of network
        for i in 0..self.layers[0].len() {
            self.layers[0][i] = input[i];
        }

        let cnt_layers = self.layers.len();

        // for every neuron
        for layer in 1..cnt_layers-1{
            for neuron in 0..self.layers[layer].len(){
                let mut sum = self.weights[layer-1][neuron][0];
                // calculate sum of inputs for current neuron
                for prev_neuron in 0..self.layers[layer-1].len(){
                    sum += self.layers[layer-1][prev_neuron] *
                        self.weights[layer-1][neuron][prev_neuron+1];
                }

                self.sums[layer][neuron] = sum;
                // calculate sigmoid function
                self.layers[layer][neuron] = sigmoid(sum);
            }
        }

        for neuron in 0..self.layers[cnt_layers-1].len(){
            let mut sum = self.weights[cnt_layers-2][neuron][0];

            for prev_neuron in 0..self.layers[cnt_layers-2].len(){
                sum += self.layers[cnt_layers-2][prev_neuron] *
                    self.weights[cnt_layers-2][neuron][prev_neuron+1];
            }

            self.sums[cnt_layers-1][neuron] = sum;
            self.layers[cnt_layers-1][neuron] = sigmoid(sum);
        }
    }

    fn step_backward(&mut self, output: &Vec<f64>) {
        let cnt_layers = self.layers.len();
        // backpropagation for the output layer
        let mut denom = 1.0;

        for prev_neuron in 0..self.layers[cnt_layers - 2].len() {
            denom += self.layers[cnt_layers - 2][prev_neuron].powi(2);
        }

        for neuron in 0..self.layers[cnt_layers - 1].len() {
            let error = self.layers[cnt_layers - 1][neuron] - output[neuron];
            let delta: f64;
            self.deltas[cnt_layers - 1][neuron] = error;

            if output[neuron] == 1.0 {
                delta = (self.sums[cnt_layers - 1][neuron] - LN_VALUE) / denom;
            } else {
                delta = (self.sums[cnt_layers - 1][neuron] + LN_VALUE) / denom;
            }
            // changing the bias
            self.weights[cnt_layers - 2][neuron][0] += KOEFF * delta;

            // changing the weights of other edges
            for edge in 1..self.weights[cnt_layers - 2][neuron].len() {
                self.weights[cnt_layers - 2][neuron][edge] -=
                    KOEFF * delta * self.layers[cnt_layers - 2][edge - 1];
            }
        }

        // backpropagation for other layers
        for layer in (1..(cnt_layers - 1)).rev() {
            for neuron in 0..self.layers[layer].len() {
                let activate_value = self.layers[layer][neuron];
                let diff = sigmoid_diff(activate_value);
                let mut error = 0.0;

                for next_neuron in 0..self.layers[layer + 1].len() {
                    error += self.deltas[layer + 1][next_neuron] *
                        self.weights[layer][next_neuron][neuron + 1];
                }

                error *= diff;
                self.deltas[layer][neuron] = error;
                self.weights[layer - 1][neuron][0] -= LEARNING_RATE *
                    (error + 2.0 * LAMBDA * self.weights[layer - 1][neuron][0]);

                for edge in 1..self.weights[layer - 1][neuron].len() {
                    self.weights[layer - 1][neuron][edge] -= LEARNING_RATE *
                        (error + 2.0 * LAMBDA * self.weights[layer - 1][neuron][edge]) *
                        self.layers[layer - 1][edge - 1];
                }
            }
        }
    }


    fn mini_batch_step_backward(&mut self, gradients: &mut Vec<Vec<Vec<f64>>>, output: &Vec<f64>) {
        // the array must have the same size as the weights array
        let cnt_layers = self.layers.len();
        let mut denom = 1.0;

        for prev_neuron in 0..self.layers[cnt_layers-2].len(){
            denom += self.layers[cnt_layers-2][prev_neuron].powi(2);
        }

        for neuron in 0..self.layers[cnt_layers-1].len(){
            let error = self.layers[cnt_layers-1][neuron] - output[neuron];
            let delta: f64;
            self.deltas[cnt_layers-1][neuron] = error;

            if output[neuron] == 1.0{
                delta = (self.sums[cnt_layers-1][neuron] - LN_VALUE) / denom;
            } else {
                delta = (self.sums[cnt_layers-1][neuron] + LN_VALUE) / denom;
            }
            gradients[cnt_layers-2][neuron][0] += KOEFF * delta;

            for edge in 1..self.weights[cnt_layers-2][neuron].len(){
                gradients[cnt_layers-2][neuron][edge] -=
                    KOEFF * delta * self.layers[cnt_layers-2][edge-1];
            }
        }

        for layer in (1..(cnt_layers - 1)).rev(){
            for neuron in 0..self.layers[layer].len(){
                let activate_value = self.layers[layer][neuron];
                let diff = sigmoid_diff(activate_value);
                let mut error = 0.0;

                for next_neuron in 0..self.layers[layer+1].len(){
                    error += self.deltas[layer+1][next_neuron] *
                        self.weights[layer][next_neuron][neuron+1];
                }

                error *= diff;
                self.deltas[layer][neuron] = error;
                gradients[layer-1][neuron][0] += LEARNING_RATE * error;

                for edge in 1..self.weights[layer-1][neuron].len(){
                    gradients[layer-1][neuron][edge] -= LEARNING_RATE *
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