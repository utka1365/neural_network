use std::{
    io::{Error, ErrorKind},
};
use std::f64::consts::E;
use ndarray::{Array1, Array2};
use ndarray::parallel::prelude::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use rand::prelude::SliceRandom;
use rand::random_range;

pub const LEARNING_RATE: f64 = 0.01;
pub const RELU_KOEFF: f64 = 0.01;
pub const LN_VALUE: f64 = 6.90675;

#[derive(Clone)]
pub enum Activation {
    Sigmoid,
    ReLU
}

pub struct DataSet {
    size: i32,
    inputs: Vec<Array1<f64>>,
    outputs: Vec<Array1<f64>>,
}

pub struct Layer {
    output: Array1<f64>,
    sums: Array1<f64>,
    gammas: Array1<f64>,
    weights: Array2<f64>,
    bias: Array1<f64>,
    activation: Activation,
}

impl Layer {
    pub fn new(
        cnt_neurons: i32, prev_cnt_neurons: i32, activation: Activation
    ) -> Result<Self, Error>{
        let output = Array1::<f64>::zeros(cnt_neurons as usize);
        let sums = Array1::<f64>::zeros(cnt_neurons as usize);
        let gammas = Array1::<f64>::zeros(cnt_neurons as usize);
        let mut weights =
            Array2::<f64>::zeros((cnt_neurons as usize, prev_cnt_neurons as usize));
        let mut bias = Array1::<f64>::zeros(cnt_neurons as usize);

        for i in 0..cnt_neurons as usize {
            for j in 0..prev_cnt_neurons as usize {
                weights[[i, j]] = random_range(0.0..1.0);
            }

            bias[i] = random_range(0.0..1.0);
        }

        Ok(Self{output, sums, gammas, weights, bias, activation})
    }

    fn forward(&mut self, input: &Array1<f64>) {
        self.sums = input.dot(&self.weights) - &self.bias;
        self.output = self.sums.clone();
        self.output.par_map_inplace(|x| *x = match self.activation {
            Activation::Sigmoid => sigmoid(*x),
            Activation::ReLU => ReLU(*x)
        }
        );
    }

    fn backward(&mut self, next_layer: &Self) -> Array1<f64> {
        self.gammas = Array1::from_vec(
            self
                .output
                .par_iter_mut()
                .map(|x| match self.activation {
                    Activation::Sigmoid => sigmoid_diff(*x),
                    Activation::ReLU => ReLU_diff(*x)
                })
                .collect::<Vec<f64>>()
        );

        let deltas = LEARNING_RATE * &self.gammas *
            &next_layer.gammas.dot(&next_layer.weights);
        self.gammas = &self.gammas * &self.output;

        deltas
    }

    fn output_backward(&mut self, input: &Array1<f64>, output: &Array1<f64>) -> Array1<f64> {
        let denom = 1.0 + input
            .into_par_iter()
            .map(|x| x.powi(2))
            .sum::<f64>();
        let deltas = (&self.sums - LN_VALUE) / denom;
        self.gammas = &self.output - output;
        self.gammas = &self.gammas * &self.output;

        deltas
    }
}

pub struct Network {
    layers: Vec<Layer>
}

impl Network {
    pub fn new(cnt_neurons: Vec<i32>, activation_types: Vec<Activation>) -> Result<Self, Error> {
        if cnt_neurons.len() - 1 != activation_types.len() {
            return Err(Error::new(
                    ErrorKind::Other,
                    "The number of layers must be equal to the number of activation types."
            ));
        } else if cnt_neurons.len() < 2 {
            return Err(Error::new(ErrorKind::Other, "network must have at least two layers"));
        }

        let mut layers = Vec::new();

        for i in 0..cnt_neurons.len() - 1 {
            layers.push(Layer::new(
                cnt_neurons[i+1],
                cnt_neurons[i],
                activation_types[i].clone()

            )?);
        }

        Ok(Self{
            layers
        })
    }

    pub fn train(mut self, data: &DataSet, epoch_count: i32, batch_size: usize) {
        let mut rng = rand::rng();
        let mut indices = (0..data.size as usize).collect::<Vec<usize>>();
        let cnt_layers = self.layers.len();

        for _ in 0..epoch_count {
            indices.shuffle(&mut rng);

            for batch in 0..data.size as usize / batch_size {
                let mut deltas: Vec<Array1<f64>> = Vec::new();

                for layer in 0..cnt_layers {
                    deltas.push(Array1::<f64>::zeros(self.layers[layer].output.len()));
                }

                for point in 0..batch_size as usize {
                    self.layers[0].forward(&data.inputs[indices[batch * batch_size + point]]);

                    for layer in 1..cnt_layers {
                        let (left, right) = self.layers.split_at_mut(layer);
                        right[0].forward(&left[layer-1].output);
                    }

                    let penultimate_layer = self.layers[cnt_layers-2].output.clone();
                    deltas[cnt_layers-1] = deltas[cnt_layers-1].clone() + self.layers[cnt_layers-1]
                        .output_backward(
                            &penultimate_layer,
                            &data.outputs[batch * batch_size + point]
                        );

                    for layer in (0..cnt_layers - 1).rev() {
                        let (left, right) = self.layers.split_at_mut(layer+1);
                        deltas[] left[layer].backward(&right[0]);
                    }
                }
            }
        }
    }
}

pub fn sigmoid(x: f64) -> f64{
    if x >= 10.0{ 0.999 }
    else if x <= -10.0 { 0.001 }
    else { 1.0 / (1.0 + E.powf(-x)) }
}

pub fn sigmoid_diff(y: f64) -> f64{
    y * (1.0 - y)
}

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