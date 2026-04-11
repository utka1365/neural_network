use std::{
    io::{Error, ErrorKind},
};
use std::alloc::LayoutErr;
use std::f64::consts::E;
use ndarray::{Array1, Array2};
use ndarray::parallel::prelude::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use rand::random_range;
use crate::only_std::base::RELU_KOEFF;

pub const LEARNING_RATE: f64 = 0.01;
pub const LN_VALUE: f64 = 6.90675;

pub enum Activation {
    Sigmoid,
    ReLU
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

    fn backward(&mut self, input: &Array1<f64>, output_layer: &Self) {
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
            &output_layer.gammas.dot(&output_layer.weights);
        self.bias = &self.bias - &deltas;
        self.weights = &self.weights - deltas * input.t();
        self.gammas = &self.gammas * &self.output;
    }

    fn output_backward(&mut self, input: &Array1<f64>, output: &Array1<f64>) {
        let denom = 1.0 + input
            .into_par_iter()
            .map(|x| x.powi(2))
            .sum::<f64>();
        let deltas = (&self.sums - LN_VALUE) / denom;
        self.gammas = &self.output - output;
        self.bias = &self.bias - &deltas;
        self.weights = &self.weights - deltas * input.t();
        self.gammas = &self.gammas * &self.output;
    }
}

pub struct Network {
    layers: Array1<Layer>,
    input: Array1<f64>,
    output: Array1<f64>,
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

        let input = Array1::<f64>::zeros(cnt_neurons[0] as usize);
        let output =
            Array1::<f64>::zeros(cnt_neurons[cnt_neurons.len()-1] as usize);
        let mut layers = Vec::new();

        for i in 0..cnt_neurons.len() - 1 {
            layers.push(Layer::new(cnt_neurons[i+1], cnt_neurons[i], &activation_types[i])?);
        }

        let layers = Array1::from_vec(layers);
        Ok(Self{
            layers,
            input,
            output
        })
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