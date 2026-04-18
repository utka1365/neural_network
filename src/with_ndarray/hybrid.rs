use std::{
    io::{Error, ErrorKind},
};
use std::f32::consts::E;
use ndarray::{Array1, Array2, Axis, Zip};
use ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator};
use rand::prelude::SliceRandom;
use rand::random_range;

pub const LEARNING_RATE: f32 = 0.001;
pub const RELU_KOEFF: f32 = 0.01;
pub const LN_VALUE: f32 = 6.90675;
pub const LAMBDA: f32 = 0.0001;
pub const BETA1: f32 = 0.9;
pub const BETA2: f32 = 0.999;
pub const EPSILON: f32 = 0.00000001;

#[derive(Clone)]
pub enum Activation {
    Sigmoid,
    ReLU
}

pub struct DataSet {
    size: i32,
    inputs: Vec<Array1<f32>>,
    outputs: Vec<Array1<f32>>,
}

impl DataSet {
    pub fn new(mut inputs: Vec<Vec<f32>>, mut outputs: Vec<Vec<f32>>) -> Result<Self, Error> {
        if inputs.len() != outputs.len() {
            return Err(Error::new(ErrorKind::Other, "inputs and outputs must be the same"));
        }
        
        let mut converted_inputs: Vec<Array1<f32>> = Vec::new();
        let mut converted_outputs: Vec<Array1<f32>> = Vec::new();
        let size = inputs.len();
        
        for _ in 0..size {
            converted_inputs.push(Array1::from_vec(inputs.remove(0)));
            converted_outputs.push(Array1::from_vec(outputs.remove(0)));
        }
        
        Ok(Self {
            size: size as i32,
            inputs: converted_inputs,
            outputs: converted_outputs,
        })
    }
}

pub struct Layer {
    output: Array1<f32>,
    sums: Array1<f32>,
    deltas: Array1<f32>,
    weights: Array2<f32>,
    bias: Array1<f32>,
    activation: Activation,
}

impl Layer {
    pub fn new(
        cnt_neurons: i32, prev_cnt_neurons: i32, activation: Activation
    ) -> Result<Self, Error>{
        let output = Array1::<f32>::zeros(cnt_neurons as usize);
        let sums = Array1::<f32>::zeros(cnt_neurons as usize);
        let deltas = Array1::<f32>::zeros(cnt_neurons as usize);
        let mut weights =
            Array2::<f32>::zeros((cnt_neurons as usize, prev_cnt_neurons as usize));
        let mut bias = Array1::<f32>::zeros(cnt_neurons as usize);

        for i in 0..cnt_neurons as usize {
            for j in 0..prev_cnt_neurons as usize {
                weights[[i, j]] = random_range(
                    -1.0 / prev_cnt_neurons as f32..1.0 / prev_cnt_neurons as f32
                );
            }

            bias[i] = random_range(
                -1.0 / prev_cnt_neurons as f32..1.0 / prev_cnt_neurons as f32
            );
        }

        Ok(Self{output, sums, deltas, weights, bias, activation})
    }

    fn forward(&mut self, input: &Array1<f32>) {
        self.sums = self.weights.dot(input) + &self.bias;
        self.output = self.sums.mapv(|x| match self.activation {
            Activation::Sigmoid => sigmoid(x),
            Activation::ReLU => ReLU(x)
        });
    }

    fn backward(&mut self, input: &Array1<f32>, next_layer: &Self) -> (Array2<f32>, Array1<f32>) {
        self.deltas = self.output.mapv(|x| match self.activation {
                    Activation::Sigmoid => sigmoid_diff(x),
                    Activation::ReLU => ReLU_diff(x)
            });
        let deltas = &next_layer.weights.t().dot(&next_layer.deltas) * &self.deltas;
        self.deltas = deltas.clone();

        (
            deltas.to_owned().insert_axis(Axis(1)) * input.to_owned().insert_axis(Axis(0)),
            deltas
        )
    }

    fn adaptive_output_backward(
        &mut self, input: &Array1<f32>, labels: &Array1<f32>
    ) -> (Array2<f32>, Array1<f32>) {
        let denom = 1.0 + input
            .into_par_iter()
            .map(|x| x.powi(2))
            .sum::<f32>();
        let mut deltas = Array1::<f32>::zeros(self.output.len());
        Zip::from(&mut deltas).and(&self.sums).and(labels).par_for_each(|delta, sum, label| {
            if *label == 1.0 {
                *delta = (sum - LN_VALUE) / denom;
            } else {
                *delta = (sum + LN_VALUE) / denom;
            }
        });
        Zip::from(&mut self.deltas)
            .and(&self.output)
            .and(labels)
            .par_for_each(|gamma, activate_value, label|
                *gamma = (activate_value - label) * match self.activation {
            Activation::Sigmoid => sigmoid_diff(*activate_value),
            Activation::ReLU => ReLU_diff(*activate_value)
        });

        (
            deltas.to_owned().insert_axis(Axis(1)) * input.to_owned().insert_axis(Axis(0)),
            deltas
        )
    }

    fn classic_output_backward(
        &mut self, input: &Array1<f32>, labels: &Array1<f32>
    ) -> (Array2<f32>, Array1<f32>) {
        self.deltas = (&self.output - labels) * self.output.mapv(|x| match self.activation {
            Activation::Sigmoid => sigmoid_diff(x),
            Activation::ReLU => ReLU_diff(x)
        });
        let deltas = self.deltas.clone();

        (
            deltas.to_owned().insert_axis(Axis(1)) * input.to_owned().insert_axis(Axis(0)),
            deltas
        )
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

    pub fn train(&mut self, data: &DataSet, epoch_count: i32, batch_size: usize) {
        let mut rng = rand::rng();
        let mut indices = (0..data.size as usize).collect::<Vec<usize>>();
        let mut weights_gradients: Vec<Array2<f32>> = Vec::new();
        let mut biases_gradients: Vec<Array1<f32>> = Vec::new();
        let cnt_layers = self.layers.len();

        for i in 0..self.layers.len() {
            weights_gradients.push(Array2::<f32>::zeros(self.layers[i].weights.dim()));
            biases_gradients.push(Array1::<f32>::zeros(self.layers[i].bias.dim()));
        }

        for _ in 0..epoch_count {
            indices.shuffle(&mut rng);

            for batch_ind in 0..data.size as usize / batch_size {
                let batch =
                    &indices[batch_ind * batch_size..(batch_ind + 1) * batch_size];

                for point in 0..batch_size {
                    self.layers[0].forward(&data.inputs[batch[point]]);

                    for layer in 1..cnt_layers {
                        let (left, right) =
                            self.layers.split_at_mut(layer);
                        right[0].forward(&left[layer-1].output);
                    }

                    {
                        let (left, right) =
                            self.layers.split_at_mut(cnt_layers - 1);
                        let (weights, bias) = right[0]
                            .adaptive_output_backward(
                                &left[cnt_layers-2].output,
                                &data.outputs[batch[point]],
                            );
                        weights_gradients[cnt_layers-1] += &weights;
                        biases_gradients[cnt_layers-1] += &bias;
                    }

                    for layer in (1..cnt_layers - 1).rev() {
                        let (left, right) =
                            self.layers.split_at_mut(layer+1);
                        let (left, mid) = left.split_at_mut(layer);
                        let (weights, bias) = mid[0].backward(
                            &left[layer-1].output,
                            &right[0]
                        );
                        weights_gradients[layer] += &weights;
                        biases_gradients[layer] += &bias;
                    }

                    {
                        let (left, right) =
                            self.layers.split_at_mut(1);
                        let (weights, bias) = left[0].backward(
                            &data.inputs[batch[point]],
                            &right[0]
                        );
                        weights_gradients[0] += &weights;
                        biases_gradients[0] += &bias;
                    }
                }

                let learning_rate_scaled = LEARNING_RATE / batch_size as f32;
                let lambda_scaled = LAMBDA / batch_size as f32;

                Zip::from(&mut self.layers)
                    .and(&weights_gradients)
                    .and(&biases_gradients)
                    .par_for_each(|layer, weights, bias| {
                        layer.weights -= &(learning_rate_scaled * weights +
                            2.0 * lambda_scaled * &layer.weights);
                        layer.bias -= &(learning_rate_scaled * bias);
                    });
                Zip::from(&mut weights_gradients)
                    .and(&mut biases_gradients)
                    .par_for_each(|weights, bias| {
                    weights.fill(0.0);
                    bias.fill(0.0);
                });
            }
        }
    }

    pub fn train_adam(&mut self, data: &DataSet, epoch_count: i32, batch_size: usize) {
        let mut rng = rand::rng();
        let mut indices = (0..data.size as usize).collect::<Vec<usize>>();
        let mut weights_gradients: Vec<Array2<f32>> = Vec::new();
        let mut biases_gradients: Vec<Array1<f32>> = Vec::new();
        let mut momentums: Vec<Array2<f32>> = Vec::new();
        let mut adaptive_scales: Vec<Array2<f32>> = Vec::new();
        let mut bias_momentums: Vec<Array1<f32>> = Vec::new();
        let mut bias_adaptive_scales: Vec<Array1<f32>> = Vec::new();
        let mut t = 0;
        let cnt_layers = self.layers.len();

        for i in 0..self.layers.len() {
            weights_gradients.push(Array2::<f32>::zeros(self.layers[i].weights.dim()));
            biases_gradients.push(Array1::<f32>::zeros(self.layers[i].bias.dim()));
            momentums.push(Array2::<f32>::zeros(self.layers[i].weights.dim()));
            adaptive_scales.push(Array2::<f32>::zeros(self.layers[i].weights.dim()));
            bias_momentums.push(Array1::<f32>::zeros(self.layers[i].output.dim()));
            bias_adaptive_scales.push(Array1::<f32>::zeros(self.layers[i].output.dim()));
        }

        for _ in 0..epoch_count {
            indices.shuffle(&mut rng);

            for batch_ind in 0..data.size as usize / batch_size {
                t += 1;
                let batch =
                    &indices[batch_ind * batch_size..(batch_ind + 1) * batch_size];

                for point in 0..batch_size {
                    self.layers[0].forward(&data.inputs[batch[point]]);

                    for layer in 1..cnt_layers {
                        let (left, right) =
                            self.layers.split_at_mut(layer);
                        right[0].forward(&left[layer-1].output);
                    }

                    {
                        let (left, right) =
                            self.layers.split_at_mut(cnt_layers - 1);
                        let (weights, bias) = right[0]
                            .adaptive_output_backward(
                                &left[cnt_layers-2].output,
                                &data.outputs[batch[point]]
                            );
                        weights_gradients[cnt_layers-1] += &weights;
                        biases_gradients[cnt_layers-1] += &bias;
                    }

                    for layer in (1..cnt_layers - 1).rev() {
                        let (left, right) =
                            self.layers.split_at_mut(layer+1);
                        let (left, mid) =
                            left.split_at_mut(layer);
                        let (weights, bias) = mid[0].backward(
                            &left[layer-1].output,
                            &right[0]
                        );
                        weights_gradients[layer] += &weights;
                        biases_gradients[layer] += &bias;
                    }

                    {
                        let (left, right) =
                            self.layers.split_at_mut(1);
                        let (weights, bias) = left[0].backward(
                            &data.inputs[batch[point]],
                            &right[0]
                        );
                        weights_gradients[0] += &weights;
                        biases_gradients[0] += &bias;
                    }
                }

                let learning_rate_scaled = LEARNING_RATE / batch_size as f32;
                let lambda_scaled = LAMBDA / batch_size as f32;

                Zip::from(&mut momentums)
                    .and(&mut adaptive_scales)
                    .and(&mut bias_momentums)
                    .and(&mut bias_adaptive_scales)
                    .and(&weights_gradients)
                    .and(&biases_gradients)
                    .par_for_each(|m,
                                   v,
                                   bias_m,
                                   bias_v,
                                   w_gradients,
                                   b_gradients| {
                        *m *= BETA1;
                        *bias_m *= BETA1;
                        *v *= BETA2;
                        *bias_v *= BETA2;

                        *m += &((1.0 - BETA1) * w_gradients);
                        *bias_m += &((1.0 - BETA1) * b_gradients);
                        *v += &((1.0 - BETA2) * w_gradients.mapv(|g | g * g));
                        *bias_v += &((1.0 - BETA2) * b_gradients.mapv(|g| g * g));
                });

                Zip::from(&mut self.layers)
                    .and(&momentums)
                    .and(&adaptive_scales)
                    .and(&bias_momentums)
                    .and(&bias_adaptive_scales)
                    .par_for_each(|layer,
                        m,
                        v,
                        bias_m,
                        bias_v| {
                        let correct_m = m / (1.0 - BETA1.powi(t));
                        let correct_bias_m = bias_m / (1.0 - BETA1.powi(t));
                        let correct_v = (v / (1.0 - BETA2.powi(t))).sqrt() + EPSILON;
                        let correct_bias_v =
                            (bias_v / (1.0 - BETA2.powi(t))).sqrt() + EPSILON;
                        layer.weights -= &(learning_rate_scaled * ((correct_m / correct_v) +
                            lambda_scaled * &layer.weights));
                        layer.bias -= &(learning_rate_scaled * correct_bias_m / correct_bias_v);
                    });
                Zip::from(&mut weights_gradients)
                    .and(&mut biases_gradients)
                    .par_for_each(|weights, bias| {
                        weights.fill(0.0);
                        bias.fill(0.0);
                    });
            }
        }
    }
    
    pub fn test(&mut self, data: &DataSet) {
        let mut accuracy = 0.0;
        let mut cnt = 0;
        let cnt_layers = self.layers.len();
        
        for vector in 0..data.size as usize {
            self.layers[0].forward(&data.inputs[vector]);

            for layer in 1..cnt_layers {
                let (left, right) =
                    self.layers.split_at_mut(layer);
                right[0].forward(&left[layer-1].output);
            }
            
            let mut max = 0.0;
            let mut max_ind = 0;
            let mut label_ind = 0;
            let mut i = 0;
            
            Zip::from(&self.layers[cnt_layers-1].output)
                .and(&data.outputs[vector])
                .for_each(|prediction, label| {
                    if *label == 1.0 {
                        label_ind = i;
                    }

                    if *prediction > max {
                        max = *prediction;
                        max_ind = i;
                    }
                    
                    accuracy += (prediction - label).powi(2);
                    i += 1;
                });

            if max_ind == label_ind {
                cnt += 1;
            }
        }
        
        println!("{}", accuracy / 100000.0);
        println!("{}", cnt as f64 / 10000.0);
    }
}

pub fn sigmoid(x: f32) -> f32{
    if x >= 10.0{ 0.999 }
    else if x <= -10.0 { 0.001 }
    else { 1.0 / (1.0 + E.powf(-x)) }
}

pub fn sigmoid_diff(y: f32) -> f32{
    y * (1.0 - y)
}

pub fn ReLU(x: f32) -> f32{
    if x < 0.0{
        RELU_KOEFF * x
    } else {
        x
    }
}

pub fn ReLU_diff(x: f32) -> f32{
    if x < 0.0{
        RELU_KOEFF
    } else {
        1.0
    }
}