extern crate rand;
use HaltCondition::{ Epochs, MSE };
use UpdateRule::{ Stochastic, Batch };

#[test]
fn it_works() {
}

type Node = Vec<f64>;
type Layer = Vec<Node>;

#[derive(Debug)]
pub enum HaltCondition {
    Epochs(usize),
    MSE(f64),
}

#[derive(Debug)]
pub enum UpdateRule {
    Stochastic,
    Batch
}

#[derive(Debug)]
struct Trainer {
    rate: f64,
    momentum: f64,
    log_interval: Option<usize>,
    halt_condition: HaltCondition,
    update_rule: UpdateRule,
    threads: usize,
}

#[derive(Debug)]
#[derive(Clone)]
pub struct NN {
    layers: Vec<Layer>,
    num_inputs: usize,
}

impl NN {
    fn train(&self) -> Trainer {
        unimplemented!()
    }

    // returns a results with ok being the error rate of the network found at the end and
    // Err is a string with an explaination
    fn train_option(&mut self, examples: &[(&[f64], &[f64])], rate: f64, momentum: f64, log_interval: Option<usize>,
                    halt_condition: HaltCondition, update_rule: UpdateRule, threads: usize) -> Result<usize, &str> {

        let mut epochs = 0;
        let mut error_rate = 0.0;

        loop {

            let mut prev_deltas = make_weights_tracker(&self.layers, 0.0f64);
            let mut batch_data = make_weights_tracker(&self.layers, 0.0f64);
            let mut error_sum: f64 = 0.0;

            for &(inputs, targets) in examples.iter() {
                let results = do_run(self, inputs);
                let weight_updates = calculate_weight_update_info(self, &results, &targets);
                error_sum += calculate_error(&results, targets);
                match update_rule {
                    Batch => update_batch_data(&mut batch_data, &weight_updates),
                    Stochastic => update_weights(self, &weight_updates, &mut prev_deltas),
                }
            }

            // if we're in batch mode, update the weights now
            match update_rule {
                Batch => update_weights(self, &batch_data, &mut prev_deltas),
                Stochastic => (),
            }

            epochs += 1;

            // log error rate if neccessary
            match log_interval {
                Some(interval) if epochs % interval == 0 => {
                    println!("error rate: {}", error_rate);
                },
                _ => (),
            }

            // check if we've met the halt condition yet
            match halt_condition {
                Epochs(epochs_halt) => {
                    if epochs == epochs_halt { break; }
                },
                MSE(target_error) => {
                    if target_error == error_rate { break; }
                }
            }
        }
        
        unimplemented!();
    }

    fn run(&self, inputs: &[f64]) -> &[f64]{
        unimplemented!()
    }
}

fn do_run(nn: &NN, inputs: &[f64]) -> Vec<Vec<f64>>{
    unimplemented!()
}

fn update_batch_data(batch_data: &mut Vec<Vec<Vec<f64>>> , weight_updates: &Vec<Vec<Vec<f64>>>) {
    unimplemented!()
}

fn update_weights(nn: &mut NN, weight_updates: &Vec<Vec<Vec<f64>>>, prev_deltas: &mut Vec<Vec<Vec<f64>>>) {
    unimplemented!()
}


fn calculate_weight_update_info(nn: &NN, results: &Vec<Vec<f64>>, targets: &[f64]) -> Vec<Vec<Vec<f64>>> {
    unimplemented!()
}

fn calculate_error(results: &Vec<Vec<f64>>, targets: &[f64]) -> f64 {
    unimplemented!()
}

fn make_weights_tracker<T: Clone>(layers: &Vec<Layer>, place_holder: T) -> Vec<Vec<Vec<T>>> {
    let mut network_level = Vec::new(); 
    for layer in layers.iter() {
        let mut layer_level = Vec::new();
        for node in layer.iter() {
            let mut node_level = Vec::new();
            for weight in node.iter() {
                node_level.push(place_holder.clone());
            }
            layer_level.push(node_level);
        }
        network_level.push(layer_level);
    }
    
    network_level
}

fn new_nn(layers_sizes: &[usize]) -> NN {
        if layers_sizes.len() < 2 {
            panic!("must have at least two layers");
        }
        let mut layers = Vec::new();
        let mut it = layers_sizes.iter();                
        // get the first layer size
        let first_layer_size = *it.next().unwrap();
        
        // setup the rest of the layers
        let mut prev_layer_size = first_layer_size;
        for &layer_size in it {
            let mut layer: Layer = Vec::new();
            for _ in 0..layer_size {
                let mut node: Node = vec![rand::random(); prev_layer_size];
                node.shrink_to_fit();
                layer.push(node)
            }
            layer.shrink_to_fit();
            layers.push(layer);
            prev_layer_size = layer_size;
        }
        layers.shrink_to_fit();
        NN { layers: layers, num_inputs: first_layer_size }
    }