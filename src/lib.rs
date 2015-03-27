extern crate rand;
extern crate threadpool;
extern crate rustc_serialize;

use HaltCondition::{ Epochs, MSE };
use UpdateRule::{ Stochastic, Batch };
use std::iter::{Zip, Enumerate};
use std::slice;
use std::num::Float;
use threadpool::{ScopedPool};
use std::sync::mpsc::channel;
use std::sync::{Arc, RwLock};
use std::cmp;
use rustc_serialize::json;

static DEFAULT_LEARNING_RATE: f64 = 0.3f64;
static DEFAULT_MOMENTUM: f64 = 0f64;
static DEFAULT_EPOCHS: u32 = 1000;

#[derive(Debug, Copy, Clone)]
pub enum HaltCondition {
    Epochs(u32),
    MSE(f64),
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum UpdateRule {
    Stochastic,
    Batch(u32)
}

#[derive(Debug)]
pub struct Trainer<'a,'b> {
    examples: &'b [(Vec<f64>, Vec<f64>)],
    rate: f64,
    momentum: f64,
    log_interval: Option<u32>,
    halt_condition: HaltCondition,
    update_rule: UpdateRule,
    nn: &'a mut NN,
}

impl<'a,'b> Trainer<'a,'b>  {
    
    pub fn rate(&mut self, rate: f64) -> &mut Trainer<'a,'b> {
        self.rate = rate;
        self
    }

    pub fn momentum(&mut self, momentum: f64) -> &mut Trainer<'a,'b> {
        self.momentum = momentum;
        self
    }

    pub fn log_interval(&mut self, log_interval: Option<u32>) -> &mut Trainer<'a,'b> {
        self.log_interval = log_interval;
        self
    }
    
    pub fn halt_condition(&mut self, halt_condition: HaltCondition) -> &mut Trainer<'a,'b> {
        self.halt_condition = halt_condition;
        self
    }

    pub fn update_rule(&mut self, update_rule: UpdateRule) -> &mut Trainer<'a,'b> {
        self.update_rule = update_rule;
        self
    }

    pub fn go(&mut self) -> f64 {
        self.nn.train_details(
            self.examples,
            self.rate,
            self.momentum,
            self.log_interval,
            self.halt_condition,
            self.update_rule,
        )
    }

}

#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub struct NN {
    layers: Vec<Vec<Vec<f64>>>,
    num_inputs: u32,
}

impl NN {

    pub fn new(layers_sizes: &[u32]) -> NN {
        if layers_sizes.len() < 2 {
            panic!("must have at least two layers");
        }

        for &layer_size in layers_sizes.iter() {
            if layer_size < 1 {
                panic!("can't have any empty layers");
            } 
        }


        let mut layers = Vec::new();
        let mut it = layers_sizes.iter();                
        // get the first layer size
        let first_layer_size = *it.next().unwrap();
        
        // setup the rest of the layers
        let mut prev_layer_size = first_layer_size;
        for &layer_size in it {
            let mut layer: Vec<Vec<f64>> = Vec::new();
            for _ in 0..layer_size {
                let mut node: Vec<f64> = Vec::new();
                for _ in 0..prev_layer_size+1 {
                    let random_weight: f64 = rand::random() - 0.5;
                    node.push(random_weight);
                }
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

    pub fn run(&self, inputs: &[f64]) -> Vec<f64> {
        if inputs.len() as u32 != self.num_inputs {
            panic!("input has a different length than the network's input layer");
        }
        self.do_run(inputs).pop().unwrap()
    }

    pub fn train<'b>(&'b mut self, examples: &'b [(Vec<f64>, Vec<f64>)]) -> Trainer {
        Trainer {
            examples: examples,
            rate: DEFAULT_LEARNING_RATE,
            momentum: DEFAULT_MOMENTUM,
            log_interval: None,
            halt_condition: Epochs(DEFAULT_EPOCHS),
            update_rule: Stochastic,
            nn: self
        }
    }

    fn train_details(&mut self, examples: &[(Vec<f64>, Vec<f64>)], rate: f64, momentum: f64, log_interval: Option<u32>,
                    halt_condition: HaltCondition, update_rule: UpdateRule) -> f64 {

        // check that input and output sizes are correct
        let input_layer_size = self.num_inputs;
        let output_layer_size = self.layers[self.layers.len() - 1].len();
        for &(ref inputs, ref outputs) in examples.iter() {
            if inputs.len() as u32 != input_layer_size {
                panic!("input has a different length than the network's input layer");
            }
            if outputs.len() != output_layer_size {
                panic!("output has a different length than the network's output layer");
            }
        }
        
        match update_rule {
            Stochastic => self.train_stochastic(examples, rate, momentum, log_interval, halt_condition),
            Batch(threads) => self.train_batch(examples, rate, momentum, log_interval, halt_condition, threads)
        }

    }

    fn train_stochastic(&mut self, examples: &[(Vec<f64>, Vec<f64>)], rate: f64, momentum: f64, log_interval: Option<u32>,
                    halt_condition: HaltCondition) -> f64 {
        
        let mut prev_deltas = self.make_weights_tracker(0.0f64);
        let mut epochs = 0u32;
        let mut training_error_rate = 0f64;

        loop {

            // log error rate if neccessary
            match log_interval {
                Some(interval) if epochs>0 && epochs % interval == 0 => {
                    println!("error rate: {}", training_error_rate);
                },
                _ => (),
            }

            // check if we've met the halt condition yet
            match halt_condition {
                Epochs(epochs_halt) => {
                    if epochs == epochs_halt { break }
                },
                MSE(target_error) => {
                    if training_error_rate <= target_error { break }
                }
            }

            training_error_rate = 0f64;
            
            for &(ref inputs, ref targets) in examples.iter() {
                let results = self.do_run(&inputs);
                let weight_updates = self.calculate_weight_updates(&results, &targets);
                training_error_rate += calculate_error(&results, &targets);
                self.update_weights(&weight_updates, &mut prev_deltas, rate, momentum)
            }

            epochs += 1;
        }

        training_error_rate
    }

    fn train_batch(&mut self, examples: &[(Vec<f64>, Vec<f64>)], rate: f64, momentum: f64, log_interval: Option<u32>,
                    halt_condition: HaltCondition, mut threads: u32) -> f64 {
        
        threads = cmp::min(threads, examples.len() as u32);

        let mut prev_deltas = self.make_weights_tracker(0.0f64);
        let mut epochs = 0;
        let mut training_error_rate = 0.0f64;

        let mut split_examples = Vec::new();
        {
            let small_num = examples.len() / threads as usize;
            let large_num = small_num + 1;
            let larges = examples.len() % threads as usize;
            let mut prev_start = 0;
            for i in 0..threads {
                let start = prev_start;
                let end = start + if i < larges as u32 { large_num } else { small_num };
                prev_start = end;
                let slc = &examples[start..end];
                split_examples.push(slc);
            }
        }        

        let pool = ScopedPool::new(threads);
        let self_lock = Arc::new(RwLock::new(self));
        let (tx, rx) = channel();

        loop {

            // log error rate if neccessary
            match log_interval {
                Some(interval) if epochs>0 && epochs % interval == 0 => {
                    println!("error rate: {}", training_error_rate);
                },
                _ => (),
            }

            // check if we've met the halt condition yet
            match halt_condition {
                Epochs(epochs_halt) => {
                    if epochs == epochs_halt { break }
                },
                MSE(target_error) => {
                    if training_error_rate <= target_error { break }
                }
            }

            training_error_rate = 0f64;

            // init batch data
            let mut batch_weight_updates =
                self_lock.read().unwrap().make_weights_tracker(0.0f64);
            
            // run each example using the thread pool
            for examples in split_examples.iter() {
                let self_lock = self_lock.clone();
                let tx = tx.clone();

                let mut local_weight_updates = self_lock.read().unwrap().make_weights_tracker(0.0f64);
                let mut local_error_rate = 0.0f64;

                pool.execute(move || { 
                    let read_self = self_lock.read().unwrap();

                    for &(ref inputs, ref targets) in examples.iter() {
                        let results = read_self.do_run(&inputs);
                        let new_weight_updates =
                            read_self.calculate_weight_updates(&results, &targets);
                        
                        let new_error_rate = calculate_error(&results, &targets);
                        
                        update_batch_data(&mut local_weight_updates, &new_weight_updates);
                        local_error_rate += new_error_rate;
                    }

                    tx.send((local_weight_updates, local_error_rate)).unwrap();
                });
            }

            // collect the results from the thread pool
            for _ in 0..threads {
                let (weight_updates, error_rate) = rx.recv().unwrap();
                training_error_rate += error_rate;
                update_batch_data(&mut batch_weight_updates, &weight_updates);
            }

            // update weights in the network
            self_lock.write().unwrap()
                .update_weights(&batch_weight_updates,
                                &mut prev_deltas,
                                rate, momentum);

            epochs += 1;
        }

        training_error_rate
    }
    

    fn do_run(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
        let mut results = Vec::new();
        results.push(inputs.to_vec());
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let mut layer_results = Vec::new();
            for node in layer.iter() {
                layer_results.push( sigmoid(modified_dotprod(&node, &results[layer_index])) )
            }
            results.push(layer_results);
        }
        results
    }

    // updates all weights in the network
    fn update_weights(&mut self, network_weight_updates: &Vec<Vec<Vec<f64>>>, prev_deltas: &mut Vec<Vec<Vec<f64>>>, rate: f64, momentum: f64) {
        for layer_index in 0..self.layers.len() {
            let mut layer = &mut self.layers[layer_index];
            let layer_weight_updates = &network_weight_updates[layer_index];
            for node_index in 0..layer.len() {
                let mut node = &mut layer[node_index];
                let node_weight_updates = &layer_weight_updates[node_index];
                for weight_index in 0..node.len() {
                    let weight_update = node_weight_updates[weight_index];
                    let prev_delta = prev_deltas[layer_index][node_index][weight_index];
                    let delta = (rate * weight_update) + (momentum * prev_delta);
                    node[weight_index] += delta;
                    prev_deltas[layer_index][node_index][weight_index] = delta;
                }
            }
        } 
    }

    // calculates all weight updates by backpropagation
    fn calculate_weight_updates(&self, results: &Vec<Vec<f64>>, targets: &[f64]) -> Vec<Vec<Vec<f64>>> {
        let mut network_errors:Vec<Vec<f64>> = Vec::new();
        let mut network_weight_updates = Vec::new();
        let layers = &self.layers;
        let network_results = &results[1..]; // skip the input layer

        let mut next_layer_nodes: Option<&Vec<Vec<f64>>> = None;
        
        for (layer_index, (layer_nodes, layer_results)) in iter_zip_enum(layers, network_results).rev() {
            let prev_layer_results = &results[layer_index];
            let mut layer_errors = Vec::new();
            let mut layer_weight_updates = Vec::new();
            

            for (node_index, (node, &result)) in iter_zip_enum(layer_nodes, layer_results) {
                let mut node_weight_updates = Vec::new();
                let mut node_error;
                
                // calculate error for this node
                if layer_index == layers.len() - 1 {
                    node_error = result * (1f64 - result) * (targets[node_index] - result);
                } else {
                    let mut sum = 0f64;
                    for (next_node, &next_node_error_data) in next_layer_nodes.unwrap().iter().zip((&network_errors[layer_index]).iter()) {
                        sum += next_node[node_index+1] * next_node_error_data; // +1 because the 0th weight is the threshold
                    }
                    node_error = result * (1f64 - result) * sum;
                }

                // calculate weight updates for this node
                for weight_index in 0..node.len() {
                    let mut prev_layer_result;
                    if weight_index == 0 {
                        prev_layer_result = 1f64; // theshold
                    } else {
                        prev_layer_result = prev_layer_results[weight_index-1];
                    }
                    let weight_update = node_error * prev_layer_result;
                    node_weight_updates.push(weight_update);
                }

                layer_errors.push(node_error);
                layer_weight_updates.push(node_weight_updates);
            }

            network_errors.push(layer_errors);
            network_weight_updates.push(layer_weight_updates);
            next_layer_nodes = Some(&layer_nodes);
        }

        // updates were built by backpropagation so reverse them
        network_weight_updates.reverse();

        network_weight_updates
    }    
    
    fn make_weights_tracker<T: Clone>(&self, place_holder: T) -> Vec<Vec<Vec<T>>> {
        let mut network_level = Vec::new(); 
        for layer in self.layers.iter() {
            let mut layer_level = Vec::new();
            for node in layer.iter() {
                let mut node_level = Vec::new();
                for _ in node.iter() {
                    node_level.push(place_holder.clone());
                }
                layer_level.push(node_level);
            }
            network_level.push(layer_level);
        }
        
        network_level
    }

    pub fn to_json(&self) -> String {
        json::encode(self).unwrap()
    }

    pub fn from_json(encoded: &str) -> NN {
        let network: NN = json::decode(encoded).unwrap();
        network
    }
}

fn modified_dotprod(node: &Vec<f64>, values: &Vec<f64>) -> f64 {
    let mut it = node.iter();
    let mut total = *it.next().unwrap(); // for the threshold weight
    for (weight, value) in it.zip(values.iter()) {
        total += weight * value;
    }
    total
}

fn sigmoid(y: f64) -> f64 {
    1f64 / (1f64 + Float::exp(-y))
}

// adds new network weight updates into the update updates already collected
fn update_batch_data(batch_data: &mut Vec<Vec<Vec<f64>>> , network_weight_updates: &Vec<Vec<Vec<f64>>>) {
    for layer_index in 0..batch_data.len() {
        let mut batch_layer = &mut batch_data[layer_index];
        let layer_weight_updates = &network_weight_updates[layer_index];
        for node_index in 0..batch_layer.len() {
            let mut batch_node = &mut batch_layer[node_index];
            let node_weight_updates = &layer_weight_updates[node_index];
            for weight_index in 0..batch_node.len() {
                batch_node[weight_index] += node_weight_updates[weight_index];
            }
        }
    } 
}


// takes two arrays and enumerates the iterator produced by zipping each of
// their iterators together
fn iter_zip_enum<'s, 't, S: 's, T: 't>(s: &'s [S], t: &'t [T]) ->
    Enumerate<Zip<slice::Iter<'s, S>, slice::Iter<'t, T>>>  {
    s.iter().zip(t.iter()).enumerate()
}

fn calculate_error(results: &Vec<Vec<f64>>, targets: &[f64]) -> f64 {
    let ref last_results = results[results.len()-1];
    let mut total:f64 = 0f64;
    for (&result, &target) in last_results.iter().zip(targets.iter()) {
        total += (target - result).powi(2);
    }
    total / (last_results.len() as f64)
}