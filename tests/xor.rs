#![feature(std_misc)]

extern crate nn;
use std::num::Float;
use nn::{NN, HaltCondition};
use std::time::Duration;

#[test]
fn xor_timed() {
    // create examples of the xor function
    let examples = [
        (vec![0f64, 0f64], vec![0f64]),
        (vec![0f64, 1f64], vec![1f64]),
        (vec![1f64, 0f64], vec![1f64]),
        (vec![1f64, 1f64], vec![0f64]),
    ];

    // create a new neural network
    let mut net = NN::new(&[2,3,1]);
    
    // train the network
    net.train(&examples)
        .halt_condition( HaltCondition::Timer(Duration::seconds(10)) )
        .log_interval(None)
        .momentum(0.1)
        .go();
    
    // test the trained network
    for &(ref inputs, ref outputs) in examples.iter() {
        let results = net.run(inputs);
        let (result, key) = (Float::round(results[0]), outputs[0]);
        assert!(result == key);
    }
}
