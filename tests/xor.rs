extern crate nn;
use std::num::Float;
use nn::{NN, HaltCondition};

#[test]
fn xor_test() {
    // create examples of the xor function
    let examples = [
        (vec![0f64, 0f64], vec![0f64]),
        (vec![0f64, 1f64], vec![1f64]),
        (vec![1f64, 0f64], vec![1f64]),
        (vec![1f64, 1f64], vec![0f64]),
    ];

    // create a new neural network
    let mut net = NN::new(&[2,2,1]);
    
    // train the network
    net
        .train(&examples)
        .halt_condition( HaltCondition::Epochs(10000) )
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
