extern crate nn;
extern crate time;

use std::num::Float;
use nn::{NN, HaltCondition, LearningMode};
use time::Duration;

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
    let mut net1 = NN::new(&[2,3,1]);

    // train the network
    net1.train(&examples)
        .halt_condition( HaltCondition::Timer(Duration::seconds(10)) )
        .log_interval(None)
        .learning_mode( LearningMode::Batch(2) )
        .momentum(0.1)
        .go();

    // make sure json encoding/decoding works as expected
    let json = net1.to_json();
    let net2 = NN::from_json(&json);

    // test the trained network
    for &(ref inputs, ref outputs) in examples.iter() {
        let results = net2.run(inputs);
        let (result, key) = (Float::round(results[0]), outputs[0]);
        assert!(result == key);
    }
}
