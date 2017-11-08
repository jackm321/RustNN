# RustNN

An easy to use neural network library written in Rust.

For the documentation take a look at the original library or generate it using "cargo doc".

## Description
RustNN is a [feedforward neural network ](http://en.wikipedia.org/wiki/Feedforward_neural_network)
library. The library
generates fully connected multi-layer artificial neural networks that
are trained via [backpropagation](http://en.wikipedia.org/wiki/Backpropagation).
Networks are trained using an incremental training mode.

## Fork
This fork adds L2 regularization and several activation functions to the original crate. Additionally, there are a few minor improvements.
Lambda can be set just like the learning rate. The activation functions for hidden and output gets set in NN::new as second and third parameter respectively.

## XOR example

This example creates a neural network with `2` nodes in the input layer,
a single hidden layer containing `3` nodes, and `1` node in the output layer.
The network is then trained on examples of the `XOR` function. All of the
methods called after `train(&examples)` are optional and are just used
to specify various options that dictate how the network should be trained.
When the `go()` method is called the network will begin training on the
given examples. See the documentation for the `NN` and `Trainer` structs
for more details.

```rust
use nn::{NN, HaltCondition, Activation};

// create examples of the XOR function
// the network is trained on tuples of vectors where the first vector
// is the inputs and the second vector is the expected outputs
let examples = [
    (vec![0f64, 0f64], vec![0f64]),
    (vec![0f64, 1f64], vec![1f64]),
    (vec![1f64, 0f64], vec![1f64]),
    (vec![1f64, 1f64], vec![0f64]),
];

// create a new neural network by passing a pointer to an array
// that specifies the number of layers and the number of nodes in each layer
// in this case we have an input layer with 2 nodes, one hidden layer
// with 3 nodes and the output layer has 1 node
let mut net = NN::new(&[2, 3, 1], Activation::PELU, Activation::Sigmoid);
    
// train the network on the examples of the XOR function
// all methods seen here are optional except go() which must be called to begin training
// see the documentation for the Trainer struct for more info on what each method does
net.train(&examples)
    .halt_condition( HaltCondition::Epochs(10000) )
    .log_interval( Some(100) )
    .momentum( 0.1 )
    .rate( 0.3 )
    .go();
    
// evaluate the network to see if it learned the XOR function
for &(ref inputs, ref outputs) in examples.iter() {
    let results = net.run(inputs);
    let (result, key) = (results[0].round(), outputs[0]);
    assert!(result == key);
}
```