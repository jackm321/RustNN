extern crate nn;

use nn::{NN, HaltCondition, Activation};

const ACTIONS:u32 = 10;


fn main()
{
	// create examples of the xor function
	let mut examples = Vec::new();
	for i in 0..ACTIONS
	{
		let mut result = Vec::new();
		for j in 0..ACTIONS
		{
			if j == i { result.push(1.0); }
			else { result.push(0.0); }
		}
		let example = (vec![i as f64], result);
		examples.push(example);
	}

	// create a new neural network
	let mut nn = NN::new(&[1, 10, ACTIONS], Activation::PELU, Activation::Sigmoid);

	// train the network
	nn.train(&examples)
		.log_interval(Some(1000))
		.halt_condition( HaltCondition::MSE(0.01) )
		.rate(0.025)
		.momentum(0.5)
		.lambda(0.00005)
		.go();

	// print results of the trained network
	for &(ref input, _) in examples.iter()
	{
		let result = nn.run(input);
		let print:Vec<String> = result.iter().map(|x:&f64| { format!("{:4.2}", (*x * 100.0).round() / 100.0) }).collect();
		println!("{:1.0} -> {:?}", input[0], print);
	}
}
