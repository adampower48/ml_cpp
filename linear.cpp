#include "linear.h"

#include <iostream>
#include <ctime>


Linear::Linear(int input_size, int units){
	Linear::input_size = input_size;
	output_size = units;

	weights = new Matrix(input_size, units);
	biases = new Matrix(1, units);
}

void Linear::initRange(){
	weights->initRange();
	biases->initRange();
}

void Linear::initNormal(){
	const auto seed = static_cast<int>(time(NULL)); // Seed with current time
	weights->initNormal(seed);
	biases->initNormal(seed + 1);
}

void Linear::print(){
	std::cout << "Weights:\n";
	weights->print();

	std::cout << "Biases:\n";
	biases->print();
}

Matrix Linear::forward(Matrix input){
	// Y = WX + B
	// input: array of size (height, width)

	return input.matmul(*weights).add(*biases);
}
