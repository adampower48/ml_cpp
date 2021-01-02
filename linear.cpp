#include "matrix.h"

#include "linear.h"

#include <iostream>

#include "matrix.h"

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
