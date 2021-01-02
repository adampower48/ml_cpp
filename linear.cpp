#include "linear.h"

#include <ctime>
#include <iostream>
#include <tuple>


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

std::tuple<Matrix, Matrix> Linear::calculateGradient(Matrix input, Matrix linearOutput, Matrix target){
	// Computes gradient

	// D = Y - Y*
	Matrix diff = target.sub(linearOutput);
	std::cout << "Difference:\n";
	diff.print();

	// Bias gradient
	// Gb = D
	Matrix gradBias = Matrix(1, biases->width);
	for (int i = 0; i < input.height; ++i) {
		// Over batch
		for (int j = 0; j < biases->width; ++j) {
			// Over nodes
			gradBias.data[j] += diff.data[i * biases->width + j];
		}
	}


	// Weight gradient
	// Gw = DX
	Matrix gradWeights = Matrix(weights->height, weights->width);

	for (int k = 0; k < input.height; ++k) {
		// Over batch
		for (int i = 0; i < weights->width; ++i) {
			// Over nodes
			for (int j = 0; j < weights->height; ++j) {
				// Over params
				gradWeights.data[j * weights->width + i] += diff.data[i] *
					input.data[k * input.width + j];
			}
		}
	}

	return std::make_tuple(gradWeights, gradBias);

}

void Linear::updateWeights(Matrix gradWeights, Matrix gradBiases, float learningRate){
	// Update biases
	for (int i = 0; i < biases->width; ++i) {
		biases->data[i] += gradWeights.data[i] * learningRate;
	}

	// Update weights
	for (int i = 0; i < weights->height * weights->width; ++i) {
		weights->data[i] += gradWeights.data[i] * learningRate;
	}
}
