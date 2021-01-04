#include "linear.h"

#include <ctime>
#include <iostream>
#include <tuple>

Linear::Linear(size_t input_size, size_t units){
	Linear::input_size = input_size;
	output_size = units;

	std::vector<size_t> weightShape{input_size, units};
	std::vector<size_t> biasShape{1, units};
	weights = new Tensor(weightShape);
	biases = new Tensor(biasShape);
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

Tensor Linear::forward(Tensor input){
	// Y = XW + B
	// input: array of size (height, width)

	return input.matmul(*weights).add(*biases);
}

std::tuple<Tensor, Tensor> Linear::calculateGradient(Tensor input, Tensor nextGrads){
	// Computes gradient


	// Bias gradient
	// Gb = D
	Tensor gradBias = Tensor(biases->shape);
	for (size_t i = 0; i < biases->shape[0]; ++i) {
		// Over batch
		for (size_t j = 0; j < biases->shape[1]; ++j) {
			// Over nodes
			gradBias.data[j] += nextGrads.data[i * nextGrads.shape[1] + j];
		}
	}


	// Weight gradient
	// Gw = DX
	Tensor gradWeights = Tensor(weights->shape);
	for (size_t k = 0; k < input.shape[0]; ++k) {
		// Over batch
		for (size_t i = 0; i < weights->shape[1]; ++i) {
			// Over nodes
			for (size_t j = 0; j < weights->shape[0]; ++j) {
				// Over params
				gradWeights.data[j * weights->shape[1] + i] += nextGrads.data[k * nextGrads.shape[1] + i] *
					input.data[k * input.shape[1] + j];
			}
		}
	}

	return std::make_tuple(gradWeights, gradBias);

}


void Linear::updateWeights(Tensor gradWeights, Tensor gradBiases, float learningRate){
	// Update biases
	for (size_t i = 0; i < biases->shape[1]; ++i) {
		biases->data[i] += gradBiases.data[i] * learningRate;
	}

	// Update weights
	for (size_t i = 0; i < weights->shape[0] * weights->shape[1]; ++i) {
		weights->data[i] += gradWeights.data[i] * learningRate;
	}
}
