#pragma once
#include <tuple>


#include "matrix.h"



class Linear {
public:
	size_t input_size;
	size_t output_size;
	Tensor* weights;
	Tensor* biases;

	Linear(size_t input_size, size_t units);

	void initRange();

	void initNormal();

	void print();

	Tensor forward(Tensor input);

	std::tuple<Tensor, Tensor> calculateGradient(Tensor input, Tensor nextGrads);

	void updateWeights(Tensor gradWeights, Tensor gradBiases, float learningRate);
};
