#include "activation.h"

#include <algorithm>

#include "matrix.h"

Matrix ReLU::forward(Matrix input){
	Matrix out = Matrix(input.height, input.width);
	for (int i = 0; i < input.height * input.width; ++i) {
		out.data[i] = std::max(0.0f, input.data[i]);
	}

	return out;
}

Matrix ReLU::gradient(Matrix input, Matrix nextGrads){
	Matrix grads = Matrix(input.height, input.width);
	for (int i = 0; i < input.height * input.width; ++i) {
		grads.data[i] = input.data[i] > 0 ? nextGrads.data[i] : 0;
	}

	return grads;
}
