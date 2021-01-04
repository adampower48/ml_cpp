#include "activation.h"

#include <algorithm>

#include "matrix.h"


Tensor ReLU::forward(Tensor input){
	Tensor out = Tensor(input.shape);
	for (int i = 0; i < input.shape[0] * input.shape[1]; ++i) {
		out.data[i] = std::max(0.0f, input.data[i]);
	}

	return out;
}

Tensor ReLU::gradient(Tensor input, Tensor nextGrads){
	Tensor grads = Tensor(input.shape);
	for (int i = 0; i < input.shape[0] * input.shape[1]; ++i) {
		grads.data[i] = input.data[i] > 0 ? nextGrads.data[i] : 0;
	}

	return grads;
}
