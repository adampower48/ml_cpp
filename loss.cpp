#include "loss.h"


float MeanSquaredError::loss(Tensor truth, Tensor pred){
	// MSE for matrices with shape (batch, values)
	// MSE = (Y - Y*)^2

	float total = 0;
	for (size_t i = 0; i < truth.shape[0] * truth.shape[1]; ++i) {
		float diff = truth.data[i] - pred.data[i];
		total += diff * diff;
	}

	return total / truth.shape[0];
}

Tensor MeanSquaredError::gradient(Tensor truth, Tensor pred){
	// MSE = (Y - Y*)^2
	// grad ~ Y - Y*

	Tensor grad = Tensor(truth.shape);
	for (size_t i = 0; i < truth.shape[0] * truth.shape[1]; ++i) {
		grad.data[i] = truth.data[i] - pred.data[i];
	}

	return grad;
}
