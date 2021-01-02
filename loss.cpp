#include "loss.h"



float MeanSquaredError::loss(Matrix truth, Matrix pred){
	// MSE for matrices with shape (batch, values)
	// MSE = (Y - Y*)^2

	float total = 0;
	for (int i = 0; i < truth.height * truth.width; ++i) {
		float diff = truth.data[i] - pred.data[i];
		total += diff * diff;
	}

	return total / truth.height;
}

Matrix MeanSquaredError::gradient(Matrix truth, Matrix pred){
	// MSE = (Y - Y*)^2
	// grad ~ Y - Y*

	Matrix grad = Matrix(truth.height, truth.width);
	for (int i = 0; i < truth.height * truth.width; ++i) {
		grad.data[i] = truth.data[i] - pred.data[i];
	}

	return grad;
}
