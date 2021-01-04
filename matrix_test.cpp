// matrix_ops.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "matrix.h"
#include "matrix.h"

#include <iostream>
#include <tuple>

#include "activation.h"
#include "helpers.h"
#include "linear.h"
#include "loss.h"


void testMatOps(){
	Matrix a(4, 3), b(3, 5), d(1, 5);
	a.initRange();
	b.initRange();
	d.initRange();

	std::cout << "A\n";
	a.print();

	std::cout << "B\n";
	b.print();

	std::cout << "D\n";
	d.print();

	// MatMul
	Matrix c = a.matmul(b);
	std::cout << "C = AB\n";
	c.print();

	// Mat-Mat Addition
	Matrix e = a.add(a);
	std::cout << "E = A + A\n";
	e.print();

	// Mat-Vec Addition
	Matrix f = c.add(d);
	std::cout << "F = C + D\n";
	f.print();

	// Mat-Mat Subtraction
	Matrix g = a.sub(a);
	std::cout << "G = A - A\n";
	g.print();

	// Mat-Vec Subtraction
	Matrix h = c.sub(d);
	std::cout << "H = C - D\n";
	h.print();

	// * Operator
	Matrix i = a * b;
	std::cout << "I = AB\n";
	i.print();

	// + Operator (Mat-Mat)
	Matrix j = a + a;
	std::cout << "J = A + A\n";
	j.print();

	// + Operator (Mat-Vec)
	Matrix k = c + d;
	std::cout << "K = C + D\n";
	k.print();

	// - Operator (Mat-Mat)
	Matrix l = a - a;
	std::cout << "L = A - A\n";
	l.print();

	// - Operator (Mat-Vec)
	Matrix m = c - d;
	std::cout << "M = C - D\n";
	m.print();

	// [] Operator (read)
	std::cout << "M[0]: " << m[0] << "\n";
	std::cout << "M[1]: " << m[1] << "\n";
	std::cout << "M[2]: " << m[2] << "\n";

	// [] Operator (write)
	m[0] = 100;
	std::cout << "M[0] = 100:\n";
	m.print();

}

void testLinear(){
	Linear linear = Linear(4, 3);
	// linear.initRange();
	linear.initNormal();
	linear.print();

	Matrix input = Matrix(3, 4);
	input.initRange();
	std::cout << "Input:\n";
	input.print();

	Matrix targets = Matrix(3, 3);
	targets.initRange();
	std::cout << "Targets:\n";
	targets.print();

	MeanSquaredError mse;

	std::cout << "===================================================================\n";
	for (int i = 0; i < 1000; ++i) {
		Matrix out = linear.forward(input);
		// std::cout << "Output:\n";
		// out.print();

		// D = Y - Y*
		Matrix lossGrads = mse.gradient(targets, out);
		// std::cout << "Loss Gradients:\n";
		// lossGrads.print();

		auto [gradWeights, gradBiases] = linear.calculateGradient(input, lossGrads);
		// std::cout << "\nGradients:\n" << "Weights:\n";
		// gradWeights.print();
		// std::cout << "Biases:\n";
		// gradBiases.print();

		linear.updateWeights(gradWeights, gradBiases, 0.01f);
		// std::cout << "Updated:\n";
		// linear.print();

		out = linear.forward(input);
		std::cout << "MSE: " << mse.loss(targets, out) <<
			"\n=====================================================================\n";
	}


}

void testActivation(){
	ReLU relu;
	MeanSquaredError mse;


	Matrix input = Matrix(3, 4);
	input.initRange();
	std::cout << "Input:\n";
	input.print();

	Matrix targets = Matrix(3, 4);
	targets.initRange();
	std::cout << "Targets:\n";
	targets.print();

	Matrix out = relu.forward(input);
	std::cout << "Outputs:\n";
	out.print();

	Matrix mseGrads = mse.gradient(targets, out);
	Matrix grads = relu.gradient(input, mseGrads);
	std::cout << "Gradients:\nMSE:\n";
	mseGrads.print();
	std::cout << "relu:\n";
	grads.print();
}

void testNN(){
	Matrix input = Matrix(3, 4);
	input.initRange();
	std::cout << "Input:\n";
	input.print();

	Matrix targets = Matrix(3, 2);
	targets.initRange();
	std::cout << "Targets:\n";
	targets.print();


	ReLU relu;
	MeanSquaredError mse;

	// Layers
	Linear linear1 = Linear(4, 8);
	linear1.initNormal();
	Linear linear2 = Linear(8, 8);
	linear2.initNormal();
	Linear linear3 = Linear(8, 2);
	linear3.initNormal();

	float lr = 0.0001f;

	// Training
	for (int i = 0; i < 100; ++i) {
		// Forward
		Matrix outLinear1 = linear1.forward(input);
		Matrix outRelu1 = relu.forward(outLinear1);
		Matrix outLinear2 = linear2.forward(outRelu1);
		Matrix outRelu2 = relu.forward(outLinear2);
		Matrix outLinear3 = linear3.forward(outRelu2);

		// Gradients
		Matrix mseGrad = mse.gradient(targets, outLinear3);
		auto [linearGradWeights3, linearGradBiases3] = linear3.calculateGradient(outRelu2, mseGrad);
		Matrix reluGrad2 = relu.gradient(outLinear2, linearGradWeights3);
		auto [linearGradWeights2, linearGradBiases2] = linear2.calculateGradient(outRelu1, reluGrad2);
		Matrix reluGrad1 = relu.gradient(outLinear1, linearGradWeights2);
		auto [linearGradWeights1, linearGradBiases1] = linear1.calculateGradient(input, reluGrad1);


		// Update weights
		linear1.updateWeights(linearGradWeights1, linearGradBiases1, lr);
		linear2.updateWeights(linearGradWeights2, linearGradBiases2, lr);
		linear3.updateWeights(linearGradWeights3, linearGradBiases3, lr);

		// Loss
		float loss = mse.loss(targets, outLinear3);
		std::cout << loss << "\n";

	}


}

void testNDMatOps(){
	// Constructor
	std::vector<size_t> aShape{1, 2, 3};
	std::vector<size_t> bShape{1, 3, 4};
	Tensor a(aShape), b(bShape);

	// Init Range
	a.initRange();
	std::cout << "A:\n";
	a.print();

	// Init Normal
	b.initNormal();
	std::cout << "B:\n";
	b.print();

	// Indexing
	std::vector<size_t> idx{0, 1, 1};
	std::cout << "A[0]: " << a[0] << "\n";
	std::cout << "A[1]: " << a[1] << "\n";
	std::cout << "A[2]: " << a[2] << "\n";
	std::cout << "A[0, 1, 2]: " << a[&idx] << "\n";

	// Reshaping
	std::vector<size_t> cShape{2, 1, 3};
	Tensor c = a.reshape(cShape);
	std::cout << "C = A (1, 2, 3) -> (2, 1, 3):\n";
	c.print();

	// Matrix multiplication
	Tensor d = a.matmul(b);
	std::cout << "D = AB:\n";
	d.print();

	// Addition
	Tensor e = a.add(a);
	std::cout << "E = A + A:\n";
	e.print();

	// Addition w/ broadcasting
	Tensor f = a.add(c);
	std::cout << "F = A + C:\n";
	f.print();

	// Subtraction
	Tensor g = a.sub(a);
	std::cout << "G = A - A:\n";
	g.print();

	// Subtraction w/ broadcasting
	Tensor h = a.sub(c);
	std::cout << "H = A - C:\n";
	h.print();
	

}

int main(){
	std::cout << "Hello World!\n";
	// testMatOps();
	// testLinear();
	// testActivation();
	// testNN();
	testNDMatOps();
}
