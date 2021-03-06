// matrix_ops.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "tensor.h"

#include <iostream>
#include <tuple>

#include "activation.h"
#include "linear.h"
#include "loss.h"

void testLinear(){
	Linear linear = Linear(4, 3);
	// linear.initRange();
	linear.initNormal();
	linear.print();

	std::vector<size_t> inputShape{3, 4};
	Tensor input = Tensor(inputShape);
	input.initRange();
	std::cout << "Input:\n";
	input.print();

	std::vector<size_t> targetShape{3, 3};
	Tensor targets = Tensor(targetShape);
	targets.initRange();
	std::cout << "Targets:\n";
	targets.print();

	MeanSquaredError mse;

	std::cout << "===================================================================\n";
	for (int i = 0; i < 1000; ++i) {
		Tensor out = linear.forward(input);
		// std::cout << "Output:\n";
		// out.print();

		// D = Y - Y*
		Tensor lossGrads = mse.gradient(targets, out);
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

	std::vector<size_t> inputShape{3, 4};
	Tensor input = Tensor(inputShape);
	input.initRange();
	std::cout << "Input:\n";
	input.print();

	std::vector<size_t> targetsShape{3, 4};
	Tensor targets = Tensor(targetsShape);
	targets.initRange();
	std::cout << "Targets:\n";
	targets.print();

	Tensor out = relu.forward(input);
	std::cout << "Outputs:\n";
	out.print();

	Tensor mseGrads = mse.gradient(targets, out);
	Tensor grads = relu.gradient(input, mseGrads);
	std::cout << "Gradients:\nMSE:\n";
	mseGrads.print();
	std::cout << "relu:\n";
	grads.print();
}


void testNN(){

	std::vector<size_t> inputShape{3, 4};
	Tensor input = Tensor(inputShape);
	input.initRange();
	std::cout << "Input:\n";
	input.print();

	std::vector<size_t> targetsShape{3, 2};
	Tensor targets = Tensor(targetsShape);
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
		Tensor outLinear1 = linear1.forward(input);
		Tensor outRelu1 = relu.forward(outLinear1);
		Tensor outLinear2 = linear2.forward(outRelu1);
		Tensor outRelu2 = relu.forward(outLinear2);
		Tensor outLinear3 = linear3.forward(outRelu2);


		// Gradients
		Tensor mseGrad = mse.gradient(targets, outLinear3);
		auto [linearGradWeights3, linearGradBiases3] = linear3.calculateGradient(outRelu2, mseGrad);
		Tensor reluGrad2 = relu.gradient(outLinear2, linearGradWeights3);
		auto [linearGradWeights2, linearGradBiases2] = linear2.calculateGradient(outRelu1, reluGrad2);
		Tensor reluGrad1 = relu.gradient(outLinear1, linearGradWeights2);
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

void testTensorOps(){
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


	// * Operator
	Tensor i = a * b;
	std::cout << "I = AB\n";
	i.print();

	// + Operator (Mat-Vec)
	Tensor j = a + c;
	std::cout << "J = A + C\n";
	j.print();

	// - Operator (Mat-Mat)
	Tensor k = a - a;
	std::cout << "K = A - A\n";
	k.print();


}

int main(){
	std::cout << "Hello World!\n";
	// testTensorOps();
	// testLinear();
	// testActivation();
	testNN();
}
