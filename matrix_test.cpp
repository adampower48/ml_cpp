// matrix_ops.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "matrix.h"

#include <iostream>
#include <tuple>

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
		std::cout << "MSE: " << mse.loss(targets, out) << "\n=====================================================================\n";
	}


}

int main(){
	std::cout << "Hello World!\n";
	// testMatOps();
	testLinear();
}
