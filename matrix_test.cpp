// matrix_ops.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include "matrix.h"
#include "linear.h"
#include <tuple>

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
	Linear linear = Linear(4, 8);
	// linear.initRange();
	linear.initNormal();
	linear.print();

	Matrix input = Matrix(3, 4);
	input.initRange();
	std::cout << "Input:\n";
	input.print();

	Matrix out = linear.forward(input);
	std::cout << "Output:\n";
	out.print();

	Matrix targets = Matrix(3, 8);
	targets.initRange();
	std::cout << "Targets:\n";
	targets.print();

	auto [gradWeights, gradBiases] = linear.calculateGradient(input, out, targets);
	std::cout << "Gradients:\n" << "Weights:\n";
	gradWeights.print();
	std::cout << "Biases:\n";
	gradBiases.print();


}

int main(){
	std::cout << "Hello World!\n";
	// testMatOps();
	testLinear();
}
