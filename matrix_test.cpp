// matrix_ops.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include "matrix.h"
#include "linear.h"

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

	Matrix c = a.matmul(b);
	std::cout << "C = AB\n";
	c.print();


	Matrix e = a.add(a);
	std::cout << "E = A + A\n";
	e.print();

	Matrix f = c.add(d);
	std::cout << "F = C + D\n";
	f.print();
	
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
	
	
}

int main(){
	std::cout << "Hello World!\n";
	testLinear();
}
