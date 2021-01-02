#pragma once
class Linear {
public:
	int input_size;
	int output_size;
	Matrix *weights;
	Matrix *biases;

	Linear(int input_size, int units);

	void initRange();

	void print();

	Matrix forward(Matrix input);
};
