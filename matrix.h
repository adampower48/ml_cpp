#pragma once
class Matrix {
public:
	int height;
	int width;
	int* data;

	Matrix(int, int);

	void print();

	void initRange();

	Matrix matmul(Matrix other);

	Matrix add(Matrix other);
};

void matmul(const int* a, const int* b, int* out, const int height, const int width, const int common);

void addMatrixMatrix(const int* a, const int* b, int* out, const int height, const int width);

void addMatrixVector(const int* a, const int* b, int* out, const int height, const int width);
