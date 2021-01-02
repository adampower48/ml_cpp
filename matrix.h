#pragma once
class Matrix {
public:
	int height;
	int width;
	float* data;

	Matrix(int, int);

	void print();

	void initRange();

	void initNormal(const int seed = 0);

	Matrix matmul(Matrix other);

	Matrix add(Matrix other);
};

void matmul(const float* a, const float* b, float* out, const int height, const int width, const int common);

void addMatrixMatrix(const float* a, const float* b, float* out, const int height, const int width);

void addMatrixVector(const float* a, const float* b, float* out, const int height, const int width);
