#pragma once
#include <vector>

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

	Matrix sub(Matrix other);

	float& operator[](const std::size_t i){ return data[i]; };

	Matrix operator*(Matrix other){ return matmul(other); };

	Matrix operator+(Matrix other){ return add(other); };

	Matrix operator-(Matrix other){ return sub(other); };
};


void matmul(const float* a, const float* b, float* out, const int height, const int width, const int common);

void addMatrixMatrix(const float* a, const float* b, float* out, const int height, const int width);

void addMatrixVector(const float* a, const float* b, float* out, const int height, const int width);

void subMatrixMatrix(const float* a, const float* b, float* out, const int height, const int width);

void subMatrixVector(const float* a, const float* b, float* out, const int height, const int width);
