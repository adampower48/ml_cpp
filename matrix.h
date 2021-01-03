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

	void initNormal(int seed = 0);

	Matrix matmul(Matrix other);

	Matrix add(Matrix other);

	Matrix sub(Matrix other);

	float& operator[](const std::size_t i){ return data[i]; };

	Matrix operator*(Matrix other){ return matmul(other); };

	Matrix operator+(Matrix other){ return add(other); };

	Matrix operator-(Matrix other){ return sub(other); };
};

class Tensor {
public:
	std::vector<size_t> shape;
	size_t nDims;
	size_t size;
	float* data;

	Tensor(std::vector<size_t> shape, bool copy = true);

	void print();

	void initRange();

	void initNormal(int seed = 0);

	Tensor reshape(std::vector<size_t> shape);

	Tensor matmul(Tensor other);

	Tensor add(Tensor other);

	Tensor sub(Tensor other);


	float& operator[](const std::size_t i){ return data[i]; }

	float& operator[](const int i){ return operator[](static_cast<size_t>(i)); }

	float& operator[](const std::vector<size_t>* idx);

};

void matmul(const float* a, const float* b, float* out, int height, int width, int common);

void matmul(const float* a, const float* b, float* out, size_t batchDims, size_t height, size_t width, size_t common);

void addMatrixMatrix(const float* a, const float* b, float* out, int height, int width);

void addMatrixVector(const float* a, const float* b, float* out, int height, int width);

void subMatrixMatrix(const float* a, const float* b, float* out, int height, int width);

void subMatrixVector(const float* a, const float* b, float* out, int height, int width);
