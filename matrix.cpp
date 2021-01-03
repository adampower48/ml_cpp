#include "matrix.h"

#include <iostream>
#include <random>

#include "helpers.h"


Matrix::Matrix(int h, int w){
	height = h;
	width = w;
	data = new float[h * w]{0}; // init to 0
}

void Matrix::print(){
	// Pretty printing for matrix
	for (auto i = 0; i < height; ++i) {
		for (auto j = 0; j < width; ++j) {
			std::cout << data[i * width + j] << "\t";
		}
		std::cout << "\n";
	}
}

void Matrix::initRange(){
	// Initialise the matrix with an increasing sequence of values
	for (auto i = 0; i < height * width; ++i) {
		data[i] = static_cast<float>(i) / (height * width);
	}
}

void Matrix::initNormal(const int seed){
	// Initialise the matrix with random values from a Normal(0, 1) distribution.
	std::default_random_engine gen(seed);
	std::normal_distribution<float> normal;

	for (auto i = 0; i < height * width; ++i) {
		data[i] = normal(gen);
	}
}

Matrix Matrix::matmul(Matrix other){
	// Matrix multiplication
	// C = AB

	// Check that dimensions are valid for matmul
	if (width != other.height) {
		printf("Shape mismatch for matrices: (%d, %d), (%d, %d)", height, width, other.height, other.width);
		throw std::invalid_argument("Shape mismatch for matrices.");
	}

	// Create new matrix and perform matmul
	Matrix newMat(height, other.width);
	::matmul(data, other.data, newMat.data, height, other.width, width);

	return newMat;
}

Matrix Matrix::add(Matrix other){
	// Matrix Addition
	// C = A + B

	// Check that dimensions are valid for addition
	if (width != other.width) {
		printf("Shape mismatch for matrices: (%d, %d), (%d, %d)", height, width, other.height, other.width);
		throw std::invalid_argument("Shape mismatch for matrices.");
	}


	Matrix newMat(height, width);
	if (height == other.height) {
		::addMatrixMatrix(data, other.data, newMat.data, height, width);
	} else {
		::addMatrixVector(data, other.data, newMat.data, height, width);
	}

	return newMat;

}

Matrix Matrix::sub(Matrix other){
	// Check that dimensions are valid for addition
	if (width != other.width) {
		printf("Shape mismatch for matrices: (%d, %d), (%d, %d)", height, width, other.height, other.width);
		throw std::invalid_argument("Shape mismatch for matrices.");
	}


	Matrix newMat(height, width);
	if (height == other.height) {
		::subMatrixMatrix(data, other.data, newMat.data, height, width);
	} else {
		::subMatrixVector(data, other.data, newMat.data, height, width);
	}

	return newMat;

}

void matmul(const float* a, const float* b, float* out, const int height, const int width, const int common){
	// Matrix multiplication
	// a: array of shape (height, common)
	// b: array of shape (common, width)
	// out: array of shape (height, width), output of calculation


	for (auto i = 0; i < height; ++i) {
		for (auto j = 0; j < width; ++j) {
			for (auto c = 0; c < common; ++c) {
				out[i * width + j] += a[i * common + c] * b[c * width + j];
			}
		}
	}
}

void addMatrixMatrix(const float* a, const float* b, float* out, const int height, const int width){
	// Matrix Addition
	// a: array of shape (height, width)
	// b: array of shape (height, width)
	// out: array of shape (height, width), output of calculation

	for (auto i = 0; i < height; ++i) {
		for (auto j = 0; j < width; ++j) {
			out[i * width + j] = a[i * width + j] + b[i * width + j];
		}
	}
}

void addMatrixVector(const float* a, const float* b, float* out, const int height, const int width){
	// Matrix-Vector Addition
	// a: array of shape (height, width)
	// b: array of shape (width,)
	// out: array of shape (height, width), output of calculation

	for (auto i = 0; i < height; ++i) {
		for (auto j = 0; j < width; ++j) {
			out[i * width + j] = a[i * width + j] + b[j];
		}
	}
}


void subMatrixMatrix(const float* a, const float* b, float* out, const int height, const int width){
	// Matrix Subtraction A - B
	// a: array of shape (height, width)
	// b: array of shape (height, width)
	// out: array of shape (height, width), output of calculation

	for (auto i = 0; i < height; ++i) {
		for (auto j = 0; j < width; ++j) {
			out[i * width + j] = a[i * width + j] - b[i * width + j];
		}
	}
}

void subMatrixVector(const float* a, const float* b, float* out, const int height, const int width){
	// Matrix-Vector Subtraction A - B
	// a: array of shape (height, width)
	// b: array of shape (width,)
	// out: array of shape (height, width), output of calculation

	for (auto i = 0; i < height; ++i) {
		for (auto j = 0; j < width; ++j) {
			out[i * width + j] = a[i * width + j] - b[j];
		}
	}
}

Tensor::Tensor(std::vector<size_t> shape, bool copy){
	Tensor::shape = shape;
	size = 1;
	for (size_t dim : shape) {
		size *= dim;
	}

	if (copy) {
		data = new float[size]{0};
	}

}

void Tensor::print(){
	// Pretty printing for tensor

	// Print shape
	std::cout << "Shape: " << size << " " << strVector<size_t>(shape) << "\n";


	// Print elements TODO: Pretty printing
	for (size_t i = 0; i < size; ++i) {
		std::cout << data[i] << "\t";
	}
	std::cout << "\n";


}

void Tensor::initRange(){
	// Initialise the tensor with an increasing sequence of values
	for (size_t i = 0; i < size; ++i) {
		data[i] = static_cast<float>(i) / size;
	}
}

void Tensor::initNormal(const int seed){
	// Initialise the tensor with random values from a Normal(0, 1) distribution.
	std::default_random_engine gen(seed);
	std::normal_distribution<float> normal;

	for (size_t i = 0; i < size; ++i) {
		data[i] = normal(gen);
	}
}

Tensor Tensor::reshape(std::vector<size_t> shape){
	// Reshape the tensor, keeping same underlying data

	// Verify same total elements
	size_t tmpSize = 1;
	for (size_t dim : shape) {
		tmpSize *= dim;
	}

	if (size != tmpSize) {
		std::cout << "Number of elements do not match: " << strVector<size_t>(Tensor::shape) << ", " << strVector<size_t
		>(shape) << "\n";
		throw std::invalid_argument("Size mismatch for tensor reshaping.");
	}

	Tensor newTensor(shape, false);
	newTensor.data = data;

	return newTensor;
}
