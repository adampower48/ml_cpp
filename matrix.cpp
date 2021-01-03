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
	nDims = shape.size();

	size = 1;
	for (size_t dim : shape) {
		size *= dim;
	}

	if (copy) {
		data = new float[size]{0};
	}

}

float& Tensor::operator[](const std::vector<size_t>* idx){
	size_t i = 0;
	for (size_t j = 0; j < nDims; ++j) {
		i += idx->at(j) * vectorProd<size_t>(&shape, j + 1, idx->size());
	}

	return data[i];
};

void Tensor::print(){
	// Pretty printing for tensor

	// Print shape
	std::cout << "Shape: " << size << " " << strVector<size_t>(shape) << "\n";


	// Print elements
	size_t batchDims = size / (shape.back() * shape[nDims - 2]);
	for (size_t i = 0; i < batchDims; ++i) {
		for (size_t j = 0; j < shape[nDims - 2]; ++j) {
			for (size_t k = 0; k < shape.back(); ++k) {
				std::cout << data[i * shape.back() * shape[nDims - 2] + j * shape.back() + k] << "\t";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}

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


Tensor Tensor::matmul(Tensor other){

	// Check that dimensions are valid for matmul
	if (nDims < 2 || other.nDims < 2 ||
		nDims != other.nDims ||
		shape.back() != other.shape[other.nDims - 2] ||
		!vectorEq<size_t>(&shape, &other.shape, 0, nDims - 2)) {

		std::cout << "Shape mismatch for tensors: " << strVector<size_t>(shape) << ", " << strVector<size_t
		>(shape) << "\n";
		throw std::invalid_argument("Shape mismatch for tensor matrix multiplication.");
	}

	std::vector<size_t> newSize(shape);
	newSize.back() = other.shape.back();
	Tensor newTensor(newSize);
	::matmul(data, other.data, newTensor.data, size / (shape.back() * shape[nDims - 2]), shape[nDims - 2],
	         other.shape.back(), shape.back());

	return newTensor;
}


void matmul(const float* a, const float* b, float* out, const size_t batchDims, const size_t height, const size_t width,
            const size_t common){
	// Tensor matrix multiplication over last 2 dimensions
	// a: array of shape (batchDims..., height, common)
	// b: array of shape (batchDims..., common, width)
	// out: array of shape (batchDims..., height, width), output of calculation

	for (size_t batch = 0; batch < batchDims; ++batch) {
		for (size_t i = 0; i < height; ++i) {
			for (size_t j = 0; j < width; ++j) {
				for (size_t c = 0; c < common; ++c) {
					out[batch * height * width + i * width + j] +=
						a[batch * height * width + i * common + c] *
						b[batch * height * width + c * width + j];
				}
			}
		}
	}
}
