#include "tensor.h"

#include <cmath>
#include <iostream>
#include <random>

#include "vector.h"


Tensor::Tensor(std::vector<size_t> shape, bool copy){
	Tensor::shape = shape;
	indexer = buildIndexer(&shape);
	nDims = shape.size();

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

Tensor Tensor::add(Tensor other){
	// Check dimensions are valid
	if (nDims != other.nDims) {
		std::cout << "Number of dimensions don't match: " << nDims << ", " << other.nDims << "\n";
		throw std::invalid_argument("Shape mismatch for tensor addition.");
	}

	// Get new dims
	std::vector<size_t> newShape(nDims);
	for (size_t i = 0; i < nDims; ++i) {
		if (shape[i] == other.shape[i]) {
			// Same dimensions
			newShape[i] = shape[i];
		} else if (shape[i] == 1 || other.shape[i] == 1) {
			// Broadcasting
			newShape[i] = std::max(shape[i], other.shape[i]);
		} else {
			// Invalid dims
			std::cout << "Non-broadcastable dimensions: " << strVector(shape) << ", " << strVector(other.shape) << "\n";
			throw std::invalid_argument("Shape mismatch for tensor addition.");
		}
	}

	Tensor newTensor(newShape);
	::add(data, other.data, newTensor.data, &indexer, &other.indexer, &newTensor.indexer, &newShape);

	return newTensor;
}

Tensor Tensor::sub(Tensor other){
	// Check dimensions are valid
	if (nDims != other.nDims) {
		std::cout << "Number of dimensions don't match: " << nDims << ", " << other.nDims << "\n";
		throw std::invalid_argument("Shape mismatch for tensor addition.");
	}

	// Get new dims
	std::vector<size_t> newShape(nDims);
	for (size_t i = 0; i < nDims; ++i) {
		if (shape[i] == other.shape[i]) {
			// Same dimensions
			newShape[i] = shape[i];
		} else if (shape[i] == 1 || other.shape[i] == 1) {
			// Broadcasting
			newShape[i] = std::max(shape[i], other.shape[i]);
		} else {
			// Invalid dims
			std::cout << "Non-broadcastable dimensions: " << strVector(shape) << ", " << strVector(other.shape) << "\n";
			throw std::invalid_argument("Shape mismatch for tensor addition.");
		}
	}

	Tensor newTensor(newShape);
	::sub(data, other.data, newTensor.data, &indexer, &other.indexer, &newTensor.indexer, &newShape);

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


void add(const float* a, const float* b, float* out, const std::vector<size_t>* aIndexer,
         const std::vector<size_t>* bIndexer, const std::vector<size_t>* outIndexer,
         const std::vector<size_t>* outShape){

	for (size_t i = 0; i < vectorProd<size_t>(outShape, 0, outShape->size()); ++i) {
		auto x = vectorUnravel(outIndexer, i);
		auto aIdx = vectorDot<size_t>(&x, aIndexer);
		auto bIdx = vectorDot<size_t>(&x, bIndexer);

		out[i] = a[aIdx] + b[bIdx];
	}
}

void sub(const float* a, const float* b, float* out, const std::vector<size_t>* aIndexer,
         const std::vector<size_t>* bIndexer, const std::vector<size_t>* outIndexer,
         const std::vector<size_t>* outShape){

	for (size_t i = 0; i < vectorProd<size_t>(outShape, 0, outShape->size()); ++i) {
		auto x = vectorUnravel(outIndexer, i);
		auto aIdx = vectorDot<size_t>(&x, aIndexer);
		auto bIdx = vectorDot<size_t>(&x, bIndexer);

		out[i] = a[aIdx] - b[bIdx];
	}
}
