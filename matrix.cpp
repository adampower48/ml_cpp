#include "matrix.h"

#include <iostream>
#include <vector>

#include "helpers.h"


Matrix::Matrix(int h, int w){
	height = h;
	width = w;
	data = new int[h * w]{0}; // init to 0
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
		data[i] = i;
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

void matmul(const int* a, const int* b, int* out, const int height, const int width, const int common){
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

void addMatrixMatrix(const int* a, const int* b, int* out, const int height, const int width){
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

void addMatrixVector(const int* a, const int* b, int* out, const int height, const int width){
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
