#pragma once
#include <vector>

#include "vector.h"

class Tensor {
public:
	std::vector<size_t> shape;
	std::vector<size_t> indexer;
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

	float& operator[](const std::vector<size_t>* idx){ return data[vectorDot(&indexer, idx)]; }


	Tensor operator*(Tensor other){ return matmul(other); };

	Tensor operator+(Tensor other){ return add(other); };

	Tensor operator-(Tensor other){ return sub(other); };

};


void matmul(const float* a, const float* b, float* out, size_t batchDims, size_t height, size_t width, size_t common);


void add(const float* a, const float* b, float* out, const std::vector<size_t>* aIndexer,
         const std::vector<size_t>* bIndexer, const std::vector<size_t>* outIndexer,
         const std::vector<size_t>* outShape);

void sub(const float* a, const float* b, float* out, const std::vector<size_t>* aIndexer,
         const std::vector<size_t>* bIndexer, const std::vector<size_t>* outIndexer,
         const std::vector<size_t>* outShape);
