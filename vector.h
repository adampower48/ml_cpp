#pragma once

#include <string>
#include <vector>

#include "helpers.h"


template <typename T>
std::string strVector(std::vector<T> vec){
	std::string out = "(";

	for (size_t i = 0; i < vec.size() - 1; ++i) {
		out += std::to_string(vec[i]) + ", ";
	}
	out += std::to_string(vec.back()) + ")";

	return out;
}


template <typename T>
bool vectorEq(std::vector<T>* a, std::vector<T>* b, size_t start, size_t end){
	for (size_t i = start; i < end; ++i) {
		if (a->at(i) != b->at(i)) {
			return false;
		}
	}
	return true;
}

template <typename T>
T vectorProd(const std::vector<T>* a, size_t start, size_t end){
	T out = 1;
	for (size_t i = start; i < end; ++i) {
		out *= a->at(i);
	}

	return out;
}


template <typename T>
T vectorDot(const std::vector<T>* a, const std::vector<T>* b){
	T out = 0;

	for (int i = 0; i < a->size(); ++i) {
		out += a->at(i) * b->at(i);
	}

	return out;
}

std::vector<size_t> buildIndexer(const std::vector<size_t>* shape);

std::vector<size_t> vectorUnravel(const std::vector<size_t>* indexer, size_t idx);
