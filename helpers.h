#pragma once
#include <string>
#include <vector>


template <typename T>
std::string strVector(std::vector<T> vec){
	std::string out = "(";

	for (size_t i = 0; i < vec.size() - 1; ++i) {
		out += std::to_string(vec[i]) + ", ";
	}
	out += std::to_string(vec.back()) + ")";

	return out;
}
