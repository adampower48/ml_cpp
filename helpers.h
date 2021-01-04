#pragma once
#include <vector>


inline std::tuple<size_t, size_t> divmod(size_t a, size_t b){
	return std::make_tuple(a / b, a % b);
}
