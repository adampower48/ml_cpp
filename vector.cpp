#include "vector.h"

std::vector<size_t> buildIndexer(const std::vector<size_t>* shape){
	// Builds indexes used to broadcast tensors
	// Example:
	// vec A: shape (2, 1, 3)
	// indexer A (iA): (3, 0, 1)
	// iA[1, 1, 1] = iA[1*3 + 1*0 + 1] = iA[4]

	std::vector<size_t> indexer(shape->size());
	size_t prod = 1;
	for (size_t i = shape->size() - 1; i > 0; --i) {
		// Cumprod + masking
		indexer[i] = prod * (shape->at(i) > 1);
		prod *= shape->at(i);
	}
	indexer[0] = prod * (shape->at(0) > 1);

	return indexer;
}

std::vector<size_t> vectorUnravel(const std::vector<size_t>* indexer, size_t idx){
	std::vector<size_t> unraveled(indexer->size());
	for (size_t i = 0; i < indexer->size(); ++i) {
		if (indexer->at(i) == 0) {
			unraveled[i] = 0;
		} else {
			auto [dv, rem] = divmod(idx, indexer->at(i));
			idx = rem;
			unraveled[i] = dv;
		}
	}

	return unraveled;
}
