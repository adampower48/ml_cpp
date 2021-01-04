#pragma once
#include "matrix.h"


class Loss {
public:
	virtual float loss(Tensor truth, Tensor pred) = 0;

	virtual Tensor gradient(Tensor truth, Tensor pred) = 0;

};

class MeanSquaredError : public Loss {
public:
	float loss(Tensor truth, Tensor pred) override;

	Tensor gradient(Tensor truth, Tensor pred) override;
};
