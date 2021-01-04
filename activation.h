#pragma once
#include "tensor.h"


class Activation {
public:
	virtual Tensor forward(Tensor input) = 0;

	virtual Tensor gradient(Tensor input, Tensor nextGrads) = 0;

};

class ReLU : public Activation {
public:
	Tensor forward(Tensor input) override;

	Tensor gradient(Tensor input, Tensor nextGrads) override;

};
