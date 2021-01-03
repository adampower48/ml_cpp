#pragma once
#include "matrix.h"

class Activation {
public:
	virtual Matrix forward(Matrix input) = 0;

	virtual Matrix gradient(Matrix input, Matrix nextGrads) = 0;

};

class ReLU : public Activation {
public:
	virtual Matrix forward(Matrix input) override;

	virtual Matrix gradient(Matrix input, Matrix nextGrads) override;
	
};
