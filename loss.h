#pragma once
#include "matrix.h"


class Loss {
public:
	virtual float loss(Matrix truth, Matrix pred) = 0;

	virtual Matrix gradient(Matrix truth, Matrix pred) = 0;

};

class MeanSquaredError : public Loss {
public:
	float loss(Matrix truth, Matrix pred) override;

	Matrix gradient(Matrix truth, Matrix pred) override;
};
