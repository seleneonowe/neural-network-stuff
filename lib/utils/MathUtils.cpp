#include "MathUtils.hpp"

void MathUtils::clip(MatrixXd &inputs, const double &min, const double &max) {
	applyToAll(inputs,min,max,&clip);
}

double MathUtils::clip(double input, double min, double max) {
	if(input < min) {
		return min;
	} else if (input > max) {
		return max;
	} else {
		return input;
	}
}

void MathUtils::applyToAll(MatrixXd &inputs, double (*func)(double)) {
	for(unsigned j = 0; j < inputs.cols(); j++) {
		for(unsigned i = 0; i < inputs.rows(); i++) {
			inputs(i,j) = func(inputs(i,j));
		}
	}
}
void MathUtils::applyToAll(MatrixXd &inputs, double arg1, double arg2, double (*func)(double,double,double)) {
	for(unsigned j = 0; j < inputs.cols(); j++) {
		for(unsigned i = 0; i < inputs.rows(); i++) {
			inputs(i,j) = func(inputs(i,j),arg1,arg2);
		}
	}
}