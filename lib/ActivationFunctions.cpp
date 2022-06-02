#include "ActivationFunctions.hpp"
#include "MathUtils.hpp"
#include <iostream>

void ActivationFunctions::forward_ReLU(MatrixXd &inputs) {
	MathUtils::applyToAll(inputs,&ReLU);
}

void ActivationFunctions::backward_ReLU(MatrixXd &dvalues) {
	MathUtils::applyToAll(dvalues,&dReLU);
}

double ActivationFunctions::dReLU(double dvalue) {
	if (dvalue > 0) {
		return 1;
	} else {
		return 0;
	}
}

double ActivationFunctions::ReLU(double input) {
	if (input < 0) {
		return 0;
	} else {
		return input;
	}
}

void ActivationFunctions::forward_softmax(MatrixXd &inputs) {
	std::cout << "inputs to forward_softmax: \n" << inputs << std::endl;
	for (int j = 0; j < inputs.cols(); j++) {
		// we need the total of all elements in a column (each column represents a batch)
		double colTotal=0;

		// find largest element in column so that we can prevent integer overflow by capping exp input to 0
		double largest = inputs.col(j).maxCoeff();
		std::cout << "max coeff = " << largest << std::endl;

		//loop to exponentiate each element then add it to the running total
		for (int i = 0; i < inputs.rows(); i++) {
			inputs(i,j) -= largest;
			inputs(i,j) = exp(inputs(i,j));
			colTotal+=inputs(i,j);
		}
		std::cout << "colTotal = " << colTotal << std::endl;
		//loop to divide each element by the total (normalization)
		for (int i = 0; i < inputs.rows(); i++) {
			inputs(i,j) /= colTotal;
		}
	}

	std::cout << "inputs AFTER forward_softmax: \n" << inputs << std::endl;
	
}