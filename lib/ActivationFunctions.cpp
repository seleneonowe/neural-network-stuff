#include "ActivationFunctions.hpp"

void ActivationFunctions::forward_ReLU(MatrixXd &inputs) {
	for(int j = 0; j < inputs.cols(); j++) {
		for(int i = 0; i < inputs.rows(); i++) {
			ReLU(inputs(i,j));
		}
	}
}

void ActivationFunctions::ReLU(double &input) {
	if (input < 0) {
		input = 0;
	}
}

void ActivationFunctions::forward_softmax(MatrixXd &inputs) {
	for (int j = 0; j < inputs.cols(); j++) {
		// we need the total of all elements in a column (each column represents a batch)
		double colTotal=0;

		//loop to exponentiate each element then add it to the running total
		for (int i = 0; i < inputs.rows(); i++) {
			inputs(i,j) = exp(inputs(i,j));
			colTotal+=inputs(i,j);
		}

		//loop to divide each element by the total (normalization)
		for (int i = 0; i < inputs.rows(); i++) {
			inputs(i,j) /= colTotal;
		}
	}
}