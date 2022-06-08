/*
 *   Copyright (c) 2022 seleneonowe
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *   
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *   
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "ActivationFunctions.hpp"
#include "MathUtils.hpp"

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
	for (int j = 0; j < inputs.cols(); j++) {
		// we need the total of all elements in a column (each column represents a batch)
		double colTotal=0;

		// find largest element in column so that we can prevent integer overflow by capping exp input to 0
		double largest = inputs.col(j).maxCoeff();

		//loop to exponentiate each element then add it to the running total
		for (int i = 0; i < inputs.rows(); i++) {
			inputs(i,j) -= largest;
			inputs(i,j) = exp(inputs(i,j));
			colTotal+=inputs(i,j);
		}
		
		//loop to divide each element by the total (normalization)
		for (int i = 0; i < inputs.rows(); i++) {
			inputs(i,j) /= colTotal;
		}
	}
	
}