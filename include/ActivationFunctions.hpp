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


#pragma once
#include <Eigen/Dense>

using Eigen::MatrixXd;

class ActivationFunctions
{
public:
	// The ReLU activation function applied to a matrix during a forward pass
	static void forward_ReLU(MatrixXd &inputs);

	// The derivative of the ReLU function.
	static void backward_ReLU(MatrixXd &dvalues);

	// The softmax activation function applied during a forward pass
	static void forward_softmax(MatrixXd &inputs);

// TODO: write backwards softmax function
//	static void backward_softmax(MatrixXd &dvalues);
	
	// The ReLU activation function applied to a scalar
	static double ReLU(double input);

	// The derivative of the ReLU function applied to a scalar
	static double dReLU(double dvalue);
};