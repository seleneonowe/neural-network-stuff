#pragma once
#include <Eigen/Dense>

using Eigen::MatrixXd;

class ActivationFunctions
{
public:
	// The ReLU activation function applied to a matrix during a forward pass
	static void forward_ReLU(MatrixXd &inputs);

	static MatrixXd backward_ReLU(MatrixXd dvalues);

	// The softmax activation function applied during a forward pass
	static void forward_softmax(MatrixXd &inputs);

	static MatrixXd backward_softmax(MatrixXd dvalues);
	
	// The ReLU activation function applied to a scalar
	static void ReLU(double &input);

	// The derivative of the ReLU function applied to a scalar
	static double dReLU(double dvalue);
};