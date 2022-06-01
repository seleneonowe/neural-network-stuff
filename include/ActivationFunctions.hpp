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