#pragma once

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class InitFunctions
{
public:
	static void HeInitialize(MatrixXd &input);
	static void HeInitialize(VectorXd &input);
};