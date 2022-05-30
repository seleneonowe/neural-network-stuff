#pragma once

#include <Eigen/Dense>

using Eigen::MatrixXd;

class InitFunctions
{
public:
	static void HeInitialize(MatrixXd &input);
};