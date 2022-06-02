#pragma once

#include <Eigen/Dense>
#include <tuple>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::tuple;

class DataSetGenerator {
	public:
		// makes <classes> sets of spiral data with <samples> samples.
		static tuple<MatrixXd,VectorXd> createSpiralData(unsigned int samples, unsigned int classes);
};