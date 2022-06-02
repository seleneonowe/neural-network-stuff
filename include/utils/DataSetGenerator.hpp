#pragma once

#include <Eigen/Dense>
#include <tuple>

using Eigen::MatrixXd;
using Eigen::RowVectorXi;
using std::tuple;

class DataSetGenerator {
	public:
		// makes <classes> sets of spiral data with <samples> samples.
		static tuple<MatrixXd,RowVectorXi> createSpiralData(unsigned int samples, unsigned int classes);
};