#pragma once
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

class MathUtils {
	public:
	static void clip(MatrixXd &inputs, const double &min, const double &max);

	static double clip(double input, double min, double max);

	static bool isShapeEqual(MatrixXd &matrix1, MatrixXd &matrix2);

	static void convertToOneHotEncoded(MatrixXd &inputs, unsigned );

	static std::vector<std::tuple<MatrixXd,VectorXd>> createRandomBatchesFromDataSet(std::tuple<MatrixXd,VectorXd> &dataSet,unsigned numberOfBatches);
	
	static void applyToAll(MatrixXd &inputs, double (*func)(double));

	static void applyToAll(MatrixXd &inputs, double, double, double (*func)(double,double,double));

	static std::tuple<VectorXd, double> linspace(double start, double stop, unsigned int num=10, bool endpoint=false);
};